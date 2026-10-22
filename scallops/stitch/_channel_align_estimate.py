"""Estimate each channel's alignment to a reference (nuclear) channel from the tiles of one well.

The model is the one applied by :mod:`scallops.stitch._channel_align`: a magnification ``m`` about the tile centre
and a translation ``(ty, tx)``, in pixels, such that ``corrected(p) = raw(c + (1 + m) * (p - c) + (ty, tx))``.

This module does not depend on the stitching code, so it can also be used on its own, e.g. to build a parameter file
for a new microscope::

    import numpy as np
    import tifffile
    from scallops.stitch._channel_align_estimate import estimate_well


    def read_tile(i, channels):
        ref = tifffile.imread(dapi_files[i])
        return ref, {c: tifffile.imread(channel_files[c][i]) for c in channels}


    result = estimate_well(
        read_tile, positions_um, channels=["FITC", "CY7"], pixel_size_um=0.3222
    )
    print(result["channels"]["CY7"])  # m, ty, tx, quality values, tiles used

Per tile (:func:`estimate_tile`):

- nuclei are found once on the reference channel;
- **S** gives ``m`` (point symmetry of the channel's mean patch around the nuclei, in 5 x 5 regions, model fitted to
  the regional shifts, 3 iterations);
- **ET** gives ``(ty, tx)`` with ``m`` fixed at the tile's S value (geometry of the nucleus outline in the mean
  patch, 10 iterations from S's translation).

A tile is valid for a channel when its reference channel has nuclei, S's regional fit rms is < 3 px, ET converged
(last step < 0.01 px), and the tile has at least 25% of the median nucleus count of the well's first-drawn tiles. An
exception while estimating a tile is not caught: it stops the estimation. Pixel sizes
of all settings scale with the nucleus radius measured on the reference channel (anchored to the validated settings
of a 20x IXM, nucleus radius 12 px); the minimum nucleus area scales with the pixel size when it is known.

Per well (:func:`estimate_well`): up to 100 valid tiles, sampled at random, spread over 4 quadrants around the well
centre (25 each, minimum spacing for the first draws), interior tiles only for wells with more than 100 tiles;
invalid tiles are replaced from the same quadrant; each quadrant stops after 50 tries and gives its unfilled slots to
the other quadrants. The well's value is the median over the valid tiles (m from S, translation from ET), with
quality checks (see :data:`CHECKS`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import dask
import numpy as np
from dask.diagnostics import ProgressBar
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage import feature, filters, measure, morphology, segmentation
from skimage.registration import phase_cross_correlation

logger = logging.getLogger("scallops")

# validated settings (pheno screen IXM, 20x, 0.3222 um/px), and the nucleus radius they correspond to
REF_NUCLEUS_RADIUS = 12.0  # px: median-area radius of the reference channel's nuclei
REF_PIXEL_SIZE = 0.3222  # um
REF_MIN_AREA = 150.0  # px
REF_PATCH_RADIUS = 48  # px: half-size of the mean patch around each nucleus
REF_NORM_RADIUS = 20.0  # px: disk for each nucleus' brightness normalisation
REF_SEG_SIGMA = 4.0  # px (full resolution): smoothing for nucleus segmentation
REF_PEAK_DISTANCE = 16.0  # px (full resolution): watershed seed spacing
REF_RADIUS_SEARCH = (4.0, 45.0)  # px: where the nucleus edge is searched for
REF_BORDER = 8.0  # px: extra margin beyond the patch at the tile border
N_REGIONS = 5  # regions per axis
S_ITERATIONS = 3
ET_ITERATIONS = 10
ET_STEP_TOL = 0.01  # px
MAX_REGION_RMS = 3.0  # px
NUCLEI_FLOOR = 0.25  # fraction of the median nucleus count of the first-drawn tiles

N_TILES = 100
N_QUADRANTS = 4
QUADRANT_CAP = 50  # tries per quadrant
SPACING_FACTOR = 0.6  # minimum spacing of first draws: SPACING_FACTOR * sqrt(pool / N_TILES) tile steps
INTERIOR_FRACTION = 0.8
INTERIOR_DEPTHS = (
    3,
    2,
    None,
)  # tried in order for wells with more than N_TILES tiles; None = rank rule only
N_SCALE_TILES = 10  # tiles whose reference channel sets the well's nucleus scale

CHECKS = dict(
    min_valid=20, min_valid_fraction=0.5, max_ci95_corner_px=0.5, max_corner_px=10.0
)
N_BOOTSTRAP = 1000


class ChannelAlignmentEstimationError(ValueError):
    """Raised when the alignment of one or more channels can't be estimated reliably."""


# ---- nucleus scale ------------------------------------------------------------------------------------------------


def nucleus_radius(ref: np.ndarray) -> float:
    """Median-area radius (px) of the reference channel's foreground objects, without size assumptions."""
    sm = ndi.gaussian_filter(np.asarray(ref, np.float32), 1.0)
    if not np.isfinite(sm).all() or sm.max() <= sm.min():
        return float("nan")
    lab, n = ndi.label(sm > filters.threshold_otsu(sm))
    if n == 0:
        return float("nan")
    area = np.bincount(lab.ravel())[1:]
    area = area[area >= 4]
    return float(np.sqrt(np.median(area) / np.pi)) if len(area) else float("nan")


def scale_settings(radius: float, pixel_size_um: float | None = None) -> dict:
    """Estimator settings for nuclei of the given radius (px).

    Every length scales with ``radius / REF_NUCLEUS_RADIUS``. The minimum nucleus area scales with the pixel size
    when it is known, otherwise with the nucleus size.
    """
    s = radius / REF_NUCLEUS_RADIUS
    downsample = 2 if radius >= 8 else 1
    if pixel_size_um:
        min_area = REF_MIN_AREA * (REF_PIXEL_SIZE / pixel_size_um) ** 2
    else:
        min_area = REF_MIN_AREA * s**2
    patch = max(4, int(round(REF_PATCH_RADIUS * s)))
    return dict(
        radius=float(radius),
        scale=float(s),
        downsample=downsample,
        min_area=float(min_area),
        patch_radius=patch,
        norm_radius=REF_NORM_RADIUS * s,
        seg_sigma=REF_SEG_SIGMA * s / downsample,
        peak_distance=max(1, int(round(REF_PEAK_DISTANCE * s / downsample))),
        radius_search=(
            REF_RADIUS_SEARCH[0] * s,
            min(REF_RADIUS_SEARCH[1] * s, patch - 1.0),
        ),
        border=patch + REF_BORDER * s,
    )


# ---- per-tile estimators --------------------------------------------------------------------------------------------


def _prep(img: np.ndarray) -> np.ndarray:
    """Background-subtract and mildly smooth."""
    img = ndi.gaussian_filter(np.asarray(img, np.float32), 1.0)
    return img - np.percentile(img, 5)


def nucleus_centres(ref: np.ndarray, settings: dict) -> np.ndarray:
    """(N, 2) nucleus centroids (y, x) in full-resolution pixels: Otsu + distance-transform watershed."""
    f = settings["downsample"]
    h, w = (ref.shape[0] // f) * f, (ref.shape[1] // f) * f
    small = (
        np.asarray(ref[:h, :w], np.float32)
        .reshape(h // f, f, w // f, f)
        .mean(axis=(1, 3))
    )
    sm = ndi.gaussian_filter(small, settings["seg_sigma"])
    if sm.max() <= sm.min():
        return np.zeros((0, 2))
    fg = sm > filters.threshold_otsu(sm)
    fg = morphology.remove_small_objects(
        ndi.binary_fill_holes(fg), max_size=settings["min_area"] / (f * f)
    )
    dist = ndi.distance_transform_edt(fg)
    peaks = feature.peak_local_max(
        dist,
        min_distance=settings["peak_distance"],
        labels=measure.label(fg),
        exclude_border=False,
    )
    markers = np.zeros(fg.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = segmentation.watershed(-dist, markers, mask=fg)
    c = np.array([p.centroid for p in measure.regionprops(lab)]).reshape(-1, 2)
    return c * f + (f - 1) / 2


def _regions(cy, cx, shape):
    return np.minimum(
        (cy / shape[0] * N_REGIONS).astype(int), N_REGIONS - 1
    ) * N_REGIONS + np.minimum((cx / shape[1] * N_REGIONS).astype(int), N_REGIONS - 1)


def _region_mean_patches(
    img, ys, xs, region, R, norm_radius, norm=True, norm_floor_pct=50
):
    """Mean bilinear-sampled (2R+1)^2 patch around (ys, xs) per region; each nucleus capped at the median level."""
    from numpy.lib.stride_tricks import sliding_window_view

    n_regions = N_REGIONS * N_REGIONS
    S = 2 * R + 2
    view = sliding_window_view(img, (S, S))
    iy, ix = np.floor(ys).astype(int), np.floor(xs).astype(int)
    keep = (iy >= R) & (ix >= R) & (iy - R < view.shape[0]) & (ix - R < view.shape[1])
    if not keep.all():  # patch (partly) outside the tile: drop the nucleus
        ys, xs, region, iy, ix = ys[keep], xs[keep], region[keep], iy[keep], ix[keep]
    if len(ys) == 0:  # no patch inside the tile
        return np.zeros((n_regions, S - 1, S - 1), np.float32), np.zeros(
            n_regions, np.float32
        )
    fy, fx = (ys - iy).astype(np.float32), (xs - ix).astype(np.float32)
    P = view[iy - R, ix - R].reshape(len(ys), S * S)
    wn = np.ones(len(ys), np.float32)
    if norm:
        yy, xx = np.mgrid[0:S, 0:S]
        disk = (np.hypot(yy - R, xx - R) <= norm_radius).ravel().astype(np.float32)
        level = P @ disk / disk.sum()
        floor = max(
            np.percentile(level, norm_floor_pct), 1e-6 * np.abs(level).max() + 1e-9
        )
        wn = (1.0 / np.maximum(level, floor)).astype(np.float32)
    Wm = np.zeros((n_regions * 4, len(ys)), np.float32)
    cols = np.arange(len(ys))
    for c, w in enumerate(((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx)):
        Wm[region * 4 + c, cols] = w * wn
    A = (Wm @ P).reshape(n_regions, 4, S, S)
    M = A[:, 0, :-1, :-1] + A[:, 1, :-1, 1:] + A[:, 2, 1:, :-1] + A[:, 3, 1:, 1:]
    counts = np.bincount(region, minlength=n_regions).astype(np.float32)
    wsum = np.bincount(region, weights=wn, minlength=n_regions).astype(np.float32)
    return M / np.maximum(wsum, 1e-12)[:, None, None], counts


def _sym_shift(P):
    """Shift t such that P(x) = S(x - t) with S point-symmetric: half the offset to P rotated by 180 degrees."""
    P = P - P.mean()
    win = np.outer(np.hanning(P.shape[0]), np.hanning(P.shape[1]))
    shift, _, _ = phase_cross_correlation(
        P * win, P[::-1, ::-1] * win, upsample_factor=50, normalization=None
    )
    return shift / 2


def _disp(prm, y, x, shape):
    vy, vx = y - shape[0] / 2, x - shape[1] / 2
    return vy * prm["m"] + prm["ty"], vx * prm["m"] + prm["tx"]


def _fit(y, x, dy, dx, n, shape):
    """Robust (soft-L1) fit of (m, ty, tx) to regional shifts, weighted by the number of nuclei."""
    w = np.sqrt(np.asarray(n, float))

    def resid(p):
        py, px = _disp(dict(ty=p[0], tx=p[1], m=p[2]), y, x, shape)
        return np.concatenate([w * (py - dy), w * (px - dx)])

    p = least_squares(resid, np.zeros(3), loss="soft_l1", f_scale=0.5, x_scale="jac").x
    return dict(ty=float(p[0]), tx=float(p[1]), m=float(p[2]))


def _inside(cen, shape, border):
    return (
        (cen[:, 0] > border)
        & (cen[:, 0] < shape[0] - border)
        & (cen[:, 1] > border)
        & (cen[:, 1] < shape[1] - border)
    )


def method_s(ref_centres, cp, shape, settings):
    """S: (m, ty, tx) and the regional fit rms (px). ``cp``: the channel prepared by :func:`_prep`."""
    ok = _inside(ref_centres, shape, settings["border"])
    cy, cx = ref_centres[ok, 0], ref_centres[ok, 1]
    region = _regions(cy, cx, shape)
    R = settings["patch_radius"]
    prm = dict(m=0.0, ty=0.0, tx=0.0)
    for _ in range(S_ITERATIONS):
        py, px = _disp(prm, cy, cx, shape)
        M, counts = _region_mean_patches(
            cp, cy + py, cx + px, region, R, settings["norm_radius"]
        )
        rs = np.flatnonzero(counts > 0)
        if (
            rs.size == 0
        ):  # every shifted patch left the tile: the tile fails the rms rule
            return prm, float("inf"), int(ok.sum())
        t = np.array([_sym_shift(M[r].astype(np.float64)) for r in rs])
        ry = np.array([cy[region == r].mean() for r in rs])
        rx = np.array([cx[region == r].mean() for r in rs])
        step = _fit(ry, rx, t[:, 0], t[:, 1], counts[rs], shape)
        prm = {k: prm[k] + step[k] for k in prm}
    # regional residuals around the last step's own fit
    fy, fx = _disp(step, ry, rx, shape)
    rms = float(np.sqrt(np.mean((t[:, 0] - fy) ** 2 + (t[:, 1] - fx) ** 2)))
    return prm, rms, int(ok.sum())


def nucleus_edge_radius(ref, centres, shape, settings, n_ang=72, dr=0.5):
    """Mean nucleus radius (px) from the falling edge of the reference channel's mean patch."""
    ok = _inside(centres, shape, settings["border"])
    R = settings["patch_radius"]
    M, _ = _region_mean_patches(
        _prep(ref),
        centres[ok, 0],
        centres[ok, 1],
        np.zeros(ok.sum(), int),
        R,
        settings["norm_radius"],
        norm=False,
    )
    P = M[0].astype(np.float64)
    r_lo, r_hi = settings["radius_search"]
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rad = np.arange(r_lo, r_hi + dr / 2, dr)
    c = (P.shape[0] - 1) / 2
    intensity = ndi.map_coordinates(
        P,
        [
            c + rad[None, :] * np.sin(ang)[:, None],
            c + rad[None, :] * np.cos(ang)[:, None],
        ],
        order=1,
    ).mean(0)
    w = np.clip(-np.gradient(intensity, dr), 0, None)
    return float((w * rad).sum() / max(w.sum(), 1e-12))


def _edge_shift_fn(r_lo, r_hi, n_ang=72, dr=0.5):
    """Shift of a mean patch's centre from the geometry of its main radial edge (first harmonic of r(theta))."""
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rad = np.arange(r_lo, r_hi + dr / 2, dr)
    s, co = np.sin(ang), np.cos(ang)
    A = np.column_stack([np.ones(n_ang), co, s])

    def f(P):
        c = (P.shape[0] - 1) / 2
        intensity = ndi.map_coordinates(
            P, [c + rad[None, :] * s[:, None], c + rad[None, :] * co[:, None]], order=1
        )
        g = np.gradient(intensity, dr, axis=1)
        w = np.clip(np.sign(g.mean()) * g, 0, None)
        r = (w * rad).sum(1) / np.maximum(w.sum(1), 1e-12)
        coef = np.linalg.lstsq(A, r, rcond=None)[0]
        return np.array([coef[2], coef[1]])

    return f


def method_et(ref_centres, cp, shape, settings, start, edge_radius):
    """ET: translation with m fixed at start['m'], from start's translation; returns (ty, tx, last step).

    ``cp``: the channel prepared by :func:`_prep`.
    """
    ok = _inside(ref_centres, shape, settings["border"])
    cy, cx = ref_centres[ok, 0], ref_centres[ok, 1]
    region = _regions(cy, cx, shape)
    shift_fn = _edge_shift_fn(0.6 * edge_radius, 1.25 * edge_radius)
    prm = dict(start)
    step = float("inf")
    for _ in range(ET_ITERATIONS):
        py, px = _disp(prm, cy, cx, shape)
        M, counts = _region_mean_patches(
            cp,
            cy + py,
            cx + px,
            region,
            settings["patch_radius"],
            settings["norm_radius"],
        )
        rs = np.flatnonzero(counts > 0)
        if rs.size == 0:  # every shifted patch left the tile: not converged
            return prm["ty"], prm["tx"], float("inf")
        t = np.array([shift_fn(M[r].astype(np.float64)) for r in rs])
        dt = (t * counts[rs, None]).sum(0) / counts[rs].sum()
        prm["ty"] += float(dt[0])
        prm["tx"] += float(dt[1])
        step = float(np.hypot(*dt))
    return prm["ty"], prm["tx"], step


def estimate_tile(ref: np.ndarray, images: dict, settings: dict) -> dict:
    """Estimate every channel of one tile.

    :param ref: Reference (nuclear) channel, 2-D.
    :param images: Channel key -> 2-D image of the same shape.
    :param settings: :func:`scale_settings`.
    :return: ``dict(n_nuclei, channels={key: dict(m, ty, tx, s_ty, s_tx, rms, et_step, error)})``. ``error`` is
        the reason a tile can't be used ("no nuclei in the reference channel"), else ``None``. Exceptions propagate.
    """
    ref = np.asarray(ref, np.float64)
    shape = ref.shape
    cen = nucleus_centres(ref, settings)
    n_nuclei = int(_inside(cen, shape, settings["border"]).sum())
    out = dict(n_nuclei=n_nuclei, shape=shape, channels={})
    if n_nuclei < N_REGIONS:
        for c in images:
            out["channels"][c] = dict(error="no nuclei in the reference channel")
        return out
    edge_radius = nucleus_edge_radius(ref, cen, shape, settings)
    for c, img in images.items():
        cp = _prep(img)  # shared by S and ET
        s_prm, rms, _ = method_s(cen, cp, shape, settings)
        ty, tx, step = method_et(cen, cp, shape, settings, s_prm, edge_radius)
        out["channels"][c] = dict(
            m=s_prm["m"],
            ty=ty,
            tx=tx,
            s_ty=s_prm["ty"],
            s_tx=s_prm["tx"],
            rms=rms,
            et_step=step,
            error=None,
        )
    return out


# ---- tile sampling --------------------------------------------------------------------------------------------------


def grid_depth(positions: np.ndarray) -> np.ndarray | None:
    """Chessboard distance (tiles) of each tile to the nearest unimaged grid cell, or None if not a grid."""

    def snap(v):
        u = np.unique(np.round(v, 3))
        d = np.diff(u)
        big = d[d > 0.25 * d.max()] if len(d) else d
        step = np.median(big) if len(big) else 1.0
        return np.round((v - v.min()) / step).astype(int)

    iy, ix = snap(positions[:, 0]), snap(positions[:, 1])
    if len(set(zip(iy.tolist(), ix.tolist()))) != len(positions):
        return None
    occ = np.zeros((iy.max() + 3, ix.max() + 3), bool)
    occ[iy + 1, ix + 1] = True
    return ndi.distance_transform_cdt(occ, metric="chessboard")[iy + 1, ix + 1]


def tile_pool(positions: np.ndarray, n_tiles: int = N_TILES) -> tuple[np.ndarray, str]:
    """Tiles that may be sampled, and the rule used.

    Wells with at most ``n_tiles`` tiles use all tiles. Larger wells use the 80% of tiles closest to the well centre
    that are also at least 3 tiles from the edge of the imaged area; if that leaves fewer than ``n_tiles``, at least
    2 tiles from the edge, then the 80% rule alone.
    """
    n = len(positions)
    if n <= n_tiles:
        return np.ones(n, bool), "all tiles (well has <= %d tiles)" % n_tiles
    d = np.hypot(*(positions - positions.mean(axis=0)).T)
    rank = np.zeros(n, bool)
    rank[np.argsort(d, kind="stable")[: int(INTERIOR_FRACTION * n)]] = True
    depth = grid_depth(positions)
    if depth is None:
        logger.warning(
            "Tile positions don't form a grid: sampling uses the distance rule alone."
        )
        return rank, "80% closest to the well centre (positions not on a grid)"
    for k in INTERIOR_DEPTHS:
        pool = rank if k is None else rank & (depth >= k)
        rule = "80% closest to the well centre" + (
            "" if k is None else f", >= {k} tiles from the edge"
        )
        if pool.sum() >= n_tiles or k is None:
            return pool, rule
    raise AssertionError  # unreachable


def sampling_plan(
    positions: np.ndarray, pool: np.ndarray, n_tiles: int = N_TILES, seed: int = 0
) -> dict:
    """Channel-independent draw order: per quadrant, the spaced first draws, then the replacement order."""
    rng = np.random.default_rng(seed)
    centre = positions.mean(axis=0)
    angle = np.mod(
        np.arctan2(positions[:, 0] - centre[0], positions[:, 1] - centre[1]), 2 * np.pi
    )
    quadrant = np.minimum(
        (angle / (2 * np.pi) * N_QUADRANTS).astype(int), N_QUADRANTS - 1
    )
    quota = [
        n_tiles // N_QUADRANTS + (1 if q < n_tiles % N_QUADRANTS else 0)
        for q in range(N_QUADRANTS)
    ]
    all_tiles = pool.sum() <= n_tiles
    steps = np.diff(np.unique(np.round(positions[:, 1], 3)))
    step = float(np.median(steps[steps > 0.25 * steps.max()])) if len(steps) else 1.0
    spacing = SPACING_FACTOR * math.sqrt(pool.sum() / n_tiles) * step
    first, rest = {}, {}
    drawn: list[int] = []
    order = {
        q: list(rng.permutation(np.flatnonzero(pool & (quadrant == q))))
        for q in range(N_QUADRANTS)
    }
    for q in range(N_QUADRANTS):
        first[q], rest[q] = [], []
    if all_tiles:
        for q in range(N_QUADRANTS):
            first[q] = order[q]
        return dict(
            quadrant=quadrant,
            first=first,
            rest=rest,
            quota=quota,
            all_tiles=True,
            spacing=0.0,
        )
    # spaced first draws, quadrants taking turns
    pending = {q: list(order[q]) for q in range(N_QUADRANTS)}
    while any(len(first[q]) < quota[q] and pending[q] for q in range(N_QUADRANTS)):
        for q in range(N_QUADRANTS):
            if len(first[q]) >= quota[q] or not pending[q]:
                continue
            pick = 0
            if drawn:
                cand = np.array(pending[q])
                d = np.hypot(
                    positions[cand, 0][:, None] - positions[drawn, 0][None, :],
                    positions[cand, 1][:, None] - positions[drawn, 1][None, :],
                ).min(1)
                far = np.flatnonzero(d >= spacing)
                pick = int(far[0]) if len(far) else 0
            i = pending[q].pop(pick)
            first[q].append(int(i))
            drawn.append(int(i))
    for q in range(N_QUADRANTS):
        rest[q] = [int(i) for i in pending[q]]  # replacement order: random, no spacing
    return dict(
        quadrant=quadrant,
        first=first,
        rest=rest,
        quota=quota,
        all_tiles=False,
        spacing=spacing,
    )


def _replay(plan: dict, valid_of: Callable[[int], bool | None]):
    """One channel's sampling, given validity per tile (None = not evaluated yet).

    Deterministic: quadrants consume their first draws, then their replacement order, until their quota is met, they
    reach QUADRANT_CAP tries or run out of tiles. Unfilled slots of closed quadrants go to the open ones (in quadrant
    order), which continue. Returns (done, requests, used valid tiles, tried tiles).
    """
    seq = {q: plan["first"][q] + plan["rest"][q] for q in range(N_QUADRANTS)}
    quota = list(plan["quota"])
    pos = [0] * N_QUADRANTS
    valid = {q: [] for q in range(N_QUADRANTS)}
    tried = {q: [] for q in range(N_QUADRANTS)}
    closed = [False] * N_QUADRANTS
    requests: list[int] = []
    cap = None if plan["all_tiles"] else QUADRANT_CAP
    if plan["all_tiles"]:  # every tile is used; no replacements
        for q in range(N_QUADRANTS):
            for i in seq[q]:
                v = valid_of(i)
                if v is None:
                    requests.append(i)
                else:
                    tried[q].append(i)
                    if v:
                        valid[q].append(i)
        return not requests, requests, valid, tried
    while True:
        changed = False
        for q in range(N_QUADRANTS):
            if closed[q]:
                continue
            while (
                len(valid[q]) < quota[q]
                and pos[q] < len(seq[q])
                and (cap is None or len(tried[q]) < cap)
            ):
                i = seq[q][pos[q]]
                v = valid_of(i)
                if (
                    v is None
                ):  # request enough tiles to fill the quadrant's remaining need
                    need = quota[q] - len(valid[q])
                    room = (cap - len(tried[q])) if cap is not None else need
                    requests.extend(seq[q][pos[q] : pos[q] + max(1, min(need, room))])
                    break
                pos[q] += 1
                tried[q].append(i)
                if v:
                    valid[q].append(i)
            else:
                if (
                    len(valid[q]) < quota[q]
                ):  # cap reached or no tiles left: close, hand slots on
                    closed[q] = True
                    deficit = quota[q] - len(valid[q])
                    quota[q] = len(valid[q])
                    open_q = [p for p in range(N_QUADRANTS) if not closed[p]]
                    for k in range(deficit):
                        if open_q:
                            quota[open_q[k % len(open_q)]] += 1
                    changed = True
        if requests:
            return False, sorted(set(requests)), valid, tried
        if not changed:
            return True, [], valid, tried


def _corner_field(m, ty, tx, shape):
    """Largest correction magnitude over the four tile corners (px)."""
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * (np.array(shape) / 2)
    return float(np.max(np.hypot(m * corners[:, 0] + ty, m * corners[:, 1] + tx)))


def _summarise(key, tiles, valid, tried, quadrant, shape, rng):
    used = sorted(i for q in valid for i in valid[q])
    n_tried = sum(len(v) for v in tried.values())
    res = dict(
        n_tried=n_tried,
        n_valid=len(used),
        valid_fraction=len(used) / n_tried if n_tried else 0.0,
        tiles_used=[int(i) for i in used],
        valid_per_quadrant=[len(valid[q]) for q in range(N_QUADRANTS)],
        tried_per_quadrant=[len(tried[q]) for q in range(N_QUADRANTS)],
        m=None,
        ty=None,
        tx=None,
        ci95_corner_px=None,
        corner_px=None,
    )
    if used:
        v = np.array(
            [[tiles[i]["channels"][key][k] for k in ("m", "ty", "tx")] for i in used]
        )
        med = np.median(v, axis=0)
        res.update(
            m=float(med[0]),
            ty=float(med[1]),
            tx=float(med[2]),
            corner_px=_corner_field(*med, shape),
        )
        boot = [
            _corner_field(
                *(np.median(v[rng.integers(0, len(v), len(v))], axis=0) - med), shape
            )
            for _ in range(N_BOOTSTRAP)
        ]
        res["ci95_corner_px"] = float(np.percentile(boot, 95))
    failures = []  # short reasons, for the record and the error message
    if res["n_valid"] < CHECKS["min_valid"]:
        failures.append(
            f"{res['n_valid']} valid tiles (at least {CHECKS['min_valid']} required)"
        )
    if res["valid_fraction"] < CHECKS["min_valid_fraction"]:
        failures.append(
            f"{res['n_valid']} of {n_tried} tiles valid ({res['valid_fraction']:.0%}; at least "
            f"{CHECKS['min_valid_fraction']:.0%} required)"
        )
    if (
        res["ci95_corner_px"] is not None
        and res["ci95_corner_px"] > CHECKS["max_ci95_corner_px"]
    ):
        failures.append(
            f"95% uncertainty {res['ci95_corner_px']:.2f} px at the tile corner (at most "
            f"{CHECKS['max_ci95_corner_px']:g} px allowed)"
        )
    res["implausible"] = (
        res["corner_px"] is not None and res["corner_px"] > CHECKS["max_corner_px"]
    )
    if res["implausible"]:
        failures.append(
            f"implausibly large correction (>{CHECKS['max_corner_px']:g} px); pre-calculated parameters are required"
        )

    res["passed"] = not failures
    res["failures"] = failures
    return res


def estimate_well(
    read_tile: Callable[[int, Sequence], tuple[np.ndarray, dict]],
    positions: np.ndarray,
    channels: Sequence,
    pixel_size_um: float | None = None,
    seed: int = 0,
) -> dict:
    """Estimate the alignment of ``channels`` to the reference channel from one well's tiles.

    :param read_tile: ``read_tile(i, channels) -> (reference image, {channel: image})`` for tile index ``i``; called
        with an empty channel list when only the reference is needed. Images are 2-D arrays of one tile shape.
    :param positions: (N, 2) tile positions (y, x), any unit (only the geometry is used), e.g. stage positions (um).
    :param channels: Channel keys to estimate (passed to ``read_tile``).
    :param pixel_size_um: Pixel size, to scale the minimum nucleus area; ``None`` scales it with nucleus size.
    :param seed: Random seed for the tile sampling and the bootstrap.
    :return: ``dict(channels={key: summary}, settings, pool_rule, n_pool, n_tiles, ...)``; each summary has
        ``m, ty, tx``, the quality values, ``passed`` and ``failures``. Use :func:`raise_for_failures` to stop on
        failures.

    Tiles are read and measured in parallel with ``dask.compute``, using the current dask scheduler.
    """
    positions = np.asarray(positions, float)
    channels = list(channels)
    pool, rule = tile_pool(positions)
    plan = sampling_plan(positions, pool, seed=seed)
    first_order = []  # first draws, quadrants interleaved
    for k in range(max(len(v) for v in plan["first"].values())):
        first_order += [
            plan["first"][q][k] for q in range(N_QUADRANTS) if k < len(plan["first"][q])
        ]
    if not first_order:
        raise ChannelAlignmentEstimationError(
            "The well has no tiles to estimate the channel alignment from."
        )

    def radius(i):
        return nucleus_radius(read_tile(i, [])[0])

    radii = dask.compute(
        *[dask.delayed(radius)(i) for i in first_order[:N_SCALE_TILES]]
    )
    radii = [r for r in radii if np.isfinite(r)]
    if not radii:
        raise ChannelAlignmentEstimationError(
            "The reference channel has no detectable nuclei in this well, so the channel alignment can't be "
            "estimated. Check --align-channel (it must be a nuclear stain such as DAPI), or give pre-calculated "
            "parameters with --alignment-correction-params."
        )
    settings = scale_settings(float(np.median(radii)), pixel_size_um)

    tiles: dict[int, dict] = {}

    def evaluate(batch, keys):
        def one(i):
            ref, imgs = read_tile(i, keys)
            return estimate_tile(ref, imgs, settings)

        with ProgressBar():
            results = dask.compute(*[dask.delayed(one)(i) for i in batch])
        for i, r in zip(batch, results):
            if i in tiles:  # evaluated before for other channels: add these
                tiles[i]["channels"].update(r["channels"])
            else:
                tiles[i] = r

    evaluate(first_order, channels)  # every channel needs the first draws
    shape = tiles[first_order[0]]["shape"]
    floor = NUCLEI_FLOOR * float(np.median([tiles[i]["n_nuclei"] for i in first_order]))

    def valid_of(key):
        def f(i):
            if i not in tiles or key not in tiles[i]["channels"]:
                return None
            r = tiles[i]["channels"][key]
            return (
                r.get("error") is None
                and r["rms"] < MAX_REGION_RMS
                and r["et_step"] < ET_STEP_TOL
                and tiles[i]["n_nuclei"] >= floor
            )

        return f

    while True:
        requests: set[int] = set()
        active = []
        for key in channels:
            done, req, _, _ = _replay(plan, valid_of(key))
            if not done:
                active.append(key)
                requests.update(req)
        if not requests:
            break
        # a tile that has to be read anyway is evaluated for every channel that still needs tiles
        evaluate(sorted(requests), active)

    rng = np.random.default_rng(seed)
    summaries = {}
    for key in channels:
        _, _, valid, tried = _replay(plan, valid_of(key))
        summaries[key] = _summarise(
            key, tiles, valid, tried, plan["quadrant"], shape, rng
        )
    return dict(
        channels=summaries,
        settings={
            k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()
        },
        pool_rule=rule,
        n_pool=int(pool.sum()),
        n_tiles=len(positions),
        n_tiles_read=len(tiles),
        nuclei_floor=floor,
        seed=seed,
        checks=dict(CHECKS),
    )


def raise_for_failures(
    result: dict, context: str = "", names: dict | None = None
) -> None:
    """Raise :class:`ChannelAlignmentEstimationError` with one line per channel that failed its checks.

    :param context: What was estimated, e.g. the well or image key.
    :param names: Optional display name per channel key.
    """
    failed = {k: v for k, v in result["channels"].items() if not v["passed"]}
    if not failed:
        return
    of = f" of {context}" if context else ""
    lines = []
    for k, v in failed.items():
        label = names.get(k, k) if names else k
        if v.get("implausible"):
            lines.append(
                f"Estimated correction for channel {label}{of} is implausibly large; "
                f">{CHECKS['max_corner_px']:g} pixel correction, m = {v['m']:.3e}, ty = {v['ty']:.2f} px, "
                f"tx = {v['tx']:.2f} px. The estimate is not applied; pre-calculated parameters are required to "
                "correct this channel."
            )
        else:
            lines.append(
                f"Alignment estimate for channel {label}{of} failed the quality check: {'; '.join(v['failures'])}. "
                "Give pre-calculated parameters for this channel (--alignment-correction-params) or leave it out of "
                "--alignment-correction-channels."
            )
    raise ChannelAlignmentEstimationError("\n".join(lines))
