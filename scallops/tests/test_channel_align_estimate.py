import json
import subprocess

import numpy as np
import pytest
import xarray as xr
from skimage.registration import phase_cross_correlation

import scallops.stitch._channel_align_estimate as E
from scallops.io import read_image
from scallops.zarr_io import _write_zarr_image

M, TY, TX = 0.002, 1.3, -0.8


def _render(shape, centres, radius, brightness, rng):
    """Soft-edged disks (nucleus-like) at (y, x) centres, plus camera-like noise."""
    img = np.zeros(shape, np.float64)
    r = int(np.ceil(radius + 3))
    for (cy, cx), b in zip(centres, brightness):
        y0, x0 = int(np.floor(cy)) - r, int(np.floor(cx)) - r
        ys, xs = (
            slice(max(y0, 0), min(y0 + 2 * r + 2, shape[0])),
            slice(max(x0, 0), min(x0 + 2 * r + 2, shape[1])),
        )
        if ys.stop <= ys.start or xs.stop <= xs.start:  # entirely outside the tile
            continue
        yy, xx = np.mgrid[ys, xs]
        d = np.hypot(yy - cy, xx - cx)
        img[ys, xs] += b * np.clip(radius + 0.5 - d, 0, 1)
    return img * 4000 + 300 + rng.normal(0, 30, shape)


def _synthetic_tile(shape, nuclei, rng, m=M, ty=TY, tx=TX, radius=9.0):
    """Reference channel with nuclei at `nuclei`; channel with the same nuclei at s(P) = c + (1 + m)(P - c) + t, so
    that the alignment (m, ty, tx) maps it back onto the reference."""
    c = np.array(shape) / 2
    moved = c + (1 + m) * (nuclei - c) + np.array([ty, tx])
    brightness = rng.uniform(0.3, 1.0, len(nuclei))
    ref = _render(shape, nuclei, radius, np.ones(len(nuclei)), rng)
    img = _render(shape, moved, radius, brightness, rng)
    return ref, img


def _nuclei(shape, n, rng):
    """n random nucleus centres. Keep coverage realistic (~17%, like the pheno screen's ~20%): heavily overlapping
    nuclei of mixed brightness (~40% coverage) bias S's magnification (+0.25 px at the corner of a 768 px tile)."""
    return rng.uniform(30, np.array(shape) - 30, (n, 2))


def _circle_grid(radius_tiles=22):
    yy, xx = np.mgrid[
        -radius_tiles : radius_tiles + 1, -radius_tiles : radius_tiles + 1
    ]
    keep = np.hypot(yy, xx) <= radius_tiles
    return (
        np.column_stack([yy[keep], xx[keep]]).astype(float) * 700.0
    )  # stage positions, um


def test_scale_settings_reproduce_validated_constants():
    s = E.scale_settings(E.REF_NUCLEUS_RADIUS, E.REF_PIXEL_SIZE)
    assert s["patch_radius"] == 48 and s["norm_radius"] == 20 and s["min_area"] == 150
    assert (
        s["seg_sigma"] == 2
        and s["peak_distance"] == 8
        and s["radius_search"] == (4, 45)
        and s["border"] == 56
    )
    small = E.scale_settings(
        6.0
    )  # e.g. lower magnification: everything halves, full-resolution segmentation
    assert (
        small["patch_radius"] == 24
        and small["downsample"] == 1
        and small["min_area"] == pytest.approx(37.5)
    )


def test_nucleus_radius():
    rng = np.random.default_rng(0)
    ref, _ = _synthetic_tile((512, 512), _nuclei((512, 512), 150, rng), rng)
    assert E.nucleus_radius(ref) == pytest.approx(9.0, abs=1.0)
    assert np.isnan(E.nucleus_radius(np.zeros((64, 64))))


def test_estimate_tile_recovers_known_alignment():
    rng = np.random.default_rng(1)
    shape = (768, 768)
    ref, img = _synthetic_tile(shape, _nuclei(shape, 400, rng), rng)
    settings = E.scale_settings(E.nucleus_radius(ref))
    r = E.estimate_tile(ref, {"ch": img}, settings)["channels"]["ch"]
    assert r["error"] is None and r["rms"] < 1 and r["et_step"] < E.ET_STEP_TOL
    corner = np.hypot(*shape) / 2
    assert abs(r["m"] - M) * corner < 0.1
    assert abs(r["ty"] - TY) < 0.06 and abs(r["tx"] - TX) < 0.06


def test_magnification_needs_three_occupied_regions():
    # one occupied region can't separate m from (ty, tx): the fit is exact (rms 0) whatever m is
    rng = np.random.default_rng(3)
    shape = (768, 768)  # regions of 153.6 px
    settings = E.scale_settings(9.0)
    cp = E._prep(rng.normal(300, 30, shape))

    def rms(centres):
        return E.method_s(np.asarray(centres, float), cp, shape, settings)[1]

    one = rng.uniform(170, 290, (30, 2))  # all in region (1, 1)
    two = np.vstack([one, one + [0, 307]])  # regions (1, 1) and (1, 3)
    row = np.column_stack([rng.uniform(170, 290, 60), rng.uniform(60, 700, 60)])
    assert rms(one) == float("inf") and rms(two) == float("inf")
    assert np.isfinite(
        rms(row)
    )  # one row of regions is enough: m is one value for both axes


def test_region_mean_patches_drops_patches_outside_the_tile():
    rng = np.random.default_rng(3)
    img = rng.random((200, 200)).astype(np.float32)
    R = 10
    inside_y, inside_x = np.array([60.3, 120.7]), np.array([80.2, 140.9])
    # patches beyond the top, left (negative indices would wrap around), bottom and right edges
    out_y, out_x = (
        np.array([5.0, 100.0, 195.0, 100.0]),
        np.array([100.0, 4.0, 100.0, 196.5]),
    )
    M_all, counts = E._region_mean_patches(
        img,
        np.concatenate([inside_y, out_y]),
        np.concatenate([inside_x, out_x]),
        np.zeros(6, int),
        R,
        5.0,
    )
    M_in, _ = E._region_mean_patches(img, inside_y, inside_x, np.zeros(2, int), R, 5.0)
    assert counts[0] == 2
    np.testing.assert_array_equal(M_all, M_in)


def test_tile_pool_rules():
    pos = _circle_grid()
    pool, rule = E.tile_pool(pos)
    assert pool.sum() >= E.N_TILES and ">= 3 tiles from the edge" in rule
    assert not pool[
        np.hypot(*pos.T) > 20 * 700
    ].any()  # the outer rings are never drawn
    small = _circle_grid(4)
    pool, rule = E.tile_pool(small)
    assert len(small) <= E.N_TILES and pool.all() and rule.startswith("all tiles")
    sq = np.column_stack(np.unravel_index(np.arange(144), (12, 12))).astype(
        float
    )  # 12 x 12 square grid
    pool, rule = E.tile_pool(sq)
    assert (
        ">= 3" not in rule
    )  # relaxed, because >= 3 from the edge leaves only 36 tiles


def _plan():
    pos = _circle_grid()
    pool, _ = E.tile_pool(pos)
    return pos, pool, E.sampling_plan(pos, pool, seed=3)


def test_sampling_first_draws_are_spaced_and_balanced():
    pos, pool, plan = _plan()
    first = [i for q in range(4) for i in plan["first"][q]]
    assert [len(plan["first"][q]) for q in range(4)] == [25, 25, 25, 25] and pool[
        first
    ].all()
    d = np.hypot(*(pos[first][:, None, :] - pos[first][None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(len(first), 1)].min() >= plan["spacing"] - 1e-9
    done, req, valid, tried = E._replay(plan, lambda i: True)
    assert (
        done
        and not req
        and [len(valid[q]) for q in range(4)] == [25] * 4
        and sum(map(len, tried.values())) == 100
    )


def test_sampling_replaces_within_quadrant_without_spacing():
    pos, pool, plan = _plan()
    bad = set(plan["first"][0][:5] + plan["first"][2][:3])
    done, _, valid, tried = E._replay(plan, lambda i: i not in bad)
    assert done and [len(valid[q]) for q in range(4)] == [25] * 4
    assert [len(tried[q]) for q in range(4)] == [30, 25, 28, 25]
    assert set(valid[0]) - set(plan["first"][0]) <= set(
        plan["rest"][0][:5]
    )  # replacements: next in the order


def test_sampling_caps_a_bad_quadrant():
    pos, pool, plan = _plan()
    bad_q = 1
    done, _, valid, tried = E._replay(plan, lambda i: plan["quadrant"][i] != bad_q)
    assert done and len(tried[bad_q]) == E.QUADRANT_CAP and len(valid[bad_q]) == 0
    assert sum(len(valid[q]) for q in range(4)) == 100


def test_sampling_requests_until_evaluated():
    pos, pool, plan = _plan()
    known = {}
    rounds = 0
    while True:
        done, req, valid, _ = E._replay(plan, lambda i: known.get(i))
        if done:
            break
        rounds += 1
        for i in req:
            known[i] = plan["quadrant"][i] != 3 or i % 3  # quadrant 3: two thirds valid
    assert sum(len(v) for v in valid.values()) == 100 and rounds <= 10


def _well(n_side, shape=(512, 512)):
    yy, xx = np.mgrid[0:n_side, 0:n_side]
    pos = np.column_stack([yy.ravel(), xx.ravel()]).astype(float) * 500.0
    tiles = {}

    def read_tile(i, channels):
        if i not in tiles:
            r = np.random.default_rng(1000 + i)
            ref, img = _synthetic_tile(shape, _nuclei(shape, 180, r), r)
            tiles[i] = (ref, img, 300 + r.normal(0, 30, shape))
        ref, img, blank = tiles[i]
        return ref, {c: {"CY7": img, "blank": blank}[c] for c in channels}

    return pos, read_tile


def test_estimate_well_small_well_uses_all_tiles_and_fails_blank_channel():
    pos, read_tile = _well(5)
    result = E.estimate_well(
        read_tile, pos, ["CY7", "blank"], pixel_size_um=None, seed=0
    )
    assert result["pool_rule"].startswith("all tiles") and result["n_tiles_read"] == 25
    cy7 = result["channels"]["CY7"]
    assert cy7["passed"] and cy7["n_valid"] >= 20 and cy7["n_tried"] == 25
    corner = np.hypot(512, 512) / 2
    assert (
        abs(cy7["m"] - M) * corner < 0.1
        and abs(cy7["ty"] - TY) < 0.1
        and abs(cy7["tx"] - TX) < 0.1
    )
    assert not result["channels"]["blank"]["passed"]
    with pytest.raises(E.ChannelAlignmentEstimationError, match="blank") as err:
        E.raise_for_failures(result, context="A1")
    assert "failed the quality check" in str(err.value) and "leave it out" in str(
        err.value
    )


def test_exception_on_a_tile_stops_estimation(monkeypatch):
    pos, read_tile = _well(5)

    def fails(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(E, "method_s", fails)
    with pytest.raises(RuntimeError, match="simulated failure"):
        E.estimate_well(read_tile, pos, ["CY7"], seed=0)


def test_tile_without_nuclei_is_invalid_not_an_error():
    pos, read_tile = _well(5)
    blank = {3, 11}

    def read(i, channels):
        ref, images = read_tile(i, channels)
        return (np.full_like(ref, 300.0), images) if i in blank else (ref, images)

    cy7 = E.estimate_well(read, pos, ["CY7"], seed=0)["channels"]["CY7"]
    assert cy7["passed"] and not blank & set(cy7["tiles_used"])


def test_implausible_estimate_requires_parameters():
    tiles = {i: dict(channels={"c": dict(m=0.0, ty=12.0, tx=0.0)}) for i in range(30)}
    valid = {0: list(range(30)), 1: [], 2: [], 3: []}
    res = E._summarise(
        "c", tiles, valid, valid, None, (512, 512), np.random.default_rng(0)
    )
    assert not res["passed"] and any(
        "pre-calculated parameters are required" in f for f in res["failures"]
    )


def test_no_nuclei_in_reference_is_explained():
    pos = _circle_grid(3)

    def read_tile(i, channels):
        z = np.full((256, 256), 300.0)
        return z, {c: z for c in channels}

    with pytest.raises(E.ChannelAlignmentEstimationError, match="no detectable nuclei"):
        E.estimate_well(read_tile, pos, ["c"])


@pytest.mark.io
def test_stitch_estimates_alignment_per_well(tmp_path):
    shape, n_side, step = (512, 512), 5, 448
    rng = np.random.default_rng(5)
    world = rng.uniform(
        0, n_side * step + 64, (n_side * n_side * 140, 2)
    )  # nuclei in well coordinates (px), ~17%
    c = np.array(shape) / 2
    k = 0
    for i in range(n_side):
        for j in range(n_side):
            origin = np.array([i * step, j * step], float)
            local = world - origin
            local = local[(local > -20).all(1) & (local < np.array(shape) + 20).all(1)]
            moved = c + (1 + M) * (local - c) + np.array([TY, TX])
            ref = _render(shape, local, 9.0, np.ones(len(local)), rng)
            img = _render(shape, moved, 9.0, rng.uniform(0.3, 1, len(local)), rng)
            tile = np.stack([ref, img]).clip(0, 65535).astype(np.uint16)
            _write_zarr_image(
                name=None,
                root=tmp_path / "input" / f"A1-{k}.zarr",
                image=xr.DataArray(tile, dims=["c", "y", "x"]),
                metadata=dict(
                    position_y=origin[0],
                    position_x=origin[1],
                    position_x_unit="micrometer",
                    position_y_unit="micrometer",
                    physical_size_y=1,
                    physical_size_x=1,
                    physical_size_y_unit="micrometer",
                    physical_size_x_unit="micrometer",
                ),
                group=None,
            )
            k += 1
    cmd = [
        "scallops",
        "stitch",
        "--images",
        str(tmp_path / "input"),
        "--image-pattern",
        "{well}-{skip}.zarr",
        "--groupby",
        "well",
        "--image-output",
        str(tmp_path / "stitch.zarr"),
        "--report-output",
        str(tmp_path / "stitch"),
        "--no-evaluate",
        "--radial-correction-k",
        "none",
        "--alignment-correction-channels",
        "all",
        "--alignment-correction-channel-order",
        "DAPI",
        "CY7",
    ]
    subprocess.check_call(cmd)
    record = json.loads((tmp_path / "stitch" / "A1-channel-alignment.json").read_text())
    (ch,) = record["channels"]
    assert (
        ch["source"] == "estimated"
        and ch["estimate"]["passed"]
        and ch["estimate"]["n_valid"] >= 20
    )
    assert abs(ch["ty"] - TY) < 0.1 and abs(ch["tx"] - TX) < 0.1
    assert (
        record["estimation"]["pool_rule"].startswith("all tiles")
        and record["params_file"] is None
    )
    fused = read_image(str(tmp_path / "stitch.zarr")).values.astype(float)
    inner = (slice(60, -60), slice(60, -60))
    shift, _, _ = phase_cross_correlation(
        fused[0][inner], fused[1][inner], upsample_factor=20
    )
    assert np.abs(shift).max() < 0.15, shift
