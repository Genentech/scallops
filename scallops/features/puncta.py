"""Custom scallops feature module: puncta detection and feature extraction.

Detection pipeline:
    1. Gaussian highpass filter (skip if sigma=0)
    2. Adaptive threshold: foreground = raw > alpha * local_background
       When sigma=0, uses raw > alpha * mean(raw within ROI).
    3. Size filter on connected components (roi-restricted hole-fill)
    4. Watershed segmentation with peak_local_max seeding (exclude_border=False;
       peak_min_distance >= 1 controls minimum separation) and fallback seeding
    5. Post-hoc small puncta merge: watershed regions smaller than
       min_size_fg_objects are absorbed by their most-touching neighbor,
       tie-broken by the mean intensity of the neighbor's boundary pixels.
       Merges only join touching pieces of one foreground component and every
       component keeps at least one punctum, so PunctaCount >= FgObjectCount.
    6. Saturated-plateau merge: watershed pieces that overlap one flat-topped
       (saturated) plateau, i.e. connected foreground pixels within 1% of the ROI's
       dynamic range of its maximum, become one punctum.

Features computed per cell per channel (33 total: 32 measurements plus the
FlagNoNuclearCentroid quality flag). Pipeline parameters are
encoded in the feature name suffix (e.g. ``Puncta_PunctaCount_Channel3_s10.0_a1.5_m2_d2``).

How this module fits into scallops
----------------------------------
Scallops processes images in tiled blocks. For each block, ``_feature_block()``
in generate.py slices out the block, relabels the cells in it sequentially
(1, 2, 3, ...), and calls every registered feature function with keyword
arguments (unique_labels, unique_labels_original, channel_names, label_image,
label_image_original, intensity_image, offset). ``puncta()`` below is registered in
constants.py and returns a dict of feature name -> array with one value per cell.

``label_features()`` additionally binds ``nuclei_label_image`` (the full-image nuclei
labels) to ``puncta()`` only, when the scallops CLI supplies it (for every ROI type:
nuclei, cell, cytoplasm) and a sample of the image shows the ROI labels share IDs
with the nuclei. No other feature function receives it.

The feature string in the pipeline config (e.g. ``puncta_1,2,3_10.0_1.5_2_2``) is
split on underscores by ``_get_params()`` and its tokens are bound POSITIONALLY to
``puncta()``'s parameters: c, sigma, alpha, min_size_fg_objects, peak_min_distance.
The framework arguments (channel_names, label_image, ...) default to None so that
``_get_params()`` skips them; ``_feature_block()`` supplies them at call time.
Do not reorder the user-tunable parameters: their order is the config-string format.

Spatial anchor (nucleus)
------------------------
The spatial features are measured from each cell's nucleus. For every ROI, ``puncta()``
finds the nucleus with the same label ID in ``nuclei_label_image`` and uses the mean
pixel coordinate of that nucleus as the anchor (for a nucleus in one piece, the same
definition as ``Nuclei_AreaShape_Center`` in the scallops objects table; a same-ID
fragment that does not touch the search window is not included). The nucleus is used only if
it overlaps or 4-touches the ROI; otherwise (no nucleus, a nucleus belonging to a
different cell, or no nuclei labels supplied) the ROI's own centroid is the anchor,
and the ``FlagNoNuclearCentroid`` feature is 1 for that ROI (0 when the nucleus was
used). For nuclei ROIs the nucleus is the ROI itself, so the anchor is unchanged and
the flag is 0 whenever the nuclei labels are supplied.

Coordinate systems
------------------
    1. Full image: the coordinates of ``nuclei_label_image``.
    2. Block: the tile being processed; its position in the full image is ``offset``.
    3. Cell crop: the tight bounding box of one ROI inside the block. Detection and
       intensity features are computed here.
    4. Frame: the cell crop enlarged to also contain the whole nucleus. The spatial
       geometry (anchor, cell boundary, radial fraction map) is computed here so the
       anchor always lies inside the array; results are cropped back to the cell crop.

Feature names
-------------
The parameters are appended to every feature name so that running the module
several times with different parameters yields distinct, self-describing columns:
    Puncta_<Feature>_<Channel>_s<sigma>_a<alpha>_m<min_size>_d<peak_min_distance>
Dots in the suffix are safe because ``_get_params()`` splits on "_" only.
"""

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    minimum_filter,
)
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import convex_hull_image
from skimage.segmentation import watershed

# 4-connected (plus-shaped) structuring element. Used to find a cell's inner
# boundary: ROI pixels with at least one 4-connected neighbor outside the ROI.
_CROSS_FOOTPRINT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# Ring density cutoffs on the rescaled [0, 1] radial fraction map:
# inner [0, 0.45), middle [0.45, 0.75), outer [0.75, 1]. These are value cutoffs,
# not equal-area partitions, so ring areas depend on cell shape and anchor position.
# Chosen to give visually balanced ring widths: 1/3-radius cutoffs leave too little
# area in the inner ring, and equal-area cutoffs make the outer ring too narrow.
_RING_CUTOFF_LO = 0.45
_RING_CUTOFF_HI = 0.75

# Relative margin above alpha x local background a pixel must exceed. On a flat
# signal the float32 normalized convolution can put local_bg up to ~2e-7 (relative)
# below the pixel value, so with alpha = 1.0 flat regions would pass the threshold.
_BG_MARGIN = 1.0 + 1e-6

# Saturated (flat-topped) puncta: pixels within this fraction of the ROI's dynamic
# range (max - median) of its maximum count as part of a plateau. Relative to the
# range, not the maximum, so it scales with the cell's contrast; > 0 so a plateau
# stays intact after flatfield correction makes its pixels slightly unequal.
_SATURATION_TOL = 0.01

# Minimum fraction of sampled ROI labels whose same-ID nucleus overlaps or touches them
# for the nuclei labels to be used at all (see nuclei_label_match_fraction). ROI labels
# derived from the nuclei score ~1; independently numbered labels (cellpose cells) ~0.
NUCLEI_MATCH_MIN_FRACTION = 0.5


# The 33 output features. FgObjectCount (connected blobs before watershed) and
# PunctaCount (individual puncta after watershed + merge) differ on purpose.
# All per-punctum statistics use the post-watershed labels and the RAW image.
#   Counts:                  FgObjectCount, PunctaCount
#   Area:                    TotalFgArea, TotalFgAreaFraction, Mean/Std/MedianPunctaSize
#   Foreground intensity:    TotalFg, MeanFg, StdFg, Pct25/50/75FgIntensity
#   Background intensity:    MeanBg, StdBg, MedianBg, RatioFgBgMeanIntensities
#   Per-punctum intensity:   Mean/Std/MedianPunctaMeans, Mean/Std/MedianPunctaMaxes
#   Spatial (see _extract_cell_features): radial position, extent, polarization,
#                            and ring densities
#   Quality flag:            FlagNoNuclearCentroid (set in puncta(), not measured):
#                            1 if the spatial features are NOT anchored on the cell's
#                            nucleus (ROI-centroid fallback), else 0
_FEATURE_KEYS = [
    "FgObjectCount",
    "PunctaCount",
    "TotalFgArea",
    "TotalFgAreaFraction",
    "MeanPunctaSize",
    "StdPunctaSize",
    "MedianPunctaSize",
    "TotalFgIntensity",
    "MeanFgIntensity",
    "StdFgIntensity",
    "Pct25FgIntensity",
    "Pct50FgIntensity",
    "Pct75FgIntensity",
    "MeanBgIntensity",
    "StdBgIntensity",
    "MedianBgIntensity",
    "RatioFgBgMeanIntensities",
    "MeanPunctaMeans",
    "StdPunctaMeans",
    "MedianPunctaMeans",
    "MeanPunctaMaxes",
    "StdPunctaMaxes",
    "MedianPunctaMaxes",
    "MeanFgRadialPosPx",
    "FgRadialEnrichment",
    "MeanFgExtentPx",
    "FgExtentEnrichment",
    "FgPolarizationPx",
    "FgPolarizationNorm",
    "DensityRingInner",
    "DensityRingMiddle",
    "DensityRingOuter",
    "FlagNoNuclearCentroid",
]

# Precomputed at module load — used in the no-puncta early return path.
_ZEROS_SPATIAL = {
    k: 0.0
    for k in _FEATURE_KEYS
    if "Radial" in k or "Extent" in k or "Polarization" in k or "Density" in k
}


def _padded_edt(mask):
    """Euclidean distance to the nearest False pixel, treating the outside as False.

    distance_transform_edt does not assume background beyond the array edge. A cell
    ROI fills its tight bounding box on every side, so edge pixels would otherwise
    measure distance to a far-away in-array False pixel (e.g. at a corner), inflating
    the distance and pulling the radial fraction below 1.0 at the boundary. Padding
    with one False pixel and cropping back makes every boundary pixel distance 1.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)
    return dist[1:-1, 1:-1]


def _geometry_mask(roi, anchor_centroid, nucleus=None):
    """Return the cell shape whose outer edge the spatial features use as the boundary.

    Holes are always filled so only the OUTER cell edge counts as boundary (otherwise
    cytoplasm pixels next to the nucleus would look "close to the boundary").

    1. Anchor outside the ROI and a nucleus given (cytoplasm ROIs): the hole-filled
       union of ROI and nucleus, i.e. the whole cell. Unlike filling the ROI alone this
       also closes a crescent-shaped cytoplasm whose nucleus touches the cell edge.
    2. Otherwise: the hole-filled ROI.
    3. If the anchor is not inside the shape chosen in 1 or 2 (e.g. the centroid of a
       non-convex ROI, or the centroid of an oddly concave nucleus): the convex hull of
       that shape plus the anchor pixel, so the anchor lies inside the shape.

    The anchor must lie inside the array; puncta() guarantees this by computing the
    geometry in a frame that contains the whole nucleus.
    """
    ay = int(round(anchor_centroid[0]))
    ax = int(round(anchor_centroid[1]))
    shape = binary_fill_holes(roi)
    if nucleus is not None and not roi[ay, ax]:
        shape = binary_fill_holes(roi | nucleus)
    if shape[ay, ax]:
        return shape
    combined = shape.copy()
    combined[ay, ax] = True
    # convex_hull_image needs >= 3 non-collinear pixels
    if combined.sum() < 3:
        return combined
    return convex_hull_image(combined)


def _compute_radial_frac_map(roi, anchor_centroid, geometry_mask=None):
    """Compute rescaled radial fraction map and expected radial distance.

    For each ROI pixel: fraction = d_anchor / (d_anchor + d_boundary), i.e. 0 at the
    anchor and approaching 1 at the cell edge, where d_boundary is the distance to the
    edge of ``geometry_mask`` (default: _geometry_mask(roi, anchor_centroid)). The
    fractions are then rescaled to span [0, 1] within the ROI so the ring cutoffs
    always split the real ROI. This matters for cytoplasm ROIs, where no pixel sits at
    the anchor (the nucleus is excluded) and raw fractions would start around
    0.15-0.25, leaving the inner ring almost empty.

    Also returns expected_radial, the mean anchor distance of all ROI pixels, which
    is the null expectation for FgRadialEnrichment.

    Known limitation: a cytoplasm ROI only ~1 px thick along the cell edge has no
    radial gradient. All fractions are equal, so every pixel is assigned 0.5 (middle
    ring) and DensityRingInner/Outer report 0.0. Those zeros are not informative.

    Returns (frac_map, expected_radial).
    """
    # A 1-2 px ROI has no meaningful gradient. Use the same 0.5 degenerate fallback
    # as below.
    if roi.sum() < 3:
        frac_map = np.zeros(roi.shape, dtype=np.float64)
        frac_map[roi] = 0.5
        return frac_map, 1e-5

    h, w = roi.shape
    if geometry_mask is None:
        geometry_mask = _geometry_mask(roi, anchor_centroid)
    dist_to_boundary = _padded_edt(geometry_mask)

    # Broadcasting two 1-D ranges avoids allocating full (h, w) coordinate grids.
    dy = np.arange(h, dtype=np.float64) - anchor_centroid[0]
    dx = np.arange(w, dtype=np.float64) - anchor_centroid[1]
    dist_from_anchor = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)

    expected_radial = max(float(np.mean(dist_from_anchor[roi])), 1e-5)

    denom = dist_from_anchor + dist_to_boundary
    frac_map = np.zeros_like(denom)
    valid = denom > 0
    frac_map[valid] = dist_from_anchor[valid] / denom[valid]

    # Re-mask holes and everything outside the original ROI.
    frac_map[~roi] = 0

    # Rescale ROI values to [0, 1]; a constant map (no gradient) gets 0.5.
    roi_vals = frac_map[roi]
    fmin, fmax = roi_vals.min(), roi_vals.max()
    if fmax - fmin > 1e-10:
        frac_map[roi] = (roi_vals - fmin) / (fmax - fmin)
    else:
        frac_map[roi] = 0.5

    return frac_map, expected_radial


def _read_label_region(labels, y0, y1, x0, x1):
    """Return ``labels[y0:y1, x0:x1]`` as a numpy array (numpy, zarr, or dask input)."""
    region = labels[y0:y1, x0:x1]
    if hasattr(region, "compute"):
        # dask: compute in the current thread, which is safe inside a running dask task
        region = region.compute(scheduler="synchronous")
    return np.asarray(region)


def _find_nucleus(
    nuclei_label_image, nuclei_block, block_origin, roi_origin, roi, label
):
    """Find the nucleus with ID ``label`` for one ROI, in a frame containing both.

    The search window starts as the ROI's bounding box plus a 1-pixel margin, so a
    nucleus that only 4-touches the ROI (the hole of a cytoplasm ROI) is seen. The
    nucleus is rejected if none of its pixels overlap or 4-touch the ROI: the cell has
    no nucleus, or the same-ID nucleus belongs to another cell. Otherwise the window is
    enlarged on every side where the nucleus reaches the window edge (unless that edge
    is the image edge) until the whole nucleus is inside.

    Pixels are taken from ``nuclei_block`` (the nuclei labels of the current block, read
    once per block) when the window lies inside the block, and read from the full
    ``nuclei_label_image`` only when the window extends beyond the block.

    :param block_origin: (row, col) of ``nuclei_block`` in the full image.
    :param roi_origin: (row, col) of the ROI's bounding box in the full image.
    :return: (roi_frame, nucleus_frame, (row, col) of the ROI's bounding box within the
        frame), or None if there is no usable nucleus.
    """
    full_h, full_w = nuclei_label_image.shape[:2]
    by, bx = block_origin
    bh, bw = nuclei_block.shape
    rh, rw = roi.shape

    def read_window(y0, y1, x0, x1):
        if by <= y0 and y1 <= by + bh and bx <= x0 and x1 <= bx + bw:
            return nuclei_block[y0 - by : y1 - by, x0 - bx : x1 - bx]
        return _read_label_region(nuclei_label_image, y0, y1, x0, x1)

    def roi_in_window(y0, x0, shape):
        frame = np.zeros(shape, dtype=bool)
        frame[roi_origin[0] - y0 : roi_origin[0] - y0 + rh,
              roi_origin[1] - x0 : roi_origin[1] - x0 + rw] = roi  # fmt: skip
        return frame

    y0, x0 = max(roi_origin[0] - 1, 0), max(roi_origin[1] - 1, 0)
    y1, x1 = min(roi_origin[0] + rh + 1, full_h), min(roi_origin[1] + rw + 1, full_w)
    nucleus = read_window(y0, y1, x0, x1) == label
    roi_frame = roi_in_window(y0, x0, nucleus.shape)
    if not (binary_dilation(roi_frame, structure=_CROSS_FOOTPRINT) & nucleus).any():
        return None

    while True:
        grow_top = y0 > 0 and nucleus[0].any()
        grow_bottom = y1 < full_h and nucleus[-1].any()
        grow_left = x0 > 0 and nucleus[:, 0].any()
        grow_right = x1 < full_w and nucleus[:, -1].any()
        if not (grow_top or grow_bottom or grow_left or grow_right):
            break
        # Double the window on each side the nucleus reaches, then re-read.
        step = max(y1 - y0, x1 - x0)
        y0 = max(y0 - step, 0) if grow_top else y0
        y1 = min(y1 + step, full_h) if grow_bottom else y1
        x0 = max(x0 - step, 0) if grow_left else x0
        x1 = min(x1 + step, full_w) if grow_right else x1
        nucleus = read_window(y0, y1, x0, x1) == label
        roi_frame = roi_in_window(y0, x0, nucleus.shape)

    return roi_frame, nucleus, (roi_origin[0] - y0, roi_origin[1] - x0)


def nuclei_label_match_fraction(label_image, nuclei_label_image, region, labels=None):
    """Fraction of ROI labels in ``region`` whose same-ID nucleus overlaps or touches them.

    Cell labels from propagation or watershed segmentation (grown from the nuclei) and
    cytoplasm labels (cells minus nuclei) share IDs with their nuclei and score ~1.
    Cellpose cell labels are numbered independently of the nuclei and score ~0; for
    such images the nuclei must not be used, or cells would be paired with unrelated
    nuclei. Compare against NUCLEI_MATCH_MIN_FRACTION.

    :param region: (row slice, col slice) to sample, in full-image coordinates.
    :param labels: ROI IDs to evaluate (default: every ROI in ``region``). Pass ROIs
        whose bounding box plus a 1-pixel margin lies inside ``region``: a ROI cut by
        the region's edge can have its nucleus outside the region and would be
        counted as unmatched.
    """
    (rows, cols) = region
    rois = _read_label_region(label_image, rows.start, rows.stop, cols.start, cols.stop)
    nuclei = _read_label_region(
        nuclei_label_image, rows.start, rows.stop, cols.start, cols.stop
    )
    ids = np.unique(rois) if labels is None else np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return 0.0
    # Same pixel, or the 4 neighbours (ROI pixel next to a nucleus pixel of its ID).
    pairs = [
        (rois, nuclei),
        (rois[1:], nuclei[:-1]),
        (rois[:-1], nuclei[1:]),
        (rois[:, 1:], nuclei[:, :-1]),
        (rois[:, :-1], nuclei[:, 1:]),
    ]
    matched = np.unique(np.concatenate([a[(a == b) & (a > 0)] for a, b in pairs]))
    return float(np.isin(ids, matched).mean())


def _validate_params(sigma, alpha, min_size_fg_objects, peak_min_distance):
    """Reject invalid user-tunable parameters before any computation starts."""
    for name, val in [
        ("sigma", sigma),
        ("alpha", alpha),
        ("min_size_fg_objects", min_size_fg_objects),
    ]:
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"{name} must be finite, got {val}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    # With sigma > 0 the watershed runs on the highpass image (raw - local_bg, clipped
    # at 0). If alpha < 1 a pixel can pass the threshold with raw < local_bg, i.e. a
    # highpass value of 0. An all-zero foreground has std 0, which would make every
    # pixel a peak and over-segment catastrophically. alpha >= 1 guarantees
    # raw > local_bg for every foreground pixel, so its highpass value is > 0.
    if sigma > 0 and alpha < 1:
        raise ValueError(
            f"alpha must be >= 1 when sigma > 0, got alpha={alpha}. "
            "With highpass filtering, alpha < 1 permits foreground pixels whose "
            "filtered value is zero or negative, definitive over-segmentation. "
            "Foreground formula, foreground = raw > alpha * local_background "
        )
    if min_size_fg_objects < 1:
        raise ValueError(f"min_size_fg_objects must be >= 1, got {min_size_fg_objects}")
    if not isinstance(peak_min_distance, (int, float)) or isinstance(
        peak_min_distance, bool
    ):
        raise TypeError(
            f"peak_min_distance must be a positive integer, got {type(peak_min_distance).__name__}"
        )
    if isinstance(peak_min_distance, float) and not peak_min_distance.is_integer():
        raise ValueError(
            f"peak_min_distance must be a whole number, got {peak_min_distance}"
        )
    if peak_min_distance < 1:
        raise ValueError(f"peak_min_distance must be >= 1, got {peak_min_distance}")


def _highpass_filter(image, roi, sigma, blurred_mask=None):
    """Remove slowly varying background by normalized Gaussian convolution.

        local_bg = gaussian(image * roi) / gaussian(roi)
        filtered = max(image * roi - local_bg, 0)

    Dividing by gaussian(roi) corrects for the kernel extending past the cell edge.
    Without it, pixels near the edge would get an artificially low background (the
    kernel averages in zeros from outside the ROI) and be over-called as foreground.

    Returns (filtered, local_bg). The foreground threshold compares the RAW image to
    local_bg; ``filtered`` is only used to find peaks for watershed seeds.
    ``blurred_mask`` (the clipped gaussian of the ROI) depends only on the cell, so
    puncta() computes it once and reuses it for every channel.
    """
    img = image.astype(np.float32)
    mask = roi.astype(np.float32)
    img_masked = img * mask
    blurred_img = gaussian_filter(img_masked, sigma=sigma, mode="constant", cval=0)
    if blurred_mask is None:
        blurred_mask = gaussian_filter(mask, sigma=sigma, mode="constant", cval=0)
        np.clip(blurred_mask, 1e-10, None, out=blurred_mask)
    local_bg = blurred_img / blurred_mask
    filtered = img_masked - local_bg
    np.maximum(filtered, 0, out=filtered)
    filtered[~roi] = 0
    local_bg[~roi] = 0
    return filtered, local_bg


def _size_filter(fg_mask, min_size, roi):
    """Drop foreground components smaller than min_size and fill tiny interior holes.

    Returns (filtered_mask, filtered_labeled, n_objects). filtered_labeled is passed
    to _watershed_segment so it does not have to relabel, and n_objects (the number
    of connected components, 4-connectivity) becomes FgObjectCount.

    Filling a hole can join two objects separated by a 1-2 px gap, so n_objects is
    counted after filling.
    """
    labeled_raw = label(fg_mask, connectivity=1)
    sizes = np.bincount(labeled_raw.ravel())  # index 0 is background
    keep = np.zeros(len(sizes), dtype=bool)
    keep[1:] = sizes[1:] >= min_size
    out = keep[labeled_raw]
    # Fill small interior holes: use binary_fill_holes so only regions completely
    # enclosed by foreground (not merely bounded by the ROI edge) are candidates.
    # This prevents diagonal spur pixels — which are isolated within the ROI but
    # surrounded by background outside it — from being wrongly filled.
    # (Labeling "roi & ~out" instead would treat such a spur as a size-1 hole, which
    # is <= 3 px, and so always turn it into foreground regardless of its intensity.)
    truly_interior = binary_fill_holes(out) & ~out & roi
    hl = label(truly_interior, connectivity=1)
    hs = np.bincount(hl.ravel())
    fill = np.zeros(len(hs), dtype=bool)
    fill[1:] = hs[1:] <= 3
    filtered = out | fill[hl]
    if filtered.any():
        filtered_labeled = label(filtered, connectivity=1)
        n_objects = int(filtered_labeled.max())
    else:
        filtered_labeled = np.zeros_like(fg_mask, dtype=int)
        n_objects = 0
    return filtered, filtered_labeled, n_objects


def _merge_small_puncta(seg_labels, working_img, min_size):
    """Merge watershed regions smaller than min_size into their best-fitting neighbor.

    Processing order: ascending (size, n_distinct_neighbors) — most constrained
    merges first. Target neighbor: most shared border pixels. Tie-break: highest
    mean intensity of the neighbor's boundary pixels. Labels are reindexed to
    close gaps only when at least one merge occurred.
    """
    sizes = np.bincount(seg_labels.ravel())
    small_mask = np.zeros(len(sizes), dtype=bool)
    small_mask[1:] = (sizes[1:] > 0) & (sizes[1:] < min_size)
    small_ids = np.where(small_mask)[0]

    if len(small_ids) == 0:
        return seg_labels

    H, W = seg_labels.shape
    n_labels = len(sizes)

    # Bounding box (row0, row1, col0, col1) of every label, kept up to date as labels
    # merge. Each small region is examined inside its own box plus a 1-pixel margin
    # instead of the whole crop, so the cost does not grow with the crop size.
    boxes = {}
    for i, sl in enumerate(find_objects(seg_labels), start=1):
        if sl is not None:
            boxes[i] = [sl[0].start, sl[0].stop, sl[1].start, sl[1].stop]

    def window(lid):
        y0, y1, x0, x1 = boxes[lid]
        return max(y0 - 1, 0), min(y1 + 1, H), max(x0 - 1, 0), min(x1 + 1, W)

    # Number of distinct neighboring puncta per small region (sort key below). The
    # window is zero-padded, so a region at the crop edge sees no wrapped-around
    # neighbours from the opposite edge.
    n_distinct = np.zeros(n_labels, dtype=int)
    for lid in small_ids:
        y0, y1, x0, x1 = window(lid)
        sub = seg_labels[y0:y1, x0:x1]
        own = sub == lid
        p = np.pad(sub, 1)
        bl = np.concatenate(
            [p[2:, 1:-1][own], p[:-2, 1:-1][own], p[1:-1, 2:][own], p[1:-1, :-2][own]]
        )
        bl = bl[(bl > 0) & (bl != lid)]
        n_distinct[lid] = len(np.unique(bl))

    sorted_ids = sorted(small_ids, key=lambda x: (int(sizes[x]), int(n_distinct[x])))

    any_merged = False
    _dy = np.array([-1, 1, 0, 0])
    _dx = np.array([0, 0, -1, 1])

    for lid in sorted_ids:
        if sizes[lid] == 0:
            continue  # already absorbed by an earlier merge
        if sizes[lid] >= min_size:
            continue  # grew large after being targeted by an earlier merge

        y0, y1, x0, x1 = window(lid)
        ys, xs = np.nonzero(seg_labels[y0:y1, x0:x1] == lid)
        ys += y0
        xs += x0

        # Vectorized 4-connectivity neighbor collection from current seg_labels
        ny = (ys[:, None] + _dy).ravel()
        nx = (xs[:, None] + _dx).ravel()
        in_bounds = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        ny, nx = ny[in_bounds], nx[in_bounds]

        border_labels = seg_labels[ny, nx]
        border_intensities = working_img[ny, nx]

        # Exclude background (0) and self-borders (same label as lid). Self-borders
        # arise from 4-adjacent pixels inside lid; keeping them would make lid its
        # own best target, "merging" into itself and zeroing its size entry.
        fg_border = (border_labels > 0) & (border_labels != lid)
        border_labels = border_labels[fg_border]
        border_intensities = border_intensities[fg_border]

        if len(border_labels) == 0:
            continue  # isolated — no foreground neighbor; leave in place

        counts = np.bincount(border_labels, minlength=n_labels)
        intensity_sums = np.bincount(
            border_labels,
            weights=border_intensities.astype(np.float64),
            minlength=n_labels,
        )

        # Primary: most shared border pixels. Tie-break: brightest boundary mean.
        max_count = int(counts.max())
        tied = np.where(counts == max_count)[0]
        tied = tied[tied > 0]  # exclude background (label 0)

        if len(tied) == 1:
            best = int(tied[0])
        else:
            mean_int = np.where(
                counts[tied] > 0,
                intensity_sums[tied] / counts[tied],
                0.0,
            )
            best = int(tied[np.argmax(mean_int)])

        seg_labels[ys, xs] = best
        sizes[best] += sizes[lid]  # additive update; no recount needed
        b, m = boxes[best], boxes[lid]
        boxes[best] = [
            min(b[0], m[0]),
            max(b[1], m[1]),
            min(b[2], m[2]),
            max(b[3], m[3]),
        ]
        sizes[lid] = 0
        any_merged = True

    if not any_merged:
        return seg_labels

    # Reindex: close gaps left by merged-away labels
    live = np.where(sizes[1:] > 0)[0] + 1
    if len(live) < int(seg_labels.max()):
        mapping = np.zeros(int(seg_labels.max()) + 1, dtype=int)
        for new_id, old_id in enumerate(live, start=1):
            mapping[old_id] = new_id
        seg_labels = mapping[seg_labels]

    return seg_labels


def _watershed_segment(
    working_img, fg_mask, fg_labeled, n_components, min_distance, min_size
):
    """Split touching foreground blobs into individual puncta, then merge tiny pieces.

    1. Seeds: local maxima of the highpass image inside the foreground, at least
       min_distance apart and above one std of the foreground values (so uniform
       noise does not seed).
    2. Fallback: every connected component with no seed gets one at its first pixel,
       so no component is lost. Where a lone seed sits does not matter, since a single
       seed floods its whole component.
    3. Watershed on the inverted image (bright = basin bottom), restricted to
       fg_mask so puncta never grow into background.
    4. _merge_small_puncta absorbs regions smaller than min_size.

    fg_labeled / n_components come from _size_filter to avoid relabeling here.
    """
    if not fg_mask.any():
        return np.zeros_like(fg_mask, dtype=int)

    # Peak threshold: one std of the foreground values (len == 1 only for a lone pixel).
    pos_px = working_img[fg_mask]
    std_filt = np.std(pos_px) if len(pos_px) > 1 else 1.0
    # std == 0 (all foreground values identical, e.g. quantized or flat data) would
    # make threshold_abs = 0 and every pixel a peak. Raise it above the maximum so
    # no peak qualifies and the fallback below seeds each component exactly once.
    if std_filt == 0.0:
        std_filt = float(pos_px.max()) + 1.0

    coords = peak_local_max(
        working_img,
        min_distance=min_distance,
        threshold_abs=std_filt,
        labels=fg_mask.astype(int),
        exclude_border=False,
    )
    peaks_mask = np.zeros_like(fg_mask, dtype=bool)
    if len(coords) > 0:
        peaks_mask[coords[:, 0], coords[:, 1]] = True

    # Only iterate over components that have no seed yet (a set lookup is cheaper
    # than testing every component's pixels against peaks_mask).
    seeded = set(fg_labeled[peaks_mask].tolist()) if peaks_mask.any() else set()
    for comp_id in range(1, n_components + 1):
        if comp_id not in seeded:
            # argmax of a boolean array = flat index of the first True pixel
            flat_idx = np.argmax(fg_labeled == comp_id)
            peaks_mask.flat[flat_idx] = True

    if not peaks_mask.any():
        return fg_labeled

    markers = label(peaks_mask, connectivity=1)
    # Invert so the brightest pixel is the lowest "elevation" in each basin.
    seg_labels = watershed(
        working_img.max() - working_img, markers=markers, mask=fg_mask
    )

    return _merge_small_puncta(seg_labels, working_img, min_size)


def _merge_saturated_plateaus(seg_labels, raw, roi):
    """Join watershed pieces that share one saturated (flat-topped) plateau.

    Camera saturation clips a bright punctum to a flat top, and watershed then seeds
    several times across it; each piece is >= min_size, so the small-piece merge does
    not rejoin them. A plateau is a 4-connected group of >= 2 foreground pixels within
    _SATURATION_TOL x (max - median) of the ROI's maximum; all pieces that overlap one
    plateau become one punctum. Seeding is unchanged. Only foreground pixels form
    plateaus, so a merge never joins two foreground components and
    PunctaCount >= FgObjectCount still holds.
    """
    if seg_labels.max() < 2:
        return seg_labels
    values = raw[roi]
    top = values.max()
    tol = _SATURATION_TOL * (top - np.median(values))
    near_top = (raw >= top - tol) & (seg_labels > 0)
    if near_top.sum() < 2:
        return seg_labels
    plateaus = label(near_top, connectivity=1)
    merged = False
    for pid, sl in enumerate(find_objects(plateaus), start=1):
        if sl is None:
            continue
        # A plateau that lies in one piece (e.g. a single brightest pixel) leaves
        # len(ids) == 1, so nothing is merged.
        ids = np.unique(seg_labels[sl][plateaus[sl] == pid])
        if len(ids) > 1:
            seg_labels[np.isin(seg_labels, ids)] = ids[0]
            merged = True
    if not merged:
        return seg_labels
    live = np.unique(seg_labels[seg_labels > 0])
    lut = np.zeros(int(seg_labels.max()) + 1, dtype=seg_labels.dtype)
    lut[live] = np.arange(1, len(live) + 1)
    return lut[seg_labels]


def _detect_puncta(
    raw, roi, sigma, alpha, min_size_fg_objects, peak_min_distance, blurred_mask=None
):
    """Run the detection pipeline on a single cell crop.

    Steps: highpass -> adaptive threshold -> size filter -> watershed -> merge
    -> saturated-plateau merge.
    Returns (seg_labels, n_fg_objects).

    seg_labels is an instance label image (0 = background, 1..N = puncta), where
    N = PunctaCount. n_fg_objects is the connected-component count after the size
    filter but BEFORE watershed (FgObjectCount). Watershed only splits components
    and the merge only rejoins touching pieces of one component, so
    PunctaCount >= FgObjectCount.
    """
    zeros = np.zeros_like(roi, dtype=int)

    # --- Step 1: Gaussian highpass (skip if sigma == 0) ---
    # working_img is what watershed climbs: highpass values, or masked raw if sigma == 0.
    if sigma > 0:
        filtered, local_bg = _highpass_filter(
            raw, roi, sigma, blurred_mask=blurred_mask
        )
        working_img = filtered
    else:
        working_img = raw.astype(np.float32) * roi.astype(np.float32)

    # --- Step 2: Adaptive foreground mask ---
    # Threshold the RAW image against alpha x local background, so alpha is a plain
    # "times brighter than background" ratio (1.5 = 50% above background). With
    # sigma == 0 there is no local estimate; compare against the cell mean instead.
    if sigma > 0:
        # The margin stops float32 rounding putting flat signal above its background.
        fg_mask = (raw > alpha * local_bg * _BG_MARGIN) & roi
    else:
        mean_val = float(np.mean(raw[roi].astype(np.float64)))
        fg_mask = (raw > alpha * mean_val) & roi

    if not fg_mask.any():
        return zeros, 0

    # If the whole ROI is foreground, detection is meaningless (very low alpha with
    # sigma == 0, or a near-zero-background channel where local_bg is clipped to
    # ~1e-10 and any nonzero pixel passes). Report no puncta instead.
    if fg_mask.sum() >= roi.sum():
        return zeros, 0

    # --- Step 3: Size filter ---
    fg_mask, fg_labeled, n_fg_objects = _size_filter(fg_mask, min_size_fg_objects, roi)
    if not fg_mask.any():
        return zeros, 0
    # The size filter fills small holes, which can complete a ROI that was not fully
    # foreground at the check above; detection would be meaningless again.
    if fg_mask.sum() >= roi.sum():
        return zeros, 0

    # --- Step 4: Watershed + post-hoc small puncta merge ---
    # min_size_fg_objects doubles as the post-watershed merge threshold.
    seg_labels = _watershed_segment(
        working_img,
        fg_mask,
        fg_labeled,
        n_fg_objects,
        int(peak_min_distance),
        min_size_fg_objects,
    )
    seg_labels = _merge_saturated_plateaus(seg_labels, raw, roi)

    return seg_labels, n_fg_objects


def _extract_cell_features(
    raw,
    seg_labels,
    roi,
    anchor_centroid,
    cell_centroid,
    expected_radial,
    mean_boundary_dist,
    radial_frac_map,
    n_fg_objects,
    roi_coords,
    cell_area,
    expected_extent,
):
    """Compute per-cell-channel features from instance-segmented puncta using raw intensity.

    All intensity statistics use the RAW image, not the highpass image. Per-cell
    geometry (centroids, expected distances, radial map, ...) is precomputed once in
    puncta() and passed in so it is not recomputed for every channel.

    Spatial reference points:
      - anchor_centroid: nuclear centroid (or ROI centroid fallback); used for radial
        position, polarization, and ring density.
      - cell_centroid: ROI mask centroid; only used for expected_extent, the null
        expectation for the spread of foreground that is uniformly distributed over
        the ROI. This is a property of cell shape, so it is channel-independent.
    """
    n_puncta = int(seg_labels.max())
    combined_fg = seg_labels > 0
    total_area = int(combined_fg.sum())

    roi_pixels = raw[roi].astype(np.float64)

    # No puncta: detection features are zero, but background stats are still valid
    # (computed from the whole ROI) and the spatial features are zero.
    if n_puncta == 0 or total_area == 0:
        return {
            "FgObjectCount": n_fg_objects,
            "PunctaCount": 0,
            "TotalFgArea": 0,
            "TotalFgAreaFraction": 0.0,
            "MeanPunctaSize": 0.0,
            "StdPunctaSize": 0.0,
            "MedianPunctaSize": 0.0,
            "TotalFgIntensity": 0.0,
            "MeanFgIntensity": 0.0,
            "StdFgIntensity": 0.0,
            "Pct25FgIntensity": 0.0,
            "Pct50FgIntensity": 0.0,
            "Pct75FgIntensity": 0.0,
            "MeanBgIntensity": float(roi_pixels.mean()) if len(roi_pixels) > 0 else 0.0,
            "StdBgIntensity": float(roi_pixels.std()) if len(roi_pixels) > 1 else 0.0,
            "MedianBgIntensity": float(np.median(roi_pixels))
            if len(roi_pixels) > 0
            else 0.0,
            "RatioFgBgMeanIntensities": 0.0,
            "MeanPunctaMeans": 0.0,
            "StdPunctaMeans": 0.0,
            "MedianPunctaMeans": 0.0,
            "MeanPunctaMaxes": 0.0,
            "StdPunctaMaxes": 0.0,
            "MedianPunctaMaxes": 0.0,
            **_ZEROS_SPATIAL,
        }

    # --- Per-punctum statistics ---
    # bincount / maximum.at aggregate every punctum in one pass, which is much faster
    # than looping over labels. Index 0 (background) is dropped with [1:].
    flat_labels = seg_labels.ravel()
    flat_raw = raw.ravel().astype(np.float64)

    counts = np.bincount(flat_labels, minlength=n_puncta + 1)[1:].astype(np.float64)
    sums_raw = np.bincount(flat_labels, weights=flat_raw, minlength=n_puncta + 1)[1:]
    valid = counts > 0
    means_raw = np.zeros(n_puncta)
    means_raw[valid] = sums_raw[valid] / counts[valid]
    maxes_raw = np.full(n_puncta, -np.inf)
    fg_idx = flat_labels > 0
    np.maximum.at(maxes_raw, flat_labels[fg_idx] - 1, flat_raw[fg_idx])

    sizes_valid = counts[valid]
    means_valid = means_raw[valid]
    maxes_valid = maxes_raw[valid]

    # --- Foreground vs background intensity ---
    # Foreground statistics pool all foreground pixels (large puncta weigh more);
    # the per-punctum statistics above weigh each punctum equally.
    fg_y, fg_x = np.where(combined_fg)
    fg_raw = raw[fg_y, fg_x].astype(np.float64)

    # Background = ROI pixels that are not foreground. The ratio is reported as 0.0
    # when the background mean is 0, to avoid dividing by zero.
    non_fg_mask = roi & ~combined_fg
    non_fg_pixels = raw[non_fg_mask].astype(np.float64)
    bg_mean = float(non_fg_pixels.mean()) if len(non_fg_pixels) > 0 else 0.0
    bg_std = float(non_fg_pixels.std()) if len(non_fg_pixels) > 1 else 0.0
    bg_median = float(np.median(non_fg_pixels)) if len(non_fg_pixels) > 0 else 0.0
    ratio_fg_bg = float(fg_raw.mean()) / bg_mean if bg_mean > 0 else 0.0

    # --- Radial position ---
    # Mean distance of foreground pixels from the anchor. Enrichment > 1 means the
    # foreground is farther from the anchor than the average ROI pixel (peripheral);
    # < 1 means closer (perinuclear).
    fg_coords = np.column_stack([fg_y, fg_x]).astype(np.float64)
    fg_dists = np.linalg.norm(fg_coords - anchor_centroid, axis=1)
    radial_mean = float(np.mean(fg_dists))
    radial_enrichment = radial_mean / expected_radial if expected_radial > 1e-5 else 0.0

    # --- Extent (spread) ---
    # How spread out the foreground is around its OWN centroid, relative to how spread
    # out the whole ROI is. > 1 = more dispersed than the cell; < 1 = clustered.
    fg_centroid = np.array([float(fg_y.mean()), float(fg_x.mean())])
    extent_dists = np.linalg.norm(fg_coords - fg_centroid, axis=1)
    extent_mean = float(np.mean(extent_dists))
    extent_enrichment = extent_mean / expected_extent if expected_extent > 1e-5 else 0.0

    # --- Polarization ---
    # Offset of the foreground centroid from the anchor (0 = symmetric around it).
    # FgPolarizationNorm divides by the mean anchor-to-boundary distance below, so
    # it is comparable across cells of different sizes.
    fg_pol_px = float(np.linalg.norm(fg_centroid - anchor_centroid))

    # --- Ring density ---
    # Fraction of each ring's ROI area covered by foreground, using the rescaled
    # radial fraction map (inner / middle / outer; see _RING_CUTOFF_*).
    fg_frac = radial_frac_map[combined_fg]
    roi_frac = radial_frac_map[roi]

    ring_densities = np.zeros(3)
    for ring_idx, (lo, hi) in enumerate(
        [
            (0, _RING_CUTOFF_LO),
            (_RING_CUTOFF_LO, _RING_CUTOFF_HI),
            (_RING_CUTOFF_HI, np.inf),
        ]
    ):
        ring_roi_area = int(np.sum((roi_frac >= lo) & (roi_frac < hi)))
        ring_fg_area = int(np.sum((fg_frac >= lo) & (fg_frac < hi)))
        ring_densities[ring_idx] = (
            ring_fg_area / ring_roi_area if ring_roi_area > 0 else 0.0
        )

    # One percentile call sorts fg_raw once instead of three times.
    pct25, pct50, pct75 = np.percentile(fg_raw, [25, 50, 75])

    return {
        "FgObjectCount": n_fg_objects,
        "PunctaCount": n_puncta,
        "TotalFgArea": total_area,
        "TotalFgAreaFraction": total_area / cell_area if cell_area > 0 else 0.0,
        "MeanPunctaSize": float(np.mean(sizes_valid)),
        "StdPunctaSize": float(np.std(sizes_valid)),
        "MedianPunctaSize": float(np.median(sizes_valid)),
        "TotalFgIntensity": float(np.sum(fg_raw)),
        "MeanFgIntensity": float(np.mean(fg_raw)),
        "StdFgIntensity": float(np.std(fg_raw)),
        "Pct25FgIntensity": float(pct25),
        "Pct50FgIntensity": float(pct50),
        "Pct75FgIntensity": float(pct75),
        "MeanBgIntensity": bg_mean,
        "StdBgIntensity": bg_std,
        "MedianBgIntensity": bg_median,
        "RatioFgBgMeanIntensities": ratio_fg_bg,
        "MeanPunctaMeans": float(np.mean(means_valid)),
        "StdPunctaMeans": float(np.std(means_valid)),
        "MedianPunctaMeans": float(np.median(means_valid)),
        "MeanPunctaMaxes": float(np.mean(maxes_valid)),
        "StdPunctaMaxes": float(np.std(maxes_valid)),
        "MedianPunctaMaxes": float(np.median(maxes_valid)),
        "MeanFgRadialPosPx": radial_mean,
        "FgRadialEnrichment": radial_enrichment,
        "MeanFgExtentPx": extent_mean,
        "FgExtentEnrichment": extent_enrichment,
        "FgPolarizationPx": fg_pol_px,
        "FgPolarizationNorm": fg_pol_px / mean_boundary_dist,
        "DensityRingInner": ring_densities[0],
        "DensityRingMiddle": ring_densities[1],
        "DensityRingOuter": ring_densities[2],
    }


def puncta(
    c: Sequence[int],
    sigma: float = 10.0,
    alpha: float = 1.5,
    min_size_fg_objects: int = 2,
    peak_min_distance: int = 2,
    channel_names: Sequence[str] | None = None,
    unique_labels: np.ndarray = None,
    unique_labels_original: np.ndarray = None,
    label_image: np.ndarray = None,
    label_image_original: np.ndarray = None,
    intensity_image: np.ndarray = None,
    offset: tuple[int, int] | None = None,
    nuclei_label_image: np.ndarray = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Puncta detection and feature extraction using adaptive thresholding.

    Foreground pixels satisfy: raw > alpha * local_background, where
    local_background is the per-pixel normalized Gaussian blur of the cell
    (sigma controls the spatial scale of background removal). When sigma=0,
    uses raw > alpha * mean(raw within ROI) instead.

    Pipeline: highpass -> adaptive threshold -> size filter -> watershed.
    Computes 33 features per cell per channel (incl. FlagNoNuclearCentroid).
    Pipeline parameters are encoded
    in the feature name suffix (e.g. ``Puncta_PunctaCount_Channel3_s10.0_a1.5_m2_d2``).

    :param c: Channel indices to process.
    :param sigma: Gaussian sigma for highpass filter; 0 skips highpass (default 10.0).
    :param alpha: Threshold multiplier on local background; foreground = raw > alpha *
        local_bg. Values > 1 require signal to exceed background (default 1.5).
    :param min_size_fg_objects: Minimum connected-component area in pixels; smaller
        foreground objects are discarded, and watershed pieces smaller than this are
        merged into a neighbouring piece (default 2).
    :param peak_min_distance: Minimum pixel distance between watershed seed peaks.
        Controls puncta splitting — larger values merge nearby peaks into one punctum
        (default 2).
    :param channel_names: Names for all channels in intensity_image.
    :param unique_labels: Unique non-zero labels present in label_image (sequential).
    :param unique_labels_original: Original cell IDs before sequential relabeling.
    :param label_image: 2D integer label mask (sequentially relabeled).
    :param label_image_original: 2D integer label mask (original labels).
    :param intensity_image: 3D array (Y, X, C) of intensity data.
    :param offset: (row, col) offset of this block within the full image.
    :param nuclei_label_image: Full-image nuclei labels (numpy, zarr, or dask array)
        whose IDs match the ROI labels. Spatial features are anchored on the centroid
        of each ROI's same-ID nucleus; if None, or for a ROI whose nucleus is missing
        or does not overlap or touch it, the ROI's own centroid is used and that ROI's
        FlagNoNuclearCentroid is 1. Pass it for nuclei ROIs too, so their flag is 0.
    :param kwargs: Absorbs other framework arguments.
    :return: Dictionary mapping feature names to arrays of length len(unique_labels).
    """
    _validate_params(sigma, alpha, min_size_fg_objects, peak_min_distance)
    # Scallops always passes channel_names; the column names need them.
    assert channel_names is not None, "channel_names is required"
    peak_min_distance = int(peak_min_distance)  # accept integer-valued floats like 2.0

    n_cells = len(unique_labels)
    results = {}
    # One bounding-box slice per sequential label (None if a label is absent).
    objects = find_objects(label_image)

    # Row i of every output array belongs to the i-th label present in label_image, and
    # scallops pairs row i with unique_labels[i] (it combines all feature modules by
    # row). A label in unique_labels that is absent from label_image, or the reverse,
    # would silently put features on the wrong cells, so fail instead.
    present = np.flatnonzero([s is not None for s in objects]) + 1
    if not np.array_equal(present, np.asarray(unique_labels)):
        absent = np.setdiff1d(unique_labels, present)
        unlisted = np.setdiff1d(present, unique_labels)
        raise ValueError(
            "unique_labels must equal the sorted labels present in label_image: "
            f"{len(absent)} listed but absent (e.g. {absent[:5]}), {len(unlisted)} "
            f"present but not listed (e.g. {unlisted[:5]}); if both are 0, "
            "unique_labels is not sorted."
        )

    if unique_labels_original is None:
        unique_labels_original = unique_labels

    block_offset_y = offset[0] if offset is not None else 0
    block_offset_x = offset[1] if offset is not None else 0

    # Nuclei labels of this block, read once; ROIs whose nucleus extends beyond the
    # block read the rest from the full image (see _find_nucleus).
    nuclei_block = None
    if nuclei_label_image is not None:
        nuclei_block = _read_label_region(
            nuclei_label_image,
            block_offset_y,
            block_offset_y + label_image.shape[0],
            block_offset_x,
            block_offset_x + label_image.shape[1],
        )

    # Pre-allocate one output array per feature per channel (every row is written
    # below; the zeros are only kept by the defensive empty-ROI guard).
    all_feat_arrays = {}
    for j in c:
        all_feat_arrays[j] = {
            k: np.zeros(n_cells, dtype=np.float64) for k in _FEATURE_KEYS
        }

    # Loop structure: outer loop over cells (per-cell geometry is computed once),
    # inner loop over channels (detection + features reuse that geometry).
    # `index` is the cell's row in the output arrays; `object_index + 1` is its
    # sequential label. They differ when some labels are absent from the block.
    index = 0
    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        cell_label = object_index + 1
        roi = label_image[sl] == cell_label  # boolean mask inside the cell's bbox

        # Defensive guard; unreachable via find_objects, which returns a slice only for
        # labels with pixels. If reached: features stay zero and, as no anchor was
        # used, the flag is 1.
        if not roi.any():
            for j in c:
                all_feat_arrays[j]["FlagNoNuclearCentroid"][index] = 1.0
            index += 1
            continue

        ry, rx = np.where(roi)
        cell_centroid = np.array([ry.mean(), rx.mean()])

        # Spatial anchor: the centroid of the ROI's same-ID nucleus, computed in a frame
        # that contains both the ROI and the whole nucleus. Without a usable nucleus the
        # frame is the cell crop itself and the anchor is the ROI centroid.
        found = None
        if nuclei_block is not None:
            # The ROI's original ID; label_image holds sequential labels.
            orig_label = (
                label_image_original[sl][roi][0]
                if label_image_original is not None
                else unique_labels_original[index]
            )
            found = _find_nucleus(
                nuclei_label_image,
                nuclei_block,
                (block_offset_y, block_offset_x),
                (block_offset_y + sl[0].start, block_offset_x + sl[1].start),
                roi,
                orig_label,
            )
        if found is None:
            roi_frame, nucleus_frame, (crop_y, crop_x) = roi, None, (0, 0)
            anchor_frame = cell_centroid
        else:
            roi_frame, nucleus_frame, (crop_y, crop_x) = found
            ny, nx = np.nonzero(nucleus_frame)
            anchor_frame = np.array([ny.mean(), nx.mean()])

        geometry = _geometry_mask(roi_frame, anchor_frame, nucleus_frame)
        radial_frame, expected_radial = _compute_radial_frac_map(
            roi_frame, anchor_frame, geometry
        )
        radial_frac_map = radial_frame[
            crop_y : crop_y + roi.shape[0], crop_x : crop_x + roi.shape[1]
        ]
        anchor_centroid = anchor_frame - np.array([crop_y, crop_x])

        roi_coords = np.column_stack([ry, rx]).astype(np.float64)
        cell_area = int(roi.sum())

        # Mean distance from the anchor to the boundary of the same cell shape the
        # radial features use (outer cell edge only, e.g. not the nuclear hole of a
        # cytoplasm ROI); normalizes FgPolarizationNorm.
        inner_bnd = geometry & ~minimum_filter(
            geometry, footprint=_CROSS_FOOTPRINT, mode="constant", cval=False
        )
        bc = np.argwhere(inner_bnd)
        mean_boundary_dist = (
            max(float(np.mean(np.linalg.norm(bc - anchor_frame, axis=1))), 1e-5)
            if len(bc) > 0
            else 1.0
        )
        # Mean distance of all ROI pixels from the ROI centroid; a cell-shape constant.
        expected_extent = float(
            np.mean(np.linalg.norm(roi_coords - cell_centroid, axis=1))
        )

        # Blurred ROI mask for the highpass normalization. It depends only on the cell,
        # so one gaussian_filter per cell replaces one per channel.
        mask_f = roi.astype(np.float32)
        if sigma > 0:
            blurred_mask = gaussian_filter(mask_f, sigma=sigma, mode="constant", cval=0)
            np.clip(blurred_mask, 1e-10, None, out=blurred_mask)
        else:
            blurred_mask = None

        for j in c:
            raw = intensity_image[sl][..., j].astype(np.float32)

            seg_labels, n_fg_objects = _detect_puncta(
                raw,
                roi,
                sigma,
                alpha,
                min_size_fg_objects,
                peak_min_distance,
                blurred_mask=blurred_mask,
            )

            feats = _extract_cell_features(
                raw,
                seg_labels,
                roi,
                anchor_centroid,
                cell_centroid,
                expected_radial,
                mean_boundary_dist,
                radial_frac_map,
                n_fg_objects,
                roi_coords,
                cell_area,
                expected_extent,
            )
            feats["FlagNoNuclearCentroid"] = 1.0 if found is None else 0.0
            for k in _FEATURE_KEYS:
                all_feat_arrays[j][k][index] = feats[k]

        index += 1

    # Encode the parameters in each feature name so runs with different parameters
    # produce distinct columns. The scallops CLI later prepends the ROI prefix
    # (Cells_, Nuclei_, Cytoplasm_) to every column from every feature module.
    params_suffix = f"s{sigma}_a{alpha}_m{min_size_fg_objects}_d{peak_min_distance}"

    for j in c:
        ch_name = channel_names[j]
        for k in _FEATURE_KEYS:
            results[f"Puncta_{k}_{ch_name}_{params_suffix}"] = all_feat_arrays[j][k]

    return results
