"""Tests for scallops.features.puncta: synthetic edge cases, unit tests, and
label_features integration with nuclei labels."""

import logging

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import skimage.util
import zarr
from scipy.ndimage import minimum_filter

from scallops.features.find_objects import find_objects
from scallops.features.generate import (
    _bind_nuclei_labels,
    _create_funcs,
    label_features,
)
from scallops.features.puncta import (
    _FEATURE_KEYS,
    NUCLEI_MATCH_MIN_FRACTION,
    _compute_radial_frac_map,
    _detect_puncta,
    _extract_cell_features,
    _find_nucleus,
    _geometry_mask,
    _highpass_filter,
    _merge_saturated_plateaus,
    _merge_small_puncta,
    _size_filter,
    _validate_params,
    _watershed_segment,
    nuclei_label_match_fraction,
    puncta,
)

# Every test in this file is a feature test (scallops' `features` marker).
pytestmark = pytest.mark.features

ALL_CHANNEL_NAMES = [f"Channel{i}" for i in range(16)]


# =====================================================================
# Shared helpers
# =====================================================================


def _make_label_and_intensity(
    n_cells=1, cell_size=30, n_channels=4, bg_val=0.1, bright_val=0.6
):
    """Create a minimal label image and (Y, X, C) intensity image.

    Cells are stacked vertically. Channel 1 gets two bright 2x2 clusters
    to give puncta-like signal; all other channels are uniform background.
    """
    H = cell_size * n_cells + 10
    W = cell_size + 10
    label_img = np.zeros((H, W), dtype=int)
    intensity = np.full((H, W, n_channels), bg_val, dtype=np.float32)

    ch_idx = min(1, n_channels - 1)  # use ch 1 when available, else ch 0
    for i in range(n_cells):
        r0 = i * cell_size + 5
        r1 = r0 + cell_size
        c0, c1 = 5, 5 + cell_size
        label_img[r0:r1, c0:c1] = i + 1

        # Two bright spots separated by > peak_min_distance
        intensity[r0 + 6 : r0 + 9, c0 + 6 : c0 + 9, ch_idx] = bright_val
        intensity[r0 + 18 : r0 + 21, c0 + 18 : c0 + 21, ch_idx] = bright_val

    return label_img, intensity


def _make_extract_inputs(shape=(20, 20)):
    """Build the positional inputs expected by _extract_cell_features."""
    h, w = shape
    roi = np.ones(shape, dtype=bool)
    anchor = np.array([h / 2.0, w / 2.0])
    cell_centroid = anchor.copy()
    radial_frac_map, expected_radial = _compute_radial_frac_map(roi, anchor)
    ry, rx = np.where(roi)
    roi_coords = np.column_stack([ry, rx]).astype(np.float64)
    cell_area = int(roi.sum())
    expected_extent = float(np.mean(np.linalg.norm(roi_coords - cell_centroid, axis=1)))
    mean_boundary_dist = float(min(h, w) / 2.0)
    return dict(
        roi=roi,
        anchor_centroid=anchor,
        cell_centroid=cell_centroid,
        expected_radial=expected_radial,
        mean_boundary_dist=mean_boundary_dist,
        radial_frac_map=radial_frac_map,
        roi_coords=roi_coords,
        cell_area=cell_area,
        expected_extent=expected_extent,
    )


def _call_extract(raw, seg_labels, n_fg_objects, inputs):
    """Call _extract_cell_features with a prepared inputs dict."""
    return _extract_cell_features(
        raw,
        seg_labels,
        inputs["roi"],
        inputs["anchor_centroid"],
        inputs["cell_centroid"],
        inputs["expected_radial"],
        inputs["mean_boundary_dist"],
        inputs["radial_frac_map"],
        n_fg_objects,
        inputs["roi_coords"],
        inputs["cell_area"],
        inputs["expected_extent"],
    )


def _run_puncta(
    n_cells=1, n_channels=3, sigma=5.0, alpha=1.5, min_size=2, pmd=2, cell_size=30
):
    label_img, intensity = _make_label_and_intensity(
        n_cells=n_cells, cell_size=cell_size, n_channels=n_channels
    )
    unique_labels = np.arange(1, n_cells + 1)
    channel_names = [f"Channel{i}" for i in range(n_channels)]
    return puncta(
        c=list(range(n_channels)),
        sigma=sigma,
        alpha=alpha,
        min_size_fg_objects=min_size,
        peak_min_distance=pmd,
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_img,
        intensity_image=intensity,
        offset=(0, 0),
    )


# =====================================================================
# _validate_params
# =====================================================================


class TestValidateParams:
    # --- valid inputs that must not raise ---

    def test_defaults_ok(self):
        _validate_params(10.0, 1.5, 2, 2)

    def test_sigma_zero_alpha_lt1_ok(self):
        _validate_params(0, 0.5, 2, 2)

    def test_sigma_zero_alpha_exactly_one_ok(self):
        _validate_params(0, 1.0, 2, 2)

    def test_peak_min_distance_one_ok(self):
        _validate_params(10.0, 1.5, 2, 1)

    def test_peak_min_distance_float_integer_ok(self):
        _validate_params(10.0, 1.5, 2, 2.0)

    def test_min_size_one_ok(self):
        _validate_params(10.0, 1.5, 1, 2)

    # --- sigma errors ---

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            _validate_params(-0.1, 1.5, 2, 2)

    def test_sigma_nan_raises(self):
        with pytest.raises(ValueError, match="sigma must be finite"):
            _validate_params(float("nan"), 1.5, 2, 2)

    def test_sigma_inf_raises(self):
        with pytest.raises(ValueError, match="sigma must be finite"):
            _validate_params(float("inf"), 1.5, 2, 2)

    def test_sigma_not_number_raises(self):
        with pytest.raises(TypeError, match="sigma must be a number"):
            _validate_params("10", 1.5, 2, 2)

    # --- alpha errors ---

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            _validate_params(10.0, 0.0, 2, 2)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            _validate_params(10.0, -1.0, 2, 2)

    def test_alpha_nan_raises(self):
        with pytest.raises(ValueError, match="alpha must be finite"):
            _validate_params(10.0, float("nan"), 2, 2)

    def test_sigma_gt0_alpha_lt1_raises(self):
        with pytest.raises(ValueError, match="alpha must be >= 1 when sigma > 0"):
            _validate_params(5.0, 0.9, 2, 2)

    def test_sigma_gt0_alpha_exactly1_ok(self):
        _validate_params(5.0, 1.0, 2, 2)

    # --- min_size errors ---

    def test_min_size_zero_raises(self):
        with pytest.raises(ValueError, match="min_size_fg_objects must be >= 1"):
            _validate_params(10.0, 1.5, 0, 2)

    def test_min_size_negative_raises(self):
        with pytest.raises(ValueError, match="min_size_fg_objects must be >= 1"):
            _validate_params(10.0, 1.5, -1, 2)

    # --- peak_min_distance errors ---

    def test_pmd_bool_true_raises(self):
        with pytest.raises(
            TypeError, match="peak_min_distance must be a positive integer"
        ):
            _validate_params(10.0, 1.5, 2, True)

    def test_pmd_bool_false_raises(self):
        with pytest.raises(TypeError):
            _validate_params(10.0, 1.5, 2, False)

    def test_pmd_noninteger_float_raises(self):
        with pytest.raises(
            ValueError, match="peak_min_distance must be a whole number"
        ):
            _validate_params(10.0, 1.5, 2, 1.5)

    def test_pmd_zero_raises(self):
        with pytest.raises(ValueError, match="peak_min_distance must be >= 1"):
            _validate_params(10.0, 1.5, 2, 0)

    def test_pmd_negative_raises(self):
        with pytest.raises(ValueError, match="peak_min_distance must be >= 1"):
            _validate_params(10.0, 1.5, 2, -1)


# =====================================================================
# _highpass_filter
# =====================================================================


class TestHighpassFilter:
    def test_uniform_image_filtered_is_zero(self):
        img = np.ones((20, 20), dtype=np.float32) * 0.5
        roi = np.ones((20, 20), dtype=bool)
        filtered, local_bg = _highpass_filter(img, roi, sigma=5)
        assert np.allclose(filtered, 0, atol=1e-4)

    def test_bright_pixel_survives_filter(self):
        img = np.ones((30, 30), dtype=np.float32) * 0.1
        roi = np.ones((30, 30), dtype=bool)
        img[15, 15] = 0.9
        filtered, local_bg = _highpass_filter(img, roi, sigma=3)
        assert filtered[15, 15] > 0

    def test_outside_roi_is_zero(self):
        img = np.ones((20, 20), dtype=np.float32) * 0.5
        roi = np.zeros((20, 20), dtype=bool)
        roi[5:15, 5:15] = True
        filtered, local_bg = _highpass_filter(img, roi, sigma=3)
        assert filtered[~roi].max() == 0.0
        assert local_bg[~roi].max() == 0.0

    def test_local_bg_positive_inside_roi(self):
        img = np.ones((20, 20), dtype=np.float32) * 0.5
        roi = np.ones((20, 20), dtype=bool)
        _, local_bg = _highpass_filter(img, roi, sigma=3)
        assert np.all(local_bg[roi] > 0)

    def test_filtered_nonneg(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (30, 30)).astype(np.float32)
        roi = np.ones((30, 30), dtype=bool)
        filtered, _ = _highpass_filter(img, roi, sigma=5)
        assert np.all(filtered >= 0)

    def test_blurred_mask_reuse_gives_same_result(self):
        from scipy.ndimage import gaussian_filter as gf

        img = np.ones((20, 20), dtype=np.float32) * 0.5
        roi = np.ones((20, 20), dtype=bool)
        bm = gf(roi.astype(np.float32), sigma=3, mode="constant", cval=0)
        np.clip(bm, 1e-10, None, out=bm)
        f1, bg1 = _highpass_filter(img, roi, sigma=3)
        f2, bg2 = _highpass_filter(img, roi, sigma=3, blurred_mask=bm)
        assert np.allclose(f1, f2, atol=1e-5)
        assert np.allclose(bg1, bg2, atol=1e-5)


# =====================================================================
# _size_filter
# =====================================================================


class TestSizeFilter:
    # In real use fg_mask = (threshold) & roi, so roi always contains all fg pixels.
    # Tests mirror this: roi is the cell region, fg is a subset of roi.

    def test_empty_input_returns_zeros(self):
        roi = np.ones((10, 10), dtype=bool)
        fg = np.zeros((10, 10), dtype=bool)
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert not filtered.any()
        assert n == 0
        assert labeled.max() == 0

    def test_single_pixel_filtered_out(self):
        roi = np.ones((10, 10), dtype=bool)
        fg = np.zeros((10, 10), dtype=bool)
        fg[5, 5] = True
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert not filtered.any()
        assert n == 0

    def test_two_pixel_component_kept(self):
        roi = np.ones((10, 10), dtype=bool)
        fg = np.zeros((10, 10), dtype=bool)
        fg[5, 5] = True
        fg[5, 6] = True
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert filtered.sum() == 2
        assert n == 1

    def test_exactly_min_size_kept(self):
        roi = np.ones((10, 10), dtype=bool)
        fg = np.zeros((10, 10), dtype=bool)
        fg[5, 5] = True
        fg[5, 6] = True
        fg[6, 5] = True  # 3 pixels, min_size=3
        filtered, labeled, n = _size_filter(fg, min_size=3, roi=roi)
        assert filtered.sum() == 3
        assert n == 1

    def test_one_below_min_size_removed(self):
        roi = np.ones((10, 10), dtype=bool)
        fg = np.zeros((10, 10), dtype=bool)
        fg[5, 5] = True
        fg[5, 6] = True  # 2 pixels, min_size=3
        filtered, labeled, n = _size_filter(fg, min_size=3, roi=roi)
        assert not filtered.any()
        assert n == 0

    def test_small_hole_1px_filled(self):
        roi = np.ones((10, 10), dtype=bool)
        fg = np.zeros((10, 10), dtype=bool)
        fg[3:7, 3:7] = True
        fg[5, 5] = False  # 1-pixel interior hole
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert filtered[5, 5], "1-pixel hole should be filled"

    def test_small_hole_3px_filled(self):
        roi = np.ones((15, 15), dtype=bool)
        fg = np.zeros((15, 15), dtype=bool)
        fg[2:12, 2:12] = True
        fg[5, 5] = False
        fg[5, 6] = False
        fg[5, 7] = False  # 3-pixel linear hole (interior)
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert filtered[5, 5] and filtered[5, 6] and filtered[5, 7]

    def test_large_hole_not_filled(self):
        roi = np.ones((15, 15), dtype=bool)
        fg = np.zeros((15, 15), dtype=bool)
        fg[2:12, 2:12] = True
        fg[5:9, 5:9] = False  # 16-pixel hole
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert not filtered[6, 6], "16-pixel hole must not be filled"

    def test_multiple_components_partial_filter(self):
        roi = np.ones((20, 20), dtype=bool)
        fg = np.zeros((20, 20), dtype=bool)
        fg[2, 2] = True  # 1 px — filtered out
        fg[10:13, 10:13] = True  # 9 px — kept
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert not filtered[2, 2]
        assert filtered[10:13, 10:13].all()
        assert n == 1

    def test_n_objects_matches_labeled_components(self):
        roi = np.ones((20, 20), dtype=bool)
        fg = np.zeros((20, 20), dtype=bool)
        fg[2:5, 2:5] = True  # 9 px
        fg[10:13, 10:13] = True  # 9 px
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        assert n == 2
        assert labeled.max() == 2

    def test_output_labeled_has_sequential_ids(self):
        roi = np.ones((20, 20), dtype=bool)
        fg = np.zeros((20, 20), dtype=bool)
        fg[2:5, 2:5] = True
        fg[10:13, 10:13] = True
        filtered, labeled, n = _size_filter(fg, min_size=2, roi=roi)
        ids = set(labeled[labeled > 0].ravel())
        assert ids == {1, 2}

    def test_non_roi_hole_not_filled(self):
        # A 1-pixel hole that is enclosed by foreground but lies OUTSIDE the ROI (a
        # gap in an irregular ROI) must not be filled: only ROI pixels can become
        # foreground (the "& roi" in _size_filter's hole fill).
        roi = np.ones((10, 10), dtype=bool)
        roi[5, 5] = False
        fg = roi.copy()  # every ROI pixel is foreground
        filtered, labeled, n = _size_filter(fg, min_size=1, roi=roi)
        assert not filtered[5, 5], "non-ROI pixel must not be filled by hole-fill"
        assert filtered[roi].all()


# =====================================================================
# _watershed_segment
# =====================================================================


class TestWatershedSegment:
    def test_empty_fg_returns_zeros(self):
        img = np.zeros((10, 10), dtype=np.float32)
        fg = np.zeros((10, 10), dtype=bool)
        fg_labeled = np.zeros((10, 10), dtype=int)
        result = _watershed_segment(img, fg, fg_labeled, 0, min_distance=2, min_size=1)
        assert result.max() == 0
        assert result.shape == img.shape

    def test_single_component_gets_at_least_one_label(self):
        img = np.zeros((20, 20), dtype=np.float32)
        img[5:15, 5:15] = 0.3
        img[10, 10] = 0.9
        fg = img > 0.1
        from skimage.measure import label as sk_label

        fg_lab = sk_label(fg, connectivity=1)
        n = int(fg_lab.max())
        result = _watershed_segment(img, fg, fg_lab, n, min_distance=2, min_size=1)
        assert result.max() >= 1
        assert (result > 0).sum() == fg.sum()

    def test_two_separated_components_each_get_label(self):
        # Each fg component must receive at least one distinct watershed label.
        # min_distance=1 can produce more than 2 total labels (boundary pixels of
        # a uniform-valued region are local maxima), so assert >= 2, not == 2.
        img = np.zeros((30, 15), dtype=np.float32)
        img[3:8, 4:10] = 0.3
        img[20:25, 4:10] = 0.3
        img[5, 7] = 0.9
        img[22, 7] = 0.9
        fg = img > 0.1
        from skimage.measure import label as sk_label

        fg_lab = sk_label(fg, connectivity=1)
        n = int(fg_lab.max())
        result = _watershed_segment(img, fg, fg_lab, n, min_distance=1, min_size=1)
        # Both fg components must be represented; result may have > 2 labels due to
        # multiple local maxima within uniform-intensity regions.
        assert result.max() >= 2
        for comp_id in range(1, n + 1):
            assert result[fg_lab == comp_id].max() > 0, (
                f"Component {comp_id} has no watershed label"
            )

    def test_uniform_foreground_std_zero_still_seeds(self):
        # All foreground pixels identical → std=0 → uses max+1 fallback
        img = np.zeros((20, 20), dtype=np.float32)
        img[5:15, 5:15] = 0.5  # perfectly uniform foreground
        fg = img > 0.1
        from skimage.measure import label as sk_label

        fg_lab = sk_label(fg, connectivity=1)
        n = int(fg_lab.max())
        result = _watershed_segment(img, fg, fg_lab, n, min_distance=2, min_size=1)
        # Without the guard every pixel would pass threshold_abs = 0 and seed; with it
        # no peak qualifies and the fallback seeds the single component exactly once.
        assert result.max() == 1

    def test_unseeded_component_fallback(self):
        # Two components; peak_local_max misses one because it's below threshold_abs.
        # The fallback should still seed both.
        img = np.zeros((20, 20), dtype=np.float32)
        # Large bright component → peak_local_max will find its peak
        img[2:8, 2:8] = 0.8
        # Tiny uniform component far away — peak_local_max may skip it
        img[14, 14] = 0.01  # single pixel, very dim
        fg = img > 0.001
        from skimage.measure import label as sk_label

        fg_lab = sk_label(fg, connectivity=1)
        n = int(fg_lab.max())
        result = _watershed_segment(img, fg, fg_lab, n, min_distance=5, min_size=1)
        # Both components must receive a label
        for comp_id in range(1, n + 1):
            comp_pixels = result[fg_lab == comp_id]
            assert comp_pixels.max() > 0, (
                f"Component {comp_id} has no label (fallback failed)"
            )

    def test_result_covers_all_fg_pixels(self):
        img = np.zeros((20, 20), dtype=np.float32)
        img[5:15, 5:15] = 0.5
        img[10, 10] = 0.9
        fg = img > 0.1
        from skimage.measure import label as sk_label

        fg_lab = sk_label(fg, connectivity=1)
        n = int(fg_lab.max())
        result = _watershed_segment(img, fg, fg_lab, n, min_distance=2, min_size=1)
        assert np.all((result > 0) == fg)

    def test_smaller_min_distance_gives_more_or_equal_labels(self):
        # Smaller min_distance suppresses fewer peaks, so it should produce
        # at least as many watershed labels as a larger min_distance.
        # Note: min_distance=0 is a special case (scikit-image treats it as
        # "every pixel above threshold is a peak") and is intentionally excluded
        # from this test — with exclude_border=False all uniform-background
        # pixels can flood peaks_mask, merge into one marker, and produce fewer
        # labels than a well-chosen positive min_distance.
        img = np.zeros((30, 30), dtype=np.float32)
        img[5:25, 5:25] = 0.3
        for r, c in [(8, 8), (8, 20), (20, 8), (20, 20), (14, 14)]:
            img[r, c] = 0.9
        fg = img > 0.1
        from skimage.measure import label as sk_label

        fg_lab = sk_label(fg, connectivity=1)
        n = int(fg_lab.max())
        result_d2 = _watershed_segment(img, fg, fg_lab, n, min_distance=2, min_size=1)
        result_d10 = _watershed_segment(img, fg, fg_lab, n, min_distance=10, min_size=1)
        assert result_d2.max() >= result_d10.max()


# =====================================================================
# _merge_small_puncta
# =====================================================================


class TestMergeSmallPuncta:
    def test_no_small_puncta_returns_unchanged(self):
        seg = np.zeros((10, 10), dtype=int)
        seg[2:6, 2:6] = 1  # 16 pixels — well above any threshold
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _merge_small_puncta(seg.copy(), img, min_size=2)
        assert np.array_equal(result, seg)

    def test_single_small_merges_into_large_neighbor(self):
        seg = np.zeros((10, 10), dtype=int)
        seg[2:7, 2:7] = 1  # 25-pixel large region
        seg[2, 7] = 2  # 1-pixel small region adjacent to label 1
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _merge_small_puncta(seg, img, min_size=2)
        assert result[2, 7] == 1, "small punctum should merge into large neighbor"
        assert result.max() == 1

    def test_two_small_adjacent_consolidate(self):
        # 4-pixel foreground split into two 2-pixel regions — no large neighbor.
        # Both are below min_size=3. They should consolidate into one region.
        seg = np.zeros((10, 10), dtype=int)
        seg[5, 5] = 1
        seg[5, 6] = 1  # label 1: 2 pixels
        seg[5, 7] = 2
        seg[5, 8] = 2  # label 2: 2 pixels, adjacent to label 1
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _merge_small_puncta(seg, img, min_size=3)
        assert result.max() == 1, "both sub-threshold regions should merge into one"
        assert (result > 0).sum() == 4, "total foreground pixel count preserved"

    def test_isolated_small_punctum_stays(self):
        # Single-pixel region surrounded only by background — no merge target.
        seg = np.zeros((10, 10), dtype=int)
        seg[5, 5] = 1
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _merge_small_puncta(seg, img, min_size=2)
        assert result[5, 5] == 1, "isolated small punctum must remain (no neighbor)"

    def test_labels_sequential_after_merge(self):
        seg = np.zeros((12, 12), dtype=int)
        seg[2:6, 2:6] = 1  # large
        seg[2, 6] = 2  # small, adjacent to 1
        seg[8:11, 8:11] = 3  # large, isolated
        img = np.ones((12, 12), dtype=np.float32) * 0.5
        result = _merge_small_puncta(seg, img, min_size=2)
        unique = np.unique(result[result > 0])
        assert np.array_equal(unique, np.arange(1, len(unique) + 1)), (
            f"labels not sequential after merge: {unique}"
        )

    def test_size_tracking_correct_after_cascade(self):
        # Small 1 (1 px) and small 2 (1 px) touch; 2 also touches large 3 (5 px).
        # 1 has fewer neighbours, so it merges first, into 2. Label 2 then has 2 px,
        # which meets min_size, so it must stay a separate punctum. If the size of 2
        # were not updated after absorbing 1, it would still look small and be
        # merged into 3, leaving one label.
        seg = np.zeros((10, 10), dtype=int)
        seg[5, 2] = 1
        seg[5, 3] = 2
        seg[5, 4:9] = 3
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _merge_small_puncta(seg, img, min_size=2)
        sizes = sorted(np.bincount(result.ravel())[1:])
        assert sizes == [2, 5], f"expected pieces of 2 and 5 px, got {sizes}"

    @pytest.mark.parametrize("brighter", ["left", "right"])
    def test_intensity_tiebreak(self, brighter):
        # Small label 1 (1 px) at (5,5) shares exactly 1 border pixel with each of
        # two large neighbours (4 px each, so neither is itself merged): label 2 on
        # the left, label 3 on the right. The tie is broken by the brighter boundary.
        seg = np.zeros((10, 10), dtype=int)
        seg[5, 5] = 1
        seg[5, 1:5] = 2
        seg[5, 6:10] = 3
        img = np.zeros((10, 10), dtype=np.float32)
        img[5, 4], img[5, 6] = (0.8, 0.2) if brighter == "left" else (0.2, 0.8)
        result = _merge_small_puncta(seg, img, min_size=2)
        winner = (5, 4) if brighter == "left" else (5, 6)
        assert result[5, 5] == result[winner], "small label must join the brighter side"
        assert len(np.unique(result[result > 0])) == 2

    def test_foreground_coverage_preserved(self):
        # Total foreground pixel count must not change after merging.
        seg = np.zeros((15, 15), dtype=int)
        seg[3:8, 3:8] = 1  # large
        seg[3, 8] = 2  # small
        img = np.ones((15, 15), dtype=np.float32) * 0.5
        fg_before = (seg > 0).sum()
        result = _merge_small_puncta(seg, img, min_size=2)
        fg_after = (result > 0).sum()
        assert fg_before == fg_after, "merging must not gain or lose foreground pixels"


# =====================================================================
# _merge_saturated_plateaus
# =====================================================================


class TestMergeSaturatedPlateaus:
    def test_pieces_sharing_a_plateau_become_one(self):
        # Pieces 1 and 2 split a 4-pixel plateau at the ROI maximum; piece 3 is a
        # separate, dimmer punctum.
        seg = np.zeros((6, 12), dtype=int)
        seg[1:5, 1:3], seg[1:5, 3:5], seg[1:5, 8:10] = 1, 2, 3
        raw = np.full((6, 12), 0.1, dtype=np.float32)
        raw[seg > 0] = 0.5
        raw[2:4, 2:4] = 1.0  # plateau across the 1|2 boundary
        out = _merge_saturated_plateaus(seg.copy(), raw, np.ones_like(seg, dtype=bool))
        assert out[2, 2] == out[2, 3] != out[2, 8]
        assert sorted(np.unique(out[out > 0])) == [1, 2]  # reindexed

    def test_pixels_within_tolerance_count_as_plateau(self):
        # Flatfield correction can make saturated pixels slightly unequal. Here
        # max - median = 1.0 - 0.5, so pixels >= 1.0 - 0.01 * 0.5 = 0.995 qualify.
        seg = np.zeros((6, 8), dtype=int)
        seg[1:5, 1:4], seg[1:5, 4:7] = 1, 2
        raw = np.full((6, 8), 0.5, dtype=np.float32)
        raw[0, :] = raw[5, :] = 0.1
        raw[2, 3], raw[2, 4] = 1.0, 0.997
        roi = np.ones_like(seg, dtype=bool)
        assert np.median(raw) == np.float32(0.5)
        assert _merge_saturated_plateaus(seg.copy(), raw, roi).max() == 1
        raw[2, 4] = 0.99  # outside the tolerance: no plateau of >= 2 pixels
        assert _merge_saturated_plateaus(seg.copy(), raw, roi).max() == 2

    def test_plateau_never_bridges_through_background(self):
        # Two foreground pieces with a near-max BACKGROUND pixel between them: only
        # foreground pixels form plateaus, so separate components stay separate.
        seg = np.zeros((3, 7), dtype=int)
        seg[1, 0:3], seg[1, 4:7] = 1, 2
        raw = np.full((3, 7), 0.1, dtype=np.float32)
        raw[1, :] = 1.0  # whole row at the maximum, including background (1, 3)
        out = _merge_saturated_plateaus(seg.copy(), raw, np.ones_like(seg, dtype=bool))
        assert out.max() == 2

    def test_clipped_spot_is_one_punctum(self):
        # A Gaussian spot clipped at 1.0 (81 saturated px) split into 8 (sigma 10) or
        # 21 (sigma 0) pieces before the plateau merge. Also with a +-2% flatfield gain.
        yy, xx = np.mgrid[0:41, 0:41]
        spot = np.clip(
            0.2 + 6 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 12.5), 0, 1
        )
        roi = np.ones((41, 41), dtype=bool)
        for gain in (0.0, 0.02):
            raw = (spot * (1 + gain * (xx - 20) / 20)).astype(np.float32)
            for sigma in (10.0, 5.0, 0.0):
                seg, n = _detect_puncta(raw, roi, sigma, 1.5, 2, 2)
                assert (n, seg.max()) == (1, 1), (gain, sigma)

    def test_separate_saturated_spots_stay_separate(self):
        yy, xx = np.mgrid[0:41, 0:41]
        raw = 0.2 + 6 * (
            np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2) / 12.5)
            + np.exp(-((yy - 29) ** 2 + (xx - 29) ** 2) / 12.5)
        )
        raw = np.clip(raw, 0, 1).astype(np.float32)
        for sigma in (10.0, 0.0):
            seg, n = _detect_puncta(
                raw, np.ones((41, 41), dtype=bool), sigma, 1.5, 2, 2
            )
            assert (n, seg.max()) == (2, 2), sigma


# =====================================================================
# _compute_radial_frac_map
# =====================================================================


class TestComputeRadialFracMap:
    def test_tiny_roi_returns_half(self):
        roi = np.zeros((10, 10), dtype=bool)
        roi[5, 5] = True
        roi[5, 6] = True  # 2 pixels → < 3, use fallback
        anchor = np.array([5.0, 5.5])
        frac_map, exp_radial = _compute_radial_frac_map(roi, anchor)
        assert np.allclose(frac_map[roi], 0.5)
        assert frac_map[~roi].max() == 0.0

    def test_outside_roi_always_zero(self):
        roi = np.zeros((20, 20), dtype=bool)
        roi[4:16, 4:16] = True
        anchor = np.array([10.0, 10.0])
        frac_map, _ = _compute_radial_frac_map(roi, anchor)
        assert frac_map[~roi].max() == 0.0

    def test_values_in_unit_range(self):
        roi = np.zeros((25, 25), dtype=bool)
        roi[3:22, 3:22] = True
        anchor = np.array([12.0, 12.0])
        frac_map, _ = _compute_radial_frac_map(roi, anchor)
        roi_vals = frac_map[roi]
        assert roi_vals.min() >= -1e-10
        assert roi_vals.max() <= 1 + 1e-10

    def test_anchor_at_center_inner_pixels_lower(self):
        roi = np.zeros((21, 21), dtype=bool)
        roi[2:19, 2:19] = True
        anchor = np.array([10.0, 10.0])
        frac_map, _ = _compute_radial_frac_map(roi, anchor)
        # Pixel at anchor should have low fraction (near center)
        center_val = frac_map[10, 10]
        # Pixel at corner of roi should have high fraction (near boundary)
        corner_val = frac_map[2, 2]
        assert center_val < corner_val

    def test_anchor_outside_roi_convex_hull_fallback(self):
        roi = np.zeros((25, 25), dtype=bool)
        roi[12:22, 12:22] = True  # bottom-right quadrant
        anchor = np.array([2.0, 2.0])  # top-left, outside roi
        frac_map, exp_radial = _compute_radial_frac_map(roi, anchor)
        roi_vals = frac_map[roi]
        # Must not crash and values must be valid
        assert roi_vals.min() >= -1e-10
        assert roi_vals.max() <= 1 + 1e-10
        assert exp_radial > 1e-5

    def test_expected_radial_positive(self):
        roi = np.zeros((20, 20), dtype=bool)
        roi[5:15, 5:15] = True
        anchor = np.array([10.0, 10.0])
        _, exp_radial = _compute_radial_frac_map(roi, anchor)
        assert exp_radial > 1e-5

    def test_circular_roi_spans_unit_range(self):
        # Perfectly circular ROI with center anchor → frac_map range ≈ [0, 1]
        # after rescaling; min should be near 0, max near 1
        roi = np.zeros((21, 21), dtype=bool)
        for r in range(21):
            for c in range(21):
                if (r - 10) ** 2 + (c - 10) ** 2 <= 81:
                    roi[r, c] = True
        anchor = np.array([10.0, 10.0])
        frac_map, _ = _compute_radial_frac_map(roi, anchor)
        roi_vals = frac_map[roi]
        assert roi_vals.min() < 0.1
        assert roi_vals.max() > 0.9


# =====================================================================
# _detect_puncta — pipeline integration
# =====================================================================


class TestDetectPuncta:
    def test_all_zeros_no_puncta(self):
        raw = np.zeros((30, 30), dtype=np.float32)
        roi = np.ones((30, 30), dtype=bool)
        seg, n = _detect_puncta(
            raw, roi, sigma=5, alpha=1.5, min_size_fg_objects=2, peak_min_distance=2
        )
        assert seg.max() == 0
        assert n == 0

    def test_uniform_image_sigma_gt0_no_puncta(self):
        # Highpass of uniform image is zero everywhere → no foreground
        raw = np.ones((30, 30), dtype=np.float32) * 0.5
        roi = np.ones((30, 30), dtype=bool)
        seg, n = _detect_puncta(
            raw, roi, sigma=5, alpha=1.5, min_size_fg_objects=2, peak_min_distance=2
        )
        assert seg.max() == 0
        assert n == 0

    def test_whole_roi_foreground_guard(self):
        # sigma=0, alpha=0.5 → raw > 0.5 * mean(raw). For uniform image, all
        # pixels pass → fires the whole-roi guard → returns (zeros, 0)
        raw = np.ones((20, 20), dtype=np.float32) * 0.5
        roi = np.ones((20, 20), dtype=bool)
        seg, n = _detect_puncta(
            raw, roi, sigma=0, alpha=0.5, min_size_fg_objects=1, peak_min_distance=1
        )
        assert seg.max() == 0
        assert n == 0

    def test_two_bright_clusters_detected(self):
        raw = np.ones((50, 50), dtype=np.float32) * 0.1
        roi = np.ones((50, 50), dtype=bool)
        raw[8:11, 8:11] = 0.8
        raw[35:38, 35:38] = 0.8
        seg, n = _detect_puncta(
            raw, roi, sigma=5, alpha=1.5, min_size_fg_objects=2, peak_min_distance=2
        )
        assert n == 2
        assert seg.max() == 2

    def test_sigma_zero_mean_threshold(self):
        # sigma=0: threshold = alpha * mean(raw[roi])
        # mean = (398 * 0.1 + 2 * 0.5) / 400 = 0.102, threshold = 1.2 * 0.102 = 0.1224
        # the two 0.5 pixels pass -> one 2-pixel object, kept with min_size=2
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        roi = np.ones((20, 20), dtype=bool)
        raw[5, 5] = 0.5
        raw[5, 6] = 0.5
        seg, n = _detect_puncta(
            raw, roi, sigma=0, alpha=1.2, min_size_fg_objects=2, peak_min_distance=1
        )
        assert n == 1
        assert seg.max() == 1
        assert (seg > 0).sum() == 2

    def test_size_filter_applied(self):
        # Single bright pixel: may be filtered out by min_size=2
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        roi = np.ones((20, 20), dtype=bool)
        raw[10, 10] = 0.9  # single pixel — below min_size=2
        seg_filtered, n_filtered = _detect_puncta(
            raw, roi, sigma=3, alpha=1.5, min_size_fg_objects=2, peak_min_distance=1
        )
        seg_kept, n_kept = _detect_puncta(
            raw, roi, sigma=3, alpha=1.5, min_size_fg_objects=1, peak_min_distance=1
        )
        # The 1-pixel spot is removed with min_size=2 and kept with min_size=1.
        assert (n_filtered, seg_filtered.max()) == (0, 0)
        assert (n_kept, seg_kept.max()) == (1, 1)

    def test_output_shape_matches_input(self):
        raw = np.ones((25, 30), dtype=np.float32)
        roi = np.ones((25, 30), dtype=bool)
        seg, n = _detect_puncta(
            raw, roi, sigma=3, alpha=1.5, min_size_fg_objects=2, peak_min_distance=2
        )
        assert seg.shape == raw.shape

    def test_blurred_mask_reuse_gives_same_result(self):
        from scipy.ndimage import gaussian_filter as gf

        raw = np.ones((25, 25), dtype=np.float32) * 0.1
        roi = np.ones((25, 25), dtype=bool)
        raw[10:13, 10:13] = 0.8
        mask_f = roi.astype(np.float32)
        bm = gf(mask_f, sigma=5, mode="constant", cval=0)
        np.clip(bm, 1e-10, None, out=bm)
        seg1, n1 = _detect_puncta(
            raw, roi, sigma=5, alpha=1.5, min_size_fg_objects=2, peak_min_distance=2
        )
        seg2, n2 = _detect_puncta(
            raw,
            roi,
            sigma=5,
            alpha=1.5,
            min_size_fg_objects=2,
            peak_min_distance=2,
            blurred_mask=bm,
        )
        assert np.array_equal(seg1, seg2)
        assert n1 == n2

    @staticmethod
    def _two_spots(separation):
        """Two Gaussian spots (peak 0.9) on a 0.1 background, `separation` px apart."""
        yy, xx = np.mgrid[0:40, 0:40]
        raw = 0.1 + 0.8 * (
            np.exp(-((yy - 20) ** 2 + (xx - 20 + separation / 2) ** 2) / 4)
            + np.exp(-((yy - 20) ** 2 + (xx - 20 - separation / 2) ** 2) / 4)
        )
        return raw.astype(np.float32), np.ones((40, 40), dtype=bool)

    def test_watershed_splits_one_blob_into_two_puncta(self):
        # Spots 4 px apart merge into one foreground blob; watershed splits it.
        raw, roi = self._two_spots(4)
        seg, n = _detect_puncta(raw, roi, 5.0, 1.5, 2, 2)
        assert (n, seg.max()) == (1, 2)

    def test_larger_peak_min_distance_keeps_close_spots_together(self):
        # Peaks 4 px apart are closer than peak_min_distance=6: one seed, one punctum.
        raw, roi = self._two_spots(4)
        seg, n = _detect_puncta(raw, roi, 5.0, 1.5, 2, 6)
        assert (n, seg.max()) == (1, 1)

    def test_hole_fill_completing_the_roi_reports_no_puncta(self):
        # Two dim pixels inside a uniformly bright cell: with sigma 0 and alpha 0.9 all
        # other pixels pass, so the ROI is not entirely foreground until the size
        # filter fills the two 1-pixel holes. Detection is then meaningless and must
        # report nothing (without the second whole-ROI check it gave 34 puncta).
        roi = np.ones((12, 12), dtype=bool)
        raw = np.full((12, 12), 0.5, dtype=np.float32)
        raw[4, 4] = raw[7, 8] = 0.1
        seg, n = _detect_puncta(raw, roi, 0, 0.9, 2, 2)
        assert (n, seg.max()) == (0, 0)

    def test_all_foreground_roi_in_two_pieces_reports_no_puncta(self):
        # Every pixel of a two-piece ROI passes; the 1-pixel piece is dropped by the
        # size filter, so only the check BEFORE the size filter can catch this.
        roi = np.zeros((14, 14), dtype=bool)
        roi[1:11, 1:11] = True
        roi[13, 13] = True
        raw = np.where(roi, 0.5, 0).astype(np.float32)
        raw[5, 5] = 0.45
        seg, n = _detect_puncta(raw, roi, 0, 0.5, 2, 2)
        assert (n, seg.max()) == (0, 0)

    def test_alpha_one_flat_signal_has_no_foreground(self):
        # alpha = 1.0 means "brighter than background". On a perfectly flat disk the
        # float32 local background can round slightly below the pixel value; without
        # the _BG_MARGIN tolerance this produced dozens of false puncta.
        yy, xx = np.mgrid[0:41, 0:41]
        disk = (yy - 20) ** 2 + (xx - 20) ** 2 <= 18**2
        for value in (0.123, 0.5, 1.0):
            raw = np.where(disk, value, 0.0).astype(np.float32)
            seg, n = _detect_puncta(raw, disk, 10.0, 1.0, 2, 2)
            assert (n, seg.max()) == (0, 0), value


# =====================================================================
# _extract_cell_features
# =====================================================================


class TestExtractCellFeatures:
    def test_returns_all_measured_keys(self):
        # Every feature except the flag, which puncta() sets itself.
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        feats = _call_extract(raw, seg, 0, inputs)
        assert set(feats.keys()) == set(_FEATURE_KEYS) - {"FlagNoNuclearCentroid"}
        assert len(feats) == 32

    def test_no_puncta_puncta_counts_zero(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        feats = _call_extract(raw, seg, 0, inputs)
        assert feats["PunctaCount"] == 0
        assert feats["TotalFgArea"] == 0
        assert feats["MeanFgIntensity"] == 0.0

    def test_no_puncta_bg_computed_from_whole_roi(self):
        # When no foreground, background = whole-ROI mean
        inputs = _make_extract_inputs()
        val = 0.3
        raw = np.ones((20, 20), dtype=np.float32) * val
        seg = np.zeros((20, 20), dtype=int)
        feats = _call_extract(raw, seg, 0, inputs)
        assert feats["MeanBgIntensity"] == pytest.approx(val, abs=1e-5)

    def test_single_punctum_count(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        assert feats["PunctaCount"] == 1
        assert feats["FgObjectCount"] == 1
        assert feats["TotalFgArea"] == 9

    def test_two_puncta_counts(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[3:5, 3:5] = 1
        seg[14:16, 14:16] = 2
        raw[3:5, 3:5] = 0.8
        raw[14:16, 14:16] = 0.9
        feats = _call_extract(raw, seg, 1, inputs)
        assert feats["PunctaCount"] == 2

    def test_fg_intensity_higher_than_bg(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        assert feats["MeanFgIntensity"] > feats["MeanBgIntensity"]

    def test_ratio_fg_bg_positive_with_signal(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        assert feats["RatioFgBgMeanIntensities"] > 1.0

    def test_fg_area_fraction_in_unit_range(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:10, 5:10] = 1
        raw[5:10, 5:10] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        assert 0.0 <= feats["TotalFgAreaFraction"] <= 1.0

    def test_ring_densities_in_unit_range(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        for key in ["DensityRingInner", "DensityRingMiddle", "DensityRingOuter"]:
            assert 0.0 <= feats[key] <= 1.0, f"{key} = {feats[key]} is out of [0, 1]"

    def test_polarization_nonneg(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        assert feats["FgPolarizationPx"] >= 0.0
        assert feats["FgPolarizationNorm"] >= 0.0

    def test_puncta_size_stats_consistent(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[3:6, 3:6] = 1  # 9 px
        seg[13:16, 13:16] = 2  # 9 px
        raw[3:6, 3:6] = 0.8
        raw[13:16, 13:16] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        assert feats["MeanPunctaSize"] == pytest.approx(9.0, abs=0.5)
        assert feats["StdPunctaSize"] == pytest.approx(0.0, abs=0.5)

    def test_no_nan_values(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        feats = _call_extract(raw, seg, 1, inputs)
        for k, v in feats.items():
            assert not np.isnan(v), f"NaN in {k}"
            assert not np.isinf(v), f"Inf in {k}"

    def test_fg_object_count_passed_through(self):
        inputs = _make_extract_inputs()
        raw = np.ones((20, 20), dtype=np.float32) * 0.1
        seg = np.zeros((20, 20), dtype=int)
        seg[5:8, 5:8] = 1
        raw[5:8, 5:8] = 0.8
        # n_fg_objects=3 passed explicitly (as if watershed split 1 blob into 1 punctum
        # but there were 3 foreground objects before watershed)
        feats = _call_extract(raw, seg, 3, inputs)
        assert feats["FgObjectCount"] == 3


# =====================================================================
# puncta() — full function integration
# =====================================================================


class TestPunctaFunction:
    def test_returns_all_features_per_channel(self):
        result = _run_puncta(n_channels=2)
        assert len(_FEATURE_KEYS) == 33
        assert len(result) == len(_FEATURE_KEYS) * 2

    def test_feature_name_format(self):
        result = _run_puncta(n_channels=1, sigma=5.0, alpha=1.5, min_size=2, pmd=2)
        for key in result:
            assert key.startswith("Puncta_"), f"Key missing Puncta_ prefix: {key}"
            assert "_s5.0_a1.5_m2_d2" in key, f"Suffix wrong in: {key}"
            # Format: Puncta_FeatureName_ChannelName_suffix
            parts = key.split("_")
            assert parts[1] in _FEATURE_KEYS and parts[2] == "Channel0"

    @staticmethod
    def _counted_cells(labels):
        """One 20x20 cell per label (left to right); cell k has k separated puncta."""
        label_img = np.zeros((24, 24 * len(labels)), dtype=np.int32)
        intensity = np.full(label_img.shape + (2,), 0.1, dtype=np.float32)
        for k, lab in enumerate(labels):
            c0 = 2 + 24 * k
            label_img[2:22, c0 : c0 + 20] = lab
            for s in range(k):
                y, x = 3 + 6 * (s // 2), c0 + 3 + 11 * (s % 2)
                intensity[y : y + 3, x : x + 3, 1] = 0.8
        return label_img, intensity

    def _counts(self, label_img, intensity, unique_labels):
        result = puncta(
            c=[1],
            sigma=5.0,
            channel_names=["Channel0", "Channel1"],
            unique_labels=np.asarray(unique_labels),
            label_image=label_img,
            intensity_image=intensity,
            offset=(0, 0),
        )
        return result["Puncta_PunctaCount_Channel1_s5.0_a1.5_m2_d2"]

    def test_each_row_holds_its_own_cells_values(self):
        # Cells drawn in label order 1..6 have 0..5 puncta; row i must report i.
        label_img, intensity = self._counted_cells(range(1, 7))
        counts = self._counts(label_img, intensity, range(1, 7))
        np.testing.assert_array_equal(counts, np.arange(6))

    def test_rows_follow_label_order_not_position(self):
        # Labels drawn left to right as 3, 1, 2 (0, 1, 2 puncta): rows are by label.
        label_img, intensity = self._counted_cells([3, 1, 2])
        counts = self._counts(label_img, intensity, [1, 2, 3])
        np.testing.assert_array_equal(counts, [1, 2, 0])

    def test_gap_in_labels_keeps_rows_aligned(self):
        label_img, intensity = self._counted_cells([1, 5, 9])
        counts = self._counts(label_img, intensity, [1, 5, 9])
        np.testing.assert_array_equal(counts, [0, 1, 2])

    @pytest.mark.parametrize(
        "unique_labels",
        [[1, 2, 3, 4], [1, 3], [3, 2, 1]],
        ids=["listed_but_absent", "present_but_unlisted", "unsorted"],
    )
    def test_label_mismatch_raises(self, unique_labels):
        # Would otherwise shift rows silently (e.g. label 3's values on label 2's row).
        label_img, intensity = self._counted_cells([1, 2, 3])
        with pytest.raises(ValueError, match="unique_labels must equal"):
            self._counts(label_img, intensity, unique_labels)

    @staticmethod
    def _sigma_zero(alpha):
        label_img = np.zeros((40, 40), dtype=np.int32)
        label_img[2:38, 2:38] = 1
        intensity = np.full((40, 40, 2), 0.1, dtype=np.float32)
        intensity[8:11, 8:11, 1] = 0.8
        intensity[28:31, 28:31, 1] = 0.8
        return puncta(
            c=[1],
            sigma=0.0,
            alpha=alpha,
            min_size_fg_objects=2,
            peak_min_distance=2,
            channel_names=["Channel0", "Channel1"],
            unique_labels=np.array([1]),
            label_image=label_img,
            intensity_image=intensity,
            offset=(0, 0),
        )

    def test_sigma_zero_detects_puncta_and_names_columns(self):
        # sigma = 0 skips the high-pass filter and thresholds against the ROI mean.
        result = self._sigma_zero(alpha=1.5)
        assert all(k.endswith("_Channel1_s0.0_a1.5_m2_d2") for k in result)
        assert result["Puncta_PunctaCount_Channel1_s0.0_a1.5_m2_d2"][0] == 2

    def test_sigma_zero_allows_alpha_below_one(self):
        # alpha < 1 is rejected only when sigma > 0. Here every ROI pixel exceeds
        # 0.9 x the ROI mean, so the whole-ROI guard reports no puncta.
        result = self._sigma_zero(alpha=0.9)
        assert result["Puncta_PunctaCount_Channel1_s0.0_a0.9_m2_d2"][0] == 0

    def test_all_arrays_length_equals_n_cells(self):
        n_cells = 3
        result = _run_puncta(n_cells=n_cells, n_channels=2)
        for key, arr in result.items():
            assert len(arr) == n_cells, (
                f"{key}: expected length {n_cells}, got {len(arr)}"
            )

    def test_all_outputs_finite(self):
        result = _run_puncta(n_cells=2, n_channels=2)
        for key, arr in result.items():
            assert np.all(np.isfinite(arr)), f"NaN or Inf in {key}"

    def test_puncta_count_at_least_fg_object_count(self):
        # Watershed only splits foreground components and the merge only rejoins
        # touching pieces of one component, so PunctaCount >= FgObjectCount.
        result = _run_puncta(n_cells=2, n_channels=1)
        pc = next(v for k, v in result.items() if "PunctaCount" in k)
        fg = next(v for k, v in result.items() if "FgObjectCount" in k)
        assert np.all(pc >= fg) and np.all(fg >= 0)

    def test_validation_error_propagated(self):
        with pytest.raises(ValueError, match="alpha must be >= 1 when sigma > 0"):
            _run_puncta(sigma=5.0, alpha=0.5)

    def test_peak_min_distance_float_integer_accepted(self):
        # 2.0 is a valid integer-valued float; it is stored as the integer 2, so the
        # feature names end in "_d2", not "_d2.0".
        result = _run_puncta(pmd=2.0)
        assert all(k.endswith("_d2") for k in result)

    def test_unknown_kwargs_absorbed(self):
        # puncta() must accept arbitrary **kwargs (scallops passes extra ones)
        label_img, intensity = _make_label_and_intensity(n_cells=1, n_channels=1)
        result = puncta(
            c=[0],
            sigma=5.0,
            alpha=1.5,
            channel_names=["Channel0"],
            unique_labels=np.array([1]),
            label_image=label_img,
            intensity_image=intensity,
            offset=(0, 0),
            label_filter=None,  # passed by _feature_block in newer scallops
            some_future_kwarg=True,  # simulated future kwarg
        )
        assert len(result) == len(_FEATURE_KEYS)

    def test_all_zero_intensity_all_features_zero(self):
        label_img = np.zeros((30, 30), dtype=int)
        label_img[5:25, 5:25] = 1
        intensity = np.zeros((30, 30, 1), dtype=np.float32)
        result = puncta(
            c=[0],
            sigma=5.0,
            alpha=1.5,
            channel_names=["Channel0"],
            unique_labels=np.array([1]),
            label_image=label_img,
            intensity_image=intensity,
            offset=(0, 0),
        )
        # Every measured feature is zero; only the flag is 1 (no nuclei were given,
        # so the ROI centroid was used).
        for key, arr in result.items():
            expected = 1.0 if "FlagNoNuclearCentroid" in key else 0.0
            assert arr[0] == expected, key

    def test_fraction_features_in_unit_range(self):
        result = _run_puncta(n_cells=2, n_channels=2)
        for key, arr in result.items():
            if "Fraction" in key or "Density" in key:
                assert np.all(arr >= 0.0), f"{key} has negative value"
                assert np.all(arr <= 1.0), f"{key} exceeds 1.0"

    def test_count_and_area_features_nonneg(self):
        result = _run_puncta(n_cells=2, n_channels=2)
        for key, arr in result.items():
            if "Count" in key or "Area" in key:
                assert np.all(arr >= 0), f"{key} has negative value"

    def test_offset_none_runs_ok(self):
        label_img, intensity = _make_label_and_intensity(n_cells=1, n_channels=1)
        result = puncta(
            c=[0],
            sigma=5.0,
            alpha=1.5,
            channel_names=["Channel0"],
            unique_labels=np.array([1]),
            label_image=label_img,
            intensity_image=intensity,
            offset=None,
        )
        assert len(result) == len(_FEATURE_KEYS)

    def test_unique_labels_original_none_uses_unique_labels(self):
        label_img, intensity = _make_label_and_intensity(n_cells=1, n_channels=1)
        result = puncta(
            c=[0],
            sigma=5.0,
            alpha=1.5,
            channel_names=["Channel0"],
            unique_labels=np.array([1]),
            unique_labels_original=None,
            label_image=label_img,
            intensity_image=intensity,
            offset=(0, 0),
            nuclei_label_image=label_img,  # nucleus IDs = the sequential labels
        )
        assert len(result) == len(_FEATURE_KEYS)
        # Without unique_labels_original (or label_image_original) the nucleus is
        # looked up by the sequential label, which here matches: anchored, flag 0.
        flag = next(v for k, v in result.items() if "FlagNoNuclearCentroid" in k)
        assert flag[0] == 0


# =====================================================================
# Nucleus anchor (nuclei_label_image)
# =====================================================================

_LABEL = 7  # non-sequential original ID, as in scallops blocks
_SUFFIX = "Channel1_s5.0_a1.5_m2_d2"
_CELL_BOX = (10, 40, 10, 40)  # rows 10-39, cols 10-39


def _nucleus_scene(nucleus_box, cytoplasm=False, shape=(60, 60)):
    """Full-image ROI labels, nuclei labels, and intensity for one cell.

    Boxes are (row0, row1, col0, col1). With cytoplasm=True the ROI is the cell minus
    the nucleus. Channel 1 has two bright spots inside the cytoplasm.
    """
    r0, r1, c0, c1 = _CELL_BOX
    cells = np.zeros(shape, dtype=np.int32)
    cells[r0:r1, c0:c1] = _LABEL
    nuclei = np.zeros(shape, dtype=np.int32)
    if nucleus_box is not None:
        n0, n1, m0, m1 = nucleus_box
        nuclei[n0:n1, m0:m1] = _LABEL
    rois = np.where(nuclei > 0, 0, cells) if cytoplasm else cells
    intensity = np.full(shape + (2,), 0.1, dtype=np.float32)
    intensity[13:16, 13:16, 1] = 0.6
    intensity[33:36, 13:16, 1] = 0.6
    return rois, nuclei, intensity


def _run_block(rois, nuclei, intensity, block=None, pass_original=True):
    """Call puncta() on one block of the full image, the way _feature_block does."""
    y0, y1, x0, x1 = (
        block if block is not None else (0, rois.shape[0], 0, rois.shape[1])
    )
    original = rois[y0:y1, x0:x1].copy()
    sequential = (original == _LABEL).astype(np.int32)
    return puncta(
        c=[1],
        sigma=5.0,
        alpha=1.5,
        min_size_fg_objects=2,
        peak_min_distance=2,
        channel_names=["Channel0", "Channel1"],
        unique_labels=np.array([1]),
        unique_labels_original=np.array([_LABEL]),
        label_image=sequential,
        label_image_original=original if pass_original else None,
        intensity_image=intensity[y0:y1, x0:x1],
        offset=(y0, x0),
        nuclei_label_image=nuclei,
    )


def _feat(result, name):
    return result[f"Puncta_{name}_{_SUFFIX}"][0]


def _roi_crop(rois):
    ys, xs = np.nonzero(rois == _LABEL)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def _fg_pixels(rois, intensity):
    """Foreground pixel coordinates (full image) exactly as puncta() detects them."""
    crop = _roi_crop(rois)
    roi = rois[crop] == _LABEL
    seg, _ = _detect_puncta(intensity[crop][..., 1], roi, 5.0, 1.5, 2, 2)
    fy, fx = np.nonzero(seg)
    return fy + crop[0].start, fx + crop[1].start


def _mean_boundary_dist(mask, anchor):
    """Mean distance from anchor to the inner boundary of mask (full-image coords)."""
    inner = mask & ~minimum_filter(mask, footprint=_CROSS, mode="constant", cval=False)
    by, bx = np.nonzero(inner)
    return float(np.mean(np.hypot(by - anchor[0], bx - anchor[1])))


_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _assert_same_features(a, b, ignore=()):
    assert a.keys() == b.keys()
    for k in a:
        if not any(name in k for name in ignore):
            np.testing.assert_allclose(a[k], b[k], err_msg=k)


class TestNucleusAnchor:
    def test_anchor_is_nucleus_centroid(self):
        rois, nuclei, intensity = _nucleus_scene((18, 26, 24, 32))
        r = _run_block(rois, nuclei, intensity)
        fy, fx = _fg_pixels(rois, intensity)
        ay, ax = 21.5, 27.5  # centroid of the nucleus box
        assert _feat(r, "MeanFgRadialPosPx") == pytest.approx(
            np.mean(np.hypot(fy - ay, fx - ax))
        )
        assert _feat(r, "FgPolarizationPx") == pytest.approx(
            np.hypot(fy.mean() - ay, fx.mean() - ax)
        )

    def test_detection_features_unchanged_by_nucleus(self):
        rois, nuclei, intensity = _nucleus_scene((18, 26, 24, 32))
        with_nuc = _run_block(rois, nuclei, intensity)
        without = _run_block(rois, None, intensity)
        spatial = ("Radial", "Extent", "Polarization", "Density", "Flag")
        for k in with_nuc:
            if not any(s in k for s in spatial):
                np.testing.assert_allclose(with_nuc[k], without[k], err_msg=k)
        assert _feat(with_nuc, "FgPolarizationPx") != pytest.approx(
            _feat(without, "FgPolarizationPx")
        )

    def test_nucleus_at_roi_centroid_matches_no_nuclei(self):
        # Cell centroid is (24.5, 24.5); a centred nucleus gives the same anchor.
        rois, nuclei, intensity = _nucleus_scene((20, 30, 20, 30))
        _assert_same_features(
            _run_block(rois, nuclei, intensity),
            _run_block(rois, None, intensity),
            ignore=("FlagNoNuclearCentroid",),
        )

    def test_missing_nucleus_falls_back_to_roi_centroid(self):
        # No nucleus with the cell's ID: same result as passing no nuclei labels at
        # all, and both are flagged.
        rois, nuclei, intensity = _nucleus_scene(None)
        with_nuclei = _run_block(rois, nuclei, intensity)
        no_labels = _run_block(rois, None, intensity)
        _assert_same_features(with_nuclei, no_labels)
        assert _feat(with_nuclei, "FlagNoNuclearCentroid") == 1
        assert _feat(no_labels, "FlagNoNuclearCentroid") == 1

    def test_nucleus_not_touching_roi_falls_back(self):
        # Same-ID nucleus that belongs to another cell (outside this cell).
        rois, nuclei, intensity = _nucleus_scene((45, 55, 45, 55))
        with_nuclei = _run_block(rois, nuclei, intensity)
        _assert_same_features(with_nuclei, _run_block(rois, None, intensity))
        assert _feat(with_nuclei, "FlagNoNuclearCentroid") == 1

    def test_nucleus_inside_bbox_but_not_touching_falls_back(self):
        # L-shaped cell; the same-ID nucleus sits in the empty corner of its bounding
        # box, separated from the cell by a 1-pixel gap (row 24 and column 25 are empty).
        rois, nuclei, intensity = _nucleus_scene(None)
        rois[10:25, 25:40] = 0
        nuclei[10:24, 26:40] = _LABEL
        with_nuclei = _run_block(rois, nuclei, intensity)
        _assert_same_features(with_nuclei, _run_block(rois, None, intensity))
        assert _feat(with_nuclei, "FlagNoNuclearCentroid") == 1

    def test_original_id_from_unique_labels_original(self):
        rois, nuclei, intensity = _nucleus_scene((18, 26, 24, 32))
        _assert_same_features(
            _run_block(rois, nuclei, intensity, pass_original=False),
            _run_block(rois, nuclei, intensity),
        )

    def test_cytoplasm_enclosed_nucleus_uses_whole_cell_boundary(self):
        rois, nuclei, intensity = _nucleus_scene((20, 30, 20, 30), cytoplasm=True)
        r = _run_block(rois, nuclei, intensity)
        anchor = (24.5, 24.5)
        cell = np.zeros(rois.shape, dtype=bool)
        cell[10:40, 10:40] = True
        norm_cell = _mean_boundary_dist(cell, anchor)
        assert _feat(r, "FgPolarizationNorm") == pytest.approx(
            _feat(r, "FgPolarizationPx") / norm_cell
        )
        # The cytoplasm's own boundary (incl. the nuclear hole) gives a different value
        assert norm_cell != pytest.approx(_mean_boundary_dist(rois == _LABEL, anchor))

    def test_crescent_cytoplasm_extends_frame_and_uses_whole_cell(self):
        # Nucleus fills the cell's right edge: the cytoplasm is cols 10-33 and the
        # nucleus (cols 34-39) lies entirely outside the cytoplasm's bounding box.
        rois, nuclei, intensity = _nucleus_scene((10, 40, 34, 40), cytoplasm=True)
        r = _run_block(rois, nuclei, intensity)
        fy, fx = _fg_pixels(rois, intensity)
        anchor = (24.5, 36.5)
        assert _feat(r, "FgPolarizationPx") == pytest.approx(
            np.hypot(fy.mean() - anchor[0], fx.mean() - anchor[1])
        )
        cell = np.zeros(rois.shape, dtype=bool)
        cell[10:40, 10:40] = True
        assert _feat(r, "FgPolarizationNorm") == pytest.approx(
            _feat(r, "FgPolarizationPx") / _mean_boundary_dist(cell, anchor)
        )

    @pytest.mark.parametrize("array_type", ["numpy", "zarr", "dask"])
    def test_nucleus_beyond_block_read_from_full_image(self, array_type):
        rois, nuclei, intensity = _nucleus_scene((10, 40, 34, 40), cytoplasm=True)
        if array_type == "zarr":
            import zarr

            full = zarr.array(nuclei)
        elif array_type == "dask":
            import dask.array as da

            full = da.from_array(nuclei, chunks=16)
        else:
            full = nuclei
        # Block = the cytoplasm's bounding box only; the nucleus is outside it.
        small = _run_block(rois, full, intensity, block=(10, 40, 10, 34))
        _assert_same_features(small, _run_block(rois, nuclei, intensity))

    def test_flag_zero_when_nucleus_used(self):
        rois, nuclei, intensity = _nucleus_scene((18, 26, 24, 32))
        assert _feat(_run_block(rois, nuclei, intensity), "FlagNoNuclearCentroid") == 0

    def test_flag_same_for_every_channel(self):
        label_img, intensity = _make_label_and_intensity(n_cells=2, n_channels=3)
        nuclei = np.where(label_img == 1, 1, 0)  # cell 1 has a nucleus, cell 2 not
        r = puncta(
            c=[0, 1, 2],
            sigma=5.0,
            channel_names=["Channel0", "Channel1", "Channel2"],
            unique_labels=np.array([1, 2]),
            label_image=label_img,
            intensity_image=intensity,
            offset=(0, 0),
            nuclei_label_image=nuclei,
        )
        flags = [v for k, v in r.items() if "FlagNoNuclearCentroid" in k]
        assert len(flags) == 3
        for f in flags:
            np.testing.assert_array_equal(f, [0.0, 1.0])

    def test_nuclei_roi_finds_itself(self):
        # Nuclei ROIs: passing the nuclei labels gives flag 0 and identical features.
        rois, nuclei, intensity = _nucleus_scene((18, 26, 24, 32))
        r_with = _run_block(nuclei, nuclei, intensity)
        r_without = _run_block(nuclei, None, intensity)
        assert _feat(r_with, "FlagNoNuclearCentroid") == 0
        assert _feat(r_without, "FlagNoNuclearCentroid") == 1
        _assert_same_features(r_with, r_without, ignore=("FlagNoNuclearCentroid",))


class TestGeometryMask:
    def test_anchor_inside_roi_uses_filled_roi(self):
        roi = np.zeros((20, 20), dtype=bool)
        roi[2:18, 2:18] = True
        roi[8:12, 8:12] = False  # hole
        filled = np.zeros_like(roi)
        filled[2:18, 2:18] = True
        np.testing.assert_array_equal(_geometry_mask(roi, (5.0, 5.0)), filled)

    def test_cytoplasm_with_nucleus_uses_union(self):
        cell = np.zeros((20, 20), dtype=bool)
        cell[2:18, 2:18] = True
        nucleus = np.zeros_like(cell)
        nucleus[2:18, 12:18] = True  # touches the cell edge: crescent cytoplasm
        roi = cell & ~nucleus
        geom = _geometry_mask(roi, (9.5, 14.5), nucleus)
        np.testing.assert_array_equal(geom, cell)

    def test_anchor_outside_nonconvex_roi_uses_convex_hull(self):
        roi = np.zeros((20, 20), dtype=bool)  # U shape; centroid in the gap
        roi[2:18, 2:6] = True
        roi[2:18, 14:18] = True
        roi[14:18, 2:18] = True
        geom = _geometry_mask(roi, (10.0, 10.0))
        assert geom[10, 10] and geom[roi].all()


class TestFindNucleus:
    def test_window_grows_until_whole_nucleus_inside(self):
        nuclei = np.zeros((100, 100), dtype=np.int32)
        nuclei[40:45, 20:90] = 3  # long nucleus reaching far beyond the ROI box
        roi = np.ones((10, 10), dtype=bool)  # ROI at rows 38-47, cols 20-29
        block = nuclei[30:60, 15:35]
        roi_frame, nucleus_frame, (ry, rx) = _find_nucleus(
            nuclei, block, (30, 15), (38, 20), roi, 3
        )
        assert nucleus_frame.sum() == (nuclei == 3).sum()
        assert roi_frame[ry : ry + 10, rx : rx + 10].all()
        assert roi_frame.sum() == roi.sum()

    def test_diagonal_contact_is_not_touching(self):
        nuclei = np.zeros((20, 20), dtype=np.int32)
        nuclei[10, 10] = 3
        roi = np.zeros((3, 3), dtype=bool)
        roi[0, 0] = True  # ROI pixel at (11, 11): diagonal neighbour only
        roi[2, 2] = True
        assert _find_nucleus(nuclei, nuclei, (0, 0), (11, 11), roi, 3) is None


class TestNucleiLabelMatchFraction:
    def _scene(self):
        cells = np.zeros((40, 40), dtype=np.int32)
        nuclei = np.zeros_like(cells)
        for i, (r, c) in enumerate([(2, 2), (2, 22), (22, 2), (22, 22)], start=1):
            cells[r : r + 16, c : c + 16] = i
            nuclei[r + 5 : r + 11, c + 5 : c + 11] = i
        return cells, nuclei

    def test_cells_grown_from_nuclei_match(self):
        cells, nuclei = self._scene()
        region = (slice(0, 40), slice(0, 40))
        assert nuclei_label_match_fraction(cells, nuclei, region) == 1.0

    def test_cytoplasm_touching_nuclei_matches(self):
        cells, nuclei = self._scene()
        cytoplasm = np.where(nuclei > 0, 0, cells)
        region = (slice(0, 40), slice(0, 40))
        assert nuclei_label_match_fraction(cytoplasm, nuclei, region) == 1.0

    def test_independent_numbering_does_not_match(self):
        cells, nuclei = self._scene()
        scrambled = np.array([0, 3, 4, 1, 2])[cells]
        region = (slice(0, 40), slice(0, 40))
        frac = nuclei_label_match_fraction(scrambled, nuclei, region)
        assert frac < NUCLEI_MATCH_MIN_FRACTION

    def test_empty_region_is_zero(self):
        cells, nuclei = self._scene()
        assert (
            nuclei_label_match_fraction(cells, nuclei, (slice(0, 2), slice(0, 2))) == 0
        )

    def test_labels_excludes_rois_cut_by_region_edge(self):
        # Columns 0-25: cells 1 and 3 (cols 2-17) are whole; cells 2 and 4 (cols 22-37)
        # are cut at col 26, before their nuclei (cols 27-32) start.
        cells, nuclei = self._scene()
        region = (slice(0, 40), slice(0, 26))
        assert nuclei_label_match_fraction(cells, nuclei, region) == 0.5
        assert nuclei_label_match_fraction(cells, nuclei, region, labels=[1, 3]) == 1.0


# =====================================================================
# label_features integration (nuclei_label_image)
# =====================================================================

_LF_FEATURE = "puncta_1_5_1.5_2_2"
_LF_IDS = np.array([3, 17, 42, 1000, 2001, 5005])  # non-sequential, as in real images
# (row0, row1, col0, col1) per cell. 2001 straddles the 64-px chunk edge in both axes.
_LF_CELLS = {
    3: (4, 36, 4, 36),
    17: (4, 36, 44, 76),
    42: (4, 36, 84, 124),
    1000: (44, 84, 4, 36),
    2001: (44, 84, 44, 76),
    5005: (92, 124, 50, 120),
}
# 42's nucleus extends past its cell to the image edge (read beyond the block);
# 5005 has no nucleus (ROI-centroid fallback).
_LF_NUCLEI = {
    3: (10, 22, 8, 20),
    17: (12, 24, 50, 62),
    42: (14, 26, 116, 128),
    1000: (58, 74, 10, 26),
    2001: (56, 72, 56, 72),
}
# Detection and intensity features: do not depend on the spatial anchor.
_LF_DETECTION_KEYS = _FEATURE_KEYS[: _FEATURE_KEYS.index("MeanFgRadialPosPx")]


def _lf_scene(cytoplasm=False):
    """Cells (or cytoplasm), nuclei, and a 2-channel intensity image (128 x 128)."""
    rng = np.random.default_rng(0)
    shape = (128, 128)
    cells = np.zeros(shape, dtype=np.int32)
    nuclei = np.zeros(shape, dtype=np.int32)
    intensity = (0.1 + 0.01 * rng.random(shape + (2,))).astype(np.float32)
    for label, (r0, r1, c0, c1) in _LF_CELLS.items():
        cells[r0:r1, c0:c1] = label
        for _ in range(4):
            y = rng.integers(r0 + 2, r1 - 5)
            x = rng.integers(c0 + 2, c1 - 5)
            intensity[y : y + 3, x : x + 3, 1] = 0.6
    for label, (r0, r1, c0, c1) in _LF_NUCLEI.items():
        nuclei[r0:r1, c0:c1] = label
    rois = np.where(nuclei > 0, 0, cells) if cytoplasm else cells
    return rois, nuclei, intensity


def _lf_run(rois, intensity, nuclei=None, features=(_LF_FEATURE,), chunks=64):
    """Run label_features on dask inputs, as the features CLI does."""
    objects_df = find_objects(rois)
    if isinstance(objects_df, dd.DataFrame):
        objects_df = objects_df.compute()
    df = label_features(
        objects_df=objects_df,
        label_image=da.from_array(rois, chunks=chunks),
        intensity_image=da.from_array(intensity, chunks=(chunks, chunks, -1)),
        features=list(features),
        normalize=False,
        nuclei_label_image=nuclei,
    )
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    return df.sort_index()


def _lf_direct(rois, intensity, nuclei):
    """Call puncta() once on the whole image (a single block) as the reference."""
    ids = np.unique(rois)
    ids = ids[ids > 0]
    sequential = np.searchsorted(ids, rois) + 1
    sequential[rois == 0] = 0
    result = puncta(
        c=[1],
        sigma=5.0,
        alpha=1.5,
        min_size_fg_objects=2,
        peak_min_distance=2,
        channel_names=["Channel0", "Channel1"],
        unique_labels=np.arange(1, len(ids) + 1),
        unique_labels_original=ids,
        label_image=sequential,
        label_image_original=rois,
        intensity_image=intensity,
        offset=(0, 0),
        nuclei_label_image=nuclei,
    )
    return pd.DataFrame(result, index=ids)


def _lf_col(key):
    return f"Puncta_{key}_{_SUFFIX}"


def _assert_frames_equal(a, b, keys=None):
    columns = a.columns if keys is None else [_lf_col(k) for k in keys]
    np.testing.assert_array_equal(a.index.values, b.index.values)
    for column in columns:
        np.testing.assert_allclose(
            a[column].to_numpy(dtype=float),
            b[column].to_numpy(dtype=float),
            equal_nan=True,
            err_msg=column,
        )


class TestLabelFeaturesNuclei:
    @pytest.mark.parametrize("cytoplasm", [False, True])
    def test_matches_direct_call(self, cytoplasm):
        rois, nuclei, intensity = _lf_scene(cytoplasm)
        result = _lf_run(rois, intensity, nuclei)
        expected = _lf_direct(rois, intensity, nuclei)
        assert len(result.columns) == len(_FEATURE_KEYS)
        _assert_frames_equal(result, expected[result.columns])

    @pytest.mark.parametrize("cytoplasm", [False, True])
    @pytest.mark.parametrize("chunks", [32, 128])
    def test_chunking_does_not_change_results(self, cytoplasm, chunks):
        rois, nuclei, intensity = _lf_scene(cytoplasm)
        _assert_frames_equal(
            _lf_run(rois, intensity, nuclei, chunks=chunks),
            _lf_run(rois, intensity, nuclei, chunks=64),
        )

    @pytest.mark.parametrize("cytoplasm", [False, True])
    def test_flag(self, cytoplasm):
        rois, nuclei, intensity = _lf_scene(cytoplasm)
        flag = _lf_run(rois, intensity, nuclei)[_lf_col("FlagNoNuclearCentroid")]
        expected = [0.0 if label in _LF_NUCLEI else 1.0 for label in _LF_IDS]
        np.testing.assert_array_equal(flag.to_numpy(), expected)

    def test_without_nuclei_all_flagged_detection_unchanged(self):
        rois, nuclei, intensity = _lf_scene(cytoplasm=True)
        with_nuclei = _lf_run(rois, intensity, nuclei)
        without = _lf_run(rois, intensity, None)
        assert (without[_lf_col("FlagNoNuclearCentroid")] == 1).all()
        _assert_frames_equal(with_nuclei, without, keys=_LF_DETECTION_KEYS)
        # the spatial features do change for the cells that have a nucleus
        radial = _lf_col("MeanFgRadialPosPx")
        assert not np.allclose(with_nuclei[radial], without[radial])

    def test_independently_numbered_cells_disable_nuclei(self, caplog):
        rois, nuclei, intensity = _lf_scene()
        renumbered = np.where(rois > 0, rois + 10000, 0)  # cellpose-like IDs
        with caplog.at_level(logging.WARNING, logger="scallops"):
            result = _lf_run(renumbered, intensity, nuclei)
        assert "puncta spatial features will use each ROI's centroid" in caplog.text
        _assert_frames_equal(result, _lf_run(renumbered, intensity, None))
        assert (result[_lf_col("FlagNoNuclearCentroid")] == 1).all()

    def test_nuclei_rois_find_themselves(self):
        _, nuclei, intensity = _lf_scene()
        with_nuclei = _lf_run(nuclei, intensity, nuclei)
        without = _lf_run(nuclei, intensity, None)
        assert (with_nuclei[_lf_col("FlagNoNuclearCentroid")] == 0).all()
        keys = [k for k in _FEATURE_KEYS if k != "FlagNoNuclearCentroid"]
        _assert_frames_equal(with_nuclei, without, keys=keys)

    @pytest.mark.parametrize("kind", ["zarr", "dask"])
    def test_nuclei_array_types(self, kind):
        rois, nuclei, intensity = _lf_scene(cytoplasm=True)
        if kind == "zarr":
            nuclei_array = zarr.create_array(
                store={}, shape=nuclei.shape, chunks=(32, 32), dtype=nuclei.dtype
            )
            nuclei_array[:] = nuclei
        else:
            nuclei_array = da.from_array(nuclei, chunks=32)
        _assert_frames_equal(
            _lf_run(rois, intensity, nuclei_array), _lf_run(rois, intensity, nuclei)
        )

    def test_other_features_unchanged(self):
        rois, nuclei, intensity = _lf_scene(cytoplasm=True)
        others = ["intensity_1", "sizeshape"]
        alone = _lf_run(rois, intensity, None, features=others)
        combined = _lf_run(rois, intensity, nuclei, features=others + [_LF_FEATURE])
        assert len(combined.columns) == len(alone.columns) + len(_FEATURE_KEYS)
        pd.testing.assert_frame_equal(alone, combined[alone.columns])

    def test_nuclei_ignored_without_puncta(self):
        rois, _, intensity = _lf_scene()
        wrong_shape = np.zeros((10, 10), dtype=np.int32)  # would fail the shape check
        result = _lf_run(rois, intensity, wrong_shape, features=["intensity_1"])
        pd.testing.assert_frame_equal(
            result, _lf_run(rois, intensity, None, features=["intensity_1"])
        )

    def test_two_parameter_sets(self):
        rois, nuclei, intensity = _lf_scene()
        result = _lf_run(
            rois, intensity, nuclei, features=[_LF_FEATURE, "puncta_1_10_2_2_2"]
        )
        assert len(result.columns) == 2 * len(_FEATURE_KEYS)
        suffixes = {c.split("_", 2)[2] for c in result.columns}
        assert suffixes == {"Channel1_s5.0_a1.5_m2_d2", "Channel1_s10.0_a2.0_m2_d2"}

    def test_only_puncta_receives_nuclei(self):
        rois, nuclei, _ = _lf_scene()
        funcs, _ = _create_funcs(
            [_LF_FEATURE, "intensity_1", "sizeshape"], n_channels=2
        )
        objects_df = find_objects(rois)
        min_cols = ["AreaShape_BoundingBoxMinimum_Y", "AreaShape_BoundingBoxMinimum_X"]
        max_cols = ["AreaShape_BoundingBoxMaximum_Y", "AreaShape_BoundingBoxMaximum_X"]
        bound = _bind_nuclei_labels(
            funcs, rois, nuclei, objects_df, min_cols, max_cols, ((64, 64), (64, 64))
        )
        for f in bound:
            assert ("nuclei_label_image" in f.keywords) == (f.func is puncta)
        assert sum(f.func is puncta for f in bound) == 1

    @staticmethod
    def _grid_scene():
        """16 cells (40 x 40) with scattered IDs on a 4 x 4 grid crossing 64-px chunks.

        Cell k has k % 6 puncta in channel 1; channel 0 encodes the label ID.
        """
        rng = np.random.default_rng(1)
        cells = np.zeros((200, 200), dtype=np.int32)
        nuclei = np.zeros_like(cells)
        image = (0.1 + 0.005 * rng.random((200, 200, 2))).astype(np.float32)
        ids = rng.permutation(np.arange(2, 5000))[:16]
        truth = {}
        for k, label in enumerate(ids):
            r0, c0 = 8 + (k // 4) * 48, 8 + (k % 4) * 48
            cells[r0 : r0 + 40, c0 : c0 + 40] = label
            nuclei[r0 + 15 : r0 + 25, c0 + 15 : c0 + 25] = label
            for s in range(k % 6):
                y, x = r0 + 3 + 7 * s, c0 + 3 + (s % 2) * 26
                image[y : y + 3, x : x + 3, 1] = 0.8
            image[r0 : r0 + 40, c0 : c0 + 40, 0] = label / 10000.0
            truth[label] = k % 6
        return cells, nuclei, image, truth

    @pytest.mark.parametrize("chunks", [64, 200])
    def test_each_row_holds_its_own_cells_values(self, chunks):
        cells, nuclei, image, truth = self._grid_scene()
        df = _lf_run(
            cells, image, nuclei, features=[_LF_FEATURE, "intensity_0"], chunks=chunks
        )
        assert sorted(df.index) == sorted(truth)
        counts = df[_lf_col("PunctaCount")]
        assert all(counts[label] == n for label, n in truth.items())
        # the intensity module's row for each label carries that label's own ID
        decoded = (df["Intensity_MeanIntensity_Channel0"] * 10000).round().astype(int)
        np.testing.assert_array_equal(decoded.to_numpy(), df.index.to_numpy())

    def test_cells_cut_by_sampled_chunk_do_not_disable_nuclei(self, caplog):
        # With 64-px chunks most cells cross a chunk edge, often with their nucleus
        # outside the densest chunk; only the ROIs assigned to it, read whole, count.
        cells, nuclei, image, _ = self._grid_scene()
        with caplog.at_level(logging.WARNING, logger="scallops"):
            df = _lf_run(cells, image, nuclei, chunks=64)
        assert "sampled labels overlap" not in caplog.text
        assert (df[_lf_col("FlagNoNuclearCentroid")] == 0).all()

    @pytest.mark.parametrize("n_matched", [3, 2])
    def test_match_fraction_threshold(self, n_matched, caplog):
        # All 6 cells sampled (one chunk). 3 of 6 matched = 0.5 >= threshold: nuclei
        # used, only cells without a nucleus flagged. 2 of 6 < 0.5: nuclei dropped for
        # the whole image, every cell flagged, warning logged.
        rois, nuclei, intensity = _lf_scene()
        keep = list(_LF_NUCLEI)[:n_matched]
        nuclei = np.where(np.isin(nuclei, keep), nuclei, 0)
        with caplog.at_level(logging.WARNING, logger="scallops"):
            flag = _lf_run(rois, intensity, nuclei, chunks=128)[
                _lf_col("FlagNoNuclearCentroid")
            ]
        warned = "sampled labels overlap" in caplog.text
        if n_matched / len(_LF_IDS) >= NUCLEI_MATCH_MIN_FRACTION:
            assert not warned
            expected = [0.0 if label in keep else 1.0 for label in _LF_IDS]
        else:
            assert warned
            expected = [1.0] * len(_LF_IDS)
        np.testing.assert_array_equal(flag.to_numpy(), expected)

    def test_match_check_reads_about_one_chunk_despite_giant_object(self, monkeypatch):
        # A thin L-shaped object starts in chunk (0, 0) but spans the whole image.
        # The check must skip it, so the region read stays within ~2 chunks.
        from scallops.features import generate

        cells = np.zeros((256, 256), dtype=np.int32)
        nuclei = np.zeros_like(cells)
        for k, (r, c) in enumerate([(4, 4), (4, 26), (26, 4)], start=1):
            cells[r : r + 18, c : c + 18] = k
            nuclei[r + 6 : r + 12, c + 6 : c + 12] = k
        cells[2, 30:250] = 99  # giant object: bbox from (2, 30) to (250, 250)
        cells[2:250, 249] = 99
        seen = {}

        def spy(label_image, nuclei_label_image, region, labels=None):
            seen["region"], seen["labels"] = region, labels
            return nuclei_label_match_fraction(
                label_image, nuclei_label_image, region, labels
            )

        monkeypatch.setattr(generate, "nuclei_label_match_fraction", spy)
        intensity = np.full((256, 256, 2), 0.1, dtype=np.float32)
        flag = _lf_run(cells, intensity, nuclei, chunks=64)[
            _lf_col("FlagNoNuclearCentroid")
        ]
        rows, cols = seen["region"]
        assert rows.stop - rows.start <= 128 and cols.stop - cols.start <= 128
        assert 99 not in seen["labels"]
        assert (flag.drop(99) == 0).all()

    def test_sigma_zero_from_feature_string(self):
        # "puncta_1_0_0.9_2_2": sigma 0 is parsed as 0.0, and alpha < 1 is accepted
        # because it is only rejected when sigma > 0.
        rois, nuclei, intensity = _lf_scene()
        df = _lf_run(rois, intensity, nuclei, features=["puncta_1_0_0.9_2_2"])
        assert all(c.endswith("_Channel1_s0.0_a0.9_m2_d2") for c in df.columns)
        with pytest.raises(ValueError, match="alpha must be >= 1 when sigma > 0"):
            _lf_run(rois, intensity, nuclei, features=["puncta_1_10_0.9_2_2"])

    def test_normalize_uint16_matches_float_input(self):
        # With normalize=True scallops converts the intensity image with
        # skimage.util.img_as_float; puncta must see exactly that.
        rois, nuclei, intensity = _lf_scene(cytoplasm=True)
        as_uint16 = (intensity * 65535).round().astype(np.uint16)
        objects_df = find_objects(rois)
        kwargs = dict(
            objects_df=objects_df,
            label_image=da.from_array(rois, chunks=64),
            features=[_LF_FEATURE],
            nuclei_label_image=nuclei,
        )
        normalized = label_features(
            intensity_image=da.from_array(as_uint16, chunks=(64, 64, -1)),
            normalize=True,
            **kwargs,
        ).compute()
        as_float = skimage.util.img_as_float(as_uint16)
        expected = label_features(
            intensity_image=da.from_array(as_float, chunks=(64, 64, -1)),
            normalize=False,
            **kwargs,
        ).compute()
        _assert_frames_equal(normalized.sort_index(), expected.sort_index())

    def test_image_smaller_than_metadata_block(self):
        # label_features computes its output metadata on a synthetic 50 x 50 block;
        # with nuclei bound, that used to read the real nuclei image and crash for
        # images under ~51 px. Results must equal a direct puncta() call.
        rois = np.zeros((40, 40), dtype=np.int32)
        rois[2:20, 2:20] = 7
        nuclei = np.zeros_like(rois)
        nuclei[8:14, 8:14] = 7
        intensity = np.full((40, 40, 2), 0.1, dtype=np.float32)
        intensity[4:7, 4:7, 1] = 0.6
        result = _lf_run(rois, intensity, nuclei)
        _assert_frames_equal(
            result, _lf_direct(rois, intensity, nuclei)[result.columns]
        )
        assert result[_lf_col("FlagNoNuclearCentroid")].iloc[0] == 0

    def test_empty_objects_table_behaves_as_without_nuclei(self):
        # With no objects there is nothing to sample; the nuclei must not change what
        # label_features does (whatever scallops itself does for an empty table).
        rois, nuclei, intensity = _lf_scene()

        def outcome(nuclei_labels):
            try:
                df = label_features(
                    objects_df=find_objects(rois).iloc[:0],
                    label_image=da.from_array(rois, chunks=64),
                    intensity_image=da.from_array(intensity, chunks=(64, 64, -1)),
                    features=[_LF_FEATURE],
                    normalize=False,
                    nuclei_label_image=nuclei_labels,
                )
                return ("result", len(df.compute() if hasattr(df, "compute") else df))
            except Exception as e:
                return (type(e).__name__, str(e))

        assert outcome(nuclei) == outcome(None)
