# Adapter from https://github.com/afermg/cp_measure/blob/main/src/cp_measure/core/measureobjectintensitydistribution.py
from collections.abc import Sequence

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import numpy as np
import scipy.ndimage
import scipy.sparse
from centrosome import zernike
from cp_measure.core.measureobjectintensitydistribution import (
    M_CATEGORY,
    MF_FRAC_AT_D,
    MF_MEAN_FRAC,
    MF_RADIAL_CV,
    OF_FRAC_AT_D,
    OF_MEAN_FRAC,
    OF_RADIAL_CV,
    _maximum_position_of_labels,
)
from numpy.typing import NDArray


def intensity_distribution_radial(
    c: Sequence[int],
    bin_count: int = 4,
    scaled: bool = True,
    maximum_radius: int = 100,
    channel_names: Sequence[str] = None,
    unique_labels: np.ndarray = None,
    label_image: np.ndarray = None,
    intensity_image: np.ndarray = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    # reimplemented from cp_measure to compute information about labels once rather than once per channel
    return get_radial_distribution(
        labels=label_image,
        intensity_image=intensity_image,
        channels=c,
        channel_names=channel_names,
        scaled=scaled,
        bin_count=bin_count,
        maximum_radius=maximum_radius,
    )


def get_radial_distribution(
    labels: NDArray[np.integer],
    intensity_image: NDArray,
    channels: Sequence[int],
    channel_names: Sequence[str],
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
) -> dict[str, NDArray[np.floating]]:
    nobjects = int(labels.max())
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    # Find the point in each object farthest away from the edge.
    # This does better than the centroid:
    # * The center is within the object
    # * The center tends to be an interesting point, like the
    #   center of the nucleus or the center of one or the other
    #   of two touching cells.
    # Tied maxima use the first position in C order so an object's center is
    # independent of values under other labels (SciPy gh-25279 / Issue #22).
    i, j = _maximum_position_of_labels(d_to_edge, labels, nobjects)

    center_labels = np.zeros(labels.shape, int)

    center_labels[i, j] = labels[i, j]

    #
    # Use the coloring trick here to process touching objects
    # in separate operations
    #
    colors = centrosome.cpmorphology.color_labels(labels)

    ncolors = np.max(colors)

    d_from_center = np.zeros(labels.shape)

    cl = np.zeros(labels.shape, int)

    for color in range(1, ncolors + 1):
        mask = colors == color
        l_, d = centrosome.propagate.propagate(
            np.zeros(center_labels.shape), center_labels, mask, 1
        )

        d_from_center[mask] = d[mask]

        cl[mask] = l_[mask]

    good_mask = cl > 0

    i_center = np.zeros(cl.shape)

    i_center[good_mask] = i[cl[good_mask] - 1]

    j_center = np.zeros(cl.shape)

    j_center[good_mask] = j[cl[good_mask] - 1]

    normalized_distance = np.zeros(labels.shape)

    if scaled:
        total_distance = d_from_center + d_to_edge

        normalized_distance[good_mask] = d_from_center[good_mask] / (
            total_distance[good_mask] + 0.001
        )
    else:
        normalized_distance[good_mask] = d_from_center[good_mask] / maximum_radius

    ngood_pixels = np.sum(good_mask)

    good_labels = labels[good_mask]

    bin_indexes = (normalized_distance * bin_count).astype(int)

    bin_indexes[bin_indexes > bin_count] = bin_count

    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])
    mean_pixel_fractions = []
    fraction_at_distances = []
    for c in channels:
        pixels = intensity_image[..., c]

        histogram = scipy.sparse.coo_matrix(
            (pixels[good_mask], labels_and_bins), (nobjects, bin_count + 1)
        ).toarray()

        sum_by_object = np.sum(histogram, 1)

        sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

        fraction_at_distance = histogram / sum_by_object_per_bin

        number_at_distance = scipy.sparse.coo_matrix(
            (np.ones(int(ngood_pixels)), labels_and_bins), (nobjects, bin_count + 1)
        ).toarray()

        sum_by_object = np.sum(number_at_distance, 1)

        sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

        fraction_at_bin = number_at_distance / sum_by_object_per_bin

        mean_pixel_fraction = fraction_at_distance / (
            fraction_at_bin + np.finfo(float).eps
        )
        mean_pixel_fractions.append(mean_pixel_fraction)
        fraction_at_distances.append(fraction_at_distance)

    # Anisotropy calculation.  Split each cell into eight wedges, then
    # compute coefficient of variation of the wedges' mean intensities
    # in each ring.
    #
    # Compute each pixel's delta from the center object's centroid
    i, j = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

    imask = i[good_mask] > i_center[good_mask]

    jmask = j[good_mask] > j_center[good_mask]

    absmask = abs(i[good_mask] - i_center[good_mask]) > abs(
        j[good_mask] - j_center[good_mask]
    )

    radial_index = imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4

    results = {}

    for bin in range(bin_count + (0 if scaled else 1)):
        bin_mask = good_mask & (bin_indexes == bin)

        bin_pixels = np.sum(bin_mask)

        bin_labels = labels[bin_mask]

        bin_radial_index = radial_index[bin_indexes[good_mask] == bin]

        labels_and_radii = (bin_labels - 1, bin_radial_index)
        pixel_count = scipy.sparse.coo_matrix(
            (np.ones(bin_pixels), labels_and_radii), (nobjects, 8)
        ).toarray()

        mask = pixel_count == 0
        for channel_index, c in enumerate(channels):
            pixels = intensity_image[..., c]
            radial_values = scipy.sparse.coo_matrix(
                (pixels[bin_mask], labels_and_radii), (nobjects, 8)
            ).toarray()

            radial_means: np.ma.MaskedArray = np.ma.MaskedArray(
                radial_values / pixel_count, mask
            )

            radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)

            radial_cv[np.sum(~mask, 1) == 0] = 0
            fraction_at_distance = fraction_at_distances[channel_index]
            mean_pixel_fraction = mean_pixel_fractions[channel_index]
            for measurement, feature, overflow_feature in (
                (fraction_at_distance[:, bin], MF_FRAC_AT_D, OF_FRAC_AT_D),
                (mean_pixel_fraction[:, bin], MF_MEAN_FRAC, OF_MEAN_FRAC),
                (np.array(radial_cv), MF_RADIAL_CV, OF_RADIAL_CV),
            ):
                if bin == bin_count:
                    measurement_name = overflow_feature
                else:
                    measurement_name = feature % (bin + 1, bin_count)
                tokens = measurement_name.split("_")  # RadialDistribution_FracAtD_1of4_
                assert len(tokens) == 3, tokens
                measurement_name = f"{tokens[0]}_{tokens[1]}_{channel_names[channel_index]}_{tokens[2]}"
                results[measurement_name] = measurement

    return results


def _zernike_scores(
    masks: NDArray[np.integer],
    zernike_indexes: NDArray[np.integer],
    intensity_image: NDArray[np.floating] | None,
    channels: Sequence[int],
) -> tuple[
    list[NDArray[np.floating]],
    list[NDArray[np.floating]],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    # Contiguous 1..N contract: segment index is label - 1, so the label->index lookup is a
    # plain arange. masks.max() raises on a size-0 array, so guard it.
    n = 0 if masks.size == 0 else int(masks.max())
    k = len(zernike_indexes)
    labels = np.arange(1, n + 1)
    lut = np.arange(-1, n)  # lut[0] = -1 (background); lut[label] = label - 1
    centers, radii = zernike.minimum_enclosing_circle(masks, labels)
    radii = np.asarray(radii, dtype=float)

    # Foreground pixels, their object row, and unit-disk coordinates relative to each
    # object's enclosing circle — no full (H, W) coordinate grid is materialised.
    seg_full = lut[masks]
    keep = seg_full >= 0
    rows, cols = np.nonzero(keep)
    seg = seg_full[keep]
    counts = np.bincount(seg, minlength=n).astype(float)
    # Single-pixel objects have an enclosing-circle radius of 0; the resulting 0/0 yields NaN
    # coordinates (which the r**2 > 1 cutoff later discards), matching centrosome — suppress the
    # warning since it is expected, not a fault.
    with np.errstate(invalid="ignore", divide="ignore"):
        ym = (rows - centers[seg, 0]) / radii[seg]
        xm = (cols - centers[seg, 1]) / radii[seg]

    coeffs = zernike.construct_zernike_lookuptable(zernike_indexes)
    r_square = xm * xm + ym * ym
    z = ym + 1j * xm

    # z**m via `**` is ~20x slower than repeated multiply; build the powers iteratively.
    z_pows = {}
    zp = z
    real_sums_list = []
    imag_sums_list = []
    for _ in channels:
        real_sums = np.zeros((n, k))
        imag_sums = np.zeros((n, k))
        real_sums_list.append(real_sums)
        imag_sums_list.append(imag_sums)
    for m in range(1, int(zernike_indexes[:, 1].max()) + 1):
        z_pows[m] = zp
        zp = zp * z
    for idx, (zn, zm) in enumerate(zernike_indexes):
        s_ = np.zeros_like(xm)
        for c in coeffs[idx, : (zn - zm) // 2 + 1]:  # Horner scheme on r**2
            s_ *= r_square
            s_ += c
        s_[r_square > 1] = 0
        for channel_index, channel in enumerate(channels):
            w = intensity_image[keep, channel].astype(float)
            real_sums = real_sums_list[channel_index]
            imag_sums = imag_sums_list[channel_index]
            s = w * s_
            if (
                zm == 0
            ):  # purely real moment; the imaginary segment-sum is identically zero
                real_sums[:, idx] = np.bincount(seg, weights=s, minlength=n)
            else:
                zf = s * z_pows[zm]
                real_sums[:, idx] = np.bincount(seg, weights=zf.real, minlength=n)
                imag_sums[:, idx] = np.bincount(seg, weights=zf.imag, minlength=n)

    return real_sums_list, imag_sums_list, radii, counts


def intensity_distribution_zernike(
    c: Sequence[int],
    zernike_degree: int = 9,
    channel_names: Sequence[str] = None,
    unique_labels: np.ndarray = None,
    label_image: np.ndarray = None,
    intensity_image: np.ndarray = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)

    # Intensity-weighted moment sums via the shared helper (pixels as the per-pixel weight);
    # radial Zernikes normalise by pixel count, not the enclosing-circle area. See _zernike_scores.
    vr_list, vi_list, _radii, counts = _zernike_scores(
        label_image, zernike_indexes, intensity_image=intensity_image, channels=c
    )
    #
    # Results will be formatted in a dictionary with the following keys:
    # Zernike{Magnitude|Phase}_{n}_{m}
    # n - the radial moment of the Zernike
    # m - the azimuthal moment of the Zernike
    #
    results: dict[str, NDArray[np.floating]] = {}
    for channel_index in range(len(c)):
        vr = vr_list[channel_index]
        vi = vi_list[channel_index]
        magnitude = np.sqrt(vr * vr + vi * vi) / counts[:, np.newaxis]
        # CellProfiler convention: arctan2(real, imag), not textbook arctan2(imag, real).
        # Kept for 1:1 parity — do not "fix" the argument order.
        phase = np.arctan2(vr, vi)

        for i, (n, m) in enumerate(zernike_indexes):
            results[
                f"{M_CATEGORY}_ZernikeMagnitude_{channel_names[channel_index]}_{n}_{m}"
            ] = magnitude[:, i]
            results[
                f"{M_CATEGORY}_ZernikePhase_{channel_names[channel_index]}_{n}_{m}"
            ] = phase[:, i]

    return results
