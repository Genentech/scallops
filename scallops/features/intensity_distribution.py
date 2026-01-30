from collections.abc import Sequence

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import numpy as np
import scipy.ndimage
import scipy.sparse
from cp_measure.core.utils import masks_to_ijv
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

""""
============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **MeasureObjectIntensity** and **/MeasureTexture**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *FracAtD:* Fraction of total stain in an object at a given radius.
-  *MeanFrac:* Mean fractional intensity at a given radius; calculated
   as fraction of total intensity normalized by fraction of pixels at a
   given radius.
-  *RadialCV:* Coefficient of variation of intensity within a ring,
   calculated across 8 slices.
-  *Zernike:* The Zernike features characterize the distribution of
   intensity across the object. For instance, Zernike 1,1 has a high
   value if the intensity is low on one side of the object and high on
   the other. The ZernikeMagnitude feature records the rotationally
   invariant degree magnitude of the moment and the ZernikePhase feature
   gives the moment’s orientation.

"""

Z_NONE = "None"
Z_MAGNITUDES = "Magnitudes only"
Z_MAGNITUDES_AND_PHASE = "Magnitudes and phase"
Z_ALL = [Z_NONE, Z_MAGNITUDES, Z_MAGNITUDES_AND_PHASE]

M_CATEGORY = "RadialDistribution"
F_FRAC_AT_D = "FracAtD"
F_MEAN_FRAC = "MeanFrac"
F_RADIAL_CV = "RadialCV"
F_ALL = [F_FRAC_AT_D, F_MEAN_FRAC, F_RADIAL_CV]

FF_SCALE = "%dof%d"
FF_OVERFLOW = "Overflow"
FF_GENERIC = FF_SCALE

MF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, FF_GENERIC))
MF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, FF_GENERIC))
MF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, FF_GENERIC))
OF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, FF_OVERFLOW))
OF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, FF_OVERFLOW))
OF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, FF_OVERFLOW))

A_FRAC_AT_D = "Fraction at Distance"
A_MEAN_FRAC = "Mean Fraction"
A_RADIAL_CV = "Radial CV"
MEASUREMENT_CHOICES = [A_FRAC_AT_D, A_MEAN_FRAC, A_RADIAL_CV]

MEASUREMENT_ALIASES = {
    A_FRAC_AT_D: MF_FRAC_AT_D,
    A_MEAN_FRAC: MF_MEAN_FRAC,
    A_RADIAL_CV: MF_RADIAL_CV,
}


def intensity_distribution(
    c: Sequence[int],
    calculate_zernike: bool = False,
    channel_names: Sequence[str] = None,
    unique_labels: np.ndarray = None,
    label_image: np.ndarray = None,
    intensity_image: np.ndarray = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    results = {}
    results.update(
        _radial_distribution(
            c=c,
            channel_names=channel_names,
            unique_labels=unique_labels,
            label_image=label_image,
            intensity_image=intensity_image,
        )
    )
    if calculate_zernike:
        results.update(
            _radial_zernikes(
                c=c,
                channel_names=channel_names,
                unique_labels=unique_labels,
                label_image=label_image,
                intensity_image=intensity_image,
            )
        )

    return results


def _radial_distribution(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    unique_labels: np.ndarray,
    intensity_image: np.ndarray,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    """
    :param scaled: Scale the bins
    :param bin_count: Number of bins
    :param maximum_radius: Maximum radius
    """
    (
        good_mask,
        labels_and_bins,
        nobjects,
        ngood_pixels,
        i_center,
        j_center,
        bin_indexes,
    ) = _radial_distribution_labels(
        label_image, unique_labels, scaled, bin_count, maximum_radius
    )
    results = {}
    for channel in c:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = _radial_distribution_intensity(
                intensity_image[..., channel],
                label_image,
                scaled,
                bin_count,
                good_mask,
                labels_and_bins,
                nobjects,
                ngood_pixels,
                i_center,
                j_center,
                bin_indexes,
            )
        for key in result:
            tokens = key.split("_")  # RadialDistribution_FracAtD_1of4_
            results[f"{tokens[0]}_{tokens[1]}_{channel_names[channel]}_{tokens[2]}"] = (
                result[key]
            )
    return results


def _radial_distribution_labels(
    labels,
    unique_labels,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    if labels.dtype == bool:
        labels = labels.astype(np.integer)

    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    # Find the point in each object farthest away from the edge.
    # This does better than the centroid:
    # * The center is within the object
    # * The center tends to be an interesting point, like the
    #   center of the nucleus or the center of one or the other
    #   of two touching cells.
    #
    # MODIFICATION: Delegated label indices to maximum_position_of_labels
    # This should not affect this one-mask/object function
    i, j = centrosome.cpmorphology.maximum_position_of_labels(
        # d_to_edge, labels, indices=[1]
        d_to_edge,
        labels,
        indices=unique_labels,
    )

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
    return (
        good_mask,
        labels_and_bins,
        len(unique_labels),
        ngood_pixels,
        i_center,
        j_center,
        bin_indexes,
    )


def _radial_distribution_intensity(
    pixels,
    labels,
    scaled,
    bin_count,
    good_mask,
    labels_and_bins,
    nobjects,
    ngood_pixels,
    i_center,
    j_center,
    bin_indexes,
):
    histogram = scipy.sparse.coo_matrix(
        (pixels[good_mask], labels_and_bins), (nobjects, bin_count + 1)
    ).toarray()

    sum_by_object = np.sum(histogram, 1)

    sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_distance = histogram / sum_by_object_per_bin

    number_at_distance = scipy.sparse.coo_matrix(
        (np.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count + 1)
    ).toarray()

    sum_by_object = np.sum(number_at_distance, 1)

    sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_bin = number_at_distance / sum_by_object_per_bin

    mean_pixel_fraction = fraction_at_distance / (fraction_at_bin + np.finfo(float).eps)

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

        radial_values = scipy.sparse.coo_matrix(
            (pixels[bin_mask], labels_and_radii), (nobjects, 8)
        ).toarray()

        pixel_count = scipy.sparse.coo_matrix(
            (np.ones(bin_pixels), labels_and_radii), (nobjects, 8)
        ).toarray()

        mask = pixel_count == 0

        radial_means = np.ma.masked_array(radial_values / pixel_count, mask)

        radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)

        radial_cv[np.sum(~mask, 1) == 0] = 0

        for measurement, feature, overflow_feature in (
            (fraction_at_distance[:, bin], MF_FRAC_AT_D, OF_FRAC_AT_D),
            (mean_pixel_fraction[:, bin], MF_MEAN_FRAC, OF_MEAN_FRAC),
            (np.array(radial_cv), MF_RADIAL_CV, OF_RADIAL_CV),
        ):
            if bin == bin_count:
                measurement_name = overflow_feature
            else:
                measurement_name = feature % (bin + 1, bin_count)

            results[measurement_name] = measurement

    return results


def _radial_zernikes(
    c: Sequence[int],
    zernike_degree: int = 9,
    channel_names: Sequence[str] = None,
    label_image: np.ndarray = None,
    unique_labels: np.ndarray = None,
    intensity_image: np.ndarray = None,
):
    ijv, l_, z, zernike_indexes = _radial_zernikes_labels(
        label_image, unique_labels, zernike_degree
    )
    results = {}
    for channel in c:
        result = _radial_zernikes_intensity(
            intensity_image[..., channel],
            unique_labels,
            ijv,
            l_,
            z,
            zernike_indexes,
        )
        for key in result:
            tokens = key.split("_")
            results[f"{tokens[0]}_{tokens[1]}_{channel_names[channel]}_{tokens[2]}"] = (
                result[key]
            )
    return results


def _radial_zernikes_labels(labels, unique_labels, zernike_degree: int = 9):
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)

    # MODIFIED: Delegate index generation to the minimum_enclosing_circle
    # MODIFIED: We assume non-overlapping labels for now
    # TODO Support label overlap (i.e., format in ijv)
    # MODIFIED: Delegate indexes to minimum_enclosing_circle
    ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels, unique_labels)

    #
    # Then compute x and y, the position of each labeled pixel
    # within a unit circle around the object
    #
    ijv = masks_to_ijv(labels)

    l_ = ijv[:, 2]  # (N,1) vector with labels

    yx = (ijv[:, :2] - ij[l_ - 1, :]) / r[l_ - 1, np.newaxis]

    z = centrosome.zernike.construct_zernike_polynomials(
        yx[:, 1], yx[:, 0], zernike_indexes
    )
    return ijv, l_, z, zernike_indexes


def _radial_zernikes_intensity(pixels, unique_labels, ijv, l_, z, zernike_indexes):
    # Filter ijv-formatted items to keep the ones inside the pixels boundary
    ijv_mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])
    # ijv_mask[ijv_mask] = pixels[ijv[ijv_mask,0], ijv[ijv_mask, 1]]

    l_ = l_[ijv_mask]
    z_ = z[ijv_mask, :]

    if len(l_) == 0:
        # Cover fringe case in which all labels were filtered out
        results = {}
        for mag_or_phase in ("Magnitude", "Phase"):
            for n, m in zernike_indexes:
                name = f"{M_CATEGORY}_Zernike{mag_or_phase}_{n}_{m}"
                results[name] = np.zeros(0)
    else:
        # MODIFIED: Replaced sum with the updated sum_labels
        areas = scipy.ndimage.sum_labels(
            np.ones(l_.shape, int), labels=l_, index=unique_labels
        )

        #
        # Results will be formatted in a dictionary with the following keys:
        # Zernike{Magniture|Phase}_{n}_{m}
        # n - the radial moment of the Zernike
        # m - the azimuthal moment of the Zernike
        #
        results = {}
        for i, (n, m) in enumerate(zernike_indexes):
            vr = scipy.ndimage.sum_labels(
                pixels[ijv[:, 0], ijv[:, 1]] * z_[:, i].real,
                labels=l_,
                index=unique_labels,
            )

            vi = scipy.ndimage.sum_labels(
                pixels[ijv[:, 0], ijv[:, 1]] * z[:, i].imag,
                labels=l_,
                index=unique_labels,
            )

            magnitude = np.sqrt(vr * vr + vi * vi) / areas
            phase = np.arctan2(vr, vi)

            results[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"] = magnitude
            results[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"] = phase

    return results


def _binned_rings(
    filled_image: np.ndarray, image: np.ndarray, bins: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Separate input filled image into binned rings.

    Here, the ring sizes are controlled by the number of bins and are normalized by the radius at
    that approximate angle. Taken from:
    https://github.com/lukebfunk/OpticalPooledScreens/blob/02812a74e183e2aeba2198804b577fd32ec673f6/ops/cp_emulator.py#L908

    :param filled_image: binary filled region image bound by box
    :param image: binary object image bound by box
    :param bins: number of bins to split object
    :return: binned object image and object center coordinates
    """

    # normalized_distance_to_center returns distance to center point,
    # normalized by distance to edge along that direction, [0,1];
    # 0 = center point, 1 = points outside the image
    normalized_distance, center = normalized_distance_to_center(filled_image)
    binned = np.ceil(normalized_distance * bins)
    binned[binned == 0] = 1

    return np.multiply(np.ceil(binned), image), center


def normalized_distance_to_center(
    filled_image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    distance_to_edge = distance_transform_edt(np.pad(filled_image, 1, "constant"))[
        1:-1, 1:-1
    ]
    max_distance = distance_to_edge.max()

    # median of all points furthest from edge
    center = tuple(
        np.median(np.where(distance_to_edge == max_distance), axis=1).astype(int)
    )
    mask = np.ones(filled_image.shape)
    mask[center[0], center[1]] = 0
    distance_to_center = distance_transform_edt(mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_dist = distance_to_center / (distance_to_center + distance_to_edge)
    norm_dist[np.isnan(norm_dist)] = 0
    return norm_dist, center


def _radial_wedges(image: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    """Returns input object shape divided into 8 radial wedges.

    From the center of the input labeled object, slices are generated along 45°. The labeling
    convention and function were taken from:
    https://github.com/lukebfunk/OpticalPooledScreens/blob/02812a74e183e2aeba2198804b577fd32ec673f6/ops/cp_emulator.py#L940

    :param image: np.ndarray representing labeled object
    :param center: center of object as tuple
    :return: object divided into eight wedges
    """

    i, j = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    positive_i, positive_j = (i > center[0], j > center[1])
    abs_i_greater_j = abs(i - center[0]) > abs(j - center[1])

    return ((positive_i + positive_j * 2 + abs_i_greater_j * 4 + 1) * image).astype(int)


def _intensity_distribution_old(
    c: Sequence[int],
    bins: int = 4,
    calculate_zernike: bool = False,
    channel_names: Sequence[str] = None,
    label_image: np.ndarray = None,
    unique_labels: np.ndarray = None,
    intensity_image: np.ndarray = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    props = regionprops(label_image, intensity_image=intensity_image[..., c])
    frac_at_d_values = np.zeros((len(props), bins, len(c)))
    mean_frac_values = np.zeros((len(props), bins, len(c)))
    radial_cv_values = np.zeros((len(props), bins, len(c)))

    for i in range(len(props)):
        r = props[i]
        image = r.image
        binned, center = _binned_rings(r.image_filled, image, bins)
        wedges = _radial_wedges(image, center)
        image_sum = image.sum()
        frac_pixels_at_d = (
            np.array([(binned == bin_ring).sum() for bin_ring in range(1, bins + 1)])
            / image_sum
        )
        frac_pixels_at_d = np.expand_dims(frac_pixels_at_d, 1)
        intensity_image = r.image_intensity

        with np.errstate(divide="ignore", invalid="ignore"):
            # Calculate fractional signal intensity per channel
            frac_at_d = np.array(
                [
                    intensity_image[binned == bin_ring].sum(axis=0)
                    for bin_ring in range(1, bins + 1)
                ]
            ) / intensity_image[image].sum(axis=0)
            # Calculate mean signal intensity within bin
            mean_frac = frac_at_d / frac_pixels_at_d

            # Split ring into radial wedges for computing variance metric

            mean_binned_wedges = np.array(
                [
                    np.array(
                        [
                            intensity_image[
                                (wedges == wedge) & (binned == bin_ring)
                            ].mean(axis=0)
                            for wedge in range(1, 9)
                        ]
                    )
                    for bin_ring in range(1, bins + 1)
                ]
            )
            radial_cv = np.nanstd(mean_binned_wedges, axis=1) / np.nanmean(
                mean_binned_wedges, axis=1
            )
            frac_at_d_values[i] = frac_at_d
            radial_cv_values[i] = radial_cv
            mean_frac_values[i] = mean_frac
    results = {}
    for channel_index in range(len(c)):
        channel_name = channel_names[c[channel_index]]
        for b in range(bins):
            results[f"RadialDistribution_FracAtD_{channel_name}_{b + 1}of{bins}"] = (
                frac_at_d_values[:, b, channel_index]
            )
            results[f"RadialDistribution_MeanFrac_{channel_name}_{b + 1}of{bins}"] = (
                mean_frac_values[:, b, channel_index]
            )
            results[f"RadialDistribution_RadialCV_{channel_name}_{b + 1}of{bins}"] = (
                radial_cv_values[:, b, channel_index]
            )
    if calculate_zernike:
        results.update(
            _radial_zernikes(
                c=c,
                channel_names=channel_names,
                unique_labels=unique_labels,
                label_image=label_image,
                intensity_image=intensity_image,
            )
        )
    return results
