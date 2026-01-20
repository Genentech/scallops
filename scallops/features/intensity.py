from collections.abc import Sequence

import numpy as np
import scipy.ndimage

__doc__ = """
MeasureObjectIntensity
======================

**MeasureObjectIntensity** measures several intensity features for
identified objects.

Given an image with objects identified (e.g., nuclei or cells), this
module extracts intensity features for each object based on one or more
corresponding grayscale images. Measurements are recorded for each
object.

Intensity measurements are made for all combinations of the images and
objects entered. If you want only specific image/object measurements,
you can use multiple MeasureObjectIntensity modules for each group of
measurements desired.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **NamesAndTypes**, **MeasureImageIntensity**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *IntegratedIntensity:* The sum of the pixel intensities within an
   object.
-  *MeanIntensity:* The average pixel intensity within an object.
-  *StdIntensity:* The standard deviation of the pixel intensities
   within an object.
-  *MaxIntensity:* The maximal pixel intensity within an object.
-  *MinIntensity:* The minimal pixel intensity within an object.
-  *IntegratedIntensityEdge:* The sum of the edge pixel intensities of
   an object.
-  *MeanIntensityEdge:* The average edge pixel intensity of an object.
-  *StdIntensityEdge:* The standard deviation of the edge pixel
   intensities of an object.
-  *MaxIntensityEdge:* The maximal edge pixel intensity of an object.
-  *MinIntensityEdge:* The minimal edge pixel intensity of an object.
-  *MassDisplacement:* The distance between the centers of gravity in
   the gray-level representation of the object and the binary
   representation of the object.
-  *LowerQuartileIntensity:* The intensity value of the pixel for which
   25% of the pixels in the object have lower values.
-  *MedianIntensity:* The median intensity value within the object.
-  *MADIntensity:* The median absolute deviation (MAD) value of the
   intensities within the object. The MAD is defined as the
   median(|x :sub:`i` - median(x)|).
-  *UpperQuartileIntensity:* The intensity value of the pixel for which
   75% of the pixels in the object have lower values.
-  *Location_CenterMassIntensity_X, Location_CenterMassIntensity_Y:*
   The (X,Y) coordinates of the intensity weighted centroid (=
   center of mass = first moment) of all pixels within the object.
-  *Location_MaxIntensity_X, Location_MaxIntensity_Y:* The
   (X,Y) coordinates of the pixel with the maximum intensity within the
   object.
"""

from scipy.ndimage import find_objects
from skimage.segmentation import find_boundaries

INTENSITY = "Intensity"
INTEGRATED_INTENSITY = "IntegratedIntensity"
MEAN_INTENSITY = "MeanIntensity"
STD_INTENSITY = "StdIntensity"
MIN_INTENSITY = "MinIntensity"
MAX_INTENSITY = "MaxIntensity"
INTEGRATED_INTENSITY_EDGE = "IntegratedIntensityEdge"
MEAN_INTENSITY_EDGE = "MeanIntensityEdge"
STD_INTENSITY_EDGE = "StdIntensityEdge"
MIN_INTENSITY_EDGE = "MinIntensityEdge"
MAX_INTENSITY_EDGE = "MaxIntensityEdge"
MASS_DISPLACEMENT = "MassDisplacement"
LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"
MEDIAN_INTENSITY = "MedianIntensity"
MAD_INTENSITY = "MADIntensity"
UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
C_LOCATION = "Location"
LOC_CMI_X = "CenterMassIntensity_X"
LOC_CMI_Y = "CenterMassIntensity_Y"
LOC_CMI_Z = "CenterMassIntensity_Z"
LOC_MAX_X = "MaxIntensity_X"
LOC_MAX_Y = "MaxIntensity_Y"
LOC_MAX_Z = "MaxIntensity_Z"

ALL_MEASUREMENTS = [
    INTEGRATED_INTENSITY,
    MEAN_INTENSITY,
    STD_INTENSITY,
    MIN_INTENSITY,
    MAX_INTENSITY,
    INTEGRATED_INTENSITY_EDGE,
    MEAN_INTENSITY_EDGE,
    STD_INTENSITY_EDGE,
    MIN_INTENSITY_EDGE,
    MAX_INTENSITY_EDGE,
    MASS_DISPLACEMENT,
    LOWER_QUARTILE_INTENSITY,
    MEDIAN_INTENSITY,
    MAD_INTENSITY,
    UPPER_QUARTILE_INTENSITY,
]
ALL_LOCATION_MEASUREMENTS = [
    LOC_CMI_X,
    LOC_CMI_Y,
    LOC_CMI_Z,
    LOC_MAX_X,
    LOC_MAX_Y,
    LOC_MAX_Z,
]


def intensity(
    c: Sequence[int],
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    label_image_original: np.ndarray,
    intensity_image: np.ndarray,
    offset: tuple[int, int],
    **kwargs,
) -> dict[str, float]:
    intensity_image = intensity_image[..., c]
    integrated_intensity = np.zeros((len(unique_labels), len(c)))
    std_intensity = np.zeros((len(unique_labels), len(c)))
    mad_intensity = np.zeros((len(unique_labels), len(c)))
    mean_intensity = np.zeros((len(unique_labels), len(c)))
    quantiles = np.zeros((len(unique_labels), 5, len(c)))

    objects = find_objects(label_image)
    index = 0
    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        label = object_index + 1
        image = label_image[sl] == label
        intensity_image_sl = intensity_image[sl][image]

        integrated_intensity[index] = intensity_image_sl.sum(axis=0)
        mean_intensity[index] = intensity_image_sl.mean(axis=0)
        std_intensity[index] = intensity_image_sl.std(axis=0)
        quantiles[index] = np.percentile(
            intensity_image_sl, q=[0, 25, 50, 75, 100], axis=0
        )
        mad_intensity[index] = np.median(
            np.abs(intensity_image_sl - quantiles[index, 2]), axis=0
        )
        index = index + 1

    com_y = np.zeros((len(unique_labels), len(c)))
    com_x = np.zeros((len(unique_labels), len(c)))
    mass_displacement = np.zeros((len(unique_labels), len(c)))
    max_position_y = np.zeros((len(unique_labels), len(c)))
    max_position_x = np.zeros((len(unique_labels), len(c)))
    for channel_index in range(intensity_image.shape[-1]):
        max_position = np.array(
            scipy.ndimage.maximum_position(
                intensity_image[..., channel_index], label_image, unique_labels
            )
        )
        max_position_y[:, channel_index] = max_position[:, 0] + offset[0]
        max_position_x[:, channel_index] = max_position[:, 1] + offset[1]
        com = np.array(
            scipy.ndimage.center_of_mass(
                intensity_image[..., channel_index], label_image, unique_labels
            )
        )

        com_y_ = com[:, 0]
        com_x_ = com[:, 1]

        binary_com = np.array(
            scipy.ndimage.center_of_mass(
                intensity_image[..., channel_index].astype(bool),
                label_image,
                unique_labels,
            )
        )

        binary_com_y = binary_com[:, 0]
        binary_com_x = binary_com[:, 1]
        diff_x = com_y_ - binary_com_y
        diff_y = com_x_ - binary_com_x

        com_y[:, channel_index] = com_y_ + offset[0]

        com_x[:, channel_index] = com_x_ + offset[1]

        mass_displacement[:, channel_index] = np.sqrt(diff_x * diff_x + diff_y * diff_y)

    results = {}
    for channel_index in range(len(c)):
        channel_name = channel_names[c[channel_index]]
        results[f"{INTENSITY}_{INTEGRATED_INTENSITY}_{channel_name}"] = (
            integrated_intensity[:, channel_index]
        )
        results[f"{INTENSITY}_{MEAN_INTENSITY}_{channel_name}"] = mean_intensity[
            :, channel_index
        ]
        results[f"{INTENSITY}_{STD_INTENSITY}_{channel_name}"] = std_intensity[
            :, channel_index
        ]
        results[f"{INTENSITY}_{MAD_INTENSITY}_{channel_name}"] = mad_intensity[
            :, channel_index
        ]
        results[f"{INTENSITY}_{MIN_INTENSITY}_{channel_name}"] = quantiles[
            ..., 0, channel_index
        ]
        results[f"{INTENSITY}_{LOWER_QUARTILE_INTENSITY}_{channel_name}"] = quantiles[
            ..., 1, channel_index
        ]
        results[f"{INTENSITY}_{MEDIAN_INTENSITY}_{channel_name}"] = quantiles[
            ..., 2, channel_index
        ]
        results[f"{INTENSITY}_{UPPER_QUARTILE_INTENSITY}_{channel_name}"] = quantiles[
            ..., 3, channel_index
        ]
        results[f"{INTENSITY}_{MAX_INTENSITY}_{channel_name}"] = quantiles[
            ..., 4, channel_index
        ]
        results[f"{INTENSITY}_{MASS_DISPLACEMENT}_{channel_name}"] = mass_displacement[
            ..., channel_index
        ]
        results[f"{C_LOCATION}_{LOC_CMI_Y}_{channel_name}"] = com_y[..., channel_index]
        results[f"{C_LOCATION}_{LOC_CMI_X}_{channel_name}"] = com_x[..., channel_index]
        results[f"{C_LOCATION}_{LOC_MAX_Y}_{channel_name}"] = max_position_y[
            ..., channel_index
        ]
        results[f"{C_LOCATION}_{LOC_MAX_X}_{channel_name}"] = max_position_x[
            ..., channel_index
        ]

    # edges
    # outline the pixels *just inside* of objects, leaving background pixels untouched

    boundaries = find_boundaries(label_image_original, mode="inner", background=0)
    label_image = label_image[boundaries]
    intensity_image = intensity_image[boundaries]
    integrated_intensity = np.zeros((len(unique_labels), len(c)))
    mean_intensity = np.zeros((len(unique_labels), len(c)))
    std_intensity = np.zeros((len(unique_labels), len(c)))
    min_intensity = np.zeros((len(unique_labels), len(c)))
    max_intensity = np.zeros((len(unique_labels), len(c)))

    objects = find_objects(label_image)
    index = 0
    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        label = object_index + 1
        image = label_image[sl] == label
        intensity_image_sl = intensity_image[sl][image]
        integrated_intensity[index] = intensity_image_sl.sum(axis=0)
        mean_intensity[index] = intensity_image_sl.mean(axis=0)
        std_intensity[index] = intensity_image_sl.std(axis=0)
        min_intensity[index] = intensity_image_sl.min(axis=0)
        max_intensity[index] = intensity_image_sl.max(axis=0)
        index = index + 1

    for channel_index in range(len(c)):
        channel_name = channel_names[c[channel_index]]
        results[f"{INTENSITY}_{INTEGRATED_INTENSITY_EDGE}_{channel_name}"] = (
            integrated_intensity[..., channel_index]
        )
        results[f"{INTENSITY}_{MEAN_INTENSITY_EDGE}_{channel_name}"] = mean_intensity[
            ..., channel_index
        ]
        results[f"{INTENSITY}_{STD_INTENSITY_EDGE}_{channel_name}"] = std_intensity[
            ..., channel_index
        ]
        results[f"{INTENSITY}_{MIN_INTENSITY_EDGE}_{channel_name}"] = min_intensity[
            ..., channel_index
        ]
        results[f"{INTENSITY}_{MAX_INTENSITY_EDGE}_{channel_name}"] = max_intensity[
            ..., channel_index
        ]
    return results
