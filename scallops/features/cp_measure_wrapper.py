"""Provides functions for calculating various intensity-based statistics for image regions.

Authors:
    - The SCALLOPS development team
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from cp_measure.core.measurecolocalization import (
    get_correlation_costes,
    get_correlation_manders_fold,
    get_correlation_overlap,
    get_correlation_pearson,
    get_correlation_rwc,
)
from cp_measure.core.measuregranularity import get_granularity
from cp_measure.core.measureobjectintensity import get_intensity
from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_distribution,
    get_radial_zernikes,
)
from cp_measure.core.measureobjectsizeshape import (
    get_ferret,
    get_sizeshape,
    get_zernike,
)
from cp_measure.core.measuretexture import get_texture
from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors


def cp_neighbors(
    label_image1: np.ndarray,
    label_image2: np.ndarray,
) -> dict[str, Any]:
    return measureobjectneighbors(label_image1, label_image2)


def cp_granularity(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_granularity(label_image, intensity_image[..., c[j]])
        for key in results_:
            results[f"{key}_{channel_names[c[j]]}"] = results_[key]
    return results


def cp_intensity_distribution(
    c: Sequence[int],
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    results = {}
    results.update(_radial_distribution(c, channel_names, label_image, intensity_image))
    results.update(_radial_zernikes(c, channel_names, label_image, intensity_image))

    return results


def _radial_distribution(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_radial_distribution(label_image, intensity_image[..., c[j]])
        for key in results_:
            tokens = key.split("_")
            results[f"{tokens[0]}_{tokens[1]}_{channel_names[c[j]]}_{tokens[2]}"] = (
                results_[key]
            )
    return results


def _radial_zernikes(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_radial_zernikes(label_image, intensity_image[..., c[j]])
        for key in results_:
            tokens = key.split("_")
            results[f"{tokens[0]}_{tokens[1]}_{channel_names[c[j]]}_{tokens[2]}"] = (
                results_[key]
            )
    return results


def cp_intensity(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_intensity(label_image, intensity_image[..., c[j]])
        for key in results_:
            results[f"{key}_{channel_names[c[j]]}"] = results_[key]
    return results


def _texture_rename(key, channel_names, c):
    index = key.index("_")
    return f"Texture_{key[:index]}_{channel_names[c]}_{key[index + 1 :]}"


def cp_texture(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_texture(label_image, intensity_image[..., c[j]])
        for key in results_:
            results[_texture_rename(key, channel_names, c[j])] = results_[key]
    return results


def _radial_distribution_rename(key, channel_names, c):
    index = key.rindex("_")
    return f"{key[:index]}_{channel_names[c]}_{key[index + 1 :]}"


def _size_shape_rename(key):
    return f"AreaShape_{key}"


size_shape_skip = {
    "Area",
    "BoundingBoxMinimum_X",
    "BoundingBoxMaximum_X",
    "BoundingBoxMinimum_Y",
    "BoundingBoxMaximum_Y",
    "Center_X",
    "Center_Y",
}


def cp_size_shape(
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    remove_objects: bool = True,
    **kwargs,
) -> dict[str, Any]:
    results_ = get_sizeshape(label_image, None)
    results = {}

    for key in results_:
        results[_size_shape_rename(key)] = results_[key]
    if remove_objects:
        for key in size_shape_skip:
            del results[f"AreaShape_{key}"]
    results.update(_zernike(label_image))
    results.update(_ferret(label_image))
    return results


def _zernike(label_image: np.ndarray) -> dict[str, Any]:
    results_ = get_zernike(label_image, None)
    results = {}
    for key in results_:
        results[_size_shape_rename(key)] = results_[key]
    return results


def _ferret(label_image: np.ndarray) -> dict[str, Any]:
    results_ = get_ferret(label_image, None)
    results = {}
    for key in results_:
        results[_size_shape_rename(key)] = results_[key]
    return results


def cp_colocalization(
    c1: int, c2: int, label_image: np.ndarray, intensity_image: np.ndarray, **kwargs
) -> dict[str, Any]:
    results = {}
    results.update(
        get_correlation_overlap(
            intensity_image[..., c1], intensity_image[..., c2], label_image
        )
    )
    results.update(
        get_correlation_pearson(
            intensity_image[..., c1], intensity_image[..., c2], label_image
        )
    )
    results.update(
        get_correlation_manders_fold(
            intensity_image[..., c1], intensity_image[..., c2], label_image
        )
    )
    results.update(
        get_correlation_rwc(
            intensity_image[..., c1], intensity_image[..., c2], label_image
        )
    )
    results.update(
        get_correlation_costes(
            intensity_image[..., c1], intensity_image[..., c2], label_image
        )
    )
    return results
