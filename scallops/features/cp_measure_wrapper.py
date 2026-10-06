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
    get_feret,
    get_sizeshape,
    get_zernike,
)
from cp_measure.core.measuretexture import get_texture
from cp_measure.multimask.measureobjectneighbors import D_EXPAND, measureobjectneighbors


def cp_neighbors(
    label_image: np.ndarray,
    distance: int = 5,
    distance_method: str = D_EXPAND,
    **kwargs,
) -> dict[str, Any]:
    return measureobjectneighbors(label_image, label_image, distance_method, distance)


def cp_granularity(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        # Granularity_{granularity_id}_Channel
        results_ = get_granularity(label_image, intensity_image[..., c[j]])
        for key in results_:
            channel_name = channel_names[c[j]]
            results[f"{key}_{channel_name}"] = results_[key]
    return results


def cp_colocalization(
    c1: int,
    c2: int,
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
) -> dict[str, np.ndarray]:
    pass


def _cp_colocalization_pairs(
    c: list[tuple[int, int]],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    all_results = {}
    for c_pair in c:
        results = {}
        img1 = intensity_image[..., c_pair[0]]
        img2 = intensity_image[..., c_pair[1]]
        channel_name1 = channel_names[c_pair[0]]
        channel_name2 = channel_names[c_pair[1]]
        results.update(get_correlation_costes(img1, img2, label_image))
        results.update(get_correlation_manders_fold(img1, img2, label_image))
        results.update(get_correlation_overlap(img1, img2, label_image))
        results.update(get_correlation_pearson(img1, img2, label_image))
        results.update(get_correlation_rwc(img1, img2, label_image))
        for key in results:
            # cp_measure suffixes directional measurements with _1 (first channel
            # relative to second) or _2 (second relative to first)
            if key.endswith("_1"):
                new_key = f"{key[:-2]}_{channel_name1}_{channel_name2}"
            elif key.endswith("_2"):
                new_key = f"{key[:-2]}_{channel_name2}_{channel_name1}"
            else:
                new_key = f"{key}_{channel_name1}_{channel_name2}"
            all_results[new_key] = results[key]
    return all_results


def cp_intensity_distribution_radial(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
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


def cp_intensity_distribution_zernike(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_radial_zernikes(label_image, intensity_image[..., c[j]])
        for key in results_:
            tokens = key.split("_")
            results[
                f"{tokens[0]}_{tokens[1]}_{channel_names[c[j]]}_{tokens[2]}_{tokens[3]}"
            ] = results_[key]
    return results


def cp_intensity(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    **kwargs,
) -> dict[str, Any]:
    results = {}
    for j in range(len(c)):
        results_ = get_intensity(label_image, intensity_image[..., c[j]])
        for key in results_:
            value = results_[key]

            # translate block-local locations to global image coordinates
            if key.startswith("Location_") and offset != (0, 0):
                if key.endswith("_Y"):
                    value = value + offset[0]
                elif key.endswith("_X"):
                    value = value + offset[1]
            results[f"{key}_{channel_names[c[j]]}"] = value
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


size_shape_skip = {
    "AreaShape_Area",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_BoundingBoxMaximum_Y",
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
}


def cp_size_shape(
    label_image: np.ndarray,
    remove_objects: bool = True,
    **kwargs,
) -> dict[str, Any]:
    results_ = get_sizeshape(label_image, None)
    results = {}

    for key in results_:
        results[f"AreaShape_{key}"] = results_[key]
    if remove_objects:
        for key in size_shape_skip:
            del results[key]
    results.update(_zernike(label_image))
    results.update(_feret(label_image))
    return results


def _zernike(label_image: np.ndarray) -> dict[str, Any]:
    results_ = get_zernike(label_image, None)
    results = {}
    for key in results_:
        results[f"AreaShape_{key}"] = results_[key]
    return results


def _feret(label_image: np.ndarray) -> dict[str, Any]:
    results_ = get_feret(label_image, None)
    results = {}
    for key in results_:
        results[f"AreaShape_{key}"] = results_[key]
    return results
