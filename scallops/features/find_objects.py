"""This module provides functions for finding and aggregating objects and their associated
properties (such as bounding boxes, centroids, and intensities) from labeled images using Dask.

Authors:
    - The SCALLOPS development team
"""

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import zarr
from dask import delayed
from numpy.typing import NDArray


def _find_bounding_boxes(
    label_image: NDArray | zarr.Array,
    intensity_image: NDArray | zarr.Array | None,
    sl: tuple[slice, ...],
    store_label_image: bool = False,
) -> pd.DataFrame:
    """Find bounding boxes, centroids, and optionally extract label and intensity images.

    :param label_image: Dask array representing labeled regions.
    :param intensity_image: Dask array of intensity values corresponding to the label
        image.
    :param sl: Slices specifying the bounding box coordinates.
    :param store_label_image: Boolean flag indicating whether to store the label image.
    :return: DataFrame with bounding box coordinates, centroids, and (optional)
        label/intensity data.
    """
    if isinstance(label_image, zarr.Array):
        label_image = label_image[sl]
    if isinstance(intensity_image, zarr.Array):
        intensity_image = intensity_image[..., sl]
    array_location = [s.start for s in sl]
    unique_vals = np.unique(label_image)
    unique_vals = unique_vals[unique_vals != 0]
    result = {}
    for val in unique_vals:
        positions = np.where(label_image == val)
        values = []
        slices = []
        for i, pos in enumerate(positions):
            min_pos = np.min(pos)
            max_pos = np.max(pos)
            slices.append(slice(min_pos, max_pos + 1))
            global_start = min_pos + array_location[i]
            global_stop = max_pos + array_location[i] + 1
            values.append(global_start)
            values.append(global_stop)
            total = (pos + array_location[i]).sum()
            values.append(total)
            values.append(len(pos))
        bbox = tuple(slices)
        if store_label_image:
            image_ = label_image[bbox] == val
            values.append(image_.reshape(-1))
        if intensity_image is not None:
            intensity_image_ = intensity_image[bbox].reshape(-1)
            values.append(intensity_image_)

        result[val] = values
    columns = []
    dim_names = ["Y", "X"]
    for i in range(label_image.ndim):
        dim_name = dim_names[i]
        columns.append(f"AreaShape_BoundingBoxMinimum_{dim_name}")
        columns.append(f"AreaShape_BoundingBoxMaximum_{dim_name}")

        columns.append(f"AreaShape_Center_{dim_name}-sum")
        columns.append(f"AreaShape_Center_{dim_name}-count")
    if store_label_image:
        columns.append("label_image")
    if intensity_image is not None:
        columns.append("intensity_image")
    return pd.DataFrame.from_dict(result, orient="index", columns=columns)


def _find_objects(
    label_image: da.Array | zarr.Array,
    intensity_image: da.Array | zarr.Array | None,
    store_label_image: bool,
):
    """Identify objects in the labeled image and compute their properties.

    :param label_image: Dask array of labeled regions.
    :param intensity_image: Dask array of intensity values.
    :param store_label_image: Boolean flag to store label images.
    :return: Grouped DataFrame of object properties.
    """

    assert (
        label_image.ndim == 2
    )  # TODO only tested for 2 dimensions but should work with n dims

    if intensity_image is not None:  # y,x,c
        assert intensity_image.shape[:-1] == label_image.shape, (
            f"{intensity_image.shape} != {label_image.shape}"
        )
        if isinstance(intensity_image, da.Array):  # match chunking
            intensity_image = intensity_image.rechunk(
                intensity_image.chunksize[:-1] + (-1,)
            )
            label_image = label_image.rechunk(intensity_image.chunksize[:-1])
            assert intensity_image.chunksize[:-1] == label_image.chunksize

    meta = {
        "AreaShape_BoundingBoxMinimum_Y": [1],
        "AreaShape_BoundingBoxMinimum_X": [1],
        "AreaShape_BoundingBoxMaximum_Y": [1],
        "AreaShape_BoundingBoxMaximum_X": [1],
        "AreaShape_Center_Y-sum": [1.0],
        "AreaShape_Center_Y-count": [1],
        "AreaShape_Center_X-sum": [1.0],
        "AreaShape_Center_X-count": [1],
    }
    if store_label_image:
        meta["label_image"] = [np.zeros(1)]
    if intensity_image is not None:
        meta["intensity_image"] = [np.zeros(1)]

    meta = dd.utils.make_meta(
        pd.DataFrame(meta, index=np.zeros(1, dtype=label_image.dtype))
    )
    is_dask_array = isinstance(label_image, da.Array)
    if not is_dask_array:
        assert isinstance(label_image, zarr.Array), (
            "Expected label image to be a Dask or Zarr array."
        )
    slices = da.core.slices_from_chunks(
        label_image.chunks if is_dask_array else da.from_zarr(label_image).chunks
    )
    results = []
    _find_bounding_boxes_delayed = delayed(_find_bounding_boxes)
    for sl in slices:
        if is_dask_array:
            label_block = label_image[sl]
            image_block = (
                intensity_image[..., sl] if intensity_image is not None else None
            )

        results.append(
            _find_bounding_boxes_delayed(
                label_block if is_dask_array else label_image,
                image_block if is_dask_array else intensity_image,
                sl,
                store_label_image,
            )
        )
    df = dd.from_delayed(results, meta=meta, verify_meta=False)
    grouped = df.groupby(df.index, group_keys=False, sort=False, dropna=False)
    return grouped


def _agg_objects(grouped) -> dd.DataFrame:
    """Aggregate object properties into a summary DataFrame.

    :param grouped: Grouped DataFrame of object properties.
    :return: Aggregated DataFrame with bounding boxes, centroids, and areas.
    """
    objects_df = grouped.agg(
        {
            "AreaShape_BoundingBoxMinimum_Y": "min",
            "AreaShape_BoundingBoxMaximum_Y": "max",
            "AreaShape_BoundingBoxMinimum_X": "min",
            "AreaShape_BoundingBoxMaximum_X": "max",
            "AreaShape_Center_Y-sum": "sum",
            "AreaShape_Center_Y-count": "sum",
            "AreaShape_Center_X-sum": "sum",
            "AreaShape_Center_X-count": "sum",
        }
    )

    objects_df["AreaShape_Center_Y"] = (
        objects_df["AreaShape_Center_Y-sum"] / objects_df["AreaShape_Center_Y-count"]
    )
    objects_df["AreaShape_Center_X"] = (
        objects_df["AreaShape_Center_X-sum"] / objects_df["AreaShape_Center_X-count"]
    )
    objects_df["AreaShape_Area"] = (
        objects_df["AreaShape_Center_Y-count"] + objects_df["AreaShape_Center_X-count"]
    ) / 2
    for f in [
        "AreaShape_BoundingBoxMinimum_Y",
        "AreaShape_BoundingBoxMinimum_X",
        "AreaShape_BoundingBoxMaximum_Y",
        "AreaShape_BoundingBoxMaximum_X",
    ]:
        objects_df[f] = objects_df[f].astype(int)
    objects_df = objects_df.drop(
        columns=[
            "AreaShape_Center_Y-sum",
            "AreaShape_Center_Y-count",
            "AreaShape_Center_X-sum",
            "AreaShape_Center_X-count",
        ],
        axis=1,
    )
    objects_df.index.name = "label"
    return objects_df


def find_objects(label_image: da.Array | NDArray) -> dd.DataFrame | pd.DataFrame:
    """Find objects in a labeled array.

    :param label_image: Image labels noted by integers.
    :return: Objects data frame containing bounding box, centroid, and area for
        each label. A pandas data frame if `label_image` is a NumPy array, otherwise a
        Dask data frame.
    """
    is_numpy = False
    if isinstance(label_image, np.ndarray):
        label_image = da.from_array(label_image)
        is_numpy = True
    grouped = _find_objects(label_image, None, False)
    result = _agg_objects(grouped)
    return result if not is_numpy else result.compute()
