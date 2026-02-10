import gc
import logging
import math
import os
import threading
from collections.abc import Sequence
from typing import Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
import psutil
import shapely
import zarr
from dask import delayed
from dask.diagnostics import ProgressBar
from skimage.util import img_as_float
from sklearn.cluster import AgglomerativeClustering

from scallops.io import _images2fov, _localize_path, pluralize
from scallops.stitch._radial import radial_correct
from scallops.stitch.utils import dtype_convert
from scallops.utils import _cpu_count, _dask_from_array_no_copy
from scallops.zarr_io import default_zarr_format, get_zarr_array_kwargs

logger = logging.getLogger("scallops")


def _create_label_ome_metadata(image_spacing: tuple[float, float], label_name: str):
    fmt = default_zarr_format()
    d = {
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {
                                "scale": (
                                    float(image_spacing[0]),
                                    float(image_spacing[1]),
                                ),
                                "type": "scale",
                            }
                        ],
                        "path": "0",
                    }
                ],
                "name": f"/labels/{label_name}",
                "version": fmt.version,
            }
        ]
    }
    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        return d

    return {"ome": d}


def _create_ome_metadata(
    image_spacing: tuple[float, float],
    stitch_coords: pd.DataFrame,
    image_key: str,
    **kwargs,
):
    metadata = {}
    metadata.update(**kwargs)
    metadata["stitch_coords"] = dict()
    fmt = default_zarr_format()
    for c in stitch_coords:  # convert to dict
        metadata["stitch_coords"][c] = stitch_coords[c].to_list()
    d = {
        "multiscales": [
            {
                "metadata": metadata,
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {
                                "scale": (
                                    1.0,
                                    float(image_spacing[0]),
                                    float(image_spacing[1]),
                                ),
                                "type": "scale",
                            }
                        ],
                        "path": "0",
                    }
                ],
                "name": f"/images/{image_key}",
                "version": fmt.version,
            }
        ]
    }
    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        return d
    return {"ome": d}


def _fuse(
    df: pd.DataFrame,
    group: zarr.Group,
    z_index: int | Literal["max"] | None = None,
    blend: Literal["none", "linear"] = "none",
    output_channels: Sequence[int] | None = None,
    ffp: np.ndarray | None = None,
    dfp: np.ndarray | None = None,
    crop_width: int | None = None,
    radial_correction_k: float | None = None,
    chunk_size: tuple[int, int] | None = None,
    channels_per_batch: int | None = None,
    scenes: bool = False,
):
    """Use stitching coordinates to fuse tiles.

    :param df: Stitch output dataframe.
    :param group: Group to save the result.
    :param z_index: z-index or 'max'. Ignored when df contains `z_index` column.
    :param blend: Blending mode
    :param output_channels: Optional output channels to include
    :param ffp: ffp image for illumination correction
    :param dfp: dfp image for illumination correction
    :param crop_width: Image crop width
    :param radial_correction_k: Image radial correction k
    :param chunk_size: Image y, x, chunk size. If `None`, auto determine by clustering
        tile positions into a grid to estimate number of rows and columns.
    :param channels_per_batch: Number of channels to output per batch. If `None`,
        auto determine from available memory
    :param scenes: If `True`, input contains single image containing all tiles
    """
    assert blend in ["none", "linear"]
    if crop_width is not None and crop_width <= 0:
        crop_width = None

    df = df.copy()
    image0_paths = df["source"].values[0]
    image0_attrs = (
        df["source_metadata"].values[0] if "source_metadata" in df.columns else None
    )

    local_image0_paths = [_localize_path(path) for path in image0_paths]
    image0_paths = [
        local_image0_paths[i] if local_image0_paths[i] is not None else image0_paths[i]
        for i in range(len(image0_paths))
    ]
    img = _images2fov(image0_paths, image0_attrs)

    [os.remove(path) for path in local_image0_paths if path is not None]

    n_channels = img.sizes["c"]
    size_z = img.sizes["z"] if "z" in img.dims else 1
    img = img.isel(t=0, z=0, missing_dims="ignore")
    if output_channels is None:
        output_channels = list(range(n_channels))
    img = img.values
    weights = None
    if blend != "none":
        # weights are cropped when fusing image so use full tile size
        weights = _tile_blending_weights(img.shape[1:])
    if crop_width is not None:
        img = img[..., crop_width:-crop_width, crop_width:-crop_width]
    target_dtype = img.dtype

    tile_shape = img.shape[1:]
    del img

    # set tile size to cropped tile size
    ysize = tile_shape[0]
    xsize = tile_shape[1]
    if blend == "none":
        # tiles closest to well center have higher priority
        df = df.sort_values("distance_to_center", ascending=False)
        df["priority"] = np.arange(len(df))

    df["x"] = df["x"].round().values.astype(int)
    df["y"] = df["y"].round().values.astype(int)
    fused_y_size = int((df["y"] + ysize).max())
    fused_x_size = int((df["x"] + xsize).max())

    if channels_per_batch is None:
        if blend == "none":
            size_per_channel = fused_y_size * fused_x_size * target_dtype.itemsize
        else:
            size_per_channel = (
                np.dtype("float").itemsize * 2 * fused_y_size * fused_x_size
            )
        gc.collect()
        available_mem = 0.7 * psutil.virtual_memory().available - (
            _cpu_count() * size_z * n_channels * ysize * xsize * target_dtype.itemsize
        )
        channels_per_batch = max(1, int(available_mem / size_per_channel))
        n_batches = math.ceil(len(output_channels) / channels_per_batch)
        channels_per_batch = max(1, int(len(output_channels) / n_batches))
        logger.debug(
            f"Available memory {available_mem:,}, memory per channel: {size_per_channel:,}"
        )
    if channels_per_batch < 0:
        channels_per_batch = len(output_channels)
    channels_per_batch = min(channels_per_batch, len(output_channels))
    cluster = AgglomerativeClustering(
        n_clusters=None, distance_threshold=tile_shape[0] * 0.1, linkage="single"
    )
    cluster.fit_predict(df[["y"]])
    y_step_size = math.ceil(fused_y_size / len(np.unique(cluster.labels_)))

    if chunk_size is None:
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=tile_shape[1] * 0.1, linkage="single"
        )
        cluster.fit_predict(df[["x"]])
        x_step_size = math.ceil(fused_x_size / len(np.unique(cluster.labels_)))
        chunk_size = y_step_size, x_step_size
    partition_tree = None
    locks = None

    if blend != "none":
        partition_boxes = []
        locks = []
        for i in np.arange(0, fused_y_size, y_step_size):
            b = shapely.box(0, i, fused_x_size, i + y_step_size)
            partition_boxes.append(b)
            locks.append(threading.Lock())
        locks = np.array(locks)
        partition_tree = shapely.STRtree(partition_boxes)
    output_shape = (len(output_channels), fused_y_size, fused_x_size)
    fmt = default_zarr_format()

    result = group.create_array(
        shape=output_shape,
        dtype=target_dtype,
        chunks=(1,) + chunk_size,
        name="0",
        overwrite=True,
        **get_zarr_array_kwargs(fmt),
    )

    _fuse_image_delayed = delayed(_fuse_image)

    if blend != "none":
        # shuffle to minimize lock contention
        df = df.sample(frac=1)

    if ffp is not None:
        if radial_correction_k is not None:
            ffp = radial_correct(ffp, radial_correction_k, cval=1)

    if dfp is not None:
        dfp = dfp / np.iinfo(target_dtype).max
        if radial_correction_k is not None:
            dfp = radial_correct(dfp, radial_correction_k, cval=0)

    y = df["y"].values
    x = df["x"].values
    source = df["source"].values

    source_attrs = (
        df["source_metadata"].values if "source_metadata" in df.columns else None
    )
    different_z_per_tile = "z_index" in df.columns
    if different_z_per_tile:
        z_index = df["z_index"].values

    tile_priorities = df["priority"].values if blend == "none" else None

    boxes = None
    tile_tree = None
    if blend == "none":
        boxes = []
        for i in range(len(x)):
            boxes.append(
                shapely.box(
                    x[i],
                    y[i],
                    min(fused_x_size - 1, x[i] + xsize - 1),
                    min(fused_y_size - 1, y[i] + ysize - 1),
                )
            )
        tile_tree = shapely.STRtree(boxes)
    for channel_batch in range(0, len(output_channels), channels_per_batch):
        channels = output_channels[channel_batch : channel_batch + channels_per_batch]
        _ffp = ffp[channels] if ffp is not None else None
        _dfp = dfp[channels] if dfp is not None else None

        logger.info(
            f"Fusing {pluralize('channel', len(channels))} {', '.join([str(c) for c in channels])}."
        )

        delayed_results = []
        target_shape = (len(channels), fused_y_size, fused_x_size)
        if blend != "none":
            weights_sum = np.zeros(
                (fused_y_size, fused_x_size),
            )
            target = np.zeros(
                target_shape,
                dtype="float",
            )
        else:
            weights_sum = None
            target = np.zeros(
                target_shape,
                dtype=target_dtype,
            )

        for i in range(len(y)):
            tile_box = shapely.box(
                x[i],
                y[i],
                min(fused_x_size - 1, x[i] + xsize - 1),
                min(fused_y_size - 1, y[i] + ysize - 1),
            )
            if blend != "none":
                lock_indices = partition_tree.query(tile_box, predicate="intersects")
            intersecting_boxes = []
            if blend == "none":
                intersect_indices = tile_tree.query(tile_box, predicate="intersects")
                # higher tile priority values have priority
                for intersect_index in intersect_indices:
                    if tile_priorities[intersect_index] > tile_priorities[i]:
                        b = shapely.intersection(
                            tile_box, boxes[intersect_index]
                        ).bounds
                        # change order to y1, x1, y2, x2
                        # bounding box is inclusive
                        b = (
                            int(b[1]),
                            int(b[0]),
                            int(b[3]),
                            int(b[2]),
                        )
                        intersecting_boxes.append(b)
            d = _fuse_image_delayed(
                image_paths=source[i],
                image_attrs=source_attrs[i] if source_attrs is not None else None,
                y=y[i],
                x=x[i],
                radial_correction_k=radial_correction_k,
                crop_width=crop_width,
                ffp=_ffp,
                dfp=_dfp,
                blend=blend,
                z_index=z_index[i] if different_z_per_tile else z_index,
                output_channels=channels,
                target_shape=target_shape,
                target_dtype=target_dtype,
                weights=weights,
                locks=locks[lock_indices] if blend != "none" else None,
                weights_sum=weights_sum,
                target=target,
                intersecting_boxes=intersecting_boxes
                if len(intersecting_boxes) > 0
                else None,
                scene=i if scenes else None,
            )

            delayed_results.append(d)

        with ProgressBar():
            dask.compute(*delayed_results)

        if blend != "none":
            weights_sum[weights_sum == 0] = 1
            target = target / weights_sum
            target = np.round(target).astype(target_dtype)

        with ProgressBar():
            logger.info("Writing to disk.")
            target = _dask_from_array_no_copy(target, chunks=(1,) + chunk_size)
            da.to_zarr(
                arr=target,
                url=result,
                region=(slice(channel_batch, channel_batch + channels_per_batch),),
                compute=True,
                dimension_separator="/",
            )


def _fuse_image(
    image_paths: list[str] | zarr.Group,
    image_attrs: dict[str, str | list[str]],
    y: int,
    x: int,
    target_shape: tuple[int, int],
    target: np.ndarray,
    target_dtype: np.dtype,
    radial_correction_k: float | None = None,
    output_channels: Sequence[int] | int | None = None,
    crop_width: int | None = None,
    dfp: np.ndarray | None = None,
    ffp: np.ndarray | None = None,
    z_index: int | Literal["max"] = "max",
    locks: Sequence[threading.Lock] | None = None,
    blend: Literal["none", "linear"] = "none",
    intersecting_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    weights: np.ndarray | None = None,
    weights_sum: np.ndarray | None = None,
    scene: int | None = None,
):
    local_image_paths = [_localize_path(path) for path in image_paths]
    image_paths = [
        local_image_paths[i] if local_image_paths[i] is not None else image_paths[i]
        for i in range(len(image_paths))
    ]

    img = _images2fov(image_paths, image_attrs, scene_id=scene).isel(
        t=0, missing_dims="ignore"
    )
    [os.remove(path) for path in local_image_paths if path is not None]

    if output_channels is not None:
        img = img.isel(c=output_channels)
    if "z" in img.dims:
        img = img.max(dim="z") if not isinstance(z_index, int) else img.isel(z=z_index)
    img = img.values

    # order: radial, illumination, crop
    if radial_correction_k is not None:
        img = radial_correct(img, radial_correction_k)
        img = dtype_convert(img, target_dtype)

    img = img_as_float(img, force_copy=True)
    if dfp is not None:
        img -= dfp
    if ffp is not None:
        img /= ffp
    img.clip(0, 1, out=img)
    if crop_width is not None:
        img = img[..., crop_width:-crop_width, crop_width:-crop_width]
        if weights is not None:
            weights = weights[crop_width:-crop_width, crop_width:-crop_width]

    target_slice_y = (y, min(target_shape[-2], y + img.shape[-2]))
    target_slice_x = (x, min(target_shape[-1], x + img.shape[-1]))
    target_shape = (
        target_slice_y[1] - target_slice_y[0],
        target_slice_x[1] - target_slice_x[0],
    )
    img_shape = img.shape[-2:]
    if target_shape != img_shape:
        img = img[
            ...,
            0 : min(img_shape[0], target_shape[0]),
            0 : min(img_shape[1], target_shape[1]),
        ]
        if weights is not None:
            weights = weights[
                0 : min(img_shape[0], target_shape[0]),
                0 : min(img_shape[1], target_shape[1]),
            ]

    target_include = None

    if intersecting_boxes is not None:
        # list of (y1, x2, y2, x2) in global coordinates where we don't write values for this tile
        target_include = np.ones(
            (
                target_slice_y[1] - target_slice_y[0],
                target_slice_x[1] - target_slice_x[0],
            ),
            dtype=bool,
        )

        for box in intersecting_boxes:
            target_include[
                box[0] - target_slice_y[0] : box[2] - target_slice_y[0] + 1,
                box[1] - target_slice_x[0] : box[3] - target_slice_x[0] + 1,
            ] = False

    img = dtype_convert(img, target_dtype)

    if blend == "none":
        if target_include is not None:
            target_y, target_x = np.where(target_include)
            target_y += target_slice_y[0]
            target_x += target_slice_x[0]
            target[
                ...,
                target_y,
                target_x,
            ] = img[..., target_include]
        else:
            target[
                ...,
                target_slice_y[0] : target_slice_y[1],
                target_slice_x[0] : target_slice_x[1],
            ] = img

    else:
        try:
            for lock in locks:
                lock.acquire()

            weights_sum[
                target_slice_y[0] : target_slice_y[1],
                target_slice_x[0] : target_slice_x[1],
            ] += weights
            target[
                ...,
                target_slice_y[0] : target_slice_y[1],
                target_slice_x[0] : target_slice_x[1],
            ] += img * weights
        finally:
            for lock in locks:
                lock.release()


def _dist_from_tile_edge(i: int, j: int, tile_width: int, tile_height: int) -> int:
    """Calculates a distance metric from pixel coordinates to tile edges.

    :param i: Vertical (row) coordinate of the pixel
    :param j: Horizontal (column) coordinate of the pixel
    :param tile_width: Width of the tile in pixels
    :param tile_height:  Height of the tile in pixels
    :return: Distance metric calculated as product of minimum horizontal and vertical distances to edges
    """

    left = j + 1
    up = i + 1
    right = tile_width - j
    bottom = tile_height - i
    hor = min(right, left)
    vert = min(up, bottom)
    return hor * vert


def _tile_blending_weights(tile_shape: tuple[int, int]) -> np.ndarray:
    weights = np.zeros(tile_shape)
    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            weights[i, j] = _dist_from_tile_edge(i, j, tile_shape[0], tile_shape[1])
    weights /= weights.max()
    return weights
