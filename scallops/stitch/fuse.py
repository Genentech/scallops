import logging
import math
import os
from collections.abc import Sequence
from typing import Any, Literal

import dask
import numpy as np
import pandas as pd
import shapely
import zarr
from dask import delayed
from dask.diagnostics import ProgressBar
from numpy.typing import NDArray
from skimage.transform import SimilarityTransform, warp
from skimage.util import img_as_float
from sklearn.cluster import AgglomerativeClustering

from scallops.io import _images2fov, _images2fov_array, _localize_path, pluralize
from scallops.registration.crosscorrelation import _apply_window, _filter_percentiles
from scallops.stitch._channel_align import channel_coords, warp_image
from scallops.stitch._radial import radial_correct
from scallops.stitch.shift_utils import calc_best_shift
from scallops.stitch.utils import (
    _crop_image,
    _select_t_index,
    dtype_convert,
)
from scallops.zarr_io import (
    _attrs_axes_scales,
    _create_array_kwargs,
    _create_zarr_attrs,
    _current_format,
)

logger = logging.getLogger("scallops")


def _fuse(
    df: pd.DataFrame,
    group: zarr.Group,
    z_index: int | Literal["max"] | None = None,
    blend: Literal["none", "linear"] = "none",
    output_channels: Sequence[int] | None = None,
    channel_names: Sequence[str] | None = None,
    output_metadata: dict[str, Any] | None = None,
    ffp: NDArray | None = None,
    dfp: NDArray | None = None,
    crop_width: tuple[int, int] | None = None,
    radial_correction_k: float | None = None,
    chunk_size: tuple[int, int] | None = None,
    channels_per_batch: int | None = None,
    channel_reference: int | None = None,
    channel_window: int = 2,
    channel_cross_correlation_upsample: int = 2,
    channel_filter_percentiles: tuple[float, float] | None = None,
    t_index: int | None = None,
    channel_alignment: dict[int, dict] | None = None,
):
    """Use stitching coordinates to fuse tiles.

    :param df: Stitch output dataframe.
    :param group: Group to save the result.
    :param z_index: z-index or 'max'. Ignored when df contains `z_index` column.
    :param blend: Blending mode
    :param output_channels: Optional output channels to include
    :param channel_names: Optional channel names to output, overwriting channel names derived from image tiles
    :param output_metadata: Optional output metadata
    :param ffp: ffp image for illumination correction
    :param dfp: dfp image for illumination correction
    :param crop_width: Image crop width
    :param radial_correction_k: Image radial correction k
    :param chunk_size: Image y, x, chunk size. If `None`, auto determine by clustering
        tile positions into a grid to estimate number of rows and columns.
    :param channels_per_batch: Number of channels to output per batch. If `None`,
        auto determine from available memory
    :param channel_reference: Reference channel to use to register across channels within a tile. If `None`, no
        registration across channels is performed
    :param channel_window: Use a centered subset of data for registration across channels if greater than one
    :param channel_cross_correlation_upsample: Perform subpixel alignment for registration across channels if greater than one
    :param channel_filter_percentiles: Replace data outside of percentile range [q1, q2] with uniform noise over the range
        [q1,q2] for registration across channels.
    :param t_index: The t-index to select from images with a time dimension. Required
        when images have more than one timepoint (e.g. live imaging data).
    :param channel_alignment: Optional per-channel alignment to the reference channel, keyed by input channel
        index (``dict(m, ty, tx, ...)``; see :mod:`scallops.stitch._channel_align`). When given, every output
        channel is resampled once (alignment composed with the radial correction) with a quintic B-spline instead of
        the bilinear radial correction.
    """
    assert blend in ["none", "linear"]
    if channel_alignment is not None and channel_reference is not None:
        raise ValueError(
            "channel_alignment and channel_reference cannot be used together."
        )
    if crop_width is not None and crop_width[0] <= 0 and crop_width[1] <= 0:
        crop_width = None

    df = df.copy()
    image0_paths = df["source"].values[0]
    image0_attrs = (
        df["source_metadata"].values[0] if "source_metadata" in df.columns else None
    )
    has_scenes = "scene" in df.columns
    local_image0_paths = [_localize_path(path) for path in image0_paths]
    image0_paths = [
        local_image0_paths[i] if local_image0_paths[i] is not None else image0_paths[i]
        for i in range(len(image0_paths))
    ]
    img = _images2fov_array(
        image0_paths, image0_attrs, scene_id=0 if has_scenes else None
    )

    for path in local_image0_paths:
        if path is not None:
            os.remove(path)

    n_channels = img.sizes["c"]
    img = _select_t_index(img, t_index).isel(z=0, missing_dims="ignore")
    # Capture channel names from the tiles (e.g. from nd2 metadata via bioio) so
    # they can be written to the output OMERO metadata. Skip integer/positional
    # coordinates since those carry no channel name information.
    tile_channel_names = (
        img.coords["c"].values
        if "c" in img.coords and img.coords["c"].dtype.kind not in "iuf"
        else None
    )
    if output_channels is None:
        output_channels = list(range(n_channels))
    img = img.values
    weights = None
    if blend != "none":
        # weights are cropped when fusing image so use full tile size
        weights = _tile_blending_weights(img.shape[1:])
    full_tile_shape = img.shape[1:]
    img = _crop_image(img, crop_width)
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
    fused_y_size = (df["y"] + ysize).max()
    fused_x_size = (df["x"] + xsize).max()

    # each task fuses all output channels in a batch for one y, x chunk, so the batch size only bounds the memory
    # of a single task
    if channels_per_batch is None or channels_per_batch < 0:
        channels_per_batch = len(output_channels)
    channels_per_batch = min(channels_per_batch, len(output_channels))

    if chunk_size is None:
        # AgglomerativeClustering requires at least 2 samples; a well with a single tile
        # (e.g. a sparse live-imaging FOV) trivially has one partition.
        if len(df) <= 1:
            chunk_size = fused_y_size, fused_x_size
        else:
            step_sizes = []
            for dim, size, fused_size in zip(
                ["y", "x"], tile_shape, [fused_y_size, fused_x_size]
            ):
                cluster = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=size * 0.1, linkage="single"
                )
                cluster.fit_predict(df[[dim]])
                step_sizes.append(
                    math.ceil(fused_size / len(np.unique(cluster.labels_)))
                )
            chunk_size = step_sizes[0], step_sizes[1]

    fmt = _current_format()
    result_zarr_array = group.create_array(
        shape=(
            len(output_channels),  # c
            fused_y_size,
            fused_x_size,
        ),
        dtype=target_dtype,
        chunks=(1,) + chunk_size,
        name="s0",
        overwrite=True,
        **_create_array_kwargs(fmt),
    )

    coords = None
    if channel_alignment is not None:
        # one coordinate map per output channel, computed once and shared by all tiles
        coords = channel_coords(
            full_tile_shape, radial_correction_k, output_channels, channel_alignment
        )

    if ffp is not None:
        if coords is not None:
            ffp = img_as_float(ffp, force_copy=True)
            for c in output_channels:
                ffp[c] = warp_image(ffp[c], coords[c], cval=1)
        elif radial_correction_k is not None:
            ffp = radial_correct(ffp, radial_correction_k, cval=1)

    if dfp is not None:
        dfp = dfp / np.iinfo(target_dtype).max
        if coords is not None:
            for c in output_channels:
                dfp[c] = warp_image(dfp[c], coords[c], cval=0)
        elif radial_correction_k is not None:
            dfp = radial_correct(dfp, radial_correction_k, cval=0)

    # weights are computed on the full tile so crop them like the tiles
    if weights is not None:
        weights = _crop_image(weights, crop_width)

    has_z_index = "z_index" in df.columns
    separate_z = False
    if "separate_z" in df.columns:
        separate_z = df["separate_z"].iloc[0]
        if not df["separate_z"].eq(separate_z).all():
            raise ValueError("separate_z must be all equal to True or False")
    if separate_z:
        has_z_index = False
        z_index = 0

    # with blend "none", df is sorted by ascending priority. _fuse_chunk paints tiles in the given order so tiles
    # with higher priority win where tiles overlap
    source = df["source"].values
    source_attrs = (
        df["source_metadata"].values if "source_metadata" in df.columns else None
    )
    # convert scenes to list to convert numpy int to int (read_image does not like numpy ints for scene ids)
    scene_ids = df["scene"].tolist() if has_scenes else None
    tile_z_indices = df["z_index"].values if has_z_index else None
    tiles = [
        dict(
            image_paths=source[i],
            image_attrs=source_attrs[i] if source_attrs is not None else None,
            y=int(df["y"].values[i]),
            x=int(df["x"].values[i]),
            z_index=tile_z_indices[i] if tile_z_indices is not None else z_index,
            scene_id=scene_ids[i] if scene_ids is not None else None,
        )
        for i in range(len(df))
    ]
    tile_tree = shapely.STRtree(
        [
            shapely.box(
                tile["x"],
                tile["y"],
                min(fused_x_size, tile["x"] + xsize) - 1,
                min(fused_y_size, tile["y"] + ysize) - 1,
            )
            for tile in tiles
        ]
    )

    _fuse_chunk_delayed = delayed(_fuse_chunk)
    # wrap shared arrays once so they are single graph keys instead of being copied into every task
    weights_delayed = delayed(weights) if weights is not None else None
    for channel_batch in range(0, len(output_channels), channels_per_batch):
        channels = output_channels[channel_batch : channel_batch + channels_per_batch]
        # with channel_reference, _load_tile corrects all input channels before registering across channels and
        # selects the output channels afterwards, so it needs the full ffp/dfp
        select = channels if channel_reference is None else slice(None)
        _ffp = delayed(ffp[select]) if ffp is not None else None
        _dfp = delayed(dfp[select]) if dfp is not None else None
        _coords = delayed([coords[c] for c in channels]) if coords is not None else None
        channel_slice = slice(channel_batch, channel_batch + len(channels))
        logger.info(
            f"Fusing {pluralize('channel', len(channels))} {', '.join([str(c) for c in channels])}."
        )

        delayed_results = []
        for y0 in range(0, fused_y_size, chunk_size[0]):
            y1 = min(fused_y_size, y0 + chunk_size[0])
            for x0 in range(0, fused_x_size, chunk_size[1]):
                x1 = min(fused_x_size, x0 + chunk_size[1])
                tile_indices = np.sort(
                    tile_tree.query(
                        shapely.box(x0, y0, x1 - 1, y1 - 1), predicate="intersects"
                    )
                )
                if len(tile_indices) == 0:
                    # the zarr array fill value (0) is the output where no tile is present
                    continue
                delayed_results.append(
                    _fuse_chunk_delayed(
                        out=result_zarr_array,
                        channel_slice=channel_slice,
                        box=(y0, x0, y1, x1),
                        tiles=[tiles[i] for i in tile_indices],
                        target_dtype=target_dtype,
                        blend=blend,
                        weights=weights_delayed,
                        radial_correction_k=radial_correction_k,
                        output_channels=channels,
                        crop_width=crop_width,
                        ffp=_ffp,
                        dfp=_dfp,
                        channel_reference=channel_reference,
                        channel_cross_correlation_upsample=channel_cross_correlation_upsample,
                        channel_window=channel_window,
                        channel_filter_percentiles=channel_filter_percentiles,
                        t_index=t_index,
                        channel_coords=_coords,
                    )
                )

        with ProgressBar():
            dask.compute(*delayed_results)

    # Fall back to the channel names read from the source tiles when the user
    # did not explicitly pass --channel-name.
    if channel_names is None:
        channel_names = tile_channel_names
    if channel_names is not None:
        channel_names = [channel_names[index] for index in output_channels]
        channel_names = [
            name.replace("_", "-").replace(" ", "-") for name in channel_names
        ]

    image_attrs, axes, scale_dict = _attrs_axes_scales(
        output_metadata,
        {"c": channel_names},
        ["c", "y", "x"],
        np.uint16,
    )
    zarr_attrs = _create_zarr_attrs(
        fmt, group, ["c", "y", "x"], image_attrs, axes, scale_dict
    )
    group.attrs.update(zarr_attrs)


def _register_across_channels(
    img: NDArray,
    channel_reference: int | None = None,
    channel_cross_correlation_upsample: int = 1,
    channel_window: int = 1,
    channel_filter_percentiles: tuple[float, float] | None = None,
):
    ref_values = img[channel_reference]
    img_dtype = ref_values.dtype
    if channel_window > 1:
        ref_values = _apply_window(ref_values, channel_window)
    if channel_filter_percentiles is not None:
        ref_values = _filter_percentiles(
            ref_values,
            q1=channel_filter_percentiles[0],
            q2=channel_filter_percentiles[1],
        )

    for c in range(img.shape[0]):
        if c != channel_reference:
            moving = img[c]
            if channel_window > 1:
                moving = _apply_window(moving, channel_window)
            if channel_filter_percentiles is not None:
                moving = _filter_percentiles(
                    moving,
                    q1=channel_filter_percentiles[0],
                    q2=channel_filter_percentiles[1],
                )

            offset, _ = calc_best_shift(
                moving,
                ref_values,
                upsample_factor=channel_cross_correlation_upsample,
                overlap_min=0,
            )
            #   offset, _, _ = phase_cross_correlation(moving, ref_values)

            if not np.all(offset == 0.0):
                # skimage SimilarityTransform has (x,y,[z]) convention
                st = SimilarityTransform(translation=offset[::-1])
                img[c] = warp(img[c], st, preserve_range=True).astype(img_dtype)


def _load_tile(
    image_paths: list[str] | zarr.Group,
    image_attrs: dict[str, str | list[str]] | None,
    target_dtype: np.dtype,
    radial_correction_k: float | None = None,
    output_channels: Sequence[int] | int | None = None,
    crop_width: tuple[int, int] | None = None,
    dfp: NDArray | None = None,
    ffp: NDArray | None = None,
    z_index: int | Literal["max"] = "max",
    scene_id: int | None = None,
    channel_reference: int | None = None,
    channel_cross_correlation_upsample: int = 1,
    channel_window: int = 1,
    channel_filter_percentiles: tuple[float, float] | None = None,
    t_index: int | None = None,
    channel_coords: Sequence[np.ndarray | None] | None = None,
) -> NDArray:
    """Read a tile and apply radial, illumination, and channel corrections.

    :return: Corrected and cropped float tile with dimensions (c, y, x) in [0, 1].
    """
    local_image_paths = [_localize_path(path) for path in image_paths]
    image_paths = [
        local_image_paths[i] if local_image_paths[i] is not None else image_paths[i]
        for i in range(len(image_paths))
    ]

    img = _images2fov(image_paths, image_attrs, scene_id=scene_id)
    img = _select_t_index(img, t_index)
    for path in local_image_paths:
        if path is not None:
            os.remove(path)
    if "z" in img.dims:
        img = (
            img.isel(z=z_index)
            if isinstance(z_index, (int, np.integer))
            else img.max(dim="z")
        )
    img = img.values

    if output_channels is not None and channel_reference is None:
        img = img[output_channels]

    # order: radial (with channel alignment if given), illumination, channel registration, crop
    if channel_coords is not None:
        # one quintic B-spline resampling per channel; channel_coords follows the output channel order. The result
        # stays float (a new array) until the final conversion to target_dtype
        img = np.stack(
            [warp_image(img[i], channel_coords[i]) for i in range(img.shape[0])]
        )
    elif radial_correction_k is not None:
        img = radial_correct(img, radial_correction_k)
        img = dtype_convert(img, target_dtype)

    img = img_as_float(img, force_copy=channel_coords is None)
    # dfp/ffp match img's channels ( the output channels), or contain all input channels when registering across channels
    if dfp is not None:
        img -= dfp
    if ffp is not None:
        img /= ffp
    img.clip(0, 1, out=img)
    if channel_reference is not None:
        _register_across_channels(
            img=img,
            channel_reference=channel_reference,
            channel_cross_correlation_upsample=channel_cross_correlation_upsample,
            channel_window=channel_window,
            channel_filter_percentiles=channel_filter_percentiles,
        )
        if output_channels is not None:
            img = img[output_channels]

    img = _crop_image(img, crop_width)
    return img


def _fuse_chunk(
    out: zarr.Array | NDArray,
    channel_slice: slice,
    box: tuple[int, int, int, int],
    tiles: Sequence[dict[str, Any]],
    target_dtype: np.dtype,
    blend: Literal["none", "linear"] = "none",
    weights: NDArray | None = None,
    **tile_kwargs,
):
    """Fuse the tiles intersecting one output chunk and write the chunk.

    Each call reads every tile it needs and writes only the chunk(s) in `box`, so calls
    can run concurrently on any scheduler (including distributed) without locks.

    :param out: Output array with dimensions (c, y, x).
    :param channel_slice: Channel slice in `out` to write.
    :param box: Chunk bounds (y1, x1, y2, x2) in `out`, exclusive of y2 and x2.
    :param tiles: Tiles intersecting `box`: dicts with keys image_paths, image_attrs,
        y, x, z_index, and scene_id. When `blend` is "none", tiles are painted in order
        so later tiles have priority.
    :param target_dtype: Output dtype.
    :param blend: Blending mode.
    :param weights: Cropped tile blending weights, required when blending.
    :param tile_kwargs: Additional arguments passed to `_load_tile`.
    """
    y0, x0, y1, x1 = box
    shape = (channel_slice.stop - channel_slice.start, y1 - y0, x1 - x0)
    weights_sum = None
    if blend == "none":
        target = np.zeros(shape, dtype=target_dtype)
    else:
        if weights is None:
            raise ValueError("weights are required when blending")
        target = np.zeros(shape)
        weights_sum = np.zeros(shape[1:])
    for tile in tiles:
        img = _load_tile(
            image_paths=tile["image_paths"],
            image_attrs=tile["image_attrs"],
            target_dtype=target_dtype,
            z_index=tile["z_index"],
            scene_id=tile["scene_id"],
            **tile_kwargs,
        )
        y, x = tile["y"], tile["x"]
        # intersection of tile and chunk in output coordinates
        iy0, iy1 = max(y0, y), min(y1, y + img.shape[-2])
        ix0, ix1 = max(x0, x), min(x1, x + img.shape[-1])
        if iy0 >= iy1 or ix0 >= ix1:
            continue
        tile_sl = (slice(iy0 - y, iy1 - y), slice(ix0 - x, ix1 - x))
        target_sl = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        img = dtype_convert(img[(...,) + tile_sl], target_dtype)
        if blend == "none":
            target[(...,) + target_sl] = img
        else:
            assert weights is not None and weights_sum is not None
            tile_weights = weights[tile_sl]
            weights_sum[target_sl] += tile_weights
            target[(...,) + target_sl] += img * tile_weights
    if weights_sum is not None:
        weights_sum[weights_sum == 0] = 1
        target = np.round(target / weights_sum).astype(target_dtype)
    out[channel_slice, y0:y1, x0:x1] = target


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


def _tile_blending_weights(tile_shape: tuple[int, int]) -> NDArray:
    weights = np.zeros(tile_shape)
    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            weights[i, j] = _dist_from_tile_edge(i, j, tile_shape[0], tile_shape[1])
    weights /= weights.max()
    return weights
