import argparse
from collections.abc import Sequence
from typing import Literal

import dask.array as da
import fsspec
import numpy as np
from array_api_compat import get_namespace
from numpy.typing import NDArray
from skimage.util import img_as_ubyte
from zarr import Group

from scallops.cli.features import _read_merged_or_objects, get_labels
from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.features.constants import (
    _label_name_to_prefix,
)
from scallops.io import (
    _images2fov_array,
    _pd_to_parquet,
    _set_up_experiment,
    is_parquet_file,
    to_label_crops,
)

logger = _get_cli_logger()


def _norm_block(image: NDArray, percentiles) -> NDArray:
    xp = get_namespace(image)
    percentiles = xp.percentile(image, percentiles, axis=(1, 2), keepdims=True)
    image = (image - percentiles[0]) / (percentiles[1] - percentiles[0])
    image = xp.clip(image, 0, 1)
    return image


def single_crop(
    group: tuple[str, ...],  # NOT USED
    file_list: Sequence[str],
    metadata: dict,
    label_groups: list[Group] | None,
    output_dir: str,
    output_sep: str,
    merge_dir: str,
    merge_dir_sep: str,
    crop_size: tuple[int, int],
    output_format: Literal["tiff", "npy"],
    label_name: str,
    label_filter: str | None,
    percentile_normalize: tuple[float, float] | None,
    local_percentile_normalize: bool,
    local_normalize_overlap: int | None,
    gaussian_sigma: int | None,
    chunks: int | None,
    no_version: bool,
    force: bool,
):
    image_key = metadata["id"]

    output_dir = f"{output_dir}{output_sep}{label_name}{output_sep}{image_key}"

    output_parquet_path = f"{output_dir}.parquet"
    if not force and is_parquet_file(output_parquet_path):
        logger.info(f"Skipping features for {image_key} {label_name}")
        return

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_fs.makedirs(output_dir, exist_ok=True)
    image = _images2fov_array(file_list, metadata, dask=True).squeeze().data
    logger.info(f"{image_key} image shape {image.shape}")
    label_image = None
    if label_groups is not None and len(label_groups) > 0:
        zarr_labels = get_labels(
            label_groups=label_groups,
            name=image_key,
            suffix=label_name,  # e.g. nuclei
        )
        label_image = da.from_zarr(zarr_labels)
    merged_df = _read_merged_or_objects(
        merge_dir=merge_dir,
        merge_dir_sep=merge_dir_sep,
        label_name=label_name,
        image_key=image_key,
        label_filter=label_filter,
    )
    if merged_df is None:
        raise ValueError(f"Unable to read merged data for {image_key}.")
    n_labels_before_filtering = len(merged_df)
    if label_filter is not None:
        merged_df = merged_df.query(label_filter)
    label_prefix = _label_name_to_prefix[label_name]
    area_column = f"{label_prefix}_AreaShape_Area"
    merged_df = merged_df.query(f"{area_column}>=2")
    n_labels_filtered = n_labels_before_filtering - len(merged_df)
    logger.info(
        f"Removed {n_labels_filtered:,} out of {n_labels_before_filtering:,} labels for {image_key}."
    )
    if len(merged_df) == 0:
        raise ValueError(f"No labels found for {image_key}.")

    if percentile_normalize is not None:
        if local_percentile_normalize:
            chunksize = list(image.chunksize)
            for i in range(len(chunksize) - 2):
                chunksize[i] = -1
            if chunks is not None:
                chunksize[-2] = chunks
                chunksize[-1] = chunks
            else:
                logger.info(
                    f"{image_key} chunk size: {chunksize[-2]:,} by {chunksize[-1]:,}"
                )
            image = image.rechunk(tuple(chunksize))
            depth = None
            if local_normalize_overlap is not None and local_normalize_overlap > 0:
                depth = {
                    image.ndim - 2: local_normalize_overlap,
                    image.ndim - 1: local_normalize_overlap,
                }
            image = da.map_overlap(
                _norm_block,
                image,
                percentiles=percentile_normalize,
                depth=depth,
                dtype=float,
            )
        else:
            percentiles = da.percentile(
                image, percentile_normalize, axis=(1, 2), keepdims=True
            )
            image = (image - percentiles[0]) / (percentiles[1] - percentiles[0])
            image = da.clip(image, 0, 1)
    image = da.map_blocks(img_as_ubyte, image, dtype=np.uint8)
    label_col = "label" if "label" in merged_df.columns else None
    merged_df = to_label_crops(
        label_image=label_image,
        intensity_image=image,
        df=merged_df,
        label_col=label_col,
        output_dir=output_dir,
        crop_size=crop_size,
        output_format=output_format,
        centroid_cols=[
            f"{label_prefix}_AreaShape_Center_Y",
            f"{label_prefix}_AreaShape_Center_X",
        ],
        gaussian_sigma=gaussian_sigma,
    )

    _pd_to_parquet(
        merged_df, output_parquet_path, cli_metadata() if not no_version else None
    )


def run_pipeline_extract_crops(arguments: argparse.Namespace):
    images_paths = arguments.images
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()
    image_patterns = arguments.image_pattern
    output_dir = arguments.output
    merge_dir = arguments.merge
    subset = arguments.subset
    force = arguments.force
    groupby = arguments.groupby
    crop_size = arguments.crop_size
    crop_size = (crop_size, crop_size)
    label_filter = arguments.label_filter
    mask = arguments.mask
    percentile_min = arguments.percentile_min
    percentile_max = arguments.percentile_max
    output_format = arguments.output_format
    local_percentile_normalize = arguments.local_percentile_normalize
    local_normalize_overlap = arguments.local_percentile_overlap
    percentile_normalize = None
    if percentile_min <= 0 or percentile_min >= 100:
        percentile_min = None
    if percentile_max <= 0 or percentile_max >= 100:
        percentile_max = None
    if percentile_min is not None or percentile_max is not None:
        if percentile_min is None:
            percentile_min = 0
        if percentile_max is None:
            percentile_max = 100
        percentile_normalize = (percentile_min, percentile_max)
    gaussian_sigma = arguments.gaussian_sigma
    if gaussian_sigma is not None and not mask:
        raise ValueError("Please specify `mask` flag when providing `gaussian sigma`")
    label_name = arguments.label_name  # cell, cytosol, nuclei
    chunks = arguments.chunks

    merge_dir_sep = fsspec.core.url_to_fs(merge_dir)[0].sep
    merge_dir = merge_dir.rstrip(merge_dir_sep)

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)

    label_groups = arguments.labels
    no_version = arguments.no_version

    if not mask and label_groups is not None and len(label_groups) > 0:
        logger.warning("Ignoring provided labels")
        label_groups = []
    if (label_groups is None or len(label_groups) == 0) and mask:
        raise ValueError("Labels must be provided when `mask` is true.")

    with (
        _create_default_dask_config(
            {"distributed.admin.large-graph-warning-threshold": "250MB"}
        ),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        for image_tuple in _set_up_experiment(
            images_paths,
            files_pattern=image_patterns,
            subset=subset,
            group_by=groupby,
        ):
            single_crop(
                *image_tuple,
                output_dir=output_dir,
                output_sep=output_fs.sep,
                merge_dir=merge_dir,
                merge_dir_sep=merge_dir_sep,
                label_groups=label_groups,
                label_filter=label_filter,
                label_name=label_name,
                percentile_normalize=percentile_normalize,
                local_normalize_overlap=local_normalize_overlap,
                local_percentile_normalize=local_percentile_normalize,
                crop_size=crop_size,
                output_format=output_format,
                gaussian_sigma=gaussian_sigma,
                chunks=chunks,
                force=force,
                no_version=no_version,
            )
