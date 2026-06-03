import json
from typing import Literal

import dask.array as da
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from array_api_compat import get_namespace
from skimage.util import img_as_ubyte
from zarr import Group

from scallops.cli.features import _read_merged_or_objects, get_labels
from scallops.cli.util import (
    _get_cli_logger,
    cli_metadata,
)
from scallops.features.constants import (
    _label_name_to_prefix,
)
from scallops.io import (
    _images2fov,
    is_parquet_file,
    to_label_crops,
)

logger = _get_cli_logger()


def _norm_block(image: np.ndarray, percentiles) -> np.ndarray:
    xp = get_namespace(image)
    percentiles = xp.percentile(image, percentiles, axis=(1, 2), keepdims=True)
    image = (image - percentiles[0]) / (percentiles[1] - percentiles[0])
    image = xp.clip(image, 0, 1)
    return image


def single_crop(
    group: str,  # NOT USED
    file_list: list[str],
    metadata: dict,
    labels_group: Group,
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
    image = _images2fov(file_list, metadata, dask=True).squeeze().data
    logger.info(f"{image_key} image shape {image.shape}")
    zarr_labels = get_labels(
        labels_group=labels_group,
        name=image_key,
        suffix=label_name,  # e.g. nuclei
    )

    if zarr_labels is None:
        raise ValueError(f"Unable to read {label_name} labels for {image_key}.")
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
    # e.g. CHAMMI-75

    if percentile_normalize is not None:
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
        if local_percentile_normalize:
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
    image = da.map_blocks(img_as_ubyte, image)
    label_col = "label" if "label" in merged_df.columns else None
    merged_df = to_label_crops(
        label_image=da.from_zarr(zarr_labels),
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

    output_metadata = cli_metadata() if not no_version else dict()

    table = pa.Table.from_pandas(merged_df, preserve_index=True)
    table = table.replace_schema_metadata(
        {
            "scallops".encode(): json.dumps(output_metadata).encode(),
            **table.schema.metadata,
        }
    )

    fs, output_parquet_path = fsspec.url_to_fs(output_parquet_path)
    pq.write_table(
        table,
        output_parquet_path,
        filesystem=fs,
    )
