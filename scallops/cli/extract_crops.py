import json
from typing import Literal

import dask.array as da
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from array_api_compat import get_namespace
from zarr import Group

from scallops.cli.features import get_labels
from scallops.cli.util import (
    _get_cli_logger,
    cli_metadata,
)
from scallops.features.constants import (
    _label_name_to_prefix,
)
from scallops.features.util import _get_names_from_pd_query
from scallops.io import (
    _images2fov,
    is_parquet_file,
    read_anndata_zarr,
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

    if label_filter is not None:
        label_filter_ = label_filter.format(**metadata["file_metadata"][0])
        if fsspec.url_to_fs(label_filter_)[0].exists(label_filter_):
            label_filter = pd.read_parquet(label_filter_, columns=["label"])["label"]

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_fs.makedirs(output_dir, exist_ok=True)
    image = _images2fov(file_list, metadata, dask=True).squeeze().data
    logger.info(f"Image shape {image.shape}")
    zarr_labels = get_labels(
        labels_group=labels_group,
        name=image_key,
        suffix=label_name,  # e.g. nuclei
    )

    if zarr_labels is None:
        raise ValueError(f"Unable to read {label_name} labels for {image_key}.")
    merge_paths = [
        f"{merge_dir}{merge_dir_sep}{label_name}{merge_dir_sep}{image_key}.parquet",
        f"{merge_dir}{merge_dir_sep}{label_name}{merge_dir_sep}{image_key}.zarr",
        f"{merge_dir}{merge_dir_sep}{label_name}{merge_dir_sep}{image_key}-objects.parquet",
    ]
    merge_path = None
    for path in merge_paths:
        if fsspec.core.url_to_fs(merge_path)[0].exists(merge_path):
            merge_path = path
            break
    if merge_path is None:
        raise ValueError(f"Unable to read merged data for {image_key}.")

    label_prefix = _label_name_to_prefix[label_name]

    area_column = f"{label_prefix}_AreaShape_Area"
    if merge_path.lower().endswith(".zarr"):
        data = read_anndata_zarr(merge_path, dask=True)
        merged_df = data.obs
        columns = {area_column}
        assert area_column in data.var.index
        if isinstance(label_filter, str):
            query_columns = _get_names_from_pd_query(label_filter)
            query_columns = [
                c
                for c in query_columns
                if c not in merged_df.columns and c in data.var.index
            ]
            columns.update(query_columns)
        columns = list(columns)
        if len(columns) > 0:
            values = data[:, columns].X.compute()
            for i in range(len(columns)):
                merged_df[columns[i]] = values[:, i]

    else:
        merged_df = pd.read_parquet(merge_path)
    #  merged_df=adata.obs.query(
    #             "barcode_count_0/barcode_count>0.5 & Nuclei_Correlation_PearsonBox_ISS_PHENO>=0.9 & Cells_Location_IntersectsBoundary_Channel0==False"
    #         )
    n_labels_before_filtering = len(merged_df)
    if label_filter is not None:
        if isinstance(label_filter, str):
            merged_df = merged_df.query(label_filter)
        else:
            merged_df = merged_df[merged_df.index.isin(label_filter)]

    merged_df = merged_df.query(f"{area_column}>=2")

    n_labels_filtered = n_labels_before_filtering - len(merged_df)
    logger.info(f"Removed {n_labels_filtered:,} out of {n_labels_before_filtering:,}.")
    if len(merged_df) == 0:
        raise ValueError("No labels.")
    # e.g. CHAMMI-75

    if percentile_normalize is not None:
        chunksize = list(image.chunksize)
        for i in range(len(chunksize) - 2):
            chunksize[i] = -1
        if chunks is not None:
            chunksize[-2] = chunks
            chunksize[-1] = chunks
        else:
            logger.info(f"Chunk size: {(chunksize[-2],)} by {(chunksize[-1],)}")
        image = image.rechunk(tuple(chunksize))
        if local_percentile_normalize:
            depth = None
            if local_normalize_overlap is not None and local_normalize_overlap > 0:
                depth = {
                    image.ndim - 2: local_normalize_overlap,
                    image.ndim - 1: local_normalize_overlap,
                }
            image = da.map_overlap(
                _norm_block, image, percentiles=percentile_normalize, depth=depth
            )
        else:
            percentiles = da.percentile(
                image, percentile_normalize, axis=(1, 2), keepdims=True
            )
            image = (image - percentiles[0]) / (percentiles[1] - percentiles[0])
            image = da.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    # gaussian_sigma=2
    index = to_label_crops(
        label_image=da.from_zarr(zarr_labels),
        intensity_image=image,
        merged_df=merged_df,
        output_dir=output_dir,
        crop_size=crop_size,
        output_format=output_format,
        centroid_cols=[
            f"{label_prefix}_AreaShape_Center_Y",
            f"{label_prefix}_AreaShape_Center_X",
        ],
        gaussian_sigma=None,
    )
    merged_df = merged_df.loc[index]
    output_metadata = cli_metadata() if not no_version else dict()
    merged_df["crop_url"] = (
        output_dir + "/" + merged_df.index.astype(str) + "." + output_format
    )
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
