import json
from typing import Literal

import dask.array as da
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from array_api_compat import get_namespace
from skimage.util import img_as_ubyte

from scallops.cli.features import (
    _find_labels,
    _image_key_without_time_and_selected_time,
    _read_merged_or_objects,
)
from scallops.cli.find_objects import get_path
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


def _norm_block(intensity_image: np.ndarray, percentiles) -> np.ndarray:
    xp = get_namespace(intensity_image)
    percentiles = xp.percentile(
        intensity_image, percentiles, axis=(1, 2), keepdims=True
    )
    intensity_image = (intensity_image - percentiles[0]) / (
        percentiles[1] - percentiles[0]
    )
    intensity_image = xp.clip(intensity_image, 0, 1)
    return intensity_image


def single_crop(
    group: str,  # NOT USED
    file_list: list[str],
    metadata: dict,
    label_paths: list[str],
    output_dir: str,
    merge_dirs: list[str],
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
    image = _images2fov(file_list, metadata, dask=True).squeeze().data

    image_key_without_t, selected_timepoint = _image_key_without_time_and_selected_time(
        metadata
    )

    g, timepoints = _find_labels(
        label_paths=label_paths,
        image_key=image_key,
        label_name=label_name,
        image_key_without_t=image_key_without_t,
        selected_timepoint=selected_timepoint,
    )
    if g is None:
        logger.info(f"No labels found for {image_key}")
        return
    labels_array = da.from_array(g[list(g.keys())[0]])
    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_sep = output_fs.sep
    for timepoint in timepoints:
        features_path = get_path(
            output_dir, output_sep, label_name, image_key, timepoint, ".parquet"
        )
        output_dir = get_path(
            output_dir, output_sep, label_name, image_key, timepoint, ""
        )

        if not force and is_parquet_file(features_path):
            logger.info(
                f"Skipping crops for {image_key} {label_name}{' at t=' + timepoint if timepoint is not None else ''}."
            )
            continue
        if timepoint is not None and labels_array.ndim == 3:
            timepoint_index = timepoints.index(timepoint)
            label_image = labels_array[timepoint_index]
        else:
            label_image = labels_array
        intensity_image = (
            image.sel(t=timepoint)
            if timepoint is not None and image.sizes.get("t", 0) > 1
            else image
        )
        merged_df = _read_merged_or_objects(
            paths=merge_dirs,
            label_name=label_name,
            timepoint=timepoint,
            image_key=image_key,
            image_key_without_t=image_key_without_t,
            label_filter=label_filter,
        )
        if merged_df is None:
            raise ValueError(f"Unable to read metadata for {image_key}.")
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
            if local_percentile_normalize:
                chunksize = list(intensity_image.chunksize)
                for i in range(len(chunksize) - 2):
                    chunksize[i] = -1
                if chunks is not None:
                    chunksize[-2] = chunks
                    chunksize[-1] = chunks
                else:
                    logger.info(
                        f"{image_key} chunk size: {chunksize[-2]:,} by {chunksize[-1]:,}"
                    )
                intensity_image = intensity_image.rechunk(tuple(chunksize))
                depth = None
                if local_normalize_overlap is not None and local_normalize_overlap > 0:
                    depth = {
                        intensity_image.ndim - 2: local_normalize_overlap,
                        intensity_image.ndim - 1: local_normalize_overlap,
                    }
                intensity_image = da.map_overlap(
                    _norm_block,
                    intensity_image,
                    percentiles=percentile_normalize,
                    depth=depth,
                    dtype=float,
                )
            else:
                percentiles = da.percentile(
                    intensity_image, percentile_normalize, axis=(1, 2), keepdims=True
                )
                intensity_image = (intensity_image - percentiles[0]) / (
                    percentiles[1] - percentiles[0]
                )
                intensity_image = da.clip(intensity_image, 0, 1)
        intensity_image = da.map_blocks(img_as_ubyte, intensity_image)
        label_col = "label" if "label" in merged_df.columns else None
        merged_df = to_label_crops(
            label_image=label_image,
            intensity_image=intensity_image,
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

        fs, features_path = fsspec.url_to_fs(features_path)
        pq.write_table(
            table,
            features_path,
            filesystem=fs,
        )
