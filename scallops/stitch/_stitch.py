import gc
import json
import math
import os
import shutil
from collections.abc import Sequence
from typing import Literal

import dask.array as da
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from sklearn.cluster import AgglomerativeClustering
from zarr.errors import PathNotFoundError

from scallops.cli.util import _get_cli_logger, cli_metadata
from scallops.io import is_parquet_file, read_image
from scallops.stitch._align import stitch_align
from scallops.stitch._plots import _qc_report
from scallops.stitch.fuse import _create_label_ome_metadata, _create_ome_metadata, _fuse
from scallops.stitch.shift_utils import _zncc, convert_stage_positions
from scallops.stitch.utils import (
    _download_path,
    _init_tiles,
    _stage_positions_from_image_metadata,
    get_pixel_size,
    read_stage_positions,
    tile_overlap_mask,
    tile_source_labels,
)
from scallops.utils import _dask_from_array_no_copy
from scallops.zarr_io import is_ome_zarr_array

logger = _get_cli_logger()


def _single_stitch(
    image_tuple: tuple[tuple[str, ...], list[str], dict],
    stage_positions_path: str | None,
    image_output_root: zarr.Group | None,
    ffp_path: str | None,
    dfp_path: str | None,
    channel: int,
    upsample_factor: int,
    z_index: int | Literal["focus", "max"] | str,
    image_spacing: tuple[float, float] | None,
    radial_correction_k: float | None | Literal["auto", "none"],
    blend: Literal["none", "linear"],
    crop_width: int | None,
    stitch_alpha: float,
    evaluate: bool,
    no_save_labels: bool,
    no_save_image: bool,
    output_channels: list[int] | None,
    no_version: bool,
    other_output_path: str,
    rename: dict[str, str] | None,
    expected_images: int | None,
    force: bool,
    min_overlap_fraction: float | None,
    random_seed: int,
    max_shifts: Sequence[float],
    flip_y_axis: int | None,
    flip_x_axis: int | None,
    swap_axes: bool | None,
):
    """Process a single cycle of images."""
    _, image_filepaths, image_metadata = image_tuple

    image_key = (
        rename.get(image_metadata["id"], image_metadata["id"])
        if rename
        else image_metadata["id"]
    )
    image_key = image_key.replace("/", "_")
    if not force:
        if not no_save_image:
            try:
                if is_ome_zarr_array(image_output_root.get(f"images/{image_key}")):
                    logger.info(f"Skipping stitching for {image_key}.")
                    return
            except PathNotFoundError:
                pass
        elif not no_save_labels:
            try:
                if is_ome_zarr_array(image_output_root.get(f"labels/{image_key}-mask")):
                    logger.info(f"Skipping stitching for {image_key}.")
                    return
            except PathNotFoundError:
                pass
        elif is_parquet_file(f"{other_output_path}{image_key}-positions.parquet"):
            logger.info(f"Skipping stitching for {image_key}.")
            return
    logger.info(f"Running stitching for {image_key}.")
    if ffp_path:
        ffp_path = ffp_path.format(**image_metadata["file_metadata"][0])
        assert fsspec.url_to_fs(ffp_path)[0].exists(ffp_path), f"{ffp_path} not found."
    if dfp_path:
        dfp_path = dfp_path.format(**image_metadata["file_metadata"][0])
        assert fsspec.url_to_fs(dfp_path)[0].exists(dfp_path), f"{dfp_path} not found."
    output_metadata = {
        "blend": blend,
        "z_index": z_index,
        "channel": channel,
        "upsample_factor": upsample_factor,
        "random_seed": random_seed,
    }

    init = _init_tiles(
        image_filepaths=image_filepaths,
        image_metadata=image_metadata,
        channel=channel,
        z_index=z_index,
        expected_images=expected_images,
    )
    z_index = init["z_index"]

    metadata_fields = init["metadata_fields"]
    n_scenes = init["n_scenes"]
    tmp_dir = init["tmp_dir"]
    filepaths = init["filepaths"]
    fileattrs = init["fileattrs"]
    original_filepaths = init["original_filepaths"]

    separate_z = (
        "z" in metadata_fields
        and isinstance(z_index, (Sequence, np.ndarray))
        and not isinstance(z_index, str)
    )

    if radial_correction_k == "none" or (
        isinstance(radial_correction_k, float) and radial_correction_k <= 0
    ):
        radial_correction_k = None
    if crop_width is not None and crop_width <= 0:
        crop_width = None

    primary_filepaths = [paths[0] for paths in filepaths]
    stage_positions = None
    if stage_positions_path is not None:
        stage_positions_path = stage_positions_path.format(
            **image_metadata["file_metadata"][0]
        )
        stage_positions = read_stage_positions(primary_filepaths, stage_positions_path)

    if stage_positions is None:
        stage_positions = _stage_positions_from_image_metadata(primary_filepaths)

    if image_spacing is None:
        image_spacing = get_pixel_size(primary_filepaths, stage_positions_path)

    output_metadata["image_spacing"] = (
        image_spacing.tolist()
        if isinstance(image_spacing, np.ndarray)
        else image_spacing
    )

    auto_radial_correction = radial_correction_k == "auto"
    stitch_result = stitch_align(
        filepaths=filepaths,
        fileattrs=fileattrs,
        n_scenes=n_scenes,
        radial_correction_k=radial_correction_k,
        crop_width=crop_width,
        image_spacing=image_spacing,
        channel=channel,
        z_index=z_index if not separate_z else 0,
        stage_positions=stage_positions,
        upsample_factor=upsample_factor,
        ncc_func=_zncc,
        stitch_alpha=stitch_alpha,
        evaluate_stitching=evaluate,
        min_overlap_fraction=min_overlap_fraction,
        random_seed=random_seed,
        max_shifts=max_shifts,
        flip_y_axis=flip_y_axis,
        flip_x_axis=flip_x_axis,
        swap_axes=swap_axes,
    )

    max_shift = stitch_result["max_shift"]
    crop_width = stitch_result["crop_width"]
    # convert from numpy type to int for JSON serialization
    fuse_crop_width = int(stitch_result["fuse_crop_width"])
    if crop_width is not None:
        crop_width = int(crop_width)
    tile_shape_no_crop = stitch_result["tile_shape"]
    align_tile_shape = stitch_result["align_tile_shape"]
    spanning_tree_edges = stitch_result["spanning_tree_edges"]
    null_params = stitch_result["null_params"]
    z_threshold = stitch_result["z_threshold"]
    zncc_val = stitch_result["zncc_val"]
    radial_correction_k = stitch_result["radial_correction_k"]
    output_metadata["area_fraction"] = stitch_result["area_fraction"]
    output_metadata["null_params"] = null_params
    output_metadata["radial_correction_k"] = radial_correction_k
    output_metadata["min_overlap_fraction"] = stitch_result["min_overlap_fraction"]
    swap = stitch_result["swap"]
    flip_y = stitch_result["flip_y"]
    flip_x = stitch_result["flip_x"]
    final_shifts = stitch_result["final_shifts"]
    pairs = stitch_result["pairs"]
    delta_shifts = stitch_result["delta_shifts"]
    nccs_before_stitching = stitch_result["nccs"]
    zncc_values = stitch_result["zncc_values"]
    min_overlap_fraction = stitch_result["min_overlap_fraction"]
    valid_edges = stitch_result["valid_edges"]
    pairs_after_stitching = stitch_result["pairs_after_stitching"]
    shifts_after_stitching = stitch_result["shifts_after_stitching"]
    fractions_after_stitching = stitch_result["fractions_after_stitching"]

    output_metadata["max_shift"] = max_shift
    output_metadata["crop_width"] = crop_width
    output_metadata["fuse_crop_width"] = fuse_crop_width
    output_metadata["align_tile_shape"] = align_tile_shape
    output_metadata["tile_shape"] = tile_shape_no_crop
    if z_threshold is not None:
        output_metadata["z_threshold"] = z_threshold

    if spanning_tree_edges is not None:
        output_metadata["spanning_tree_edges"] = spanning_tree_edges

    if zncc_val is not None:
        output_metadata["zncc"] = zncc_val

    stitch_positions_df = pd.DataFrame(
        stitch_result["stitched_positions"], columns=["y", "x"]
    )
    if isinstance(z_index, (Sequence, np.ndarray)) and not isinstance(z_index, str):
        stitch_positions_df["z_index"] = z_index
    del stitch_result

    stitch_position_coords = stitch_positions_df[["y", "x"]].values
    center = stitch_position_coords.mean(axis=0)
    stitch_positions_df["distance_to_center"] = np.sqrt(
        np.sum(stitch_position_coords - center, axis=1) ** 2
    )

    stitch_positions_df["source"] = original_filepaths
    if fileattrs is not None:
        stitch_positions_df["source_metadata"] = fileattrs
    stitch_positions_df["tile"] = np.arange(len(stitch_positions_df))
    stitch_positions_df["original_y"] = stage_positions[:, 0]
    stitch_positions_df["original_x"] = stage_positions[:, 1]

    if not no_version:
        output_metadata.update(cli_metadata())
    stitch_positions_table = pa.Table.from_pandas(
        stitch_positions_df, preserve_index=False
    )
    stitch_positions_table = stitch_positions_table.replace_schema_metadata(
        {
            "scallops".encode(): json.dumps(output_metadata).encode(),
            **stitch_positions_table.schema.metadata,
        }
    )
    position_path = f"{other_output_path}{image_key}-positions.parquet"
    fs, position_path = fsspec.url_to_fs(position_path)
    pq.write_table(
        stitch_positions_table,
        position_path,
        filesystem=fs,
    )

    cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tile_shape_no_crop[0] * 0.1,
        linkage="single",
    )
    cluster.fit_predict(stitch_positions_df[["y"]])
    n_partitions_y = len(np.unique(cluster.labels_))

    cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tile_shape_no_crop[1] * 0.1,
        linkage="single",
    )
    cluster.fit_predict(stitch_positions_df[["x"]])
    n_partitions_x = len(np.unique(cluster.labels_))

    # replace source with local source
    stitch_positions_df_local = stitch_positions_df.copy()
    stitch_positions_df_local["source"] = filepaths
    logger.info(f"Saving report to {other_output_path}{image_key}.pdf.")
    _qc_report(
        path=f"{other_output_path}{image_key}.pdf",
        stage_positions=convert_stage_positions(
            stage_positions,
            swap,
            flip_y,
            flip_x,
            (1, 1),
        ),
        final_shifts=final_shifts,
        spanning_tree_edges=spanning_tree_edges,
        pairs=pairs,
        zncc_val=zncc_val,
        shifts_before_stitching=(delta_shifts * image_spacing)
        if delta_shifts is not None
        else None,
        nccs_before_stitching=nccs_before_stitching,
        final_positions=stitch_positions_df[["y", "x"]].values,
        no_version=no_version,
        zncc_values=zncc_values,
        pairs_after_stitching=pairs_after_stitching,
        shifts_after_stitching=shifts_after_stitching,
        fractions_after_stitching=fractions_after_stitching,
        min_overlap_fraction=min_overlap_fraction,
        tile_shape=align_tile_shape,
        valid_edges=valid_edges,
        max_shift=max_shift,
        radial_correction_k=radial_correction_k if auto_radial_correction else None,
    )
    if pairs_after_stitching is not None:
        eval_df = pd.DataFrame(
            data=dict(
                zncc=zncc_values,
                pair=pairs_after_stitching.tolist(),
                shift=shifts_after_stitching.tolist(),
                fraction=fractions_after_stitching,
            )
        )
        eval_table = pa.Table.from_pandas(eval_df, preserve_index=False)
        eval_table = eval_table.replace_schema_metadata(
            {
                "scallops".encode(): json.dumps(output_metadata).encode(),
                **eval_table.schema.metadata,
            }
        )
        eval_path = f"{other_output_path}{image_key}-eval.parquet"
        fs, eval_path = fsspec.url_to_fs(eval_path)
        pq.write_table(
            eval_table,
            eval_path,
            filesystem=fs,
        )
    fused_tile_shape = (
        tile_shape_no_crop[0] - fuse_crop_width * 2,
        tile_shape_no_crop[1] - fuse_crop_width * 2,
    )
    fused_y_size = (
        np.round(stitch_positions_df["y"].max()).astype(int) + fused_tile_shape[0]
    )
    fused_x_size = (
        np.round(stitch_positions_df["x"].max()).astype(int) + fused_tile_shape[1]
    )

    chunk_size = (
        math.ceil(fused_y_size / n_partitions_y),
        math.ceil(fused_x_size / n_partitions_x),
    )
    _write_arrays(
        stitch_positions_df,
        stitch_positions_df_local,
        blend,
        image_output_root,
        image_key,
        fused_y_size,
        fused_x_size,
        fused_tile_shape,
        chunk_size,
        image_spacing,
        no_save_labels,
        no_save_image,
        ffp_path,
        dfp_path,
        z_index,
        output_channels,
        fuse_crop_width,
        radial_correction_k,
        n_scenes is not None,
        output_metadata,
    )
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_arrays(
    stitch_positions_df,
    stitch_positions_df_local,
    blend,
    image_output_root,
    image_key,
    fused_y_size,
    fused_x_size,
    fused_tile_shape,
    chunk_size,
    image_spacing,
    no_save_labels,
    no_save_image,
    ffp_path,
    dfp_path,
    z_index,
    output_channels,
    fuse_crop_width,
    radial_correction_k,
    scenes,
    metadata,
):
    gc.collect()
    if not no_save_labels:
        labels_group = image_output_root.require_group("labels")
        group = labels_group.create_group(image_key + "-mask", overwrite=True)

        array = group.create_dataset(
            name="0",
            shape=(fused_y_size, fused_x_size),
            chunks=chunk_size,
            dtype=np.uint8,
            dimension_separator="/",
            overwrite=True,
        )

        da.to_zarr(
            arr=_dask_from_array_no_copy(
                tile_overlap_mask(
                    stitch_positions_df,
                    fill=blend != "none",
                    tile_shape=fused_tile_shape,
                ),
                chunks=chunk_size,
            ),
            url=array,
            compute=True,
            dimension_separator="/",
        )
        group.attrs.update(
            _create_label_ome_metadata(image_spacing, image_key + "-mask")
        )
        if blend == "none":
            group = labels_group.create_group(image_key + "-tile", overwrite=True)
            array = group.create_dataset(
                name="0",
                shape=(fused_y_size, fused_x_size),
                chunks=chunk_size,
                dtype=np.uint16,
                dimension_separator="/",
                overwrite=True,
            )

            da.to_zarr(
                arr=_dask_from_array_no_copy(
                    tile_source_labels(stitch_positions_df, fused_tile_shape),
                    chunks=chunk_size,
                ),
                url=array,
                compute=True,
                dimension_separator="/",
            )
            label_metadata = _create_label_ome_metadata(
                image_spacing, image_key + "-tile"
            )
            label_metadata["multiscales"][0]["metadata"] = {
                "source": f"../../images/{image_key}"
            }
            group.attrs.update(label_metadata)
    cleanup_paths = []
    if not no_save_image:
        group = image_output_root.require_group("images").require_group(
            image_key, overwrite=True
        )

        ffp = None
        dfp = None
        if ffp_path is not None:
            local_path = _download_path(ffp_path)
            if local_path is not None:
                cleanup_paths.append(local_path)
                ffp_path = local_path
            ffp = read_image(ffp_path)
            # check for swap
            if ffp.sizes["z"] > 1 and ffp.sizes["c"] == 1:
                ffp = ffp.rename({"z": "c", "c": "z"})
            squeeze_dims = [d for d in ["t", "z"] if d in ffp.dims]

            if len(squeeze_dims) > 0:
                ffp = ffp.squeeze(squeeze_dims)
            if "z" in ffp.dims:
                if isinstance(z_index, str) and z_index == "max":
                    ffp = ffp.max(dim="z")
                elif isinstance(z_index, int):
                    ffp = ffp.isel(z=z_index)
                else:
                    raise ValueError(f"Unknown z index:{z_index}")

            ffp = ffp.values

        if dfp_path is not None:
            local_path = _download_path(dfp_path)
            if local_path is not None:
                cleanup_paths.append(local_path)
                dfp_path = local_path
            dfp = read_image(dfp_path)
            # check for swap
            if dfp.sizes["z"] > 1 and dfp.sizes["c"] == 1:
                dfp = dfp.rename({"z": "c", "c": "z"})
            squeeze_dims = [d for d in ["t", "z"] if d in dfp.dims]
            if len(squeeze_dims) > 0:
                dfp = dfp.squeeze(squeeze_dims)
            if "z" in dfp.dims:
                if isinstance(z_index, str) and z_index == "max":
                    dfp = dfp.max(dim="z")
                elif isinstance(z_index, int):
                    dfp = dfp.isel(z=z_index)
                else:
                    raise ValueError(f"Unknown z index:{z_index}")

            dfp = dfp.values

        _fuse(
            df=stitch_positions_df_local,
            group=group,
            z_index=z_index,
            blend=blend,
            output_channels=output_channels,
            ffp=ffp,
            dfp=dfp,
            crop_width=fuse_crop_width,
            radial_correction_k=radial_correction_k,
            chunk_size=chunk_size,
            scenes=scenes,
        )

        ome_metadata = _create_ome_metadata(
            image_key=image_key,
            stitch_coords=stitch_positions_df,
            **metadata,
        )
        group.attrs.update(ome_metadata)

    # cleanup
    for path in cleanup_paths:
        if os.path.exists(path):
            os.remove(path)
