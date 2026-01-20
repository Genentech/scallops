"""
Module: scallops.cli.stitch

Provides a command-line interface (CLI) for performing stitching of microscope images using Ashlar.

Differences from original ashlar code:
 - Reads stage positions directly from Bioformats supported images (e.g., nd2).
 - Outputs OME-ZARR files.
 - Outputs stitched and original positions.
 - Option to disable blending when stitching.
 - Option to specify z-index or perform max-z projection.
 - Compatible with newer versions of skimage.
 - Option to exit or warn when alignment fails on enough edges.
"""

import argparse
import shutil
from collections.abc import Sequence
from typing import Literal

import fsspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
import skimage.util
from dask.bag import from_sequence
from distributed import Client
from skimage.util import img_as_uint

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _get_cli_logger,
    load_json,
)
from scallops.io import _add_suffix, _images2fov, _set_up_experiment
from scallops.stitch._align import _get_read_images
from scallops.stitch._stitch import _single_stitch
from scallops.stitch.shift_utils import (
    convert_stage_positions,
    determine_layout,
)
from scallops.stitch.utils import (
    _init_tiles,
    _stage_positions_from_image_metadata,
    get_pixel_size,
    read_stage_positions,
)
from scallops.utils import _cpu_count
from scallops.xr import _z_projection
from scallops.zarr_io import open_ome_zarr

logger = _get_cli_logger()


def single_stitch_preview(
    image_tuple: tuple[tuple[str, ...], list[str], dict],
    output_path: str,
    channel: int,
    bounds: bool,
    numbers: bool,
    log: bool,
    downsample: float,
    show_image: bool,
    stage_positions_path: str | None = None,
    z_index: Literal["max", "focus"] | int = "max",
) -> None:
    """Generate a preview of the stitching.

    :param image_tuple: A tuple containing file metadata, file list, and metadata dictionary.
    :param output_path: Path to the output directory for the preview.
    :param channel: Channel index to visualize.
    :param stage_positions_path: Optional stage positions.
    :param bounds: Whether to display tile boundaries.
    :param numbers: Whether to display tile numbers.
    :param log: Whether to apply log transformation.
    :param downsample: Downsampling factor for the preview.
    :param show_image: Whether to show images.
    """
    _, image_filepaths, image_metadata = image_tuple
    init = _init_tiles(
        image_filepaths=image_filepaths,
        image_metadata=image_metadata,
        channel=channel,
        z_index=z_index,
        expected_images=None,
    )
    z_index = init["z_index"]
    n_scenes = init["n_scenes"]
    tmp_dir = init["tmp_dir"]
    filepaths = init["filepaths"]
    fileattrs = init["fileattrs"]
    metadata_fields = init["metadata_fields"]
    primary_filepaths = [paths[0] for paths in filepaths]
    image_key = image_metadata["id"]

    stage_positions = None
    if stage_positions_path is not None:
        stage_positions_path = stage_positions_path.format(
            **image_metadata["file_metadata"][0]
        )

        stage_positions = read_stage_positions(primary_filepaths, stage_positions_path)

    if stage_positions is None:
        stage_positions = _stage_positions_from_image_metadata(primary_filepaths)
    logger.info(f"Previewing {image_key} with {len(stage_positions):,} tiles")

    read_images = _get_read_images(
        filepaths=filepaths,
        fileattrs=fileattrs,
        channel=channel,
        n_scenes=n_scenes,
    )
    image_spacing = get_pixel_size(primary_filepaths, stage_positions_path)
    swap, flip_y, flip_x, area_fraction, tile_shape_no_crop, center_tile, max_shift = (
        determine_layout(
            read_images=read_images,
            stage_positions=stage_positions,
            image_spacing=image_spacing,
            max_shifts=[np.inf],
        )
    )

    logger.info(
        f"Flip y axis: {str(flip_y == -1).lower()}, "
        f"flip x axis: {str(flip_x == -1).lower()}, swap axes: {str(swap).lower()}."
    )

    if image_spacing is None:
        image_spacing = get_pixel_size(primary_filepaths, stage_positions_path)
    stage_positions = convert_stage_positions(
        stage_positions, swap, flip_y, flip_x, image_spacing
    )

    origin = stage_positions.min(axis=0)
    stage_positions -= origin

    resolution_scale = 1 / downsample

    stage_positions = stage_positions * resolution_scale
    stage_positions = np.round(stage_positions).astype(int)

    fig, ax = plt.subplots()
    ax.axis("off")
    tile_shape = np.round(np.array(tile_shape_no_crop) * resolution_scale).astype(int)
    z_index_per_tile = isinstance(z_index, (Sequence, np.ndarray)) and not isinstance(
        z_index, str
    )
    z_tiles_removed = (
        "z" in metadata_fields
        and isinstance(z_index, (Sequence, np.ndarray, int))
        and not isinstance(z_index, str)
    )
    if z_tiles_removed:
        z_index = 0
    mosaic_image_shape = stage_positions.max(axis=0) + tile_shape
    if show_image:
        mosaic_image = np.zeros(mosaic_image_shape, dtype=np.uint16)
        n = n_scenes if n_scenes is not None else len(filepaths)
        for i in range(n):
            img = _images2fov(
                filepaths[i],
                None,
                dask=False,
                scene_id=i if n_scenes is not None else None,
            )
            img = img.isel(t=0, c=channel, missing_dims="ignore")
            z_index_ = z_index[i] if z_index_per_tile else z_index
            img = (
                _z_projection(img, z_index_)
                if not z_tiles_removed
                else img.isel(z=0, missing_dims="ignore")
            )
            img = img.values
            img = skimage.transform.rescale(img, resolution_scale, anti_aliasing=False)
            img = img_as_uint(img)
            if log:
                img = np.log1p(np.maximum(img, 1)).astype(np.uint16)
            y, x = stage_positions[i]
            mosaic_image[y : y + img.shape[0], x : x + img.shape[1]] = img

        ax.imshow(mosaic_image)

    if bounds or numbers:
        if not show_image:
            ax.set_ylim(0, mosaic_image_shape[0])
            ax.set_xlim(0, mosaic_image_shape[1])
            ax.invert_yaxis()
        for i in range(len(stage_positions)):
            y, x = stage_positions[i]
            if bounds:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x, y), *tile_shape, edgecolor="black", fill=False
                    )
                )
            if numbers:
                ax.add_patch(
                    mpatches.Circle(
                        (x + tile_shape[1] / 2, y + tile_shape[0] / 2),
                        5,
                        color="salmon",
                    )
                )
                ax.text(
                    x + tile_shape[1] / 2,
                    y + tile_shape[0] / 2,
                    str(i),
                    ha="center",
                    va="center",
                )

    fs = fsspec.url_to_fs(output_path)[0]
    output_path = output_path.rstrip(fs.sep)
    if not fs.exists(output_path):
        fs.mkdirs(output_path, exist_ok=True)
    with fs.open(f"{output_path}{fs.sep}{image_key}.png", "wb") as f:
        plt.savefig(f)
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_stitch_preview(args: argparse.Namespace) -> None:
    """Run the stitch preview pipeline."""
    if not args.bounds and not args.numbers and args.no_tiles:
        raise ValueError("Please provide bounds or numbers or disable tiles.")
    from_sequence(
        _set_up_experiment(
            args.images, args.image_pattern, args.groupby, subset=args.subset
        )
    ).map(
        single_stitch_preview,
        output_path=args.output,
        channel=args.channel,
        numbers=args.numbers,
        bounds=args.bounds,
        downsample=args.downsample,
        log=args.log,
        stage_positions_path=args.stage_positions,
        show_image=not args.no_tiles,
        z_index=args.z_index,
    ).compute()


def run_stitch(args: argparse.Namespace) -> None:
    """Run the full stitching pipeline."""
    dask_server_url = args.client
    dask_cluster_parameters = (
        load_json(args.dask_cluster) if args.dask_cluster is not None else {}
    )
    no_save_image = args.no_save_image

    if dask_server_url is None and args.dask_cluster is None:
        dask_cluster_parameters = dict(n_workers=1, threads_per_worker=_cpu_count())

    if (
        not no_save_image
        and dask_server_url is None
        and dask_cluster_parameters.get("n_workers") != 1
    ):
        raise ValueError("Stitching can only be run with 1 worker.")
    blend = args.blend
    output_channels = args.output_channels

    no_save_labels = args.no_save_labels
    image_output = args.image_output
    other_output_path = args.report_output
    stage_positions_path = args.stage_positions
    dfp = args.dfp
    ffp = args.ffp
    flip_y_axis = args.flip_y_axis
    flip_x_axis = args.flip_x_axis
    swap_axes = args.swap_axes
    if flip_y_axis is not None:
        flip_y_axis = 1 if flip_y_axis == 0 else -1
    if flip_x_axis is not None:
        flip_x_axis = 1 if flip_x_axis == 0 else -1

    if swap_axes is not None:
        swap_axes = bool(swap_axes)
    radial_correction_k = args.radial_correction_k
    if radial_correction_k is not None:
        if radial_correction_k not in ("auto", "none"):
            radial_correction_k = float(radial_correction_k)
        elif radial_correction_k == "none":
            radial_correction_k = None
    force = args.force
    evaluate = not args.no_evaluate
    rename = args.rename
    if rename is not None:
        rename = pd.read_csv(rename, header=None, index_col=0).to_dict()[1]
    no_version = args.no_version
    stitch_alpha = args.stitch_alpha

    image_spacing = args.image_spacing
    channel = args.align_channel
    upsample_factor = args.cross_correlation_upsample
    z_index = args.z_index
    if z_index not in ("max", "focus"):
        try:
            z_index = int(z_index)
        except ValueError:
            pass

    expected_images = args.expected_images
    crop_width = args.crop
    min_overlap_fraction = args.min_overlap_fraction
    random_seed = args.random_seed
    max_shifts = args.max_shift
    assert all(shift >= 0 for shift in max_shifts), "Max shift must be non-negative."

    if crop_width is not None:
        assert crop_width >= 0, "Crop must be positive."
    if crop_width == 0:
        crop_width = None
    assert 0 <= stitch_alpha <= 1, "stitch alpha must be between 0 and 1"
    if min_overlap_fraction is not None:
        assert 0 <= min_overlap_fraction <= 1, (
            "min overlap fraction must be between 0 and 1"
        )
    assert upsample_factor >= 1, "upsample factor must be greater than or equal to 1"

    fs, _ = fsspec.core.url_to_fs(other_output_path)
    other_output_path = other_output_path.rstrip(fs.sep) + fs.sep
    fs.makedirs(other_output_path, exist_ok=True)
    image_output_root = None

    if image_output is None and (not no_save_image or not no_save_labels):
        raise ValueError("Please provide output zarr directory")
    elif not no_save_image or not no_save_labels:
        image_output = _add_suffix(image_output, ".zarr")
        image_output_root = open_ome_zarr(image_output, mode="a")
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters) as client,
    ):
        if (
            no_save_image
            and isinstance(client, Client)
            and len(client.cluster.workers) != 1
        ):
            raise ValueError("Stitching can only be run with 1 worker.")
        from_sequence(
            _set_up_experiment(
                args.images, args.image_pattern, args.groupby, subset=args.subset
            )
        ).map(
            _single_stitch,
            image_output_root=image_output_root,
            other_output_path=other_output_path,
            radial_correction_k=radial_correction_k,
            dfp_path=dfp,
            ffp_path=ffp,
            stitch_alpha=stitch_alpha,
            z_index=z_index,
            force=force,
            image_spacing=image_spacing,
            upsample_factor=upsample_factor,
            expected_images=expected_images,
            output_channels=output_channels,
            channel=channel,
            rename=rename,
            crop_width=crop_width,
            no_save_image=no_save_image,
            no_save_labels=no_save_labels,
            no_version=no_version,
            stage_positions_path=stage_positions_path,
            blend=blend,
            evaluate=evaluate,
            min_overlap_fraction=min_overlap_fraction,
            random_seed=random_seed,
            max_shifts=max_shifts,
            flip_y_axis=flip_y_axis,
            flip_x_axis=flip_x_axis,
            swap_axes=swap_axes,
        ).compute()
