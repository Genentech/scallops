"""Command-line interface (CLI) module for illumination correction in scallops.

This module provides CLI commands to perform illumination correction using various
methods such as mean-based correction.

Illumination correction is a crucial preprocessing step in image analysis, particularly in
biomedical image processing. It aims to compensate for uneven illumination across an image,
which can arise due to variations in lighting conditions, optics, or sensor sensitivity.
Uneven illumination can introduce artifacts and affect the accuracy of downstream analysis
tasks such as segmentation and feature extraction.

This module provides the illumination correction method:


1. **Mean-Based Illumination Correction:**
   This method calculates illumination correction by computing an aggregation
   (e.g. mean), followed by a median filter and optional rescaling.
   It offers a simple and effective approach for
   addressing illumination variations.


Authors:
    - The SCALLOPS development team
"""

import argparse
from typing import Literal, Optional, Union

import fsspec
import pandas as pd
import zarr
from dask.bag import from_sequence

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.illumination_correction import illumination_correction
from scallops.io import _get_fs_protocol, _set_up_experiment, save_ome_tiff
from scallops.zarr_io import (
    _get_fs,
    _get_sep,
    _write_zarr_image,
    is_ome_zarr_array,
    open_ome_zarr,
)

logger = _get_cli_logger()


def single_agg_illumination_correction(
    image_tuple: tuple[tuple[str, ...], list[str], dict],
    root: Union[zarr.Group, str],
    smooth: Optional[int] = None,
    rescale: bool = True,
    output_image_format: str = "zarr",
    z_index: Literal["max", "focus"] | int | str = "max",
    channel: int = 0,
    force: bool = False,
    agg_method: Literal["mean", "median"] = "mean",
    expected_images: int | None = None,
    no_version: bool = False,
) -> None:
    """Run illumination correction by aggregation for a group of images (typically all
    tiles in a well).

    :param image_tuple: Placeholder argument.
    :param root: Output root directory or zarr group.
    :param smooth: The radius of the disk-shaped footprint for the median filter. If None, no
        smoothing is applied.
    :param rescale: Whether to use the 2nd percentile for robust minimum rescaling.
    :param output_image_format: Output image format, either 'zarr' or 'tiff'.
    :param z_index: Either 'max' to use the maximum z-projection or a specific z-index.
    :param channel: Channel to select best focus z index.
    :param force: Whether to overwrite existing output if it already exists.
    :param agg_method: Method to aggregate images, either 'mean' or 'median'.
    :param expected_images: Number of expected images.
    :param no_version: Whether to skip version/CLI information in output.
    """
    _, image_filepaths, image_metadata = image_tuple
    image_key = image_metadata["id"]
    if not force and (
        (
            output_image_format == "zarr"
            and is_ome_zarr_array(root.get(f"images/{image_key}"))
        )
        or (
            output_image_format != "zarr"
            and fsspec.core.url_to_fs(root)[0].exists(f"{root}{image_key}.ome.tiff")
        )
    ):
        return logger.info(f"Skipping illumination correction for {image_key}")

    logger.info(f"Running illumination correction for {image_key}.")
    save_z_index = isinstance(z_index, str) and z_index == "focus"

    image, keys, z_index = illumination_correction(
        image_tuple=image_tuple,
        smooth=smooth,
        rescale=rescale,
        channel=channel,
        z_index=z_index,
        expected_images=expected_images,
        agg_method=agg_method,
    )
    if save_z_index:
        if output_image_format == "zarr":
            path = root.store.path.rstrip(_get_sep(root))
            if path.endswith(".zarr"):
                path = path[: -len(".zarr")]
            protocol = _get_fs_protocol(_get_fs(root))
            if protocol != "file":
                path = f"{protocol}://{path}"
            path = f"{path}{_get_sep(root)}{image_key}-zindex.parquet"
        else:
            path = f"{root}{image_key}-zindex.parquet"
        pd.DataFrame(data=dict(key=keys, z_index=z_index)).to_parquet(path, index=False)
    output_metadata = {}

    if not no_version:
        output_metadata.update(cli_metadata())

    if output_image_format == "zarr":
        _write_zarr_image(
            name=image_key, root=root, image=image, metadata=output_metadata
        )
    else:
        save_ome_tiff(
            data=image.data,
            dim_order="CYX",
            uri=f"{root}{image_key}.ome.tiff",
            attrs=output_metadata,
        )


def run_illumination_correction_agg(arguments: argparse.Namespace):
    """Run mean-based illumination correction for a set of images.

    :param arguments: Parsed command-line arguments.
    """
    image_path = arguments.images
    dask_server_url = arguments.client
    default_dask_cluster_parameters = dict(processes=False)
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster)
        if arguments.dask_cluster is not None
        else default_dask_cluster_parameters
    )

    image_pattern = arguments.image_pattern
    group_by = arguments.groupby
    if len(group_by) == 0:
        group_by = ("*",)  # group all files together
    subset = arguments.subset
    output = arguments.output
    output_image_format = arguments.output_image_format
    z_index = arguments.z_index
    channel = arguments.channel
    if z_index not in ("max", "focus"):
        try:
            z_index = int(z_index)
        except ValueError:
            pass
    smooth = arguments.smooth
    force = arguments.force
    agg_method = arguments.agg_method
    expected_images = arguments.expected_images
    rescale = not arguments.no_rescale

    if output_image_format == "zarr":
        output = open_ome_zarr(output, mode="a")
    else:
        fs, _ = fsspec.core.url_to_fs(output)
        output = output.rstrip(fs.sep)
        if output != "":
            output += fs.sep
        fs.makedirs(output, exist_ok=True)

    generator = _set_up_experiment(image_path, image_pattern, group_by, subset=subset)
    image_bag = from_sequence(generator)
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        image_bag.map(
            single_agg_illumination_correction,
            root=output,
            smooth=smooth,
            rescale=rescale,
            agg_method=agg_method,
            z_index=z_index,
            channel=channel,
            output_image_format=output_image_format,
            force=force,
            expected_images=expected_images,
        ).compute()
