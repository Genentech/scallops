"""The `register` module provides functionality for image registration using ITK and performing
cross-correlation-based registration across different cycles."""

import argparse
import os
from collections.abc import Sequence
from itertools import zip_longest
from typing import Literal

import fsspec
import itk
import numpy as np
import xarray as xr
import zarr
from dask.bag import from_sequence
from zarr import Group

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.io import (
    _add_suffix,
    _get_fs_protocol,
    _images2fov,
    _set_up_experiment,
    get_image_spacing,
    pluralize,
)
from scallops.registration.crosscorrelation import align_image
from scallops.registration.itk import (
    _itk_align_reference_time_zarr,
    _itk_transform_image_zarr,
    _load_itk_parameters,
    _load_itk_parameters_from_dir,
    itk_align,
    itk_transform_labels,
    set_automatic_transform_initialization,
)
from scallops.registration.landmarks import _get_translation, find_landmarks
from scallops.utils import _cpu_count
from scallops.xr import _z_projection
from scallops.zarr_io import (
    _get_fs,
    _write_zarr_image,
    is_ome_zarr_array,
    open_ome_zarr,
    read_ome_zarr_array,
)

logger = _get_cli_logger()


def single_registration(
    fixed_tuple: tuple[tuple[str, ...], list[str | Group], dict] | None,
    moving_tuple: tuple[tuple[str, ...], list[str | Group], dict],
    moving_channel: list[int],
    fixed_channel: int | None,
    label_output_root: zarr.Group | None,
    image_output_root: zarr.Group | None,
    transform_output_dir: str,
    transform_fs: fsspec.AbstractFileSystem,
    itk_parameters: list[str],
    moving_labels: list[str],
    moving_image_spacing: tuple[float, float] | None,
    fixed_image_spacing: tuple[float, float] | None,
    reference_timepoint: int | str,
    unroll_channels: bool = False,
    force: bool = False,
    z_index: int | str = "max",
    initial_transform: str | None = None,
    landmarks_initialize: bool = False,
    landmark_slice_size: float = 200,
    landmark_template_padding: Sequence[float] = 500,
    landmark_step_size: float = 1000,
    landmark_min_score: float = 0.6,
    landmark_initializations: Sequence[Literal["com", "none"]] = ("com", "none"),
    landmark_com_min_quantile: float | None = None,
    landmark_com_max_quantile: float | None = None,
    landmark_min_count: int = 100,
    output_aligned_channels_only: bool = False,
    itk_channel_parameters: list[str] | None = None,
    no_version: bool = False,
) -> str:
    """Perform single registration between fixed and moving images.

    This function executes a registration process between a fixed and a moving image. It supports
    registration of images with different channels, and the result is saved to specified output
    roots.

    :param fixed_tuple: Tuple containing fixed image information.
    :param moving_tuple: Tuple containing moving image information.
    :param moving_channel: Index of the moving image channel.
    :param fixed_channel: Index of the fixed image channel.
    :param label_output_root: Root for output labels.
    :param image_output_root: Root for output images.
    :param z_index: Either 'max' or z-index
    :param transform_output_dir: Directory for saving transformation parameters.
    :param transform_fs: Filesystem for the transform directory.
    :param itk_parameters: List of ITK parameter files.
    :param moving_labels: Paths to moving labels.
    :param moving_image_spacing: Spacing of the moving image.
    :param fixed_image_spacing: Spacing of the fixed image.
    :param force: Whether to overwrite existing output
    :param reference_timepoint: Index or value of timepoint to register to when registering
        across timepoints
    :param unroll_channels: Whether to unroll channels across timepoints.
    :param landmarks_initialize: Use landmarks to initialize registration.
    :param landmark_slice_size: The slice size in physical coordinates
    :param landmark_template_padding: Template padding in physical coordinates.
    :param landmark_step_size: Image step sizes.
    :param landmark_min_score: Minimum score threshold.
    :param landmark_initializations: Automatic estimation of initial alignment for landmark
        estimation
    :param landmark_com_max_quantile: Cap values above specified quantile for center of
        mass computation.
    :param landmark_min_count: Ensure minimum number of landmarks
    :param initial_transform: Path to initial transformation file.
    :param output_aligned_channels_only: Whether to output aligned channels only
    :param no_version: Whether to skip version/CLI information in output.
    :return: Key of the registered image.
    """

    _, moving_file_list, moving_metadata = moving_tuple
    image_key = moving_metadata["id"]
    register_self = fixed_tuple is None
    transform_dest = f"{transform_output_dir}{transform_fs.sep}{image_key}"
    moving_label_keys = []

    if moving_labels is not None:
        for moving_label in moving_labels:
            moving_label_keys.extend(
                get_matching_names(
                    image_key=image_key, image_dir=moving_label, labels=True
                )
            )
    moving_label_keys = sorted(moving_label_keys)
    if len(moving_label_keys) > 0 and len(moving_label_keys) == 0:
        logger.warning(f"No labels found for {image_key}")

    if not force:
        labels_exist = True
        if label_output_root is not None:
            if not register_self:
                for key in moving_label_keys:
                    key = os.path.basename(key)
                    if not is_ome_zarr_array(label_output_root.get(f"labels/{key}")):
                        labels_exist = False
                        break
            # TODO check for transformed labels when register_self

        image_exists = True
        if image_output_root is not None:
            if not is_ome_zarr_array(image_output_root.get(f"images/{image_key}")):
                image_exists = False
        elif label_output_root is None:
            image_exists = False
        if labels_exist and image_exists:
            logger.info(f"Skipping registration for {image_key}")
            return image_key

    if register_self:
        logger.info(f"Running registration for {image_key} t={reference_timepoint}")
        logger.info(
            f"{len(moving_file_list):,} {pluralize('input', len(moving_file_list))}:"
            f" {', '.join([s.name.replace('/images/', '') if isinstance(s, zarr.Group) else str(s) for s in moving_file_list])}"
        )
    else:
        logger.info(f"Running registration for {image_key}")

    if not register_self:
        _, fixed_file_list, fixed_metadata = fixed_tuple

        assert fixed_metadata["id"] == moving_metadata["id"], (
            f"{fixed_metadata['id']} != {moving_metadata['id']}"
        )

        fixed_image = _images2fov(
            fixed_file_list,
            fixed_metadata,
            concat_dims=("c",),
            dask=True,
        )
        if isinstance(fixed_image, Sequence):
            fixed_image = fixed_image[0]
        fixed_image = _z_projection(fixed_image, z_index).isel(
            t=0, c=fixed_channel, missing_dims="ignore"
        )
    moving_image = _images2fov(
        moving_file_list,
        moving_metadata,
        dask=True,
        concat_dims=("c",),
    )

    parameter_object = _load_itk_parameters(itk_parameters)
    parameter_object_across_channels = (
        _load_itk_parameters(itk_channel_parameters)
        if itk_channel_parameters is not None and len(itk_channel_parameters) > 0
        else None
    )
    if parameter_object_across_channels is not None:
        set_automatic_transform_initialization(parameter_object_across_channels, False)

    transform_fs.makedirs(transform_dest, exist_ok=True)

    if not register_self:
        if isinstance(moving_image, Sequence):
            moving_image = moving_image[0]
        if (
            moving_image_spacing is None
            and get_image_spacing(moving_image.attrs) is None
        ):
            raise ValueError(
                f"Physical size not found for moving image for {image_key}."
            )

        if fixed_image_spacing is None and get_image_spacing(fixed_image.attrs) is None:
            raise ValueError(
                f"Physical size not found for fixed image for {image_key}."
            )

        moving_image_align = _z_projection(moving_image, z_index).isel(
            t=0, c=moving_channel, missing_dims="ignore"
        )
        if "c" in moving_image_align.dims and moving_image_align.sizes["c"] > 1:
            moving_image_align = moving_image_align.median(dim="c", keep_attrs=True)

        if "c" in fixed_image.dims and fixed_image.sizes["c"] > 1:
            fixed_image = fixed_image.median(dim="c", keep_attrs=True)
        if np.issubdtype(fixed_image.dtype, np.floating) and not np.issubdtype(
            moving_image_align.dtype, np.floating
        ):
            moving_image_align = moving_image_align.astype(fixed_image.dtype)
        if np.issubdtype(moving_image_align.dtype, np.floating) and not np.issubdtype(
            fixed_image.dtype, np.floating
        ):
            fixed_image = fixed_image.astype(moving_image_align.dtype)
        fixed_image = fixed_image.squeeze()
        moving_image_align = moving_image_align.squeeze()
        chunksize = fixed_image.data.chunksize[-2:]
        image_attrs = dict(
            physical_pixel_sizes=fixed_image_spacing
            or get_image_spacing(fixed_image.attrs)
        )
        if not no_version:
            image_attrs.update(cli_metadata())

        set_automatic_transform_initialization(
            parameter_object, initial_transform is None and not landmarks_initialize
        )
        landmarks = None
        if transform_fs.isdir(transform_dest):
            transform_fs.rm(transform_dest, recursive=True)
        transform_fs.makedirs(transform_dest)
        if landmarks_initialize:
            template_labels = None
            if len(moving_label_keys) > 0:
                template_labels = read_ome_zarr_array(moving_label_keys[-1], dask=True)
            landmarks_found = False
            grid_results = None
            for landmark_translation_attempt in range(len(landmark_initializations)):
                landmark_translation = landmark_initializations[
                    landmark_translation_attempt
                ]
                translation = _get_translation(
                    translation=landmark_translation,
                    image=fixed_image,
                    template=moving_image_align,
                    image_spacing=fixed_image_spacing,
                    template_spacing=moving_image_spacing,
                    com_min_quantile=landmark_com_min_quantile,
                    com_max_quantile=landmark_com_max_quantile,
                )
                for padding_attempt in range(len(landmark_template_padding)):
                    grid_results = find_landmarks(
                        image=fixed_image,
                        template=moving_image_align,
                        template_labels=template_labels,
                        slice_size=landmark_slice_size,
                        template_padding=landmark_template_padding[padding_attempt],
                        step_size=landmark_step_size,
                        image_labels=None,
                        translation=translation,
                        image_spacing=fixed_image_spacing,
                        template_spacing=moving_image_spacing,
                    ).compute()

                    query = ["inlier", f"score>{landmark_min_score}"]
                    if template_labels is not None:
                        query.append("n_template_labels>0")
                    grid_results_filtered = grid_results.query(" & ".join(query))
                    landmarks = dict(
                        fixed_y=grid_results_filtered["y_start_microns"].values,
                        fixed_x=grid_results_filtered["x_start_microns"].values,
                        moving_y=grid_results_filtered["moving_y_microns"].values,
                        moving_x=grid_results_filtered["moving_x_microns"].values,
                    )
                    if len(grid_results_filtered) >= landmark_min_count:
                        landmarks_found = True
                        break
                if landmarks_found:
                    break

            if not landmarks_found:
                raise ValueError(f"Not enough landmarks found for {image_key}.")
            if grid_results is not None:
                grid_results.to_parquet(
                    f"{transform_dest}{transform_fs.sep}landmarks.parquet", index=False
                )
        elastix_object = itk_align(
            fixed_image=fixed_image,
            moving_image=moving_image_align,
            parameter_object=parameter_object,
            fixed_image_spacing=fixed_image_spacing,
            moving_image_spacing=moving_image_spacing,
            landmarks=landmarks,
            initial_transform=initial_transform,
            output_directory=transform_dest,
        )

        del fixed_image
        if not output_aligned_channels_only:
            del moving_image_align
        if image_output_root is not None:  # save moving image
            _itk_transform_image_zarr(
                image=moving_image
                if not output_aligned_channels_only
                else moving_image_align.expand_dims({"c": 1}, axis=0),
                transform_parameter_object=elastix_object.GetTransformParameterObject(),
                image_attrs=image_attrs,
                image_spacing=moving_image_spacing,
                image_name=image_key,
                image_root=image_output_root,
                chunksize=chunksize,
            )
        if output_aligned_channels_only:
            del moving_image_align
        if moving_labels is not None:
            transform_all_labels(
                transform_parameter_object=elastix_object.GetTransformParameterObject(),
                attrs=moving_image.attrs,
                matching_keys=moving_label_keys,
                moving_image_spacing=moving_image_spacing,
                output_root=label_output_root,
            )

    else:  # align to t=reference_timepoint
        if isinstance(reference_timepoint, str):
            for i in range(len(moving_image)):
                if moving_image[i].coords["t"].values[0] == reference_timepoint:
                    reference_timepoint = i
                    break
        set_automatic_transform_initialization(parameter_object, False)
        if output_aligned_channels_only and not isinstance(moving_image, xr.DataArray):
            new_moving_image = []
            for img in moving_image:
                new_moving_image.append(img.isel(c=[moving_channel]))
            moving_channel = 0
            moving_image = new_moving_image
        if not no_version:
            moving_image[reference_timepoint].attrs.update(cli_metadata())
        _itk_align_reference_time_zarr(
            unroll_channels=unroll_channels,
            reference_timepoint=reference_timepoint,
            moving_image=moving_image,
            moving_channel=moving_channel,
            parameter_object=parameter_object,
            moving_image_spacing=moving_image_spacing,
            output_dir=transform_dest,
            image_name=image_key,
            image_root=image_output_root,
            landmarks_initialize=landmarks_initialize,
            landmark_com_min_quantile=landmark_com_min_quantile,
            landmark_com_max_quantile=landmark_com_max_quantile,
            landmark_slice_size=landmark_slice_size,
            landmark_template_padding=landmark_template_padding,
            landmark_step_size=landmark_step_size,
            landmark_min_score=landmark_min_score,
            landmark_translations=landmark_initializations,
            landmark_min_count=landmark_min_count,
            parameter_object_across_channels=parameter_object_across_channels,
        )
        moving_image_attrs = moving_image[0].attrs.copy()
        del moving_image

        if len(moving_label_keys) > 0:
            # transform_dest structure is image_key/t=1
            # assume labels are named image_key-t-suffix
            transform_fs_protocol = _get_fs_protocol(transform_fs)
            for transform_file in transform_fs.ls(
                transform_dest, detail=True, refresh=True
            ):
                if transform_file["type"] == "directory":
                    transform_name = transform_file["name"]
                    basename = os.path.basename(transform_name)
                    if basename.startswith("t="):
                        time = basename[2:]
                        moving_label_keys_t = []
                        label_prefix = f"{image_key}-{time}-"
                        for moving_label_key in moving_label_keys:
                            basename = os.path.basename(moving_label_key)
                            if basename.startswith(label_prefix):
                                moving_label_keys_t.append(moving_label_key)

                        if len(moving_label_keys_t) > 0:
                            if transform_fs_protocol != "file":
                                transform_name = (
                                    f"{transform_fs_protocol}://{transform_name}"
                                )

                            transform_parameter_object = _load_itk_parameters_from_dir(
                                transform_name
                            )
                            if (
                                transform_parameter_object.GetNumberOfParameterMaps()
                                > 0
                            ):
                                transform_all_labels(
                                    transform_parameter_object=transform_parameter_object,
                                    attrs=moving_image_attrs,
                                    matching_keys=moving_label_keys_t,
                                    moving_image_spacing=moving_image_spacing,
                                    output_root=label_output_root,
                                )

    return image_key


def get_matching_names(
    image_key: str, image_dir: str | Group, labels: bool = True
) -> list[str]:
    """Get matching keys for the given image key and directory.

    This function retrieves matching keys for a specified image key and directory. It is
    particularly useful when searching for labels associated with a given image.

    :param image_key: Key of the image.
    :param image_dir: Directory containing images.
    :param labels: Whether to look for labels.
    :return: Matching keys.
    """
    # look for f'labels/image_key-{suffix} or  f'images/image_key
    zarr_dir = "labels" if labels else "images"
    if isinstance(image_dir, Group):
        protocol = _get_fs_protocol(_get_fs(image_dir))
        image_dir = f"{image_dir.store.path}{image_dir.name}"
        if protocol != "file":
            image_dir = f"{protocol}://{image_dir}"

    image_fs, _ = fsspec.core.url_to_fs(image_dir)
    image_dir = image_dir.rstrip(image_fs.sep)

    glob_pattern = f"{image_dir}{image_fs.sep}{zarr_dir}{image_fs.sep}{image_key}"
    if labels:
        glob_pattern += "-*"
    paths = image_fs.glob(glob_pattern)
    protocol = _get_fs_protocol(image_fs)
    if protocol != "file":
        paths = [f"{protocol}://{x}" for x in paths]
    results = []
    for path in paths:
        name = os.path.basename(path)
        if not name.startswith(".") and is_ome_zarr_array(zarr.open(path, "r")):
            results.append(path)
    return results


def transform_all_images(
    image_key: str,
    transform_parameter_object: itk.ParameterObject,
    matching_keys: Sequence[str],
    output_root: zarr.Group,
    moving_image_spacing: None | tuple[float, float],
    attrs: None | dict,
    channels_transform_parameter_objects: dict[int, itk.ParameterObject] | None = None,
):
    """Transform and save images.

    This function applies a specified transformation and saves the results.
    It is designed to work with ITK transformations and Zarr storage.

    :param image_key: Key of the image.
    :param transform_parameter_object: ITK parameter object.
    :param matching_keys: Matching keys for transformation.
    :param output_root: Root for output storage.
    :param moving_image_spacing: Spacing of the moving image.
    :param attrs: Additional attributes for the transformed array.
    :param channels_transform_parameter_objects: Maps channel index to ITK parameter
        object
    """

    for key in matching_keys:
        array = read_ome_zarr_array(key, dask=True)
        chunksize = array.data.chunksize[-2:]
        if attrs is not None:
            array.attrs = attrs  # e.g. copy physical size
        logger.info(f"Running transformation for {image_key}.")
        array = array.compute()
        _itk_transform_image_zarr(
            image=array,
            transform_parameter_object=transform_parameter_object,
            channels_transform_parameter_objects=channels_transform_parameter_objects,
            image_attrs=dict(),
            image_spacing=moving_image_spacing,
            image_name=image_key,
            image_root=output_root,
            chunksize=chunksize,
        )


def transform_all_labels(
    transform_parameter_object: itk.ParameterObject,
    matching_keys: Sequence[str],
    output_root: zarr.Group,
    moving_image_spacing: None | tuple[float, float],
    attrs: None | dict,
):
    """Transform and save labels.

    This function applies a specified transformation and saves the results.
    It is designed to work with ITK transformations and Zarr storage.

    :param transform_parameter_object: ITK parameter object.
    :param matching_keys: Matching keys for transformation.
    :param output_root: Root for output storage.
    :param moving_image_spacing: Spacing of the moving image.
    :param attrs: Additional attributes for the transformed array.
    """
    for key in matching_keys:
        name = os.path.basename(key)
        array = read_ome_zarr_array(key)

        if attrs is not None:
            array.attrs = attrs  # e.g. copy physical size

        logger.info(f"Running transformation for {name}.")
        transformed_array = itk_transform_labels(
            image=array,
            transform_parameter_object=transform_parameter_object,
            image_spacing=moving_image_spacing,
        )
        del array

        _write_zarr_image(
            name=name,
            root=output_root,
            image=transformed_array,
            group="labels",
        )


def single_transform(
    transform_dir: str,
    image_dir: str,
    image_spacing: None | tuple[float, float],
    transform_type: Literal["images", "labels"],
    output_root: zarr.Group,
    force: bool = False,
) -> None:
    """Perform transformation for a single image.

    This function applies a transformation to a single image based on the specified parameters. It
    is used in scenarios where transformation parameters are stored in a directory.

    :param transform_dir: Directory containing transform information.
    :param image_dir: Directory containing images.
    :param image_spacing: Spacing of the image.
    :param transform_type: Type of transformation (images or labels).
    :param output_root: Root for output storage.
    :param force: Whether to overwrite existing outputs.
    """

    image_key = os.path.basename(transform_dir)
    matching_keys = get_matching_names(
        image_key=image_key, image_dir=image_dir, labels=transform_type == "labels"
    )
    labels = transform_type == "labels"
    if len(matching_keys) == 0:  # see if directory is A1/t=0 for example
        tokens = transform_dir.split("/")
        if len(tokens) >= 2:
            image_key = tokens[-2]
            matching_keys = get_matching_names(
                image_key=image_key,
                image_dir=image_dir,
                labels=transform_type == "labels",
            )

    if len(matching_keys) == 0:
        logger.info(f"No matching {transform_type} to transform found for {image_key}.")

    if not force:
        _matching_keys = []
        if labels:
            for key in matching_keys:
                if output_root.get(f"labels/{os.path.basename(key)}") is not None:
                    logger.info(f"Skipping transformation for {os.path.basename(key)}.")
                else:
                    _matching_keys.append(key)
        else:
            if output_root.get(f"images/{image_key}") is not None:
                logger.info(f"Skipping transformation for {image_key}.")
            else:
                _matching_keys = matching_keys
        matching_keys = _matching_keys
    if labels:
        transform_all_labels(
            transform_parameter_object=_load_itk_parameters_from_dir(transform_dir),
            matching_keys=matching_keys,
            output_root=output_root,
            moving_image_spacing=image_spacing,
            attrs=None,
        )
    else:
        # see if transform dir has subdirectory describing channel transformation
        transform_fs, _ = fsspec.url_to_fs(transform_dir)
        channels_transform_parameter_objects = dict()
        for d in transform_fs.ls(transform_dir, detail=True, refresh=True):
            if d["type"] == "directory" and os.path.basename(
                d["name"].startswith("c=")
            ):
                channel = int(os.path.basename(d["name"]).split("=")[1])
                channels_transform_parameter_objects[channel] = (
                    _load_itk_parameters_from_dir(d["name"])
                )
        if len(channels_transform_parameter_objects) > 0:
            raise ValueError(f"Channel transformations not supported for {image_key}")
        transform_all_images(
            image_key=image_key,
            transform_parameter_object=_load_itk_parameters_from_dir(transform_dir),
            matching_keys=matching_keys,
            output_root=output_root,
            moving_image_spacing=image_spacing,
            attrs=None,
            channels_transform_parameter_objects=channels_transform_parameter_objects,
        )


def run_itk_transform(arguments: argparse.Namespace) -> None:
    """Transform moving images using previously computed ITK transformations.

    This function processes command-line arguments to transform a collection of images using ITK
    transformations. It takes into account the transformation type (images or labels), input image
    spacing, and the output directory to store the transformed images.

    :param arguments: An argparse.Namespace object containing command-line arguments.
    """
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )

    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = dict(n_workers=1, threads_per_worker=_cpu_count())

    images = arguments.images
    transform_dir = arguments.transform_dir
    output_dir = arguments.output.rstrip("/")
    transform_type = arguments.type
    force = arguments.force
    image_spacing = arguments.image_spacing
    output_dir = _add_suffix(output_dir, ".zarr")

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)

    output_root = open_ome_zarr(output_dir, mode="a")
    transform_fs, _ = fsspec.core.url_to_fs(transform_dir)
    transform_dir = transform_dir.rstrip(transform_fs.sep)
    transform_dirs = transform_fs.ls(transform_dir, detail=True, refresh=True)
    transform_dirs = [
        transform_dir
        for transform_dir in transform_dirs
        if transform_dir["type"] == "directory"
        and os.path.basename(transform_dir["name"])[0] != "."
    ]
    if len(transform_dirs) == 0:
        transform_dirs = [transform_fs.info(transform_dir)]
    transform_fs_protocol = _get_fs_protocol(transform_fs)
    transform_dirs = [f"{transform_fs_protocol}://{x['name']}" for x in transform_dirs]
    transform_dirs = from_sequence(transform_dirs)
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        transform_dirs.map(
            single_transform,
            image_dir=images,
            image_spacing=image_spacing,
            transform_type=transform_type,
            output_root=output_root,
            force=force,
        ).compute()


def run_itk_registration(arguments: argparse.Namespace) -> None:
    """Perform image registration using ITK.

    This function orchestrates the image registration process, aligning a set of moving images to a
    fixed image. It handles various parameters such as image spacing, channel indices, and output
    directories for transformed images and labels.

    :param arguments: An argparse.Namespace object containing command-line arguments.
    """
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url == "none" and dask_cluster_parameters:
        # If a JSON is passed to dask_cluster avoid null_context
        dask_server_url = None

    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = dict(threads_per_worker=1, n_workers=_cpu_count())

    sort_order = arguments.sort
    moving_image_path = arguments.moving
    fixed_image_path = arguments.fixed
    moving_image_pattern = arguments.moving_image_pattern
    fixed_image_pattern = arguments.fixed_image_pattern
    group_by = arguments.groupby
    reference_timepoint = arguments.time
    if reference_timepoint is not None:
        try:
            reference_timepoint = int(reference_timepoint)
        except ValueError:
            pass
    unroll_channels = arguments.unroll_channels
    transform_output_dir = arguments.transform_output_dir
    subset = arguments.subset
    moving_channel = arguments.moving_channel
    label_output_dir = arguments.label_output_dir
    moving_output_dir = arguments.moving_output_dir
    fixed_channel = arguments.fixed_channel
    itk_parameters = arguments.itk_parameters
    moving_labels = arguments.moving_label
    no_version = arguments.no_version
    z_index = arguments.z_index
    moving_image_spacing = arguments.moving_image_spacing
    fixed_image_spacing = arguments.fixed_image_spacing
    force = arguments.force

    landmark_min_count = arguments.landmark_min_count
    landmarks_initialize = not arguments.no_landmarks
    landmark_slice_size = arguments.landmark_image_chunk_size
    landmark_template_padding = arguments.landmark_template_padding
    landmark_step_size = arguments.landmark_step_size
    landmark_min_score = arguments.landmark_min_score
    landmark_initializations = arguments.landmark_initialization
    landmark_com_min_quantile = arguments.landmark_com_min_quantile
    landmark_com_max_quantile = arguments.landmark_com_max_quantile

    itk_channel_parameters = arguments.itk_channels
    if (
        itk_channel_parameters is not None
        and len(itk_channel_parameters) > 0
        and fixed_image_path is not None
    ):
        raise ValueError(
            "Registering across channels not supported when fixed and moving images are provided."
        )
    if landmark_com_min_quantile is not None and (
        landmark_com_min_quantile <= 0.0 or landmark_com_min_quantile >= 1
    ):
        landmark_com_min_quantile = None
    if landmark_com_max_quantile is not None and (
        landmark_com_max_quantile <= 0.0 or landmark_com_max_quantile >= 1
    ):
        landmark_com_max_quantile = None
    initial_transform = arguments.initial_transform
    output_aligned_channels_only = arguments.output_aligned_channels_only

    if initial_transform is not None and fixed_image_path is None:
        raise ValueError(
            "Initial transformation not supported when fixed and moving images are provided."
        )
    transform_fs, _ = fsspec.core.url_to_fs(transform_output_dir)
    transform_output_dir = transform_output_dir.rstrip(transform_fs.sep)
    transform_fs.makedirs(transform_output_dir, exist_ok=True)
    label_output_root = None
    if moving_labels is not None:
        if label_output_dir is None:
            raise ValueError("Please provide label output")

        label_output_dir = _add_suffix(label_output_dir, ".zarr")

        label_output_root = open_ome_zarr(label_output_dir, mode="a")
        label_output_root.require_group("labels", overwrite=False)
    image_output_root = None
    if moving_output_dir is not None:
        moving_output_dir = _add_suffix(moving_output_dir, ".zarr")

        image_output_root = open_ome_zarr(moving_output_dir, mode="a")
    if image_output_root is not None:
        image_output_root.require_group("images", overwrite=False)
    moving_image_gen = _set_up_experiment(
        moving_image_path,
        moving_image_pattern,
        group_by,
        subset=subset,
        file_sort_order=sort_order,
    )

    if fixed_image_path is not None:  # aligned moving image to self at specified time
        fixed_image_gen = _set_up_experiment(
            fixed_image_path,
            fixed_image_pattern,
            group_by,
            subset=subset,
            file_sort_order=sort_order,
        )
        image_bag = from_sequence(zip(fixed_image_gen, moving_image_gen))

    else:
        image_bag = from_sequence(zip_longest([None], moving_image_gen))
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        bag = image_bag.starmap(
            single_registration,
            unroll_channels=unroll_channels,
            z_index=z_index,
            initial_transform=initial_transform,
            transform_output_dir=transform_output_dir,
            transform_fs=transform_fs,
            moving_channel=moving_channel,
            fixed_channel=fixed_channel,
            itk_parameters=itk_parameters,
            moving_labels=moving_labels,
            label_output_root=label_output_root,
            image_output_root=image_output_root,
            moving_image_spacing=moving_image_spacing,
            fixed_image_spacing=fixed_image_spacing,
            reference_timepoint=reference_timepoint,
            landmarks_initialize=landmarks_initialize,
            landmark_slice_size=landmark_slice_size,
            landmark_min_count=landmark_min_count,
            landmark_template_padding=landmark_template_padding,
            landmark_step_size=landmark_step_size,
            landmark_min_score=landmark_min_score,
            landmark_initializations=landmark_initializations,
            landmark_com_max_quantile=landmark_com_max_quantile,
            landmark_com_min_quantile=landmark_com_min_quantile,
            output_aligned_channels_only=output_aligned_channels_only,
            force=force,
            no_version=no_version,
            itk_channel_parameters=itk_channel_parameters,
        )
        bag.compute()


def single_cross_correlation(
    _,
    file_list: list[str],
    metadata: dict,
    across_t_channel: None | int,
    within_t_channel: None | list[int],
    filter_percentiles: tuple[float, float],
    output_dir: zarr.Group,
    force: bool = False,
) -> None:
    """Perform cross-correlation-based registration for a single image.

    This function aligns a single image across or within timepoints using cross-correlation. It
    allows filtering based on specified percentiles to refine the alignment and saves the aligned
    image to the provided Zarr output directory.

    :param _: Unused placeholder parameter required by the Dask starmap call.
    :param file_list: List of file paths for the images to process.
    :param metadata: Metadata dictionary containing information about the image.
    :param across_t_channel: Channel index used for alignment across timepoints. If None, no across-
        time alignment is performed.
    :param within_t_channel: List of channel indices to align within the same timepoint. If None, no
        within-time alignment is performed.
    :param filter_percentiles: Tuple containing the lower and upper percentiles for filtering pixel
        intensities (e.g., (0.1, 0.9)).
    :param output_dir: Zarr group where the aligned images will be saved.
    :param force: If True, forces the re-alignment even if the output already exists.
    """
    image_key = metadata["id"]
    if not force and output_dir.get(f"images/{image_key}") is not None:
        return logger.info(f"Skipping cross correlation for {image_key}")
    logger.info(f"Running cross correlation for {image_key}")
    image = _images2fov(file_list, metadata)
    image = align_image(
        image,
        align_within_time_channels=within_t_channel,
        align_between_time_channel=across_t_channel,
        filter_percentiles=filter_percentiles,
    )
    _write_zarr_image(name=image_key, root=output_dir, image=image)


def run_cross_correlation_registration(arguments: argparse.Namespace) -> None:
    """Run image registration using cross-correlation.

    This function orchestrates the image registration process using cross-correlation, aligning a
    collection of images across and within cycles. It considers parameters such as filter
    percentiles, within-time channels, and output directories for storing the registered images.

    :param arguments: An argparse.Namespace object containing command-line arguments.
    """
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url == "none" and dask_cluster_parameters:
        # If a JSON is passed to dask_cluster avoid null_context
        dask_server_url = None

    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = dict(threads_per_worker=1, n_workers=_cpu_count())

    filter_percentiles = (
        arguments.registration_filter_min,
        arguments.registration_filter_max,
    )
    within_t_channel = arguments.within_t_channel
    images = arguments.images
    image_pattern = arguments.image_pattern
    across_t_channel = arguments.across_t_channel
    group_by = arguments.groupby
    subset = arguments.subset
    force = arguments.force
    output_dir = arguments.output
    output_dir = _add_suffix(output_dir, ".zarr")

    output_dir = open_ome_zarr(output_dir, mode="a")

    moving_image_gen = _set_up_experiment(
        images, image_pattern, group_by, subset=subset
    )

    image_bag = from_sequence(moving_image_gen)
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        bag = image_bag.starmap(
            single_cross_correlation,
            output_dir=output_dir,
            filter_percentiles=filter_percentiles,
            within_t_channel=within_t_channel,
            across_t_channel=across_t_channel,
            force=force,
        )
        bag.compute()
