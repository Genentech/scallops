"""The `register` module provides functionality for image registration using ITK and performing
cross-correlation-based registration across different cycles."""

import argparse
import os
import re
from collections.abc import Iterable, Sequence
from itertools import zip_longest
from typing import Literal, cast

import fsspec
import itk
import numpy as np
import xarray as xr
import zarr
from array_api_compat import get_namespace
from zarr import Group

from scallops.cli.util import (
    ImageTuple,
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
from scallops.registration.itk import (
    _combine_channels,
    _get_chunk_size,
    _itk_align_reference_time_zarr,
    _itk_transform_image_zarr,
    _load_itk_parameters,
    _load_itk_parameters_from_dir,
    _padded_to_local_channel,
    itk_align,
    itk_transform_labels,
    set_automatic_transform_initialization,
)
from scallops.registration.landmarks import _get_translation, find_landmarks
from scallops.utils import _cpu_count
from scallops.xr import _z_projection
from scallops.zarr_io import (
    _get_fs,
    _get_store_path,
    _require_group,
    _write_zarr_labels,
    is_ome_zarr_array,
    open_ome_zarr,
    read_ome_zarr_array,
)

logger = _get_cli_logger()


def _flip(image: xr.DataArray, flip_y: bool, flip_x: bool) -> xr.DataArray:
    """Flip the y and/or x axes of an image with any number of dimensions.

    The axes are found by the ``y`` and ``x`` dimension names, defaulting to the last two
    axes.
    """
    if not (flip_x or flip_y):
        return image

    def _axis(name: str, default: int) -> int:
        return image.dims.index(name) if name in image.dims else image.ndim + default

    axes = []
    if flip_y:
        axes.append(_axis("y", -2))
    if flip_x:
        axes.append(_axis("x", -1))
    xp = get_namespace(image.data)
    return image.copy(data=xp.flip(image.data, axis=tuple(axes)))


def single_registration(
    fixed_tuple: ImageTuple | None,
    moving_tuple: ImageTuple,
    moving_channel: int | Sequence[int],
    fixed_channel: int | Sequence[int] | None,
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
    z_index: int | Literal["max"] = "max",
    initial_transform: str | None = None,
    landmarks_initialize: bool = False,
    landmark_slice_size: float = 200,
    landmark_template_padding: Sequence[float] = (500,),
    landmark_step_size: float = 1000,
    landmark_min_score: float = 0.6,
    landmark_initializations: Sequence[Literal["com", "none"]] = ("com", "none"),
    landmark_com_min_quantile: float | None = None,
    landmark_com_max_quantile: float | None = None,
    landmark_min_count: int = 100,
    output_aligned_channels_only: bool = False,
    itk_channel_parameters: list[str] | None = None,
    flip_moving_y: bool = False,
    flip_moving_x: bool = False,
    no_version: bool = False,
    channel_combine: Literal["median", "mean"] = "median",
    channel_weights: Sequence[float] | None = None,
    equalize_channels: bool = False,
) -> str:
    """Perform single registration between fixed and moving images.

    This function executes a registration process between a fixed and a moving image. It supports
    registration of images with different channels, and the result is saved to specified output
    roots.

    :param fixed_tuple: Tuple containing fixed image information.
    :param moving_tuple: Tuple containing moving image information.
    :param moving_channel: Index (or indices) of the moving image channel(s). Several channels
        are combined with ``channel_combine``.
    :param fixed_channel: Index (or indices) of the fixed image channel(s).
    :param channel_combine: How several moving (or fixed) channels are combined into the image
        used for registration: the pixelwise ``median`` or ``mean``.
    :param channel_weights: Weight of each moving channel for the ``mean`` (default: equal).
    :param equalize_channels: Scale every channel to a common range before combining.
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
    :param flip_moving_y: Flip moving y axis
    :param flip_moving_x: Flip moving x axis
    :param no_version: Whether to skip version/CLI information in output.
    :return: Key of the registered image.
    """

    _, moving_file_list, moving_metadata = moving_tuple
    image_key = moving_metadata["id"]
    register_self = fixed_tuple is None
    transform_dest = f"{transform_output_dir}{transform_fs.sep}{image_key}"
    moving_label_keys = []

    if moving_labels is not None:
        matching_label_prefix = image_key
        for moving_label in moving_labels:
            moving_label_keys.extend(
                get_matching_names(
                    image_key=matching_label_prefix, image_dir=moving_label, labels=True
                )
            )

        if len(moving_label_keys) == 0:
            logger.warning(f"No labels found for {image_key}")

    moving_image = None
    times: list[str] = []
    reference_time: str | None = None
    labels_by_time: dict[str, tuple[list[str], list[str]]] = {}
    if register_self:
        # times are needed to decide which labels to transform and whether to skip
        moving_image = _images2fov(
            moving_file_list,
            moving_metadata,
            dask=True,
            concat_dims=("c",),
        )
        times = _image_times(moving_image)
        if len(times) > 0:
            reference_timepoint = _reference_time_index(times, reference_timepoint)
            reference_time = times[reference_timepoint]
            if len(moving_label_keys) > 0:
                labels_by_time = _labels_to_transform_by_time(
                    moving_label_keys, image_key, times, reference_time
                )

    if not force:
        labels_exist = True
        if label_output_root is not None:
            if register_self:
                expected_label_names = [
                    name for _, names in labels_by_time.values() for name in names
                ]
            else:
                expected_label_names = [os.path.basename(k) for k in moving_label_keys]
            for key in expected_label_names:
                if not is_ome_zarr_array(label_output_root.get(f"labels/{key}")):
                    labels_exist = False
                    break
        transforms_exist = True
        if register_self:
            # a transform directory for every non-reference time
            transforms_exist = len(times) > 0 and all(
                transform_fs.isdir(f"{transform_dest}{transform_fs.sep}t={t}")
                for t in times
                if t != reference_time
            )

        image_exists = True
        if image_output_root is not None:
            if not is_ome_zarr_array(image_output_root.get(f"images/{image_key}")):
                image_exists = False
        elif label_output_root is None:
            image_exists = False
        if labels_exist and image_exists and transforms_exist:
            logger.info(f"Skipping registration for {image_key}")
            return image_key

    if register_self:
        logger.info(f"Running registration for {image_key} t={reference_time}")
        logger.info(
            f"{len(moving_file_list):,} {pluralize('input', len(moving_file_list))}:"
            f" {', '.join([s.name.replace('/images/', '') if isinstance(s, zarr.Group) else str(s) for s in moving_file_list])}"
        )
    else:
        logger.info(f"Running registration for {image_key}")

    if not register_self:  # register fixed t=0 to moving
        assert fixed_tuple is not None
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

    if moving_image is None:
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
        moving_image = cast(xr.DataArray, moving_image)
        fixed_image = cast(xr.DataArray, fixed_image)
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
        if output_aligned_channels_only:
            # write the raw selected channels, not the combined registration image
            moving_image_out = _z_projection(moving_image, z_index).isel(
                t=0,
                c=np.atleast_1d(moving_channel).tolist(),
                missing_dims="ignore",
            )
            if "c" not in moving_image_out.dims:
                moving_image_out = moving_image_out.expand_dims({"c": 1}, axis=0)
        else:
            moving_image_out = moving_image
        moving_image_align = _combine_channels(
            moving_image_align, channel_combine, channel_weights, equalize_channels
        )
        fixed_image = _combine_channels(
            fixed_image,
            channel_combine,
            _fixed_channel_weights(fixed_image, channel_weights),
            equalize_channels,
        )
        if fixed_image.dtype != moving_image_align.dtype and (
            np.issubdtype(fixed_image.dtype, np.floating)
            or np.issubdtype(moving_image_align.dtype, np.floating)
        ):
            # elastix needs both images of the same pixel type and computes in float32
            fixed_image = fixed_image.astype(np.float32)
            moving_image_align = moving_image_align.astype(np.float32)
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
        moving_image_align = moving_image_align.compute()
        moving_image_align = _flip(moving_image_align, flip_moving_y, flip_moving_x)

        if landmarks_initialize:
            template_labels = None
            if len(moving_label_keys) > 0:
                nuclei_key = None
                cell_key = None

                for key in reversed(moving_label_keys):
                    if nuclei_key is None and key.endswith("-nuclei"):
                        nuclei_key = key

                    if cell_key is None and key.endswith("-cell"):
                        cell_key = key

                template_label_key = nuclei_key if nuclei_key is not None else cell_key
                if template_label_key is not None:
                    template_labels = read_ome_zarr_array(
                        template_label_key, dask=True
                    ).compute()
                    template_labels = _flip(
                        template_labels, flip_moving_y, flip_moving_x
                    )

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
        del moving_image_align
        if image_output_root is not None:  # save moving image
            _itk_transform_image_zarr(
                image=_flip(moving_image_out, flip_moving_y, flip_moving_x),
                transform_parameter_object=elastix_object.GetTransformParameterObject(),
                image_attrs=image_attrs,
                image_spacing=moving_image_spacing,
                image_name=image_key,
                image_root=image_output_root,
                chunksize=chunksize,
            )
        del moving_image_out
        if moving_labels is not None:
            _transform_labels(
                transform_parameter_object=elastix_object.GetTransformParameterObject(),
                attrs=moving_image.attrs,
                matching_keys=moving_label_keys,
                output_names=None,
                moving_image_spacing=moving_image_spacing,
                output_root=label_output_root,
                flip_y=flip_moving_y,
                flip_x=flip_moving_x,
                chunksize=chunksize,
            )

    else:  # align to t=reference_timepoint
        if len(times) == 0:
            raise ValueError(
                f"No t coordinate found for {image_key}; registering across times "
                "requires a {t} field in the moving image pattern."
            )
        assert isinstance(reference_timepoint, int)

        set_automatic_transform_initialization(parameter_object, False)
        if output_aligned_channels_only and isinstance(moving_image, xr.DataArray):
            moving_image = moving_image.isel(c=np.atleast_1d(moving_channel).tolist())
            n_moving = len(np.atleast_1d(moving_channel))
            moving_channel = 0 if n_moving == 1 else list(range(n_moving))
        elif output_aligned_channels_only:
            new_moving_image = []
            max_nchannels = max(img.sizes["c"] for img in moving_image)
            for img in moving_image:
                # without unrolling, moving_channel indexes the layout padded with
                # leading empty channels
                c = _padded_to_local_channel(
                    moving_channel,
                    offset=0 if unroll_channels else max_nchannels - img.sizes["c"],
                    nchannels=img.sizes["c"],
                    t=img.coords["t"].values.ravel()[0],
                )
                new_moving_image.append(img.isel(c=np.atleast_1d(c).tolist()))
            n_moving = len(np.atleast_1d(moving_channel))
            moving_channel = 0 if n_moving == 1 else list(range(n_moving))
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
            channel_combine=channel_combine,
            channel_weights=channel_weights,
            equalize_channels=equalize_channels,
        )

        moving_image_attrs = moving_image[0].attrs.copy()
        chunk_size = _get_chunk_size(moving_image[reference_timepoint])
        del moving_image
        if len(labels_by_time) > 0:
            _transform_labels_t(
                transform_fs=transform_fs,
                transform_dest=transform_dest,
                labels_by_time=labels_by_time,
                moving_image_attrs=moving_image_attrs,
                moving_image_spacing=moving_image_spacing,
                label_output_root=label_output_root,
                chunksize=chunk_size,
            )

    return image_key


def _transform_dir_times(transform_fs, transform_dest: str) -> list[str]:
    """Return the times that have a ``t=`` transform directory in ``transform_dest``."""
    if not transform_fs.isdir(transform_dest):
        return []
    times = []
    for transform_file in transform_fs.ls(transform_dest, detail=True, refresh=True):
        if transform_file["type"] == "directory":
            basename = os.path.basename(transform_file["name"].rstrip("/"))
            if basename.startswith("t="):
                times.append(basename[2:])
    return times


def _image_times(moving_image: xr.DataArray | Sequence[xr.DataArray]) -> list[str]:
    """Return the time values of an image as strings (as used in ``t=`` directories).

    Returns an empty list when the image has no ``t`` coordinate.
    """
    if isinstance(moving_image, xr.DataArray):
        if "t" not in moving_image.dims and "t" not in moving_image.coords:
            return []
        # a t dim without coordinate values gives the index
        return [str(t) for t in np.atleast_1d(moving_image["t"].values)]
    times = []
    for img in moving_image:
        if "t" not in img.dims and "t" not in img.coords:
            return []
        t = img["t"].values
        if t.ndim > 0:
            t = t[0]
        times.append(str(t))
    return times


def _reference_time_index(times: Sequence[str], reference_timepoint: int | str) -> int:
    """Return the index of the reference timepoint (given as an index or a time value)."""
    if isinstance(reference_timepoint, str):
        if reference_timepoint not in times:
            raise ValueError(f"Reference timepoint not found: {reference_timepoint}.")
        return list(times).index(reference_timepoint)
    if not 0 <= reference_timepoint < len(times):
        raise ValueError(f"Reference timepoint not found: {reference_timepoint}.")
    return reference_timepoint


def _moving_labels_by_time(
    moving_label_keys: Sequence[str], image_key: str, times: Sequence[str]
) -> tuple[dict[str, tuple[list[str], list[str]]], list[str]]:
    """Group moving labels by the time they were acquired at.

    Labels are assumed to be named ``{image_key}-{t}-{suffix}``. A label belongs to
    time ``t`` if its name starts with ``{image_key}-{t}-``; when several times match
    (e.g. ``IF`` and ``IF-1``), the longest one wins.

    :param moving_label_keys: Paths to moving labels.
    :param image_key: Image key the labels belong to.
    :param times: All times of the image, including the reference time. Integer
        times also match labels whose time has leading zeros.
    :return: Dict mapping time to (label paths, output names), and the paths of
        labels that match no time.
    """
    result: dict[str, tuple[list[str], list[str]]] = {}
    unmatched = []
    times_longest_first = sorted(times, key=len, reverse=True)
    prefix = f"{image_key}-"
    for moving_label_key in moving_label_keys:
        basename = os.path.basename(moving_label_key.rstrip("/"))
        rest = basename[len(prefix) :] if basename.startswith(prefix) else None
        matched_time = None
        if rest is not None:
            for time in times_longest_first:
                if rest.startswith(f"{time}-"):
                    matched_time = time
                    break
            if matched_time is None:
                # integer times lose leading zeros when read (01 -> 1)
                digits = re.match(r"(\d+)-", rest)
                if digits is not None:
                    for time in times_longest_first:
                        if time.isdecimal() and int(time) == int(digits[1]):
                            matched_time = time
                            break
        if matched_time is None:
            unmatched.append(moving_label_key)
        else:
            paths, names = result.setdefault(matched_time, ([], []))
            paths.append(moving_label_key)
            names.append(basename)
    return result, unmatched


def _labels_to_transform_by_time(
    moving_label_keys: Sequence[str],
    image_key: str,
    times: Sequence[str],
    reference_time: str,
) -> dict[str, tuple[list[str], list[str]]]:
    """Return the labels to transform to the reference time, grouped by time.

    Logs a warning for labels that are not transformed: labels at the reference time
    and labels whose name doesn't start with ``{image_key}-{t}-`` for any time.
    """
    labels_by_time, unmatched = _moving_labels_by_time(
        moving_label_keys, image_key, times
    )
    reference_labels = labels_by_time.pop(reference_time, ([], []))[1]
    if len(reference_labels) > 0:
        logger.warning(
            f"Not transforming {', '.join(reference_labels)}: labels are at the "
            f"reference time {reference_time}."
        )
    if len(unmatched) > 0:
        logger.warning(
            f"Not transforming {', '.join(os.path.basename(k) for k in unmatched)}: "
            f"label names must start with {image_key}-{{t}}- for one of the times "
            f"{', '.join(times)}."
        )
    return labels_by_time


def _transform_labels_t(
    transform_fs,
    transform_dest: str,
    labels_by_time: dict[str, tuple[list[str], list[str]]],
    moving_image_attrs,
    moving_image_spacing,
    label_output_root,
    chunksize: tuple[int, int] | None = (1024, 1024),
):
    # transform_dest structure is image_key/t=1
    # dest labels keep the same name so that A1-IF1-nuclei and A1-IF2-nuclei can
    # both be transformed to the reference t
    for time, (moving_label_paths, output_names) in labels_by_time.items():
        transform_dir = f"{transform_dest}{transform_fs.sep}t={time}"
        transform_parameter_object = None
        if transform_fs.isdir(transform_dir):
            transform_parameter_object = _load_itk_parameters_from_dir(
                transform_fs.unstrip_protocol(transform_dir)
            )
        if (
            transform_parameter_object is None
            or transform_parameter_object.GetNumberOfParameterMaps() == 0
        ):
            logger.warning(
                f"Not transforming {', '.join(output_names)}: no transform found for "
                f"t={time}."
            )
            continue
        _transform_labels(
            transform_parameter_object=transform_parameter_object,
            attrs=moving_image_attrs,
            matching_keys=moving_label_paths,
            output_names=output_names,
            moving_image_spacing=moving_image_spacing,
            output_root=label_output_root,
            chunksize=chunksize,
        )


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
        image_dir = f"{_get_store_path(image_dir)}{image_dir.name}"
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
        if not name.startswith(".") and is_ome_zarr_array(zarr.open(path, mode="r")):
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
            image_attrs=dict(),
            image_spacing=moving_image_spacing,
            image_name=image_key,
            image_root=output_root,
            chunksize=chunksize,
        )


def _transform_labels(
    transform_parameter_object: itk.ParameterObject,
    matching_keys: Sequence[str],
    output_names: Sequence[str] | None,
    output_root: zarr.Group,
    moving_image_spacing: None | tuple[float, float],
    attrs: None | dict,
    flip_y: bool = False,
    flip_x: bool = False,
    chunksize: tuple[int, int] | None = (1024, 1024),
):
    """Transform and save labels.

    This function applies a specified transformation and saves the results.
    It is designed to work with ITK transformations and Zarr storage.

    :param transform_parameter_object: ITK parameter object.
    :param matching_keys: Paths to zarr labels.
    :param output_names: Names of labels.
    :param output_root: Root for output storage.
    :param moving_image_spacing: Spacing of the moving image.
    :param attrs: Additional attributes for the transformed array.
    :param flip_y: Flip y axis.
    :param flip_x: Flip x axis.
    """
    if output_names is None:
        output_names = [os.path.basename(key) for key in matching_keys]
    # if keys clash, keep the last one
    unique_outputs = set()
    for i in range(len(matching_keys) - 1, -1, -1):
        key = matching_keys[i]

        if output_names[i] in unique_outputs:
            logger.info(f"Skipping transforming {matching_keys[i]} labels.")
            continue
        unique_outputs.add(output_names[i])
        array = read_ome_zarr_array(key)

        if attrs is not None:
            array.attrs = attrs  # e.g. copy physical size

        logger.info(f"Transforming {matching_keys[i]} labels.")
        transformed_array = itk_transform_labels(
            image=_flip(array, flip_y, flip_x),
            transform_parameter_object=transform_parameter_object,
            image_spacing=moving_image_spacing,
        )
        del array
        # match chunk size of fixed image
        _write_zarr_labels(
            name=output_names[i],
            root=output_root,
            labels=transformed_array,
            storage_options=dict(chunks=chunksize) if chunksize is not None else None,
        )


def single_transformix(
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
    # TODO see if transform dir has subdirectory for t
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
        _transform_labels(
            transform_parameter_object=_load_itk_parameters_from_dir(transform_dir),
            matching_keys=matching_keys,
            output_names=None,
            output_root=output_root,
            moving_image_spacing=image_spacing,
            attrs=None,
        )
    else:
        # see if transform dir has subdirectory describing channel transformation
        transform_fs, _ = fsspec.url_to_fs(transform_dir)
        for d in transform_fs.ls(transform_dir, detail=True, refresh=True):
            if d["type"] == "directory" and os.path.basename(
                d["name"].rstrip("/")
            ).startswith("c="):
                raise ValueError(
                    f"Channel transformations not supported for {image_key}"
                )
        transform_all_images(
            image_key=image_key,
            transform_parameter_object=_load_itk_parameters_from_dir(transform_dir),
            matching_keys=matching_keys,
            output_root=output_root,
            moving_image_spacing=image_spacing,
            attrs=None,
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
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        for transform_dir in transform_dirs:
            single_transformix(
                transform_dir,
                image_dir=images,
                image_spacing=image_spacing,
                transform_type=transform_type,
                output_root=output_root,
                force=force,
            )


def _fixed_channel_weights(
    fixed_image: xr.DataArray, channel_weights: Sequence[float] | None
) -> Sequence[float] | None:
    """Weights to combine the fixed channels: the moving weights, used channel for channel.

    :raises ValueError: If weights are given and the fixed image has several channels but not
        one per weight.
    """
    if channel_weights is None or "c" not in fixed_image.dims:
        return None
    n_fixed = fixed_image.sizes["c"]
    if n_fixed == 1:
        return None
    if n_fixed != len(channel_weights):
        raise ValueError(
            f"--channel-weights has {len(channel_weights)} values but {n_fixed} fixed "
            "channels are selected; select one fixed channel or as many as moving channels."
        )
    return channel_weights


def _channel_arg(channel: int | Sequence[int]) -> int | list[int]:
    """Single channel index as an int, several as a list (``--moving-channel 1 2 3 4``)."""
    if isinstance(channel, int):
        return channel
    return channel[0] if len(channel) == 1 else list(channel)


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

    moving_sort_order = arguments.sort
    if moving_sort_order is not None:
        moving_sort_order = dict(t=moving_sort_order)
    # fixed_sort_order = arguments.fixed_sort
    # if fixed_sort_order is not None:
    #     fixed_sort_order = dict(t=fixed_sort_order)
    moving_image_path = arguments.moving
    fixed_image_path = arguments.fixed
    flip_moving_y = arguments.flip_moving_y
    flip_moving_x = arguments.flip_moving_x

    if fixed_image_path is None and (flip_moving_y or flip_moving_x):
        raise ValueError(
            "Flipping moving image only supported when fixed image is provided."
        )
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
    moving_channel = _channel_arg(arguments.moving_channel)
    label_output_dir = arguments.label_output_dir
    moving_output_dir = arguments.moving_output_dir
    fixed_channel = _channel_arg(arguments.fixed_channel)
    channel_combine = arguments.combine_channels
    channel_weights = arguments.channel_weights
    equalize_channels = arguments.equalize_channels
    if channel_weights is not None:
        if channel_combine != "mean":
            raise ValueError("--channel-weights requires --combine-channels mean.")
        n_moving = len(np.atleast_1d(moving_channel))
        if len(channel_weights) != n_moving:
            raise ValueError(
                f"--channel-weights has {len(channel_weights)} values but "
                f"{n_moving} --moving-channel {'index' if n_moving == 1 else 'indices'} given."
            )
        n_fixed = len(np.atleast_1d(fixed_channel))
        if fixed_image_path is not None and n_fixed not in (1, n_moving):
            raise ValueError(
                f"--channel-weights is used for the fixed channels too, so give one "
                f"--fixed-channel or {n_moving} (got {n_fixed})."
            )
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
        _require_group(label_output_root, "labels")
    image_output_root = None
    if moving_output_dir is not None:
        moving_output_dir = _add_suffix(moving_output_dir, ".zarr")

        image_output_root = open_ome_zarr(moving_output_dir, mode="a")
    if image_output_root is not None:
        _require_group(image_output_root, "images")
    moving_image_gen = _set_up_experiment(
        moving_image_path,
        moving_image_pattern,
        group_by,
        subset=subset,
        file_sort_order=moving_sort_order,
    )

    image_pairs: Iterable[tuple[ImageTuple | None, ImageTuple]]
    if fixed_image_path is not None:  # aligned moving image to self at specified time
        fixed_image_gen = _set_up_experiment(
            fixed_image_path,
            fixed_image_pattern,
            group_by,
            subset=subset,
        )
        image_pairs = zip(fixed_image_gen, moving_image_gen)

    else:
        image_pairs = zip_longest([None], moving_image_gen)
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        for fixed_image_tuple, moving_image_tuple in image_pairs:
            single_registration(
                fixed_image_tuple,
                moving_image_tuple,
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
                flip_moving_y=flip_moving_y,
                flip_moving_x=flip_moving_x,
                itk_channel_parameters=itk_channel_parameters,
                channel_combine=channel_combine,
                channel_weights=channel_weights,
                equalize_channels=equalize_channels,
            )
