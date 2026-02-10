"""SCALLOPS Registration ITK Submodule.

Provides image registration functionality using the ITK library.


Authors:
    - The SCALLOPS development team
"""

import itertools
import json
import logging
import os
import re
import shutil
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal

import dask
import dask.array as da
import fsspec
import itk
import numpy as np
import xarray as xr
import zarr
from natsort import natsorted

from scallops.io import _download_file, _get_fs_protocol, get_image_spacing
from scallops.registration.landmarks import _get_translation, find_landmarks
from scallops.utils import _dask_from_array_no_copy
from scallops.xr import _get_dims
from scallops.zarr_io import (
    _zarr_v3,
    default_zarr_format,
    get_zarr_array_kwargs,
    open_ome_zarr,
    write_zarr,
)

logger = logging.getLogger("scallops")


def _load_parameter_object(paths: list[str], tmp_paths: list[str]):
    """Load an ITK ParameterObject from parameter files and remove temporary paths.

    This function creates a new ITK ParameterObject, adds parameter files from
    the provided paths, and ensures the number of parameter maps matches the input paths.
    It also deletes any temporary paths provided.

    :param paths: A list of file paths to the parameter files to be added.
    :param tmp_paths: A list of temporary file paths to be removed.
    :return: An ITK ParameterObject with the loaded parameter maps.

    :raises AssertionError: If the number of parameter maps does not match the length of `paths`.

    :example:

    .. code-block:: python

        parameter_paths = ["params1.txt", "params2.txt"]
        tmp_files = ["tmp1.txt"]
        parameter_obj = _load_parameter_object(parameter_paths, tmp_files)
    """
    parameter_object = itk.ParameterObject.New()
    for path in paths:
        parameter_object.AddParameterFile(path)

    assert parameter_object.GetNumberOfParameterMaps() == len(paths), (
        f"{parameter_object.GetNumberOfParameterMaps()} != {len(paths)}"
    )

    for tmp_path in tmp_paths:
        os.remove(tmp_path)

    return parameter_object


def _get_chunk_size(image: xr.DataArray) -> tuple[int, int]:
    """Determine the optimal chunk size for a given xarray DataArray.

    This function checks the chunk sizes of a Dask-backed DataArray and adjusts
    them if the chunk size differs significantly from the image dimensions along
    the 'y' and 'x' axes. If the chunk size along an axis is less than 99% of the
    corresponding image size, it updates the chunk size for that axis.

    :param image: An xarray DataArray representing the image.
    :return: A tuple containing the optimized chunk size for the 'y' and 'x' axes.

    :example:

    .. code-block:: python

        image = xr.DataArray(da.ones((2000, 2000), chunks=(500, 500)), dims=("y", "x"))
        chunk_size = _get_chunk_size(image)
        print(chunk_size)  # Output: (500, 500)
    """
    chunk_size = [1360, 1360]  # Default chunk size

    if isinstance(image.data, da.Array):
        if (
            image.data.chunksize[-2] != image.sizes["y"]
            and (image.data.chunksize[-2] / image.sizes["y"]) < 0.99
        ):
            chunk_size[0] = image.data.chunksize[-2]
        if (
            image.data.chunksize[-1] != image.sizes["x"]
            and (image.data.chunksize[-1] / image.sizes["x"]) < 0.99
        ):
            chunk_size[1] = image.data.chunksize[-1]

    return tuple(chunk_size)


def _load_itk_parameters(itk_parameters: list[str] | None) -> itk.ParameterObject:
    """Load ITK parameter maps from provided paths or predefined parameter maps.

    This function creates an ITK ParameterObject by loading parameter maps
    from local files, remote URLs, or predefined maps. If the file is remote
    and not directly accessible, it downloads the file temporarily and ensures
    it is removed after loading.

    :param itk_parameters: List of paths to parameter files or predefined map keys.
    :return: An ITK ParameterObject containing all the loaded parameter maps.

    :raises ValueError: If a path is not found and is not a predefined parameter.

    :example:

    .. code-block:: python

        itk_params = [
            "default_map",
            "/local/path/params.json",
            "s3://bucket/params.txt",
        ]
        parameter_object = _load_itk_parameters(itk_params)
    """
    parameter_object = itk.ParameterObject.New()
    tmp_paths = []
    if itk_parameters is not None:
        for path in itk_parameters:
            fs, _ = fsspec.core.url_to_fs(path)

            if not fs.isfile(path):
                m = DEFAULT_REG_PARAM_MAPS.get(path)
                if m is None:
                    raise ValueError(
                        f"{path} not found and is not a predefined parameter"
                    )
                parameter_object.AddParameterMap(m)
            else:
                if path.lower().endswith(".json"):
                    with fs.open(path, "rt") as f:
                        m = json.load(f)
                        if isinstance(m, list):
                            for pm in m:
                                parameter_object.AddParameterMap(pm)
                        else:
                            parameter_object.AddParameterMap(m)
                else:
                    if _get_fs_protocol(fs) != "file":
                        path = _download_file(fs, path)
                        tmp_paths.append(path)
                    parameter_object.AddParameterFile(path)

    for tmp_path in tmp_paths:
        os.remove(tmp_path)

    return parameter_object


def _get_itk_transform_paths_from_dir(directory: str) -> tuple[list[str], list[str]]:
    """Retrieve paths to ITK transform parameter files from a specified directory.

    This function searches the provided directory for transform parameter files
    matching the pattern `TransformParameters.[0-9]+.txt`. If the directory is remote,
    it downloads the files temporarily and ensures their paths are returned for further use.

    :param directory: Directory to search for transform parameter files.
    :return: A tuple containing:
             - List of transform parameter file paths.
             - List of temporary file paths that were downloaded (if any).

    """
    paths = []
    fs, _ = fsspec.core.url_to_fs(directory)
    directory = directory.rstrip(fs.sep)
    tmp_paths = []

    transform_parameters_pattern = re.compile(
        "TransformParameters.[0-9]+.txt|InitialTransformParameters.0.txt"
    )

    matches = fs.glob(f"{directory}{fs.sep}*TransformParameters*.txt")

    if _get_fs_protocol(fs) != "file":
        matches = [f"{_get_fs_protocol(fs)}://{x}" for x in matches]

    for path in natsorted(matches):
        name = os.path.basename(path)

        if name[0] != "." and transform_parameters_pattern.match(
            name
        ):  # Ignore hidden files
            if _get_fs_protocol(fs) != "file":
                path = _download_file(fs, path)
                tmp_paths.append(path)
            paths.append(path)
    if len(paths) == 0:
        raise ValueError(f"No transform parameters found in {directory}.")
    return paths, tmp_paths


def _load_itk_parameters_from_dir(directory: str) -> itk.ParameterObject:
    """Load ITK parameter object from a directory containing transform parameter files.

    :param directory: Directory to search for transform parameter files.
    :return: An ITK ParameterObject containing all the loaded parameter maps.
    """
    paths, tmp_paths = _get_itk_transform_paths_from_dir(directory)
    return _load_parameter_object(paths, tmp_paths)


def _array_to_itk(
    image: xr.DataArray | np.ndarray, image_spacing: None | tuple[float, float] = None
) -> itk.Image:
    """Convert a NumPy or xarray image to an ITK image with specified spacing.

    :param image: Input image as an xarray DataArray or NumPy array.
    :param image_spacing: Tuple of physical spacing (y, x). If not provided, spacing is extracted
        from the image's attributes if image is a DataArray.
    :return: An ITK image object with the specified spacing.
    :raises ValueError: If physical size (spacing) is not found.
    """
    if isinstance(image, itk.Image):
        return image
    if image.dtype in (np.int32, np.uint32, np.int64, np.uint64):
        image = image.astype(np.float32)
    spacing = image_spacing
    if spacing is None:
        if isinstance(image, xr.DataArray):
            spacing = get_image_spacing(image.attrs)
        if spacing is None:
            raise ValueError("Physical size not found")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Handle non-contiguous array warning from ITK.
        data = (
            image.squeeze().isel(c=0, t=0, z=0, missing_dims="ignore").values
            if isinstance(image, xr.DataArray)
            else image.squeeze()
        )

        itk_image = itk.image_view_from_array(data)

    itk_image.SetSpacing(spacing)
    return itk_image


def set_automatic_transform_initialization(
    parameter_object: itk.ParameterObject, first_parameter_map: bool = True
) -> None:
    """Set the automatic transform initialization flag in the provided ITK parameter object.

    This function updates the given ITK parameter object to enable automatic transform
    initialization for the first parameter map only if mask is False

    :param parameter_object: The ITK parameter object representing the registration parameters.
    :param first_parameter_map: Whether to enable automatic transform initialization for first
        parameter map.
    """

    for i in range(parameter_object.GetNumberOfParameterMaps()):
        p = parameter_object.GetParameterMap(i)
        p["AutomaticTransformInitialization"] = [
            "true" if i == 0 and first_parameter_map else "false"
        ]


def _itk_align_reference_time_zarr(
    moving_image: xr.DataArray | list[xr.DataArray],
    moving_channel: int | list[int],
    parameter_object: itk.ParameterObject,
    image_root: zarr.Group,
    image_name: str,
    moving_image_spacing: None | tuple[float, float] = None,
    output_dir: str | None = None,
    unroll_channels: bool = False,
    reference_timepoint: int = 0,
    landmarks_initialize: bool = False,
    landmark_slice_size: float = 200,
    landmark_template_padding: Sequence[float] = (500,),
    landmark_step_size: float = 1000,
    landmark_min_score: float = 0.6,
    landmark_translations: Sequence[Literal["com", "none"]] = ("com", "none"),
    landmark_com_min_quantile: float | None = None,
    landmark_com_max_quantile: float | None = None,
    landmark_min_count: int = 100,
    parameter_object_across_channels: itk.ParameterObject | None = None,
):
    """Aligns a moving image to a reference timepoint and stores the result in Zarr format.

    This function performs image alignment using ITK and stores the aligned image in a Zarr dataset.
    The result is stored chunk by chunk to optimize memory usage.

    :param moving_image: A single xarray DataArray or a list of DataArrays representing the moving
        image(s).
    :param moving_channel: Channel(s) to align.
    :param parameter_object: ITK ParameterObject containing transformation parameters.
    :param image_root: Zarr group where the aligned image will be stored.
    :param image_name: Name of the image to store within the Zarr group.
    :param moving_image_spacing: Physical spacing (y, x) for the moving image, or None to infer from
        metadata.
    :param output_dir: Optional output directory for intermediate results.
    :param unroll_channels: If True, unrolls channels in the output.
    :param reference_timepoint: Timepoint index to use as the reference for alignment.
    :param parameter_object_across_channels: Align across channels.
    """

    def _init_callback(init_params: dict[str, Any]) -> dict[str, Any]:
        dims = init_params["dims"]
        coords = init_params["coords"]
        shape = init_params["shape"]
        attrs = init_params["attrs"]
        dtype = init_params["dtype"]
        chunk_size = init_params["chunk_size"]
        zarr_dataset = None
        group = None
        if image_root is not None:
            images_group = image_root.require_group("images", overwrite=False)
            fmt = default_zarr_format()
            group = images_group.create_group(
                image_name.replace("/", "-"), overwrite=True
            )

            zarr_dataset = (
                group.create_array(
                    "0",
                    shape=shape,
                    chunks=(1,) * (len(shape) - 2) + chunk_size,
                    dtype=dtype,
                    overwrite=True,
                    **get_zarr_array_kwargs(fmt),
                )
                if _zarr_v3()
                else group.create_dataset(
                    "0",
                    shape=shape,
                    chunks=(1,) * (len(shape) - 2) + chunk_size,
                    dtype=dtype,
                    overwrite=True,
                    **get_zarr_array_kwargs(fmt),
                )
            )

        return {
            "data": zarr_dataset,
            "group": group,
            "dims": dims,
            "coords": coords,
            "attrs": attrs,
        }

    def done(d: dict[str, Any]):
        """Finalize the Zarr dataset by writing metadata.

        :param d: Dictionary containing dataset and metadata information.
        """
        data = d["data"]
        group = d["group"]
        dims = d["dims"]
        coords = d["coords"]
        image_attrs = d["attrs"]
        if data is not None:
            write_zarr(
                grp=group,
                data=data,
                image_attrs=image_attrs,
                coords=coords,
                dims=dims,
                zarr_format="zarr",
            )

    def _write_callback(x, idx, val):
        if x is None:
            return
        if isinstance(idx, int):
            idx = (idx,)
        if isinstance(val, xr.DataArray):
            val = val.data
        if not isinstance(val, da.Array):
            val = _dask_from_array_no_copy(val, chunks=x.chunks[-2:])
        da.store(val, x, regions=idx, compute=True)

    _itk_align_reference_time(
        moving_image=moving_image,
        moving_channel=moving_channel,
        parameter_object=parameter_object,
        moving_image_spacing=moving_image_spacing,
        output_dir=output_dir,
        unroll_channels=unroll_channels,
        init_callback=_init_callback,
        done_callback=done,
        reference_timepoint=reference_timepoint,
        landmarks_initialize=landmarks_initialize,
        landmark_slice_size=landmark_slice_size,
        landmark_template_padding=landmark_template_padding,
        landmark_step_size=landmark_step_size,
        landmark_min_score=landmark_min_score,
        landmark_translations=landmark_translations,
        landmark_com_min_quantile=landmark_com_min_quantile,
        landmark_com_max_quantile=landmark_com_max_quantile,
        landmark_min_count=landmark_min_count,
        parameter_object_across_channels=parameter_object_across_channels,
        write_callback=_write_callback,
    )


def itk_align_to_reference_time(
    moving_image: xr.DataArray | list[xr.DataArray],
    moving_channel: int | list[int],
    parameter_object: itk.ParameterObject,
    moving_image_spacing: None | tuple[float, float] = None,
    output_dir: str | None = None,
    unroll_channels: bool = False,
    reference_timepoint: int = 0,
) -> xr.DataArray:
    """Align a time-series of moving images to the specified time point using ITK registration.

    This function performs time-dependent registration using the ITK library. It aligns each time
    point of a moving image stack to the specified time point

    :param moving_image: The time-series of moving images to be aligned, represented as a DataArray
        or a list of data arrays representing each timepoint.
    :param moving_channel: The channel index to consider during registration.
    :param parameter_object: The ITK parameter object containing registration parameters.
    :param moving_image_spacing: Optional spacing information for the moving images. If not
        provided, it is determined from the image metadata.
    :param output_dir: Optional directory to save intermediate outputs and ITK parameters per
        timepoint. If None, outputs are not saved.
    :param unroll_channels: Whether to concatenate time along the channel dimension.
    :param reference_timepoint: Index of the timepoint to align to
    :return: The aligned DataArray.
    """

    def _init_callback(init_params: dict[str, Any]):
        dims = init_params["dims"]
        coords = init_params["coords"]
        shape = init_params["shape"]
        attrs = init_params["attrs"]
        dtype = init_params["dtype"]
        result_image = xr.DataArray(
            np.zeros(
                shape=shape,
                dtype=dtype,
            ),
            coords=coords,
            dims=dims,
        )

        return {"data": result_image.data, "other": result_image, "attrs": attrs}

    def done(d):
        d["other"].attrs = d["attrs"]
        return d["other"]

    return _itk_align_reference_time(
        moving_image=moving_image,
        moving_channel=moving_channel,
        parameter_object=parameter_object,
        moving_image_spacing=moving_image_spacing,
        output_dir=output_dir,
        unroll_channels=unroll_channels,
        init_callback=_init_callback,
        done_callback=done,
        reference_timepoint=reference_timepoint,
    )


def _read_itk_point_set(path: str) -> tuple[list[float], list[float]]:
    # point set file format:
    # <index, point>
    # <number of points>
    # point1 x point1 y [point1 z]
    y = []
    x = []
    fs, _ = fsspec.core.url_to_fs(path)
    with fs.open(path, "rt") as f:
        header = f.readline().strip()
        assert header in ("point", "index"), f"{header} is not a valid header"
        npoints = int(f.readline().strip())
        for i in range(npoints):
            tokens = f.readline().strip().split(" ")
            x.append(float(tokens[0]))
            y.append(float(tokens[1]))
    return x, y


def _write_itk_point_set(
    path: str,
    y: Sequence[float],
    x: Sequence[float],
    point_set_type: Literal["point", "index"] = "point",
):
    # point set file format:
    # <index, point>
    # <number of points>
    # point1 x point1 y [point1 z]

    with open(path, "wt") as f:
        f.write(f"{point_set_type}\n")
        f.write(f"{len(y)}\n")
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]}\n")


def _read_itk_transformix_point_set_output(path: str) -> tuple[np.ndarray, np.ndarray]:
    # Point	0	; InputIndex = [ 1 1 ]	; InputPoint = [ 1.000000 1.000000 ]	; OutputIndexFixed = [ 26 51 ]	; OutputPoint = [ 25.712395 51.221179 ]	; Deformation = [ 24.712395 50.221180 ]	; OutputIndexMoving = [ 26 51 ]
    y = []
    x = []
    leading_offset = len("OutputPoint = [ ")
    with open(path, "rt") as f:
        tokens = f.readline().split(";")
        output_point = tokens[4].strip()  # 'OutputPoint = [ 25.712395 51.221179 ]'
        x_point, y_point = output_point[leading_offset:-2].split(" ")
        x.append(float(x_point))
        y.append(float(y_point))
    return np.asarray(y), np.asarray(x)


def _itk_align_reference_time(
    moving_image: xr.DataArray | list[xr.DataArray],
    moving_channel: int | list[int],
    parameter_object: itk.ParameterObject,
    init_callback: Callable[
        [
            dict[str, Any],
        ],
        dict[str, Any],
    ],
    done_callback: Callable[[dict[str, Any]], Any],
    moving_image_spacing: None | tuple[float, float] = None,
    output_dir: str | None = None,
    unroll_channels: bool = False,
    reference_timepoint: int = 0,
    landmarks_initialize: bool = False,
    landmark_slice_size: float = 200,
    landmark_template_padding: Sequence[float] = (500,),
    landmark_step_size: float = 1000,
    landmark_min_score: float = 0.6,
    landmark_translations: Sequence[Literal["com", "none"]] = ("com", "none"),
    landmark_com_min_quantile: float | None = None,
    landmark_com_max_quantile: float | None = None,
    landmark_min_count: int = 100,
    attrs_keep: Sequence[str] | None = ("stitch_coords",),
    parameter_object_across_channels: itk.ParameterObject | None = None,
    write_callback: Callable[
        [np.ndarray | zarr.Array, tuple[int, ...], np.ndarray], None
    ] = None,
):
    """Align a time-series of moving images to the reference time point using ITK registration.

    This function performs time-dependent registration using the ITK library. It aligns each time
    point of a moving image stack to the first time point, considering a specific channel.

    :param moving_image: The time-series of moving images to be aligned, represented as a DataArray
        or a list of data arrays representing each timepoint.
    :param moving_channel: The channel index to consider during registration.
    :param parameter_object: The ITK parameter object containing registration parameters.
    :param moving_image_spacing: Optional spacing information for the moving images. If not
        provided, it is determined from the metadata.
    :param output_dir: Optional directory to save intermediate outputs and ITK parameters per
        timepoint. If None, outputs are not saved.
    :param unroll_channels: Whether to concatenate time along the channel dimension.
    :param reference_timepoint: Reference timepoint to align to
    :param attrs_keep: Sequence of attributes to keep. Stored as list in result dataset attributes.
    :param parameter_object_across_channels: The ITK parameter object containing
    registration parameters for aligning across channels or None.
    :return: The aligned DataArray.
    """
    if write_callback is None:

        def _write_value(x, idx, val):
            x[idx] = val

        write_callback = _write_value

    if isinstance(moving_image, xr.DataArray):
        if "z" in moving_image.dims:
            moving_image = moving_image.squeeze("z", drop=True)
    else:
        moving_images = []
        for i in range(len(moving_image)):
            img = moving_image[i]
            if "z" in img.dims:
                img = img.squeeze("z", drop=True)
            moving_images.append(img)
        moving_image = moving_images

    moving_image_reference_t_moving_c = (
        moving_image.isel(t=reference_timepoint, c=moving_channel)
        if isinstance(moving_image, xr.DataArray)
        else moving_image[reference_timepoint].isel(c=moving_channel)
    )
    if (
        "c" in moving_image_reference_t_moving_c.dims
        and moving_image_reference_t_moving_c.sizes["c"] > 1
    ):
        moving_image_reference_t_moving_c = moving_image_reference_t_moving_c.median(
            dim="c", keep_attrs=True
        )
    if isinstance(moving_image, xr.DataArray):
        times = moving_image.coords["t"].values
    else:
        times = []
        for img in moving_image:
            t = img.coords["t"].values
            if t.ndim > 0:
                t = t[0]
            times.append(t)
    if len(times) < 2 and parameter_object_across_channels is None:
        logger.warning("Only one timepoint provided.")

    nchannels = (
        [moving_image.sizes["c"] for i in range(moving_image.sizes["t"])]
        if isinstance(moving_image, xr.DataArray)
        else [moving_image[i].sizes["c"] for i in range(len(moving_image))]
    )
    if not unroll_channels:
        if not all(x == nchannels[0] for x in nchannels):
            unroll_channels = True
            logger.info(
                "Unrolling channel dimension as time points have different number of "
                "channels."
            )

    channels = (
        [moving_image.coords["c"].values for i in range(moving_image.sizes["t"])]
        if isinstance(moving_image, xr.DataArray)
        else [moving_image[i].coords["c"].values for i in range(len(moving_image))]
    )

    if unroll_channels:
        logger.info(
            f"Channels per time point: {', '.join([str(c) for c in nchannels])}."
        )
        dims = ["c", "y", "x"]

        all_channels = []
        unique_channels = set()
        for i in range(len(times)):
            for c in channels[i]:
                base_channel_name = f"{times[i]}-{c}"
                channel_name = base_channel_name
                counter = 1
                while channel_name in unique_channels:
                    channel_name = f"{base_channel_name}-{counter}"
                    counter += 1
                unique_channels.add(channel_name)
                all_channels.append(channel_name)
        coords = dict(c=all_channels)
        leading_dims_shape = (len(all_channels),)
    else:  # all images must have the same number of channels
        dims = ["t", "c", "y", "x"]
        leading_dims_shape = len(times), nchannels[0]
        coords = dict(t=times, c=channels[0])

    result_image_shape = leading_dims_shape + (
        moving_image_reference_t_moving_c.sizes["y"],
        moving_image_reference_t_moving_c.sizes["x"],
    )

    init_dict = init_callback(
        dict(
            dims=dims,
            coords=coords,
            chunk_size=_get_chunk_size(moving_image_reference_t_moving_c),
            shape=result_image_shape,
            attrs=moving_image_reference_t_moving_c.attrs.copy(),
            dtype=moving_image_reference_t_moving_c.dtype,
        )
    )

    result_data = init_dict["data"]
    output_fs = fsspec.core.url_to_fs(output_dir)[0] if output_dir is not None else None
    unrolled_t_index = 0

    attr_name_to_values = defaultdict(list)
    if attrs_keep is None:
        attrs_keep = []
    moving_image_reference_t_moving_c = moving_image_reference_t_moving_c.squeeze()
    for i in range(len(times)):
        moving_image_ti_moving_c = (
            moving_image.isel(t=i, c=moving_channel)
            if isinstance(moving_image, xr.DataArray)
            else moving_image[i].isel(c=moving_channel)
        )

        for attr_name in attrs_keep:
            if attr_name in moving_image_ti_moving_c.attrs:
                attr_name_to_values[attr_name].append(
                    moving_image_ti_moving_c.attrs[attr_name]
                )
        transform_parameter_object = None
        if i != reference_timepoint:
            logger.info(f"Registering t={i} to t={reference_timepoint}.")

            if (
                "c" in moving_image_ti_moving_c.dims
                and moving_image_ti_moving_c.sizes["c"] > 1
            ):
                moving_image_ti_moving_c = moving_image_ti_moving_c.median(
                    dim="c", keep_attrs=True
                )
            moving_image_ti_moving_c = moving_image_ti_moving_c.squeeze()
            dest = None
            if output_dir is not None:
                dest = f"{output_dir}{output_fs.sep}t={times[i]}{output_fs.sep}"
                if output_fs.isdir(dest):
                    output_fs.rm(dest, recursive=True)
                output_fs.makedirs(dest, exist_ok=True)
            landmarks = None
            if landmarks_initialize:
                landmarks_found = False
                grid_results = None
                for landmark_translation_attempt in range(len(landmark_translations)):
                    landmark_translation = landmark_translations[
                        landmark_translation_attempt
                    ]
                    translation = _get_translation(
                        translation=landmark_translation,
                        image=moving_image_reference_t_moving_c,
                        template=moving_image_ti_moving_c,
                        image_spacing=moving_image_spacing,
                        template_spacing=moving_image_spacing,
                        com_min_quantile=landmark_com_min_quantile,
                        com_max_quantile=landmark_com_max_quantile,
                    )
                    for padding_attempt in range(len(landmark_template_padding)):
                        grid_results = find_landmarks(
                            image=moving_image_reference_t_moving_c,
                            template=moving_image_ti_moving_c,
                            slice_size=landmark_slice_size,
                            template_padding=landmark_template_padding[padding_attempt],
                            step_size=landmark_step_size,
                            image_labels=None,
                            translation=translation,
                            image_spacing=moving_image_spacing,
                            template_spacing=moving_image_spacing,
                        ).compute()
                        query = ["inlier", f"score>{landmark_min_score}"]
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

                if dest is not None and grid_results is not None:
                    grid_results.to_parquet(f"{dest}landmarks.parquet", index=False)
                if not landmarks_found:
                    raise ValueError("Not enough landmarks found.")

            elastix_object = itk_align(
                fixed_image=moving_image_reference_t_moving_c,
                moving_image=moving_image_ti_moving_c,
                parameter_object=parameter_object,
                fixed_image_spacing=moving_image_spacing,
                moving_image_spacing=moving_image_spacing,
                output_directory=dest,
                landmarks=landmarks,
            )
            if parameter_object_across_channels is None:
                del moving_image_ti_moving_c
            transform_parameter_object = _get_transform_parameter_object(elastix_object)
            del elastix_object
            # transform every c+t separately

        for j in range(nchannels[i]):
            image_i_j = (
                moving_image.isel(t=i, c=j)
                if isinstance(moving_image, xr.DataArray)
                else moving_image[i].isel(c=j)
            ).squeeze()
            if parameter_object_across_channels is not None and j != moving_channel:
                # first align and transform to reference channel
                logger.info(f"Registering c={j} to c={moving_channel} at t={i}.")
                dest_channel = None
                if output_fs is not None:
                    dest_channel = (
                        f"{output_dir}{output_fs.sep}t={times[i]}{output_fs.sep}c={j}"
                    )
                    if output_fs.isdir(dest_channel):
                        output_fs.rm(dest_channel, recursive=True)
                    output_fs.makedirs(dest_channel, exist_ok=True)

                elastix_object_across_channels = itk_align(
                    fixed_image=moving_image_ti_moving_c,
                    moving_image=image_i_j,
                    parameter_object=parameter_object_across_channels,
                    fixed_image_spacing=moving_image_spacing,
                    moving_image_spacing=moving_image_spacing,
                    output_directory=dest_channel,
                )
                transform_parameter_object_across_channels = (
                    _get_transform_parameter_object(elastix_object_across_channels)
                )
                del elastix_object_across_channels
                image_i_j = _itk_transform(
                    image=image_i_j,
                    transform_parameter_object=transform_parameter_object_across_channels,
                    labels=False,
                    image_spacing=moving_image_spacing
                    or get_image_spacing(moving_image_reference_t_moving_c.attrs),
                )["image"]

            if transform_parameter_object is not None:
                image_i_j = _itk_transform(
                    image=image_i_j,
                    transform_parameter_object=transform_parameter_object,
                    labels=False,
                    image_spacing=moving_image_spacing
                    or get_image_spacing(moving_image_reference_t_moving_c.attrs),
                )["image"]

            index = (i, j) if not unroll_channels else unrolled_t_index + j
            logger.info(f"Writing t={i}, c={j}.")
            write_callback(result_data, index, image_i_j)
            del image_i_j

        del transform_parameter_object
        unrolled_t_index += nchannels[i]
    dataset_attrs = init_dict["attrs"]
    for attr_name in attr_name_to_values.keys():
        if len(attr_name_to_values[attr_name]) == len(times):
            dataset_attrs[attr_name] = attr_name_to_values[attr_name]
    return done_callback(init_dict)


def _itk_align_landmarks(
    fixed_y: np.ndarray,
    fixed_x: np.ndarray,
    moving_y: np.ndarray,
    moving_x: np.ndarray,
    transform: Literal["affine", "rigid"] = "affine",
):
    if transform == "affine":
        transform = itk.AffineTransform.New()
    elif transform == "rigid":
        transform = itk.Rigid2DTransform.New()
    else:
        raise ValueError(f"Unknown transform: {transform}")
    LandmarkPointType = itk.Point[itk.D, 2]
    LandmarkContainerType = itk.vector[LandmarkPointType]
    fixed_landmarks = LandmarkContainerType()
    moving_landmarks = LandmarkContainerType()

    fixed_point = LandmarkPointType()
    moving_point = LandmarkPointType()

    for i in range(len(moving_y)):
        fixed_point[0] = float(fixed_x[i])
        fixed_point[1] = float(fixed_y[i])
        moving_point[0] = float(moving_x[i])
        moving_point[1] = float(moving_y[i])
        fixed_landmarks.push_back(fixed_point)
        moving_landmarks.push_back(moving_point)

    transform_initializer = itk.LandmarkBasedTransformInitializer.New()
    transform_initializer.SetFixedLandmarks(fixed_landmarks)
    transform_initializer.SetMovingLandmarks(moving_landmarks)
    transform_initializer.SetTransform(transform)
    transform_initializer.InitializeTransform()
    return transform


def itk_align(
    fixed_image: xr.DataArray | np.ndarray,
    moving_image: xr.DataArray | np.ndarray,
    parameter_object: itk.ParameterObject,
    fixed_image_spacing: None | tuple[float, float] = None,
    moving_image_spacing: None | tuple[float, float] = None,
    output_directory: None | str = None,
    fixed_mask: xr.DataArray | np.ndarray[np.uint8] | da.Array | None = None,
    moving_mask: xr.DataArray | np.ndarray[np.uint8] | da.Array | None = None,
    initial_transform: str | None = None,
    landmarks: dict[str, np.ndarray[float]] | None = None,
    additional_fixed_image: xr.DataArray | np.ndarray | None = None,
    additional_moving_image: xr.DataArray | np.ndarray | None = None,
) -> itk.ElastixRegistrationMethod:
    """Align a moving image to a fixed image using ITK registration.

    This function uses the ITK library to perform image registration, aligning a moving image to a
    fixed image.

    :param fixed_image: The fixed image represented as an xarray DataArray.
    :param moving_image: The moving image represented as an xarray DataArray.
    :param parameter_object: The ITK parameter object containing registration parameters.
    :param fixed_image_spacing: Optional spacing information for the fixed image. If not provided,
        it is determined from the image metadata.
    :param moving_image_spacing: Optional spacing information for the moving image. If not provided,
        it is determined from the image metadata.
    :param output_directory: Optional directory to save intermediate outputs. If None, no outputs
        are saved.
    :param fixed_mask: Optional fixed mask of 0's and 1's of type np.uint8
    :param moving_mask: Optional moving mask of 0's and 1's of type np.uint8
    :param additional_fixed_image: Optional labels for registration
    :param additional_moving_image: Optional labels for registration
    :param initial_transform: Optional path to initial transformation
    :param landmarks: Optional landmarks dictionary with keys 'fixed_x', 'fixed_y', 'moving_x', and
        'moving_y' used to compute intial affine tranformation.
    :return: An ITK ElastixRegistrationMethod object representing the registration.
    """
    if fixed_image_spacing is None:
        fixed_image_spacing = get_image_spacing(fixed_image.attrs)
    if moving_image_spacing is None:
        moving_image_spacing = get_image_spacing(moving_image.attrs)
    fixed_image = _array_to_itk(fixed_image, fixed_image_spacing)
    moving_image = _array_to_itk(moving_image, moving_image_spacing)
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    if (
        parameter_object.GetNumberOfParameterMaps() == 0
    ):  # hack for elastix to work with just initial transform
        tmp_pm = parameter_object.GetDefaultParameterMap("translation", 1)
        tmp_pm["AutomaticParameterEstimation"] = ["false"]
        tmp_pm["AutomaticScalesEstimation"] = ["false"]
        tmp_pm["AutomaticTransformInitialization"] = ["false"]
        tmp_pm["BSplineInterpolationOrder"] = ["1"]
        tmp_pm["FinalBSplineInterpolationOrder"] = ["1"]
        tmp_pm["FixedInternalImagePixelType"] = ["float"]
        tmp_pm["HowToCombineTransforms"] = ["Compose"]
        tmp_pm["Interpolator"] = ["LinearInterpolator"]
        tmp_pm["MaximumNumberOfIterations"] = ["0"]
        tmp_pm["MovingInternalImagePixelType"] = ["float"]
        tmp_pm["Resampler"] = ["DefaultResampler"]
        tmp_pm["ResultImageFormat"] = ["mha"]
        tmp_pm["ResultImagePixelType"] = ["short"]
        tmp_pm["WriteResultImage"] = ["false"]
        parameter_object.AddParameterMap(tmp_pm)
    if additional_fixed_image is not None:
        if additional_moving_image is None:
            raise ValueError(
                "additional_moving_image cannot be None if additional_fixed_image is not None"
            )

        additional_fixed_image = _array_to_itk(
            additional_fixed_image, fixed_image_spacing
        )
        assert additional_fixed_image.shape == fixed_image.shape
        additional_moving_image = _array_to_itk(
            additional_moving_image, moving_image_spacing
        )
        assert additional_moving_image.shape == moving_image.shape
        elastix_object.AddFixedImage(additional_fixed_image)
        elastix_object.AddMovingImage(additional_moving_image)

        for i in range(parameter_object.GetNumberOfParameterMaps()):
            p = parameter_object.GetParameterMap(i)
            metrics = list(p["Metric"])
            metrics.append("AdvancedKappaStatistic")
            p["Metric"] = metrics
            parameter_object.SetParameterMap(i, p)
    tmp_paths = []
    if landmarks is not None:
        # for i in range(parameter_object.GetNumberOfParameterMaps() - 1):
        #     p = parameter_object.GetParameterMap(i)
        #     metrics = list(p["Metric"])
        #     metrics.append("CorrespondingPointsEuclideanDistanceMetric")
        #     p["Metric"] = metrics
        #     parameter_object.SetParameterMap(i, p)
        # elastix_object.SetFixedPointSetFileName(fixed_point_set)
        # elastix_object.SetMovingPointSetFileName(moving_point_set)

        landmarks_transform = _itk_align_landmarks(
            fixed_y=landmarks["fixed_y"],
            fixed_x=landmarks["fixed_x"],
            moving_y=landmarks["moving_y"],
            moving_x=landmarks["moving_x"],
        )
        matrix = np.array(landmarks_transform.GetParameters()).tolist()
        matrix = ", ".join([f"{d:.2f}" for d in matrix])
        logger.info(
            f"Initialized registration using {len(landmarks['fixed_y'])} landmarks. "
            f"Transformation: {matrix}."
        )
        elastix_object.SetInitialTransform(landmarks_transform)

    if additional_fixed_image is not None:
        fields_update_multi_metric = [
            "MovingImagePyramid",
            "FixedImagePyramid",
            "ImageSampler",
            "Interpolator",
        ]
        for i in range(parameter_object.GetNumberOfParameterMaps()):
            p = parameter_object.GetParameterMap(i)
            p["Registration"] = ["MultiMetricMultiResolutionRegistration"]
            n_metrics = len(p["Metric"])
            for field in fields_update_multi_metric:
                value = p[field]
                if len(value) == 1:
                    p[field] = value * n_metrics
            parameter_object.SetParameterMap(i, p)
    if initial_transform is not None:
        fs = fsspec.core.url_to_fs(initial_transform)[0]
        if _get_fs_protocol(fs) != "file":
            tmp_path = _download_file(initial_transform)
            initial_transform = tmp_path
            tmp_paths.append(tmp_path)
        elastix_object.SetInitialTransformParameterFileName(initial_transform)
    if moving_mask is not None:
        mask = (
            moving_mask.values if isinstance(moving_mask, xr.DataArray) else moving_mask
        )
        elastix_object.SetMovingMask(
            itk.image_view_from_array(mask.astype(np.uint8, copy=False))
        )
    if fixed_mask is not None:
        mask = fixed_mask.values if isinstance(fixed_mask, xr.DataArray) else fixed_mask
        elastix_object.SetFixedMask(
            itk.image_view_from_array(mask.astype(np.uint8, copy=False))
        )

    elastix_object.SetParameterObject(parameter_object)
    local_directory = output_directory
    if os.environ.get("scallops_elastix") == "DEBUG":
        elastix_object.SetLogToConsole(True)
    if output_directory is not None:  # save elastix.log and transforms
        fs, _ = fsspec.core.url_to_fs(output_directory)
        if _get_fs_protocol(fs) != "file":
            output_directory = output_directory.rstrip(fs.sep)
            local_directory = tempfile.mkdtemp()
        else:
            os.makedirs(local_directory, exist_ok=True)
        elastix_object.SetLogToFile(True)
        elastix_object.SetOutputDirectory(local_directory)
    elastix_object.UpdateLargestPossibleRegion()
    for tmp_path in tmp_paths:
        os.remove(tmp_path)
    if output_directory != local_directory:
        for name in os.listdir(local_directory):
            fs.put(
                os.path.join(local_directory, name), output_directory + fs.sep + name
            )
        shutil.rmtree(local_directory, ignore_errors=True)
    return elastix_object


def _get_transform_parameter_object(
    elastix_object: itk.ElastixRegistrationMethod, is_distributed: bool | None = None
) -> Sequence[dict] | itk.ParameterObject:
    """Retrieve the transform parameter object from an Elastix registration method.

    This function handles both local and distributed environments. If running in a distributed
    environment, the transform parameter object is converted to a list of dictionaries to ensure
    compatibility with serialization.

    :param elastix_object: The Elastix registration object from which to retrieve the transform
        parameters.
    :param is_distributed: A boolean flag to indicate if the operation is distributed. If None, the
        function checks the Dask configuration.
    :return: A sequence of parameter maps as dictionaries (if distributed) or an ITK ParameterObject
        (if not distributed).
    """

    transform_parameter_object = elastix_object.GetTransformParameterObject()

    if is_distributed is None:
        is_distributed = "distributed" in dask.config.config

    if is_distributed:
        # ITK objects cannot be serialized using cloudpickle for distributed processing.
        transform_parameter_object = [
            transform_parameter_object.GetParameterMap(i).asdict()
            for i in range(transform_parameter_object.GetNumberOfParameterMaps())
        ]

    return transform_parameter_object


def itk_transform_labels(
    image: xr.DataArray,
    transform_parameter_object: itk.ParameterObject | str | Sequence[dict],
    image_spacing: None | tuple[float, float] = None,
) -> np.ndarray:
    """Apply an ITK transform to image labels.

    :param image: The input image represented as an xarray DataArray.
    :param transform_parameter_object: The ITK parameter object containing transformation parameters
        or path to directory of parameter objects
    :param image_spacing: Optional spacing information for the image. If not provided, it is
        determined from the metadata.
    :return: The transformed image as a NumPy array.
    """

    return _itk_transform(
        image=image,
        transform_parameter_object=transform_parameter_object,
        labels=True,
        image_spacing=image_spacing,
    )["image"]


def itk_transform_image(
    image: xr.DataArray,
    transform_parameter_object: itk.ParameterObject | str | Sequence[dict],
    image_spacing: None | tuple[float, float] = None,
) -> xr.DataArray:
    """Apply an ITK transform to an image.

    :param image: The input image represented as an xarray DataArray.
    :param transform_parameter_object: The ITK parameter object containing transformation parameters
        or path to directory of parameter objects
    :param image_spacing: Optional spacing information for the image. If not provided, it is
        determined from the metadata.
    :return: The transformed image as a DataArray.
    """
    output_size = _get_transform_size(transform_parameter_object)
    dims = _get_dims(image, ["t", "c", "z"])
    dim_sizes = tuple([image.sizes[d] for d in dims])
    coords = {
        c: image.coords[c]
        for c in image.coords
        if c not in ["y", "x", "z"] and c in dims
    }
    output = xr.DataArray(
        np.zeros(dim_sizes + output_size, dtype=image.dtype),
        dims=image.dims,
        attrs=dict(image.attrs),
        coords=coords,
    )

    _itk_transform_image(
        image=image,
        transform_parameter_object=transform_parameter_object,
        image_spacing=image_spacing,
        result=output.data,
    )
    return output


def _itk_transform_image_zarr(
    image: xr.DataArray,
    transform_parameter_object: itk.ParameterObject | str | Sequence[dict],
    image_root: zarr.Group | str,
    image_name: str,
    chunksize: tuple[int, int] | None,
    image_attrs: dict,
    image_spacing: None | tuple[float, float] = None,
    channels_transform_parameter_objects: dict[int, itk.ParameterObject] | None = None,
):
    """Apply an ITK-based transformation to an image and store the result in a Zarr group.

    :param image: Input image as an `xarray.DataArray`.
    :param transform_parameter_object: ITK parameter object, string path, or a sequence of parameter dictionaries.
    :param image_root: Zarr group or path to the Zarr root.
    :param image_name: Name to use for the transformed image in the Zarr store.
    :param chunksize: Chunk size to use for the Zarr dataset (default is (1024, 1024)).
    :param image_attrs: Attributes to store with the transformed image.
    :param image_spacing: Physical spacing of the input image.
    :param channels_transform_parameter_objects: Maps channel index to ITK parameter
        object
    """
    if not isinstance(image_root, zarr.Group):
        image_root = open_ome_zarr(image_root, mode="a")

    output_size = _get_transform_size(transform_parameter_object)
    transform_dims = _get_dims(image, ["t", "c", "z"])
    dim_sizes = tuple([image.sizes[d] for d in transform_dims])

    group = image_root.require_group("images").require_group(
        image_name.replace("/", "-"), overwrite=True
    )
    chunks = (1,) * len(transform_dims) + (chunksize or (1024, 1024))
    fmt = default_zarr_format()

    data = (
        group.create_array(
            "0",
            shape=dim_sizes + output_size,
            chunks=chunks,
            dtype=image.dtype,
            overwrite=True,
            **get_zarr_array_kwargs(fmt),
        )
        if _zarr_v3()
        else group.create_dataset(
            "0",
            shape=dim_sizes + output_size,
            chunks=chunks,
            dtype=image.dtype,
            overwrite=True,
            **get_zarr_array_kwargs(fmt),
        )
    )

    _itk_transform_image(
        image=image,
        transform_parameter_object=transform_parameter_object,
        image_spacing=image_spacing,
        result=data,
    )
    coords = {
        c: image.coords[c]
        for c in image.coords
        if c not in ["y", "x", "z"] and c in transform_dims
    }
    write_zarr(
        grp=group,
        data=data,
        image_attrs=image_attrs,
        coords=coords,
        dims=image.dims,
        zarr_format="zarr",
    )


def _itk_transform_image(
    image: xr.DataArray,
    transform_parameter_object: itk.ParameterObject | str | Sequence[dict],
    image_spacing: None | tuple[float],
    result: np.ndarray | zarr.Array,
):
    """Apply ITK-based transformation to each (t, c, z) slice individually.

    :param image: Input image as an `xarray.DataArray`.
    :param transform_parameter_object: ITK parameter object, string path, or sequence of parameter dictionaries.
    :param image_spacing: Physical spacing of the input image.
    :param result: Output array or Zarr dataset to store the transformed result.
    """
    if image_spacing is None:
        image_spacing = get_image_spacing(image.attrs)
    for index in itertools.product(
        *[range(image.shape[i]) for i in range(image.ndim - 2)]
    ):
        result[index] = _itk_transform(
            image=image.data[index],
            transform_parameter_object=transform_parameter_object,
            labels=False,
            image_spacing=image_spacing,
        )["image"]


def _parameter_object(
    parameter_object: itk.ParameterObject | str | Sequence[dict],
) -> itk.ParameterObject:
    """Convert a parameter object input into an ITK ParameterObject.

    :param parameter_object: ITK parameter object, string path, or sequence of parameter
        dictionaries.
    :return: ITK ParameterObject.
    """
    if isinstance(parameter_object, Sequence):
        tmp = itk.ParameterObject.New()
        for p in parameter_object:
            if isinstance(p, itk.ParameterObject):
                for i in range(p.GetNumberOfParameterMaps()):
                    tmp.AddParameterMap(p.GetParameterMap(i))
            else:
                tmp.AddParameterMap(p)
        parameter_object = tmp
    elif isinstance(parameter_object, str):
        tmp = parameter_object
        parameter_object = _load_itk_parameters_from_dir(parameter_object)
        if parameter_object.GetNumberOfParameterMaps() == 0:
            time.sleep(0.5)
            parameter_object = _load_itk_parameters_from_dir(tmp)
    return parameter_object


def _itk_transform(
    image: xr.DataArray,
    transform_parameter_object: itk.ParameterObject | str | list,
    labels: bool,
    image_spacing: None | tuple[float, float] = None,
    deformation_field: bool = False,
) -> dict[str, Any]:
    """Apply an ITK transformation to an image and return the result as a NumPy array.

    :param image: Input image as an `xarray.DataArray`.
    :param transform_parameter_object: ITK parameter object, string path, or list of parameter dictionaries.
    :param labels: Whether the image contains label data.
    :param image_spacing: Physical spacing of the input image.
    :param deformation_field: Whether to compute deformation field.
    :return: Transformed image as a NumPy array.
    """
    transform_parameter_object = _parameter_object(transform_parameter_object)
    image_spacing = (
        get_image_spacing(image.attrs) if image_spacing is None else image_spacing
    )
    itk_image = _array_to_itk(image, image_spacing)

    transformix_object = itk.TransformixFilter.New(itk_image)
    transform_parameter_object.SetParameter(
        "FinalBSplineInterpolationOrder", ["0"] if labels else ["3"]
    )
    local_directory = None
    if deformation_field:
        local_directory = tempfile.mkdtemp()
        transformix_object.SetOutputDirectory(local_directory)
        transformix_object.SetComputeDeformationField(True)

    transformix_object.SetTransformParameterObject(transform_parameter_object)
    transformix_object.UpdateLargestPossibleRegion()
    output = transformix_object.GetOutput()
    output = itk.GetArrayViewFromImage(output)
    if labels:
        output = (
            output.astype(image.dtype, copy=False)
            if output.dtype != image.dtype
            else output
        )

    result = dict(image=output)
    if deformation_field:
        output_deformation_field = transformix_object.GetOutputDeformationField()

        result["deformation_field"] = output_deformation_field
    if local_directory is not None:
        shutil.rmtree(local_directory, ignore_errors=True)
    return result


def _get_transform_size(
    transform_parameter_object: itk.ParameterObject | str | Sequence[dict],
) -> tuple[int, int]:
    """Retrieve the output size of the transformed image from the ITK parameter object.

    :param transform_parameter_object: ITK parameter object, string path, or sequence of parameter
        dictionaries.
    :return: Tuple representing the size of the transformed image.
    """
    if not isinstance(transform_parameter_object, itk.ParameterObject):
        transform_parameter_object = _parameter_object(transform_parameter_object)
    size = transform_parameter_object.GetParameter(
        transform_parameter_object.GetNumberOfParameterMaps() - 1, "Size"
    )
    return int(size[1]), int(size[0])


# from https://github.com/NHPatterson/wsireg/blob/master/wsireg/parameter_maps/reg_params.py
WSIREG_PARAM_MAPS = {
    "rigid": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["true"],
        "AutomaticTransformInitializationMethod": ["CenterOfGravity"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["200"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "100.0",
            "75.0",
            "66.0",
            "50.0",
            "25.0",
            "15.0",
            "10.0",
            "10.0",
            "5.0",
            "1.0",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["16"],
        "NumberOfResolutions": ["10"],
        "NumberOfSpatialSamples": ["10000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["EulerTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "affine": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["true"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["200"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "100.0",
            "75.0",
            "66.0",
            "50.0",
            "25.0",
            "15.0",
            "10.0",
            "10.0",
            "5.0",
            "1.0",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["10"],
        "NumberOfSpatialSamples": ["10000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["AffineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "similarity": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["true"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["200"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "100.0",
            "75.0",
            "66.0",
            "50.0",
            "25.0",
            "15.0",
            "10.0",
            "10.0",
            "5.0",
            "1.0",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["10"],
        "NumberOfSpatialSamples": ["10000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["SimilarityTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "nl": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FinalGridSpacingInPhysicalUnits": ["100"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "GridSpacingSchedule": [
            "512",
            "512",
            "392",
            "392",
            "256",
            "256",
            "128",
            "128",
            "64",
            "64",
            "32",
            "32",
            "16",
            "16",
            "4",
            "4",
            "2",
            "2",
            "1",
            "1",
        ],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["200"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "100",
            "90",
            "70",
            "50",
            "40",
            "30",
            "20",
            "10",
            "1",
            "1",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["10"],
        "NumberOfSpatialSamples": ["50000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["BSplineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "fi_correction": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImagePyramidSchedule": ["8", "8", "4", "4", "2", "2", "1", "1"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["75"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": ["100", "50", "20", "10"],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["16"],
        "NumberOfResolutions": ["4"],
        "NumberOfSpatialSamples": ["10000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["EulerTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "nl-reduced": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FinalGridSpacingInPhysicalUnits": ["100"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "GridSpacingSchedule": [
            "392",
            "392",
            "256",
            "256",
            "128",
            "128",
            "64",
            "64",
            "32",
            "32",
        ],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["150"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "50",
            "40",
            "30",
            "20",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["5"],
        "NumberOfSpatialSamples": ["50000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["BSplineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "nl-mid": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FinalGridSpacingInPhysicalUnits": ["150"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "GridSpacingSchedule": [
            "512",
            "512",
            "128",
            "128",
            "64",
            "64",
            "32",
            "32",
            "2",
            "2",
        ],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["200"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "100",
            "70",
            "50",
            "30",
            "10",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["5"],
        "NumberOfSpatialSamples": ["50000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["BSplineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "rigid-expanded": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["true"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["500"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "100.0",
            "75.0",
            "66.0",
            "50.0",
            "25.0",
            "15.0",
            "10.0",
            "10.0",
            "5.0",
            "1.0",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["16"],
        "NumberOfResolutions": ["10"],
        "NumberOfSpatialSamples": ["30000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["EulerTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "nl3": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FinalGridSpacingInPhysicalUnits": ["200"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedImagePyramidSchedule": [str(int(2**2)), str(int(2**2))],
        "FixedInternalImagePixelType": ["float"],
        "GridSpacingSchedule": [
            "8",
            "8",
        ],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["1000"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "10",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingImagePyramidSchedule": [str(int(2**2)), str(int(2**2))],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["1"],
        "NumberOfSpatialSamples": ["125000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["BSplineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "nl2": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FinalGridSpacingInPhysicalUnits": ["75"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "GridSpacingSchedule": [
            "512",
            "512",
            "392",
            "392",
            "256",
            "256",
            "128",
            "128",
            "64",
            "64",
            "32",
            "32",
            "16",
            "16",
            "4",
            "4",
            "2",
            "2",
            "1",
            "1",
        ],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["200"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": [
            "25",
            "20",
            "15",
            "10",
            "10",
            "8",
            "5",
            "5",
            "5",
            "2",
        ],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["10"],
        "NumberOfSpatialSamples": ["15000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["BSplineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
}
SCALLOPS_PARAM_MAPS = {
    "nl-100": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FinalGridSpacingInPhysicalUnits": ["100"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "GridSpacingSchedule": ["1", "1"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["500"],
        "MaximumNumberOfSamplingAttempts": ["10"],
        "MaximumStepLength": ["1"],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["1"],
        "NumberOfSpatialSamples": ["50000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["RecursiveBSplineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "affine": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["false"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["500"],
        "MaximumNumberOfSamplingAttempts": ["10"],
        "MaximumStepLength": ["1.0"],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["32"],
        "NumberOfResolutions": ["1"],
        "NumberOfSpatialSamples": ["10000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["AffineTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
    "rigid": {
        "AutomaticScalesEstimation": ["true"],
        "AutomaticTransformInitialization": ["true"],
        "AutomaticTransformInitializationMethod": ["CenterOfGravity"],
        "BSplineInterpolationOrder": ["1"],
        "CompressResultImage": ["true"],
        "DefaultPixelValue": ["0"],
        "ErodeMask": ["true"],
        "FinalBSplineInterpolationOrder": ["1"],
        "FixedImageDimension": ["2"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "FixedInternalImagePixelType": ["float"],
        "HowToCombineTransforms": ["Compose"],
        "ImageSampler": ["Random"],
        "Interpolator": ["LinearInterpolator"],
        "MaximumNumberOfIterations": ["500"],
        "MaximumNumberOfSamplingAttempts": [
            "10",
        ],
        "MaximumStepLength": ["1.0"],
        "Metric": ["AdvancedMattesMutualInformation"],
        "MovingImageDimension": ["2"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "MovingInternalImagePixelType": ["float"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfHistogramBins": ["16"],
        "NumberOfResolutions": ["1"],
        "NumberOfSpatialSamples": ["10000"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "Registration": ["MultiResolutionRegistration"],
        "RequiredRatioOfValidSamples": ["0.05"],
        "ResampleInterpolator": ["FinalNearestNeighborInterpolator"],
        "Resampler": ["DefaultResampler"],
        "ResultImageFormat": ["mha"],
        "ResultImagePixelType": ["short"],
        "Transform": ["EulerTransform"],
        "UseDirectionCosines": ["true"],
        "WriteResultImage": ["false"],
        "WriteTransformParametersEachResolution": ["false"],
    },
}
# add advanced mean squares (ams) and normalized correlation (anc)
metrics = [("AdvancedMeanSquares", "ams"), ("AdvancedNormalizedCorrelation", "anc")]


def _add_metrics(d):
    for k in list(d.keys()):
        for metric, suffix in metrics:
            p = d[k].copy()
            p["Metric"] = [metric]
            d[f"{k}-{suffix}"] = p


_add_metrics(WSIREG_PARAM_MAPS)
_add_metrics(SCALLOPS_PARAM_MAPS)
# add -wsireg suffix
for key in list(WSIREG_PARAM_MAPS.keys()):
    WSIREG_PARAM_MAPS[key + "-wsireg"] = WSIREG_PARAM_MAPS[key]
    del WSIREG_PARAM_MAPS[key]
DEFAULT_REG_PARAM_MAPS = dict()
DEFAULT_REG_PARAM_MAPS.update(WSIREG_PARAM_MAPS)
DEFAULT_REG_PARAM_MAPS.update(SCALLOPS_PARAM_MAPS)
