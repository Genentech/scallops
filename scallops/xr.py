"""SCALLOPS Xarray Utility Functions Module.

Provides a collection of utility functions for working with Xarray
DataArrays. These functions are designed to facilitate common operations such as
applying functions over dimensions, computing grouped quantiles, and cropping images.

Authors:
- The SCALLOPS development team
"""

import itertools
from collections.abc import Callable, Sequence
from typing import Any, Literal, Union

import numpy as np
import xarray as xr


def _z_projection(
    data: xr.DataArray, z_index: int | Literal["max", "focus"]
) -> xr.DataArray:
    """Project data using specified z index.

    :param data: The data to be projected.
    :param z_index: The z index to be selected, `max`, or 'focus`
    :return: The projected data.
    """
    if "z" not in data.dims:
        return data
    if isinstance(z_index, str):
        if z_index == "focus":
            from scallops.stitch.utils import _best_focus_z_index

            return data.isel(z=_best_focus_z_index(data))
        return data.max(dim="z", keep_attrs=True)
    return data.isel(z=z_index)


def _get_dims(
    array: xr.DataArray,
    dims: list[str],
    missing_dims: Literal["ignore", "error"] = "ignore",
) -> list[str]:
    """Retrieve dimensions from a DataArray, with options to ignore or raise errors
    for missing dimensions.

    This function checks if the specified dimensions are present in the given
    DataArray. If a dimension is missing, it either ignores the missing dimension or
    raises an error based on the `missing_dims` parameter.

    :param array: The input DataArray from which to retrieve dimensions.
    :param dims: List of dimensions to retrieve from the DataArray.
    :param missing_dims: Specifies the behavior when a dimension is missing.
        If "ignore", missing dimensions are ignored. If "error", a ValueError is
        raised for missing dimensions. Default is "ignore".
    :return: List of dimensions that are present in the DataArray.

    :raises ValueError:
        If `missing_dims` is set to "error" and a specified dimension is not found in
        the DataArray.

    :example:

        .. code-block:: python

            import xarray as xr
            import numpy as np

            data = np.random.rand(4, 3, 2)
            array = xr.DataArray(data, dims=["x", "y", "z"])

            # Retrieve dimensions, ignoring missing ones
            dims = _get_dims(array, ["x", "y", "time"], missing_dims="ignore")
            print(dims)  # Output: ['x', 'y']

            # Retrieve dimensions, raising an error for missing ones
            try:
                dims = _get_dims(array, ["x", "y", "time"], missing_dims="error")
            except ValueError as e:
                print(e)  # Output: Dimension time not found
    """
    _dims = []
    for d in dims:
        if d not in array.dims:
            if missing_dims == "error":
                raise ValueError(f"Dimension {d} not found")
        else:
            _dims.append(d)
    return _dims


def apply_data_array(
    array: xr.DataArray,
    dims: list[str],
    func: Callable[[xr.DataArray, Any], np.ndarray | xr.DataArray],
    missing_dims: Literal["ignore", "error"] = "ignore",
    **kwargs: Any,
) -> xr.DataArray:
    """Apply a function to all combinations of the values for the given dimensions.

    :param array: The data array
    :param dims: List of dimensions to apply the function over
    :param func: Function to apply
    :param missing_dims: Whether to ignore or raise error for missing dims
    :param kwargs: Keyword arguments to pass to `func`
    :return: DataArray containing the same dimensions as the input array

    :example:

        .. code-block:: python

            apply_data_array(
                image,
                ["c", "t"],
                lambda x: skimage.exposure.rescale_intensity(
                    x, in_range=tuple(x.quantile([0, 0.95]).values)
                ),
            )
    """
    assert isinstance(array, xr.DataArray)
    dims = _get_dims(array=array, dims=dims, missing_dims=missing_dims)
    array_copy = array.copy()
    if len(dims) == 0:
        array_copy[...] = func(array, **kwargs)
        return array_copy
    dim_vals = [array[d].values for d in dims]
    for dim_val in itertools.product(*dim_vals):
        sel = dict(zip(dims, dim_val))

        array_copy.loc[sel] = func(array.sel(sel), **kwargs)
    return array_copy


def iter_data_array(
    array: xr.DataArray,
    dims: list[str],
    func: Callable[[xr.DataArray, dict], None],
    missing_dims: Literal["ignore", "error"] = "ignore",
):
    """Call a function for all combinations of the values for the given dimensions.

    :param array: The data array
    :param dims: List of dimensions to apply the function over
    :param missing_dims: Whether to ignore or raise error for missing dims
    :param func: The function to apply. Invoked with the sliced DataArray and selector
        used to create the slice.
    """
    dims = _get_dims(array=array, dims=dims, missing_dims=missing_dims)
    if len(dims) == 0:
        func(array, dict())
    else:
        dim_vals = [array[d].values for d in dims]
        for dim_val in itertools.product(*dim_vals):
            sel = dict(zip(dims, dim_val))
            func(array.sel(sel), sel)


def _crop(
    image: xr.DataArray, crop: Union[int, tuple[int, int, int, int]]
) -> xr.DataArray:
    """Crop an image DataArray to specified dimensions.

    This function crops the input image DataArray to the specified dimensions.
    If a single integer is provided, the image is cropped to a square centered at the
    middle of the image. If a tuple of four integers is provided, the image is cropped
    to the specified rectangle.

    :param image: The input image DataArray to be cropped.
    :param crop: The crop dimensions. If an integer, the image is cropped
        to a square of that size centered at the middle of the image. If a tuple of
        four integers, it specifies the
        (xstart, ystart, crop_width, crop_height).
    :return: The cropped image DataArray.

    :example:

        .. code-block:: python

            import xarray as xr
            import numpy as np

            data = np.random.rand(100, 100)
            image = xr.DataArray(data, dims=["y", "x"])

            # Crop to a 50x50 square centered at the middle
            cropped_image = _crop(image, 50)
            print(cropped_image)

            # Crop to a specified rectangle
            cropped_image = _crop(image, (10, 10, 60, 60))
            print(cropped_image)
    """
    if not isinstance(crop, Sequence):
        half_crop = round(crop / 2)
        xstart = round(image.sizes["x"] / 2) - half_crop
        ystart = round(image.sizes["y"] / 2) - half_crop
        crop_width = xstart + crop
        crop_height = ystart + crop
    else:
        xstart, ystart, crop_width, crop_height = crop
    return (
        image.isel(x=slice(xstart, crop_width), y=slice(ystart, crop_height))
        if isinstance(image, xr.DataArray)
        else image[ystart:crop_height, xstart:crop_width]
    )
