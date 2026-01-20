"""Image Alignment using Cross-Correlation.

Provides functions for aligning images based on cross-correlation, a
fundamental step in image processing pipelines. Image alignment is particularly crucial in scenarios involving
time-series data or multi-channel images, where shifts or movements may occur.


Authors:
    - The SCALLOPS development team.
"""

import itertools
from collections.abc import Sequence
from typing import Any, Union

import numpy as np
import xarray as xr
from numpy.random import RandomState
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp

from scallops.utils import applyIJ, match_size


def _apply_window(
    data: Union[xr.DataArray, np.ndarray], window: int = 2
) -> Union[xr.DataArray, np.ndarray]:
    """Find borders in window slices.

    :param data: Array with image data
    :param window: Size of the window
    :return: sliced data within window
    """
    height, width = data.shape[-2:]
    factor = 1 - (1 / float(window))
    i = int((height / 2.0) * factor)
    j = int((width / 2.0) * factor)
    return data[..., i : height - i, j : width - j]


def _align_between_t(
    image: xr.DataArray,
    upsample_factor: int = 2,
    window: int = 2,
    channel_index: int = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """Align data between timepoints.

    :param channel_index: Channel index to align
    :param image: :DataArray: stack of images
    :param window: Size of the window
    :param upsample_factor: Level of upsampling
    """
    image = image.copy()
    transforms = _align_between_t_transform(
        image=image,
        upsample_factor=upsample_factor,
        window=window,
        channel_index=channel_index,
    )
    coordinate_transformations = image.attrs.get("coordinateTransformations")
    if coordinate_transformations is None:
        coordinate_transformations = []
        image.attrs["coordinateTransformations"] = coordinate_transformations
    coordinate_transformations += transforms
    _apply_transforms(image, transforms)
    return image


def _align_between_t_transform(
    image: xr.DataArray,
    upsample_factor: int = 2,
    window: int = 2,
    channel_index: int = 0,
) -> list[dict]:
    """Align data between timepoints.

    :param channel_index: Channel index to align
    :param image: :DataArray: stack of images
    :param window: Size of the window
    :param upsample_factor: Level of upsampling
    :return: list of dicts describing transformations to apply
    """
    assert "t" in image.dims
    times = image.t.values
    channels = image.c.values

    dims = ["t"]
    dims += [d for d in ["z"] if d in image.dims]
    dim_vals = [times[1:]]
    dim_vals += [image[d].values for d in dims[1:]]
    transforms = []
    image = _apply_window(image.isel(c=channel_index), window)

    for dim_val in itertools.product(*dim_vals):
        src_selector = dict(zip(dims, dim_val))
        # target is t=0
        target_selector = src_selector.copy()
        target_selector["t"] = times[0]
        offset, _, _ = phase_cross_correlation(
            image.sel(src_selector).values,
            image.sel(target_selector).values,
            upsample_factor=upsample_factor,
        )

        if not np.all(offset == 0.0):
            for c in channels:  # apply the translation across all channels
                selector = src_selector.copy()
                selector["c"] = c
                transforms.append(dict(translation=offset.tolist(), sel=selector))
    return transforms


def _align_within_t(
    image: xr.DataArray,
    align_within_time_channels,
    upsample_factor: int = 4,
    window: int = 1,
    filter_percentiles: Sequence[float] | None = (0, 90),
) -> xr.DataArray:
    """Align data within each time point (cycle)

    :param image: The image to align
    :param upsample_factor: Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
    :param window: Size of the window to sample
    :param filter_percentiles: Replaces data outside of percentile range [q1, q2] with uniform noise over the range [q1,q2].
    :return:Aligned image
    """
    image = image.copy()
    transforms = _align_within_t_transform(
        image=image,
        align_within_time_channels=align_within_time_channels,
        upsample_factor=upsample_factor,
        window=window,
        filter_percentiles=filter_percentiles,
    )
    coordinate_transformations = image.attrs.get("coordinateTransformations")
    if coordinate_transformations is None:
        coordinate_transformations = []
        image.attrs["coordinateTransformations"] = coordinate_transformations
    coordinate_transformations += transforms
    _apply_transforms(image, transforms)
    return image


def _align_within_t_transform(
    image: xr.DataArray,
    align_within_time_channels,
    upsample_factor: int = 4,
    window: int = 1,
    filter_percentiles: Sequence[float] | None = [0, 90],
) -> list[dict]:
    """Align data within each time point (cycle)

    :param image: The image to align
    :param upsample_factor: Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
    :param window: Size of the window to sample
    :return: list of dicts describing transformations to apply
    """

    image = _apply_window(image, window)
    reference_channel = align_within_time_channels[0]
    channels = image.c.values
    dims = [d for d in ["t", "z"] if d in image.dims]
    dim_vals = [image[d].values for d in dims]
    if len(dims) == 0:  # image does not contain t or z
        dim_vals = [[""]]
    transforms = []
    for dim_val in itertools.product(*dim_vals):
        sel = dict(zip(dims, dim_val)) if len(dims) > 0 else dict()
        data = image.sel(sel)
        if filter_percentiles is not None:
            data = xr.DataArray(
                _filter_percentiles(
                    data.copy(), q1=filter_percentiles[0], q2=filter_percentiles[1]
                ),
                dims=data.dims,
            )
        for i in range(len(align_within_time_channels)):
            c = align_within_time_channels[i]
            offset, _, _ = phase_cross_correlation(
                data.isel(c=c).values,
                data.isel(c=reference_channel).values,
                upsample_factor=upsample_factor,
            )

            if not np.all(offset == 0.0):
                selector = sel.copy()
                selector["c"] = channels[c]
                transforms.append(dict(translation=offset.tolist(), sel=selector))

    return transforms


def _apply_transforms(data: xr.DataArray, transforms: list[dict[str, Any]]) -> None:
    """Apply a series of transformations to an xarray DataArray.

    This function iterates over a list of transformation dictionaries and applies
    each transformation to the corresponding selection within the provided DataArray.
    It uses `skimage`'s `SimilarityTransform` and `warp` to perform the transformations.

    :param data: An xarray DataArray containing the data to be transformed.
    :param transforms: A list of dictionaries, where each dictionary represents a
        transformation. Each dictionary must contain:
        - `"translation"`: A sequence representing the translation offsets (x, y, [z]).
        - `"sel"`: A selection used to index the DataArray (e.g., slice or boolean mask).

    :example:

    .. code-block:: python

        data = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"])
        transforms = [
            {"translation": [5, 3], "sel": {"x": slice(0, 5), "y": slice(0, 5)}},
            {"translation": [-2, -1], "sel": {"x": slice(5, 10), "y": slice(5, 10)}},
        ]
        _apply_transforms(data, transforms)

    :return: None. The function modifies the input DataArray in place.
    """
    for transform in transforms:
        offset = transform["translation"]
        # skimage SimilarityTransform uses (x, y, [z]) convention; reverse the offset
        translation = offset[::-1]
        sel = transform["sel"]
        st = SimilarityTransform(translation=translation)

        # Apply the warp transform to the selected data, preserving the original range and dtype
        data.loc[sel] = warp(data.sel(sel).values, st, preserve_range=True).astype(
            data.dtype
        )


@applyIJ
def _filter_percentiles(data: np.ndarray, q1: float, q2: float) -> np.ndarray:
    """Replaces data outside of percentile range [q1, q2] with uniform noise over the range [q1,
    q2].

    Useful for eliminating alignment artifacts due to bright features or regions of zeros.
    """
    x1, x2 = np.percentile(data, [q1, q2])
    mask = (x1 > data) | (x2 < data)
    return _fill_noise(data, mask, x1, x2)


def _fill_noise(data: np.ndarray, mask: np.ndarray, x1: float, x2: float) -> np.ndarray:
    """Replace the masked indexes with random uniform values.

    :param data: Array with the image data
    :param mask: Boolean array with the conditions for replacement
    :param x1: Lower quantile
    :param x2: Upper quantile
    :return: Array with the filled data
    """
    filtered = data.copy()
    rs = RandomState(0)
    filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
    return filtered


def align_image(
    image: xr.DataArray,
    align_within_time_channels: Sequence[int] | None = None,
    align_between_time_channel: int | None = 0,
    window: int = 2,
    upsample_factor: int = 2,
    filter_percentiles: Sequence[float] | None = None,
) -> xr.DataArray:
    """Align an image within cycles (timepoints) and then between cycles.

    This function aligns an image stack within cycles (timepoints) and between cycles using cross correlation.
    The returned image has attributes containing the key 'coordinateTransformations', which describes the applied
    transformations.

    :param image:
        DataArray containing the data to align with dimensions of (t,c,z,y,x)
    :param align_within_time_channels:
        List of channel indices for aligning within cycles. Set to `None` to skip aligning within a timepoint.
    :param align_between_time_channel:
        Channel index to use for aligning between cycles. Set to `None` to skip aligning between timepoints.
    :param window: The Window size.
    :param upsample_factor:
        Subpixel alignment is done if `upsample_factor` is greater than one. Images will be registered to
        within ``1 / upsample_factor`` of a pixel. For example, ``upsample_factor == 20`` means the images will be
        registered within 1/20th of a pixel. Parameter passed to :skimage.registration.phase_cross_correlation:
    :param filter_percentiles:
        Replaces data outside of percentile range [q1, q2] with uniform noise over the range [q1,q2] when aligning
        within cycle.
    :return: Aligned image

    :example:

    .. code-block:: python

        import xarray as xr
        from scallops.registration.crosscorrelation import align_image

        # Create a sample DataArray
        channels = ["ChannelA", "ChannelT", "ChannelG", "ChannelC"]
        data = xr.DataArray(
            np.random.rand(100, 10, len(channels)),
            dims=("t", "c", "z", "y", "x"),
            coords={"c": channels},
        )

        # Generate aligned image
        aligned_image = align_image(
            data,
            align_within_time_channels=[0, 1, 2],
            align_between_time_channel=0,
            window=2,
            upsample_factor=2,
            filter_percentiles=(0, 90),
        )
    """
    registration_attrs = dict()
    if align_within_time_channels is not None and len(align_within_time_channels) > 1:
        # if filter percentiles is not None, values outside of filter_percentiles are modified
        image = _align_within_t(
            image,
            align_within_time_channels=align_within_time_channels,
            window=window,
            upsample_factor=upsample_factor,
            filter_percentiles=filter_percentiles,
        )
        registration_attrs["filter_percentiles"] = filter_percentiles
        registration_attrs["within_t"] = align_within_time_channels

    if (
        align_between_time_channel is not None
        and "t" in image.dims
        and image.sizes["t"] > 1
    ):
        image = _align_between_t(
            image,
            window=window,
            upsample_factor=upsample_factor,
            channel_index=align_between_time_channel,
        )
        registration_attrs["between_t"] = align_between_time_channel
    if len(registration_attrs) > 0:
        image.attrs["registration"] = registration_attrs
    return image


def align_images(
    image_1: xr.DataArray,
    image_2: xr.DataArray,
    channel_index: int = 0,
    upsample_factor: int = 2,
    autoscale: bool = True,
) -> xr.DataArray:
    """Align two images, using the channel at position `channel_index` (typically DAPI).

    This function aligns two images using cross correlation with optional autoscaling.

    :param image_1:
        First image to align
    :param image_2:
        Second image to align
    :param channel_index:
        Index of the desired channel to align by
    :param upsample_factor:
        Subpixel alignment is done if `upsample_factor` is greater than one. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example, ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.
    :param autoscale:
        Automatically scale `image_2` prior to alignment. Offsets are applied to the unscaled image so no resolution is
        lost.
    :return: `image_2` with calculated offsets applied to all channels.

    :example:

    .. code-block:: python

        import xarray as xr
        from your_module import align_images

        # Create two sample DataArrays
        channels = ["ChannelA", "ChannelT", "ChannelG", "ChannelC"]
        data_1 = xr.DataArray(
            np.random.rand(100, 10, len(channels)),
            dims=("t", "c", "z", "y", "x"),
            coords={"c": channels},
        )
        data_2 = xr.DataArray(
            np.random.rand(100, 10, len(channels)),
            dims=("t", "c", "z", "y", "x"),
            coords={"c": channels},
        )

        # Generate aligned image
        aligned_image = align_images(
            data_1, data_2, channel_index=0, upsample_factor=2, autoscale=True
        )
    """
    result = image_2.copy()
    image_1 = image_1.isel(c=channel_index)
    image_2 = image_2.isel(c=channel_index)
    dims = [d for d in ["t", "z"] if d in result.dims]
    dim_vals = [result[d].values for d in dims]
    assert all([image_1.sizes[d] == image_2.sizes[d] for d in dims]), "Sizes differ"
    if autoscale:
        scale_factor = image_2.shape[-1] / image_1.shape[-1]
        image_2 = xr.DataArray(
            match_size(image_2.values, image_1.values), dims=image_2.dims
        )

    channels = result.c.values
    if len(dims) == 0:  # images do not contain t or z
        dim_vals = [[""]]

    for dim_val in itertools.product(*dim_vals):
        sel = dict(zip(dims, dim_val)) if len(dims) > 0 else dict()

        offset, _, _ = phase_cross_correlation(
            image_2.sel(sel).values,
            image_1.sel(sel).values,
            upsample_factor=upsample_factor,
        )
        if autoscale:
            offset *= scale_factor
        for c in range(result.sizes["c"]):
            # skimage SimilarityTransform has (x,y,[z]) convention
            st = SimilarityTransform(translation=offset[::-1])
            selector = sel.copy()
            selector["c"] = channels[c]
            result.loc[selector] = warp(
                result.sel(selector).values, st, preserve_range=True
            ).astype(result.dtype)

    return result
