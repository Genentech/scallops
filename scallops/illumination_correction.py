"""Image illumination correction.

This module provides functionalities to correct illumination artifacts
in microscopy or other image data, ensuring uniform brightness and contrast.

Authors:
    - The SCALLOPS development team
"""

import logging
import shutil
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import skimage
import xarray as xr

from scallops.io import _images2fov
from scallops.stitch.utils import _init_tiles
from scallops.xr import _z_projection, apply_data_array

logger = logging.getLogger("scallops")


def apply_illumination_correction(
    img: np.ndarray,
    ffp: np.ndarray | None = None,
    dfp: np.ndarray | None = None,
) -> np.ndarray:
    """Apply illumination correction.

    :param img: Input image
    :param ffp: FFP image
    :param dfp: DFP image
    :return: Corrected image
    """

    img = skimage.util.img_as_float(img, force_copy=True)
    if dfp is not None:
        img -= dfp
    if ffp is not None:
        img /= ffp
    img.clip(0, 1, out=img)
    return img


def _smooth_and_rescale(
    im: xr.DataArray, smooth: int | None, rescale: bool
) -> xr.DataArray:
    """Smooth and rescale an image using a median filter and robust rescaling.

    :param im: Input image as an xarray DataArray.
    :param smooth: Disk size for the median filter; if None, a default value is computed.
    :param rescale: Boolean indicating whether to apply robust rescaling.
    :return: Smoothed and rescaled image as an xarray DataArray.
    """
    apply_by = ["c"]
    if smooth is None:
        # Default smooth size is 1/20th of the image area
        smooth = int(np.sqrt((im.shape[-1] * im.shape[-2]) / (np.pi * 20)))
        logger.info(f"Smooth: {smooth}")

    if smooth > 0:

        def _median_filter(x, disk):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress large bin warning
                return skimage.filters.median(x, disk, behavior="rank")

        im = apply_data_array(
            im.astype(np.uint16, copy=False),
            apply_by,
            _median_filter,
            disk=skimage.morphology.disk(smooth),
        )

    if rescale:

        def _rescale_channels(x: xr.DataArray) -> xr.DataArray:
            """Rescale each channel using the 2nd percentile as the robust minimum. Ensures no
            division by zero by replacing 0 with 1.

            :param x: Input data array.
            :return: Rescaled data array.
            """
            robust_min = x.quantile(q=0.02)
            robust_min = 1 if robust_min == 0 else robust_min
            x = x / robust_min
            return x.clip(min=1)

        im = apply_data_array(im.astype(float, copy=False), apply_by, _rescale_channels)

    return im


def illumination_correction(
    image_tuple: tuple[tuple[str, ...], list[str], dict],
    smooth: int = None,
    rescale: bool = True,
    z_index: Literal["max", "focus"] | str | int = "max",
    channel: int = 0,
    agg_method: str = Literal["mean", "median", "min"],
    expected_images: int | None = None,
) -> tuple[xr.DataArray, list[str], str | int | Sequence[int]]:
    """Calculate illumination correction.

    :param image_tuple: Image tuple.
    :param smooth: The radius of the disk-shaped footprint for median filter
    :param rescale: Whether to use 2nd percentile for robust image minimum
    :param z_index: Either 'max', 'focus', or z-index
    :param channel: The channel select the best focus z-index if z_index is `focus`.
    :param agg_method: Method to aggregate images

    Equivalent to CellProfiler's CorrectIlluminationCalculate module with option "Regular", "All", "Median Filter"

    Note: algorithm originally benchmarked using ~250 images per plate to calculate plate-wise
    illumination correction functions (Singh et al. J Microscopy, 256(3):231-236, 2014)
    """
    warnings.filterwarnings(
        "ignore", message="ND2File file not closed before garbage collection"
    )
    _, image_filepaths, image_metadata = image_tuple

    init = _init_tiles(
        image_filepaths=image_filepaths,
        image_metadata=image_metadata,
        channel=channel,
        z_index=z_index,
        expected_images=expected_images,
        download_suffixes={".nd2", ".tif", ".tiff"},
    )
    z_index = init["z_index"]
    z_index_per_tile = isinstance(z_index, (Sequence, np.ndarray)) and not isinstance(
        z_index, str
    )
    metadata_fields = init["metadata_fields"]
    n_scenes = init["n_scenes"]
    tmp_dir = init["tmp_dir"]
    filepaths = init["filepaths"]
    fileattrs = init["fileattrs"]
    keys = init["keys"]

    images = []
    z_tiles_removed = (
        "z" in metadata_fields
        and isinstance(z_index, (Sequence, np.ndarray, int))
        and not isinstance(z_index, str)
    )
    n = n_scenes if n_scenes is not None else len(filepaths)
    for i in range(n):
        file_list = filepaths[0] if n_scenes is not None else filepaths[i]
        attrs = fileattrs[i] if fileattrs is not None else None
        z_index_ = z_index[i] if z_index_per_tile else z_index

        image = _images2fov(
            file_list,
            attrs,
            dask=True,
            scene_id=i if n_scenes is not None else None,
        )
        image = (
            _z_projection(image, z_index_)
            if not z_tiles_removed
            else image.isel(z=0, missing_dims="ignore")
        )
        image = image.squeeze(d for d in ("t", "z") if d in image.dims)
        assert "t" not in image.dims
        images.append(image)
    image = xr.concat(images, dim="i")

    if agg_method == "mean":
        image = image.mean(dim="i")
    elif agg_method == "median":
        image = image.median(dim="i")
    elif agg_method == "min":
        image = image.min(dim="i")
    image = image.compute()  # image is now small so load into memory
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    image = _smooth_and_rescale(image, smooth, rescale)
    return image, keys, z_index
