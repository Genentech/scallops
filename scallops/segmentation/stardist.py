"""Nuclei segmentation using stardist.

Provides functions for segmenting nuclei and cells in an image using the StarDist model.


Authors:
    - The SCALLOPS development team
"""

import logging
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from filelock import FileLock

from scallops.segmentation.util import _download_model

logger = logging.getLogger("scallops")


def _normalize(
    x: np.ndarray | da.Array,
    pmin: float = 3,
    pmax: float = 99.8,
    axis: int | tuple[int, ...] | None = None,
    clip: bool = False,
    eps: float = 1e-20,
    dtype: np.dtype | None = np.float32,
) -> np.ndarray:
    """Perform percentile-based image normalization.

    :param x: Input image as a NumPy array or Dask array.
    :param pmin: Minimum percentile for image normalization.
    :param pmax: Maximum percentile for image normalization.
    :param axis: Axis or axes along which to compute percentiles.
    :param clip: If True, clip the normalized values to the range [0, 1].
    :param eps: Small value added to the denominator for numerical stability.
    :param dtype: Data type of the output image.
    :return: Normalized image.
    """
    inplace = False
    if isinstance(
        x, da.Array
    ):  # loading one channel in one well does not use too much memory
        x = x.compute()
        inplace = True

    mi, ma = np.percentile(x, [pmin, pmax], axis=axis, keepdims=True)
    return _normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype, inplace=inplace)


def _normalize_mi_ma(
    x: np.ndarray,
    mi: np.ndarray,
    ma: np.ndarray,
    clip: bool = False,
    eps: float = 1e-20,
    dtype: np.dtype | None = np.float32,
    inplace: bool = False,
) -> np.ndarray:
    """Normalize image based on minimum and maximum values.

    :param x: Input image as a NumPy array.
    :param mi: Minimum values for normalization.
    :param ma: Maximum values for normalization.
    :param clip: If True, clip the normalized values to the range [0, 1].
    :param eps: Small value added to the denominator for numerical stability.
    :param dtype: Data type of the output image.
    :return: Normalized image.
    """

    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        if inplace:
            x -= mi
            x /= ma - mi + eps
        else:
            x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def segment_nuclei_stardist(
    image: xr.DataArray,
    nuclei_channel: int = 0,
    pmin: float = 3,
    pmax: float = 99.8,
    clip: bool = False,
    chunks: tuple[int, int] | None = (4096, 4096),
    depth: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Segment nuclei using StarDist.

    :param image: Input image with the dimensions (t, c, z, y, x).
    :param nuclei_channel: Index of the nuclei channel (typically DAPI).
    :param pmin: Minimum percentile for image normalization.
    :param pmax: Maximum percentile for image normalization.
    :param clip: Whether to clip normalized values.
    :param chunks: Chunk size to perform segmentation using overlapping chunks.
    :param depth: The number of elements that each block should share with its neighbors when using
        dask
    :param kwargs: Additional keyword arguments to pass to StarDist predict_instances.
    :return: Nuclei labels as a NumPy array.
    """

    n_tiles = None
    data = image.isel(t=0, c=nuclei_channel, z=0, missing_dims="ignore").squeeze().data
    if isinstance(data, da.Array) and chunks is None:
        chunks = data.chunksize

    with redirect_stdout(StringIO()):  # hide loading message
        from stardist.models import StarDist2D

        # prevent error when multiple processes load model using hdf5 simultaneously
        with FileLock(".scallops-stardist.lock"):
            local_model_dest = (
                Path.home() / ".keras" / "models" / "StarDist2D" / "2D_versatile_fluo"
            )

            _download_model(local_model_dest, "2D_versatile_fluo.zip")
            model = StarDist2D.from_pretrained("2D_versatile_fluo")

    if chunks is not None:
        n_tiles = (
            np.ceil((image.sizes["y"]) / chunks[0]),
            np.ceil((image.sizes["x"]) / chunks[1]),
        )
    if depth is not None:
        model._tile_overlap = [(depth, depth), (depth, depth)]
    elif hasattr(model, "_tile_overlap"):
        delattr(model, "_tile_overlap")

    return model.predict_instances(
        _normalize(
            data,
            pmin=pmin,
            pmax=pmax,
            clip=clip,
        ),
        n_tiles=n_tiles,
        **kwargs,
    )[0]
