"""Cellpose Segmentation Submodule.

Provides functions for nuclei and cell segmentation using the Cellpose library.


Note:
    Before using this submodule, ensure that the Cellpose library is installed.

Authors:
    - The SCALLOPS development team
"""

import inspect
import logging
import math
from collections.abc import Sequence
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from filelock import FileLock

from scallops.segmentation.dask import segment_blocks
from scallops.segmentation.util import (
    _download_model,
    cyto_channel_summary,
)

logger = logging.getLogger("scallops")
try:
    cellpose_version = version("cellpose")
    cellpose_version = int(cellpose_version[0])

except PackageNotFoundError:
    logger.info("Please install cellpose")


def _process_block(
    x: np.ndarray,
    model: object,
    eval_kwargs: dict[str, Any],
) -> np.ndarray:
    """Process a block of the input image using the provided model.

    :param x: Input image block as a NumPy array.
    :param model: Model object with an `eval` method for segmentation.
    :param eval_kwargs: Keyword arguments for the model evaluation.
    :return: Labeled image block as a NumPy array of type int32.
    """

    if (x.max() - x.min()) == 0:
        return np.zeros(x.shape[-2:], dtype=np.int32)
    result = model.eval(x, **eval_kwargs)
    return result[0].astype(np.int32)


def _cellpose(
    image: xr.DataArray,
    channels: tuple[int, int],
    model: object,
    nuclei_channel: int = 0,
    cyto_channel: None | int | Sequence[int] = None,
    diameter: float | None = None,
    chunks: None | tuple[int, int] = None,
    **kwargs: dict[str, Any],
) -> np.ndarray | da.Array:
    """Perform segmentation using the Cellpose model on the input image.

    :param image: Input image as an `xarray.DataArray`.
    :param channels: Tuple specifying the channels to use for segmentation (e.g., (cyto, nuclei)).
    :param model: Cellpose model object.
    :param nuclei_channel: Channel index for nuclei (default is 0).
    :param cyto_channel: Channel index or sequence for cytoplasm (optional).
    :param diameter: Expected diameter of objects to detect.
    :param chunks: Chunk size for processing with Dask (optional).
    :param kwargs: Additional keyword arguments for model evaluation.
    :return: Segmentation labels.
    """
    image = image.squeeze()  # May drop t and z dimensions if present
    nuclei = image.isel(t=0, z=0, c=nuclei_channel, missing_dims="ignore").data
    cyto = None

    if cyto_channel is not None:
        cyto = cyto_channel_summary(
            image.isel(c=cyto_channel, z=0, missing_dims="ignore")
        ).data

    if isinstance(nuclei, da.Array):
        if chunks is None:
            chunks = nuclei.chunksize
        else:
            nuclei = nuclei.rechunk(chunks)

        if cyto is not None:
            cyto = cyto.rechunk(chunks)
            array = da.stack([cyto, nuclei])
        else:
            array = nuclei

    else:
        array = np.stack([cyto, nuclei]) if cyto is not None else nuclei
    depth = kwargs.pop("depth", None)
    eval_kwargs = dict(diameter=diameter)
    if cellpose_version < 4:
        eval_kwargs["channels"] = channels
    eval_kwargs.update(kwargs)

    if isinstance(array, da.Array):
        if depth is None:
            depth = math.ceil(diameter) + 1 if diameter is not None else 50

        return segment_blocks(
            array,
            seg_fn=_process_block,
            depth=depth,
            spatial_dims=2,
            model=model,
            eval_kwargs=eval_kwargs,
        )

    else:
        labels = model.eval(array, **eval_kwargs)[0]

    return labels


def _create_cellpose_model(**kwargs):
    import torch

    with FileLock(".scallops-cellpose.lock"):
        if cellpose_version < 4:
            from cellpose.models import Cellpose

            if kwargs["model_type"] == "cyto3":
                _download_model(
                    Path.home() / ".cellpose" / "models", ["cyto3", "size_cyto3.npy"]
                )
            elif kwargs["model_type"] == "nuclei":
                _download_model(
                    Path.home() / ".cellpose" / "models",
                    ["nucleitorch_0", "size_nucleitorch_0.npy"],
                )
            model_cls = Cellpose
        else:
            from cellpose.models import CellposeModel

            model_cls = CellposeModel
        # Split constructor arguments from eval arguments; unspecified constructor
        # arguments use the library defaults
        constructor_params = inspect.signature(model_cls).parameters
        cellpose_kwargs = dict(gpu=torch.cuda.is_available())
        for key in list(kwargs):
            if key in constructor_params:
                cellpose_kwargs[key] = kwargs.pop(key)
        with redirect_stdout(StringIO()):
            model = model_cls(**cellpose_kwargs)
        return model, kwargs


def segment_cells_cellpose(
    image: xr.DataArray,
    nuclei_channel: int = 0,
    cyto_channel: int | list[int] | None = None,
    diameter: float | None = None,
    chunks: tuple[int, int] | None = None,
    **kwargs: Any,
) -> np.ndarray | da.Array:
    """Segment cells using cellpose.

    :param image: Input image with dimensions (t, c, z, y, x).
    :param nuclei_channel: Index of nuclei channel (typically DAPI).
    :param cyto_channel: Index or list of indices for cyto channel(s). If not provided,
     all channels except for the nuclei channel are used.
    :param diameter: The diameter parameter for cellpose. Also used as the default
     chunk overlap (`depth`) when segmenting with chunks.
    :param chunks: Chunk size to perform segmentation using overlapping chunks.
    :param kwargs: Additional arguments to pass to model.eval or model constructor.
     Pass `depth` to set the chunk overlap.
    :return: Cell labels.
    """

    if cyto_channel is None:
        cyto_channel = np.delete(np.arange(image.sizes["c"]), nuclei_channel)
    if cellpose_version < 4 and "model_type" not in kwargs:
        kwargs["model_type"] = "cyto3"

    model, kwargs = _create_cellpose_model(**kwargs)

    return _cellpose(
        image=image,
        channels=(1, 2),
        model=model,
        nuclei_channel=nuclei_channel,
        cyto_channel=cyto_channel,
        diameter=diameter,
        chunks=chunks,
        **kwargs,
    )


def segment_nuclei_cellpose(
    image: xr.DataArray,
    nuclei_channel: int = 0,
    diameter: float | None = None,
    chunks: tuple[int, int] | None = None,
    **kwargs: Any,
) -> np.ndarray | da.Array:
    """Segment nuclei using cellpose.

    :param image: Input image with dimensions (t, c, z, y, x).
    :param nuclei_channel: Index of nuclei channel (typically DAPI).
    :param diameter: The diameter parameter for cellpose. Also used as the default
     chunk overlap (`depth`) when segmenting with chunks.
    :param chunks: Chunk size to perform segmentation using overlapping chunks.
    :param kwargs: Additional arguments to pass to model.eval or model constructor.
     Pass `depth` to set the chunk overlap.
    :return: Nuclei labels.
    """

    if cellpose_version < 4 and "model_type" not in kwargs:
        kwargs["model_type"] = "nuclei"
    model, kwargs = _create_cellpose_model(**kwargs)
    return _cellpose(
        image=image,
        channels=(0, 0),
        model=model,
        nuclei_channel=nuclei_channel,
        cyto_channel=None,
        diameter=diameter,
        chunks=chunks,
        **kwargs,
    )
