"""Cellpose Segmentation Submodule.

Provides functions for nuclei and cell segmentation using the Cellpose library.


Note:
    Before using this submodule, ensure that the Cellpose library is installed.

Authors:
    - The SCALLOPS development team
"""

import logging
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

from scallops.dask_relabeling.relabel import image2labels
from scallops.segmentation.util import (
    _download_model,
    cyto_channel_summary,
    dask_relabel_sequential,
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
    diameter: float,
    eval_kwargs: dict[str, Any],
) -> np.ndarray:
    """Process a block of the input image using the provided model.

    :param x: Input image block as a NumPy array.
    :param model: Model object with an `eval` method for segmentation.
    :param diameter: Diameter of objects to detect.
    :param eval_kwargs: Additional keyword arguments for the model evaluation.
    :return: Labeled image block as a NumPy array of type int32.
    """

    if (x.max() - x.min()) == 0:
        return np.zeros(x.shape[-2:], dtype=np.int32)
    result = model.eval(x, diameter=diameter, **eval_kwargs)
    return result[0].astype(np.int32)


def _cellpose(
    image: xr.DataArray,
    channels: tuple[int, int],
    model: object,
    nuclei_channel: int = 0,
    cyto_channel: None | int | Sequence[int] = None,
    diameter: None | float = None,
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

    if isinstance(array, da.Array):
        depth = kwargs.pop("depth", 30)
        if diameter is not None and depth is None:
            depth = np.ceil(diameter).astype(np.int64)
        segmentation_fn_kwargs = {
            "diameter": diameter,
            "eval_kwargs": kwargs,
            "model": model,
        }
        if cellpose_version < 4:
            kwargs["channels"] = channels
        labels = image2labels(
            array,
            seg_fn=_process_block,
            overlaps=[depth, depth],
            spatial_dims=2,
            segmentation_fn_kwargs=segmentation_fn_kwargs,
        )
        return dask_relabel_sequential(labels)
    else:
        model_args = dict()
        if cellpose_version < 4:
            model_args["channels"] = channels
        model_args.update(kwargs)
        labels = model.eval(array, diameter=diameter, **model_args)[0]

    return labels


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
    :param diameter: The diameter parameter for cellpose.
    :param chunks: Chunk size to perform segmentation using overlapping chunks.
    :param kwargs: Additional arguments to pass to model.eval.
    :return: Nuclei labels.
    """
    if cellpose_version < 4:
        from cellpose.models import Cellpose
    else:
        from cellpose.models import CellposeModel
    import torch

    with FileLock(".scallops-cellpose.lock"):
        if cellpose_version < 4:
            _download_model(
                Path.home() / ".cellpose" / "models",
                ["nucleitorch_0", "size_nucleitorch_0.npy"],
            )
        with redirect_stdout(StringIO()):
            model = (
                Cellpose(model_type="nuclei", gpu=torch.cuda.is_available())
                if cellpose_version < 4
                else CellposeModel(gpu=torch.cuda.is_available())
            )

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
    :param diameter: The diameter parameter for cellpose.
    :param chunks: Chunk size to perform segmentation using overlapping chunks.
    :param kwargs: Additional arguments to pass to model.eval.
    :return: Cell labels.
    """

    if cellpose_version < 4:
        from cellpose.models import Cellpose
    else:
        from cellpose.models import CellposeModel
    import torch

    with FileLock(".scallops-cellpose.lock"):
        if cellpose_version < 4:
            _download_model(
                Path.home() / ".cellpose" / "models", ["cyto3", "size_cyto3.npy"]
            )
        with redirect_stdout(StringIO()):
            model = (
                Cellpose(model_type="cyto3", gpu=torch.cuda.is_available())
                if cellpose_version < 4
                else CellposeModel(gpu=torch.cuda.is_available())
            )
    if cyto_channel is None:
        cyto_channel = np.delete(np.arange(image.sizes["c"]), nuclei_channel)
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
