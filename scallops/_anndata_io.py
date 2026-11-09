"""scallops' anndata writers, registered in place of anndata's when imported.

`scallops/__init__.py` imports this module, so it should import only anndata and
modules that anndata itself imports.
"""

import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np
import zarr
from anndata._core.views import DaskArrayView
from anndata._io.specs import _REGISTRY
from anndata._io.specs.methods import zarr_v3_compressor_compat
from anndata._io.specs.registry import IOSpec, Writer
from anndata.compat import DaskArray


@_REGISTRY.register_write(zarr.Group, DaskArrayView, IOSpec("array", "0.2.0"))
@_REGISTRY.register_write(zarr.Group, DaskArray, IOSpec("array", "0.2.0"))
def write_basic_dask_dask_dense(
    f: zarr.Group | h5py.Group,
    k: str,
    elem: DaskArray,
    *,
    _writer: Writer,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
):
    # Removes hard-coded scheduler, respects dask chunk sizes, sets fill value, allows rectilinear chunks if array.rectilinear_chunks is set
    import dask.array as da

    dataset_kwargs = dict(dataset_kwargs)
    fill_value = "fill_value" if not isinstance(f, h5py.Group) else "fillvalue"
    if fill_value not in dataset_kwargs and np.isdtype(elem.dtype, "real floating"):
        dataset_kwargs[fill_value] = np.nan
    if (
        not isinstance(f, h5py.Group)
        and "chunks" not in dataset_kwargs
        and "shards" not in dataset_kwargs
    ):
        # logic based on code in da.to_zarr
        if zarr.config.get("array.rectilinear_chunks", False):
            dataset_kwargs["chunks"] = elem.chunks
        else:
            if not da.core._check_regular_chunks(elem.chunks):
                warnings.warn(
                    "The array uses irregular chunk sizes. Rechunking to regular (uniform) chunks "
                    "to ensure the data can be written safely. If you want to avoid this automatic "
                    "rechunking, manually rechunk the array so that all chunks, except possibly the "
                    "final chunk, in each dimension—have the same size (e.g., arr = arr.rechunk(...)).",
                    UserWarning,
                    stacklevel=2,
                )

                elem = elem.rechunk(tuple(map(max, elem.chunks)))
            # zarr requires min chunk size 1

            min_chunk_value = 0 if isinstance(f, h5py.Group) else 1
            dataset_kwargs["chunks"] = tuple(
                max(c[0], min_chunk_value) for c in elem.chunks
            )

    if isinstance(f, h5py.Group):
        g = f.require_dataset(k, shape=elem.shape, dtype=elem.dtype, **dataset_kwargs)
    else:
        dataset_kwargs = zarr_v3_compressor_compat(dataset_kwargs)
        g = f.require_array(k, shape=elem.shape, dtype=elem.dtype, **dataset_kwargs)

    if isinstance(f, h5py.Group):
        da.store(elem, g, scheduler="threads")
    else:
        da.store(elem, g)
