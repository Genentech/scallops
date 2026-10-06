from collections.abc import Mapping

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from scallops.utils import map_overlap_ragged


def example_func(block, left_pads, starts, stops):
    data = {}
    mean_block = block / block.mean()
    core = block[
        tuple(
            slice(pad, pad + (stop - start))
            for pad, start, stop in zip(left_pads, starts, stops)
        )
    ]
    core2 = mean_block[
        tuple(
            slice(pad, pad + (stop - start))
            for pad, start, stop in zip(left_pads, starts, stops)
        )
    ]

    # Global index of every element along each axis
    indices = np.indices(core.shape).reshape(core.ndim, -1)

    for ax, (start, stop) in enumerate(zip(starts, stops)):
        data[f"chunk_start_{ax}"] = start
        data[f"chunk_end_{ax}"] = stop
    for ax, start in enumerate(starts):
        data[f"mapped_index_{ax}"] = indices[ax] + start
    data["val"] = core.ravel()
    data["val2"] = core2.ravel()
    return pd.DataFrame(data)


def _check_map_overlap_ragged(x: da.Array, depth: Mapping[int, int], boundary: str):
    meta = {}
    for ax in range(x.ndim):
        meta[f"chunk_start_{ax}"] = pd.Series(dtype="int64")
        meta[f"chunk_end_{ax}"] = pd.Series(dtype="int64")
    for ax in range(x.ndim):
        meta[f"mapped_index_{ax}"] = pd.Series(dtype="int64")
    meta["val"] = pd.Series(dtype=x.dtype)
    meta["val2"] = pd.Series(dtype="float64")

    ddf = map_overlap_ragged(
        example_func, array=x, depth=depth, boundary=boundary, meta=pd.DataFrame(meta)
    )
    df = ddf.compute()

    # Check every element of x appears exactly once, with the value at its mapped index
    x_np = x.compute()
    mapped = tuple(df[f"mapped_index_{ax}"].to_numpy() for ax in range(x.ndim))
    assert len(df) == x_np.size
    assert len(set(zip(*mapped))) == x_np.size
    assert (x_np[mapped] == df["val"].to_numpy()).all()

    # Compare with da.map_overlap
    x2 = da.map_overlap(
        lambda block: block / block.mean(), x, depth=depth, boundary=boundary
    )
    assembled = np.empty(x.shape, dtype=df["val2"].dtype)
    assembled[mapped] = df["val2"].to_numpy()
    assert np.array_equal(assembled, x2.compute())


# Every value is unique (its flat index), so mapping mistakes show up in "val"
shape = (10, 4, 100, 100)
BOUNDARIES = ["reflect", "periodic", "nearest", "none"]


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_map_overlap_ragged_no_rechunk(boundary):
    # Every chunk is at least as large as the depth, so da.overlap.overlap keeps the chunks as is
    x = da.arange(np.prod(shape)).reshape(shape).rechunk((-1, -1, 10, 10))
    depth = {0: 0, 1: 0, 2: 1, 3: 1}
    da.overlap.overlap(
        x, depth=depth, boundary=boundary, allow_rechunk=False
    )  # Raises if a rechunk is needed
    _check_map_overlap_ragged(x, depth, boundary)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_map_overlap_ragged_rechunk(boundary):
    # The last chunk along axis 2 (4) is smaller than the depth (5), so da.overlap.overlap rechunks
    x = da.arange(np.prod(shape)).reshape(shape).rechunk((-1, -1, (48, 48, 4), 10))
    depth = {0: 0, 1: 0, 2: 5, 3: 1}
    with pytest.raises(
        ValueError, match="Overlap depth is larger than smallest chunksize"
    ):
        da.overlap.overlap(x, depth=depth, boundary=boundary, allow_rechunk=False)
    _check_map_overlap_ragged(x, depth, boundary)
