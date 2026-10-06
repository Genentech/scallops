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
        example_func, x, meta=pd.DataFrame(meta), depth=depth, boundary=boundary
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


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_map_overlap_ragged_multiple_arrays(boundary):
    # Arrays with different chunks are aligned so each call gets the same region of both
    x = da.arange(np.prod(shape)).reshape(shape).rechunk((-1, -1, (48, 48, 4), 10))
    y = (-x).rechunk((-1, -1, 25, 30))
    depth = {0: 0, 1: 0, 2: 5, 3: 1}

    def func(x_block, y_block, left_pads, starts, stops):
        assert x_block.shape == y_block.shape
        core = tuple(
            slice(pad, pad + (stop - start))
            for pad, start, stop in zip(left_pads, starts, stops)
        )
        indices = np.indices(x_block[core].shape).reshape(x_block.ndim, -1)
        data = {
            f"mapped_index_{ax}": indices[ax] + start for ax, start in enumerate(starts)
        }
        data["x"] = x_block[core].ravel()
        data["y"] = y_block[core].ravel()
        data["sum_with_overlap"] = (x_block + y_block).sum()
        return pd.DataFrame(data)

    meta = {f"mapped_index_{ax}": pd.Series(dtype="int64") for ax in range(x.ndim)}
    meta["x"] = pd.Series(dtype=x.dtype)
    meta["y"] = pd.Series(dtype=y.dtype)
    meta["sum_with_overlap"] = pd.Series(dtype=x.dtype)
    df = map_overlap_ragged(
        func, x, y, meta=pd.DataFrame(meta), depth=depth, boundary=boundary
    ).compute()

    x_np = x.compute()
    mapped = tuple(df[f"mapped_index_{ax}"].to_numpy() for ax in range(x.ndim))
    assert len(df) == x_np.size
    assert len(set(zip(*mapped))) == x_np.size
    assert (x_np[mapped] == df["x"].to_numpy()).all()
    assert (-x_np[mapped] == df["y"].to_numpy()).all()
    # y == -x everywhere, so the overlap regions line up only if the sums are 0
    assert (df["sum_with_overlap"] == 0).all()


def _sum_func(*args):
    # One row per element of the main region, with each block's value and its sum including
    # the overlap (so overlap mistakes show up too)
    *blocks, left_pads, starts, stops = args
    assert all(b.shape == blocks[0].shape for b in blocks)
    core = tuple(
        slice(pad, pad + (stop - start))
        for pad, start, stop in zip(left_pads, starts, stops)
    )
    indices = np.indices(blocks[0][core].shape).reshape(blocks[0].ndim, -1)
    data = {
        f"mapped_index_{ax}": indices[ax] + start for ax, start in enumerate(starts)
    }
    for i, b in enumerate(blocks):
        data[f"val_{i}"] = b[core].ravel().astype("float64")
        data[f"sum_{i}"] = float(b.sum())
    return pd.DataFrame(data)


def _run_sum_func(arrays, depth, boundary):
    ndim = max(a.ndim for a in arrays)
    meta = {f"mapped_index_{ax}": pd.Series(dtype="int64") for ax in range(ndim)}
    for i in range(len(arrays)):
        meta[f"val_{i}"] = pd.Series(dtype="float64")
        meta[f"sum_{i}"] = pd.Series(dtype="float64")
    df = map_overlap_ragged(
        _sum_func, *arrays, meta=pd.DataFrame(meta), depth=depth, boundary=boundary
    ).compute()
    return df.sort_values(list(df.columns[:ndim]), ignore_index=True)


@pytest.mark.parametrize("boundary", [*BOUNDARIES, 7])
@pytest.mark.parametrize(
    "y_shape,y_chunks",
    [
        ((100, 100), (30, 25)),  # Missing leading axes
        ((1, 4, 100, 100), (1, 2, 50, 50)),  # Size-1 axis
        ((10, 1, 1, 100), (5, 1, 1, 100)),  # Size-1 axes with overlap depth
    ],
)
def test_map_overlap_ragged_broadcast(boundary, y_shape, y_chunks):
    x = da.arange(np.prod(shape)).reshape(shape).rechunk((-1, -1, (48, 48, 4), 10))
    y = da.arange(np.prod(y_shape)).reshape(y_shape).rechunk(y_chunks) * 0.5
    depth = {0: 0, 1: 0, 2: 5, 3: 1}

    df = _run_sum_func([x, y], depth, boundary)
    # da.broadcast_arrays also unifies chunks, so the blocks (and overlap sums) match
    x_b, y_b = da.broadcast_arrays(x, y)
    expected = _run_sum_func([x_b, y_b], depth, boundary)
    pd.testing.assert_frame_equal(df, expected)
    assert len(df) == x.size


def test_map_overlap_ragged_broadcast_lower_ndim_first():
    # The broadcast shape doesn't depend on which array comes first
    x = da.arange(np.prod(shape)).reshape(shape).rechunk((-1, -1, 25, 25))
    y = da.arange(100 * 100).reshape((100, 100)).rechunk(40)
    depth = {0: 0, 1: 0, 2: 3, 3: 3}
    df = _run_sum_func([y, x], depth, "reflect")
    y_b, x_b = da.broadcast_arrays(y, x)
    expected = _run_sum_func([y_b, x_b], depth, "reflect")
    pd.testing.assert_frame_equal(df, expected)


def test_map_overlap_ragged_shape_mismatch():
    x = da.zeros((10, 10), chunks=5)
    y = da.zeros((10, 9), chunks=5)
    with pytest.raises(ValueError, match="broadcast"):
        map_overlap_ragged(
            lambda *args: pd.DataFrame(), x, y, meta=pd.DataFrame(), depth={0: 1, 1: 1}
        )


def test_map_overlap_ragged_kwargs():
    # Extra keyword arguments are passed to func; depth and boundary accept the same forms
    # as da.map_overlap
    x = da.arange(100 * 100).reshape((100, 100)).rechunk(30)

    def func(block, left_pads, starts, stops, scale, offset=0):
        core = block[
            tuple(slice(p, p + (e - s)) for p, s, e in zip(left_pads, starts, stops))
        ]
        return pd.DataFrame({"val": core.ravel() * scale + offset})

    meta = pd.DataFrame({"val": pd.Series(dtype="float64")})
    df = map_overlap_ragged(func, x, meta=meta, depth=2, scale=0.5, offset=1).compute()
    np.testing.assert_array_equal(
        np.sort(df["val"].to_numpy()), np.arange(100 * 100) * 0.5 + 1
    )


@pytest.mark.parametrize(
    "depth,boundary",
    [
        (1, {2: "reflect", 3: 7}),  # Per-axis boundary, missing axes are "none"
        ((0, 0, 5, 1), ("none", "none", "periodic", "nearest")),
        ({2: 5, 3: 1}, None),  # Missing depth axes are 0, boundary None is "none"
    ],
)
def test_map_overlap_ragged_depth_boundary_forms(depth, boundary):
    x = da.arange(np.prod(shape)).reshape(shape).rechunk((-1, -1, (48, 48, 4), 10))
    y = da.arange(100 * 100).reshape((100, 100)).rechunk(25) * 0.5
    df = _run_sum_func([x, y], depth, boundary)
    x_b, y_b = da.broadcast_arrays(x, y)
    expected = _run_sum_func([x_b, y_b], depth, boundary)
    pd.testing.assert_frame_equal(df, expected)

    # Compare each block's mean (including overlap) with da.map_overlap, which supports the
    # same depth and boundary forms and merges small chunks the same way
    def block_mean(block, left_pads, starts, stops):
        return pd.DataFrame(
            {
                **{f"start_{ax}": [s] for ax, s in enumerate(starts)},
                "mean": [block.mean()],
            }
        )

    meta = {f"start_{ax}": pd.Series(dtype="int64") for ax in range(x.ndim)}
    meta["mean"] = pd.Series(dtype="float64")
    df = map_overlap_ragged(
        block_mean, x, meta=pd.DataFrame(meta), depth=depth, boundary=boundary
    ).compute()
    x2 = da.map_overlap(
        lambda block: np.full(block.shape, block.mean()),
        x,
        depth=depth,
        boundary=boundary,
        dtype="float64",
    ).compute()
    starts = tuple(df[f"start_{ax}"].to_numpy() for ax in range(x.ndim))
    np.testing.assert_array_equal(x2[starts], df["mean"].to_numpy())


def test_map_overlap_ragged_type_errors():
    meta = pd.DataFrame()
    with pytest.raises(TypeError, match="callable"):
        map_overlap_ragged(da.zeros(10, chunks=5), meta=meta)
    with pytest.raises(TypeError, match="dask arrays"):
        map_overlap_ragged(lambda *args: meta, np.zeros(10), meta=meta)
    with pytest.raises(ValueError, match="At least one array"):
        map_overlap_ragged(lambda *args: meta, meta=meta)
