"""SCALLOPS Utility Functions Module.

This module provides a collection of utility functions for various tasks such as image processing,
data manipulation, and statistical analysis. These functions are designed to support the SCALLOPS
framework and can be used independently or as part of larger workflows.

Authors:
- The SCALLOPS development team
"""

import itertools
import json
import logging
import os
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import chain, product
from pathlib import Path
from statistics import mode
from typing import Any, Optional, Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import skimage
from dask.array.overlap import ensure_minimum_chunksize
from dask.system import CPU_COUNT
from decorator import decorator
from kneed import KneeLocator
from skimage import restoration
from skimage.feature import hog
from skimage.measure import label
from skimage.morphology import dilation, flood
from skimage.transform import rescale, resize
from xarray import DataArray

logger = logging.getLogger("scallops")


# func(*blocks, left_pads, starts, stops, **kwargs) -> DataFrame of results for the blocks'
# main region
BlockFunc = Callable[..., pd.DataFrame]


def _broadcast_block(
    block: np.ndarray,
    broadcast_axes: Sequence[int],
    starts: Sequence[int],
    stops: Sequence[int],
    shape: Sequence[int],
    left_pads: Sequence[int],
    right_pads: Sequence[int],
    boundary: Mapping[int, str | int],
) -> np.ndarray:
    """Stretch a block's size-1 broadcast axes to the size of the overlapped block.

    :param block: Block with size 1 and no overlap along ``broadcast_axes``.
    :param broadcast_axes: Axes to broadcast.
    :param starts: Start of the block's main region along each axis.
    :param stops: Stop of the block's main region along each axis.
    :param shape: Broadcast shape of the input arrays.
    :param left_pads: Overlap before the main region along each axis.
    :param right_pads: Overlap after the main region along each axis.
    :param boundary: Boundary setting along each axis used to pad the input arrays.
    :return: Read-only view of the block, or a copy if constant padding had to be added.
    """
    if len(broadcast_axes) == 0:
        return block
    # Every boundary mode just repeats the single value along the axis, except a constant
    # boundary, which pads with the constant at the outer edges of the array
    target_shape = list(block.shape)
    pad_widths = {}
    for ax in broadcast_axes:
        constant = not isinstance(boundary[ax], str)
        left = 0 if constant and starts[ax] == 0 else left_pads[ax]
        right = 0 if constant and stops[ax] == shape[ax] else right_pads[ax]
        target_shape[ax] = left + stops[ax] - starts[ax] + right
        if left_pads[ax] - left + right_pads[ax] - right > 0:
            pad_widths[ax] = (left_pads[ax] - left, right_pads[ax] - right)
    block = np.broadcast_to(block, target_shape)
    for ax, pad_width in pad_widths.items():
        block = np.pad(
            block,
            [pad_width if i == ax else (0, 0) for i in range(block.ndim)],
            constant_values=boundary[ax],
        )
    return block


def _block_wrapper(
    func: BlockFunc,
    blocks: Sequence[np.ndarray],
    broadcast_axes: Sequence[Sequence[int]],
    chunk_location,
    boundary: Mapping[int, str | int],
    depth: Mapping[int, int],
    shape: Sequence[int],
    kwargs: Mapping[str, Any],
) -> pd.DataFrame:
    """Call ``func`` on one block of each array, passing the location of the blocks' main region.

    :param func: Function to apply to the blocks (see :func:`map_overlap_ragged`).
    :param blocks: The block from each input array as NumPy arrays, including their overlap.
    :param broadcast_axes: For each block, the axes along which it is broadcast.
    :param chunk_location: One slice per axis giving the block's main region in the broadcast
        shape (e.g. ``[slice(0, 3), slice(4, 8)]``).
    :param boundary: Boundary setting along each axis used to pad the input arrays.
    :param depth: Overlap depth along each axis.
    :param shape: Broadcast shape of the input arrays.
    :param kwargs: Extra keyword arguments to pass to ``func``.
    :return: The DataFrame returned by ``func``.
    """

    starts = [s.start for s in chunk_location]
    stops = [s.stop for s in chunk_location]

    # Strip the overlap so only the block's main rows remain
    # (Note: with boundary="none", edge blocks get no padding on their outer side)
    left_pads = [
        min(depth[ax], start) if boundary[ax] == "none" else depth[ax]
        for ax, start in enumerate(starts)
    ]
    right_pads = [
        min(depth[ax], shape[ax] - stop) if boundary[ax] == "none" else depth[ax]
        for ax, stop in enumerate(stops)
    ]
    blocks = [
        _broadcast_block(
            block, axes, starts, stops, shape, left_pads, right_pads, boundary
        )
        for block, axes in zip(blocks, broadcast_axes)
    ]
    return func(*blocks, left_pads, starts, stops, **kwargs)


def map_overlap_ragged(
    func: BlockFunc,
    *args: da.Array,
    meta: pd.DataFrame,
    depth: int | tuple[int, ...] | Mapping[int, int] | None = None,
    boundary: str | int | tuple | Mapping[int, str | int] | None = "reflect",
    **kwargs: Any,
) -> dd.DataFrame:
    """Apply a function to overlapping blocks of one or more arrays and collect the results in a
    DataFrame.

    Like :func:`dask.array.map_overlap`, each block is passed to ``func`` with ``depth`` extra
    elements from its neighbors along each axis. Unlike ``map_overlap``, ``func`` returns a
    DataFrame instead of an array, so each block can produce any number of rows (e.g. one row per
    object or per element). The results for all blocks are concatenated into one Dask DataFrame.

    :param func: Called as ``func(*blocks, left_pads, starts, stops, **kwargs)`` for each block,
        where:

        - ``blocks`` are the corresponding blocks of each input array as NumPy arrays, including
          their overlap. Every block has the full number of dimensions and the same shape;
          blocks of broadcast arrays are read-only.
        - ``left_pads[ax]`` is the number of overlap elements before the blocks' main region
          along axis ``ax``. The main region along that axis is
          ``block[left_pads[ax] : left_pads[ax] + stops[ax] - starts[ax]]``.
        - ``starts[ax]`` / ``stops[ax]`` give the main region's location in the input arrays
          (broadcast to a common shape) along axis ``ax`` (stop is exclusive). Add
          ``starts[ax]`` to an index within the main region to get its index in the input arrays.

        ``func`` must return a DataFrame with the same columns and dtypes as ``meta``. To avoid
        duplicate rows, only return results for the main region; the overlap is there to provide
        context.
    :param args: Input dask arrays. The arrays are broadcast against each other using NumPy rules
        (e.g. a ``(y, x)`` label image with a ``(t, c, y, x)`` image). Broadcasting happens per
        block, so broadcast arrays are never copied along the broadcast axes. The arrays are
        rechunked to common chunks, and chunks smaller than the overlap depth are merged with
        their neighbors, the same way :func:`dask.array.overlap.overlap` does.
    :param meta: Empty DataFrame with the columns and dtypes that ``func`` returns.
    :param depth: Number of overlap elements along each axis of the broadcast shape, as an int for
        all axes, a tuple with one value per axis, or a dict mapping axes to depths (missing axes
        get 0). Applied to every array.
    :param boundary: How to pad the outer edges of the arrays: ``"reflect"``, ``"periodic"``,
        ``"nearest"``, ``"none"`` (no padding, so blocks at the arrays' edges have no overlap on
        their outer side), or a constant value. Either one setting for all axes, a tuple with one
        value per axis, or a dict mapping axes to settings (missing axes get ``"none"``). Applied
        to every array.
    :param kwargs: Extra keyword arguments to pass to ``func``.
    :return: Dask DataFrame with one partition per block, containing the rows returned by ``func``.

    :example:

    .. code-block:: python

        def count_labels(image, labels, left_pads, starts, stops, threshold): ...


        ddf = map_overlap_ragged(
            count_labels, image, labels, meta=meta, depth={2: 16, 3: 16}, threshold=0.5
        )
    """
    if not callable(func):
        raise TypeError(
            f"First argument must be callable function, not {type(func).__name__}"
        )
    if len(args) == 0:
        raise ValueError("At least one array is required")
    if not all(isinstance(a, da.Array) for a in args):
        raise TypeError(
            f"All variadic arguments must be dask arrays, not {[type(a).__name__ for a in args]}"
        )
    shape = np.broadcast_shapes(*(a.shape for a in args))
    ndim = len(shape)
    depth = da.overlap.coerce_depth(ndim, depth)
    if any(isinstance(d, tuple) for d in depth.values()):
        raise NotImplementedError("Asymmetric overlap depth is not supported")
    boundary = da.overlap.coerce_boundary(ndim, boundary)
    # Add missing leading axes, and find the size-1 axes each array is broadcast along
    arrays = [a[(None,) * (ndim - a.ndim)] for a in args]
    broadcast_axes = [
        [ax for ax in range(ndim) if a.shape[ax] != shape[ax]] for a in arrays
    ]

    # Find common chunks so block b_id covers the same region in every array, by splitting
    # each axis at every chunk boundary of the arrays that are not broadcast along it
    chunks = []
    for ax in range(ndim):
        boundaries = sorted(
            {
                offset
                for a, axes in zip(arrays, broadcast_axes)
                if ax not in axes
                for offset in itertools.accumulate(a.chunks[ax])
            }
        )
        chunks.append(tuple(np.diff([0, *boundaries]).tolist()) or (shape[ax],))
    # Rechunk up front the same way da.overlap.overlap would (every chunk must be >= depth),
    # so the block locations below are computed from the chunks actually used
    chunks = tuple(
        ensure_minimum_chunksize(depth[ax], c) for ax, c in enumerate(chunks)
    )

    delayed_blocks = []
    for a, axes in zip(arrays, broadcast_axes):
        # Broadcast axes stay as one size-1 chunk with no overlap
        a = a.rechunk(tuple(-1 if ax in axes else c for ax, c in enumerate(chunks)))
        a_depth = {ax: 0 if ax in axes else depth[ax] for ax in range(ndim)}
        a_boundary = {ax: boundary[ax] for ax in range(ndim) if ax not in axes}
        delayed_blocks.append(
            da.overlap.overlap(a, depth=a_depth, boundary=a_boundary).to_delayed()
        )

    # Dask tracks array indexing schemas via x.chunks
    # Compute the global slice of each block in the original (non-overlapped) arrays
    offsets = [[0, *itertools.accumulate(c)] for c in chunks]

    delayed_dfs = []
    _block_wrapper_delayed = dask.delayed(_block_wrapper)
    for b_id in np.ndindex(*(len(c) for c in chunks)):
        # List of slices per dimension giving this block's main region
        chunk_slice = [
            slice(offsets[ax][i], offsets[ax][i + 1]) for ax, i in enumerate(b_id)
        ]
        d_df = _block_wrapper_delayed(
            func,
            [
                blocks[tuple(0 if ax in axes else i for ax, i in enumerate(b_id))]
                for blocks, axes in zip(delayed_blocks, broadcast_axes)
            ],
            broadcast_axes=broadcast_axes,
            chunk_location=chunk_slice,
            depth=depth,
            boundary=boundary,
            shape=shape,
            kwargs=kwargs,
        )
        delayed_dfs.append(d_df)

    return dd.from_delayed(delayed_dfs, meta=meta)


def _cpu_count():
    count = os.environ.get("SCALLOPS_CPU_COUNT")
    if count is not None:
        return int(count)

    return CPU_COUNT


def _tqdm_shim(iterator, *args, **kwargs):
    return iterator


def tqdm_func(progress: bool | str = True):
    progress_args = dict()
    tqdm_ = _tqdm_shim
    if progress != False:  # noqa: E712
        try:
            from tqdm import tqdm as tqdm_

            if isinstance(progress, str):
                progress_args["desc"] = progress
        except ImportError:
            pass
    return tqdm_, progress_args


def _fix_json(d):
    """Attempts to serialize and deserialize a dictionary to ensure it can be safely converted to
    JSON.

    This function first tries to use the faster `ujson` library from `pandas` if available.
    If `ujson` is not available, it defaults to the standard `json` library. If serialization
    fails due to an `OverflowError`, a warning is logged, and an empty dictionary is returned.

    :param d: The dictionary to be serialized and deserialized.
    :return: A deserialized version of the input dictionary or an empty dictionary
             if serialization fails.

    :raises OverflowError: If the data exceeds the size limits for JSON serialization.

    :example:

    .. code-block:: python

        data = {"key": "value", "large_number": 1e400}
        fixed_data = _fix_json(data)
        print(fixed_data)
        # Output: {}
    """
    import pandas._libs.json as ujson

    try:
        # Try to use ujson for faster performance
        dumps = ujson.ujson_dumps
    except AttributeError:
        # Fallback to standard JSON library's dumps if ujson is not available
        dumps = ujson.dumps

    try:
        # Serialize and deserialize the dictionary to ensure JSON compatibility
        d = json.loads(dumps(d, ensure_ascii=True))
    except OverflowError:
        # Log a warning if serialization fails
        logger.warning("Unable to serialize to JSON")
        d = {}

    return d


def is_dask_distributed():
    return "distributed" in dask.config.config


def high_pass_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """High pass filter typically used to remove background.

    :param image: Input image to filter.
    :param sigma: Standard deviation for Gaussian kernel.
    :return: Filtered image.
    """
    lowpass = skimage.filters.gaussian(image, sigma=sigma, preserve_range=True)
    highpass = image - lowpass
    highpass[lowpass > image] = 0
    return highpass


def gaussian_kernel(size: tuple[int, ...] = (3, 3), sigma: float = 0.5) -> np.ndarray:
    """Returns a gaussian kernel of specified size and standard deviation.

    The kernel is normalized to one.
    :param size: Kernel size
    :param sigma: Standard deviation
    :return: Gaussian kernel
    """
    m, n = [(ss - 1.0) / 2.0 for ss in size]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


@decorator
def applyIJ(
    f: Callable, arr: Union[np.ndarray, DataArray], *args: Any, **kwargs: Any
) -> np.ndarray:
    """Apply a function that expects 2D input to the trailing two dimensions of an array. The
    function must output an array whose shape depends only on the input shape.

    :param f: Function to be decorated.
    :param arr: Array being trimmed.
    :param args: Positional arguments to the function.
    :param kwargs: Keyword arguments to the function.
    :return: Reshaped array with trimmed dimensions.
    """
    if isinstance(arr, DataArray):
        arr = arr.values
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))
    # kwargs are not actually getting passed in?
    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]
    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


def match_size(
    image: np.ndarray, target: np.ndarray, order: Optional[str] = None
) -> np.ndarray:
    """Resize image to target without changing data range or type.

    :param image: Array with image data.
    :param target: Targeted array to match size with.
    :param order: The order of the spline interpolation, default is 0 if image.dtype is bool and 1
        otherwise. The order has to be in the range 0-5. See :skimage.transform.warp: for detail.
    :return: Resized version of image
    """
    return resize(image, target.shape, preserve_range=True, order=order).astype(
        image.dtype
    )


def mlcs(strings: Sequence[str]):
    """Return a long common subsequence of the strings. Uses a greedy algorithm, so the result is
    not necessarily the longest common subsequence.

    :param strings: list of strings to compare
    """
    if not isinstance(strings, list) or len(strings) < 1:
        return strings
    if len(strings) == 1:
        return strings[0]
    if not strings:
        raise ValueError("mlcs() argument is an empty sequence")
    strings = list(set(strings))  # deduplicate
    alphabet = set.intersection(*(set(s) for s in strings))

    indexes = {letter: [[] for _ in strings] for letter in alphabet}
    for i, s in enumerate(strings):
        for j, letter in enumerate(s):
            if letter in alphabet:
                indexes[letter][i].append(j)

    # pos[right] is current position of search in strings[right].
    pos = [len(s) for s in strings]

    # Generate candidate positions for next step in search.
    def _candidates():
        for letter, letter_indexes in indexes.items():
            distance, candidate = 0, []
            for ind, p in zip(letter_indexes, pos):
                right = bisect_right(ind, p - 1) - 1
                q = ind[right]
                if right < 0 or q > p - 1:
                    break
                candidate.append(q)
                distance += (p - q) ** 2
            else:
                yield distance, letter, candidate

    result = []
    while True:
        try:
            # Choose the closest candidate position, if any.
            _, letter, pos = min(_candidates())
        except ValueError:
            combo = ["--", "-_", "-.", "_-", "__", "_.", ".-", "._", ".."]
            res = "".join(reversed(result))
            for c in combo:
                res = res.replace(c, "_*_")
            return res
        result.append(letter)


def id_well_edge(data_: np.ndarray) -> np.ndarray:
    """ID the bright edge of a well and return the mask of it.

    :param data_: Image array with potentially an edge of a well to be identified.
    """
    eroded = data_.copy()
    for m in np.unique(eroded)[::-1][: max(eroded.shape)]:
        x, y = np.unravel_index(np.where(eroded.ravel() == m), eroded.shape)
        if x.size > 0 and y.size > 0:
            mask = flood(
                eroded, (x[0][0], y[0][0]), tolerance=np.power(10, int(np.log10(m)))
            )
            eroded[mask] = 0
    labeled = label(eroded == 0)
    counts = Counter(labeled.ravel().tolist())
    del counts[0]
    lab, ps = zip(*counts.most_common())
    new_mask = np.zeros_like(eroded)
    if len(ps) > 2:
        knee = KneeLocator(
            range(len(ps)), ps, curve="convex", direction="decreasing"
        ).knee
    else:
        knee = 1
    for lbl in lab[:knee]:
        temp = labeled == lbl
        if temp.sum() >= 15000:
            new_mask += temp
    return new_mask


def rm_edge(dt, rm_bkgr=True) -> tuple[np.ndarray, np.ndarray]:
    """Remove the well edge from an image.

    :param dt: Image array with potentially an edge of a well to be removed.
    :param rm_bkgr: Whether to use rolling_ball background removal.
    """
    d = dt.copy()
    new_mask = id_well_edge(d)
    if rm_bkgr:
        background = restoration.rolling_ball(d)
        d = d - background
    d[new_mask] = 0
    return d, new_mask


def id_edge_hog(stack: np.ndarray) -> np.ndarray:
    """Identify the edge of a well and return its mask using Histogram of Oriented Gradients (HOG).
    This is faster, but coarser than the :func:`id_well_edge`. It is is also *less* prone to
    overcorrecting.

    :param stack: Single channeled image array with potentially an edge of a well to be removed.
    """
    img = rescale(stack, 1 / 3)
    dim = img.shape[0]
    divisor = next(
        chain.from_iterable((i, dim // i) for i in range(9, 21) if dim % i == 0)
    )
    hog_fd = hog(
        img,
        pixels_per_cell=(divisor, divisor),
        cells_per_block=(1, 1),
        orientations=2,
        feature_vector=False,
    )
    divisor *= 3
    a, b = np.split(hog_fd, 2, axis=-1)
    hog_mask = rescale(np.isclose(a, b).squeeze(), divisor)
    mask = label(~hog_mask, connectivity=1)
    ravelled = mask.ravel()
    total_pixels = ravelled.shape[0]
    counts = Counter(ravelled)
    for lab, pix in counts.items():
        if lab == 0:
            continue
        elif pix / total_pixels <= 0.004:
            mask[mask == lab] = 0

    return mask


def curate_segmentation(seg_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Given segmentation data with potential edge of a well, curate the segmentation given the well
    mask.

    :param seg_data: Array with the segmentation data (often produced with the results of :func:`rm_edge`).
    :param mask: Mask with the information of the edge (often produced with the results of :func:`rm_edge`).
    """
    mask = dilation(mask).astype(bool)
    bool_mask = np.isin(seg_data, seg_data[mask])
    return np.where(bool_mask.ravel(), 0, seg_data.ravel()).reshape(bool_mask.shape)


class AssignTiles(object):
    """Class for assigning tile IDs to coordinates within squares based on chunk size.

    :param coordinates: Path or str or None
        Path to a CSV file containing 'X', 'Y', and 'Tile' columns or None. If None, it generates coordinates
        based on the maximum 'nuclei_x' and 'nuclei_y' values from the provided DataFrame and chunk size.
    :param chunksize: int
        Size of each square/tile.
    :param df: pd.DataFrame
        DataFrame containing the data with `coordinates_names` columns.
    :param xy_df_names: tuple of str (default: ('nuclei_x', 'nuclei_y'))
        Tuple specifying the names of the x and y columns in the DataFrame.

    :methods:
    - :meth:`find_tiles_within_squares(self) -> pd.Series`:
        Finds tile IDs for coordinates within squares based on chunk size.

    :example:

    .. code-block:: python
        import pandas as pd

        # Sample DataFrame
        data = {"nuclei_x": [1, 5, 10, 15, 20], "nuclei_y": [2, 6, 11, 16, 21]}
        df = pd.DataFrame(data)
        # Instantiate AssignTiles
        tile_assigner = AssignTiles(coordinates=None, chunksize=5, df=df)
        # Find tiles within squares
        result_tiles = tile_assigner.find_tiles_within_squares()
        print(result_tiles)
    """

    def __init__(
        self,
        coordinates: Path | str | None,
        df: pd.DataFrame,
        chunksize: int | None = None,
        xy_df_names: tuple[str, str] = ("nuclei_x", "nuclei_y"),
    ):
        self.coordinates_names = list(xy_df_names)
        self.data = df
        if chunksize is None:
            assert coordinates is not None, (
                "You need to provide either chunksize or coordinates"
            )
            self.coordinates = coordinates
            self.chunksize = None
        else:
            assert chunksize is not None, (
                "You need to provide either chunksize or coordinates"
            )
            self.chunksize = chunksize
            self.coordinates = coordinates

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, chunksize):
        if chunksize is None:
            chunksize = mode(
                np.abs(
                    np.diff(self.coordinates.loc[:, ["X", "Y"]].values, axis=0)
                    .round()
                    .ravel()
                    .astype(int)
                )
            )
        self._chunksize = chunksize

    @property
    def data(self) -> pd.DataFrame:
        """Property to access the data DataFrame.

        :return: pd.DataFrame
            DataFrame containing the data with `self.xy_df_names` columns.
        """
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame):
        """Setter for the data DataFrame. Checks if the required columns are present.

        :param data: pd.DataFrame
            DataFrame containing the data.

        :raises AssertionError:
            If `self.xy_df_names` columns are not present in the DataFrame.
        """
        assert set(self.coordinates_names).issubset(data.columns), (
            f"Data needs {self.coordinates_names} in columns"
        )
        self._data = data

    @property
    def coordinates(self) -> pd.DataFrame:
        """Property to access the coordinates DataFrame.

        :return: pd.DataFrame DataFrame containing 'X', 'Y', and 'Tile' columns representing
            coordinates and tile IDs.
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Path | str | None):
        """Setter for the coordinates DataFrame. If None, generates coordinates based on maximum
        values and chunk size.

        :param coordinates: Path or str or None Path to a CSV file containing 'X', 'Y', and 'Tile'
            columns or None.
        """
        if coordinates is None:
            assert self.chunksize is not None, (
                "You need to provide either chunksize or coordinates"
            )
            # Code to generate coordinates
            max_x = np.ceil(self.data[self.coordinates_names[0]].max()).astype(int)
            max_y = np.ceil(self.data[self.coordinates_names[1]].max()).astype(int)
            coords = product(
                range(0, max_x, self.chunksize), range(0, max_y, self.chunksize)
            )
            self._coordinates = pd.DataFrame(
                [{"X": x, "Y": y, "Tile": tile} for tile, (x, y) in enumerate(coords)]
            )
        else:
            columns_to_read = ["X", "Y", "Tile"]
            self._coordinates = pd.read_csv(coordinates, usecols=columns_to_read)[
                columns_to_read
            ]

    def find_tiles_within_squares(self) -> pd.Series:
        """Finds tile IDs for coordinates within squares based on chunk size.

        :return: Series containing the resulting tile IDs.
        """
        pos_coordinates = self.coordinates.loc[:, ["X", "Y"]].values
        row_coordinates = self.data[self.coordinates_names].values[:, None, :]
        square_max = pos_coordinates + self.chunksize
        within_bounds = np.all(
            np.logical_and(
                pos_coordinates <= row_coordinates, row_coordinates < square_max
            ),
            axis=2,
        )
        indices = np.where(within_bounds, self.coordinates.Tile, np.nan)
        # return a series so we can have an int32 with nan
        result_tiles = pd.Series(
            data=np.nanmax(indices, axis=1), name="tile", index=self.data.index
        ).astype("Int32")
        return result_tiles


def forceTCZYX(array: DataArray, requires_dims: str = "tczyx") -> DataArray:
    """Ensure that the given xarray DataArray has the specified dimensions.

    This function expands the dimensions of the input DataArray to match the specified dimensions,
    if they are not already present.

    :param array:
        The input DataArray.
    :param requires_dims:
        A string specifying the required dimensions. Each character represents a dimension. Defaults to 'tczyx'.
    :return:
        The DataArray with dimensions matching the specified requirements.

    :example:

    .. code-block:: python

        import xarray as xr
        import numpy as np
        from scallops.io import forceTCZYX

        # Create a sample DataArray with dimensions 'z', 'y', and 'x'
        data = np.random.rand(3, 10, 512, 512)
        dims = ("z", "y", "x")
        array = xr.DataArray(data, dims=dims)

        # Force the DataArray to have dimensions 'tczyx'
        array_tczyx = forceTCZYX(array)

        print(array_tczyx.dims)  # Output: ('t', 'c', 'z', 'y', 'x')
    """
    for i, d in enumerate(requires_dims):
        if d not in array.dims:
            array = array.expand_dims(dim=d, axis=i)
    return array


def _block_full(
    a: np.ndarray,
    b: np.ndarray | None = None,
    f: Callable[[np.ndarray, np.ndarray | None], float] = None,
    ndim: int = None,
) -> np.ndarray:
    return np.full((1,) * ndim, f(a, b))


def dask_chunk_stats(
    f: Callable[[np.ndarray, np.ndarray | None], float],
    a: da.Array,
    b: da.Array | None = None,
) -> da.Array:
    """Wrap a function that returns a float for use with dask.array.map_blocks.

    :param a: First image of dimensions (y,x) or (t,y,x) if second image is None
    :param b: Optional second image
    :param f: Function that computes statistics
    :return: Array with values returned `f`
    """

    f = partial(_block_full, f=f, ndim=a.ndim - 1 if b is None else a.ndim)
    return da.map_blocks(
        f,
        a,
        b,
        dtype=float,
        drop_axis=0 if b is None else None,
    ).squeeze()


def _write_img_size(file_list: list[str]):
    from scallops.io import _images2fov, _localize_path

    local_file_list = []
    cleanup_file_list = []
    for path in file_list:
        local_path = _localize_path(path)
        if local_path is not None:
            cleanup_file_list.append(local_path)
            local_file_list.append(local_path)
        else:
            local_file_list.append(path)
    sizes = _images2fov(local_file_list, dask=True).sizes
    for path in cleanup_file_list:
        os.remove(path)
    with open("img_size.txt", "wt") as f:
        for dim in ["t", "c", "z", "y", "x"]:
            s = sizes[dim] if dim in sizes else 0
            f.write(f"{s}")
            f.write("\n")


def _write_group_size(metadata: dict):
    n_tiles = len(metadata["file_metadata"])
    metadata_fields = [v for v in ("c", "z") if v in metadata["file_metadata"][0]]
    if len(metadata_fields) > 0:
        from scallops.cli.util import _group_src_attrs

        keys, channel_sources, filepaths = _group_src_attrs(
            metadata=metadata, metadata_fields=tuple(metadata_fields)
        )
        n_tiles = len(filepaths)
    with open("group_size.txt", "wt") as f:
        f.write(f"{n_tiles}")
        f.write("\n")


def _list_images_wdl(
    image_pattern: str,
    urls: list[str],
    groupby: list[str],
    subset: list[str],
    batch_size_str: str,
    save_group_size: bool = False,
    expected_cycles_str: int | None = None,
):
    """Used by WDL workflow to output info about images"""
    from scallops.io import _set_up_experiment

    batch_size = 1
    expected_cycles = None
    if expected_cycles_str != "":
        expected_cycles = int(expected_cycles_str)
    if batch_size_str != "":
        batch_size = int(batch_size_str)

    if len(subset) == 0 or (len(subset) == 1 and subset[0] == ""):
        subset = None
    if image_pattern != "":
        groupby = [g for g in groupby if "{" + g + "}" in image_pattern]
    exp_gen = _set_up_experiment(
        image_path=urls, files_pattern=image_pattern, group_by=groupby, subset=subset
    )
    # "groups.txt" is passed to --subset in cli
    # "groupby.txt" filtered groupby
    groupby_t = "t" in groupby
    t = []

    if not save_group_size:
        with open("group_size.txt", "wt") as f:
            f.write("0\n")

    first = True
    subset_ids = []
    for g, file_list, metadata in exp_gen:
        subset_ids.append(metadata["id"])

        if first:
            first = False
            if save_group_size:
                _write_group_size(metadata)
            if not groupby_t and "t" in metadata["file_metadata"][0]:
                t = [md["t"] for md in metadata["file_metadata"]]
                if expected_cycles is not None:
                    assert len(t) == expected_cycles

    with open("groups.txt", "wt") as f:  # ["plate1-A1", "plate1-A2", ...]
        for i in range(0, len(subset_ids), batch_size):
            selected = subset_ids[i : i + batch_size]
            f.write(" ".join(selected))
            f.write("\n")
    with open("groupby.txt", "wt") as f:
        for g in groupby:
            f.write(g)
            f.write("\n")

    with open("t.txt", "wt") as f:
        for val in t:
            f.write(str(val))
            f.write("\n")

    with open("groupby_pattern.txt", "wt") as f:
        first = True
        for g in groupby:
            if not first:
                f.write("-")
            first = False
            f.write("{")
            f.write(g)
            f.write("}")
