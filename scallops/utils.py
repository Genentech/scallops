"""SCALLOPS Utility Functions Module.

This module provides a collection of utility functions for various tasks such as image processing,
data manipulation, and statistical analysis. These functions are designed to support the SCALLOPS
framework and can be used independently or as part of larger workflows.

Authors:
- The SCALLOPS development team
"""

import json
import logging
import os
import uuid
import warnings
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Sequence
from functools import partial
from itertools import chain, product
from pathlib import Path
from statistics import mode
from typing import Any, Optional, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import skimage
from dask import is_dask_collection
from dask.array.core import (
    getter,
    getter_nofancy,
    graph_from_arraylike,
    normalize_chunks,
    slices_from_chunks,
)
from dask.system import CPU_COUNT
from dask.tokenize import tokenize
from dask.utils import SerializableLock
from decorator import decorator
from kneed import KneeLocator
from ome_zarr.writer import write_image
from skimage import restoration
from skimage.feature import hog
from skimage.measure import label
from skimage.morphology import dilation, flood
from skimage.transform import rescale, resize
from xarray import DataArray
from xarray import concat as xr_concat

logger = logging.getLogger("scallops")


def _cpu_count():
    count = os.environ.get("SCALLOPS_CPU_COUNT")
    if count is not None:
        return int(count)

    return CPU_COUNT


def _tqdm_shim(iterator, *args, **kwargs):
    return iterator


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


def concatenate_arrays(
    arrays: Sequence[DataArray],
    suppl_attr: Optional[Sequence[dict[str, np.ndarray]]] = None,
    dim: str = "Image",
    swap: Optional[dict[str, str]] = None,
) -> DataArray:
    """Concatenate a list of arrays over the new dimension `dim`, and setting new attributes with
    the option of extra images in dim.

    :param swap: Swap dimensions based on the mapping AFTER concatenation
    :param arrays: List of DataArrays with the images to be concatenated
    :param suppl_attr: Supplemental attributes as dictionary of numpy arrays
    :param dim: Name of the dimension to create the concatenation by
    :return: A :DataArray: with the concatenation
    """
    assert isinstance(arrays, list), (
        "A list of DataArrays to concatenate must be supplied"
    )
    new_attr = {i: x.attrs for i, x in enumerate(arrays)}
    if suppl_attr is not None:
        assert len(suppl_attr) == len(arrays), (
            "arrays and suppl_coordinates must match!!"
        )
        for i, x in enumerate(suppl_attr):
            new_attr[i]["supplementary"] = x
    new_arr = xr_concat(arrays, dim=dim, combine_attrs="drop", coords="all")
    if swap is not None:
        new_arr = new_arr.swap_dims(swap)
    return new_arr.assign_attrs(new_attr)


def grid_search(
    function: Callable,
    parameters_ranges: dict[str, Union[range, Sequence]],
    cpus: int = -1,
    to_zarr: Optional[str] = None,
    **kwargs,
) -> dict[Any, Union[Union[str, Path], np.ndarray]]:
    """Generate a line or grid search of the `parameters` of `function`

    :param function: Function to be evaluated.
    :param parameters_ranges: dictionary of ranges (or lists) with the parameter space.
    :param cpus: Number of cpus to use. By default, is -1 which means all available cpus.
    :param to_zarr: If not None, dump the function's results to specified path.
    :param kwargs: All argument as keyword arguments for `function`. Positional arguments can be passed as keyword
                   arguments with their name.
    :return: A dictionary with the parameter combination as keys and the function results as values.
    """
    from joblib import Parallel
    from joblib import delayed as joblib_delayed

    pairs = [product([k], v) for k, v in parameters_ranges.items()]
    grid = list(product(*pairs))
    keys = ["-".join("{0}_{1}".format(*x) for x in y) for y in grid]
    results = Parallel(n_jobs=cpus)(
        joblib_delayed(function)(**dict(parameter), **kwargs) for parameter in grid
    )
    res_dict = dict(zip(keys, results))
    if to_zarr is not None:
        from scallops.zarr_io import open_ome_zarr

        ome_zarr_root = open_ome_zarr(to_zarr, mode="w")
        image_keys = []
        for thresh, image in res_dict:
            name = str(thresh)
            image_keys.append(name)
            write_image(
                image=image,
                group=ome_zarr_root.create_group(name),
                scaler=None,
                axes=["t", "c", "z", "y", "x"],
                storage_options=dict(dimension_separator="/"),
            )
        ome_zarr_root.create_group("OME").attrs["series"] = image_keys
    return res_dict


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


def _dask_from_array_no_copy(
    x,
    chunks="auto",
    name=None,
    lock=False,
    asarray=False,
    fancy=True,
    getitem=None,
    meta=None,
    inline_array=False,
):
    """Create dask array from something that looks like an array without copying."""

    if isinstance(x, da.Array):
        raise ValueError(
            "Array is already a dask array. Use 'asarray' or 'rechunk' instead."
        )

    elif is_dask_collection(x):
        warnings.warn(
            "Passing an object to dask.array.from_array which is already a "
            "Dask collection. This can lead to unexpected behavior."
        )

    if isinstance(x, (list, tuple, memoryview) + np.ScalarType):
        x = np.array(x)

    # if is_arraylike(x) and hasattr(x, "copy"):
    #     x = x.copy()

    if asarray is None:
        asarray = not hasattr(x, "__array_function__")

    previous_chunks = getattr(x, "chunks", None)

    chunks = normalize_chunks(
        chunks, x.shape, dtype=x.dtype, previous_chunks=previous_chunks
    )

    if name in (None, True):
        token = tokenize(x, chunks, lock, asarray, fancy, getitem, inline_array)
        name = name or "array-" + token
    elif name is False:
        name = "array-" + str(uuid.uuid1())

    if lock is True:
        lock = SerializableLock()

    is_ndarray = type(x) in (np.ndarray, np.ma.core.MaskedArray)
    is_single_block = all(len(c) == 1 for c in chunks)
    # Always use the getter for h5py etc. Not using isinstance(x, np.ndarray)
    # because np.matrix is a subclass of np.ndarray.
    if is_ndarray and not is_single_block and not lock:
        # eagerly slice numpy arrays to prevent memory blowup
        # GH5367, GH5601
        slices = slices_from_chunks(chunks)
        keys = product([name], *(range(len(bds)) for bds in chunks))
        values = [x[slc] for slc in slices]
        dsk = dict(zip(keys, values))

    elif is_ndarray and is_single_block:
        # No slicing needed
        dsk = {(name,) + (0,) * x.ndim: x}
    else:
        if getitem is None:
            if fancy:
                getitem = getter
            else:
                getitem = getter_nofancy

        dsk = graph_from_arraylike(
            x,
            chunks,
            x.shape,
            name,
            getitem=getitem,
            lock=lock,
            asarray=asarray,
            dtype=x.dtype,
            inline_array=inline_array,
        )

    # Workaround for TileDB, its indexing is 1-based,
    # and doesn't seems to support 0-length slicing
    if x.__class__.__module__.split(".")[0] == "tiledb" and hasattr(x, "_ctx_"):
        return da.Array(dsk, name, chunks, dtype=x.dtype)

    if meta is None:
        meta = x

    return da.Array(dsk, name, chunks, meta=meta, dtype=getattr(x, "dtype", None))


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

    batch_size = 0
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
    if batch_size > 0:
        with open("groups.txt", "wt") as f:
            ids = []
            first = True
            for g, file_list, metadata in exp_gen:
                if first:
                    first = False
                    if save_group_size:
                        _write_group_size(metadata)
                if not groupby_t and "t" in metadata["file_metadata"][0]:
                    t = [md["t"] for md in metadata["file_metadata"]]
                    if expected_cycles is not None:
                        assert len(t) == expected_cycles

                ids.append('"' + metadata["id"] + '"')
                if len(ids) == batch_size:
                    f.write(" ".join(ids))
                    f.write("\n")
                    ids = []
            if len(ids) > 0:
                f.write(" ".join(ids))
                f.write("\n")
    else:
        with open("groups.txt", "wt") as f:
            first = True
            for g, file_list, metadata in exp_gen:
                f.write(metadata["id"])
                f.write("\n")
                if first:
                    first = False
                    if save_group_size:
                        _write_group_size(metadata)
                    if not groupby_t and "t" in metadata["file_metadata"][0]:
                        t = [md["t"] for md in metadata["file_metadata"]]

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
