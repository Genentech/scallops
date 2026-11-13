"""Reads Processing Module.

This module provides functionalities for processing and analyzing sequencing reads.
It includes tools for filtering, mapping, and summarizing reads, as well as for generating
various quality control metrics.

Authors:
- The SCALLOPS development team
"""

import copy
import itertools
import logging
import math
import random
from collections.abc import Callable, Sequence
from functools import cache
from itertools import product
from typing import Literal, Tuple

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from dask.utils import cached_cumsum
from numpy import ndarray
from numpy.typing import NDArray
from scipy.special import softmax
from skimage.segmentation import expand_labels
from skimage.util import img_as_float
from sklearn.linear_model import QuantileRegressor
from sklearn.neighbors import NearestNeighbors

from scallops.io import CYAN, GRAY, GREEN, MAGENTA, RED, save_stack_imagej
from scallops.utils import _dask_total_threads

logger = logging.getLogger("scallops")

DEFAULT_MIN_P_ERROR = 1e-6

# dtype pandas infers for a column of strings (object, or str with pandas >= 3)
_STR_DTYPE = pd.Series([""]).dtype


def _encode_barcodes(barcodes: Sequence[str] | NDArray) -> NDArray:
    """Encode equal-length barcodes as a 2-D array of character codes.

    :param barcodes: Barcodes, all of the same length (not checked).
    :return: Array of shape (n_barcodes, barcode_length).
    """
    barcodes = list(barcodes)
    length = len(barcodes[0]) if barcodes else 0
    codes = np.frombuffer("".join(barcodes).encode("utf-32-le"), dtype=np.uint32)
    return codes.reshape(len(barcodes), length)


def _fit_whitelist(whitelist_barcodes: NDArray) -> tuple[NDArray, NearestNeighbors]:
    """Encode the whitelist and fit the nearest neighbors index used by
    :func:`_hamming_distance`.

    :param whitelist_barcodes: Whitelist barcodes.
    :return: The whitelist as a numpy object array and the fitted index.
    """
    # numpy object array (pandas >= 3 returns a string extension array from .values)
    whitelist_barcodes = np.asarray(whitelist_barcodes, dtype=object)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="hamming").fit(
        _encode_barcodes(whitelist_barcodes)
    )
    return whitelist_barcodes, nbrs


def _hamming_distance(
    whitelist_index: tuple[NDArray, NearestNeighbors],
    read_barcodes: NDArray,
) -> pd.DataFrame:
    """Computes the Hamming distance between read barcodes and a whitelist of barcodes.

    :param whitelist_index: The whitelist barcodes and nearest neighbors index fitted
        on them, from :func:`_fit_whitelist`. Fit once and reuse it when calling
        repeatedly with the same whitelist.
    :param read_barcodes: A numpy array of read barcodes to compare against the whitelist.
    :return: A DataFrame containing the mismatches and closest matching barcodes from
    the whitelist.
    """
    if len(read_barcodes) == 0:
        # object dtype, as returned by _fit_whitelist
        return pd.DataFrame(
            {
                "mismatches": np.ndarray((0,), dtype=int),
                "closest_match": np.ndarray((0,), dtype=object),
                "mismatches2": np.ndarray((0,), dtype=int),
                "closest_match2": np.ndarray((0,), dtype=object),
            }
        )

    # the barcodes the index was fitted on, so neighbor positions index into them
    whitelist_barcodes, nbrs = whitelist_index
    assert len(whitelist_barcodes[0]) == len(read_barcodes[0]), (
        f"Length of whitelist barcode ({len(whitelist_barcodes[0])}) != length of read ({len(read_barcodes[0])})."
    )
    distances, argmin = nbrs.kneighbors(_encode_barcodes(read_barcodes))
    # hamming distances are fractions of the length; round, since e.g.
    # 15 / 22 * 22 == 14.999... would truncate to 14
    distances = np.rint(distances * len(whitelist_barcodes[0])).astype(int)
    return pd.DataFrame(
        index=read_barcodes,
        data=dict(
            mismatches=distances[:, 0],
            closest_match=whitelist_barcodes[argmin[:, 0]],
            mismatches2=distances[:, 1],
            closest_match2=whitelist_barcodes[argmin[:, 1]],
        ),
    )


def summarize_base_call_mismatches(
    reads_df: pd.DataFrame,
    barcodes_df: pd.DataFrame,
    n_mismatches: int = 1,
    delta_mismatches: int = 1,
) -> pd.DataFrame:
    """Summarize single base call mismatches in called barcodes in reads_df against whitelist in
    barcodes_df.

    :param reads_df: DataFrame containing called reads
    :param barcodes_df: DataFrame containing whitelisted barcodes
    :param n_mismatches: Include reads where number of mismatches to the closest barcode is less than or equal to
        `n_mismatches`
    :param delta_mismatches: Include reads where number of mismatches to 2nd closest barcode minus number of mismatches
        to the closest barcode is greater than or equal to `delta_mismatches`

    :return: DataFrame containing called_base, whitelist_base, read_position (0-based), count, and
        fraction
    """
    reads_df = reads_df.query("label > 0")
    reads_df = reads_df[~reads_df["barcode"].isin(barcodes_df["barcode"])]
    if len(reads_df) == 0:  # no mismatches
        return pd.DataFrame()
    read_value_counts = reads_df.value_counts("barcode")
    df = _hamming_distance(
        whitelist_index=_fit_whitelist(barcodes_df["barcode"].values),
        read_barcodes=read_value_counts.index.values,
    )
    df = df[
        (
            (df["mismatches"] > 0)
            & (df["mismatches"] <= n_mismatches)
            & ((df["mismatches2"] - df["mismatches"]) >= delta_mismatches)
        )
    ]
    barcode_len = len(read_value_counts.index.values[0])
    results = []
    for row_index in range(len(df)):
        closest_match = df["closest_match"].iloc[row_index]
        read = df.index[row_index]
        for str_idx in range(barcode_len):
            if closest_match[str_idx] != read[str_idx]:
                break
        called_base = read[str_idx]
        whitelist_base = closest_match[str_idx]
        count = read_value_counts.loc[read]
        results.append([called_base, whitelist_base, str_idx, count])
    df = pd.DataFrame(
        results, columns=["called_base", "whitelist_base", "read_position", "count"]
    )
    df = (
        df.groupby(["called_base", "whitelist_base", "read_position"])
        .agg("sum")
        .reset_index()
    )
    total = df["count"].sum()
    df["fraction"] = df["count"] / total
    return df


def quality_softmax(x: NDArray, min_error: float = 1e-6) -> NDArray:
    """Computes the phred quality score of transformed data using the softmax function.

    :param x: Array with transformed data (read, cycle, channel).
    :param min_error: Minimum p-value error.
    :return: Array with computed quality scores (higher is better) with shape (read, cycle)
    """

    p = np.max(softmax(x, axis=2), axis=2)

    p_error = 1 - p
    p_error[p_error < min_error] = min_error
    return -10 * np.log10(p_error)


def make_encoding(
    channel_bases: list[str],
    dark_bases: list[str],
) -> tuple[NDArray, NDArray]:
    """Build an encoding matrix in which each bright base fires only its own channel.

    The encoding matrix ``E`` says which channels each base lights up: one row per
    base, one column per channel, with ``E[b, c] = 1`` if base ``b`` fires channel
    ``c`` and 0 otherwise.  Here each base in ``channel_bases`` fires only its own
    channel, and each base in ``dark_bases`` fires none (an all-zero row), e.g.
    G in 3-color NIS-seq:

    >>> E, labels = make_encoding(["A", "T", "C"], dark_bases=["G"])
    >>> labels
    array(['A', 'T', 'C', 'G'], dtype='<U1')
    >>> E
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 0.]])

    :func:`decode_se`, :func:`decode_polar` and :func:`decode_em` call this
    themselves when given ``dark_bases``, using ``spots.c`` as ``channel_bases``.
    Use at most one dark base: dark bases have identical rows and cannot be told
    apart, so the decoders raise an error for more.

    This function cannot express a base that fires several channels.  For those
    chemistries, write the matrix directly and pass it as ``encoding`` with
    ``base_labels`` instead.  For example, in 2-color chemistry where A fires both
    channels: ``E = [[0, 0], [0, 1], [1, 1], [1, 0]]`` with
    ``base_labels=["G", "T", "A", "C"]``.

    :param channel_bases: Base imaged in each channel, in channel order.
    :param dark_bases: Bases with no channel.
    :return: ``(E, base_labels)``: ``E`` is a float array of shape
        ``(n_bases, n_channels)``, and ``base_labels`` names its rows
        (``channel_bases`` followed by ``dark_bases``).
    """
    all_bases = list(channel_bases) + list(dark_bases)
    E = np.eye(len(all_bases), len(channel_bases), dtype=float)
    return E, np.array(all_bases)


def channel_crosstalk_matrix(
    a: xr.DataArray,
    method: Literal["median", "li_and_speed"] = "median",
    by_t: bool = False,
    **kwargs,
) -> ndarray | da.Array | dict[str, list[str] | NDArray]:
    """Estimate and correct differences in channel intensity and spectral overlap among sequencing
    channels using either median or Li and Speed method.

    Describe with linear transformation w so that w * a = y, where y is the corrected data.

    :param a: data to compute crosstalk matrix for (read, t, c)
    :param by_t: Compute separate matrices per cycle.
    :param method: Either median or li_and_speed
    :param kwargs: Passed to the ``li_and_speed`` estimator (``n_iter_max``,
        ``slope_threshold``, ``quantile_range``).  Ignored by ``median``.
    :return: The inverse matrix, w (c, c), or an array of (t, c, c) if `by_t` — except that with
        `method="median"` on a dask-backed `a`, a single cycle still returns (c, c) even if `by_t`
        is set, since there is nothing to stack.
    """

    method_name = str(method).lower()
    assert method_name in ["median", "li_and_speed"]

    if isinstance(a.data, da.Array) and method_name == "median":
        return _channel_crosstalk_matrix_median_dask(a, by_t=by_t)

    correct_crosstalk = (
        _correct_channel_crosstalk_li_and_speed
        if method_name == "li_and_speed"
        else _correct_channel_crosstalk_median
    )
    if by_t:
        dims = ["t"]
        dim_vals = [a[d].values for d in dims]
        w_arrays = []

        for dim_val in itertools.product(*dim_vals):
            sel = dict(zip(dims, dim_val))
            w = correct_crosstalk(a.sel(sel).data.reshape(-1, a.sizes["c"]), **kwargs)
            w_arrays.append(w)

        return np.array(w_arrays)
    return correct_crosstalk(a.data.reshape(-1, a.sizes["c"]), **kwargs)


def _channel_crosstalk_matrix_median_dask(a: xr.DataArray, by_t: bool) -> da.Array:
    """Exact median crosstalk matrix for a dask-backed array without loading all reads
    into one chunk.

    Each block is split by the channel with the maximum intensity; the parts for each
    channel are then concatenated and reduced with :func:`numpy.median` in a separate
    task. A median depends only on the values, not their order, so the result is the
    same as :func:`_crosstalk_median_ratio` on the whole array for any chunking.

    :param a: Data (read, t, c).
    :param by_t: Compute separate matrices per cycle.
    :return: The inverse matrix (c, c), or (t, c, c) if `by_t` and there is more than
        one cycle.
    """
    nchannels = a.sizes["c"]
    data = a.transpose("read", "t", "c").data
    # t and c unchunked, or one cycle per chunk if `by_t`
    data = data.rechunk({1: 1 if by_t else -1, 2: -1})
    blocks = data.to_delayed()[:, :, 0]  # (read blocks, t blocks)
    split = delayed(_split_by_max_channel, nout=nchannels, pure=True)
    concat_median = delayed(_concat_median, pure=True)
    inv_median_ratio = delayed(_inv_median_ratio, pure=True)

    w_inv = []
    for t in range(blocks.shape[1]):
        parts = [split(block, nchannels) for block in blocks[:, t]]
        medians = [concat_median(*[p[i] for p in parts]) for i in range(nchannels)]
        w_inv.append(
            da.from_delayed(
                inv_median_ratio(*medians),
                shape=(nchannels, nchannels),
                dtype=np.float64,
            )
        )
    # a single cycle returns (c, c), as the previous dask implementation
    return da.stack(w_inv, axis=0) if len(w_inv) > 1 else w_inv[0]


def _split_by_max_channel(x: NDArray, nchannels: int) -> tuple[NDArray, ...]:
    """Split rows by the channel with the maximum intensity.

    :param x: Array (..., c).
    :param nchannels: Number of channels.
    :return: Tuple of `nchannels` arrays (rows, c).
    """
    x = x.reshape(-1, nchannels)
    max_indices = x.argmax(axis=1)
    return tuple(x[max_indices == i] for i in range(nchannels))


def _concat_median(*parts: NDArray) -> NDArray:
    """Median over the rows of the concatenated parts.

    :param parts: Arrays (rows, c).
    :return: Median for each channel.
    """
    return np.median(np.concatenate(parts), axis=0)


def _normalize_median_array(median_array: NDArray) -> NDArray:
    """Normalize the columns of the median array to sum to one.

    :param median_array: Array (c, c) where column i holds the channel medians of rows
        where channel i is the maximum.
    :return: Normalized array with NaNs replaced by 1.
    """
    totals = median_array.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        median_array = median_array / totals

    median_array[np.isnan(median_array)] = 1  # Replace NaNs with 1
    return median_array


def _inv_median_ratio(*medians: NDArray) -> NDArray:
    """Inverse crosstalk matrix from per-channel medians.

    :param medians: For each channel i, the channel medians of rows where channel i is
        the maximum.
    :return: The inverse matrix (c, c).
    """
    median_ratio = _normalize_median_array(np.array(medians).T)
    return np.linalg.inv(median_ratio.astype(np.float64, copy=False))


def _correct_channel_crosstalk_median(a: NDArray) -> NDArray:
    """Estimate and correct differences in channel intensity and spectral overlap among
    sequencing channels. For each channel, find points where the largest signal is from
    that channel. Use the median of these points to define new basis vectors.
    Describe with linear transformation w, so that w * x_array = y, where y is the
    corrected data.

    :param a: raw data to transform (read + t, c)
    :return: The inverse matrix, w
    """

    return np.linalg.inv(_crosstalk_median_ratio(a))


def n_unique_bases(bases_array: xr.DataArray) -> NDArray:
    """Number of distinct bases called (channel with the maximum intensity) across cycles
    for each read.

    :param bases_array: Intensities with dimensions (read, t, c).
    :return: Array with the number of distinct bases for each read.
    """
    data = bases_array.transpose("read", "t", "c").data
    if isinstance(data, da.Array):
        data = data.rechunk({1: -1, 2: -1})
        n_unique = data.map_blocks(
            _n_unique_bases_chunk, drop_axis=[1, 2], dtype=np.int64
        ).compute()
    else:
        n_unique = _n_unique_bases_chunk(data)
    return n_unique


def _n_unique_bases_chunk(x: NDArray) -> NDArray:
    return _n_unique_per_row(x.argmax(axis=2))  # calls (read, t)


def _n_unique_per_row(codes: NDArray) -> NDArray:
    """Number of distinct values in each row of a 2D array.

    :param codes: Array (rows, values).
    :return: Number of distinct values for each row.
    """
    if codes.shape[1] == 0:
        return np.zeros(codes.shape[0], dtype=np.int64)
    codes = np.sort(codes, axis=1)
    return 1 + (np.diff(codes, axis=1) != 0).sum(axis=1)


def barcode_n_unique_bases(barcodes: Sequence[str]) -> NDArray:
    """Number of distinct bases in each barcode.

    Counterpart of :func:`n_unique_bases` for barcode strings, e.g. whitelist barcodes
    or barcodes called by :func:`decode_max`.

    :param barcodes: Barcodes, all of the same length.
    :return: Number of distinct bases for each barcode.

    >>> barcode_n_unique_bases(["AAAA", "ACGT", "AACC"]).tolist()
    [1, 4, 2]
    """
    return _n_unique_per_row(_encode_barcodes(barcodes))


def filter_low_complexity_reads(
    bases_array: xr.DataArray, min_unique_bases: int = 2
) -> xr.DataArray:
    """Remove reads whose called bases take fewer than `min_unique_bases` distinct values
    across cycles.

    Reads with the same brightest channel in every cycle are typically
    autofluorescence, debris, or other false positive spots rather than sequencing
    reads.

    :param bases_array: Intensities with dimensions (read, t, c).
    :param min_unique_bases: Minimum number of distinct bases. Use 2 to remove reads
        with the same base in every cycle.
    :return: The filtered array.
    """
    if min_unique_bases <= 1:
        return bases_array
    keep = n_unique_bases(bases_array) >= min_unique_bases
    return bases_array.isel(read=keep)


def apply_channel_crosstalk_matrix(
    a: xr.DataArray, w: NDArray | da.Array, dtype=np.float32
) -> xr.DataArray:
    """Applies a linear transformation w * a = y, where y is the corrected data.

    :param a: data to correct
    :param w: Crosstalk compensation matrix
    :param dtype: Corrected data type
    :return: The corrected data
    """

    def _apply(x, w, nchannels, result_dtype):
        if w.ndim == 2:
            return (
                w.dot(x.reshape(-1, nchannels).T)
                .T.reshape(x.shape)
                .astype(result_dtype, copy=False)
            )
        data = []
        for t in range(len(w)):
            x_i = x[:, t, :]
            w_i = w[t]
            data.append(w_i.dot(x_i.reshape(-1, nchannels).T).T.reshape(x_i.shape))
        return np.stack(data, axis=1).astype(result_dtype, copy=False)

    nchannels = a.sizes["c"]
    coords = a.coords.copy()
    dims = a.dims
    attrs = a.attrs.copy()
    a = a.data
    if isinstance(a, da.Array):
        chunksize_list = list(a.chunksize)
        # no chunking in t or c dimensions
        no_chunk_dims = ["c"]
        if w.ndim == 3:
            no_chunk_dims.append("t")
        for i in range(len(dims)):
            if dims[i] in no_chunk_dims and a.chunksize[i] != a.shape[i]:
                chunksize_list[i] = -1
        chunksize = tuple(chunksize_list)
        if chunksize != a.chunksize:
            a = a.rechunk(chunksize)

        vals = da.map_blocks(
            _apply,
            a,
            w=w,
            nchannels=nchannels,
            dtype=dtype,
            result_dtype=dtype,
        )
    else:
        vals = _apply(a, w, nchannels, dtype)
    return xr.DataArray(data=vals, coords=coords, dims=dims, attrs=attrs)


def _crosstalk_median_ratio(a: NDArray) -> NDArray:
    """Compute the median ratio of the input array to quantify crosstalk between channels.

    :param a: Input 2D array where each row represents data for a specific observation, and each
        column corresponds to a channel.
    :return: A normalized 2D array of median ratios for each channel.
    """
    max_indices = a.argmax(axis=1)  # Indices of maximum values per row
    median_array = np.array(
        [np.median(a[max_indices == i], axis=0) for i in range(a.shape[1])]
    ).T
    return _normalize_median_array(median_array)


def _assign_barcodes_to_labels_df(
    df: pd.DataFrame,
    sort_by: list[str],
    agg_columns: list[str],
    top2_columns: list[str],
) -> pd.DataFrame:
    """Call the top two barcodes for each label in an in-memory data frame.

    Labels with at least one perfect match (``mismatches`` is NA) use only perfect
    matches; other labels use their corrected/mismatched reads and are flagged with
    ``mismatch``. Reads are summed per (label, barcode) and barcodes are ranked by the
    summed ``sort_by`` columns in descending order, ties broken by the order in which
    the barcode first appears for the label. Labels are returned in order of first
    appearance in `df`.

    :param df: Reads with `label` and `barcode` columns plus `sort_by`, `agg_columns`,
        and `top2_columns`.
    :param sort_by: Columns to rank barcodes by.
    :param agg_columns: Scalar columns to sum and report for the top two barcodes.
    :param top2_columns: Array-valued columns (e.g. per-cycle `Q`) to sum element-wise
        and report for the top two barcodes.
    :return: One row per label.
    """
    # labels are returned in order of first appearance before filtering
    _, label_order = pd.factorize(df["label"], use_na_sentinel=False)
    if "mismatches" in df.columns:
        perfect = df["mismatches"].isna()
        has_perfect = perfect.groupby(df["label"], sort=False, dropna=False).transform(
            "any"
        )
        keep = (perfect | ~has_perfect).to_numpy()
        df = df[keep]
        label_mismatch = ~has_perfect[keep].to_numpy()
    else:
        label_mismatch = np.zeros(len(df), dtype=bool)

    barcode_dtype = df["barcode"].dtype
    sum_columns = list(dict.fromkeys(agg_columns + sort_by))
    groupby_keys = [df["label"], df["barcode"]]
    groupby = df.groupby(groupby_keys, sort=False, dropna=False)
    # groups are numbered in order of first appearance
    group_ids = groupby.ngroup().to_numpy()
    # sum in float64 (e.g. Q_mean is float32 when computed from float32 intensities)
    sum_df = df[sum_columns].astype(
        {c: np.float64 for c in sum_columns if df[c].dtype.kind == "f"}
    )
    sums = sum_df.groupby(groupby_keys, sort=False, dropna=False).sum()
    n_groups = len(sums)
    group_labels = sums.index.get_level_values("label")
    group_barcodes = sums.index.get_level_values("barcode").to_numpy()
    group_sizes = np.bincount(group_ids, minlength=n_groups)

    # labels numbered in order of first appearance
    label_codes, labels = pd.factorize(group_labels, use_na_sentinel=False)
    n_labels = len(labels)
    label_counts = np.bincount(label_codes, weights=group_sizes, minlength=n_labels)
    # label_mismatch is constant within a label
    mismatch = np.zeros(n_labels, dtype=bool)
    mismatch[label_codes[group_ids]] = label_mismatch

    # rank barcodes within each label (stable, so ties keep first appearance order)
    sort_keys = [np.arange(n_groups)]
    for c in reversed(sort_by):
        # float64 so negation doesn't wrap around for unsigned integer sums
        sort_keys.append(-sums[c].to_numpy(dtype=np.float64))
    sort_keys.append(label_codes)
    order = np.lexsort(sort_keys)
    sorted_label_codes = label_codes[order]
    is_first = np.ones(n_groups, dtype=bool)
    is_first[1:] = sorted_label_codes[1:] != sorted_label_codes[:-1]
    top0 = order[is_first]  # one per label, in label code order
    is_second = np.zeros(n_groups, dtype=bool)
    is_second[1:] = (~is_first[1:]) & is_first[:-1]
    has_second = np.zeros(n_labels, dtype=bool)
    has_second[sorted_label_codes[is_second]] = True
    top1 = np.zeros(n_labels, dtype=np.int64)
    top1[sorted_label_codes[is_second]] = order[is_second]

    top2_sums = {}
    for c in top2_columns:
        values = df[c].to_numpy()
        if len(values) > 0:
            # sum in float64 so results don't depend on the input dtype, which can
            # change from float32 to float64 when dask shuffles the reads
            values = np.stack(values).astype(np.float64, copy=False)
            c_sums = np.zeros((n_groups,) + values.shape[1:], dtype=np.float64)
            np.add.at(c_sums, group_ids, values)
        else:
            c_sums = np.zeros((0,), dtype=np.float64)
        top2_sums[c] = c_sums

    def _to_object_array(values: NDArray, mask: NDArray | None = None) -> NDArray:
        # 1-D object array of the rows of `values` (None where `mask` is False);
        # object dtype even when empty so partitions match the dask meta
        out = np.empty(len(values), dtype=object)
        # slice assignment keeps each row as an array (masked assignment doesn't)
        out[:] = list(values)
        if mask is not None:
            out[~mask] = None
        return out

    result = {
        "label": labels,
        # same dtype as the input (object, or str with pandas >= 3)
        "barcode_0": pd.array(group_barcodes[top0], dtype=barcode_dtype),
        "barcode_1": pd.array(
            np.where(has_second, group_barcodes[top1], ""), dtype=barcode_dtype
        ),
        "mismatch": mismatch,
        "barcode_count": label_counts.astype(np.int64),
        "barcode_count_0": group_sizes[top0].astype(np.int64),
    }
    for c in agg_columns:
        result[f"barcode_{c}_0"] = sums[c].to_numpy(dtype=np.float64)[top0]
    for c in top2_columns:
        result[f"barcode_{c}_0"] = _to_object_array(top2_sums[c][top0])

    result["barcode_count_1"] = np.where(has_second, group_sizes[top1], 0).astype(
        np.int64
    )
    default_values = {"Q_min": 0, "Q_mean": 0, "peak": np.nan}
    for c in agg_columns:
        result[f"barcode_{c}_1"] = np.where(
            has_second,
            sums[c].to_numpy()[top1],
            default_values.get(c, np.nan),
        ).astype(np.float64)
    for c in top2_columns:
        result[f"barcode_{c}_1"] = _to_object_array(
            top2_sums[c][top1] if n_labels > 0 else top2_sums[c], mask=has_second
        )
    result_df = pd.DataFrame(result)
    order = np.argsort(pd.Index(label_order).get_indexer(labels), kind="stable")
    return result_df.take(order).reset_index(drop=True)


def assign_barcodes_to_labels(
    df_reads: pd.DataFrame | dd.DataFrame,
    sort_by: str | list[str] = ["Q_mean", "peak"],
) -> pd.DataFrame | dd.DataFrame:
    """Call the barcode for each label.

    :param df_reads: Table of all called reads from one of the ``decode_*`` functions.
    :param sort_by: Which column(s) to sort aggregated barcodes by.
    :return: One row per label, with columns ``label``, ``barcode_0``/``barcode_1`` (the top two
        barcodes by `sort_by`), ``barcode_count`` (total reads in the label),
        ``barcode_count_0``/``barcode_count_1`` (reads with that barcode), ``mismatch`` (whether no
        read exactly matched a barcode), the summed per-cycle ``barcode_Q_{0,1}``, and, for each
        of ``Q_mean``, ``Q_min``, and ``peak`` present in `df_reads`, a summed
        ``barcode_{column}_{0,1}``. See :ref:`outputs` for the full column reference.
    """
    sort_by = [sort_by] if isinstance(sort_by, str) else list(sort_by)
    top2_columns = ["Q"]
    agg_columns = ["Q_mean", "Q_min", "peak"]
    agg_columns = [c for c in agg_columns if c in df_reads.columns]
    top2_columns = [c for c in top2_columns if c in df_reads.columns]

    columns_needed = list(
        dict.fromkeys(["label", "barcode"] + agg_columns + top2_columns + sort_by)
    )
    if "mismatches" in df_reads.columns:
        columns_needed.append("mismatches")
    df_reads = df_reads[columns_needed]
    kwargs = dict(sort_by=sort_by, agg_columns=agg_columns, top2_columns=top2_columns)
    if not isinstance(df_reads, dd.DataFrame):
        return _assign_barcodes_to_labels_df(df_reads, **kwargs)

    columns = []
    columns.append(("label", df_reads["label"].dtype))
    columns.append(("barcode_0", df_reads["barcode"].dtype))
    columns.append(("barcode_1", df_reads["barcode"].dtype))
    columns.append(("mismatch", bool))
    columns.append(("barcode_count", np.int64))
    columns.append(("barcode_count_0", np.int64))
    # barcode 0
    for c in agg_columns:
        columns.append((f"barcode_{c}_0", np.float64))
    for c in top2_columns:
        columns.append((f"barcode_{c}_0", object))
    # barcode 1
    columns.append(("barcode_count_1", np.int64))
    for c in agg_columns:
        columns.append((f"barcode_{c}_1", np.float64))
    for c in top2_columns:
        columns.append((f"barcode_{c}_1", object))
    meta = dd.utils.make_meta(columns)
    if df_reads.npartitions == 1:
        return df_reads.map_partitions(
            _assign_barcodes_to_labels_df, meta=meta, **kwargs
        )
    # Put all reads for a label in the same partition. Shuffling does not preserve row
    # order, so record each row's position and restore it to keep results identical
    # to the in-memory version (row order breaks ties and sets summation order).
    position_meta = df_reads._meta.assign(
        _partition=np.int64(0), _row=np.int64(0)
    ).iloc[:0]
    df_reads = df_reads.map_partitions(_add_row_position, meta=position_meta).shuffle(
        "label"
    )
    return df_reads.map_partitions(
        _assign_barcodes_to_labels_sorted_df, meta=meta, **kwargs
    )


def _add_row_position(df: pd.DataFrame, partition_info=None) -> pd.DataFrame:
    partition = partition_info["number"] if partition_info is not None else 0
    return df.assign(
        _partition=np.int64(partition), _row=np.arange(len(df), dtype=np.int64)
    )


def _assign_barcodes_to_labels_sorted_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = df.sort_values(["_partition", "_row"], kind="stable").drop(
        columns=["_partition", "_row"]
    )
    return _assign_barcodes_to_labels_df(df, **kwargs)


def correct_mismatches(
    reads: pd.DataFrame | dd.DataFrame,
    barcodes: pd.DataFrame,
    n_mismatches: int = 1,
    delta_mismatches: int = 1,
) -> pd.DataFrame | dd.DataFrame:
    """Correct mismatches between called barcodes and barcodes in a whitelist.

    Note that if a read is equidistant to more than one barcode, it will not be corrected.

    :param reads: Reads from one of the ``decode_*`` functions.
    :param barcodes: Data frame of designed barcode sequences. Expected to have column 'barcode'
    :param n_mismatches: Correct mismatch if number of mismatches to the closest barcode is less than or equal to
        `n_mismatches`
    :param delta_mismatches: Correct mismatch if number of mismatches to 2nd closest barcode minus number of mismatches
        to the closest barcode is greater than or equal to `delta_mismatches`
    :return: Adds the columns barcode_uncorrected, mismatches, mismatches2, closest_match, and closest_match2.
             Updates the `barcode` column to the closest match and the boolean column `barcode_match` if `n_mismatches`
             and `delta_mismatches` criteria are satisfied.
    """
    is_dask = isinstance(reads, dd.DataFrame)
    if not is_dask:
        reads = reads.copy()
    if "barcode_match" not in reads.columns:
        reads["barcode_match"] = reads["barcode"].isin(barcodes["barcode"])

    def _single_partition(reads_df, whitelist_index):
        reads_df = reads_df.join(
            _hamming_distance(
                whitelist_index=whitelist_index,
                read_barcodes=reads_df.query("~barcode_match")["barcode"].unique(),
            ),
            on="barcode",
        )

        correct = (
            (reads_df["mismatches"] > 0)
            & (reads_df["mismatches"] <= n_mismatches)
            & ((reads_df["mismatches2"] - reads_df["mismatches"]) >= delta_mismatches)
        )
        reads_df.loc[correct, "barcode_uncorrected"] = reads_df.loc[correct]["barcode"]
        reads_df.loc[correct, "barcode"] = reads_df.loc[correct]["closest_match"]
        reads_df.loc[correct, "barcode_match"] = True  # update after correction
        return reads_df.astype(output_dtypes)

    # mismatches are NaN for reads that exactly match a whitelist barcode
    output_dtypes = {
        "mismatches": np.float64,
        "closest_match": _STR_DTYPE,
        "mismatches2": np.float64,
        "closest_match2": _STR_DTYPE,
        "barcode_uncorrected": _STR_DTYPE,
    }
    # encode the whitelist and fit the index once rather than in every partition
    whitelist_index = _fit_whitelist(barcodes["barcode"].values)
    if is_dask:
        meta = reads._meta.copy()
        for c, dtype in output_dtypes.items():
            meta[c] = pd.Series(dtype=dtype)
        return dd.map_partitions(
            _single_partition, reads, delayed(whitelist_index), meta=meta
        )
    else:
        return _single_partition(reads, whitelist_index)


def _decode_max_chunk(
    spots: NDArray,
    bases: list[str],
    meta_df: pd.DataFrame,
    offset: slice | None,
    whitelist: list[str] | None,
    thresholded_bases: Sequence[dict] | None = None,
) -> pd.DataFrame:
    Q = quality_softmax(spots)
    channel_calls = np.argmax(spots, axis=2)
    if thresholded_bases is not None:
        min_error = DEFAULT_MIN_P_ERROR
        bases = list(bases)
        n_bases = len(bases)
        for thresholded_base_index in range(len(thresholded_bases)):
            thresholded_base = thresholded_bases[thresholded_base_index]
            base = thresholded_base["base"]
            channels = thresholded_base["channels"]
            threshold = thresholded_base["threshold"]
            direction = thresholded_base["direction"]
            base_offset = thresholded_base_index + n_bases

            max_spots = np.max(spots[..., channels], axis=2)  # (read, t)
            max_spots_per_read = np.max(max_spots, axis=1)  # (read,)
            cutoff = max_spots_per_read * threshold  # (read,)

            # create fake 2 channel data with cutoff and max for p value
            p_thresholded = softmax(
                np.concatenate(
                    (
                        np.expand_dims(max_spots, axis=2),
                        np.repeat(
                            np.expand_dims(np.expand_dims(cutoff, axis=1), axis=2),
                            max_spots.shape[1],
                            axis=1,
                        ),
                    ),
                    axis=2,
                ),
                axis=2,
            )  # (read, t, 2)
            # Q is only used where the base is called, i.e. where the called class
            # (cutoff for "<", max_spots for ">") wins the softmax, so take the max
            p_thresholded = np.max(p_thresholded, axis=2)
            p_thresholded_error = 1 - p_thresholded
            p_thresholded_error[p_thresholded_error < min_error] = min_error
            Q_thresholded = -10 * np.log10(p_thresholded_error)
            if direction == "<":
                base_present = max_spots < np.expand_dims(cutoff, axis=1)
            else:
                base_present = max_spots > np.expand_dims(cutoff, axis=1)
            channel_calls[base_present] = base_offset
            Q[base_present] = Q_thresholded[base_present]
            bases.append(base)
        bases = np.array(bases)
    calls = bases[channel_calls]

    df = (
        meta_df.iloc[offset.start : offset.stop].copy()
        if offset is not None
        else meta_df.copy()
    )
    df.index.name = None
    df["barcode"] = ["".join(x) for x in calls]
    df["Q"] = list(Q)
    df["Q_mean"] = Q.mean(axis=1)
    df["Q_min"] = Q.min(axis=1)
    if whitelist is not None:
        df["barcode_match"] = df["barcode"].isin(whitelist)
    return df


def _decode(
    spots: xr.DataArray, barcodes: pd.DataFrame | None, f: Callable, **kwargs
) -> pd.DataFrame | dd.DataFrame:
    """Call reads by assigning the base with the highest intensity.

    :param spots: Spots returned from peaks_to_bases containing dimensions (read,t,c)
    :param barcodes: Table of designed barcode sequences used for indicating whether a
        barcode is an exact match. Expected to have column 'barcode'.
    :return: One row per read, with the coordinates of `spots["read"]` (e.g. ``y``, ``x``,
        ``sigma``, ``peak``, ``label``, if present) plus the called ``barcode`` (one base per
        cycle), ``Q`` (per-cycle quality), ``Q_mean``, ``Q_min``, and, when `barcodes` is given,
        ``barcode_match`` (whether `barcode` is in `barcodes`).
    """

    whitelist = barcodes["barcode"].values if barcodes is not None else None
    meta_df = spots["read"].to_dataframe()  # index is read
    decode_args = dict(whitelist=whitelist, meta_df=meta_df)
    decode_args.update(kwargs)
    if not isinstance(spots.data, da.Array):
        decode_args["spots"] = spots.data
        df = f(
            **decode_args,
            offset=None,
        )
    else:
        # no chunking in t or c dimension
        dims = spots.dims
        chunksize_list = list(spots.data.chunksize)
        for i in range(len(dims)):
            if dims[i] in ("c", "t") and spots.data.chunksize[i] != spots.data.shape[i]:
                chunksize_list[i] = -1
        chunksize = tuple(chunksize_list)
        if chunksize != spots.data.chunksize:
            spots.data = spots.data.rechunk(chunksize)

        columns = []
        for col in meta_df.columns:
            columns.append((col, meta_df[col].dtype))
        columns.append(("barcode", _STR_DTYPE))
        columns.append(("Q", object))
        columns.append(("Q_mean", np.float64))
        columns.append(("Q_min", np.float64))
        if whitelist is not None:
            columns.append(("barcode_match", bool))
        meta = dd.utils.make_meta(columns)
        f_delayed = delayed(f)
        for key in decode_args.keys():
            decode_args[key] = delayed(decode_args[key])

        starts = [cached_cumsum(bds, initial_zero=True) for bds in spots.data.chunks]
        ndim = len(starts)
        results = []
        for block in spots.data.to_delayed().ravel():
            key = np.array(block.key[1:])
            start = []
            stop = []
            for i in range(ndim):
                start.append(starts[i][key[i]])
                stop.append(starts[i][key[i] + 1])
            decode_args["spots"] = block
            results.append(
                f_delayed(
                    **decode_args,
                    offset=slice(start[0], stop[0]),
                )
            )
        df = dd.from_delayed(results, meta=meta, verify_meta=False)
    return df


def decode_max(
    spots: xr.DataArray,
    barcodes: pd.DataFrame | None = None,
    thresholded_bases: Sequence[dict] | None = None,
) -> pd.DataFrame | dd.DataFrame:
    """Call reads by assigning the base with the highest intensity (softmax argmax).

    :param spots: Spots returned from peaks_to_bases containing dimensions (read, t, c).
    :param barcodes: Table of designed barcode sequences used for indicating whether a
        barcode is an exact match. Expected to have column 'barcode'.
    :param thresholded_bases: Optional list of thresholded bases when number of bases is greater than number of
        imaged channels (e.g. NIS-Seq). For example, `dict(base="G", channels=[0,1,2], direction='<', threshold=0.2)`,
        calls a `G` when all intensities are below 20% of the spot’s maximum intensity across all cycles
    :return: The reads data frame.
    """
    return _decode(
        spots=spots,
        barcodes=barcodes,
        f=_decode_max_chunk,
        bases=spots.c.values,
        thresholded_bases=thresholded_bases,
    )


def _decode_se_chunk(
    spots: NDArray,
    encoding: NDArray,
    base_labels: NDArray,
    meta_df: pd.DataFrame,
    offset: slice | None,
    whitelist: list[str] | None,
) -> pd.DataFrame:
    """Signed-encoding chunk decoder with data-adaptive f = alpha/3.

    For each cycle assigns the base that maximizes
    ``score(b,t) = Σ_c (2E[b,c]-1) · (x_ct - l_c - f(h_c-l_c))``,
    where f = alpha/3 and alpha = min_c(mean top-phi_c normalized bright cycles).

    :param spots: ``(read, T, n_ch)`` spot intensities (xtalk-corrected).
    :param encoding: ``(n_bases, n_ch)`` binary encoding matrix; all-zero rows → dark bases.
    :param base_labels: Base label array of length n_bases.
    :param meta_df: Metadata DataFrame (index = read integer).
    :param offset: Slice into meta_df for this chunk.
    :param whitelist: Barcode whitelist for barcode_match column.
    :return: DataFrame with barcode, Q, Q_mean, Q_min, optionally barcode_match.
    """
    bright_fraction = encoding.sum(axis=0) / encoding.shape[0]
    spots_c = np.clip(spots, 0.0, None)
    lo = spots_c.min(axis=1, keepdims=True)
    hi = spots_c.max(axis=1, keepdims=True)
    rng = hi - lo
    x_norm = np.divide(spots_c - lo, rng, out=np.zeros_like(spots_c), where=rng > 0)
    T = spots.shape[1]
    alpha_per_ch = np.zeros(spots.shape[2])
    for c_idx, phi_c in enumerate(bright_fraction):
        if phi_c <= 0.0:
            continue
        k = max(1, int(np.ceil(float(phi_c) * T)))
        alpha_per_ch[c_idx] = np.sort(x_norm[:, :, c_idx], axis=1)[:, -k:].mean()
    bright_mask = bright_fraction > 0.0
    f = float(alpha_per_ch[bright_mask].min()) / 3.0 if bright_mask.any() else 0.5
    log_bp = (spots_c - (lo + f * (hi - lo))) @ (2 * encoding - 1).T
    bp = softmax(log_bp, axis=-1)
    p_best = np.clip(bp.max(axis=2), 1e-6, 1 - 1e-6)
    Q = -10 * np.log10(1 - p_best)
    calls = base_labels[np.argmax(bp, axis=2)]

    df = (
        meta_df.iloc[offset.start : offset.stop].copy()
        if offset is not None
        else meta_df.copy()
    )
    df.index.name = None
    df["barcode"] = ["".join(x) for x in calls]
    df["Q"] = list(Q)
    df["Q_mean"] = Q.mean(axis=1)
    df["Q_min"] = Q.min(axis=1)
    if whitelist is not None:
        df["barcode_match"] = df["barcode"].isin(whitelist)
    return df


def decode_se(
    spots: xr.DataArray,
    barcodes: pd.DataFrame | None = None,
    encoding: NDArray | None = None,
    base_labels: list[str] | NDArray | None = None,
    dark_bases: list[str] | None = None,
) -> pd.DataFrame | dd.DataFrame:
    """Call reads using the signed-encoding (SE) decoder with data-adaptive f = alpha/3.

    Handles dark bases (all-zero rows in ``E``) automatically — their score
    ``-Σ d_ct`` is maximized at quiet cycles.  Per-channel range weighting acts as an
    SNR proxy, down-weighting dim channels.

    :param spots: Spots DataArray with dims ``(read, t, c)``.
    :param barcodes: Whitelist DataFrame with column ``'barcode'``; adds ``barcode_match``.
    :param encoding: ``(n_bases, n_channels)`` binary encoding matrix.  All-zero rows
        indicate dark bases; at most one is allowed.  When omitted and ``dark_bases``
        is also omitted, defaults to the identity (one bright base per channel, no
        dark bases).
    :param base_labels: Ordered base labels of length n_bases, required with ``encoding``.
    :param dark_bases: Shorthand for 1:1 channel→base chemistry with named dark bases.
        Builds the encoding via :func:`make_encoding`.
    :return: DataFrame with columns ``barcode``, ``Q``, ``Q_mean``, ``Q_min``, and
        optionally ``barcode_match``.
    """
    if encoding is None and dark_bases is not None:
        encoding, base_labels = make_encoding(
            channel_bases=list(spots.c.values), dark_bases=dark_bases
        )
    if encoding is None:
        n_ch = len(spots.c.values)
        encoding = np.eye(n_ch, dtype=float)
        base_labels = np.array(list(spots.c.values))

    encoding = np.asarray(encoding, dtype=float)
    base_labels = np.asarray(base_labels)
    dark_rows = np.where(encoding.sum(axis=1) == 0)[0]
    if len(dark_rows) > 1:
        raise ValueError(
            "decode_se supports at most one dark base, got "
            f"{list(base_labels[dark_rows])}: dark bases have no signal and "
            "cannot be told apart"
        )
    return _decode(
        spots=spots,
        barcodes=barcodes,
        f=_decode_se_chunk,
        encoding=encoding,
        base_labels=base_labels,
    )


def _resolve_encoding(
    spots: xr.DataArray,
    encoding: NDArray | None,
    base_labels: list[str] | NDArray | None,
    dark_bases: list[str] | None,
) -> tuple[NDArray, NDArray]:
    """Resolve ``encoding``/``base_labels``/``dark_bases`` as in :func:`decode_se`."""
    if encoding is None and dark_bases is not None:
        encoding, base_labels = make_encoding(
            channel_bases=list(spots.c.values), dark_bases=dark_bases
        )
    if encoding is None:
        encoding = np.eye(len(spots.c.values), dtype=float)
        base_labels = np.array(list(spots.c.values))
    if base_labels is None:
        raise ValueError("base_labels is required with encoding")
    return np.asarray(encoding, dtype=float), np.asarray(base_labels)


def _em_cycle_gain(x: NDArray, q: float = 0.9) -> NDArray:
    """Per-cycle, per-channel gain from the signal alone.

    Every cycle has a similar mix of bases, so the ``q``-quantile of a cycle's
    intensities across reads tracks that cycle's brightness (e.g. a bright first
    cycle).  Returned relative to the median over cycles.

    :param x: ``(read, t, c)`` intensities.
    :return: ``(t, c)`` gain.
    """
    g = np.quantile(x, q, axis=0)  # (t, c)
    med = np.median(g, axis=0, keepdims=True)
    return np.where((g > 0) & (med > 0), g / np.where(med > 0, med, 1.0), 1.0)


def _em_split_levels(x: NDArray, min_snr: float) -> tuple[NDArray, NDArray, NDArray]:
    """Split each read's cycles into low and high groups per channel, without calls.

    For each read and channel, the cycles are split at the point that maximizes the
    between-group variance (Otsu's method on the read's T values).  The split is
    accepted when the gap between the group means is at least ``min_snr`` times
    the pooled within-group standard deviation; otherwise the channel is treated as
    never on in that read.

    :param x: ``(read, t, c)`` intensities.
    :return: ``(b, gap, ok)``, each ``(read, c)``: background (low-group mean, or
        the mean of all cycles when the split is rejected), gap between group means,
        and whether the split was accepted.
    """
    v = np.sort(x, axis=1)  # (read, t, c)
    T = v.shape[1]
    k = np.arange(1, T)[np.newaxis, :, np.newaxis]  # size of the low group
    csum = np.cumsum(v, axis=1)
    total = csum[:, -1:, :]
    lo_mean = csum[:, :-1, :] / k
    hi_mean = (total - csum[:, :-1, :]) / (T - k)
    between = k * (T - k) / T * (hi_mean - lo_mean) ** 2
    best = between.argmax(axis=1)  # (read, c)
    take = best[:, np.newaxis, :]
    b = np.take_along_axis(lo_mean, take, axis=1)[:, 0, :]
    gap = np.take_along_axis(hi_mean, take, axis=1)[:, 0, :] - b
    rank = np.arange(T)[np.newaxis, :, np.newaxis]
    is_lo = rank <= best[:, np.newaxis, :]
    level = np.where(is_lo, b[:, np.newaxis, :], (b + gap)[:, np.newaxis, :])
    pooled_sd = np.sqrt(((v - level) ** 2).sum(axis=1) / max(T - 2, 1))
    ok = gap >= min_snr * np.maximum(pooled_sd, 1e-12)
    b = np.where(ok, b, x.mean(axis=1))
    return b, gap, ok


def _em_global_scale(x: NDArray, min_snr: float) -> NDArray:
    """Global bright scale per channel: median gap over reads whose split is accepted."""
    _, gap, ok = _em_split_levels(x, min_snr)
    S = np.array(
        [
            np.median(gap[ok[:, c], c]) if ok[:, c].any() else 1.0
            for c in range(x.shape[2])
        ]
    )
    return np.where(S > 0, S, 1.0)


def _em_initial_levels(
    x: NDArray, S: NDArray, min_snr: float, min_gap: float, k_low: NDArray
) -> tuple[NDArray, NDArray]:
    """Initial per-read background and bright level from :func:`_em_split_levels`.

    The background is the mean of the read's ``k_low[c]`` lowest cycles in each
    channel (about half the cycles the encoding expects to be off), rather than the
    low group's mean, which can include dim on cycles when a channel is fired by
    several bases with different dye brightness.  The bright level is the high
    group's mean above that background.

    A spot's brightness comes mostly from its size and amplification, which affect
    every dye alike, so each read gets one amplitude ``a`` (the median over its
    accepted channels of ``gap / S``).  Channels whose split is rejected, or whose
    gap is below ``min_gap * a * S`` (e.g. channels that are mostly clipped to 0),
    are treated as never on in the read and get ``a * S`` as their bright level, so
    they are neither stretched to look bright nor given another read's scale.

    :return: ``(b, s)``, each ``(read, c)``.
    """
    b_split, gap, ok = _em_split_levels(x, min_snr)
    v = np.sort(x, axis=1)
    rank = np.arange(x.shape[1])[np.newaxis, :, np.newaxis]
    b = np.where(rank < k_low, v, 0.0).sum(axis=1) / k_low  # (read, c)
    gap = np.where(ok, gap + b_split - b, gap)  # high-group mean minus background
    ok &= gap >= min_gap * S
    log_rel = np.where(ok, np.log(np.where(ok, gap, 1.0) / S), np.nan)
    n_ok = ok.sum(axis=1)
    a = np.ones(len(x))
    has = n_ok > 0
    if has.any():
        a[has] = np.exp(np.nanmedian(log_rel[has], axis=1))
    s_read = a[:, np.newaxis] * S
    ok &= gap >= min_gap * s_read
    return b, np.where(ok, gap, s_read)


def _em_update_levels(
    x: NDArray,
    w_on: NDArray,
    b0: NDArray,
    s0: NDArray,
    S: NDArray,
    n0: float,
    min_gap: float,
) -> tuple[NDArray, NDArray]:
    """Per-read background and bright level from the cycles the model calls off/on.

    The background is the posterior-weighted mean of the read's off cycles and the
    bright level the posterior-weighted mean of its on cycles above that
    background.  Each is shrunk with ``n0`` pseudo-cycles
    toward the read's initial levels ``b0`` and ``s0`` (which fall back to the global
    scale for channels that never fire), so reads with few on or off cycles in a
    channel keep sensible values, and the bright level is floored at ``min_gap * S``.

    :param x: ``(read, t, c)`` intensities.
    :param w_on: ``(read, t, c)`` posterior probability that each channel is on.
    :return: ``(b, s)``, each ``(read, c)``.
    """
    w_off = 1 - w_on
    b = ((w_off * x).sum(axis=1) + n0 * b0) / (w_off.sum(axis=1) + n0)
    d = x - b[:, np.newaxis, :]
    s = ((w_on * d).sum(axis=1) + n0 * s0) / (w_on.sum(axis=1) + n0)
    return b, np.maximum(s, min_gap * S)


def _em_update_amplitude(
    x: NDArray,
    post: NDArray,
    model: dict,
    b0: NDArray,
    s0: NDArray,
    n0: float,
) -> tuple[NDArray, NDArray]:
    """Per-read background and one amplitude shared across channels.

    The background is updated as in :func:`_em_update_levels`.  The bright level is
    ``a * S``, with a single amplitude ``a`` per read estimated from all of its on
    channels and cycles, relative to each base's fitted on-state mean.  In encodings
    where a channel is fired by several bases with different dye brightness (e.g.
    2-color, where A and T both fire one channel), a per-channel bright level would
    average over whichever of those bases the read contains, so the same cycle would
    be normalized differently depending on the rest of the read.

    :param x: ``(read, t, c)`` intensities, divided by the cycle gain.
    :param post: ``(read, t, base)`` posterior.
    :param b0: ``(read, c)`` initial background.
    :param s0: ``(read, c)`` initial bright level, used for the initial amplitude.
    :return: ``(b, s)``, each ``(read, c)``.
    """
    E, S = model["encoding"], model["S"]
    w_on = post @ E
    w_off = 1 - w_on
    b = ((w_off * x).sum(axis=1) + n0 * b0) / (w_off.sum(axis=1) + n0)
    log_a0 = np.median(np.log(s0 / S), axis=1)  # (read,)
    d = x - b[:, np.newaxis, :]
    pos = d > 0
    log_d = np.log(np.where(pos, d, 1.0))[..., np.newaxis, :]  # (read, t, 1, c)
    mu_on = np.where(E > 0, model["mu"], 0.0)  # (base, c), log space
    resid = log_d - np.log(S) - mu_on  # (read, t, base, c)
    wt = post[..., np.newaxis] * (E > 0) * pos[..., np.newaxis, :]
    log_a = ((wt * resid).sum(axis=(1, 2, 3)) + n0 * log_a0) / (
        wt.sum(axis=(1, 2, 3)) + n0
    )
    return b, np.exp(log_a)[:, np.newaxis] * S


def _em_signal(x: NDArray, b: NDArray, s: NDArray) -> NDArray:
    """Normalized signal ``(x - b) / s``: on near 1, off near 0."""
    return (x - b[:, np.newaxis, :]) / s[:, np.newaxis, :]


_EM_U_RANGE = (-1.0, 3.0)  # support of the outlier component, in normalized units


def _em_values(u: NDArray, E: NDArray) -> tuple[NDArray, NDArray]:
    """Values modelled by each base's channel states, ``(read, t, base, c)``.

    Off states model ``u`` directly; on states model ``log u`` (lognormal), since on
    signal varies multiplicatively and a symmetric on state would put spurious
    density near 0, where it captures dark cycles.  ``u`` is clipped to
    :data:`_EM_U_RANGE`.

    :return: ``(val, valid)``: modelled value, and whether it is in the state's
        support (on states need ``u > 0``).
    """
    lo, hi = _EM_U_RANGE
    uc = np.clip(u, lo, hi)[..., np.newaxis, :]
    on = E > 0  # (base, c)
    pos = uc > 0
    val = np.where(on, np.log(np.where(pos, uc, 1.0)), uc)
    valid = np.where(on, pos, True)
    return val, valid


def _em_robust_logpdf(
    val: NDArray,
    valid: NDArray,
    mu: NDArray,
    sd: NDArray,
    jac: NDArray | float,
    eps: float,
) -> tuple[NDArray, NDArray]:
    """Elementwise robust log density ``(1 - eps) * f + eps * U`` and inlier weight.

    ``f`` is a Gaussian in ``val`` with mean ``mu`` and standard deviation ``sd``
    (minus the Jacobian ``jac``, ``val`` for lognormal on states), and ``U`` is
    uniform on :data:`_EM_U_RANGE`.  Values outside a state's support (``~valid``)
    get density ``eps * U`` and inlier weight 0.
    """
    lo, hi = _EM_U_RANGE
    const = np.log1p(-eps) - np.log(sd) - 0.5 * np.log(2 * np.pi)
    log_in = const - 0.5 * ((val - mu) / sd) ** 2 - jac
    log_in = np.where(valid, log_in, -np.inf)
    log_out = np.log(eps) - np.log(hi - lo) if eps > 0 else -np.inf
    log_p = np.logaddexp(log_in, log_out)
    return log_p, np.exp(log_in - log_p)


def _em_channel_logpdf(
    u: NDArray,
    model: dict,
    val: NDArray | None = None,
    valid: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Robust per-channel log densities ``(read, t, base, c)`` and inlier weights.

    Each channel's density is ``(1 - eps) * f + eps * U``, with ``f`` a Gaussian in
    ``u`` for off states and a lognormal for on states (:func:`_em_values`), and
    ``U`` uniform on :data:`_EM_U_RANGE`, so a few outlying values (saturated
    pixels, misestimated read levels) cannot dominate the fit.  The inlier weight is
    the posterior probability of ``f``.

    :param val: ``val, valid`` from :func:`_em_values`, if already computed.
    """
    E = model["encoding"]
    if val is None or valid is None:
        val, valid = _em_values(u, E)
    jac = np.where(E > 0, val, 0.0)  # Jacobian of log u for on states
    return _em_robust_logpdf(
        val, valid, model["mu"], model["sd"], jac, model["outlier"]
    )


def _em_tied_values(u: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Modelled values of the on and off states per channel, ``(read, t, c)``.

    The same values as :func:`_em_values`, without the base axis: ``log u`` for on
    states and ``u`` for off states, ``u`` clipped to :data:`_EM_U_RANGE`.

    :return: ``(val_on, valid_on, val_off)``.
    """
    lo, hi = _EM_U_RANGE
    uc = np.clip(u, lo, hi)
    pos = uc > 0
    return np.log(np.where(pos, uc, 1.0)), pos, uc


def _em_tied_params(model: dict) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Per-channel on and off ``(mu, sd)`` of a tied model, each ``(c,)``.

    Channels without an on (or off) base get the values of the other state; they
    receive no weight.
    """
    on = model["encoding"] > 0
    mu, sd = model["mu"], model["sd"]
    first_on = np.where(on.any(axis=0), on.argmax(axis=0), 0)
    first_off = np.where((~on).any(axis=0), (~on).argmax(axis=0), 0)
    cols = np.arange(on.shape[1])
    return (
        mu[first_on, cols],
        sd[first_on, cols],
        mu[first_off, cols],
        sd[first_off, cols],
    )


def _em_is_tied(model: dict) -> bool:
    """Whether every base shares one on and one off state per channel."""
    if "cov" in model:
        return False
    on = model["encoding"] > 0
    mu_on, sd_on, mu_off, sd_off = _em_tied_params(model)
    return bool(
        np.array_equal(model["mu"], np.where(on, mu_on, mu_off))
        and np.array_equal(model["sd"], np.where(on, sd_on, sd_off))
    )


def _em_tied_logpdf(
    u: NDArray, model: dict, vals: tuple[NDArray, NDArray, NDArray] | None = None
) -> tuple[NDArray, NDArray, NDArray]:
    """:func:`_em_channel_logpdf` for a tied model, computed per state not per base.

    A tied model has one on and one off state per channel, so the densities are
    computed once per state on ``(read, t, c)`` arrays, a factor ``n_bases / 2``
    less work, and summed over channels with the encoding.

    :param vals: Output of :func:`_em_tied_values`, if already computed.
    :return: ``(log_p, inl_on, inl_off)``: ``(read, t, base)`` log densities summed
        over channels and the ``(read, t, c)`` inlier weights of each state.
    """
    E = model["encoding"]
    eps = model["outlier"]
    val_on, valid_on, val_off = _em_tied_values(u) if vals is None else vals
    mu_on, sd_on, mu_off, sd_off = _em_tied_params(model)
    lp_on, inl_on = _em_robust_logpdf(val_on, valid_on, mu_on, sd_on, val_on, eps)
    lp_off, inl_off = _em_robust_logpdf(val_off, True, mu_off, sd_off, 0.0, eps)
    if eps > 0:  # finite densities
        log_p = lp_on @ E.T + lp_off @ (1 - E).T
    else:  # avoid 0 * -inf
        on = E > 0
        log_p = np.where(on, lp_on[..., np.newaxis, :], lp_off[..., np.newaxis, :])
        log_p = log_p.sum(axis=-1)
    return log_p, inl_on, inl_off


def _em_joint_logpdf(
    u: NDArray,
    model: dict,
    val: NDArray | None = None,
    valid: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Robust joint log densities ``(read, t, base)`` and inlier weights.

    Like :func:`_em_channel_logpdf` but with a full covariance ``cov[b]`` across
    channels per base (in the space of :func:`_em_values`), so channels that move
    together for a base (e.g. both channels of a base that fires two) are modelled
    jointly.  With ``t_dof``, each base's density is a multivariate t rather than a
    Gaussian: a single wild channel then inflates the distance to every base but
    the bases stay ranked by the rest of the cycle, whereas a per-cycle outlier
    component alone would make the call uninformative.  The uniform outlier
    component, per cycle, remains for values outside a state's support.

    :param val: ``val, valid`` from :func:`_em_values`, if already computed.
    :return: ``(log_p, weight)``: log densities and M-step weights (inlier
        probability times the t-distribution weight).
    """
    from scipy.special import gammaln

    lo, hi = _EM_U_RANGE
    eps = model["outlier"]
    nu = model.get("t_dof")
    E = model["encoding"]
    if val is None or valid is None:
        val, valid = _em_values(u, E)  # (read, t, base, c)
    C = val.shape[-1]
    chol = np.linalg.cholesky(model["cov"])  # (base, c, c)
    # whiten with the inverse factor of each base: one small inverse per base
    # instead of a batched solve per value
    chol_inv = np.linalg.inv(chol)
    diff = val - model["mu"]
    z = np.einsum("bcd,ntbd->ntbc", chol_inv, diff, optimize=True)
    q = (z**2).sum(axis=-1)  # squared Mahalanobis distance
    logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)  # (base,)
    if nu is None:
        log_f = -0.5 * q - 0.5 * C * np.log(2 * np.pi)
        tau = np.ones_like(q)
    else:  # multivariate t
        log_f = (
            gammaln((nu + C) / 2)
            - gammaln(nu / 2)
            - 0.5 * C * np.log(nu * np.pi)
            - 0.5 * (nu + C) * np.log1p(q / nu)
        )
        tau = (nu + C) / (nu + q)
    log_f = log_f - 0.5 * logdet - (np.where(E > 0, val, 0.0)).sum(axis=-1)
    log_f = np.where(valid.all(axis=-1), log_f, -np.inf)
    log_in = np.log1p(-eps) + log_f
    log_out = np.log(eps) - C * np.log(hi - lo) if eps > 0 else -np.inf
    log_p = np.logaddexp(log_in, log_out)
    return log_p, np.exp(log_in - log_p) * tau


def _em_loglik(u: NDArray, model: dict) -> NDArray:
    """``(read, t, c)`` normalized signal -> ``(read, t, base)`` log P(signal | base).

    Robust per-base off (Gaussian) and on (lognormal) states, with a full covariance
    across channels (:func:`_em_joint_logpdf`) when the model has ``cov``, and
    channels independent given the base (:func:`_em_channel_logpdf`) otherwise.
    """
    if "cov" in model:
        return _em_joint_logpdf(u, model)[0]
    if _em_is_tied(model):
        return _em_tied_logpdf(u, model)[0]
    return _em_channel_logpdf(u, model)[0].sum(axis=-1)


def _em_logsumexp_softmax(L: NDArray) -> tuple[NDArray, NDArray]:
    """``logsumexp`` and ``softmax`` of ``L`` over the last axis, in one pass."""
    m = L.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(L - m)
    s = e.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(s) + m)[..., 0], e / s


def _em_run(
    u: NDArray,
    E: NDArray,
    model: dict,
    n_iter: int,
    tol: float,
    var_floor: float,
    tied: bool,
    anneal: bool,
    full: bool = False,
    fixed_w: NDArray | None = None,
) -> tuple[dict, float]:
    """EM for the encoding-structured mixture from one starting point.

    With ``tied``, each channel has one on and one off state shared across bases;
    otherwise each base has its own parameters per channel, and with ``full`` a
    full covariance across channels per base.  Parameters are weighted by the
    inlier probability of each value, so outliers barely move them.  With
    ``fixed_w`` (``(read, t, base)``, e.g. one-hot labels from whitelist-matched
    reads), the base posteriors are held fixed and only the parameters (and
    their inlier weights) are updated.
    """
    model = {
        k: np.array(v, dtype=float) if isinstance(v, np.ndarray) else v
        for k, v in model.items()
    }
    model["encoding"] = E
    on = E > 0
    tied = tied and not full
    if tied:
        # one on and one off state per channel: work on (read, t, c) arrays
        vals = _em_tied_values(u)
        mu_on, sd_on, mu_off, sd_off = _em_tied_params(model)
        model["mu"] = np.where(on, mu_on, mu_off)
        model["sd"] = np.where(on, sd_on, sd_off)
    else:
        val, valid = _em_values(u, E)
    n_anneal = max(1, n_iter // 4) if anneal else 0
    prev = -np.inf
    ll = -np.inf
    for it in range(n_iter):
        beta = min(1.0, 0.5 + 0.5 * it / n_anneal) if n_anneal else 1.0
        if tied:
            log_p, inl_on, inl_off = _em_tied_logpdf(u, model, vals)
        elif full:
            log_p, inlier = _em_joint_logpdf(u, model, val, valid)
        else:
            log_p, inlier = _em_channel_logpdf(u, model, val, valid)
            log_p = log_p.sum(axis=-1)
        L = log_p + np.log(model["pi"])[np.newaxis]
        lse, post = _em_logsumexp_softmax(L)
        ll = float(lse.mean())
        if beta == 1.0 and abs(ll - prev) < tol * max(1.0, abs(prev)):
            break
        prev = ll if beta == 1.0 else -np.inf
        if fixed_w is not None:
            w = fixed_w
        elif beta == 1.0:
            w = post
        else:
            w = softmax(beta * L, axis=-1)
        pi = w.mean(axis=0) + 1e-6
        model["pi"] = pi / pi.sum(axis=1, keepdims=True)
        if full:
            wb = w * inlier  # (read, t, base): includes the t weight
            tot = wb.sum(axis=(0, 1)) + 1e-12
            mu = np.einsum("ntb,ntbc->bc", wb, val) / tot[:, np.newaxis]
            dv = val - mu
            n_b = (w * (inlier > 0)).sum(axis=(0, 1)) + 1e-12
            cov = np.einsum("ntb,ntbc,ntbd->bcd", wb, dv, dv) / n_b[:, None, None]
            cov += var_floor * np.eye(val.shape[-1])
            model["mu"], model["cov"] = mu, cov
            model["sd"] = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
            continue
        if tied:  # pool on states and off states across bases, per channel
            val_on, _, val_off = vals
            mu = np.zeros(E.shape)
            var = np.zeros(E.shape)
            for state, w_s, v_s, inl in (
                (on, w @ E, val_on, inl_on),
                (~on, w @ (1 - E), val_off, inl_off),
            ):
                wc = w_s * inl  # (read, t, c)
                n_s = wc.sum(axis=(0, 1)) + 1e-12
                m_s = (wc * v_s).sum(axis=(0, 1)) / n_s
                var_s = (wc * v_s**2).sum(axis=(0, 1)) / n_s - m_s**2
                mu = np.where(state, m_s, mu)
                var = np.where(state, var_s, var)
        else:
            wc = w[..., np.newaxis] * inlier  # (read, t, base, c)
            sw = wc.sum(axis=(0, 1))  # (base, c)
            sx = np.einsum("ntbc,ntbc->bc", wc, val)
            sxx = np.einsum("ntbc,ntbc->bc", wc, val**2)
            mu = sx / (sw + 1e-12)
            var = sxx / (sw + 1e-12) - mu**2
        model["mu"] = mu
        model["sd"] = np.sqrt(np.maximum(var, var_floor))
    return model, ll


def _em_posterior(x: NDArray, model: dict, n_rescale: int) -> NDArray:
    """Posterior P(base | signal) with per-read levels refined from the model.

    :param x: ``(read, t, c)`` intensities, divided by the cycle gain.
    :return: ``(read, t, base)`` posterior.
    """
    return _em_posterior_signal(x, model, n_rescale)[0]


def _em_posterior_signal(
    x: NDArray, model: dict, n_rescale: int
) -> tuple[NDArray, NDArray]:
    """:func:`_em_posterior`, also returning the normalized signal ``u``.

    :return: ``(post, u)``: ``(read, t, base)`` posterior and ``(read, t, c)``
        signal at the final per-read levels.
    """
    E = model["encoding"]
    b0, s0 = _em_initial_levels(
        x, model["S"], model["min_snr"], model["min_gap"], model["k_low"]
    )
    b, s = b0, s0
    for it in range(n_rescale + 1):
        u = _em_signal(x, b, s)
        post = softmax(_em_loglik(u, model) + np.log(model["pi"])[np.newaxis], axis=-1)
        if it < n_rescale:
            if model.get("read_scale", "channel") == "shared":
                b, s = _em_update_amplitude(x, post, model, b0, s0, model["n0"])
            else:
                b, s = _em_update_levels(
                    x, post @ E, b0, s0, model["S"], model["n0"], model["min_gap"]
                )
    return post, u


def fit_base_mixture(
    spots: xr.DataArray,
    encoding: NDArray | None = None,
    base_labels: list[str] | NDArray | None = None,
    dark_bases: list[str] | None = None,
    n_reads: int = 500_000,
    fit_peak_quantile: float | None = 0.5,
    n_iter: int = 200,
    n_restarts: int = 4,
    n_rescale: int = 3,
    tied: bool = False,
    covariance: Literal["full", "diag"] = "full",
    t_dof: float | None = 4.0,
    read_scale: Literal["shared", "channel"] = "shared",
    tol: float = 1e-7,
    min_snr: float = 4.0,
    min_gap: float = 0.2,
    outlier: float = 0.01,
    var_floor: float = 1e-3,
    n0: float = 1.0,
    whitelist: Sequence[str] | None = None,
    n_whitelist_rounds: int = 3,
    xtalk_tol: float | None = 0.15,
    seed: int = 0,
) -> dict:
    """Fit an unsupervised model of log P(signal | base), without a whitelist or calls.

    Each read's signal is normalized as :math:`u_{c,t} = (x_{c,t} / g_{c,t} - b_c) /
    s_c`, with a per-cycle gain :math:`g_{c,t}` and a per-read background
    :math:`b_c` and bright level :math:`s_c` per channel, so on channels sit near 1
    and off channels near 0.  Channels are independent given the base, with a
    Gaussian off state and a lognormal on state (on signal varies
    multiplicatively, and a symmetric on state would put spurious density near 0,
    where it captures dark cycles):

    .. math::

        \\log P(\\mathbf{u} \\mid b) = \\sum_c \\log \\Big[(1 - \\epsilon)\\,
        f_{b,c}(u_c) + \\epsilon\\, U(u_c)\\Big], \\quad
        f_{b,c} = \\begin{cases}
            \\text{LogNormal}(\\mu_{b,c}, \\sigma_{b,c}) & E[b,c] = 1 \\\\
            \\mathcal{N}(\\mu_{b,c}, \\sigma_{b,c}) & E[b,c] = 0
        \\end{cases}

    with a small uniform outlier component :math:`U` so saturated pixels or
    misestimated read levels cannot dominate the fit.  With ``covariance="full"``
    (the default), the per-base stage instead models each base's channels jointly,
    with a full covariance in the same space (log for on channels, linear for off)
    and a multivariate t density (``t_dof``), so channels that move together for a
    base, such as both channels of a base that fires two, are modelled jointly,
    while a single wild channel only lowers the fit of every base instead of making
    the call uninformative.

    The encoding matrix ``E`` gives the structure that makes this identifiable
    without labels: fitting starts from a *tied* model in which each channel has one
    on and one off state shared across bases, and is then released to per-base
    parameters (unless ``tied``) warm-started from it.  Dark bases are the all-off
    rows of ``E`` and need no special handling or radius threshold; the dark/bright
    boundary is where the fitted posteriors cross.

    Fitting uses no whitelist or other decoder:

    * **Fit reads**: a random sample of up to ``n_reads`` reads, restricted to reads
      with ``peak`` at or above the ``fit_peak_quantile`` quantile when ``spots``
      has a ``peak`` coordinate, since low-peak spots are mostly not real reads.
    * **Crosstalk** (``xtalk_tol``): for orthogonal encodings (each channel fired
      by exactly one base, as in 4-color or 3-color with a dark base), the
      crosstalk matrix of :func:`channel_crosstalk_matrix` (median method) is
      estimated on the fit reads.  It is applied, in fitting and calling, only when
      its largest off-diagonal term relative to the diagonal exceeds
      ``xtalk_tol``, so spots already corrected with
      :func:`apply_channel_crosstalk_matrix` are not corrected twice (which costs
      0.1 to 0.3 points of mapping rate on the scallops test data).  Non-orthogonal
      encodings are not corrected, since the median method would take a base that
      fires several channels (2-color A) for crosstalk.
    * **Per-cycle gain** (:func:`_em_cycle_gain`): every cycle has a similar mix of
      bases, so a high quantile of each cycle's intensities tracks its brightness.
      Intensities are not clipped at 0, since values pushed negative by xtalk
      correction are off-state noise.
    * **Initial levels** (:func:`_em_initial_levels`): the background is the mean of
      the read's lowest cycles per channel (about half the cycles the encoding
      expects to be off), and the bright level comes from splitting the read's
      cycles into low and high groups (Otsu's method).  Brightness is shared across
      dyes (it comes mostly from spot size and amplification), so channels whose
      split is rejected get the read's amplitude times the channel's scale.
    * **Initialization** from the signal alone: a 1-D two-component mixture per
      channel and a quantile split at the expected on-fraction
      :math:`\\phi_c = \\sum_b E[b,c] / n_\\text{bases}`, plus jittered copies.
      The tied model is fit by annealed EM from each, keeping the highest
      log-likelihood.
    * **Level refinement**: ``n_rescale`` times, each read's background and bright
      level are re-estimated from the cycles the model calls off and on and the
      model is refit.  With ``read_scale="shared"`` (the default), the bright level
      is one amplitude per read times each channel's scale, estimated relative to
      each base's fitted on-state mean (:func:`_em_update_amplitude`), so a channel
      fired by several bases with different dye brightness is normalized the same
      way whatever else the read contains; ``"channel"`` estimates it per channel
      (:func:`_em_update_levels`).  Calls repeat the same
      refinement per read, so they do not depend on Dask chunking.

    :param spots: Spots DataArray with dims ``(read, t, c)``, with or without xtalk
        correction (see ``xtalk_tol``).
    :param encoding: ``(n_bases, n_channels)`` binary encoding matrix.  All-zero
        rows indicate dark bases.  Rows must be distinct.
    :param base_labels: Base labels of length ``n_bases``, required with
        ``encoding``.
    :param dark_bases: Shorthand for 1:1 channel→base chemistry with named dark
        bases, via :func:`make_encoding`.
    :param n_reads: Number of reads sampled to fit the model.
    :param fit_peak_quantile: Fit only on reads whose ``peak`` is at or above this
        quantile (``None`` to use all reads).
    :param n_iter: Maximum EM iterations per fit.
    :param n_restarts: Number of starting points for the tied model (at least the
        mixture and quantile starts; the rest are jittered copies).  The restarts
        run in parallel as tasks on the current Dask scheduler, so peak memory grows
        with the number running at once.
    :param n_rescale: Rounds of per-read level refinement, in fitting and calling.
    :param tied: Keep one on and one off state per channel shared across bases
        instead of releasing to per-base parameters.
    :param covariance: ``"full"`` to model each base's channels jointly in the
        per-base stage, or ``"diag"`` for channels independent given the base.
    :param t_dof: Degrees of freedom of the multivariate t used with
        ``covariance="full"``; ``None`` for a Gaussian.
    :param read_scale: ``"shared"`` for one amplitude per read across channels, or
        ``"channel"`` for a bright level per read and channel.
    :param tol: Relative change in mean log-likelihood at which EM stops.
    :param min_snr: Minimum gap between a read's low and high groups, in units of
        their pooled standard deviation, for a channel to count as on in the read.
    :param min_gap: Minimum gap, as a fraction of the read's expected bright level,
        for a channel to count as on in the read; also floors refined bright levels.
    :param outlier: Weight :math:`\\epsilon` of the uniform outlier component.
    :param var_floor: Minimum variance of each state.
    :param n0: Pseudo-cycles shrinking refined per-read levels toward the initial
        levels.
    :param whitelist: Optional barcode whitelist.  After the unsupervised fit, the
        model is refit ``n_whitelist_rounds`` times on the fit reads whose calls
        match it, holding their calls fixed: those calls are correct by
        construction, so they act as labelled training data.  Only barcodes as long
        as the number of cycles are used.  ``None`` (default) fits without labels.
    :param n_whitelist_rounds: Number of refits on whitelist-matched reads.
    :param xtalk_tol: Largest off-diagonal term of the estimated crosstalk matrix,
        relative to its diagonal, that is left uncorrected; ``None`` to never
        correct.  On the scallops test data, corrected spots measure 0.03 to 0.06
        and uncorrected spots 0.43 to 0.55.  Ignored for non-orthogonal encodings.
    :param seed: Random seed for read sampling and jitter.
    :return: Model dict with per-base, per-channel ``mu`` and ``sd`` (log space for
        on states), per-base ``cov`` and ``t_dof`` with ``covariance="full"``,
        per-cycle base frequencies ``pi`` ``(t, base)``, the crosstalk matrix
        ``xtalk`` applied to the spots (``None`` if not corrected), the per-cycle
        ``gain``, level parameters ``S``, ``k_low``, ``min_snr``, ``min_gap``, ``n0``,
        ``n_rescale``, ``read_scale``, ``outlier``, the ``encoding`` and
        ``base_labels``, the mean log-likelihood ``loglik``, and the number of
        whitelist-matched fit reads per refit, ``n_whitelist_matched``.
    """
    from sklearn.mixture import GaussianMixture

    E, labels = _resolve_encoding(spots, encoding, base_labels, dark_bases)
    if len(np.unique(E, axis=0)) != len(E):
        raise ValueError(
            "Rows of the encoding must be distinct; identical rows (e.g. several "
            "dark bases) cannot be told apart"
        )
    B, n_c = E.shape
    phi = E.sum(axis=0) / B

    candidates = np.arange(spots.sizes["read"])
    if fit_peak_quantile is not None and "peak" in spots.coords:
        peak = np.asarray(spots["peak"].values)
        candidates = np.flatnonzero(peak >= np.quantile(peak, fit_peak_quantile))
    rs = random.Random(seed)
    sample = np.sort(rs.sample(list(candidates), min(n_reads, len(candidates))))
    sp = spots.isel(read=sample).data
    if isinstance(sp, da.Array):
        sp = sp.compute()
    # not clipped at 0: values pushed negative by xtalk correction are off-state
    # noise, and clipping them would pile them into a spike the off state collapses on
    x = np.asarray(sp, dtype=float)
    T = x.shape[1]
    xtalk = None
    orthogonal = (E.sum(axis=1) <= 1).all() and (E.sum(axis=0) == 1).all()
    if xtalk_tol is not None and orthogonal and n_c > 1:
        w = _correct_channel_crosstalk_median(x.reshape(-1, n_c))
        resid = np.abs(w / np.diag(w)[:, np.newaxis] - np.eye(n_c)).max()
        if resid > xtalk_tol:
            logger.info(f"Correcting residual crosstalk ({resid:.2f}) in the spots")
            xtalk = w
            x = x @ w.T
    gain = _em_cycle_gain(x)
    x = x / gain[np.newaxis]

    S = _em_global_scale(x, min_snr)
    # background from about half the cycles the encoding expects to be off
    k_low = np.maximum(1, np.floor(T * (1 - phi) / 2)).astype(int)
    b0, s0 = _em_initial_levels(x, S, min_snr, min_gap, k_low)
    s = s0
    u = _em_signal(x, b0, s)

    # Starting points for the tied model, from the signal alone
    rng = np.random.default_rng(seed)

    gm = np.zeros((4, n_c))  # rows: m, z, s_on, s_off
    qt = np.zeros((4, n_c))
    for c in range(n_c):
        v = u[..., c].ravel()
        v_fit = rng.choice(v, size=min(len(v), 200_000), replace=False)[:, np.newaxis]
        g2 = GaussianMixture(2, random_state=seed).fit(v_fit)
        g1 = GaussianMixture(1, random_state=seed).fit(v_fit)
        if g2.bic(v_fit) >= g1.bic(v_fit):
            logger.warning(
                f"Channel {spots.c.values[c]} does not look bimodal; on and off "
                "states may be hard to separate"
            )
        off, on = np.argsort(g2.means_.ravel())
        sd = np.sqrt(g2.covariances_.ravel())
        gm[:, c] = g2.means_[on, 0], g2.means_[off, 0], sd[on], sd[off]
        cut = np.quantile(v, 1 - phi[c]) if phi[c] > 0 else np.inf
        hi, lo = v[v > cut], v[v <= cut]
        qt[:, c] = (
            hi.mean() if len(hi) else 1.0,
            lo.mean(),
            hi.std() if len(hi) > 1 else 0.3,
            lo.std(),
        )
    starts = [gm, qt]
    for i in range(max(0, n_restarts - 2)):
        jitter = rng.uniform([[0.9], [0.9], [0.8], [0.8]], [[1.1], [1.1], [1.2], [1.2]])
        starts.append(starts[i % 2] * jitter)

    def tied_model(p: NDArray) -> dict:
        m, z, s_on, s_off = p
        # on states are lognormal: match the linear-space mean and sd
        m = np.maximum(m, 0.05)
        s_log = np.sqrt(np.log1p((s_on / m) ** 2))
        m_log = np.log(m) - s_log**2 / 2
        return {
            "mu": E * m_log + (1 - E) * z,
            "sd": np.sqrt(np.maximum((E * s_log + (1 - E) * s_off) ** 2, var_floor)),
            "pi": np.full((T, B), 1.0 / B),
            "outlier": outlier,
            "encoding": E,
        }

    # the restarts are independent: run them as tasks on the current dask
    # scheduler, keeping the first with the highest log-likelihood
    from distributed import get_client

    try:  # send u to the workers once, rather than in the graph
        u_task = get_client().scatter(u, broadcast=True, hash=False)
    except ValueError:  # no distributed client
        u_task = delayed(u)  # one graph node shared by every restart
    em_run = delayed(_em_run, pure=True)
    fits = dask.compute(
        *[
            em_run(
                u_task, E, tied_model(p), n_iter, tol, var_floor, tied=True, anneal=True
            )
            for p in starts[: max(n_restarts, 2)]
        ]
    )
    best, best_ll = max(
        fits, key=lambda fit: fit[1] if np.isfinite(fit[1]) else -np.inf
    )
    best.update(
        encoding=E,
        S=S,
        gain=gain,
        xtalk=xtalk,
        min_snr=min_snr,
        min_gap=min_gap,
        n0=n0,
        read_scale=read_scale,
        k_low=k_low,
    )

    # Refine per-read levels from the model and refit; then
    # release to per-base parameters
    stages = [True] * n_rescale + ([] if tied else [False] * max(1, n_rescale))
    b = b0
    for stage_tied in stages:
        u = _em_signal(x, b, s)
        post = softmax(_em_loglik(u, best) + np.log(best["pi"])[np.newaxis], axis=-1)
        if read_scale == "shared":
            b, s = _em_update_amplitude(x, post, best, b0, s0, n0)
            # the overall scale is shared with the on-state means: pin its median
            shift = np.median(np.log(s / best["S"])) - np.median(np.log(s0 / best["S"]))
            s = s * np.exp(-shift)
        else:
            b, s = _em_update_levels(x, post @ E, b0, s0, best["S"], n0, min_gap)
        full = covariance == "full" and not stage_tied
        params = {k: best[k] for k in ("mu", "sd", "pi", "outlier")}
        if full:
            params["cov"] = best.get(
                "cov", np.stack([np.diag(v**2) for v in best["sd"]])
            )
            params["t_dof"] = t_dof
        refit, best_ll = _em_run(
            _em_signal(x, b, s),
            E,
            params,
            n_iter,
            tol,
            var_floor,
            tied=stage_tied,
            anneal=False,
            full=full,
        )
        best.update(refit)

    best.update(base_labels=labels, loglik=best_ll, n_rescale=n_rescale, tied=tied)

    # Optionally refit on reads whose calls match the whitelist: their calls are
    # correct by construction, so they serve as labelled training data
    best["n_whitelist_matched"] = []
    if whitelist is not None:
        wl = {bc for bc in whitelist if len(bc) == T}
        if not wl:
            logger.warning(f"No whitelist barcodes of length {T}; fitting unsupervised")
        for _ in range(n_whitelist_rounds if wl else 0):
            post, u = _em_posterior_signal(x, best, n_rescale)
            idx = post.argmax(axis=-1)
            matched = np.isin(["".join(r) for r in labels[idx]], list(wl))
            best["n_whitelist_matched"].append(int(matched.sum()))
            if matched.sum() < 10:
                logger.warning("Too few whitelist matches to refit on; stopping")
                break
            fixed_w = np.eye(B)[idx[matched]]  # (read, t, base) one-hot
            full = covariance == "full" and not tied
            params = {k: best[k] for k in ("mu", "sd", "pi", "outlier")}
            if full:
                params["cov"] = best.get(
                    "cov", np.stack([np.diag(v**2) for v in best["sd"]])
                )
                params["t_dof"] = t_dof
            refit, best_ll = _em_run(
                u[matched],
                E,
                params,
                n_iter,
                tol,
                var_floor,
                tied=tied,
                anneal=False,
                full=full,
                fixed_w=fixed_w,
            )
            best.update(refit)
        best["loglik"] = best_ll
    return best


def _decode_em_chunk(
    spots: NDArray,
    model: dict,
    meta_df: pd.DataFrame,
    whitelist: NDArray | None,
    offset: slice | None,
    min_error: float = DEFAULT_MIN_P_ERROR,
) -> pd.DataFrame:
    """Call one chunk of spots with a model from :func:`fit_base_mixture`."""
    x = np.asarray(spots, dtype=float)
    if model.get("xtalk") is not None:
        x = x @ model["xtalk"].T
    x = x / model["gain"][np.newaxis]
    post = _em_posterior(x, model, model["n_rescale"])
    calls = model["base_labels"][post.argmax(axis=-1)]
    p_error = 1 - post.max(axis=-1)
    p_error[p_error < min_error] = min_error
    Q = -10 * np.log10(p_error)

    df = (
        meta_df.iloc[offset.start : offset.stop].copy()
        if offset is not None
        else meta_df.copy()
    )
    df.index.name = None
    df["barcode"] = ["".join(x) for x in calls]
    df["Q"] = list(Q)
    df["Q_mean"] = Q.mean(axis=1)
    df["Q_min"] = Q.min(axis=1)
    if whitelist is not None:
        df["barcode_match"] = df["barcode"].isin(whitelist)
    return df


def decode_em(
    spots: xr.DataArray,
    barcodes: pd.DataFrame | None = None,
    encoding: NDArray | None = None,
    base_labels: list[str] | NDArray | None = None,
    dark_bases: list[str] | None = None,
    model: dict | None = None,
    use_whitelist: bool = False,
    **fit_kwargs,
) -> pd.DataFrame | dd.DataFrame:
    """Call reads with a mixture model of log P(signal | base) fitted to the spots.

    Fits :func:`fit_base_mixture` (unless ``model`` is given) and calls each cycle
    as the base with the highest posterior.  Each base is a combination of channel
    states set by its row of the encoding matrix, off (Gaussian near 0) or on
    (lognormal near 1), fitted by EM after normalizing each read by its background
    and brightness.  Dark bases (all-zero rows) need no threshold, and
    non-orthogonal encodings (a base that fires several channels, as A does in
    2-color chemistry) are handled by modelling each base's channels jointly.  See
    :func:`fit_base_mixture` for the model and fitting details.

    No whitelist or other decoder is needed.  On the scallops test data, this maps
    more reads than :func:`decode_se` and :func:`decode_polar` on 4-color and
    3-color NIS-seq (dark G) data, and about as many as :func:`decode_polar` on
    2-color data (slightly more with ``use_whitelist``).

    **Inputs.** If ``spots`` has a ``peak`` coordinate (as from
    :func:`peaks_to_bases`), the model is fit on the higher-peak half of spots
    (``fit_peak_quantile``), since low-peak spots are mostly not real reads; fitting
    on all spots can lower the mapping rate substantially.  All reads are called.

    **Crosstalk.** Spots need not be xtalk-corrected first.  For orthogonal
    encodings (4-color, or 3-color with a dark base), :func:`fit_base_mixture`
    estimates the crosstalk matrix on its fit reads, as
    :func:`channel_crosstalk_matrix` does, and corrects the spots when the estimated
    crosstalk exceeds ``xtalk_tol``; spots already corrected with
    :func:`apply_channel_crosstalk_matrix` are left as they are.  Uncorrected spots
    then map about as well as corrected ones.  The matrix is stored in the model as
    ``xtalk`` and applied to every chunk, so calls do not depend on chunking.

    Crosstalk that remains (non-orthogonal encodings, or residual bleed after
    correction) matters much less than for :func:`decode_se` or
    :func:`decode_max`, because the model absorbs crosstalk itself.  Bleed-through
    from a base's dye into another channel makes that channel's *off* state for the
    base sit above zero, and each base has its own mean (and, with
    ``covariance="full"``, its own covariance) per channel, so the bleed simply
    becomes part of the base's fitted signature: a base is identified by its whole
    pattern across channels, not by which channel is brightest.  Correction is an
    invertible linear transform, so it adds little information for a model that
    learns each base's pattern (it is not exactly neutral, because the model and
    its per-read normalization act on each channel separately); decoders that
    compare channels directly need it.

    Absorbing crosstalk needs per-base off states, which matters when the built-in
    correction is off (``xtalk_tol=None``) or does not apply.  The fit starts from a
    *tied* model, in which each channel has a single off state shared by every base;
    there the bleed cannot be represented (channel T's off state would have to fit
    both A cycles, with bleed, and C and G cycles, without), so it only inflates the
    off state's spread.  The second stage gives each base its own parameters, and
    the bleed becomes part of the base's signature.

    **Whitelist mode.** With ``use_whitelist=True`` and ``barcodes``, the model is
    refit on the fit reads whose calls match the whitelist, holding those calls
    fixed (see ``whitelist`` in :func:`fit_base_mixture`).  This usually helps a
    little, but the whitelist then both trains the model and scores it, so mapping
    rates measured against the same whitelist favor this mode somewhat.

    **Q.** The phred score of each call's posterior probability given the read's
    signal, ``-10 log10(1 - P(base | signal))``, capped at 60.

    Useful ``fit_kwargs``: ``fit_peak_quantile``, ``n_reads`` (reads sampled to fit,
    default 50 000), ``covariance`` (``"full"`` or ``"diag"``), ``t_dof``,
    ``read_scale``, ``xtalk_tol`` and ``seed``.  To call several datasets with one
    model, or to inspect it, fit it once with :func:`fit_base_mixture` and pass it
    as ``model``; ``encoding``, ``base_labels``, ``dark_bases``, ``use_whitelist``
    and ``fit_kwargs`` are then ignored.

    :example:

    .. code-block:: python

        from scallops.reads import decode_em, fit_base_mixture, read_statistics

        # 3-color NIS-seq: A, T and C channels, G dark
        reads = decode_em(spots, barcodes=barcodes, dark_bases=["G"])

        # 2-color: rows G (dark), T (ch1), A (both), C (ch0)
        E = [[0, 0], [0, 1], [1, 1], [1, 0]]
        reads = decode_em(
            spots_2col,
            barcodes=barcodes,
            encoding=E,
            base_labels=["G", "T", "A", "C"],
            use_whitelist=True,
        )

        # fit once, inspect, and reuse
        model = fit_base_mixture(spots, dark_bases=["G"])
        print(dict(zip(model["base_labels"], model["pi"].mean(axis=0))))
        reads = decode_em(spots, barcodes=barcodes, model=model)
        read_statistics(reads)

    :param spots: Spots DataArray with dims ``(read, t, c)``, with or without xtalk
        correction (see **Crosstalk** above).
    :param barcodes: Whitelist DataFrame with column ``'barcode'``.  Adds the
        ``barcode_match`` column, and is used for fitting only with
        ``use_whitelist``.
    :param encoding: ``(n_bases, n_channels)`` binary encoding matrix.  All-zero
        rows indicate dark bases (at most one, since dark bases cannot be told
        apart).  When omitted with ``dark_bases`` also omitted, defaults to one base
        per channel, named by ``spots.c``.
    :param base_labels: Base labels of length ``n_bases``, required with
        ``encoding``.
    :param dark_bases: Shorthand for one base per channel (named by ``spots.c``)
        plus the named dark bases, via :func:`make_encoding`.
    :param model: A model from :func:`fit_base_mixture`; fitted if omitted.
    :param use_whitelist: When fitting, also refit the model on reads whose calls
        match ``barcodes``.  Off by default, so calls need no whitelist.  Ignored
        without ``barcodes`` or with ``model``.
    :param fit_kwargs: Passed to :func:`fit_base_mixture`.
    :return: DataFrame (Dask DataFrame for Dask-backed ``spots``) with the read
        metadata and columns ``barcode``, ``Q`` (per-cycle quality scores),
        ``Q_mean``, ``Q_min``, and ``barcode_match`` when ``barcodes`` is given.
    """
    if model is None:
        if use_whitelist and barcodes is not None:
            fit_kwargs.setdefault("whitelist", barcodes["barcode"].values)
        model = fit_base_mixture(
            spots,
            encoding=encoding,
            base_labels=base_labels,
            dark_bases=dark_bases,
            **fit_kwargs,
        )
    return _decode(spots=spots, barcodes=barcodes, f=_decode_em_chunk, model=model)


def _polar_thresholds_from_wcor(
    w_cor: NDArray,
    E: NDArray,
    base_labels: NDArray,
    bright_rows: NDArray,
    w_cor_bases: Sequence[str] | None = None,
) -> NDArray:
    """Infer the expected polar angle of each bright base from the xtalk matrix.

    Computes the expected signal of each dye via the forward xtalk model
    (``W_fwd = inv(w_cor)``) and projects it onto the 2 data channels. Angle
    boundaries are then placed at the midpoints between adjacent bases by
    :func:`_polar_angle_bins`. Fully parameter-free.

    Two cases, chosen from the size of ``w_cor``:

    * ``w_cor`` is over the 2 data channels: each data channel is one dye, and a
      base's signal is the mixture of the dyes in its encoding row,
      ``E[b] @ e_soft``. If ``w_cor_bases`` is given, ``w_cor`` is reordered to the
      data channel order, otherwise it is assumed to already be in that order.
    * ``w_cor`` is over more physical channels than data channels (e.g. 2-color
      data synthesized from 3 single-base channels): each physical channel is the
      dye of the base named in ``w_cor_bases``, and data channel ``k`` is taken as
      the max over the physical channels of the bases with ``E[b, k] > 0``.

    :param w_cor: ``(n, n)`` xtalk correction matrix from
        :func:`channel_crosstalk_matrix`.
    :param E: ``(n_bases, 2)`` encoding matrix.
    :param base_labels: Base label for each row of ``E``.
    :param bright_rows: Indices of the bright rows of ``E``.
    :param w_cor_bases: Base imaged in each physical channel of ``w_cor``, e.g.
        the ``c`` coordinate of the spots ``w_cor`` was estimated from. Required
        when ``w_cor`` has more channels than ``E``. When ``w_cor`` is over the
        data channels, each named base must fire exactly one data channel.
    :return: Expected angle in degrees of each base in ``bright_rows``.
    """
    n_xtalk = w_cor.shape[0]
    n_data = E.shape[1]
    labels = list(base_labels)

    if w_cor_bases is None:
        if n_xtalk != n_data:
            raise ValueError(
                f"w_cor has {n_xtalk} channels but the data has {n_data}; pass "
                f"w_cor_bases with the base imaged in each of the {n_xtalk} channels"
            )
    else:
        if len(w_cor_bases) != n_xtalk:
            raise ValueError(
                f"w_cor_bases has {len(w_cor_bases)} entries but w_cor has "
                f"{n_xtalk} channels"
            )
        unknown = [b for b in w_cor_bases if b not in labels]
        if unknown:
            raise ValueError(f"w_cor_bases {unknown} are not in the base labels")
        phys_rows = np.array([labels.index(b) for b in w_cor_bases])
        if n_xtalk == n_data:
            # data channel fired by the base in each w_cor channel
            data_ch = []
            for b, row in zip(w_cor_bases, phys_rows):
                fired = np.where(E[row] > 0)[0]
                if len(fired) != 1:
                    raise ValueError(
                        f"Base {b} in w_cor_bases must fire exactly one data "
                        "channel when w_cor is over the data channels"
                    )
                data_ch.append(fired[0])
            if sorted(data_ch) != list(range(n_data)):
                raise ValueError(
                    "w_cor_bases must name one base per data channel when w_cor "
                    "is over the data channels"
                )
            order = np.argsort(data_ch)  # w_cor channel for each data channel
            w_cor = w_cor[np.ix_(order, order)]

    w_fwd = np.linalg.inv(w_cor)
    col_sums = w_fwd.sum(axis=0, keepdims=True).clip(1e-9)
    e_soft = (w_fwd / col_sums).T  # (n_xtalk, n_xtalk): rows = dyes, cols = channels

    if n_xtalk == n_data:
        centroids = E[bright_rows] @ e_soft  # (n_bright, 2)
    else:
        # physical channels that make up each data channel
        groups = [np.where(E[phys_rows, k] > 0)[0] for k in range(n_data)]
        centroids = np.zeros((len(bright_rows), n_data))
        for i, row in enumerate(bright_rows):
            dye = np.where(phys_rows == row)[0]
            if len(dye) != 1:
                raise ValueError(
                    f"Base {labels[row]} must appear exactly once in w_cor_bases"
                )
            for k, group in enumerate(groups):
                centroids[i, k] = e_soft[dye[0], group].max() if len(group) else 0.0

    return np.degrees(np.arctan2(centroids[:, 1] + 1e-9, centroids[:, 0] + 1e-9))


def _rfrac_from_sweep(
    Rn: NDArray,
    bright_calls: NDArray,
    dark_base_idx: int,
    base_labels_arr: NDArray,
    whitelist: NDArray,
    n_grid: int = 20,
) -> float | None:
    """Find the r_frac that maximizes whitelist mapping rate via a vectorized grid sweep.

    All r_frac values in ``[0.05, 0.50]`` are evaluated simultaneously using
    broadcasting — no Python loop over the grid.  Cost is O(n·T·G) in memory
    and O(n·T·G + n·G·log|WL|) in time, where G = ``n_grid``.

    :param Rn: ``(n, T)`` per-read normalized radius ``R / R_max``.
    :param bright_calls: ``(n, T)`` base index (into ``base_labels_arr``) when
        the cycle is classified as bright.
    :param dark_base_idx: Index into ``base_labels_arr`` for the dark base.
    :param base_labels_arr: Ordered base labels (length = alphabet size).
    :param whitelist: 1-D array of barcode strings.
    :param n_grid: Number of evenly-spaced r_frac candidates in ``[0.05, 0.50]``.
    :return: Optimal r_frac as a float, or ``None`` if no whitelist barcode has
        length ``T``.
    """
    T = Rn.shape[1]
    n_bases = len(base_labels_arr)
    powers = (n_bases ** np.arange(T)).astype(np.int64)

    # Encode whitelist as sorted int64 array.
    # Single join+encode avoids one Python .encode() call per barcode.
    char_to_idx = {label: i for i, label in enumerate(base_labels_arr)}
    wl_valid = [s for s in whitelist if len(s) == T]
    if not wl_valid:
        return None
    cmap = np.zeros(256, dtype=np.int64)
    for label, idx in char_to_idx.items():
        cmap[ord(label)] = idx
    bc_bytes = np.frombuffer("".join(wl_valid).encode("ascii"), dtype=np.uint8).reshape(
        -1, T
    )
    wl_ints = np.sort((cmap[bc_bytes] * powers).sum(axis=1))

    # Precompute base barcode (all bright) and per-cycle dark correction
    bright_ints = (bright_calls.astype(np.int64) * powers).sum(axis=1)  # (n,)
    delta = (np.int64(dark_base_idx) - bright_calls.astype(np.int64)) * powers  # (n, T)

    # Vectorized evaluation — search [0.05, 0.40]; optimal is never above 0.35
    rfrac_grid = np.linspace(0.05, 0.40, n_grid)
    mask = Rn[:, :, np.newaxis] < rfrac_grid[np.newaxis, np.newaxis, :]  # (n, T, G)
    corr = (delta[:, :, np.newaxis] * mask).sum(axis=1)  # (n, G)
    bc = bright_ints[:, np.newaxis] + corr  # (n, G)

    pos = np.searchsorted(wl_ints, bc)
    pos = np.clip(pos, 0, len(wl_ints) - 1)
    rates = (wl_ints[pos] == bc).mean(axis=0)  # (G,)

    return float(rfrac_grid[np.argmax(rates)])


def _polar_angle_bins(
    E: NDArray, bright_rows: NDArray, centroid_angles: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Map angle bins to bright bases for a non-orthogonal 2-channel encoding.

    Bright rows are ordered by the angle of their encoding vector
    (ch0-only = 0°, both = 45°, ch1-only = 90°), so calls do not depend on the
    row order of ``E``. One bin edge is placed at the midpoint between the
    expected angles of each pair of adjacent bases.

    :param E: ``(n_bases, 2)`` encoding matrix.
    :param bright_rows: Indices of the bright rows of ``E``.
    :param centroid_angles: Expected angle in degrees of each base in
        ``bright_rows``, e.g. from :func:`_polar_thresholds_from_wcor`. Defaults to
        the encoding angles, which are also used, with a warning, if these angles
        are not in the same order as the encoding angles (e.g. very strong
        crosstalk or a ``w_cor_bases`` that does not match ``w_cor``).
    :return: ``(angle_rows, edges)``: bright rows sorted by angle and the
        ``len(angle_rows) - 1`` ascending bin edges in degrees.
    """
    E_bright = E[bright_rows]
    phi = np.degrees(np.arctan2(E_bright[:, 1], E_bright[:, 0]))
    order = np.argsort(phi)
    if np.any(np.isclose(np.diff(phi[order]), 0)):
        raise ValueError("Bright bases in the encoding must have distinct directions")
    centers = phi[order]
    if centroid_angles is not None:
        w_cor_centers = np.asarray(centroid_angles)[order]
        if np.all(np.diff(w_cor_centers) > 0):
            centers = w_cor_centers
        else:
            logger.warning(
                "Expected base angles from w_cor are not in encoding order, using "
                "encoding angles instead; check that w_cor_bases matches the "
                "channels of w_cor"
            )
    return bright_rows[order], (centers[:-1] + centers[1:]) / 2


def _polar_angle_calls(
    d: NDArray, angle_rows: NDArray, edges: NDArray
) -> tuple[NDArray, NDArray]:
    """Call bright bases from the angle of 2-channel background-subtracted spots.

    :param d: ``(read, t, 2)`` background-subtracted intensities.
    :param angle_rows: Bright rows sorted by angle, from :func:`_polar_angle_bins`.
    :param edges: Ascending bin edges in degrees, from :func:`_polar_angle_bins`.
    :return: ``(theta, calls)``: angle in degrees and called row of ``E``, both
        ``(read, t)``.
    """
    theta = np.degrees(np.arctan2(d[..., 1] + 1e-9, d[..., 0] + 1e-9))
    return theta, angle_rows[np.searchsorted(edges, theta)]


def _decode_polar_chunk(
    spots: NDArray,
    E: NDArray,
    base_labels_arr: NDArray,
    bright_rows: NDArray,
    dark_rows: NDArray,
    has_dark: bool,
    r_frac: float,
    non_orthogonal: bool,
    angle_rows: NDArray,
    angle_edges: NDArray,
    meta_df: pd.DataFrame,
    whitelist: NDArray | None,
    offset: slice | None,
    min_error: float = DEFAULT_MIN_P_ERROR,
) -> pd.DataFrame:
    """Process one chunk of spots through the polar decoder (numpy-only).

    Q is the phred score of P(call) = P(dark/bright call) * P(base | bright), where
    each factor is a two-class softmax on the margin of the decision actually made
    (in intensity units, consistent with :func:`quality_softmax`).
    """
    sp = np.clip(spots, 0.0, None)
    lo = sp.min(axis=1, keepdims=True)
    d = sp - lo

    E_bright = E[bright_rows]
    R = np.sqrt((d**2).sum(axis=-1))  # (read, t)
    if non_orthogonal:
        theta, bright_calls = _polar_angle_calls(d, angle_rows, angle_edges)
        if len(angle_edges) == 0:
            p_base = np.ones_like(R)  # single bright base
        else:
            # margin = perpendicular distance to the nearest angle bin edge;
            # 2-class softmax of [margin, -margin] = sigmoid(2 * margin), margin >= 0
            dtheta = np.radians(
                np.abs(theta[..., np.newaxis] - angle_edges).min(axis=-1)
            )
            margin = R * np.sin(dtheta)
            p_base = 1 / (1 + np.exp(-2 * margin))
    else:
        bright_scores = d @ E_bright.T
        bright_calls = bright_rows[bright_scores.argmax(axis=-1)]
        p_base = softmax(bright_scores, axis=-1).max(axis=-1)

    if has_dark:
        R_max = R.max(axis=1, keepdims=True)
        Rn = np.divide(R, R_max, out=np.zeros_like(R), where=R_max > 0)
        dark_mask = Rn < r_frac
        cutoff = r_frac * R_max
        # compare R with the cutoff in intensity units (not Rn with r_frac) so that
        # p is not capped by the [0, 1] range of Rn
        p_bright = softmax(
            np.stack([R, np.broadcast_to(cutoff, R.shape)], axis=-1), axis=-1
        )[..., 0]
        p = np.where(dark_mask, 1 - p_bright, p_bright * p_base)
    else:
        dark_mask = np.zeros(d.shape[:2], dtype=bool)
        p = p_base

    calls = np.where(dark_mask, dark_rows[0] if has_dark else 0, bright_calls)

    p_error = 1 - p
    p_error[p_error < min_error] = min_error
    Q = -10 * np.log10(p_error)
    df = (
        meta_df.iloc[offset.start : offset.stop].copy()
        if offset is not None
        else meta_df.copy()
    )
    df.index.name = None
    df["barcode"] = ["".join(base_labels_arr[row]) for row in calls.astype(int)]
    df["Q"] = list(Q)
    df["Q_mean"] = Q.mean(axis=1)
    df["Q_min"] = Q.min(axis=1)
    if whitelist is not None:
        df["barcode_match"] = df["barcode"].isin(whitelist)
    return df


_PEAK_LABEL_COL = "__peak_label__"


def _peaks_in_labels(peaks: pd.DataFrame, labels: NDArray) -> pd.DataFrame:
    """Keep peaks that fall inside a label.

    :param peaks: Peaks with `y` and `x` columns.
    :param labels: Segmentation array (y,x).
    :return: Peaks where the label is > 0, with the label in column
        `_PEAK_LABEL_COL`.
    """
    peak_labels = labels[peaks["y"].values, peaks["x"].values]
    in_labels = peak_labels > 0
    peaks = peaks[in_labels].copy()
    peaks[_PEAK_LABEL_COL] = peak_labels[in_labels]
    return peaks


def _auto_read_chunk_size(
    n_reads: int,
    n_threads: int,
    chunks_per_thread: int = 3,
    min_chunk_size: int = 50_000,
    max_chunk_size: int = 500_000,
) -> int:
    """Number of reads per chunk giving about `chunks_per_thread` chunks for each thread.

    The size is clamped to [`min_chunk_size`, `max_chunk_size`]: small chunks
    duplicate per-chunk work (e.g. mismatch correction of the same barcodes) and
    create many small output files; large chunks limit parallelism.

    :param n_reads: Total number of reads.
    :param n_threads: Number of threads available for processing.
    :param chunks_per_thread: Target number of chunks per thread.
    :param min_chunk_size: Minimum number of reads per chunk.
    :param max_chunk_size: Maximum number of reads per chunk.
    :return: Number of reads per chunk.

    >>> _auto_read_chunk_size(10_000_000, 32)
    104167
    >>> _auto_read_chunk_size(20_000, 32)
    50000
    >>> _auto_read_chunk_size(100_000_000, 8)
    500000
    """
    chunk_size = math.ceil(n_reads / (max(1, n_threads) * max(1, chunks_per_thread)))
    return int(min(max(chunk_size, min_chunk_size), max_chunk_size))


def decode_polar(
    spots: xr.DataArray,
    barcodes: pd.DataFrame | None = None,
    encoding: NDArray | None = None,
    base_labels: list[str] | NDArray | None = None,
    dark_bases: list[str] | None = None,
    w_cor: NDArray | None = None,
    w_cor_bases: Sequence[str] | None = None,
    r_frac: float | None = None,
    n_reads: int = 50_000,
) -> pd.DataFrame | dd.DataFrame:
    """Call reads using polar-coordinate classification.

    Separates signal into two independent properties:

    * **Radius** :math:`R = \\|\\mathbf{x} - \\mathbf{lo}\\|_2` — detects dark bases
      (all channels quiet below :math:`r_{\\text{frac}} \\cdot R_{\\max}`).
    * **Direction** — determines which bright base fired.

    Two direction strategies, chosen automatically from the encoding matrix:

    **Orthogonal encoding** (each bright base fires exactly one channel,
    e.g. 4-color identity or 3-color dark-base):

    .. math::

        \\text{calls} = \\arg\\max_b\\; (\\mathbf{x} - \\mathbf{lo}) \\cdot E[b,:]

    **Non-orthogonal encoding** (some bright base fires multiple channels,
    e.g. Illumina 2-color where A fires both red and green):

    .. math::

        \\theta = \\arctan\\!\\left(\\frac{d_{c_1}}{d_{c_0}}\\right),
        \\quad \\text{classified into one angle bin per bright base}

    The branching depends on the **encoding matrix**, not on channel count —
    orthogonal chemistries use the dot-product regardless of dimensionality.

    Non-orthogonal encodings require exactly 2 channels.  Bright bases are
    ordered by the angle of their encoding vector (ch0-only = 0°, both = 45°,
    ch1-only = 90°), so the row order of ``encoding`` does not matter, and the
    :math:`n_{\\text{bright}} - 1` bin edges are placed at the midpoints between
    the expected angles of adjacent bases.  The expected angles are derived
    parameter-free from ``w_cor`` and ``w_cor_bases`` via
    :func:`_polar_thresholds_from_wcor`; if ``w_cor`` is ``None``, the encoding
    angles are used, giving edges at 22.5° and 67.5° for the usual 3 bright
    bases.

    **Quality scores.** Per-cycle ``Q`` is the phred score of the probability
    that the call made is correct, factored into the two decisions above:

    .. math::

        p = \\begin{cases}
            p_{\\text{dark}} & \\text{dark call} \\\\
            p_{\\text{bright}} \\cdot p_{\\text{base}} & \\text{bright call}
        \\end{cases},
        \\qquad Q = -10 \\log_{10} \\max(1 - p,\\; 10^{-6})

    so ``Q`` is capped at 60. Each factor is a two-class softmax on the margin of
    its decision, in intensity units (as in :func:`quality_softmax`):

    * **Dark vs. bright** — softmax of :math:`R` against the cutoff
      :math:`r_{\\text{frac}} \\cdot R_{\\max}`; :math:`p_{\\text{dark}}` and
      :math:`p_{\\text{bright}}` are the probabilities of the side called.
      Omitted when there are no dark bases.
    * **Bright base, orthogonal** — :math:`p_{\\text{base}} = \\max_b
      \\operatorname{softmax}_b\\big((\\mathbf{x} - \\mathbf{lo}) \\cdot E[b,:]\\big)`
      over the bright bases.
    * **Bright base, non-orthogonal** — with :math:`\\Delta\\theta` the angle to the
      nearest angle bin edge,
      the margin is the perpendicular distance to that edge,
      :math:`m = R \\sin\\Delta\\theta`, and
      :math:`p_{\\text{base}} = \\operatorname{softmax}(m, -m)_0 = 1 / (1 + e^{-2m})`.

    Cycles near the dark cutoff or a bin edge therefore score low, and clearly
    bright or clearly dark cycles score high. Because the margins are in intensity
    units, ``Q`` depends on the intensity scale of ``spots`` and is a relative
    confidence rather than a calibrated error rate.

    :param spots: Spot DataArray from :func:`peaks_to_bases`, dims ``(read, t, c)``.
    :param barcodes: Whitelist DataFrame with column ``'barcode'``; adds a
        ``barcode_match`` column to the result when provided.
    :param encoding: ``(n_bases, n_channels)`` encoding matrix.  All-zero rows
        indicate dark bases; at most one is allowed.  Takes priority over
        ``dark_bases``.
    :param base_labels: Base labels of length ``n_bases``, required with
        ``encoding``.
    :param dark_bases: Shorthand for 1:1 channel→base chemistry with named dark
        bases.  Builds the encoding matrix via :func:`make_encoding`.
    :param w_cor: Xtalk correction matrix from :func:`channel_crosstalk_matrix`.
        Used to derive angle thresholds for non-orthogonal encodings.
    :param w_cor_bases: Base imaged in each channel of ``w_cor``, e.g.
        ``list(spots_used_for_w_cor.c.values)``. Required when ``w_cor`` has more
        channels than ``spots`` (e.g. 2-color data synthesized from 3 channels).
        When ``w_cor`` is over the channels of ``spots``, it is optional and is
        used to reorder ``w_cor`` to the channel order of ``spots``; each named
        base must then fire exactly one channel.
    :param r_frac: Per-read fraction of :math:`R_{\\max}` below which a cycle
        is classified as the dark base.  Ignored when no dark bases exist.
        When ``None`` (default) and dark bases are present, the optimal value
        is found automatically via :func:`_rfrac_from_sweep` if a barcode
        whitelist with barcodes of length :math:`T` is provided, otherwise falls
        back to :math:`0.11\\sqrt{n_c}`.
    :param n_reads: Number of reads to use for :func:`_rfrac_from_sweep`
    :return: DataFrame with columns ``barcode``, ``Q`` (per-cycle quality scores),
        ``Q_mean``, ``Q_min``, and optionally ``barcode_match``.
    """
    channel_bases = list(spots.c.values)

    # Build encoding matrix
    if encoding is not None:
        E = np.asarray(encoding, dtype=float)
        base_labels_arr = np.asarray(base_labels)
    elif dark_bases is not None and len(dark_bases) > 0:
        E, base_labels_arr = make_encoding(channel_bases, list(dark_bases))
    else:
        E = np.eye(len(channel_bases), dtype=float)
        base_labels_arr = np.array(channel_bases)

    dark_rows = np.where(E.sum(axis=1) == 0)[0]
    bright_rows = np.where(E.sum(axis=1) > 0)[0]
    has_dark = len(dark_rows) > 0
    if len(dark_rows) > 1:
        raise ValueError(
            "decode_polar supports at most one dark base, got "
            f"{list(base_labels_arr[dark_rows])}: dark bases have no signal and "
            "cannot be told apart"
        )

    #  Angle thresholds (non-orthogonal only, parameter-free from w_cor) ─
    E_bright = E[bright_rows]
    non_orthogonal = bool(np.any(E_bright.sum(axis=1) > 1))
    if non_orthogonal:
        if E.shape[1] != 2:
            raise ValueError(
                "Non-orthogonal encodings are only supported with 2 channels, "
                f"got {E.shape[1]}"
            )
        centroid_angles = (
            _polar_thresholds_from_wcor(
                w_cor, E, base_labels_arr, bright_rows, w_cor_bases
            )
            if w_cor is not None
            else None
        )
        angle_rows, angle_edges = _polar_angle_bins(E, bright_rows, centroid_angles)
    else:
        # unused for orthogonal
        angle_rows, angle_edges = bright_rows, np.array([], dtype=float)

    #  r_frac: auto-compute from a sample when the input is dask
    whitelist_arr = barcodes["barcode"].values if barcodes is not None else None
    if r_frac is None and has_dark:
        if whitelist_arr is not None:
            # Use up to n_reads spots for the sweep; materialize only that sample
            rng = random.Random(239753)
            random_reads = rng.sample(
                range(0, spots.sizes["read"]), min(n_reads, spots.sizes["read"])
            )
            random_reads = np.sort(random_reads)
            sp_sample = np.clip(spots.isel(read=random_reads).data, 0.0, None)
            if isinstance(sp_sample, da.Array):
                sp_sample = sp_sample.compute()
            lo_s = sp_sample.min(axis=1, keepdims=True)
            d_s = sp_sample - lo_s
            R_s = np.sqrt((d_s**2).sum(axis=-1))
            Rmax_s = R_s.max(axis=1, keepdims=True)
            Rn_s = np.divide(R_s, Rmax_s, out=np.zeros_like(R_s), where=Rmax_s > 0)
            if non_orthogonal:
                _, bc_s = _polar_angle_calls(d_s, angle_rows, angle_edges)
            else:
                bc_s = bright_rows[(d_s @ E_bright.T).argmax(axis=-1)]
            r_frac = _rfrac_from_sweep(
                Rn_s, bc_s, int(dark_rows[0]), base_labels_arr, whitelist_arr
            )
            if r_frac is None:
                logger.warning(
                    f"No whitelist barcodes of length {Rn_s.shape[1]}, "
                    "using default r_frac."
                )
        if r_frac is None:
            r_frac = 0.11 * np.sqrt(len(channel_bases))

    return _decode(
        spots=spots,
        barcodes=barcodes,
        f=_decode_polar_chunk,
        E=E,
        base_labels_arr=base_labels_arr,
        bright_rows=bright_rows,
        dark_rows=dark_rows,
        has_dark=has_dark,
        r_frac=r_frac if r_frac is not None else 0.0,
        non_orthogonal=non_orthogonal,
        angle_rows=angle_rows,
        angle_edges=angle_edges,
    )


def peaks_to_bases(
    maxed: xr.DataArray,
    peaks: pd.DataFrame | dd.DataFrame,
    labels: NDArray | da.Array | xr.DataArray | None = None,
    labels_only: bool = True,
    bases: Sequence[str] | None = ("G", "T", "A", "C"),
    read_chunk_size: Literal["auto"] | int | None = "auto",
) -> xr.DataArray:
    """Convert peaks to bases.

    Memory usage:

    - ``peaks`` is loaded into memory. If it is a dask data frame and ``labels_only``
      is true, peaks outside labels are dropped per partition before loading, so only
      peaks inside labels are held in memory.
    - ``labels`` is loaded into memory as a numpy array if dask-backed. When a
      distributed client is active and ``peaks`` is a dask data frame, ``labels`` is
      scattered to the workers for filtering.
    - ``maxed`` is not loaded. The returned array is lazy if ``maxed`` is dask-backed;
      computing it reads the ``maxed`` chunks that contain peaks, which is typically
      the whole image.

    :param maxed: Maxed array (sigma,t,c,y,x) or (t,c,y,x)
    :param peaks: Peaks data frame which has been filtered to retain only peaks of
        interest.
    :param labels: Segmentation array (y,x)
    :param labels_only: If true, only return peaks where labels are present.
    :param bases: List of bases.
    :param read_chunk_size: Number of reads per chunk for parallel processing, or
        "auto" to choose from the number of reads and dask worker threads. Use None or
        <= 0 for a single chunk. Ignored when `maxed` is not dask-backed.
    :return: DataArray with dimensions (read,t,c) where read is spot with coordinates for every
        column of `peaks` (typically ``y``, ``x``, ``sigma``, ``peak``), plus ``label`` when
        `labels` is given.
    """
    if isinstance(labels, xr.DataArray):
        labels = labels.data
    if isinstance(labels, da.Array):
        labels = labels.compute()
    peak_labels = None
    if isinstance(peaks, dd.DataFrame):
        if labels_only and labels is not None:
            from distributed import get_client

            # drop peaks outside labels before loading them into memory
            try:
                labels_arg = get_client().scatter(labels, hash=False)
            except ValueError:  # no distributed client
                labels_arg = labels
            meta = peaks._meta.copy()
            meta[_PEAK_LABEL_COL] = pd.Series(dtype=labels.dtype)
            peaks = peaks.map_partitions(_peaks_in_labels, labels_arg, meta=meta)
            peaks = peaks.compute()
            peak_labels = peaks.pop(_PEAK_LABEL_COL).values
        else:
            peaks = peaks.compute()
    if labels is not None and peak_labels is None:
        peak_labels = labels[peaks["y"].values, peaks["x"].values]
        if labels_only:
            in_labels = peak_labels > 0
            peaks = peaks[in_labels]
            peak_labels = peak_labels[in_labels]
    if "t" not in maxed.dims:
        maxed = maxed.expand_dims("t")

    assert maxed.dims == ("sigma", "t", "c", "y", "x") or maxed.dims == (
        "t",
        "c",
        "y",
        "x",
    ), f"Found dimensions: {maxed.dims}"

    if "sigma" in maxed.dims and maxed.sizes["sigma"] == 1:
        maxed = maxed.squeeze("sigma", drop=True)

    sigma_indices = None
    if "sigma" in maxed.dims:
        sigma_indices = pd.Index(maxed.coords["sigma"].values).get_indexer(
            peaks["sigma"]
        )
        if (sigma_indices < 0).any():
            raise ValueError("Peaks contain sigma values not found in `maxed`")
    bases_array_reads = (
        (
            maxed.isel(
                y=xr.DataArray(peaks["y"]),
                x=xr.DataArray(peaks["x"]),
                sigma=xr.DataArray(sigma_indices),
            )
        )
        if sigma_indices is not None
        else maxed.isel(y=xr.DataArray(peaks["y"]), x=xr.DataArray(peaks["x"]))
    )
    bases_array_reads = bases_array_reads.rename({"dim_0": "read"}).transpose(
        "read", ...
    )
    bases_array_reads.name = "maxed"

    for c in peaks.columns:
        # add columns in peaks
        bases_array_reads.coords[c] = ("read", peaks[c])
    if peak_labels is not None:
        bases_array_reads.coords["label"] = ("read", peak_labels)
    if bases is not None:
        bases_array_reads = bases_array_reads.assign_coords(c=list(bases))
    if not isinstance(bases_array_reads.data, da.Array):
        return bases_array_reads  # in memory: nothing to split
    # Split the gathered spots along `read` so decoding, crosstalk application, mismatch
    # correction, and writing run in parallel.
    n_reads = bases_array_reads.sizes["read"]
    if read_chunk_size == "auto":
        read_chunk_size = _auto_read_chunk_size(n_reads, _dask_total_threads())
    if isinstance(read_chunk_size, str):
        raise ValueError(f"Invalid read chunk size: {read_chunk_size}")
    if read_chunk_size is None or read_chunk_size <= 0:
        read_chunk_size = -1  # single chunk
    elif read_chunk_size < n_reads:
        logger.info(
            f"Processing reads in chunks of {read_chunk_size:,} "
            f"({math.ceil(n_reads / read_chunk_size)} chunks)."
        )
    # rechunk the data only; `.chunk` would also make the read coords lazy. No-op when
    # the chunks already match.
    bases_array_reads = bases_array_reads.copy(
        data=bases_array_reads.data.rechunk(
            {bases_array_reads.get_axis_num("read"): read_chunk_size}
        )
    )
    return bases_array_reads


def merge_sbs_phenotype(
    df_labels: pd.DataFrame | dd.DataFrame,
    df_phenotype: pd.DataFrame | dd.DataFrame,
    df_barcode: pd.DataFrame,
    sbs_cycles: Sequence[int],
    how: Literal["left", "right", "inner", "outer", "cross"] = "outer",
    barcode_prefix: str | None = None,
) -> pd.DataFrame | dd.DataFrame:
    """Combine sequencing and phenotype tables with one row per label.

    The index must be the same in both tables (e.g., both tables generated from the
    same segmentation).

    The barcode table is then joined using its `barcode` column to the most abundant
    (`barcode_0`) and second-most abundant (`barcode_1`) barcodes for each label.
    The substring (prefix) of `barcode` used for joining is determined by the
    `sbs_cycles` index. Duplicate prefixes are dropped for the joined table
    (e.g., if insufficient sequencing is available to disambiguate two barcodes).

    :param df_labels: Data frame containing SBS reads:
    :param df_phenotype: Data frame with phenotype calls
    :param df_barcode: Barcode information data frame
    :param sbs_cycles: List of cycles used (starting at 1)
    :param how: How to merge
    :param barcode_prefix: Optional prefix to prepend to the barcode table columns
    :return: Combined table
    """

    df_barcode = (
        df_barcode.assign(
            prefix=lambda x: x["barcode"].apply(barcode_to_prefix, args=(sbs_cycles,))
        )
    ).set_index("prefix")
    df_barcode["duplicate_prefix"] = df_barcode.index.duplicated(keep=False)

    if isinstance(df_barcode, pd.DataFrame):
        n_barcodes = len(df_barcode)
    df_barcode = df_barcode[~df_barcode.index.duplicated(keep="first")]
    if isinstance(df_barcode, pd.DataFrame):
        n_barcodes_duplicated = len(df_barcode)
        if n_barcodes_duplicated != n_barcodes:
            from scallops.io import pluralize

            n_removed = n_barcodes - n_barcodes_duplicated
            logger.info(
                f"Removed {n_removed:,} duplicate {pluralize('barcode', n_removed)}"
            )

    if barcode_prefix:
        df_barcode = df_barcode.rename(columns=lambda x: barcode_prefix + x)

    df_combined = (
        df_labels.join(df_phenotype, how=how)
        .join(df_barcode, on="barcode_0", rsuffix="_barcode_0")
        .join(
            df_barcode.rename(columns=lambda x: x + "_1"),
            on="barcode_1",
            rsuffix="_barcode_1",
        )
    )
    return df_combined


def barcode_to_prefix(barcode: str, sbs_cycles: Sequence[int]) -> str:
    """Utility function to generate prefixes based on barcodes.

    Intended to be called per-row via ``Series.apply(barcode_to_prefix, args=(sbs_cycles,))``, so
    `barcode` is a single barcode string, not a Series.

    :param barcode: A single barcode string, one base per cycle.
    :param sbs_cycles: Cycles to work on (starting from 1)
    :return: Prefix combining all channels
    """
    return "".join(barcode[c - 1] for c in sbs_cycles)


def read_statistics(reads_df: pd.DataFrame | dd.DataFrame) -> dict[str, float | int]:
    """Compute read statistics, such as mapped_reads, mapped_reads_within_labels, and
    average_reads_per_label.

    :param reads_df: Reads data frame
    :return: Dictionary with keys ``mapped_reads``, ``number_of_reads``, ``mapping_rate``,
        ``mapping_rate_within_labels``, ``mapped_reads_within_labels``,
        ``average_reads_per_label``, ``average_mapped_reads_per_label``,
        ``number_of_unique_barcodes_in_labels``, ``mean_barcode_count_in_labels``,
        ``labels_with_reads``, and ``labels_with_mapped_reads``; plus, only when `reads_df` has
        reads outside any label (``label == 0``), ``mapping_rate_outside_labels`` and
        ``mapped_reads_outside_labels``.
    """
    unique_cell_col = (
        "label" if "unique_label" not in reads_df.columns else "unique_label"
    )
    in_labels = reads_df.query("label!=0")
    outside_labels = reads_df.query("label==0")
    mapping_rate = reads_df.query("barcode_match").shape[0] / reads_df.shape[0]
    mapping_rate_within_labels = (
        (in_labels.query("barcode_match").shape[0] / in_labels.shape[0])
        if in_labels.shape[0] > 0
        else 0
    )

    barcode_matches = in_labels.query("barcode_match==1")

    data = {
        "mapped_reads": reads_df["barcode_match"].sum(),
        "number_of_reads": reads_df.shape[0],
        "mapping_rate": mapping_rate,
        "mapping_rate_within_labels": mapping_rate_within_labels,
        "mapped_reads_within_labels": in_labels["barcode_match"].sum(),
        "average_reads_per_label": in_labels.pipe(len)
        / reads_df[unique_cell_col].nunique(),
        "average_mapped_reads_per_label": barcode_matches.pipe(len)
        / reads_df[unique_cell_col].nunique(),
        "number_of_unique_barcodes_in_labels": barcode_matches.pipe(
            lambda x: x["barcode"].nunique()
        ),
        "mean_barcode_count_in_labels": barcode_matches.pipe(
            lambda x: x["barcode"].value_counts().mean()
        ),
        "labels_with_reads": in_labels[unique_cell_col].nunique(),
        "labels_with_mapped_reads": barcode_matches[unique_cell_col].nunique(),
    }

    if outside_labels.shape[0] > 0:
        mapping_rate_outside_labels = outside_labels.query("barcode_match").shape[
            0
        ] / max(1, outside_labels.shape[0])
        data["mapping_rate_outside_labels"] = mapping_rate_outside_labels
        data["mapped_reads_outside_labels"] = outside_labels["barcode_match"].sum()
    return data


def base_counts(reads_df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """Compute base counts per cycle.

    :param reads_df: Data frame containing reads
    :param normalize: Whether to normalize the counts
    :return: One row per base and cycle, with columns ``base``, ``t`` (cycle), and ``count`` (the
        number of reads with that base at that cycle, or its fraction of the cycle's reads if
        `normalize`).
    """
    read_len = reads_df["barcode"].str.len().max()
    counts_df = pd.DataFrame()
    for i in range(read_len):
        counts_df_ = pd.DataFrame()
        counts_df_["count"] = (
            reads_df["barcode"].str[i].value_counts(normalize=normalize)
        )
        counts_df_["t"] = i
        counts_df = pd.concat((counts_df, counts_df_))
    counts_df.index.name = "base"
    return counts_df.reset_index()


def peaks_to_spot_labels(
    peaks: dd.DataFrame | pd.DataFrame, shape: tuple[int, int]
) -> NDArray:
    """Convert peaks to spot labels.

    :param peaks: Peaks data frame which has been filtered to retain only peaks of interest
    :param shape: Shape of the spots output
    :return: Array with spot labels
    """

    spots_labels = np.zeros(shape, dtype=int)
    spots_labels[peaks["y"], peaks["x"]] = np.arange(1, len(peaks) + 1)
    return spots_labels


def _save_annotated_spots(df_reads, maxed, bases, output_annotated_spots):
    """Save annotated spots as an image stack with color-coded bases.

    :param df_reads: DataFrame containing read data.
    :param maxed: Max-filtered image.
    :param bases: List of bases.
    :param output_annotated_spots: Tuple with output filename and expansion width.
    """
    outfn, width = output_annotated_spots
    ann = annotated_spots(
        df_reads, shape=maxed.shape, bases_order=bases, expand_width=int(width)
    )
    dim_order = "".join([x.upper() for x in maxed.dims])[:-2]
    # DAPI(gray), G(green), T(red), A(magenta), C(cyan)
    LUTS = {"G": GREEN, "T": RED, "A": MAGENTA, "C": CYAN}
    luts = (GRAY,)
    for base in bases:
        luts += (LUTS[base.upper()],)
    dims = [x for x in maxed.dims if x != "c"]
    maxs = maxed.max(dim=dims).values
    mins = maxed.min(dim=dims).values
    display_ranges = tuple()
    for i in range(maxed.c.size):
        display_ranges += ((mins[i], maxs[i]),)
    save_stack_imagej(
        outfn,
        img_as_float(ann),
        luts=luts,
        display_ranges=display_ranges,
        dimensions=dim_order,
        compress=1,
    )


def annotated_spots(reads_df, shape, bases_order, expand_width=3):
    """Generate an annotated image of spots based on decoded reads.

    This function takes a DataFrame of decoded reads produced by one of the ``decode_*`` functions
    and generates an annotated image of spots based on the decoded information.

    :param reads_df: DataFrame containing decoded reads with columns 'y', 'x', and 'barcode'.
    :param shape: Shape of the output annotated image.
    :param bases_order: Order of bases for mapping to channels.
    :param expand_width: Width to expand each spot label for better visualization. Default is 3.
    :return: Annotated image of spots.

    :example:

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from scallops import annotated_spots, reads

        # Create a synthetic reads DataFrame
        reads_df = pd.DataFrame(
            {"y": [10, 20, 30], "x": [15, 25, 35], "barcode": ["ACGT", "TCGA", "GCTA"]}
        )

        # Define shape and bases order
        shape = (50, 50, 4, 4)
        bases_order = ["A", "C", "G", "T"]

        # Generate annotated spots
        annotated_image = annotated_spots(reads_df, shape, bases_order)
    """
    spots = np.zeros(shape, dtype=np.uint16)
    decoded = reads_df[["y", "x", "barcode"]]
    decoded = decoded.set_index(["y", "x"])
    decoded = decoded.barcode.apply(lambda x: list(x)).explode()
    decoded = decoded.reset_index()
    decoded["t"] = decoded.groupby(["y", "x"]).cumcount()
    decoded["c"] = decoded.barcode.map(dict(zip(bases_order, range(len(bases_order)))))
    spots[decoded.t, decoded.c, decoded.y, decoded.x] = 255
    return np.array(
        [
            [
                expand_labels(spots[t, c, ...], distance=expand_width)
                for c in range(spots.shape[1])
            ]
            for t in range(spots.shape[0])
        ]
    )


def query_spots(
    spots_labels: NDArray,
    calls: NDArray | pd.DataFrame,
    query: Sequence[str],
    expand_width: int = 1,
) -> NDArray:
    """Returns expanded (optional) labels that match a set of query barcodes.

    :param spots_labels: An array of spot labels to be queried.
    :param calls: An array or DataFrame of barcode calls.
    :param query: A sequence of barcode strings to query.
    :param expand_width: The width by which to expand the labels (default is 1).
    :return: An array with expanded labels that match the query barcodes.

    :example:

    .. code-block:: python

        # Example usage of query_spots
        expanded_labels = query_spots(
            spots_labels, calls, query=["ACTG", "TGCA"], expand_width=2
        )
    """
    error = f"calls is of type {type(calls)} but array or dataframe were expected"

    assert isinstance(calls, (np.ndarray, pd.DataFrame)), error

    if isinstance(calls, np.ndarray):
        joined = np.apply_along_axis("".join, 1, calls)
        matches = np.where(np.isin(joined, query))[0] + 1
    else:
        matches = calls.query("barcode.isin(@query)").read.values + 1
    return expand_labels(np.where(np.isin(spots_labels, matches), 255, 0), expand_width)


def li_speed_slope(
    x: NDArray, y: NDArray, quantile_range: tuple[float, float] = (0.6, 0.999)
) -> Tuple[pd.DataFrame, float]:
    """Computes the slope using the Li and Speed method for crosstalk correction.

    :param x: The independent variable data (e.g., signal intensity).
    :param y: The dependent variable data (e.g., signal intensity in another channel).
    :param quantile_range: The range of quantiles to consider for the analysis (default is (0.6, 0.999)).
    :return: A tuple containing the DataFrame with binned data and the computed slope.

    :reference:

    This method is based on the work of Li and Speed in their paper on crosstalk correction:
    Li, C., & Speed, T. P. (1999). "Crosstalk correction for cDNA microarray data."
    Nature Biotechnology, 17(9), 884-885. doi:10.1038/12813

    :example:

    .. code-block:: python

        # Example usage of li_speed_slope
        df, slope = li_speed_slope(x_data, y_data, quantile_range=(0.6, 0.999))
    """

    df = pd.DataFrame(data=dict(x=x, y=y))
    quantiles = df["x"].quantile(quantile_range).values
    df = df[(df.x >= quantiles[0]) & (df.x <= quantiles[1])]
    # convert x to bins
    # for each bin, find (x, y) where y is minimum
    n_bins = int(np.ceil(2 * np.power(df.x.size, 2 / 5)))
    df["bin"] = pd.cut(df.x, bins=n_bins, labels=False)

    def get_points(_df):
        return _df.iloc[_df.y.argmin()]

    df = df.groupby("bin").apply(get_points)

    qr = QuantileRegressor(solver="highs", fit_intercept=True)
    X = df.x.values.reshape(-1, 1)
    df["y_pred"] = qr.fit(X, df.y).predict(X)
    slope = qr.coef_[0]
    # intercept = qr.intercept_[0]
    return df, slope


def _correct_channel_crosstalk_li_and_speed(
    a: np.array,
    n_iter_max: int = 15,
    slope_threshold: float = 0.05,
    quantile_range: tuple[float, float] = (0.6, 0.999),
    normalize_w: bool = True,
) -> tuple[NDArray, NDArray]:
    """Estimate and correct differences in channel intensity and spectral overlap among sequencing
    channels using the method in Li and Speed.

    Describe with linear transformation w so that w * data = y, where y is the corrected data.

    :param a: data to transform (read + t, c)
    :param n_iter_max: Maximum number of iterations to perform
    :param slope_threshold: Stop iterating when the maximum of the absolute values of the 12 estimated slopes <= `slope_threshold`
    :param quantile_range:Lower and upper quantiles to include
    :param normalize_w: Whether to normalize the w matrix on every iteration
    :return: The inverse matrix, w
    """

    only_inverse = True
    n_channels = a.shape[1]
    working_data = a
    _w = None
    _inverse = None
    for li_speed_iter in range(n_iter_max):
        max_slope = 0.0
        w = np.ones((n_channels, n_channels))
        for i in range(n_channels):
            x = working_data[:, i]
            for j in range(n_channels):
                if i != j:
                    y = working_data[:, j]
                    _, slope = li_speed_slope(x, y, quantile_range=quantile_range)
                    max_slope = max(max_slope, abs(slope))
                    w[j, i] = slope

        if max_slope <= slope_threshold:
            break
        if normalize_w:
            w = w / w.sum(axis=0)
            inverse = np.linalg.inv(w)
            _inverse = inverse @ _inverse if _inverse is not None else inverse
            working_data = inverse.dot(working_data.T).T
        else:
            working_data = np.linalg.inv(w).dot(working_data.T).T
            _w = w @ _w if _w is not None else w

    if normalize_w:
        if _inverse is None:
            _inverse = np.zeros((n_channels, n_channels))
            np.fill_diagonal(_inverse, 1)
        if only_inverse:
            return _inverse
        return working_data, _inverse
    else:
        if _w is None:
            _w = np.zeros((n_channels, n_channels))
            np.fill_diagonal(_w, 1)
        w = _w / _w.sum(axis=0)
        inverse = np.linalg.inv(w)

    if only_inverse:
        return inverse
    return inverse.dot(a.T).T, inverse


def li_and_speed_cc_number(bases: xr.DataArray) -> float:
    """Calculates the maximum of the absolute values of the (12 for 4-channels) estimated slopes
    using the Li and Speed method.

    :param bases: Array containing corrected intensities.
    :return: The cc number.
    """
    a = bases.where(bases.label > 0, drop=True)
    channels = a.c.values
    max_slope: float = 0
    for i, j in product(range(len(channels)), repeat=2):
        x = a.sel(c=channels[i]).to_numpy().flatten()
        if i != j:
            y = a.sel(c=channels[j]).to_numpy().flatten()
            _, slope = li_speed_slope(x, y)
            max_slope = max(max_slope, abs(slope))
    return max_slope


class _OneMismatchIndex:
    """Exact index of equal-length barcodes for lookups within one mismatch.

    Two barcodes are at most one mismatch apart exactly when they agree with some
    one position masked, so each query is a sorted lookup per position rather than
    a nearest-neighbor search.  Barcodes are packed into integer keys (or byte
    strings, for long barcodes).

    :param codes: ``(barcode, position)`` codes from :func:`_encode_alphabet`.
    :param alphabet: Alphabet used to encode ``codes``.
    """

    def __init__(self, codes: NDArray, alphabet: Sequence[str]):
        self.n_alphabet = len(alphabet)
        self.codes = codes
        self.mask = self.n_alphabet + 2  # codes are 1..A, A + 1 for unknown
        base = self.mask + 1
        self.length = codes.shape[1]
        self.int_keys = base**self.length < 2**62
        self.powers = base ** np.arange(self.length, dtype=np.int64)
        self.n = len(codes)
        sorted_keys, order = [], []
        for k in range(self.length):
            key = self._keys(codes, k)
            o = np.argsort(key)
            sorted_keys.append(key[o])
            order.append(o)
        self._set_keys(np.sort(self._keys(codes)), sorted_keys, order)

    def _set_keys(
        self, exact_keys: NDArray, sorted_keys: list[NDArray], order: list[NDArray]
    ) -> None:
        self.exact_keys = exact_keys
        self.exact = self._runs(exact_keys)
        self.sorted_keys = sorted_keys
        self.order = order
        self.masked = [self._runs(key) for key in sorted_keys]
        self._conflicts: tuple[NDArray, NDArray, NDArray] | None = None

    def extend(self, codes: NDArray) -> "_OneMismatchIndex":
        """A new index of the indexed barcodes followed by ``codes``.

        The new keys are sorted and merged into the existing sorted keys, so the
        indexed barcodes are not sorted again.
        """
        new = copy.copy(self)
        new.codes = np.vstack([self.codes, codes])
        new.n = len(new.codes)

        def merge(old: NDArray, keys: NDArray) -> tuple[NDArray, NDArray, NDArray]:
            o = np.argsort(keys)
            keys = keys[o]
            at = np.searchsorted(old, keys)
            return np.insert(old, at, keys), at, o

        exact_keys = merge(self.exact_keys, self._keys(codes))[0]
        sorted_keys, order = [], []
        for k in range(self.length):
            key, at, o = merge(self.sorted_keys[k], self._keys(codes, k))
            sorted_keys.append(key)
            order.append(np.insert(self.order[k], at, o + self.n))
        new._set_keys(exact_keys, sorted_keys, order)
        return new

    @property
    def n_neighbors(self) -> int:
        """Number of one-mismatch neighbors of each barcode, ``L * (A - 1)``."""
        return self.length * (self.n_alphabet - 1)

    def _keys(self, codes: NDArray, masked: int | list[int] | None = None) -> NDArray:
        if masked is not None:
            codes = codes.copy()
            codes[:, masked] = self.mask
        if self.int_keys:
            return codes.astype(np.int64) @ self.powers
        return np.ascontiguousarray(codes).view(f"S{self.length}").ravel()

    @staticmethod
    def _runs(sorted_keys: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Unique keys with the start and length of each one's run."""
        new = np.ones(len(sorted_keys), dtype=bool)
        new[1:] = sorted_keys[1:] != sorted_keys[:-1]
        start = np.flatnonzero(new)
        return sorted_keys[start], start, np.diff(start, append=len(sorted_keys))

    @staticmethod
    def _count(runs: tuple[NDArray, NDArray, NDArray], q: NDArray) -> NDArray:
        """Number of sorted keys equal to each query."""
        uniq, _, cnt = runs
        if len(uniq) == 0:
            return np.zeros(len(q), dtype=np.int64)
        i = np.minimum(np.searchsorted(uniq, q), len(uniq) - 1)
        return np.where(uniq[i] == q, cnt[i], 0)

    def count_within(self, codes: NDArray) -> NDArray:
        """Number of indexed barcodes at most one mismatch from each query (an
        indexed barcode equal to the query included)."""
        n_exact = self._count(self.exact, self._keys(codes))
        n_within = np.zeros(len(codes), dtype=np.int64)
        for k in range(self.length):
            n_within += self._count(self.masked[k], self._keys(codes, k))
        # an exact match matches every mask
        return n_within - (self.length - 1) * n_exact

    @staticmethod
    def _group_pairs(
        order: NDArray, runs: tuple[NDArray, NDArray, NDArray]
    ) -> tuple[NDArray, NDArray]:
        """All ordered pairs of distinct barcodes with equal (sorted) keys."""
        _, start, cnt = runs
        big = cnt > 1
        s, c = start[big], cnt[big]

        def expand(first: NDArray, size: NDArray) -> NDArray:  # first + 0..size-1
            return np.repeat(first - (np.cumsum(size) - size), size) + np.arange(
                size.sum()
            )

        elem = expand(s, c)
        elem_start, elem_cnt = np.repeat(s, c), np.repeat(c, c)
        a = np.repeat(elem, elem_cnt)
        b = expand(elem_start, elem_cnt)
        keep = a != b
        return order[a[keep]], order[b[keep]]

    def _neighbor_conflicts(self) -> tuple[NDArray, NDArray, NDArray]:
        """For each indexed barcode's one-mismatch neighbor, the other indexed
        barcodes within one mismatch of it.

        Barcode ``y`` is within one mismatch of a neighbor of ``x`` exactly when
        ``x`` and ``y`` are at most two mismatches apart: one mismatch, at ``p``,
        blocks all of ``x``'s neighbors at ``p``; two, at ``p`` and ``q``, block the
        neighbors taking ``y``'s base at ``p`` or at ``q``; duplicates block all.
        So conflicts come from grouping barcodes with one or two positions masked.

        :return: ``(neighbor, owner, blocker)``: for each ``(neighbor, blocker)``
            pair, an index into the blocked neighbors, whose ``owner`` is the
            barcode they are a neighbor of.
        """
        A = self.n_alphabet
        m = self.n_neighbors
        codes = self.codes.astype(np.int64)
        # neighbor j * (A - 1) + d - 1 of barcode i (base d steps on at position j)
        # has id i * m + j * (A - 1) + d - 1
        all_d = np.arange(A - 1)
        neighbors, blockers = [], []

        def toward(x: NDArray, y: NDArray, j: int) -> NDArray:
            return x * m + j * (A - 1) + (codes[y, j] - codes[x, j]) % A - 1

        for k in range(self.length):
            x, y = self._group_pairs(self.order[k], self.masked[k])
            one = codes[x, k] != codes[y, k]
            if k == 0:  # duplicates agree at every position
                xd, yd = x[~one], y[~one]
                neighbors.append((xd[:, None] * m + np.arange(m)).ravel())
                blockers.append(np.repeat(yd, m))
            x, y = x[one], y[one]
            neighbors.append((x[:, None] * m + k * (A - 1) + all_d).ravel())
            blockers.append(np.repeat(y, A - 1))
        for p in range(self.length):
            for q in range(p + 1, self.length):
                key = self._keys(self.codes, [p, q])
                o = np.argsort(key)
                x, y = self._group_pairs(o, self._runs(key[o]))
                two = (codes[x, p] != codes[y, p]) & (codes[x, q] != codes[y, q])
                x, y = x[two], y[two]
                neighbors.extend([toward(x, y, p), toward(x, y, q)])
                blockers.extend([y, y])
        ids, neighbor = np.unique(np.concatenate(neighbors), return_inverse=True)
        blocker = np.concatenate(blockers)
        return neighbor.astype(np.int32), ids // m, blocker.astype(np.int32)

    def capture(self, selected: NDArray | None = None) -> NDArray:
        """Number of each indexed barcode's one-mismatch neighbors that are within
        one mismatch of no other indexed barcode, i.e. that one-mismatch correction
        (:func:`correct_mismatches` with ``delta_mismatches=1``) assigns to it.

        The neighbor conflicts are computed once, so capture within many subsets of
        the indexed barcodes is cheap.

        :param selected: Boolean mask of the indexed barcodes to correct against
            (all by default); capture is meaningful only for selected barcodes.
        """
        if self._conflicts is None:
            self._conflicts = self._neighbor_conflicts()
        neighbor, owner, blocker = self._conflicts
        if selected is not None:
            neighbor = neighbor[selected[blocker]]
        blocked = np.zeros(len(owner), dtype=bool)
        blocked[neighbor] = True
        return self.n_neighbors - np.bincount(owner[blocked], minlength=self.n)


def _encode_alphabet(
    barcodes: Sequence[str] | NDArray, alphabet: Sequence[str]
) -> NDArray:
    """``(barcode, position)`` codes, 1 to A for the alphabet, A + 1 otherwise."""
    lut = np.full(256, len(alphabet) + 1, np.uint8)
    for i, a in enumerate(alphabet):
        lut[ord(a)] = i + 1
    chars = _encode_barcodes(barcodes)
    return lut[np.minimum(chars, 255)].astype(np.uint8)


def _barcode_alphabet(*barcode_sets: Sequence[str] | NDArray) -> list[str]:
    return sorted(set().union(*({c for b in bs for c in b} for bs in barcode_sets)))


def make_decoy_barcodes(
    barcodes: Sequence[str],
    n: int | None = None,
    min_mismatches: int = 2,
    max_rounds: int = 100,
    match_density: bool = True,
    seed: int = 0,
) -> NDArray:
    """Decoy barcodes for estimating how often spots match the whitelist by chance.

    Decoys are made by permuting each position independently across the whitelist,
    so each position has about the same base frequencies as the whitelist.  Each
    permutation keeps them exactly, but dropping candidates shifts them: candidates
    near the whitelist are rich in each position's common bases, so in skewed
    libraries decoys have fewer of them (e.g. by up to 0.18 at a position where
    the whitelist is 70% one base; under 0.03 for uniform libraries).
    Candidates within ``min_mismatches - 1`` mismatches of a whitelist barcode are
    dropped, as are whitelist barcodes and duplicates.  With the default of 2, a
    real read with one sequencing error cannot match a decoy exactly.

    With ``match_density``, the decoys are also as crowded as the whitelist, so
    one-mismatch correction (:func:`correct_mismatches` with
    ``delta_mismatches=1``) corrects as many one-mismatch neighbors, on average, to
    each decoy as to each target.  A barcode's one-mismatch neighbor is corrected
    to it only if no other barcode is within one mismatch of the neighbor, so
    barcodes with close neighbors collect fewer sequences.  Libraries are often
    more crowded than random barcodes, and permuted decoys can be more or less
    crowded among themselves than the whitelist is.  To match, a pool of
    candidates is drawn:

    - the decoys from permutation,
    - as many more permuted candidates (sparse choices), and
    - as many partners of the first decoys, exactly one or two mismatches away, at
      distinct positions (crowded choices).  Positions are chosen in proportion to
      the chance that redrawing them from the whitelist's base frequencies changes
      them, and each new base is drawn by frequency among the other bases, so
      partners nearly keep the whitelist's frequencies (only one or two positions
      change, and those lean slightly away from common bases).  Positions that
      are constant in the whitelist never change.

    Each batch may come up short when the whitelist leaves little room.  Decoys
    are then sampled from the pool, without replacement, with weights
    ``exp(lam * crowding)``, where crowding is the number of a candidate's
    one-mismatch neighbors that are also within one mismatch of another candidate
    (crowding between targets and decoys affects both alike, so it is not
    counted).  ``lam`` is set by bisection so that, with targets and decoys
    corrected together, the mean number of neighbors corrected to each decoy and
    to each target agree within 0.02; if they cannot, the closest pick is kept and
    a warning is logged.  As many decoys are picked as the first permutation
    made, so neither the extra candidates nor the partners make up a shortfall.

    :param barcodes: Whitelist barcodes, a one-dimensional sequence of strings
        (e.g. a list, array or :class:`pandas.Series`), all of the same length.
        Duplicates are ignored.
    :param n: Number of decoys (at least 1); defaults to the number of unique
        whitelist barcodes.
    :param min_mismatches: Minimum Hamming distance from every whitelist barcode.
        Values of 1 or less only require decoys to differ from the whitelist.
        Values above 2 need at least 2 unique whitelist barcodes.
    :param max_rounds: Maximum rounds of drawing for each batch of candidates
        before giving up.
    :param match_density: Match the whitelist's density of one-mismatch neighbors.
    :param seed: Random seed; the result is deterministic for a given seed.
    :return: Object array of unique decoy barcodes (Python strings), fewer than
        ``n`` (with a warning) if the whitelist leaves too little room.  The order
        carries no meaning.
    :raises ValueError: If ``barcodes`` is not one-dimensional (e.g. a
        :class:`pandas.DataFrame`; pass its barcode column), is empty, or the
        barcodes differ in length; if ``n`` is less than 1; or if
        ``min_mismatches`` is above 2 and the whitelist has only one unique
        barcode.
    """

    barcode_array = np.asarray(barcodes, dtype=object)
    if barcode_array.ndim != 1:
        raise ValueError(
            f"barcodes must be one-dimensional, got shape {barcode_array.shape}; "
            "pass a single column (e.g. df['barcode'])"
        )
    targets = np.unique(barcode_array)
    if len(targets) == 0:
        raise ValueError("barcodes is empty")
    lengths = {len(b) for b in targets}
    if len(lengths) != 1:
        raise ValueError(f"Barcodes must have the same length, got {sorted(lengths)}")
    n = len(targets) if n is None else int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if min_mismatches > 2 and len(targets) < 2:
        raise ValueError(
            "min_mismatches above 2 needs at least 2 unique whitelist barcodes"
        )
    # candidates only use whitelist bases, so the whitelist's alphabet covers them
    alphabet = _barcode_alphabet(targets)
    letters = np.array(alphabet)
    target_codes = _encode_alphabet(targets, alphabet)  # (barcode, position), 1..A
    length = target_codes.shape[1]

    @cache
    def target_index() -> _OneMismatchIndex:
        """Index of the whitelist, built on first use (for the min_mismatches == 2
        filter and density matching)."""
        return _OneMismatchIndex(target_codes, alphabet)

    def to_str(codes: NDArray) -> list[str]:
        """Barcodes from ``(barcode, position)`` codes (1..A)."""
        rows = np.ascontiguousarray(letters[codes - 1])
        return rows.view(f"<U{length}").ravel().tolist()

    # whether each row of candidate codes is far enough from the whitelist
    far_enough: Callable[[NDArray], NDArray]
    if min_mismatches == 2:
        # one sorted lookup per position instead of a nearest-neighbor search
        def far_enough(codes: NDArray) -> NDArray:
            return target_index().count_within(codes) == 0

    elif min_mismatches > 2:
        index = _fit_whitelist(targets)

        def far_enough(codes: NDArray) -> NDArray:
            reads = np.asarray(to_str(codes), dtype=object)
            mismatches = _hamming_distance(index, reads)["mismatches"]
            return mismatches.values >= min_mismatches

    else:

        def far_enough(codes: NDArray) -> NDArray:
            return np.ones(len(codes), dtype=bool)

    rng = np.random.default_rng(seed)
    seen = set(targets)

    def keep_new(codes: NDArray, k: int) -> list[str]:
        """Up to ``k`` of the barcodes encoded in ``codes`` (1..A) that are new and
        far enough from the whitelist, which are then marked as seen.  Codes are
        deduplicated and filtered before any strings are made."""
        # codes are 1..A, never 0, so fixed-width bytes keep every row intact
        uniq = np.unique(np.ascontiguousarray(codes, np.uint8).view(f"S{length}"))
        codes = uniq.view(np.uint8).reshape(-1, length)
        codes = codes[far_enough(codes)]
        cand = [c for c in to_str(codes) if c not in seen]
        cand = [cand[i] for i in rng.permutation(len(cand))[:k]]
        seen.update(cand)
        return cand

    def permuted(k: int) -> list[str]:
        """Up to ``k`` new candidates from permuting each position."""
        out: list[str] = []
        for _ in range(max_rounds):
            if len(out) >= k:
                break
            perm = np.stack(
                [rng.permutation(target_codes[:, j]) for j in range(length)], axis=1
            )
            out.extend(keep_new(perm, k - len(out)))
        return out

    decoys = permuted(n)
    if match_density and decoys:
        # more permuted candidates, so the pool also has sparse candidates to pick
        # from (partners make the candidates they come from crowded too)
        extra = permuted(len(decoys))
        # partners 1 or 2 mismatches from the decoys, so the pool has crowded
        # candidates to pick from (random barcodes rarely have close neighbors)
        decoy_codes = _encode_alphabet(decoys, alphabet)
        freq = _base_frequencies(target_codes, len(alphabet))
        varies = ((freq > 0).sum(axis=1) > 1).any()  # else no partners exist
        partners: list[str] = []
        for _ in range(max_rounds if varies else 0):
            need = len(decoys) - len(partners)
            if need <= 0:
                break
            # partners of a random subset of decoys, twice as many as still
            # needed to allow for duplicates and candidates too close to targets
            rows = rng.permutation(len(decoy_codes))[: 2 * need]
            mut = _decoy_partners(decoy_codes[rows], freq, rng)
            partners.extend(keep_new(mut, need))
        # pick as many decoys as the first permutation made, so neither extra
        # candidates nor partners stand in for missing decoys (partners would make
        # the decoys more crowded than intended)
        n_pick = len(decoys)
        pool = np.array(decoys + extra + partners)
        # Python strings, as without match_density
        decoys = _match_decoy_density(
            target_index(), pool, alphabet, n_pick, rng
        ).tolist()
    if len(decoys) < n:
        logger.warning(
            f"Made {len(decoys)} of {n} decoys at least {min_mismatches} mismatches "
            "from the whitelist"
        )
    return np.array(decoys, dtype=object)


def _draw(weights: NDArray, rng: np.random.Generator) -> NDArray:
    """Column drawn from each row of nonnegative ``weights`` (each row with some
    positive weight), in proportion to the weights; never a zero-weight column."""
    cum = np.cumsum(weights, axis=1)
    u = rng.random(len(cum)) * cum[:, -1]
    idx = (cum <= u[:, None]).sum(axis=1)
    # u can round up to the row total; fall back to the last positive column
    last = weights.shape[1] - 1 - np.argmax(weights[:, ::-1] > 0, axis=1)
    return np.minimum(idx, last)


def _base_frequencies(codes: NDArray, n_alphabet: int) -> NDArray:
    """``(position, base)`` frequencies of ``(barcode, position)`` codes (1..A)."""
    return np.stack(
        [
            np.bincount(codes[:, j] - 1, minlength=n_alphabet)
            for j in range(codes.shape[1])
        ]
    ) / len(codes)


def _decoy_partners(codes: NDArray, freq: NDArray, rng: np.random.Generator) -> NDArray:
    """One partner of each barcode, one or two mismatches away.

    This is equivalent to redrawing a uniformly chosen position from the
    whitelist's base frequencies and keeping only redraws that change the base,
    without the wasted draws: positions are chosen in proportion to how likely a
    redraw changes them, and each new base is drawn by frequency among the other
    bases.  Keeping only changes makes the bases written lean away from common
    bases, but since only one or two positions change, partners nearly keep the
    whitelist's frequencies.  The two positions differ, and positions that are
    constant in the whitelist never change.

    :param codes: ``(barcode, position)`` codes (1..A) of the barcodes, using only
        bases that occur at each position of the whitelist.
    :param freq: The whitelist's base frequencies from :func:`_base_frequencies`;
        some position must have two or more bases.
    :return: Partner codes.
    """
    change = 1 - freq[np.arange(freq.shape[0]), codes - 1]  # (barcode, position)
    n_changeable = (change > 0).sum(axis=1)
    mut = codes.copy()
    n_mut = np.minimum(rng.integers(1, 3, size=len(mut)), n_changeable)
    for i in range(2):
        rows = np.flatnonzero(n_mut > i)
        pos = _draw(change[rows], rng)
        change[rows, pos] = 0  # the second mutation goes elsewhere
        base_w = freq[pos].copy()
        base_w[np.arange(len(rows)), mut[rows, pos] - 1] = 0
        mut[rows, pos] = _draw(base_w, rng) + 1
    return mut


def _match_decoy_density(
    target_index: _OneMismatchIndex,
    pool: NDArray,
    alphabet: Sequence[str],
    n: int,
    rng: np.random.Generator,
    tol: float = 0.02,
) -> NDArray:
    """Pick ``n`` decoys from ``pool`` as crowded as the targets.

    Crowding is the number of a candidate's one-mismatch neighbors that other
    candidates take from it (:meth:`_OneMismatchIndex.capture`).  Decoys are the top ``n``
    candidates by ``lam * crowding + g``, with ``g`` fixed Gumbel noise (sampling
    without replacement with weights ``exp(lam * crowding)``), and ``lam`` is set by
    bisection so the mean capture of decoys and targets, indexed together, agree
    within ``tol``.  If they cannot, the closest pick is returned with a warning.

    :param target_index: Index of the whitelist codes from :func:`_encode_alphabet`.
    :param pool: Candidate decoys, using only bases in ``alphabet``.
    :param alphabet: Alphabet used to encode the whitelist.
    """
    nt = target_index.n
    # targets and the whole pool indexed once (the pool merged into the target
    # index); each lam scores a subset of it
    index = target_index.extend(_encode_alphabet(pool, alphabet))
    codes = index.codes
    # crowding by other candidates only: crowding between targets and decoys is
    # mutual, so it changes both sides' capture alike and cannot close the gap
    in_pool = np.arange(len(codes)) >= nt
    crowd = index.n_neighbors - index.capture(in_pool)[nt:]
    g = rng.gumbel(size=len(pool))

    def pick(lam: float) -> NDArray:
        return np.sort(np.argsort(-(lam * crowd + g), kind="stable")[:n])

    gaps: dict[float, float] = {}

    def gap(lam: float) -> float:
        if lam not in gaps:
            selected = np.zeros(len(codes), dtype=bool)
            selected[:nt] = True
            selected[nt + pick(lam)] = True
            cap = index.capture(selected)
            gaps[lam] = float(cap[nt:][selected[nt:]].mean() - cap[:nt].mean())
        return gaps[lam]

    # gap falls as lam rises (more crowded decoys collect fewer neighbors)
    lo, hi = -1.0, 1.0
    while gap(lo) < 0 and lo > -16:
        lo *= 2
    while gap(hi) > 0 and hi < 16:
        hi *= 2
    if gap(lo) >= 0 and gap(hi) <= 0:  # bracketed
        for _ in range(12):
            mid = (lo + hi) / 2
            if abs(gap(mid)) < tol:
                break
            lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    logger.debug(f"density gaps by tilt: {sorted(gaps.items())}")
    lam = min(gaps, key=lambda x: abs(gaps[x]))
    if abs(gaps[lam]) >= tol:
        logger.warning(
            "Could not match the whitelist's density of one-mismatch neighbors; "
            f"decoys differ by {gaps[lam]:+.2f} neighbors per barcode"
        )
    return pool[pick(lam)]
