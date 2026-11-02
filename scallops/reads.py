"""Reads Processing Module.

This module provides functionalities for processing and analyzing sequencing reads.
It includes tools for filtering, mapping, and summarizing reads, as well as for generating
various quality control metrics.

Authors:
- The SCALLOPS development team
"""

import itertools
import logging
import math
from collections.abc import Sequence
from itertools import product
from typing import Literal, Tuple

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

# dtype pandas infers for a column of strings (object, or str with pandas >= 3)
_STR_DTYPE = pd.Series([""]).dtype


def _encode_barcodes(barcodes: Sequence[str]) -> NDArray:
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

    :param: reads_df: DataFrame containing called reads
    :param: barcodes_df: DataFrame containing whitelisted barcodes
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
    :return: Array with computed quality scores (higher is better).
    """

    p = np.max(softmax(x, axis=2), axis=2)
    p_error = 1 - p
    p_error[p_error < min_error] = min_error
    return -10 * np.log10(p_error)


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
    :return: The inverse matrix, w (c,c) or an array of (t, c, c) if `by_t`
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

    :param df_reads: Table of all called reads. See :func:`~decode_max`
    :param sort_by: Which column(s) to sort aggregated barcodes by.
    :return: Table of all labels containing sequencing reads.
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

    :param reads: reads from decode_max
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
) -> pd.DataFrame:
    """Decode the maximum intensity chunk from the input spot data and compute base
    quality scores.

    :param spots: Spot data.
    :param bases: List of bases.
    :param meta_df: Metadata dataframe.
    :param offset: Offset into metadata.
    :param whitelist: List of whitelisted barcodes.
    :return: A pandas DataFrame with decoded barcode sequences and quality metrics.
    """
    Q = quality_softmax(spots)
    channel_calls = np.argmax(spots, axis=2)
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


def decode_max(
    spots: xr.DataArray,
    barcodes: pd.DataFrame | None = None,
) -> pd.DataFrame | dd.DataFrame:
    """Call reads by assigning the base with the highest intensity.

    :param spots: Spots returned from peaks_to_bases containing dimensions (read,t,c)
    :param barcodes: Table of designed barcode sequences used for indicating whether a
        barcode is an exact match. Expected to have column 'barcode'.
    :return: The reads data frame
    """

    whitelist = barcodes["barcode"].values if barcodes is not None else None
    meta_df = spots["read"].to_dataframe()  # index is read
    bases = spots.c.values
    if not isinstance(spots.data, da.Array):
        df = _decode_max_chunk(
            spots=spots.data,
            bases=bases,
            offset=None,
            meta_df=meta_df,
            whitelist=whitelist,
        )
    else:
        # # no chunking in t or c dimension
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
        _decode_max_chunk_delayed = delayed(_decode_max_chunk)
        whitelist = delayed(whitelist)
        bases = delayed(bases)
        meta_df = delayed(meta_df)
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
            results.append(
                _decode_max_chunk_delayed(
                    spots=block,
                    whitelist=whitelist,
                    meta_df=meta_df,
                    offset=slice(start[0], stop[0]),
                    bases=bases,
                )
            )
        df = dd.from_delayed(results, meta=meta, verify_meta=False)

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
    :return: DataArray with dimensions (read,t,c) where read is spot with coordinates y, x, peak,
        and label
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

            n_removed = n_barcodes_duplicated != n_barcodes
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


def barcode_to_prefix(barcode: pd.Series, sbs_cycles: Sequence[int]) -> str:
    """Utility function to generate prefixes based on barcodes.

    :param barcode: pd.Series with barcode information
    :param sbs_cycles: Cycles to work on (starting from 1)
    :return: Prefix combining all channels
    """
    return "".join(barcode[c - 1] for c in sbs_cycles)


def read_statistics(reads_df: pd.DataFrame | dd.DataFrame) -> dict[str, float | int]:
    """Compute read statistics, such as mapped_reads, mapped_reads_within_labels, and
    average_reads_per_label.

    :param reads_df: Reads data frame
    :return: Dictionary containing statistics
    """
    unique_cell_col = (
        "label" if "unique_label" not in reads_df.columns else "unique_label"
    )
    in_labels = reads_df.query("label!=0")
    outside_labels = reads_df.query("label==0")
    mapping_rate = reads_df.query("barcode_match").shape[0] / reads_df.shape[0]
    mapping_rate_within_labels = (
        in_labels.query("barcode_match").shape[0] / in_labels.shape[0]
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
    :return: Data frame containing read index and counts
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

    This function takes a DataFrame of decoded reads produced by the `scallops.reads.decode_max` function
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
