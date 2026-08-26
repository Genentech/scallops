"""Reads Processing Module.

This module provides functionalities for processing and analyzing sequencing reads.
It includes tools for filtering, mapping, and summarizing reads, as well as for generating
various quality control metrics.

Authors:
- The SCALLOPS development team
"""

import itertools
import logging
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
from scipy.special import softmax
from skimage.segmentation import expand_labels
from skimage.util import img_as_float
from sklearn.linear_model import QuantileRegressor
from sklearn.neighbors import NearestNeighbors

from scallops.io import CYAN, GRAY, GREEN, MAGENTA, RED, save_stack_imagej

logger = logging.getLogger("scallops")


def _hamming_distance(
    whitelist_barcodes: np.ndarray, read_barcodes: np.ndarray
) -> pd.DataFrame:
    """Computes the Hamming distance between read barcodes and a whitelist of barcodes.

    :param whitelist_barcodes: A numpy array of whitelist barcodes.
    :param read_barcodes: A numpy array of read barcodes to compare against the whitelist.
    :return: A DataFrame containing the mismatches and closest matching barcodes from
    the whitelist.
    """
    if len(read_barcodes) == 0:
        return pd.DataFrame(
            {
                "mismatches": np.ndarray((0,), dtype=int),
                "closest_match": np.ndarray((0,), dtype=whitelist_barcodes.dtype),
                "mismatches2": np.ndarray((0,), dtype=int),
                "closest_match2": np.ndarray((0,), dtype=whitelist_barcodes.dtype),
            }
        )

    assert len(whitelist_barcodes[0]) == len(read_barcodes[0]), (
        f"Length of whitelist barcode ({len(whitelist_barcodes[0])}) != length of read ({len(read_barcodes[0])})."
    )
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="hamming").fit(
        np.array([[ord(b) for b in read] for read in whitelist_barcodes])
    )
    distances, argmin = nbrs.kneighbors(
        np.array([[ord(b) for b in read] for read in read_barcodes])
    )
    distances = (distances * len(whitelist_barcodes[0])).astype(int)
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
        whitelist_barcodes=barcodes_df["barcode"].values,
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


def quality_softmax(x: np.ndarray, min_error: float = 1e-6) -> np.ndarray:
    """Computes the phred quality score of transformed data using the softmax function.

    :param x: Array with transformed data (read, cycle, channel).
    :param min_error: Minimum p-value error.
    :return: Array with computed quality scores (higher is better) with shape (read, cycle)
    """

    p = np.max(softmax(x, axis=2), axis=2)

    p_error = 1 - p
    p_error[p_error < min_error] = min_error
    return -10 * np.log10(p_error)


def channel_probs(x: np.ndarray, min_error: float = 1e-6) -> np.ndarray:
    """Step 1 of Bernoulli basecalling: per-channel P(bright) via two-class softmax
    against the per-read dark reference (minimum intensity across cycles).

    :param x: (read, cycle, n_channels) intensities after crosstalk correction.
    :param min_error: Probability clipping floor to avoid log(0).
    :return: (read, cycle, n_channels) brightness probabilities.
    """
    # Per-channel midpoint: each channel is normalized against its own per-read range,
    # so cycles where that channel fires (bright) score > 0.5 and cycles where it doesn't
    # (dark) score < 0.5. A global reference breaks 4-color data because all channels
    # compete against the single brightest channel in the whole read.
    dark = x.min(axis=1, keepdims=True)  # (read, 1, n_channels)
    bright = x.max(axis=1, keepdims=True)  # (read, 1, n_channels)
    mid = np.broadcast_to((dark + bright) / 2, x.shape)  # (read, cycle, n_channels)
    p = softmax(np.stack([x, mid], axis=-1), axis=-1)  # (read, cycle, n_channels, 2)
    return np.clip(p[..., 0], min_error, 1 - min_error)


def base_probs(p: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Step 2 of Bernoulli basecalling: per-base probabilities via encoding matrix.

    For each base b, P(b) = product_c [ p_c^E[b,c] * (1-p_c)^(1-E[b,c]) ], normalised
    across bases. In log space this is a matrix multiply, avoiding numerical underflow.

    :param p: (read, cycle, n_channels) from channel_probs.
    :param E: (n_bases, n_channels) binary encoding matrix defining the chemistry.
    :return: (read, cycle, n_bases) normalised base probabilities.
    """
    log_p = np.log(p)
    log_1mp = np.log(1 - p)
    log_bp = log_p @ E.T + log_1mp @ (1 - E).T
    bp = np.exp(log_bp)
    return bp / bp.sum(axis=-1, keepdims=True)


def make_encoding(
    channel_bases: list[str],
    dark_bases: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a Bernoulli encoding matrix for the common case where each channel maps
    1:1 to one base and remaining bases are dark (no fluorescent signal).

    For chemistries where a base lights up multiple channels (e.g. Illumina 2-color
    where A appears in both red and green), construct the encoding matrix directly
    and pass it to decode_max via the `encoding` and `base_labels` parameters.

    :param channel_bases: Ordered base label for each channel, length = n_channels.
    :param dark_bases: Bases that have no dedicated channel.
    :return: (E, base_labels) where E is (n_bases, n_channels) float array and
        base_labels is the full ordered base array (channel_bases + dark_bases).
    """
    all_bases = list(channel_bases) + list(dark_bases)
    E = np.eye(len(all_bases), len(channel_bases), dtype=float)
    return E, np.array(all_bases)


def channel_crosstalk_matrix(
    a: xr.DataArray,
    method: Literal["median", "li_and_speed"] = "median",
    by_t: bool = False,
    **kwargs,
) -> ndarray | da.Array | dict[str, list[str] | np.ndarray]:
    """Estimate and correct differences in channel intensity and spectral overlap among sequencing
    channels using either median or Li and Speed method.

    Describe with linear transformation w so that w * a = y, where y is the corrected data.

    :param a: data to compute crosstalk matrix for (read, t, c)
    :param by_t: Compute separate matrices per cycle.
    :param method: Either median or li_and_speed
    :return: The inverse matrix, w (c,c) or an array of (t, c, c) if `by_t`
    """

    method = str(method).lower()
    assert method in ["median", "li_and_speed"]

    nchannels = a.sizes["c"]
    if isinstance(a.data, da.Array) and method == "median":
        dims = a.dims
        a = a.data
        chunksize = list(a.chunksize)
        dims_no_chunk = ["c"]
        if not by_t:
            dims_no_chunk.append("t")
        else:
            # timepoints separate
            chunksize[dims.index("t")] = 1
        # no chunking in t or c dimensions
        for i in range(len(dims)):
            if dims[i] in dims_no_chunk and a.chunksize[i] != a.shape[i]:
                chunksize[i] = -1

        chunksize = tuple(chunksize)
        if chunksize != a.chunksize:
            a = a.rechunk(chunksize)
        chunks = list(a.chunks)
        reads_per_chunk = np.array(chunks[0])
        chunks[dims.index("read")] = (1,) * len(chunks[dims.index("read")])
        if not by_t:
            chunks[dims.index("t")] = (1,) * len(chunks[dims.index("t")])
        # drop c dimension
        chunks[dims.index("c")] = (nchannels * nchannels,)
        # w is now (chunk, t, 16)
        w = da.map_blocks(
            _crosstalk_median_ratio_per_chunk, a, chunks=tuple(chunks), dtype=float
        )
        # weighted by number of reads in each chunk
        w = da.average(w, weights=reads_per_chunk, axis=0)
        # w is now (t, 16)
        w = w.reshape(-1, nchannels, nchannels).squeeze()
        if w.ndim == 2:
            return da.linalg.inv(w)
        else:
            w_inv = []
            for t in range(len(w)):
                w_inv.append(da.linalg.inv(w[t]))
            w_inv = da.stack(w_inv, axis=0)
            return w_inv

    method = (
        _correct_channel_crosstalk_li_and_speed
        if method == "li_and_speed"
        else _correct_channel_crosstalk_median
    )
    if by_t:
        dims = ["t"]
        dim_vals = [a[d].values for d in dims]
        w_arrays = []

        for dim_val in itertools.product(*dim_vals):
            sel = dict(zip(dims, dim_val))
            w = method(a.sel(sel).data.reshape(-1, a.sizes["c"]), **kwargs)
            w_arrays.append(w)

        return np.array(w_arrays)
    return method(a.data.reshape(-1, a.sizes["c"]), **kwargs)


def _crosstalk_median_ratio_per_chunk(x: np.ndarray) -> np.ndarray:
    """Compute the crosstalk median ratio for a given chunk of data.

    :param x: Input 2 or 3d array representing the chunk data.
    :return: The median ratio array.
    """
    nchannels = x.shape[-1]
    x = x.reshape(-1, nchannels)
    return _crosstalk_median_ratio(x).reshape(1, 1, nchannels * nchannels)


def _correct_channel_crosstalk_median(a: np.ndarray) -> np.ndarray:
    """Estimate and correct differences in channel intensity and spectral overlap among
    sequencing channels. For each channel, find points where the largest signal is from
    that channel. Use the median of these points to define new basis vectors.
    Describe with linear transformation w, so that w * x_array = y, where y is the
    corrected data.

    :param a: raw data to transform (read + t, c)
    :return: The inverse matrix, w
    """

    return np.linalg.inv(_crosstalk_median_ratio(a))


def apply_channel_crosstalk_matrix(
    a: xr.DataArray, w: np.ndarray | da.Array, dtype=np.float32
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
        chunksize = list(a.chunksize)
        # no chunking in t or c dimensions
        no_chunk_dims = ["c"]
        if w.ndim == 3:
            no_chunk_dims.append("t")
        for i in range(len(dims)):
            if dims[i] in no_chunk_dims and a.chunksize[i] != a.shape[i]:
                chunksize[i] = -1
        chunksize = tuple(chunksize)
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


def _crosstalk_median_ratio(a: np.ndarray) -> np.ndarray:
    """Compute the median ratio of the input array to quantify crosstalk between channels.

    :param a: Input 2D array where each row represents data for a specific observation, and each
        column corresponds to a channel.
    :return: A normalized 2D array of median ratios for each channel.
    """
    max_indices = a.argmax(axis=1)  # Indices of maximum values per row
    median_array = np.array(
        [np.median(a[max_indices == i], axis=0) for i in range(a.shape[1])]
    ).T
    totals = median_array.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        median_array = median_array / totals

    median_array[np.isnan(median_array)] = 1  # Replace NaNs with 1
    return median_array


def _agg_barcodes(df: pd.DataFrame, sort_by: str | list[str]) -> pd.DataFrame:
    """Aggregate barcode counts/intensities from a dataframe.

    :param df: Dataframe containing `label`, `barcode`, and `peak` columns.
    :param sort_by: Column(s) to sort by
    :return: Aggregated dataframe with top 2 barcodes for each label.
    """
    df_perfect_match = df[df["mismatches"].isna()] if "mismatches" in df.columns else df
    mismatch = False
    if len(df_perfect_match) > 0:
        df = df_perfect_match
    else:
        df = df[~df["mismatches"].isna()]
        mismatch = True
    if len(df) == 0:
        return pd.DataFrame()

    peak_sum = df["peak"].sum()
    count_sum = len(df)
    q_mean_sum = df["Q_mean"].sum()
    q_min_sum = df["Q_min"].sum()
    label = df.iloc[0]["label"]

    barcode_groupby = df.groupby(
        "barcode", as_index=True, group_keys=True, sort=False, dropna=False
    )
    barcode_sizes = barcode_groupby.size()
    top2 = barcode_groupby.agg("sum").nlargest(n=2, columns=sort_by)
    barcode_0 = top2.iloc[0].name
    barcode_1 = top2.iloc[1].name if len(top2) > 1 else ""

    return pd.DataFrame.from_dict(
        {
            "label": [label],
            "mismatch": [mismatch],
            "barcode_Q_mean": [q_mean_sum],
            "barcode_Q_min": [q_min_sum],
            "barcode_peak": [peak_sum],
            "barcode_count": [count_sum],
            "barcode_0": [barcode_0],
            "barcode_Q_mean_0": [top2.iloc[0]["Q_mean"]],
            "barcode_Q_min_0": [top2.iloc[0]["Q_min"]],
            "barcode_peak_0": [top2.iloc[0]["peak"]],
            "barcode_count_0": [barcode_sizes.loc[barcode_0]],
            "barcode_Q_0": [top2.iloc[0]["Q"]],
            "barcode_1": [barcode_1],
            "barcode_Q_mean_1": [top2.iloc[1]["Q_mean"] if len(top2) > 1 else 0],
            "barcode_Q_min_1": [top2.iloc[1]["Q_min"] if len(top2) > 1 else 0],
            "barcode_peak_1": [top2.iloc[1]["peak"] if len(top2) > 1 else np.nan],
            "barcode_count_1": [barcode_sizes.loc[barcode_1] if len(top2) > 1 else 0],
            "barcode_Q_1": [top2.iloc[1]["Q"] if len(top2) > 1 else None],
        }
    )


def assign_barcodes_to_labels(
    df_reads: pd.DataFrame | dd.DataFrame,
    sort_by: str | list[str] = ["Q_mean", "peak"],
) -> pd.DataFrame | dd.DataFrame:
    """Call the barcode for each label.

    :param df_reads: Table of all called reads. See :func:`~decode_max`
    :param sort_by: Which column(s) to sort aggregated barcodes by.
    :return: Table of all labels containing sequencing reads.
    """

    columns = []
    columns.append(("label", df_reads["label"].dtype))
    columns.append(("mismatch", bool))
    columns.append(("barcode_Q_mean", np.float64))
    columns.append(("barcode_Q_min", np.float64))
    columns.append(("barcode_peak", np.float64))
    columns.append(("barcode_count", np.int64))

    for i in range(2):
        columns.append((f"barcode_{i}", object))
        columns.append((f"barcode_Q_mean_{i}", np.float64))
        columns.append((f"barcode_Q_min_{i}", np.float64))
        columns.append((f"barcode_peak_{i}", np.float64))
        columns.append((f"barcode_count_{i}", np.int64))
        columns.append((f"barcode_Q_{i}", object))

    apply_args = (
        dict(meta=dd.utils.make_meta(columns))
        if isinstance(df_reads, dd.DataFrame)
        else dict()
    )
    apply_args["sort_by"] = sort_by
    reads_columns = ["label", "peak", "barcode", "Q", "Q_min", "Q_mean"]
    if "mismatches" in df_reads.columns:
        reads_columns.append("mismatches")
    return df_reads.groupby("label", group_keys=False, sort=False, dropna=False)[
        reads_columns
    ].apply(_agg_barcodes, **apply_args)


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

    def _single_partition(reads_df, whitelist_barcodes):
        reads_df = reads_df.join(
            _hamming_distance(
                whitelist_barcodes=whitelist_barcodes,
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
        return reads_df

    if is_dask:
        meta = dd.utils.make_meta(reads)
        meta["mismatches"] = pd.Series(dtype=int)
        meta["closest_match"] = pd.Series(dtype=object)
        meta["mismatches2"] = pd.Series(dtype=int)
        meta["closest_match2"] = pd.Series(dtype=object)
        meta["barcode_uncorrected"] = pd.Series(dtype=object)
        reads = dd.map_partitions(
            _single_partition, reads, barcodes["barcode"].values, meta=meta
        )
        return reads
    else:
        return _single_partition(reads, barcodes["barcode"].values)


def _decode_max_chunk(
    spots: np.ndarray,
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
    """Call reads by assigning the base with the highest intensity (softmax argmax).

    :param spots: Spots returned from peaks_to_bases containing dimensions (read, t, c).
    :param barcodes: Table of designed barcode sequences used for indicating whether a
        barcode is an exact match. Expected to have column 'barcode'.
    :return: The reads data frame.
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
        # no chunking in t or c dimension
        dims = spots.dims
        chunksize = list(spots.data.chunksize)
        for i in range(len(dims)):
            if dims[i] in ("c", "t") and spots.data.chunksize[i] != spots.data.shape[i]:
                chunksize[i] = -1
        chunksize = tuple(chunksize)
        if chunksize != spots.data.chunksize:
            spots.data = spots.data.rechunk(chunksize)

        columns = []
        for col in meta_df.columns:
            columns.append((col, meta_df[col].dtype))
        columns.append(("barcode", object))
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


def _decode_se_chunk(
    spots: np.ndarray,
    encoding: np.ndarray,
    base_labels: np.ndarray,
    meta_df: pd.DataFrame,
    offset: slice | None,
    whitelist: list[str] | None,
) -> pd.DataFrame:
    """Signed-encoding chunk decoder with data-adaptive f = alpha/3.

    For each cycle assigns the base that maximises
    ``score(b,t) = Σ_c (2E[b,c]-1) · (x_ct - l_c - f(h_c-l_c))``,
    where f = alpha/3 and alpha = min_c(mean top-phi_c normalised bright cycles).

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
    encoding: np.ndarray | None = None,
    base_labels: list[str] | np.ndarray | None = None,
    dark_bases: list[str] | None = None,
) -> pd.DataFrame | dd.DataFrame:
    """Call reads using the signed-encoding (SE) decoder with data-adaptive f = alpha/3.

    Outperforms softmax argmax for all SBS chemistries.  Handles dark bases (all-zero
    rows in ``E``) automatically — their score ``-Σ d_ct`` is maximised at quiet cycles.
    Per-channel range weighting acts as a SNR proxy, down-weighting dim channels.

    :param spots: Spots DataArray with dims ``(read, t, c)``.
    :param barcodes: Whitelist DataFrame with column ``'barcode'``; adds ``barcode_match``.
    :param encoding: ``(n_bases, n_channels)`` binary encoding matrix.  All-zero rows
        indicate dark bases.  When omitted and ``dark_bases`` is also omitted, defaults
        to the identity (one bright base per channel, no dark bases).
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

    whitelist = barcodes["barcode"].values if barcodes is not None else None
    meta_df = spots["read"].to_dataframe()

    if not isinstance(spots.data, da.Array):
        df = _decode_se_chunk(
            spots=spots.data,
            encoding=encoding,
            base_labels=base_labels,
            offset=None,
            meta_df=meta_df,
            whitelist=whitelist,
        )
    else:
        dims = spots.dims
        chunksize = list(spots.data.chunksize)
        for i in range(len(dims)):
            if dims[i] in ("c", "t") and spots.data.chunksize[i] != spots.data.shape[i]:
                chunksize[i] = -1
        chunksize = tuple(chunksize)
        if chunksize != spots.data.chunksize:
            spots.data = spots.data.rechunk(chunksize)

        columns = [(col, meta_df[col].dtype) for col in meta_df.columns]
        columns += [
            ("barcode", object),
            ("Q", object),
            ("Q_mean", np.float64),
            ("Q_min", np.float64),
        ]
        if whitelist is not None:
            columns.append(("barcode_match", bool))
        meta = dd.utils.make_meta(columns)
        _chunk_d = delayed(_decode_se_chunk)
        wl_d = delayed(whitelist)
        mdf_d = delayed(meta_df)
        enc_d = delayed(encoding)
        bla_d = delayed(base_labels)
        starts = [cached_cumsum(bds, initial_zero=True) for bds in spots.data.chunks]
        results = []
        for block in spots.data.to_delayed().ravel():
            key = np.array(block.key[1:])
            start = [starts[i][key[i]] for i in range(len(starts))]
            stop = [starts[i][key[i] + 1] for i in range(len(starts))]
            results.append(
                _chunk_d(
                    spots=block,
                    encoding=enc_d,
                    base_labels=bla_d,
                    meta_df=mdf_d,
                    whitelist=wl_d,
                    offset=slice(start[0], stop[0]),
                )
            )
        df = dd.from_delayed(results, meta=meta, verify_meta=False)

    return df


def _polar_thresholds_from_wcor(
    w_cor: np.ndarray,
    channel_bases: list[str],
) -> dict:
    """Infer polar angle thresholds from the xtalk correction matrix.

    Computes the expected signal direction for each base via the forward xtalk
    model (``W_fwd = inv(w_cor)``), then places angle boundaries at the midpoint
    between adjacent centroid directions.  Fully parameter-free.

    The xtalk matrix size (``w_cor.shape[0]``) determines the signal model.
    The number of channels in the target data (``len(channel_bases)``) determines
    which polar coordinate system to use.  These may differ — for example, a
    3-channel xtalk matrix can be used to compute 2-colour angle thresholds when
    the 2-colour channels are synthesised from the 3 original channels via max().

    :param w_cor: ``(n, n)`` xtalk correction matrix from
        :func:`channel_crosstalk_matrix`, where ``n`` is the number of physical
        channels used for correction.
    :param channel_bases: Ordered base labels for each **data** channel (may be
        fewer than ``n`` for synthesised channels such as 2-colour Illumina).
    :return: Dict with keys ``t1``, ``t2`` (3-channel spherical) or ``t_lo``,
        ``t_hi`` (2-channel planar), giving angle boundaries in degrees.
    """
    # Use the xtalk matrix size for E_soft computation (physical channels)
    n_xtalk = w_cor.shape[0]
    n_data = len(channel_bases)

    w_fwd = np.linalg.inv(w_cor)
    col_sums = w_fwd.sum(axis=0, keepdims=True).clip(1e-9)
    e_soft = (w_fwd / col_sums).T  # (n_xtalk, n_xtalk): rows = bases

    e_norm = e_soft / np.linalg.norm(e_soft, axis=1, keepdims=True).clip(1e-9)

    if n_data == 3 or (n_data != 2 and n_xtalk == 3):
        # Spherical: theta1 from axis-0, theta2 in transverse 1-2 plane
        theta1 = np.degrees(np.arccos(np.clip(e_norm[:, 0], 0.0, 1.0)))
        theta2 = np.degrees(np.arctan2(e_norm[:, 2] + 1e-9, e_norm[:, 1] + 1e-9))
        t1 = (theta1[0] + min(theta1[1], theta1[2])) / 2
        t2 = (theta2[1] + theta2[2]) / 2
        return {"t1": float(t1), "t2": float(t2)}

    # Planar (2-colour Illumina): channels are synthesised as
    #   ch0 = max(A_ch, C_ch),  ch1 = max(A_ch, T_ch)
    # E_soft rows: A=0, T=1, C=2  (for 3-channel physical xtalk)
    # 2-col encoding bl=[G,T,A,C]: G=dark, T=ch1-only, A=both, C=ch0-only
    e_2col = np.zeros((4, 2))
    if n_xtalk >= 3:
        # Use all 3 physical channel directions
        e_2col[1, 0] = max(e_soft[1, 0], e_soft[1, 2])  # T in ch0=max(A,C)
        e_2col[1, 1] = max(e_soft[1, 0], e_soft[1, 1])  # T in ch1=max(A,T)
        e_2col[2, 0] = max(e_soft[0, 0], e_soft[0, 2])  # A in ch0
        e_2col[2, 1] = max(e_soft[0, 0], e_soft[0, 1])  # A in ch1
        e_2col[3, 0] = max(e_soft[2, 0], e_soft[2, 2])  # C in ch0
        e_2col[3, 1] = max(e_soft[2, 0], e_soft[2, 1])  # C in ch1
    else:
        # Fallback: use 2-channel xtalk directly
        e_2col[1, :] = e_soft[min(1, n_xtalk - 1)]
        e_2col[2, :] = e_soft[0]
        e_2col[3, 0] = e_soft[min(2, n_xtalk - 1), 0]
        e_2col[3, 1] = e_soft[min(2, n_xtalk - 1), min(1, n_xtalk - 1)]

    e_n2 = e_2col / np.linalg.norm(e_2col, axis=1, keepdims=True).clip(1e-9)
    theta_cent = np.degrees(np.arctan2(e_n2[:, 1] + 1e-9, e_n2[:, 0] + 1e-9))
    t_lo = (theta_cent[3] + theta_cent[2]) / 2  # C / A boundary
    t_hi = (theta_cent[2] + theta_cent[1]) / 2  # A / T boundary
    return {"t_lo": float(t_lo), "t_hi": float(t_hi)}


def _rfrac_from_sweep(
    Rn: np.ndarray,
    bright_calls: np.ndarray,
    dark_base_idx: int,
    base_labels_arr: np.ndarray,
    whitelist: np.ndarray,
    n_grid: int = 20,
) -> float:
    """Find the r_frac that maximises whitelist mapping rate via a vectorised grid sweep.

    All r_frac values in ``[0.05, 0.50]`` are evaluated simultaneously using
    broadcasting — no Python loop over the grid.  Cost is O(n·T·G) in memory
    and O(n·T·G + n·G·log|WL|) in time, where G = ``n_grid``.

    :param Rn: ``(n, T)`` per-read normalised radius ``R / R_max``.
    :param bright_calls: ``(n, T)`` base index (into ``base_labels_arr``) when
        the cycle is classified as bright.
    :param dark_base_idx: Index into ``base_labels_arr`` for the dark base.
    :param base_labels_arr: Ordered base labels (length = alphabet size).
    :param whitelist: 1-D array of barcode strings.
    :param n_grid: Number of evenly-spaced r_frac candidates in ``[0.05, 0.50]``.
    :return: Optimal r_frac as a float.
    """
    T = Rn.shape[1]
    n_bases = len(base_labels_arr)
    powers = (n_bases ** np.arange(T)).astype(np.int64)

    # Encode whitelist as sorted int64 array.
    # Single join+encode avoids one Python .encode() call per barcode.
    char_to_idx = {label: i for i, label in enumerate(base_labels_arr)}
    wl_valid = [s for s in whitelist if len(s) == T]
    if not wl_valid:
        return 0.11 * np.sqrt(Rn.shape[-1] if Rn.ndim > 1 else 1)
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

    # Vectorised evaluation — search [0.05, 0.40]; optimal is never above 0.35
    rfrac_grid = np.linspace(0.05, 0.40, n_grid)
    mask = Rn[:, :, np.newaxis] < rfrac_grid[np.newaxis, np.newaxis, :]  # (n, T, G)
    corr = (delta[:, :, np.newaxis] * mask).sum(axis=1)  # (n, G)
    bc = bright_ints[:, np.newaxis] + corr  # (n, G)

    pos = np.searchsorted(wl_ints, bc)
    pos = np.clip(pos, 0, len(wl_ints) - 1)
    rates = (wl_ints[pos] == bc).mean(axis=0)  # (G,)

    return float(rfrac_grid[np.argmax(rates)])


def _decode_polar_chunk(
    spots: np.ndarray,
    E: np.ndarray,
    base_labels_arr: np.ndarray,
    bright_rows: np.ndarray,
    dark_rows: np.ndarray,
    has_dark: bool,
    r_frac: float,
    non_orthogonal: bool,
    t_lo: float,
    t_hi: float,
    meta_df: pd.DataFrame,
    whitelist: np.ndarray | None,
    offset: slice | None,
) -> pd.DataFrame:
    """Process one chunk of spots through the polar decoder (numpy-only)."""
    sp = np.clip(spots, 0.0, None)
    lo = sp.min(axis=1, keepdims=True)
    d = sp - lo

    E_bright = E[bright_rows]
    if non_orthogonal:
        c0, c1 = d[..., 0] + 1e-9, d[..., 1] + 1e-9
        theta = np.degrees(np.arctan2(c1, c0))
        bright_calls = np.where(
            theta <= t_lo,
            bright_rows[2],
            np.where(theta >= t_hi, bright_rows[0], bright_rows[1]),
        )
    else:
        bright_scores = d @ E_bright.T
        bright_calls = bright_rows[bright_scores.argmax(axis=-1)]

    if has_dark:
        R = np.sqrt((d**2).sum(axis=-1))
        R_max = R.max(axis=1, keepdims=True)
        Rn = np.divide(R, R_max, out=np.zeros_like(R), where=R_max > 0)
        dark_mask = Rn < r_frac
    else:
        dark_mask = np.zeros(d.shape[:2], dtype=bool)

    calls = np.where(dark_mask, dark_rows[0] if has_dark else 0, bright_calls)

    Q = quality_softmax(sp)
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


def decode_polar(
    spots: xr.DataArray,
    barcodes: pd.DataFrame | None = None,
    encoding: np.ndarray | None = None,
    base_labels: list[str] | np.ndarray | None = None,
    dark_bases: list[str] | None = None,
    w_cor: np.ndarray | None = None,
    r_frac: float | None = None,
) -> pd.DataFrame:
    """Call reads using polar-coordinate classification.

    .. note::
        For **orthogonal** encodings (each bright base fires exactly one channel),
        :func:`decode_max` with ``dark_bases`` or ``encoding`` is the preferred
        method, its SE formula accounts for per-channel dynamic range
        (SNR proxy) and consistently outperforms the polar dot-product argmax.
        Use ``decode_polar`` when the encoding is **non-orthogonal**, i.e. when at
        least one bright base fires more than one channel (e.g. Illumina 2-colour
        where A appears in both the red and green channels).  In that case the
        amplitude-independent angle :math:`\\theta` is the only reliable discriminant.

    Separates signal into two independent properties:

    * **Radius** :math:`R = \\|\\mathbf{x} - \\mathbf{lo}\\|_2` — detects dark bases
      (all channels quiet below :math:`r_{\\text{frac}} \\cdot R_{\\max}`).
    * **Direction** — determines which bright base fired.

    Two direction strategies, chosen automatically from the encoding matrix:

    **Orthogonal encoding** (each bright base fires exactly one channel,
    e.g. 4-colour identity or 3-colour dark-base):

    .. math::

        \\text{calls} = \\arg\\max_b\\; (\\mathbf{x} - \\mathbf{lo}) \\cdot E[b,:]

    **Non-orthogonal encoding** (some bright base fires multiple channels,
    e.g. Illumina 2-colour where A fires both red and green):

    .. math::

        \\theta = \\arctan\\!\\left(\\frac{d_{c_1}}{d_{c_0}}\\right),
        \\quad \\text{classified by xtalk-inferred} \\; \\theta_{\\text{low}}, \\theta_{\\text{high}}

    The branching depends on the **encoding matrix**, not on channel count —
    orthogonal chemistries use the dot-product regardless of dimensionality.

    Angle thresholds for non-orthogonal encodings are derived parameter-free
    from ``w_cor`` via :func:`_polar_thresholds_from_wcor`; if ``w_cor`` is
    ``None`` the theoretical midpoints (45°, etc.) are used.

    :param spots: Spot DataArray from :func:`peaks_to_bases`, dims ``(read, t, c)``.
    :param barcodes: Whitelist DataFrame with column ``'barcode'``; adds a
        ``barcode_match`` column to the result when provided.
    :param encoding: ``(n_bases, n_channels)`` encoding matrix.  All-zero rows
        indicate dark bases.  Takes priority over ``dark_bases``.
    :param base_labels: Base labels of length ``n_bases``, required with
        ``encoding``.
    :param dark_bases: Shorthand for 1:1 channel→base chemistry with named dark
        bases.  Builds the encoding matrix via :func:`make_encoding`.
    :param w_cor: Xtalk correction matrix from :func:`channel_crosstalk_matrix`.
        Used to derive angle thresholds for non-orthogonal encodings.
    :param r_frac: Per-read fraction of :math:`R_{\\max}` below which a cycle
        is classified as the dark base.  Ignored when no dark bases exist.
        When ``None`` (default) and dark bases are present, the optimal value
        is found automatically via :func:`_rfrac_from_sweep` if a barcode
        whitelist is provided, otherwise falls back to :math:`0.11\\sqrt{n_c}`.
    :return: DataFrame with columns ``barcode``, ``Q_mean``, ``Q_min``, and
        optionally ``barcode_match``.
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

    #  Angle thresholds (non-orthogonal only, parameter-free from w_cor) ─
    E_bright = E[bright_rows]
    non_orthogonal = bool(np.any(E_bright.sum(axis=1) > 1))
    if non_orthogonal:
        thresholds = (
            _polar_thresholds_from_wcor(w_cor, channel_bases)
            if w_cor is not None
            else {"t_lo": 22.5, "t_hi": 67.5}
        )
        t_lo, t_hi = thresholds["t_lo"], thresholds["t_hi"]
    else:
        t_lo, t_hi = 0.0, 0.0  # unused for orthogonal

    #  r_frac: auto-compute from a sample when the input is dask
    whitelist_arr = barcodes["barcode"].values if barcodes is not None else None
    if r_frac is None and has_dark:
        if whitelist_arr is not None:
            # Use up to 50 k spots for the sweep; materialise only that sample
            n_sample = min(50_000, spots.sizes["read"])
            sp_sample = np.clip(spots.isel(read=slice(0, n_sample)).data, 0.0, None)
            if hasattr(sp_sample, "compute"):
                sp_sample = sp_sample.compute()
            lo_s = sp_sample.min(axis=1, keepdims=True)
            d_s = sp_sample - lo_s
            R_s = np.sqrt((d_s**2).sum(axis=-1))
            Rmax_s = R_s.max(axis=1, keepdims=True)
            Rn_s = np.divide(R_s, Rmax_s, out=np.zeros_like(R_s), where=Rmax_s > 0)
            if non_orthogonal:
                c0s, c1s = d_s[..., 0] + 1e-9, d_s[..., 1] + 1e-9
                theta_s = np.degrees(np.arctan2(c1s, c0s))
                bc_s = np.where(
                    theta_s <= t_lo,
                    bright_rows[2],
                    np.where(theta_s >= t_hi, bright_rows[0], bright_rows[1]),
                )
            else:
                bc_s = bright_rows[(d_s @ E_bright.T).argmax(axis=-1)]
            r_frac = _rfrac_from_sweep(
                Rn_s, bc_s, int(dark_rows[0]), base_labels_arr, whitelist_arr
            )
        else:
            r_frac = 0.11 * np.sqrt(len(channel_bases))

    #  Dispatch: dask (chunked) or numpy (all at once)
    meta_df = spots["read"].to_dataframe()

    if not isinstance(spots.data, da.Array):
        df = _decode_polar_chunk(
            spots=spots.data,
            E=E,
            base_labels_arr=base_labels_arr,
            bright_rows=bright_rows,
            dark_rows=dark_rows,
            has_dark=has_dark,
            r_frac=r_frac if r_frac is not None else 0.0,
            non_orthogonal=non_orthogonal,
            t_lo=t_lo,
            t_hi=t_hi,
            meta_df=meta_df,
            whitelist=whitelist_arr,
            offset=None,
        )
    else:
        # Mirror the decode_max dask pattern: rechunk so t and c are whole,
        # then dispatch one delayed _decode_polar_chunk per read-chunk.
        dims = spots.dims
        chunksize = list(spots.data.chunksize)
        for i, dim in enumerate(dims):
            if dim in ("c", "t") and spots.data.chunksize[i] != spots.data.shape[i]:
                chunksize[i] = -1
        chunksize = tuple(chunksize)
        if chunksize != spots.data.chunksize:
            spots = spots.copy(data=spots.data.rechunk(chunksize))

        columns = [(col, meta_df[col].dtype) for col in meta_df.columns]
        columns += [
            ("barcode", object),
            ("Q", object),
            ("Q_mean", np.float64),
            ("Q_min", np.float64),
        ]
        if whitelist_arr is not None:
            columns.append(("barcode_match", bool))
        meta = dd.utils.make_meta(columns)

        _chunk_delayed = delayed(_decode_polar_chunk)
        meta_df_d = delayed(meta_df)
        whitelist_d = delayed(whitelist_arr)
        E_d = delayed(E)
        bla_d = delayed(base_labels_arr)
        br_d = delayed(bright_rows)
        dr_d = delayed(dark_rows)

        starts = [cached_cumsum(bds, initial_zero=True) for bds in spots.data.chunks]
        results = []
        for block in spots.data.to_delayed().ravel():
            key = np.array(block.key[1:])
            start = [starts[i][key[i]] for i in range(len(starts))]
            stop = [starts[i][key[i] + 1] for i in range(len(starts))]
            results.append(
                _chunk_delayed(
                    spots=block,
                    E=E_d,
                    base_labels_arr=bla_d,
                    bright_rows=br_d,
                    dark_rows=dr_d,
                    has_dark=has_dark,
                    r_frac=r_frac if r_frac is not None else 0.0,
                    non_orthogonal=non_orthogonal,
                    t_lo=t_lo,
                    t_hi=t_hi,
                    meta_df=meta_df_d,
                    whitelist=whitelist_d,
                    offset=slice(start[0], stop[0]),
                )
            )
        df = dd.from_delayed(results, meta=meta, verify_meta=False)

    return df


def peaks_to_bases(
    maxed: xr.DataArray,
    peaks: pd.DataFrame | dd.DataFrame,
    labels: np.ndarray | xr.DataArray | None = None,
    labels_only: bool = True,
    bases: Sequence[str] | None = ("G", "T", "A", "C"),
) -> xr.DataArray:
    """Convert peaks to bases.

    :param maxed: Maxed array (sigma,t,c,y,x) or (t,c,y,x)
    :param peaks: Peaks data frame which has been filtered to retain only peaks of interest. Note
        that if peaks is a dask data frame, it is loaded into memory using dask.compute.
    :param labels: Segmentation array (y,x)
    :param labels_only: If true, only return peaks where labels are present.
    :param bases: List of bases.
    :return: DataArray with dimensions (read,t,c) where read is spot with coordinates y, x, peak,
        and label
    """

    if "t" not in maxed.dims:
        maxed = maxed.expand_dims("t")

    assert maxed.dims == ("sigma", "t", "c", "y", "x") or maxed.dims == (
        "t",
        "c",
        "y",
        "x",
    ), f"Found dimensions: {maxed.dims}"

    if isinstance(peaks, dd.DataFrame):
        peaks = peaks.compute()

    if "sigma" in maxed.dims and maxed.sizes["sigma"] == 1:
        maxed = maxed.squeeze("sigma", drop=True)

    if isinstance(labels, xr.DataArray):
        labels = labels.values

    if labels_only and labels is not None:
        peaks = peaks[labels[peaks["y"], peaks["x"]] > 0]
    sigma_indices = None
    if "sigma" in maxed.dims:
        sigma = maxed.coords["sigma"].values
        sigma_to_index = {}
        for i in range(len(sigma)):
            sigma_to_index[sigma[i]] = i
        sigma_indices = peaks["sigma"].replace(sigma_to_index).values.astype(int)
    maxed_spots = (
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
    maxed_spots = maxed_spots.rename({"dim_0": "read"}).transpose("read", ...)
    maxed_spots.name = "maxed"

    for c in peaks.columns:
        # add columns in peaks
        maxed_spots.coords[c] = ("read", peaks[c])
    if labels is not None:
        maxed_spots.coords["label"] = ("read", labels[peaks["y"], peaks["x"]])
    if bases is not None:
        maxed_spots = maxed_spots.assign_coords(c=list(bases))
    return maxed_spots


def merge_sbs_phenotype(
    df_labels: pd.DataFrame | dd.DataFrame,
    df_phenotype: pd.DataFrame | dd.DataFrame,
    df_barcode: pd.DataFrame,
    sbs_cycles: Sequence[int],
    how: Literal["left", "right", "inner", "outer", "cross"] = "outer",
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
) -> np.ndarray:
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
    spots_labels: np.ndarray,
    calls: np.ndarray | pd.DataFrame,
    query: Sequence[str],
    expand_width: int = 1,
) -> np.ndarray:
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
    x: np.ndarray, y: np.ndarray, quantile_range: tuple[float, float] = (0.6, 0.999)
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
) -> tuple[np.ndarray, np.ndarray]:
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
    max_slope = 0
    for i, j in product(range(len(channels)), repeat=2):
        x = a.sel(c=channels[i]).to_numpy().flatten()
        if i != j:
            y = a.sel(c=channels[j]).to_numpy().flatten()
            _, slope = li_speed_slope(x, y)
            max_slope = max(max_slope, abs(slope))
    return max_slope
