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


def peaks_to_bases(
    maxed: xr.DataArray,
    peaks: pd.DataFrame | dd.DataFrame,
    labels: np.ndarray | xr.DataArray | None = None,
    labels_only: bool = True,
    bases: Sequence[str] | None = ("G", "T", "A", "C"),
) -> xr.DataArray:
    """Convert peaks to bases.

    :param maxed: Maxed array (sigma,t,c,y,x)
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

    maxed_spots = (
        maxed.isel(y=xr.DataArray(peaks["y"]), x=xr.DataArray(peaks["x"]))
        .rename({"dim_0": "read"})
        .transpose("read", ...)
    )
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
