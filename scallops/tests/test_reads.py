import argparse
from pathlib import Path

import dask.array as da
import dask.config
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from scallops.io import read_image
from scallops.reads import (
    annotated_spots,
    apply_channel_crosstalk_matrix,
    assign_barcodes_to_labels,
    channel_crosstalk_matrix,
    correct_mismatches,
    decode_max,
    peaks_to_bases,
)
from scallops.spots import (
    find_peaks,
    max_filter,
    peak_thresholds_from_bases,
    peak_thresholds_from_reads,
    std,
    transform_log,
)

__root__ = Path(__file__).resolve().parent


def diff_reads(test_df_bases, test_df_reads, test_df_cell):
    known_good_df_bases = (
        pd.read_csv(
            str(__root__.joinpath("data", "process_fig4", "10X_A1_Tile-102.bases.csv"))
        )
        .rename(
            {"i": "y", "j": "x", "channel": "c", "cycle": "t", "cell": "label"}, axis=1
        )
        .reset_index()
        .drop(["index", "read", "tile", "well"], axis=1)
    )

    if "channel" in test_df_bases.columns:
        test_df_bases = test_df_bases.rename({"channel": "c", "cycle": "t"}, axis=1)

    test_df_bases = test_df_bases.sort_values(["label", "t", "c", "y", "x"])
    known_good_df_bases = known_good_df_bases.sort_values(["label", "t", "c", "y", "x"])

    pd.testing.assert_frame_equal(
        test_df_bases[known_good_df_bases.columns].set_index(
            pd.RangeIndex(len(test_df_bases))
        ),
        known_good_df_bases.set_index(pd.RangeIndex(len(known_good_df_bases))),
        check_exact=True,
        check_dtype=False,
    )
    skip_cols = [f"Q_{i}" for i in range(9)] + ["Q_min"]  # we use a different method
    # reads
    known_good_df_reads = (
        pd.read_csv(
            str(__root__.joinpath("data", "process_fig4", "10X_A1_Tile-102.reads.csv"))
        )
        .rename({"i": "y", "j": "x", "cell": "label"}, axis=1)
        .reset_index()
        .drop(["index", "read", "peak", "tile", "well"] + skip_cols, axis=1)
    ).sort_values(["label", "y", "x"])

    test_df_reads = test_df_reads.sort_values(["label", "y", "x"]).drop(
        ["read"], axis=1
    )
    test_df_reads = test_df_reads[known_good_df_reads.columns]

    pd.testing.assert_frame_equal(
        test_df_reads.set_index(pd.RangeIndex(len(test_df_reads))),
        known_good_df_reads.set_index(pd.RangeIndex(len(known_good_df_reads))),
        check_dtype=False,
    )

    # assigned cell barcodes can differ due to ties
    test_df_cell = (
        test_df_cell.reset_index(drop=True)
        .query("barcode_count_0 != barcode_count_1")
        .drop(["barcode_1"], axis=1)
        .sort_values("label")
    )
    known_good_df_cell = (
        pd.read_csv(
            str(__root__.joinpath("data", "process_fig4", "10X_A1_Tile-102.cells.csv"))
        )
        .rename(
            {
                "cell": "label",
                "cell_barcode_count_0": "barcode_count_0",
                "cell_barcode_count_1": "barcode_count_1",
                "cell_barcode_0": "barcode_0",
                "cell_barcode_1": "barcode_1",
            },
            axis=1,
        )
        .drop(["peak", "tile", "well"], axis=1)
    )
    known_good_df_cell = (
        known_good_df_cell.reset_index(drop=True)
        .query("barcode_count_0 != barcode_count_1")
        .drop(["barcode_1"], axis=1)
        .sort_values("label")
    )
    pd.testing.assert_frame_equal(
        test_df_cell[known_good_df_cell.columns].set_index(
            pd.RangeIndex(len(test_df_cell))
        ),
        known_good_df_cell.set_index(pd.RangeIndex(len(known_good_df_cell))),
        check_dtype=False,
    )


@pytest.mark.basecalls
def test_correct_mismatches():
    barcodes = pd.DataFrame(data=["AAAA", "GGGG", "CCCC", "TTTT"], columns=["barcode"])
    reads = pd.DataFrame(
        data=["AAAA", "AAAA", "AAAG", "ACCC", "ACAC", "ACAC"], columns=["barcode"]
    )

    corrected_reads = correct_mismatches(reads=reads, barcodes=barcodes, n_mismatches=2)
    # ACAC is equidistant to more than one barcode so not corrected
    # AAAG -> AAAA
    # ACCC -> CCCC
    expected_result = pd.DataFrame(
        data=[
            ["AAAA", True, np.nan, np.nan, np.nan, np.nan, np.nan],
            ["AAAA", True, np.nan, np.nan, np.nan, np.nan, np.nan],
            ["AAAA", True, "AAAG", 1, "AAAA", 3, "GGGG"],
            ["CCCC", True, "ACCC", 1, "CCCC", 3, "AAAA"],
            ["ACAC", False, np.nan, 2, "AAAA", 2, "CCCC"],
            ["ACAC", False, np.nan, 2, "AAAA", 2, "CCCC"],
        ],
        columns=[
            "barcode",
            "barcode_match",
            "barcode_uncorrected",
            "mismatches",
            "closest_match",
            "mismatches2",
            "closest_match2",
        ],
    )
    pd.testing.assert_frame_equal(
        corrected_reads[expected_result.columns], expected_result
    )


@pytest.mark.basecalls
def test_peaks_to_bases(experiment_c_A1_102_aligned, experiment_c_A1_102_cells):
    with dask.config.set({"dataframe.convert-string": False}):
        image = experiment_c_A1_102_aligned

        image = image.isel(z=0, c=np.delete(np.arange(image.sizes["c"]), 0))
        loged = transform_log(image)
        std_arr = std(loged)
        peaks = find_peaks(std_arr)
        maxed = max_filter(loged)
        bases_array = peaks_to_bases(
            maxed=maxed,
            peaks=peaks[peaks["peak"] >= 50],
            labels=experiment_c_A1_102_cells.squeeze().values,
        )
        bases_array = bases_array.sortby(["y", "x"])

        maxed2 = maxed.copy()
        maxed2.data = da.from_array(maxed2.data, chunks=(-1, -1, -1, 255, 255))
        bases_array_dask = peaks_to_bases(
            maxed=maxed2,
            peaks=peaks[peaks["peak"] >= 50],
            labels=experiment_c_A1_102_cells.squeeze().values,
        )
        df_reads = decode_max(bases_array).sort_values(["y", "x"])
        df_reads_dask = decode_max(bases_array_dask).sort_values(["y", "x"])

        bases_array_dask = bases_array_dask.sortby(["y", "x"])
        np.testing.assert_array_equal(bases_array_dask.values, bases_array.values)
        for c in ["y", "x", "sigma", "peak", "label"]:
            np.testing.assert_array_equal(
                bases_array_dask[c].values, bases_array[c].values
            )

        df_reads_dask = (
            df_reads_dask.compute().reset_index(drop=True).drop("read", axis=1)
        )
        df_reads = df_reads.reset_index(drop=True).drop("read", axis=1)

        pd.testing.assert_frame_equal(
            df_reads_dask,
            df_reads,
            check_dtype=False,
        )


@pytest.mark.basecalls
def test_peak_thresholds_from_reads():
    df = pd.read_csv("scallops/tests/data/process_fig4/10X_A1_Tile-102.reads.csv")
    df["Q_mean"] = df["Q_min"].apply(lambda p: -10 * np.log10(p + 1e-6))
    df_cutoff = peak_thresholds_from_reads(df)
    cutoff = df_cutoff.iloc[0]["threshold"]
    assert abs(cutoff - 50) < 1


@pytest.mark.basecalls
def test_peak_thresholds_from_reads_empty():
    df = pd.DataFrame({"peak": np.zeros(0), "Q_mean": np.zeros(0)})
    df_cutoff = peak_thresholds_from_reads(df)
    assert len(df_cutoff) > 0
    assert np.isnan(df_cutoff["threshold"]).all()
    assert (df_cutoff[["tp", "fp", "tn", "fn"]] == 0).all().all()


def _run_pipeline(image, cells):
    with dask.config.set({"dataframe.convert-string": False}):
        image = image.isel(c=np.delete(np.arange(image.sizes["c"]), 0))
        loged = transform_log(image)
        std_arr = std(loged)
        peaks = find_peaks(std_arr)
        peaks = peaks.sort_values(["y", "x"]).reset_index(
            drop=True
        )  # match dask and non-dask
        maxed = max_filter(loged)

        bases_array = peaks_to_bases(
            maxed=maxed,
            peaks=peaks,
            labels=cells,
        )
        df_cutoff = peak_thresholds_from_bases(
            bases_array=bases_array, min_unique_bases=0
        )
        cutoff = df_cutoff.iloc[0]["threshold"]
        assert np.abs(cutoff - 85) < 1, f"cutoff is {cutoff}"
        bases_array = bases_array.query(dict(read=f"peak>{cutoff}"))

        w = channel_crosstalk_matrix(bases_array)

        corrected_bases_array = apply_channel_crosstalk_matrix(bases_array, w)
        df_reads = decode_max(corrected_bases_array)
        df_cells = assign_barcodes_to_labels(df_reads)

        return {
            "w": w,
            "loged": loged,
            "maxed": maxed,
            "std_arr": std_arr,
            "peaks": peaks,
            "bases_array": bases_array,
            "corrected_bases_array": corrected_bases_array,
            "df_reads": df_reads,
            "df_cells": df_cells,
        }


@pytest.mark.basecalls
def test_sbs_dask(experiment_c_A1_102_aligned, experiment_c_A1_102_cells):
    image = experiment_c_A1_102_aligned
    image = image.isel(z=0)
    image1 = image.copy()
    image1.data = da.from_array(image1.data, chunks=(-1, -1, 256, 256))
    image2 = image.copy()

    np_results = _run_pipeline(image2, experiment_c_A1_102_cells.data.squeeze())
    dask_results = _run_pipeline(image1, experiment_c_A1_102_cells.data.squeeze())

    for k in ["loged", "maxed", "std_arr", "bases_array", "corrected_bases_array"]:
        np.testing.assert_array_equal(
            dask_results[k], np_results[k], err_msg=f"{k} not equal"
        )

    dask_w = dask_results["w"].compute()
    w_delta = np.max(np.abs(dask_w - np_results["w"]))

    assert w_delta < 4.45e-16
    np.testing.assert_array_almost_equal(dask_w, np_results["w"], err_msg="w not equal")
    for k in ["peaks", "df_reads", "df_cells"]:
        dask_df = dask_results[k]
        np_df = np_results[k]

        if isinstance(dask_df, dd.DataFrame):
            dask_df = dask_df.compute()
        if "label" in np_df.columns:
            np_df = np_df.sort_values("label")
            dask_df = dask_df.sort_values("label")
        if "read" in np_df.columns:
            np_df = np_df.drop("read", axis=1)
            dask_df = dask_df.drop("read", axis=1)
        dask_df = dask_df.reset_index(drop=True)
        np_df = np_df.reset_index(drop=True)

        pd.testing.assert_frame_equal(np_df, dask_df, check_dtype=False)


@pytest.mark.basecalls
def test_annotated(experiment_c_A1_102_cells):
    path = __root__.joinpath("data", "annotated", "10X_A1_Tile-102.annotated.npz")
    aln_path = __root__.joinpath("data", "process_fig4")
    cells = experiment_c_A1_102_cells.values.squeeze()
    maxed = read_image(f"{aln_path.joinpath('10X_A1_Tile-102.maxed.tif')}").isel(z=0)
    peaks = read_image(f"{aln_path.joinpath('10X_A1_Tile-102.peaks.tif')}").squeeze()
    peaks = peaks.to_dataframe(name="peak").reset_index()
    peaks = peaks[peaks["peak"] >= 50]
    bases_array = peaks_to_bases(
        maxed=maxed, peaks=peaks, labels=cells, labels_only=False
    ).sortby(["y", "x"])
    bases_array = bases_array.assign_coords(c=["G", "T", "A", "C"])
    bases_array = bases_array.sel(c=["A", "C", "G", "T"])
    bases_array = bases_array.assign_coords(t=np.arange(1, 1 + len(bases_array.t)))
    w = channel_crosstalk_matrix(bases_array.where(bases_array.label > 0, drop=True))
    corrected_bases_array = apply_channel_crosstalk_matrix(bases_array, w)
    corrected_bases_array = corrected_bases_array.astype(int)
    df_reads = decode_max(corrected_bases_array)
    outputs = np.load(f"{path}")
    for width in range(4):
        annotated = annotated_spots(
            df_reads,
            shape=maxed.shape,
            bases_order=["A", "C", "G", "T"],
            expand_width=width,
        )
        expected = outputs[f"width_{width}"]
        np.testing.assert_array_equal(expected, annotated)


def _assign_barcodes_to_labels_reference(
    df_reads: pd.DataFrame, sort_by=("Q_mean", "peak")
) -> pd.DataFrame:
    """Original per-label groupby/apply implementation of assign_barcodes_to_labels."""
    sort_by = list(sort_by)
    agg_columns = [c for c in ["Q_mean", "Q_min", "peak"] if c in df_reads.columns]
    top2_columns = [c for c in ["Q"] if c in df_reads.columns]

    def _agg(df):
        df_perfect_match = (
            df[df["mismatches"].isna()] if "mismatches" in df.columns else df
        )
        mismatch = False
        if len(df_perfect_match) > 0:
            df = df_perfect_match
        else:
            df = df[~df["mismatches"].isna()]
            mismatch = True
        barcode_groupby = df.groupby("barcode", sort=False, dropna=False)
        barcode_sizes = barcode_groupby.size()
        top2 = barcode_groupby.agg("sum").nlargest(n=2, columns=sort_by)
        barcode_0 = top2.iloc[0].name
        barcode_1 = top2.iloc[1].name if len(top2) > 1 else ""
        d = {
            "label": [df.iloc[0]["label"]],
            "barcode_0": [barcode_0],
            "barcode_1": [barcode_1],
            "mismatch": [mismatch],
            "barcode_count": [len(df)],
            "barcode_count_0": [barcode_sizes.loc[barcode_0]],
        }
        for c in agg_columns + top2_columns:
            d[f"barcode_{c}_0"] = [top2.iloc[0][c]]
        d["barcode_count_1"] = [barcode_sizes.loc[barcode_1] if len(top2) > 1 else 0]
        default_values = {"Q_min": 0.0, "Q_mean": 0.0, "peak": np.nan}
        for c in agg_columns + top2_columns:
            d[f"barcode_{c}_1"] = [
                top2.iloc[1][c] if len(top2) > 1 else default_values.get(c)
            ]
        return pd.DataFrame.from_dict(d)

    return df_reads.groupby("label", group_keys=False, sort=False, dropna=False)[
        list(df_reads.columns)
    ].apply(_agg)


def _synthetic_reads(n_reads=20_000, n_labels=2_000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "label": rng.integers(1, n_labels, n_reads),
            "barcode": rng.choice(["AAAA", "CCCC", "GGGG", "TTTT", "ACGT"], n_reads),
            # coarse values to create many exact ties between barcodes
            "Q_mean": rng.choice([10.0, 20.0, 30.0], n_reads),
            "Q_min": rng.random(n_reads),
            "peak": rng.choice([1.0, 2.0], n_reads),
        }
    )
    df["Q"] = list(rng.random((n_reads, 4)))
    df["mismatches"] = np.where(rng.random(n_reads) < 0.5, np.nan, 1.0)
    df.loc[df["label"] % 7 == 0, "mismatches"] = 1.0  # labels with only mismatches
    return df


def _assert_labels_equal(expected: pd.DataFrame, actual: pd.DataFrame):
    expected = expected.reset_index(drop=True)
    actual = actual.reset_index(drop=True)
    assert list(expected.columns) == list(actual.columns)
    for c in ["barcode_Q_0", "barcode_Q_1"]:
        for x, y in zip(expected[c], actual[c]):
            assert (x is None) == (y is None)
            if x is not None:
                np.testing.assert_allclose(x, y, rtol=1e-14, atol=0)
    q_columns = ["barcode_Q_0", "barcode_Q_1"]
    # sums can differ in the last bit depending on how pandas groups the summation
    pd.testing.assert_frame_equal(
        expected.drop(columns=q_columns),
        actual.drop(columns=q_columns),
        check_exact=False,
        rtol=1e-14,
        atol=0,
        check_dtype=False,
    )


@pytest.mark.basecalls
@pytest.mark.parametrize("mismatches", [True, False])
def test_assign_barcodes_to_labels_matches_reference(mismatches):
    df = _synthetic_reads()
    if not mismatches:
        df = df.drop(columns="mismatches")
    _assert_labels_equal(
        _assign_barcodes_to_labels_reference(df), assign_barcodes_to_labels(df)
    )


@pytest.mark.basecalls
def test_assign_barcodes_to_labels_dask():
    df = _synthetic_reads()
    expected = assign_barcodes_to_labels(df).sort_values("label")
    with dask.config.set(
        {"dataframe.convert-string": False, "dataframe.shuffle.method": "tasks"}
    ):
        actual = (
            assign_barcodes_to_labels(dd.from_pandas(df, npartitions=8))
            .compute()
            .sort_values("label")
        )
    _assert_labels_equal(expected, actual)


@pytest.mark.basecalls
@pytest.mark.parametrize("by_t", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_channel_crosstalk_matrix_median_dask_exact(by_t, dtype):
    import xarray as xr

    from scallops.reads import _crosstalk_median_ratio

    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 100.0, (5_001, 3, 4)).astype(dtype)
    x[::3, :, 2] *= 3  # uneven number of rows per max channel
    if by_t:
        expected = np.stack(
            [
                np.linalg.inv(_crosstalk_median_ratio(x[:, t]).astype(np.float64))
                for t in range(x.shape[1])
            ]
        )
    else:
        expected = np.linalg.inv(
            _crosstalk_median_ratio(x.reshape(-1, 4)).astype(np.float64)
        )
    for read_chunks in [-1, 997, 2000]:
        a = xr.DataArray(
            da.from_array(x, chunks=(read_chunks, -1, -1)), dims=("read", "t", "c")
        )
        w = channel_crosstalk_matrix(a, by_t=by_t)
        assert isinstance(w, da.Array)
        # exact global median regardless of chunking
        np.testing.assert_array_equal(w.compute(), expected)


@pytest.mark.basecalls
@pytest.mark.parametrize("use_dask", [False, True])
def test_filter_low_complexity_reads(use_dask):
    import xarray as xr

    from scallops.reads import filter_low_complexity_reads, n_unique_bases

    # brightest channel per cycle for each read
    calls = np.array(
        [
            [0, 0, 0],  # 1 unique base
            [1, 1, 2],  # 2
            [3, 2, 1],  # 3
            [2, 2, 2],  # 1
            [0, 3, 0],  # 2
        ]
    )
    x = np.ones(calls.shape + (4,), dtype=np.float32)
    np.put_along_axis(x, calls[..., np.newaxis], 10.0, axis=2)
    data = da.from_array(x, chunks=(2, 1, -1)) if use_dask else x
    a = xr.DataArray(
        data,
        dims=("read", "t", "c"),
        coords={"read": np.arange(len(x)), "peak": ("read", np.arange(len(x)) * 1.0)},
    )
    np.testing.assert_array_equal(n_unique_bases(a), [1, 2, 3, 1, 2])
    np.testing.assert_array_equal(
        filter_low_complexity_reads(a, 2)["read"].values, [1, 2, 4]
    )
    np.testing.assert_array_equal(
        filter_low_complexity_reads(a, 3)["peak"].values, [2.0]
    )
    assert filter_low_complexity_reads(a, 1) is a


def _label_columns_meta() -> dict[str, str]:
    # barcode columns keep the input string dtype (object, or str with pandas >= 3)
    barcode_dtype = str(pd.Series(["A"]).dtype)
    return {
        "label": "int32",
        "barcode_0": barcode_dtype,
        "barcode_1": barcode_dtype,
        "mismatch": "bool",
        "barcode_count": "int64",
        "barcode_count_0": "int64",
        "barcode_Q_mean_0": "float64",
        "barcode_Q_min_0": "float64",
        "barcode_peak_0": "float64",
        "barcode_Q_0": "object",
        "barcode_count_1": "int64",
        "barcode_Q_mean_1": "float64",
        "barcode_Q_min_1": "float64",
        "barcode_peak_1": "float64",
        "barcode_Q_1": "object",
    }


def _to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Use float32 Q columns, as produced by decode_max from float32 intensities."""
    df = df.copy()
    df["label"] = df["label"].astype(np.int32)
    df["Q_mean"] = df["Q_mean"].astype(np.float32)
    df["Q_min"] = df["Q_min"].astype(np.float32)
    df["Q"] = [np.asarray(q, dtype=np.float32) for q in df["Q"]]
    return df


@pytest.mark.basecalls
def test_assign_barcodes_to_labels_float32():
    df32 = _to_float32(_synthetic_reads())
    # same values in float64 (float32 -> float64 is exact)
    df64 = df32.astype({"Q_mean": np.float64, "Q_min": np.float64})
    df64["Q"] = [q.astype(np.float64) for q in df64["Q"]]

    result32 = assign_barcodes_to_labels(df32)
    # sums are computed in float64, so the result doesn't depend on the input dtype,
    # which can change from float32 to float64 when dask shuffles reads
    _assert_labels_equal(assign_barcodes_to_labels(df64), result32)
    _assert_labels_equal(_assign_barcodes_to_labels_reference(df64), result32)
    assert result32.dtypes.astype(str).to_dict() == _label_columns_meta()
    assert all(q.dtype == np.float64 for q in result32["barcode_Q_0"])


@pytest.mark.basecalls
def test_assign_barcodes_to_labels_empty():
    df = _to_float32(_synthetic_reads()).iloc[:0]
    result = assign_barcodes_to_labels(df)
    assert len(result) == 0
    assert result.dtypes.astype(str).to_dict() == _label_columns_meta()


@pytest.mark.basecalls
def test_assign_barcodes_to_labels_single_barcode_per_label():
    # labels 3 and 1 each have one barcode; label order is first appearance
    df = pd.DataFrame(
        {
            "label": np.array([3, 1, 3], dtype=np.int32),
            "barcode": ["ACGT", "TTTT", "ACGT"],
            "Q_mean": np.array([30.0, 20.0, 40.0], dtype=np.float32),
            "Q_min": np.array([10.0, 5.0, 15.0], dtype=np.float32),
            "peak": [100.0, 50.0, 200.0],
            "Q": [
                np.float32([30, 30]),
                np.float32([20, 20]),
                np.float32([40, 40]),
            ],
        }
    )
    result = assign_barcodes_to_labels(df)
    assert result.dtypes.astype(str).to_dict() == _label_columns_meta()
    assert result["label"].tolist() == [3, 1]
    assert result["barcode_0"].tolist() == ["ACGT", "TTTT"]
    assert result["barcode_count"].tolist() == [2, 1]
    assert result["barcode_count_0"].tolist() == [2, 1]
    assert result["barcode_Q_mean_0"].tolist() == [70.0, 20.0]
    assert result["barcode_peak_0"].tolist() == [300.0, 50.0]
    np.testing.assert_array_equal(result["barcode_Q_0"].iloc[0], [70.0, 70.0])
    # no second barcode
    assert result["barcode_1"].tolist() == ["", ""]
    assert result["barcode_count_1"].tolist() == [0, 0]
    assert result["barcode_Q_mean_1"].tolist() == [0.0, 0.0]
    assert result["barcode_Q_min_1"].tolist() == [0.0, 0.0]
    assert result["barcode_peak_1"].isna().all()
    assert result["barcode_Q_1"].tolist() == [None, None]
    assert not result["mismatch"].any()


@pytest.mark.basecalls
def test_assign_barcodes_to_labels_distributed(client):
    # the distributed (p2p) shuffle converts partitions to Arrow and back, which can
    # change dtypes and leave some partitions empty
    df = _to_float32(_synthetic_reads(n_reads=5_000, n_labels=20))
    expected = assign_barcodes_to_labels(df).sort_values("label")
    with dask.config.set(
        {"dataframe.convert-string": False, "dataframe.shuffle.method": "p2p"}
    ):
        result = assign_barcodes_to_labels(dd.from_pandas(df, npartitions=8))
        partitions = client.compute(result.to_delayed(), sync=True)
    assert any(len(p) == 0 for p in partitions)  # only 20 labels in 8 partitions
    for p in partitions:
        assert p.dtypes.astype(str).to_dict() == _label_columns_meta()
    actual = pd.concat(partitions).sort_values("label")
    _assert_labels_equal(expected, actual)


@pytest.mark.basecalls
def test_auto_read_chunk_size():
    from scallops.cli.pooled_if_sbs_main import _read_chunk_size
    from scallops.reads import _auto_read_chunk_size

    # ~3 chunks per thread
    assert _auto_read_chunk_size(9_000_000, 30) == 100_000
    # clamped to the minimum: small wells stay in one chunk
    assert _auto_read_chunk_size(5_000, 8) == 50_000
    # clamped to the maximum
    assert _auto_read_chunk_size(100_000_000, 4) == 500_000
    assert _auto_read_chunk_size(1_000_000, 0) == 333_334  # at least one thread
    assert _read_chunk_size("auto") == "auto"
    assert _read_chunk_size("-1") == -1
    with pytest.raises(argparse.ArgumentTypeError):
        _read_chunk_size("big")


@pytest.mark.basecalls
def test_channel_crosstalk_matrix_by_t_single_cycle():
    import xarray as xr

    x = np.random.default_rng(0).gamma(2.0, 100.0, (1_000, 1, 4))
    a = xr.DataArray(da.from_array(x, chunks=(300, -1, -1)), dims=("read", "t", "c"))
    w_by_t = channel_crosstalk_matrix(a, by_t=True).compute()
    assert w_by_t.shape == (4, 4)
    np.testing.assert_array_equal(w_by_t, channel_crosstalk_matrix(a).compute())


@pytest.mark.basecalls
def test_barcode_n_unique_bases():
    from scallops.reads import barcode_n_unique_bases, n_unique_bases

    barcodes = ["AAA", "ACA", "GTC", "TTG"]
    np.testing.assert_array_equal(barcode_n_unique_bases(barcodes), [1, 2, 3, 2])
    np.testing.assert_array_equal(
        barcode_n_unique_bases(pd.Series(["AC", "GG", "TA"])), [2, 1, 2]
    )
    assert len(barcode_n_unique_bases([])) == 0
    # same counts as n_unique_bases on intensities whose maximum gives the barcodes
    import xarray as xr

    bases = "GTAC"
    x = np.zeros((len(barcodes), 3, 4))
    for i, barcode in enumerate(barcodes):
        for t, base in enumerate(barcode):
            x[i, t, bases.index(base)] = 1
    np.testing.assert_array_equal(
        n_unique_bases(xr.DataArray(x, dims=("read", "t", "c"))),
        barcode_n_unique_bases(barcodes),
    )


@pytest.mark.basecalls
def test_dask_total_threads_cpu_count(monkeypatch):
    from scallops.utils import _dask_total_threads

    monkeypatch.setenv("SCALLOPS_CPU_COUNT", "3")
    try:
        from distributed import get_client

        get_client()
        pytest.skip("distributed client active")
    except ValueError:
        pass
    assert _dask_total_threads() == 3
