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
from scallops.segmentation.watershed import (
    segment_cells_watershed,
    segment_nuclei_watershed,
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
def test_peaks_to_bases(array_A1_102_aln, array_A1_102_cells):
    with dask.config.set({"dataframe.convert-string": False}):
        image = array_A1_102_aln.transpose(*("z", "c", "t", "y", "x")).rename(
            {"z": "t", "t": "z"}
        )  # ops swaps z and t in saved tif

        image = image.isel(z=0, c=np.delete(np.arange(image.sizes["c"]), 0))
        loged = transform_log(image)
        std_arr = std(loged)
        peaks = find_peaks(std_arr)
        maxed = max_filter(loged)
        bases_array = peaks_to_bases(
            maxed=maxed,
            peaks=peaks[peaks["peak"] >= 50],
            labels=array_A1_102_cells.squeeze().values,
        )
        bases_array = bases_array.sortby(["y", "x"])

        maxed2 = maxed.copy()
        maxed2.data = da.from_array(maxed2.data, chunks=(-1, -1, -1, 255, 255))
        bases_array_dask = peaks_to_bases(
            maxed=maxed2,
            peaks=peaks[peaks["peak"] >= 50],
            labels=array_A1_102_cells.squeeze().values,
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
    assert abs(cutoff - 53) < 1


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
            bases_array=bases_array, remove_zero_entropy_barcodes=False
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
def test_sbs_dask(array_A1_102_aln):
    image = array_A1_102_aln.transpose(*("z", "c", "t", "y", "x")).rename(
        {"z": "t", "t": "z"}
    )  # ops swaps z and t in saved tif

    image = image.isel(z=0)
    image1 = image.copy()
    image1.data = da.from_array(image1.data, chunks=(-1, -1, 256, 256))
    image2 = image.copy()
    nuclei = segment_nuclei_watershed(image=image2, nuclei_channel=0)
    cells, _ = segment_cells_watershed(
        image2, nuclei, threshold=600, at_least_nuclei=False, watershed_method="binary"
    )
    np_results = _run_pipeline(image2, cells)
    dask_results = _run_pipeline(image1, cells)

    for k in ["loged", "maxed", "std_arr", "bases_array", "corrected_bases_array"]:
        np.testing.assert_array_equal(
            dask_results[k], np_results[k], err_msg=f"{k} not equal"
        )

    dask_w = dask_results["w"].compute()
    w_delta = np.max(np.abs(dask_w - np_results["w"]))
    assert w_delta < 3e-16
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
def test_annotated(array_A1_102_cells):
    path = __root__.joinpath("data", "annotated", "10X_A1_Tile-102.annotated.npz")
    aln_path = __root__.joinpath("data", "process_fig4")
    cells = array_A1_102_cells.values.squeeze()
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
