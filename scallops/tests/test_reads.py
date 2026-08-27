from pathlib import Path

import dask.array as da
import dask.config
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scallops.io import read_image
from scallops.reads import (
    annotated_spots,
    apply_channel_crosstalk_matrix,
    assign_barcodes_to_labels,
    channel_crosstalk_matrix,
    correct_mismatches,
    decode_max,
    decode_se,
    peaks_to_bases,
    read_statistics,
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
def test_dark_bases(aligned_A1_102, barcodes_A1_102):
    """3-color simulation: G/T/A have dedicated channels; C detected by absence of signal.

    Uses aligned ExperimentC image (z already selected in fixture), channels G/T/A only.
    """
    image = aligned_A1_102.isel(c=[1, 2, 3])  # z=0 already selected in fixture
    loged = transform_log(image)
    bases_array = peaks_to_bases(
        maxed=max_filter(loged),
        peaks=find_peaks(std(loged))[lambda p: p["peak"] >= 50],
        bases=["G", "T", "A"],
    )
    df_reads = decode_se(bases_array, barcodes=barcodes_A1_102, dark_bases=["C"])
    df_reads["label"] = 0
    stats = read_statistics(df_reads)
    assert stats["mapping_rate"] > 0.40  # ~0.47 on raw (non-xtalk-corrected) data


@pytest.mark.basecalls
def test_dark_bases_two_color(aligned_A1_102, barcodes_A1_102):
    """2-color Illumina simulation: combine 4 SBS channels into red (A+C) and green (A+T).

    Encoding:  G → dark | T → green only | A → red+green | C → red only
    Uses aligned ExperimentC image (z already selected in fixture).
    """
    image = aligned_A1_102.isel(c=[1, 2, 3, 4])  # z=0 already selected
    loged = transform_log(image)
    bases_array = peaks_to_bases(
        maxed=max_filter(loged),
        peaks=find_peaks(std(loged))[lambda p: p["peak"] >= 50],
        bases=["G", "T", "A", "C"],
    )
    w = channel_crosstalk_matrix(bases_array)
    bases_array_c = apply_channel_crosstalk_matrix(bases_array, w)

    spots = bases_array_c.values  # (read, t, 4): G=0, T=1, A=2, C=3
    red = np.maximum(spots[..., 2], spots[..., 3])  # max(A, C)
    green = np.maximum(spots[..., 2], spots[..., 1])  # max(A, T)
    two_color_array = xr.DataArray(
        np.stack([red, green], axis=-1),
        dims=bases_array.dims,
        coords={k: v for k, v in bases_array.coords.items() if k != "c"},
    ).assign_coords(c=["red", "green"])

    E = np.array(
        [[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float
    )  # G, T, A, C × red, green
    df_reads = decode_se(
        two_color_array,
        barcodes=barcodes_A1_102,
        encoding=E,
        base_labels=["G", "T", "A", "C"],
    )
    df_reads["label"] = 0
    stats = read_statistics(df_reads)
    assert stats["mapping_rate"] > 0.40  # ~0.48 on xtalk-corrected 2-col data


@pytest.mark.basecalls
def test_decoders_4ch(aligned_A1_102, barcodes_A1_102, dask_A1_102_cells):
    """SE and polar decoders on 4-channel ExperimentC, shared preprocessing.

    Notebook polar_basecalling.ipynb values:
      decode_se    map=83.69%   cells/2,612 = 74.78%
      decode_polar map=80.59%   cells/2,612 = 74.04%
    """
    from scallops.reads import decode_polar

    cells = dask_A1_102_cells.squeeze()
    n_cells = int(cells.max())

    image = aligned_A1_102.isel(c=[1, 2, 3, 4])
    loged = transform_log(image)
    bases_array = peaks_to_bases(
        maxed=max_filter(loged),
        peaks=find_peaks(std(loged))[lambda p: p["peak"] >= 50],
        bases=["G", "T", "A", "C"],
        labels=cells,
    )
    thr_x = peak_thresholds_from_bases(bases_array=bases_array).iloc[0]["threshold"]
    w = channel_crosstalk_matrix(bases_array.where(bases_array.peak > thr_x, drop=True))
    corrected = apply_channel_crosstalk_matrix(bases_array, w)
    df_sm = decode_max(corrected, barcodes=barcodes_A1_102)
    thr_r = peak_thresholds_from_reads(df_sm.query("barcode_match")).iloc[0][
        "threshold"
    ]

    def s(df):
        return read_statistics(df.query(f"peak>{thr_r}"))

    se_stats = s(decode_se(corrected, barcodes=barcodes_A1_102))
    pol_stats = s(decode_polar(corrected, barcodes=barcodes_A1_102))

    assert se_stats["mapping_rate"] > 0.80  # notebook: ~83.69%
    assert se_stats["labels_with_mapped_reads"] / n_cells > 0.71
    assert pol_stats["mapping_rate"] > 0.77  # notebook: ~80.59%
    assert pol_stats["labels_with_mapped_reads"] / n_cells > 0.70


@pytest.mark.basecalls
def test_decoders_3ch_nis_seq(nis_seq_fixtures):
    """SE and polar decoders on NIS-seq 3-channel (G dark), shared xtalk-corrected input.

    Notebook polar_basecalling.ipynb values:
      decode_se    map=44.90%   nuclei/1,001 = 84.58%
      decode_polar map=43.84%   nuclei/1,001 = 85.39%
    """
    from scallops.reads import decode_polar

    f = nis_seq_fixtures

    def s(df):
        return read_statistics(df.query(f"peak>{f.thr_r3}"))

    se_stats = s(decode_se(f.cor3, barcodes=f.df_bcn, dark_bases=["G"]))
    pol_stats = s(decode_polar(f.cor3, barcodes=f.df_bcn, dark_bases=["G"], w_cor=f.w3))

    assert se_stats["mapping_rate"] > 0.43  # notebook: ~45.1%
    assert se_stats["labels_with_mapped_reads"] / f.n_nuc > 0.82  # notebook: ~84.2%
    assert pol_stats["mapping_rate"] > 0.42  # notebook: ~44.0%
    assert pol_stats["labels_with_mapped_reads"] / f.n_nuc > 0.83  # notebook: ~85.2%


@pytest.mark.basecalls
def test_decoders_2col_nis_seq(nis_seq_fixtures):
    """SE and polar decoders on synthesised 2-colour NIS-seq (G dark, non-orthogonal).

    Channels: ch0=max(A,C), ch1=max(A,T).  Notebook polar_basecalling.ipynb values:
      decode_se    map=17.66%   nuclei/1,001 = 71.07%
      decode_polar map=29.46%   nuclei/1,001 = 84.08%   (recommended — atan2 is amplitude-independent)
    Polar outperforms SE because θ=atan2(ch1,ch0) resolves the A/C score ambiguity.
    """
    from scallops.reads import decode_polar

    f = nis_seq_fixtures
    sp3 = f.cor3.values  # (n, T, 3): A=0, T=1, C=2
    ch0 = np.maximum(sp3[..., 0], sp3[..., 2])  # max(A, C)
    ch1 = np.maximum(sp3[..., 0], sp3[..., 1])  # max(A, T)
    spots_2col = xr.DataArray(
        np.stack([ch0, ch1], axis=-1),
        dims=["read", "t", "c"],
        coords={
            "read": f.cor3.read.values,
            "t": f.cor3.t.values,
            "c": ["ch0", "ch1"],
            "peak": ("read", f.cor3.peak.values),
            "label": ("read", f.cor3.label.values),
            "y": ("read", f.cor3.y.values),
            "x": ("read", f.cor3.x.values),
        },
    )
    E2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)

    def s(df):
        return read_statistics(
            df.query(f"peak>{f.thr_r2}")
        )  # 2-col uses its own threshold

    se_stats = s(
        decode_se(
            spots_2col, barcodes=f.df_bcn, encoding=E2, base_labels=["G", "T", "A", "C"]
        )
    )
    pol_stats = s(
        decode_polar(
            spots_2col,
            barcodes=f.df_bcn,
            encoding=E2,
            base_labels=["G", "T", "A", "C"],
            w_cor=f.w3,
        )
    )

    assert se_stats["mapping_rate"] > 0.15  # notebook: ~17.7%
    assert se_stats["labels_with_mapped_reads"] / f.n_nuc > 0.68  # notebook: ~70.9%
    assert pol_stats["mapping_rate"] > 0.27  # notebook: ~29.5%
    assert pol_stats["labels_with_mapped_reads"] / f.n_nuc > 0.81  # notebook: ~83.9%


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
