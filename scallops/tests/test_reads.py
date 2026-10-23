from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import dask.config
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scallops.io import read_barcodes, read_image
from scallops.reads import (
    annotated_spots,
    apply_channel_crosstalk_matrix,
    assign_barcodes_to_labels,
    channel_crosstalk_matrix,
    correct_mismatches,
    decode_em,
    decode_max,
    decode_polar,
    decode_se,
    fit_base_mixture,
    peaks_to_bases,
    read_statistics,
)
from scallops.registration.crosscorrelation import align_image
from scallops.spots import (
    find_peaks,
    max_filter,
    peak_thresholds_from_bases,
    peak_thresholds_from_reads,
    std,
    transform_log,
)

__root__ = Path("scallops/tests")  # Path(__file__).resolve().parent


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
def test_decoders_4ch(experiment_c_A1_102_aligned, experiment_c_A1_102_cells):
    """SE, max, and polar decoders on 4-channel ExperimentC, shared preprocessing."""

    cells = experiment_c_A1_102_cells.squeeze()
    n_cells = int(cells.max())

    image = experiment_c_A1_102_aligned.isel(z=0, c=[1, 2, 3, 4])
    barcodes_A1_102 = read_barcodes(
        str(__root__.joinpath("data", "experimentC", "barcodes.csv")),
        image.t.values - 1,
    )
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
    df_max = decode_max(corrected, barcodes=barcodes_A1_102)
    thr_r = peak_thresholds_from_reads(df_max.query("barcode_match")).iloc[0][
        "threshold"
    ]

    def s(df):
        return read_statistics(df.query(f"peak>{thr_r}"))

    se_stats = s(decode_se(corrected, barcodes=barcodes_A1_102))
    pol_stats = s(decode_polar(corrected, barcodes=barcodes_A1_102))
    max_stats = s(df_max)
    assert se_stats["mapping_rate"] > 0.80  # 0.84
    assert se_stats["labels_with_mapped_reads"] / n_cells > 0.80  # 0.85
    assert pol_stats["mapping_rate"] > 0.77  # 0.81
    assert pol_stats["labels_with_mapped_reads"] / n_cells > 0.80  # 0.85
    assert max_stats["mapping_rate"] > 0.77  # 0.81
    assert max_stats["labels_with_mapped_reads"] / n_cells > 0.80  # 0.85


@pytest.fixture(scope="module")
def nis_seq_fixtures(nis_seq_experiment, nis_seq_nuclear_mask, nis_seq_barcodes):
    """Fully preprocessed NIS-seq fixtures: aligned, xtalk-corrected spots,
    nuclear labels, whitelist and thresholds.

    Attributes
    ----------
    iss : xr.DataArray
        Raw aligned NIS-seq image (well C10, tile 0040).
    nuclei : np.ndarray
        2D CellPose nuclear label mask.
    df_bcn : pd.DataFrame
        Whitelist with column 'barcode' (14-mer RC).
    ba3 : xr.DataArray
        peaks_to_bases output (3 SBS channels, labels=nuclei).
    cor3 : xr.DataArray
        Xtalk-corrected peaks_to_bases output.
    w3 : np.ndarray
        Xtalk correction matrix (3×3).
    thr_r3 : float
        Secondary peak threshold for 3-ch.
    n_nuc : int
        Number of nuclei in the mask.
    """

    nuclei = nis_seq_nuclear_mask
    df_bcn = nis_seq_barcodes
    n_nuc = int(nuclei.max())

    # Align across cycles (no within-cycle channel alignment for NIS-seq)
    iss = nis_seq_experiment.images["C10-0040"].squeeze()
    iss = align_image(
        iss,
        align_within_time_channels=None,
        align_between_time_channel=0,
        filter_percentiles=[0, 90],
    )

    # Channels 1,2,3 → ch03(A), ch04(T), ch06(C)
    loged3 = transform_log(iss.isel(c=[1, 2, 3]))
    ba3 = peaks_to_bases(
        maxed=max_filter(loged3, width=5),
        peaks=find_peaks(std(loged3)),
        labels=nuclei,
        bases=["A", "T", "C"],
    )
    thr_x3 = peak_thresholds_from_bases(ba3).iloc[0]["threshold"]
    w3 = channel_crosstalk_matrix(ba3.where(ba3.peak > thr_x3, drop=True))
    cor3 = apply_channel_crosstalk_matrix(ba3, w3)

    df_tmp = decode_max(cor3, barcodes=df_bcn)
    thr_r3 = peak_thresholds_from_reads(df_tmp.query("barcode_match")).iloc[0][
        "threshold"
    ]

    # the synthesized 2-color spots are the 3-color spots, so they share its threshold
    thr_r2 = thr_r3

    return SimpleNamespace(
        iss=iss,
        nuclei=nuclei,
        df_bcn=df_bcn,
        ba3=ba3,
        cor3=cor3,
        w3=w3,
        thr_r3=thr_r3,
        thr_r2=thr_r2,
        n_nuc=n_nuc,
    )


@pytest.mark.basecalls
def test_decoders_3ch_nis_seq(nis_seq_fixtures):
    """SE, max, and polar decoders on NIS-seq 3-channel (G dark), shared xtalk-corrected input.

    Notebook base_calling_methods.ipynb values:
      decode_se    map=44.90%   nuclei/1,001 = 84.58%
      decode_polar map=43.84%   nuclei/1,001 = 85.39%
    """

    f = nis_seq_fixtures

    def s(df):
        return read_statistics(df.query(f"peak>{f.thr_r3}"))

    max_stats = s(
        decode_max(
            f.cor3,
            barcodes=f.df_bcn,
            thresholded_bases=[
                dict(base="G", channels=[0, 1, 2], direction="<", threshold=0.2)
            ],
        )
    )
    se_stats = s(decode_se(f.cor3, barcodes=f.df_bcn, dark_bases=["G"]))
    pol_stats = s(decode_polar(f.cor3, barcodes=f.df_bcn, dark_bases=["G"], w_cor=f.w3))

    assert se_stats["mapping_rate"] > 0.43  # 0.46
    assert se_stats["labels_with_mapped_reads"] / f.n_nuc > 0.80  # 0.83
    assert pol_stats["mapping_rate"] > 0.42  # 0.45
    assert pol_stats["labels_with_mapped_reads"] / f.n_nuc > 0.81  # 0.84
    assert max_stats["mapping_rate"] > 0.40  # 0.43
    assert max_stats["labels_with_mapped_reads"] / f.n_nuc > 0.80  # 0.83


@pytest.mark.basecalls
def test_decoders_2col_nis_seq(nis_seq_fixtures):
    """SE and polar decoders on synthesized 2-color NIS-seq (G dark, non-orthogonal).

    Channels: ch0=max(A,C), ch1=max(A,T).  Notebook base_calling_methods.ipynb values:
      decode_se    map=23.9%   nuclei/1,001 = 70.2%
      decode_polar map=39.7%   nuclei/1,001 = 83.1%
    Reads are counted above the 3-color threshold, since the spots are the same.
    Polar outperforms SE because θ=atan2(ch1,ch0) resolves the A/C score ambiguity.
    """

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
        return read_statistics(df.query(f"peak>{f.thr_r2}"))

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
            w_cor_bases=list(f.ba3.c.values),
        )
    )

    assert se_stats["mapping_rate"] > 0.21  # notebook: ~23.9%
    assert se_stats["labels_with_mapped_reads"] / f.n_nuc > 0.68  # notebook: ~70.2%
    assert pol_stats["mapping_rate"] > 0.37  # notebook: ~39.7%
    assert pol_stats["labels_with_mapped_reads"] / f.n_nuc > 0.81  # notebook: ~83.1%


@pytest.mark.basecalls
def test_decode_polar_quality():
    """Q reflects the margin of the polar call: ambiguous calls score lower than clear ones."""
    # orthogonal: clear A, A vs C ambiguous, dark, near dark cutoff, clear C, clear T
    x = np.array(
        [[[100, 0, 0], [100, 95, 0], [0, 0, 0], [21, 0, 0], [0, 100, 0], [0, 0, 100]]],
        dtype=float,
    )
    spots = xr.DataArray(
        x, dims=("read", "t", "c"), coords={"read": [0], "c": list("ACT")}
    )
    df = decode_polar(spots, dark_bases=["G"], r_frac=0.2)
    q = df["Q"].iloc[0]
    assert df["barcode"].iloc[0] == "AAGGCT"
    clear = q[[0, 2, 4, 5]]
    assert clear.min() > 50
    assert q[1] < clear.min() and q[3] < clear.min()
    assert q[1] > 3 and q[3] > 3  # not capped like the old 2-class min

    # non-orthogonal 2-color: clear T/A/C/dark, then a call near the t_lo bin edge
    E = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
    x2 = np.array(
        [[[100, 100], [100, 0], [0, 100], [0, 0], [100, 70], [100, 45]]], dtype=float
    )
    spots2 = xr.DataArray(x2, dims=("read", "t", "c"), coords={"read": [0]})
    df2 = decode_polar(spots2, encoding=E, base_labels=list("TACG"), r_frac=0.2)
    q2 = df2["Q"].iloc[0]
    assert df2["barcode"].iloc[0] == "ACTGAA"
    assert q2[:5].min() > 50
    assert q2[5] < q2[:5].min()


@pytest.mark.basecalls
@pytest.mark.parametrize(
    "encoding, base_labels, x, expected",
    [
        # ch0-only = C, both = A, ch1-only = T, none = G; only differs in row order
        pytest.param(
            [[0, 1], [1, 1], [1, 0], [0, 0]],
            "TACG",
            [[100, 100], [100, 0], [0, 100], [0, 0]],
            "ACTG",
            id="rows-TAC",
        ),
        pytest.param(
            [[1, 1], [0, 1], [1, 0], [0, 0]],
            "ATCG",
            [[100, 100], [100, 0], [0, 100], [0, 0]],
            "ACTG",
            id="rows-ATC",
        ),
        pytest.param(
            [[1, 0], [1, 1], [0, 1], [0, 0]],
            "CATG",
            [[100, 100], [100, 0], [0, 100], [0, 0]],
            "ACTG",
            id="rows-CAT",
        ),
        # only 2 bright bases
        pytest.param(
            [[1, 1], [1, 0], [0, 0]],
            "ACG",
            [[100, 100], [100, 0], [0, 0]],
            "ACG",
            id="2-bright-bases",
        ),
    ],
)
def test_decode_polar_non_orthogonal_row_order(encoding, base_labels, x, expected):
    """Non-orthogonal calls should not depend on the row order of the encoding matrix."""
    spots = xr.DataArray(
        np.array([x], dtype=float), dims=("read", "t", "c"), coords={"read": [0]}
    )
    df = decode_polar(
        spots,
        encoding=np.array(encoding, dtype=float),
        base_labels=list(base_labels),
        r_frac=0.2,
    )
    assert df["barcode"].iloc[0] == expected


def _polar_2col_spots(n_reads=200, n_t=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, (n_reads, n_t, 2))
    return xr.DataArray(x, dims=("read", "t", "c"), coords={"read": np.arange(n_reads)})


@pytest.mark.basecalls
def test_decode_polar_w_cor_channel_order():
    """Thresholds from a 3-channel w_cor depend on channel names, not channel order."""
    E = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    labels = ["G", "T", "A", "C"]
    # forward model over physical channels A, T, C: A bleeds into T, C into A
    fwd = np.array([[1.0, 0.0, 0.3], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])
    w = np.linalg.inv(fwd)
    p = [2, 0, 1]  # same physics with channels stored as C, A, T
    spots = _polar_2col_spots()
    kwargs = dict(encoding=E, base_labels=labels, r_frac=0.2)
    df = decode_polar(spots, w_cor=w, w_cor_bases=["A", "T", "C"], **kwargs)
    df_perm = decode_polar(
        spots, w_cor=w[np.ix_(p, p)], w_cor_bases=["C", "A", "T"], **kwargs
    )
    df_default = decode_polar(spots, **kwargs)
    assert (df["barcode"] == df_perm["barcode"]).all()
    assert (df["barcode"] != df_default["barcode"]).any()  # w_cor is used

    with pytest.raises(ValueError, match="w_cor_bases"):
        decode_polar(spots, w_cor=w, **kwargs)


@pytest.mark.basecalls
def test_decode_polar_w_cor_2_channels():
    """A 2x2 w_cor (channels = dyes) gives distinct bin edges, so A is still called."""
    E = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    # ch1 dye bleeds 20% into ch0: expected angles C = 0, A ~ 35.5, T ~ 78.7 degrees
    w = np.linalg.inv(np.array([[1.0, 0.2], [0.0, 1.0]]))
    x = np.array([[[100, 70], [100, 0], [20, 100], [0, 0]]], dtype=float)
    spots = xr.DataArray(x, dims=("read", "t", "c"), coords={"read": [0]})
    df = decode_polar(spots, encoding=E, base_labels=list("GTAC"), w_cor=w, r_frac=0.2)
    assert df["barcode"].iloc[0] == "ACTG"


@pytest.mark.basecalls
def test_decode_polar_w_cor_2_channels_order():
    """A 2x2 w_cor stored in the opposite channel order is reordered by w_cor_bases."""
    E = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    labels = list("GTAC")  # C fires ch0, T fires ch1
    w = np.linalg.inv(np.array([[1.0, 0.2], [0.0, 1.0]]))
    w_swapped = w[::-1, ::-1]  # same physics, w_cor channels stored as T, C
    spots = _polar_2col_spots()
    kwargs = dict(encoding=E, base_labels=labels, r_frac=0.2)
    df = decode_polar(spots, w_cor=w, **kwargs)
    df_named = decode_polar(spots, w_cor=w, w_cor_bases=["C", "T"], **kwargs)
    df_swapped = decode_polar(spots, w_cor=w_swapped, w_cor_bases=["T", "C"], **kwargs)
    assert (df["barcode"] == df_named["barcode"]).all()
    assert (df["barcode"] == df_swapped["barcode"]).all()

    with pytest.raises(ValueError, match="exactly one data channel"):
        decode_polar(spots, w_cor=w, w_cor_bases=["A", "T"], **kwargs)
    with pytest.raises(ValueError, match="one base per data channel"):
        decode_polar(spots, w_cor=w, w_cor_bases=["T", "T"], **kwargs)


@pytest.mark.basecalls
def test_decode_polar_single_bright_base():
    """A non-orthogonal encoding with one bright base has no angle bin edges."""
    E = np.array([[1, 1], [0, 0]], dtype=float)
    x = np.array([[[100, 100], [0, 0], [100, 90]]], dtype=float)
    spots = xr.DataArray(x, dims=("read", "t", "c"), coords={"read": [0]})
    df = decode_polar(spots, encoding=E, base_labels=["A", "G"], r_frac=0.2)
    assert df["barcode"].iloc[0] == "AGA"
    assert np.all(df["Q"].iloc[0] > 50)


@pytest.mark.basecalls
def test_decode_polar_w_cor_out_of_order_falls_back(caplog):
    """w_cor angles out of encoding order fall back to the encoding angles."""
    E = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    # w_cor channels swapped relative to the data: C's dye points along ch1 and
    # T's along ch0, so C's expected angle is above T's
    w = np.linalg.inv(np.array([[0.2, 1.0], [1.0, 0.2]]))
    spots = _polar_2col_spots()
    kwargs = dict(encoding=E, base_labels=list("GTAC"), r_frac=0.2)
    with caplog.at_level("WARNING", logger="scallops"):
        df = decode_polar(spots, w_cor=w, **kwargs)
    assert "not in encoding order" in caplog.text
    assert (df["barcode"] == decode_polar(spots, **kwargs)["barcode"]).all()


@pytest.mark.basecalls
def test_decode_multiple_dark_bases():
    """Dark bases cannot be told apart, so more than one is rejected."""
    spots = _polar_2col_spots()
    with pytest.raises(ValueError, match="at most one dark base"):
        decode_polar(spots, dark_bases=["G", "N"])
    with pytest.raises(ValueError, match="at most one dark base"):
        decode_se(spots, dark_bases=["G", "N"])


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


def _simulate_spots(E, labels, n=1500, T=12, gains=None, noise=0.05, seed=0):
    """Simulated spots with known calls: per-read amplitude, background, decay."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=float)
    B, C = E.shape
    gains = np.ones(C) if gains is None else np.asarray(gains, dtype=float)
    truth = rng.integers(0, B, (n, T))
    amp = rng.lognormal(0, 0.4, (n, 1, 1))
    signal = E[truth] * gains * amp * 0.97 ** np.arange(T)[np.newaxis, :, np.newaxis]
    x = (
        0.2 * rng.uniform(0.5, 1.5, (n, 1, C))
        + signal
        + rng.normal(0, noise, (n, T, C)) * (1 + 0.5 * signal)
    )
    spots = xr.DataArray(
        x,
        dims=("read", "t", "c"),
        coords={"read": np.arange(n), "peak": ("read", amp.ravel())},
    )
    return spots, np.asarray(labels)[truth]


def _calls(df):
    return np.array([list(b) for b in df["barcode"]])


@pytest.mark.basecalls
def test_decode_em_dark_base():
    """Unsupervised EM recovers a dark base without a whitelist or other decoder."""
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    spots, truth = _simulate_spots(E, list("ATCG"))
    spots = spots.assign_coords(c=list("ATC"))
    calls = _calls(decode_em(spots, dark_bases=["G"], n_restarts=2))
    assert np.mean(calls == truth) > 0.99  # 0.999
    assert np.mean(calls[truth == "G"] == "G") > 0.98  # 0.999
    assert np.mean((calls == truth).all(axis=1)) > 0.95  # 0.996
    # better than the signed-encoding decoder on the same reads (0.967)
    se = _calls(decode_se(spots, dark_bases=["G"]))
    assert np.mean(calls == truth) > np.mean(se == truth)


@pytest.mark.basecalls
def test_decode_em_non_orthogonal():
    """EM handles a 2-color encoding where A fires both channels and G is dark."""
    E = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    labels = list("GTAC")
    spots, truth = _simulate_spots(E, labels, gains=[1, 0.6], seed=1)
    calls = _calls(decode_em(spots, encoding=E, base_labels=labels, n_restarts=2))
    assert np.mean(calls == truth) > 0.98  # 0.993


@pytest.mark.basecalls
def test_decode_em_chunking_and_whitelist():
    """Calls do not depend on Dask chunking, and barcodes are only used for matching."""
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    spots, truth = _simulate_spots(E, list("ATCG"), n=600)
    spots = spots.assign_coords(c=list("ATC"))
    model = fit_base_mixture(spots, dark_bases=["G"], n_restarts=2)
    df = decode_em(spots, model=model)
    chunked = spots.copy(data=da.from_array(spots.data, chunks=(100, -1, -1)))
    df_dask = decode_em(chunked, model=model).compute()
    assert (df["barcode"].values == df_dask["barcode"].values).all()

    whitelist = pd.DataFrame({"barcode": ["".join(r) for r in truth[:50]]})
    df_wl = decode_em(spots, barcodes=whitelist, model=model)
    assert (df["barcode"].values == df_wl["barcode"].values).all()
    assert df_wl["barcode_match"].iloc[:50].mean() > 0.8


@pytest.mark.basecalls
def test_decode_em_use_whitelist():
    """Refitting on whitelist-matched reads keeps calls accurate and uses the matches."""
    E = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    labels = list("GTAC")
    spots, truth = _simulate_spots(E, labels, gains=[1, 0.6], seed=1)
    whitelist = pd.DataFrame({"barcode": np.unique(["".join(r) for r in truth])})
    model = fit_base_mixture(
        spots,
        encoding=E,
        base_labels=labels,
        n_restarts=2,
        whitelist=whitelist["barcode"],
    )
    assert len(model["n_whitelist_matched"]) > 0
    assert model["n_whitelist_matched"][-1] > 0.9 * len(truth) / 2  # most fit reads
    calls = _calls(decode_em(spots, barcodes=whitelist, model=model))
    assert np.mean(calls == truth) > 0.98

    # use_whitelist passes the barcodes to the fit
    df = decode_em(
        spots,
        barcodes=whitelist,
        encoding=E,
        base_labels=labels,
        n_restarts=2,
        use_whitelist=True,
    )
    assert np.mean(_calls(df) == truth) > 0.98


@pytest.mark.basecalls
def test_decode_em_crosstalk():
    """Uncorrected crosstalk is estimated and corrected; corrected spots are left as is."""
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    spots, truth = _simulate_spots(E, list("ATCG"), n=800)
    spots = spots.assign_coords(c=list("ATC"))
    # A bleeds into T and C into A, as in the NIS-seq test data
    M = np.array([[1.0, 0.0, 0.35], [0.35, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mixed = spots.copy(data=spots.data @ M.T)
    model = fit_base_mixture(mixed, dark_bases=["G"], n_restarts=2)
    assert model["xtalk"] is not None
    df = decode_em(mixed, model=model)
    assert np.mean(_calls(df) == truth) > 0.99
    chunked = mixed.copy(data=da.from_array(mixed.data, chunks=(100, -1, -1)))
    df_dask = decode_em(chunked, model=model).compute()
    assert (df["barcode"].values == df_dask["barcode"].values).all()

    corrected = apply_channel_crosstalk_matrix(mixed, model["xtalk"])
    assert fit_base_mixture(corrected, dark_bases=["G"], n_restarts=2)["xtalk"] is None
    off = fit_base_mixture(mixed, dark_bases=["G"], n_restarts=2, xtalk_tol=None)
    assert off["xtalk"] is None

    # non-orthogonal encodings are not corrected
    E2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    spots2, _ = _simulate_spots(E2, list("GTAC"), n=400, seed=1)
    model2 = fit_base_mixture(spots2, encoding=E2, base_labels=list("GTAC"))
    assert model2["xtalk"] is None


@pytest.mark.basecalls
def test_fit_base_mixture_duplicate_rows():
    spots, _ = _simulate_spots(np.eye(2), ["A", "C"], n=50)
    with pytest.raises(ValueError, match="distinct"):
        fit_base_mixture(
            spots, encoding=[[1, 0], [0, 0], [0, 0]], base_labels=list("ACG")
        )
