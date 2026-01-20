from pathlib import Path

import dask.array as da
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from scallops.reads import (
    apply_channel_crosstalk_matrix,
    channel_crosstalk_matrix,
    decode_max,
    li_and_speed_cc_number,
)
from scallops.visualize.crosstalk import pairwise_channel_scatter_plot

mpl.use("Agg")

data_path = Path("scallops") / "tests" / "data"


def bases_df_to_xarray(df):
    df = df.rename({"cycle": "t", "channel": "c", "i": "y", "j": "x"}, axis=1).drop(
        ["well", "tile"], axis=1
    )
    cells = df.drop_duplicates("read")["cell"]
    df = df.set_index(["read", "t", "c"])
    # pivot to (read,t,c) where read has label, y, x
    a = df["intensity"].to_xarray()

    coords = dict(label=("read", cells), c=a.c.values, t=a.t.values)
    a = xr.DataArray(data=a, dims=a.dims, coords=coords)
    return a


@pytest.mark.basecalls
def test_correction_per_chunk():
    barcode_indices = np.concatenate((np.arange(0, 5), np.arange(6, 10)))
    barcodes_df = pd.read_csv(data_path.joinpath("experimentC", "barcodes.csv"))
    _barcodes = barcodes_df["barcode"]
    barcodes = ""
    for i in barcode_indices:
        barcodes += _barcodes.str[i]
    barcodes_df["barcode"] = barcodes

    df_bases = pd.read_csv(
        data_path.joinpath("process_fig4", "10X_A1_Tile-102.bases.csv")
    )
    bases = bases_df_to_xarray(df_bases)
    bases = bases.where(bases.label > 0, drop=True)  # (10284, 9, 4)
    w = channel_crosstalk_matrix(bases, method="median")

    w_by_t = channel_crosstalk_matrix(bases, method="median", by_t=True)
    corrected = apply_channel_crosstalk_matrix(bases, w)
    corrected_by_t = apply_channel_crosstalk_matrix(bases, w_by_t)

    bases.data = da.from_array(bases.data).rechunk((500, -1, -1))
    w_dask = channel_crosstalk_matrix(bases, method="median").compute()
    w_dask_by_t = channel_crosstalk_matrix(bases, method="median", by_t=True).compute()
    corrected_dask = apply_channel_crosstalk_matrix(bases, w).compute()
    np.testing.assert_array_equal(corrected_dask, corrected)

    corrected_by_t_dask = apply_channel_crosstalk_matrix(bases, w_by_t).compute()
    np.testing.assert_array_equal(corrected_by_t_dask, corrected_by_t)

    w_diff = np.abs(w - w_dask).max()
    w_diff_by_t = np.abs(w_by_t - w_dask_by_t).max()
    assert w_diff < 0.02, w_diff
    assert w_diff_by_t < 0.09, w_diff_by_t


@pytest.mark.basecalls
def test_li_speed_no_iters():
    df_bases = pd.read_csv(
        data_path.joinpath("process_fig4", "10X_A1_Tile-102.bases.csv")
    )
    bases = bases_df_to_xarray(df_bases)
    w = channel_crosstalk_matrix(
        bases.where(bases.label > 0, drop=True),
        method="li_and_speed",
        n_iter_max=0,
    )
    corrected_intensity = apply_channel_crosstalk_matrix(bases, w)
    corrected_intensity = corrected_intensity.to_numpy()
    intensity = bases.to_numpy()
    np.testing.assert_equal(corrected_intensity, intensity)


@pytest.mark.basecalls
def test_li_and_speed_correct_multiple():
    df_bases = pd.read_csv(
        data_path.joinpath("process_fig4", "10X_A1_Tile-102.bases.csv")
    )
    bases = bases_df_to_xarray(df_bases)
    inverse1 = channel_crosstalk_matrix(
        bases.where(bases.label > 0, drop=True),
        method="li_and_speed",
        n_iter_max=1,
        normalize_w=True,
    )
    corrected_intensity = apply_channel_crosstalk_matrix(
        bases, inverse1, dtype=np.float64
    )
    inverse2 = channel_crosstalk_matrix(
        corrected_intensity.where(corrected_intensity.label > 0, drop=True),
        method="li_and_speed",
        n_iter_max=1,
        normalize_w=True,
    )

    corrected_intensity = apply_channel_crosstalk_matrix(
        corrected_intensity, inverse2, dtype=np.float64
    )
    inverse3 = channel_crosstalk_matrix(
        bases.where(bases.label > 0, drop=True),
        method="li_and_speed",
        n_iter_max=2,
        normalize_w=True,
    )
    corrected_intensity3 = apply_channel_crosstalk_matrix(
        bases, inverse3, dtype=np.float64
    )
    np.testing.assert_allclose(corrected_intensity.data, corrected_intensity3.data)


@pytest.mark.basecalls
def test_li_and_speed_correct():
    df_bases = pd.read_csv(
        data_path.joinpath("process_fig4", "10X_A1_Tile-102.bases.csv")
    )
    bases = bases_df_to_xarray(df_bases)

    inverse = channel_crosstalk_matrix(
        bases.where(bases.label > 0, drop=True),
        method="li_and_speed",
        quantile_range=[0.6, 1],
    )
    corrected_intensity = apply_channel_crosstalk_matrix(bases, inverse)

    assert li_and_speed_cc_number(bases) >= 0.3
    assert li_and_speed_cc_number(corrected_intensity) <= 0.11
    fig, axes = pairwise_channel_scatter_plot(corrected_intensity, q=[0.6, 0.999])
    plt.close(fig)


@pytest.mark.basecalls
def test_correction_is_better():
    # no data at t=6 for this exp
    # test that we get more cells by performing crosstalk correction
    barcode_indices = np.concatenate((np.arange(0, 5), np.arange(6, 10)))
    barcodes_df = pd.read_csv(data_path.joinpath("experimentC", "barcodes.csv"))
    _barcodes = barcodes_df["barcode"]
    barcodes = ""
    for i in barcode_indices:
        barcodes += _barcodes.str[i]
    barcodes_df["barcode"] = barcodes

    df_bases = pd.read_csv(
        data_path.joinpath("process_fig4", "10X_A1_Tile-102.bases.csv")
    )
    bases = bases_df_to_xarray(df_bases)
    w = channel_crosstalk_matrix(
        bases.where(bases.label > 0, drop=True), method="median"
    )

    corrected_reads = apply_channel_crosstalk_matrix(bases, w)
    df_reads_median = decode_max(corrected_reads)
    median_total = df_reads_median.query("label > 0")["barcode"].isin(barcodes).sum()

    df_reads_none = decode_max(bases)
    none_total = df_reads_none.query("label > 0")["barcode"].isin(barcodes).sum()

    assert median_total > none_total

    w = channel_crosstalk_matrix(
        bases.where(bases.label > 0, drop=True),
        method="li_and_speed",
        quantile_range=[0.5, 1],
    )
    corrected_reads = apply_channel_crosstalk_matrix(bases, w)
    df_reads_li_speed = decode_max(corrected_reads)

    li_speed_total = (
        df_reads_li_speed.query("label > 0")["barcode"].isin(barcodes).sum()
    )
    assert li_speed_total > none_total
