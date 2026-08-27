import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scallops.io import read_barcodes, read_experiment, read_image

__root__ = Path(__file__).parent
__data_dir__ = __root__.joinpath("data")
__experimentc_dir__ = __data_dir__.joinpath("experimentC")
__pheno_dir__ = __experimentc_dir__.joinpath("10X_c0-DAPI-p65ab")
__processfig4_dir__ = __data_dir__.joinpath("process_fig4")
__nisseq_dir__ = __data_dir__.joinpath("nis-seq")
__nisseq_tile__ = __nisseq_dir__.joinpath("Fig1E_NIS_HeLa_tile40")

assert __root__.joinpath(
    "data", "experimentC", "input", "10X_c1-SBS-1", "10X_c1-SBS-1_A1_Tile-102.sbs.tif"
).exists(), "Test files not found. Please ensure you have Git LFS installed"


# ── Raw experiment fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def experiment_c():
    return read_experiment(
        str(__experimentc_dir__.joinpath("input")),
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
    )


@pytest.fixture(scope="module", autouse=True)
def experiment_c_dask():
    return read_experiment(
        str(__experimentc_dir__.joinpath("input")),
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        dask=True,
    )


# ── Preprocessed ExperimentC fixtures ────────────────────────────────────


@pytest.fixture(scope="module")
def aligned_A1_102(experiment_c):
    """Aligned 4-ch SBS image for tile A1-102 (z=0, all time-channels)."""
    from scallops.registration.crosscorrelation import align_image

    image = experiment_c.images["A1-102"].isel(z=0)
    return align_image(
        image,
        align_within_time_channels=[1, 2, 3, 4],
        align_between_time_channel=0,
        filter_percentiles=[0, 90],
    )


@pytest.fixture(scope="module")
def barcodes_A1_102(aligned_A1_102):
    """Barcode whitelist DataFrame for ExperimentC tile A1-102."""
    return read_barcodes(
        str(__experimentc_dir__.joinpath("barcodes.csv")),
        aligned_A1_102.t.values - 1,
    )


# ── Image fixtures from process_fig4 ─────────────────────────────────────


@pytest.fixture(scope="module", autouse=False)
def dask_A1_102_cells():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.cells.tif")), dask=True
    )


@pytest.fixture(scope="module", autouse=False)
def array_A1_102_cells():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.cells.tif")), dask=False
    )


@pytest.fixture(scope="module", autouse=False)
def array_A1_102_alnpheno():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.phenotype_aligned.tif")),
        dask=False,
    )


@pytest.fixture(scope="module", autouse=False)
def array_A1_102_pheno():
    return read_image(
        str(__pheno_dir__.joinpath("10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif")),
        dask=False,
    )


@pytest.fixture(scope="module", autouse=False)
def array_A1_103_pheno():
    return read_image(
        str(__pheno_dir__.joinpath("10X_c0-DAPI-p65ab_A1_Tile-103.phenotype.tif")),
        dask=False,
    )


@pytest.fixture(scope="module", autouse=False)
def dask_A1_102_alnpheno():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.phenotype_aligned.tif")),
        dask=True,
    )


@pytest.fixture(scope="module", autouse=False)
def array_A1_102_aln():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.aligned.tif")), dask=False
    )


@pytest.fixture(scope="module", autouse=False)
def array_A1_102_nuclei():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.nuclei.tif")), dask=False
    )


# ── NIS-seq fixtures (raw TIF data) ──────────────────────────────────────


@pytest.fixture(scope="module")
def nis_seq_experiment():
    """Raw NIS-seq experiment loaded from TIF images (3 SBS channels: C=ch03, A=ch04, T=ch06)."""
    return read_experiment(
        str(__nisseq_tile__.joinpath("NIS-Seq-raw-images")),
        "cycle{t}_{well}_time001_tile{tile}_channel{c}.tif",
        group_by=("well", "tile"),
    )


@pytest.fixture(scope="module")
def nis_seq_nuclear_mask():
    """CellPose nuclear segmentation mask for NIS-seq HeLa tile40."""
    return (
        read_image(
            str(
                __nisseq_tile__.joinpath(
                    "NIS-Seq-cellpose-masks",
                    "nuclear_mask_cycle1_C10_time001_tile0040_channel02.tif",
                )
            )
        )
        .squeeze()
        .data.astype("int32")
    )


@pytest.fixture(scope="module")
def nis_seq_barcodes():
    """Brunello sgRNA barcode whitelist (library + scrambled), RC-trimmed to 14 mer."""

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    bru = pd.read_csv(
        str(__nisseq_dir__.joinpath("NIS-Seq_Brunello_sgRNAs", "Brunello_sgRNAs.txt")),
        sep="\t",
        header=None,
        names=["gene", "full_barcode"],
    )
    scr = pd.read_csv(
        str(
            __nisseq_dir__.joinpath(
                "NIS-Seq_Brunello_sgRNAs", "Brunello_sgRNAs_scrambled.txt"
            )
        ),
        sep="\t",
        header=None,
        names=["gene", "full_barcode"],
    )
    df = pd.concat([bru, scr], ignore_index=True)
    df["barcode"] = df["full_barcode"].map(rc).str[:14]
    return df


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
    from scallops.reads import (
        apply_channel_crosstalk_matrix,
        channel_crosstalk_matrix,
        decode_max,
        peaks_to_bases,
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

    # Channels 1,2,3 → ch03(C), ch04(A), ch06(T)
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

    # 2-col secondary threshold (from synthesised ch0=max(A,C), ch1=max(A,T) baseline)
    sp3 = np.clip(cor3.data, 0, None)
    bl2 = np.array(["G", "T", "A", "C"])
    ch0 = np.maximum(sp3[..., 0], sp3[..., 2])
    ch1 = np.maximum(sp3[..., 0], sp3[..., 1])
    sp2 = np.stack([ch0, ch1], axis=-1)
    above2 = sp2.max(axis=2) > 0.20 * sp2.max(axis=(1, 2), keepdims=True).squeeze(-1)
    df_t2 = pd.DataFrame(
        {
            "peak": cor3.peak.values,
            "label": cor3.label.values,
            "Q_mean": 60.0,
            "Q_min": 60.0,
            "barcode": [
                "".join(bl2[r]) for r in np.where(above2, sp2.argmax(axis=2), 2)
            ],
        }
    )
    df_t2["barcode_match"] = df_t2["barcode"].isin(set(df_bcn["barcode"]))
    try:
        thr_r2 = peak_thresholds_from_reads(df_t2.query("barcode_match")).iloc[0][
            "threshold"
        ]
    except Exception:
        thr_r2 = 3.0

    return types.SimpleNamespace(
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
