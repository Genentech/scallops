from pathlib import Path

import pandas as pd
import pytest

from scallops.io import read_experiment, read_image

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


# ── Raw experiment fixtures


@pytest.fixture(scope="module", autouse=True)
def experiment_c():
    return read_experiment(
        str(__experimentc_dir__.joinpath("input")),
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
    )


@pytest.fixture(scope="module", autouse=False)
def experiment_c_A1_102_cells():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.cells.tif")), dask=False
    )


@pytest.fixture(scope="module", autouse=False)
def experiment_c_A1_102_pheno_aligned():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.phenotype_aligned.tif")),
        dask=False,
    )


@pytest.fixture(scope="module", autouse=False)
def experiment_c_A1_102_pheno():
    return read_image(
        str(__pheno_dir__.joinpath("10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif")),
        dask=False,
    )


@pytest.fixture(scope="module", autouse=False)
def experiment_c_A1_102_aligned(experiment_c):
    """Pre-saved aligned image with t-coordinates corrected to match the live experiment.

    The saved TIF uses sequential 0-indexed t values; we reassign them from
    experiment_c so that read_barcodes extracts the correct barcode characters
    (including the cycle-6 gap present in ExperimentC).
    """
    img = (
        read_image(
            str(__processfig4_dir__.joinpath("10X_A1_Tile-102.aligned.tif")), dask=False
        )
        .transpose(*("z", "c", "t", "y", "x"))
        .rename({"z": "t", "t": "z"})
    )  # ops swaps z and t in saved tif
    return img.assign_coords(t=experiment_c.images["A1-102"].t.values)


@pytest.fixture(scope="module", autouse=False)
def experiment_c_A1_102_nuclei():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.nuclei.tif")), dask=False
    )


# ── NIS-seq fixtures (raw TIF data) ──


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
