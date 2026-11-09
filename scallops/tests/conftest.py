from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from distributed import Client, LocalCluster

from scallops.datasets._registry import cache_dir
from scallops.io import read_experiment, read_image

__root__ = Path(__file__).parent

__data_dir__ = cache_dir()

__experimentc_dir__ = __data_dir__.joinpath("experimentC")
__pheno_dir__ = __experimentc_dir__.joinpath("10X_c0-DAPI-p65ab")

__processfig4_dir__ = __data_dir__.joinpath("process_fig4")

# NIS-seq test data is not yet in the GitHub release assets; it is tracked in the repo.
__nisseq_dir__ = __root__.joinpath("data", "nis-seq")
__nisseq_tile__ = __nisseq_dir__.joinpath("Fig1E_NIS_HeLa_tile40")


def pytest_configure(config):
    # Download test data from GitHub release assets to the scallops cache directory
    # (override with SCALLOPS_CACHE_DIR); up-to-date files are skipped. Run only in the
    # controller process so pytest-xdist workers don't download concurrently.
    if hasattr(config, "workerinput"):
        return
    from scallops.datasets._registry import fetch

    fetch(verbose=True)


@pytest.fixture(scope="module")
def client():
    # Start a local Dask cluster for the entire test module
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    yield client  # Provide the client to the tests
    client.close()
    cluster.close()


@pytest.fixture(scope="module")
def test_feature_table():
    return pd.DataFrame(
        data=dict(
            label=np.arange(6),
            Cells_Intensity_feature_1=[1, 2, 4, 8, 16, 32],
            Cells_Intensity_feature_2=[10, 20, 40, 80, 160, 320],
            gene_symbol=["a", "NTC", "a", "NTC", "a", "NTC"],
            well=["a", "a", "a", "b", "b", "b"],
            plate=["a", "a", "a", "b", "b", "b"],
            Nuclei_AreaShape_Center_Y=[1, 7, 12, 16, 19, 21],
            Nuclei_AreaShape_Center_X=[1, 7, 12, 16, 19, 21],
        ),
    )


@pytest.fixture(scope="module")
def experiment_c():
    return read_experiment(
        str(__experimentc_dir__.joinpath("input")),
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
    )


@pytest.fixture(scope="module")
def experiment_c_A1_102_cells():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.cells.tif")), dask=False
    )


@pytest.fixture(scope="module")
def experiment_c_A1_102_pheno_aligned():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.phenotype_aligned.tif")),
        dask=False,
    )


@pytest.fixture(scope="module")
def experiment_c_A1_102_pheno():
    return read_image(
        str(__pheno_dir__.joinpath("10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif")),
        dask=False,
    )


@pytest.fixture(scope="module")
def experiment_c_A1_102_aligned():
    return (
        read_image(
            str(__processfig4_dir__.joinpath("10X_A1_Tile-102.aligned.tif")), dask=False
        )
        .transpose(*("z", "c", "t", "y", "x"))
        .rename({"z": "t", "t": "z"})
    ).assign_coords(t=[1, 2, 3, 4, 5, 7, 8, 9, 10])


@pytest.fixture(scope="module")
def experiment_c_A1_102_nuclei():
    return read_image(
        str(__processfig4_dir__.joinpath("10X_A1_Tile-102.nuclei.tif")), dask=False
    )


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
