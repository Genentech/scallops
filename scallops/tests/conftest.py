from pathlib import Path

import pytest

from scallops.io import read_experiment, read_image

__root__ = Path(__file__).parent

__data_dir__ = __root__.joinpath("data")

__experimentc_dir__ = __data_dir__.joinpath("experimentC")
__pheno_dir__ = __experimentc_dir__.joinpath("10X_c0-DAPI-p65ab")

__processfig4_dir__ = __data_dir__.joinpath("process_fig4")

assert __root__.joinpath(
    "data", "experimentC", "input", "10X_c1-SBS-1", "10X_c1-SBS-1_A1_Tile-102.sbs.tif"
).exists(), "Test files not found. Please ensure you have Git LFS installed"


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
