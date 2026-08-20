from pathlib import Path


def example_feature_summary_stats() -> Path:
    """Example phenotype feature summary statistics.

    :return: Path to Parquet file.
    """
    import pooch

    registry = {
        "features/plotting-notebook-example-summary-stats.pq": None,
    }
    path = pooch.os_cache("scallops")
    p = pooch.create(
        path=path,
        base_url="https://github.com/Genentech/scallops/raw/refs/heads/main/scallops/tests/data/",
        registry=registry,
    )
    for key in registry.keys():
        p.fetch(key)
    return path / "features" / "plotting-notebook-example-summary-stats.pq"


def feldman_2019_small() -> Path:
    """Example SBS and phenotype tiles from Optical Pooled Screens in Human Cells
    by Feldman et al. (https://www.cell.com/cell/fulltext/S0092-8674(19)31067-0).

    :return: Path to root data directory.
    """

    import pooch

    registry = {
        "experimentC/input/10X_c10-SBS-10/10X_c10-SBS-10_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c10-SBS-10/10X_c10-SBS-10_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c7-SBS-7/10X_c7-SBS-7_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c7-SBS-7/10X_c7-SBS-7_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c2-SBS-2/10X_c2-SBS-2_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c2-SBS-2/10X_c2-SBS-2_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c9-SBS-9/10X_c9-SBS-9_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c9-SBS-9/10X_c9-SBS-9_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c4-SBS-4/10X_c4-SBS-4_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c4-SBS-4/10X_c4-SBS-4_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c3-SBS-3/10X_c3-SBS-3_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c3-SBS-3/10X_c3-SBS-3_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c8-SBS-8/10X_c8-SBS-8_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c8-SBS-8/10X_c8-SBS-8_A1_Tile-103.sbs.tif": None,
        "experimentC/input/10X_c5-SBS-5/10X_c5-SBS-5_A1_Tile-102.sbs.tif": None,
        "experimentC/input/10X_c5-SBS-5/10X_c5-SBS-5_A1_Tile-103.sbs.tif": None,
        "experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-103.phenotype.tif": None,
        "experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif": None,
        "experimentC/barcodes.csv": None,
    }
    path = pooch.os_cache("scallops")
    p = pooch.create(
        path=path,
        base_url="https://github.com/Genentech/scallops/raw/refs/heads/main/scallops/tests/data/",
        registry=registry,
    )
    for key in registry.keys():
        p.fetch(key)
    return path / "experimentC"


def nis_seq_schmid_burgk() -> Path:
    """Example SBS tile from Schmid-Burgk et al.
    (https://www.nature.com/articles/s41587-024-02516-5#MOESM3).

    :return: Path to root data directory.
    """

    import pooch

    registry = {
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle1_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle1_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle1_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle1_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle2_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle2_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle2_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle2_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle3_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle3_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle3_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle3_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle4_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle4_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle4_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle4_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle5_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle5_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle5_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle5_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle6_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle6_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle6_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle6_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle7_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle7_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle7_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle7_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle8_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle8_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle8_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle8_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle9_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle9_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle9_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle9_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle10_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle10_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle10_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle10_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle11_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle11_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle11_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle11_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle12_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle12_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle12_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle12_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle13_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle13_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle13_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle13_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle14_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle14_C10_time001_tile0040_channel03.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle14_C10_time001_tile0040_channel04.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-raw-images/cycle14_C10_time001_tile0040_channel06.tif": None,
        "nis-seq/Fig1E_NIS_HeLa_tile40/NIS-Seq-cellpose-masks/nuclear_mask_cycle1_C10_time001_tile0040_channel02.tif": None,
        "nis-seq/NIS-Seq_Brunello_sgRNAs/Brunello_sgRNAs.txt": None,
        # "nis-seq/results/test_NuclearSequences.txt": None,
    }
    path = pooch.os_cache("scallops")
    p = pooch.create(
        path=path,
        base_url="https://github.com/Genentech/scallops/raw/refs/heads/dark-bases/scallops/tests/data/",  # FIXME
        registry=registry,
    )
    for key in registry.keys():
        p.fetch(key)
    return path / "nis-seq"
