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
