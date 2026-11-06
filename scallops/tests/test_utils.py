from pathlib import Path

import pandas as pd
import pytest

from scallops.datasets._registry import cache_dir
from scallops.utils import AssignTiles

DATA_DIR = cache_dir()

root_path = Path(__file__).parent


@pytest.mark.utils
def test_AssignTiles():
    df_example = pd.read_parquet(DATA_DIR.joinpath("dataframe", "sample_df_well1.pq"))
    coordinate = DATA_DIR.joinpath("dataframe", "well1_stitching_coords.csv.gz")
    chunk_size = None  # infer from coordinates
    tile_assigner = AssignTiles(
        coordinates=coordinate, chunksize=chunk_size, df=df_example
    )
    tiles = tile_assigner.find_tiles_within_squares()
    pd.testing.assert_series_equal(tiles, df_example.tile)
