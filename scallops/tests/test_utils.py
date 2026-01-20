from pathlib import Path

import pandas as pd
import pytest

from scallops.utils import AssignTiles

root_path = Path(__file__).parent


@pytest.mark.utils
def test_AssignTiles():
    df_example = pd.read_parquet(
        root_path.joinpath("data", "dataframe", "sample_df_well1.pq")
    )
    coordinate = root_path.joinpath(
        "data", "dataframe", "well1_stitching_coords.csv.gz"
    )
    chunk_size = None  # infer from coordinates
    tile_assigner = AssignTiles(
        coordinates=coordinate, chunksize=chunk_size, df=df_example
    )
    tiles = tile_assigner.find_tiles_within_squares()
    pd.testing.assert_series_equal(tiles, df_example.tile)
