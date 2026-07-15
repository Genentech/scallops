import dask.array as da
import numpy as np
import pandas as pd
import pytest

from scallops.features.map_eval import recall


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.features
def test_compute_recall(use_dask):
    null_distribution = np.array([1, 2, 3, 4, 5])
    query_distribution = np.array([1, 5])
    if use_dask:
        null_distribution = da.from_array(null_distribution)
        query_distribution = da.from_array(query_distribution)
    recall_threshold_pairs = [(0.1, 0.9), (0.2, 0.8)]
    expected_result = pd.DataFrame(
        data=dict(threshold=[(0.1, 0.9), (0.2, 0.8)], recall=[0.0, 1.0])
    )
    result = recall(null_distribution, query_distribution, recall_threshold_pairs)
    pd.testing.assert_frame_equal(result, expected_result)
