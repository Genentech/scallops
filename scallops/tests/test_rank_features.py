import anndata
import dask.array as da
import pandas as pd
import pytest

from scallops.features.rank import rank_features
from scallops.features.util import pandas_to_anndata


@pytest.fixture
def data(test_feature_table):
    return pandas_to_anndata(
        test_feature_table, ["Cells_Intensity_feature_1", "Cells_Intensity_feature_2"]
    )


@pytest.mark.parametrize("by", [None, ["well"]])
@pytest.mark.features
def testrank_features(client, data, by):
    reference_value = "NTC"
    perturbation_column = "gene_symbol"
    method = "welch_t"
    min_labels = 0

    rank_results = rank_features(
        data,
        by=by,
        perturbation_column=perturbation_column,
        reference_value=reference_value,
        method=method,
        min_labels=min_labels,
        iqr_multiplier=None,
    )
    rank_dask_results = rank_features(
        anndata.AnnData(
            X=da.from_array(data.X, chunks=(1, 2)), obs=data.obs, var=data.var
        ),
        by=by,
        perturbation_column=perturbation_column,
        reference_value=reference_value,
        method=method,
        min_labels=min_labels,
        iqr_multiplier=None,
    ).compute()
    pd.testing.assert_frame_equal(
        rank_dask_results[rank_results.columns],
        rank_results,
        check_dtype=False,
    )
