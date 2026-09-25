import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from scallops.features.agg import agg_features


@pytest.mark.parametrize("by", ["pert", ["pert", "well"]])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("agg_func", ["mean", "median"])
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.features
def test_agg_features(by, weighted, agg_func, use_dask):
    adata = anndata.AnnData(
        X=da.arange(8, chunks=(1,)).reshape((4, 2))
        if use_dask
        else np.arange(8).reshape((4, 2)),
        obs=pd.DataFrame(
            data=dict(
                pert=["pert1", "pert2", "pert1", "pert2"],
                well=["well1", "well2", "well1", "well2"],
                weight=[2, 4, 8, 6],
            )
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata2 = adata.copy()
    if isinstance(adata2.X, da.Array):
        adata2.X = adata2.X.compute()

    df = adata2.to_df().join(adata2.obs)
    grouped = df.groupby(by)
    if weighted:
        if agg_func == "mean":
            result_df = grouped.agg(
                gene1=(
                    "gene1",
                    lambda x: np.average(x, weights=df.loc[x.index, "weight"]),
                ),
                gene2=(
                    "gene2",
                    lambda x: np.average(x, weights=df.loc[x.index, "weight"]),
                ),
            )
        else:
            result_df = grouped.agg(
                gene1=(
                    "gene1",
                    lambda x: DescrStatsW(
                        data=x, weights=df.loc[x.index, "weight"]
                    ).quantile(probs=[0.5]),
                ),
                gene2=(
                    "gene2",
                    lambda x: DescrStatsW(
                        data=x, weights=df.loc[x.index, "weight"]
                    ).quantile(probs=[0.5]),
                ),
            )
    else:
        result_df = grouped.agg({"gene1": agg_func, "gene2": agg_func})
    result_df = result_df.reset_index().sort_values("pert")
    result_df.index = result_df.index.astype(str)
    agg_d = agg_features(
        adata, by, weights_col="weight" if weighted else None, agg_func=agg_func
    )
    if isinstance(agg_d.X, da.Array):
        agg_d.X = agg_d.X.compute()
    assert agg_d.shape == (2, 2)
    agg_df = agg_d.to_df().join(agg_d.obs).sort_values("pert").drop("count", axis=1)
    pd.testing.assert_frame_equal(
        result_df[agg_df.columns].reset_index(drop=True), agg_df.reset_index(drop=True)
    )
