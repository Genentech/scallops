import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from scallops.features.preprocessing import filter_data, transform_features_yj


@pytest.mark.parametrize("by", [None, "well"])
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.features
def test_filter_data(use_dask, by):
    adata = anndata.AnnData(
        X=da.arange(8, chunks=(1,)).reshape((4, 2))
        if use_dask
        else np.arange(8).reshape((4, 2)),
        obs=pd.DataFrame(
            data=dict(
                pert=["pert1", "pert2", "pert1", "pert2"],
                well=["well1", "well2", "well1", "well2"],
            )
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata.X = adata.X.astype(np.float32)
    adata.X[1, 0] = 100
    adata.X[0, 0] = np.nan
    # np.var(adata.X, axis=0) array([nan,  5.], dtype=float32)
    test_nan_filter = filter_data(adata, max_fraction_not_finite=0, min_variance=None)
    assert test_nan_filter.shape == (3, 2)
    # np.var(adata.X, axis=0) # array([nan,  5.]
    # np.var(adata[adata.obs['well'] == 'well1'].X, axis=0)  # array([nan,  4.])
    # np.var(adata[adata.obs['well'] == 'well2'].X, axis=0)  # array([2209.,    4.]
    d1 = filter_data(adata, max_fraction_not_finite=None, min_variance=0, by=by)
    # np.var(adata[1:].X, axis=0)  array([2006.2222, 2.6666667]
    d2 = filter_data(adata, max_fraction_not_finite=0, min_variance=5, by=by)

    assert d1.shape == (4, 1)
    assert d2.shape == (3, 1)
    assert d1.var.index.values[0] == "gene2"
    assert d2.var.index.values[0] == "gene1"


@pytest.mark.parametrize("by", [None, ["pert", "well"], ["well"]])
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.features
def test_transform_features_yj(by, use_dask):
    adata = anndata.AnnData(
        X=da.arange(8, chunks=(1,)).reshape((4, 2))
        if use_dask
        else np.arange(8).reshape((4, 2)),
        obs=pd.DataFrame(
            data=dict(
                pert=["pert1", "pert2", "pert1", "pert2"],
                well=["well1", "well2", "well1", "well2"],
            )
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata2 = adata.copy()
    if isinstance(adata2.X, da.Array):
        adata2.X = adata2.X.compute()
    df = adata2.to_df().join(adata2.obs)

    if by is not None:
        grouped = df.groupby(by)

        def single_group(x):
            x = x.copy()
            x["gene1"] = (
                PowerTransformer(method="yeo-johnson")
                .fit_transform(x["gene1"].values.reshape(-1, 1))
                .squeeze()
            )
            x["gene2"] = (
                PowerTransformer(method="yeo-johnson")
                .fit_transform(x["gene2"].values.reshape(-1, 1))
                .squeeze()
            )
            return x

        df = grouped.apply(single_group, include_groups=False).reset_index()

    else:
        df["gene1"] = (
            PowerTransformer(method="yeo-johnson")
            .fit_transform(df["gene1"].values.reshape(-1, 1))
            .squeeze()
        )
        df["gene2"] = (
            PowerTransformer(method="yeo-johnson")
            .fit_transform(df["gene2"].values.reshape(-1, 1))
            .squeeze()
        )
        df = df.reset_index(drop=True)
    columns_drop = df.columns[df.columns.str.startswith("level_")]
    if len(columns_drop) > 0:
        df = df.drop(columns_drop, axis=1)

    adata_transformed = transform_features_yj(adata, by=by)

    if isinstance(adata_transformed.X, da.Array):
        adata_transformed.X = adata_transformed.X.compute()
    df_test = (
        adata_transformed.to_df()
        .join(adata_transformed.obs)
        .sort_values(["pert", "well"])
    )
    df_test = df_test.sort_values(["pert", "well"]).reset_index(drop=True)
    df = df.sort_values(["pert", "well"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_test[df.columns], df)
