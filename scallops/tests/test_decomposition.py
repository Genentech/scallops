import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import IncrementalPCA

from scallops.features.decomposition import pca


@pytest.mark.features
def test_decomposition():
    X = da.random.random((10, 10), chunks=(2, 2))
    obs = pd.DataFrame(
        index=np.arange(10).astype(str),
        data=dict(plate="test", well=["a"] * 5 + ["b"] * 5),
    )
    adata = anndata.AnnData(X=X, obs=obs)
    result = pca(
        data=adata,
        n_components=2,
        min_std=0,
        standardize=True,
        standardize_by=["plate", "well"],
        max_value=10,
        progress=False,
        batch_size=2,
    )
    np.testing.assert_array_equal(result.obs.columns, ("plate", "well"))
    assert result.X.shape == (10, 2)


@pytest.mark.features
def test_decomposition_compare_numpy():
    X = da.random.random((10, 10), chunks=(2, 2))
    adata = anndata.AnnData(X=X)
    result = pca(
        data=adata,
        n_components=2,
        standardize=False,
        progress=False,
        batch_size=2,
    )

    d = IncrementalPCA(n_components=2, batch_size=2)
    result2 = d.fit_transform(X.compute())
    np.testing.assert_array_equal(result.uns["pca"]["mean"], d.mean_)
    np.testing.assert_array_equal(result.uns["pca"]["variance"], d.explained_variance_)
    np.testing.assert_array_equal(
        result.uns["pca"]["variance_ratio"], d.explained_variance_ratio_
    )
    np.testing.assert_array_equal(result.uns["pca"]["PCs"], d.components_)
    np.testing.assert_almost_equal(result.X, result2)
