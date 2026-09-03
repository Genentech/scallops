import anndata
import dask.array as da
import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA

from scallops.features.decomposition import PCA


@pytest.mark.features
def test_decomposition_compare_numpy():
    X = da.random.random((10, 10), chunks=(2, 2))
    adata = anndata.AnnData(X=X)
    pca = PCA(
        n_components=2,
        progress=False,
        batch_size=2,
    )
    pca.fit(adata.X)
    adata = anndata.AnnData(X=pca.transform(adata.X), obs=adata.obs.copy())
    pca.add_uns(adata)
    d = IncrementalPCA(n_components=2, batch_size=2)
    result2 = d.fit_transform(X.compute())
    np.testing.assert_array_equal(adata.uns["pca"]["mean"], d.mean_)
    np.testing.assert_array_equal(adata.uns["pca"]["variance"], d.explained_variance_)
    np.testing.assert_array_equal(
        adata.uns["pca"]["variance_ratio"], d.explained_variance_ratio_
    )
    np.testing.assert_array_equal(adata.uns["pca"]["PCs"], d.components_)
    np.testing.assert_almost_equal(adata.X, result2)
