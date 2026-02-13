import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest

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
        adata=adata,
        n_components=2,
        min_std=0,
        standardize=True,
        standardize_by=["plate", "well"],
        max_value=10,
        progress=False,
        batch_size=2,
    )
    assert result.X.shape == (10, 2)
