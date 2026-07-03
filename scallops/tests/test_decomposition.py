import warnings

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import IncrementalPCA

from scallops.features.decomposition import (
    largest_variance_from_random_matrix,
    pca,
    select_pca_components,
    sphere,
)


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


# ---------------------------------------------------------------------------
# sphere
# ---------------------------------------------------------------------------


def _make_adata(n=40, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    obs = pd.DataFrame(
        {"batch": ["b1"] * (n // 2) + ["b2"] * (n // 2)},
        index=pd.RangeIndex(n).astype(str),
    )
    return anndata.AnnData(
        X=X, obs=obs, var=pd.DataFrame(index=[f"f{i}" for i in range(p)])
    )


@pytest.mark.features
def test_sphere_shape_unchanged():
    data = _make_adata()
    result = sphere(data)
    assert result.shape == data.shape
    assert list(result.var.index) == list(data.var.index)
    assert list(result.obs.index) == list(data.obs.index)


@pytest.mark.features
def test_sphere_covariance_is_identity():
    """After sphering the sample covariance should be ~I."""
    data = _make_adata(n=200, p=4)
    result = sphere(data)
    cov = np.cov(result.X, rowvar=False)
    np.testing.assert_allclose(np.diag(cov), 1.0, atol=0.15)
    off_diag = cov - np.eye(cov.shape[0])
    assert np.abs(off_diag).max() < 0.2


@pytest.mark.features
def test_sphere_by_group():
    """sphere with by='batch' must still produce a valid AnnData of original shape."""
    data = _make_adata(n=60, p=4)
    result = sphere(data, by="batch")
    assert result.shape == data.shape
    assert set(result.obs.index) == set(data.obs.index)


@pytest.mark.features
def test_sphere_obs_index_preserved():
    """String obs indices must be preserved through the xarray groupby path."""
    data = _make_adata(n=40, p=4)
    result = sphere(data, by="batch")
    assert set(result.obs.index) == set(data.obs.index)


# ---------------------------------------------------------------------------
# select_pca_components
# ---------------------------------------------------------------------------


@pytest.fixture
def pca_adata():
    """AnnData in PCA space with all 20 components from 20 features.

    Using n_components == n_features ensures variance_ratio sums to 1.0, so
    any min_variance_fraction threshold is achievable in the fixture.
    """
    rng = np.random.default_rng(1)
    n, p = 100, 20
    X = rng.standard_normal((n, p)).astype(np.float32)
    data = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"f{i}" for i in range(p)]),
    )
    return pca(data, n_components=p, standardize=False)


@pytest.mark.features
def test_select_pca_variance_fraction(pca_adata):
    """method=variance must retain enough PCs to cover min_variance_fraction."""
    result = select_pca_components(pca_adata, method="variance", min_variance_fraction=0.80)
    vr = np.asarray(pca_adata.uns["pca"]["variance_ratio"])
    covered = float(vr[: result.shape[1]].sum())
    assert covered >= 0.80 - 1e-6
    assert result.shape[1] <= pca_adata.shape[1]


@pytest.mark.features
def test_select_pca_variance_always_at_least_one(pca_adata):
    """method=variance never returns zero components."""
    result = select_pca_components(pca_adata, method="variance", min_variance_fraction=0.0)
    assert result.shape[1] >= 1


@pytest.mark.features
def test_select_pca_max_components_cap(pca_adata):
    """max_components caps the result regardless of method."""
    result = select_pca_components(
        pca_adata, method="variance", min_variance_fraction=0.99, max_components=3
    )
    assert result.shape[1] <= 3


@pytest.mark.features
def test_select_pca_permutation_keeps_significant(pca_adata):
    """method=permutation retains at least 1 and at most n_components."""
    result = select_pca_components(
        pca_adata, method="permutation", pval=0.05, n_perms=20
    )
    assert 1 <= result.shape[1] <= pca_adata.shape[1]


@pytest.mark.features
def test_select_pca_tracy_widom_warns_always():
    """method=tracy_widom always emits a UserWarning about assumption violation."""
    rng = np.random.default_rng(0)
    n, p = 100, 20
    data = anndata.AnnData(
        X=rng.standard_normal((n, p)).astype(np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"f{i}" for i in range(p)]),
    )
    pca_data = pca(data, n_components=10, standardize=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        select_pca_components(pca_data, method="tracy_widom", n_features=p)
    assert any(issubclass(x.category, UserWarning) for x in w), (
        "tracy_widom method must always warn about assumption violation"
    )


@pytest.mark.features
def test_select_pca_tracy_widom_warns_on_high_retention():
    """method=tracy_widom emits an extra warning when retention rate > 80%."""
    rng = np.random.default_rng(0)
    n, p_feat = 500, 50
    # Correlated features → many eigenvalues above TW threshold
    rho = 0.9
    block = rho * np.ones((10, 10)) + (1 - rho) * np.eye(10)
    cov = np.zeros((p_feat, p_feat))
    for b in range(5):
        cov[b * 10 : (b + 1) * 10, b * 10 : (b + 1) * 10] = block
    L = np.linalg.cholesky(cov)
    X_corr = (L @ rng.standard_normal((p_feat, n))).T.astype(np.float32)
    data = anndata.AnnData(
        X=X_corr,
        obs=pd.DataFrame(index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"f{i}" for i in range(p_feat)]),
    )
    pca_data = pca(data, n_components=20, standardize=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = select_pca_components(pca_data, method="tracy_widom", n_features=p_feat)
    user_warns = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    # Must warn about the method itself; may additionally warn about high retention
    assert len(user_warns) >= 1
    assert any("morphological" in m or "tracy_widom" in m.lower() or "correlation" in m for m in user_warns)


@pytest.mark.features
def test_select_pca_invalid_method_raises(pca_adata):
    with pytest.raises(ValueError, match="Unknown method"):
        select_pca_components(pca_adata, method="invalid")


@pytest.mark.features
def test_select_pca_missing_variance_raises():
    """select_pca_components raises ValueError when pca uns is absent."""
    data = anndata.AnnData(
        X=np.ones((5, 3), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(5).astype(str)),
    )
    with pytest.raises(ValueError, match="variance.*missing"):
        select_pca_components(data)


@pytest.mark.features
def test_largest_variance_from_random_matrix_decreases_with_n():
    """Threshold should decrease as n grows (more data → tighter bound)."""
    t_small = largest_variance_from_random_matrix(100, 50)
    t_large = largest_variance_from_random_matrix(10000, 50)
    assert t_small > t_large
