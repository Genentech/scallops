import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest

from scallops.features.normalize import (
    typical_variation_normalization,
)

_TVN_EXPECTED = {
    None: np.array(
        [
            [-0.04291203, 0.18638646, 0.27220447, 0.67209463],
            [-0.55894541, 0.19174692, 1.38901053, -0.90084445],
            [-0.82425444, 0.38479927, -1.47732763, 3.14749198],
            [-1.11969741, -0.89830164, -1.90069852, -0.86270471],
            [-0.88068808, 0.23178571, 0.48233403, 1.53127634],
            [-0.19957925, -1.86270047, 0.87084285, -0.32437467],
            [0.64100803, -0.86808427, -0.74919245, 1.83509943],
            [-0.36284177, -1.11425827, -0.63895275, 1.05378921],
            [-0.23559986, -0.63306106, 0.67389772, -0.03479035],
            [-1.05332758, 0.31060587, 1.08879044, 1.11501746],
            [-0.59387476, -0.76407084, 2.00735328, -4.09244801],
            [-0.77007312, 0.88541763, 1.82289621, 2.29165866],
            [1.61351123, -1.24891423, 0.76755379, -0.90811570],
            [-0.39717464, -1.63492881, 0.07223129, 1.92812669],
            [0.33832026, -1.63599051, -0.44894328, 1.04266704],
            [-0.04774458, 1.71078293, -0.51303077, -0.32999887],
            [-0.40287516, 0.89580278, 1.51440777, 1.36796794],
            [-0.57148655, 1.60403197, -1.10150313, 6.38441994],
            [1.14876341, 1.26853257, -0.25066526, -0.53454907],
            [-1.21177281, -1.01978647, 1.87709182, 0.77113411],
            [0.30107182, -0.03582408, -1.41166936, 1.53527611],
            [-1.13960107, -0.46100769, 1.28503830, -0.98684316],
            [-0.95921666, -1.87359843, 1.97705132, -0.74135491],
            [-0.92897222, 0.48926766, 0.09891010, 2.60322085],
        ]
    ),
    "batch": np.array(
        [
            [-1.33471544, 0.12457799, 0.62343086, 0.36699049],
            [0.91357505, -0.20069304, 1.31926465, -1.04606469],
            [-2.21105043, 0.14328926, -2.07053861, 1.96162958],
            [-0.50493502, -0.82786092, -1.30877609, -0.28382954],
            [-2.58176047, 0.05376568, 1.23209613, 1.15732938],
            [2.38167396, -2.97736476, 1.14773720, -0.24436883],
            [-0.24194042, -0.62351638, -1.13156358, 1.11585414],
            [1.42445862, -2.03942167, -0.47186131, 1.63318310],
            [-1.99714981, -0.90575020, 1.08667222, -0.44818073],
            [-0.59012595, 0.05774033, 0.46008318, 1.51242204],
            [-3.39490751, -1.92184950, 2.50949242, -4.39059572],
            [-0.09910835, 0.30268977, 0.79948526, 2.69815685],
            [0.85265137, -0.87464330, 1.02609194, -0.87452368],
            [1.50770817, -2.81400017, -0.06373809, 2.59937579],
            [-1.10176232, -1.70154845, -0.88454640, 0.12587120],
            [1.42498802, 1.28276823, -0.28868392, -0.27011708],
            [-1.57871193, 1.21030524, 3.27860927, 1.83163329],
            [-0.55524132, 0.64468106, -2.39426611, 7.95117995],
            [0.72400449, 1.37358169, -0.51795923, -0.60832095],
            [-0.42605324, -1.42081271, 1.24925298, 1.09036546],
            [-0.70328053, -0.16656110, -2.39175146, 0.49805824],
            [-0.32992705, -0.51264764, 1.13737683, -0.95847542],
            [-3.63858797, -2.39135497, 3.20460946, -1.05630559],
            [-0.58963059, 0.03749456, -0.61699016, 3.40329392],
        ]
    ),
}


def _tvn_fixture(chunks=None):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((24, 4)) * 5 + 100
    return anndata.AnnData(
        X=da.from_array(x, chunks=chunks) if chunks is not None else x,
        obs=pd.DataFrame(
            data=dict(pert=["1", "2", "2"] * 8, batch=["1", "2"] * 12),
            index=[str(i) for i in range(24)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )


@pytest.mark.parametrize("chunks", [None, (24, 4), (6, 4), (7, 2)])
@pytest.mark.parametrize("by", [None, "batch"])
@pytest.mark.features
def test_typical_variation_normalization_values(by, chunks):
    """Well-conditioned input, pinned to the pre-refactor output."""
    data = _tvn_fixture(chunks)
    result = typical_variation_normalization(data, "pert=='1'", by)
    x = result.X.compute() if isinstance(result.X, da.Array) else np.asarray(result.X)

    # rows come back in input order, so obs needs no reindexing to line up
    assert list(result.obs.index) == [str(i) for i in range(24)]
    assert list(result.var.index) == [f"g{i}" for i in range(4)]
    assert sorted(result.uns["pca"]) == ["PCs", "mean", "variance", "variance_ratio"]
    # tolerance is set by the 8 decimals the expected values are written to, not by
    # the agreement itself, which is ~5e-9
    np.testing.assert_allclose(x, _TVN_EXPECTED[by], rtol=1e-6, atol=1e-7)


def _efaar_fixture(
    n_obs=120,
    n_features=8,
    n_groups=3,
    dtype=np.float64,
    seed=0,
    interleave=False,
):
    """Full-rank input with enough reference rows for PCA to keep every component.

    `interleave` scatters the groups through the rows instead of laying them out in
    contiguous blocks, which is what a chunk holding several groups looks like.
    """
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n_obs, n_features)) * 5 + 100).astype(dtype)
    groups = (
        np.tile(np.arange(n_groups), n_obs // n_groups)
        if interleave
        else np.repeat(np.arange(n_groups), n_obs // n_groups)
    )
    obs = pd.DataFrame(
        data={
            # 1 in 2 is a control, so the reference stays larger than n_features
            "pert": np.resize(["1", "2"], n_obs),
            "batch": [f"b{g}" for g in groups],
            "plate": [f"p{g % 2}" for g in groups],
        },
        index=[str(i) for i in range(n_obs)],
    )
    obs["plate_batch"] = obs["plate"] + "_" + obs["batch"]
    return anndata.AnnData(
        X=x, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_features)])
    )


def _assert_matches_efaar(data, by, efaar_batch_col, chunks=None, atol=1e-11):
    """Run both implementations on `data` and compare."""
    efaar = pytest.importorskip("efaar_benchmarking.efaar")
    expected = efaar.tvn_on_controls(
        embeddings=np.asarray(data.X).copy(),
        metadata=data.obs.copy(),
        pert_col="pert",
        control_key="1",
        batch_col=efaar_batch_col,
    )
    if chunks is not None:
        data = anndata.AnnData(
            X=da.from_array(np.asarray(data.X), chunks=chunks),
            obs=data.obs.copy(),
            var=data.var.copy(),
        )
    result = typical_variation_normalization(data, "pert=='1'", by)
    actual = (
        result.X.compute() if isinstance(result.X, da.Array) else np.asarray(result.X)
    )

    # guard against the comparison passing on degenerate output
    assert np.isfinite(actual).all()
    assert actual.std() > 0.1
    assert list(result.obs.index) == list(data.obs.index)
    np.testing.assert_allclose(actual, np.asarray(expected), atol=atol, rtol=1e-9)
    return actual


@pytest.mark.parametrize(
    ("n_obs", "n_features", "n_groups"), [(60, 4, 2), (120, 8, 3), (200, 16, 5)]
)
@pytest.mark.parametrize("by", [None, "batch"])
@pytest.mark.features
def test_tvn_matches_efaar(by, n_obs, n_features, n_groups):
    """Cross-check against the implementation this was adapted from.

    `efaar_benchmarking` is not a declared dependency, so these skip unless it is
    installed. The input is full rank, unlike the rank-1 fixture the hardcoded arrays
    above come from, so every component is meaningful.
    """
    data = _efaar_fixture(n_obs=n_obs, n_features=n_features, n_groups=n_groups)
    _assert_matches_efaar(data, by, by)


@pytest.mark.features
def test_tvn_matches_efaar_interleaved_groups():
    """Groups scattered through the rows, so a block holds several of them."""
    data = _efaar_fixture(interleave=True)
    _assert_matches_efaar(data, "batch", "batch")


@pytest.mark.features
def test_tvn_matches_efaar_uneven_groups():
    """Groups of different sizes, including one barely larger than n_features."""
    data = _efaar_fixture(n_obs=120, n_features=4)
    batch = np.array(["b0"] * 12 + ["b1"] * 40 + ["b2"] * 68)
    data.obs["batch"] = batch
    _assert_matches_efaar(data, "batch", "batch")


@pytest.mark.parametrize("chunks", [(120, 8), (30, 8), (17, 3), (7, 2)])
@pytest.mark.parametrize("by", [None, "batch"])
@pytest.mark.features
def test_tvn_matches_efaar_dask(by, chunks):
    """Dask input, including chunking on both axes as the production stores are."""
    data = _efaar_fixture()
    _assert_matches_efaar(data, by, by, chunks=chunks)


@pytest.mark.parametrize("by", [None, ["plate", "batch"]])
@pytest.mark.features
def test_tvn_matches_efaar_multi_column_by(by):
    """A list `by` groups on the combination, matching a single combined column."""
    data = _efaar_fixture(n_obs=120, n_features=4, n_groups=6)
    _assert_matches_efaar(data, by, None if by is None else "plate_batch")


@pytest.mark.parametrize("by", [None, "batch"])
@pytest.mark.features
def test_tvn_matches_efaar_float32(by):
    """float32 input stays float32.

    EFAAR promotes to float64 partway through, so the two diverge at float32 epsilon —
    measured at 1.3e-5 on values of scale 5. The tolerance is set just above that, not
    loosened until it passes.
    """
    data = _efaar_fixture(dtype=np.float32)
    actual = _assert_matches_efaar(data, by, by, atol=1e-4)
    assert actual.dtype == np.float32


@pytest.mark.features
def test_typical_variation_normalization_missing_by_value():
    """A missing `by` value forms its own group rather than dropping those rows."""
    data = _tvn_fixture()
    batch = data.obs["batch"].astype(object).to_numpy()
    batch[:4] = None
    data.obs["batch"] = batch
    result = typical_variation_normalization(data, "pert=='1'", "batch")
    assert result.shape == data.shape
    assert np.isfinite(np.asarray(result.X)).all()
