import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from scipy.stats import median_abs_deviation

from scallops.features.constants import _centroid_column_names
from scallops.features.normalize import (
    _nearest_neighbors_indices,
    normalize_features,
)
from scallops.features.util import _slice_anndata, pandas_to_anndata


@pytest.fixture
def data(test_feature_table):
    return pandas_to_anndata(
        test_feature_table, ["Cells_Intensity_feature_1", "Cells_Intensity_feature_2"]
    )


def _diff_values(ds, normed_data, normalize, robust, reference, scaling, n_neighbors):
    values = ds.X.copy()
    ref_ds = ds
    if reference is not None:
        ref_ds = _slice_anndata(ds, ds.obs.query(reference).index)

    ref_values = ref_ds.X
    if normalize == "zscore":
        mean = np.median(ref_values, axis=0) if robust else np.mean(ref_values, axis=0)
        std = (
            median_abs_deviation(ref_values, axis=0, scale="normal")
            if robust
            else np.std(ref_values, axis=0)
        )
    else:
        if normalize == "nn-zscore":
            query = ds.X
            ref = ref_ds.X
        elif normalize == "local-zscore":
            query = np.stack(
                (
                    ds.obs[_centroid_column_names[0]].values,
                    ds.obs[_centroid_column_names[1]].values,
                ),
                axis=1,
            )

            ref = np.stack(
                (
                    ref_ds.obs[_centroid_column_names[0]].values,
                    ref_ds.obs[_centroid_column_names[1]].values,
                ),
                axis=1,
            )

        indices = _nearest_neighbors_indices(ref, query, n_neighbors=n_neighbors)
        ref_values = ref_values[indices]
        # ref_values dims are (labels,neighbors,features)
        if robust:
            mean = np.median(ref_values, axis=1)
            std = median_abs_deviation(ref_values, axis=1, scale="normal")

        else:
            mean = np.mean(ref_values, axis=1)
            std = np.std(ref_values, axis=1)
    values = values - mean

    if scaling:
        values = values / std

    np.testing.assert_allclose(
        values,
        _slice_anndata(normed_data, ds.obs.index).X,
        rtol=2.07703709e-15,
        atol=2.22044605e-15,
        err_msg="Expected values not equal",
    )


def _compare_anndata(data1: anndata.AnnData, data2: anndata.AnnData):
    data2 = _slice_anndata(data2, data1.obs.index)

    np.testing.assert_allclose(
        data1.X,
        data2.X,
        rtol=2.07703709e-16,
        atol=2.22044605e-16,
        err_msg="Expected values not equal",
    )
    pd.testing.assert_frame_equal(data1.obs, data2.obs)
    pd.testing.assert_frame_equal(data1.var, data2.var)


@pytest.mark.parametrize("normalize", ["zscore", "local-zscore"])
@pytest.mark.parametrize("reference", ["gene_symbol=='NTC'", None])
@pytest.mark.parametrize("robust", [True, False])
@pytest.mark.parametrize("by", [["plate", "well"], None])
@pytest.mark.parametrize("sort", ["well", None])
@pytest.mark.features
def test_norm_features(client, data, normalize, by, robust, reference, sort, tmp_path):
    if sort is not None:
        data = _slice_anndata(data, data.obs.sort_values(sort).index)
    n_neighbors = 3
    if by is not None and reference is not None:
        n_neighbors = 1
    scaling = n_neighbors > 1

    normed_data = normalize_features(
        data,
        reference_query=reference,
        normalize=normalize,
        robust=robust,
        by=by,
        n_neighbors=n_neighbors,
        scaling=scaling,
    )
    if by is not None:
        indices = data.obs.groupby(by).indices
        for name in indices:
            query = []
            for i in range(len(by)):
                query.append(f"{by[i]}=='{name[i]}'")

            _diff_values(
                ds=_slice_anndata(data, indices[name]),
                normed_data=_slice_anndata(
                    normed_data, normed_data.obs.query("&".join(query)).index
                ),
                normalize=normalize,
                robust=robust,
                reference=reference,
                scaling=scaling,
                n_neighbors=n_neighbors,
            )

    else:
        _diff_values(
            ds=data,
            normed_data=normed_data,
            normalize=normalize,
            robust=robust,
            reference=reference,
            scaling=scaling,
            n_neighbors=n_neighbors,
        )
    if normalize == "local-zscore":
        normed_data2 = normalize_features(
            data,
            reference_query=reference,
            normalize=normalize,
            robust=robust,
            by=by,
            n_neighbors=n_neighbors,
            scaling=scaling,
            tile_bytes=0,
        )
        _compare_anndata(normed_data, normed_data2)

    dask_data = anndata.AnnData(
        X=da.from_array(data.X, chunks=(1, 1)), obs=data.obs, var=data.var
    )

    normed_data_dask = normalize_features(
        dask_data,
        reference_query=reference,
        normalize=normalize,
        robust=robust,
        by=by,
        n_neighbors=n_neighbors,
        scaling=scaling,
    )
    normed_data_dask.X = normed_data_dask.X.compute()
    if normalize == "local-zscore":
        normed_data_dask2 = normalize_features(
            dask_data,
            reference_query=reference,
            normalize=normalize,
            robust=robust,
            by=by,
            n_neighbors=n_neighbors,
            scaling=scaling,
            tile_bytes=0,
        )
        normed_data_dask2.X = normed_data_dask2.X.compute()
        _compare_anndata(normed_data_dask, normed_data_dask2)

    normed_data = _slice_anndata(
        normed_data, normed_data.obs.sort_values("label").index
    )
    normed_data_dask = _slice_anndata(
        normed_data_dask, normed_data_dask.obs.sort_values("label").index
    )
    _compare_anndata(normed_data, normed_data_dask)


@pytest.mark.parametrize("groups_per_chunk", [1, 2, 4])
@pytest.mark.parametrize("reference", ["gene_symbol=='NTC'", None])
@pytest.mark.parametrize("robust", [True, False])
@pytest.mark.features
def test_norm_local_zscore_groups_per_chunk(
    groups_per_chunk, reference, robust, caplog
):
    """Blocks may hold more than one group, so neighbor indices must be per block."""
    n_groups = 8
    group_size = 6
    n_features = 3
    n_neighbors = 2
    rng = np.random.default_rng(0)
    n = n_groups * group_size
    x = rng.normal(size=(n, n_features)) * 10
    obs = pd.DataFrame(
        data={
            "well": np.repeat([f"well{i}" for i in range(n_groups)], group_size),
            "gene_symbol": np.tile(["NTC", "NTC", "NTC", "a", "b", "c"], n_groups),
            _centroid_column_names[0]: rng.normal(size=n),
            _centroid_column_names[1]: rng.normal(size=n),
        },
        index=[str(i) for i in range(n)],
    )
    kwargs = dict(
        reference_query=reference,
        normalize="local-zscore",
        robust=robust,
        by=["well"],
        n_neighbors=n_neighbors,
    )
    expected = normalize_features(
        anndata.AnnData(X=x.copy(), obs=obs.copy()),
        **kwargs,
    )
    caplog.clear()
    dask_data = anndata.AnnData(
        X=da.from_array(x.copy(), chunks=(group_size * groups_per_chunk, n_features)),
        obs=obs.copy(),
    )
    with caplog.at_level("WARNING", logger="scallops"):
        result = normalize_features(dask_data, **kwargs)
    # the slow path would hide the bug this test is about
    assert "slower code" not in caplog.text
    result.X = result.X.compute()
    _compare_anndata(expected, result)


def _local_zscore_data(n_groups=4, group_size=30, n_features=5, dtype=np.float64):
    rng = np.random.default_rng(0)
    n = n_groups * group_size
    # a large offset relative to the spread is the hard case for one-pass moments
    x = (rng.normal(size=(n, n_features)) * 10 + 3000).astype(dtype)
    obs = pd.DataFrame(
        data={
            "well": np.repeat([f"well{i}" for i in range(n_groups)], group_size),
            "gene_symbol": np.resize(["NTC", "NTC", "NTC", "a", "b", "c"], n),
            _centroid_column_names[0]: rng.normal(size=n),
            _centroid_column_names[1]: rng.normal(size=n),
        },
        index=[str(i) for i in range(n)],
    )
    return anndata.AnnData(X=x, obs=obs)


@pytest.mark.parametrize("n_neighbors", [3, 4, 10])
@pytest.mark.parametrize("reference", ["gene_symbol=='NTC'", None])
@pytest.mark.parametrize("robust", [True, False])
@pytest.mark.features
def test_norm_local_zscore_fast_moments(n_neighbors, reference, robust):
    """`fast_moments` computes the same statistics a different, faster way."""
    data = _local_zscore_data()
    kwargs = dict(
        reference_query=reference,
        normalize="local-zscore",
        robust=robust,
        by=["well"],
        n_neighbors=n_neighbors,
    )
    exact = normalize_features(data, **kwargs)
    fast = normalize_features(data, fast_moments=True, **kwargs)
    if robust:
        # no sufficient-statistics form for the median, so it must fall back
        _compare_anndata(exact, fast)
    else:
        np.testing.assert_allclose(exact.X, fast.X, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n_neighbors", [3, 4])
@pytest.mark.features
def test_norm_local_zscore_robust_matches_scipy(n_neighbors):
    """The partitioned median and MAD are exact for both odd and even neighbor counts."""
    data = _local_zscore_data(n_groups=1, group_size=40)
    result = normalize_features(
        data,
        normalize="local-zscore",
        robust=True,
        n_neighbors=n_neighbors,
    )
    coordinates = data.obs[list(_centroid_column_names)].values
    neighbors = data.X[
        _nearest_neighbors_indices(coordinates, coordinates, n_neighbors)
    ]
    expected = (data.X - np.median(neighbors, axis=1)) / median_abs_deviation(
        neighbors, axis=1, scale="normal"
    )
    np.testing.assert_array_equal(result.X, expected)


@pytest.mark.parametrize("sorted_groups", [True, False])
@pytest.mark.features
def test_norm_local_zscore_feature_chunks(sorted_groups):
    """A wide matrix is split along features, so a block holds only some of them."""
    group_size = 30
    data = _local_zscore_data(group_size=group_size, n_features=40)
    kwargs = dict(normalize="local-zscore", by=["well"], n_neighbors=5)
    expected = normalize_features(data, **kwargs)

    dask_data = anndata.AnnData(
        X=da.from_array(
            data.X, chunks=((group_size if sorted_groups else 7), data.shape[1])
        ),
        obs=data.obs.copy(),
    )
    # 40 features in chunks of 8 gives five blocks per group
    result = normalize_features(dask_data, feature_chunk_size=8, **kwargs)
    assert len(result.X.chunks[1]) == 5
    result.X = result.X.compute()
    _compare_anndata(expected, result)


@pytest.mark.parametrize("chunks", [None, (30, 2), (7, 2)])
@pytest.mark.features
def test_norm_local_zscore_preserves_dtype(chunks):
    """A float32 matrix must not come back as float64, on any path."""
    data = _local_zscore_data(dtype=np.float32)
    if chunks is not None:
        data = anndata.AnnData(
            X=da.from_array(data.X, chunks=chunks), obs=data.obs.copy()
        )
    result = normalize_features(
        data, normalize="local-zscore", by=["well"], n_neighbors=3
    )
    assert result.X.dtype == np.float32
    if chunks is not None:
        # the Dask meta must agree with what the blocks actually produce
        assert result.X.compute().dtype == np.float32
