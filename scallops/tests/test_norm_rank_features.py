import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from distributed import Client, LocalCluster
from scipy.stats import median_abs_deviation

from scallops.features.agg import agg_features
from scallops.features.constants import _centroid_column_names
from scallops.features.normalize import (
    _nearest_neighbors_indices,
    normalize_features,
    typical_variation_normalization,
)
from scallops.features.rank import rank_features
from scallops.features.util import _slice_anndata, pandas_to_anndata


@pytest.fixture(scope="module")
def client():
    # Start a local Dask cluster for the entire test module
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    yield client  # Provide the client to the tests
    client.close()
    cluster.close()


@pytest.fixture(params=[["plate", "well"], None])
def normalize_groups(request):
    return request.param


@pytest.fixture(params=[True, False])
def robust(request):
    return request.param


@pytest.fixture(params=["zscore", "local-zscore"])
def normalize(request):
    return request.param


@pytest.fixture(params=["gene_symbol=='NTC'", None])
def reference(request):
    return request.param


@pytest.fixture
def data():
    df = pd.DataFrame(
        data=dict(
            label=np.arange(6),
            Cells_Intensity_feature_1=[1, 2, 4, 8, 16, 32],
            Cells_Intensity_feature_2=[10, 20, 40, 80, 160, 320],
            gene_symbol=["a", "NTC", "a", "NTC", "a", "NTC"],
            well=["a", "a", "a", "b", "b", "b"],
            plate=["a", "a", "a", "b", "b", "b"],
            Nuclei_AreaShape_Center_Y=[1, 7, 12, 16, 19, 21],
            Nuclei_AreaShape_Center_X=[1, 7, 12, 16, 19, 21],
        ),
    )
    return pandas_to_anndata(
        df, ["Cells_Intensity_feature_1", "Cells_Intensity_feature_2"]
    )


def _diff_values(ds, normed_data, normalize, robust, reference, scaling, n_neighbors):
    values = ds.X.copy()
    ref_ds = ds
    if reference is not None:
        ref_ds = _slice_anndata(ds, ds.obs.query(reference))

    ref_values = ref_ds.X
    if normalize == "zscore":
        mean = np.median(ref_values, axis=0) if robust else np.mean(ref_values, axis=0)
        std = (
            median_abs_deviation(ref_values, axis=0, scale="normal")
            if robust
            else np.std(ref_values, axis=0)
        )
    else:
        query_centroids = np.stack(
            (
                ds.obs[_centroid_column_names[0]].values,
                ds.obs[_centroid_column_names[1]].values,
            ),
            axis=1,
        )

        ref_centroids = np.stack(
            (
                ref_ds.obs[_centroid_column_names[0]].values,
                ref_ds.obs[_centroid_column_names[1]].values,
            ),
            axis=1,
        )

        indices = _nearest_neighbors_indices(
            ref_centroids, query_centroids, n_neighbors=n_neighbors
        )
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
    np.testing.assert_array_equal(
        values,
        normed_data.X,
        err_msg="Not equal",
    )


@pytest.mark.features
def test_norm_features(client, data, normalize, normalize_groups, robust, reference):
    n_neighbors = 2 if normalize_groups is None else 1
    scaling = n_neighbors > 1
    normed_data = normalize_features(
        data,
        reference_query=reference,
        normalize=normalize,
        robust=robust,
        normalize_groups=normalize_groups,
        n_neighbors=n_neighbors,
        scaling=scaling,
    )

    dask_data = anndata.AnnData(
        X=da.from_array(data.X, chunks=(1, 2)), obs=data.obs, var=data.var
    )

    normed_data_dask = normalize_features(
        dask_data,
        reference_query=reference,
        normalize=normalize,
        robust=robust,
        normalize_groups=normalize_groups,
        n_neighbors=n_neighbors,
        scaling=scaling,
    )
    normed_data_dask.X = normed_data_dask.X.compute()
    normed_data = _slice_anndata(normed_data, normed_data.obs.sort_values("label"))
    normed_data_dask = _slice_anndata(
        normed_data_dask, normed_data_dask.obs.sort_values("label")
    )
    np.testing.assert_array_equal(normed_data.X, normed_data_dask.X)
    pd.testing.assert_frame_equal(normed_data.obs, normed_data_dask.obs)
    pd.testing.assert_frame_equal(normed_data.var, normed_data_dask.var)
    if normalize_groups is not None:
        indices = data.obs.groupby(normalize_groups).indices
        for name in indices:
            query = []
            for i in range(len(normalize_groups)):
                query.append(f"{normalize_groups[i]}=='{name[i]}'")

            _diff_values(
                _slice_anndata(data, indices[name]),
                _slice_anndata(normed_data, normed_data.obs.query("&".join(query))),
                normalize,
                robust,
                reference,
                scaling,
                n_neighbors,
            )
    else:
        _diff_values(
            data, normed_data, normalize, robust, reference, scaling, n_neighbors
        )


@pytest.fixture(params=[None, ["well"]])
def rank_groups(request):
    return request.param


@pytest.mark.features
def testrank_features(client, data, rank_groups):
    reference_value = "NTC"
    perturbation_column = "gene_symbol"
    method = "welch_t"
    min_labels = 0

    rank_results = rank_features(
        data,
        rank_groups=rank_groups,
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
        rank_groups=rank_groups,
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


@pytest.mark.features
def test_anndata_slice():
    d = anndata.AnnData(
        X=np.arange(4).reshape((2, 2)),
        obs=pd.DataFrame(index=["1", "2"]),
        var=pd.DataFrame(index=["1", "2"]),
    )
    data1 = d[[1, 0], [1, 0]]
    data2 = _slice_anndata(d, [1, 0], [1, 0])
    np.testing.assert_array_equal(data1.X, data2.X)
    pd.testing.assert_frame_equal(data1.obs, data2.obs)
    pd.testing.assert_frame_equal(data1.var, data2.var)

    data1 = d[d.obs.index.isin(["2"]), d.var.index.isin(["1"])]
    data2 = _slice_anndata(d, d.obs.index.isin(["2"]), d.var.index.isin(["1"]))
    np.testing.assert_array_equal(data1.X, data2.X)
    pd.testing.assert_frame_equal(data1.obs, data2.obs)
    pd.testing.assert_frame_equal(data1.var, data2.var)


@pytest.mark.features
def test_agg_features():
    d = anndata.AnnData(
        X=np.arange(8).reshape((4, 2)),
        obs=pd.DataFrame(data=dict(pert=["1", "2", "1", "2"])),
        var=pd.DataFrame(index=["1", "2"]),
    )
    agg_d = agg_features(d, "pert")
    assert agg_d.shape == (2, 2)
    agg_d = agg_d[["1", "2"]]
    np.testing.assert_array_equal(agg_d.X, np.array([[2.0, 3.0], [4.0, 5.0]]))


@pytest.mark.features
def test_typical_variation_normalization():
    d = anndata.AnnData(
        X=np.arange(64).reshape((32, 2)),
        obs=pd.DataFrame(
            data=dict(pert=["1", "2"] * 16, batch=["1", "2", "2", "1"] * 8)
        ),
    )
    # from efaar_benchmarking.efaar import tvn_on_controls

    # ref_tvn = tvn_on_controls(
    #     embeddings=d.X,
    #     metadata=d.obs,
    #     pert_col="pert",
    #     control_key="1",
    #     batch_col=None)
    # ref_tvn_batch = tvn_on_controls(
    #     embeddings=d.X,
    #     metadata=d.obs,
    #     pert_col="pert",
    #     control_key="1",
    #     batch_col="batch")
    ref_tvn = np.array(
        [
            [-1.62697843e00, -1.64249999e00],
            [-1.51851320e00, -1.69953762e00],
            [-1.41004798e00, -1.36437737e00],
            [-1.30158275e00, -1.17409425e00],
            [-1.19311752e00, -1.08625474e00],
            [-1.08465229e00, -1.14329237e00],
            [-9.76187060e-01, -1.05545286e00],
            [-8.67721831e-01, -8.65169749e-01],
            [-7.59256602e-01, -8.49768810e-01],
            [-6.50791373e-01, -5.87047125e-01],
            [-5.42326145e-01, -5.71646186e-01],
            [-4.33860916e-01, -4.32584874e-01],
            [-3.25395687e-01, -2.93523562e-01],
            [-2.16930458e-01, -2.16292437e-01],
            [-1.08465229e-01, -1.08146219e-01],
            [-1.00511831e-16, -5.83485025e-17],
            [1.08465229e-01, 1.08146219e-01],
            [2.16930458e-01, 2.16292437e-01],
            [3.25395687e-01, 2.93523562e-01],
            [4.33860916e-01, 4.32584874e-01],
            [5.42326145e-01, 5.71646186e-01],
            [6.50791373e-01, 5.87047125e-01],
            [7.59256602e-01, 8.49768810e-01],
            [8.67721831e-01, 8.65169749e-01],
            [9.76187060e-01, 1.05545286e00],
            [1.08465229e00, 1.14329237e00],
            [1.19311752e00, 1.08625474e00],
            [1.30158275e00, 1.17409425e00],
            [1.41004798e00, 1.36437737e00],
            [1.51851320e00, 1.69953762e00],
            [1.62697843e00, 1.64249999e00],
            [1.73544366e00, 1.73033950e00],
        ]
    )
    ref_tvn_batch = np.array(
        [
            [-1.48481205, -1.51048701],
            [-1.58893347, -1.75679769],
            [-1.48599043, -1.42626166],
            [-1.16853876, -1.05095595],
            [-1.06216772, -0.96543912],
            [-1.16628913, -1.21174981],
            [-1.05991808, -1.12623297],
            [-0.74246641, -0.75092727],
            [-0.63509132, -0.7371749],
            [-0.74364479, -0.66670192],
            [-0.6362697, -0.65294955],
            [-0.31810807, -0.32838899],
            [-0.21244699, -0.19212702],
            [-0.31842945, -0.30541844],
            [-0.21233986, -0.19978387],
            [0.10625027, 0.0941493],
            [0.21233986, 0.19978387],
            [0.1059289, 0.11711985],
            [0.21244699, 0.19212702],
            [0.53060862, 0.51668758],
            [0.6362697, 0.65294955],
            [0.53114425, 0.47840333],
            [0.63509132, 0.7371749],
            [0.95496696, 0.93922586],
            [1.05991808, 1.12623297],
            [0.95378858, 1.02345121],
            [1.06216772, 0.96543912],
            [1.38103931, 1.23925455],
            [1.48599043, 1.42626166],
            [1.37643292, 1.5684991],
            [1.48481205, 1.51048701],
            [1.80368365, 1.78430243],
        ]
    )

    result = typical_variation_normalization(d, "pert=='1'")
    result_batch = typical_variation_normalization(d, "pert=='1'", "batch")

    np.testing.assert_allclose(
        ref_tvn,
        result.X,
        rtol=3.16,
        atol=0.18,
    )
    np.testing.assert_allclose(
        result_batch.X,
        ref_tvn_batch,
        rtol=3.16,
        atol=0.18,
    )
