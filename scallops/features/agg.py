from collections.abc import Sequence
from typing import Literal

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from array_api_compat import get_namespace
from dask.array.numpy_compat import NUMPY_GE_200
from numba import jit, prange
from pandas import MultiIndex
from scipy.stats import wasserstein_distance_nd
from statsmodels.stats.weightstats import DescrStatsW
from xarray.core.indexes import PandasMultiIndex


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_pairwise_mean_between(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mean pairwise euclidean distance between two groups (X to Y)."""
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    if n_samples_X == 0 or n_samples_Y == 0:
        return 0.0

    total_distance = 0.0
    n_pairs = n_samples_X * n_samples_Y

    for i in prange(n_samples_X):
        for j in range(n_samples_Y):
            total_distance += _euclidean_distance(X[i], Y[j])

    return total_distance / n_pairs


@jit(nopython=True, cache=True)
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute euclidean distance between two vectors."""
    dist_sq = 0.0
    for k in range(x.shape[0]):
        diff = x[k] - y[k]
        dist_sq += diff * diff
    return np.sqrt(dist_sq)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_pairwise_mean_within(X: np.ndarray) -> float:
    """Compute mean pairwise euclidean distance within a group (X to X)."""
    n_samples = X.shape[0]
    if n_samples < 2:
        return 0.0

    total_distance = 0.0
    n_pairs = n_samples * (n_samples - 1) / 2.0

    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            total_distance += _euclidean_distance(X[i], X[j])

    return total_distance / n_pairs


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_pairwise_mean_between(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mean pairwise euclidean distance between two groups (X to Y)."""
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    if n_samples_X == 0 or n_samples_Y == 0:
        return 0.0

    total_distance = 0.0
    n_pairs = n_samples_X * n_samples_Y

    for i in prange(n_samples_X):
        for j in range(n_samples_Y):
            total_distance += _euclidean_distance(X[i], Y[j])

    return total_distance / n_pairs


def _weighted_median(x, weights):
    d = DescrStatsW(data=x, weights=weights).quantile(probs=0.5, return_pandas=False)
    return d


def _compute_weights_one_perturbation(
    x: np.ndarray, group_values: np.ndarray, metric: Literal["wasserstein", "energy"]
) -> pd.DataFrame:
    df = pd.DataFrame(index=group_values)
    indices = df.groupby(df.index).indices
    keys = list(indices.keys())
    if len(keys) <= 2:
        return pd.DataFrame(index=keys, data=dict(weight=1.0))

    distances = np.zeros((len(keys), len(keys)))
    if metric == "energy":
        within = np.zeros(len(keys))
        for i in range(len(keys)):
            within = _euclidean_pairwise_mean_within(x[indices[keys[i]]])
    for i in range(len(keys)):
        X = x[indices[keys[i]]]
        for j in range(i):
            Y = x[indices[keys[j]]]
            if metric == "energy":
                between = _euclidean_pairwise_mean_between(X, Y)
                dist = 2 * between - within[i] - within[j]
            else:
                dist = wasserstein_distance_nd(X, Y)
            distances[i, j] = dist
            distances[j, i] = dist
    distances = 1 / (1 + distances)
    np.fill_diagonal(distances, 0)
    distances = distances.mean(axis=0)
    distances = distances / distances.sum()
    return df.join(pd.DataFrame(index=keys, data=dict(weight=distances)))["weight"]


def _weighted_distance(
    x: np.ndarray,
    group_values: np.ndarray,
    agg_func: Literal["mean", "median"],
    metric: Literal["wasserstein", "energy"],
):
    weights = _compute_weights_one_perturbation(
        x=x, group_values=group_values, metric=metric
    )["weight"].values
    if agg_func == "mean":
        x = np.average(x, weights=weights, axis=0)
    else:
        x = np.quantile(x, weights=weights, q=0.5, axis=0, method="inverted_cdf")
    return x


def _weighted_agg(
    x: xr.DataArray,
    weights_col: str,
    agg_func: Literal["mean", "median"],
    metric: Literal["wasserstein", "energy"] | None,
):
    weights = x.coords[weights_col].values
    x = x.data
    xp = get_namespace(x)

    if metric in ("wasserstein", "energy"):
        if isinstance(x, da.Array):
            chunks = list(x.chunksize)
            if chunks[0] != x.shape[0]:
                chunks[0] = -1
                x = x.rechunk(tuple(chunks))
            x = da.map_blocks(
                _weighted_distance,
                x,
                group_values=weights,
                agg_func=agg_func,
                metric=metric,
                meta=np.array((), dtype=np.float64),
                drop_axis=0,
            )
        else:
            x = _weighted_distance(
                x, group_values=weights, agg_func=agg_func, metric=metric
            )
    elif agg_func == "mean":
        x = xp.average(x, weights=weights, axis=0)
    elif agg_func == "median":
        if isinstance(x, da.Array):
            chunks = list(x.chunksize)
            if chunks[0] != x.shape[0]:
                chunks[0] = -1
                x = x.rechunk(tuple(chunks))
        if NUMPY_GE_200 and not isinstance(x, da.Array):
            # np.quantile weights parameter added in numpy 2
            # https://github.com/dask/dask/issues/12322
            x = xp.quantile(x, weights=weights, q=0.5, axis=0, method="inverted_cdf")
        else:
            if isinstance(x, da.Array):
                x = da.map_blocks(
                    _weighted_median,
                    x,
                    weights=weights,
                    meta=np.array((), dtype=np.int64),
                    drop_axis=0,
                )
            else:
                x = _weighted_median(x, weights=weights).squeeze()

    return xr.DataArray(x, dims=("var",), name="")


def _get_obs_with_counts(data: xr.DataArray, grouped, by: str | Sequence[str]):
    counts = []
    groups = []
    for group in grouped.groups:
        val = grouped.groups[group]
        if isinstance(val, slice):
            count = (
                val.stop - val.start
                if val.step is None
                else len(val.indices(data.shape[0]))
            )
        else:
            count = len(val)
        groups.append(group)
        counts.append(count)

    obs = data.coords["obs"].to_dataframe()
    group_counts = pd.DataFrame(
        data={"count": counts},
        index=pd.MultiIndex.from_tuples(groups, names=obs.index.names)
        if isinstance(obs.index, MultiIndex)
        else pd.Index(groups),
    )
    obs = (
        obs.drop("obs", errors="ignore", axis=1)
        .join(group_counts, rsuffix="_1")
        .reset_index()
    )
    group_by_multi = not isinstance(by, str) and isinstance(by, Sequence)
    if not group_by_multi and "obs" in obs.columns:
        obs = obs.rename({"obs": by}, axis=1)
    return obs.set_index(pd.RangeIndex(len(obs)).astype(str))


def agg_features(
    data: anndata.AnnData,
    by: str | Sequence[str],
    weights_col: str | None = None,
    agg_func: Literal["mean", "median"] = "mean",
    metric: Literal["wasserstein", "energy"] | None = None,
):
    """Aggregate features

    :param data: Annotated data matrix.
    :param by: Perturbation column(s) in `data.obs` to aggregate by.
    :param weights_col: If provided, perform weighted aggregation using provided
     weights if `metric` is None, or compute weights using specified metric within
    `weights_col` (e.g. barcode).
    :param agg_func: Aggregation method.
    :param metric: Distance method to use to compute weights within each feature
    :return: Aggregated data
    """
    assert agg_func in ("mean", "median")
    if metric is not None and weights_col is None:
        raise ValueError("Please provide `weights_col`")
    group_by_multi = not isinstance(by, str) and isinstance(by, Sequence)
    if not group_by_multi:
        coords = {"obs": data.obs[by]}
    else:
        coords = {"obs": np.arange(data.shape[0])}
        for col in by:
            coords[col] = ("obs", data.obs[col])
    if weights_col is not None:
        coords[weights_col] = ("obs", data.obs[weights_col])
    xdata = xr.DataArray(data=data.X, dims=("obs", "var"), coords=coords, name="")
    if group_by_multi:
        xdata = xdata.set_xindex(by, PandasMultiIndex)
    if isinstance(xdata.data, da.Array) and (
        agg_func == "median" or metric is not None
    ):
        xdata = xdata.groupby("obs").shuffle_to_chunks()

    grouped = xdata.groupby("obs")

    if weights_col is not None:
        result = grouped.map(
            _weighted_agg, weights_col=weights_col, agg_func=agg_func, metric=metric
        )
    else:
        result = grouped.mean() if agg_func == "mean" else grouped.median()
    X = result.data
    obs = _get_obs_with_counts(result, grouped, by)
    return anndata.AnnData(
        X=X,
        obs=obs,
        var=data.var.copy(),
    )
