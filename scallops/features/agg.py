from collections.abc import Sequence
from typing import Literal

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from array_api_compat import get_namespace
from dask.array.numpy_compat import NUMPY_GE_200
from statsmodels.stats.weightstats import DescrStatsW


def _weighted_median(x, weights):
    d = DescrStatsW(data=x, weights=weights).quantile(probs=0.5, return_pandas=False)
    return d


def agg_features(
    data: anndata.AnnData,
    by: str | Sequence[str],
    weights_col: str | None = None,
    agg_func: Literal["mean", "median"] = "mean",
):
    """Aggregate features

    :param data: Annotated data matrix.
    :param by: Perturbation column(s) in `data.obs` to aggregate by.
    :param weights_col: If provided, perform weighted aggregation
    :param agg_func: Aggregation method.
    :return: Aggregated data
    """
    assert agg_func in ("mean", "median")

    group_by_multi = not isinstance(by, str) and isinstance(by, Sequence)
    if not group_by_multi:
        by = list(by)
        if len(by) == 1:
            by = by[0]
            group_by_multi = False

    if not group_by_multi:
        coords = {"obs": data.obs[by]}
    else:
        coords = {"obs": data.obs[by].apply(tuple, axis=1)}
    if weights_col is not None:
        coords[weights_col] = ("obs", data.obs[weights_col])
    xdata = xr.DataArray(data=data.X, dims=("obs", "var"), coords=coords, name="")

    grouped = xdata.groupby("obs")
    xp = get_namespace(xdata.data)
    if weights_col is not None:

        def weighted_agg(x):
            weights = x.coords[weights_col].values
            x = x.data

            if agg_func == "mean":
                x = xp.average(x, weights=weights, axis=0)
            else:
                if isinstance(x, da.Array):
                    chunks = list(x.chunksize)
                    if chunks[0] != x.shape[0]:
                        chunks[0] = -1
                        x = x.rechunk(tuple(chunks))
                if NUMPY_GE_200 and not isinstance(x, da.Array):
                    # np.quantile weights parameter added in numpy 2
                    # https://github.com/dask/dask/issues/12322
                    x = xp.quantile(
                        x, weights=weights, q=0.5, axis=0, method="inverted_cdf"
                    )
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

        result = grouped.map(weighted_agg)
    else:
        result = grouped.mean() if agg_func == "mean" else grouped.median()
    X = result.data
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

    counts = []
    groups = []
    for group in grouped.groups:
        val = grouped.groups[group]
        if isinstance(val, slice):
            count = (
                val.stop - val.start
                if val.step is None
                else len(val.indices(X.shape[0]))
            )
        else:
            count = len(val)
        groups.append(group)
        counts.append(count)

    obs = result.coords["obs"].to_dataframe()
    group_counts = pd.DataFrame(
        data={"count": counts},
        index=groups,
    )
    obs = obs.join(group_counts, rsuffix="_1").reset_index(drop=True)
    if group_by_multi:
        new_obs = pd.DataFrame(obs["obs"].tolist(), columns=by)
        for c in obs.columns:
            if c.startswith("count") and c not in new_obs.columns:
                new_obs[c] = obs[c]
        obs = new_obs
    else:
        obs = obs.rename({"obs": by}, axis=1)
    obs = obs.set_index(pd.RangeIndex(len(obs)).astype(str))
    return anndata.AnnData(
        X=X,
        obs=obs,
        var=data.var.copy(),
    )
