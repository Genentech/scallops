import math
from collections.abc import Sequence
from typing import Literal

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from pandas import MultiIndex
from xarray.core.indexes import PandasMultiIndex


def agg_features(
    data: anndata.AnnData,
    by: str | Sequence[str],
    agg_func: Literal["mean", "median"] = "mean",
):
    """Aggregate features

    :param data: Annotated data matrix.
    :param by: Perturbation column(s) in `data.obs` to aggregate by.
    :param agg_func: Aggregation method.
    :return: Aggregated data
    """
    assert agg_func in ("mean", "median")

    group_by_multi = not isinstance(by, str) and isinstance(by, Sequence)
    if not group_by_multi:
        coords = {"obs": data.obs[by]}
    else:
        coords = {"obs": np.arange(data.shape[0])}
        for col in by:
            coords[col] = ("obs", data.obs[col])
    xdata = xr.DataArray(data=data.X, dims=("obs", "var"), coords=coords, name="")
    if group_by_multi:
        xdata = xdata.set_xindex(by, PandasMultiIndex)
    grouped = xdata.groupby("obs")
    result = grouped.mean() if agg_func == "mean" else grouped.median()
    X = result.data
    group_counts = []
    for group in grouped.groups:
        val = grouped.groups[group]
        if isinstance(val, slice):
            count = val.stop - val.start
            if val.step is not None:
                count = math.ceil(count / val.step)
        else:
            count = len(val)
        group_counts.append((group, count))
    obs = result.coords["obs"].to_dataframe()
    group_counts = pd.DataFrame(group_counts, columns=["obs", "count"]).set_index("obs")
    if isinstance(obs.index, MultiIndex):
        obs.index = obs.index.to_flat_index()
    obs = obs.drop("obs", axis=1).join(group_counts).reset_index()
    if not group_by_multi:
        obs = obs.rename({"obs": by}, axis=1)
    obs = obs.set_index(pd.RangeIndex(len(obs)).astype(str))
    return anndata.AnnData(
        X=X,
        obs=obs,
        var=data.var.copy(),
    )
