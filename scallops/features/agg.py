from typing import Literal

import anndata
import pandas as pd
import xarray as xr


def agg_features(
    data: anndata.AnnData,
    perturbation_column: str,
    agg_func: Literal["mean", "median"] = "mean",
):
    """Aggregate features

    :param data: Annotated data matrix.
    :param perturbation_column: Column in `data.obs` containing perturbation.
    :param agg_func: Aggregation method.
    :return: Aggregated data
    """
    assert agg_func in ["mean", "median"]
    coords = dict(obs=data.obs[perturbation_column])
    x = xr.DataArray(data.X, dims=["obs", "var"], coords=coords, name="")
    grouped = x.groupby("obs")
    result = grouped.mean() if agg_func == "mean" else grouped.median()
    result_obs = pd.DataFrame(index=result.coords["obs"].values)
    return anndata.AnnData(
        X=result.data,
        obs=result_obs,
        var=data.var.copy(),
    )
