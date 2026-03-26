from collections.abc import Sequence

import anndata
import dask
import dask.array as da
import numpy as np
import xarray as xr
from array_api_compat import get_namespace
from sklearn.preprocessing import PowerTransformer

from scallops.features.util import _anndata_to_xr, _slice_anndata


def transform_features_yj(
    data: anndata.AnnData, by: str | Sequence | None = None
) -> anndata.AnnData:
    """Transform features using yeo-johnson transform

    :param data: AnnData object
    :param by: Column(s) in `data.obs` to stratify by.
    :return: Transformed AnnData object
    """

    def _transform_block(x):
        return PowerTransformer(method="yeo-johnson").fit_transform(x)

    def _transform_feature_group(x):
        d = x.data
        if isinstance(d, da.Array):
            chunks = list(d.chunksize)
            if chunks[0] != d.shape[0]:
                chunks[0] = -1
                d = d.rechunk(tuple(chunks))
            d = da.map_blocks(_transform_block, d, meta=np.array((), dtype=np.float64))
        else:
            d = _transform_block(d)
        return x.copy(data=d, deep=False)

    xdata = _anndata_to_xr(data, by)
    if by is not None:
        result = xdata.groupby(by).map(_transform_feature_group)
        return anndata.AnnData(
            X=result.data,
            obs=data.obs.loc[result.coords["obs"].values],
            var=data.var.copy(),
        )

    return anndata.AnnData(
        X=_transform_feature_group(xdata).data,
        obs=data.obs.copy(),
        var=data.var.copy(),
    )


def filter_data(
    data: anndata.AnnData,
    max_fraction_nans: float | None = 0.25,
    min_variance: float | None = 0.1,
    by: str | Sequence | None = None,
) -> anndata.AnnData:
    """Filter cells using `max_fraction_nans` then filter features using `min_variance`

    :param data: AnnData object
    :param max_fraction_nans: Keep cells with <= `max_fraction_nans` missing values
    :param min_variance: Keep features with variance >= `min_variance`
    :param by: Column(s) in `data.obs` to stratify by when computing variance. If
    provided, the median variance is used for filtering.
    :return: Filtered AnnData object
    """
    xp = get_namespace(data.X)
    keep_cells = None
    keep_features = None
    if max_fraction_nans is not None:
        nan_counts_per_cell = xp.isnan(data.X).sum(axis=1)
        max_nans = int(data.shape[1] * max_fraction_nans)
        keep_cells = nan_counts_per_cell <= max_nans
    if min_variance is not None:
        if by is not None:
            if isinstance(keep_cells, da.Array):
                keep_cells = keep_cells.compute()

            if not isinstance(by, str) and isinstance(by, Sequence):
                # xarray outputs all combinations, even ones that don't exist
                # https://github.com/pydata/xarray/issues/11264
                xdata = xr.DataArray(
                    data.X,
                    dims=("obs", "var"),
                    name="",
                    coords={"obs": data.obs[by].apply(tuple, axis=1)},
                )
                by = "obs"
            else:
                xdata = _anndata_to_xr(data, by)
            if keep_cells is not None:
                xdata = xdata[keep_cells]

            variance = xdata.groupby(by).var(skipna=False)  # dims (by, 'var')
            variance = xp.median(variance.data, axis=0)
        else:
            variance = (
                xp.var(data.X[keep_cells], axis=0)
                if keep_cells is not None
                else xp.var(data.X, axis=0)
            )
        keep_features = variance >= min_variance

    if isinstance(data.X, da.Array):
        keep_features, keep_cells = dask.compute(keep_features, keep_cells)
    return _slice_anndata(data, keep_cells, keep_features)
