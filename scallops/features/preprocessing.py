from collections.abc import Sequence

import anndata
import dask
import dask.array as da
from array_api_compat import get_namespace
from sklearn.preprocessing import PowerTransformer

from scallops.features.util import _anndata_to_xr, _slice_anndata


def transform_features_yj(
    adata: anndata.AnnData, by: str | Sequence | None = None
) -> anndata.AnnData:
    """Transform features using yeo-johnson transform

    :param adata: AnnData object
    :param by: Column(s) in `adata.obs` to stratify by.
    :return: Transformed AnnData object
    """

    def _transform_feature_block(x):
        return PowerTransformer(method="yeo-johnson").fit_transform(x)

    def _transform_feature_group(x):
        if isinstance(x.data, da.Array):
            chunks = list(x.data.chunksize)
            if chunks[0] != x.data.shape[0]:
                chunks[0] = -1
                chunks = tuple(chunks)
                x.data = x.data.rechunk(chunks)
            transformed_x = da.map_blocks(_transform_feature_block, x.data)
        else:
            transformed_x = _transform_feature_block(x.data)
        return x.copy(data=transformed_x, deep=False)

    xdata = _anndata_to_xr(adata, by)
    if by is not None:
        xdata = xdata.groupby(by).shuffle_to_chunks()
        result = xdata.groupby(by).map(_transform_feature_group)
        return anndata.AnnData(
            X=result.data,
            obs=adata.obs.loc[result.coords["obs"].values],
            var=adata.var.copy(),
        )
    return anndata.AnnData(
        X=_transform_feature_group(xdata).data,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )


def filter_data(
    adata: anndata.AnnData,
    max_fraction_nans: float | None = 0.25,
    min_variance: float | None = 0.1,
) -> anndata.AnnData:
    """Filter cells using `max_fraction_nans` then filter features using `min_variance`

    :param adata: AnnData object
    :param max_fraction_nans: Keep cells with <= `max_fraction_nans` missing values
    :param min_variance: Keep features with variance >= `min_variance`
    :return: Filtered AnnData object
    """
    xp = get_namespace(adata.X)
    keep_cells = None
    keep_features = None
    if max_fraction_nans is not None:
        nan_counts_per_cell = xp.isnan(adata.X).sum(axis=1)
        max_nans = int(adata.shape[1] * max_fraction_nans)
        keep_cells = nan_counts_per_cell <= max_nans
    if min_variance is not None:
        variance = (
            xp.var(adata.X[keep_cells], axis=0)
            if keep_cells is not None
            else xp.var(adata.X, axis=0)
        )
        keep_features = variance >= min_variance
    if isinstance(adata.X, da.Array):
        keep_features, keep_cells = dask.compute(keep_features, keep_cells)
    return _slice_anndata(adata, keep_cells, keep_features)
