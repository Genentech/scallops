import logging
from collections.abc import Sequence
from typing import Literal

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from anndata._core.index import _normalize_index
from array_api_compat import get_namespace
from flox import rechunk_for_blockwise
from flox.lib import _issorted
from scipy.linalg import fractional_matrix_power
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scallops.features.util import (
    _get_names_from_pd_query,
    _slice_anndata,
    _trim_by,
    _xarray_by_values,
)
from scallops.utils import tqdm_func

logger = logging.getLogger("scallops")


def _convert_scale(mad_scale):
    if isinstance(mad_scale, str):
        if mad_scale.lower() == "normal":
            mad_scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            raise ValueError(f"{mad_scale} is not a valid mad_scale value.")
    return mad_scale


def normalize_features(
    data: anndata.AnnData,
    reference_query: str | None = None,
    by: Sequence[str] | str | None = None,
    normalize: Literal["zscore", "local-zscore"] = "zscore",
    n_neighbors: int = 100,
    neighbors_metric: str = "minkowski",
    robust: bool = False,
    mad_scale: float | str = "normal",
    max_value: float | None = None,
    centering: bool = True,
    scaling: bool = True,
    batch_size: int | None = 25000,
    centroid_column_names: tuple[str, str] = (
        "Nuclei_AreaShape_Center_Y",
        "Nuclei_AreaShape_Center_X",
    ),
) -> anndata.AnnData:
    """Normalize features

    :param data: Annotated data matrix.
    :param reference_query: Query to extract reference observations
        (e.g. "gene_symbol=='NTC'")
    :param by: Column(s) in `data.obs` to stratify by.
    :param normalize: Normalization method to use where `local` uses nearest neighbors by location.
    :param n_neighbors: Number of neighbors for local and nearest neighbor zscore.
    :param neighbors_metric: Nearest neighbor metric to use when normalize is
        `local-zscore`.
    :param robust: Use robust statistics.
    :param mad_scale: Numerical scale factor to divide median absolute deviation. The
        string “normal” is also accepted, and results in scale being the inverse of the
        standard normal quantile function at 0.75
    :param centering: Whether to center the data before scaling.
    :param max_value: Truncate to this value after scaling
    :param scaling: Whether to scale the data by dividing by the standard deviation.
    :param batch_size: Batch size to use for local z-score scaling to conserve memory.
    :param centroid_column_names: Columns for y and x centroids to use for local zscore.
    :return: Normalized data
    """
    assert normalize in ["zscore", "local-zscore"]
    mad_scale = _convert_scale(mad_scale)
    centroid_column_names = list(centroid_column_names)
    is_dask = isinstance(data.X, da.Array)
    use_map_blocks = False
    if by is not None:
        by = _trim_by(by)
        by_values = _xarray_by_values(data, by)
        series = pd.Series(by_values, dtype="category")
        use_map_blocks = (
            is_dask and len(data.X.chunks[0]) > 1 and _issorted(series.cat.codes.values)
        )
        if normalize != "zscore":
            group_indices = series.groupby(
                series, observed=True, sort=False, dropna=False
            ).indices
    else:
        group_indices = {None: None}
    if normalize == "zscore":
        coords = {}
        if by is not None:
            coords["obs"] = by_values
        xdata = xr.DataArray(data.X, dims=["obs", "var"], coords=coords)
        x_ref_data = xdata
        if reference_query is not None:
            refererence_data = _slice_anndata(
                data, data.obs.query(reference_query).index
            )
            coords = dict()
            if by is not None:
                coords["obs"] = _xarray_by_values(refererence_data, by)
            x_ref_data = xr.DataArray(
                refererence_data.X,
                dims=["obs", "var"],
                coords=coords,
            )
        kwargs = dict()
        if by is not None:
            grouped_ref = x_ref_data.groupby("obs")
            grouped_values = (
                xdata.groupby("obs") if reference_query is not None else grouped_ref
            )
        else:
            kwargs["dim"] = "obs"
            grouped_ref = x_ref_data
            grouped_values = xdata

        means = None
        stds = None
        xp = get_namespace(data.X)
        if robust:
            if centering:
                means = grouped_ref.median(**kwargs)

            if scaling:
                if by is not None:
                    results = []

                    for key, group in grouped_ref:
                        value = median_abs_deviation(
                            group.data, axis=0, scale=mad_scale
                        )
                        value = xp.expand_dims(value, axis=0)
                        key_values = np.empty(
                            1,
                            dtype=object if isinstance(key, tuple) else by_values.dtype,
                        )
                        key_values[0] = key
                        coords = dict(obs=key_values)
                        results.append(
                            xr.DataArray(
                                value,
                                dims=("obs", "var"),
                                coords=coords,
                                name="",
                            )
                        )
                    stds = xr.concat(results, dim="obs")
                else:
                    stds = median_abs_deviation(grouped_ref, axis=0, scale=mad_scale)

        else:
            if centering:
                means = grouped_ref.mean(**kwargs)
            if scaling:
                stds = grouped_ref.std(**kwargs)
        if by is not None:
            results = []
            indices = []
            for key, group in grouped_values:
                values = group.data
                if centering:
                    values = values - means.sel(obs=key).data
                if scaling:
                    values = values / stds.sel(obs=key).data

                    if max_value is not None:
                        values = xp.clip(values, -max_value, max_value)
                results.append(values)
                indices.append(grouped_values.groups[key])
            results = xp.concatenate(results)
            indices = np.concatenate(indices)
            obsm = dict()
            for key in data.obsm.keys():
                obsm[key] = data.obsm[key][indices]
            return anndata.AnnData(
                X=results,
                obs=data.obs.iloc[indices],
                var=data.var.copy(),
                uns=data.uns.copy(),
                obsm=obsm,
                varm=data.varm.copy(),
            )
        else:
            if centering:
                grouped_values = grouped_values - means
            if scaling:
                grouped_values = grouped_values / stds
            if max_value is not None:
                grouped_values = grouped_values.clip(-max_value, max_value)
            return anndata.AnnData(
                X=grouped_values.data,
                obs=data.obs.copy(),
                var=data.var.copy(),
                uns=data.uns.copy(),
                obsm=data.obsm.copy(),
                varm=data.varm.copy(),
            )
    indices = [] if use_map_blocks else None
    results = [] if not use_map_blocks else None
    obs_list = [] if not use_map_blocks else None
    for key in group_indices.keys():
        if by is not None:
            group_indices_ = group_indices[key]
            if not use_map_blocks:
                array_subset = group_indices_
                if np.all(np.diff(group_indices_) == 1):
                    array_subset = slice(group_indices_[0], group_indices_[-1] + 1)
                x = data.X[array_subset]
            df = data.obs.iloc[group_indices_]
        else:
            if not use_map_blocks:
                x = data.X
            df = data.obs

        local_reference_indices = None
        if reference_query is not None:
            local_reference_indices = _normalize_index(
                df.query(reference_query).index, df.index
            )

        if normalize == "local-zscore":
            query_coordinates = df[centroid_column_names].values
            reference_coordinates = (
                df.iloc[local_reference_indices][centroid_column_names].values
                if local_reference_indices is not None
                else query_coordinates
            )
            reference_indices = _nearest_neighbors_indices(
                query=query_coordinates,
                reference=reference_coordinates,
                n_neighbors=n_neighbors,
                metric=neighbors_metric,
            )
            if use_map_blocks:
                if local_reference_indices is not None:
                    reference_indices = local_reference_indices[reference_indices]
                indices.append(reference_indices)
            else:
                if local_reference_indices is not None:
                    reference_indices = local_reference_indices[reference_indices]
                if is_dask:
                    reference_indices = da.from_array(reference_indices, chunks=-1)

                # memory = (x.shape[0] * x.shape[1] * n_neighbors) / batch_size + (x.shape[0] * x.shape[1])
                # memory *= 8
                result = _local_z_batched(
                    x=x,
                    reference_indices=reference_indices,  # indices into x
                    robust=robust,
                    mad_scale=mad_scale,
                    centering=centering,
                    scaling=scaling,
                    max_value=max_value,
                    batch_size=batch_size,
                )

        # else:
        #     if use_map_blocks:
        #         if global_reference_indices is not None:
        #             indices.append(global_reference_indices)
        #     else:
        #         if is_dask and local_reference_indices is not None:
        #             local_reference_indices = da.from_array(local_reference_indices)
        #
        #         result = _normalize_features_array(
        #             values=x,
        #             reference_indices=local_reference_indices,
        #             robust=robust,
        #             mad_scale=mad_scale,
        #             centering=centering,
        #             scaling=scaling,
        #             max_value=max_value,
        #             local_zscore=False,
        #         )

        if not use_map_blocks:
            results.append(result)
            obs_list.append(df)

    if use_map_blocks:
        rechunked_data = rechunk_for_blockwise(data.X, 0, series.cat.codes.values)[1]
        indices = np.concatenate(indices, axis=0)
        chunks = [(rechunked_data.chunks[0])]
        for s in indices.shape[1:]:
            chunks.append((s,))

        indices = da.from_array(indices, chunks=tuple(chunks))
        assert indices.shape[0] == rechunked_data.shape[0]
        kwargs = dict(
            robust=robust,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
            max_value=max_value,
        )

        result = da.map_blocks(
            _local_z_batched, rechunked_data, indices, **kwargs, dtype=np.float64
        )
        return anndata.AnnData(
            X=result,
            obs=data.obs.copy(),
            var=data.var.copy(),
            uns=data.uns.copy(),
            obsm=data.obsm.copy(),
            varm=data.varm.copy(),
        )
    return anndata.AnnData(
        X=get_namespace(data.X).vstack(results),
        obs=pd.concat(obs_list),
        var=data.var.copy(),
        uns=data.uns.copy(),
        obsm=data.obsm.copy(),
        varm=data.varm.copy(),
    )


def _local_z_batched(
    x: np.ndarray | da.Array,
    reference_indices: np.ndarray | da.Array,
    scaling: bool = True,
    centering: bool = True,
    max_value: float | None = None,
    mad_scale: float | str = "normal",
    robust: bool = False,
    batch_size: int | None = None,
    progress: bool | str = False,
):
    if isinstance(x, da.Array):
        batch_size = None

    if batch_size is None:
        batch_size = x.shape[0]
    result_arrays = []

    tqdm, progress_args = tqdm_func(progress)
    for batch in tqdm(range(0, x.shape[0], batch_size), **progress_args):
        sl = slice(batch, batch + batch_size)
        reference_indices_ = reference_indices[sl]
        x_ = x[sl]
        if isinstance(x, da.Array):
            n_labels = reference_indices_.shape[0]
            n_neighbors = reference_indices_.shape[1]
            reference_indices_ = reference_indices_.flatten()
            if not isinstance(reference_indices_, da.Array):
                reference_indices_ = da.from_array(reference_indices_)
            # (labels,neighbors,features)
            reference_data_ = x[reference_indices_].reshape((n_labels, n_neighbors, -1))
        else:
            reference_data_ = x[reference_indices_]
        result = _normalize_features_array(
            values=x_,
            # reference_indices=reference_data_,
            reference_values=reference_data_,
            robust=robust,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
            max_value=max_value,
            local_zscore=reference_indices is not None,
        )
        result_arrays.append(result)
    return (
        get_namespace(x).vstack(result_arrays)
        if len(result_arrays) > 1
        else result_arrays[0]
    )


def _normalize_features_array(
    values: np.ndarray | da.Array,
    reference_indices: np.ndarray | da.Array | None = None,
    reference_values: np.ndarray | da.Array | None = None,
    scaling: bool = True,
    centering: bool = True,
    max_value: float | None = None,
    local_zscore: bool = False,
    mad_scale: float | str = "normal",
    robust: bool = False,
):
    mad_scale = _convert_scale(mad_scale) if robust else None
    xp = get_namespace(values)
    if reference_values is None:
        reference_values = (
            values if reference_indices is None else values[reference_indices]
        )
    means = None
    stds = None
    if not local_zscore:
        if robust:
            if centering:
                means = xp.nanmedian(reference_values, axis=0)
            if scaling:
                stds = (
                    xp.nanmedian(xp.abs(reference_values - means), axis=0) / mad_scale
                )
        else:
            if centering:
                means = xp.nanmean(reference_values, axis=0)
            if scaling:
                stds = xp.nanstd(reference_values, axis=0)
        if centering:
            means = xp.expand_dims(means, 0)
        if scaling:
            stds = xp.expand_dims(stds, 0)
    else:
        # reference_values dims are (labels,neighbors,features)
        if robust:
            means = xp.nanmedian(reference_values, axis=1)

            if scaling:
                stds = (
                    xp.nanmedian(
                        xp.abs(reference_values - xp.expand_dims(means, axis=1)),
                        axis=1,
                    )
                    / mad_scale
                )

        else:
            if centering:
                means = xp.nanmean(reference_values, axis=1)
            if scaling:
                stds = xp.nanstd(reference_values, axis=1)

    if centering:
        values = values - means
    if scaling:
        stds[stds == 0] = 1.0
        values = values / stds
        if max_value is not None:
            values = xp.clip(values, -max_value, max_value)
    return values


def _nearest_neighbors_indices(
    reference: np.ndarray,
    query: np.ndarray,
    n_neighbors: int = 100,
    metric: str = "minkowski",
) -> np.ndarray:
    if n_neighbors > len(reference):
        raise ValueError(f"n_neighbors: {n_neighbors}, n points: {len(reference)}")
    # shape is reference.shape[0], n_neighbors
    return (
        NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        .fit(reference)
        .kneighbors(query, return_distance=False)
    )


def typical_variation_normalization(
    data: anndata.AnnData,
    reference_query: str,
    by: Sequence[str] | str | None = None,
    pca_kwargs: dict | None = None,
) -> anndata.AnnData:
    """
    Apply Typical Variation Normalization based on control
    perturbations.

    Note that the data is first centered and scaled based on the control units.

    :param data: Annotated data matrix.
    :param reference_query: Query to extract reference observations
        (e.g. "gene_symbol=='NTC'")
    :param by: Further align control and treatments in each group,
        using the covariance matrix of all negative (reference) controls as the target
        and the covariance matrix of each group of negative controls as the source.
    :param pca_kwargs: Arguments to pass to PCA.
    :return: Annotated data matrix.
    """
    # Adapted from EFAAR_benchmarking <https://github.com/recursionpharma/EFAAR_benchmarking/blob/trunk/efaar_benchmarking/efaar.py>_

    columns = _get_names_from_pd_query(reference_query)
    columns = [c for c in columns if c in data.obs.columns]

    if by is not None:
        by = _trim_by(by)
        by_values = _xarray_by_values(data, by)
        coords = dict(obs=by_values)
        coords["index"] = ("obs", data.obs.index.values)

        for c in columns:
            coords[c] = ("obs", data.obs[c].to_numpy(copy=False))
        xdata = xr.DataArray(data.X, dims=("obs", "var"), coords=coords, name="")
        xdata_ref = xdata.query(obs=reference_query)
        ref_means = xdata_ref.mean(dim="obs")
        ref_stds = xdata_ref.std(dim="obs")
        xdata = (xdata - ref_means) / ref_stds
        xdata_ref = xdata.query(obs=reference_query)
    else:
        coords = dict(obs=data.obs.index.values)

        for c in columns:
            coords[c] = ("obs", data.obs[c].to_numpy(copy=False))
        xdata = xr.DataArray(data.X, dims=("obs", "var"), coords=coords, name="")
        xdata_ref = xdata.query(obs=reference_query)
        means = xdata_ref.mean(dim="obs")
        stds = xdata_ref.std(dim="obs")
        xdata = (xdata - means) / stds
        xdata_ref = xdata.query(obs=reference_query)
    default_pca_kwargs = dict(random_state=239753)
    if isinstance(xdata_ref.data, da.Array):
        xdata_ref.data = xdata_ref.data.compute()
        default_pca_kwargs["copy"] = False
    if pca_kwargs is not None:
        default_pca_kwargs.update(pca_kwargs)
    d = PCA(**default_pca_kwargs)

    d.fit(xdata_ref.data)
    if isinstance(xdata.data, da.Array):
        del xdata_ref

    xdata.data = d.transform(xdata.data)
    xdata_ref = xdata.query(obs=reference_query)

    components_ = d.components_
    mean_ = d.mean_

    variance_ratio = d.explained_variance_ratio_
    variance = d.explained_variance_
    uns = {
        "pca": {
            "variance_ratio": variance_ratio,
            "variance": variance,
            "mean": mean_,
            "PCs": components_,
        }
    }
    xp = get_namespace(xdata.data)
    if by is not None:
        ref_grouped = xdata_ref.groupby("obs")
        ref_mean = ref_grouped.mean()
        ref_std = ref_grouped.std()
        results = []
        grouped = xdata.groupby("obs")
        for key, group in grouped:
            value = (group.data - ref_mean.sel(obs=key).data) / ref_std.sel(
                obs=key
            ).data
            results.append(group.copy(data=value))

        xdata = xr.concat(results, dim="obs")
        xdata_ref = xdata.query(obs=reference_query)
        n_features = xdata.shape[1]
        target_cov = xp.cov(xdata_ref.data, rowvar=False, ddof=1) + 0.5 * xp.eye(
            n_features
        )
        target_cov = fractional_matrix_power(target_cov, 0.5)
        if isinstance(xdata.data, da.Array):
            target_cov = da.from_array(target_cov)
        grouped = xdata.groupby("obs")
        results = []
        for key, group in grouped:
            source_cov = xp.cov(
                group.query(obs=reference_query).data, rowvar=False, ddof=1
            ) + 0.5 * xp.eye(n_features)
            X = group.data
            X = X @ fractional_matrix_power(source_cov, -0.5)
            X = X @ target_cov
            results.append(group.copy(data=X))

        xdata = xr.concat(results, dim="obs")
        return anndata.AnnData(
            X=xdata.data,
            obs=data.obs.loc[xdata.coords["index"].values],
            var=data.var.copy(),
            uns=uns,
        )
    else:
        means = xdata_ref.mean(dim="obs")
        stds = xdata_ref.std(dim="obs")
        xdata = (xdata - means) / stds
        return anndata.AnnData(
            X=xdata.data,
            obs=data.obs.copy(),
            var=data.var.copy(),
            uns=uns,
        )
