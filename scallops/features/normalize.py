import logging
from collections.abc import Sequence
from typing import Literal

import anndata
import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy
from anndata._core.index import _normalize_index
from array_api_compat import get_namespace
from dask import delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

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

    mad_scale = _convert_scale(mad_scale)
    centroid_column_names = list(centroid_column_names)
    is_dask = isinstance(data.X, da.Array)

    if by is not None:
        group_indices = data.obs.groupby(
            by, as_index=False, sort=False, group_keys=True, observed=True, dropna=False
        ).indices
    else:
        group_indices = {None: None}
    result_obs = []
    result_arrays = []

    _normalize_features_array_func = (
        delayed(_normalize_features_array) if is_dask else _normalize_features_array
    )
    _local_z_batched_func = delayed(_local_z_batched) if is_dask else _local_z_batched
    for key in group_indices.keys():
        if by is not None:
            progress = str(key)
            group_indices_ = group_indices[key]
            array_subset = group_indices_
            if np.all(np.diff(group_indices_) == 1):
                array_subset = slice(group_indices_[0], group_indices_[-1] + 1)
            x = data.X[array_subset]
            df = data.obs.iloc[group_indices_]
        else:
            progress = True
            x = data.X
            df = data.obs
        if is_dask:
            progress = False
        ref_indices = (
            _normalize_index(df.query(reference_query).index, df.index)
            if reference_query is not None
            else None
        )

        if normalize == "local-zscore":
            query_coordinates = df[centroid_column_names].values
            reference_coordinates = (
                df.iloc[ref_indices][centroid_column_names].values
                if reference_query is not None
                else None
            )

            # memory = (x.shape[0] * x.shape[1] * n_neighbors) / batch_size + (x.shape[0] * x.shape[1])
            # memory *= 8

            with dask.annotate(resources={"process": 1}):
                result = _local_z_batched_func(
                    x=x,
                    ref_indices=ref_indices,
                    query_coordinates=query_coordinates,
                    ref_coordinates=reference_coordinates,
                    neighbors_metric=neighbors_metric,
                    n_neighbors=n_neighbors,
                    robust=robust,
                    mad_scale=mad_scale,
                    centering=centering,
                    scaling=scaling,
                    max_value=max_value,
                    progress=progress,
                    batch_size=batch_size,
                )

                if is_dask:
                    result = da.from_delayed(
                        result,
                        shape=x.shape,
                        dtype=np.float64 if scaling or robust else x.dtype,
                    )
            result_arrays.append(result)
        else:
            reference_data = x[ref_indices] if ref_indices is not None else x
            result = _normalize_features_array_func(
                values=x,
                reference_values=reference_data,
                robust=robust,
                mad_scale=mad_scale,
                centering=centering,
                scaling=scaling,
                max_value=max_value,
                local_zscore=False,
            )
            if is_dask:
                result = da.from_delayed(
                    result,
                    shape=x.shape,
                    dtype=np.float64 if scaling or robust else x.dtype,
                )
            result_arrays.append(result)
        result_obs.append(df)

    return anndata.AnnData(
        X=get_namespace(data.X).vstack(result_arrays),
        obs=pd.concat(result_obs),
        var=data.var.copy(),
        uns=data.uns.copy(),
        obsm=data.obsm.copy(),
        varm=data.varm.copy(),
    )


def _local_z_batched(
    x: np.ndarray | da.Array,
    ref_indices: np.ndarray | None,
    query_coordinates: np.ndarray,
    ref_coordinates: np.ndarray | None,
    n_neighbors: int = 75,
    neighbors_metric: str = "minkowski",
    scaling: bool = True,
    centering: bool = True,
    max_value: float | None = None,
    mad_scale: float | str = "normal",
    robust: bool = False,
    batch_size: int | None = 25000,
    progress: bool | str = False,
):
    if ref_coordinates is None:
        ref_coordinates = query_coordinates
    nn_indices = _nearest_neighbors_indices(
        query=query_coordinates,
        reference=ref_coordinates,
        n_neighbors=n_neighbors,
        metric=neighbors_metric,
    )
    if batch_size is None:
        batch_size = x.shape[0]
    result_arrays = []
    reference_data = x[ref_indices] if ref_indices is not None else x
    tqdm, progress_args = tqdm_func(progress)
    for batch in tqdm(range(0, nn_indices.shape[0], batch_size), **progress_args):
        sl = slice(batch, batch + batch_size)
        nn_indices_ = nn_indices[sl]
        x_ = x[sl]
        if isinstance(reference_data, da.Array):
            n_labels = nn_indices_.shape[0]
            n_neighbors = nn_indices_.shape[1]
            nn_indices_ = nn_indices_.flatten()
            nn_indices_ = da.from_array(nn_indices_)
            # (labels,neighbors,features)
            reference_data_ = reference_data[nn_indices_].reshape(
                (n_labels, n_neighbors, -1)
            )
        else:
            reference_data_ = reference_data[nn_indices_]
        result = _normalize_features_array(
            values=x_,
            reference_values=reference_data_,
            robust=robust,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
            max_value=max_value,
            local_zscore=nn_indices is not None,
        )
        result_arrays.append(result)
    return (
        get_namespace(x).vstack(result_arrays)
        if len(result_arrays) > 1
        else result_arrays[0]
    )


def _normalize_features_array(
    values: np.ndarray | da.Array,
    reference_values: np.ndarray | da.Array | None,
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
        reference_values = values
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
    if not isinstance(values, da.Array):
        del means, stds
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
    """
    # Adapted from EFAAR_benchmarking <https://github.com/recursionpharma/EFAAR_benchmarking/blob/trunk/efaar_benchmarking/efaar.py>_
    X = data.X
    ref_indices = data.obs.index.get_indexer_for(data.obs.query(reference_query).index)
    X = _normalize_features_array(
        X,
        X[ref_indices],
        robust=False,
        mad_scale="normal",
        centering=True,
        scaling=True,
        max_value=None,
        local_zscore=False,
    )
    d = PCA()
    X = d.fit(X[ref_indices]).transform(X)
    components_ = d.components_
    mean_ = d.mean_
    variance_ratio = d.explained_variance_ratio_
    variance = d.explained_variance_

    if by is not None:
        group_to_indices = data.obs.groupby(by, observed=True, sort=False).indices
        for group in group_to_indices.keys():
            group_indices = group_to_indices[group]
            group_control_indices = group_indices[np.isin(group_indices, ref_indices)]
            X[group_indices] = _normalize_features_array(
                X[group_indices],
                X[group_control_indices],
                local_zscore=False,
                robust=False,
                mad_scale="normal",
                centering=True,
                scaling=True,
                max_value=None,
            )

        target_cov = np.cov(X[ref_indices], rowvar=False, ddof=1) + 0.5 * np.eye(
            X.shape[1]
        )

        for group in group_to_indices.keys():
            group_indices = group_to_indices[group]
            group_control_indices = group_indices[np.isin(group_indices, ref_indices)]

            source_cov = np.cov(
                X[group_control_indices], rowvar=False, ddof=1
            ) + 0.5 * np.eye(X.shape[1])

            X[group_indices] = np.matmul(
                X[group_indices], scipy.linalg.fractional_matrix_power(source_cov, -0.5)
            )
            X[group_indices] = np.matmul(
                X[group_indices], scipy.linalg.fractional_matrix_power(target_cov, 0.5)
            )
    else:
        X = _normalize_features_array(
            X,
            X[ref_indices],
            local_zscore=False,
            robust=False,
            mad_scale="normal",
            centering=True,
            scaling=True,
            max_value=None,
        )
    return anndata.AnnData(
        X=X,
        obs=data.obs.copy(),
        var=data.var.copy(),
        uns={
            "pca": {
                "variance_ratio": variance_ratio,
                "variance": variance,
                "mean": mean_,
                "PCs": components_,
            }
        },
    )
