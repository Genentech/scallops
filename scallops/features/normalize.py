import logging
from collections.abc import Sequence
from typing import Literal

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import scipy
from anndata._core.index import _normalize_index
from array_api_compat import get_namespace
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("scallops")


def _convert_scale(mad_scale):
    if isinstance(mad_scale, str):
        if mad_scale.lower() == "normal":
            mad_scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            raise ValueError(f"{mad_scale} is not a valid mad_scale value.")
    return mad_scale


def _normalize_features_array(
    values: np.ndarray | da.Array,
    reference_values: np.ndarray | da.Array,
    indices: np.ndarray | None,
    mad_scale: float | str,
    robust: bool,
    scaling: bool,
    centering: bool,
    max_value: float | None,
):
    """Normalize 2d labels by features array.

    :param values: Array of values to normalize
    :param reference_values: Array of reference values
    :param indices: Array of nearest neighbor indices for local-zscore
    :param mad_scale: The numerical value of mad_scale will be divided out of the final
         result of the median absolute deviation. The default is 1.0. The string
         "normal" is also accepted,and results in `mad_scale` being the inverse of the
         standard normal quantile function at 0.75, which is approximately 0.67449
     :param robust: Use robust statistics
     :return: Array of normalized values

    """

    mad_scale = _convert_scale(mad_scale)
    xp = get_namespace(values)
    if reference_values is None:
        reference_values = values
    means = None
    stds = None
    if indices is None:
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
        if isinstance(reference_values, da.Array):
            # neighbors = indices.shape[1]
            # features = reference_values.shape[1]
            # labels = indices.shape[0]
            # reference_values = reference_values[indices.flatten()].reshape(
            #     (labels, neighbors, features)
            # )
            reference_values = reference_values.vindex[indices]
        else:
            reference_values = reference_values[indices]
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
        indices=None,
        robust=False,
        mad_scale="normal",
        centering=True,
        scaling=True,
        max_value=None,
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
                indices=None,
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
            indices=None,
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
    batch_size: int | None = None,
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
    :param normalize: Normalization method to use where `local` uses nearest
        neighbors by location and `nn` uses nearest neighbors by `neighbors_metric`.
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
    if normalize == "local-zscore":
        indices = _nearest_neighbors_indices_by_group(
            df=data.obs,
            centroid_column_names=centroid_column_names,
            by=by,
            reference_query=reference_query,
            n_neighbors=n_neighbors,
            metric=neighbors_metric,
        )

        result = _normalize_group(
            data.X,
            reference_data=data.X
            if reference_query is None
            else data.X[
                data.obs.index.get_indexer_for(data.obs.query(reference_query).index)
            ],
            indices=indices,
            robust=robust,
            max_value=max_value,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
            batch_size=batch_size,
        )
        return anndata.AnnData(
            X=result,
            obs=data.obs.copy(),
            var=data.var.copy(),
            uns=data.uns.copy(),
        )
    if by is not None:
        group_indices = data.obs.groupby(
            by, as_index=False, sort=False, group_keys=True, observed=True, dropna=False
        ).indices
        result_obs = []
        result_arrays = []

        for key in group_indices.keys():
            indices = group_indices[key]
            sl = indices

            if np.all(np.diff(indices) == 1):
                sl = slice(indices[0], indices[-1] + 1)
            data_slice = data.X[sl]
            data_obs_slice = data.obs.iloc[indices]
            reference_data_slice = None

            if reference_query is not None:
                ref_indices = _normalize_index(
                    data_obs_slice.query(reference_query).index, data_obs_slice.index
                )
                reference_data_slice = data_slice[ref_indices]

            result = _normalize_group(
                data_slice,
                reference_data=reference_data_slice,
                indices=None,
                robust=robust,
                max_value=max_value,
                mad_scale=mad_scale,
                centering=centering,
                scaling=scaling,
                batch_size=batch_size,
            )

            result_arrays.append(result)
            result_obs.append(data_obs_slice)

        return anndata.AnnData(
            X=get_namespace(data.X).vstack(result_arrays),
            obs=pd.concat(result_obs),
            var=data.var.copy(),
            uns=data.uns.copy(),
            obsm=data.obsm.copy(),
            varm=data.varm.copy(),
        )

    result = _normalize_group(
        data.X,
        reference_data=data.X
        if reference_query is None
        else data.X[
            data.obs.index.get_indexer_for(data.obs.query(reference_query).index)
        ],
        indices=None,
        robust=robust,
        max_value=max_value,
        mad_scale=mad_scale,
        centering=centering,
        scaling=scaling,
        batch_size=batch_size,
    )
    return anndata.AnnData(
        X=result,
        obs=data.obs.copy(),
        var=data.var.copy(),
        uns=data.uns.copy(),
        obsm=data.obsm.copy(),
        varm=data.varm.copy(),
    )


def _normalize_group(
    data: np.ndarray | da.Array,
    indices: np.ndarray | None,
    reference_data: np.ndarray | da.Array | None,
    robust: bool,
    mad_scale: float | str,
    centering: bool,
    max_value: float | None,
    scaling: bool,
    batch_size: int | None,
) -> np.ndarray | da.Array:
    if (
        batch_size is not None
        and not isinstance(data, da.Array)
        and indices is not None
        and indices.shape[0] > batch_size
    ):
        value_list = []
        if reference_data is None:
            reference_data = data
        xp = get_namespace(data)
        for i in range(0, indices.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            values = _normalize_features_array(
                data[sl],
                reference_data,
                indices=indices[sl],
                robust=robust,
                mad_scale=mad_scale,
                centering=centering,
                scaling=scaling,
                max_value=max_value,
            )
            value_list.append(values)
        values = xp.concatenate(value_list)
    else:
        values = _normalize_features_array(
            data,
            reference_data if reference_data is not None else None,
            indices=indices,
            robust=robust,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
            max_value=max_value,
        )
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


def _nearest_neighbors_indices_by_group(
    df: pd.DataFrame,
    centroid_column_names: tuple[str, str],
    by: str | list[str] | None,
    reference_query: str | None,
    n_neighbors: int = 100,
    metric: str = "minkowski",
) -> np.ndarray:
    df_ref = df.query(reference_query) if reference_query is not None else df
    if by is None:
        nn_indices = _nearest_neighbors_indices(
            query=df[centroid_column_names].values,
            reference=df_ref[centroid_column_names].values,
            n_neighbors=n_neighbors,
            metric=metric,
        )
        return nn_indices

    query_indices = df.groupby(
        by, as_index=False, sort=False, group_keys=True, observed=True, dropna=False
    ).indices
    ref_indices = (
        df_ref.groupby(
            by, as_index=False, sort=False, group_keys=True, observed=True, dropna=False
        ).indices
        if reference_query is not None
        else query_indices
    )
    result = []
    for key in query_indices.keys():
        query_indices_ = query_indices[key]
        ref_indices_ = ref_indices[key]
        nn_indices = _nearest_neighbors_indices(
            query=df.iloc[query_indices_][centroid_column_names].values,
            reference=df_ref.iloc[ref_indices_][centroid_column_names].values,
            n_neighbors=n_neighbors,
            metric=metric,
        )
        nn_indices = ref_indices_[nn_indices]
        result.append(nn_indices)
    result = np.vstack(result)
    # reference.shape[0], n_neighbors
    return result
