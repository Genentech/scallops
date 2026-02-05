import logging
from collections.abc import Sequence
from typing import Literal

import anndata
import dask.array as da
import numpy as np
import scipy
import xarray as xr
from dask.delayed import delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scallops.features.constants import _centroid_column_names

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
    if isinstance(values, da.Array):
        chunks = list(values.chunksize)
        if chunks[0] != values.shape[0]:
            chunks[0] = -1
        chunks = tuple(chunks)
        if chunks != values.chunksize:
            values = values.rechunk(chunks)
        if reference_values is not None:
            ref_chunks = list(reference_values.chunksize)
            if ref_chunks[0] != reference_values.shape[0]:
                ref_chunks[0] = -1
            if ref_chunks[1] != chunks[1]:
                ref_chunks[1] = chunks[1]
            ref_chunks = tuple(ref_chunks)
            if ref_chunks != reference_values.chunksize:
                reference_values = reference_values.rechunk(ref_chunks)
        arrays = (
            (values, reference_values) if reference_values is not None else (values,)
        )

        return da.map_blocks(
            _normalize_features_np,
            *arrays,
            indices=delayed(indices),
            robust=robust,
            mad_scale=mad_scale,
            scaling=scaling,
            centering=centering,
            max_value=max_value,
            meta=values._meta,
        )

    return _normalize_features_np(
        values=values,
        ref_values=reference_values,
        indices=indices,
        robust=robust,
        mad_scale=mad_scale,
        scaling=scaling,
        max_value=max_value,
        centering=centering,
    )


def typical_variation_normalization(
    data: anndata.AnnData,
    reference_query: str,
    normalize_groups: Sequence[str] | str | None = None,
) -> anndata.AnnData:
    """
    Apply Typical Variation Normalization based on control
    perturbations.

    Note that the data is first centered and scaled based on the control units.

    :param data: Annotated data matrix.
    :param reference_query: Query to extract reference observations
        (e.g. "gene_symbol=='NTC'")
    :param normalize_groups: Further align control and treatments in each group,
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
    X = PCA().fit(X[ref_indices]).transform(X)

    if normalize_groups is not None:
        group_to_indices = data.obs.groupby(
            normalize_groups, observed=True, sort=False
        ).indices
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
    return anndata.AnnData(X=X, obs=data.obs.copy(), var=data.var.copy())


def normalize_features(
    data: anndata.AnnData,
    reference_query: str | None,
    normalize_groups: Sequence[str] | str | None = None,
    normalize: Literal["zscore", "local-zscore", "nn-zscore"] = "zscore",
    n_neighbors: int | None = 100,
    neighbors_metric: str = "minkowski",
    robust: bool = False,
    mad_scale: float | str = "normal",
    max_value: float | None = None,
    centering: bool = True,
    scaling: bool = True,
) -> anndata.AnnData:
    """Normalize features

    :param data: Annotated data matrix.
    :param reference_query: Query to extract reference observations
        (e.g. "gene_symbol=='NTC'")
    :param normalize_groups: Column(s) in `data.obs` to stratify by.
    :param normalize: Normalization method to use where `local` uses nearest
        neighbors by location and `nn` uses nearest neighbors by `neighbors_metric`.
    :param n_neighbors: Number of neighbors for local and nearest neighbor zscore.
    :param neighbors_metric: Nearest neighbor metric to use when normalize is
        `nn-zscore`.
    :param robust: Use robust statistics.
    :param mad_scale: Numerical scale factor to divide median absolute deviation. The
        string “normal” is also accepted, and results in scale being the inverse of the
        standard normal quantile function at 0.75
    :param centering: Whether to center the data before scaling.
    :param max_value: Truncate to this value after scaling
    :param scaling: Whether to scale the data by dividing by the standard deviation.
    :return: Normalized data
    """

    mad_scale = _convert_scale(mad_scale)
    coords = dict()
    obs = data.obs
    coords["obs"] = obs.index
    for c in data.obs.columns:
        coords[c] = ("obs", obs[c])

    x_data = xr.DataArray(data.X, dims=["obs", "var"], name="", coords=coords)
    if normalize_groups is not None:
        group_result = x_data.groupby(normalize_groups).map(
            lambda x: _normalize_group(
                x,
                reference_query=reference_query,
                normalize=normalize,
                n_neighbors=n_neighbors,
                neighbors_metric=neighbors_metric,
                robust=robust,
                max_value=max_value,
                mad_scale=mad_scale,
                centering=centering,
                scaling=scaling,
            )
        )

        group_obs = obs.loc[group_result.coords["obs"].values]
        data = anndata.AnnData(X=group_result.data, obs=group_obs, var=data.var.copy())
    else:
        result = _normalize_group(
            x_data,
            reference_query=reference_query,
            normalize=normalize,
            n_neighbors=n_neighbors,
            neighbors_metric=neighbors_metric,
            robust=robust,
            max_value=max_value,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
        )
        data = anndata.AnnData(X=result.data, obs=data.obs.copy(), var=data.var.copy())

    return data


def _normalize_group(
    data: xr.DataArray,
    reference_query: str | None,
    normalize: Literal["zscore", "local-zscore", "nn-zscore"],
    n_neighbors: int | None,
    neighbors_metric: str,
    robust: bool,
    mad_scale: float | str,
    centering: bool,
    max_value: float | None,
    scaling: bool,
) -> xr.DataArray:
    indices = None
    reference_data = (
        data.query(dict(obs=reference_query)) if reference_query is not None else None
    )

    if reference_data is not None:
        if reference_data.shape[0] == 0:
            raise ValueError("No reference data found.")
    if normalize == "nn-zscore":
        # nearest neighbors in PCA space
        nn_query = data.data
        nn_ref = nn_query if reference_data is not None else reference_data.data
        indices = _nearest_neighbors_indices(
            nn_ref, nn_query, n_neighbors=n_neighbors, metric=neighbors_metric
        )
    elif normalize == "local-zscore":
        nn_query = np.stack(
            (
                data.coords[_centroid_column_names[0]].values,
                data.coords[_centroid_column_names[1]].values,
            ),
            axis=1,
        )
        nn_ref = nn_query
        if reference_data is not None:
            nn_ref = np.stack(
                (
                    reference_data.coords[_centroid_column_names[0]].values,
                    reference_data.coords[_centroid_column_names[1]].values,
                ),
                axis=1,
            )

        indices = _nearest_neighbors_indices(
            nn_ref, nn_query, n_neighbors=n_neighbors, metric=neighbors_metric
        )

    values = _normalize_features_array(
        data.data,
        reference_data.data if reference_data is not None else None,
        indices=indices,
        robust=robust,
        mad_scale=mad_scale,
        centering=centering,
        scaling=scaling,
        max_value=max_value,
    )
    return data.copy(data=values)


def _nearest_neighbors_indices(
    reference: np.ndarray,
    query: np.ndarray,
    n_neighbors: int = 100,
    metric: str = "minkowski",
) -> np.ndarray:
    if n_neighbors > len(reference):
        raise ValueError(f"n_neighbors: {n_neighbors}, n points: {len(reference)}")
    return (
        NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        .fit(reference)
        .kneighbors(query, return_distance=False)
    )


def _normalize_features_np(
    values: np.ndarray,
    ref_values: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    mad_scale: float | str = "normal",
    centering: bool = True,
    scaling: bool = True,
    robust: bool = True,
    max_value: float | None = None,
) -> np.ndarray:
    mad_scale = _convert_scale(mad_scale)

    if ref_values is None:
        ref_values = values
    means = None
    stds = None
    if indices is None:
        if robust:
            if centering:
                means = np.nanmedian(ref_values, axis=0)
            if scaling:
                stds = np.nanmedian(np.abs(ref_values - means), axis=0) / mad_scale
        else:
            if centering:
                means = np.nanmean(ref_values, axis=0)
            if scaling:
                stds = np.nanstd(ref_values, axis=0)
        if centering:
            means = np.expand_dims(means, 0)
        if scaling:
            stds = np.expand_dims(stds, 0)
    else:
        ref_values = ref_values[indices]
        # ref_values dims are (labels,neighbors,features)
        if robust:
            means = np.nanmedian(ref_values, axis=1)

            if scaling:
                stds = (
                    np.nanmedian(
                        np.abs(ref_values - np.expand_dims(means, axis=1)),
                        axis=1,
                    )
                    / mad_scale
                )

        else:
            if centering:
                means = np.nanmean(ref_values, axis=1)
            if scaling:
                stds = np.nanstd(ref_values, axis=1)

    if centering:
        values = values - means
    if scaling:
        stds[stds == 0] = 1.0
        values = values / stds
        if max_value is not None:
            values[values > max_value] = max_value
            values[values < -max_value] = -max_value
    return values
