import logging
from collections.abc import Sequence
from typing import Literal

import anndata
import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from anndata._core.index import _normalize_index
from array_api_compat import get_namespace
from dask.utils import parse_bytes
from flox.lib import _issorted
from scipy.stats import median_abs_deviation
from sklearn.neighbors import NearestNeighbors

from scallops.features.decomposition import PCA
from scallops.features.util import (
    _slice_anndata,
    _trim_by,
    _xarray_by_values,
)
from scallops.utils import tqdm_func

logger = logging.getLogger("scallops")


def _get_group_chunks(values):
    """Chunk sizes that put each run of equal `values` in its own chunk.

    Should only be used with sorted input.
    """
    # a boundary at i means the split is after row i, so the chunk sizes are the
    # differences between successive split points
    boundaries = np.where(values[:-1] != values[1:])[0] + 1
    split_points = np.concatenate(([0], boundaries, [len(values)]))
    return tuple(int(chunk) for chunk in np.diff(split_points))


def _local_dtype(dtype: np.dtype) -> np.dtype:
    """Dtype the neighborhood statistics are accumulated and returned in."""

    return np.result_type(dtype, np.float32)


def _rows_per_tile(
    n_rows: int, n_neighbors: int, n_features: int, itemsize: int, tile_bytes: int
) -> int:
    """Rows per tile that keep the neighborhood gather within ``tile_bytes``."""

    per_row = max(1, n_neighbors * n_features * itemsize)
    return max(1, min(n_rows, tile_bytes // per_row))


def _feature_chunk_size(
    x: da.Array, max_group_rows: int, override: int | None = None
) -> int:
    """Features per local z-score task.

    A neighborhood spans a whole group, so rows cannot be split and the only knob left
    for task size is the feature axis. The result is snapped to a whole number of the
    column chunks `x` already has, so the rechunk never makes a task read part of a
    source chunk — re-reading a wide source chunk once per feature slice costs far more
    than a larger task does.
    """
    n_features = x.shape[1]
    if override is not None:
        return max(1, min(int(override), n_features))
    target = parse_bytes(dask.config.get("array.chunk-size"))
    itemsize = np.dtype(_local_dtype(x.dtype)).itemsize
    size = max(1, int(target // max(1, max_group_rows * itemsize)))
    widths = x.chunks[1]
    if widths and len(set(widths)) == 1:
        width = widths[0]
        size = max(width, (size // width) * width)
    return max(1, min(size, n_features))


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
    feature_chunk_size: int | None = None,
    centroid_column_names: tuple[str, str] = (
        "Nuclei_AreaShape_Center_Y",
        "Nuclei_AreaShape_Center_X",
    ),
    tile_bytes: int = 128 * 1024 * 1024,
    fast_moments: bool = False,
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
    :param fast_moments: Compute local means and standard deviations from sufficient
        statistics instead of gathering each neighborhood. Several times faster and
        uses far less memory, at the cost of round-off: results differ from the exact
        reduction by ~1e-12 on float64 input, and are at least as close to the true
        value as it on float32 input. Ignored when `robust` is set.
    :param feature_chunk_size: Features per task for the local z-score of a sorted
        Dask array. The default derives one from the `array.chunk-size` Dask config
        value and the largest group.
    :param centroid_column_names: Columns for y and x centroids to use for local zscore.
    :param tile_bytes: Bytes the (rows, neighbors, features) neighborhood gather is allowed to reach in local zscore.
        Should be small enough to stay friendly to cache and to many concurrent workers.
    :return: Normalized data
    """

    assert normalize in ["zscore", "local-zscore"]
    mad_scale = _convert_scale(mad_scale)
    centroid_column_names = list(centroid_column_names)
    is_dask = isinstance(data.X, da.Array)
    use_map_blocks = False
    if max_value is not None and not scaling:
        raise ValueError("max_value only applied when scaling")
    if by is not None:
        by = _trim_by(by)
        by_values = _xarray_by_values(data, by)
        series = pd.Series(by_values, dtype="category")
        use_map_blocks = is_dask and _issorted(series.cat.codes.values)
        if normalize != "zscore":
            # grouping a categorical drops missing values even with dropna=False, so
            # group on the codes, where missing values get their own code of -1
            codes = series.cat.codes
            group_indices = codes.groupby(codes, sort=False).indices
    else:
        group_indices = {None: None}
    if (
        normalize == "local-zscore"
        and is_dask
        and not use_map_blocks
        and by is not None
    ):
        logger.warning(
            "Using slower code for local z-score since data is not sorted. Sort the "
            f"observations by {by} so each group lands in its own chunk."
        )
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

                    if max_value is not None:  # only clip when scaling
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
    # one row per observation, filled in place per group so the full index array is
    # never duplicated by a concatenate
    indices = (
        np.empty(
            (data.n_obs, n_neighbors),
            dtype=np.int32 if data.n_obs <= np.iinfo(np.int32).max else np.int64,
        )
        if use_map_blocks
        else None
    )
    results = [] if not use_map_blocks else None
    obs_list = [] if not use_map_blocks else None
    obs_indices = [] if not use_map_blocks else None
    resolved_feature_chunk = (
        _feature_chunk_size(
            data.X,
            max((len(i) for i in group_indices.values() if i is not None), default=0)
            or data.n_obs,
            feature_chunk_size,
        )
        if is_dask
        else None
    )
    for key in group_indices.keys():
        if by is not None:
            group_indices_ = group_indices[key]
            if not use_map_blocks:
                array_subset = group_indices_
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
            if local_reference_indices is not None:
                reference_indices = local_reference_indices[reference_indices]
            if use_map_blocks:
                # use_map_blocks is only ever set when grouping
                indices[group_indices_] = reference_indices
            else:
                kwargs = dict(
                    robust=robust,
                    mad_scale=mad_scale,
                    centering=centering,
                    scaling=scaling,
                    max_value=max_value,
                    fast_moments=fast_moments,
                    tile_bytes=tile_bytes,
                )
                if is_dask:
                    # a neighborhood spans the whole group, so the group has to be a
                    # single row chunk; gathering through a Dask fancy index instead
                    # would build a task per neighbor
                    x = x.rechunk({0: -1, 1: resolved_feature_chunk})
                    result = da.map_blocks(
                        _local_z_batched,
                        x,
                        da.from_array(reference_indices, chunks=-1),
                        **kwargs,
                        dtype=_local_dtype(data.X.dtype),
                    )
                else:
                    result = _local_z_batched(
                        x=x,
                        reference_indices=reference_indices,  # indices into x
                        **kwargs,
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
            if by is not None:
                obs_indices.append(group_indices_)

    if use_map_blocks:
        # group boundaries line up perfectly with chunk boundaries
        chunks = _get_group_chunks(series.cat.codes.values)
        rechunked_data = data.X.rechunk({0: chunks, 1: resolved_feature_chunk})

        # one chunk per group along the labels dim, neighbors are never split
        indices = da.from_array(indices, chunks=(chunks, -1))
        assert indices.shape[0] == rechunked_data.shape[0]
        kwargs = dict(
            robust=robust,
            mad_scale=mad_scale,
            centering=centering,
            scaling=scaling,
            max_value=max_value,
            fast_moments=fast_moments,
            tile_bytes=tile_bytes,
        )

        result = da.map_blocks(
            _local_z_batched,
            rechunked_data,
            indices,
            **kwargs,
            dtype=_local_dtype(data.X.dtype),
        )
        return anndata.AnnData(
            X=result,
            obs=data.obs.copy(),
            var=data.var.copy(),
            uns=data.uns.copy(),
            obsm=data.obsm.copy(),
            varm=data.varm.copy(),
        )
    obsm = data.obsm.copy()
    if by is not None:
        # rows come back in group order, so obsm has to be reordered to match
        row_indices = np.concatenate(obs_indices)
        obsm = {key: value[row_indices] for key, value in data.obsm.items()}
    return anndata.AnnData(
        X=get_namespace(data.X).vstack(results),
        obs=pd.concat(obs_list),
        var=data.var.copy(),
        uns=data.uns.copy(),
        obsm=obsm,
        varm=data.varm.copy(),
    )


def _run_tiles(func, n_rows: int, tile: int, progress: bool | str):
    """Apply `func` to each row tile in turn.

    Tiling is for memory, not parallelism: Dask parallelizes across blocks, and the
    array library parallelizes each reduction.
    """
    tqdm, progress_args = tqdm_func(progress)
    for start in tqdm(range(0, n_rows, tile), **progress_args):
        func(start)


def _local_moments_gather(
    x,
    indices,
    means,
    stds,
    robust: bool,
    mad_scale: float | None,
    tile: int,
    progress: bool | str = False,
) -> None:
    """Neighborhood statistics by gathering ``x[indices]`` one row tile at a time.

    Mathematically identical to reducing the full ``(rows, neighbors, features)``
    array, but the gather is bounded by ``tile``. The gather is only ever read, never
    written, so it does not matter whether indexing `x` returned a copy or a view.
    """
    xp = get_namespace(x)
    # the robust scale is a deviation about the median, so the median is needed even
    # when the caller does not want the values centered
    want_center = means is not None or (robust and stds is not None)

    def _tile(start: int) -> None:
        stop = min(start + tile, indices.shape[0])
        block = x[indices[start:stop]]  # (rows, neighbors, features)
        center = None
        if want_center:
            center = xp.median(block, axis=1) if robust else xp.mean(block, axis=1)
            if means is not None:
                means[start:stop] = center
        if stds is not None:
            if robust:
                deviation = xp.abs(block - center[:, None, :])
                scale = xp.median(deviation, axis=1) / mad_scale
            else:
                scale = xp.std(block, axis=1)
            stds[start:stop] = scale

    _run_tiles(_tile, indices.shape[0], tile, progress)


def _local_moments_sparse(
    x: np.ndarray,
    indices: np.ndarray,
    means: np.ndarray | None,
    stds: np.ndarray | None,
    tile: int,
    progress: bool | str = False,
) -> None:
    """Neighborhood mean and standard deviation from sufficient statistics.

    Averaging over neighbors is a sparse matrix product: with ``S`` the row-normalized
    neighbor incidence matrix, the local means are ``S @ x`` and the local variances are
    ``S @ x**2 - (S @ x)**2``. That never materializes the ``(rows, neighbors, features)``
    gather :func:`_local_moments_gather` needs, which is what makes it several times
    faster. ``x`` is shifted by its column means first so the one-pass variance stays
    well conditioned.

    NumPy only — it is built on :mod:`scipy.sparse`. The caller dispatches elsewhere
    for other array types and for robust statistics.
    """
    n_rows, n_neighbors = indices.shape
    dtype = _local_dtype(x.dtype)
    shift = x.mean(axis=0, dtype=np.float64).astype(dtype)
    centered = np.subtract(x, shift, dtype=dtype)
    squared = centered * centered if stds is not None else None
    weights = np.full(tile * n_neighbors, 1.0 / n_neighbors, dtype=dtype)
    indptr = np.arange(0, tile * n_neighbors + 1, n_neighbors, dtype=indices.dtype)

    def _tile(start: int) -> None:
        stop = min(start + tile, n_rows)
        rows = stop - start
        operator = sp.csr_matrix(
            (
                weights[: rows * n_neighbors],
                indices[start:stop].ravel(),
                indptr[: rows + 1],
            ),
            shape=(rows, x.shape[0]),
        )
        local_mean = operator @ centered
        if stds is not None:
            variance = operator @ squared
            variance -= local_mean * local_mean
            np.maximum(variance, 0, out=variance)  # guard against round-off below zero
            np.sqrt(variance, out=variance)
            stds[start:stop] = variance
        if means is not None:
            local_mean += shift
            means[start:stop] = local_mean

    _run_tiles(_tile, n_rows, tile, progress)


def _local_z_batched(
    x,
    reference_indices,
    tile_bytes: int,
    scaling: bool = True,
    centering: bool = True,
    max_value: float | None = None,
    mad_scale: float | str = "normal",
    robust: bool = False,
    fast_moments: bool = False,
    progress: bool | str = False,
):
    """Z-score every row of ``x`` against the neighborhood given by its row of indices.

    ``reference_indices`` has shape ``(x.shape[0], n_neighbors)`` and indexes into
    ``x`` itself. Works for any array namespace `get_namespace` understands; the
    NumPy-only fast paths below are guarded.
    """
    xp = get_namespace(x)
    is_numpy = isinstance(x, np.ndarray)
    mad_scale = _convert_scale(mad_scale) if robust else None
    n_rows, n_features = x.shape
    n_neighbors = reference_indices.shape[1]
    dtype = _local_dtype(x.dtype)
    tile = _rows_per_tile(
        n_rows, n_neighbors, n_features, np.dtype(dtype).itemsize, tile_bytes
    )
    if not is_numpy:
        # the indices have to live wherever `x` does for it to gather them
        reference_indices = xp.asarray(reference_indices)

    # the local means are written straight into the output buffer and subtracted in
    # place, so a matrix this size is only ever allocated twice (means and stds)
    out = xp.empty((n_rows, n_features), dtype=dtype)
    means = out if centering else None
    stds = xp.empty_like(out) if scaling else None

    if fast_moments and not robust and is_numpy:
        _local_moments_sparse(x, reference_indices, means, stds, tile, progress)
    else:
        _local_moments_gather(
            x,
            reference_indices,
            means,
            stds,
            robust,
            mad_scale,
            tile,
            progress,
        )

    # tiled so every temporary stays bounded rather than allocating a full-size array
    def _finalize(start: int) -> None:
        stop = min(start + finalize_tile, n_rows)
        block = out[start:stop]
        if centering:
            xp.subtract(x[start:stop], block, out=block)
        else:
            block[...] = x[start:stop]
        if scaling:
            xp.divide(block, stds[start:stop], out=block)
            if max_value is not None:
                xp.clip(block, -max_value, max_value, out=block)

    finalize_tile = _rows_per_tile(
        n_rows, 1, n_features, np.dtype(dtype).itemsize, tile_bytes
    )
    _run_tiles(_finalize, n_rows, finalize_tile, False)
    return out


def _normalize_features_array(
    values: np.ndarray | da.Array,
    reference_indices: np.ndarray | da.Array | None = None,
    reference_values: np.ndarray | da.Array | None = None,
    scaling: bool = True,
    centering: bool = True,
    max_value: float | None = None,
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
    if robust:
        if centering:
            means = xp.nanmedian(reference_values, axis=0)
        if scaling:
            stds = xp.nanmedian(xp.abs(reference_values - means), axis=0) / mad_scale
    else:
        if centering:
            means = xp.nanmean(reference_values, axis=0)
        if scaling:
            stds = xp.nanstd(reference_values, axis=0)
    if centering:
        means = xp.expand_dims(means, 0)
    if scaling:
        stds = xp.expand_dims(stds, 0)

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
    indices = (
        NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        .fit(reference)
        .kneighbors(query, return_distance=False)
    )
    # int64 indices are half the memory of the feature matrix itself at scale
    if len(reference) <= np.iinfo(np.int32).max:
        indices = indices.astype(np.int32, copy=False)
    return indices


def _symmetric_matrix_power(a: np.ndarray, power: float) -> np.ndarray:
    """Fractional power of a symmetric positive-definite matrix.

    :func:`scipy.linalg.fractional_matrix_power` accepts any matrix and pays for it
    with a Schur decomposition. The covariance matrices here are symmetric and shifted
    by ``0.5 * I``, so an eigendecomposition gives the same answer — measured to 1e-14
    — around six times faster at the sizes this is used at.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(a)
    np.clip(eigenvalues, np.finfo(eigenvalues.dtype).tiny, None, out=eigenvalues)
    return (eigenvectors * eigenvalues**power) @ eigenvectors.T


def _apply_affine(
    x: np.ndarray, codes: np.ndarray, maps: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """``x @ maps[g] + offsets[g]``, with `g` the group each row of `x` belongs to."""
    codes = codes.ravel()
    out = np.empty((x.shape[0], maps.shape[2]), dtype=maps.dtype)
    # normally a single group per block, since groups are far larger than row chunks
    for group in np.unique(codes):
        rows = slice(None) if (codes == group).all() else codes == group
        out[rows] = x[rows] @ maps[group] + offsets[group]
    return out


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
    #
    # Every step here — standardizing on the reference, the PCA rotation, the per-group
    # standardization, and the CORAL alignment — is an affine map, so they compose into
    # a single `X @ maps[g] + offsets[g]` per group. Only the reference is needed to
    # derive those, which leaves one pass over the full matrix instead of three.

    reference_rows = _normalize_index(
        data.obs.query(reference_query).index, data.obs.index
    )
    reference = data.X[reference_rows]
    if isinstance(reference, da.Array):
        reference = reference.compute()

    mean_ = reference.mean(axis=0)
    std_ = reference.std(axis=0)
    standardized = (reference - mean_) / std_
    del reference

    default_pca_kwargs = dict(pca_kwargs) if pca_kwargs is not None else dict()
    d = PCA(**default_pca_kwargs)
    d.fit(standardized)
    logger.info(f"TVN: fit PCA with {standardized.shape[0]:,} labels.")
    uns = {
        "pca": {
            "variance_ratio": d.explained_variance_ratio_,
            "variance": d.explained_variance_,
            "mean": d.mean_,
            "PCs": d.components_,
        }
    }
    # the reference is the only thing that has to be rotated to build the maps
    rotated = d.transform(standardized)
    del standardized
    n_components = rotated.shape[1]
    if n_components != data.shape[1]:
        # the output keeps `data.var`, so it has to stay in the original feature space.
        # PCA returns min(n_samples, n_features) components, so this also trips when
        # the reference has fewer rows than there are features.
        raise ValueError(
            f"PCA kept {n_components} of {data.shape[1]} components, but "
            "typical_variation_normalization only supports a full-rank rotation. The "
            f"reference has {reference_rows.shape[0]} observations; it needs at least "
            "as many as there are features, and pca_kwargs must not set n_components."
        )

    if by is None:
        codes = np.zeros(data.n_obs, dtype=np.intp)
        n_groups = 1
    else:
        by = _trim_by(by)
        series = pd.Series(_xarray_by_values(data, by), dtype="category")
        codes = series.cat.codes.to_numpy()
        n_groups = len(series.cat.categories)
        if (codes < 0).any():
            # a missing `by` value gets code -1; give it a group rather than dropping
            codes = np.where(codes < 0, n_groups, codes)
            n_groups += 1
    reference_codes = codes[reference_rows]

    dtype = np.result_type(data.X.dtype, np.float32)
    # diag(1 / std_) @ components_.T, and the constant the standardizations contribute
    base = d.components_.T / std_[:, None]
    base_offset = -(mean_ / std_ + d.mean_) @ d.components_.T

    maps = np.empty((n_groups, data.shape[1], n_components), dtype=dtype)
    offsets = np.empty((n_groups, n_components), dtype=dtype)
    group_rows = [reference_codes == group for group in range(n_groups)]

    if by is None:
        group_mean = rotated.mean(axis=0)
        group_std = rotated.std(axis=0)
        maps[0] = base / group_std
        offsets[0] = (base_offset - group_mean) / group_std
    else:
        group_means = np.empty((n_groups, n_components))
        group_stds = np.empty((n_groups, n_components))
        for group, rows in enumerate(group_rows):
            group_means[group] = rotated[rows].mean(axis=0)
            group_stds[group] = rotated[rows].std(axis=0)
            rotated[rows] = (rotated[rows] - group_means[group]) / group_stds[group]

        identity = 0.5 * np.eye(n_components)
        target_cov = _symmetric_matrix_power(
            np.cov(rotated, rowvar=False, ddof=1) + identity, 0.5
        )
        for group, rows in enumerate(group_rows):
            source_cov = np.cov(rotated[rows], rowvar=False, ddof=1) + identity
            align = _symmetric_matrix_power(source_cov, -0.5) @ target_cov
            maps[group] = (base / group_stds[group]) @ align
            offsets[group] = (
                (base_offset - group_means[group]) / group_stds[group]
            ) @ align

    if isinstance(data.X, da.Array):
        # a row needs every feature for the matmul; widening the column axis is free,
        # it only merges the column chunks already inside each row block
        wide = data.X.rechunk({1: -1})
        result = da.map_blocks(
            _apply_affine,
            wide,
            da.from_array(codes[:, None], chunks=(wide.chunks[0], 1)),
            maps=maps,
            offsets=offsets,
            chunks=(wide.chunks[0], (n_components,)),
            dtype=dtype,
        )
    else:
        result = _apply_affine(data.X, codes, maps, offsets)

    return anndata.AnnData(
        X=result,
        obs=data.obs.copy(),
        var=data.var.copy(),
        uns=uns,
    )
