import logging
from collections.abc import Sequence

import anndata
import dask
import dask.array as da
import numpy as np
import xarray as xr
from array_api_compat import get_namespace
from sklearn.preprocessing import PowerTransformer

from scallops.features.util import _anndata_to_xr, _query_anndata, _slice_anndata

logger = logging.getLogger("scallops")


def transform_features_yj(
    data: anndata.AnnData,
    by: str | Sequence | None = None,
    standardize: bool = False,
) -> anndata.AnnData:
    """Transform features using yeo-johnson transform

    :param data: AnnData object
    :param by: Column(s) in `data.obs` to stratify by.
    :param standardize: Set to True to apply zero-mean, unit-variance normalization to the
        transformed output
    :return: Transformed AnnData object
    """

    def _transform_block(x):
        return PowerTransformer(
            method="yeo-johnson", standardize=standardize
        ).fit_transform(x)

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


def feature_variance(
    data: anndata.AnnData, by: str | Sequence
) -> np.ndarray | da.Array:
    """Compute feature variance stratified by column(s) in `data.obs`

    :param data: AnnData object
    :param by: Column(s) in `data.obs` to stratify by when computing variance.
    :return: Median feature variance
    """

    xp = get_namespace(data.X)

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

    variance = xdata.groupby(by).var(skipna=False)  # dims (by, 'var')
    variance = xp.median(variance.data, axis=0)
    return variance


def filter_data(
    data: anndata.AnnData,
    max_fraction_not_finite: float | None = 0.25,
    min_variance: float | None = 0.1,
    max_variance: float | None = None,
    by: str | Sequence | None = None,
) -> anndata.AnnData:
    """Filter cells using `max_fraction_not_finite` then filter features using variance

    :param data: AnnData object
    :param max_fraction_not_finite: Keep cells with <= `max_fraction_not_finite`
    missing or infinite values
    :param min_variance: Keep features with variance >= `min_variance`
    :param max_variance: Keep features with variance <= `max_variance`
    :param by: Column(s) in `data.obs` to stratify by when computing variance. If
    provided, the median variance is used for filtering.
    :return: Filtered AnnData object
    """
    xp = get_namespace(data.X)
    keep_cells = None
    keep_features = None
    if max_fraction_not_finite is not None:
        invalid_counts_per_cell = (~xp.isfinite(data.X)).sum(axis=1)
        max_counts = int(data.shape[1] * max_fraction_not_finite)
        keep_cells = invalid_counts_per_cell <= max_counts
    if min_variance is not None or max_variance is not None:
        if min_variance is None:
            min_variance = -np.inf
        if max_variance is None:
            max_variance = np.inf
        if isinstance(keep_cells, da.Array):
            keep_cells = keep_cells.compute()
        if keep_cells is not None:
            data = _slice_anndata(data, keep_cells)
            keep_cells = None
        if by is not None:
            variance = feature_variance(data, by)

        else:
            variance = xp.var(data.X, axis=0)

        keep_features = (
            (variance >= min_variance)
            & (variance <= max_variance)
            & (xp.isfinite(variance))
        )

    if isinstance(data.X, da.Array):
        keep_features, keep_cells = dask.compute(keep_features, keep_cells)
    return _slice_anndata(data, keep_cells, keep_features)


# ---------------------------------------------------------------------------
# Streaming filter helpers (used when data.X is a large dask array)
# ---------------------------------------------------------------------------


def _streaming_cell_filter(
    X_dask: "da.Array",
    max_fraction_not_finite: float,
    n_prefetch: int = 3,
) -> "np.ndarray":
    """Boolean cell-keep mask with bounded parallel chunk prefetching.

    Reads ``n_prefetch`` chunks concurrently so S3 I/O overlaps with
    computation.  Peak memory ≈ ``n_prefetch`` × one chunk.

    :param X_dask: Dask array of shape (n_obs, n_feat).
    :param max_fraction_not_finite: Fraction threshold; cells above this are
        dropped.
    :param n_prefetch: Max concurrent chunk reads (higher = faster but more RAM).
    :return: Boolean numpy array of length n_obs.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    n_feat = X_dask.shape[1]
    max_bad = int(n_feat * max_fraction_not_finite)
    n_chunks = X_dask.numblocks[0]
    chunk_gb = [X_dask.chunks[0][ci] * n_feat * 4 / 1e9 for ci in range(n_chunks)]

    logger.info(
        "  [cell filter] %d chunks, %d concurrent reads (peak ≈ %.0f GB)",
        n_chunks, n_prefetch, n_prefetch * (chunk_gb[0] if chunk_gb else 0),
    )

    keeps = [None] * n_chunks
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=n_prefetch) as pool:
        # Submit all chunks up-front; executor limits to n_prefetch concurrent.
        futures = [pool.submit(X_dask.blocks[ci].compute) for ci in range(n_chunks)]
        # Consume IN ORDER so row offsets align with obs_df later.
        for ci, fut in enumerate(futures):
            chunk = fut.result()
            bad = (~np.isfinite(chunk)).sum(axis=1)
            keeps[ci] = bad <= max_bad
            del chunk
            done = ci + 1
            eta  = (time.monotonic() - t0) / done * (n_chunks - done) / 60
            logger.info(
                "  [cell filter] %d/%d done — %s bad cells — ETA: %.0f min",
                done, n_chunks, f"{int((~keeps[ci]).sum()):,}", eta,
            )

    return np.concatenate(keeps)


def _streaming_feature_variance_by_group(
    X_dask: "da.Array",
    obs_df: "pd.DataFrame",
    cell_keep: "np.ndarray",
    by: list,
) -> "np.ndarray":
    """Median per-group feature variance via Welford's parallel online algorithm.

    Reads one dask chunk at a time — peak memory ≈ one chunk (no xarray copies,
    no materialisation of the full array).  Preserves the stratified variance
    semantics (median across plate × well groups).

    The Welford parallel update formula is used to combine statistics from
    different chunk slices of the same group:

    .. code-block:: text

        n_new   = n_a + n_b
        mean_c  = (n_a * mean_a + n_b * mean_b) / n_new
        M2_c    = M2_a + M2_b + delta^2 * n_a * n_b / n_new
        (where delta = mean_b - mean_a)

    :param X_dask: Dask array of shape (n_obs, n_feat).
    :param obs_df: DataFrame with obs metadata (same row order as X_dask).
    :param cell_keep: Boolean mask (n_obs,) of cells to include.
    :param by: List of obs columns to stratify by (e.g. ['plate', 'well']).
    :return: Float numpy array of length n_feat (median variance across groups).
    """
    import pandas as pd

    import time
    from concurrent.futures import ThreadPoolExecutor

    n_feat   = X_dask.shape[1]
    n_chunks = X_dask.numblocks[0]
    n_prefetch = 3  # same default as cell filter
    # group_key -> {'n': int, 'mean': ndarray(n_feat), 'M2': ndarray(n_feat)}
    group_stats: dict = {}
    row_offset = 0
    t0 = time.monotonic()

    logger.info(
        "  [variance] %d chunks, %d concurrent reads", n_chunks, n_prefetch,
    )

    with ThreadPoolExecutor(max_workers=n_prefetch) as pool:
        futures = [pool.submit(X_dask.blocks[ci].compute) for ci in range(n_chunks)]
        for ci, fut in enumerate(futures):
            chunk_rows = X_dask.chunks[0][ci]
            cell_mask  = cell_keep[row_offset : row_offset + chunk_rows]
            obs_ci     = obs_df.iloc[row_offset : row_offset + chunk_rows]
            row_offset += chunk_rows

            raw_chunk = fut.result()

            if not cell_mask.any():
                del raw_chunk
                continue

            chunk    = raw_chunk[cell_mask].astype(np.float64)
            del raw_chunk
            obs_filt = obs_ci.iloc[cell_mask]

            for group_key, idx in obs_filt.groupby(by, observed=True).indices.items():
                X_g = chunk[idx]
                if len(X_g) == 0:
                    continue
                n_g    = len(X_g)
                mean_g = np.nanmean(X_g, axis=0)
                var_g  = np.nanvar(X_g, axis=0, ddof=0)

                if group_key not in group_stats:
                    group_stats[group_key] = {
                        "n":    n_g,
                        "mean": mean_g.copy(),
                        "M2":   var_g * n_g,
                    }
                else:
                    s      = group_stats[group_key]
                    n_total = s["n"] + n_g
                    delta   = mean_g - s["mean"]
                    s["mean"] = s["mean"] + delta * n_g / n_total
                    s["M2"]   = s["M2"] + var_g * n_g + delta ** 2 * s["n"] * n_g / n_total
                    s["n"]    = n_total

            del chunk
            done = ci + 1
            eta  = (time.monotonic() - t0) / done * (n_chunks - done) / 60
            logger.info(
                "  [variance] %d/%d done — ETA: %.0f min", done, n_chunks, eta,
            )

    if not group_stats:
        return np.zeros(n_feat)

    group_variances = []
    for s in group_stats.values():
        if s["n"] > 1:
            group_variances.append(s["M2"] / (s["n"] - 1))  # unbiased
        else:
            group_variances.append(np.zeros(n_feat))

    return np.median(np.stack(group_variances, axis=0), axis=0)


def _streaming_materialise(
    X_dask: "da.Array",
    cell_keep: "np.ndarray",
    feat_keep: "np.ndarray",
) -> "np.ndarray":
    """Pre-allocate the filtered output and fill it one chunk at a time.

    Pre-allocation avoids the peak-memory spike from ``np.concatenate`` at the
    end (which would momentarily hold two copies of the filtered array).

    :param X_dask: Dask array of shape (n_obs, n_feat).
    :param cell_keep: Boolean mask (n_obs,) of cells to keep.
    :param feat_keep: Boolean mask (n_feat,) of features to keep.
    :return: Float32 numpy array of shape (n_keep_obs, n_keep_feat).
    """
    from concurrent.futures import ThreadPoolExecutor

    n_out_obs  = int(cell_keep.sum())
    n_out_feat = int(feat_keep.sum())
    n_chunks   = X_dask.numblocks[0]
    n_prefetch = 3
    result     = np.empty((n_out_obs, n_out_feat), dtype=np.float32)
    out_row    = 0
    row_offset = 0

    with ThreadPoolExecutor(max_workers=n_prefetch) as pool:
        futures = [pool.submit(X_dask.blocks[ci].compute) for ci in range(n_chunks)]
        for ci, fut in enumerate(futures):
            chunk_rows = X_dask.chunks[0][ci]
            cell_mask  = cell_keep[row_offset : row_offset + chunk_rows]
            row_offset += chunk_rows

            raw_chunk = fut.result()
            if not cell_mask.any():
                del raw_chunk
                continue

            filtered = raw_chunk[cell_mask][:, feat_keep]
            del raw_chunk
            n_rows = filtered.shape[0]
            result[out_row : out_row + n_rows] = filtered
            out_row += n_rows
            del filtered

    return result


# ---------------------------------------------------------------------------
# Additional feature filters
# ---------------------------------------------------------------------------


def remove_correlated_features(
    data: anndata.AnnData,
    threshold: float = 0.9,
    reference_query: str | None = None,
    chunk_size: int = 512,
) -> anndata.AnnData:
    """Remove features that are pairwise-correlated above *threshold*.

    Uses a variance-ordered greedy algorithm: features are visited in
    decreasing variance order; when a pair of kept features exceeds the
    absolute Pearson correlation threshold the *lower-variance* one is
    dropped.  The correlation matrix is computed in column blocks of
    *chunk_size* to bound peak memory to ``O(p × chunk_size × 4 bytes)``
    plus one ``float32 (p, p)`` matrix for the final greedy pass.

    :param data: AnnData object (may have dask ``X``; will be computed).
    :param threshold: Maximum allowed absolute Pearson correlation between any
        two retained features.  Values closer to 1.0 are more permissive.
    :param reference_query: If given, Pearson correlations are computed on the
        subset of observations matching this query expression (e.g. NTC
        controls), avoiding inflation by biological signal.
    :param chunk_size: Column block size for the blocked matrix-multiply
        correlation estimate.  Larger values are faster but use more memory.
    :return: Filtered AnnData with correlated features removed.
    """
    X = data.X
    if isinstance(X, da.Array):
        X = X.compute()
    X = np.asarray(X, dtype=np.float32)

    if reference_query is not None:
        ref_idx = data.obs.index.get_indexer_for(
            _query_anndata(data, reference_query).index
        )
        if len(ref_idx) == 0:
            logger.warning(
                "remove_correlated_features: reference_query matched zero cells; "
                "using all cells."
            )
            X_ref = X
        else:
            X_ref = X[ref_idx]
    else:
        X_ref = X

    n, p = X_ref.shape
    if p < 2:
        return data

    # Centre and normalise → each column has mean≈0, std≈1
    means = np.nanmean(X_ref, axis=0, keepdims=True)
    stds = np.nanstd(X_ref, axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    X_norm = ((X_ref - means) / stds).astype(np.float32)

    # Compute (p, p) Pearson correlation in column blocks → float32
    corr = np.empty((p, p), dtype=np.float32)
    for ci in range(0, p, chunk_size):
        ci_end = min(ci + chunk_size, p)
        block_i = X_norm[:, ci:ci_end]
        for cj in range(ci, p, chunk_size):
            cj_end = min(cj + chunk_size, p)
            block_j = X_norm[:, cj:cj_end]
            block = (block_i.T @ block_j) / max(n - 1, 1)
            corr[ci:ci_end, cj:cj_end] = block
            if ci != cj:
                corr[cj:cj_end, ci:ci_end] = block.T

    # Greedy pass: process features in decreasing variance order
    variances = np.nanvar(X_ref, axis=0)
    order = np.argsort(variances)[::-1]  # highest variance first
    keep = np.ones(p, dtype=bool)
    np.fill_diagonal(corr, 0.0)

    for i in order:
        if not keep[i]:
            continue
        # Drop all still-kept features that are too correlated with feature i
        mask = (np.abs(corr[i]) > threshold) & keep
        keep[mask] = False

    n_removed = int((~keep).sum())
    logger.info(
        f"remove_correlated_features: removed {n_removed:,} features "
        f"(threshold={threshold}), kept {int(keep.sum()):,}"
    )
    return _slice_anndata(data, None, keep)


def filter_zero_inflated(
    data: anndata.AnnData,
    max_zero_fraction: float = 0.5,
    near_zero_threshold: float = 0.0,
    by: str | Sequence | None = None,
) -> anndata.AnnData:
    """Remove features with an excessive fraction of zero or near-zero values.

    A value ``v`` is considered *near-zero* when ``|v| <= near_zero_threshold``.
    When *by* is given the zero fraction is computed per group and the
    **maximum** group-wise fraction must fall below *max_zero_fraction*.

    :param data: AnnData object.
    :param max_zero_fraction: Maximum fraction of near-zero values a feature
        may have before it is removed (0–1).
    :param near_zero_threshold: Values with absolute magnitude ≤ this are
        counted as zero.
    :param by: Column(s) in ``obs`` to stratify the zero-fraction computation.
        The maximum fraction across groups is used.
    :return: Filtered AnnData.
    """
    xp = get_namespace(data.X)

    if by is None:
        zero_mask = xp.abs(data.X) <= near_zero_threshold
        zero_fractions = zero_mask.mean(axis=0)
        if isinstance(zero_fractions, da.Array):
            zero_fractions = zero_fractions.compute()
    else:
        # Compute per-group maximum zero fraction
        if isinstance(by, str):
            by_list = [by]
        else:
            by_list = list(by)
        groups = data.obs.groupby(by_list, observed=True, sort=False).indices
        group_fractions = []
        for idx in groups.values():
            if len(idx) == 0:
                continue
            X_g = data.X[idx]
            frac = (xp.abs(X_g) <= near_zero_threshold).mean(axis=0)
            if isinstance(frac, da.Array):
                frac = frac.compute()
            group_fractions.append(np.asarray(frac))
        zero_fractions = np.max(np.stack(group_fractions, axis=0), axis=0)

    keep = np.asarray(zero_fractions) <= max_zero_fraction
    n_removed = int((~keep).sum())
    logger.info(
        f"filter_zero_inflated: removed {n_removed:,} features "
        f"(max_zero_fraction={max_zero_fraction}), kept {int(keep.sum()):,}"
    )
    return _slice_anndata(data, None, keep)


def filter_low_cardinality(
    data: anndata.AnnData,
    min_unique: int = 20,
) -> anndata.AnnData:
    """Remove features that appear categorical (too few distinct values).

    Features with fewer than *min_unique* unique values across the dataset
    are likely integer-coded or Boolean flags that should not be treated as
    continuous morphological measurements.

    :param data: AnnData object.  ``X`` is materialised if it is a dask array.
    :param min_unique: Minimum number of distinct values a feature must have
        to be retained.
    :return: Filtered AnnData.
    """
    X = data.X
    if isinstance(X, da.Array):
        X = X.compute()
    X = np.asarray(X)

    # Count unique values per column; NaNs are ignored
    n_unique = np.array(
        [np.unique(X[:, j][np.isfinite(X[:, j])]).size for j in range(X.shape[1])]
    )
    keep = n_unique >= min_unique
    n_removed = int((~keep).sum())
    logger.info(
        f"filter_low_cardinality: removed {n_removed:,} features "
        f"(min_unique={min_unique}), kept {int(keep.sum()):,}"
    )
    return _slice_anndata(data, None, keep)


def filter_batch_correlated(
    data: anndata.AnnData,
    batch_column: str | Sequence[str],
    reference_query: str | None = None,
    pvalue_threshold: float = 0.05,
    method: str = "kruskal",
) -> anndata.AnnData:
    """Remove features that are significantly associated with batch identity.

    For each feature a statistical test is performed across batches
    (defined by *batch_column*) to detect batch-driven variation.
    Features significant at *pvalue_threshold* are removed.

    When *reference_query* is given the test is restricted to those
    observations (e.g. NTC controls), where any remaining between-batch
    variation must be purely technical.

    :param data: AnnData object.
    :param batch_column: Column name (or list of names) in ``obs`` that
        identifies the batch.  Multiple columns are concatenated into a single
        composite batch label.
    :param reference_query: Query expression selecting reference observations
        (e.g. ``"gene_symbol=='NTC'"``) on which the test is performed.
        If *None* all observations are used.
    :param pvalue_threshold: Features with p-value < this threshold (after
        the batch association test) are removed.
    :param method: Test method: ``"kruskal"`` (Kruskal-Wallis, non-parametric,
        recommended) or ``"anova"`` (one-way ANOVA).
    :return: Filtered AnnData.
    """
    from scipy.stats import f_oneway, kruskal

    X = data.X
    if isinstance(X, da.Array):
        X = X.compute()
    X = np.asarray(X, dtype=np.float64)
    obs = data.obs

    # Build composite batch label
    if isinstance(batch_column, str):
        batch_labels = obs[batch_column].astype(str).values
    else:
        batch_labels = obs[list(batch_column)].astype(str).agg("-".join, axis=1).values

    # Optionally restrict to reference cells
    if reference_query is not None:
        ref_mask = obs.index.isin(_query_anndata(data, reference_query).index)
        X_test = X[ref_mask]
        batch_test = batch_labels[ref_mask]
    else:
        X_test = X
        batch_test = batch_labels

    unique_batches = np.unique(batch_test)
    if len(unique_batches) < 2:
        logger.warning(
            "filter_batch_correlated: fewer than 2 unique batches found; "
            "skipping batch-correlation filter."
        )
        return data

    # Build group arrays: list of (n_batch_i, n_features) arrays
    groups = [X_test[batch_test == b] for b in unique_batches]

    # Vectorised test across all features at once
    if method == "kruskal":
        # kruskal accepts *groups per call*, not vectorised → use anova approximation
        # for vectorised operation, then fall back to kruskal for confirmation
        # Use a vectorised one-way ANOVA to get p-values quickly, then keep
        # features passing anova (conservative direction: harder to remove)
        test_fn = f_oneway
    elif method == "anova":
        test_fn = f_oneway
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'kruskal' or 'anova'.")

    _, pvalues = test_fn(*groups)  # pvalues shape: (n_features,)

    if method == "kruskal":
        # Re-test only the features that appeared significant under anova
        # using the proper kruskal test (slower but correct)
        potentially_significant = np.where(pvalues < pvalue_threshold)[0]
        for j in potentially_significant:
            col_groups = [g[:, j] for g in groups]
            try:
                _, pvalues[j] = kruskal(*col_groups)
            except ValueError:
                pvalues[j] = 1.0

    keep = ~np.isfinite(pvalues) | (pvalues >= pvalue_threshold)
    n_removed = int((~keep).sum())
    logger.info(
        f"filter_batch_correlated: removed {n_removed:,} features "
        f"(method={method}, p<{pvalue_threshold}), kept {int(keep.sum()):,}"
    )
    return _slice_anndata(data, None, keep)
