import logging
from collections.abc import Sequence

import anndata
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from array_api_compat import get_namespace
from sklearn.preprocessing import PowerTransformer

from scallops.features.util import _anndata_to_xr, _query_anndata, _slice_anndata

logger = logging.getLogger("scallops")


_YJ_OUTPUT_CAP = None  # no output cap by default; only meaningful with standardize=True


def _yj_fit_transform_col(col: np.ndarray, standardize: bool,
                          output_cap: float = _YJ_OUTPUT_CAP) -> np.ndarray:
    """Fit Yeo-Johnson on the *finite* values of a column, then apply to all
    values while propagating NaN for missing entries.

    Top-level for joblib pickling.  Uses scipy directly so it releases the
    GIL during numerical optimisation.
    """
    from scipy.stats import yeojohnson as _yj
    col = col.astype(np.float64)
    finite_mask = np.isfinite(col)
    if not finite_mask.all():
        # Fit on non-NaN only, apply to finite subset, leave NaN as-is
        valid = col[finite_mask]
        if len(valid) < 2:
            return col.astype(np.float32)
        transformed_valid, lmbda = _yj(valid)
        # Apply the fitted lambda to all finite cells using the forward transform
        from scipy.stats import yeojohnson_normmax as _yjnorm
        out = np.full_like(col, np.nan)
        # yeojohnson with known lambda: use boxcox1p / sign-aware formula
        pos = valid >= 0
        neg = ~pos
        out_valid = np.empty_like(valid)
        if pos.any():
            if lmbda != 0:
                out_valid[pos] = (np.power(valid[pos] + 1, lmbda) - 1) / lmbda
            else:
                out_valid[pos] = np.log1p(valid[pos])
        if neg.any():
            if lmbda != 2:
                out_valid[neg] = -(np.power(-valid[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
            else:
                out_valid[neg] = -np.log1p(-valid[neg])
        out[finite_mask] = out_valid
        if standardize:
            std = np.nanstd(out)
            if std > 0:
                out = (out - np.nanmean(out)) / std
        if output_cap is not None and np.isfinite(output_cap):
            np.clip(out, -output_cap, output_cap, out=out)
        return out.astype(np.float32)
    # No NaN: standard path
    transformed, _ = _yj(col)
    if standardize:
        std = transformed.std()
        if std > 0:
            transformed = (transformed - transformed.mean()) / std
    # Cap output to prevent extreme values from corrupting downstream z-scores.
    # Default ±5 is already extreme for any normalised biological measurement.
    if output_cap is not None and np.isfinite(output_cap):
        np.clip(transformed, -output_cap, output_cap, out=transformed)
    return transformed.astype(np.float32)


def transform_features_yj(
    data: anndata.AnnData,
    by: str | Sequence | None = None,
    standardize: bool = False,
    n_jobs: int = -1,
    output_cap: float | None = _YJ_OUTPUT_CAP,
) -> anndata.AnnData:
    """Transform features using Yeo-Johnson transform.

    The transform is fitted independently per feature (and per group when
    *by* is set), parallelised using Dask's threaded scheduler.  scipy's
    yeojohnson optimiser releases the GIL in C, so Dask threads achieve true
    CPU-level parallelism without data copying.

    :param data: AnnData object
    :param by: Column(s) in `data.obs` to stratify by.
    :param standardize: Apply zero-mean, unit-variance normalisation after YJ.
    :param n_jobs: Number of Dask worker threads (−1 = all CPUs, 1 = serial).
    :return: Transformed AnnData object
    """
    import os
    import dask
    from dask import delayed as _dd

    if n_jobs == -1:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, n_jobs)

    by_cols = ([by] if isinstance(by, str) else list(by)) if by else []
    X_full = np.asarray(data.X, dtype=np.float64)
    n_obs, n_feat = X_full.shape

    if not by_cols:
        # Single-group: parallelise over features with Dask threads
        tasks = [_dd(_yj_fit_transform_col)(X_full[:, j], standardize,
                                             output_cap=output_cap)
                 for j in range(n_feat)]
        cols_out = dask.compute(*tasks, scheduler="threads", num_workers=n_workers)
        X_out = np.column_stack(cols_out).astype(np.float32)
        return anndata.AnnData(X=X_out, obs=data.obs.copy(), var=data.var.copy())

    # Multi-group: parallelise all (group × feature) pairs
    if len(by_cols) == 1:
        group_keys = data.obs[by_cols[0]].astype(str).values
    else:
        group_keys = data.obs[by_cols].astype(str).agg("-".join, axis=1).values
    unique_groups = list(dict.fromkeys(group_keys))

    # Pre-compute masks once (avoids repeated string comparisons in workers)
    group_masks = {gk: group_keys == gk for gk in unique_groups}

    def _process(gk: str, j: int) -> tuple[str, int, np.ndarray]:
        return gk, j, _yj_fit_transform_col(X_full[group_masks[gk], j], standardize,
                                             output_cap=output_cap)

    tasks = [_dd(_process)(gk, j)
             for gk in unique_groups
             for j in range(n_feat)]
    results = dask.compute(*tasks, scheduler="threads", num_workers=n_workers)

    X_out = np.empty((n_obs, n_feat), dtype=np.float32)
    for gk, j, col in results:
        X_out[group_masks[gk], j] = col

    return anndata.AnnData(X=X_out, obs=data.obs.copy(), var=data.var.copy())


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


def _streaming_cell_and_variance_filter(
    X_dask: "da.Array",
    obs_df: "pd.DataFrame",
    label_mask: "np.ndarray",
    by: list | None,
    max_fraction_not_finite: float | None,
    *,
    n_prefetch: int = 3,
) -> "tuple[np.ndarray, np.ndarray]":
    """Single S3 pass: cell-keep mask + per-group Welford variance.

    Merges what were previously two separate passes (``_streaming_cell_filter``
    and ``_streaming_feature_variance_by_group``) into one read.  Each chunk is
    fetched once; the cell filter and variance update happen on the same copy.

    :param X_dask: Dask array of shape (n_obs, n_feat).
    :param obs_df: DataFrame with obs metadata, same row order as X_dask.
    :param label_mask: Boolean mask (n_obs,) from the label-filter expression
        (obs-only, no I/O required to compute).
    :param by: Obs columns to stratify variance by, or None for global.
    :param max_fraction_not_finite: Cell-level infinite/NaN fraction threshold.
    :param n_prefetch: Max concurrent S3 chunk reads.
    :return: ``(cell_keep, feat_var)`` — boolean mask (n_obs,) and float array
        (n_feat,) of median group variance.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    n_feat   = X_dask.shape[1]
    n_chunks = X_dask.numblocks[0]
    max_bad  = int(n_feat * max_fraction_not_finite) if max_fraction_not_finite is not None else n_feat

    group_stats: dict = {}
    cell_keep_parts: list = []
    row_offset = 0
    t0 = time.monotonic()

    logger.info(
        "  [filter pass 1/2] %d chunks, %d concurrent reads"
        " — cell filter + %s variance in one S3 pass",
        n_chunks, n_prefetch,
        f"per-{'×'.join(by)}" if by else "global",
    )

    with ThreadPoolExecutor(max_workers=n_prefetch) as pool:
        futures = [pool.submit(X_dask.blocks[ci].compute) for ci in range(n_chunks)]

        for ci, fut in enumerate(futures):
            chunk_rows  = X_dask.chunks[0][ci]
            label_ci    = label_mask[row_offset : row_offset + chunk_rows]
            obs_ci      = obs_df.iloc[row_offset : row_offset + chunk_rows]
            row_offset += chunk_rows

            chunk = fut.result()

            # ── Cell filter: label mask ∩ finite-value mask ────────────────
            bad      = (~np.isfinite(chunk)).sum(axis=1)
            cell_mask = label_ci & (bad <= max_bad)
            cell_keep_parts.append(cell_mask)

            # ── Welford variance update (only kept cells) ──────────────────
            if cell_mask.any():
                kept     = chunk[cell_mask].astype(np.float64)
                obs_filt = obs_ci.iloc[cell_mask]

                groups = obs_filt.groupby(by, observed=True).indices if by else {"_all_": np.arange(len(kept))}
                for gk, idx in groups.items():
                    X_g = kept[idx]
                    if len(X_g) == 0:
                        continue
                    n_g    = len(X_g)
                    mean_g = np.nanmean(X_g, axis=0)
                    var_g  = np.nanvar(X_g, axis=0, ddof=0)
                    if gk not in group_stats:
                        group_stats[gk] = {"n": n_g, "mean": mean_g.copy(), "M2": var_g * n_g}
                    else:
                        s = group_stats[gk]
                        nt    = s["n"] + n_g
                        delta = mean_g - s["mean"]
                        s["mean"] += delta * n_g / nt
                        s["M2"]   += var_g * n_g + delta ** 2 * s["n"] * n_g / nt
                        s["n"]     = nt
                del kept

            del chunk
            done = ci + 1
            eta  = (time.monotonic() - t0) / done * (n_chunks - done) / 60
            logger.info(
                "  [filter pass 1/2] %d/%d done — %s cells kept — ETA: %.0f min",
                done, n_chunks, f"{int(cell_mask.sum()):,}", eta,
            )

    cell_keep = np.concatenate(cell_keep_parts)

    if not group_stats:
        return cell_keep, np.zeros(n_feat)

    vars_per_group = [
        s["M2"] / (s["n"] - 1) if s["n"] > 1 else np.zeros(n_feat)
        for s in group_stats.values()
    ]
    feat_var = np.median(np.stack(vars_per_group, axis=0), axis=0)
    return cell_keep, feat_var


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
    *,
    n_prefetch: int = 3,
) -> "np.ndarray":
    """Pre-allocate the filtered output and fill it one chunk at a time.

    Pre-allocation avoids the peak-memory spike from ``np.concatenate`` at the
    end (which would momentarily hold two copies of the filtered array).

    :param X_dask: Dask array of shape (n_obs, n_feat).
    :param cell_keep: Boolean mask (n_obs,) of cells to keep.
    :param feat_keep: Boolean mask (n_feat,) of features to keep.
    :param n_prefetch: Max concurrent chunk reads (higher = faster but more RAM).
    :return: Float32 numpy array of shape (n_keep_obs, n_keep_feat).
    """
    from concurrent.futures import ThreadPoolExecutor

    n_out_obs  = int(cell_keep.sum())
    n_out_feat = int(feat_keep.sum())
    n_chunks   = X_dask.numblocks[0]
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
# Column-batch filter for parquet (all row groups parallel per feature batch)
# ---------------------------------------------------------------------------


def _col_batch_filter_parquet(
    sources: list,
    obs_df: "pd.DataFrame",
    label_mask: "np.ndarray",
    by: list | None,
    max_fraction_not_finite: float | None,
    min_variance: float | None,
    max_variance: float | None,
    feat_cols: "list[str] | None" = None,
    batch_size: int = 500_000,
    max_memory_gb: float | None = None,
    max_feature_nan_fraction: float | None = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]":
    """Two-pass sequential streaming filter for parquet files (local or S3).

    Works for both parquet AND any format accepted by the PyArrow dataset API.
    Uses sequential HTTP GETs (one per S3 object) rather than per-column range
    requests, and handles files with mismatched schemas automatically.

    Uses sequential S3 reads (one HTTP GET per file rather than one range
    request per column chunk) and handles files with different schemas by
    letting the dataset scanner align them automatically.

    Pass 1 — stream all batches once: accumulate per-cell AND per-feature
    NaN/Inf counts, plus per-group Welford variance.

    Feature filtering order (applied before cell filtering):
    1. ``max_feature_nan_fraction``: drop features where the NaN/Inf fraction
       across ALL cells exceeds this threshold.  These features are unreliable
       regardless of variance (e.g. edge-cell colocalization features).
    2. Variance filter (``min_variance``, ``max_variance``): applied only on the
       reliable (low-NaN) features.

    Cell filtering order:
    3. ``max_fraction_not_finite``: applied on the *remaining* reliable features
       only, so cells that appeared NaN-heavy solely due to dropped features are
       correctly retained.

    Pass 2 — stream again with only surviving columns: materialise the filtered
    (cell_keep × feat_keep) matrix.

    :param sources: List of dicts with ``path`` and ``feat_cols`` keys, as
        produced by ``_read_parquet_for_map``.
    :param obs_df: DataFrame with obs metadata, row-aligned with the concatenated
        sources.
    :param label_mask: Boolean mask (n_obs,) from an obs-only label filter.
    :param by: Obs columns to stratify variance by, or None for global.
    :param max_fraction_not_finite: Cell-level infinite/NaN fraction threshold.
    :param min_variance: Minimum variance threshold (None = disabled).
    :param max_variance: Maximum variance threshold (None = disabled).
    :param feat_cols: Authoritative feature column list (intersection across all
        files after ``anndata.concat``). Defaults to ``sources[0]['feat_cols']``.
    :return: ``(X_filtered, cell_keep, feat_keep, report_df)`` — float32 array,
        two bool masks, and a :class:`pandas.DataFrame` with one row per input
        feature recording ``feature``, ``compartment``, ``measurement_type``,
        ``kept``, ``drop_step``, ``nan_frac_all_cells``, ``nan_frac_bad_cells``,
        and ``median_variance``.
    """
    import time
    import pyarrow.dataset as _ds
    import pyarrow.fs    as _pafs

    if feat_cols is None:
        feat_cols = list(sources[0]["feat_cols"])
    n_feat  = len(feat_cols)
    n_cells = len(obs_df)
    max_bad = int(n_feat * max_fraction_not_finite) if max_fraction_not_finite is not None else n_feat

    # ── Build a single multi-file dataset ────────────────────────────────────
    # The dataset scanner reads files sequentially (one HTTP GET per S3 object,
    # not one range request per column chunk), handles schema differences across
    # files automatically, and uses internal threading for decompression.
    _pa_fs, _ = _pafs.FileSystem.from_uri(sources[0]["path"])
    _pa_paths  = [_pafs.FileSystem.from_uri(src["path"])[1] for src in sources]
    dataset    = _ds.dataset(_pa_paths, filesystem=_pa_fs, format="parquet")

    logger.info(
        "  [dataset filter] %d files, %d features, sequential streaming …",
        len(sources), n_feat,
    )

    # Precompute label-filtered obs once (in RAM, no I/O)
    obs_label  = obs_df.iloc[label_mask]

    # ── Pass 1: count NaN per cell and per feature (no Welford — variance computed
    # in step 4 on the already-clean materialised matrix, which is more accurate).
    bad_counts        = np.zeros(n_cells, dtype=np.int32)
    nan_per_feat      = np.zeros(n_feat,  dtype=np.int64)
    nan_per_feat_bad  = np.zeros(n_feat,  dtype=np.int64)
    row_offset  = 0

    # Compute batch_readahead / fragment_readahead from the memory budget.
    #
    # max_memory_gb (explicit):  Use this many GB for the read-ahead buffer.
    #   Pass a value ≤ available physical RAM for dedicated machines, or a
    #   smaller fraction for shared/cluster environments (e.g. --max-memory 32
    #   on a 256 GB node with 8 concurrent jobs).
    #
    # max_memory_gb is None (auto):  Use 70 % of currently available RAM.
    #   This is the right default for dedicated compute nodes; if you share the
    #   node set --max-memory explicitly.
    #
    # float64 parquet → worst-case 8 bytes/element per batch.
    _batch_gb = batch_size * n_feat * 8 / 1e9
    # _avail_gb is always computed so it can be logged regardless of whether
    # max_memory_gb was supplied explicitly.
    try:
        import psutil as _psutil
        _avail_gb = _psutil.virtual_memory().available / 1e9
    except Exception:
        _avail_gb = 64.0   # conservative fallback

    if max_memory_gb is not None:
        _budget_gb = float(max_memory_gb)
    else:
        # 40% keeps the pre-fetch buffer small enough that PyArrow yields the
        # first batch quickly. Larger fractions stall the scan waiting to fill
        # hundreds of GB before the first batch is delivered to Python.
        _budget_gb = _avail_gb * 0.40
    _budget_batches = max(2, int(_budget_gb / max(_batch_gb, 0.1)))
    # Hard cap frag_ra at 4: beyond 4 concurrent S3 streams the combined
    # buffer (frag_ra × batch_ra × batch_gb) can exceed available RAM even
    # when each term looks safe on paper.
    _frag_ra   = max(1, min(min(len(sources), 4), _budget_batches // 3))
    _batch_ra  = max(2, min(8, _budget_batches // max(1, _frag_ra)))
    logger.info(
        "  [scanner] available=%.0f GB → fragment_readahead=%d, batch_readahead=%d"
        " (budget ≈%.0f GB)",
        _avail_gb, _frag_ra, _batch_ra, _frag_ra * _batch_ra * _batch_gb,
    )

    scanner1 = dataset.scanner(
        columns=feat_cols,
        batch_size=batch_size,
        use_threads=True,
        batch_readahead=_batch_ra,
        fragment_readahead=_frag_ra,
    )
    t0 = time.monotonic()
    n_done = 0

    for batch in scanner1.to_batches():
        n_b = len(batch)
        X_b = batch.to_pandas().to_numpy(np.float32)
        del batch
        label_b = label_mask[row_offset : row_offset + n_b]
        row_offset += n_b

        not_finite_b = ~np.isfinite(X_b)
        per_cell_nan = not_finite_b.sum(axis=1).astype(np.int32)
        bad_counts[row_offset - n_b : row_offset] = per_cell_nan
        nan_per_feat += not_finite_b.sum(axis=0).astype(np.int64)
        # Track NaN in cells that will be dropped (>max_bad NaN) — identifies
        # which features are causing cell attrition.
        bad_mask = per_cell_nan > max_bad
        if bad_mask.any():
            nan_per_feat_bad += not_finite_b[bad_mask].sum(axis=0).astype(np.int64)

        # No Welford streaming needed — variance computed on the clean matrix after
        # materialisation (step 4), which is more accurate and avoids nanvar bias.
        del X_b
        n_done += 1
        if n_done % 5 == 0:
            eta = (time.monotonic() - t0) / row_offset * max(n_cells - row_offset, 0) / 60
            logger.info("  [pass 1/2] %.0f%% — ETA: %.0f min",
                        row_offset / n_cells * 100, eta)

    # ── Four-step filter (applied in the correct order) ──────────────────────────
    #
    # Step 1 — drop features with >max_feature_nan_fraction NaN across ALL cells.
    #   Applied first so that the cell filter (step 2) is not corrupted by features
    #   that are fundamentally broken (e.g. edge-cell colocalization).
    feat_pass1 = np.ones(n_feat, dtype=bool)
    if max_feature_nan_fraction is not None:
        feat_nan_frac = nan_per_feat / max(n_cells, 1)
        feat_pass1 &= feat_nan_frac <= max_feature_nan_fraction
        n_s1_drop = int((~feat_pass1).sum())
        if n_s1_drop:
            logger.info(
                "  [step 1] feature NaN filter (>%.0f%%): %d / %d features dropped",
                max_feature_nan_fraction * 100, n_s1_drop, n_feat,
            )

    # Step 2 — drop cells with >max_fraction_not_finite NaN across step-1-surviving
    #   features.  bad_counts was accumulated over all n_feat features in pass 1,
    #   so we recompute max_bad using the step-1 survivor count as the denominator.
    #   This ensures a cell with NaN only in step-1-dropped features is not
    #   incorrectly discarded (its bad_counts includes NaN in features that are gone).
    n_feat_step1 = int(feat_pass1.sum())
    max_bad_step2 = (
        int(n_feat_step1 * max_fraction_not_finite)
        if max_fraction_not_finite is not None else n_feat_step1
    )
    cell_keep = label_mask & (bad_counts <= max_bad_step2)

    # Steps 3 and 4 run on the materialised clean matrix — see post-pass-2 section.
    feat_keep = feat_pass1   # pass 2 materialises step-1-surviving features

    n_cells_out = int(cell_keep.sum())
    n_feat_out  = int(feat_keep.sum())
    logger.info(
        "  [feature NaN + cell NaN filters done] %s / %s cells · %s / %s features"
        " → materialising …",
        f"{n_cells_out:,}", f"{n_cells:,}", f"{n_feat_out:,}", f"{n_feat:,}",
    )

    # ── Pass 2: materialise ───────────────────────────────────────────────────
    kept_feat_cols = [feat_cols[i] for i, k in enumerate(feat_keep) if k]
    result      = np.empty((n_cells_out, n_feat_out), dtype=np.float32)
    out_row     = 0
    row_offset  = 0

    # Pass 2 reads only surviving columns — re-derive readahead with smaller batch
    _batch_gb2 = batch_size * n_feat_out * 8 / 1e9
    _budget2   = max(2, int(_avail_gb * 0.40 / max(_batch_gb2, 0.1)))
    _frag_ra2  = max(1, min(len(sources), _budget2 // 3))
    _batch_ra2 = max(2, min(8, _budget2 // max(1, _frag_ra2)))

    scanner2 = dataset.scanner(
        columns=kept_feat_cols,
        batch_size=batch_size,
        use_threads=True,
        batch_readahead=_batch_ra2,
        fragment_readahead=_frag_ra2,
    )
    t0 = time.monotonic()
    n_done = 0

    for batch in scanner2.to_batches():
        n_b    = len(batch)
        cell_b = cell_keep[row_offset : row_offset + n_b]
        row_offset += n_b
        if cell_b.any():
            X_b = batch.to_pandas().values.astype(np.float32)
            X_f = X_b[cell_b]
            result[out_row : out_row + X_f.shape[0]] = X_f
            out_row += X_f.shape[0]
            del X_b, X_f
        n_done += 1
        if n_done % 5 == 0:
            eta = (time.monotonic() - t0) / row_offset * max(n_cells - row_offset, 0) / 60
            logger.info("  [pass 2/2] %.0f%% — ETA: %.0f min",
                        row_offset / n_cells * 100, eta)

    logger.info("  [pass 2/2 done] materialised %.1f GB", result.nbytes / 1e9)

    # ── Step 3 — remove features with any remaining NaN in the kept cells ───────
    # After removing bad cells, features whose NaN came solely from those cells
    # are now clean.  Dropping the remainder gives a NaN-free matrix without
    # imputation or an extra S3 scan.
    nan_per_feat_final = np.isnan(result).sum(axis=0)
    feat_still_nan = nan_per_feat_final > 0
    if feat_still_nan.any():
        n_drop = int(feat_still_nan.sum())
        logger.info(
            "  [residual NaN feature removal] %d features still have NaN in kept"
            " cells → dropped, %d remain",
            n_drop, n_feat_out - n_drop,
        )
        result = result[:, ~feat_still_nan]
        feat_keep_indices = np.where(feat_keep)[0]
        feat_keep[feat_keep_indices[feat_still_nan]] = False

    # ── Step 4 — variance filter on the clean NaN-free matrix ───────────────────
    # Computing variance HERE (not during streaming pass 1) gives unbiased estimates
    # because the matrix is already clean: no nanvar, no Welford approximation.
    feat_var = np.zeros(result.shape[1], dtype=np.float64)
    if result.shape[0] > 1:
        if by:
            obs_kept = obs_df.iloc[cell_keep].copy()
            obs_kept = obs_kept.reset_index(drop=True)
            group_vars = []
            for _, grp_idx in obs_kept.groupby(by, observed=True).groups.items():
                X_grp = result[grp_idx.values]
                if len(X_grp) > 1:
                    group_vars.append(np.var(X_grp, axis=0, ddof=1))
            if group_vars:
                feat_var = np.median(np.stack(group_vars), axis=0)
        else:
            feat_var = np.var(result, axis=0, ddof=1)

    feat_var_keep = np.isfinite(feat_var)
    if min_variance is not None:
        feat_var_keep &= feat_var >= min_variance
    if max_variance is not None:
        feat_var_keep &= feat_var <= max_variance

    if not feat_var_keep.all():
        n_var_drop = int((~feat_var_keep).sum())
        logger.info(
            "  [variance filter] dropped %d features (var<%.2f or var>threshold)"
            " → %d remain",
            n_var_drop, min_variance or 0.0, int(feat_var_keep.sum()),
        )
        result = result[:, feat_var_keep]
        feat_keep_indices = np.where(feat_keep)[0]
        feat_keep[feat_keep_indices[~feat_var_keep]] = False

    # ── Build the feature-drop report ─────────────────────────────────────────
    _feat_names = list(feat_cols) if feat_cols else [f"feat_{i}" for i in range(n_feat)]

    def _parse_feature_name(name: str) -> dict:
        parts = name.split("_")
        compartment = parts[0] if parts else "Unknown"
        mtype = parts[1] if len(parts) > 1 else "Unknown"
        return {"compartment": compartment, "measurement_type": mtype}

    # ── Build the feature-drop report ─────────────────────────────────────────
    records = []
    feat_nan_frac_all = nan_per_feat / max(n_cells, 1)
    feat_nan_frac_bad = nan_per_feat_bad / max(int((~cell_keep).sum()), 1)

    # feat_var is indexed by step-3 survivors (before step-4 variance filter),
    # NOT by step-4 survivors.  Build a mapping from original feature index to
    # the correct feat_var position before the report loop to avoid the
    # off-by-one that would occur when using feat_keep[:i+1].sum()-1 after
    # step-4 has further narrowed feat_keep.
    _step3_survivors = np.where(feat_pass1)[0]   # indices after steps 1 only
    # feat_var has len == result_after_step3.shape[1] == len(_step3_survivors),
    # so we map original-feature-index → feat_var position.
    _feat_var_by_orig: dict[int, float] = {}
    if len(feat_var) == len(_step3_survivors):
        for _pos, _orig_idx in enumerate(_step3_survivors):
            _feat_var_by_orig[int(_orig_idx)] = float(feat_var[_pos])

    for i, name in enumerate(_feat_names):
        kept = bool(feat_keep[i])
        parsed = _parse_feature_name(name)
        # Determine which filter step dropped this feature (correct order now)
        # 1=feature NaN · 2=cell NaN (cells, not features) · 3=residual NaN · 4=variance
        if kept:
            drop_step = None
        elif max_feature_nan_fraction is not None and feat_nan_frac_all[i] > max_feature_nan_fraction:
            drop_step = "1_feature_nan_gt50pct"
        elif feat_nan_frac_all[i] > 0:   # had NaN but survived step1; dropped at step3
            drop_step = "3_residual_nan_in_kept_cells"
        else:
            drop_step = "4_low_variance"

        # Look up variance by original feature index (not by step-4 survivor count)
        feat_var_i = _feat_var_by_orig.get(i, float("nan"))

        records.append({
            "feature":             name,
            "compartment":         parsed["compartment"],
            "measurement_type":    parsed["measurement_type"],
            "kept":                kept,
            "drop_step":           drop_step,
            "nan_frac_all_cells":  round(float(feat_nan_frac_all[i]), 5),
            "nan_frac_bad_cells":  round(float(feat_nan_frac_bad[i]), 5),
            "median_variance":     round(feat_var_i if np.isfinite(feat_var_i) else 0.0, 6),
        })

    report_df = pd.DataFrame(records)

    # Log a compartment × drop_step breakdown for the analysis
    if not report_df.empty:
        dropped = report_df[~report_df["kept"]]
        if not dropped.empty:
            summary = (dropped.groupby(["compartment", "drop_step"])
                       .size().reset_index(name="n_dropped"))
            logger.info("  Feature drop breakdown:\n%s", summary.to_string(index=False))

            # Top features driving cell dropping (high NaN in bad cells)
            top_culprits = (
                dropped.sort_values("nan_frac_bad_cells", ascending=False)
                .head(10)[["feature", "compartment", "measurement_type",
                            "nan_frac_all_cells", "nan_frac_bad_cells"]]
            )
            logger.info("  Top features driving cell dropout:\n%s",
                        top_culprits.to_string(index=False))

    return result, cell_keep, feat_keep, report_df


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
