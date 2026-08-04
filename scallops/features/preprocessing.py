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
    output_zarr_path: "str | None" = None,
    feat_block_bytes: int = 2_000_000_000,
) -> anndata.AnnData:
    """Transform features using Yeo-Johnson.

    :param output_zarr_path: When provided AND ``data.X`` is a dask array,
        write the transformed X directly to this zarr path per feature block
        instead of pre-allocating a full float32 output array.  The returned
        AnnData is backed by that zarr.  Peak RAM ≈ ``feat_block_bytes × 2``
        instead of ``n_obs × n_feat × 4``.
    :param feat_block_bytes: Target bytes per feature block for streaming
        (default 2 GB).  Derived from ``--memory-budget-gb`` by the caller.
    """
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
    n_obs, n_feat = data.shape

    # ── Phase-2 streaming path: dask X + output_zarr_path ────────────────────
    # When output_zarr_path is given, write the transformed X directly to zarr
    # per feature block.  Peak RAM = 2 × feat_block × n_cells × dtype_bytes
    # (one float64 input block + one float32 output block) instead of the full
    # n_cells × n_feat pre-allocation.  The caller receives a zarr-backed
    # AnnData whose X is never fully in RAM.
    # ── Dask path: zarr write-through for all dask inputs ────────────────────
    # Replaces Phase-1 (which pre-allocated X_out = np.empty((n_obs, n_feat))
    # defeating streaming) and consolidates Phase-2.  Works for both:
    #   • Caller-supplied output_zarr_path (no temp file needed)
    #   • No path supplied: creates a temp zarr, returns dask-backed AnnData
    # Zarr column chunk = _fblock so no single chunk covers all n_feat columns.
    if isinstance(data.X, da.Array):
        import tempfile as _tmp_mod, zarr as _zarr
        _zarr_path = output_zarr_path or (
            _tmp_mod.mkdtemp(prefix="scallops_yj_") + "/yj_out.zarr"
        )
        _fblock    = max(1, int(feat_block_bytes / (max(n_obs, 1) * 12)))
        _col_chunk = min(_fblock, n_feat)   # chunk ≤ block → no full-width alloc
        _zgrp = _zarr.open_group(_zarr_path, mode="a")
        if "X" not in _zgrp:
            _zgrp.create_dataset("X", shape=(n_obs, n_feat), dtype="float32",
                                 chunks=(min(50_000, n_obs), _col_chunk),
                                 overwrite=True)
        _zX = _zgrp["X"]
        for _f0 in range(0, n_feat, _fblock):
            _f1  = min(_f0 + _fblock, n_feat)
            _blk = np.asarray(data.X[:, _f0:_f1].compute(), dtype=np.float64)
            _tmp = anndata.AnnData(X=_blk, obs=data.obs)
            if not by_cols:
                _tasks = [_dd(_yj_fit_transform_col)(_blk[:, j], standardize,
                                                      output_cap=output_cap)
                          for j in range(_f1 - _f0)]
                _cols = dask.compute(*_tasks, scheduler="threads",
                                     num_workers=n_workers)
                _zX[:, _f0:_f1] = np.column_stack(_cols).astype(np.float32)
            else:
                _res = transform_features_yj(_tmp, by=by, standardize=standardize,
                                              n_jobs=n_jobs, output_cap=output_cap)
                _zX[:, _f0:_f1] = _res.X
            del _blk, _tmp
        _X_da = da.from_zarr(_zarr_path, component="X")
        return anndata.AnnData(X=_X_da, obs=data.obs.copy(), var=data.var.copy())
    # ── Standard path: small / numpy-backed X ────────────────────────────────

    X_full = np.asarray(data.X, dtype=np.float64)

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


def _dask_nan_scan(
    X_dask: "da.Array",
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute per-cell and per-feature NaN counts using dask-native ops.

    Replaces the manual ThreadPoolExecutor streaming loop for zarr inputs.
    Dask parallelises across ALL chunks (both row and column dimensions)
    in a single fused task graph, avoiding GIL contention and sequential
    result collection.  On S3 this is 5-10× faster than the streaming loop
    because dask issues concurrent requests for every chunk simultaneously.

    :param X_dask: Dask array of shape (n_cells, n_feat).
    :return: ``(bad_counts, nan_per_feat)`` — same semantics as
        :func:`_streaming_cell_and_variance_filter`.
    """
    import dask.array as _da
    not_finite   = ~_da.isfinite(X_dask)
    bad_counts   = not_finite.sum(axis=1).compute().astype(np.int32)
    nan_per_feat = not_finite.sum(axis=0).compute().astype(np.int64)
    return bad_counts, nan_per_feat


def _dask_materialise(
    X_dask: "da.Array",
    cell_keep: "np.ndarray",
    feat_keep: "np.ndarray",
) -> "np.ndarray":
    """Materialise a filtered subset of a dask array using native indexing.

    Replaces :func:`_streaming_materialise` for zarr inputs.  Uses dask
    fancy indexing so the scheduler can parallelise all chunk reads.

    :param X_dask: Dask array (n_cells, n_feat).
    :param cell_keep: Boolean mask (n_cells,) of cells to keep.
    :param feat_keep: Boolean mask (n_feat,) of features to keep.
    :return: Dense float32 numpy array (n_kept_cells, n_kept_feat).
    """
    cell_idx = np.where(cell_keep)[0]
    feat_idx = np.where(feat_keep)[0]
    return X_dask[cell_idx, :][:, feat_idx].compute().astype(np.float32)


def _streaming_cell_and_variance_filter(
    X_dask: "da.Array",
    obs_df: "pd.DataFrame",
    label_mask: "np.ndarray",
    by: list | None,
    max_fraction_not_finite: float | None,
    *,
    n_prefetch: int = 3,
) -> "tuple[np.ndarray, np.ndarray]":
    """Streaming scan: accumulate per-cell and per-feature NaN counts (zarr path).

    Pure statistics accumulator — no filter decisions made here.  Both
    ``bad_counts`` and ``nan_per_feat`` are passed to
    :func:`_apply_filter_steps_1_2` which applies the same step-1 and step-2
    logic as the parquet column-batch path, guaranteeing identical behaviour.

    :param X_dask: Dask array of shape (n_obs, n_feat).
    :param obs_df: DataFrame with obs metadata (row-aligned with X_dask).
    :param label_mask: Boolean mask (n_obs,) from obs-only label filter.
    :param by: Unused (variance now computed post-materialise); kept for
        call-site compatibility.
    :param max_fraction_not_finite: Unused here; passed through to
        ``_apply_filter_steps_1_2`` by the caller.
    :param n_prefetch: Max concurrent S3 chunk reads.
    :return: ``(bad_counts, nan_per_feat)`` — per-cell NaN count (n_obs,)
        and per-feature NaN count (n_feat,), both accumulated across ALL cells
        (including those that will be dropped by step 2).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    n_cells  = X_dask.shape[0]
    n_feat   = X_dask.shape[1]
    n_chunks = X_dask.numblocks[0]

    bad_counts   = np.zeros(n_cells, dtype=np.int32)
    nan_per_feat = np.zeros(n_feat,  dtype=np.int64)
    row_offset   = 0
    t0 = time.monotonic()

    logger.info(
        "  [filter scan] %d chunks, %d concurrent reads — accumulating NaN counts",
        n_chunks, n_prefetch,
    )

    with ThreadPoolExecutor(max_workers=n_prefetch) as pool:
        futures = [pool.submit(X_dask.blocks[ci].compute) for ci in range(n_chunks)]

        for ci, fut in enumerate(futures):
            chunk_rows = X_dask.chunks[0][ci]
            row_offset_end = row_offset + chunk_rows

            chunk = fut.result()
            not_fin = ~np.isfinite(chunk)
            bad_counts[row_offset:row_offset_end] = not_fin.sum(axis=1).astype(np.int32)
            nan_per_feat += not_fin.sum(axis=0).astype(np.int64)
            del chunk, not_fin

            row_offset += chunk_rows
            done = ci + 1
            eta  = (time.monotonic() - t0) / done * (n_chunks - done) / 60
            if done % 5 == 0 or done == n_chunks:
                logger.info(
                    "  [filter scan] %d/%d done — ETA: %.0f min", done, n_chunks, eta,
                )

    return bad_counts, nan_per_feat


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
# Shared step-3 residual-NaN handler (called by parquet AND zarr paths)
# ---------------------------------------------------------------------------


def _apply_residual_nan_step(
    result: "np.ndarray",
    feat_keep: "np.ndarray",
    obs_kept: "pd.DataFrame | None",
    max_residual_nan_fraction: "float | None" = 0.0,
    residual_nan_impute: "str" = "zero",
    perturbation_column: "str | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Apply step-3 residual-NaN logic to the materialised filter matrix.

    This is the single source of truth for residual-NaN handling shared by
    the parquet column-batch path and the zarr row-batch path.  Changing the
    logic here automatically applies to both.

    Three modes controlled by *max_residual_nan_fraction*:

    ``None`` (**recommended, per-well median**)
        **Per-well median mode** — no explicit step-3 drop.  The caller is
        expected to run the step-4 variance filter on the matrix with residual
        NaN still present (``np.var`` propagates NaN within a well; the median
        across wells is finite when fewer than half the wells have NaN, matching
        ``feature_variance(skipna=False)``).  After step 4, surviving NaN cells
        are imputed to 0.  Features whose NaN is concentrated in ≤50% of wells
        survive; those with majority-NaN wells are dropped by ``isfinite``.

    ``0.0`` (default)
        **Zero-tolerance drop** — any feature with ≥ 1 NaN cell is removed.
        Conservative; keeps the output fully NaN-free before the variance step.

    ``0 < f ≤ 1``
        **Fraction-threshold** — features whose residual NaN fraction exceeds
        *f* are dropped; surviving NaN cells are imputed (see
        *residual_nan_impute*).

    :param result: Materialised float32 array ``(n_cells, n_feat)``.
        Modified **in place** for the imputation modes.
    :param feat_keep: Boolean mask ``(n_original_feat,)`` tracking which
        features are still kept.  Updated in place when features are dropped.
    :param obs_kept: DataFrame of the *kept* cells (row-aligned with *result*).
        Required for perturbation imputation; may be ``None`` for zero-mode.
    :param max_residual_nan_fraction: Tolerance threshold (see above).
    :param residual_nan_impute: ``"zero"`` or ``"perturbation"``.
    :param perturbation_column: obs column naming each cell's perturbation.
    :return: ``(result, feat_keep)`` — the (possibly sliced) matrix and the
        updated keep mask.  In ``None`` mode the matrix is returned unchanged
        (NaN cells still present) so the caller's step-4 variance filter can
        see them.
    """
    # Use ~isfinite (not isnan) to also catch Inf values — Inf in a minority of
    # wells can survive step-4 isfinite(median_var) because np.var(…, inf) = nan
    # but np.median([finite,…, nan]) is finite when nan is in the minority.
    nan_per_feat = (~np.isfinite(result)).sum(axis=0)   # (n_feat,) counts NaN+Inf
    n_cells = result.shape[0]

    if max_residual_nan_fraction is None:
        # ── Per-well-median mode: pass through unchanged, step-4 handles via isfinite ──
        n_with_nan = int((nan_per_feat > 0).sum())
        if n_with_nan:
            logger.info(
                "  [residual NaN] per-well-median: %d features have residual non-finite "
                "→ step-4 variance (isfinite) decides fate",
                n_with_nan,
            )
        # No mutation — caller runs step-4 on data that may contain NaN/Inf
        return result, feat_keep

    if max_residual_nan_fraction == 0.0:
        # ── Zero-tolerance: drop any feature with residual non-finite value ──
        feat_still_nan = nan_per_feat > 0
        if feat_still_nan.any():
            n_drop = int(feat_still_nan.sum())
            logger.info(
                "  [residual NaN] zero-tolerance: dropped %d features → %d remain",
                n_drop, int(feat_keep.sum()) - n_drop,
            )
            result = result[:, ~feat_still_nan]
            _fk = np.where(feat_keep)[0]
            feat_keep[_fk[feat_still_nan]] = False

    else:
        # ── Fraction-threshold: drop only features exceeding the limit ─────
        nan_frac = nan_per_feat / max(n_cells, 1)
        feat_too_nan = nan_frac > max_residual_nan_fraction
        if feat_too_nan.any():
            n_drop = int(feat_too_nan.sum())
            logger.info(
                "  [residual NaN] fraction-mode (>%.1f%%): dropped %d → %d remain",
                max_residual_nan_fraction * 100, n_drop, int(feat_keep.sum()) - n_drop,
            )
            result = result[:, ~feat_too_nan]
            _fk = np.where(feat_keep)[0]
            feat_keep[_fk[feat_too_nan]] = False

        # ── Impute surviving residual non-finite cells (NaN + Inf) ───────────
        nan_mask = ~np.isfinite(result)
        if nan_mask.any():
            n_nan_cells = int(nan_mask.sum())
            n_nan_feats = int(nan_mask.any(axis=0).sum())

            if (residual_nan_impute == "perturbation"
                    and perturbation_column is not None
                    and obs_kept is not None
                    and perturbation_column in obs_kept.columns):
                pv = obs_kept[perturbation_column].values
                for pk in np.unique(pv):
                    grp = pv == pk
                    X_g = result[grp].copy()
                    ng  = ~np.isfinite(X_g)
                    if not ng.any():
                        continue
                    X_g[ng] = np.nan          # treat Inf as NaN for mean computation
                    mg = np.nanmean(X_g, axis=0)
                    np.nan_to_num(mg, nan=0.0, copy=False)
                    result[grp] = np.where(ng, mg[np.newaxis, :], result[grp])
                logger.info(
                    "  [residual NaN] perturbation-mean imputed %d cells"
                    " across %d features", n_nan_cells, n_nan_feats,
                )
            else:
                if residual_nan_impute == "perturbation":
                    logger.warning(
                        "  [residual NaN] perturbation impute requested but "
                        "column '%s' unavailable — falling back to zero",
                        perturbation_column,
                    )
                result[nan_mask] = 0.0
                logger.info(
                    "  [residual NaN] zero-imputed %d cells across %d features",
                    n_nan_cells, n_nan_feats,
                )

    return result, feat_keep


# ---------------------------------------------------------------------------
# Column-batch filter for parquet (all row groups parallel per feature batch)
# ---------------------------------------------------------------------------


# ── Shared variance building blocks (used by both parquet and zarr paths) ────

def _clip_bounds_mean3sd(
    sum_: "np.ndarray",
    sq: "np.ndarray",
    cnt: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Mean ± 3σ clip bounds from online sum / sum-of-squares / count.

    O(1) in memory — no sorting required.  For near-normal features this
    approximates the [0.15th, 99.85th] percentile interval; for skewed
    features it is somewhat looser, which is acceptable for clipping purposes.
    """
    mu  = np.where(cnt > 0, sum_ / np.maximum(cnt, 1.0), np.nan)
    var = np.where(cnt > 1, sq   / np.maximum(cnt, 1.0) - mu ** 2, 0.0)
    sd  = np.sqrt(np.maximum(var, 0.0))
    return mu - 3.0 * sd, mu + 3.0 * sd


def _scale_to_01(
    X: "np.ndarray",
    lo: "np.ndarray",
    hi: "np.ndarray",
) -> "np.ndarray":
    """Clip X to [lo, hi] and minmax-scale to [0, 1].

    Features where lo == hi (zero range, i.e. constant) are set to 0.
    NaN propagation: NaN in X → NaN in output (caller must mask separately).
    """
    rng   = hi - lo
    valid = rng > 0.0
    return np.where(
        valid,
        (np.clip(X, lo, hi) - lo) / np.where(valid, rng, 1.0),
        0.0,
    )


def _nanvar_from_accum(
    sum_: "np.ndarray",
    sq: "np.ndarray",
    cnt: "np.ndarray",
) -> "np.ndarray":
    """Variance from online sum / sum-of-squares / count accumulators.

    Returns NaN for groups with ≤ 1 finite observation.
    """
    mu = np.where(cnt > 0, sum_ / np.maximum(cnt, 1.0), np.nan)
    return np.where(cnt > 1, sq / np.maximum(cnt, 1.0) - mu ** 2, np.nan)


def _scaled_nanvar_per_group(X_grp: "np.ndarray") -> "np.ndarray":
    """Per-well clip-and-scale nanvar for the in-memory variance filter path.

    Uses the same mean ± 3σ clip bounds and _scale_to_01 helper as the zarr
    streaming path so both paths are guaranteed to produce identical results.

    :param X_grp: float64 array (n_cells, n_feat) for one group (well).
    :return: float64 array (n_feat,) of scaled within-well nanvar values.
    """
    fm   = np.isfinite(X_grp)
    cnt  = fm.sum(axis=0).astype(np.float64)
    sum_ = np.nansum(X_grp,      axis=0)
    sq   = np.nansum(X_grp ** 2, axis=0)
    lo, hi = _clip_bounds_mean3sd(sum_, sq, cnt)
    Xs = _scale_to_01(X_grp, lo, hi)
    Xs = np.where(fm, Xs, np.nan)          # restore NaN positions
    return np.nanvar(Xs, axis=0, ddof=0)


def _apply_variance_filter(
    result: "np.ndarray",
    feat_keep: "np.ndarray",
    obs_kept: "pd.DataFrame | None",
    by: "list | None" = None,
    min_variance: "float | None" = 0.001,
    max_variance: "float | None" = None,
    max_residual_nan_fraction: "float | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Step-4 variance filter — in-memory path (parquet + zarr after materialise).

    Calls _scaled_nanvar_per_group (mean±3σ clip → [0,1] scale → nanvar) per
    group, takes nanmedian across groups, applies threshold.  The zarr streaming
    path uses the same _clip_bounds_mean3sd / _scale_to_01 / _nanvar_from_accum
    helpers so both paths are identical in logic, differing only in how data is
    read (full array vs chunked zarr).

    :param result: Float32 array (n_cells, n_feat).
    :param feat_keep: Boolean mask over original features; updated on drop.
    :param obs_kept: DataFrame of kept cells for groupby; None = no grouping.
    :param by: obs columns to stratify by (e.g. ['plate', 'well']).
    :param min_variance: Threshold on [0,1]-scaled variance. None = disabled.
    :param max_variance: Upper threshold. None = disabled.
    :param max_residual_nan_fraction: Controls post-filter NaN imputation.
    :return: (result, feat_keep, feat_var).
    """
    n_feat = result.shape[1]
    feat_var = np.zeros(n_feat, dtype=np.float64)

    if result.shape[0] > 1:
        if by and obs_kept is not None:
            group_vars = []
            for _, grp_idx in obs_kept.groupby(by, observed=True).groups.items():
                X_grp = result[grp_idx.values].astype(np.float64)
                if len(X_grp) > 1:
                    group_vars.append(_scaled_nanvar_per_group(X_grp))
            if group_vars:
                feat_var = np.nanmedian(np.stack(group_vars), axis=0)
        else:
            feat_var = _scaled_nanvar_per_group(result.astype(np.float64))

    _min_var = min_variance if min_variance is not None else 0.0
    feat_var_keep = np.isfinite(feat_var) & (feat_var >= _min_var)
    if max_variance is not None:
        feat_var_keep &= feat_var <= max_variance

    if not feat_var_keep.all():
        n_drop = int((~feat_var_keep).sum())
        logger.info(
            "  [variance filter] dropped %d features (scaled var < %.4g or not finite)"
            " → %d remain",
            n_drop, _min_var, int(feat_var_keep.sum()),
        )
        result = result[:, feat_var_keep]
        _fk = np.where(feat_keep)[0]
        feat_keep[_fk[~feat_var_keep]] = False

    # Per-well-median mode: impute surviving non-finite cells (NaN + Inf) to 0
    if max_residual_nan_fraction is None:
        nan_mask = ~np.isfinite(result)
        if nan_mask.any():
            logger.info(
                "  [variance filter] per-well-median: zero-imputed %d residual "
                "non-finite cells across %d features",
                int(nan_mask.sum()), int(nan_mask.any(axis=0).sum()),
            )
            result[nan_mask] = 0.0

    return result, feat_keep, feat_var


def _apply_filter_steps_1_2(
    bad_counts: "np.ndarray",
    nan_per_feat: "np.ndarray",
    label_mask: "np.ndarray",
    n_cells: int,
    n_feat: int,
    max_feature_nan_fraction: "float | None" = 0.50,
    max_fraction_not_finite: "float | None" = 0.25,
) -> "tuple[np.ndarray, np.ndarray]":
    """Apply steps 1 and 2 from accumulated scan statistics.

    Decouples the decision logic from the I/O mechanism: both the parquet
    column-batch scanner and the zarr row-batch scanner accumulate
    ``bad_counts`` (per-cell NaN count) and ``nan_per_feat`` (per-feature NaN
    count) during their streaming read, then call this function identically.

    Step 1 — feature NaN filter:
        Drop features where the NaN fraction across all cells exceeds
        ``max_feature_nan_fraction``.  Disabled when ``None``.

    Step 2 — cell filter:
        Drop cells whose NaN count across all original features exceeds
        ``max_fraction_not_finite × n_feat``.  The denominator is always
        the *total* original feature count (not step-1 survivors) so that
        cells are not unfairly penalised for having NaN in features that were
        already going to be dropped.

    :param bad_counts: Per-cell NaN count over all features, shape ``(n_obs,)``.
    :param nan_per_feat: Per-feature NaN count over all cells, shape ``(n_feat,)``.
    :param label_mask: Boolean obs-only mask applied before both steps.
    :param n_cells: Total cell count (len of ``bad_counts``).
    :param n_feat: Total feature count (len of ``nan_per_feat``).
    :param max_feature_nan_fraction: Step-1 threshold; ``None`` skips step 1.
    :param max_fraction_not_finite: Step-2 threshold; ``None`` keeps all cells.
    :return: ``(feat_pass1, cell_keep)`` — feature and cell boolean masks.
    """
    # Step 1
    feat_pass1 = np.ones(n_feat, dtype=bool)
    if max_feature_nan_fraction is not None:
        feat_nan_frac = nan_per_feat / max(n_cells, 1)
        feat_pass1 = feat_nan_frac <= max_feature_nan_fraction
        n_s1 = int((~feat_pass1).sum())
        if n_s1:
            logger.info(
                "  [step 1] feature NaN filter (>%.0f%%): %d / %d features dropped",
                max_feature_nan_fraction * 100, n_s1, n_feat,
            )

    # Step 2 — denominator = n_feat (total), never step-1-survivors
    max_bad = (
        int(n_feat * max_fraction_not_finite)
        if max_fraction_not_finite is not None else n_feat
    )
    cell_keep = label_mask & (bad_counts <= max_bad)
    logger.info(
        "  [step 2] cell filter (max_bad=%d, denom=%d): %s / %s cells kept",
        max_bad, n_feat, f"{int(cell_keep.sum()):,}", f"{n_cells:,}",
    )

    return feat_pass1, cell_keep


def _apply_filter_post_materialise(
    result: "np.ndarray",
    feat_keep: "np.ndarray",
    obs_kept: "pd.DataFrame | None",
    by: "list | None" = None,
    min_variance: "float | None" = 0.001,
    max_variance: "float | None" = None,
    max_residual_nan_fraction: "float | None" = None,
    residual_nan_impute: "str" = "zero",
    perturbation_column: "str | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Apply variance selection then NaN cleanup on the materialised filter matrix.

    This is the **single entry point** used by both the parquet column-batch
    path and the zarr row-batch path after ``_streaming_materialise``.  Having
    one function guarantees both paths receive identical post-materialisation
    processing and can never diverge.

    **Ordering: variance first, NaN cleanup second.**

    Step 4 (``_apply_variance_filter``) — per-well nanvar + nanmedian + isfinite:
        Uses ``np.nanvar`` so that a feature with occasional NaN cells still
        gets a finite within-well variance (NaN cells are ignored, not
        propagated).  ``np.nanmedian`` across wells means only features that
        are *entirely* NaN in *most* wells are dropped.  This selects features
        based on biological signal independently of NaN sparsity.

    Step 3 (``_apply_residual_nan_step``) — residual-NaN cleanup:
        Applied *after* variance selection so we know which features are worth
        keeping before deciding how to handle their NaN cells.

        * ``0.0`` *(recommended)*: zero-tolerance — drop any feature with ≥1
          NaN cell remaining after the cell filter.  Features whose NaN cells
          were all in "bad" cells (already removed by the cell filter) survive
          cleanly.  Produces a fully finite matrix without imputation.
        * ``None``: no-op — NaN cells remain (useful when the caller will
          impute them externally, e.g. during YJ transform).
        * ``> 0``: fraction-threshold drop + imputation of survivors.

    :return: ``(result, feat_keep, feat_var)`` — clean matrix, updated feature
        mask, and per-feature variance array.
    """
    # Step 4 first: variance filter on NaN-containing data (nanvar ignores NaN)
    result, feat_keep, feat_var = _apply_variance_filter(
        result, feat_keep,
        obs_kept=obs_kept,
        by=by,
        min_variance=min_variance,
        max_variance=max_variance,
        max_residual_nan_fraction=max_residual_nan_fraction,
    )
    # Step 3 second: NaN cleanup on variance-selected features
    result, feat_keep = _apply_residual_nan_step(
        result, feat_keep,
        obs_kept=obs_kept,
        max_residual_nan_fraction=max_residual_nan_fraction,
        residual_nan_impute=residual_nan_impute,
        perturbation_column=perturbation_column,
    )
    return result, feat_keep, feat_var


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
    max_residual_nan_fraction: float | None = 0.0,
    residual_nan_impute: "str | None" = "zero",
    perturbation_column: "str | None" = None,
    output_zarr_path: "str | None" = None,
    streaming_chunk_gb: float = 2.0,
) -> "tuple[np.ndarray | None, np.ndarray, np.ndarray, pd.DataFrame]":
    """Two-pass streaming filter.

    When *output_zarr_path* is provided, pass 2 writes filtered rows
    directly to a zarr store (no ``np.empty`` pre-allocation) and steps
    3+4 (variance + residual-NaN) are computed from that zarr in
    ``streaming_chunk_gb``-sized row-chunks.  Return value is the same
    except the first element is ``None``; the caller reads the result
    from the zarr as a dask-backed AnnData.
    """
    """Two-pass sequential streaming filter for parquet files (local or S3).

    Works for both parquet AND any format accepted by the PyArrow dataset API.
    Uses sequential HTTP GETs (one per S3 object) rather than per-column range
    requests, and handles files with mismatched schemas automatically.

    Uses sequential S3 reads (one HTTP GET per file rather than one range
    request per column chunk) and handles files with different schemas by
    letting the dataset scanner align them automatically.

    Pass 1 — stream all batches once: accumulate per-cell AND per-feature
    NaN/Inf counts.

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
        produced by ``_read_map_inputs``.
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
    import os
    import pyarrow.dataset as _ds
    import pyarrow.fs    as _pafs

    if feat_cols is None:
        feat_cols = list(sources[0]["feat_cols"])
    n_feat  = len(feat_cols)
    n_cells = len(obs_df)
    max_bad = int(n_feat * max_fraction_not_finite) if max_fraction_not_finite is not None else n_feat

    # ── Per-file row offsets (no I/O — derived from row_group_sizes) ─────────
    # Process one source file at a time so peak RAM = one file, not all files.
    # Peak for 12 × 1.4M-cell parquets: ~50 GB instead of ~700 GB.
    _file_offsets = [0]
    for _src in sources:
        _file_offsets.append(_file_offsets[-1] + sum(_src["row_group_sizes"]))

    logger.info(
        "  [dataset filter] %d files, %d features, per-file streaming …",
        len(sources), n_feat,
    )

    # ── Memory budget → batch_size + n_workers ───────────────────────────────
    # Safety factor 3: one batch being read + one being processed + zarr buffer.
    # batch_readahead=1 means ONE extra batch pre-fetched → always ≤ 2 batches
    # in PyArrow's internal buffers.  Total peak = 2×batch + to_pandas copy =
    # 3×batch ≤ budget_gb.
    try:
        import psutil as _psutil
        _avail_gb = _psutil.virtual_memory().available / 1e9
    except Exception:
        _avail_gb = 64.0
    _budget_gb = float(max_memory_gb) if max_memory_gb is not None else _avail_gb * 0.40
    n_workers = min(
        max(1, int(os.cpu_count() or 4)),
        len(sources),
        max(1, int(_budget_gb / max(0.1, n_feat * 4 * 3 / 1e9))),
    )
    # batch_size: each worker gets a budget-bounded share.
    # Safety factor 10: accounts for PyArrow internal buffer (×1) + numpy copy
    # (×1) + Python virtual-memory baseline for heavy libraries (×8 headroom).
    # Without this headroom, even a 200 GB RLIMIT_AS budget can be exhausted by
    # Python's own mmap'd libraries before any data is allocated.
    # Hard cap at 500K rows: efficient for S3 (one HTTP GET per batch) without
    # allocating excessively large contiguous buffers.
    _bytes_per_row = n_feat * 4   # float32
    _safety        = 10
    batch_size = max(10_000, min(500_000, int(_budget_gb * 1e9 / (n_workers * _bytes_per_row * _safety))))
    # batch_readahead=0: no pre-fetching.  With large parquet files (76+ GB),
    # PyArrow column decompressors consume ~80–160 GB VmSize per batch.
    # With readahead=1, two batches are decompressed simultaneously, which can
    # double VmSize and hit the RLIMIT_AS limit.  Sequential read is slightly
    # slower for S3 (no pipelining) but critical for local large files.
    _batch_ra  = 0
    _frag_ra   = 1   # one file at a time — no cross-file read-ahead
    logger.info(
        "  [scanner] budget=%.0f GB  n_workers=%d  batch_size=%s  "
        "peak_per_worker≈%.1f GB",
        _budget_gb, n_workers, f"{batch_size:,}",
        n_workers * batch_size * _bytes_per_row * _safety / 1e9,
    )

    # ── Group-key helper (used in both passes) ───────────────────────────────
    _by_list = ([by] if isinstance(by, str) else list(by)) if by else []

    def _group_keys_for(obs_slice: "pd.DataFrame") -> "np.ndarray":
        return (obs_slice[_by_list].astype(str).agg("-".join, axis=1).values
                if _by_list else np.full(len(obs_slice), "__all__"))

    # ── Pass 1: NaN scan + raw stats for clip bounds ──────────────────────────
    # Each file is processed by one worker thread; stats merged afterwards.
    # n_workers files read in parallel → saturates S3 bandwidth.
    bad_counts        = np.zeros(n_cells, dtype=np.int32)
    nan_per_feat      = np.zeros(n_feat,  dtype=np.int64)
    nan_per_feat_bad  = np.zeros(n_feat,  dtype=np.int64)
    import threading, concurrent.futures as _cf
    _lock = threading.Lock()

    def _pass1_file(fi: int, src: dict) -> None:
        """Pass 1: NaN scan only.  No group stats — avoids 24+ GB peak allocation
        when PyArrow's internal column-page buffers already consume ~196 GB of
        virtual memory for large local parquet files (per VmSize measurement).
        Clip bounds for the variance filter are computed from the materialised
        zarr in passes A+B after pass 2.
        """
        _pa_fs, _pa_path = _pafs.FileSystem.from_uri(src["path"])
        _ds_fi  = _ds.dataset(_pa_path, filesystem=_pa_fs, format="parquet")
        _scan   = _ds_fi.scanner(
            columns=feat_cols, batch_size=batch_size,
            use_threads=True, batch_readahead=_batch_ra, fragment_readahead=1,
        )
        _file_bad   = np.zeros(_file_offsets[fi + 1] - _file_offsets[fi], dtype=np.int32)
        _file_nf    = np.zeros(n_feat, dtype=np.int64)
        _file_nfb   = np.zeros(n_feat, dtype=np.int64)
        _local_off  = 0

        for _batch in _scan.to_batches():
            _n_b = len(_batch)
            _X_b = _batch.to_pandas().to_numpy(np.float32)
            del _batch
            _nf_b   = ~np.isfinite(_X_b)
            _pc_nan = _nf_b.sum(axis=1).astype(np.int32)
            _file_bad[_local_off:_local_off + _n_b] = _pc_nan
            _file_nf  += _nf_b.sum(axis=0).astype(np.int64)
            _bad_m     = _pc_nan > max_bad
            if _bad_m.any():
                _file_nfb += _nf_b[_bad_m].sum(axis=0).astype(np.int64)
            del _X_b, _nf_b
            _local_off += _n_b

        # Merge into shared accumulators (lock protects concurrent writes).
        # Use slice assignment (__setitem__) instead of += so Python does not
        # mark nan_per_feat as a local variable in the nested function scope.
        with _lock:
            bad_counts[_file_offsets[fi]:_file_offsets[fi + 1]] = _file_bad
            nan_per_feat[:]     += _file_nf
            nan_per_feat_bad[:] += _file_nfb

    t0 = time.monotonic()
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as _exe:
        _futs = [_exe.submit(_pass1_file, fi, src) for fi, src in enumerate(sources)]
        for fi, _fut in enumerate(_futs):
            _fut.result()   # re-raise any exception from the worker
            elapsed = time.monotonic() - t0
            cells_done = _file_offsets[fi + 1]
            eta = elapsed / max(cells_done, 1) * max(n_cells - cells_done, 0) / 60
            logger.info("  [pass 1/2] file %d/%d done — %.0f%% — ETA: %.0f min",
                        fi + 1, len(sources), cells_done / n_cells * 100, eta)

    # ── Steps 1 + 2: shared filter logic ─────────────────────────────────────
    feat_pass1, cell_keep = _apply_filter_steps_1_2(
        bad_counts, nan_per_feat, label_mask, n_cells, n_feat,
        max_feature_nan_fraction, max_fraction_not_finite,
    )
    feat_keep = feat_pass1   # pass 2 materialises step-1-surviving features

    n_cells_out = int(cell_keep.sum())
    n_feat_out  = int(feat_keep.sum())
    logger.info(
        "  [feature NaN + cell NaN filters done] %s / %s cells · %s / %s features"
        " → materialising …",
        f"{n_cells_out:,}", f"{n_cells:,}", f"{n_feat_out:,}", f"{n_feat:,}",
    )

    # ── Pass 2: parallel materialise to local zarr or numpy ──────────────────
    # Each worker thread reads its assigned file, filters rows/cols, and writes
    # the result to a pre-allocated output store (zarr slab or numpy slice).
    # Scaled-variance and residual-NaN computation happen in passes A+B below
    # (reading the already-local zarr) — keeps pass-2 peak memory small.
    kept_feat_cols = [feat_cols[i] for i, k in enumerate(feat_keep) if k]
    if output_zarr_path is not None:
        import zarr as _zarr
        _zg = _zarr.open_group(output_zarr_path, mode="a")
        _zX = _zg.require_dataset(
            "X", shape=(n_cells_out, n_feat_out), dtype="float32",
            chunks=(min(50_000, n_cells_out), n_feat_out), overwrite=True,
        )
        result = None
    else:
        result = np.empty((n_cells_out, n_feat_out), dtype=np.float32)
        _zX    = None

    # Pre-compute per-file output-row offsets (disjoint slabs, thread-safe)
    _n_out_per_file = [int(cell_keep[_file_offsets[fi]:_file_offsets[fi + 1]].sum())
                       for fi in range(len(sources))]
    _out_offsets = [0]
    for _n in _n_out_per_file:
        _out_offsets.append(_out_offsets[-1] + _n)

    def _pass2_file(fi: int, src: dict) -> None:
        _pa_fs, _pa_path = _pafs.FileSystem.from_uri(src["path"])
        _ds_fi = _ds.dataset(_pa_path, filesystem=_pa_fs, format="parquet")
        _scan  = _ds_fi.scanner(
            columns=kept_feat_cols, batch_size=batch_size,
            use_threads=True, batch_readahead=_batch_ra, fragment_readahead=1,
        )
        _global_off = _file_offsets[fi]
        _out_row    = _out_offsets[fi]

        for _batch in _scan.to_batches():
            _n_b    = len(_batch)
            _cell_b = cell_keep[_global_off:_global_off + _n_b]
            _global_off += _n_b
            if not _cell_b.any():
                del _batch
                continue
            _X_b = _batch.to_pandas().values.astype(np.float32)
            del _batch
            _X_f = _X_b[_cell_b]
            del _X_b
            _n_f = _X_f.shape[0]
            if _zX is not None:
                _zX[_out_row:_out_row + _n_f] = _X_f
            else:
                result[_out_row:_out_row + _n_f] = _X_f
            _out_row += _n_f
            del _X_f

    t0 = time.monotonic()
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as _exe:
        _futs2 = [_exe.submit(_pass2_file, fi, src) for fi, src in enumerate(sources)]
        for fi, _fut in enumerate(_futs2):
            _fut.result()
            elapsed = time.monotonic() - t0
            cells_done = _file_offsets[fi + 1]
            eta = elapsed / max(cells_done, 1) * max(n_cells - cells_done, 0) / 60
            logger.info("  [pass 2/2] file %d/%d — %.0f%% — ETA: %.0f min",
                        fi + 1, len(sources), cells_done / n_cells * 100, eta)

    obs_kept = obs_df.iloc[cell_keep].reset_index(drop=True)

    # Index of surviving zarr columns (used to select correct cols from zarr)
    _zarr_col_idx: "np.ndarray | None" = None

    if _zX is not None:
        # ── Steps 3+4 from zarr: two streaming passes using shared helpers ─────
        # Pass A: raw stats per group → clip bounds.
        # Pass B: scaled stats per group → variance + residual NaN.
        # Both read the LOCAL zarr written in pass 2 — much faster than reading
        # the S3 zarr that the old code used for this.
        logger.info("  [pass 2/2 done] streamed to zarr — computing variance "
                    "from zarr in %.0f GB chunks", streaming_chunk_gb)
        # Cap the variance chunk at 2 GB of float64 regardless of budget.
        # Each chunk requires ~4× its size in temporary arrays (_Xg copy,
        # _Xg² copy, nansum copy) and PyArrow column decompressors may still
        # hold ~180 GB VmSize from passes 1+2 — small chunks keep total under
        # the RLIMIT_AS ceiling.
        _var_chunk_gb = min(streaming_chunk_gb, 2.0)
        chunk_rows = max(1, int(_var_chunk_gb * 1e9 / (n_feat_out * 8)))

        def _zkeys_for(obs_slice: "pd.DataFrame") -> "np.ndarray":
            return (_keys_for_obs := (_by_list and obs_slice[_by_list].astype(str).agg("-".join, axis=1).values)
                    or np.full(len(obs_slice), "__all__"))

        # Helper that avoids the walrus-assignment; just call group_keys_for
        def _zk(obs_sl):
            return _group_keys_for(obs_sl)

        # Pass A — accumulate raw stats per group for clip bounds
        _raw_sum:  "dict[str, np.ndarray]" = {}
        _raw_sq:   "dict[str, np.ndarray]" = {}
        _raw_cnt:  "dict[str, np.ndarray]" = {}
        for _r0 in range(0, n_cells_out, chunk_rows):
            _r1  = min(_r0 + chunk_rows, n_cells_out)
            _Xc  = np.asarray(_zX[_r0:_r1], dtype=np.float64)
            _keys = _zk(obs_kept.iloc[_r0:_r1])
            for _g in np.unique(_keys):
                _Xg = _Xc[_keys == _g]
                _fm = np.isfinite(_Xg)
                _Xgc = np.where(_fm, _Xg, 0.0)       # float64, NaN→0, no nansum copy
                _raw_sum[_g] = _raw_sum.get(_g, np.zeros(n_feat_out)) + _Xgc.sum(axis=0)
                _raw_sq[_g]  = _raw_sq.get(_g,  np.zeros(n_feat_out)) + (_Xgc * _Xgc).sum(axis=0)
                _raw_cnt[_g] = _raw_cnt.get(_g, np.zeros(n_feat_out)) + _fm.sum(axis=0).astype(np.float64)
                del _Xgc
            del _Xc

        _gclip = {_g: _clip_bounds_mean3sd(_raw_sum[_g], _raw_sq[_g], _raw_cnt[_g])
                  for _g in _raw_sum}

        # Pass B — scaled stats per group + residual NaN flag
        _sc_sum:  "dict[str, np.ndarray]" = {}
        _sc_sq:   "dict[str, np.ndarray]" = {}
        _sc_cnt:  "dict[str, np.ndarray]" = {}
        _nan_feat_all = np.zeros(n_feat_out, dtype=bool)
        for _r0 in range(0, n_cells_out, chunk_rows):
            _r1  = min(_r0 + chunk_rows, n_cells_out)
            _Xc  = np.asarray(_zX[_r0:_r1], dtype=np.float64)
            _nan_feat_all |= ~np.isfinite(_Xc).all(axis=0)
            _keys = _zk(obs_kept.iloc[_r0:_r1])
            for _g in np.unique(_keys):
                _Xg = _Xc[_keys == _g]
                _fm = np.isfinite(_Xg)
                _Xs = np.where(_fm, _scale_to_01(_Xg, *_gclip[_g]), 0.0)   # NaN→0
                _sc_sum[_g] = _sc_sum.get(_g, np.zeros(n_feat_out)) + _Xs.sum(axis=0)
                _sc_sq[_g]  = _sc_sq.get(_g,  np.zeros(n_feat_out)) + (_Xs * _Xs).sum(axis=0)
                _sc_cnt[_g] = _sc_cnt.get(_g, np.zeros(n_feat_out)) + _fm.sum(axis=0).astype(np.float64)
                del _Xs
            del _Xc

        _gvars = [_nanvar_from_accum(_sc_sum[_g], _sc_sq[_g], _sc_cnt[_g]) for _g in _sc_sum]
        feat_var  = np.nanmedian(np.stack(_gvars), axis=0) if _gvars else np.zeros(n_feat_out)
        _min_var  = min_variance if min_variance is not None else 0.0
        _vk       = np.isfinite(feat_var) & (feat_var >= _min_var)
        if max_variance is not None:
            _vk &= feat_var <= max_variance
        if not _vk.all():
            logger.info(
                "  [variance filter] dropped %d features (scaled var < %.4g) → %d remain",
                int((~_vk).sum()), _min_var, int(_vk.sum()),
            )
        _fk_idx = np.where(feat_keep)[0]
        feat_keep[_fk_idx[~_vk]] = False
        feat_var  = feat_var[_vk]

        _kept_idx = np.where(_vk)[0]
        _nan_feat = _nan_feat_all[_kept_idx]
        if _nan_feat.any() and max_residual_nan_fraction == 0.0:
            logger.info("  [residual NaN] zero-tolerance: dropped %d features", int(_nan_feat.sum()))
            _fk2 = np.where(feat_keep)[0]
            feat_keep[_fk2[_nan_feat]] = False
            _kept_idx = _kept_idx[~_nan_feat]

        # Record surviving zarr column indices so the caller selects the correct
        # columns (not simply the first _n_final cols, which would be wrong if
        # the variance filter dropped non-contiguous columns).
        _zarr_col_idx = _kept_idx

        # result stays None; caller builds dask AnnData from zarr
        result = None
    else:
        logger.info("  [pass 2/2 done] materialised %.1f GB", result.nbytes / 1e9)
        result, feat_keep, feat_var = _apply_filter_post_materialise(
            result, feat_keep, obs_kept,
            by=by,
            min_variance=min_variance,
            max_variance=max_variance,
            max_residual_nan_fraction=max_residual_nan_fraction,
            residual_nan_impute=residual_nan_impute,
            perturbation_column=perturbation_column,
        )

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

    return result, cell_keep, feat_keep, report_df, _zarr_col_idx


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
