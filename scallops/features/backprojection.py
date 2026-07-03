"""Backprojection from TVN-normalized embeddings to the original feature space.

After Typical Variation Normalization (TVN), each observation lives in a
PCA-rotated, covariance-aligned space.  The parameters needed to reverse this
transformation are stored by :func:`~scallops.features.normalize.typical_variation_normalization`:

* ``uns["pca"]["PCs"]``              — PCA components (n_components × n_features)
* ``uns["pca"]["mean"]``             — PCA training mean (n_features,)
* ``uns["tvn_pre_scale_mean"]``      — reference mean before z-scoring (n_features,)
* ``uns["tvn_pre_scale_std"]``       — reference std before z-scoring (n_features,)
* ``uns["covariance_alignment_inv"]``— {group → inverse alignment matrix}
* ``uns["normalization_arguments"]`` — {reference_query, by}
* ``varm["PCs"]``                    — transposed components (n_features × n_components)

Design principle
----------------
Feature importance is *never* derived by testing individual z-score features
(which are correlated).  Instead:

1. The centroid difference in TVN/PCA space is computed — the PCs are
   orthogonal by construction, so this is a clean, decorrelated signal.
2. An optional statistical filter can be applied *on the PC scores*
   (orthogonal) to retain only significant components.
3. The (filtered) centroid difference is projected back to z-score feature
   space.  This backprojected vector is the feature importance ranking.

Public API
----------
backproject_tvn
    Reverse the TVN transform to recover z-score or original-scale profiles.
top_features_from_backprojection
    Given a set of perturbations or a cluster assignment, rank original
    features by how much they discriminate the query from a reference set.

Authors:
    - The SCALLOPS development team
"""

import warnings
from collections.abc import Sequence
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_tvn_uns(data: anndata.AnnData) -> None:
    """Raise :class:`ValueError` if any required backprojection parameter is absent.

    :param data: AnnData whose ``uns`` dict is inspected.
    :raises ValueError: When ``uns["pca"]``, ``uns["tvn_pre_scale_mean"]``, or
        ``uns["tvn_pre_scale_std"]`` are missing, or when ``uns["pca"]`` is
        missing the ``"PCs"`` or ``"mean"`` keys.
    """
    required_uns = ("pca", "tvn_pre_scale_mean", "tvn_pre_scale_std")
    missing = [k for k in required_uns if k not in data.uns]
    if missing:
        raise ValueError(
            f"Cannot backproject: missing uns keys {missing!r}.  "
            "Ensure the AnnData was produced by typical_variation_normalization "
            "(and that downstream pipeline steps propagated uns correctly)."
        )
    pca_info = data.uns["pca"]
    for k in ("PCs", "mean"):
        if k not in pca_info:
            raise ValueError(
                f"Cannot backproject: uns['pca']['{k}'] is missing.  "
                "This key is set by typical_variation_normalization."
            )


def _resolve_cov_alignment(
    cov_alignment_inv: dict,
    group: str | None,
) -> np.ndarray | None:
    """Return the inverse covariance-alignment matrix for *group*.

    :param cov_alignment_inv: Dict mapping group keys to inverse alignment
        matrices, as stored in ``uns["covariance_alignment_inv"]``.
    :param group: Key to look up.  If *None* and exactly one group exists it
        is used automatically.  If *None* and multiple groups exist the
        alignment inversion step is skipped with a :class:`UserWarning`.
    :return: 2-D alignment inverse matrix, or *None* to skip the step.
    :raises KeyError: When *group* is specified but not present in
        *cov_alignment_inv*.
    """
    if not cov_alignment_inv:
        return None
    if group is not None:
        key = str(group)
        if key not in cov_alignment_inv:
            raise KeyError(
                f"Group {group!r} not found in uns['covariance_alignment_inv']. "
                f"Available: {list(cov_alignment_inv)}"
            )
        return np.asarray(cov_alignment_inv[key], dtype=np.float64)
    keys = list(cov_alignment_inv)
    if len(keys) == 1:
        return np.asarray(cov_alignment_inv[keys[0]], dtype=np.float64)
    warnings.warn(
        f"TVN was applied with multiple groups ({keys!r}) but `group` was not "
        "specified for backprojection.  The covariance-alignment inversion step "
        "is skipped.  Pass `group=<key>` to apply a group-specific inversion.",
        UserWarning,
        stacklevel=3,
    )
    return None


# ---------------------------------------------------------------------------
# Core backprojection
# ---------------------------------------------------------------------------


def backproject_tvn(
    data: anndata.AnnData,
    group: str | None = None,
    to_original_scale: bool = False,
) -> np.ndarray:
    """Reverse the TVN transform to recover feature-space profiles.

    The TVN forward transform is:

    1. **Z-score**:          ``X_z = (X − pre_mean) / pre_std``
    2. **PCA**:              ``X_p = (X_z − pca_mean) @ PCs.T``
    3. **Cov alignment**:    ``X_t[g] = X_p[g] @ fwd_mat[g]``  *(if ``by`` was set)*

    The inverse is applied in reverse order:

    1. **Inv cov alignment**: ``X_p = X_t @ cov_alignment_inv``
    2. **Inv PCA**:           ``X_z = X_p @ PCs + pca_mean``
    3. **Inv z-score**:       ``X   = X_z * pre_std + pre_mean`` *(only if to_original_scale)*

    :param data: AnnData produced by
        :func:`~scallops.features.normalize.typical_variation_normalization`.
        Can be cell-level or perturbation-level (after
        :func:`~scallops.features.agg.agg_features`).  Must contain TVN
        backprojection parameters in ``uns``.
    :param group: Key in ``uns["covariance_alignment_inv"]`` to use for the
        inverse covariance-alignment step.  If *None* and a single group
        exists it is used automatically.  With multiple groups and no *group*
        specified the alignment inversion is skipped with a warning.
    :param to_original_scale: If *True*, also reverse the initial z-scoring step
        to recover the original measurement scale.  Defaults to *False*
        (returns z-score space, which is scale-independent and easier to
        interpret across features).
    :return: Array of shape ``(n_obs, n_features)`` in z-score or original
        feature scale, where *n_features* corresponds to ``data.var.index``.
    """
    _validate_tvn_uns(data)

    pca_info = data.uns["pca"]
    # Use float32 for large arrays to halve memory; PCs are typically float64
    # but casting them to float32 for the matmul is safe given the tolerance of
    # the downstream feature-importance ranking.
    X = np.asarray(data.X)
    if X.dtype == np.float32:
        compute_dtype = np.float32
    else:
        compute_dtype = np.float64

    PCs = np.asarray(pca_info["PCs"], dtype=compute_dtype)       # (n_pcs, n_features)
    pca_mean = np.asarray(pca_info["mean"], dtype=compute_dtype) # (n_features,)

    if X.dtype != compute_dtype:
        X = X.astype(compute_dtype)

    # --- Inverse covariance alignment ---
    cov_alignment_inv = data.uns.get("covariance_alignment_inv", {})
    if cov_alignment_inv:
        cov_inv_mat = _resolve_cov_alignment(cov_alignment_inv, group)
        if cov_inv_mat is not None:
            X = X @ np.asarray(cov_inv_mat, dtype=compute_dtype)

    # --- Inverse PCA ---
    # Forward: X_p = (X_z - pca_mean) @ PCs.T    [PCs shape: n_pcs × n_features]
    # Inverse: X_z = X_p @ PCs + pca_mean
    X_zscored = X @ PCs + pca_mean  # (n_obs, n_features)

    if not to_original_scale:
        return X_zscored

    # --- Inverse z-score ---
    pre_mean = np.asarray(data.uns["tvn_pre_scale_mean"], dtype=compute_dtype)
    pre_std = np.asarray(data.uns["tvn_pre_scale_std"], dtype=compute_dtype)
    pre_std = np.where(pre_std == 0.0, compute_dtype(1.0), pre_std)
    return X_zscored * pre_std + pre_mean


# ---------------------------------------------------------------------------
# Observation mask builders
# ---------------------------------------------------------------------------


def _build_obs_mask(
    data: anndata.AnnData,
    genes: Sequence[str] | None,
    perturbation_column: str,
    cluster_labels: np.ndarray | pd.Series | None,
    cluster_values: str | int | Sequence | None,
) -> np.ndarray | None:
    """Return a boolean array selecting observations, or *None* if no selector given.

    :param data: AnnData being queried.
    :param genes: Perturbation names matched against ``data.obs[perturbation_column]``.
    :param perturbation_column: Column in ``obs`` that identifies perturbations.
    :param cluster_labels: Cluster-assignment array (one entry per observation).
    :param cluster_values: Cluster value(s) to select when *cluster_labels* is given.
    :return: Boolean array of length ``n_obs``, or *None* when all inputs are *None*.
    """
    if genes is not None:
        if perturbation_column not in data.obs.columns:
            raise ValueError(
                f"Column {perturbation_column!r} not found in obs.  "
                f"Available columns: {list(data.obs.columns)}"
            )
        return data.obs[perturbation_column].isin(genes).values
    if cluster_values is not None:
        if cluster_labels is None:
            raise ValueError(
                "`cluster_labels` must be provided when `cluster_values` is given."
            )
        cluster_arr = np.asarray(cluster_labels)
        if len(cluster_arr) != data.n_obs:
            raise ValueError(
                f"cluster_labels length ({len(cluster_arr)}) != n_obs ({data.n_obs})"
            )
        if not isinstance(cluster_values, (list, tuple, np.ndarray)):
            cluster_values = [cluster_values]
        return np.isin(cluster_arr, cluster_values)
    return None


# ---------------------------------------------------------------------------
# PC-level statistical tests (orthogonal space)
# ---------------------------------------------------------------------------


def _pc_ttest(
    query: np.ndarray, ref: np.ndarray
) -> np.ndarray:
    """Vectorised Welch t-test on each PC dimension (query vs. reference).

    :param query: PC scores for the query set, shape ``(n_query, n_pcs)``.
    :param ref: PC scores for the reference set, shape ``(n_ref, n_pcs)``.
    :return: p-value array of length ``n_pcs``.
    """
    result = stats.ttest_ind(query, ref, axis=0, equal_var=False)
    return result.pvalue


def _pc_mannwhitney(
    query: np.ndarray, ref: np.ndarray
) -> np.ndarray:
    """Mann-Whitney U test on each PC dimension (query vs. reference).

    :param query: PC scores for the query set, shape ``(n_query, n_pcs)``.
    :param ref: PC scores for the reference set, shape ``(n_ref, n_pcs)``.
    :return: p-value array of length ``n_pcs``.
    """
    n_pcs = query.shape[1]
    pvalues = np.zeros(n_pcs)
    for i in range(n_pcs):
        res = stats.mannwhitneyu(query[:, i], ref[:, i], alternative="two-sided")
        pvalues[i] = res.pvalue
    return pvalues


# ---------------------------------------------------------------------------
# Public analysis function
# ---------------------------------------------------------------------------


def top_features_from_backprojection(
    data: anndata.AnnData,
    genes: Sequence[str] | None = None,
    perturbation_column: str = "gene_symbol",
    cluster_labels: np.ndarray | pd.Series | None = None,
    cluster_query: str | int | Sequence | None = None,
    genes_ref: Sequence[str] | None = None,
    cluster_ref: str | int | Sequence | None = None,
    top_k: int | None = None,
    pc_stat_filter: Literal["ttest", "mannwhitney"] | None = None,
    pc_pvalue_threshold: float = 0.05,
    group: str | None = None,
    to_original_scale: bool = False,
) -> pd.DataFrame:
    """Find original features that best discriminate a perturbation set or cluster.

    Feature importance is derived from the centroid difference in TVN/PCA space
    (an orthogonal basis), which is then projected back to the original z-score
    feature space.  This is mathematically equivalent to, but conceptually
    cleaner than, computing the centroid difference directly in the correlated
    z-score space.

    Formally, the score for each original feature *f* is::

        score[f] = (mean(X_query_pca) − mean(X_ref_pca)) @ PCs[:, f]

    where ``X_query_pca`` and ``X_ref_pca`` are the query and reference
    observations in pure PCA space (after inverse covariance alignment if
    ``by`` was used in TVN).

    An optional *pc_stat_filter* prunes PCA components that are not
    statistically different between query and reference *before* backprojection.
    Because PCA components are orthogonal by construction, these tests avoid
    the correlated-feature multiple-testing problem inherent in direct feature
    testing.

    :param data: AnnData with TVN backprojection parameters in ``uns``.
        Works on cell-level data and on aggregated perturbation profiles
        (output of :func:`~scallops.features.agg.agg_features`).
    :param genes: Perturbation names identifying the *query* set.  Matched
        against ``data.obs[perturbation_column]``.  Mutually exclusive with
        ``cluster_labels`` + ``cluster_query``.
    :param perturbation_column: Column in ``obs`` that identifies perturbations.
        Only used when *genes* or *genes_ref* are provided.  Default
        ``"gene_symbol"``.
    :param cluster_labels: Array of cluster labels, one per observation (same
        order as ``data.obs``).  Use with ``cluster_query`` to select the
        query set and optionally ``cluster_ref`` to select the reference set.
    :param cluster_query: Cluster value(s) that define the query set when
        *cluster_labels* is provided.  A scalar or a list.
    :param genes_ref: Perturbation names identifying the *reference* set.
        When *None* (default) the reference is all observations not in the
        query.  Mutually exclusive with ``cluster_ref``.
    :param cluster_ref: Cluster value(s) identifying the *reference* set
        when *cluster_labels* is provided.  When *None* (default) the
        reference is all non-query observations.
    :param top_k: Return only the top-*k* features by ``|score|``.  *None*
        returns all features.
    :param pc_stat_filter: If set, applies a statistical test to each PC
        component (in the orthogonal PC space) to retain only those that are
        significantly different between query and reference.  Non-significant
        components (``p ≥ pc_pvalue_threshold``) are zeroed before
        backprojection.  Choose from:

        * ``"ttest"``       — Welch t-test (vectorised); recommended for ≥ 3
          samples per group.
        * ``"mannwhitney"`` — Mann-Whitney U; non-parametric, slower.
        * *None*            — no filtering (default); all PCs contribute.

    :param pc_pvalue_threshold: Significance threshold for PC retention when
        *pc_stat_filter* is set.  Default ``0.05``.
    :param group: Covariance-alignment group key used when inverting the TVN
        alignment step (see :func:`backproject_tvn`).  Required when TVN was
        run with ``by`` and multiple groups are present.
    :param to_original_scale: If *True*, backproject past the z-scoring step
        to recover the original measurement scale.
    :return: :class:`pandas.DataFrame` with columns:

        * ``feature``  — original feature name (from ``data.var.index``).
        * ``score``    — signed backprojected centroid difference in z-score
          (or original) space.  ``|score|`` gives the feature importance rank.
        * ``pvalue``   — per-feature significance derived as the contribution-
          weighted average of PC p-values when *pc_stat_filter* is set;
          ``NaN`` otherwise.

        Rows are sorted by ``|score|`` descending.
    """
    _validate_tvn_uns(data)

    # --- Build query mask ---
    if genes is not None and cluster_query is not None and cluster_labels is None:
        raise ValueError(
            "Provide `cluster_labels` when using `cluster_query`."
        )
    if genes is not None and cluster_labels is not None and cluster_query is not None:
        raise ValueError(
            "Specify the query using either `genes` or "
            "`cluster_labels` + `cluster_query`, not both."
        )
    query_mask = _build_obs_mask(
        data, genes, perturbation_column, cluster_labels, cluster_query
    )
    if query_mask is None:
        raise ValueError(
            "A query must be specified.  Provide `genes` or "
            "`cluster_labels` + `cluster_query`."
        )
    if query_mask.sum() == 0:
        raise ValueError("The query selector matched zero observations.")

    # --- Build reference mask ---
    if genes_ref is not None and cluster_ref is not None:
        raise ValueError(
            "Specify the reference using either `genes_ref` or `cluster_ref`, not both."
        )
    ref_mask_explicit = _build_obs_mask(
        data, genes_ref, perturbation_column, cluster_labels, cluster_ref
    )
    if ref_mask_explicit is not None:
        ref_mask = ref_mask_explicit
    else:
        ref_mask = ~query_mask

    if ref_mask.sum() == 0:
        raise ValueError(
            "The reference selector matched zero observations.  "
            "Check `genes_ref` / `cluster_ref` or ensure non-query data exists."
        )

    # --- Resolve pure PCA space (inverse covariance alignment) ---
    # Preserve float32 when possible; the ranking result is invariant to precision.
    X = np.asarray(data.X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float32)
    cov_alignment_inv = data.uns.get("covariance_alignment_inv", {})
    if cov_alignment_inv:
        cov_inv_mat = _resolve_cov_alignment(cov_alignment_inv, group)
        if cov_inv_mat is not None:
            X_pca = X @ np.asarray(cov_inv_mat, dtype=X.dtype)
        else:
            X_pca = X
    else:
        X_pca = X

    # --- Centroid difference in PCA space ---
    X_query = X_pca[query_mask]   # (n_query, n_pcs)
    X_ref = X_pca[ref_mask]       # (n_ref, n_pcs)
    diff_pca = X_query.mean(0) - X_ref.mean(0)  # (n_pcs,)

    # --- Optional PC-level statistical filter ---
    pc_pvalues: np.ndarray | None = None
    if pc_stat_filter is not None:
        if len(X_query) < 2 or len(X_ref) < 2:
            warnings.warn(
                f"PC-level stat filter requires ≥ 2 samples in each group "
                f"(query={len(X_query)}, ref={len(X_ref)}).  "
                "Skipping PC stat filter.",
                UserWarning,
                stacklevel=2,
            )
        else:
            if pc_stat_filter == "ttest":
                pc_pvalues = _pc_ttest(X_query, X_ref)
            elif pc_stat_filter == "mannwhitney":
                pc_pvalues = _pc_mannwhitney(X_query, X_ref)
            else:
                raise ValueError(
                    f"Unknown pc_stat_filter {pc_stat_filter!r}. "
                    "Choose from 'ttest', 'mannwhitney', or None."
                )
            # Zero out non-significant PCs before backprojection
            diff_pca = diff_pca.copy()
            diff_pca[pc_pvalues >= pc_pvalue_threshold] = 0.0

    # --- Backproject centroid difference to feature space ---
    # Step 1: TVN internal PCA (K × K in PC space)
    PCs_tvn = np.asarray(data.uns["pca"]["PCs"], dtype=X.dtype)  # (K, K)
    # pca_mean cancels in differences: diff_z_pca = diff_pca @ PCs_tvn
    diff_z_pca = diff_pca @ PCs_tvn  # still in PC space (shape K)

    # Step 2 (optional): apply the original map-pca components if present.
    # ``uns["map_pca"]`` is set by the ``map run`` pipeline when data was
    # first reduced from p features → K PCs by ``map pca``.  Applying its
    # transpose brings the centroid difference back to the original p-
    # dimensional z-score feature space.
    map_pca = data.uns.get("map_pca")
    if map_pca is not None and "PCs" in map_pca:
        # map_pca["PCs"] shape: (K_in, K_out) = (p, K)
        # diff_z_pca @ map_pca["PCs"].T maps K-dim diff → p-dim diff
        # map_pca["PCs"] shape: (K', p)  where K' = selected PCs, p = original features
        # diff_z_pca shape:   (K',)
        # diff_z_pca @ PCs_map  →  (p,)   (scores in original feature space)
        PCs_map = np.asarray(map_pca["PCs"], dtype=diff_z_pca.dtype)  # (K', p)
        # If K' dims don't match (pca-select sliced obsm but not uns), align
        k_diff = len(diff_z_pca)
        if PCs_map.shape[0] != k_diff:
            PCs_map = PCs_map[:k_diff]
        feature_scores = diff_z_pca @ PCs_map     # (p,) — original features
        feature_names  = list(map_pca.get("features", data.var.index))
    else:
        # No map_pca: scores are already in the same space as data.var
        feature_scores = diff_z_pca
        feature_names  = list(data.var.index)

    if to_original_scale:
        # un-z-score using the per-feature scale stored by TVN
        pre_std = np.asarray(data.uns["tvn_pre_scale_std"], dtype=np.float64)
        pre_std = np.where(pre_std == 0.0, 1.0, pre_std)
        if len(pre_std) == len(feature_scores):
            feature_scores = feature_scores * pre_std

    # --- Per-feature p-value (contribution-weighted average of PC p-values) ---
    n_features = len(feature_scores)
    PCs = PCs_tvn   # used for contribution weighting below
    if pc_pvalues is not None:
        # Contribution of TVN PC i to output feature f:
        # diff_pca[i] × PCs_tvn[i, j] × (PCs_map[f, j] if map_pca else δ_{i,f})
        if map_pca is not None and "PCs" in map_pca:
            PCs_map_np = np.asarray(map_pca["PCs"], dtype=np.float32)  # (K', p)
            k_d = len(diff_pca)
            if PCs_map_np.shape[0] != k_d:
                PCs_map_np = PCs_map_np[:k_d]
            # PCs_tvn (K'×K') @ PCs_map_np (K'×p) → (K'×p) total contribution
            contrib = np.abs(diff_pca[:, np.newaxis]) * np.abs(PCs_tvn @ PCs_map_np)
        else:
            contrib = np.abs(diff_pca[:, np.newaxis]) * np.abs(PCs)
        total = contrib.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted_sum = (contrib * pc_pvalues[:, np.newaxis]).sum(axis=0)
            feature_pvalues = np.where(total > 0, weighted_sum / total, np.nan)
    else:
        feature_pvalues = np.full(n_features, np.nan)

    # --- Assemble and sort result ---
    result = pd.DataFrame(
        {"feature": feature_names, "score": feature_scores, "pvalue": feature_pvalues}
    )
    result = result.iloc[result["score"].abs().argsort()[::-1]].reset_index(drop=True)
    if top_k is not None:
        result = result.head(top_k)
    return result
