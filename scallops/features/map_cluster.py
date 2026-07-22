"""Clustering methods for perturbation-map similarity matrices.

Three algorithms are provided.  All accept a square similarity matrix (as
stored in the ``X`` or ``obsp["similarity"]`` of an AnnData produced by
``map-similarity``) and return cluster labels together with a reordered
AnnData where same-cluster perturbations are adjacent.

Public API
----------
cluster_similarity
    Top-level function: choose algorithm, estimate hyperparameters when
    requested, reorder the similarity matrix, and annotate ``obs["cluster"]``.

Authors:
    - The SCALLOPS development team
"""

import logging
import warnings
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

# Leaf-ordering modes for hierarchical clustering.
LeafOrdering = Literal["none", "fast", "exact"]

logger = logging.getLogger("scallops")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _find_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index of the elbow/knee in curve (x, y).

    Uses the *maximum perpendicular distance from the secant* method
    (Kneedle-style, no external dependency).

    :param x: 1-D independent-variable values (e.g. hyperparameter range).
    :param y: 1-D dependent values (e.g. number of clusters).
    :return: Index into *x* / *y* of the elbow point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    denom = np.sqrt(dx ** 2 + dy ** 2)
    if denom < 1e-12:
        return 0
    distances = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / denom
    return int(np.argmax(distances))


def _sim_to_dist(sim: np.ndarray) -> np.ndarray:
    """Convert a similarity matrix to a distance matrix.

    :param sim: Square similarity matrix with values in ``[-1, 1]``
        (e.g. cosine or Pearson correlation).
    :return: Square distance matrix ``dist = clip(1 - sim, 0, None)`` with
        the diagonal set to zero.
    """
    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# Leaf ordering helpers
# ---------------------------------------------------------------------------


def _approx_leaf_ordering(Z: np.ndarray, n: int, sim: np.ndarray) -> np.ndarray:
    """Greedy approximate leaf ordering via dendrogram subtree flipping.

    At each internal node (bottom-up), the two child subtrees are
    optionally flipped so that the boundary leaves — rightmost of the left
    subtree and leftmost of the right subtree — have the highest pairwise
    similarity.  This is equivalent to scipy's ``optimal_leaf_ordering``
    but greedy (no backtracking), making it O(n log n) for balanced
    dendrograms instead of O(n²).  In practice the result is
    indistinguishable from the exact solution for well-separated clusters.

    :param Z: Linkage matrix from ``scipy.cluster.hierarchy.linkage``.
    :param n: Number of leaf nodes (observations).
    :param sim: Full (n × n) similarity matrix; accessed read-only.
    :return: 1-D integer array of length *n* giving the optimal leaf order.
    """
    # Each entry: tuple(forward_list, reversed_list) to avoid O(n) reversals.
    # We store both orientations so any flip is an O(1) swap.
    node_fwd: dict[int, list[int]] = {}
    node_rev: dict[int, list[int]] = {}
    for i in range(n):
        node_fwd[i] = [i]
        node_rev[i] = [i]

    for merge_idx, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        node_id = n + merge_idx

        lf, lr = node_fwd.pop(left),  node_rev.pop(left)
        rf, rr = node_fwd.pop(right), node_rev.pop(right)

        # Four junction similarities: (L-end, R-start) for each flip combo.
        # lf[-1] = right end of left-forward, lr[-1] = right end of left-reversed, etc.
        # We want max sim[left_right_end, right_left_end].
        combos = [
            (lf, rf, sim[lf[-1], rf[0]]),   # left-fwd + right-fwd
            (lf, rr, sim[lf[-1], rr[0]]),   # left-fwd + right-rev
            (lr, rf, sim[lr[-1], rf[0]]),   # left-rev + right-fwd
            (lr, rr, sim[lr[-1], rr[0]]),   # left-rev + right-rev
        ]
        best_l, best_r, _ = max(combos, key=lambda t: t[2])
        merged = best_l + best_r
        node_fwd[node_id] = merged
        node_rev[node_id] = merged[::-1]

    root = n + len(Z) - 1
    return np.array(node_fwd[root])


def _apply_leaf_ordering(
    Z: np.ndarray,
    n: int,
    sim: np.ndarray,
    mode: "LeafOrdering",
) -> np.ndarray:
    """Return a leaf order array according to *mode*.

    :param Z: Linkage matrix.
    :param n: Number of leaves.
    :param sim: Full similarity matrix (used only for ``"fast"``).
    :param mode: ``"none"`` — dendrogram default order (instant);
        ``"fast"`` — greedy subtree-flip approximation, O(n log n);
        ``"exact"`` — scipy optimal leaf ordering, O(n²), slow for n > 5 000.
    :return: 1-D integer index array of length *n*.
    """
    if mode == "none":
        return leaves_list(Z)
    if mode == "fast":
        return _approx_leaf_ordering(Z, n, sim)
    # "exact"
    if n > 5_000:
        logger.warning(
            "leaf_ordering='exact' with n=%d may take many hours. "
            "Consider 'fast' instead.",
            n,
        )
    from scipy.cluster.hierarchy import optimal_leaf_ordering
    Z_ordered = optimal_leaf_ordering(Z, squareform(np.clip(1.0 - sim, 0, None), checks=False))
    return leaves_list(Z_ordered)


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------


def _hierarchical_auto_n(Z: np.ndarray, max_n: int) -> int:
    """Estimate the optimal number of clusters from a linkage matrix.

    Looks at the sequence of merge heights (descending) and finds the point
    with the largest acceleration — the biggest "jump" in the dendrogram.

    :param Z: Linkage matrix returned by ``scipy.cluster.hierarchy.linkage``.
    :param max_n: Upper bound on the number of clusters to consider.
    :return: Estimated optimal number of clusters (≥ 2).
    """
    heights = Z[:, 2][::-1]  # largest merge first
    cap = min(len(heights) - 1, max(2, max_n - 1))
    heights = heights[:cap]
    if len(heights) < 2:
        return 2
    # Gaps between consecutive merge heights (negative = decreasing height)
    gaps = np.abs(np.diff(heights))
    n_clusters = int(np.argmax(gaps)) + 2  # +2: one for 0-indexing, one for the gap step
    return max(2, min(n_clusters, max_n))


def _cluster_hierarchical(
    sim: np.ndarray,
    n_clusters: int | None,
    linkage_method: str,
    auto_n: bool,
    max_n: int,
    leaf_ordering: "LeafOrdering" = "fast",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit hierarchical clustering on a similarity matrix.

    :param sim: Square (n, n) similarity matrix.
    :param n_clusters: Target number of flat clusters.  Estimated when *None*
        and ``auto_n`` is *True*.
    :param linkage_method: Linkage criterion (``"ward"``, ``"complete"``,
        ``"average"``, ``"single"``).
    :param auto_n: Estimate ``n_clusters`` from the dendrogram when *True*.
    :param max_n: Maximum ``n_clusters`` for the auto-estimation search.
    :param leaf_ordering: How to order leaves within the dendrogram for
        visualisation.  ``"none"`` uses the raw dendrogram traversal order
        (instant); ``"fast"`` uses a greedy subtree-flip approximation
        (O(n log n), default); ``"exact"`` uses scipy's optimal leaf ordering
        (O(n²), slow for n > 5 000).
    :return: ``(labels, leaf_order, info_dict)`` — cluster assignments
        (0-indexed), the leaf traversal order to use for reordering the
        matrix, and metadata.
    """
    n = sim.shape[0]
    dist = _sim_to_dist(sim)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if n_clusters is None:
        if auto_n:
            n_clusters = _hierarchical_auto_n(Z, max_n=min(max_n, n - 1))
            logger.info(
                f"Hierarchical clustering: auto-selected n_clusters={n_clusters} "
                f"(linkage={linkage_method}, max_n={max_n})"
            )
        else:
            n_clusters = max(2, int(np.round(np.sqrt(n))))

    labels = fcluster(Z, t=int(n_clusters), criterion="maxclust") - 1  # 0-indexed
    leaf_order = _apply_leaf_ordering(Z, n, sim, leaf_ordering)
    return labels.astype(int), leaf_order, {
        "method": "hierarchical",
        "linkage": linkage_method,
        "n_clusters": int(labels.max() + 1),
        "n_clusters_requested": int(n_clusters),
        "auto_n": auto_n,
        "leaf_ordering": leaf_ordering,
    }


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------


def _hdbscan_auto_min_cluster_size(
    dist: np.ndarray,
    n_range: int,
) -> int:
    """Estimate ``min_cluster_size`` for HDBSCAN using an elbow criterion.

    Runs HDBSCAN over a grid of ``min_cluster_size`` values and finds the
    elbow point where the number of clusters stops decreasing rapidly.

    :param dist: Square (n, n) distance matrix (``float64``).
    :param n_range: Number of candidate values to evaluate.
    :return: Estimated ``min_cluster_size``.
    """
    try:
        import hdbscan as _hdbscan
    except ImportError:
        return max(2, dist.shape[0] // 10)

    n = dist.shape[0]
    sizes = np.unique(np.linspace(2, max(3, n // 4), n_range, dtype=int))
    n_clusters = []
    for s in sizes:
        lbl = _hdbscan.HDBSCAN(
            min_cluster_size=int(s), metric="precomputed"
        ).fit_predict(dist)
        nc = max(1, int((lbl >= 0).sum() > 0 and len(set(lbl[lbl >= 0]))))
        nc = len(set(lbl[lbl >= 0])) if (lbl >= 0).any() else 1
        n_clusters.append(nc)

    idx = _find_elbow(sizes.astype(float), np.array(n_clusters, dtype=float))
    chosen = int(sizes[idx])
    logger.info(
        f"HDBSCAN elbow: min_cluster_size={chosen} "
        f"(evaluated {len(sizes)} values, n_clusters={n_clusters})"
    )
    return chosen


def _hdbscan_auto_min_samples(
    dist: np.ndarray,
    min_cluster_size: int,
    n_range: int,
) -> int:
    """Estimate ``min_samples`` for HDBSCAN using an elbow criterion.

    With ``min_cluster_size`` fixed, runs HDBSCAN over a grid of
    ``min_samples`` values (1 … ``min_cluster_size``) and finds the elbow
    of the *noise-point count* curve.

    Effect of increasing ``min_samples``:
      * Fewer, denser core regions → more noise points, more robust clusters.
    The elbow identifies the point beyond which noise grows sharply, i.e. the
    transition from "robust clustering" to "over-conservative clustering".

    :param dist: Square (n, n) distance matrix (``float64``).
    :param min_cluster_size: Fixed ``min_cluster_size`` to use during search.
    :param n_range: Number of ``min_samples`` values to evaluate.
    :return: Estimated ``min_samples``.
    """
    try:
        import hdbscan as _hdbscan
    except ImportError:
        return 1

    max_ms = max(2, min_cluster_size)
    # Candidate values: integer grid from 1 to min_cluster_size
    sample_range = np.unique(np.linspace(1, max_ms, min(n_range, max_ms), dtype=int))
    noise_counts = []

    for ms in sample_range:
        lbl = _hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=int(ms),
            metric="precomputed",
        ).fit_predict(dist)
        noise_counts.append(int((lbl == -1).sum()))

    noise_arr = np.array(noise_counts, dtype=float)

    # The noise curve is monotonically non-decreasing. The elbow is the last
    # point before noise starts growing significantly — we reverse the curve
    # so _find_elbow locates the "take-off" point.
    idx = _find_elbow(sample_range.astype(float), noise_arr)
    chosen = int(sample_range[idx])
    logger.info(
        f"HDBSCAN min_samples elbow: min_samples={chosen} "
        f"(range 1-{max_ms}, noise_counts={noise_counts})"
    )
    return chosen


def _cluster_hdbscan(
    sim: np.ndarray,
    min_cluster_size: int | None,
    min_samples: int | None,
    auto_params: bool,
    n_range: int,
) -> tuple[np.ndarray, dict]:
    """Fit HDBSCAN on a similarity matrix.

    Both ``min_cluster_size`` and ``min_samples`` are auto-estimated when
    *None* and ``auto_params`` is *True*:

    * ``min_cluster_size`` is estimated first via the elbow of the
      cluster-count curve across a grid of candidate sizes.
    * ``min_samples`` is then estimated (with ``min_cluster_size`` fixed)
      via the elbow of the noise-point-count curve across values
      1 … ``min_cluster_size``.

    :param sim: Square (n, n) similarity matrix.
    :param min_cluster_size: Minimum cluster size.  Auto-estimated when *None*
        and ``auto_params`` is *True*.
    :param min_samples: Controls the conservativeness of the density estimate.
        Auto-estimated when *None* and ``auto_params`` is *True*.
    :param auto_params: Estimate both hyperparameters via elbow when *True*.
    :param n_range: Number of candidate values for each elbow search.
    :return: ``(labels, info_dict)``, noise points (−1) re-labelled as
        individual trailing clusters.
    """
    try:
        import hdbscan as _hdbscan
    except ImportError:
        raise ImportError(
            "hdbscan is not installed.  Install it with: pip install hdbscan"
        )

    dist = _sim_to_dist(sim).astype(np.float64)

    # --- Estimate min_cluster_size ---
    if min_cluster_size is None:
        if auto_params:
            min_cluster_size = _hdbscan_auto_min_cluster_size(dist, n_range=n_range)
        else:
            min_cluster_size = max(2, sim.shape[0] // 10)
        logger.info(f"HDBSCAN: min_cluster_size={min_cluster_size}")

    # --- Estimate min_samples ---
    if min_samples is None:
        if auto_params:
            min_samples = _hdbscan_auto_min_samples(dist, min_cluster_size, n_range=n_range)
        else:
            min_samples = min_cluster_size  # HDBSCAN default
        logger.info(f"HDBSCAN: min_samples={min_samples}")

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist)

    # Assign noise points (−1) to singleton clusters at the end
    n_noise = int((labels == -1).sum())
    if n_noise > 0:
        next_label = int(labels.max()) + 1
        noise_idx = np.where(labels == -1)[0]
        for i, idx in enumerate(noise_idx):
            labels[idx] = next_label + i
        warnings.warn(
            f"HDBSCAN: {n_noise} noise point(s) assigned to singleton clusters.  "
            "Consider a smaller min_cluster_size or --cluster-method hierarchical.",
            UserWarning,
            stacklevel=3,
        )

    return labels.astype(int), {
        "method": "hdbscan",
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "auto_params": auto_params,
        "n_clusters": int(labels.max() + 1),
        "n_noise": n_noise,
    }


# ---------------------------------------------------------------------------
# Leiden clustering
# ---------------------------------------------------------------------------


def _leiden_auto_resolution(
    g,
    n_range: int,
    res_min: float,
    res_max: float,
    random_state: int,
) -> float:
    """Estimate Leiden ``resolution`` via the elbow of the cluster-count curve.

    :param g: ``igraph.Graph`` with edge weights.
    :param n_range: Number of resolution values to evaluate.
    :param res_min: Minimum resolution to consider.
    :param res_max: Maximum resolution to consider.
    :param random_state: Random seed.
    :return: Estimated resolution.
    """
    try:
        import leidenalg
    except ImportError:
        return 1.0

    resolutions = np.linspace(res_min, res_max, n_range)
    n_clusters = []
    for res in resolutions:
        p = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(res),
            seed=random_state,
        )
        n_clusters.append(len(set(p.membership)))

    idx = _find_elbow(resolutions, np.array(n_clusters, dtype=float))
    chosen = float(resolutions[idx])
    logger.info(
        f"Leiden elbow: resolution={chosen:.4f} "
        f"(evaluated {n_range} values, n_clusters={n_clusters})"
    )
    return chosen


def _cluster_leiden(
    sim: np.ndarray,
    resolution: float | None,
    similarity_threshold: float,
    auto_params: bool,
    n_range: int,
    res_min: float,
    res_max: float,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    """Fit Leiden community detection on a similarity matrix.

    :param sim: Square (n, n) similarity matrix.
    :param resolution: ``resolution_parameter`` for
        ``RBConfigurationVertexPartition``.  Auto-estimated when *None* and
        ``auto_params`` is *True*.
    :param similarity_threshold: Edges with similarity < this value are
        dropped from the graph.
    :param auto_params: Estimate ``resolution`` via elbow when *True*.
    :param n_range: Number of resolution values to evaluate for the elbow.
    :param res_min: Minimum resolution for the elbow search.
    :param res_max: Maximum resolution for the elbow search.
    :param random_state: Random seed.
    :return: ``(labels, info_dict)``.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "leidenalg and python-igraph are required for Leiden clustering.  "
            "Install with:  pip install leidenalg python-igraph"
        )

    # Build weighted graph: keep only edges above the similarity threshold
    adj = sim.copy()
    adj[adj < similarity_threshold] = 0.0
    np.fill_diagonal(adj, 0.0)

    src, dst = np.nonzero(np.triu(adj, k=1))
    weights_list = adj[src, dst].tolist()
    g = ig.Graph(
        n=sim.shape[0],
        edges=list(zip(src.tolist(), dst.tolist())),
        edge_attrs={"weight": weights_list},
    )

    if resolution is None:
        if auto_params:
            resolution = _leiden_auto_resolution(
                g, n_range=n_range, res_min=res_min, res_max=res_max,
                random_state=random_state,
            )
        else:
            resolution = 1.0
        logger.info(f"Leiden: resolution={resolution:.4f}")

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=random_state,
    )
    labels = np.array(partition.membership, dtype=int)
    return labels, {
        "method": "leiden",
        "resolution": float(resolution),
        "similarity_threshold": float(similarity_threshold),
        "auto_params": auto_params,
        "n_clusters": int(labels.max() + 1),
        "random_state": random_state,
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def cluster_similarity(
    data: anndata.AnnData,
    method: Literal["hierarchical", "hdbscan", "leiden"] = "hierarchical",
    auto_params: bool = True,
    n_clusters: int | None = None,
    linkage_method: str = "ward",
    max_n_clusters: int = 50,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    resolution: float | None = None,
    similarity_threshold: float = 0.3,
    leiden_res_min: float = 0.05,
    leiden_res_max: float = 2.0,
    elbow_n_range: int = 20,
    random_state: int = 0,
    leaf_ordering: "LeafOrdering" = "fast",
) -> anndata.AnnData:
    """Cluster perturbation profiles and reorder the similarity matrix.

    Works with both output formats from ``map-similarity``:

    * **matrix format** (``X`` is the square similarity matrix): reorders both
      rows and columns; ``obs.index == var.index == perturbation labels``.
    * **anndata format** (``obsp["similarity"]`` holds the matrix): reorders
      ``X`` (profiles), ``obs``, and ``obsp["similarity"]``; ``var`` is
      unchanged.

    The cluster assignments are stored in ``obs["cluster"]`` as zero-indexed
    integer strings so they sort correctly.  Clustering metadata is stored in
    ``uns["clustering"]``.

    :param data: AnnData from ``map-similarity``.
    :param method: Clustering algorithm.

        ``"hierarchical"`` *(default)*
            Agglomerative clustering via ``scipy``.  Optimal number of
            clusters estimated from the largest gap in the dendrogram merge
            heights when ``auto_params`` is *True*.
        ``"hdbscan"``
            Density-based hierarchical clustering.  Requires the ``hdbscan``
            package.  ``min_cluster_size`` estimated via elbow of the
            cluster-count curve when ``auto_params`` is *True*.
        ``"leiden"``
            Graph community detection via the Leiden algorithm.  Requires
            ``leidenalg`` and ``python-igraph``.  ``resolution`` estimated via
            elbow when ``auto_params`` is *True*.

    :param auto_params: Automatically estimate the main hyperparameter for the
        chosen method using an elbow criterion.  Ignored when the relevant
        parameter (``n_clusters``, ``min_cluster_size``, ``resolution``) is
        explicitly given.
    :param n_clusters: Target number of clusters for hierarchical clustering.
    :param linkage_method: Linkage criterion for hierarchical clustering:
        ``"ward"`` (default), ``"complete"``, ``"average"``, ``"single"``.
    :param max_n_clusters: Upper bound for the auto-estimated cluster count in
        hierarchical clustering.
    :param min_cluster_size: Minimum cluster size for HDBSCAN.
    :param min_samples: ``min_samples`` for HDBSCAN (defaults to
        ``min_cluster_size``).
    :param resolution: Resolution for the Leiden partition objective.
    :param similarity_threshold: Minimum similarity required to include an
        edge in the graph built for Leiden clustering.
    :param leiden_res_min: Lower bound of the resolution search range for
        Leiden elbow estimation.
    :param leiden_res_max: Upper bound of the resolution search range.
    :param elbow_n_range: Number of candidate hyperparameter values to
        evaluate when running the elbow search.
    :param random_state: Random seed for Leiden.
    :param leaf_ordering: Controls the within-dendrogram leaf order for
        ``method="hierarchical"``.  ``"none"`` — raw dendrogram traversal
        (instant, adequate for most uses); ``"fast"`` — greedy subtree-flip
        approximation, O(n log n), default; ``"exact"`` — scipy optimal leaf
        ordering, O(n²), slow for n > 5 000.  Ignored for other methods.
    :return: New AnnData with observations sorted by cluster.
    """
    use_obsp = "similarity" in data.obsp

    if use_obsp:
        sim = np.asarray(data.obsp["similarity"], dtype=np.float64)
    else:
        sim = np.asarray(data.X, dtype=np.float64)

    n = sim.shape[0]
    if n < 2:
        logger.warning("cluster_similarity: n < 2, returning data unchanged.")
        return data

    # --- Run chosen algorithm ---
    leaf_order = None
    if method == "hierarchical":
        labels, leaf_order, info = _cluster_hierarchical(
            sim,
            n_clusters=n_clusters,
            linkage_method=linkage_method,
            auto_n=auto_params,
            max_n=max_n_clusters,
            leaf_ordering=leaf_ordering,
        )
    elif method == "hdbscan":
        labels, info = _cluster_hdbscan(
            sim,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            auto_params=auto_params,
            n_range=elbow_n_range,
        )
    elif method == "leiden":
        labels, info = _cluster_leiden(
            sim,
            resolution=resolution,
            similarity_threshold=similarity_threshold,
            auto_params=auto_params,
            n_range=elbow_n_range,
            res_min=leiden_res_min,
            res_max=leiden_res_max,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown clustering method {method!r}.  "
            "Choose from 'hierarchical', 'hdbscan', 'leiden'."
        )

    logger.info(
        f"Clustering ({method}): {info['n_clusters']} clusters from {n} perturbations"
    )

    # --- Reorder: hierarchical uses dendrogram leaf order (preserves within-cluster
    # similarity structure); other methods fall back to sorting by cluster label. ---
    if leaf_order is not None:
        order = leaf_order
    else:
        order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]

    # obs with cluster column; use zero-padded strings so lexicographic sort works
    n_digits = len(str(int(labels.max())))
    cluster_labels_str = [str(c).zfill(n_digits) for c in labels_sorted]

    new_obs = data.obs.iloc[order].copy()
    new_obs["cluster"] = cluster_labels_str

    # --- Build output AnnData ---
    reordered_sim = (sim[order][:, order]).astype(np.float32)

    if use_obsp:
        new_X = np.asarray(data.X, dtype=np.float32)[order]
        result = anndata.AnnData(
            X=new_X,
            obs=new_obs,
            var=data.var.copy(),
        )
        result.obsp["similarity"] = reordered_sim
        for k, v in data.varm.items():
            result.varm[k] = v
    else:
        result = anndata.AnnData(
            X=reordered_sim,
            obs=new_obs,
            var=pd.DataFrame(index=new_obs.index),
        )

    for k, v in data.uns.items():
        result.uns[k] = v
    result.uns["clustering"] = info

    return result
