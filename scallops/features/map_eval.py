import io
import logging
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Tuple

import anndata
import dask
import dask.array as da
import fsspec
import numpy as np
import pandas as pd
from array_api_compat import get_namespace
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity

from scallops.features.util import _slice_anndata

logger = logging.getLogger("scallops")


def recall(
    null_distribution: np.ndarray | da.Array,
    query_distribution: np.ndarray | da.Array,
    recall_thresholds: Sequence[Tuple[float, float] | float] = [
        (0.01, 0.99),
        (0.05, 0.95),
    ],
) -> pd.DataFrame:
    """Compute recall at given thresholds for a query distribution with respect to a
    null distribution.

    :param null_distribution: The null distribution to compare against
    :param query_distribution: The query distribution
    :param recall_thresholds: A sequence of pairs of floats (left, right) or single
    floats. Single floats are used to perform one-sided recall. Thresholds should be
    between 0 and 1.
    :return Dataframe containing recall at given thresholds
    """
    assert isinstance(query_distribution, da.Array) == isinstance(
        null_distribution, da.Array
    )
    left = False
    right = False
    # validate inputs and check which directions are needed
    for threshold in recall_thresholds:
        if np.isscalar(threshold):
            assert 0 <= threshold <= 1
            if threshold >= 0.5:
                left = True
            else:
                right = True
        else:
            left_threshold, right_threshold = np.min(threshold), np.max(threshold)
            assert 0.5 <= right_threshold <= 1
            assert 0 <= left_threshold <= 0.5
            left = True
            right = True
    if isinstance(query_distribution, da.Array) and np.isnan(
        query_distribution.shape[0]
    ):
        query_distribution = query_distribution.compute_chunk_sizes()

    sorted_null_distribution = np.sort(
        null_distribution.compute()
        if isinstance(null_distribution, da.Array)
        else null_distribution
    )
    if isinstance(null_distribution, da.Array):
        sorted_null_distribution = da.from_array(sorted_null_distribution)

    xp = get_namespace(query_distribution)
    if left:
        query_percentage_ranks_left = xp.searchsorted(
            sorted_null_distribution, query_distribution, side="left"
        ) / len(sorted_null_distribution)
    if right:
        query_percentage_ranks_right = xp.searchsorted(
            sorted_null_distribution, query_distribution, side="right"
        ) / len(sorted_null_distribution)
    results = []
    for threshold in recall_thresholds:
        result = dict()
        if np.isscalar(threshold):
            result["threshold"] = threshold
            if threshold >= 0.5:
                result["recall"] = xp.sum(
                    (query_percentage_ranks_left >= threshold)
                ) / len(query_distribution)
            else:
                result["recall"] = xp.sum(
                    (query_percentage_ranks_right <= threshold)
                ) / len(query_distribution)
        else:
            left_threshold, right_threshold = np.min(threshold), np.max(threshold)
            result["threshold"] = (left_threshold, right_threshold)
            result["recall"] = xp.sum(
                (query_percentage_ranks_right <= left_threshold)
                | (query_percentage_ranks_left >= right_threshold)
            ) / len(query_distribution)
        results.append(result)
    if isinstance(query_distribution, da.Array):
        results = dask.compute(*results)
    return pd.DataFrame(results)


def set_benchmark(
    data: anndata.AnnData,
    set_name_to_genes: dict[str, Sequence[str]],
    min_genes: int = 10,
) -> pd.DataFrame:
    """
    Tests whether distributions of similarities of within and between set are different using Kolmogorov-Smirnov test.

    :param data: AnnData object containing perturbation similarity matrix.
    :param set_name_to_genes: Dictionary that maps set names to genes in set.
    :param min_genes: Minimum number of genes per set.
    :return: DataFrame containing the results.

    """

    # Adapted from cluster_benchmark method from
    # https://github.com/recursionpharma/EFAAR_benchmarking/blob/trunk/efaar_benchmarking/benchmarking.py

    results = []
    assert np.all(data.var.index == data.obs.index)
    for set_name in set_name_to_genes:
        set_genes = set_name_to_genes[set_name]

        within_expr = data.var.index.isin(set_genes)
        within_data = _slice_anndata(data, within_expr, within_expr)
        if within_data.shape[0] < min_genes:
            continue
        within_vals = within_data.X[np.triu_indices(within_data.shape[0], k=1)]
        between_data = _slice_anndata(data, within_expr, ~within_expr)
        between_vals = between_data.X.flatten()
        ks_res = ks_2samp(within_vals, between_vals)
        results.append(
            [
                set_name,
                within_data.shape[0],
                within_vals.mean(),
                between_vals.mean(),
                ks_res.statistic,
                ks_res.pvalue,
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "name",
            "size",
            "within_mean",
            "between_mean",
            "statistic",
            "pvalue",
        ],
    )


def pairwise_similarities(
    data: anndata.AnnData, metric: Literal["cosine", "pearson"] = "cosine"
) -> np.ndarray:
    """Compute pairwise similarities between observations in data.

    :param data: Anndata object
    :param metric: Similarity metric
    :return: Array containing similarities
    """

    if metric == "cosine":
        values = cosine_similarity(data.X)
    elif metric == "pearson":
        values = np.corrcoef(data.X)
    else:
        raise ValueError(f"Metric {metric} is not supported.")
    return values


def read_gmt(path: str) -> pd.DataFrame:
    """Read gene sets stored in GMT format.

    :param path: Path to GMT file.
    :return: Dataframe containing gene sets.
    """
    results = []
    with fsspec.open(path, "r") as file:
        for line in file:
            fields = line.strip().split("\t")
            genes = fields[2:]
            genes = [x for x in genes if x]
            n_genes = len(genes)
            genes = set(genes)
            set_name = fields[0]
            set_descr = fields[1]
            assert len(genes) == n_genes, f"Duplicate gene found for {set_name}."
            results.append([set_name, set_descr, genes])
    return pd.DataFrame(results, columns=["name", "description", "genes"]).set_index(
        "name"
    )


def read_corum(path: str) -> pd.DataFrame:
    """Read CORUM CSV and return a dataframe containing pairs of genes found in CORUM.

    :param path: Path to CORUM CSV (e.g. corum_humanComplexes.txt). Available from
        https://mips.helmholtz-muenchen.de/corum/download
    :return: Dataframe containing pairs of genes found and complexes they belong to
    """

    df = pd.read_csv(path, usecols=["complex_name", "subunits_gene_name"], sep="\t")
    corum_gene_names = df["subunits_gene_name"].values
    complex_names = df["complex_name"].values
    pairs = set()
    pair_to_complex_names = defaultdict(set)

    for i in range(len(corum_gene_names)):
        cluster = corum_gene_names[i].split(";")
        complex_name = complex_names[i]
        for j in range(len(cluster)):
            for k in range(j):
                p1 = (cluster[j], cluster[k])
                p2 = (cluster[k], cluster[j])
                pairs.add(p1)
                pairs.add(p2)
                pair_to_complex_names[p1].add(complex_name)
                pair_to_complex_names[p2].add(complex_name)
    a = []
    b = []
    c = []
    for p in pairs:
        a.append(p[0])
        b.append(p[1])
        c.append(pair_to_complex_names[p])
    return pd.DataFrame(data=dict(a=a, b=b, complex_name=c))


# ---------------------------------------------------------------------------
# Pairwise recall (for interaction-network databases)
# ---------------------------------------------------------------------------


def pairwise_benchmark(
    data: anndata.AnnData,
    reference_pairs: pd.DataFrame,
    gene_a_col: str = "gene_a",
    gene_b_col: str = "gene_b",
    recall_thresholds: Sequence[Tuple[float, float] | float] = [
        (0.01, 0.99),
        (0.05, 0.95),
    ],
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Recall benchmark for pairwise interaction databases (e.g. STRING, Reactome FI).

    Tests whether known-interacting gene pairs appear at the extremes of the
    similarity distribution more often than chance.

    The similarity matrix ``data.X`` must be square with ``data.obs.index ==
    data.var.index`` (i.e. the output of ``map-similarity``).

    :param data: AnnData containing the square pairwise similarity matrix.
    :param reference_pairs: DataFrame with at least two gene-symbol columns
        identifying interacting pairs.  Interactions are treated as
        bidirectional.
    :param gene_a_col: Column name for the first gene in each pair.
    :param gene_b_col: Column name for the second gene in each pair.
    :param recall_thresholds: Thresholds passed to :func:`recall`.
    :param min_pairs: Minimum number of reference pairs present in the
        similarity matrix for the benchmark to run.  Returns an empty
        DataFrame when the threshold is not met.
    :return: DataFrame with columns ``[n_pairs, threshold, recall]``.
    """
    labels = np.asarray(data.obs.index.tolist())
    label_set = set(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # Filter pairs to genes present in the similarity matrix
    in_matrix_a = reference_pairs[gene_a_col].isin(label_set)
    in_matrix_b = reference_pairs[gene_b_col].isin(label_set)
    valid = reference_pairs[in_matrix_a & in_matrix_b]

    if len(valid) < min_pairs:
        logger.info(
            f"pairwise_benchmark: only {len(valid)} reference pairs found "
            f"in the similarity matrix (min_pairs={min_pairs}); skipping."
        )
        return pd.DataFrame()

    # Build a frozenset of sorted gene pairs for O(1) lookup
    ref_set = frozenset(
        tuple(sorted([a, b]))
        for a, b in zip(valid[gene_a_col].values, valid[gene_b_col].values)
    )

    # Extract upper-triangle pairwise similarities and mark reference pairs
    n = len(labels)
    i_idx, j_idx = np.triu_indices(n, k=1)
    X = np.asarray(data.X, dtype=np.float64)
    all_sims = X[i_idx, j_idx]

    ref_mask = np.array(
        [tuple(sorted([labels[i], labels[j]])) in ref_set for i, j in zip(i_idx, j_idx)]
    )
    query_sims = all_sims[ref_mask]

    if len(query_sims) == 0:
        return pd.DataFrame()

    n_pairs = int(ref_mask.sum())
    result = recall(
        null_distribution=all_sims,
        query_distribution=query_sims,
        recall_thresholds=recall_thresholds,
    )
    result.insert(0, "n_pairs", n_pairs)
    return result


# ---------------------------------------------------------------------------
# STRING DB
# ---------------------------------------------------------------------------


def read_string(
    path: str,
    score_threshold: int = 400,
    gene_a_col: str = "preferredName_A",
    gene_b_col: str = "preferredName_B",
    score_col: str = "score",
    sep: str = "\t",
) -> pd.DataFrame:
    """Read a STRING interaction file and return gene pairs above the threshold.

    Supports the TSV format returned by the STRING REST API as well as custom
    two-column gene-symbol files.  STRING scores can be in the 0–1 scale
    (REST API ``network`` endpoint) or the 0–1000 scale (flat files); the
    function normalises automatically by comparing the maximum observed score.

    Download STRING data from `string-db.org <https://string-db.org/>`_ or
    produce a TSV via::

        scallops map-recall --string-fetch  # query API at recall time

    :param path: Path to the STRING TSV (local or cloud URI).
    :param score_threshold: Minimum combined score to retain an interaction.
        Interpreted on the 0–1000 scale; files with scores in 0–1 are
        rescaled automatically.
    :param gene_a_col: Column name for the first protein/gene symbol.
    :param gene_b_col: Column name for the second protein/gene symbol.
    :param score_col: Column containing the combined interaction score.
    :param sep: Field separator (default tab).
    :return: DataFrame with columns ``[gene_a, gene_b, score]``.
    """
    df = pd.read_csv(path, sep=sep, comment="#")

    rename = {}
    if gene_a_col in df.columns:
        rename[gene_a_col] = "gene_a"
    if gene_b_col in df.columns:
        rename[gene_b_col] = "gene_b"
    if score_col in df.columns:
        rename[score_col] = "score"
    df = df.rename(columns=rename)

    required = {"gene_a", "gene_b"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Could not find gene columns in {path!r}.  "
            f"Expected {gene_a_col!r} and {gene_b_col!r}; "
            f"found: {list(df.columns)}"
        )

    if "score" in df.columns:
        max_score = df["score"].max()
        # Scores in 0–1 range → rescale to 0–1000 for consistent thresholding
        threshold = score_threshold / 1000.0 if max_score <= 1.0 else score_threshold
        df = df[df["score"] >= threshold]

    keep = ["gene_a", "gene_b"] + (["score"] if "score" in df.columns else [])
    return df[keep].drop_duplicates().reset_index(drop=True)


def fetch_string(
    genes: Sequence[str],
    species_id: int = 9606,
    score_threshold: int = 400,
    network_type: str = "full",
    batch_size: int = 200,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch STRING protein interactions for a list of genes via the REST API.

    Queries ``https://string-db.org/api/tsv/network`` in batches of
    *batch_size* genes and returns all interactions above *score_threshold*.

    :param genes: Gene symbols to query (HGNC / gene name format).
    :param species_id: NCBI taxonomy ID (default 9606 for human).
    :param score_threshold: Minimum combined score (0–1000) to retain.
    :param network_type: ``"full"`` (all evidence types, default) or
        ``"physical"`` (physical interactions only).
    :param batch_size: Number of genes per API request.  STRING recommends
        ≤ 200 per call.
    :param timeout: HTTP request timeout in seconds.
    :return: DataFrame with columns ``[gene_a, gene_b, score]``.
    :raises urllib.error.URLError: When the STRING API is unreachable.
    """
    BASE = "https://string-db.org/api/tsv/network"
    frames = []

    for start in range(0, len(genes), batch_size):
        batch = list(genes[start : start + batch_size])
        params = urllib.parse.urlencode(
            {
                "identifiers": "\r".join(batch),
                "species": species_id,
                "required_score": score_threshold,
                "network_type": network_type,
                "caller_identity": "scallops_map_recall",
            }
        )
        url = f"{BASE}?{params}"
        logger.info(f"Fetching STRING for {len(batch)} genes (batch {start}–{start+len(batch)})…")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
        batch_df = pd.read_csv(io.StringIO(text), sep="\t")
        if len(batch_df) and "preferredName_A" in batch_df.columns:
            frames.append(
                batch_df[["preferredName_A", "preferredName_B", "score"]].rename(
                    columns={"preferredName_A": "gene_a", "preferredName_B": "gene_b"}
                )
            )

    if not frames:
        return pd.DataFrame(columns=["gene_a", "gene_b", "score"])

    full = pd.concat(frames, ignore_index=True)
    # Scores from this endpoint are in 0–1 range; normalise to 0–1000
    if full["score"].max() <= 1.0:
        full["score"] = (full["score"] * 1000).round().astype(int)
    full = full[full["score"] >= score_threshold]

    # Deduplicate symmetric pairs (A→B and B→A both appear)
    full["_key"] = full.apply(
        lambda r: tuple(sorted([r["gene_a"], r["gene_b"]])), axis=1
    )
    full = full.drop_duplicates(subset="_key").drop(columns="_key")
    logger.info(f"STRING: {len(full):,} interactions above score ≥{score_threshold}")
    return full.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reactome Functional Interactions
# ---------------------------------------------------------------------------


def read_reactome_fi(
    path: str,
    interaction_types: set[str] | None = None,
) -> pd.DataFrame:
    """Read a Reactome Functional Interactions (ReactomeFI) file.

    ReactomeFI files can be downloaded from
    `reactomefip.wustl.edu <https://reactomefip.wustl.edu/download/>`_.
    Common column layouts are auto-detected:

    * Two-column: ``Gene1 \\t Gene2``
    * Three-column: ``Gene1 \\t Annotation \\t Gene2`` (older format)
    * Four-column: ``Gene1 \\t Gene2 \\t Annotation \\t Score``

    :param path: Path to the ReactomeFI TSV (local or cloud URI).
    :param interaction_types: Optional set of annotation values to keep
        (e.g. ``{"activate|inhibit", "catalyze"}``).  When *None* all
        interactions are returned.
    :return: DataFrame with columns ``[gene_a, gene_b]``  and optionally
        ``annotation`` when an annotation column is detected.
    """
    df = pd.read_csv(path, sep="\t", comment="#", header=0)
    cols = df.columns.tolist()

    if len(cols) < 2:
        raise ValueError(f"Expected ≥ 2 columns in {path!r}, got {cols}")

    # Heuristic: gene columns are first and last (or first two for 2-col files)
    gene_a_col = cols[0]
    gene_b_col = cols[-1] if len(cols) > 2 else cols[1]
    annotation_col = None

    # Detect annotation column (non-gene column between the two gene columns)
    for c in cols[1:-1]:
        sample = df[c].dropna().astype(str).head(10)
        # Annotation columns tend to have "|" separators or known keywords
        if sample.str.contains(r"\||catalyz|inhibit|activat|bind|complex", case=False).any():
            annotation_col = c
            break

    result = pd.DataFrame(
        {"gene_a": df[gene_a_col].astype(str), "gene_b": df[gene_b_col].astype(str)}
    )
    if annotation_col:
        result["annotation"] = df[annotation_col].astype(str)
        if interaction_types is not None:
            result = result[result["annotation"].isin(interaction_types)]

    return result.dropna(subset=["gene_a", "gene_b"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# GMT → gene-set dict
# ---------------------------------------------------------------------------


def gmt_to_gene_sets(gmt_df: pd.DataFrame) -> dict[str, list[str]]:
    """Convert a :func:`read_gmt` DataFrame to the ``{name: [gene, ...]}`` dict
    expected by :func:`set_benchmark`.

    :param gmt_df: DataFrame returned by :func:`read_gmt` (indexed by set name,
        ``"genes"`` column contains sets of strings).
    :return: Mapping from set name to list of gene symbols.
    """
    return {name: list(genes) for name, genes in gmt_df["genes"].items()}
