from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Tuple

import anndata
import fsspec
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity

from scallops.features.util import _slice_anndata


def recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
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

    sorted_null_distribution = np.sort(null_distribution)
    query_percentage_ranks_left = np.searchsorted(
        sorted_null_distribution, query_distribution, side="left"
    ) / len(sorted_null_distribution)
    query_percentage_ranks_right = np.searchsorted(
        sorted_null_distribution, query_distribution, side="right"
    ) / len(sorted_null_distribution)
    results = []
    for threshold in recall_thresholds:
        result = dict()
        if np.isscalar(threshold):
            assert 0 <= threshold <= 1
            result["threshold"] = threshold
            if threshold >= 0.5:
                result["recall"] = np.sum(
                    (query_percentage_ranks_left >= threshold)
                ) / len(query_distribution)
            else:
                result["recall"] = np.sum(
                    (query_percentage_ranks_right <= threshold)
                ) / len(query_distribution)
        else:
            left_threshold, right_threshold = np.min(threshold), np.max(threshold)
            assert 0 <= left_threshold <= 1
            assert 0 <= right_threshold <= 1
            result["threshold"] = (left_threshold, right_threshold)
            result["recall"] = np.sum(
                (query_percentage_ranks_right <= left_threshold)
                | (query_percentage_ranks_left >= right_threshold)
            ) / len(query_distribution)
        results.append(result)
    return pd.DataFrame(results)


def cluster_benchmark(
    data: anndata.AnnData,
    cluster_name_to_genes: dict[str, Sequence[str]],
    min_genes: int = 10,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> pd.DataFrame:
    """
    Perform benchmarking of a map based on known biological cluster of perturbations.

    :param data: AnnData object containing perturbation similarity matrix.
    :param cluster_name_to_genes: Dictionary that maps cluster name to genes in cluster.
    :param min_genes: Minimum number of genes per cluster.
    :param alternative: Defines the null and alternative hypotheses. Use `less` to test that
    non-cluster similarities are less than cluster similarities.
    :return: DataFrame containing the benchmarking results.

    """

    # Adapted from EFAAR_benchmarking https://github.com/recursionpharma/EFAAR_benchmarking/

    results = []
    assert np.all(data.var.index == data.obs.index)
    for cluster_name in cluster_name_to_genes:
        cluster_genes = cluster_name_to_genes[cluster_name]

        within_expr = data.var.index.isin(cluster_genes)
        within_data = _slice_anndata(data, within_expr, within_expr)
        if within_data.shape[0] < min_genes:
            continue
        within_vals = within_data.X[np.triu_indices(within_data.shape[0], k=1)]
        notin_in_data = _slice_anndata(data, ~within_expr, within_expr)
        between_vals = notin_in_data.X.flatten()
        ks_res = ks_2samp(within_vals, between_vals, alternative=alternative)
        results.append(
            [
                cluster_name,
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
            "within",
            "between",
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


def read_gmt(path: str) -> dict[str, set[str]]:
    """Read gene sets stored in GMT format.

    :param path: Path to GMT file.
    :return: Set name to genes
    """
    set_name_to_entries = dict()
    with fsspec.open(path, "r") as file:
        for line in file:
            fields = line.strip().split("\t")
            genes = fields[2:]
            genes = [x for x in genes if x]
            n_genes = len(genes)
            genes = set(genes)
            set_name = fields[0]
            #  set_descr = fields[1]
            assert len(genes) == n_genes, f"Duplicate gene found for {set_name}."
            if set_name in set_name_to_entries:
                raise ValueError(f"Duplicate gene set found: {set_name}.")
            set_name_to_entries[set_name] = genes
    return set_name_to_entries


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
