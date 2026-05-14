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
