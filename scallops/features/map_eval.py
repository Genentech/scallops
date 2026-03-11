from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Tuple

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
