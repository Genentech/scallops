from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recall(
    true_positives_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    similarity_column_true_positives: str = "value",
    similarity_column: str = "value",
    quantiles: Sequence[float] = (0.01, 0.05),
    two_sided: bool = True,
    n_true_positives: Callable[[pd.DataFrame], int] = len,
) -> pd.DataFrame:
    """Compute recall at the specified quantiles.

    :param true_positives_df: Dataframe containing true positive pairwise similarities
    (e.g. relationships from CORUM)
    :param similarity_df: Dataframe containing all pairwise similarities from which
    quantiles are computed.
    :param similarity_column_true_positives: Column in `true_positives_df` containing
    similarity scores.
    :param similarity_column: Column in `similarity_df` containing similarity scores.
    :param quantiles: List of quantiles to extract relevant gene pairs.
    :param two_sided: If two-sided, recall is computed at specific quantiles
    and 1-quantiles.
    :param n_true_positives: Function that accepts a dataframe and returns the
    number of true positives. Default is the length of the dataframe
    :return: Dataframe containing recall results.
    """
    results = []

    n_relevant = n_true_positives(true_positives_df)
    quantiles_ = quantiles
    if two_sided:
        quantiles = np.array(quantiles)
        one_minus_quantiles = 1 - np.array(quantiles)
        quantiles_ = [
            (quantiles[i], one_minus_quantiles[i]) for i in range(len(quantiles))
        ]
        quantiles = np.concatenate((quantiles, one_minus_quantiles))
    recall_thresholds = similarity_df[similarity_column].quantile(quantiles)
    for quantile in quantiles_:
        if two_sided:
            threshold_low, threshold_high = (
                recall_thresholds[quantile[0]],
                recall_thresholds[quantile[1]],
            )

            df_retrieved = true_positives_df[
                (true_positives_df[similarity_column_true_positives] >= threshold_high)
                | (true_positives_df[similarity_column_true_positives] <= threshold_low)
            ]
        elif quantile <= 0.5:
            df_retrieved = true_positives_df[
                true_positives_df[similarity_column_true_positives]
                <= recall_thresholds[quantile]
            ]
        else:
            df_retrieved = true_positives_df[
                true_positives_df[similarity_column_true_positives]
                >= recall_thresholds[quantile]
            ]
        n_relevant_retrieved = n_true_positives(df_retrieved)
        results.append(
            [quantile[0] if two_sided else quantile, n_relevant_retrieved / n_relevant]
        )
    return pd.DataFrame(results, columns=["quantile", "recall"])


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
