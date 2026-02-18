from typing import Literal

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
    :return: Dataframe containing pairs of genes found in CORUM
    """

    corum_gene_names = pd.read_csv(path, usecols=["subunits_gene_name"], sep="\t")[
        "subunits_gene_name"
    ].values
    pairs = set()
    for i in range(len(corum_gene_names)):
        cluster = corum_gene_names[i].split(";")
        for j in range(len(cluster)):
            for k in range(j):
                pairs.add((cluster[j], cluster[k]))
                pairs.add((cluster[k], cluster[j]))
    a = []
    b = []
    for p in pairs:
        a.append(p[0])
        b.append(p[1])
    return pd.DataFrame(data=dict(a=a, b=b))
