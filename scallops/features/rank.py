import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import anndata
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.delayed import delayed
from statsmodels.stats.multitest import multipletests

from scallops.features.util import _slice_anndata
from scallops.stats import _compute_pairwise_stats


def _create_rank_metadata(
    method: Literal["welch_t", "student_t", "mannwhitney"],
    reference_value: str | None,
):
    d = _compute_pairwise_stats(
        np.array([0, 1, 2]),
        np.array([10, 20, 40]) if reference_value is not None else None,
        method,
    )
    d["perturbation"] = "test"
    d["feature"] = "test"
    df = pd.DataFrame([d])
    df["FDR"] = 1.0
    df["perturbation"] = "test"
    return df


def _rank_single(
    data: anndata.AnnData,
    perturbation_column: str,
    min_labels: int,
    group_name: str | None,
    method: Literal["welch_t", "student_t", "mannwhitney"],
    reference_value: str | None,
    features: list[str],
    iqr_multiplier: float | None,
):
    perturbation_to_indices = data.obs.groupby(
        perturbation_column, observed=True
    ).indices

    if min_labels is not None and min_labels > 0:
        for key in list(perturbation_to_indices.keys()):
            n_cells = len(perturbation_to_indices[key])
            if n_cells < min_labels:
                del perturbation_to_indices[key]

    _diff_exp_func = _diff_exp

    if isinstance(data.X, da.Array):
        _diff_exp_func = delayed(_diff_exp_func)
        perturbation_to_indices = delayed(perturbation_to_indices)
        data.X = data.X.rechunk((-1, 1))
    rank_results = [
        _diff_exp_func(
            values=data.X[:, j],
            indices=perturbation_to_indices,
            method=method,
            reference_value=reference_value,
            feature=features[j],
            group_name=group_name,
            iqr_multiplier=iqr_multiplier,
        )
        for j in range(data.shape[1])
    ]
    return rank_results


def rank_features(
    data: anndata.AnnData,
    perturbation_column: str,
    reference_value: str | None,
    rank_groups: Sequence[str] | str | None = None,
    method: Literal["welch_t", "student_t", "mannwhitney"] = "welch_t",
    min_labels: int | None = None,
    iqr_multiplier: float | None = None,
):
    """Rank features for characterizing perturbations.

    :param data: Annotated data matrix.
    :param perturbation_column: Column in `data.obs` containing perturbation.
    :param reference_value: Reference value (e.g. NTC).
    :param method: Statistical method to use.
    :param rank_groups: Column(s) in `data.obs` to stratify by.
    :param min_labels: Include perturbations with at least this many observations.
    :param iqr_multiplier: Multiplier for interquartile range outlier removal.
    :return: A DataFrame with statistics for each comparison.
    """

    features = data.var.index.values
    is_dask = isinstance(data.X, da.Array)
    rank_results = []
    if rank_groups is not None:
        group_to_indices = data.obs.groupby(rank_groups, observed=True).indices
        for group_name in group_to_indices:
            rank_results += _rank_single(
                data=_slice_anndata(data, group_to_indices[group_name]),
                perturbation_column=perturbation_column,
                min_labels=min_labels,
                method=method,
                reference_value=reference_value,
                group_name=group_name,
                features=features,
                iqr_multiplier=iqr_multiplier,
            )
    else:
        rank_results += _rank_single(
            data=data,
            perturbation_column=perturbation_column,
            min_labels=min_labels,
            method=method,
            reference_value=reference_value,
            group_name=None,
            features=features,
            iqr_multiplier=iqr_multiplier,
        )

    if is_dask:
        rank_meta = _create_rank_metadata(method, reference_value)
        if rank_groups is not None and len(rank_groups) > 0:
            rank_meta["group"] = "test"
        return dd.from_delayed(rank_results, meta=rank_meta, verify_meta=False)
    else:
        return pd.concat(rank_results)


def _diff_exp(
    values: np.ndarray,
    indices: Mapping[str, np.ndarray],
    method: Literal["welch_t", "student_t", "mannwhitney"],
    reference_value: str | None,
    feature: str,
    group_name: str | None,
    iqr_multiplier: float | None,
    correction_method: Literal["fdr_bh", "fdr_by"] = "fdr_bh",
) -> dd.DataFrame:
    # resdf = _run_negative_binomial_model(df, feature, group_by, reference_group)
    if iqr_multiplier is not None:
        q1, q3 = np.nanquantile(values, [0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
    if reference_value is not None:
        reference = values[indices[reference_value]]
        reference = reference[~np.isnan(reference)]
        if iqr_multiplier is not None:
            reference = reference[
                (reference >= lower_bound) & (reference <= upper_bound)
            ]
    else:
        reference = None
    de_results = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for key in indices:
            if key != reference_value:
                treatment = values[indices[key]]
                treatment = treatment[~np.isnan(treatment)]
                if iqr_multiplier is not None:
                    treatment = treatment[
                        (treatment >= lower_bound) & (treatment <= upper_bound)
                    ]
                d = _compute_pairwise_stats(
                    treatment,
                    reference,
                    method,
                )

                d["perturbation"] = key
                d["feature"] = feature
                de_results.append(d)
    if len(de_results) == 0:
        empty_df = _create_rank_metadata(method, reference_value)
        if group_name is not None:
            empty_df["group"] = group_name
        return empty_df.query("`p-value`<0")

    de_results = pd.DataFrame(de_results)
    de_results["p-value"] = de_results["p-value"].fillna(1)
    de_results["FDR"] = multipletests(
        de_results["p-value"].values, method=correction_method
    )[1]

    de_results["FDR"] = de_results["FDR"].fillna(1)

    if group_name is not None:
        de_results["group"] = (
            "_".join([str(x) for x in group_name])
            if isinstance(group_name, tuple)
            else str(group_name)
        )
    return de_results
