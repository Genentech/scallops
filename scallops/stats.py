"""Statistical Analysis Module.

This module provides functions for conducting statistical analysis on data, including hypothesis testing,
meta-analysis, and confidence interval estimation.

Authors:
    - The SCALLOPS development team
"""

import warnings
from collections import namedtuple
from functools import partial
from math import comb
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import chi2, mannwhitneyu, median_abs_deviation, norm, ttest_ind
from sklearn.neighbors import NearestNeighbors
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.multitest import multipletests


def _compute_pairwise_stats(
    treatment: np.ndarray,
    reference: np.ndarray,
    method: Literal["welch_t", "student_t", "mannwhitney"],
    **kwargs,
) -> dict:
    """Helper to compute statistics for one pairwise comparison."""
    if method == "mannwhitney":
        result = mannwhitneyu(reference, treatment, **kwargs)
    elif method == "welch_t":
        result = ttest_ind(reference, treatment, equal_var=False, **kwargs)
    else:  # student_t
        result = ttest_ind(reference, treatment, equal_var=True, **kwargs)

    reference_mean = reference.mean()
    treatment_mean = treatment.mean()
    epsilon = np.finfo(float).eps
    fold_change = treatment_mean / (reference_mean + epsilon)

    d = {
        "statistic": result.statistic,
        "p-value": result.pvalue,
        "fold_change": fold_change,
        "log2FoldChange": np.log2(max(fold_change, epsilon)),
        "reference_mean": reference_mean,
        "treatment_mean": treatment_mean,
        "treatment_n": len(treatment),
        "reference_n": len(reference),
    }

    if method == "mannwhitney":
        auc = result.statistic / (len(reference) * len(treatment))
        d["CLES"] = auc

    return d


def _run_negative_binomial_model(
    df: pd.DataFrame,
    feature: str,
    group_by: str,
    reference_group: str,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: bool = False,
    min_n: int = 3,
    ci_level: float = 0.95,
    maxiter: int = 10000,
    exposure=None,
    offset=None,
    skip_zero_variance: bool = True,
    **kwargs,
):
    """
    Pairwise NB2 with corrected Design Matrix construction.
    """
    if verbose:
        print(f"--- NB2 Model: {feature} vs {group_by} (Ref: {reference_group}) ---")

    feat_num = pd.to_numeric(df[feature], errors="coerce")
    grp = df[group_by]
    mask = feat_num.notna() & grp.notna()

    def _aligned_series(arr):
        if arr is None:
            return None
        if isinstance(arr, pd.Series):
            s = pd.to_numeric(arr, errors="coerce").reindex(df.index)
        else:
            a = np.asarray(arr)
            if a.shape[0] != len(df):
                raise ValueError("Length mismatch")
            s = pd.Series(a, index=df.index)
            s = pd.to_numeric(s, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    exposure_s = _aligned_series(exposure)
    offset_s = _aligned_series(offset)

    if exposure_s is not None:
        mask &= exposure_s.notna()
    if offset_s is not None:
        mask &= offset_s.notna()

    # Verbose: Data Filtering
    n_dropped = len(df) - mask.sum()
    if verbose and n_dropped > 0:
        print(f"   > Dropped {n_dropped} rows due to missing values/groups.")

    if not mask.any():
        if verbose:
            print("   > No valid data remaining. Returning empty.")
        return pd.DataFrame()

    y_all = feat_num[mask].to_numpy()
    groups_all = grp[mask].to_numpy()

    # Offsets
    if exposure_s is not None:
        exp_clean = np.maximum(exposure_s[mask].to_numpy(), 1e-12)
        log_exp = np.log(exp_clean)
    else:
        log_exp = None
    off_clean = offset_s[mask].to_numpy() if offset_s is not None else None

    if log_exp is not None and off_clean is not None:
        offset_all = off_clean + log_exp
    elif log_exp is not None:
        offset_all = log_exp
    else:
        offset_all = off_clean

    # Quick Negative Check
    if (y_all < 0).any():
        if verbose:
            print("   > Negative values detected in feature. NB requires count data.")
        return pd.DataFrame()

    group_to_idx = {g: np.flatnonzero(groups_all == g) for g in pd.unique(groups_all)}
    if reference_group not in group_to_idx:
        if verbose:
            print(f"   > Reference group '{reference_group}' not found in data.")
        return pd.DataFrame()

    ref_idx = group_to_idx[reference_group]
    y_ref = y_all[ref_idx]
    n_ref = y_ref.size

    # Targets
    targets = [
        (g, group_to_idx[g])
        for g in group_to_idx.keys()
        if g != reference_group and group_to_idx[g].size >= min_n
    ]

    if not targets:
        if verbose:
            print(f"   > No target groups met min_n={min_n}.")
        return pd.DataFrame()

    if n_ref < min_n:
        if verbose:
            print(f"   > Reference n={n_ref} is below min_n={min_n}.")
        return pd.DataFrame()

    if verbose:
        print(f"   > Processing {len(targets)} comparisons (n_ref={n_ref})...")

    ref_mean = float(np.mean(y_ref))
    ci_alpha = 1.0 - ci_level

    # --- Worker Function ---
    def _fit_one(target_label, idx_tgt):
        n_target = idx_tgt.size
        idx_subset = np.concatenate([ref_idx, idx_tgt])
        y_subset = y_all[idx_subset]

        # Zero Variance Check
        if skip_zero_variance and np.allclose(y_subset, y_subset[0]):
            if verbose:
                print(f"     [Skip] {target_label}: Zero variance detected.")
            return {
                "group": target_label,
                "statistic": 0.0,
                "p-value": 1.0,
                "fold_change": 1.0,
                "log2FoldChange": 0.0,
                "conf_int_low": 0.0,
                "conf_int_high": 0.0,
                "alpha": np.nan,
                "treatment_mean": float(np.mean(y_subset[n_ref:])),
                "reference_mean": ref_mean,
                "n_ref": n_ref,
                "n_target": n_target,
            }

        # Col 0: Intercept (All 1s)
        # Col 1: Group Dummy (0 for Ref, 1 for Target)
        x_subset = np.zeros((n_ref + n_target, 2), dtype=float)
        x_subset[:, 0] = 1.0  # Intercept is always 1
        x_subset[n_ref:, 1] = 1.0  # Target group gets 1, Ref stays 0

        offset_subset = offset_all[idx_subset] if offset_all is not None else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = NegativeBinomial(
                    y_subset, x_subset, offset=offset_subset, loglike_method="nb2"
                )
                res = model.fit(method="bfgs", maxiter=maxiter, disp=0)
            # Extract Stats
            beta = float(res.params[1])
            pval = float(res.pvalues[1])
            t_stat = float(res.tvalues[1])
            ci_beta = res.conf_int(alpha=ci_alpha)[1]

            return {
                "group": target_label,
                "statistic": t_stat,
                "p-value": pval,
                "fold_change": float(np.exp(beta)) if np.isfinite(beta) else np.nan,
                "log2FoldChange": float(beta / np.log(2.0))
                if np.isfinite(beta)
                else np.nan,
                "conf_int_low": float(ci_beta[0] / np.log(2.0)),
                "conf_int_high": float(ci_beta[1] / np.log(2.0)),
                "alpha": float(res.params[-1]),
                "treatment_mean": float(np.mean(y_subset[n_ref:])),
                "reference_mean": ref_mean,
                "n_ref": n_ref,
                "n_target": n_target,
            }

        except Exception as e:
            if verbose:
                print(f"     [Error] Fit failed for {target_label}: {e}")
            return {
                "group": target_label,
                "statistic": np.nan,
                "p-value": np.nan,
                "fold_change": np.nan,
                "log2FoldChange": np.nan,
                "conf_int_low": np.nan,
                "conf_int_high": np.nan,
                "alpha": np.nan,
                "treatment_mean": float(np.mean(y_subset[n_ref:])),
                "reference_mean": ref_mean,
                "n_ref": n_ref,
                "n_target": n_target,
            }

    # Parallel Execution
    # Reduced verbosity from 10 (spammy) to 1 (progress bar) to allow custom prints to be seen
    joblib_verbose = 1 if verbose else 0

    if n_jobs != 1:
        out = Parallel(n_jobs=n_jobs, backend=backend, verbose=joblib_verbose)(
            delayed(_fit_one)(label, idx) for (label, idx) in targets
        )
    else:
        out = [_fit_one(label, idx) for (label, idx) in targets]

    out = [r for r in out if r is not None]

    if verbose:
        print(f"   > Completed. {len(out)} successful results generated.")

    return pd.DataFrame(out) if out else pd.DataFrame()


def group_differences_test(
    df: pd.DataFrame,
    feature: str,
    group_by: str,
    reference_group: str,
    method: Literal[
        "welch_t", "student_t", "mannwhitney", "negative_binomial"
    ] = "welch_t",
    correction_method: Literal["fdr_bh", "fdr_by"] = "fdr_bh",
    fdr_floor: float = 1e-10,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Test for difference between the `reference` group and all other groups.

    This function provides a unified interface for both pairwise testing (with
    multiple testing correction) and negative binomial regression models. It
    returns a standardized DataFrame with core statistical columns.

    :param df: DataFrame with feature and group columns.
    :param feature: Name of the column with the desired feature.
    :param group_by: Name of the column to group by.
    :param reference_group: Name of the group to be set as reference.
    :param method: Statistical method to use. One of "welch_t", "student_t", "mannwhitney", "negative_binomial".
        Defaults to 'welch_t'.
    :param correction_method: Method for multiple testing correction. Either "fdr_bh", "fdr_by". Defaults to fdr_bh.
    :param fdr_floor: Floor for p-values when calculating -log10FDR.
    :param n_jobs: Number of parallel jobs to run. Defaults to -1 (all cores).
    :param backend: Joblib backend to use. Defaults to 'loky'.
    :param verbose: If True, print progress messages.
    :param kwargs: Additional keyword arguments passed to the underlying statistical method.

        **For 'negative_binomial'**, valid arguments include:
        - ``min_n`` (int): Minimum samples required per group (default: 3).
        - ``ci_level`` (float): Confidence interval level (default: 0.95).
        - ``method`` (str): Optimization method, 'newton' or 'bfgs' (default: 'bfgs').
        - ``maxiter`` (int): Maximum iterations for optimization (default: 1000).
        - ``exposure`` (pd.Series or array): Exposure values for offset (default: None).
        - ``offset`` (pd.Series or array): Fixed offset values (default: None).
        - ``use_start_params`` (bool): Use heuristic start parameters to speed convergence (default: True).
        - ``skip_zero_variance`` (bool): Skip groups with zero variance (all equal values) (default: True).

        **For 'welch_t', 'student_t', 'mannwhitney'**, arguments are passed to the respective
        SciPy functions (e.g., ``alternative``, ``nan_policy``).

    :return: A DataFrame with comprehensive statistics for each comparison.
    """
    if verbose:
        print(
            f"Running {method} with {reference_group} as reference and {group_by} as targets"
        )
    if feature not in df.columns:
        raise KeyError(f"Feature column '{feature}' not found in the DataFrame.")
    if group_by not in df.columns:
        raise KeyError(f"Group-by column '{group_by}' not found in the DataFrame.")
    if reference_group not in df[group_by].unique():
        raise ValueError(
            f"Reference group '{reference_group}' not in column '{group_by}'."
        )

    if method == "negative_binomial":
        if verbose:
            print("Starting the negative binomial model")
        # Pass n_jobs and backend explicitly
        resdf = _run_negative_binomial_model(
            df,
            feature,
            group_by,
            reference_group,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            **kwargs,
        )
    else:
        if verbose:
            print(f"Starting the {method}")
        grouped = df.groupby(group_by)
        reference = grouped.get_group(reference_group)[feature].values

        # Worker function for pairwise stats
        def _per_group(tr, d_df):
            if verbose:
                print(f"{reference_group} vs {tr}")
            if tr != reference_group:
                treatment_vals = d_df[feature].values
                stats = _compute_pairwise_stats(
                    treatment_vals, reference, method, **kwargs
                )
                stats["group"] = tr
                return stats
            return None

        joblib_verbose = 10 if verbose else 0

        # Use joblib Parallel instead of list comprehension
        if n_jobs != 1:
            results = Parallel(n_jobs=n_jobs, backend=backend, verbose=joblib_verbose)(
                delayed(_per_group)(tr, d) for tr, d in grouped
            )
        else:
            results = [_per_group(tr, d) for tr, d in grouped]

        # Filter out None results (reference vs reference)
        results = [r for r in results if r is not None]
        resdf = pd.DataFrame(results).dropna(subset=["statistic", "p-value"])

    if not resdf.empty:
        resdf["FDR"] = multipletests(resdf["p-value"].values, method=correction_method)[
            1
        ]

    if resdf.empty:
        return pd.DataFrame()

    resdf["FDR"] = resdf["FDR"].fillna(1)
    resdf["-log10FDR"] = -np.log10(np.maximum(resdf["FDR"], fdr_floor))

    core_cols = [
        "group",
        "statistic",
        "p-value",
        "FDR",
        "-log10FDR",
        "fold_change",
        "log2FoldChange",
    ]
    extra_cols = [col for col in resdf.columns if col not in core_cols]

    final_df = (
        resdf[core_cols + extra_cols].sort_values(by="FDR").reset_index(drop=True)
    )

    return final_df


def fisher_combined_pvalue(p_values: Sequence[float]) -> float:
    """Combine p-values using Fisher's method[1].

    :param p_values: List of individual p-values (list of float) to be combined.
    :return: Combined p-value (float).

    .. note ::
        Fisher's method is used to combine independent p-values from multiple
        statistical tests into an overall test statistic.

        The input p-values are transformed into chi-squared statistics, and
        the combined chi-squared statistic is used to calculate the combined p-value.


        [1] Fisher, R. A. (1932). Statistical Methods for Research Workers.
        [2] https://en.wikipedia.org/wiki/Fisher%27s_method

    :example:

        .. code-block:: python

            individual_p_values = [0.01, 0.05, 0.2]
            combined_p_value = fisher_combined_pvalue(individual_p_values)
            print(f"Combined P-value: {combined_p_value}")
    """
    # Ensure p-values are not 0
    p_values = np.maximum(p_values, np.finfo(float).eps)
    # Transform p-values to chi-squared statistics
    chi_squared_stats = -2 * np.sum(np.log(p_values))
    # Calculate degrees of freedom
    degrees_of_freedom = 2 * len(p_values)
    # Calculate combined p-value
    combined_p_value = 1 - chi2.cdf(chi_squared_stats, degrees_of_freedom)
    return combined_p_value


def fixed_effects_meta_analysis_mannwhitneyu(
    U_statistics: Sequence[float], sample_sizes: Sequence[tuple[int, int]]
) -> tuple[float, float]:
    """Perform meta-analysis of Mann-Whitney U tests using inverse variance as weights.

    :param U_statistics: List of Mann-Whitney U statistics from individual studies.
    :param sample_sizes: List of sample sizes corresponding to each study.
    :return: Combined p-value.

    .. note::
        The function assumes that the U_statistics and sample_sizes lists have the same length.

        The combined p-value is calculated using the inverse variance method, where the weight
        for each study is the inverse of the variance of the corresponding U statistic.

        [1] Borenstein, M., Hedges, L. V., Higgins, J. P., & Rothstein, H. R. (2009).
        Introduction to Meta-Analysis. Wiley.

    :example:

        .. code-block:: python

            U_statistics = [20, 25, 30]
            sample_sizes = [(50, 50), (60, 60), (70, 70)]
            combined_p_value = fixed_effects_meta_analysis_mannwhitneyu(
                U_statistics, sample_sizes
            )
            print(f"Combined P-value: {combined_p_value}")
    """
    assert len(U_statistics) == len(sample_sizes), (
        "Sample sizes and U-statistics should have the same length"
    )
    assert all([len(x) == 2 for x in sample_sizes]), (
        "Sample sizes for both groups must be provided"
    )

    def _compute_variance(n1: int, n2: int) -> float:
        return ((n1 * n2) * (n1 + n2 + 1)) / 12

    # Calculate weights based on inverse variance
    weights = [
        1 / _compute_variance(n1, n2) if U != 0 else 0
        for U, (n1, n2) in zip(U_statistics, sample_sizes)
    ]
    # Calculate combined effect size
    combined_effect_size = np.sum(np.multiply(weights, U_statistics)) / np.sum(weights)
    # Calculate combined standard error
    combined_std_error = np.sqrt(1 / np.sum(weights))
    # Calculate Z-score
    z_score = combined_effect_size / combined_std_error
    return 2 * norm.cdf(-abs(z_score)), combined_effect_size  # two-tailed p-value


def auc_variance_exact(m, n, k) -> float:
    """Calculate the variance of the Area Under the Receiver Operating Characteristic Curve (AUC).

    :param m: int
        The sample size of the first group.
    :param n: int
        The sample size of the second group.
    :param k: int
        A factor related to the hypothesis testing context.

    :return: float
        The variance of the AUC estimate.

    .. note::

        This function calculates the variance of the AUC estimate based on a specific formula.
        The formula is derived from the statistical method proposed by Cortes and Mohri [1]
        for hypothesis testing involving two independent binomial proportions and the AUC.

        The parameters m, n, and k represent the sample sizes and a factor relevant to the hypothesis testing.

        .. [1] Cortes, C., & Mohri, M. (2005). Confidence intervals for the area under the ROC curve.
            In Advances in neural information processing systems (pp. 305-312).

    :example:

        .. code-block:: python

            m = 30
            n = 40
            k = 5
            var_auc = auc_variance_exact(m, n, k)
            print(f"Variance of AUC: {var_auc}")
    """

    def _z_i(i):
        c = partial(comb, mn + 1 - i)
        d = partial(comb, mn + 1)
        nc = np.vectorize(c)
        nd = np.vectorize(d)
        num = nc(np.arange(0, k - 1))
        den = nd(np.arange(0, k))
        return num.sum() / den.sum()

    mn = m + n
    mnp1 = mn + 1
    m2n2 = (m**2) * (n**2)
    t = (3 * (((m - n) ** 2) + mn)) + 2
    q0 = (
        (mnp1 * (t * k**2))
        + (
            (
                (((-3 * n**2) + (3 * m * n) + (3 * m) + 1) * t)
                - ((12 * (3 * m * n)) + mn)
                - 8
            )
            * k
        )
        + (((-3 * m**2) + (7 * m) + (10 * n) + (3 * m * n) + 10) * t)
        - (4 * ((3 * m * n) + mn + 1))
    )
    q1 = (
        (t * k**3)
        + ((3 * (m - 1)) * t * k**2)
        + (
            ((((-3 * n**2) + (3 * m * n) - (3 * m) + 8) * t) - (6 * ((6 * m * n) + mn)))
            * k
        )
        + (((-3 * m**2) + (7 * (m + n)) + (3 * m * n)) * t)
        - (2 * ((6 * m * n) + n + m))
    )
    var = (
        mnp1
        * mn
        * (mn - 1)
        * t
        * (((mn - 2) * _z_i(4)) - (((2 * m) - n + (3 * k) - 10) * _z_i(3)))
    ) / (72 * m2n2)
    var += (
        mnp1
        * mn
        * t
        * (
            (m**2)
            - (n * m)
            + (3 * k * m)
            - (5 * m)
            + (2 * k**2)
            - (n * k)
            + 12
            - (9 * k)
        )
        * _z_i(2)
    ) / (48 * m2n2)
    var -= ((mnp1**2) * ((m - n) ** 4) * (_z_i(1) ** 2)) / (16 * m2n2)
    var -= (mnp1 * q1 * _z_i(1)) / (72 * m2n2)
    var += (k * q0) / (144 * m2n2)
    return var


def iqr_threshold(image, multiplier=3.4) -> float:
    """Compute the interquartile range (IQR) threshold on an image.

    The IQR threshold is calculated as Q3 + (multiplier * IQR), where Q3 is the third quartile (75th percentile)
    and IQR is the interquartile range (Q3 - Q1). This threshold is commonly used for outlier detection.

    :param image: N-dimensional array representing the image to be analyzed.
    :param multiplier: Multiplier to adjust the significance of the IQR. By default, 3.4 is used, which is
        equivalent to approximately 2 standard deviations in a normal distribution.
    :return: IQR threshold.

    :example:

        .. code-block:: python

            import numpy as np
            from scallops.stats import iqr_threshold

            # Create a synthetic image
            image = np.random.rand(100, 100)

            # Compute the IQR threshold
            threshold = iqr_threshold(image)
    """
    q1, q3 = np.percentile(image, [25, 75])
    iqr = q3 - q1
    return q3 + (multiplier * iqr)


def iqr_outlier(image: np.ndarray, multiplier: float) -> np.ndarray:
    """Identifies the outliers using the interquartile method.

    :param image: numpy array with single channel image data
    :param multiplier: Multiplier to the IQR for adjusting the significance (if normal 1.7 is
        equivalent to 1 std)
    """
    img = image.squeeze().copy()
    outlier_thr = iqr_threshold(img[img > 0], multiplier=multiplier)
    img[img < outlier_thr] = 0
    return img


def vectorized_iqr_outlier(image: np.ndarray, multiplier: float) -> np.ndarray:
    """Detect outliers in each channel of the intensity image using a vectorized approach.

    :param image: numpy array with single channel image data
    :param multiplier: Multiplier to the IQR for adjusting the significance (if normal 1.7 is
        equivalent to 1 std)
    """
    intensity_image_with_nan = np.where(image == 0, np.nan, image)
    q1 = np.nanpercentile(intensity_image_with_nan, 25, axis=(0, 1), keepdims=True)
    q3 = np.nanpercentile(intensity_image_with_nan, 75, axis=(0, 1), keepdims=True)
    iqr = q3 - q1
    threshold = q3 + (multiplier * iqr)
    return np.where(image < threshold, 0, image)


def generate_cdfs_with_area_difference(
    target_area: float,
    n_points: int = 1000,
    n_data: int = 1000,
    iterations: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Generates two realistic CDFs from normal distributions with a specified area difference
    between them, along with the raw data corresponding to each CDF.

    :param target_area: The target area difference between the two CDFs.
    :param n_points: Number of points to use for generating the CDFs.
    :param n_data: Number of data points to generate for each distribution.
    :param iterations: Number of iterations to adjust the mean of the second distribution to achieve the target area.
    :param seed: Random seed for reproducibility.
    :return: A tuple containing two arrays representing the CDFs, the x-axis values for these CDFs, the final area,
             and the raw data for each distribution.

    :example:

        .. code-block:: python

            target_area = 0.1
            x, cdf1, cdf2, final_area, data1, data2 = (
                generate_cdfs_with_area_difference(target_area)
            )
            print(f"Final area between CDFs: {final_area}")
    """
    np.random.seed(seed)  # Set the random seed for reproducibility

    # Initial parameters for the two normal distributions
    # First distribution (reference)
    mean1, std1 = 0, 1
    # Search bounds for the second mean
    mean2_lower, mean2_upper = (
        mean1 - 5 * std1,
        mean1 + 5 * std1,
    )
    for _ in range(iterations):
        mean2 = (mean2_lower + mean2_upper) / 2
        data1 = np.random.normal(mean1, std1, n_data)
        data2 = np.random.normal(mean2, std1, n_data)

        # Compute empirical CDFs
        x = np.linspace(
            min(np.min(data1), np.min(data2)),
            max(np.max(data1), np.max(data2)),
            n_points,
        )
        cdf1 = np.array([np.mean(data1 <= v) for v in x])
        cdf2 = np.array([np.mean(data2 <= v) for v in x])

        # Calculate the area between the CDFs
        area = np.trapz(np.abs(cdf1 - cdf2), dx=1) / n_points

        # Adjust the search bounds based on the current area
        if area < target_area:
            mean2_lower = mean2
        else:
            mean2_upper = mean2

        if np.isclose(area, target_area, atol=1e-4):
            break
    assert np.isclose(area, target_area, atol=1e-4), (
        "Convergence error: Try increasing the iterations"
    )
    final_area = area

    return x, cdf1, cdf2, final_area, data1, data2


def scale_array(data, min_scale=-1, max_scale=1) -> np.ndarray:
    """Scales an array to a specified range.

    :param data: The input array to be scaled.
    :param min_scale: The minimum value of the scaled array, defaults to -1.
    :param max_scale: The maximum value of the scaled array, defaults to 1.
    :return: The scaled array.

    :example:

        .. code-block:: python

            import numpy as np

            # Create an array
            arr = np.array([0, 5, 10])
            # Scale the array to range 0 to 1
            scaled_arr = scale_array(arr, 0, 1)
            print(scaled_arr)  # Output: [0.  0.5 1. ]
    """
    min_val = np.min(data)
    max_val = np.max(data)
    scale = max_scale - min_scale
    scaled_data = ((data - min_val) / (max_val - min_val)) * scale + min_scale
    return scaled_data


def compute_area_between_cdfs(
    reference: np.ndarray,
    target: dict[str, np.ndarray] | np.ndarray,
    reference_label: str = "reference",
    npoints: int = 500,
    area_as_delta: bool = False,
    plot: bool = False,
) -> tuple[
    dict[str, float],
    dict[str, np.ndarray],
    np.ndarray,
    tuple[plt.Axes | None, Figure | None],
]:
    """Computes the area between the CDFs of a reference data set and other target data sets.

    This function calculates the area between the cumulative distribution functions (CDFs) of a reference dataset
    and one or more target datasets. It can also plot the CDFs if required.

    :param reference: np.ndarray
        Array of data points for the reference dataset.
    :param target: dict[str, np.ndarray] | np.ndarray
        Dictionary where keys are labels for each target dataset and values are arrays of data points,
        or a single array of data points for one target dataset.
    :param reference_label: str, optional
        Label for the reference dataset in the plot. Default is "reference".
    :param npoints: int, optional
        Number of points to use for generating the CDFs. Default is 500.
    :param area_as_delta: bool, optional
        If True, return the area as raw difference instead of absolute value. Default is False.
    :param plot: bool, optional
        If True, plot the CDFs. Default is False.
    :return: tuple
        - dict[str, float]: Dictionary of areas between the reference CDF and each target CDF.
        - dict[str, np.ndarray]: Dictionary of CDFs for the reference and each target dataset.
        - np.ndarray: Array of bin edges used for generating the CDFs.
        - tuple[plt.Axes | None, Figure | None]: Tuple containing the axis and figure objects if `plot` is True, otherwise (None, None).

    :example:

        .. code-block:: python

            data_sets = {
                "reference": np.random.normal(0, 1, 1000),
                "target1": np.random.normal(0.5, 1.5, 1000),
            }
            areas, cdfs, bin_edges, (ax, f) = compute_area_between_cdfs(
                data_sets["reference"], data_sets, npoints=1000
            )
            print(
                f"Computed area between the reference and target1: {areas['target1']}"
            )
    """
    areas = {}
    cdfs = {}
    ax = f = None
    if isinstance(target, (np.ndarray, pd.Series)):
        target = {"Target": target}

    def _compute_cdf(data, bin_edges):
        pdf, _ = np.histogram(data, bins=bin_edges, density=True)
        scaled_pdf = pdf * np.diff(bin_edges)
        assert np.isclose(scaled_pdf.sum(), 1), (
            f"The sum of the scaled PDF does not equal 1 ({scaled_pdf.sum()})"
        )
        cdf = np.cumsum(scaled_pdf)
        return cdf

    merged = np.concatenate([reference] + list(target.values()), axis=None)
    mini, maxi = np.quantile(merged, [0, 1])
    bin_edges = np.linspace(mini, maxi, num=npoints)
    cdf_ref = _compute_cdf(reference, bin_edges)
    cdfs["reference"] = cdf_ref
    if plot:
        f, ax = plt.subplots()
        ax.plot(bin_edges[:-1], cdf_ref, label=reference_label, linestyle="dashed")

    for label, data2 in target.items():
        cdf_target = _compute_cdf(data2, bin_edges)
        cdfs[label] = cdf_target
        if plot:
            ax.plot(bin_edges[:-1], cdf_target, label=label)
            ax.legend()
        diff_cdf = (
            cdf_ref - cdf_target if area_as_delta else np.abs(cdf_ref - cdf_target)
        )
        areas[label] = np.trapz(diff_cdf, dx=1) / (bin_edges.shape[0] - 1)

    return areas, cdfs, bin_edges, (ax, f)


def random_effects_meta_analysis_mannwhitneyu(
    U_statistics: Sequence[float] | dict[str, float],
    sample_sizes: Sequence[tuple[int, int]],
    pvalues: Sequence[float],
    estimation_method: Literal["pm", "dl"] = "pm",
    alpha: float = 0.05,
) -> namedtuple:
    """Perform meta-analysis on Mann-Whitney U statistics by converting them to CLES, and then using
    statsmodels to combine the effect sizes under a random-effects model.

    :param pvalues: Pvalues to match the Us.
    :param alpha: Significance level for the bootstrap CI estimation. Default is 0.05.
    :param U_statistics: List of Mann-Whitney U statistics from individual studies or a dictionary
        where keys are study names and values are U statistics.
    :param sample_sizes: List of tuples (n1, n2) representing sample sizes for each study.
    :param estimation_method: Method for the estimation of Tau's effect size and variance.
        Can be either 'pm' as Paule-Mandel or 'dl' as DerSimonian-Laird. Default 'pm'.
    :return: A namedtuple containing the combined effect size, its variance, CI if applicable, and the p-value.

    :example:

        .. code-block:: python

            from scallops.stats import random_effects_meta_analysis_mannwhitneyu

            U_statistics_example = [120, 150, 180]
            sample_sizes_example = [(30, 30), (40, 40), (50, 50)]
            individual_p_values = [0.1, 0.05, 0.025]
            result = random_effects_meta_analysis_mannwhitneyu(
                U_statistics_example, sample_sizes_example, individual_p_values
            )
            print(
                f"Combined effect: {result.combined_effect}, Variance: {result.combined_variance}, P-value: {result.pvalue}"
            )
    """
    result = namedtuple(
        "result",
        ["combined_effect", "combined_variance", "ci", "pvalue", "tau_squared"],
        defaults=[None],
    )
    pvalue = fisher_combined_pvalue(pvalues)
    if isinstance(U_statistics, list):
        row_names = None
    elif isinstance(U_statistics, dict):
        row_names = list(U_statistics.keys())
        U_statistics = U_statistics.values()
    else:
        raise NotImplementedError
    cles = np.zeros(len(U_statistics))
    variances = np.zeros(len(U_statistics))
    for i, (U, (n1, n2)) in enumerate(zip(U_statistics, sample_sizes)):
        cles[i] = U / (n1 * n2)
        variances[i] = (n1 + n2 + 1) / (12 * n1 * n2)
    if estimation_method == "reml":
        combined_effect, combined_variance, tau_squared = combine_effects_reml(
            cles, variances
        )
        tts = stats.t.ppf(1 - (alpha / 2), len(combined_effect) - 1) * combined_variance
        ci = (combined_effect - tts, combined_effect + tts)
    else:
        res = combine_effects(
            effect=cles,
            variance=variances,
            method_re=estimation_method,
            row_names=row_names,
        )
        combined_effect, combined_variance, tau_squared, ci = (
            res.mean_effect_re,
            np.sqrt(res.var_eff_w_re),
            res.tau2,
            res.conf_int(alpha)[1],
        )
    res = result(
        combined_effect=combined_effect,
        combined_variance=combined_variance,
        pvalue=pvalue,
        ci=ci,
        tau_squared=tau_squared,
    )
    return res


def reml_likelihood(
    tau_squared: float, effects: np.ndarray, variances: np.ndarray
) -> float:
    """Calculates the negative log-likelihood for a random-effects meta-analysis model.

    :param tau_squared: The between-study variance.
    :param effects: Array-like, the observed effect sizes from the studies.
    :param variances: Array-like, the within-study variances of the effect sizes.
    :return: The negative log-likelihood value for the given tau_squared.

    :example:

        .. code-block:: python

            effects = np.array([0.2, 0.5, 0.3])
            variances = np.array([0.01, 0.02, 0.015])
            tau_squared_initial = 0.1
            likelihood = reml_likelihood(tau_squared_initial, effects, variances)
            print(likelihood)
    """
    # Validate input data
    if len(effects) != len(variances):
        raise ValueError("Effects and variances arrays must have the same length.")

    if tau_squared < 0:
        raise ValueError("Tau squared must be non-negative.")

    # Compute total variance
    total_variance = variances + tau_squared

    # Compute weighted effects
    weighted_effects = effects / total_variance

    # Compute negative log-likelihood
    likelihood = -0.5 * (
        np.sum(np.log(total_variance)) + np.sum(effects * weighted_effects)
    )

    return -likelihood


def estimate_reml(effects: np.ndarray, variances: np.ndarray) -> float:
    """Estimates the between-study variance (tau_squared) using REML.

    :param effects: Array-like, the observed effect sizes from the studies.
    :param variances: Array-like, the within-study variances of the effect sizes.
    :return: The estimated tau_squared.

    :example:

    .. code-block:: python

            effects = np.array([0.2, 0.5, 0.3])
            variances = np.array([0.01, 0.02, 0.015])
            tau_squared = estimate_reml(effects, variances)
            print(tau_squared)
    """
    # Initial guess for tau_squared
    initial_tau_squared = np.var(effects) * 0.5

    # Perform REML estimation
    result = minimize(
        reml_likelihood,
        x0=initial_tau_squared,
        args=(effects, variances),
        bounds=[(0, None)],
    )

    # Check if optimization converged successfully
    if result.success:
        return result.x[0]
    else:
        raise RuntimeError("REML estimation failed to converge")


def combine_effects_reml(
    effects: np.ndarray, variances: np.ndarray
) -> tuple[float, float, float]:
    """Combines effect sizes using REML in a random-effects meta-analysis.

    :param effects: Array-like, the observed effect sizes from the studies.
    :param variances: Array-like, the within-study variances of the effect sizes.
    :return: Tuple containing the combined effect size, its variance, and the estimated tau_squared.

    :example:

        .. code-block:: python

            effects = np.array([0.2, 0.5, 0.3])
            variances = np.array([0.01, 0.02, 0.015])
            combined_effect, combined_variance, tau_squared = combine_effects_reml(
                effects, variances
            )
            print(
                f"Combined effect: {combined_effect}, Combined variance: {combined_variance}, Tau^2: {tau_squared}"
            )
    """
    # Estimate tau_squared using REML
    tau_squared = estimate_reml(effects, variances)
    # Calculate weights
    weights = 1 / (variances + tau_squared)
    # Combine effect sizes
    combined_effect = np.sum(weights * effects) / np.sum(weights)
    # Calculate combined variance
    combined_variance = 1 / np.sum(weights)
    return combined_effect, combined_variance, tau_squared


def guide_pvalue_ABC(
    non_targeting: np.ndarray, targeting: np.ndarray, reps: int = 10000, cpus: int = -1
) -> tuple[float, float, np.ndarray]:
    """Computes the p-value for the difference in areas between the CDFs of two datasets.

    This function compares the cumulative distribution functions (CDFs) of two datasets,
    `non_targeting` and `targeting`, by computing the area between them. It then performs
    bootstrapping to generate empirical p-values for the observed difference in areas.

    :param non_targeting: The data for the non-targeting group.
    :param targeting: The data for the targeting group.
    :param reps: The number of bootstrapping repetitions. Default is 10000.
    :param cpus: The number of CPUs to use for parallel computation. Default is -1 (use all available CPUs).
    :return: A tuple containing the observed difference in areas, the computed p-value, and the bootstrapped samples.
    """
    # Compute area between CDFs for the non-targeting group
    base = compute_area_between_cdfs(
        non_targeting, {"b": targeting}, reference_label="a", area_as_delta=True
    )[0]["b"]

    # Function for bootstrapping
    def _boot():
        rng = np.random.default_rng()
        d = {
            "Target": rng.choice(non_targeting, size=targeting.shape[0], replace=True),
        }
        return compute_area_between_cdfs(
            non_targeting, d, reference_label="NTC", area_as_delta=True
        )[0]["Target"]

    # Perform bootstrapping in parallel
    bootstrapped = Parallel(n_jobs=cpus)(delayed(_boot)() for _ in range(reps))
    bootstrapped = np.array(bootstrapped)
    # Compute empirical p-value
    pval = (
        max(min((bootstrapped > base).mean(), (bootstrapped < base).mean()), 1 / reps)
        * 2
    )
    return base, pval, bootstrapped


def _local_zscore_transform(
    values: np.ndarray,
    indices: np.ndarray,
    norm_method: Literal["classic", "robust"] = "classic",
) -> np.ndarray:
    """Computes the local z-score for a single feature.
    :param values: Feature values
    :param indices: Nearest neighbor indices
    :param norm_method: Normalization method, either 'classic' (mean and std) or
        'robust' (median and MAD).
    :return: Local z-scored values
    """
    neighborhood_intensities = values[indices]  # Shape: (n_samples, n_neighbors)
    if norm_method == "classic":
        local_means = np.nanmean(neighborhood_intensities, axis=1)
        local_stds = np.nanstd(neighborhood_intensities, axis=1)
        local_stds = np.where(local_stds == 0, 1, local_stds)
        local_zscores = (values - local_means) / local_stds
    elif norm_method == "robust":
        local_medians = np.nanmedian(neighborhood_intensities, axis=1)
        local_mads = median_abs_deviation(
            neighborhood_intensities, axis=1, nan_policy="omit"
        )
        local_mads = np.where(local_mads == 0, 1, local_mads)
        local_zscores = (values - local_medians) / local_mads
    else:
        raise ValueError(f"{norm_method} is not a valid normalization method")
    return local_zscores


def _local_z_score_indices(
    df: pd.DataFrame,
    n_neighbors: int = 100,
    reference: Literal["all", "ntcs"] = "all",
    ntc_query: str = 'type == "ntc"',
    centroid_column_names: Sequence[str] = (
        "Nuclei_AreaShape_Center_X",
        "Nuclei_AreaShape_Center_Y",
    ),
) -> np.ndarray:
    assert len(centroid_column_names) == 2, "Only a sequence of length 2 is valid"
    assert (centroid_column_names[0] in df.columns) and (
        centroid_column_names[1] in df.columns
    ), "centroid_column_names not found."
    if reference == "all":
        coordinates = df[[centroid_column_names[0], centroid_column_names[1]]].values
    elif reference == "ntcs":
        coordinates = df.query(ntc_query)[centroid_column_names].values
    else:
        raise ValueError(f"{reference} is not a valid reference")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(coordinates)
    return nbrs.kneighbors(coordinates, return_distance=False)


def local_zscore(
    dataframe: pd.DataFrame,
    features: Sequence[str],
    norm_method: Literal["classic", "robust"] = "classic",
    reference: Literal["all", "ntcs"] = "all",
    n_neighbors: int = 100,
    centroid_column_names: Sequence[str] = (
        "Nuclei_AreaShape_Center_X",
        "Nuclei_AreaShape_Center_Y",
    ),
) -> pd.DataFrame:
    """Calculate local z-scores for specified features in a dataframe based on spatial proximity.

    :param dataframe: The input dataframe containing the data.
    :param features: A list of column names for which local z-scores will be computed.
    :param norm_method: The method of normalization, either 'classic' (mean and std) or 'robust' (median and MAD).
    :param reference: The reference group for neighbor selection, either 'all' or 'ntcs'.
    :param n_neighbors: The number of neighbors to consider for local calculations.
    :param centroid_column_names: Names of the columns depicting X and Y centroid coordinates

    :returns: A dataframe with original data and additional columns for local z-scores.

    :example:

        .. code-block:: python

            from scallops.stats import local_zscore
            import pandas as pd

            # Example DataFrame
            data = {
                "nuclei_centroid-0": [1, 2, 3],
                "centroid_column_names[1]": [1, 2, 3],
                "feature1": [10, 20, 30],
                "type": ["ntc", "ntc", "control"],
            }
            df = pd.DataFrame(data)

            # Calculate local z-scores
            result = local_zscore(
                df,
                features=["feature1"],
                norm_method="classic",
                reference="all",
                n_neighbors=2,
            )
            print(result)
    """
    welldf = dataframe.copy()
    indices = _local_z_score_indices(
        dataframe,
        n_neighbors=n_neighbors,
        reference=reference,
        centroid_column_names=centroid_column_names,
    )
    local_zs = pd.DataFrame(
        {
            f"{feature}_{norm_method[0]}lzs": _local_zscore_transform(
                welldf[feature].values, indices, norm_method
            )
            for feature in features
        }
    )
    return pd.concat(
        [welldf.reset_index(drop=True), local_zs.reset_index(drop=True)], axis=1
    )
