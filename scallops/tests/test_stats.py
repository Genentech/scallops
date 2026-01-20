import numpy as np
import pandas as pd
import pytest

from scallops.stats import (
    auc_variance_exact,
    compute_area_between_cdfs,
    fisher_combined_pvalue,
    fixed_effects_meta_analysis_mannwhitneyu,
    generate_cdfs_with_area_difference,
    group_differences_test,
    random_effects_meta_analysis_mannwhitneyu,
)


@pytest.mark.stats
def test_rank_features():
    df_example = pd.DataFrame(
        {
            "gene_symbol": ["A", "A", "B", "B", "C", "C"],
            "feature": [1.2, 1.5, 2.0, 2.2, 3.5, 3.6],
        }
    )

    result_example = group_differences_test(
        df_example,
        feature="feature",
        group_by="gene_symbol",
        reference_group="A",
        method="mannwhitney",
    )

    # Check if the output has the expected columns
    assert sorted(result_example.columns) == sorted(
        [
            "group",
            "statistic",
            "reference_mean",
            "treatment_mean",
            "treatment_n",
            "reference_n",
            "fold_change",
            "log2FoldChange",
            "CLES",
            "p-value",
            "FDR",
            "-log10FDR",
        ]
    )
    # Check if the output has the expected number of rows
    assert result_example.shape[0] == 2
    # Check statistic is expected
    np.testing.assert_equal(result_example["statistic"], np.array([0.0, 0.0]))
    # Check AUC and ∆ AUC
    np.testing.assert_equal(result_example.CLES, np.array([0.0, 0.0]))
    # Check pvalue, FDR-BH pval, and  -log2FDR
    np.testing.assert_almost_equal(
        result_example["p-value"], np.array([0.3333, 0.3333]), decimal=4
    )
    np.testing.assert_almost_equal(
        result_example["FDR"], np.array([0.333333, 0.333333]), decimal=4
    )
    np.testing.assert_almost_equal(
        result_example["-log10FDR"], np.array([0.477121, 0.477121]), decimal=4
    )


@pytest.mark.stats
def test_fisher_combined_pvalue():
    individual_p_values = [0.01, 0.05, 0.2]
    combined_p_value = fisher_combined_pvalue(individual_p_values)
    # Check if the combined p-value is within a reasonable range
    assert 0 <= combined_p_value <= 1
    # Check value
    np.testing.assert_almost_equal(combined_p_value, 0.00526255)


@pytest.mark.stats
def test_meta_analysis_mannwhitneyu():
    U_statistics = [20, 25, 30]
    sample_sizes = [(50, 50), (60, 60), (70, 70)]
    combined_p_value, combined_effect_size = fixed_effects_meta_analysis_mannwhitneyu(
        U_statistics, sample_sizes
    )
    # Check  combined p-value
    np.testing.assert_almost_equal(combined_p_value, 0.82222508)
    # Check combined effect size
    np.testing.assert_almost_equal(combined_effect_size, 23.36891079)


@pytest.mark.stats
def test_auc_variance():
    m = 30
    n = 40
    k = 5
    var_auc = auc_variance_exact(m, n, k)
    # Check  variance of AUC
    np.testing.assert_almost_equal(var_auc, 3.11339108)


@pytest.mark.parametrize("target_area", [0.1, 0.2, 0.3])
def test_generate_cdfs_with_area_difference(target_area):
    seed = 123456
    x, cdf1, cdf2, final_area, data1, data2 = generate_cdfs_with_area_difference(
        target_area, iterations=1000, seed=seed
    )
    np.testing.assert_allclose(final_area, target_area, rtol=1e-04, atol=1e-03)


@pytest.mark.stats
@pytest.mark.parametrize("target_area", [0.1, 0.2, 0.3])
def test_compute_area_between_cdfs(target_area):
    seed = 123456
    iterations = 10000
    x, cdf1, cdf2, final_area, data1, data2 = generate_cdfs_with_area_difference(
        target_area, iterations=iterations, seed=seed
    )

    data_sets = {"target1": data2 if target_area != 0 else data1}
    areas, cdfs, bins, _ = compute_area_between_cdfs(
        data1, data_sets, reference_label="reference"
    )
    np.testing.assert_allclose(areas["target1"], final_area, rtol=1e-04, atol=1e-03)


@pytest.mark.stats
def test_random_effects_meta_analysis_mannwhitneyu():
    U_statistics_example = [120, 150, 180]
    sample_sizes_example = [(30, 30), (40, 40), (50, 50)]
    individual_p_values = [0.1, 0.05, 0.025]
    expected_effect = 0.0945203394988672
    expected_variance = 0.037499873194929825
    expected_pvalue = 0.006296506270915048

    result = random_effects_meta_analysis_mannwhitneyu(
        U_statistics_example, sample_sizes_example, individual_p_values
    )
    assert result.combined_effect == expected_effect
    assert result.combined_variance == expected_variance
    assert result.pvalue == expected_pvalue
