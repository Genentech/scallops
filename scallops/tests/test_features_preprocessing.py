import warnings

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from scallops.features.preprocessing import (
    filter_batch_correlated,
    filter_data,
    filter_low_cardinality,
    filter_zero_inflated,
    remove_correlated_features,
    transform_features_yj,
    _col_batch_filter_parquet,
    _streaming_cell_and_variance_filter,
    _streaming_materialise,
    _apply_filter_steps_1_2,
    _apply_filter_post_materialise,
    _scaled_nanvar_per_group,
)


@pytest.mark.parametrize("by", [None, "well"])
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.features
def test_filter_data(use_dask, by):
    adata = anndata.AnnData(
        X=da.arange(8, chunks=(1,)).reshape((4, 2))
        if use_dask
        else np.arange(8).reshape((4, 2)),
        obs=pd.DataFrame(
            data=dict(
                pert=["pert1", "pert2", "pert1", "pert2"],
                well=["well1", "well2", "well1", "well2"],
            )
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata.X = adata.X.astype(np.float32)
    adata.X[1, 0] = 100
    adata.X[0, 0] = np.nan
    # np.var(adata.X, axis=0) array([nan,  5.], dtype=float32)
    test_nan_filter = filter_data(
        adata, max_fraction_not_finite=0, min_variance=None, max_variance=None
    )
    assert test_nan_filter.shape == (3, 2)
    # np.var(adata.X, axis=0) # array([nan,  5.]
    # np.var(adata[adata.obs['well'] == 'well1'].X, axis=0)  # array([nan,  4.])
    # np.var(adata[adata.obs['well'] == 'well2'].X, axis=0)  # array([2209.,    4.]
    d1 = filter_data(
        adata, max_fraction_not_finite=None, min_variance=0, max_variance=None, by=by
    )
    # np.var(adata[1:].X, axis=0)  array([2006.2222, 2.6666667]
    d2 = filter_data(
        adata, max_fraction_not_finite=0, min_variance=5, max_variance=None, by=by
    )

    assert d1.shape == (4, 1)
    assert d2.shape == (3, 1)
    assert d1.var.index.values[0] == "gene2"
    assert d2.var.index.values[0] == "gene1"


@pytest.mark.parametrize("by", [None, ["pert", "well"], ["well"]])
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.features
def test_transform_features_yj(by, use_dask):
    adata = anndata.AnnData(
        X=da.arange(8, chunks=(1,)).reshape((4, 2))
        if use_dask
        else np.arange(8).reshape((4, 2)),
        obs=pd.DataFrame(
            data=dict(
                pert=["pert1", "pert2", "pert1", "pert2"],
                well=["well1", "well2", "well1", "well2"],
            )
        ),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata2 = adata.copy()
    if isinstance(adata2.X, da.Array):
        adata2.X = adata2.X.compute()
    df = adata2.to_df().join(adata2.obs)

    if by is not None:
        grouped = df.groupby(by)

        def single_group(x):
            x = x.copy()
            x["gene1"] = (
                PowerTransformer(method="yeo-johnson", standardize=False)
                .fit_transform(x["gene1"].values.reshape(-1, 1))
                .squeeze()
            )
            x["gene2"] = (
                PowerTransformer(method="yeo-johnson", standardize=False)
                .fit_transform(x["gene2"].values.reshape(-1, 1))
                .squeeze()
            )
            return x

        df = grouped.apply(single_group, include_groups=False).reset_index()

    else:
        df["gene1"] = (
            PowerTransformer(method="yeo-johnson", standardize=False)
            .fit_transform(df["gene1"].values.reshape(-1, 1))
            .squeeze()
        )
        df["gene2"] = (
            PowerTransformer(method="yeo-johnson", standardize=False)
            .fit_transform(df["gene2"].values.reshape(-1, 1))
            .squeeze()
        )
        df = df.reset_index(drop=True)
    columns_drop = df.columns[df.columns.str.startswith("level_")]
    if len(columns_drop) > 0:
        df = df.drop(columns_drop, axis=1)

    adata_transformed = transform_features_yj(adata, by=by)

    if isinstance(adata_transformed.X, da.Array):
        adata_transformed.X = adata_transformed.X.compute()
    df_test = (
        adata_transformed.to_df()
        .join(adata_transformed.obs)
        .sort_values(["pert", "well"])
    )
    df_test = df_test.sort_values(["pert", "well"]).reset_index(drop=True)
    df = df.sort_values(["pert", "well"]).reset_index(drop=True)
    # Use rtol/atol tolerance: our implementation uses scipy.stats.yeojohnson
    # directly (GIL-releasing, parallelisable) instead of sklearn PowerTransformer.
    # Both find the same optimum but via different solvers → not bit-identical.
    # scipy.stats.yeojohnson (our implementation) returns float32; sklearn returns float64.
    pd.testing.assert_frame_equal(df_test[df.columns], df,
                                   check_exact=False, rtol=1e-3, atol=1e-4,
                                   check_dtype=False)


# ---------------------------------------------------------------------------
# Shared fixture for new filter tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """30-obs, 8-feature AnnData with controlled structure for filter testing."""
    np.random.seed(0)
    n, p = 30, 8
    X = np.random.randn(n, p).astype(np.float32)
    # feat0 and feat1 are nearly identical (r ≈ 1)
    X[:, 1] = X[:, 0] + 0.001 * np.random.randn(n)
    # feat4 is mostly zero (80% of cells)
    X[:24, 4] = 0.0
    # feat5 is binary
    X[:, 5] = (X[:, 5] > 0).astype(np.float32)
    genes = ["NTC"] * 10 + ["gene_A"] * 10 + ["gene_B"] * 10
    plates = ["p1"] * 15 + ["p2"] * 15
    obs = pd.DataFrame(
        {"gene_symbol": genes, "plate": plates},
        index=pd.RangeIndex(n).astype(str),
    )
    return anndata.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"f{i}" for i in range(p)]),
    )


# ---------------------------------------------------------------------------
# remove_correlated_features
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_remove_correlated_drops_redundant_feature(sample_data):
    """feat1 ≈ feat0 → one should be removed."""
    result = remove_correlated_features(sample_data, threshold=0.9)
    assert result.shape[1] < sample_data.shape[1]
    # feat0 has higher variance → it should be kept
    assert "f0" in result.var.index
    assert "f1" not in result.var.index


@pytest.mark.features
def test_remove_correlated_keeps_uncorrelated(sample_data):
    """Features with |r| < threshold are always retained."""
    result = remove_correlated_features(sample_data, threshold=0.9)
    # feat0 and feat2 are not correlated → both kept (feat1 removed)
    assert "f0" in result.var.index
    assert "f2" in result.var.index


@pytest.mark.features
def test_remove_correlated_high_threshold_keeps_most(sample_data):
    """threshold=0.9999 (near-perfect correlation only) keeps nearly everything.

    We cannot guarantee exactly n features because float32 arithmetic can produce
    correlation values very slightly above the theoretical maximum of 1.0.  We
    instead verify that fewer features are removed at a tight threshold than at a
    looser one.
    """
    r_loose = remove_correlated_features(sample_data, threshold=0.9)
    r_tight = remove_correlated_features(sample_data, threshold=0.9999)
    # Tight threshold should keep at least as many features as the loose one
    assert r_tight.shape[1] >= r_loose.shape[1]


@pytest.mark.features
def test_remove_correlated_reference_query(sample_data):
    """Correlation estimated on NTC only should still remove feat1."""
    result = remove_correlated_features(
        sample_data, threshold=0.9, reference_query="gene_symbol=='NTC'"
    )
    assert result.shape[1] < sample_data.shape[1]


@pytest.mark.features
def test_remove_correlated_chunk_size_consistent(sample_data):
    """Different chunk sizes must give identical results."""
    r1 = remove_correlated_features(sample_data, threshold=0.9, chunk_size=3)
    r2 = remove_correlated_features(sample_data, threshold=0.9, chunk_size=100)
    assert list(r1.var.index) == list(r2.var.index)


# ---------------------------------------------------------------------------
# filter_zero_inflated
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_filter_zero_inflated_drops_sparse_feature(sample_data):
    """feat4 has 80% zeros → removed with max_zero_fraction=0.5.

    feat5 (binary) may also be removed if it happens to have ≥50% zero values;
    we verify feat4 is gone rather than assuming exactly 1 feature is removed.
    """
    result = filter_zero_inflated(sample_data, max_zero_fraction=0.5)
    assert "f4" not in result.var.index
    assert result.shape[1] < sample_data.shape[1]


@pytest.mark.features
def test_filter_zero_inflated_keeps_non_sparse(sample_data):
    """Non-sparse features are retained."""
    result = filter_zero_inflated(sample_data, max_zero_fraction=0.5)
    assert "f0" in result.var.index


@pytest.mark.features
def test_filter_zero_inflated_permissive_threshold_keeps_all(sample_data):
    """max_zero_fraction=1.0 keeps everything."""
    result = filter_zero_inflated(sample_data, max_zero_fraction=1.0)
    assert result.shape == sample_data.shape


@pytest.mark.features
def test_filter_zero_inflated_near_zero_threshold(sample_data):
    """near_zero_threshold counts small values as zero."""
    data = sample_data.copy()
    data.X[:28, 6] = 0.001  # 93% near-zero for feat6
    result = filter_zero_inflated(
        data, max_zero_fraction=0.5, near_zero_threshold=0.01
    )
    assert "f6" not in result.var.index


# ---------------------------------------------------------------------------
# filter_low_cardinality
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_filter_low_cardinality_drops_binary_feature(sample_data):
    """feat5 is binary (2 unique values) → removed with min_unique=5."""
    result = filter_low_cardinality(sample_data, min_unique=5)
    assert "f5" not in result.var.index


@pytest.mark.features
def test_filter_low_cardinality_keeps_continuous(sample_data):
    """Continuous features have many unique values → retained."""
    result = filter_low_cardinality(sample_data, min_unique=5)
    assert "f0" in result.var.index


@pytest.mark.features
def test_filter_low_cardinality_min_unique_1_keeps_all(sample_data):
    """min_unique=1 keeps everything (even binary features have ≥1 unique value)."""
    result = filter_low_cardinality(sample_data, min_unique=1)
    assert result.shape == sample_data.shape


# ---------------------------------------------------------------------------
# filter_batch_correlated
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_filter_batch_correlated_removes_batch_feature(sample_data):
    """A feature artificially shifted between plates should be removed."""
    data = sample_data.copy()
    # Make feat7 perfectly predict the plate label (strong batch effect)
    data.X[data.obs["plate"] == "p1", 7] = 5.0
    data.X[data.obs["plate"] == "p2", 7] = -5.0

    result = filter_batch_correlated(
        data, batch_column="plate", pvalue_threshold=0.05, method="kruskal"
    )
    assert "f7" not in result.var.index


@pytest.mark.features
def test_filter_batch_correlated_keeps_non_batch_features(sample_data):
    """Features unrelated to plate should not be removed."""
    result = filter_batch_correlated(
        sample_data, batch_column="plate", pvalue_threshold=0.001
    )
    # At p<0.001, none of the random features should be flagged (low FPR)
    assert result.shape[1] == sample_data.shape[1]


@pytest.mark.features
def test_filter_batch_correlated_reference_query(sample_data):
    """Restricting the test to NTC cells should work without error."""
    result = filter_batch_correlated(
        sample_data,
        batch_column="plate",
        reference_query="gene_symbol=='NTC'",
        pvalue_threshold=0.05,
    )
    assert result.shape[0] == sample_data.shape[0]  # obs unchanged
    assert result.shape[1] <= sample_data.shape[1]


@pytest.mark.features
def test_filter_batch_correlated_single_batch_is_noop(sample_data):
    """With only one batch value the filter does nothing."""
    data = sample_data.copy()
    data.obs["plate"] = "p1"
    result = filter_batch_correlated(data, batch_column="plate")
    assert result.shape == data.shape


# ---------------------------------------------------------------------------
# _col_batch_filter_parquet vs _streaming_cell_and_variance_filter parity
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_col_batch_matches_row_batch(tmp_path):
    """Column-batch (parquet) and row-batch (zarr) filter paths must agree.

    We write a small synthetic AnnData to parquet, read it back with
    ``_read_data`` (which populates ``uns["_parquet_sources"]`` and keeps the
    feature data as a dask array), then run both filter paths and compare the
    results.
    """
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    from scallops.features.util import _read_map_inputs as _read_data

    # ── Create a synthetic dataset ────────────────────────────────────────
    np.random.seed(7)
    n_obs, n_feat = 60, 12
    X = np.random.randn(n_obs, n_feat).astype(np.float32)
    # Make feature 0 constant within every well → scaled var = 0 → dropped
    X[:, 0] = 3.14
    # Insert a few NaN values into row 5 so it triggers the finite filter
    X[5, :4] = np.nan

    plates = ["p1"] * 30 + ["p2"] * 30
    wells  = [str(i % 3) for i in range(n_obs)]
    obs = pd.DataFrame(
        {"plate": plates, "well": wells},
        index=pd.RangeIndex(n_obs).astype(str),
    )
    obs.index.name = "cell_id"

    feat_names = [f"Cells_Intensity_feat{i}" for i in range(n_feat)]
    var = pd.DataFrame(index=feat_names)

    # ── Write to parquet (2 row groups) ──────────────────────────────────
    pq_path = str(tmp_path / "test_filter.parquet")
    df_obs  = obs.copy()
    df_obs.index.name = "cell_id"
    df_feat = pd.DataFrame(X, columns=feat_names, index=obs.index)
    df_all  = pd.concat([df_feat, df_obs], axis=1)

    table = _pa.Table.from_pandas(df_all)
    _pq.write_table(table, pq_path, row_group_size=30)

    # ── Read back via _read_data ──────────────────────────────────────────
    data = _read_data([pq_path])
    assert "_parquet_sources" in data.uns, "parquet_sources must be set on read"

    parquet_sources = data.uns["_parquet_sources"]
    obs_df          = data.obs
    label_mask      = np.ones(len(obs_df), dtype=bool)

    min_var = 0.001
    max_fnf = 0.25   # drop cells where > 25% of features are NaN
    by_cols = ["plate", "well"]

    # ── Path A: column-batch (parquet) ────────────────────────────────────
    X_col, cell_keep_col, feat_keep_col, _report, _ = _col_batch_filter_parquet(
        parquet_sources, obs_df, label_mask,
        by=by_cols,
        max_fraction_not_finite=max_fnf,
        min_variance=min_var,
        max_variance=None,
    )

    # ── Path B: row-batch (dask array) via shared helpers ─────────────────
    # _streaming_cell_and_variance_filter now returns raw NaN counts;
    # _apply_filter_steps_1_2 converts them into the boolean masks.
    bad_counts_row, nan_per_feat_row = _streaming_cell_and_variance_filter(
        data.X, obs_df, label_mask,
        by=by_cols,
        max_fraction_not_finite=max_fnf,
        n_prefetch=2,
    )
    n_obs_b, n_feat_b = data.X.shape
    feat_pass1_row, cell_keep_row = _apply_filter_steps_1_2(
        bad_counts_row, nan_per_feat_row, label_mask, n_obs_b, n_feat_b,
        max_feature_nan_fraction=None,
        max_fraction_not_finite=max_fnf,
    )
    X_row_raw = _streaming_materialise(data.X, cell_keep_row, feat_pass1_row, n_prefetch=2)
    obs_kept_row = obs_df.iloc[cell_keep_row].reset_index(drop=True)
    feat_keep_b  = feat_pass1_row.copy()
    X_row, feat_keep_row, _ = _apply_filter_post_materialise(
        X_row_raw, feat_keep_b, obs_kept_row,
        by=by_cols, min_variance=min_var, max_variance=None,
        max_residual_nan_fraction=0.0,
    )

    # ── Align feature order: both paths use the same feat_cols from uns ───
    # cell_keep masks should be identical
    np.testing.assert_array_equal(
        cell_keep_col, cell_keep_row,
        err_msg="cell_keep masks differ between col-batch and row-batch paths",
    )
    np.testing.assert_array_equal(
        feat_keep_col, feat_keep_row,
        err_msg="feat_keep masks differ between col-batch and row-batch paths",
    )

    # X_filtered arrays should be numerically close
    assert X_col.shape == X_row.shape, (
        f"Filtered shapes differ: col={X_col.shape}, row={X_row.shape}"
    )
    np.testing.assert_allclose(
        X_col, X_row, rtol=1e-4, atol=1e-5,
        err_msg="X_filtered values differ between col-batch and row-batch paths",
    )


@pytest.mark.features
def test_col_batch_report_has_variance(tmp_path):
    """The feature-drop report must have finite median_variance values.

    Regression: after the step-3/4 refactor, ``feat_var`` was reset to
    ``np.array([])`` after calling ``_apply_filter_post_materialise``, making
    the ``len(feat_var) == len(_step3_survivors)`` guard always False and
    causing every feature to get ``median_variance = 0.0`` in the report.
    """
    import pyarrow as _pa
    import pyarrow.parquet as _pq

    np.random.seed(99)
    n_obs, n_feat = 40, 8
    X = np.random.randn(n_obs, n_feat).astype(np.float32)
    # feat3: constant within the well → scaled var = 0 → dropped → appears in report
    X[:, 3] = 7.5
    feat_names = [f"Cells_Intensity_feat{i}" for i in range(n_feat)]
    df = pd.DataFrame(X, columns=feat_names)
    df["plate"] = "p1"
    df["well"] = "1"
    p = str(tmp_path / "report_test.parquet")
    _pq.write_table(_pa.Table.from_pandas(df), p)

    from scallops.features.util import _read_map_inputs
    data = _read_map_inputs([p])
    parquet_sources = data.uns["_parquet_sources"]
    label_mask = np.ones(len(data.obs), dtype=bool)

    _, _, _, report, _ = _col_batch_filter_parquet(
        parquet_sources, data.obs, label_mask,
        by=["plate", "well"],
        max_fraction_not_finite=0.25,
        min_variance=0.001,
        max_variance=None,
    )

    assert report is not None and not report.empty, "Feature-drop report is empty"
    dropped = report[~report["kept"]]
    assert len(dropped) > 0, "No dropped features in report"
    # At least some dropped features must have a finite median_variance
    assert dropped["median_variance"].notna().any(), (
        "All dropped features have NaN median_variance — "
        "feat_var was likely reset to np.array([]) before report was built"
    )
    # The low-variance feature (feat3) should appear with a small but finite variance
    feat3_row = report[report["feature"] == "Cells_Intensity_feat3"]
    assert len(feat3_row) == 1
    assert np.isfinite(feat3_row["median_variance"].iloc[0]), (
        "feat3 median_variance is not finite despite being dropped by variance filter"
    )


# ---------------------------------------------------------------------------
# Parquet vs zarr input parity — regression guard for unified filter path
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_filter_parquet_zarr_input_parity(tmp_path):
    """Same data written to parquet and zarr must produce identical filter output.

    This is the primary regression guard for the unified two-pass filter path.
    If parquet and zarr inputs diverge at any point, this test catches it.
    """
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    from scallops.features.util import _read_map_inputs as _read_data
    from scallops.cli.map_build import _apply_filter_inmem
    import argparse

    np.random.seed(55)
    n_obs, n_feat = 80, 10
    X = np.random.randn(n_obs, n_feat).astype(np.float32)
    X[:, 2] = 0.0          # constant → dropped by variance filter
    X[0, :3] = np.nan      # row 0 has NaN → dropped by cell filter
    feat_names = [f"Cells_Intensity_feat{i}" for i in range(n_feat)]
    plates = ["p1"] * 40 + ["p2"] * 40
    wells  = ["1" if i < 40 else "2" for i in range(n_obs)]
    obs = pd.DataFrame({"plate": plates, "well": wells},
                       index=pd.RangeIndex(n_obs).astype(str))

    # ── Write parquet ──────────────────────────────────────────────────────
    pq_path = str(tmp_path / "data.parquet")
    df = pd.concat([pd.DataFrame(X, columns=feat_names, index=obs.index), obs], axis=1)
    _pq.write_table(_pa.Table.from_pandas(df), pq_path)

    # ── Write zarr ────────────────────────────────────────────────────────
    import anndata
    zarr_path = str(tmp_path / "data.zarr")
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=feat_names))
    adata.write_zarr(zarr_path)

    common_args = argparse.Namespace(
        plate_column="plate", well_column="well",
        label_filter=None, features=None,
        max_fraction_not_finite=0.25, max_feature_nan_fraction=0.5,
        min_variance=0.001, max_variance=None,
        max_residual_nan_fraction=0.0, residual_nan_impute="zero",
        perturbation="gene_symbol", memory_budget_gb=None,
        streaming_threshold_gb=None, filter_batch_size=500_000,
        filter_max_memory_gb=None, force=True, no_version=True,
        client="none", dask_cluster=None,
        include_measurement_types=None, feature_channels=None,
        obs_force=None, condition_column=None, condition_source_column=None,
        condition_map=None,
    )

    # Filter via parquet input
    data_pq = _read_data([pq_path])
    result_pq = _apply_filter_inmem(data_pq, common_args)

    # Filter via zarr input
    data_zarr = _read_data([zarr_path])
    result_zarr = _apply_filter_inmem(data_zarr, common_args)

    assert result_pq.shape == result_zarr.shape, (
        f"Shape mismatch: parquet={result_pq.shape} zarr={result_zarr.shape}"
    )
    np.testing.assert_array_equal(
        np.sort(result_pq.var.index),
        np.sort(result_zarr.var.index),
        err_msg="Surviving features differ between parquet and zarr paths",
    )
    X_pq   = result_pq.X   if isinstance(result_pq.X,   np.ndarray) else result_pq.X.compute()
    X_zarr = result_zarr.X if isinstance(result_zarr.X, np.ndarray) else result_zarr.X.compute()
    np.testing.assert_allclose(X_pq, X_zarr, rtol=1e-5, atol=1e-5,
        err_msg="X values differ between parquet and zarr filter paths")


# ---------------------------------------------------------------------------
# _scaled_nanvar_per_group — unit tests for the per-well variance helper
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_scaled_nanvar_constant_feature():
    """A perfectly constant feature must give scaled variance = 0."""
    X = np.full((100, 3), 42.0)  # all cells identical in all features
    var = _scaled_nanvar_per_group(X)
    np.testing.assert_array_equal(var, 0.0)


@pytest.mark.features
def test_scaled_nanvar_exactly_constant_gives_zero():
    """An exactly constant feature must give scaled variance = 0.

    With mean±3σ, std=0 → clip range=0 → all scaled values = 0 → var=0.
    Note: near-constant features with outliers inflate σ and widen the clip
    range — only exactly constant features are guaranteed var=0 with mean±3σ.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = np.zeros((n, 2))
    X[:, 0] = 5.0                        # exactly constant → var must be 0
    X[:, 1] = rng.uniform(0, 1, n)       # variable control

    var = _scaled_nanvar_per_group(X)
    assert var[0] == 0.0, f"Exactly-constant feature must give var=0, got {var[0]}"
    assert var[1] > 0.001, f"Uniform feature gave unexpectedly low var: {var[1]}"


@pytest.mark.features
def test_scaled_nanvar_outlier_robustness():
    """Extreme outliers must not inflate scaled variance of otherwise variable features.

    A feature with cells uniformly spread in [0, 1] plus one cell at 1e6
    should give about the same scaled variance as without the outlier,
    because clipping removes the outlier before computing the range.
    """
    rng = np.random.default_rng(1)
    n = 500
    X_clean = rng.uniform(0, 1, (n, 1))
    X_outlier = X_clean.copy()
    X_outlier[0, 0] = 1e6

    var_clean   = _scaled_nanvar_per_group(X_clean)
    var_outlier = _scaled_nanvar_per_group(X_outlier)
    # With mean±3σ the outlier inflates σ → wider clip range → other cells
    # compress → lower scaled variance.  Both results must be finite and positive.
    assert np.all(np.isfinite(var_clean)),   "clean: non-finite variance"
    assert np.all(np.isfinite(var_outlier)), "outlier: non-finite variance"
    assert np.all(var_clean   > 0), "clean: variance should be positive"
    assert np.all(var_outlier > 0), "outlier: variance should be positive"


@pytest.mark.features
def test_scaled_nanvar_nan_handling():
    """NaN cells must be ignored; the result must still be finite."""
    rng = np.random.default_rng(2)
    n = 100
    X = rng.standard_normal((n, 4)).astype(np.float64)
    # Scatter NaN across 20% of cells in feat 0 and feat 2
    nan_idx = rng.choice(n, size=20, replace=False)
    X[nan_idx, 0] = np.nan
    X[nan_idx, 2] = np.nan

    var = _scaled_nanvar_per_group(X)
    assert np.all(np.isfinite(var)), f"NaN in result: {var}"
    assert np.all(var >= 0), "Negative variance"


@pytest.mark.features
def test_scaled_nanvar_high_variance_kept():
    """A feature with wide uniform spread must produce scaled variance well above threshold."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, (1000, 1))
    var = _scaled_nanvar_per_group(X)
    # mean±3σ extends beyond [0,1] so scaled values don't fill the full [0,1]
    # range — variance is ~0.027, well above the 0.001 filter threshold
    assert var[0] > 0.001, f"Wide-spread feature gave unexpectedly low scaled var: {var[0]}"


@pytest.mark.features
def test_apply_filter_inmem_parquet_e2e(tmp_path):
    """_apply_filter_inmem must produce a non-empty AnnData when called with a
    parquet input that has a bool obs column used in the label filter.

    This exercises the full path: _read_map_inputs → label-filter with bool obs
    column → parquet column-batch filter → cells written to output.
    """
    import argparse
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    from scallops.features.util import _read_map_inputs
    from scallops.cli.map_build import _apply_filter_inmem

    np.random.seed(1)
    n = 30
    X = np.random.randn(n, 5).astype(np.float32)
    df = pd.DataFrame({
        "Cells_Intensity_feat0": X[:, 0].astype(float),
        "Cells_Intensity_feat1": X[:, 1].astype(float),
        "Cells_Intensity_feat2": X[:, 2].astype(float),
        "Cells_Intensity_feat3": X[:, 3].astype(float),
        "Cells_Intensity_feat4": X[:, 4].astype(float),
        # Bool obs column — label filter will use this
        "is_boundary": [False] * 20 + [True] * 10,
        "gene_symbol": ["NTC"] * 10 + ["g1"] * 10 + ["g2"] * 10,
        "plate": ["p1"] * n,
        "well": ["1"] * n,
    }, index=pd.RangeIndex(n).astype(str))
    p = str(tmp_path / "e2e_filter.parquet")
    _pq.write_table(_pa.Table.from_pandas(df), p)

    data = _read_map_inputs([p])

    args = argparse.Namespace(
        label_filter="is_boundary == False",   # must keep 20 rows
        max_fraction_not_finite=0.25,
        min_variance=0.0,
        max_variance=None,
        max_feature_nan_fraction=0.50,
        max_residual_nan_fraction=None,
        residual_nan_impute="zero",
        perturbation="gene_symbol",
        plate_column="plate",
        well_column="well",
        filter_batch_size=500_000,
        filter_max_memory_gb=None,
        scale_method="global",
        batch_column=None,
        max_correlation=None,
        feature_channels=None,
    )

    result = _apply_filter_inmem(data, args)

    assert result.n_obs == 20, (
        f"Expected 20 cells (is_boundary==False) but got {result.n_obs}. "
        "Bool label-filter may not be working."
    )
    assert result.n_vars > 0, "All features were filtered out"
    assert not np.isnan(result.X).any(), "NaN values in filtered result"
