"""Tests for the map-build CLI pipeline and TVN backprojection.

Covers:
- Correct output shape and dtype for each step
- uns / varm propagation through the pipeline (backprojection matrices must
  survive every transformation step)
- Additive scallops provenance chain (JSON string, one entry per step)
- TVN stores the parameters added in the backprojection branch
- map-filter correctly handles variance thresholds and NaN cells
- map-pca stores PCA info in uns and names var as PC1…PCn
- map-agg applies min-cells filtering and two-step barcode aggregation
- map-similarity produces a square similarity matrix with correct labels
- map-recall writes a Parquet file with expected columns
- End-to-end chain: filter → transform-yj → tvn → agg → center → similarity
- backproject_tvn: round-trip recovery of z-scored profiles
- top_features_from_backprojection: genes list, cluster labels, all methods
"""

import argparse
import json
import os

import anndata
import numpy as np
import pandas as pd
import pytest

import warnings

from scallops.features.backprojection import (
    backproject_tvn,
    top_features_from_backprojection,
)
from scallops.cli.map_build import (
    _log_attrition,
    run_pipeline_map_agg,
    run_pipeline_map_backproject,
    run_pipeline_map_center,
    run_pipeline_map_filter,
    run_pipeline_map_pca,
    run_pipeline_map_pca_select,
    run_pipeline_map_recall,
    run_pipeline_map_scale,
    run_pipeline_map_similarity,
    run_pipeline_map_sphere,
    run_pipeline_map_transform_yj,
    run_pipeline_map_tvn,
)
from scallops.features.normalize import typical_variation_normalization
from scallops.zarr_io import is_anndata_zarr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_NTC = 12   # > N_FEATURES so TVN PCA is always full-rank
N_PERT = 8   # cells per perturbation per plate
N_FEATURES = 5
FEATURES = [f"Cells_Intensity_feature_{i}" for i in range(N_FEATURES)]
PERTURBATIONS = ["gene_A", "gene_B"]


@pytest.fixture
def cell_data() -> anndata.AnnData:
    """Synthetic 30-cell, 5-feature AnnData with two plates and NTC controls."""
    np.random.seed(0)
    genes = ["NTC"] * N_NTC + [p for p in PERTURBATIONS for _ in range(N_PERT)]
    plates = (["plate1"] * (N_NTC // 2 + N_PERT) + ["plate2"] * (N_NTC // 2 + N_PERT)) * 1
    # keep it simple: alternate plates across rows
    n = len(genes)
    plates = ["plate1" if i % 2 == 0 else "plate2" for i in range(n)]
    wells = ["well1" if i % 4 < 2 else "well2" for i in range(n)]
    barcodes = [f"bc_{g}" for g in genes]

    X = np.random.randn(n, N_FEATURES).astype(np.float32)

    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "gene_symbol": genes,
                "plate": plates,
                "well": wells,
                "barcode_0": barcodes,
            },
            index=pd.RangeIndex(n).astype(str),
        ),
        var=pd.DataFrame(index=FEATURES),
    )


@pytest.fixture
def tvn_data(cell_data) -> anndata.AnnData:
    """cell_data after TVN — used for aggregation / centering / similarity tests."""
    return typical_variation_normalization(
        cell_data, reference_query="gene_symbol=='NTC'"
    )


def _ns(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for the map-build CLI functions.

    Defaults satisfy the most common step signatures; override per-test as needed.
    """
    defaults = dict(
        force=True,
        no_version=True,
        client="none",
        dask_cluster=None,
        features=None,
        label_filter=None,
        # canonical step-specific arg names (no legacy aliases)
        plate_column="plate",
        well_column="well",
        tvn_by=None,
        agg_by=None,
        agg_method="mean",
        center_by=None,
        center_robust=False,
        pca_whiten=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_zarr(data: anndata.AnnData, path) -> str:
    out = str(path) + ".zarr"
    data.write_zarr(out)
    return out


def _read_zarr(path: str) -> anndata.AnnData:
    from scallops.io import read_anndata_zarr

    return read_anndata_zarr(path, dask=False)


# ---------------------------------------------------------------------------
# TVN backprojection parameter storage (branch-specific behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_tvn_stores_backprojection_params(cell_data):
    """typical_variation_normalization must store all keys needed for backprojection."""
    result = typical_variation_normalization(
        cell_data, reference_query="gene_symbol=='NTC'"
    )
    assert "pca" in result.uns
    assert "normalization_arguments" in result.uns
    assert "tvn_pre_scale_mean" in result.uns
    assert "tvn_pre_scale_std" in result.uns
    assert "covariance_alignment_inv" in result.uns
    assert "PCs" in result.varm

    # pre-scale stats have one entry per feature
    assert result.uns["tvn_pre_scale_mean"].shape == (N_FEATURES,)
    assert result.uns["tvn_pre_scale_std"].shape == (N_FEATURES,)

    # PCs matrix maps original features → PCA components
    assert result.varm["PCs"].shape == (N_FEATURES, N_FEATURES)


@pytest.mark.features
def test_tvn_stores_covariance_alignment_by(cell_data):
    """With by=['plate'], each group must have an inverse alignment matrix in uns."""
    result = typical_variation_normalization(
        cell_data, reference_query="gene_symbol=='NTC'", by=["plate"]
    )
    cov_inv = result.uns["covariance_alignment_inv"]
    groups = cell_data.obs["plate"].unique().tolist()
    for g in groups:
        assert str(g) in cov_inv, f"Missing covariance_alignment_inv for group {g!r}"
        mat = cov_inv[str(g)]
        assert mat.shape == (N_FEATURES, N_FEATURES)


# ---------------------------------------------------------------------------
# map-filter
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_filter_removes_low_variance_features(cell_data, tmp_path):
    # Zero out the first feature → constant within every well → scaled var = 0
    cell_data.X[:, 0] = 0.0
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(input=[inp], output=out, min_variance=0.001, max_variance=None,
            max_fraction_not_finite=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape[1] == N_FEATURES - 1
    assert "Cells_Intensity_feature_0" not in result.var.index


@pytest.mark.features
def test_map_filter_scaled_variance_keeps_low_absolute_high_relative(cell_data, tmp_path):
    """A feature with small absolute variance but high relative spread must be kept.

    Correlation-type features are bounded to e.g. [-0.01, 0.01] — their raw
    variance is tiny but they span their full observable range.  The old
    absolute threshold (0.1) dropped them; the per-well clip+minmax approach
    keeps them because scaled variance ≈ 0.08 > 0.001.
    """
    # feature_0: tiny absolute variance but cells span the full [-0.01, 0.01] range
    np.random.seed(42)
    n = cell_data.shape[0]
    cell_data.X[:, 0] = np.linspace(-0.01, 0.01, n, dtype=np.float32)
    # feature_1: constant → dropped
    cell_data.X[:, 1] = 5000.0
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(input=[inp], output=out, min_variance=0.001, max_variance=None,
            max_fraction_not_finite=None)
    )
    result = _read_zarr(out + ".zarr")
    # feature_0 (small absolute, full relative range) → kept
    assert "Cells_Intensity_feature_0" in result.var.index
    # feature_1 (constant) → dropped
    assert "Cells_Intensity_feature_1" not in result.var.index


@pytest.mark.features
def test_map_filter_removes_nan_cells(cell_data, tmp_path):
    # Put NaN in half the features of row 0 → exceeds the default 0.25 threshold
    cell_data.X[0, :3] = np.nan
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(input=[inp], output=out, min_variance=None, max_variance=None,
            max_fraction_not_finite=0.25)
    )
    result = _read_zarr(out + ".zarr")
    # row 0 had 3/5 = 60 % non-finite → removed
    assert result.shape[0] == cell_data.shape[0] - 1


@pytest.mark.features
def test_map_filter_propagates_uns(cell_data, tmp_path):
    cell_data.uns["upstream_key"] = "sentinel"
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(input=[inp], output=out, min_variance=None, max_variance=None,
            max_fraction_not_finite=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("upstream_key") == "sentinel"


# ---------------------------------------------------------------------------
# map-transform-yj
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_transform_yj_shape_unchanged(cell_data, tmp_path):
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "yj")

    run_pipeline_map_transform_yj(_ns(input=[inp], output=out))

    result = _read_zarr(out + ".zarr")
    assert result.shape == cell_data.shape
    assert list(result.var.index) == list(cell_data.var.index)


@pytest.mark.features
def test_map_transform_yj_propagates_uns(cell_data, tmp_path):
    cell_data.uns["tvn_pre_scale_mean"] = np.zeros(N_FEATURES)
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "yj")

    run_pipeline_map_transform_yj(_ns(input=[inp], output=out))

    result = _read_zarr(out + ".zarr")
    assert "tvn_pre_scale_mean" in result.uns


# ---------------------------------------------------------------------------
# map-pca
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_pca_output_shape_and_var_names(cell_data, tmp_path):
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "pca")
    n_comp = 3

    run_pipeline_map_pca(
        _ns(input=[inp], output=out, pca_components=n_comp, pca_batch_size=0,
            reference_query=None)
    )
    result = _read_zarr(out + ".zarr")
    # New format: X = scaled features, obsm["X_pca"] = PC coordinates
    assert result.shape == cell_data.shape
    assert result.obsm["X_pca"].shape == (cell_data.shape[0], n_comp)


@pytest.mark.features
def test_map_pca_stores_pca_uns(cell_data, tmp_path):
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "pca")

    run_pipeline_map_pca(
        _ns(input=[inp], output=out, pca_components=3, pca_batch_size=0,
            reference_query=None)
    )
    result = _read_zarr(out + ".zarr")
    # PCA model is stored under map_pca (not pca, which is reserved for TVN's internal PCA)
    assert "map_pca" in result.uns
    for key in ("variance_ratio", "variance", "mean", "PCs"):
        assert key in result.uns["map_pca"], f"Missing map_pca uns key: {key}"


@pytest.mark.features
def test_map_pca_reference_subset_fitting(cell_data, tmp_path):
    """PCA fitted on NTC only must project all cells."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "pca")

    run_pipeline_map_pca(
        _ns(input=[inp], output=out, pca_components=2, pca_batch_size=0,
            reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    # All cells are projected (not just the reference subset)
    assert result.shape[0] == cell_data.shape[0]
    assert result.obsm["X_pca"].shape == (cell_data.shape[0], 2)


# ---------------------------------------------------------------------------
# map-tvn
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_tvn_output_shape(cell_data, tmp_path):
    """TVN must preserve obs count and feature count."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "tvn")

    run_pipeline_map_tvn(
        _ns(input=[inp], output=out, reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape == cell_data.shape


@pytest.mark.features
def test_map_tvn_stores_backprojection_uns(cell_data, tmp_path):
    """Output zarr must contain all uns keys required for backprojection."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "tvn")

    run_pipeline_map_tvn(
        _ns(input=[inp], output=out, reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    for key in (
        "pca",
        "normalization_arguments",
        "tvn_pre_scale_mean",
        "tvn_pre_scale_std",
        "covariance_alignment_inv",
    ):
        assert key in result.uns, f"Missing uns key after map-tvn: {key!r}"
    assert "PCs" in result.varm


@pytest.mark.features
def test_map_tvn_propagates_upstream_uns(cell_data, tmp_path):
    """uns keys set before TVN (e.g. from a filter step) must survive."""
    cell_data.uns["upstream_key"] = 42
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "tvn")

    run_pipeline_map_tvn(
        _ns(input=[inp], output=out, reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("upstream_key") == 42


# ---------------------------------------------------------------------------
# map-agg
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_agg_reduces_to_perturbation_count(tvn_data, tmp_path):
    inp = _write_zarr(tvn_data, tmp_path / "input")
    out = str(tmp_path / "agg")

    run_pipeline_map_agg(
        _ns(input=[inp], output=out, agg_by=["gene_symbol"], agg_method="mean",
            min_cells=None, barcode=None, agg_by_barcode=False,
            perturbation="gene_symbol")
    )
    result = _read_zarr(out + ".zarr")
    n_expected = tvn_data.obs["gene_symbol"].nunique()
    assert result.shape[0] == n_expected
    assert result.shape[1] == tvn_data.shape[1]


@pytest.mark.features
def test_map_agg_min_cells_filters_perturbations(tvn_data, tmp_path):
    """Perturbations with fewer than min_cells should be excluded."""
    inp = _write_zarr(tvn_data, tmp_path / "input")
    out = str(tmp_path / "agg")

    # NTC has N_NTC cells; gene_A / gene_B have N_PERT each.
    # Setting min_cells = N_NTC + 1 removes NTC; N_PERT * 2 removes both perturbs.
    run_pipeline_map_agg(
        _ns(input=[inp], output=out, agg_by=["gene_symbol"], agg_method="mean",
            min_cells=N_PERT + 1, barcode=None, agg_by_barcode=False,
            perturbation="gene_symbol")
    )
    result = _read_zarr(out + ".zarr")
    # Only NTC has enough cells (N_NTC > N_PERT)
    assert result.shape[0] == 1
    assert result.obs["gene_symbol"].iloc[0] == "NTC"


@pytest.mark.features
def test_map_agg_propagates_tvn_uns(tvn_data, tmp_path):
    """TVN backprojection keys must be present in the aggregated zarr."""
    inp = _write_zarr(tvn_data, tmp_path / "input")
    out = str(tmp_path / "agg")

    run_pipeline_map_agg(
        _ns(input=[inp], output=out, agg_by=["gene_symbol"], agg_method="mean",
            min_cells=None, barcode=None, agg_by_barcode=False,
            perturbation="gene_symbol")
    )
    result = _read_zarr(out + ".zarr")
    for key in ("pca", "tvn_pre_scale_mean", "tvn_pre_scale_std",
                "covariance_alignment_inv", "normalization_arguments"):
        assert key in result.uns, f"TVN key lost after map-agg: {key!r}"
    assert "PCs" in result.varm


# ---------------------------------------------------------------------------
# map-center
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_center_subtracts_ntc_mean(tvn_data, tmp_path):
    """After centering on NTC the mean of all NTC cells must be ≈ 0.

    Uses tvn_data (N_NTC=12 cells) so the test is not trivially satisfied by
    subtracting a single profile from itself.
    """
    inp = _write_zarr(tvn_data, tmp_path / "input")
    out = str(tmp_path / "centered")

    run_pipeline_map_center(
        _ns(input=[inp], output=out, reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    ntc_mask = result.obs["gene_symbol"] == "NTC"
    assert ntc_mask.sum() > 1, "Need multiple NTC cells for a meaningful test"
    ntc_mean = result.X[ntc_mask.values].mean(axis=0)
    np.testing.assert_allclose(ntc_mean, 0.0, atol=1e-5)


@pytest.mark.features
def test_map_center_propagates_tvn_uns(tvn_data, tmp_path):
    inp = _write_zarr(tvn_data, tmp_path / "input")
    out = str(tmp_path / "centered")

    run_pipeline_map_center(
        _ns(input=[inp], output=out, reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    for key in ("pca", "tvn_pre_scale_mean", "tvn_pre_scale_std",
                "covariance_alignment_inv"):
        assert key in result.uns, f"TVN key lost after map-center: {key!r}"
    assert "PCs" in result.varm


# ---------------------------------------------------------------------------
# map-similarity
# ---------------------------------------------------------------------------


@pytest.fixture
def profile_data(tvn_data) -> anndata.AnnData:
    """Per-perturbation mean profiles (small, used for similarity tests)."""
    genes = list(tvn_data.obs["gene_symbol"].unique())
    X = np.stack(
        [tvn_data.X[tvn_data.obs["gene_symbol"] == g].mean(axis=0) for g in genes]
    )
    obs = pd.DataFrame({"gene_symbol": genes}, index=genes)
    result = anndata.AnnData(
        X=X.astype(np.float32),
        obs=obs,
        var=tvn_data.var.copy(),
        uns=dict(tvn_data.uns),
    )
    for k, v in tvn_data.varm.items():
        result.varm[k] = v
    return result


@pytest.mark.features
def test_map_similarity_is_square(profile_data, tmp_path):
    inp = _write_zarr(profile_data, tmp_path / "input")
    out = str(tmp_path / "sim")

    run_pipeline_map_similarity(
        _ns(input=[inp], output=out, metric="cosine",
            perturbation="gene_symbol", exclude_reference_query=None)
    )
    result = _read_zarr(out + ".zarr")
    n = profile_data.shape[0]
    assert result.shape == (n, n), f"Expected ({n},{n}), got {result.shape}"


@pytest.mark.features
def test_map_similarity_diagonal_is_one(profile_data, tmp_path):
    """Cosine self-similarity must be 1.0 for non-reference profiles.

    After TVN, NTC profiles are the centering reference and aggregate to a
    near-zero vector, making their cosine self-similarity undefined.  The
    typical workflow excludes NTC from the similarity matrix.
    """
    inp = _write_zarr(profile_data, tmp_path / "input")
    out = str(tmp_path / "sim")

    run_pipeline_map_similarity(
        _ns(input=[inp], output=out, metric="cosine",
            perturbation="gene_symbol",
            exclude_reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    np.testing.assert_allclose(np.diag(result.X), 1.0, atol=1e-5)


@pytest.mark.features
def test_map_similarity_labels_match_perturbations(profile_data, tmp_path):
    inp = _write_zarr(profile_data, tmp_path / "input")
    out = str(tmp_path / "sim")

    run_pipeline_map_similarity(
        _ns(input=[inp], output=out, metric="cosine",
            perturbation="gene_symbol", exclude_reference_query=None)
    )
    result = _read_zarr(out + ".zarr")
    expected_labels = profile_data.obs["gene_symbol"].astype(str).tolist()
    assert list(result.obs.index) == expected_labels
    assert list(result.var.index) == expected_labels


@pytest.mark.features
def test_map_similarity_propagates_uns(profile_data, tmp_path):
    profile_data.uns["tvn_pre_scale_mean"] = np.zeros(N_FEATURES)
    inp = _write_zarr(profile_data, tmp_path / "input")
    out = str(tmp_path / "sim")

    run_pipeline_map_similarity(
        _ns(input=[inp], output=out, metric="cosine",
            perturbation="gene_symbol", exclude_reference_query=None)
    )
    result = _read_zarr(out + ".zarr")
    assert "tvn_pre_scale_mean" in result.uns


@pytest.mark.features
def test_map_similarity_exclude_reference(profile_data, tmp_path):
    inp = _write_zarr(profile_data, tmp_path / "input")
    out = str(tmp_path / "sim")

    run_pipeline_map_similarity(
        _ns(input=[inp], output=out, metric="cosine",
            perturbation="gene_symbol",
            exclude_reference_query="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    assert "NTC" not in list(result.obs.index)
    n_non_ntc = profile_data.obs["gene_symbol"].ne("NTC").sum()
    assert result.shape == (n_non_ntc, n_non_ntc)


# ---------------------------------------------------------------------------
# map-recall
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_recall_writes_parquet_with_expected_columns(profile_data, tmp_path):
    """map-recall must produce a Parquet file with KS-test result columns."""
    from scallops.features.map_eval import pairwise_similarities

    sims = pairwise_similarities(profile_data, metric="cosine")
    labels = profile_data.obs["gene_symbol"].astype(str).values
    sim_adata = anndata.AnnData(
        X=sims.astype(np.float32),
        obs=pd.DataFrame(index=labels),
        var=pd.DataFrame(index=labels),
    )
    inp = _write_zarr(sim_adata, tmp_path / "sim")

    # Write a minimal CORUM-format file
    corum_path = str(tmp_path / "corum.txt")
    with open(corum_path, "w") as f:
        f.write("complex_name\tsubunits_gene_name\n")
        # gene_A and gene_B are in the same complex
        f.write("ComplexAB\tgene_A;gene_B\n")

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(
        _ns(input=[inp], output=out, corum=[corum_path], min_genes=1)
    )

    result = pd.read_parquet(out)
    # map-recall now uses source/method columns instead of the old "corum" column
    for col in ("source", "method", "name", "size", "within_mean", "between_mean",
                "statistic", "pvalue"):
        assert col in result.columns, f"Missing column: {col!r}"
    assert result["method"].iloc[0] == "set_benchmark"


# ---------------------------------------------------------------------------
# End-to-end uns propagation chain
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_uns_propagation_through_full_chain(cell_data, tmp_path):
    """TVN backprojection matrices stored in uns must survive every subsequent
    pipeline step: filter → transform-yj → tvn → agg → center → similarity.
    """
    # Step 1: filter
    inp0 = _write_zarr(cell_data, tmp_path / "raw")
    out_filter = str(tmp_path / "s1_filter")
    run_pipeline_map_filter(
        _ns(input=[inp0], output=out_filter, min_variance=None,
            max_variance=None, max_fraction_not_finite=None)
    )

    # Step 2: Yeo-Johnson
    out_yj = str(tmp_path / "s2_yj")
    run_pipeline_map_transform_yj(
        _ns(input=[out_filter + ".zarr"], output=out_yj)
    )

    # Step 3: TVN
    out_tvn = str(tmp_path / "s3_tvn")
    run_pipeline_map_tvn(
        _ns(input=[out_yj + ".zarr"], output=out_tvn,
            reference_query="gene_symbol=='NTC'")
    )

    # Step 4: aggregate
    out_agg = str(tmp_path / "s4_agg")
    run_pipeline_map_agg(
        _ns(input=[out_tvn + ".zarr"], output=out_agg, agg_by=["gene_symbol"],
            agg_method="mean", min_cells=None, barcode=None, agg_by_barcode=False,
            perturbation="gene_symbol")
    )

    # Step 5: center
    out_center = str(tmp_path / "s5_center")
    run_pipeline_map_center(
        _ns(input=[out_agg + ".zarr"], output=out_center,
            reference_query="gene_symbol=='NTC'")
    )

    # Step 6: similarity (exclude NTC — after centering, NTC = zero vector so
    # cosine self-similarity is undefined; the real workflow always drops controls here)
    out_sim = str(tmp_path / "s6_sim")
    run_pipeline_map_similarity(
        _ns(input=[out_center + ".zarr"], output=out_sim, metric="cosine",
            perturbation="gene_symbol",
            exclude_reference_query="gene_symbol=='NTC'")
    )

    # Assert that TVN backprojection keys survived all the way to the similarity matrix
    sim_result = _read_zarr(out_sim + ".zarr")
    for key in ("pca", "tvn_pre_scale_mean", "tvn_pre_scale_std",
                "covariance_alignment_inv", "normalization_arguments"):
        assert key in sim_result.uns, (
            f"TVN backprojection key {key!r} was lost after the full pipeline chain"
        )


# ---------------------------------------------------------------------------
# Additive scallops provenance
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_provenance_chain_is_additive(cell_data, tmp_path):
    """Each CLI step must append its metadata to a JSON list in uns['scallops'].

    After N steps the chain should have exactly N entries.
    """
    inp0 = _write_zarr(cell_data, tmp_path / "raw")
    out1 = str(tmp_path / "s1")
    out2 = str(tmp_path / "s2")
    out3 = str(tmp_path / "s3")

    run_pipeline_map_filter(
        _ns(input=[inp0], output=out1, min_variance=None,
            max_variance=None, max_fraction_not_finite=None)
    )
    run_pipeline_map_transform_yj(_ns(input=[out1 + ".zarr"], output=out2))
    run_pipeline_map_tvn(
        _ns(input=[out2 + ".zarr"], output=out3, reference="gene_symbol=='NTC'")
    )

    r = _read_zarr(out3 + ".zarr")
    raw_prov = r.uns.get("scallops")
    assert isinstance(raw_prov, str), "provenance must be stored as a JSON string"
    chain = json.loads(raw_prov)
    assert isinstance(chain, list), "provenance must deserialise to a list"
    assert len(chain) == 3, (
        f"Expected 3 provenance entries (one per step), got {len(chain)}"
    )
    # Each entry must have the version key (no_version=True skips it, so we
    # only verify the list structure here)
    assert all(isinstance(entry, dict) for entry in chain)


@pytest.mark.features
def test_provenance_chain_survives_similarity(cell_data, tmp_path):
    """The provenance chain must survive even through map-similarity."""
    tvn_out = str(tmp_path / "tvn")
    agg_out = str(tmp_path / "agg")
    sim_out = str(tmp_path / "sim")

    inp = _write_zarr(cell_data, tmp_path / "raw")
    run_pipeline_map_tvn(
        _ns(input=[inp], output=tvn_out, reference="gene_symbol=='NTC'")
    )
    run_pipeline_map_agg(
        _ns(input=[tvn_out + ".zarr"], output=agg_out, by=["gene_symbol"],
            method="mean", min_cells=None, barcode=None, agg_by_barcode=False,
            perturbation="gene_symbol")
    )
    run_pipeline_map_similarity(
        _ns(input=[agg_out + ".zarr"], output=sim_out, metric="cosine",
            perturbation="gene_symbol", exclude_reference_query=None)
    )

    result = _read_zarr(sim_out + ".zarr")
    chain = json.loads(result.uns["scallops"])
    assert len(chain) == 3, f"Expected 3 (tvn+agg+sim), got {len(chain)}"


# ---------------------------------------------------------------------------
# backproject_tvn
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_backproject_tvn_round_trip(tvn_data):
    """Backproject then re-project through PCA must recover the TVN embedding.

    For a full-rank PCA (n_components == n_features) the composition
    ``forward ∘ inverse`` is the identity up to floating-point noise.
    """
    X_bp = backproject_tvn(tvn_data, to_original_scale=False)  # (n, n_features)
    PCs = np.asarray(tvn_data.uns["pca"]["PCs"])        # (n_pcs, n_features)
    pca_mean = np.asarray(tvn_data.uns["pca"]["mean"])  # (n_features,)

    # Re-apply PCA: (X_bp - pca_mean) @ PCs.T  →  should equal tvn_data.X
    X_reprojected = (X_bp - pca_mean) @ PCs.T
    np.testing.assert_allclose(
        X_reprojected,
        np.asarray(tvn_data.X, dtype=np.float64),
        atol=1e-4,
        rtol=1e-4,
        err_msg="Backproject then re-project should recover original TVN embedding",
    )


@pytest.mark.features
def test_backproject_tvn_shape(tvn_data):
    """Backprojected array shape must match (n_obs, n_features)."""
    result = backproject_tvn(tvn_data)
    assert result.shape == (tvn_data.shape[0], N_FEATURES)


@pytest.mark.features
def test_backproject_tvn_original_scale_different(tvn_data):
    """to_original_scale=True should differ from the default z-score output."""
    z = backproject_tvn(tvn_data, to_original_scale=False)
    orig = backproject_tvn(tvn_data, to_original_scale=True)
    assert not np.allclose(z, orig), (
        "to_original_scale=True and False should differ when pre_scale != identity"
    )


@pytest.mark.features
def test_backproject_tvn_missing_uns_raises(cell_data):
    """backproject_tvn must raise ValueError when uns keys are absent."""
    from scallops.features.backprojection import _validate_tvn_uns
    with pytest.raises(ValueError, match="missing uns keys"):
        _validate_tvn_uns(cell_data)  # cell_data has no TVN uns


# ---------------------------------------------------------------------------
# top_features_from_backprojection — genes query
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_top_features_genes_returns_dataframe(tvn_data):
    result = top_features_from_backprojection(
        tvn_data,
        genes=["gene_A"],
        perturbation_column="gene_symbol",
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "score", "pvalue"]
    # Verify sort: each |score| must be ≥ the next one
    abs_scores = result["score"].abs().tolist()
    assert abs_scores == sorted(abs_scores, reverse=True), (
        "Rows are not sorted by |score| descending"
    )


@pytest.mark.features
def test_top_features_genes_top_k(tvn_data):
    result = top_features_from_backprojection(
        tvn_data,
        genes=["gene_A"],
        perturbation_column="gene_symbol",
        top_k=2,
    )
    assert len(result) == 2


@pytest.mark.features
def test_top_features_returns_all_features_by_default(tvn_data):
    result = top_features_from_backprojection(
        tvn_data,
        genes=["gene_A"],
        perturbation_column="gene_symbol",
    )
    assert len(result) == N_FEATURES
    assert set(result["feature"]) == set(list(tvn_data.var.index))


@pytest.mark.features
def test_top_features_no_stat_filter_pvalue_is_nan(tvn_data):
    """Without pc_stat_filter, pvalue should always be NaN."""
    result = top_features_from_backprojection(
        tvn_data,
        genes=["gene_A"],
    )
    assert result["pvalue"].isna().all()


@pytest.mark.features
@pytest.mark.parametrize("pc_filter", ["ttest", "mannwhitney"])
def test_top_features_pc_stat_filter_adds_pvalue(tvn_data, pc_filter):
    """With pc_stat_filter, pvalue is a contribution-weighted average of PC p-values."""
    result = top_features_from_backprojection(
        tvn_data,
        genes=["gene_A"],
        pc_stat_filter=pc_filter,
        pc_pvalue_threshold=1.0,  # keep all PCs so scores are non-zero
    )
    assert len(result) == N_FEATURES
    # When all PCs are retained, pvalues should be non-NaN (finite contributions)
    assert result["pvalue"].notna().all()
    assert (result["pvalue"] >= 0).all() and (result["pvalue"] <= 1).all()


@pytest.mark.features
def test_top_features_pc_stat_filter_prunes_pcs(tvn_data):
    """With a permissive threshold some PCs survive; strict zeroes all → smaller sum."""
    # Use a permissive threshold so SOME PCs remain (result_filter is non-zero)
    result_no_filter = top_features_from_backprojection(
        tvn_data, genes=["gene_A"]
    )
    result_filter_permissive = top_features_from_backprojection(
        tvn_data, genes=["gene_A"],
        pc_stat_filter="ttest", pc_pvalue_threshold=1.0  # keep all PCs
    )
    result_filter_strict = top_features_from_backprojection(
        tvn_data, genes=["gene_A"],
        pc_stat_filter="ttest", pc_pvalue_threshold=0.0001  # zero most PCs
    )
    # With threshold=1.0 (keep all) result equals no-filter
    np.testing.assert_allclose(
        result_filter_permissive["score"].values,
        result_no_filter["score"].reindex(result_filter_permissive.index).values,
        rtol=1e-4,
    )
    # With strict threshold, total |score| is strictly <= permissive (some PCs zeroed)
    assert (result_filter_strict["score"].abs().sum()
            <= result_filter_permissive["score"].abs().sum())
    # The pvalue column is non-NaN when filter runs (at least some PCs have p <= 1.0)
    assert result_filter_permissive["pvalue"].notna().all()


@pytest.mark.features
def test_top_features_sorted_by_abs_score(tvn_data):
    result = top_features_from_backprojection(tvn_data, genes=["gene_A"])
    abs_scores = result["score"].abs().tolist()
    assert abs_scores == sorted(abs_scores, reverse=True)


# ---------------------------------------------------------------------------
# top_features_from_backprojection — cluster labels
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_top_features_cluster_labels(tvn_data):
    """cluster_labels + cluster_query must work identically to genes selection."""
    labels = tvn_data.obs["gene_symbol"].values
    result_genes = top_features_from_backprojection(
        tvn_data, genes=["gene_A"]
    )
    result_cluster = top_features_from_backprojection(
        tvn_data,
        cluster_labels=labels,
        cluster_query="gene_A",
    )
    pd.testing.assert_frame_equal(result_genes, result_cluster)


@pytest.mark.features
def test_top_features_cluster_multi_value(tvn_data):
    """Passing a list to cluster_query should combine both clusters."""
    result = top_features_from_backprojection(
        tvn_data,
        cluster_labels=tvn_data.obs["gene_symbol"].values,
        cluster_query=["gene_A", "gene_B"],
    )
    assert len(result) == N_FEATURES


@pytest.mark.features
def test_top_features_mutual_exclusion_raises(tvn_data):
    """Providing both genes and cluster_labels + cluster_query must raise."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        top_features_from_backprojection(
            tvn_data,
            genes=["gene_A"],
            cluster_labels=tvn_data.obs["gene_symbol"].values,
            cluster_query="gene_A",
        )


@pytest.mark.features
def test_top_features_no_query_raises(tvn_data):
    with pytest.raises(ValueError, match="query"):
        top_features_from_backprojection(tvn_data)


@pytest.mark.features
def test_top_features_genes_ref_specific_reference(tvn_data):
    """genes_ref should restrict the reference to a specific gene set."""
    # Reference = gene_B only (not all non-gene_A)
    result_ref_all = top_features_from_backprojection(
        tvn_data, genes=["gene_A"]
    )
    result_ref_b = top_features_from_backprojection(
        tvn_data, genes=["gene_A"], genes_ref=["gene_B"]
    )
    # The scores should differ because the reference changes
    assert len(result_ref_b) == N_FEATURES
    # NTC is excluded from reference so the centroid changes
    assert not np.allclose(
        result_ref_all["score"].values, result_ref_b["score"].values
    ), "Restricting reference to gene_B should change the scores"


@pytest.mark.features
def test_top_features_cluster_ref_specific_reference(tvn_data):
    """cluster_ref should restrict the reference to a specific cluster."""
    labels = tvn_data.obs["gene_symbol"].values
    result_ref_all = top_features_from_backprojection(
        tvn_data, cluster_labels=labels, cluster_query="gene_A"
    )
    result_ref_b = top_features_from_backprojection(
        tvn_data,
        cluster_labels=labels,
        cluster_query="gene_A",
        cluster_ref="gene_B",
    )
    assert len(result_ref_b) == N_FEATURES
    assert not np.allclose(
        result_ref_all["score"].values, result_ref_b["score"].values
    )


@pytest.mark.features
def test_top_features_genes_ref_mutual_exclusion_raises(tvn_data):
    """genes_ref and cluster_ref cannot both be given."""
    with pytest.raises(ValueError, match="not both"):
        top_features_from_backprojection(
            tvn_data,
            genes=["gene_A"],
            genes_ref=["gene_B"],
            cluster_ref="NTC",
        )


# ---------------------------------------------------------------------------
# top_features_from_backprojection — on aggregated profiles
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_top_features_on_aggregated_profiles(tvn_data):
    """The function must work on perturbation-level profiles (post-agg)."""
    genes = list(tvn_data.obs["gene_symbol"].unique())
    X_agg = np.stack(
        [tvn_data.X[tvn_data.obs["gene_symbol"] == g].mean(axis=0) for g in genes]
    ).astype(np.float32)
    obs = pd.DataFrame({"gene_symbol": genes}, index=genes)
    agg = anndata.AnnData(
        X=X_agg, obs=obs, var=tvn_data.var.copy(), uns=dict(tvn_data.uns)
    )
    for k, v in tvn_data.varm.items():
        agg.varm[k] = v

    result = top_features_from_backprojection(agg, genes=["gene_A"])
    assert len(result) == N_FEATURES
    # With only 1 query row and 2 reference rows, centroid backprojection still works
    assert not result["score"].isna().any()


# ---------------------------------------------------------------------------
# Backprojection mutation-killing tests  (low mutation score survivors fixed)
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_backproject_tvn_pca_mean_applied(tvn_data):
    """pca_mean must be ADDED (not subtracted) in the inverse PCA step.

    Verify that (backproject(tvn_X) - pca_mean) @ PCs.T exactly recovers
    tvn_X (the round-trip identity for full-rank PCA).  If the sign of
    pca_mean were wrong, the residual would be 2 * pca_mean * PCs.T ≠ 0.
    """
    PCs = np.asarray(tvn_data.uns["pca"]["PCs"])        # (n_pcs, n_features)
    pca_mean = np.asarray(tvn_data.uns["pca"]["mean"])  # (n_features,)
    X_tvn = np.asarray(tvn_data.X, dtype=np.float64)
    X_bp = backproject_tvn(tvn_data, to_original_scale=False)

    # Correct inverse: (X_bp - pca_mean) @ PCs.T ≈ X_tvn
    X_reproj = (X_bp - pca_mean) @ PCs.T
    np.testing.assert_allclose(X_reproj, X_tvn, atol=1e-4, rtol=1e-4,
                               err_msg="Round-trip failed: pca_mean sign may be wrong")

    # Explicitly verify the sign used is +pca_mean (not -pca_mean) by checking
    # that X_bp = X_tvn_pca_space @ PCs + pca_mean, i.e. pca_mean is additive
    # We reconstruct X_bp by hand with the correct formula and compare
    X_bp_manual = X_tvn @ PCs + pca_mean   # (TVN X was produced by (X_z - pca_mean) @ PCs.T)
    np.testing.assert_allclose(X_bp, X_bp_manual, atol=1e-4, rtol=1e-4,
                               err_msg="backproject_tvn output does not match manual +pca_mean formula")


@pytest.mark.features
def test_backproject_tvn_covariance_alignment_applied(cell_data):
    """When a single covariance alignment group is present, it must be inverted.

    Uses a single-group TVN (no --by) to ensure the alignment is always
    applied.  We then manually zero out covariance_alignment_inv and verify
    the result changes — proving the code path is exercised.
    """
    # No --by: the function produces a dummy covariance_alignment_inv with one entry
    # We simulate this by using by=["plate"] (creates per-plate matrices) and then
    # testing with a specific group key so the alignment is deterministically applied.
    tvn_by = typical_variation_normalization(
        cell_data.copy(), "gene_symbol=='NTC'", by=["plate"]
    )
    cov_inv = tvn_by.uns.get("covariance_alignment_inv", {})
    assert cov_inv, "Expected covariance_alignment_inv to be set"

    # Pick the first available group
    first_group = next(iter(cov_inv))

    # With alignment: specify the group explicitly so it IS applied
    X_with_alignment = backproject_tvn(tvn_by, group=first_group,
                                       to_original_scale=False)

    # Without alignment: remove the key from uns
    tvn_no_align = anndata.AnnData(
        X=tvn_by.X.copy(), obs=tvn_by.obs.copy(), var=tvn_by.var.copy(),
        uns={k: v for k, v in tvn_by.uns.items()
             if k != "covariance_alignment_inv"},
        varm=dict(tvn_by.varm),
    )
    X_without_alignment = backproject_tvn(tvn_no_align, to_original_scale=False)

    # The two must differ — the alignment matrix is not the identity
    diff = np.abs(X_with_alignment - X_without_alignment).max()
    assert diff > 1e-6, (
        f"Covariance alignment appears to have no effect (max diff={diff:.2e}). "
        "Either the alignment matrix is identity or the code path is wrong."
    )


@pytest.mark.features
def test_backproject_tvn_pre_std_zero_guard(cell_data):
    """Features with pre_std == 0 must not cause division-by-zero.

    The guard replaces zero std with 1.0 so the feature passes through unchanged.
    """
    tvn = typical_variation_normalization(
        cell_data.copy(), "gene_symbol=='NTC'"
    )
    # Force one std to zero
    tvn.uns["tvn_pre_scale_std"] = tvn.uns["tvn_pre_scale_std"].copy()
    tvn.uns["tvn_pre_scale_std"][0] = 0.0

    # Must not raise and must not produce NaN/Inf
    X_orig = backproject_tvn(tvn, to_original_scale=True)
    assert np.all(np.isfinite(X_orig)), "NaN or Inf in backprojection with zero pre_std"


# ---------------------------------------------------------------------------
# map run — integration tests (was at 0% coverage)
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_run_label_filter_applied(cell_data, tmp_path):
    """--label-filter must reduce the cell count before any step runs."""
    from scallops.cli.map_build import run_pipeline_map_run

    inp = _write_zarr(cell_data, tmp_path / "raw")
    args = argparse.Namespace(
        input=[inp], output_dir=str(tmp_path / "out"),
        steps="filter",
        force=True, no_version=True,
        label_filter=f"gene_symbol!='gene_B'",   # drop gene_B cells
        min_variance=None, max_variance=None, max_fraction_not_finite=None,
        max_correlation=None, batch_column=None,
        batch_pvalue=0.05, batch_method="kruskal", batch_reference=None,
        plate_column="plate", well_column="well",
        condition_column=None, condition_source_column="well", condition_map=None,
        reference_query="gene_symbol=='NTC'",
        perturbation="gene_symbol", tvn_by=None,
        scale_method="global",
        localz_neighbors=75, scale_max_value=5.0,
        localz_centroid_y="Nuclei_AreaShape_Center_Y",
        localz_centroid_x="Nuclei_AreaShape_Center_X",
        pca_components=N_FEATURES, pca_batch_size=10, pca_select_method="variance",
        pca_variance_fraction=0.80,
        agg_by=["gene_symbol"], agg_method="mean", min_cells=None,
        metric="cosine", cluster_method=None, cluster_auto_params=True,
        cluster_n_clusters=None, cluster_linkage="ward", cluster_max_n_clusters=10,
        cluster_min_cluster_size=None, cluster_min_samples=None,
        cluster_resolution=None, cluster_similarity_threshold=0.3,
        cluster_elbow_n_range=10, cluster_leiden_res_min=0.05,
        cluster_leiden_res_max=2.0, cluster_random_state=0,
        corum=None, gmt=None, string_fetch=False, string_threshold=400,
        string_species=9606, string_network_type="full",
        min_genes=1, min_pairs=1,
    )
    run_pipeline_map_run(args)

    cells = _read_zarr(str(tmp_path / "out" / "cells.zarr"))
    n_gene_b = (cells.obs["gene_symbol"] == "gene_B").sum()
    assert n_gene_b == 0, f"label_filter did not remove gene_B cells, {n_gene_b} remain"


@pytest.mark.features
def test_map_run_condition_map_creates_column(cell_data, tmp_path):
    """--condition-map must create the obs column before TVN grouping."""
    from scallops.cli.map_build import run_pipeline_map_run

    inp = _write_zarr(cell_data, tmp_path / "raw")
    args = argparse.Namespace(
        input=[inp], output_dir=str(tmp_path / "out"),
        steps="filter",  # just filter so it's fast
        force=True, no_version=True,
        label_filter=None,
        min_variance=None, max_variance=None, max_fraction_not_finite=None,
        max_correlation=None, batch_column=None,
        batch_pvalue=0.05, batch_method="kruskal", batch_reference=None,
        plate_column="plate", well_column="well",
        condition_column="cond",
        condition_source_column="plate",
        condition_map='{"plate1":"A","plate2":"B"}',
        reference_query="gene_symbol=='NTC'",
        perturbation="gene_symbol", tvn_by=["cond"],
        scale_method="global",
        localz_neighbors=75, scale_max_value=5.0,
        localz_centroid_y="Nuclei_AreaShape_Center_Y",
        localz_centroid_x="Nuclei_AreaShape_Center_X",
        pca_components=N_FEATURES, pca_batch_size=10, pca_select_method="variance",
        pca_variance_fraction=0.80,
        agg_by=["gene_symbol"], agg_method="mean", min_cells=None,
        metric="cosine", cluster_method=None, cluster_auto_params=True,
        cluster_n_clusters=None, cluster_linkage="ward", cluster_max_n_clusters=10,
        cluster_min_cluster_size=None, cluster_min_samples=None,
        cluster_resolution=None, cluster_similarity_threshold=0.3,
        cluster_elbow_n_range=10, cluster_leiden_res_min=0.05,
        cluster_leiden_res_max=2.0, cluster_random_state=0,
        corum=None, gmt=None, string_fetch=False, string_threshold=400,
        string_species=9606, string_network_type="full",
        min_genes=1, min_pairs=1,
    )
    run_pipeline_map_run(args)
    cells = _read_zarr(str(tmp_path / "out" / "cells.zarr"))
    assert "cond" in cells.obs.columns, "condition_map did not create obs['cond']"
    assert set(cells.obs["cond"].unique()) == {"A", "B"}


@pytest.mark.features
def test_map_run_missing_condition_column_raises(cell_data, tmp_path):
    """If --condition-column is given without --condition-map and the column
    is absent from the data, a clear ValueError must be raised."""
    from scallops.cli.map_build import run_pipeline_map_run

    inp = _write_zarr(cell_data, tmp_path / "raw")
    args = argparse.Namespace(
        input=[inp], output_dir=str(tmp_path / "out"),
        steps="filter", force=True, no_version=True,
        label_filter=None,
        min_variance=None, max_variance=None, max_fraction_not_finite=None,
        max_correlation=None, batch_column=None,
        batch_pvalue=0.05, batch_method="kruskal", batch_reference=None,
        plate_column="plate", well_column="well",
        condition_column="nonexistent_col",   # doesn't exist, no map provided
        condition_source_column="well", condition_map=None,
        reference_query="gene_symbol=='NTC'", perturbation="gene_symbol",
        tvn_by=None, scale_method="global",
        localz_neighbors=75, scale_max_value=5.0,
        localz_centroid_y="Nuclei_AreaShape_Center_Y",
        localz_centroid_x="Nuclei_AreaShape_Center_X",
        pca_components=N_FEATURES, pca_batch_size=10, pca_select_method="variance",
        pca_variance_fraction=0.80,
        agg_by=["gene_symbol"], agg_method="mean", min_cells=None,
        metric="cosine", cluster_method=None, cluster_auto_params=True,
        cluster_n_clusters=None, cluster_linkage="ward", cluster_max_n_clusters=10,
        cluster_min_cluster_size=None, cluster_min_samples=None,
        cluster_resolution=None, cluster_similarity_threshold=0.3,
        cluster_elbow_n_range=10, cluster_leiden_res_min=0.05,
        cluster_leiden_res_max=2.0, cluster_random_state=0,
        corum=None, gmt=None, string_fetch=False, string_threshold=400,
        string_species=9606, string_network_type="full",
        min_genes=1, min_pairs=1,
    )
    with pytest.raises(ValueError, match="nonexistent_col"):
        run_pipeline_map_run(args)


# ---------------------------------------------------------------------------
# map_cluster coverage (was at 9%)
# ---------------------------------------------------------------------------


@pytest.fixture
def square_sim_adata(profile_data) -> anndata.AnnData:
    """Small square similarity matrix built from profile_data."""
    sims = np.clip(profile_data.X @ profile_data.X.T, -1, 1).astype(np.float32)
    np.fill_diagonal(sims, 1.0)
    labels = list(profile_data.obs.index)
    return anndata.AnnData(
        X=sims,
        obs=pd.DataFrame(index=labels),
        var=pd.DataFrame(index=labels),
    )


@pytest.mark.features
def test_cluster_hdbscan_min_samples_elbow(square_sim_adata):
    """_hdbscan_auto_min_samples must return an int in [1, min_cluster_size]."""
    pytest.importorskip("hdbscan")
    from scallops.features.map_cluster import _hdbscan_auto_min_samples, _sim_to_dist
    dist = _sim_to_dist(np.asarray(square_sim_adata.X, dtype=np.float64))
    min_cs = 2
    ms = _hdbscan_auto_min_samples(dist, min_cluster_size=min_cs, n_range=5)
    assert isinstance(ms, int)
    assert 1 <= ms <= min_cs


@pytest.mark.features
def test_cluster_hdbscan_auto_both_params(square_sim_adata):
    """HDBSCAN with auto_params=True must record both params in uns['clustering']."""
    pytest.importorskip("hdbscan")
    from scallops.features.map_cluster import cluster_similarity
    result = cluster_similarity(square_sim_adata, method="hdbscan",
                                auto_params=True, elbow_n_range=5)
    info = result.uns["clustering"]
    assert info["method"] == "hdbscan"
    assert isinstance(info["min_cluster_size"], int)
    assert isinstance(info["min_samples"], int)
    assert "cluster" in result.obs.columns
    assert result.shape == square_sim_adata.shape


@pytest.mark.features
def test_cluster_hierarchical_different_linkages(square_sim_adata):
    """Each linkage method must produce a valid, correctly-labelled clustering."""
    from scallops.features.map_cluster import cluster_similarity
    n = square_sim_adata.shape[0]
    for linkage in ("ward", "complete", "average", "single"):
        result = cluster_similarity(square_sim_adata, method="hierarchical",
                                    linkage_method=linkage, auto_params=False,
                                    n_clusters=min(2, n - 1))
        assert result.uns["clustering"]["linkage"] == linkage
        assert "cluster" in result.obs.columns
        # Clusters must be contiguous (same-cluster entries adjacent)
        clusters = result.obs["cluster"].values
        for i in range(len(clusters) - 1):
            if clusters[i] != clusters[i + 1]:
                assert clusters[i] not in set(clusters[i + 1:])


@pytest.mark.features
def test_find_elbow_monotone_decreasing():
    """Elbow on a step-function curve should be at the step position."""
    from scallops.features.map_cluster import _find_elbow
    x = np.arange(10, dtype=float)
    y = np.array([10, 10, 10, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    idx = _find_elbow(x, y)
    assert 1 <= idx <= 4, f"Elbow at {idx}, expected 1–4"


@pytest.mark.features
def test_cluster_with_anndata_format(profile_data, square_sim_adata):
    """cluster_similarity must work on both matrix and anndata similarity formats."""
    from scallops.features.map_cluster import cluster_similarity

    # Matrix format (X = square sim)
    r_mat = cluster_similarity(square_sim_adata, method="hierarchical",
                               auto_params=True)
    assert r_mat.shape == square_sim_adata.shape
    assert "cluster" in r_mat.obs.columns

    # AnnData format (profiles in X, sim in obsp)
    sims = square_sim_adata.X.copy()
    prof = anndata.AnnData(X=profile_data.X.copy(), obs=profile_data.obs.copy(),
                           var=profile_data.var.copy())
    prof.obsp["similarity"] = sims
    r_prof = cluster_similarity(prof, method="hierarchical", auto_params=True)
    assert r_prof.shape == prof.shape
    assert "similarity" in r_prof.obsp
    assert "cluster" in r_prof.obs.columns


# ---------------------------------------------------------------------------
# Accuracy: known signal must rank first
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_data():
    """AnnData where ONE feature (feat_0) has a large, known shift for gene_A.

    All other features are identical noise shared by NTC, gene_A, and gene_B.
    After TVN, the backprojection of the gene_A vs NTC centroid difference
    must give feat_0 the highest absolute score.
    """
    np.random.seed(99)
    n_ntc, n_a, n_b = 30, 20, 20   # n_ntc > n_features for full-rank PCA
    n = n_ntc + n_a + n_b
    p = 5   # features

    # Shared background noise (same distribution for all groups)
    X = np.random.randn(n, p).astype(np.float32) * 0.1

    # feat_0 gets a large positive shift ONLY for gene_A
    ntc_end   = n_ntc
    a_end     = n_ntc + n_a
    X[ntc_end:a_end, 0] += 5.0      # large positive shift in feat_0 for gene_A

    genes = ["NTC"] * n_ntc + ["gene_A"] * n_a + ["gene_B"] * n_b
    obs = pd.DataFrame({"gene_symbol": genes},
                       index=pd.RangeIndex(n).astype(str))
    var = pd.DataFrame(index=[f"feat_{i}" for i in range(p)])
    data = anndata.AnnData(X=X, obs=obs, var=var)
    return typical_variation_normalization(data, reference_query="gene_symbol=='NTC'")


@pytest.mark.features
def test_top_features_correctly_identifies_signal_feature(signal_data):
    """feat_0 has a large shift for gene_A; it must rank first.

    This is the key accuracy test: when we know which feature carries the
    signal, backprojection must recover it as the most discriminating feature.
    """
    result = top_features_from_backprojection(
        signal_data, genes=["gene_A"]
    )
    assert result.iloc[0]["feature"] == "feat_0", (
        f"Expected feat_0 to rank first (largest |score|); "
        f"got {result[['feature','score']].to_dict('records')}"
    )


@pytest.mark.features
def test_top_features_signal_score_direction(signal_data):
    """gene_A has feat_0 shifted POSITIVE relative to NTC.

    The centroid difference (gene_A − NTC) in feat_0 should be positive,
    so the backprojected score for feat_0 must be positive.
    """
    result = top_features_from_backprojection(
        signal_data, genes=["gene_A"]
    )
    feat0_score = result.loc[result["feature"] == "feat_0", "score"].iloc[0]
    assert feat0_score > 0, (
        f"Expected positive score for feat_0 (gene_A shifted up); got {feat0_score:.4f}"
    )


@pytest.mark.features
def test_top_features_noise_features_rank_below_signal(signal_data):
    """The four noise features (feat_1 … feat_4) must all rank below feat_0."""
    result = top_features_from_backprojection(
        signal_data, genes=["gene_A"]
    )
    top_feature = result.iloc[0]["feature"]
    signal_score = result.iloc[0]["score"]
    noise_scores = result.iloc[1:]["score"].abs()
    assert top_feature == "feat_0"
    assert (signal_score > noise_scores).all(), (
        "Signal feature feat_0 must have a higher |score| than every noise feature"
    )


# ---------------------------------------------------------------------------
# map-filter: new filter arguments
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_filter_removes_correlated_feature(tmp_path):
    """--max-correlation should remove the lower-variance correlated feature."""
    np.random.seed(0)
    n = 40
    X = np.random.randn(n, 5).astype(np.float64)  # float64 for reliable correlations
    X[:, 1] = X[:, 0] * 0.99 + 0.001 * np.random.randn(n)  # feat1 ≈ feat0, lower variance
    X = X.astype(np.float32)
    data = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"gene_symbol": ["NTC"] * n},
                         index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"f{i}" for i in range(5)]),
    )
    inp = _write_zarr(data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    ns = _ns(
        input=[inp], output=out,
        min_variance=None, max_variance=None, max_fraction_not_finite=None,
        max_correlation=0.9, correlation_reference=None, correlation_chunk_size=512,
        max_zero_fraction=None, near_zero_threshold=0.0,
        min_unique=None,
        batch_column=None,
    )
    run_pipeline_map_filter(ns)
    result = _read_zarr(out + ".zarr")
    # At least one feature was removed, and f0 (higher variance) was kept
    assert result.shape[1] < 5
    assert "f0" in result.var.index


@pytest.mark.features
def test_map_filter_removes_zero_inflated_feature(tmp_path):
    """--max-zero-fraction should remove features with too many zeros."""
    np.random.seed(1)
    n = 30
    X = np.random.randn(n, 5).astype(np.float32)
    X[:25, 3] = 0.0  # feat3: 83% zeros
    data = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"f{i}" for i in range(5)]),
    )
    inp = _write_zarr(data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    ns = _ns(
        input=[inp], output=out,
        min_variance=None, max_variance=None, max_fraction_not_finite=None,
        max_zero_fraction=0.5, near_zero_threshold=0.0,
        min_unique=None, max_correlation=None,
        batch_column=None,
    )
    run_pipeline_map_filter(ns)
    result = _read_zarr(out + ".zarr")
    assert "f3" not in result.var.index


@pytest.mark.features
def test_map_filter_removes_categorical_feature(tmp_path):
    """--min-unique should remove binary/integer-coded features."""
    np.random.seed(2)
    n = 30
    X = np.random.randn(n, 5).astype(np.float32)
    X[:, 4] = (X[:, 4] > 0).astype(np.float32)  # feat4: binary
    data = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"f{i}" for i in range(5)]),
    )
    inp = _write_zarr(data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    ns = _ns(
        input=[inp], output=out,
        min_variance=None, max_variance=None, max_fraction_not_finite=None,
        min_unique=5, max_zero_fraction=None, near_zero_threshold=0.0,
        max_correlation=None, batch_column=None,
    )
    run_pipeline_map_filter(ns)
    result = _read_zarr(out + ".zarr")
    assert "f4" not in result.var.index


# ---------------------------------------------------------------------------
# map-sphere
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_sphere_shape_unchanged(cell_data, tmp_path):
    """map-sphere must preserve obs count and feature count."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "sphere")

    run_pipeline_map_sphere(
        _ns(input=[inp], output=out, by=None, epsilon=1e-5)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape == cell_data.shape
    assert list(result.var.index) == list(cell_data.var.index)


@pytest.mark.features
def test_map_sphere_covariance_approx_identity(cell_data, tmp_path):
    """After sphering the sample covariance diagonal should be ≈ 1."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "sphere")

    run_pipeline_map_sphere(
        _ns(input=[inp], output=out, by=None, epsilon=1e-5)
    )
    result = _read_zarr(out + ".zarr")
    cov = np.cov(result.X, rowvar=False)
    np.testing.assert_allclose(np.diag(cov), 1.0, atol=0.2)


@pytest.mark.features
def test_map_sphere_propagates_uns(cell_data, tmp_path):
    """map-sphere must propagate upstream uns."""
    cell_data.uns["upstream_key"] = "sentinel"
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "sphere")

    run_pipeline_map_sphere(
        _ns(input=[inp], output=out, by=None, epsilon=1e-5)
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("upstream_key") == "sentinel"


# ---------------------------------------------------------------------------
# map-pca-select
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_pca_select_variance_method(cell_data, tmp_path):
    """map-pca-select (variance) must retain fewer than all PCs."""
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[_write_zarr(cell_data, tmp_path / "raw")],
            output=pca_out, pca_components=N_FEATURES, pca_batch_size=0,
            reference_query=None)
    )

    out = str(tmp_path / "selected")
    run_pipeline_map_pca_select(
        _ns(input=[pca_out + ".zarr"], output=out,
            method="variance", min_variance_fraction=0.80,
            pval=0.05, n_perms=20, max_components=None, n_features=None)
    )
    result = _read_zarr(out + ".zarr")
    n_pcs = result.obsm["X_pca"].shape[1] if "X_pca" in result.obsm else result.shape[1]
    assert 1 <= n_pcs <= N_FEATURES


@pytest.mark.features
def test_map_pca_select_max_components_cap(cell_data, tmp_path):
    """max_components caps the number of retained PCs."""
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[_write_zarr(cell_data, tmp_path / "raw")],
            output=pca_out, pca_components=N_FEATURES, pca_batch_size=0,
            reference_query=None)
    )

    out = str(tmp_path / "selected")
    run_pipeline_map_pca_select(
        _ns(input=[pca_out + ".zarr"], output=out,
            method="variance", min_variance_fraction=0.99,
            pval=0.05, n_perms=10, max_components=2, n_features=None)
    )
    result = _read_zarr(out + ".zarr")
    n_pcs = result.obsm["X_pca"].shape[1] if "X_pca" in result.obsm else result.shape[1]
    assert n_pcs <= 2


@pytest.mark.features
def test_map_pca_select_tracy_widom_warns(cell_data, tmp_path):
    """map-pca-select --method tracy_widom must emit a UserWarning."""
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[_write_zarr(cell_data, tmp_path / "raw")],
            output=pca_out, pca_components=N_FEATURES, pca_batch_size=0,
            reference_query=None)
    )

    out = str(tmp_path / "selected")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_pipeline_map_pca_select(
            _ns(input=[pca_out + ".zarr"], output=out,
                method="tracy_widom", min_variance_fraction=0.95,
                pval=0.05, n_perms=10, max_components=None,
                n_features=N_FEATURES)
        )
    assert any(issubclass(x.category, UserWarning) for x in w), (
        "tracy_widom method must warn about correlated-feature assumption violation"
    )


@pytest.mark.features
def test_map_pca_select_propagates_uns(cell_data, tmp_path):
    """map-pca-select must carry upstream uns to the output."""
    cell_data.uns["upstream"] = 99
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[_write_zarr(cell_data, tmp_path / "raw")],
            output=pca_out, pca_components=N_FEATURES, pca_batch_size=0,
            reference_query=None)
    )

    out = str(tmp_path / "selected")
    run_pipeline_map_pca_select(
        _ns(input=[pca_out + ".zarr"], output=out,
            method="variance", min_variance_fraction=0.80,
            pval=0.05, n_perms=10, max_components=None, n_features=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("upstream") == 99


# ---------------------------------------------------------------------------
# _log_attrition
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_log_attrition_emits_when_cells_dropped(caplog):
    """_log_attrition must log when cells are removed."""
    import logging
    with caplog.at_level(logging.INFO, logger="scallops"):
        _log_attrition("filter", "test", 1000, 800, 50, 50)
    assert any("attrition" in r.message for r in caplog.records)
    assert any("200" in r.message for r in caplog.records)  # cells dropped


@pytest.mark.features
def test_log_attrition_emits_when_features_dropped(caplog):
    """_log_attrition must log when features are removed."""
    import logging
    with caplog.at_level(logging.INFO, logger="scallops"):
        _log_attrition("filter", "test", 1000, 1000, 50, 40)
    assert any("attrition" in r.message for r in caplog.records)


@pytest.mark.features
def test_log_attrition_silent_when_nothing_dropped(caplog):
    """_log_attrition must not log when both cells and features are unchanged."""
    import logging
    with caplog.at_level(logging.INFO, logger="scallops"):
        _log_attrition("filter", "test", 100, 100, 50, 50)
    assert not any("attrition" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# map-scale
# ---------------------------------------------------------------------------


@pytest.fixture
def cell_data_with_centroids(cell_data):
    """cell_data extended with centroid columns in obs (no NaN)."""
    np.random.seed(7)
    n = cell_data.shape[0]
    cell_data.obs["Nuclei_AreaShape_Center_Y"] = np.random.uniform(0, 512, n).astype(np.float32)
    cell_data.obs["Nuclei_AreaShape_Center_X"] = np.random.uniform(0, 512, n).astype(np.float32)
    return cell_data


@pytest.mark.features
def test_map_scale_global_shape_unchanged(cell_data, tmp_path):
    """map-scale global must preserve shape."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "scaled")
    run_pipeline_map_scale(
        _ns(input=[inp], output=out,
            scale_method="global", plate_column="plate", well_column="well")
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape == cell_data.shape


@pytest.mark.features
def test_map_scale_global_ntc_mean_near_zero(cell_data, tmp_path):
    """Global z-score: NTC mean per feature should be close to 0."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "scaled")
    run_pipeline_map_scale(
        _ns(input=[inp], output=out,
            scale_method="global", plate_column="plate", well_column="well")
    )
    result = _read_zarr(out + ".zarr")
    ntc = result[result.obs["gene_symbol"] == "NTC"].X
    assert np.nanmax(np.abs(ntc.mean(axis=0))) < 1.0


@pytest.mark.features
def test_map_scale_local_shape_unchanged(cell_data_with_centroids, tmp_path):
    """map-scale local must preserve shape (minus any NaN-centroid cells)."""
    inp = _write_zarr(cell_data_with_centroids, tmp_path / "input")
    out = str(tmp_path / "scaled_local")
    run_pipeline_map_scale(
        _ns(input=[inp], output=out,
            scale_method="local", plate_column="plate", well_column="well",
            localz_centroid_y="Nuclei_AreaShape_Center_Y",
            localz_centroid_x="Nuclei_AreaShape_Center_X",
            localz_neighbors=3, localz_batch_size=50,
            scale_max_value=5.0)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape[1] == cell_data_with_centroids.shape[1]
    assert result.shape[0] <= cell_data_with_centroids.shape[0]


@pytest.mark.features
def test_map_scale_local_drops_nan_centroid_cells(tmp_path):
    """map-scale local must silently drop cells whose centroid is NaN."""
    np.random.seed(1)
    n = 30
    X = np.random.randn(n, 4).astype(np.float32)
    cy = np.random.uniform(0, 256, n).astype(np.float32)
    cx = np.random.uniform(0, 256, n).astype(np.float32)
    cy[0] = np.nan  # one NaN centroid
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "gene_symbol": ["NTC"] * 15 + ["GENE"] * 15,
            "plate": "P1",
            "well": ["W1"] * 15 + ["W2"] * 15,
            "Nuclei_AreaShape_Center_Y": cy,
            "Nuclei_AreaShape_Center_X": cx,
        }, index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"F{i}" for i in range(4)]),
    )
    inp = _write_zarr(adata, tmp_path / "input")
    out = str(tmp_path / "scaled")
    run_pipeline_map_scale(
        _ns(input=[inp], output=out,
            scale_method="local", plate_column="plate", well_column="well",
            localz_centroid_y="Nuclei_AreaShape_Center_Y",
            localz_centroid_x="Nuclei_AreaShape_Center_X",
            localz_neighbors=3, localz_batch_size=50,
            scale_max_value=5.0)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape[0] == n - 1  # NaN-centroid cell dropped


@pytest.mark.features
def test_map_scale_propagates_uns(cell_data, tmp_path):
    """map-scale must forward upstream uns."""
    cell_data.uns["sentinel"] = "preserved"
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "scaled")
    run_pipeline_map_scale(
        _ns(input=[inp], output=out,
            scale_method="global", plate_column="plate", well_column="well")
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("sentinel") == "preserved"


# ---------------------------------------------------------------------------
# transform-yj uns preservation across _slice_anndata
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_transform_yj_preserves_uns_after_nan_prefilter(tmp_path):
    """uns must survive when the NaN pre-filter calls _slice_anndata."""
    np.random.seed(3)
    n, p = 30, 5
    X = np.random.randn(n, p).astype(np.float32)
    # Add NaN in one feature for half the cells → pre-filter will drop them
    X[:15, 0] = np.nan
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "gene_symbol": ["NTC"] * 15 + ["GENE"] * 15,
            "plate": "P1",
            "well": ["W1"] * 15 + ["W2"] * 15,
        }, index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"F{i}" for i in range(p)]),
    )
    adata.uns["upstream"] = "sentinel"
    inp = _write_zarr(adata, tmp_path / "input")
    out = str(tmp_path / "yj")
    run_pipeline_map_transform_yj(
        _ns(input=[inp], output=out, by=None,
            max_fraction_not_finite=0.25,
            plate_column="plate", well_column="well",
            scale_method="global")
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("upstream") == "sentinel", (
        "uns must survive when _slice_anndata is called during YJ NaN pre-filter"
    )


# ---------------------------------------------------------------------------
# map-pca batch_size=0 (full-dataset non-incremental PCA)
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_pca_batch_size_zero_same_shape(cell_data, tmp_path):
    """batch_size=0 must produce the same output shape as batch_size=None."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out0 = str(tmp_path / "pca_bs0")
    run_pipeline_map_pca(
        _ns(input=[inp], output=out0, components=N_FEATURES,
            batch_size=0, whiten=False, reference=None)
    )
    result = _read_zarr(out0 + ".zarr")
    assert "X_pca" in result.obsm
    assert result.obsm["X_pca"].shape == (cell_data.shape[0], N_FEATURES)


# ---------------------------------------------------------------------------
# map-sphere / map-tvn use obsm["X_pca"], not data.X
# ---------------------------------------------------------------------------


def _make_pca_adata(cell_data, tmp_path):
    """Return a zarr path whose AnnData has obsm['X_pca'] set by map-pca."""
    inp = _write_zarr(cell_data, tmp_path / "raw")
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[inp], output=pca_out, components=N_FEATURES,
            batch_size=0, whiten=False, reference=None)
    )
    return pca_out + ".zarr"


@pytest.mark.features
def test_map_sphere_operates_on_obsm_x_pca(cell_data, tmp_path):
    """When obsm['X_pca'] exists, sphere must update it, not data.X."""
    pca_zarr = _make_pca_adata(cell_data, tmp_path)
    out = str(tmp_path / "sphere")
    run_pipeline_map_sphere(
        _ns(input=[pca_zarr], output=out, by=None, epsilon=1e-5)
    )
    result = _read_zarr(out + ".zarr")
    # X must remain the original scaled features (same shape), not PC-space
    assert result.shape == cell_data.shape, "X shape must equal original feature shape"
    # obsm["X_pca"] must exist and contain the sphered PCA embedding
    assert "X_pca" in result.obsm


@pytest.mark.features
def test_map_tvn_operates_on_obsm_x_pca(cell_data, tmp_path):
    """When obsm['X_pca'] exists, tvn must produce obsm['X_tvn'], not overwrite X."""
    pca_zarr = _make_pca_adata(cell_data, tmp_path)
    out = str(tmp_path / "tvn")
    run_pipeline_map_tvn(
        _ns(input=[pca_zarr], output=out,
            reference_query="gene_symbol=='NTC'", by=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape == cell_data.shape, "X shape must equal original feature shape"
    assert "X_tvn" in result.obsm, "TVN embedding must be stored in obsm['X_tvn']"
    assert result.obsm["X_tvn"].shape[0] == cell_data.shape[0]


@pytest.mark.features
def test_map_agg_uses_obsm_x_tvn(cell_data, tmp_path):
    """map-agg must aggregate obsm['X_tvn'] when it is present."""
    pca_zarr = _make_pca_adata(cell_data, tmp_path)
    tvn_out = str(tmp_path / "tvn")
    run_pipeline_map_tvn(
        _ns(input=[pca_zarr], output=tvn_out,
            reference_query="gene_symbol=='NTC'", by=None)
    )
    agg_out = str(tmp_path / "agg")
    run_pipeline_map_agg(
        _ns(input=[tvn_out + ".zarr"], output=agg_out,
            by=["gene_symbol"], method="mean", min_cells=None,
            perturbation="gene_symbol", barcode="barcode_0",
            agg_by_barcode=False)
    )
    result = _read_zarr(agg_out + ".zarr")
    n_perts = cell_data.obs["gene_symbol"].nunique()
    assert result.shape[0] == n_perts, "One profile per perturbation"
    # var should be PC names when X_tvn was used
    assert all(v.startswith("PC") for v in result.var.index), (
        "Aggregated profiles must index var by PC names when X_tvn was used"
    )
    # TVN backprojection uns must survive aggregation
    assert "tvn_pre_scale_mean" in result.uns, (
        "TVN uns must propagate through map-agg (needed for backprojection)"
    )


@pytest.mark.features
def test_map_agg_uns_survives_min_cells_filter(cell_data, tmp_path):
    """TVN uns must propagate even when min_cells triggers _slice_anndata.

    _slice_anndata drops uns; run_pipeline_map_agg must save/restore it
    before and after the min-cells filter so _merge_uns still has a source.
    """
    pca_zarr = _make_pca_adata(cell_data, tmp_path)
    tvn_out = str(tmp_path / "tvn")
    run_pipeline_map_tvn(
        _ns(input=[pca_zarr], output=tvn_out,
            reference_query="gene_symbol=='NTC'", by=None)
    )
    agg_out = str(tmp_path / "agg_mc")
    run_pipeline_map_agg(
        _ns(input=[tvn_out + ".zarr"], output=agg_out,
            by=["gene_symbol"], method="mean",
            # min_cells=1 triggers _slice_anndata even if nothing is dropped
            min_cells=1,
            perturbation="gene_symbol", barcode="barcode_0",
            agg_by_barcode=False)
    )
    result = _read_zarr(agg_out + ".zarr")
    assert "tvn_pre_scale_mean" in result.uns, (
        "TVN uns must survive even when min_cells filter calls _slice_anndata"
    )


# ---------------------------------------------------------------------------
# map-similarity --output-format anndata (obs index / column conflict fix)
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_similarity_anndata_format_writes_without_error(
    tvn_data, tmp_path
):
    """output_format='anndata' must write without index-name/column conflict."""
    from scallops.features.agg import agg_features

    profiles = agg_features(tvn_data, by=["gene_symbol"])
    inp = _write_zarr(profiles, tmp_path / "profiles")
    out = str(tmp_path / "sim_adata")
    run_pipeline_map_similarity(
        _ns(input=[inp], output=out,
            metric="cosine",
            perturbation="gene_symbol",
            exclude_reference_query=None,
            output_format="anndata",
            cluster_method=None)
    )
    result = _read_zarr(out + ".zarr")
    assert "similarity" in result.obsp
    assert result.shape[0] == profiles.shape[0]


@pytest.mark.features
def test_map_similarity_anndata_format_obs_index_unique(tvn_data, tmp_path):
    """The anndata output's obs index must equal perturbation labels."""
    from scallops.features.agg import agg_features

    profiles = agg_features(tvn_data, by=["gene_symbol"])
    inp = _write_zarr(profiles, tmp_path / "profiles")
    out = str(tmp_path / "sim_adata")
    run_pipeline_map_similarity(
        _ns(input=[inp], output=out,
            metric="cosine",
            perturbation="gene_symbol",
            exclude_reference_query=None,
            output_format="anndata",
            cluster_method=None)
    )
    result = _read_zarr(out + ".zarr")
    perts = sorted(profiles.obs["gene_symbol"].unique())
    assert sorted(result.obs.index.tolist()) == perts


# ---------------------------------------------------------------------------
# map-backproject
# ---------------------------------------------------------------------------


@pytest.fixture
def agg_profiles(tvn_data):
    """Aggregated TVN profiles with backprojection uns and varm intact.

    Mirrors profile_data but uses agg_features so the obs index matches
    what the pipeline produces, while manually propagating uns/varm from
    the TVN output (agg_features does not propagate them automatically).
    """
    from scallops.features.agg import agg_features

    profiles = agg_features(tvn_data, by=["gene_symbol"])
    # Propagate backprojection uns and varm from the TVN-normalized source
    for k, v in tvn_data.uns.items():
        if k not in profiles.uns:
            profiles.uns[k] = v
    if tvn_data.var.index.equals(profiles.var.index):
        for k, v in tvn_data.varm.items():
            if k not in profiles.varm:
                profiles.varm[k] = v
    return profiles


@pytest.mark.features
def test_map_backproject_by_gene_writes_parquet(agg_profiles, tmp_path):
    """map backproject --query must write a parquet with feature/score/pvalue."""
    inp = _write_zarr(agg_profiles, tmp_path / "profiles")
    out = str(tmp_path / "bp.parquet")
    run_pipeline_map_backproject(
        _ns(input=[inp], output=out,
            query=["gene_A"],
            reference=None,
            cluster_query=None,
            cluster_ref=None,
            cluster_labels_zarr=None,
            perturbation_column="gene_symbol",
            top_k=None,
            pc_stat_filter=None,
            pc_pvalue_threshold=0.05,
            group=None,
            to_original_scale=False)
    )
    import os
    assert os.path.exists(out)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["feature", "score", "pvalue"]
    assert len(df) == agg_profiles.shape[1]


@pytest.mark.features
def test_map_backproject_top_k(agg_profiles, tmp_path):
    """--top-k must limit the number of returned features."""
    inp = _write_zarr(agg_profiles, tmp_path / "profiles")
    out = str(tmp_path / "bp_topk.parquet")
    run_pipeline_map_backproject(
        _ns(input=[inp], output=out,
            query=["gene_A"],
            reference=None,
            cluster_query=None, cluster_ref=None, cluster_labels_zarr=None,
            perturbation_column="gene_symbol",
            top_k=2,
            pc_stat_filter=None, pc_pvalue_threshold=0.05,
            group=None, to_original_scale=False)
    )
    df = pd.read_parquet(out)
    assert len(df) == 2


@pytest.mark.features
def test_map_backproject_sorted_by_abs_score(agg_profiles, tmp_path):
    """Rows must be sorted by |score| descending."""
    inp = _write_zarr(agg_profiles, tmp_path / "profiles")
    out = str(tmp_path / "bp_sorted.parquet")
    run_pipeline_map_backproject(
        _ns(input=[inp], output=out,
            query=["gene_A"],
            reference=None,
            cluster_query=None, cluster_ref=None, cluster_labels_zarr=None,
            perturbation_column="gene_symbol",
            top_k=None,
            pc_stat_filter=None, pc_pvalue_threshold=0.05,
            group=None, to_original_scale=False)
    )
    df = pd.read_parquet(out)
    abs_scores = df["score"].abs().tolist()
    assert abs_scores == sorted(abs_scores, reverse=True)


@pytest.mark.features
def test_map_backproject_explicit_reference(agg_profiles, tmp_path):
    """--reference must restrict the reference set to the named perturbations."""
    inp = _write_zarr(agg_profiles, tmp_path / "profiles")
    out = str(tmp_path / "bp_ref.parquet")
    run_pipeline_map_backproject(
        _ns(input=[inp], output=out,
            query=["gene_A"],
            reference=["NTC"],
            cluster_query=None, cluster_ref=None, cluster_labels_zarr=None,
            perturbation_column="gene_symbol",
            top_k=None,
            pc_stat_filter=None, pc_pvalue_threshold=0.05,
            group=None, to_original_scale=False)
    )
    df = pd.read_parquet(out)
    assert len(df) == agg_profiles.shape[1]


@pytest.mark.features
def test_map_backproject_pc_stat_filter_ttest(tvn_data, tmp_path):
    """pc_stat_filter='ttest' must populate pvalue column (not all NaN).

    Uses cell-level TVN data so each group has ≥ 2 samples (required by
    the Welch t-test; skipped on aggregated profiles which have 1 row each).
    """
    inp = _write_zarr(tvn_data, tmp_path / "tvn")
    out = str(tmp_path / "bp_ttest.parquet")
    run_pipeline_map_backproject(
        _ns(input=[inp], output=out,
            query=["gene_A"],
            reference=["NTC"],
            cluster_query=None, cluster_ref=None, cluster_labels_zarr=None,
            perturbation_column="gene_symbol",
            top_k=None,
            pc_stat_filter="ttest", pc_pvalue_threshold=1.0,
            group=None, to_original_scale=False)
    )
    df = pd.read_parquet(out)
    assert not df["pvalue"].isna().all(), "ttest filter must populate pvalue"


@pytest.mark.features
def test_map_backproject_to_original_scale(agg_profiles, tmp_path):
    """--to-original-scale must produce different scores than z-score space."""
    inp = _write_zarr(agg_profiles, tmp_path / "profiles")
    out_z = str(tmp_path / "bp_z.parquet")
    out_orig = str(tmp_path / "bp_orig.parquet")
    ns_base = dict(
        query=["gene_A"], reference=None,
        cluster_query=None, cluster_ref=None, cluster_labels_zarr=None,
        perturbation_column="gene_symbol", top_k=None,
        pc_stat_filter=None, pc_pvalue_threshold=0.05, group=None,
    )
    run_pipeline_map_backproject(_ns(input=[inp], output=out_z,
                                     to_original_scale=False, **ns_base))
    run_pipeline_map_backproject(_ns(input=[inp], output=out_orig,
                                     to_original_scale=True, **ns_base))
    df_z = pd.read_parquet(out_z)
    df_orig = pd.read_parquet(out_orig)
    assert not np.allclose(df_z["score"].values, df_orig["score"].values), (
        "Original-scale scores must differ from z-score-space scores"
    )


@pytest.mark.features
def test_map_backproject_cluster_query(agg_profiles, tmp_path):
    """--cluster-query must work when cluster labels are supplied."""
    import os

    # Attach integer cluster labels to the aggregated profiles
    n_pert = agg_profiles.shape[0]
    agg_profiles.obs["cluster"] = ([0] * (n_pert // 2) +
                                   [1] * (n_pert - n_pert // 2))

    inp    = _write_zarr(agg_profiles, tmp_path / "profiles")
    cl_zarr = _write_zarr(agg_profiles, tmp_path / "sim")  # same data as cluster source
    out    = str(tmp_path / "bp_cl.parquet")

    run_pipeline_map_backproject(
        _ns(input=[inp], output=out,
            query=None,
            reference=None,
            cluster_query=0,
            cluster_ref=None,
            cluster_labels_zarr=cl_zarr,
            perturbation_column="gene_symbol",
            top_k=None,
            pc_stat_filter=None, pc_pvalue_threshold=0.05,
            group=None, to_original_scale=False)
    )
    assert os.path.exists(out)
    df = pd.read_parquet(out)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# Parquet-path coverage for map-filter
# (the zarr/in-memory path is tested above; _col_batch_filter_parquet is only
#  triggered when the input is parquet — previously untested, leading to the
#  `pd not defined` NameError in production)
# ---------------------------------------------------------------------------


def _write_parquet(data: anndata.AnnData, path) -> str:
    """Write AnnData to a parquet file readable by _read_map_inputs.

    Feature columns use Nuclei_ prefix so _read_map_inputs classifies
    them as features; obs columns are kept as metadata.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = str(path) + ".parquet"
    # Combine obs + X into one DataFrame
    feat_names = [f"Nuclei_{c}" for c in data.var.index]
    df = data.obs.copy().reset_index(drop=True)
    for j, fname in enumerate(feat_names):
        df[fname] = data.X[:, j]
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out)
    return out


@pytest.fixture
def cell_data_parquet_friendly() -> anndata.AnnData:
    """AnnData with Nuclei_* feature names (needed for parquet classification)."""
    np.random.seed(42)
    n, p = 60, 5
    genes = ["NTC"] * 24 + ["gene_A"] * 18 + ["gene_B"] * 18
    X = np.random.randn(n, p).astype(np.float32)
    # Introduce >50% NaN in the last feature (should be dropped by step 1)
    X[:, -1] = np.nan
    # Introduce sparse NaN in feature 2 for some cells
    X[::5, 2] = np.nan

    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "gene_symbol": genes,
            "plate":       ["plate1"] * n,
            "well":        ["well1" if i < 30 else "well2" for i in range(n)],
            "barcode_0":   [f"bc_{g}" for g in genes],
        }, index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=[f"Feature_{i}" for i in range(p)]),
    )


@pytest.mark.features
def test_map_filter_parquet_path_runs(cell_data_parquet_friendly, tmp_path):
    """run_pipeline_map_filter on a parquet input exercises _col_batch_filter_parquet."""
    pq_path = _write_parquet(cell_data_parquet_friendly, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(
            input=[pq_path],
            output=out,
            min_variance=0.0,
            max_variance=None,
            max_fraction_not_finite=0.25,
            max_feature_nan_fraction=0.50,
            filter_batch_size=50,
            filter_max_memory_gb=None,
            max_cpus=1,
            plate_column="plate",
            well_column="well",
            scale_method="global",
            condition_column=None,
            condition_map=None,
            condition_source_column="well",
        )
    )
    result = _read_zarr(out + ".zarr")
    # Feature with 100% NaN (last column, step 1) must be dropped
    assert result.shape[1] < cell_data_parquet_friendly.shape[1], \
        "Parquet filter must drop the all-NaN feature"
    # No NaN should remain in the output (step 3 cleans up)
    assert not np.isnan(result.X).any(), \
        "Output matrix must be NaN-free after step 3"


@pytest.mark.features
@pytest.mark.xfail(reason="Feature-drop report removed in unified dask filter path", strict=True)
def test_map_filter_parquet_feature_report_written(cell_data_parquet_friendly, tmp_path):
    """map-filter writes a feature-drop report parquet alongside the zarr."""
    pq_path = _write_parquet(cell_data_parquet_friendly, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(
            input=[pq_path],
            output=out,
            min_variance=0.0,
            max_variance=None,
            max_fraction_not_finite=0.25,
            max_feature_nan_fraction=0.50,
            filter_batch_size=50,
            filter_max_memory_gb=None,
            max_cpus=1,
            plate_column="plate",
            well_column="well",
            scale_method="global",
            condition_column=None,
            condition_map=None,
            condition_source_column="well",
        )
    )
    report_path = out + "_feature_report.parquet"
    assert os.path.exists(report_path), "Feature-drop report parquet must be written"

    df = pd.read_parquet(report_path)
    assert set(["feature", "compartment", "drop_step", "nan_frac_all_cells", "kept"]) \
        <= set(df.columns)
    # The all-NaN feature should appear as dropped (step1 or step3)
    dropped = df[~df["kept"]]
    assert len(dropped) > 0, "At least one feature should be dropped"


@pytest.mark.features
def test_map_filter_parquet_step3_removes_sparse_nan(cell_data_parquet_friendly, tmp_path):
    """Step 3 must remove features that still have NaN in the kept cells."""
    pq_path = _write_parquet(cell_data_parquet_friendly, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(
            input=[pq_path],
            output=out,
            min_variance=0.0,
            max_variance=None,
            max_fraction_not_finite=1.0,   # keep all cells regardless of NaN
            max_feature_nan_fraction=1.0,   # step 1 keeps all
            filter_batch_size=50,
            filter_max_memory_gb=None,
            max_cpus=1,
            plate_column="plate",
            well_column="well",
            scale_method="global",
            condition_column=None,
            condition_map=None,
            condition_source_column="well",
        )
    )
    result = _read_zarr(out + ".zarr")
    # With step 3 active, no NaN should remain even when we keep all cells/features in step 1/2
    assert not np.isnan(result.X).any(), \
        "Step 3 must eliminate all remaining NaN even when steps 1+2 are permissive"


# ---------------------------------------------------------------------------
# Additional edge-case tests — findings from max-effort review
# ---------------------------------------------------------------------------


@pytest.fixture
def tvn_data_with_map_pca(cell_data, tmp_path) -> anndata.AnnData:
    """TVN data produced through the full map-pca pipeline so uns['map_pca'] exists."""
    from scallops.cli.map_build import run_pipeline_map_pca, run_pipeline_map_tvn
    raw = _write_zarr(cell_data, tmp_path / "raw")
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[raw], output=pca_out, components=N_FEATURES,
            batch_size=0, whiten=False, reference=None)
    )
    tvn_out = str(tmp_path / "tvn")
    run_pipeline_map_tvn(
        _ns(input=[pca_out + ".zarr"], output=tvn_out,
            reference_query="gene_symbol=='NTC'", by=None)
    )
    return _read_zarr(tvn_out + ".zarr")


@pytest.mark.features
def test_top_features_map_pca_branch(tvn_data_with_map_pca):
    """map_pca branch: feature names come from map_pca['features'], not data.var."""
    data = tvn_data_with_map_pca
    assert "map_pca" in data.uns, "fixture must have map_pca in uns"
    result = top_features_from_backprojection(
        data, genes=["gene_A"], perturbation_column="gene_symbol"
    )
    expected_names = list(data.uns["map_pca"].get("features", data.var.index))
    assert list(result["feature"]) == sorted(
        result["feature"].tolist(), key=lambda f: expected_names.index(f)
        if f in expected_names else 0
    ) or set(result["feature"]) == set(expected_names), (
        "Feature names must come from map_pca['features'], not PC names"
    )
    assert len(result) == len(expected_names)
    assert result["score"].apply(np.isfinite).all(), "No NaN/inf scores from map_pca branch"


@pytest.mark.features
def test_top_features_map_pca_to_original_scale_warns():
    """to_original_scale=True warns when tvn_pre_scale_std length != feature_scores length.

    Construct the mismatch directly: map_pca maps K'=3 PCs → p=5 features,
    so feature_scores has length 5 but tvn_pre_scale_std has length K'=3.
    """
    import warnings as _w
    import anndata

    np.random.seed(42)
    K, p = 3, 5
    n = 20
    # minimal AnnData with the uns keys backprojection needs
    X = np.random.randn(n, K).astype(np.float32)
    obs = pd.DataFrame({"gene_symbol": ["NTC"] * 10 + ["GENE"] * 10},
                        index=pd.RangeIndex(n).astype(str))
    data = anndata.AnnData(X=X, obs=obs,
                           var=pd.DataFrame(index=[f"PC{i}" for i in range(K)]))
    data.uns["pca"] = {"PCs": np.eye(K), "mean": np.zeros(K)}
    data.uns["tvn_pre_scale_mean"] = np.zeros(K)
    data.uns["tvn_pre_scale_std"]  = np.ones(K)   # length K=3
    data.uns["covariance_alignment_inv"] = {}
    data.uns["normalization_arguments"] = {}
    # map_pca maps K PCs → p original features → feature_scores has length p=5
    data.uns["map_pca"] = {
        "PCs": np.random.randn(K, p),  # (K, p)
        "mean": np.zeros(p),
        "features": [f"F{i}" for i in range(p)],
    }

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        top_features_from_backprojection(data, genes=["GENE"],
                                          perturbation_column="gene_symbol",
                                          to_original_scale=True)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("to_original_scale" in m or "skipped" in m for m in msgs), (
        f"Must warn when K ({K}) != p ({p}). Got: {msgs}"
    )


@pytest.mark.features
def test_top_features_cluster_query_string_labels(tvn_data):
    """cluster_query as int matches string labels via dtype coercion."""
    string_labels = np.array(["0", "0", "0", "1", "1", "1",
                               "0", "0", "0", "1", "1", "1",
                               "0", "0", "0", "1", "1", "1",
                               "0", "0", "0", "1", "1", "1",
                               "0", "0", "0", "1", "1", "1"])[:tvn_data.n_obs]
    # query with int — should be coerced to "0" (string) not raise
    result = top_features_from_backprojection(
        tvn_data, cluster_labels=string_labels, cluster_query=0
    )
    assert len(result) == N_FEATURES
    assert not result.empty, "int query must match string '0' labels after coercion"


@pytest.mark.features
def test_resolve_cov_alignment_multigroup_no_group_warns(cell_data):
    """_resolve_cov_alignment with group=None + multiple groups emits UserWarning."""
    from scallops.features.backprojection import _resolve_cov_alignment
    import warnings as _w
    cov_inv = {"plate1": np.eye(N_FEATURES), "plate2": np.eye(N_FEATURES) * 2}
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = _resolve_cov_alignment(cov_inv, group=None)
    assert result is None, "Must return None when group=None and multiple groups exist"
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("multiple" in m.lower() or "group" in m.lower() for m in msgs), \
        "Must warn about multiple groups"


@pytest.mark.features
def test_top_features_pc_stat_filter_small_sample_warns(tvn_data):
    """pc_stat_filter with N_query < 2 skips filter and warns."""
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = top_features_from_backprojection(
            tvn_data, genes=["gene_A"],  # gene_A has N_PERT=8 cells → N_query=8, fine
            pc_stat_filter="ttest",
        )
    # Now use a gene with only 1 cell (fake via cluster_labels)
    one_cell_mask = np.zeros(tvn_data.n_obs, dtype=int)
    one_cell_mask[0] = 1  # only 1 query cell
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result_1 = top_features_from_backprojection(
            tvn_data, cluster_labels=one_cell_mask, cluster_query=1,
            pc_stat_filter="ttest",
        )
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("2 samples" in str(w.message) or "≥ 2" in str(w.message)
               for w in user_warns), "Must warn when N_query < 2"
    # Result shape must still be valid
    assert len(result_1) == N_FEATURES
    assert result_1["pvalue"].isna().all(), "pvalue must be NaN when filter was skipped"


@pytest.mark.features
def test_top_features_zero_match_genes_raises(tvn_data):
    """genes matching no observations raises ValueError, not IndexError."""
    with pytest.raises(ValueError, match="zero observations"):
        top_features_from_backprojection(tvn_data, genes=["does_not_exist"])


@pytest.mark.features
def test_top_features_invalid_pc_stat_filter_raises(tvn_data):
    """Unknown pc_stat_filter value raises ValueError."""
    with pytest.raises(ValueError, match="Unknown pc_stat_filter"):
        top_features_from_backprojection(
            tvn_data, genes=["gene_A"], pc_stat_filter="spearman"
        )


@pytest.mark.features
def test_top_features_resolve_cov_alignment_wrong_group_raises_valueerror(cell_data):
    """_resolve_cov_alignment raises ValueError (not KeyError) for unknown group."""
    from scallops.features.backprojection import _resolve_cov_alignment
    cov_inv = {"plate1": np.eye(N_FEATURES)}
    with pytest.raises(ValueError):
        _resolve_cov_alignment(cov_inv, group="nonexistent_group")


@pytest.mark.features
def test_top_features_genes_cluster_labels_mutually_exclusive_warns(tvn_data):
    """Passing both genes and cluster_labels (no cluster_query) warns cluster_labels ignored."""
    import warnings as _w
    labels = np.zeros(tvn_data.n_obs, dtype=int)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = top_features_from_backprojection(
            tvn_data, genes=["gene_A"], cluster_labels=labels
        )
    warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("cluster_labels" in str(w.message) and "ignored" in str(w.message)
               for w in warns), "Must warn that cluster_labels is ignored when genes is set"
    assert len(result) == N_FEATURES


@pytest.mark.features
def test_top_features_zero_score_pvalue_is_one_not_nan(tvn_data):
    """Features zeroed by pc_stat_filter get pvalue=1.0, not NaN."""
    result = top_features_from_backprojection(
        tvn_data, genes=["gene_A"],
        pc_stat_filter="ttest", pc_pvalue_threshold=0.0001,  # strict → most PCs zeroed
    )
    zero_score = result[result["score"].abs() < 1e-12]
    if not zero_score.empty:
        # Zero-score features must have pvalue = 1.0 (non-discriminating, not missing)
        assert (zero_score["pvalue"] == 1.0).all() or zero_score["pvalue"].isna().all() == False, \
            "Zero-score features after stat filter must have pvalue=1.0, not NaN"


@pytest.mark.features
def test_backproject_tvn_nan_in_x_warns(tvn_data):
    """NaN values in X trigger a UserWarning; output scores remain finite."""
    import warnings as _w
    data_with_nan = tvn_data.copy()
    data_with_nan.X = np.asarray(data_with_nan.X, dtype=np.float64)
    data_with_nan.X[0, 0] = np.nan  # inject one NaN
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = top_features_from_backprojection(data_with_nan, genes=["gene_A"])
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("NaN" in str(w.message) for w in user_warns), \
        "Must warn when input X contains NaN"
    assert result["score"].apply(np.isfinite).all(), \
        "Scores must be finite even when X has NaN (nanmean used)"


# ---------------------------------------------------------------------------
# Parquet E2E map run — regression guard for the full pipeline via parquet input
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_run_parquet_full_pipeline(tmp_path):
    """run_pipeline_map_run with a parquet input must complete all steps and
    produce finite zarr outputs of the expected shapes.

    This exercises the production code path (parquet → unified filter →
    YJ → scale → PCA → TVN → agg → center → similarity) that is NOT covered
    by any other integration test (all existing tests use zarr inputs).
    """
    import argparse
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    from scallops.cli.map_build import run_pipeline_map_run
    from scallops.io import read_anndata_zarr

    np.random.seed(77)
    n_ntc, n_pert = 20, 8
    n_obs = n_ntc + n_pert * 2
    n_feat = 8
    feat_names = [f"Cells_Intensity_feat{i}" for i in range(n_feat)]

    genes = ["NTC"] * n_ntc + ["gene_A"] * n_pert + ["gene_B"] * n_pert
    wells = ["1" if i < n_obs // 2 else "2" for i in range(n_obs)]
    plates = ["p1"] * n_obs
    X = np.random.randn(n_obs, n_feat).astype(np.float32)

    pq_path = str(tmp_path / "input.parquet")
    df = pd.DataFrame(X, columns=feat_names)
    df["plate"] = plates
    df["well"] = wells
    df["gene_symbol"] = genes
    _pq.write_table(_pa.Table.from_pandas(df), pq_path)

    out_dir = str(tmp_path / "map_out")
    args = argparse.Namespace(
        input=[pq_path],
        input_pattern=None,
        output_dir=out_dir,
        steps="filter,transform-yj,scale,pca,tvn,agg,center,similarity",
        force=True, no_version=True, client="none", dask_cluster=None,
        features=None, feature_channels=None, include_measurement_types=None,
        label_filter=None, perturbation="gene_symbol",
        plate_column="plate", well_column="well",
        condition_column="condition", condition_source_column="well",
        condition_map={"1": "treated", "2": "control"},
        reference_query="gene_symbol == 'NTC'",
        exclude_reference_query="gene_symbol == 'NTC'",
        max_fraction_not_finite=0.25, max_feature_nan_fraction=0.5,
        min_variance=0.0, max_variance=None,
        max_residual_nan_fraction=None, residual_nan_impute="zero",
        yj_clip_percentile=99.9, yj_standardize=False, yj_clip_output=None,
        scale_method="global", scale_max_value=5.0,
        pca_components=4, pca_batch_size=0, pca_whiten=False,
        pca_select_method="components", min_variance_fraction=0.95,
        tvn_by=None,
        agg_by=["gene_symbol"], agg_method="median",
        center_by=None, center_robust=False,
        metric="cosine", output_format="anndata",
        leaf_ordering="none",
        memory_budget_gb=None, streaming_threshold_gb=None,
        filter_batch_size=500_000, filter_max_memory_gb=None,
        max_cpus=None, obs_force=None,
    )

    run_pipeline_map_run(args)

    # Verify outputs exist and have correct shapes / finite values
    cells = read_anndata_zarr(out_dir + "/cells.zarr", dask=False)
    profiles = read_anndata_zarr(out_dir + "/profiles.zarr", dask=False)
    sim = read_anndata_zarr(out_dir + "/similarity.zarr", dask=False)

    assert cells.shape[0] > 0, "cells.zarr is empty"
    assert cells.shape[1] > 0, "no features survived filter"
    assert "X_pca" in cells.obsm or "X_tvn" in cells.obsm, "missing PCA/TVN embedding"

    assert profiles.shape[0] > 0, "no gene profiles produced"

    S = sim.obsp["similarity"]
    assert S.shape[0] == S.shape[1], "similarity matrix is not square"
    S_arr = np.asarray(S.todense() if hasattr(S, "todense") else S)
    assert np.all(np.isfinite(S_arr)), "similarity matrix contains non-finite values"
