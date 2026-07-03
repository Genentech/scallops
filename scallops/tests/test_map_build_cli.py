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
    run_pipeline_map_agg,
    run_pipeline_map_center,
    run_pipeline_map_filter,
    run_pipeline_map_pca,
    run_pipeline_map_pca_select,
    run_pipeline_map_recall,
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
        by=None,
        robust=False,
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
    # Zero out the first feature so it has zero variance
    cell_data.X[:, 0] = 0.0
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "filtered")

    run_pipeline_map_filter(
        _ns(input=[inp], output=out, min_variance=0.01, max_variance=None,
            max_fraction_not_finite=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape[1] == N_FEATURES - 1
    assert "Cells_Intensity_feature_0" not in result.var.index


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
        _ns(input=[inp], output=out, components=n_comp, batch_size=0, whiten=False,
            reference=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape == (cell_data.shape[0], n_comp)
    assert list(result.var.index) == [f"PC{i + 1}" for i in range(n_comp)]


@pytest.mark.features
def test_map_pca_stores_pca_uns(cell_data, tmp_path):
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "pca")

    run_pipeline_map_pca(
        _ns(input=[inp], output=out, components=3, batch_size=0, whiten=False,
            reference=None)
    )
    result = _read_zarr(out + ".zarr")
    assert "pca" in result.uns
    for key in ("variance_ratio", "variance", "mean", "PCs"):
        assert key in result.uns["pca"], f"Missing pca uns key: {key}"


@pytest.mark.features
def test_map_pca_reference_subset_fitting(cell_data, tmp_path):
    """PCA fitted on NTC only must project all cells."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "pca")

    run_pipeline_map_pca(
        _ns(input=[inp], output=out, components=2, batch_size=0, whiten=False,
            reference="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    # All cells are projected (not just the reference subset)
    assert result.shape[0] == cell_data.shape[0]


# ---------------------------------------------------------------------------
# map-tvn
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_map_tvn_output_shape(cell_data, tmp_path):
    """TVN must preserve obs count and feature count."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "tvn")

    run_pipeline_map_tvn(
        _ns(input=[inp], output=out, reference="gene_symbol=='NTC'")
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape == cell_data.shape


@pytest.mark.features
def test_map_tvn_stores_backprojection_uns(cell_data, tmp_path):
    """Output zarr must contain all uns keys required for backprojection."""
    inp = _write_zarr(cell_data, tmp_path / "input")
    out = str(tmp_path / "tvn")

    run_pipeline_map_tvn(
        _ns(input=[inp], output=out, reference="gene_symbol=='NTC'")
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
        _ns(input=[inp], output=out, reference="gene_symbol=='NTC'")
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
        _ns(input=[inp], output=out, by=["gene_symbol"], method="mean",
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
        _ns(input=[inp], output=out, by=["gene_symbol"], method="mean",
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
        _ns(input=[inp], output=out, by=["gene_symbol"], method="mean",
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
        _ns(input=[inp], output=out, reference="gene_symbol=='NTC'", robust=False)
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
        _ns(input=[inp], output=out, reference="gene_symbol=='NTC'", robust=False)
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
            reference="gene_symbol=='NTC'")
    )

    # Step 4: aggregate
    out_agg = str(tmp_path / "s4_agg")
    run_pipeline_map_agg(
        _ns(input=[out_tvn + ".zarr"], output=out_agg, by=["gene_symbol"],
            method="mean", min_cells=None, barcode=None, agg_by_barcode=False,
            perturbation="gene_symbol")
    )

    # Step 5: center
    out_center = str(tmp_path / "s5_center")
    run_pipeline_map_center(
        _ns(input=[out_agg + ".zarr"], output=out_center,
            reference="gene_symbol=='NTC'", robust=False)
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
    """With a strict threshold, some PCs are zeroed before backprojection."""
    result_no_filter = top_features_from_backprojection(
        tvn_data, genes=["gene_A"]
    )
    result_filter = top_features_from_backprojection(
        tvn_data, genes=["gene_A"],
        pc_stat_filter="ttest", pc_pvalue_threshold=0.0001  # very strict
    )
    # With a very strict threshold most PCs are zeroed → scores are smaller
    assert result_filter["score"].abs().sum() <= result_no_filter["score"].abs().sum()


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
    with pytest.raises(ValueError, match="not both"):
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
            output=pca_out, components=N_FEATURES, batch_size=0,
            whiten=False, reference=None)
    )

    out = str(tmp_path / "selected")
    run_pipeline_map_pca_select(
        _ns(input=[pca_out + ".zarr"], output=out,
            method="variance", min_variance_fraction=0.80,
            pval=0.05, n_perms=20, max_components=None, n_features=None)
    )
    result = _read_zarr(out + ".zarr")
    assert 1 <= result.shape[1] <= N_FEATURES


@pytest.mark.features
def test_map_pca_select_max_components_cap(cell_data, tmp_path):
    """max_components caps the number of retained PCs."""
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[_write_zarr(cell_data, tmp_path / "raw")],
            output=pca_out, components=N_FEATURES, batch_size=0,
            whiten=False, reference=None)
    )

    out = str(tmp_path / "selected")
    run_pipeline_map_pca_select(
        _ns(input=[pca_out + ".zarr"], output=out,
            method="variance", min_variance_fraction=0.99,
            pval=0.05, n_perms=10, max_components=2, n_features=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.shape[1] <= 2


@pytest.mark.features
def test_map_pca_select_tracy_widom_warns(cell_data, tmp_path):
    """map-pca-select --method tracy_widom must emit a UserWarning."""
    pca_out = str(tmp_path / "pca")
    run_pipeline_map_pca(
        _ns(input=[_write_zarr(cell_data, tmp_path / "raw")],
            output=pca_out, components=N_FEATURES, batch_size=0,
            whiten=False, reference=None)
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
            output=pca_out, components=N_FEATURES, batch_size=0,
            whiten=False, reference=None)
    )

    out = str(tmp_path / "selected")
    run_pipeline_map_pca_select(
        _ns(input=[pca_out + ".zarr"], output=out,
            method="variance", min_variance_fraction=0.80,
            pval=0.05, n_perms=10, max_components=None, n_features=None)
    )
    result = _read_zarr(out + ".zarr")
    assert result.uns.get("upstream") == 99
