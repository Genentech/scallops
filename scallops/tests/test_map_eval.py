"""Tests for map_eval recall functions and reference database readers.

Covers:
- pairwise_benchmark: correct recall computation, empty-pair short-circuit
- read_string: score normalisation, threshold filtering, missing columns
- read_reactome_fi: two-column and three-column formats
- gmt_to_gene_sets: round-trip from read_gmt output
- run_pipeline_map_recall: multi-source CLI integration (CORUM + GMT + STRING file)
"""

import io
import textwrap
import argparse

import anndata
import numpy as np
import pandas as pd
import pytest

from scallops.features.map_eval import (
    gmt_to_gene_sets,
    pairwise_benchmark,
    read_reactome_fi,
    read_string,
)


# ---------------------------------------------------------------------------
# Shared fixture: small similarity matrix
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_adata():
    """3×3 cosine similarity matrix: genes A, B, C.

    A-B are "true positives" (high similarity = 0.9).
    A-C and B-C are low similarity (0.1).
    """
    labels = ["gene_A", "gene_B", "gene_C"]
    X = np.array(
        [[1.0, 0.9, 0.1],
         [0.9, 1.0, 0.1],
         [0.1, 0.1, 1.0]],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=labels)
    return anndata.AnnData(X=X, obs=obs, var=obs.copy())


# ---------------------------------------------------------------------------
# pairwise_benchmark
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_pairwise_benchmark_known_pair_has_high_recall(sim_adata):
    """The known-high pair (A-B, sim=0.9) should be above the median null.

    With only 3 upper-triangle values [0.9, 0.1, 0.1], the strict two-sided
    threshold (0.05, 0.95) would require the left percentile rank ≥ 0.95 or
    right rank ≤ 0.05 — unachievable with 3 data points.  Use a one-sided
    threshold of 0.5 (above median) which is always achievable.
    """
    pairs = pd.DataFrame({"gene_a": ["gene_A"], "gene_b": ["gene_B"]})
    result = pairwise_benchmark(
        sim_adata,
        pairs,
        recall_thresholds=[0.5],  # one-sided: fraction above the median
        min_pairs=1,
    )
    assert len(result) > 0
    # A-B (sim=0.9) is the highest value → recall at 0.5 threshold should be 1
    assert result["recall"].iloc[0] == pytest.approx(1.0)


@pytest.mark.features
def test_pairwise_benchmark_returns_n_pairs_column(sim_adata):
    pairs = pd.DataFrame({"gene_a": ["gene_A"], "gene_b": ["gene_B"]})
    result = pairwise_benchmark(sim_adata, pairs, min_pairs=1)
    assert "n_pairs" in result.columns
    assert result["n_pairs"].iloc[0] == 1


@pytest.mark.features
def test_pairwise_benchmark_below_min_pairs_returns_empty(sim_adata):
    """When fewer valid pairs than min_pairs, an empty DataFrame is returned."""
    pairs = pd.DataFrame({"gene_a": ["gene_A"], "gene_b": ["gene_B"]})
    result = pairwise_benchmark(sim_adata, pairs, min_pairs=50)
    assert len(result) == 0


@pytest.mark.features
def test_pairwise_benchmark_ignores_genes_not_in_matrix(sim_adata):
    """Pairs whose genes are absent from the matrix are silently dropped."""
    pairs = pd.DataFrame(
        {"gene_a": ["gene_A", "UNKNOWN"], "gene_b": ["gene_B", "gene_A"]}
    )
    result = pairwise_benchmark(sim_adata, pairs, min_pairs=1)
    # Only A-B survives; UNKNOWN is filtered
    assert result["n_pairs"].iloc[0] == 1


@pytest.mark.features
def test_pairwise_benchmark_bidirectional(sim_adata):
    """Specifying B-A instead of A-B should give the same result."""
    pairs_ab = pd.DataFrame({"gene_a": ["gene_A"], "gene_b": ["gene_B"]})
    pairs_ba = pd.DataFrame({"gene_a": ["gene_B"], "gene_b": ["gene_A"]})
    r_ab = pairwise_benchmark(sim_adata, pairs_ab, min_pairs=1)
    r_ba = pairwise_benchmark(sim_adata, pairs_ba, min_pairs=1)
    np.testing.assert_allclose(r_ab["recall"].values, r_ba["recall"].values)


# ---------------------------------------------------------------------------
# read_string
# ---------------------------------------------------------------------------


def _make_string_tsv(scores_0_1000=True):
    """Create an in-memory STRING TSV with two interactions."""
    if scores_0_1000:
        content = (
            "preferredName_A\tpreferredName_B\tscore\n"
            "gene_A\tgene_B\t800\n"
            "gene_A\tgene_C\t300\n"
        )
    else:
        # 0–1 scale (REST API format)
        content = (
            "preferredName_A\tpreferredName_B\tscore\n"
            "gene_A\tgene_B\t0.8\n"
            "gene_A\tgene_C\t0.3\n"
        )
    return content


@pytest.mark.features
def test_read_string_filters_by_threshold(tmp_path):
    """Interactions below score_threshold must be excluded."""
    path = tmp_path / "string.tsv"
    path.write_text(_make_string_tsv(scores_0_1000=True))
    df = read_string(str(path), score_threshold=400)
    # Only gene_A–gene_B (score=800) survives threshold 400
    assert len(df) == 1
    assert set(df["gene_a"].tolist() + df["gene_b"].tolist()) == {"gene_A", "gene_B"}


@pytest.mark.features
def test_read_string_normalises_0_1_scores(tmp_path):
    """Files with scores in [0, 1] are rescaled to 0–1000 before thresholding."""
    path = tmp_path / "string_norm.tsv"
    path.write_text(_make_string_tsv(scores_0_1000=False))
    df = read_string(str(path), score_threshold=400)
    # 0.8 → 800 ≥ 400 → kept; 0.3 → 300 < 400 → dropped
    assert len(df) == 1


@pytest.mark.features
def test_read_string_custom_column_names(tmp_path):
    """Alternative column names should be mapped correctly."""
    content = "protein_a\tprotein_b\tcombined_score\ngene_X\tgene_Y\t600\n"
    path = tmp_path / "string_custom.tsv"
    path.write_text(content)
    df = read_string(
        str(path),
        score_threshold=400,
        gene_a_col="protein_a",
        gene_b_col="protein_b",
        score_col="combined_score",
    )
    assert len(df) == 1
    assert "gene_a" in df.columns and "gene_b" in df.columns


@pytest.mark.features
def test_read_string_missing_gene_columns_raises(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("col1\tcol2\ncrap\tdata\n")
    with pytest.raises(ValueError, match="gene columns"):
        read_string(str(path))


@pytest.mark.features
def test_read_string_no_score_column(tmp_path):
    """A two-column file without a score column should still load."""
    content = "preferredName_A\tpreferredName_B\ngene_A\tgene_B\n"
    path = tmp_path / "string_noscore.tsv"
    path.write_text(content)
    df = read_string(str(path))
    assert len(df) == 1
    assert "gene_a" in df.columns


# ---------------------------------------------------------------------------
# read_reactome_fi
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_read_reactome_fi_two_column(tmp_path):
    content = "Gene1\tGene2\nTP53\tMDM2\nBRCA1\tBRCA2\n"
    path = tmp_path / "reactome.tsv"
    path.write_text(content)
    df = read_reactome_fi(str(path))
    assert len(df) == 2
    assert "gene_a" in df.columns and "gene_b" in df.columns


@pytest.mark.features
def test_read_reactome_fi_three_column(tmp_path):
    content = "Gene1\tAnnotation\tGene2\nTP53\tactivate|inhibit\tMDM2\n"
    path = tmp_path / "reactome3.tsv"
    path.write_text(content)
    df = read_reactome_fi(str(path))
    assert len(df) == 1
    assert df["gene_a"].iloc[0] == "TP53"


@pytest.mark.features
def test_read_reactome_fi_annotation_filter(tmp_path):
    content = "Gene1\tAnnotation\tGene2\nTP53\tactivate|inhibit\tMDM2\nBRCA1\tbind\tBRCA2\n"
    path = tmp_path / "reactome_filt.tsv"
    path.write_text(content)
    df = read_reactome_fi(str(path), interaction_types={"bind"})
    assert len(df) == 1
    assert df["gene_a"].iloc[0] == "BRCA1"


# ---------------------------------------------------------------------------
# gmt_to_gene_sets
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_gmt_to_gene_sets_roundtrip(tmp_path):
    """gmt_to_gene_sets must produce the same gene lists as read_gmt parsed."""
    from scallops.features.map_eval import read_gmt

    content = "SET_A\tdescription\tTP53\tMDM2\nSET_B\tdescription2\tBRCA1\tBRCA2\n"
    path = tmp_path / "test.gmt"
    path.write_text(content)

    gmt_df = read_gmt(str(path))
    gene_sets = gmt_to_gene_sets(gmt_df)

    assert set(gene_sets.keys()) == {"SET_A", "SET_B"}
    assert set(gene_sets["SET_A"]) == {"TP53", "MDM2"}
    assert set(gene_sets["SET_B"]) == {"BRCA1", "BRCA2"}


# ---------------------------------------------------------------------------
# run_pipeline_map_recall — multi-source CLI integration
# ---------------------------------------------------------------------------


def _ns(**kw):
    d = dict(force=True, no_version=True,
             corum=None, gmt=None, string=None, string_fetch=False,
             string_threshold=400, string_species=9606, string_network_type="full",
             reactome=None, min_genes=1, min_pairs=1)
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.mark.features
def test_map_recall_corum_source(sim_adata, tmp_path):
    """map-recall with --corum should produce rows with method=set_benchmark."""
    from scallops.cli.map_build import run_pipeline_map_recall
    from scallops.io import read_anndata_zarr

    # Write similarity matrix zarr
    sim_zarr = str(tmp_path / "sim.zarr")
    sim_adata.write_zarr(sim_zarr)

    # CORUM file: gene_A and gene_B in the same complex
    corum = str(tmp_path / "corum.txt")
    with open(corum, "w") as f:
        f.write("complex_name\tsubunits_gene_name\n")
        f.write("ComplexAB\tgene_A;gene_B\n")

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(_ns(input=[sim_zarr], output=out, corum=[corum]))

    result = pd.read_parquet(out)
    assert "source" in result.columns
    assert "method" in result.columns
    assert result["method"].unique().tolist() == ["set_benchmark"]


@pytest.mark.features
def test_map_recall_gmt_source(sim_adata, tmp_path):
    """map-recall with --gmt should produce rows with method=set_benchmark."""
    from scallops.cli.map_build import run_pipeline_map_recall

    sim_zarr = str(tmp_path / "sim.zarr")
    sim_adata.write_zarr(sim_zarr)

    gmt = str(tmp_path / "sets.gmt")
    with open(gmt, "w") as f:
        f.write("SET_AB\tdescription\tgene_A\tgene_B\n")

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(_ns(input=[sim_zarr], output=out, gmt=[gmt]))

    result = pd.read_parquet(out)
    assert result["method"].iloc[0] == "set_benchmark"
    assert result["source"].iloc[0] == "sets.gmt"


@pytest.mark.features
def test_map_recall_string_file_source(sim_adata, tmp_path):
    """map-recall with --string should produce rows with method=pairwise_recall."""
    from scallops.cli.map_build import run_pipeline_map_recall

    sim_zarr = str(tmp_path / "sim.zarr")
    sim_adata.write_zarr(sim_zarr)

    string_file = str(tmp_path / "string.tsv")
    with open(string_file, "w") as f:
        f.write("preferredName_A\tpreferredName_B\tscore\n")
        f.write("gene_A\tgene_B\t800\n")

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(
        _ns(input=[sim_zarr], output=out, string=[string_file])
    )

    result = pd.read_parquet(out)
    assert result["method"].iloc[0] == "pairwise_recall"
    assert "n_pairs" in result.columns
    assert "recall" in result.columns


@pytest.mark.features
def test_map_recall_reactome_source(sim_adata, tmp_path):
    """map-recall with --reactome should produce rows with method=pairwise_recall."""
    from scallops.cli.map_build import run_pipeline_map_recall

    sim_zarr = str(tmp_path / "sim.zarr")
    sim_adata.write_zarr(sim_zarr)

    reactome = str(tmp_path / "reactome.tsv")
    with open(reactome, "w") as f:
        f.write("Gene1\tGene2\n")
        f.write("gene_A\tgene_B\n")

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(_ns(input=[sim_zarr], output=out, reactome=[reactome]))

    result = pd.read_parquet(out)
    assert result["method"].iloc[0] == "pairwise_recall"


@pytest.mark.features
def test_map_recall_multi_source(sim_adata, tmp_path):
    """Multiple sources produce separate rows each with the correct source name."""
    from scallops.cli.map_build import run_pipeline_map_recall

    sim_zarr = str(tmp_path / "sim.zarr")
    sim_adata.write_zarr(sim_zarr)

    corum = str(tmp_path / "corum.txt")
    with open(corum, "w") as f:
        f.write("complex_name\tsubunits_gene_name\n")
        f.write("ComplexAB\tgene_A;gene_B\n")

    string_file = str(tmp_path / "string.tsv")
    with open(string_file, "w") as f:
        f.write("preferredName_A\tpreferredName_B\tscore\n")
        f.write("gene_A\tgene_B\t800\n")

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(
        _ns(input=[sim_zarr], output=out, corum=[corum], string=[string_file])
    )

    result = pd.read_parquet(out)
    methods = set(result["method"])
    assert "set_benchmark" in methods
    assert "pairwise_recall" in methods

    sources = set(result["source"])
    assert "corum.txt" in sources
    assert "string.tsv" in sources


@pytest.mark.features
def test_map_recall_no_reference_produces_empty(sim_adata, tmp_path):
    """Providing no reference source produces an empty Parquet file without error."""
    from scallops.cli.map_build import run_pipeline_map_recall

    sim_zarr = str(tmp_path / "sim.zarr")
    sim_adata.write_zarr(sim_zarr)

    out = str(tmp_path / "recall.parquet")
    run_pipeline_map_recall(_ns(input=[sim_zarr], output=out))

    result = pd.read_parquet(out)
    assert len(result) == 0
