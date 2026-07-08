"""Tests for _read_data with parquet inputs.

Covers the new behaviour introduced to handle large CellProfiler parquet files
from S3 efficiently:
- Feature columns (Cells/Nuclei/Cytoplasm prefix) are loaded lazily as dask arrays.
- Obs columns are loaded eagerly but with column pruning (only simple types).
- list<double> and other nested-type columns are excluded from obs.
- The parquet row index (__index_level_0__) is restored as obs.index, renamed
  to 'label' if that name is not already taken.
"""

import tempfile
from pathlib import Path

import anndata
import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scallops.features.util import _read_data


# ---------------------------------------------------------------------------
# Helpers: write synthetic parquet files
# ---------------------------------------------------------------------------


def _write_parquet(path: str, n_rows: int = 10, row_group_size: int = 5) -> dict:
    """Write a synthetic CellProfiler-like parquet and return expected values."""
    np.random.seed(42)
    rng_data = {
        # Feature columns (CellProfiler compartment naming)
        "Nuclei_Intensity_feat0": np.random.randn(n_rows).astype("float64"),
        "Cells_AreaShape_feat1":  np.random.randn(n_rows).astype("float64"),
        "Cytoplasm_Texture_feat2": np.random.randn(n_rows).astype("float64"),
        # Obs columns (simple types)
        "gene_symbol": [f"g_{i % 3}" for i in range(n_rows)],
        "plate":       ["plateA"] * n_rows,
        "well":        [str(i % 3 + 1) for i in range(n_rows)],
        "barcode_count":   np.arange(n_rows, dtype="float64"),
        "barcode_count_0": (np.arange(n_rows) * 0.9).astype("float64"),
        # Nested-type column that should be EXCLUDED from obs
        "barcode_Q_0": [[0.1, 0.2]] * n_rows,   # list<double>
    }
    df = pd.DataFrame(rng_data)
    # Save with a named index (simulates cell label)
    df.index = pd.RangeIndex(n_rows, name="cell_label")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, row_group_size=row_group_size)
    return {
        "feat_cols": ["Nuclei_Intensity_feat0", "Cells_AreaShape_feat1",
                      "Cytoplasm_Texture_feat2"],
        "obs_cols":  ["gene_symbol", "plate", "well",
                      "barcode_count", "barcode_count_0"],
        "excluded":  ["barcode_Q_0"],
        "n_rows":    n_rows,
    }


def _write_parquet_no_index_name(path: str, n_rows: int = 6) -> None:
    """Write parquet whose pandas index has no name → obs.index gets name 'label'."""
    df = pd.DataFrame({
        "Cells_Intensity_X": np.random.randn(n_rows),
        "gene_symbol": ["NTC"] * n_rows,
    })
    # Default RangeIndex has name=None → our code renames to 'label'
    assert df.index.name is None
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def _write_parquet_label_collision(path: str, n_rows: int = 6) -> None:
    """Write parquet where 'label' is already a data column."""
    df = pd.DataFrame({
        "Cells_Intensity_X": np.random.randn(n_rows),
        "label": [f"cell_{i}" for i in range(n_rows)],   # 'label' already exists
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.features
def test_read_data_parquet_feature_matrix_is_dask(tmp_path):
    """Feature columns must be loaded lazily (dask array, not numpy)."""
    p = str(tmp_path / "data.parquet")
    _write_parquet(p)
    data = _read_data([p])
    assert isinstance(data.X, da.Array), (
        "Feature matrix must be a dask array so it is never fully materialised"
    )


@pytest.mark.features
def test_read_data_parquet_correct_shape(tmp_path):
    """AnnData shape must match the parquet content."""
    p = str(tmp_path / "data.parquet")
    info = _write_parquet(p, n_rows=10)
    data = _read_data([p])
    assert data.shape[0] == info["n_rows"]
    assert data.shape[1] == len(info["feat_cols"])


@pytest.mark.features
def test_read_data_parquet_feature_var_names(tmp_path):
    """var.index must contain the CellProfiler feature column names."""
    p = str(tmp_path / "data.parquet")
    info = _write_parquet(p)
    data = _read_data([p])
    assert set(data.var.index) == set(info["feat_cols"])


@pytest.mark.features
def test_read_data_parquet_obs_contains_simple_columns(tmp_path):
    """Simple-typed obs columns (str, float) must be in data.obs."""
    p = str(tmp_path / "data.parquet")
    info = _write_parquet(p)
    data = _read_data([p])
    for col in info["obs_cols"]:
        assert col in data.obs.columns, f"Expected obs column '{col}' to be present"


@pytest.mark.features
def test_read_data_parquet_list_columns_excluded_from_obs(tmp_path):
    """list<double> columns must be excluded from obs (too slow to read from S3)."""
    p = str(tmp_path / "data.parquet")
    info = _write_parquet(p)
    data = _read_data([p])
    for col in info["excluded"]:
        assert col not in data.obs.columns, (
            f"list-typed column '{col}' must not appear in obs"
        )


@pytest.mark.features
def test_read_data_parquet_named_index_preserved(tmp_path):
    """A named parquet index (e.g. 'cell_label') must become obs.index.name.

    pyarrow restores a named RangeIndex from the pandas metadata stored in the
    parquet footer — so the original name is preserved, not overwritten.
    """
    p = str(tmp_path / "data.parquet")
    _write_parquet(p, n_rows=8)
    data = _read_data([p])
    assert data.obs.index.name == "cell_label", (
        f"Expected obs.index.name='cell_label', got '{data.obs.index.name}'"
    )
    assert data.obs.index.dtype == object   # must be strings


@pytest.mark.features
def test_read_data_parquet_unnamed_index_renamed_to_label(tmp_path):
    """An unnamed parquet index (__index_level_0__) must be renamed to 'label'."""
    p = str(tmp_path / "data.parquet")
    _write_parquet_no_index_name(p)
    data = _read_data([p])
    assert data.obs.index.name == "label", (
        f"Expected obs.index.name='label', got '{data.obs.index.name}'"
    )


@pytest.mark.features
def test_read_data_parquet_label_collision_uses_fallback(tmp_path):
    """When 'label' is already a data column, a fallback name is used."""
    p = str(tmp_path / "data.parquet")
    _write_parquet_label_collision(p)
    data = _read_data([p])
    # The index must be renamed to something other than 'label'
    assert data.obs.index.name != "__index_level_0__"
    assert data.obs.index.name is not None
    # The original 'label' data column must still be in obs
    assert "label" in data.obs.columns


@pytest.mark.features
def test_read_data_parquet_correct_feature_values(tmp_path):
    """Feature values in the dask array must match what was written to parquet."""
    p = str(tmp_path / "data.parquet")
    _write_parquet(p, n_rows=6)
    data = _read_data([p])
    X = np.asarray(data.X.compute(), dtype=np.float64)

    # Re-read with plain pandas to get ground truth
    df = pd.read_parquet(p)
    feat_cols = [c for c in df.columns if c.split("_")[0] in {"Cells", "Nuclei", "Cytoplasm"}]
    expected = df[feat_cols].values.astype(np.float64)

    # Columns may be in different order — align by name
    col_order = [data.var.index.get_loc(c) for c in feat_cols]
    X_aligned = X[:, col_order]
    np.testing.assert_allclose(X_aligned, expected, rtol=1e-5)


@pytest.mark.features
def test_read_data_parquet_multiple_files_concat(tmp_path):
    """Passing multiple parquet files must concatenate them correctly."""
    p1 = str(tmp_path / "a.parquet")
    p2 = str(tmp_path / "b.parquet")
    _write_parquet(p1, n_rows=6)
    _write_parquet(p2, n_rows=4)
    data = _read_data([p1, p2])
    assert data.shape[0] == 10
    assert data.shape[1] == 3   # same 3 feature columns


@pytest.mark.features
def test_read_data_parquet_row_groups(tmp_path):
    """The dask array must have one chunk per parquet row group."""
    p = str(tmp_path / "rg.parquet")
    _write_parquet(p, n_rows=10, row_group_size=3)   # → 4 row groups: 3+3+3+1
    data = _read_data([p])
    assert isinstance(data.X, da.Array)
    # With dask.delayed per-row-group, chunks match exactly: (3, 3, 3, 1)
    assert len(data.X.chunks[0]) == 4
    assert sum(data.X.chunks[0]) == 10
