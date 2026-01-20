import numpy as np
import pandas as pd
import pytest
from pandas.core.dtypes.common import is_object_dtype

from scallops.cli.pooled_if_sbs import _merged_to_matrix
from scallops.reads import merge_sbs_phenotype


@pytest.fixture
def df_known_good():
    df = (
        pd.read_csv("scallops/tests/data/process_fig4/combined.csv")
        .rename(
            {
                "cell": "label",
                "cell_barcode_0": "barcode_0",
                "cell_barcode_count_0": "barcode_count_0",
                "cell_barcode_1": "barcode_1",
                "cell_barcode_count_1": "barcode_count_1",
            },
            axis=1,
        )
        .drop(["tile", "well"], axis=1)
    )  # (2599, 28)

    df = df.set_index("label")
    df = df[~df.index.duplicated()]
    return df


@pytest.mark.utils
def test_merge_sbs_phenotype_df(tmp_path, df_known_good):
    df_cells = (
        pd.read_csv("scallops/tests/data/process_fig4/10X_A1_Tile-102.cells.csv")
        .rename(
            {
                "cell": "label",
                "cell_barcode_0": "barcode_0",
                "cell_barcode_count_0": "barcode_count_0",
                "cell_barcode_1": "barcode_1",
                "cell_barcode_count_1": "barcode_count_1",
            },
            axis=1,
        )
        .set_index("label")
        .drop(["tile", "well"], axis=1)
    )
    df_phenotype = (
        pd.read_csv("scallops/tests/data/process_fig4/10X_A1_Tile-102.phenotype.csv")
        .rename(
            {"cell": "label"},
            axis=1,
        )
        .set_index("label")
        .drop(["tile", "well"], axis=1)
    )  # (2538, 16)
    df_barcode = pd.read_csv("scallops/tests/data/experimentC/barcodes.csv")

    test_merged_df = merge_sbs_phenotype(
        df_labels=df_cells,
        df_phenotype=df_phenotype,
        df_barcode=df_barcode,
        sbs_cycles=[1, 2, 3, 4, 5, 7, 8, 9, 10],
    )
    assert len(test_merged_df) == len(df_known_good)
    test_merged_df = test_merged_df.loc[df_known_good.index]
    pd.testing.assert_frame_equal(
        test_merged_df[df_known_good.columns],
        df_known_good,
    )


@pytest.mark.utils
def test_merge_sbs_phenotype_matrix(tmp_path, df_known_good):
    df_cells = (
        pd.read_csv("scallops/tests/data/process_fig4/10X_A1_Tile-102.cells.csv")
        .rename(
            {
                "cell": "label",
                "cell_barcode_0": "barcode_0",
                "cell_barcode_count_0": "barcode_count_0",
                "cell_barcode_1": "barcode_1",
                "cell_barcode_count_1": "barcode_count_1",
            },
            axis=1,
        )
        .set_index("label")
        .drop(["tile", "well"], axis=1)
    )
    df_phenotype = (
        pd.read_csv("scallops/tests/data/process_fig4/10X_A1_Tile-102.phenotype.csv")
        .rename(
            {"cell": "label"},
            axis=1,
        )
        .set_index("label")
        .drop(["tile", "well"], axis=1)
    )  # (2538, 16)
    df_barcode = pd.read_csv("scallops/tests/data/experimentC/barcodes.csv")
    metadata_columns = ["i_cell", "j_cell", "i_nucleus", "j_nucleus"]
    feature_names = df_phenotype.drop(metadata_columns, axis=1).columns.tolist()
    # test anndata output
    cells_path = str(tmp_path / "cell.parquet")
    pheno_path = str(tmp_path / "pheno.parquet")
    df_cells.to_parquet(cells_path)
    df_phenotype.to_parquet(pheno_path)

    data = _merged_to_matrix(
        merged_df=merge_sbs_phenotype(
            df_labels=pd.read_parquet(cells_path),
            df_phenotype=pd.read_parquet(pheno_path, columns=metadata_columns),
            df_barcode=df_barcode,
            sbs_cycles=[1, 2, 3, 4, 5, 7, 8, 9, 10],
        ),
        phenotype_paths=[pheno_path],
        feature_names=feature_names,
        feature_columns=[feature_names],
        name="test",
        format="anndata",
    )
    data_df = data.to_df()  # converts index to str
    data_df = data_df.join(data.obs)
    data_df = data_df.set_index("label")

    merged_df = merge_sbs_phenotype(
        df_labels=pd.read_parquet(cells_path),
        df_phenotype=pd.read_parquet(pheno_path),
        df_barcode=df_barcode,
        sbs_cycles=[1, 2, 3, 4, 5, 7, 8, 9, 10],
    )
    data_df = data_df.loc[merged_df.index]
    merged_df["duplicate_prefix"] = merged_df["duplicate_prefix"].astype(str)
    merged_df["duplicate_prefix_1"] = merged_df["duplicate_prefix_1"].astype(str)
    for c in merged_df.columns:
        if is_object_dtype(merged_df[c]):
            merged_df[c] = merged_df[c].replace({np.nan: ""}).replace({"nan": ""})
            data_df[c] = data_df[c].replace({"nan": ""}).replace({"None": ""})

        np.testing.assert_array_equal(data_df[c].values, merged_df[c].values, err_msg=c)
