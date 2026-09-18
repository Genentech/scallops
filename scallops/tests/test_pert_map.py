from subprocess import check_call

import pytest

from scallops.features.util import pandas_to_anndata


@pytest.mark.parametrize("input_format", ["zarr", "parquet"])
@pytest.mark.features
def test_map_filter(tmp_path, test_feature_table, input_format):
    dataset_path = str(tmp_path / f"dataset_test.{input_format}")
    output_path = tmp_path / "labels.zarr"

    if input_format == "parquet":
        test_feature_table.to_parquet(dataset_path)
    else:
        d = pandas_to_anndata(
            test_feature_table,
            ["Cells_Intensity_feature_1", "Cells_Intensity_feature_2"],
        )
        d.write_zarr(dataset_path, convert_strings_to_categoricals=False)
    cmd = [
        "scallops",
        "pert-map",
        "filter",
        "--input",
        dataset_path,
        "--output",
        str(output_path),
        "--min-feature-variance",
        "-1",
    ]
    check_call(cmd)
    assert output_path.exists()


@pytest.mark.parametrize("outut_format", ["zarr", "parquet"])
@pytest.mark.features
def test_map_norm(tmp_path, test_feature_table, outut_format):
    dataset_path = str(tmp_path / "dataset_test.zarr")
    output_path = tmp_path / f"dataset_test.{outut_format}"

    d = pandas_to_anndata(
        test_feature_table, ["Cells_Intensity_feature_1", "Cells_Intensity_feature_2"]
    )
    d.write_zarr(dataset_path, convert_strings_to_categoricals=False)
    cmd = [
        "scallops",
        "pert-map",
        "normalize",
        "--input",
        dataset_path,
        "--output",
        str(output_path),
    ]
    check_call(cmd)
    assert output_path.exists()
