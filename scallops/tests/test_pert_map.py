from subprocess import check_call

import pytest

from scallops.features.util import pandas_to_anndata


@pytest.mark.parametrize("input_format", ["zarr", "parquet"])
@pytest.mark.features
def test_map_filter(tmp_path, test_feature_table, input_format):
    dataset_path = str(tmp_path / f"dataset_test.{input_format}")
    label_path = tmp_path / "labels.parquet"
    feature_path = tmp_path / "features.parquet"
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
        "--dataset",
        dataset_path,
        "--output-label-ids",
        str(label_path),
        "--output-feature-ids",
        str(feature_path),
    ]
    check_call(cmd)
    assert feature_path.exists()
    assert label_path.exists()


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
        "--dataset",
        dataset_path,
        "--output",
        str(output_path),
    ]
    check_call(cmd)
    assert output_path.exists()
