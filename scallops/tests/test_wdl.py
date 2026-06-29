import glob
import json
import os.path
from subprocess import check_call

import numpy as np
import ome_types
import pandas as pd
import pytest
import xarray as xr

from scallops import Experiment
from scallops.cli.util import _list_images_wdl
from scallops.io import read_image, save_ome_tiff
from scallops.tests.test_stitch import _write_image_with_position


def add_physical_size(input_path, output_path):
    img = read_image(input_path)
    if img is None:
        raise ValueError(f"Input image {input_path} cannot be read")
    save_ome_tiff(img.values, uri=output_path)
    img = read_image(output_path)
    if isinstance(img.attrs["processed"], str):
        img.attrs["processed"] = ome_types.from_xml(img.attrs["processed"])
    img.attrs["processed"].images[0].pixels.physical_size_x = 1
    img.attrs["processed"].images[0].pixels.physical_size_y = 1
    save_ome_tiff(img.values, uri=output_path, ome_xml=img.attrs["processed"].to_xml())


@pytest.mark.parametrize("reference_time", ["IF", None])
@pytest.mark.cli_e2e
def test_list_images_wdl(reference_time, tmp_path, monkeypatch):
    (tmp_path / "plate1-A1-IF").touch()
    (tmp_path / "plate1-A1-FISH").touch()
    monkeypatch.chdir(tmp_path)
    _list_images_wdl(
        image_pattern="{plate}-{well}-{t}",
        reference_time=reference_time,
        urls=[str(tmp_path)],
        groupby=["plate", "well"],
        subset=None,
        batch_size_str=None,
        save_group_size=False,
        expected_cycles_str=None,
    )
    groups = pd.read_csv(tmp_path / "groups.txt", header=None)[0].values
    np.testing.assert_array_equal(groups, ["plate1-A1"])
    groupby = pd.read_csv(tmp_path / "groupby.txt", header=None)[0].values
    np.testing.assert_array_equal(groupby, ["plate", "well"])
    groupby_pattern = pd.read_csv(tmp_path / "groupby_pattern.txt", header=None)[
        0
    ].values
    np.testing.assert_array_equal(groupby_pattern, ["{plate}-{well}"])

    times = pd.read_csv(tmp_path / "t.txt", header=None)[0].values
    np.testing.assert_array_equal(times, ["FISH", "IF"])
    groups_with_times = pd.read_csv(tmp_path / "groups_with_t.txt", header=None)[
        0
    ].values
    if reference_time == "IF":
        np.testing.assert_array_equal(groups_with_times, ["plate1-A1-IF"])
    else:
        np.testing.assert_array_equal(groups_with_times, ["plate1-A1-FISH"])


@pytest.mark.cli_e2e
def test_stitch_wdl_z_stack(tmp_path):
    input_path = tmp_path / "input"
    # top-left, top-right, bottom-left, bottom-right
    coords = [(0, 0), (0, 100), (100, 0), (100, 100)]
    constant_img_val = 10000
    for i in range(len(coords)):
        c = coords[i]

        for z_index in range(2):
            if z_index == 0:  # constant large value will be selected for max projection
                img = np.zeros((1, 100, 100), dtype=np.uint16)
                img[...] = constant_img_val
            else:  # will be selected for best focus
                img = np.arange(100 * 100, dtype=np.uint16).reshape(1, 100, 100)
            _write_image_with_position(
                input_path / f"test-tile{i}-z{z_index}.zarr",
                xr.DataArray(img, dims=["c", "y", "x"]),
                c[0],
                c[1],
            )

    input_json = {
        "urls": [str(input_path)],
        "image_pattern": "{well}-tile{tile}-z{z}.zarr",
        "z_index": "focus",
        "stitch_radial_correction_k": "none",
        "output_directory": str(tmp_path / "out"),
        "docker": "",
    }

    with open(tmp_path / "inputs.json", "wt") as out:
        json.dump(input_json, out)

    cmd = [
        "miniwdl",
        "run",
        "-i",
        str(tmp_path / "inputs.json"),
        "wdl/stitch_workflow.wdl",
    ]
    env = os.environ.copy()
    env["MINIWDL__SCHEDULER__CONTAINER_BACKEND"] = "miniwdl_test_local"
    env["SCALLOPS_TEST"] = "1"
    check_call(cmd, env=env)


@pytest.mark.cli_e2e
def test_stitch_wdl(tmp_path):
    # top-left, top-right, bottom-left, bottom-right
    coords = [(0, 0), (0.5, 1024 - 50.5), (1000, 0.5), (999, 1024 - 49.5)]
    input_path = tmp_path / "input"

    for i in range(len(coords)):
        c = coords[i]
        img = np.ones((2, 1024, 1024), dtype=np.uint16)
        img[...] = i + 1
        _write_image_with_position(
            input_path / f"test-{i}.zarr",
            xr.DataArray(img, dims=["c", "y", "x"]),
            c[0],
            c[1],
        )
    output_directory = tmp_path / "out"
    input_json = {
        "urls": [str(input_path)],
        "image_pattern": "{well}-{skip}.zarr",
        "output_directory": str(output_directory),
        "channel_names": ["a", "b"],
        "docker": "",
    }

    with open(tmp_path / "inputs.json", "wt") as out:
        json.dump(input_json, out)

    cmd = [
        "miniwdl",
        "run",
        "-i",
        str(tmp_path / "inputs.json"),
        "wdl/stitch_workflow.wdl",
    ]
    env = os.environ.copy()
    env["MINIWDL__SCHEDULER__CONTAINER_BACKEND"] = "miniwdl_test_local"
    env["SCALLOPS_TEST"] = "1"
    check_call(cmd, env=env)
    image = read_image(tmp_path / "out" / "stitch" / "stitch.zarr" / "images" / "test")
    np.testing.assert_array_equal(image.coords["c"].values, ["a", "b"])


@pytest.mark.parametrize("phenotype_rounds", [1])
@pytest.mark.cli_e2e
def test_ops_wdl(phenotype_rounds, tmp_path):
    sbs_dir = tmp_path / "sbs"
    output = tmp_path / "out"
    pheno_dir = tmp_path / "pheno.zarr"
    sbs_dir.mkdir()
    output.mkdir()
    for p in glob.glob("scallops/tests/data/experimentC/input/*/*Tile-102*"):
        cycles = os.path.basename(p).split("_")[1]
        cycles = cycles.split("-")[0]
        dest = f"plateA-A1-{cycles[1:]}.tif"
        add_physical_size(p, str(sbs_dir / dest))

    pheno_img = read_image(
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    )
    pheno_img.attrs["physical_pixel_sizes"] = (1, 1)
    phenotype_mask = np.ones(
        (pheno_img.sizes["y"], pheno_img.sizes["x"]), dtype=np.uint8
    )
    phenotype_mask[10, 10] = 0
    phenotype_tile = np.ones(
        (pheno_img.sizes["y"], pheno_img.sizes["x"]), dtype=np.uint16
    )
    phenotype_tile[10, 10] = 2
    reference_phenotype_time = "IF"

    phenotype_cell_features = {"IF": ["intensity_0"]}
    phenotype_image_pattern = "{plate}-{well}-{t}"
    if phenotype_rounds == 1:
        phenotype_nuclei_features = {
            "IF": ["intensity_0", "intensity_1"],
        }
        Experiment(
            images={"plateA-A1-IF": pheno_img},
            labels={
                "plateA-A1-IF-mask": phenotype_mask,
                "plateA-A1-IF-tile": phenotype_tile,
            },
        ).save(pheno_dir)
    elif phenotype_rounds == 2:
        phenotype_nuclei_features = {
            "IF": ["intensity_0", "intensity_1"],
            "FISH": ["intensity_0", "intensity_1"],
        }
        Experiment(
            images={"plateA-A1-IF": pheno_img, "plateA-A1-FISH": pheno_img},
            labels={
                "plateA-A1-IF-mask": phenotype_mask,
                "plateA-A1-IF-tile": phenotype_tile,
                "plateA-A1-FISH-mask": phenotype_mask,
                "plateA-A1-FISH-tile": phenotype_tile,
            },
        ).save(pheno_dir)
    else:
        phenotype_nuclei_features = {
            "": ["intensity_0", "intensity_1"],
        }
        reference_phenotype_time = None
        phenotype_image_pattern = "{plate}-{well}"
        phenotype_cell_features = {"": ["intensity_0"]}
        Experiment(
            images={"plateA-A1": pheno_img},
            labels={
                "plateA-A1-mask": phenotype_mask,
                "plateA-A1-tile": phenotype_tile,
            },
        ).save(pheno_dir)

    input_json = {
        "model_dir": "",
        "iss_url": str(sbs_dir.absolute()),
        "iss_image_pattern": "{plate}-{well}-{t}.tif",
        "phenotype_image_pattern": phenotype_image_pattern,
        "output_directory": str(output.absolute()),
        "iss_registration_extra_arguments": "--no-landmarks",
        "pheno_to_iss_registration_extra_arguments": "--no-landmarks",
        "pheno_registration_extra_arguments": "--no-landmarks",
        "phenotype_cyto_channel": [1],
        "reference_phenotype_time": reference_phenotype_time,
        "phenotype_url": str(pheno_dir.absolute()),
        "phenotype_nuclei_features": phenotype_nuclei_features,
        # 2 batches
        "phenotype_cell_features": phenotype_cell_features,
        "reads_threshold_peaks": "0",
        "reads_threshold_peaks_crosstalk": "20",
        "barcodes": os.path.abspath("scallops/tests/data/experimentC/barcodes.csv"),
        "reads_labels": "cell",
        "docker": "",
    }

    with open(tmp_path / "inputs.json", "wt") as out:
        json.dump(input_json, out)

    cmd = [
        "miniwdl",
        "run",
        "-i",
        str(tmp_path / "inputs.json"),
        "wdl/ops_workflow.wdl",
    ]
    env = os.environ.copy()
    env["MINIWDL__SCHEDULER__CONTAINER_BACKEND"] = "miniwdl_test_local"
    env["SCALLOPS_TEST"] = "1"
    check_call(cmd, env=env)

    merge_sbs_metadata_df = pd.read_parquet(
        output / "merge-sbs-metadata" / "plateA-A1.parquet"
    )
    assert len(merge_sbs_metadata_df) > len(
        merge_sbs_metadata_df.query("~barcode_count_0.isna()")
    )
    assert (
        len(
            merge_sbs_metadata_df.columns[
                merge_sbs_metadata_df.columns.str.contains("iss_to_iss_qc")
            ]
        )
        > 0
    )
    assert (
        len(
            merge_sbs_metadata_df.columns[
                merge_sbs_metadata_df.columns.str.contains("Intensity")
            ]
        )
        == 0
    )

    merge_features_df = pd.read_parquet(output / "merge-features" / "plateA-A1.parquet")
    assert (
        len(
            merge_features_df.columns[
                merge_features_df.columns.str.contains("iss_to_iss_qc")
            ]
        )
        > 0
    )
    assert (
        len(
            merge_features_df.columns[
                merge_features_df.columns.str.contains("Intensity")
            ]
        )
        > 0
    )
    intensity_column = merge_features_df.columns[
        merge_features_df.columns.str.contains("Intensity")
    ][0]
    assert len(merge_features_df.query(f"~{intensity_column}.isna()")) == len(
        merge_sbs_metadata_df.query("~barcode_count_0.isna()")
    )
