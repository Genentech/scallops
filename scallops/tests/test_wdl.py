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


@pytest.mark.cli_e2e
def test_ops_wdl(tmp_path):
    sbs_dir = tmp_path / "sbs"
    pheno_dir = tmp_path / "pheno"
    output = tmp_path / "out"
    sbs_dir.mkdir()
    pheno_dir.mkdir()
    output.mkdir()
    for p in glob.glob("scallops/tests/data/experimentC/input/*/*Tile-102*"):
        add_physical_size(p, str(sbs_dir / os.path.basename(p)))

    pheno_img = read_image(
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    )
    pheno_img.attrs["physical_pixel_sizes"] = (1, 1)
    phenotype_mask = np.ones(
        (pheno_img.sizes["y"], pheno_img.sizes["x"]), dtype=np.uint8
    )
    phenotype_mask[10, 10] = 1
    phenotype_tile = np.ones(
        (pheno_img.sizes["y"], pheno_img.sizes["x"]), dtype=np.uint16
    )
    phenotype_tile[10, 10] = 2
    exp = Experiment(
        images={"A1-102-1": pheno_img, "A1-102-2": pheno_img},
        labels={
            "A1-102-1-mask": phenotype_mask,
            "A1-102-1-tile": phenotype_tile,
            "A1-102-2-mask": phenotype_mask,
            "A1-102-2-tile": phenotype_tile,
        },
    )
    exp.save(str(pheno_dir))

    input_json = {
        "model_dir": "",
        "iss_url": str(sbs_dir.absolute()),
        "iss_image_pattern": "{mag}X_c{t}-{experiment}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "output_directory": str(output.absolute()),
        "iss_registration_extra_arguments": "--no-landmarks",
        "pheno_to_iss_registration_extra_arguments": "--no-landmarks",
        "pheno_registration_extra_arguments": "--no-landmarks",
        "phenotype_cyto_channel": [1],
        "phenotype_dapi_channel": 0,
        "phenotype_url": str(pheno_dir.absolute()),
        "phenotype_nuclei_features": ["intensity_0", "intensity_1"],
        "features_label_filter": " barcode_Q_mean_0/barcode_Q_mean > 0.5",
        # 2 batches
        "phenotype_cell_features": ["intensity_0"],
        # "phenotype_cytosol_features": ["mean_0 area"], # no cytosol features
        "phenotype_image_pattern": "{well}-{tile}-{t}",
        "groupby": ["well", "tile"],
        "reads_threshold_peaks": "0",
        "reads_threshold_peaks_crosstalk": "20",
        "barcodes": os.path.abspath("scallops/tests/data/experimentC/barcodes.csv"),
        "mark_stitch_boundary_cells": False,
        "reads_labels": "cell",
        "merge_extra_arguments": "--format parquet",
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
    df = pd.read_parquet(output / "merge" / "A1-102.parquet")

    for col in [
        "Nuclei_AreaShape_Area",
        "Cells_AreaShape_Area",
        "Nuclei_Intensity_MeanIntensity_Channel0",
        "Nuclei_Intensity_MeanIntensity_Channel1",
        "Cells_Intensity_MeanIntensity_Channel0",
    ]:
        assert col in df.columns
    assert len(df) > 0
