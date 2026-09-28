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
from scallops.features.util import pandas_to_anndata
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
        images={"A1-102-IF": pheno_img, "A1-102-FISH": pheno_img},
        labels={
            "A1-102-IF-mask": phenotype_mask,
            "A1-102-IF-tile": phenotype_tile,
            "A1-102-FISH-mask": phenotype_mask,
            "A1-102-FISH-tile": phenotype_tile,
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
        "reference_phenotype_time": "IF",
        "phenotype_url": str(pheno_dir.absolute()),
        "phenotype_nuclei_features": ["intensity_0", "intensity_1"],
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

    merge_sbs_metadata_df = pd.read_parquet(
        output / "merge-sbs-metadata" / "A1-102.parquet"
    )
    assert len(merge_sbs_metadata_df) > len(
        merge_sbs_metadata_df.query("~barcode_count_0.isna()")
    )
    bbox_cols = sorted(
        merge_sbs_metadata_df.columns[
            merge_sbs_metadata_df.columns.str.contains("PearsonBox")
        ].tolist()
    )
    assert bbox_cols == [
        "Nuclei_Correlation_PearsonBox_FISH_IF",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS1",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS2",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS3",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS4",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS5",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS6",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS7",
        "Nuclei_Correlation_PearsonBox_ISS0_ISS8",
        "Nuclei_Correlation_PearsonBox_ISS_PHENO",
    ]

    for col in [
        "Nuclei_AreaShape_Area",
        "Cells_AreaShape_Area",
    ]:
        assert col in merge_sbs_metadata_df.columns
    for col in [
        "Nuclei_Intensity_MeanIntensity_Channel0",
        "Nuclei_Intensity_MeanIntensity_Channel1",
        "Cells_Intensity_MeanIntensity_Channel0",
    ]:
        assert col not in merge_sbs_metadata_df.columns
    merge_features_df = pd.read_parquet(output / "merge-features" / "A1-102.parquet")
    for col in [
        "Nuclei_AreaShape_Area",
        "Cells_AreaShape_Area",
        "Nuclei_Intensity_MeanIntensity_Channel0",
        "Nuclei_Intensity_MeanIntensity_Channel1",
        "Cells_Intensity_MeanIntensity_Channel0",
    ]:
        assert col in merge_features_df.columns
    assert len(
        merge_features_df.query("~Nuclei_Intensity_MeanIntensity_Channel0.isna()")
    ) == len(merge_sbs_metadata_df.query("~barcode_count_0.isna()"))


@pytest.mark.cli_e2e
def test_pert_map_wdl(tmp_path):
    rng = np.random.default_rng(0)
    n_labels = 1200
    n_features = 24
    # `enrichment` requires at least 10 genes per set, so use two sets of 12.
    genes = [f"GENE{i}" for i in range(24)]
    set_genes = [genes[:12], genes[12:]]
    feature_names = [f"Cells_Intensity_feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(
        data=dict(
            label=np.arange(n_labels),
            gene_symbol=rng.choice(["NTC", "INTERGENIC"] + genes, n_labels),
            plate=rng.choice(["p1", "p2"], n_labels),
            well=rng.choice(["w1", "w2"], n_labels),
            Nuclei_AreaShape_Center_Y=rng.uniform(0, 1000, n_labels),
            Nuclei_AreaShape_Center_X=rng.uniform(0, 1000, n_labels),
            **{name: rng.normal(size=n_labels) for name in feature_names},
        )
    )
    dataset_path = tmp_path / "dataset.zarr"
    pandas_to_anndata(df, feature_names).write_zarr(
        str(dataset_path), convert_strings_to_categoricals=False
    )

    gmt_path = tmp_path / "sets.gmt"
    gmt_path.write_text(
        "".join(
            "\t".join([f"C{i}", "na"] + set_genes[i]) + "\n"
            for i in range(len(set_genes))
        )
    )

    corum_path = tmp_path / "corum.txt.gz"
    pd.DataFrame(
        dict(
            complex_name=[f"C{i}" for i in range(len(set_genes))],
            subunits_gene_name=[";".join(s) for s in set_genes],
        )
    ).to_csv(corum_path, sep="\t", index=False, compression="gzip")

    output = tmp_path / "out"
    reference_query = "gene_symbol=='NTC' or gene_symbol=='INTERGENIC'"
    components = 5
    input_json = {
        "pert_map_workflow.inputs": [str(dataset_path)],
        "pert_map_workflow.output_directory": str(output),
        "pert_map_workflow.by": ["plate", "well"],
        "pert_map_workflow.aggregate_by": ["gene_symbol"],
        "pert_map_workflow.pca_components": components,
        "pert_map_workflow.filter_n_features": 20,
        "pert_map_workflow.normalize_method": "local-zscore",
        "pert_map_workflow.normalize_neighbors": 10,
        "pert_map_workflow.tvn_reference_query": reference_query,
        "pert_map_workflow.aggregate_center_reference_query": reference_query,
        "pert_map_workflow.recall_ground_truth_corum": [str(corum_path)],
        "pert_map_workflow.enrichment_sets": [str(gmt_path)],
        "pert_map_workflow.docker": "",
        **{
            f"pert_map_workflow.{step}_extra_arguments": "--client none"
            for step in [
                "filter",
                "normalize",
                "pca",
                "tvn",
                "aggregate",
                "similarity_matrix",
                "recall",
                "enrichment",
            ]
        },
    }

    with open(tmp_path / "inputs.json", "wt") as out:
        json.dump(input_json, out)

    cmd = [
        "miniwdl",
        "run",
        "-i",
        str(tmp_path / "inputs.json"),
        "wdl/pert_map_workflow.wdl",
    ]
    env = os.environ.copy()
    env["MINIWDL__SCHEDULER__CONTAINER_BACKEND"] = "miniwdl_test_local"
    env["SCALLOPS_TEST"] = "1"
    check_call(cmd, env=env)

    for name in [
        "filter.zarr",
        "normalized.zarr",
        "pca.zarr",
        "tvn.zarr",
        "agg.zarr",
        "sim.zarr",
    ]:
        assert (output / name).exists(), name

    assert len(pd.read_parquet(output / "recall.parquet")) > 0
    assert len(pd.read_parquet(output / "enrichment.parquet")) > 0
