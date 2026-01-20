import glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scallops import Experiment
from scallops.io import read_experiment, read_image
from scallops.registration.crosscorrelation import align_images
from scallops.tests.test_features import diff_pheno
from scallops.tests.test_reads import diff_reads
from scallops.zarr_io import _write_zarr_image, open_ome_zarr, read_ome_zarr_array

data_path = Path("scallops") / "tests" / "data"


@pytest.fixture(params=[False, True])
def group_by_tile(request):
    return request.param


def diff_combined(df_test: pd.DataFrame):
    # assigned cell barcodes can differ due to ties
    df_test = df_test.rename(
        {
            "Cells_AreaShape_Center_Y": "cell_centroid-0",
            "Cells_AreaShape_Center_X": "cell_centroid-1",
            "Cells_AreaShape_Area": "cell_area",
            "Nuclei_AreaShape_Center_Y": "nuclei_centroid-0",
            "Nuclei_AreaShape_Center_X": "nuclei_centroid-1",
            "Nuclei_Intensity_MaxIntensity_Channel0": "nuclei_max_0",
            "Nuclei_Intensity_MeanIntensity_Channel0": "nuclei_mean_0",
            "Nuclei_Intensity_MedianIntensity_Channel0": "nuclei_median_0",
            "Nuclei_Intensity_MaxIntensity_Channel1": "nuclei_max_1",
            "Nuclei_Intensity_MeanIntensity_Channel1": "nuclei_mean_1",
            "Nuclei_Intensity_MedianIntensity_Channel1": "nuclei_median_1",
            "Nuclei_Correlation_Pearson_Channel0_Channel1": "nuclei_corr_0_1",
            "Nuclei_AreaShape_Area": "nuclei_area",
        },
        axis=1,
    )

    df_combined_known_good = pd.read_csv(
        str(data_path.joinpath("process_fig4", "combined.csv"))
    )
    df_combined_known_good = df_combined_known_good.replace(np.nan, None)
    df_combined_known_good = df_combined_known_good.rename(
        {
            "cell": "label",
            "area_cell": "cell_area",
            "i_cell": "cell_centroid-0",
            "j_cell": "cell_centroid-1",
            "i_nucleus": "nuclei_centroid-0",
            "j_nucleus": "nuclei_centroid-1",
            "area_nucleus": "nuclei_area",
            "dapi_gfp_corr_nucleus": "nuclei_corr_0_1",
            "dapi_max_nucleus": "nuclei_max_0",
            "dapi_mean_nucleus": "nuclei_mean_0",
            "dapi_median_nucleus": "nuclei_median_0",
            "gfp_max_nucleus": "nuclei_max_1",
            "gfp_mean_nucleus": "nuclei_mean_1",
            "gfp_median_nucleus": "nuclei_median_1",
            "cell_barcode_0": "barcode_0",
            "cell_barcode_count_0": "barcode_count_0",
            "cell_barcode_1": "barcode_1",
            "cell_barcode_count_1": "barcode_count_1",
        },
        axis=1,
    )

    df_combined_known_good = df_combined_known_good.drop(
        ["tile", "well", "peak"], axis=1
    )
    drop_cols = [x for x in ("well", "tile") if x in df_test.columns]
    if len(drop_cols) > 0:
        df_test = df_test.drop(drop_cols, axis=1)
    # assigned cell barcodes can differ due to ties
    df_test["label"] = df_test["label"].astype(df_combined_known_good["label"].dtype)
    index_cols = ["label"]

    df_test = (
        df_test.drop_duplicates(index_cols)[df_combined_known_good.columns]
        .set_index(index_cols)
        .query("barcode_count_0 != barcode_count_1")
    )
    df_combined_known_good = (
        df_combined_known_good.drop_duplicates(index_cols)
        .set_index(index_cols)
        .query("barcode_count_0 != barcode_count_1")
        .drop([c for c in df_combined_known_good.columns if c.endswith("_1")], axis=1)
    )

    df_test = df_test.replace(np.nan, None)
    pd.testing.assert_frame_equal(
        df_test.loc[df_combined_known_good.index][df_combined_known_good.columns],
        df_combined_known_good,
        check_dtype=False,
    )


def diff_sbs_output_images(
    known_good_suffix, test_suffix, test_image_path, group_by_tile
):
    known_good_image = (
        (
            read_image(
                str(
                    data_path.joinpath(
                        "process_fig4", f"10X_A1_Tile-102.{known_good_suffix}.tif"
                    )
                )
            )
            .transpose(*("z", "c", "t", "y", "x"))
            .rename({"z": "t", "t": "z"})
        )
        .squeeze()
        .data
    )  # ops swaps z and t in saved tif
    key = "A1-102" if group_by_tile else "A1"
    key = key + test_suffix
    test_path = os.path.join(test_image_path, key)

    test_image = read_ome_zarr_array(open_ome_zarr(test_path, mode="r")).squeeze().data
    # ops saved dapi channel
    if (
        known_good_image.ndim == 4
        and known_good_image.shape[1] - 1 == test_image.shape[1]
    ):
        known_good_image = known_good_image[:, [1, 2, 3, 4], :, :]
    if known_good_suffix == "std":
        np.testing.assert_equal(
            known_good_image, test_image.astype("float32"), err_msg=known_good_suffix
        )
    else:
        np.testing.assert_equal(known_good_image, test_image, err_msg=known_good_suffix)


@pytest.mark.cli_spot
def test_spot_detect_invalid_t(tmp_path):
    # spot-detect
    spot_detect_zarr = str(tmp_path / "spot_detect.zarr")
    spot_detection_args = [
        "scallops",
        "pooled-sbs",
        "spot-detect",
        "--images",
        str(data_path.joinpath("experimentC", "input")),
        "--channel",
        "1",
        "2",
        "3",
        "4",
        "--output=" + spot_detect_zarr,
        "--subset=A1-102",
        "--groupby",
        "well",
        "tile",
        "--image-pattern=10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--cycles",
        "1",
        "4",
        "15",
        "--dask-cluster",
        '{"n_workers":1, "threads_per_worker":1}',
    ]

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(spot_detection_args)


@pytest.mark.cli_spot
def test_spot_detect_subset_t(tmp_path):
    # spot-detect
    spot_detect_zarr = str(tmp_path / "spot_detect.zarr")
    input_experiment = read_experiment(
        str(data_path.joinpath("experimentC", "input")),
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
    )
    input_zarr = str(tmp_path / "spot_detect_input.zarr")
    input_experiment.save(input_zarr)
    spot_detection_args = [
        "scallops",
        "pooled-sbs",
        "spot-detect",
        "--images",
        input_zarr,
        "--channel",
        "1",
        "2",
        "3",
        "4",
        "--output=" + spot_detect_zarr,
        "--subset=A1-102",
        "--cycles",
        "0",
        "3",
        "--dask-cluster",
        '{"n_workers":1, "threads_per_worker":1}',
    ]

    subprocess.check_call(spot_detection_args)
    exp = read_experiment(spot_detect_zarr)
    np.testing.assert_array_equal(exp.images["A1-102-max"].t, [1, 4])


@pytest.mark.cli_segment
def test_reads_nuclei_labels(tmp_path):
    # segment
    segment_zarr_path = str(tmp_path / "segment.zarr")
    seg_args = [
        "scallops",
        "segment",
        "nuclei",
        "--images",
        str(data_path.joinpath("experimentC", "input")),
        "--output=" + segment_zarr_path,
        "--subset=A1-102",
        "--groupby",
        "well",
        "tile",
        "--image-pattern=10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
    ]
    subprocess.check_call(seg_args)

    spot_detect_zarr = str(tmp_path / "spot_detect.zarr")
    spot_detection_args = [
        "scallops",
        "pooled-sbs",
        "spot-detect",
        "--images",
        str(data_path.joinpath("experimentC", "input")),
        "--channel",
        "1",
        "2",
        "3",
        "4",
        "--output=" + spot_detect_zarr,
        "--subset=A1-102",
        "--groupby",
        "well",
        "tile",
        "--image-pattern=10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--cycles",
        "1",
        "4",
        "--dask-cluster",
        '{"n_workers":1, "threads_per_worker":1}',
    ]

    subprocess.check_call(spot_detection_args)
    reads1 = str(tmp_path / "reads1")
    subprocess.check_call(
        [
            "scallops",
            "pooled-sbs",
            "reads",
            "--spots",
            spot_detect_zarr,
            "--labels",
            segment_zarr_path,
            "--barcodes",
            str(data_path.joinpath("experimentC", "barcodes.csv")),
            "--label-name",
            "nuclei",
            "--output",
            reads1,
            "--save-bases",
            "--threshold-peaks",
            "0.5",
            "--dask-cluster",
            '{"n_workers":1, "threads_per_worker":1}',
        ]
    )
    reads2 = str(tmp_path / "reads2")
    subprocess.check_call(
        [
            "scallops",
            "pooled-sbs",
            "reads",
            "--spots",
            spot_detect_zarr,
            "--labels",
            segment_zarr_path,
            "--barcodes",
            str(data_path.joinpath("experimentC", "barcodes.csv")),
            "--label-name",
            "nuclei",
            "--output",
            reads2,
            "--save-bases",
            "--threshold-peaks",
            "peak > 0.5 & peak > 0.1",
            "--dask-cluster",
            '{"n_workers":1, "threads_per_worker":1}',
        ]
    )
    for x in ["bases", "reads", "labels"]:
        out1 = pd.read_parquet(os.path.join(reads1, x))
        out2 = pd.read_parquet(os.path.join(reads2, x))
        pd.testing.assert_frame_equal(out1, out2)


@pytest.mark.cli_e2e
def test_e2e_cli(tmp_path, group_by_tile):
    env = os.environ.copy()
    env["SCALLOPS_IMAGE_SCALE"] = "1"
    env["SCALLOPS_BARCODES_TO_LABELS_NO_FILTER"] = "1"
    env["SCALLOPS_BASE_ORDER"] = "A,C,G,T"

    # use known good segment nuclei + cells
    segment_register_zarr_path = str(tmp_path / "segment-register.zarr")
    nuclei_labels_known_good = (
        read_image(
            str(data_path.joinpath("process_fig4", "10X_A1_Tile-102.nuclei.tif"))
        )
        .squeeze()
        .data
    )
    cell_labels_known_good = (
        read_image(str(data_path.joinpath("process_fig4", "10X_A1_Tile-102.cells.tif")))
        .squeeze()
        .data
    )
    exp = Experiment()
    exp.labels["A1-102-cell" if group_by_tile else "A1-cell"] = cell_labels_known_good
    exp.labels["A1-102-nuclei" if group_by_tile else "A1-nuclei"] = (
        nuclei_labels_known_good
    )
    exp.save(segment_register_zarr_path)

    # cross-correlation
    registration_args = [
        "scallops",
        "registration",
        "cross-correlation",
        "--images",
        str(data_path.joinpath("experimentC", "input")),
        "--across-t-channel=0",
        "--within-t-channel",
        "1",
        "2",
        "3",
        "4",
        "--output=" + segment_register_zarr_path,
    ]

    if group_by_tile:
        registration_args.append("--subset=A1-102")
        registration_args.append("--groupby")
        registration_args.append("well")
        registration_args.append("tile")
        registration_args.append(
            "--image-pattern=10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif"
        )
    else:
        registration_args.append("--groupby=well")
        registration_args.append(
            "--image-pattern=10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-102.{datatype}.tif"
        )
    subprocess.check_call(registration_args, env=env)

    # spot-detect
    spot_detect_zarr = str(tmp_path / "spot_detect.zarr")
    spot_detection_args = [
        "scallops",
        "pooled-sbs",
        "spot-detect",
        "--images",
        segment_register_zarr_path,
        "--channel",
        "1",
        "2",
        "3",
        "4",
        "--save",
        "log",
        "std",
        "--output=" + spot_detect_zarr,
        "--dask-cluster",
        '{"n_workers":1, "threads_per_worker":1}',
    ]
    subprocess.check_call(spot_detection_args, env=env)

    # reads
    reads_output = str(tmp_path / "reads")

    reads_args = [
        "scallops",
        "pooled-sbs",
        "reads",
        "--spots",
        spot_detect_zarr,
        "--labels",
        segment_register_zarr_path,
        "--barcodes",
        str(data_path.joinpath("experimentC", "barcodes.csv")),
        "--threshold-peaks",
        "50",
        "--threshold-peaks-crosstalk",
        "50",
        "--output=" + reads_output,
        "--label-name",
        "cell",
        "--all-labels",
        "--save-bases",
        "--dask-cluster",
        '{"n_workers":1, "threads_per_worker":1}',
    ]
    subprocess.check_call(reads_args, env=env)

    # align phenotype images to sbs t=0, dapi
    phenotype_image = read_image(
        str(
            data_path.joinpath(
                "experimentC",
                "10X_c0-DAPI-p65ab",
                "10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif",
            )
        )
    ).isel(t=0, z=0)
    sbs_image = read_image(
        str(
            data_path.joinpath(
                "experimentC",
                "input",
                "10X_c1-SBS-1",
                "10X_c1-SBS-1_A1_Tile-102.sbs.tif",
            )
        )
    ).isel(t=0, z=0)
    phenotype_image_aligned = align_images(sbs_image, phenotype_image)
    phenotype_aligned_zarr = str(tmp_path / "phenotype-aligned.zarr")
    image_key = "A1-102" if group_by_tile else "A1"
    _write_zarr_image(
        image_key, open_ome_zarr(phenotype_aligned_zarr), phenotype_image_aligned
    )
    # compute features using segmentation results from sbs
    shutil.copytree(
        os.path.join(segment_register_zarr_path, "labels"),
        os.path.join(phenotype_aligned_zarr, "labels"),
        dirs_exist_ok=True,
    )

    features_path = str(tmp_path / "features")
    features_args = [
        "scallops",
        "features",
        "--no-normalize",
        "--images",
        phenotype_aligned_zarr,
        "--labels",
        segment_register_zarr_path,
        "--output=" + features_path,
        "--features-cell",
    ]

    features_args += ["sizeshape"]
    features_args += ["--features-nuclei"]
    features_args += [
        "intensity_0",
        "intensity_1",
        "colocalization_0_1",
    ]
    subprocess.check_call(features_args)
    pheno_df = []
    for compartment in ["cell", "nuclei"]:
        for dataset in ["features", "objects"]:
            pattern = os.path.join(features_path, compartment, f"*-{dataset}.parquet")
            matches = glob.glob(pattern)
            assert len(matches) == 1, pattern
            pheno_df.append(pd.read_parquet(matches[0]))
    pheno_df = pd.concat(pheno_df, join="inner", axis=1)
    pheno_df = pheno_df.sort_index()
    pheno_df.index.name = "cell"
    diff_pheno(pheno_df)

    # merge
    merge_path = str(tmp_path / "merge")
    merge_args = [
        "scallops",
        "pooled-sbs",
        "merge",
        "--sbs",
        os.path.join(reads_output, "labels"),
        "--phenotype",
        os.path.join(features_path, "cell"),
        os.path.join(features_path, "nuclei"),
        "--barcodes",
        str(data_path.joinpath("experimentC", "barcodes.csv")),
        "--output=" + merge_path,
        "--join-pheno",
        "inner",
        "--format",
        "parquet",
    ]

    subprocess.check_call(merge_args)
    # our pipeline includes all bases and all reads in output

    diff_reads(
        pd.read_parquet(os.path.join(reads_output, "bases")).query("peak>50"),
        pd.read_parquet(os.path.join(reads_output, "reads")).query("peak>50"),
        pd.read_parquet(os.path.join(reads_output, "labels")),
    )

    df_test = pd.read_parquet(merge_path).sort_index()
    assert df_test.index.duplicated().sum() == 0
    df_test.index.name = "label"
    df_test = df_test.reset_index()
    diff_combined(df_test)

    images_diff = [
        ("aligned", "", os.path.join(segment_register_zarr_path, "images")),
        ("phenotype_aligned", "", os.path.join(phenotype_aligned_zarr, "images")),
        ("log", "-log", os.path.join(spot_detect_zarr, "images")),
        ("maxed", "-max", os.path.join(spot_detect_zarr, "images")),
        ("std", "-std", os.path.join(spot_detect_zarr, "images")),
    ]  # known-good suffix, test suffix

    for suffix1, suffix2, image_path in images_diff:
        diff_sbs_output_images(suffix1, suffix2, image_path, group_by_tile)
