import glob
import os
from pathlib import Path
from subprocess import check_call

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import shapely
import zarr
from scipy.stats import spearmanr
from skimage.measure import regionprops

from scallops.experiment.elements import Experiment
from scallops.features.bounding_box import bounding_box_to_edge_distance
from scallops.features.constants import (
    _cp_features_mask,
    _cp_features_multichannel,
    _cp_features_single_channel,
    _label_name_to_prefix,
    _other_features_multichannel,
    _other_features_single_channel,
)
from scallops.features.find_objects import find_objects
from scallops.features.generate import _create_funcs, label_features
from scallops.features.spots import spot_count
from scallops.features.texture import pftas
from scallops.io import read_image, to_label_crops

__this__ = Path(__file__).resolve()
__tests__ = __this__.parent
__data__ = __tests__.joinpath("data")


@pytest.mark.features
def test_to_label_crops(tmp_path, array_A1_102_cells, array_A1_102_alnpheno):
    label_image = da.from_array(array_A1_102_cells.squeeze().data)
    intensity_image = (
        array_A1_102_alnpheno.transpose(*("z", "c", "t", "y", "x"))
        .rename({"z": "t", "t": "z"})
        .isel(t=0, z=0)
    ).data  # ops swaps z and t in saved tif
    output_dir_dask = str(tmp_path / "crops-dask")
    output_dir_zarr = str(tmp_path / "crops-zarr")

    intensity_image = da.from_array(intensity_image).rechunk((-1, -1, 50, 50))
    objects_df = find_objects(label_image).compute()
    crop_size = (30, 30)

    result_df = to_label_crops(
        intensity_image=intensity_image,
        label_image=label_image,
        objects_df=objects_df.query("index==2603|index==17"),
        crop_size=crop_size,
        output_dir=output_dir_dask,
    )
    # 17 should be filtered b/c on tile edge
    assert len(result_df) == 1 and result_df.index.values[0] == 2603

    group = zarr.group()
    intensity_image_zarr = group.create_array(name="image", shape=intensity_image.shape)
    intensity_image_zarr[:] = intensity_image.compute()

    label_image_zarr = group.create_array(name="label", shape=label_image.shape)
    label_image_zarr[:] = label_image.compute()

    to_label_crops(
        intensity_image=intensity_image_zarr,
        label_image=label_image_zarr,
        objects_df=objects_df.query("index==2603"),
        crop_size=crop_size,
        output_dir=output_dir_zarr,
    )
    img_dask = read_image(os.path.join(output_dir_dask, "2603.tiff")).values.squeeze()
    img_zarr = read_image(os.path.join(output_dir_zarr, "2603.tiff")).values.squeeze()
    np.testing.assert_array_equal(img_dask, img_zarr)
    # centroid: 1007.136364  579.090909
    slice_2603 = intensity_image[
        ..., slice(1007 - 15, 1007 + 15), slice(579 - 15, 579 + 15)
    ].compute()
    slice_2603_labels = label_image[
        ..., slice(1007 - 15, 1007 + 15), slice(579 - 15, 579 + 15)
    ].compute()
    slice_2603 = slice_2603 * (slice_2603_labels == 2603)
    assert slice_2603.shape == (2, 30, 30)
    np.testing.assert_array_equal(img_dask, slice_2603, strict=True)


def _label_features(
    intensity_image, label_image, features, label_filter=None, normalize=True
):
    objects_df = find_objects(label_image)
    if isinstance(objects_df, dd.DataFrame):
        objects_df = objects_df.compute()
    if label_filter is not None:
        objects_df = objects_df[objects_df.index.isin(label_filter)]
    features_df = label_features(
        objects_df=objects_df,
        label_image=label_image,
        intensity_image=intensity_image,
        features=features,
        normalize=normalize,
    )
    if isinstance(features_df, dd.DataFrame):
        features_df = features_df.compute()

    return objects_df.join(features_df) if features_df is not None else objects_df


@pytest.mark.features
def test_features_dask(array_A1_102_cells, array_A1_102_pheno):
    label_image = array_A1_102_cells.squeeze().data
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]
    intensity_image = (
        (
            array_A1_102_pheno.transpose(*("z", "c", "t", "y", "x"))
            .rename({"z": "t", "t": "z"})
            .isel(t=0, z=0)
        )
        .transpose(*("y", "x", "c"))
        .data
    )

    region_props_features = [
        "area",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "euler_number",
        "perimeter",
        "solidity",
    ]
    cp_features = [
        "Area",
        "MajorAxisLength",
        "MinorAxisLength",
        "Eccentricity",
        "EulerNumber",
        "Perimeter",
        "Solidity",
    ]

    features = []
    for f in _cp_features_single_channel.keys():
        features.append(f"{f}_*")
    for f in _other_features_single_channel.keys():
        features.append(f"{f}_*")
    for f in _cp_features_multichannel.keys():
        features.append(f"{f}_*_*")
    for f in _other_features_multichannel.keys():
        features.append(f"{f}_*_*")
    for f in _cp_features_mask.keys():
        features.append(f)
    try:
        import ufish  # noqa: F401
    except ImportError:
        features.remove("spots_*")
    objects_df = find_objects(label_image)

    test_df = (
        label_features(
            objects_df=objects_df,
            intensity_image=da.from_array(intensity_image, chunks=(100, 100) + (-1,)),
            label_image=da.from_array(label_image, chunks=(100, 100)),
            features=features,
            normalize=False,
            overlap=20,
        )
        .compute()
        .sort_index()
    )

    np.testing.assert_array_equal(unique_labels, test_df.index.values)

    test_df_no_chunking = (
        label_features(
            objects_df=objects_df,
            intensity_image=da.from_array(intensity_image, chunks=(-1, -1) + (-1,)),
            label_image=da.from_array(label_image, chunks=(-1, -1)),
            features=features,
            normalize=False,
        )
        .compute()
        .sort_index()
    )
    # this differs due to ties
    location_cols = [
        "Location_MaxIntensity_Y_Channel0",
        "Location_MaxIntensity_X_Channel0",
        "Location_MaxIntensity_Y_Channel1",
        "Location_MaxIntensity_X_Channel1",
    ]
    # columns that are not equal if computed in chunks

    neighbors_cols = test_df.columns[test_df.columns.str.contains("Neighbors")].tolist()
    granularity_cols = test_df.columns[
        test_df.columns.str.contains("Granularity")
    ].tolist()
    zernike_cols = test_df.columns[
        test_df.columns.str.contains("AreaShape_Zernike")
    ].tolist()
    radial_dist_zernike_cols = test_df.columns[
        test_df.columns.str.contains("RadialDistribution_Zernike")
    ].tolist()
    radial_dist_cols = test_df.columns[
        test_df.columns.str.contains("RadialDistribution")
        & ~test_df.columns.str.contains("RadialDistribution_Zernike")
    ].tolist()
    spots_cols = test_df.columns[test_df.columns.str.contains("Spots_Count")].tolist()

    drop_cols = (
        neighbors_cols
        + granularity_cols
        + zernike_cols
        + radial_dist_zernike_cols
        + radial_dist_cols
        + location_cols
        + spots_cols
    )
    for col in spots_cols:
        cor = np.corrcoef(test_df[col], test_df_no_chunking[col])[0, 1]
        assert cor > 0.6, f"{col}, {cor}"
    for col in neighbors_cols:
        cor = np.corrcoef(test_df[col], test_df_no_chunking[col])[0, 1]
        assert cor > 0.98, f"{col}, {cor}"

    for col in zernike_cols:
        cor = np.corrcoef(test_df[col], test_df_no_chunking[col])[0, 1]
        assert cor > 0.85, f"{col}, {cor}"
        diff = np.max(np.abs(test_df[col] - test_df_no_chunking[col]))
        assert diff < 0.025, f"{col}, {diff}"

    for col in granularity_cols:
        # there are some values that are much different so use spearman
        cor = spearmanr(test_df[col], test_df_no_chunking[col])[0]
        assert cor > 0.9, f"{col}, {cor}"
    for col in radial_dist_cols:
        cor = np.corrcoef(test_df[col], test_df_no_chunking[col])[0, 1]
        assert cor > 0.89, f"{col}, {cor}"

    for col in radial_dist_zernike_cols:
        val1 = test_df[col]
        val2 = test_df_no_chunking[col]
        keep = ~np.isnan(val1) & ~np.isnan(val2)
        val1 = val1[keep]
        val2 = val2[keep]
        cor = np.corrcoef(val1, val2)[0, 1]
        assert cor > 0.8, f"{col}, {cor}"
    pd.testing.assert_frame_equal(
        test_df.drop(drop_cols, axis=1), test_df_no_chunking.drop(drop_cols, axis=1)
    )
    test_df = test_df.join(objects_df)

    test_labels = test_df.index.values
    test_centroid0 = test_df["AreaShape_Center_Y"].values
    test_centroid1 = test_df["AreaShape_Center_X"].values
    test_bbox0 = test_df["AreaShape_BoundingBoxMinimum_Y"].values
    test_bbox1 = test_df["AreaShape_BoundingBoxMinimum_X"].values
    test_bbox2 = test_df["AreaShape_BoundingBoxMaximum_Y"].values
    test_bbox3 = test_df["AreaShape_BoundingBoxMaximum_X"].values
    regions = regionprops(label_image=label_image, intensity_image=intensity_image)
    for i in range(len(regions)):
        r = regions[i]
        img = r.image_intensity * np.expand_dims(r.image, -1)
        max_intensity_per_channel = img.max(axis=(0, 1))
        for c in range(img.shape[-1]):
            max_count = (img[..., c] == max_intensity_per_channel[c]).sum()
            if max_count == 1:
                assert (
                    test_df[f"Location_MaxIntensity_Y_Channel{c}"].values[i]
                    == test_df_no_chunking[
                        f"Location_MaxIntensity_Y_Channel{c}"
                    ].values[i]
                )
                assert (
                    test_df[f"Location_MaxIntensity_X_Channel{c}"].values[i]
                    == test_df_no_chunking[
                        f"Location_MaxIntensity_X_Channel{c}"
                    ].values[i]
                )

        assert r.label == test_labels[i], f"{r.label} != {test_labels[i]}"
        assert r.centroid == (test_centroid0[i], test_centroid1[i])
        assert r.bbox == (
            test_bbox0[i],
            test_bbox1[i],
            test_bbox2[i],
            test_bbox3[i],
        )

        for j in range(len(region_props_features)):
            np.testing.assert_equal(
                r[region_props_features[j]],
                test_df[f"AreaShape_{cp_features[j]}"].values[i],
                err_msg=f"{region_props_features[j]}",
            )


@pytest.mark.features
def test_create_funcs():
    funcs, requires_intensity = _create_funcs(["colocalization_*_*"], 3)
    assert requires_intensity
    assert len(funcs) == 1
    assert funcs[0].keywords["c"] == [(0, 1), (0, 2), (1, 2)]
    funcs, requires_intensity = _create_funcs(["haralick_*_3", "haralick_*_5"], 3)
    assert requires_intensity
    assert len(funcs) == 2


@pytest.mark.features
def test_features_cli_multi_images(tmp_path, array_A1_102_cells, array_A1_102_alnpheno):
    # test that multiple images are stacked
    image = (
        array_A1_102_alnpheno.transpose(*("z", "c", "t", "y", "x")).rename(
            {"z": "t", "t": "z"}
        )
    ).isel(t=0, z=0)  # ops swaps z and t in saved tif

    labels = array_A1_102_cells.squeeze().copy()
    labels.values[labels.values != 17] = 0
    zarr_path1 = tmp_path.joinpath("test1.zarr")
    zarr_path2 = tmp_path.joinpath("test2.zarr")
    output_path = str(tmp_path.joinpath("features-out"))
    objects_path = str(tmp_path.joinpath("objects-out"))
    exp = Experiment()
    exp.images["test"] = image
    exp.labels["test-cell"] = labels
    exp.save(str(zarr_path1))

    exp = Experiment()
    exp.images["test"] = image
    exp.save(str(zarr_path2))

    cmd = [
        "scallops",
        "find-objects",
        "--labels",
        str(zarr_path1),
        "--label-pattern",
        "{well}",
        "--output",
        objects_path,
    ]

    check_call(cmd)

    cmd = [
        "scallops",
        "features",
        "--images",
        str(zarr_path1),
        "--image-pattern",
        "{well}",
        "--labels",
        str(zarr_path1),
        "--stack-images",
        str(zarr_path2),
        "--stack-image-pattern",
        "{well}",
        "--output",
        output_path,
        "--features-cell",
        "colocalization_0_2",
        "--objects",
        objects_path,
        "--channel-rename",
        '{"0":"A", "2":"B"}',
    ]

    check_call(cmd)
    outputs = glob.glob(os.path.join(output_path, "cell", "*.parquet"))
    dfs = []
    for output in outputs:
        dfs.append(pd.read_parquet(output))
    df = pd.concat(dfs, axis=1)
    assert len(df) == len(np.unique(labels)) - 1

    assert df["Cells_Correlation_Pearson_A_B"].min() > 0.9999


@pytest.mark.features
def test_features_cli(tmp_path, array_A1_102_cells, array_A1_102_alnpheno):
    tmp_path.mkdir(parents=True, exist_ok=True)
    # test that all features run, can be saved to disk, and diff with known good output
    image = (
        array_A1_102_alnpheno.transpose(*("z", "c", "t", "y", "x")).rename(
            {"z": "t", "t": "z"}
        )
    ).isel(t=0, z=0)  # ops swaps z and t in saved tif

    labels = array_A1_102_cells.squeeze().copy()
    labels.values[labels.values != 17] = 0
    zarr_path = str(tmp_path / "test.zarr")
    features_output_path = str(tmp_path / "features-out")
    objects_output_path = str(tmp_path / "objects-out")
    exp = Experiment()
    exp.images["test"] = image
    exp.labels["test-cell"] = labels
    exp.save(zarr_path)

    cmd = [
        "scallops",
        "find-objects",
        "--labels",
        zarr_path,
        "--label-pattern",
        "{well}",
        "--output",
        objects_output_path,
    ]
    check_call(cmd)

    cmd = [
        "scallops",
        "features",
        "--images",
        zarr_path,
        "--labels",
        zarr_path,
        "--output",
        features_output_path,
        "--features-cell",
        "intensity_*",
        "sizeshape",
        "colocalization_*_*",
        "--objects",
        objects_output_path,
    ]

    check_call(cmd)


@pytest.mark.features
def test_phenotype_ops(array_A1_102_cells, array_A1_102_alnpheno, array_A1_102_nuclei):
    cells = array_A1_102_cells.squeeze().data
    nuclei = array_A1_102_nuclei.squeeze().data
    pheno_aligned = (
        array_A1_102_alnpheno.transpose(*("z", "c", "t", "y", "x")).rename(
            {"z": "t", "t": "z"}
        )
    ).isel(t=0, z=0)  # ops swaps z and t in saved tif
    labels = dict(cell=cells, nuclei=nuclei)
    features = dict(
        cell=["sizeshape"],
        nuclei=[
            "intensity_0",
            "intensity_1",
            "colocalization_0_1",
        ],
    )
    dfs = []
    for key in features:
        df = _label_features(
            intensity_image=pheno_aligned.transpose(*("y", "x", "c")).data,
            label_image=labels[key],
            features=features[key],
            normalize=False,
        )
        df.columns = _label_name_to_prefix[key] + "_" + df.columns
        dfs.append(df)

    df = pd.concat(dfs, join="inner", axis=1)
    df.index.name = "cell"
    diff_pheno(df)


def diff_pheno(df_test):
    df_test = df_test.rename(
        {
            "Cells_AreaShape_Center_Y": "cell_y",
            "Cells_AreaShape_Center_X": "cell_x",
            "Cells_AreaShape_Area": "cell_area",
            "Nuclei_AreaShape_Center_Y": "nuclei_y",
            "Nuclei_AreaShape_Center_X": "nuclei_x",
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

    if "label" in df_test.columns:
        df_test = df_test.rename({"label": "cell"}, axis=1)
    df_known_good = pd.read_csv(
        str(__data__.joinpath("process_fig4", "10X_A1_Tile-102.phenotype.csv"))
    )
    df_known_good = df_known_good.drop(
        ["dapi_gfp_corr_nucleus", "tile", "well"], axis=1
    )  # corr implementation changed
    df_known_good = df_known_good.rename(
        {
            "area_cell": "cell_area",
            "i_cell": "cell_y",
            "j_cell": "cell_x",
            "i_nucleus": "nuclei_y",
            "j_nucleus": "nuclei_x",
            "area_nucleus": "nuclei_area",
            "dapi_gfp_corr_nucleus": "nuclei_corr_0_1",
            "dapi_max_nucleus": "nuclei_max_0",
            "dapi_mean_nucleus": "nuclei_mean_0",
            "dapi_median_nucleus": "nuclei_median_0",
            "gfp_max_nucleus": "nuclei_max_1",
            "gfp_mean_nucleus": "nuclei_mean_1",
            "gfp_median_nucleus": "nuclei_median_1",
        },
        axis=1,
    )  # ['cell', 'cell_area', 'cell_y', 'cell_x', 'nuclei_corr_0_1', 'tile', 'well']
    df_test = df_test.reset_index()[df_known_good.columns]
    df_known_good = df_known_good.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df_test,
        df_known_good,
        check_dtype=False,
    )


@pytest.mark.features
def test_pftas_features(array_A1_102_cells, array_A1_102_pheno):
    label_image = array_A1_102_cells.squeeze().data
    intensity_image = (
        (
            array_A1_102_pheno.transpose(*("z", "c", "t", "y", "x"))
            .rename({"z": "t", "t": "z"})
            .isel(t=0, z=0)
        )
        .transpose(*("y", "x", "c"))
        .data
    )
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]
    channel_names = ["c0", "c1"]

    features_scallops = pftas(
        c=[0, 1],
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        intensity_image=intensity_image,
    )
    assert len(features_scallops) == 54 * 2


@pytest.mark.features
def test_distance_from_bounding_box_to_edge():
    # no crosses boundary
    query = shapely.box(5, 5, 11, 10).bounds
    template = shapely.box(0, 0, 15, 14).bounds

    labels_df = pd.DataFrame(
        data={
            "bbox-0": [query[1]],
            "bbox-2": [query[3]],
            "bbox-1": [query[0]],
            "bbox-3": [query[2]],
        }
    )

    boundaries_df = pd.DataFrame(
        data={
            "bbox-0": [template[1]],
            "bbox-2": [template[3]],
            "bbox-1": [template[0]],
            "bbox-3": [template[2]],
        }
    )
    distance_df = bounding_box_to_edge_distance(
        objects_boxes_df=labels_df, objects_edges_df=boundaries_df
    )
    assert not distance_df.loc[0, "crosses_boundary"]
    assert distance_df.loc[0, "distance"] == 4

    # crosses boundary
    query1 = shapely.box(5, 5, 16, 10).bounds
    template1 = shapely.box(0, 0, 15, 14).bounds
    labels_df = pd.DataFrame(
        data={
            "bbox-0": [query1[1]],
            "bbox-2": [query1[3]],
            "bbox-1": [query1[0]],
            "bbox-3": [query1[2]],
        }
    )
    boundaries_df = pd.DataFrame(
        data={
            "bbox-0": [template1[1]],
            "bbox-2": [template1[3]],
            "bbox-1": [template1[0]],
            "bbox-3": [template1[2]],
        }
    )
    distance_df = bounding_box_to_edge_distance(
        objects_boxes_df=labels_df, objects_edges_df=boundaries_df
    )
    assert distance_df.loc[0, "crosses_boundary"]
    assert distance_df.loc[0, "distance"] == 0

    # does not cross boundary, just touches edge
    query2 = shapely.box(5, 5, 15, 10).bounds
    template2 = shapely.box(0, 0, 15, 14).bounds
    labels_df = pd.DataFrame(
        data={
            "bbox-0": [query2[1]],
            "bbox-2": [query2[3]],
            "bbox-1": [query2[0]],
            "bbox-3": [query2[2]],
        }
    )
    boundaries_df = pd.DataFrame(
        data={
            "bbox-0": [template2[1]],
            "bbox-2": [template2[3]],
            "bbox-1": [template2[0]],
            "bbox-3": [template2[2]],
        }
    )
    distance_df = bounding_box_to_edge_distance(
        objects_boxes_df=labels_df, objects_edges_df=boundaries_df
    )
    assert not distance_df.loc[0, "crosses_boundary"]
    assert distance_df.loc[0, "distance"] == 0


@pytest.mark.features
def test_fish_spots(array_A1_102_cells):
    pytest.importorskip("ufish")
    image = read_image(
        "scallops/tests/data/experimentC/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-102.sbs.tif"
    )
    image = image.squeeze().transpose(*("y", "x", "c")).data
    cells = array_A1_102_cells.squeeze().data
    cells = cells[500:550, 500:550]
    image = image[500:550, 500:550]
    unique_labels = np.unique(cells)
    unique_labels = unique_labels[unique_labels != 0]
    results = spot_count(
        c=[1],
        intensity_threshold=0.06,
        channel_names=["DAPI", "FISH"],
        unique_labels=unique_labels,
        label_image=cells,
        intensity_image=image,
        min_peak_distance=3,
        radius=3,
    )

    np.testing.assert_almost_equal(
        results["Spots_Count_FISH"], np.array([0, 0, 0, 0, 10, 1, 0, 0, 0, 0, 1])
    )
