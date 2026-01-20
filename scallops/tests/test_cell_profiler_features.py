import os.path
import pickle

import dask.array as da
import numpy as np
import pytest
from cp_measure.core.measurecolocalization import (
    get_correlation_costes,
    get_correlation_manders_fold,
    get_correlation_overlap,
    get_correlation_pearson,
    get_correlation_rwc,
)
from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors

from scallops.features.colocalization import _colocalization_pairs
from scallops.features.cp_measure_wrapper import (
    cp_intensity,
    cp_intensity_distribution,
    cp_size_shape,
    cp_texture,
)
from scallops.features.find_objects import find_objects
from scallops.features.generate import label_features
from scallops.features.intensity import intensity
from scallops.features.intensity_distribution import intensity_distribution
from scallops.features.neighbors import neighbors
from scallops.features.texture import haralick
from scallops.segmentation.util import relabel_sequential


@pytest.mark.features
def test_neighbors(array_A1_102_cells):
    label_image = array_A1_102_cells.squeeze().data
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]
    features_cp = measureobjectneighbors(
        relabel_sequential(label_image), relabel_sequential(label_image)
    )
    features_cp["Neighbors_FirstClosestObjectNumber_Expanded"] = unique_labels[
        features_cp["Neighbors_FirstClosestObjectNumber_Expanded"] - 1
    ]
    features_cp["Neighbors_SecondClosestObjectNumber_Expanded"] = unique_labels[
        features_cp["Neighbors_SecondClosestObjectNumber_Expanded"] - 1
    ]

    features_scallops = neighbors(label_image, label_image)
    for key in features_cp:
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[key],
            err_msg=key,
        )


@pytest.mark.features
def test_intensity(array_A1_102_cells, array_A1_102_pheno):
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
    c = [0, 1]
    channel_names = ["c0", "c1"]
    if not os.path.exists("scallops/tests/data/features/intensity.pkl"):
        label_image_relabel = relabel_sequential(label_image)
        unique_labels_relabel = np.unique(label_image_relabel)
        unique_labels_relabel = unique_labels_relabel[unique_labels_relabel > 0]
        features_cp = cp_intensity(
            c=[0],
            channel_names=channel_names,
            unique_labels=unique_labels_relabel,
            label_image=label_image_relabel,
            intensity_image=intensity_image,
        )
        with open("scallops/tests/data/features/intensity.pkl", "wb") as f:
            pickle.dump(features_cp, f)
    else:
        with open("scallops/tests/data/features/intensity.pkl", "rb") as f:
            features_cp = pickle.load(f)
    features_scallops = intensity(
        c=c,
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        label_image_original=label_image,
        intensity_image=intensity_image,
        offset=(0, 0),
    )
    # these values are computed differently in cp-measure
    inexact = {
        "Std": 0.0008,
        "LowerQuartile": 0.016,
        "UpperQuartile": 0.16,
        "Median": 0.065,
        "MAD": 0.31,
        "Location_MaxIntensity": 0.06,
    }

    for key in features_cp:
        if key in ("Location_CenterMassIntensity_Z_c0", "Location_MaxIntensity_Z_c0"):
            continue
        rtol = None
        for t in inexact.keys():
            if key.find(t) != -1:
                rtol = inexact[t]
                break
        if rtol is not None:
            np.testing.assert_allclose(
                features_cp[key],
                features_scallops[key],
                err_msg=key,
                rtol=rtol,
            )
        else:
            np.testing.assert_array_equal(
                features_cp[key],
                features_scallops[key],
                err_msg=key,
            )


@pytest.mark.features
def test_colocalization(array_A1_102_cells, array_A1_102_pheno):
    label_image = array_A1_102_cells.squeeze().data
    label_image = relabel_sequential(label_image)

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
    features_scallops = _colocalization_pairs(
        c=[(0, 1)],
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        intensity_image=intensity_image,
    )
    # pearson
    features_cp = get_correlation_pearson(
        intensity_image[..., 0], intensity_image[..., 1], label_image
    )

    for key in features_cp:
        np.testing.assert_allclose(
            features_cp[key],
            features_scallops[f"{key}_c0_c1"],
            atol=2.22044605e-16,
            rtol=2.49860215e-16,
            err_msg=key,
        )
    # manders
    features_cp = get_correlation_manders_fold(
        intensity_image[..., 0], intensity_image[..., 1], label_image
    )
    for key in features_cp:
        scallops_key = key.replace("_1", "_c0_c1").replace("_2", "_c1_c0")
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[scallops_key],
            err_msg=key,
        )

    # overlap
    features_cp = get_correlation_overlap(
        intensity_image[..., 0], intensity_image[..., 1], label_image
    )

    for key in features_cp:
        scallops_key = (
            key.replace("K_1", "K_c0_c1")
            .replace("K_2", "K_c1_c0")
            .replace("Overlap", "Overlap_c0_c1")
        )
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[scallops_key],
            err_msg=key,
        )

    # rwc
    features_cp = get_correlation_rwc(
        intensity_image[..., 0], intensity_image[..., 1], label_image
    )
    for key in features_cp:
        scallops_key = key.replace("_1", "_c0_c1").replace("_2", "_c1_c0")
        np.testing.assert_allclose(
            features_cp[key],
            features_scallops[scallops_key],
            err_msg=key,
            atol=1.77635684e-15,
            rtol=2.11831416e-15,
        )

    # costes
    features_cp = get_correlation_costes(
        intensity_image[..., 0], intensity_image[..., 1], label_image
    )
    for key in features_cp:
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[key],
            err_msg=key,
        )


@pytest.mark.features
def test_haralick_features(array_A1_102_cells, array_A1_102_pheno):
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

    features_scallops = haralick(
        c=[0, 1],
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        intensity_image=intensity_image,
    )

    features_cp = cp_texture(
        c=[0, 1],
        channel_names=channel_names,
        unique_labels=None,
        label_image=relabel_sequential(label_image),
        intensity_image=intensity_image,
    )
    assert len(features_cp) == len(features_scallops)
    for key in features_cp:
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[key],
            err_msg=key,
        )


@pytest.mark.features
def test_features_dask(array_A1_102_cells, array_A1_102_pheno):
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
    label_image_dask = da.from_array(label_image, chunks=(200, 200))
    intensity_image_dask = da.from_array(intensity_image, chunks=(200, 200, 1))
    objects_df = find_objects(label_image_dask).compute()
    features_scallops = label_features(
        objects_df=objects_df,
        label_image=label_image_dask,
        features=["colocalization_0_1", "sizeshape", "intensity_0"],
        intensity_image=intensity_image_dask,
        channel_names={0: "c0", "1": "c1"},
    ).compute()
    features_scallops = features_scallops.join(objects_df).sort_index()

    features_cp = get_correlation_pearson(
        intensity_image[..., 0], intensity_image[..., 1], label_image
    )
    # values are slightly different because arrays are ordered differently
    for key in features_cp:
        np.testing.assert_allclose(
            features_cp[key],
            features_scallops[f"{key}_c0_c1"],
            rtol=0.00014,
            err_msg=key,
        )

    features_cp = cp_size_shape(
        channel_names=None,
        unique_labels=None,
        label_image=relabel_sequential(label_image),
        intensity_image=None,
        remove_objects=False,
    )

    for key in features_cp:
        # https://github.com/afermg/cp_measure/issues/18
        if key.startswith("AreaShape_Zernike"):
            diff = np.max(np.abs(features_cp[key] - features_scallops[key]))
            assert diff < 0.025, f"{key}, {diff}"

        else:
            np.testing.assert_array_equal(
                features_cp[key],
                features_scallops[key],
                err_msg=key,
            )


@pytest.mark.features
def test_intensity_distribution(array_A1_102_cells, array_A1_102_pheno):
    label_image = relabel_sequential(array_A1_102_cells.squeeze().data)

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
    c = [0, 1]
    channel_names = ["c0", "c1"]

    features_cp = cp_intensity_distribution(
        c=c,
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        intensity_image=intensity_image,
        calculate_zernike=True,
    )

    features_scallops = intensity_distribution(
        c=c,
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        intensity_image=intensity_image,
        calculate_zernike=True,
    )

    for key in features_cp:
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[key],
            err_msg=key,
        )
