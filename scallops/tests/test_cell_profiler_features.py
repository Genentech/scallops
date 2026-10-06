import dask.array as da
import numpy as np
import pytest
from cp_measure.core.measurecolocalization import (
    get_correlation_pearson,
)
from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors

from scallops.features.cp_measure_wrapper import (
    cp_intensity_distribution_radial,
    cp_intensity_distribution_zernike,
    cp_size_shape,
    cp_texture,
)
from scallops.features.find_objects import find_objects
from scallops.features.generate import label_features
from scallops.features.intensity_distribution import (
    intensity_distribution_radial,
    intensity_distribution_zernike,
)
from scallops.features.neighbors import neighbors
from scallops.features.texture import haralick
from scallops.segmentation.util import relabel_sequential


@pytest.mark.features
def test_neighbors(experiment_c_A1_102_cells):
    label_image = experiment_c_A1_102_cells.squeeze().data
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
def test_haralick_features(experiment_c_A1_102_cells, experiment_c_A1_102_pheno):
    label_image = experiment_c_A1_102_cells.squeeze().data
    intensity_image = (
        experiment_c_A1_102_pheno.isel(t=0, z=0).transpose(*("y", "x", "c")).data
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
def test_intensity_distribution(experiment_c_A1_102_cells, experiment_c_A1_102_pheno):
    label_image = experiment_c_A1_102_cells.squeeze().data

    label_image = relabel_sequential(label_image)
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]
    intensity_image = (
        experiment_c_A1_102_pheno.isel(t=0, z=0).transpose(*("y", "x", "c")).data
    )

    c = [0, 1]
    channel_names = ["c0", "c1"]

    features_cp = cp_intensity_distribution_radial(
        c=c,
        channel_names=channel_names,
        unique_labels=None,
        label_image=label_image,
        intensity_image=intensity_image,
    )

    features_cp.update(
        cp_intensity_distribution_zernike(
            c=c,
            channel_names=channel_names,
            unique_labels=None,
            label_image=label_image,
            intensity_image=intensity_image,
        )
    )

    features_scallops = intensity_distribution_radial(
        c=c,
        channel_names=channel_names,
        unique_labels=unique_labels,
        label_image=label_image,
        intensity_image=intensity_image,
    )
    features_scallops.update(
        intensity_distribution_zernike(
            c=c,
            channel_names=channel_names,
            unique_labels=unique_labels,
            label_image=label_image,
            intensity_image=intensity_image,
        )
    )

    for key in features_cp:
        np.testing.assert_array_equal(
            features_cp[key],
            features_scallops[key],
            err_msg=key,
        )


@pytest.mark.features
def test_features_dask(experiment_c_A1_102_cells, experiment_c_A1_102_pheno):
    label_image = experiment_c_A1_102_cells.squeeze().data
    intensity_image = (
        experiment_c_A1_102_pheno.isel(t=0, z=0).transpose(*("y", "x", "c")).data
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
        intensity_image[..., 0],
        intensity_image[..., 1],
        relabel_sequential(label_image),
    )
    # values are slightly different because arrays are ordered differently
    for key in features_cp:
        np.testing.assert_allclose(
            features_cp[key],
            features_scallops[f"{key}_c0_c1"],
            err_msg=key,
        )

    features_cp = cp_size_shape(
        channel_names=None,
        unique_labels=None,
        label_image=relabel_sequential(label_image),
        intensity_image=None,
        remove_objects=False,
    )

    check_close = [
        "AreaShape_MajorAxisLength",
        "AreaShape_MinorAxisLength",
        "AreaShape_Eccentricity",
        "AreaShape_Orientation",
        "AreaShape_CentralMoment",
        "AreaShape_NormalizedMoment",
        "AreaShape_HuMoment",
        "AreaShape_InertiaTensor",
        "AreaShape_InertiaTensorEigenvalues",
        "AreaShape_Zernike",
    ]

    tolerance = {
        "AreaShape_Zernike": (0.02403081, 1e-7),
        "AreaShape_CentralMoment": (2.16004992e-12, 1e-7),
    }

    for key in features_cp:
        close = False
        atol = 0
        rtol = 1e-7

        for check in check_close:
            if key.startswith(check):
                close = True
                if check in tolerance:
                    atol, rtol = tolerance[check]
                break

        if close:
            np.testing.assert_allclose(
                features_cp[key],
                features_scallops[key],
                err_msg=key,
                rtol=rtol,
                atol=atol,
            )
        else:
            np.testing.assert_array_equal(
                features_cp[key],
                features_scallops[key],
                err_msg=key,
            )
