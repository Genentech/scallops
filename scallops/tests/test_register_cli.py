import os
import subprocess

import numpy as np
import ome_types
import pandas as pd
import pytest
import xarray as xr
from skimage.data import camera
from skimage.transform import SimilarityTransform, resize, warp

from scallops.experiment.elements import Experiment
from scallops.io import read_experiment, read_image, save_ome_tiff
from scallops.registration.itk import (
    _load_itk_parameters,
    _load_itk_parameters_from_dir,
    itk_align_to_reference_time,
    itk_transform_image,
    itk_transform_labels,
    set_automatic_transform_initialization,
)
from scallops.xr import apply_data_array
from scallops.zarr_io import read_ome_zarr_array


def add_physical_size(path, physical_size):
    img = read_image(path)
    if isinstance(img.attrs["processed"], str):
        img.attrs["processed"] = ome_types.from_xml(img.attrs["processed"])
    img.attrs["processed"].images[0].pixels.physical_size_x = physical_size
    img.attrs["processed"].images[0].pixels.physical_size_y = physical_size
    save_ome_tiff(img.values, uri=path, ome_xml=img.attrs["processed"].to_xml())


def create_itk_param_file(tmp_path):
    p = """
    (AutomaticParameterEstimation "true")
    (AutomaticTransformInitialization "true")
    (CheckNumberOfSamples "true")
    (DefaultPixelValue 0)
    (FinalBSplineInterpolationOrder 3)
    (FixedImagePyramid "FixedSmoothingImagePyramid")
    (ImageSampler "RandomCoordinate")
    (Interpolator "LinearInterpolator")
    (MaximumNumberOfIterations 256)
    (MaximumNumberOfSamplingAttempts 8)
    (Metric "AdvancedMattesMutualInformation")
    (MovingImagePyramid "MovingSmoothingImagePyramid")
    (NewSamplesEveryIteration "true")
    (NumberOfResolutions 4)
    (NumberOfSamplesForExactGradient 4096)
    (NumberOfSpatialSamples 2048)
    (Optimizer "AdaptiveStochasticGradientDescent")
    (Registration "MultiResolutionRegistration")
    (ResampleInterpolator "FinalBSplineInterpolator")
    (Resampler "DefaultResampler")
    (ResultImageFormat "nii")
    (Transform "TranslationTransform")
    (WriteIterationInfo "false")
    (WriteResultImage "false")
    """
    path = os.path.join(tmp_path, "translation.txt")
    with open(path, "wt") as out:
        out.write(p)
    return path


@pytest.mark.registration
def test_register_cross_correlation_cli(tmp_path, array_A1_102_aln):
    output_dir = os.path.join(tmp_path, "cross-correlation-output.zarr")
    cmd = [
        "scallops",
        "registration",
        "cross-correlation",
        "--images",
        "scallops/tests/data/experimentC/input",
        "--image-pattern",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--subset=A1-102",
        "--across-t-channel=0",
        "--groupby",
        "well",
        "tile",
        "--within-t-channel",
        "1",
        "2",
        "3",
        "4",
        "--output=" + output_dir,
    ]
    subprocess.check_call(cmd)
    aligned = array_A1_102_aln.transpose(*("z", "c", "t", "y", "x")).rename(
        {"z": "t", "t": "z"}
    )  # ops swaps z and t in saved tif
    registered_image = read_image(os.path.join(output_dir, "images", "A1-102"))
    np.testing.assert_equal(registered_image.data, aligned.data)


@pytest.mark.registration
def test_register_itk_cli_known_shift(tmp_path):
    image = camera()
    shifts = [(-22, 13), (40, 50)]
    # skimage SimilarityTransform has (x,y,[z]) convention
    arrays = [image]
    for shift in shifts:
        st = SimilarityTransform(translation=shift[::-1])
        arrays.append(warp(image, st, preserve_range=True).astype(image.dtype))
    data = xr.DataArray(np.array(arrays), dims=["t", "y", "x"])
    data = data.expand_dims("c", 1)
    data.attrs["processed"] = dict(
        images=[dict(pixels=dict(physical_size_x=1, physical_size_y=1))]
    )
    data_path = tmp_path / "test.zarr"
    exp = Experiment()
    exp.images["test"] = data
    exp.save(data_path)

    cmd = [
        "scallops",
        "registration",
        "elastix",
        "--time",
        "1",
        "--moving",
        str(data_path),
        "--no-landmarks",
        "--itk-parameters",
        create_itk_param_file(tmp_path),
        "--moving-output",
        str(tmp_path / "moving.zarr"),
        "--transform-output",
        str(tmp_path / "transforms"),
    ]
    subprocess.check_call(cmd)
    transformed_exp = read_experiment(tmp_path / "moving.zarr")
    transformed_image = transformed_exp.images[list(exp.images)[0]]
    assert transformed_image.sizes == data.sizes
    np.testing.assert_array_equal(transformed_image.isel(t=1), data.isel(t=1))
    # check that reference time is unchanged


@pytest.mark.registration
def test_register_itk_cli_t_reference(tmp_path, array_A1_102_nuclei):
    param_file = create_itk_param_file(tmp_path)
    transform_output_dir = os.path.join(tmp_path, "transform")
    elastix_output_dir = os.path.join(tmp_path, "elastix-output.zarr")
    registration_input_moving_labels_path = os.path.join(
        tmp_path, "registration-input.zarr"
    )
    exp = Experiment()
    reference_t = 2
    test_t = 10
    array_A1_102_nuclei = array_A1_102_nuclei.squeeze()
    exp.labels[f"A1-102-{test_t}-mask"] = array_A1_102_nuclei
    exp.save(registration_input_moving_labels_path)

    cmd = [
        "scallops",
        "registration",
        "elastix",
        "--moving",
        "scallops/tests/data/experimentC/input",
        "--no-landmarks",
        "--moving-image-spacing",
        "1,1",
        "--itk-parameters",
        param_file,
        "--moving-image-pattern",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--groupby",
        "well",
        "tile",
        "--subset",
        "A1-102",
        "--transform-output",
        transform_output_dir,
        "--moving-output",
        elastix_output_dir,
        "--moving-label",
        registration_input_moving_labels_path,
        "--label-output",
        elastix_output_dir,
        "--time",
        str(reference_t),
    ]
    subprocess.check_call(cmd)
    result_exp = read_experiment(elastix_output_dir)
    transformed_image = result_exp.images["A1-102"].squeeze()
    assert transformed_image.dtype == np.uint16
    original_image = (
        read_experiment(
            "scallops/tests/data/experimentC/input",
            "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
            group_by=("well", "tile"),
        )
        .images["A1-102"]
        .squeeze()
    )
    np.testing.assert_array_equal(transformed_image.t.values, original_image.t.values)
    np.testing.assert_array_equal(transformed_image.c.values, original_image.c.values)
    np.testing.assert_array_equal(
        transformed_image.isel(t=reference_t),
        original_image.isel(t=reference_t),
        err_msg="Reference t not equal via CLI",
    )
    for t in range(original_image.sizes["t"]):
        if t != reference_t:
            with np.testing.assert_raises(AssertionError):
                np.testing.assert_array_equal(
                    transformed_image.isel(t=t), original_image.isel(t=t)
                )
    transform_parameter_object = _load_itk_parameters_from_dir(
        os.path.join(transform_output_dir, "A1-102", f"t={test_t}")
    )

    # test load and apply saved transform for image
    warped = itk_transform_image(
        image=original_image.sel(t=test_t, c=original_image.c.values[0]),
        transform_parameter_object=transform_parameter_object,
        image_spacing=(1, 1),
    )

    np.testing.assert_array_equal(
        transformed_image.sel(t=test_t, c=transformed_image.c.values[0]).data,
        warped.values,
        err_msg=f"t {test_t} images not equal using itk_transform_image and CLI",
    )
    # test load and apply saved transform for labels
    assert len(result_exp.labels.keys()) == 1
    warped_labels = itk_transform_labels(
        image=array_A1_102_nuclei,
        transform_parameter_object=transform_parameter_object,
        image_spacing=(1, 1),
    )
    assert warped_labels.min() == 0
    np.testing.assert_array_equal(
        result_exp.labels[f"A1-102-{test_t}-mask"].values,
        warped_labels,
        err_msg=f"t {test_t} labels not equal using itk_transform_labels and CLI",
    )
    # compare results to API usage
    parameter_object = _load_itk_parameters([param_file])
    set_automatic_transform_initialization(parameter_object)
    result_np = itk_align_to_reference_time(
        moving_image=original_image,
        moving_channel=[0],
        parameter_object=parameter_object,
        moving_image_spacing=(1, 1),
        reference_timepoint=reference_t,
    )

    xr.testing.assert_equal(result_np, transformed_image)


@pytest.mark.registration
def test_register_itk_cli_concat_t(tmp_path):
    transform_output_dir = os.path.join(tmp_path, "transform")
    image = (
        read_experiment(
            "scallops/tests/data/experimentC/input",
            "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
            group_by=("well", "tile"),
        )
        .images["A1-102"]
        .squeeze()
    )
    exp = Experiment()
    img = image.isel(t=[0]).copy()
    img.attrs["stitch_coords"] = "test1"
    exp.images["A1-1"] = img
    img = image.isel(t=[1], c=[0, 1])
    img = img.copy()
    img.attrs["stitch_coords"] = "test2"
    exp.images["A1-2"] = img
    exp.save("test.zarr")
    elastix_output_dir = os.path.join(tmp_path, "elastix-output.zarr")
    cmd = [
        "scallops",
        "registration",
        "elastix",
        "--moving",
        "test.zarr",
        "--moving-image-pattern",
        "{well}-{t}",
        "--no-landmarks",
        "--groupby",
        "well",
        "--moving-image-spacing",
        "1,1",
        "--itk-parameters",
        create_itk_param_file(tmp_path),
        "--transform-output",
        transform_output_dir,
        "--moving-output",
        elastix_output_dir,
    ]
    " ".join(cmd)
    subprocess.check_call(cmd)
    transformed_image = read_ome_zarr_array(
        os.path.join(elastix_output_dir, "images/A1")
    )
    assert transformed_image.attrs["stitch_coords"] == ["test1", "test2"]
    assert transformed_image.sizes["c"] == 7


@pytest.mark.registration
def test_register_itk_cli(tmp_path, array_A1_102_nuclei):
    st = SimilarityTransform(translation=[100, 20])
    image = read_image(
        "scallops/tests/data/experimentC/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-102.sbs.tif"
    ).squeeze()
    moving_image = image.copy()

    def _warp(img):
        return warp(img.values, st, order=1, preserve_range=True)

    moving_image = apply_data_array(moving_image, ["c"], _warp)
    moving_image = xr.DataArray(
        resize(
            moving_image.values,
            (
                moving_image.sizes["c"],
                (moving_image.sizes["y"] / 2) - 1,
                (moving_image.sizes["x"] / 2) - 3,
            ),
            anti_aliasing=True,
            preserve_range=True,
        ),
        dims=["c", "y", "x"],
    )
    moving_image = moving_image.isel(c=[0, 1, 2])

    moving_labels = array_A1_102_nuclei.squeeze().values
    moving_labels = warp(moving_labels, st, order=0, preserve_range=True)
    moving_labels = xr.DataArray(
        resize(
            moving_labels,
            ((moving_labels.shape[0] / 2) - 1, (moving_labels.shape[1] / 2) - 3),
            anti_aliasing=False,
            order=0,
        ),
        dims=["y", "x"],
    )
    moving_image.coords["c"] = ["a", "b", "c"]
    moving_image.attrs["physical_pixel_sizes"] = (2, 2)
    moving_image.attrs["physical_pixel_units"] = ("micrometer", "micrometer")

    fixed_image = xr.DataArray(
        resize(
            image.values,
            (
                image.sizes["c"],
                image.sizes["y"] - 1,
                image.sizes["x"] - 3,
            ),
        ),
        dims=["c", "y", "x"],
    )

    exp = Experiment()
    exp.labels["1-nuclei"] = moving_labels
    exp.images["1"] = moving_image
    registration_input_zarr_path = os.path.join(tmp_path, "registration-input.zarr")
    exp.save(registration_input_zarr_path)

    fixed_img_path = os.path.join(tmp_path, "fixed", "1.ome.tif")

    os.makedirs(os.path.dirname(fixed_img_path), exist_ok=True)
    save_ome_tiff(fixed_image.values, uri=fixed_img_path)

    add_physical_size(fixed_img_path, 1)

    transform_output_dir = os.path.join(tmp_path, "transform")
    elastix_output_dir = os.path.join(tmp_path, "elastix-output.zarr")
    cmd = [
        "scallops",
        "registration",
        "elastix",
        "--fixed",
        os.path.dirname(fixed_img_path),
        "--moving",
        registration_input_zarr_path,
        "--moving-label",
        registration_input_zarr_path,
        "--itk-parameters",
        create_itk_param_file(tmp_path),
        "--moving-image-pattern",
        "{well}",
        "--fixed-image-pattern",
        "{well}.ome.tif",
        "--groupby",
        "well",
        "--transform-output",
        transform_output_dir,
        "--label-output",
        elastix_output_dir,
        "--moving-output",
        elastix_output_dir,
        "--landmark-step-size",
        "50",
    ]

    subprocess.check_call(cmd)
    transformed_labels = read_ome_zarr_array(
        os.path.join(elastix_output_dir, "labels/1-nuclei")
    )
    assert transformed_labels.min() == 0
    transformed_image = read_ome_zarr_array(
        os.path.join(elastix_output_dir, "images/1")
    )
    assert transformed_image.sizes["c"] == moving_image.sizes["c"]
    assert transformed_labels.shape == (fixed_image.sizes["y"], fixed_image.sizes["x"])
    assert (transformed_image.sizes["y"], transformed_image.sizes["x"]) == (
        fixed_image.sizes["y"],
        fixed_image.sizes["x"],
    )

    assert (
        _load_itk_parameters_from_dir(
            os.path.join(transform_output_dir, "1")
        ).GetNumberOfParameterMaps()
        == 2  # landmarks + params
    )

    # verify we can the same results using transformix command
    transformix_output = os.path.join(tmp_path, "transformix.zarr")
    cmd = [
        "scallops",
        "registration",
        "transformix",
        "--images",
        registration_input_zarr_path,
        "--type",
        "labels",
        "--image-spacing",
        "2,2",
        "--output",
        transformix_output,
        "--transform",
        transform_output_dir,
    ]

    subprocess.check_call(cmd)

    cmd = [
        "scallops",
        "registration",
        "transformix",
        "--images",
        registration_input_zarr_path,
        "--type",
        "images",
        "--output",
        transformix_output,
        "--transform",
        transform_output_dir,
    ]

    subprocess.check_call(cmd)

    transformix_exp = read_experiment(transformix_output)
    np.testing.assert_array_equal(
        transformed_labels.values,
        transformix_exp.labels["1-nuclei"].values,
        "Labels are not equal",
    )
    np.testing.assert_array_equal(
        transformed_image.values,
        transformix_exp.images["1"].values,
        "Arrays are not equal",
    )
    transform_output_dir2 = os.path.join(tmp_path, "transform2")
    cmd = [
        "scallops",
        "registration",
        "elastix",
        "--fixed",
        os.path.dirname(fixed_img_path),
        "--moving",
        registration_input_zarr_path,
        "--moving-image-pattern",
        "{well}",
        "--fixed-image-pattern",
        "{well}.ome.tif",
        "--groupby",
        "well",
        "--transform-output",
        transform_output_dir2,
        "--landmark-step-size",
        "50",
        "--landmark-image-chunk-size",
        "100",
        "--landmark-template-padding",
        "100",
        "--landmark-initialization",
        "none",
        "--force",
    ]
    subprocess.check_call(cmd)
    df = pd.read_parquet(os.path.join(transform_output_dir2, "1", "landmarks.parquet"))
    assert len(df) == 441
    n_inliers = len(df.query("inlier"))
    assert abs(364 - n_inliers) < 5, f"# of inliers: {n_inliers}"
