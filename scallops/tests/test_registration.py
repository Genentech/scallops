import numpy as np
import pytest
import skimage
import xarray as xr
from scipy.ndimage import fourier_shift

from scallops.io import read_experiment, read_image
from scallops.registration.crosscorrelation import align_image, align_images


@pytest.mark.registration
def test_align_images(experiment_c_A1_102_pheno_aligned):
    pheno_image = read_image(
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    )
    image = read_experiment(
        "scallops/tests/data/experimentC/input",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
    ).images["A1-102"]
    aligned_pheno_image = align_images(image.isel(t=0, z=0), pheno_image.isel(t=0, z=0))

    np.testing.assert_equal(
        experiment_c_A1_102_pheno_aligned.data.squeeze(),
        aligned_pheno_image.data.squeeze(),
    )


@pytest.mark.registration
def test_align_image(experiment_c, experiment_c_A1_102_aligned):
    align_between_time_channel = 0  # DAPI
    image = experiment_c.images["A1-102"]

    align_within_time_channels = np.delete(
        np.arange(image.sizes["c"]), align_between_time_channel
    )  # [1,2,3,4]
    aligned_image = align_image(
        image.copy(),
        align_within_time_channels=align_within_time_channels,
        align_between_time_channel=align_between_time_channel,
        filter_percentiles=[0, 90],
    )

    np.testing.assert_equal(aligned_image.data, experiment_c_A1_102_aligned.data)
    np.testing.assert_equal(aligned_image.isel(t=0).data, image.isel(t=0).data)


def _create_shifted_images():
    image = skimage.data.camera()
    shifts = [(-22.4, 13.32), (13.32, -22.4)]
    images = [xr.DataArray(image, dims=["y", "x"])]
    for shift in shifts:
        offset_image = fourier_shift(np.fft.fftn(image), shift)
        offset_image = np.fft.ifftn(offset_image)
        images.append(xr.DataArray(offset_image, dims=["y", "x"]))
    return images


@pytest.mark.registration
def test_align_between_time_coordinate_transformations():
    img = xr.concat(_create_shifted_images(), dim="t")
    img = xr.concat((img, img), dim="c")
    align_between_time_image = align_image(
        img,
        align_within_time_channels=None,
        align_between_time_channel=0,
        filter_percentiles=None,
        upsample_factor=1,
        window=1,
    )
    coordinate_transformations = align_between_time_image.attrs[
        "coordinateTransformations"
    ]
    assert len(coordinate_transformations) == 4
    assert coordinate_transformations[0]["translation"] == [
        -22,
        13,
    ] and coordinate_transformations[1]["translation"] == [-22, 13]
    assert coordinate_transformations[0]["sel"] == dict(t=1, c=0)
    assert coordinate_transformations[1]["sel"] == dict(t=1, c=1)

    assert coordinate_transformations[2]["translation"] == [
        13,
        -22,
    ] and coordinate_transformations[3]["translation"] == [13, -22]
    assert coordinate_transformations[2]["sel"] == dict(t=2, c=0)
    assert coordinate_transformations[3]["sel"] == dict(t=2, c=1)


@pytest.mark.registration
def test_align_within_time_coordinate_transformations():
    img = xr.concat(_create_shifted_images(), dim="c").expand_dims("t")
    align_within_time_image = align_image(
        img,
        align_within_time_channels=[0, 1, 2],
        align_between_time_channel=None,
        filter_percentiles=None,
        upsample_factor=1,
        window=1,
    )
    coordinate_transformations = align_within_time_image.attrs[
        "coordinateTransformations"
    ]
    assert len(coordinate_transformations) == 2
    assert coordinate_transformations[0]["translation"] == [-22, 13]
    assert coordinate_transformations[0]["sel"] == dict(t=0, c=1)
    assert coordinate_transformations[1]["translation"] == [13, -22]
    assert coordinate_transformations[1]["sel"] == dict(t=0, c=2)


@pytest.mark.registration
def test_align_image_align_within_time(experiment_c):
    align_between_time_channel = 0  # DAPI
    image = experiment_c.images["A1-102"]
    align_within_time_channels = np.delete(
        np.arange(image.sizes["c"]), align_between_time_channel
    )  # [1,2,3,4]
    image_copy = image.copy()

    result = align_image(
        image_copy,
        align_within_time_channels=align_within_time_channels,
        align_between_time_channel=None,
        filter_percentiles=[0, 90],
    )
    known_good = read_image("scallops/tests/data/align/align_within_time_channels.tif")
    np.testing.assert_array_equal(result.values, known_good.values)

    xr.testing.assert_equal(image_copy, image)


@pytest.mark.registration
def test_align_image_align_between_time(experiment_c):
    image = experiment_c.images["A1-102"]
    image_copy = image.copy()
    result = align_image(
        image,
        align_within_time_channels=None,
        align_between_time_channel=0,
        filter_percentiles=[0, 90],
    )
    known_good = read_image("scallops/tests/data/align/align_between_time_channel.tif")
    np.testing.assert_array_equal(result.values, known_good.values)
    xr.testing.assert_equal(image_copy, image)


@pytest.mark.registration
def test_array_to_itk_spacing_order():
    from scallops.registration.itk import _array_to_itk

    itk_image = _array_to_itk(np.zeros((10, 20), dtype=np.float32), (0.5, 0.25))
    # scallops spacing is (y, x); ITK spacing is (x, y)
    assert tuple(itk_image.GetSpacing()) == (0.25, 0.5)


@pytest.mark.registration
def test_itk_align_across_channels():
    import itk
    from scipy.ndimage import gaussian_filter, shift

    from scallops.registration.itk import (
        itk_align_to_reference_time,
        set_automatic_transform_initialization,
    )

    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.random((256, 256)), 4).astype(np.float32)
    base = (base - base.min()) / (base.max() - base.min())
    # inverted contrast to mimic a different stain
    moving = 1 - shift(base, (4, -6), order=1, mode="nearest")
    image = xr.DataArray(
        np.stack([base, base, moving])[np.newaxis],
        dims=["t", "c", "y", "x"],
        coords=dict(t=[0], c=["a", "b", "c"]),
    )
    parameter_object_across_channels = itk.ParameterObject.New()
    parameter_object_across_channels.AddParameterMap(
        parameter_object_across_channels.GetDefaultParameterMap("translation")
    )
    set_automatic_transform_initialization(parameter_object_across_channels, False)

    result = itk_align_to_reference_time(
        moving_image=image,
        moving_channel=[0, 1],
        parameter_object=itk.ParameterObject.New(),
        moving_image_spacing=(1.0, 1.0),
        parameter_object_across_channels=parameter_object_across_channels,
    )

    # reference channels are not registered or resampled
    np.testing.assert_array_equal(result.isel(t=0, c=0).values, base)
    np.testing.assert_array_equal(result.isel(t=0, c=1).values, base)
    interior = (slice(20, -20), slice(20, -20))
    aligned = result.isel(t=0, c=2).values[interior]
    expected = 1 - base[interior]
    assert np.corrcoef(aligned.ravel(), expected.ravel())[0, 1] > 0.99
    assert np.corrcoef(moving[interior].ravel(), expected.ravel())[0, 1] < 0.9
