import numpy as np
import pytest
import skimage
import xarray as xr
from scipy.ndimage import fourier_shift

from scallops.datasets._registry import cache_dir
from scallops.io import read_experiment, read_image
from scallops.registration.crosscorrelation import align_image, align_images

DATA_DIR = cache_dir()


@pytest.mark.registration
def test_align_images(experiment_c_A1_102_pheno_aligned):
    pheno_image = read_image(
        f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    )
    image = read_experiment(
        f"{DATA_DIR}/experimentC/input",
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
    known_good = read_image(f"{DATA_DIR}/align/align_within_time_channels.tif")
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
    known_good = read_image(f"{DATA_DIR}/align/align_between_time_channel.tif")
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


@pytest.mark.registration
@pytest.mark.parametrize("channel_combine", ["median", "mean"])
def test_itk_align_across_channels_combined_int_channels(channel_combine):
    # the combined reference channel is float while the other channels are uint16;
    # elastix needs both images of the same pixel type
    import itk
    from scipy.ndimage import gaussian_filter, shift

    from scallops.registration.itk import (
        itk_align_to_reference_time,
        set_automatic_transform_initialization,
    )

    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.random((256, 256)), 4)
    base = (base - base.min()) / (base.max() - base.min())
    moving = shift(base, (4, -6), order=1, mode="nearest")
    image = xr.DataArray(
        (np.stack([base, base, moving]) * 4000).astype(np.uint16)[np.newaxis],
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
        channel_combine=channel_combine,
        equalize_channels=channel_combine == "mean",
    )
    np.testing.assert_array_equal(result.isel(t=0, c=0).values, image.values[0, 0])
    interior = (slice(20, -20), slice(20, -20))
    aligned = result.isel(t=0, c=2).values[interior].astype(float)
    expected = base[interior]
    assert np.corrcoef(aligned.ravel(), expected.ravel())[0, 1] > 0.99
    assert np.corrcoef(moving[interior].ravel(), expected.ravel())[0, 1] < 0.9


@pytest.mark.registration
@pytest.mark.parametrize("unroll_channels", [False, True])
def test_itk_align_reference_time_different_nchannels(tmp_path, unroll_channels):
    import itk
    import zarr

    from scallops.registration.itk import (
        _itk_align_reference_time_zarr,
        itk_align_to_reference_time,
    )
    from scallops.zarr_io import read_ome_zarr_array

    rng = np.random.default_rng(0)
    shape = (32, 32)
    t0 = xr.DataArray(
        rng.random((1, 3) + shape, dtype=np.float32),
        dims=["t", "c", "y", "x"],
        coords=dict(t=[0], c=["a", "b", "c"]),
    )
    t1 = xr.DataArray(
        rng.random((1, 2) + shape, dtype=np.float32),
        dims=["t", "c", "y", "x"],
        coords=dict(t=[1], c=["a", "b"]),
    )
    # empty parameter object: images are not transformed
    kwargs = dict(
        moving_image=[t0, t1],
        # padded channel 1 is t1's channel 0
        moving_channel=0 if unroll_channels else 1,
        parameter_object=itk.ParameterObject.New(),
        moving_image_spacing=(1.0, 1.0),
        unroll_channels=unroll_channels,
    )
    result = itk_align_to_reference_time(**kwargs)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w")
    _itk_align_reference_time_zarr(image_root=root, image_name="test", **kwargs)
    result_zarr = read_ome_zarr_array(root["images"]["test"])

    for r in (result, result_zarr):
        if unroll_channels:
            assert r.dims == ("c", "y", "x")
            assert r.sizes["c"] == 5
            np.testing.assert_array_equal(r.values[:3], t0.values[0])
            np.testing.assert_array_equal(r.values[3:], t1.values[0])
        else:
            assert r.dims == ("t", "c", "y", "x")
            assert r.sizes["t"] == 2 and r.sizes["c"] == 3
            assert r.coords["c"].values.tolist() == ["a", "b", "c"]
            np.testing.assert_array_equal(r.values[0], t0.values[0])
            np.testing.assert_array_equal(r.values[1, 0], np.zeros(shape))
            np.testing.assert_array_equal(r.values[1, 1:], t1.values[0])


@pytest.mark.registration
def test_itk_align_reference_time_padded_moving_channel():
    import itk
    from scipy.ndimage import gaussian_filter, shift

    from scallops.registration.itk import (
        itk_align_to_reference_time,
        set_automatic_transform_initialization,
    )

    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.random((256, 256)), 4).astype(np.float32)
    base = (base - base.min()) / (base.max() - base.min())
    other = gaussian_filter(rng.random((256, 256)), 4).astype(np.float32)
    # t1 is missing the leading channel, so padded channel 1 is t1's channel 0
    t0 = xr.DataArray(
        np.stack([other, base])[np.newaxis],
        dims=["t", "c", "y", "x"],
        coords=dict(t=[0], c=["a", "b"]),
    )
    t1 = xr.DataArray(
        shift(base, (4, -6), order=1, mode="nearest")[np.newaxis, np.newaxis],
        dims=["t", "c", "y", "x"],
        coords=dict(t=[1], c=["b"]),
    )
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap("translation")
    )
    set_automatic_transform_initialization(parameter_object, False)
    kwargs = dict(
        moving_image=[t0, t1],
        parameter_object=parameter_object,
        moving_image_spacing=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="empty"):
        itk_align_to_reference_time(moving_channel=0, **kwargs)

    result = itk_align_to_reference_time(moving_channel=1, **kwargs)
    assert result.sizes["t"] == 2 and result.sizes["c"] == 2
    np.testing.assert_array_equal(result.isel(t=1, c=0).values, 0)
    interior = (slice(20, -20), slice(20, -20))
    aligned = result.isel(t=1, c=1).values[interior]
    assert np.corrcoef(aligned.ravel(), base[interior].ravel())[0, 1] > 0.99


def test_combine_channels():
    from scallops.registration.itk import _combine_channels

    rng = np.random.default_rng(0)
    scales = np.array([1.0, 10.0, 100.0])[:, None, None]
    image = xr.DataArray(
        rng.random((3, 32, 32)) * scales, dims=["c", "y", "x"], attrs={"a": 1}
    )

    median = _combine_channels(image)
    np.testing.assert_allclose(
        median.values, np.median(image.values, axis=0), rtol=1e-6
    )
    assert median.dims == ("y", "x") and median.attrs == {"a": 1}
    # both methods give float32, also for integer channels
    assert median.dtype == np.float32
    assert _combine_channels(image.astype(np.uint16)).dtype == np.float32

    mean = _combine_channels(image, "mean")
    assert mean.dtype == np.float32 and mean.attrs == {"a": 1}
    # attrs are kept even where xarray drops them in arithmetic (older versions)
    with xr.set_options(keep_attrs=False):
        assert _combine_channels(image, "mean", weights=[1, 2, 1]).attrs == {"a": 1}
    np.testing.assert_allclose(mean.values, image.values.mean(axis=0), rtol=1e-5)
    weighted = _combine_channels(image, "mean", weights=[1, 3, 0])
    np.testing.assert_allclose(
        weighted.values,
        (image.values[0] + 3 * image.values[1]) / 4,
        rtol=1e-5,
    )

    # equalizing makes every channel contribute on a common scale
    equalized = _combine_channels(image, "mean", equalize=True)
    shares = [
        np.corrcoef(equalized.values.ravel(), image.values[c].ravel())[0, 1]
        for c in range(3)
    ]
    assert min(shares) > 0.4
    raw_shares = [
        np.corrcoef(mean.values.ravel(), image.values[c].ravel())[0, 1]
        for c in range(3)
    ]
    assert raw_shares[2] > 0.99 > 0.4 > raw_shares[0]
    # the channel order does not matter
    np.testing.assert_allclose(
        _combine_channels(image.transpose("y", "x", "c"), "mean", equalize=True)
        .transpose("y", "x")
        .values,
        equalized.values,
        rtol=1e-5,
    )
    # a single channel is returned as it is
    single = image.isel(c=[0])
    assert _combine_channels(single, "mean", equalize=True) is single


def test_equalize_channels_sparse_signal():
    from scallops.registration.itk import _equalize_channels

    image = np.zeros((2, 64, 64), np.float32)
    image[0] = np.random.default_rng(0).random((64, 64)) * 1000
    image[1, 8, 8] = 50  # a few bright pixels: 99.5th percentile == median
    image[1, 40, 16] = 50
    equalized = _equalize_channels(xr.DataArray(image, dims=["c", "y", "x"]))
    assert float(equalized.isel(c=1).max()) == pytest.approx(1.0)
    assert float(np.abs(equalized.isel(c=0)).max()) < 10


def test_equalize_channels_dask_matches_numpy():
    import dask.array as da

    from scallops.registration.itk import _equalize_channels

    rng = np.random.default_rng(0)
    # skewed, camera-like intensities: the bright-signal percentile is in a long tail
    values = rng.gamma(1.5, 200, (2, 256, 256)).astype(np.uint16)
    values[1] *= 10
    image = xr.DataArray(values, dims=["c", "y", "x"])
    for chunks in [(1, 32, 32), (2, 100, 33)]:
        lazy = image.copy(data=da.from_array(values, chunks=chunks))
        equalized = _equalize_channels(lazy)
        assert isinstance(equalized.data, da.Array)
        np.testing.assert_allclose(
            equalized.values, _equalize_channels(image).values, rtol=1e-6
        )


def test_equalize_step():
    from scallops.registration.itk import (
        _EQUALIZE_MAX_SAMPLE,
        _equalize_stats,
        _equalize_step,
    )

    # every pixel up to the sample size, then a stride that keeps it about that size
    assert _equalize_step(1024 * 1024) == 1
    assert _equalize_step(_EQUALIZE_MAX_SAMPLE) == 1
    for n in [_EQUALIZE_MAX_SAMPLE + 1, 50_000**2]:
        step = _equalize_step(n)
        assert step > 1 and n / step**2 <= _EQUALIZE_MAX_SAMPLE < n / (step - 1) ** 2

    # small images get the exact percentiles of every pixel
    values = np.random.default_rng(0).gamma(1.5, 200, (2, 300, 200)).astype(np.uint16)
    low, high, maximum = _equalize_stats(xr.DataArray(values, dims=["c", "y", "x"]))
    expected = np.percentile(values.reshape(2, -1), [50, 99.5], axis=1)
    np.testing.assert_array_equal(low, expected[0])
    np.testing.assert_array_equal(high, expected[1])
    np.testing.assert_array_equal(maximum, values.max(axis=(1, 2)))


def test_fixed_channel_weights():
    from scallops.cli.register import _fixed_channel_weights

    image = xr.DataArray(np.ones((3, 4, 4)), dims=["c", "y", "x"])
    assert _fixed_channel_weights(image, None) is None
    assert _fixed_channel_weights(image, [1, 2, 3]) == [1, 2, 3]
    assert _fixed_channel_weights(image.isel(c=[0]), [1, 2, 3]) is None
    assert _fixed_channel_weights(image.isel(c=0), [1, 2, 3]) is None
    with pytest.raises(ValueError, match="3 fixed channels"):
        _fixed_channel_weights(image, [1, 2])


def test_combine_channels_errors():
    from scallops.registration.itk import _combine_channels

    image = xr.DataArray(np.ones((3, 8, 8)), dims=["c", "y", "x"])
    with pytest.raises(ValueError, match="require the 'mean'"):
        _combine_channels(image, "median", weights=[1, 1, 1])
    with pytest.raises(ValueError, match="Expected 3 channel weights"):
        _combine_channels(image, "mean", weights=[1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        _combine_channels(image, "mean", weights=[0, 0, 0])
    with pytest.raises(ValueError, match="Unknown"):
        _combine_channels(image, "max")
    # also checked for a single channel, which is otherwise returned as it is
    with pytest.raises(ValueError, match="require the 'mean'"):
        _combine_channels(image.isel(c=[0]), "median", weights=[1])
    with pytest.raises(ValueError, match="Expected 1 channel weights"):
        _combine_channels(image.isel(c=0), "mean", weights=[1, 2, 3])


def test_itk_align_reference_time_checks_channel_weights():
    import itk

    from scallops.registration.itk import itk_align_to_reference_time

    image = xr.DataArray(
        np.ones((2, 3, 8, 8)), dims=["t", "c", "y", "x"], coords=dict(t=[0, 1])
    )
    with pytest.raises(ValueError, match="Expected 1 channel weights"):
        itk_align_to_reference_time(
            moving_image=image,
            moving_channel=0,
            parameter_object=itk.ParameterObject.New(),
            moving_image_spacing=(1.0, 1.0),
            channel_combine="mean",
            channel_weights=[1, 2, 3],
        )


@pytest.mark.registration
@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_itk_align_reference_time_int32_keeps_values(dtype):
    # int32/uint32 are resampled in float32; the result must be rounded and clipped
    # back to the dtype, not truncated or wrapped around
    import itk
    from scipy.ndimage import gaussian_filter, shift

    from scallops.registration.itk import (
        itk_align_to_reference_time,
        set_automatic_transform_initialization,
    )

    rng = np.random.default_rng(0)
    # sharp edges make the cubic B-spline overshoot below 0
    base = np.stack(
        [(gaussian_filter(rng.random((128, 128)), 2) > 0.5) * 4000.0 for _ in range(2)]
    )
    shifted = np.stack([shift(c, (3.5, -2.5), order=1, mode="nearest") for c in base])
    image = xr.DataArray(
        np.stack([base, shifted]).astype(dtype),
        dims=["t", "c", "y", "x"],
        coords=dict(t=[0, 1]),
    )
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap("translation")
    )
    set_automatic_transform_initialization(parameter_object, False)

    def align(img):
        return itk_align_to_reference_time(
            moving_image=img,
            moving_channel=[0, 1],
            parameter_object=parameter_object,
            moving_image_spacing=(1.0, 1.0),
        )

    result = align(image)
    assert result.dtype == dtype
    # the same registration in float32, rounded and clipped to the dtype
    expected = align(image.astype(np.float32)).isel(t=1).values
    info = np.iinfo(dtype)
    expected = np.clip(np.rint(expected), info.min, info.max)
    assert (expected < 0).any() == (dtype == np.int32)  # B-spline overshoot
    np.testing.assert_array_equal(result.isel(t=1).values, expected)


@pytest.mark.registration
@pytest.mark.parametrize("channel_combine", ["median", "mean"])
def test_itk_align_reference_time_combined_channels_keep_dtype(channel_combine):
    import itk
    from scipy.ndimage import gaussian_filter, shift

    from scallops.registration.itk import (
        itk_align_to_reference_time,
        set_automatic_transform_initialization,
    )

    rng = np.random.default_rng(0)
    base = np.stack([gaussian_filter(rng.random((128, 128)), 4) for _ in range(2)])
    base = ((base - base.min()) / (base.max() - base.min()) * 4000).astype(np.uint16)
    shifted = np.stack([shift(c, (3, -2), order=0, mode="nearest") for c in base])
    image = xr.DataArray(
        np.stack([base, shifted]), dims=["t", "c", "y", "x"], coords=dict(t=[0, 1])
    )
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap("translation")
    )
    set_automatic_transform_initialization(parameter_object, False)
    result = itk_align_to_reference_time(
        moving_image=image,
        moving_channel=[0, 1],
        parameter_object=parameter_object,
        moving_image_spacing=(1.0, 1.0),
        channel_combine=channel_combine,
        equalize_channels=channel_combine == "mean",
    )
    # the raw channels are written, not the (float) image combined for registration
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result.isel(t=0).values, base)


@pytest.mark.registration
@pytest.mark.parametrize("equalize", [False, True])
def test_itk_align_reference_time_combined_moving_channels(equalize):
    import itk
    from scipy.ndimage import gaussian_filter, shift

    from scallops.registration.itk import (
        itk_align_to_reference_time,
        set_automatic_transform_initialization,
    )

    rng = np.random.default_rng(0)

    def texture():
        a = gaussian_filter(rng.random((256, 256)), 4).astype(np.float32)
        return (a - a.min()) / (a.max() - a.min())

    # three channels with different structure and very different brightness; the
    # leading padded channel of t1 is empty
    channels = [texture() * 1000, texture() * 10, texture()]
    t0 = xr.DataArray(
        np.stack([texture(), *channels])[np.newaxis],
        dims=["t", "c", "y", "x"],
        coords=dict(t=[0], c=["dapi", "a", "b", "c"]),
    )
    t1 = xr.DataArray(
        np.stack([shift(c, (4, -6), order=1, mode="nearest") for c in channels])[
            np.newaxis
        ],
        dims=["t", "c", "y", "x"],
        coords=dict(t=[1], c=["a", "b", "c"]),
    )
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap("translation")
    )
    set_automatic_transform_initialization(parameter_object, False)
    result = itk_align_to_reference_time(
        moving_image=[t0, t1],
        moving_channel=[1, 2, 3],
        parameter_object=parameter_object,
        moving_image_spacing=(1.0, 1.0),
        channel_combine="mean",
        equalize_channels=equalize,
    )
    interior = (slice(20, -20), slice(20, -20))
    for c, original in zip((1, 2, 3), channels):
        aligned = result.isel(t=1, c=c).values[interior]
        assert np.corrcoef(aligned.ravel(), original[interior].ravel())[0, 1] > 0.98
