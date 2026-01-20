import numpy as np
import pytest
import xarray as xr

from scallops.xr import apply_data_array, dask_grouped_quantiles


@pytest.fixture
def image(array_A1_102_aln):
    return array_A1_102_aln.transpose(*("z", "c", "t", "y", "x")).rename(
        {"z": "t", "t": "z"}
    )  # ops swaps z and t in saved tif


@pytest.mark.io
def test_dask_grouped_quantiles(image):
    dask_image = image.squeeze()  # ops swaps z and t in saved tif
    dask_image = dask_image.chunk(dict(t=1, y=256, x=256))
    q = [0.5, 0.75]
    dask_results = dask_grouped_quantiles(dask_image, dims=["t", "c"], q=q).compute()
    results = image.squeeze().quantile(dim=["y", "x"], q=q)
    assert np.abs(dask_results.isel(quantile=0) - results.isel(quantile=0)).max() < 0.8
    assert np.abs(dask_results.isel(quantile=1) - results.isel(quantile=1)).max() < 1.75


@pytest.mark.io
def test_data_array(image):
    def add_data_array(x: xr.DataArray, y: float):
        return x + y

    _apply_data_array(add_data_array, image)


@pytest.mark.io
def test_numpy(image):
    def add_numpy(x: xr.DataArray, y: float):
        return (x + y).values

    _apply_data_array(add_numpy, image)


def _apply_data_array(f, image):
    result = apply_data_array(image, ["t", "c"], f, **dict(y=2))
    assert result.sizes == image.sizes
    assert ((result - 2) != image).sum() == 0
