import pytest
import xarray as xr

from scallops.xr import apply_data_array


@pytest.fixture
def image(array_A1_102_aln):
    return array_A1_102_aln.transpose(*("z", "c", "t", "y", "x")).rename(
        {"z": "t", "t": "z"}
    )  # ops swaps z and t in saved tif


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
