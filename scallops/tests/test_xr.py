import pytest
import xarray as xr

from scallops.xr import apply_data_array


@pytest.mark.io
def test_data_array(experiment_c_A1_102_aligned):
    def add_data_array(x: xr.DataArray, y: float):
        return x + y

    _apply_data_array(add_data_array, experiment_c_A1_102_aligned)


@pytest.mark.io
def test_numpy(experiment_c_A1_102_aligned):
    def add_numpy(x: xr.DataArray, y: float):
        return (x + y).values

    _apply_data_array(add_numpy, experiment_c_A1_102_aligned)


def _apply_data_array(f, experiment_c_A1_102_aligned):
    result = apply_data_array(experiment_c_A1_102_aligned, ["t", "c"], f, **dict(y=2))
    assert result.sizes == experiment_c_A1_102_aligned.sizes
    assert ((result - 2) != experiment_c_A1_102_aligned).sum() == 0
