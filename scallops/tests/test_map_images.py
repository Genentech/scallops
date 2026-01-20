import pytest
import xarray as xr

from scallops.experiment.elements import Experiment
from scallops.experiment.util import map_images


@pytest.mark.io
def create_experiment(keys):
    images = {}
    for key in keys:
        images[key] = None
    return Experiment(images)


@pytest.mark.io
def test_map_images_intersection():
    exp1 = create_experiment(["a", "b", "c"])
    exp2 = create_experiment(["a", "b"])
    exp3 = create_experiment(["b"])

    def identity(data1: xr.DataArray, data2: xr.DataArray, data3: xr.DataArray):
        return data1

    result = map_images((exp1, exp2, exp3), identity)
    mapped_keys = list(result.images.keys())
    assert len(mapped_keys) == 1
    assert mapped_keys[0] == "b"


@pytest.mark.io
def test_map_images_key_order():
    exp = create_experiment(["a", "b", "c"])

    def identity(data: xr.DataArray):
        return data

    result = map_images(exp, identity)
    mapped_keys = list(result.images.keys())
    assert len(mapped_keys) == 3
    assert mapped_keys == ["a", "b", "c"]
