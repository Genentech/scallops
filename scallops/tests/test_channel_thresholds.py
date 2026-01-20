import dask.array as da
import numpy as np
import pytest
import xarray as xr

import scallops.experiment.util
import scallops.io
from scallops.experiment.elements import Experiment
from scallops.visualize.utils import channel_thresholds


@pytest.mark.basecalls
def test_thresholds():
    dims = ["t", "c", "z", "y", "x"]
    nimages = 3
    data_array_dict = {}
    dask_data_array_dict = {}
    channels = np.arange(4)
    arrays = []
    for i in range(nimages):
        data = np.random.random((1, 4, 1, 100, 50))
        arrays.append(data)
        data_array_dict["test-{}".format(i)] = xr.DataArray(
            data, dims=dims, coords={"c": channels}
        )

        dask_data_array_dict["test-{}".format(i)] = xr.DataArray(
            da.from_array(data), dims=dims, coords={"c": channels}
        )

    data = np.stack(arrays)
    ds = scallops.experiment.util._concat_images(Experiment(data_array_dict))
    dask_ds = scallops.experiment.util._concat_images(Experiment(dask_data_array_dict))
    percentile_min = 10
    percentile_max = 90
    result = channel_thresholds(
        ds,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
        pad_min=0,
        pad_max=0,
    )
    dask_result = channel_thresholds(
        dask_ds,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
        pad_min=0,
        pad_max=0,
    )
    for i in range(len(channels)):
        assert channels[i] in result
        assert channels[i] in dask_result
        np_result = np.percentile(data[:, :, i], q=[percentile_min, percentile_max])
        np.testing.assert_equal(np_result, result[channels[i]])
        dask_thresholds = dask_result[channels[i]]
        dask_thresholds = dask_thresholds[0].compute(), dask_thresholds[1].compute()
        np.testing.assert_almost_equal(np_result, dask_thresholds, 2)
