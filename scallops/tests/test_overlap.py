from subprocess import check_call

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from array_api_compat import get_namespace
from scipy.sparse import coo_array, issparse, sparray

from scallops import Experiment
from scallops.segmentation.util import (
    label_overlap_iou,
)


def _label_overlap_to_iou(
    overlap: np.ndarray | sparray | da.Array,
) -> np.ndarray | sparray:
    if issparse(overlap):  # no keepdims
        n_pixels_x = overlap.sum(axis=0)
        n_pixels_true = overlap.sum(axis=1)
        n_pixels_x.resize((1,) + n_pixels_x.shape)
        n_pixels_true.resize(n_pixels_true.shape + (1,))
    else:
        xp = get_namespace(overlap)
        n_pixels_x = xp.sum(overlap, axis=0, keepdims=True)
        n_pixels_true = xp.sum(overlap, axis=1, keepdims=True)
    return overlap / (n_pixels_x + n_pixels_true - overlap)


@pytest.mark.features
def test_label_overlap_iou(experiment_c_A1_102_cells, experiment_c_A1_102_nuclei):
    x = da.from_array(experiment_c_A1_102_cells.squeeze().data, chunks=(50, 50))
    y = da.from_array(experiment_c_A1_102_nuclei.squeeze().data, chunks=(40, 60))
    df = label_overlap_iou(x, y).compute()
    df = df.sort_values(["label_1", "label_2"])

    assert df.query("label_1 == 17 and label_2 == 17")["iou"].values[0] == 67 / 99

    x = experiment_c_A1_102_cells.squeeze().data.ravel()
    y = experiment_c_A1_102_nuclei.squeeze().data.ravel()
    np_overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        np_overlap[x[i], y[i]] += 1
    assert np_overlap[17, 17] == 67
    np_iou = _label_overlap_to_iou(np_overlap)
    assert (
        np_iou[17, 17]
        == _label_overlap_to_iou(coo_array(np_overlap)).tocsr()[17, 17]
        == 67 / 99
    )

    np_iou[0, :] = 0
    np_iou[:, 0] = 0
    i, j = np.nonzero(np_iou > 0)
    np.testing.assert_equal(df["label_1"].values, i)
    np.testing.assert_equal(df["label_2"].values, j)
    np.testing.assert_equal(df["iou"].values, np_iou[i, j])


@pytest.mark.features
def test_label_overlap_iou_cli(
    experiment_c_A1_102_cells, experiment_c_A1_102_nuclei, tmpdir
):
    x = da.from_array(experiment_c_A1_102_cells.squeeze().data, chunks=(50, 50))
    y = da.from_array(experiment_c_A1_102_nuclei.squeeze().data, chunks=(40, 60))
    labels_path = str(tmpdir / "test.zarr")
    output_path = str(tmpdir / "output")
    Experiment(labels={"test-nuclei": y, "test-cell": x}).save(labels_path)
    cmd = [
        "scallops",
        "segment",
        "overlap",
        "--labels",
        labels_path,
        "--output",
        output_path,
        "--label-suffix-1",
        "nuclei",
        "--label-suffix-2",
        "cell",
        "--label-pattern",
        "{well}",
    ]
    check_call(cmd)
    df = pd.read_parquet(output_path + "/test-overlap.parquet")
    assert df.query("label_1 == 17 and label_2 == 17")["iou"].values[0] == 67 / 99
