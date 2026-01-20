import dask.array as da
import numpy as np
import pytest
from scipy.sparse import coo_array

from scallops.segmentation.util import area_overlap, overlap_to_iou


@pytest.mark.features
def test_area_overlap(array_A1_102_cells, array_A1_102_nuclei):
    array_A1_102_cells = array_A1_102_cells.squeeze().data
    array_A1_102_nuclei = array_A1_102_nuclei.squeeze().data
    array_A1_102_cells_ = da.from_array(array_A1_102_cells, chunks=(50, 50))
    array_A1_102_nuclei_ = da.from_array(array_A1_102_nuclei, chunks=(50, 50))
    result_df = area_overlap(array_A1_102_nuclei_, array_A1_102_cells_).compute()
    mask = array_A1_102_nuclei == 17
    area = np.sum(mask)
    overlap = (array_A1_102_cells[mask] == 17).sum()
    result_df_subset = result_df.query("x==17")
    assert len(result_df_subset) == 1
    assert area == 71 == result_df_subset["area"].values[0]
    assert overlap == 67 == result_df_subset["overlap"].values[0]


@pytest.mark.features
def test_overlap_to_iou(array_A1_102_cells, array_A1_102_nuclei):
    array_A1_102_cells = array_A1_102_cells.squeeze().data
    array_A1_102_nuclei = array_A1_102_nuclei.squeeze().data
    x = array_A1_102_nuclei.ravel()
    y = array_A1_102_cells.ravel()
    np_overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        np_overlap[x[i], y[i]] += 1
    assert np_overlap[17, 17] == 67
    np_iou = overlap_to_iou(np_overlap)
    assert (
        np_iou[17, 17]
        == overlap_to_iou(coo_array(np_overlap)).tocsr()[17, 17]
        == 67 / 99
    )
