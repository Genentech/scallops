import os
from subprocess import check_call

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from array_api_compat import get_namespace
from scipy.sparse import coo_array, issparse, sparray

from scallops.io import get_image_spacing
from scallops.segmentation.util import (
    assign_labels_by_overlap,
    label_overlap,
    relabel_by_assignment,
)
from scallops.zarr_io import _write_zarr_labels, open_ome_zarr, read_ome_zarr_array


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
    df = label_overlap(x, y).compute()
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

    # labels in label_image_2 without any overlaps have label_1 set to 0
    no_overlap_df = df[df["label_1"] == 0]
    labels_2 = np.unique(y[y != 0])
    np.testing.assert_equal(
        no_overlap_df["label_2"].values,
        labels_2[np_overlap[1:, labels_2].sum(axis=0) == 0],
    )
    assert (no_overlap_df[["overlap", "iou", "fraction_overlap"]] == 0).all().all()
    # labels in label_image_1 without any overlaps have label_2 set to 0
    no_overlap_df = df[df["label_2"] == 0]
    labels_1 = np.unique(x[x != 0])
    np.testing.assert_equal(
        no_overlap_df["label_1"].values,
        labels_1[np_overlap[labels_1, 1:].sum(axis=1) == 0],
    )
    assert (no_overlap_df[["overlap", "iou", "fraction_overlap"]] == 0).all().all()
    df = df[(df["label_1"] != 0) & (df["label_2"] != 0)]

    np_iou[0, :] = 0
    np_iou[:, 0] = 0
    i, j = np.nonzero(np_iou > 0)
    np.testing.assert_equal(df["label_1"].values, i)
    np.testing.assert_equal(df["label_2"].values, j)
    np.testing.assert_equal(df["iou"].values, np_iou[i, j])
    np.testing.assert_equal(
        df["fraction_overlap"].values, np_overlap[i, j] / np_overlap.sum(axis=1)[i]
    )


@pytest.mark.features
def test_assign_labels_by_overlap():
    overlap_df = pd.DataFrame(
        {
            "label_1": [1, 1, 2, 2, 2, 3],
            "label_2": [5, 6, 7, 8, 9, 5],
            "fraction_overlap": [10, 20, 5, 5, 3, 4],
            "iou": [0.1, 0.2, 0.3, 0.5, 0.1, 0.4],
        }
    )
    df = assign_labels_by_overlap(overlap_df)
    # label 2 has a tie in overlap that is broken by IoU. Cells 7 and 9 only overlap
    # label 2, which is assigned to cell 8, so they are included with label_1 set to 0
    np.testing.assert_equal(df["label_1"].values, [1, 2, 3, 0, 0])
    np.testing.assert_equal(df["label_2"].values, [6, 8, 5, 7, 9])
    assert (df.query("label_1 == 0")[["iou", "fraction_overlap"]] == 0).all().all()


@pytest.mark.features
@pytest.mark.parametrize("npartitions", [1, 3, 7])
def test_assign_labels_by_overlap_dask(npartitions):
    rng = np.random.default_rng(0)
    n = 5000
    overlap_df = pd.DataFrame(
        {
            "label_1": rng.integers(1, 500, n),
            "label_2": rng.integers(1, 1000, n),
            "fraction_overlap": rng.integers(1, 10, n) / 10,
            "iou": rng.random(n),
        }
    ).drop_duplicates(["label_1", "label_2"])
    expected = assign_labels_by_overlap(overlap_df)
    ddf = dd.from_pandas(
        overlap_df.sample(frac=1, random_state=0), npartitions=npartitions
    )
    result = assign_labels_by_overlap(ddf)
    assert isinstance(result, dd.DataFrame)
    sort_by = ["label_1", "label_2"]
    result = result.compute().sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    assert (expected["label_1"] == 0).any()
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.features
@pytest.mark.parametrize("use_dask", [False, True])
def test_relabel_by_assignment(
    experiment_c_A1_102_cells, experiment_c_A1_102_nuclei, use_dask
):
    cells = experiment_c_A1_102_cells.squeeze().data
    nuclei = experiment_c_A1_102_nuclei.squeeze().data.copy()
    # remove nuclei in cell 17 so that cell 17 doesn't overlap any nuclei
    nuclei[np.isin(nuclei, np.unique(nuclei[cells == 17]))] = 0
    # add a nucleus outside of any cell with the largest nucleus label
    orphan_nucleus = int(nuclei.max()) + 1
    nuclei[tuple(np.argwhere(cells == 0)[0])] = orphan_nucleus
    overlap_df = label_overlap(
        da.from_array(nuclei, chunks=(50, 50)), da.from_array(cells, chunks=(50, 50))
    ).compute()
    # cells without nuclei are included in label_overlap with label_1 set to 0
    assert overlap_df.query("label_2 == 17")["label_1"].tolist() == [0]
    assignment_df = assign_labels_by_overlap(overlap_df)
    # every cell is included, with label_1 set to 0 if no nucleus is assigned to it
    np.testing.assert_equal(
        np.sort(assignment_df["label_2"].unique()), np.unique(cells)
    )
    assert assignment_df.query("label_2 == 17")["label_1"].tolist() == [0]
    # nuclei outside of cells are assigned to background
    assert assignment_df.query(f"label_1 == {orphan_nucleus}")["label_2"].tolist() == [
        0
    ]
    offset = orphan_nucleus + 1
    labels = da.from_array(cells, chunks=(40, 60)) if use_dask else cells
    result = relabel_by_assignment(labels, assignment_df)
    assert isinstance(result, da.Array) == use_dask
    result = np.asarray(result)

    assert result.shape == cells.shape
    np.testing.assert_equal(result == 0, cells == 0)
    # one-to-one mapping between old and new labels
    pairs = np.unique(np.stack([cells.ravel(), result.ravel()]), axis=1)
    assert len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]
    mapping = dict(zip(pairs[0], pairs[1]))

    assignment_df = (
        assignment_df[(assignment_df["label_1"] != 0) & (assignment_df["label_2"] != 0)]
        .sort_values(
            ["label_2", "fraction_overlap", "label_1"], ascending=[True, False, True]
        )
        .drop_duplicates("label_2")
    )
    for label_1, label_2 in zip(assignment_df["label_1"], assignment_df["label_2"]):
        assert mapping[label_2] == label_1
    # cells without an assigned nucleus get new labels after the largest nucleus
    unassigned = np.setdiff1d(np.unique(cells[cells != 0]), assignment_df["label_2"])
    assert len(unassigned) > 0
    np.testing.assert_equal(
        [mapping[c] for c in unassigned], np.arange(offset, offset + len(unassigned))
    )


def _check_renumbered_cells(
    cells: np.ndarray,
    nuclei: np.ndarray,
    renumbered_cells: np.ndarray,
    assignment_df: pd.DataFrame,
):
    assert renumbered_cells.shape == cells.shape
    np.testing.assert_equal(renumbered_cells == 0, cells == 0)
    # one-to-one mapping between old and new cell labels
    pairs = np.unique(np.stack([cells.ravel(), renumbered_cells.ravel()]), axis=1)
    assert len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]
    mapping = dict(zip(pairs[0], pairs[1]))
    cell_labels = pairs[0][pairs[0] != 0]

    # cells with a nucleus take the label of the nucleus with the highest fraction
    # overlap, with ties broken by the smallest nucleus label
    assigned = assignment_df.query("label_1 != 0 and label_2 != 0")
    for cell, group in assigned.groupby("label_2"):
        best = group[group["fraction_overlap"] == group["fraction_overlap"].max()]
        assert mapping[cell] == best["label_1"].min()

    # cells without a nucleus get unique labels after the largest nucleus label, so
    # they don't match any nucleus
    unassigned = np.setdiff1d(cell_labels, assigned["label_2"])
    np.testing.assert_equal(
        np.sort(unassigned),
        np.sort(assignment_df.query("label_1 == 0")["label_2"].values),
    )
    max_nucleus = nuclei.max()
    np.testing.assert_equal(
        [mapping[cell] for cell in unassigned],
        np.arange(max_nucleus + 1, max_nucleus + 1 + len(unassigned)),
    )


@pytest.mark.features
@pytest.mark.parametrize("large_labels", [False])
def test_relabel_by_assignment_missing_label(large_labels):
    # large labels use map_array instead of a lookup table
    offset = 2**30 if large_labels else 0
    cells = np.array([[0, 5, 5, 6], [7, 7, 0, 9]])
    cells = np.where(cells != 0, cells + offset, 0)

    assignment_df = pd.DataFrame(
        {
            "label_1": [1, 2, 0],
            "label_2": np.array([5, 6, 7]) + offset,
            "fraction_overlap": [1.0, 1.0, 0.0],
        }
    )
    with pytest.raises(IndexError):
        relabel_by_assignment(cells, assignment_df)
    # a missing label smaller than the largest label in assignment_df does not raise error
    # with pytest.raises(IndexError):
    #     relabel_by_assignment(
    #         np.where(cells == 9 + offset, 3 + offset, cells), assignment_df
    #     )


@pytest.mark.features
def test_label_overlap_cli(
    experiment_c_A1_102_cells, experiment_c_A1_102_nuclei, tmpdir
):
    cells = experiment_c_A1_102_cells.squeeze().data
    nuclei = experiment_c_A1_102_nuclei.squeeze().data.copy()

    # remove the nucleus in cell 19 so that cell 19 doesn't overlap any nuclei
    nuclei[np.isin(nuclei, np.unique(nuclei[cells == 19]))] = 0
    # add a nucleus that doesn't overlap any cells
    nuclei[tuple(np.argwhere(cells == 0)[0])] = nuclei.max() + 1  # 2644

    filtered_cells = cells.copy()
    filtered_cells[filtered_cells == 17] = 0  # remove cell 17
    x = da.from_array(cells, chunks=(50, 50))
    y = da.from_array(nuclei, chunks=(40, 60))
    z = da.from_array(filtered_cells, chunks=(40, 60))

    labels_path = str(tmpdir / "test.zarr")
    output_meta_path = str(tmpdir / "output")
    output_zarr_path = str(tmpdir / "output.zarr")
    labels_root = open_ome_zarr(labels_path, mode="w")
    for name, labels in {
        "test-nuclei": y,
        "test-cell": x,
        "test-cytosol": z,
    }.items():
        _write_zarr_labels(
            name=name,
            root=labels_root,
            labels=labels,
            metadata=dict(physical_pixel_sizes=[0.5, 0.25], scallops_version="old"),
            group_metadata={"image-label": {"source": {"image": "../../images/test"}}},
        )
    cmd = [
        "scallops",
        "segment",
        "overlap",
        "--labels",
        labels_path,
        "--meta-output",
        output_meta_path,
        "--label-output",
        output_zarr_path,
        "--label-suffix-1",
        "nuclei",
        "--label-suffix-2",
        "cell",
        "--label-pattern",
        "{well}",
        "--additional-suffix",
        "cytosol",
    ]
    check_call(cmd)
    overlap_df = pd.read_parquet(output_meta_path + "/test-overlap.parquet")
    assert (
        overlap_df.query("label_1 == 17 and label_2 == 17")["iou"].values[0] == 67 / 99
    )

    # no background pairs
    assert ((overlap_df["label_1"] != 0) | (overlap_df["label_2"] != 0)).all()

    assignment_df = pd.read_parquet(output_meta_path + "/test-assignment.parquet")
    # cell 19 does not overlap any nuclei
    assert len(assignment_df.query("label_1 == 0")) == 1
    assert assignment_df.query("label_2 == 19")["label_1"].tolist() == [0]

    # nuclei 2644 does not overlap any cells
    assert len(assignment_df.query("label_1 == 2644")) == 1
    assert assignment_df.query("label_1 == 2644")["label_2"].tolist() == [0]

    renumbered_cells = read_ome_zarr_array(
        os.path.join(output_zarr_path, "labels", "test-cell")
    ).values
    _check_renumbered_cells(
        cells=cells,
        nuclei=nuclei,
        renumbered_cells=renumbered_cells,
        assignment_df=assignment_df,
    )
    # check metadata
    for name in ("test-cell", "test-cytosol"):
        relabeled = read_ome_zarr_array(os.path.join(output_zarr_path, "labels", name))
        assert get_image_spacing(relabeled.attrs) == (0.5, 0.25)
        assert relabeled.attrs["scallops_version"] != "old"

    renumbered_cytosol = read_ome_zarr_array(
        os.path.join(output_zarr_path, "labels", "test-cytosol")
    ).values
    # cells and cytosol are equal except for cell 17
    np.testing.assert_equal(
        renumbered_cytosol[cells != 17], renumbered_cells[cells != 17]
    )

    # all cells are included, with label_1 set to 0 if no nucleus is assigned to them
    cells_without_nuclei = assignment_df.query("label_1 == 0")
    assignment_df = assignment_df.query("label_1 != 0")
    assert not cells_without_nuclei["label_2"].isin(assignment_df["label_2"]).any()
    assert assignment_df["label_1"].is_unique
    # all nuclei are included, with label_2 set to 0 if they don't overlap any cell
    np.testing.assert_equal(
        np.sort(assignment_df["label_1"].values), np.unique(nuclei[nuclei != 0])
    )
    np.testing.assert_equal(
        np.sort(assignment_df.query("label_2 != 0")["label_1"].values),
        np.unique(nuclei[(nuclei != 0) & (cells != 0)]),
    )
    df = overlap_df[overlap_df["label_1"] != 0].reset_index(drop=True)
    expected = (
        df.loc[
            df.groupby("label_1")["fraction_overlap"].idxmax(),
            ["label_1", "fraction_overlap"],
        ]
        .sort_values("label_1")
        .reset_index(drop=True)
    )
    assignment_df = assignment_df.sort_values("label_1").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        assignment_df[["label_1", "fraction_overlap"]], expected, check_dtype=False
    )
