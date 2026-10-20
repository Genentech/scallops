import subprocess

import dask.array as da
import numpy as np
import pytest
import skimage.draw
import skimage.measure
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential

from scallops.experiment.elements import Experiment
from scallops.io import read_experiment
from scallops.segmentation import remove_boundary_labels
from scallops.segmentation.dask import _match_labels, segment_blocks
from scallops.segmentation.util import (
    close_labels,
    dask_relabel_sequential,
    identify_tertiary_objects,
    remove_labels,
    remove_masked_regions,
)
from scallops.segmentation.watershed import (
    segment_cells_watershed,
    segment_nuclei_watershed,
)


def jaccard(labels1, labels2):
    intersection = ((labels1 > 0) & (labels2 > 0)).sum()
    union = ((labels1 > 0) | (labels2 > 0)).sum()
    return intersection / union


@pytest.mark.parametrize("is_dask", [True, False])
@pytest.mark.utils
def test_identify_tertiary_objects(is_dask):
    primary_labels = np.zeros((5, 5), dtype=np.uint8)
    primary_labels[1:4, 1:4] = 1
    secondary_labels = np.zeros((5, 5), dtype=np.uint8)
    secondary_labels[1:4, 1:4] = 1
    if is_dask:
        primary_labels = da.from_array(primary_labels)
        secondary_labels = da.from_array(secondary_labels)
    assert np.all(
        identify_tertiary_objects(primary_labels, secondary_labels, False) == 0
    ), "Expected empty"
    expected_tertiary_labels = primary_labels.copy()
    expected_tertiary_labels[2, 2] = 0
    np.testing.assert_array_equal(
        expected_tertiary_labels,
        identify_tertiary_objects(primary_labels, secondary_labels, True),
        err_msg="Failed with shrink_primary",
    )

    secondary_labels2 = np.ones((5, 5), dtype=np.uint8)
    if is_dask:
        secondary_labels2 = da.from_array(secondary_labels2)
    np.testing.assert_array_equal(
        secondary_labels2 - primary_labels,
        identify_tertiary_objects(primary_labels, secondary_labels2, False),
    )
    np.testing.assert_array_equal(
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        ),
        identify_tertiary_objects(primary_labels, secondary_labels2, True),
        err_msg="Failed with shrink_primary secondary_labels2",
    )


@pytest.fixture(params=[False, True])
def image(experiment_c, request):
    param = request.param
    data = experiment_c.images["A1-102"].isel(z=0)
    if param:  # check to see if segmentation works for one and multiple t
        data = data.isel(t=[0])
    return data


@pytest.mark.utils
def test_dask_relabel_sequential():
    b = da.random.random_integers(0, 1000000, size=(500, 500), chunks=(50, 50))
    np.testing.assert_array_equal(
        relabel_sequential(b.compute())[0], dask_relabel_sequential(b).compute()
    )


@pytest.mark.segmentation_stardist
def test_segment_nuclei_stardist(image):
    pytest.importorskip("tensorflow")
    pytest.importorskip("stardist")

    import scallops.segmentation.stardist

    labels = scallops.segmentation.stardist.segment_nuclei_stardist(image)
    image_dask = image.copy(
        data=da.from_array(image.data, chunks=(1,) * (image.ndim - 2) + (256, 256))
    )
    labels_chunks = scallops.segmentation.stardist.segment_nuclei_stardist(image_dask)
    assert labels.max() > 2000
    assert labels.max() == labels_chunks.max()
    area = regionprops_table(labels, properties=("area",))["area"]
    area_chunked = regionprops_table(labels_chunks, properties=("area",))["area"]
    assert area_chunked.max() == area.max(), (
        f"Max area: {area_chunked.max()}, {area.max()}"
    )
    assert area_chunked.min() == area.min(), (
        f"Min area: {area_chunked.min()}, {area.min()}"
    )
    overlap = jaccard(labels, labels_chunks)
    assert overlap >= 0.99, overlap


@pytest.mark.segmentation_cellpose
def test_segment_nuclei_cellpose(experiment_c_A1_102_pheno):
    pytest.importorskip("cellpose")

    import scallops.segmentation.cellpose

    labels = scallops.segmentation.cellpose.segment_nuclei_cellpose(
        experiment_c_A1_102_pheno
    )
    image_dask = experiment_c_A1_102_pheno.copy(
        data=da.from_array(
            experiment_c_A1_102_pheno.data,
            chunks=(1,) * (experiment_c_A1_102_pheno.data.ndim - 2) + (256, 256),
        )
    )
    labels_chunks = scallops.segmentation.cellpose.segment_nuclei_cellpose(
        image_dask,
    ).compute()
    assert len(np.unique(labels_chunks)) - 1 == labels_chunks.max()
    assert labels.max() > 2000
    assert np.abs(labels.max() - labels_chunks.max()) < 100
    area = regionprops_table(labels, properties=("area",))["area"]
    area_chunked = regionprops_table(labels_chunks, properties=("area",))["area"]

    assert np.abs(area_chunked.max() - area.max()) < 7, (
        f"Area max {area_chunked.max()}, {area.max()}"
    )

    overlap = jaccard(labels, labels_chunks)
    assert overlap >= 0.9, overlap


@pytest.mark.segmentation_cellpose
def test_segment_cells_cellpose(experiment_c_A1_102_pheno):
    pytest.importorskip("cellpose")

    import scallops.segmentation.cellpose

    labels = scallops.segmentation.cellpose.segment_cells_cellpose(
        experiment_c_A1_102_pheno
    )
    assert labels.max() > 10


@pytest.mark.segmentation_watershed
def test_segment_nuclei_watershed_ops(experiment_c, experiment_c_A1_102_nuclei):
    imagec = experiment_c.images["A1-102"]
    labels = segment_nuclei_watershed(imagec.isel(t=0, z=0))
    nuclei = experiment_c_A1_102_nuclei.squeeze().data
    np.testing.assert_equal(labels, nuclei)


@pytest.mark.segmentation_watershed
def test_segment_cells_watershed_ops(
    experiment_c, experiment_c_A1_102_cells, experiment_c_A1_102_nuclei
):
    imagec = experiment_c.images["A1-102"].isel(z=0)
    nuclei = experiment_c_A1_102_nuclei.squeeze().data
    labels, _ = segment_cells_watershed(
        imagec,
        nuclei=nuclei,
        threshold=600,
        at_least_nuclei=False,
        watershed_method="binary",
    )
    labels = remove_boundary_labels(labels)
    known_good_labels = experiment_c_A1_102_cells.squeeze().data
    n = (labels.astype(bool) != known_good_labels.astype(bool)).sum()
    # slight differences due to versions of skimage
    assert n <= 10


@pytest.mark.segmentation_watershed
def test_segment_nuclei_watershed(image):
    labels = segment_nuclei_watershed(image)
    assert labels.max() > 10


@pytest.mark.segmentation_watershed
def test_segment_cells_watershed(image):
    labels = segment_nuclei_watershed(image)
    cell_labels, threshold = segment_cells_watershed(image, nuclei=labels)
    assert cell_labels.max() > 10


@pytest.mark.segmentation_propagation
def test_segment_cells_watershed_dask(image):
    labels = segment_nuclei_watershed(image)
    cell_labels, threshold = segment_cells_watershed(image, nuclei=labels)
    image = image.copy()
    image.data = da.from_array(image.data)
    cell_labels2, threshold = segment_cells_watershed(
        image, nuclei=labels, chunks=(256, 256)
    )
    assert jaccard(cell_labels, cell_labels2) > 0.9


@pytest.mark.segmentation_propagation
def test_segment_cells_propagation(image):
    import scallops.segmentation.propagation

    labels = segment_nuclei_watershed(image)
    cell_labels, threshold = (
        scallops.segmentation.propagation.segment_cells_propagation(
            image, nuclei=labels
        )
    )
    assert cell_labels.max() > 10


@pytest.fixture(params=[False, True])
def chunks(request):
    return request.param


@pytest.mark.segmentation_cellpose
def test_segment_cmd_cellpose(tmp_path, chunks):
    pytest.importorskip("cellpose")
    run_segment_nuclei_cmd(tmp_path, "cellpose", ["--chunks", "512"] if chunks else [])


@pytest.mark.segmentation_stardist
def test_segment_cmd_stardist(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("stardist")
    run_segment_nuclei_cmd(tmp_path, "stardist")


@pytest.mark.segmentation_stardist
def test_segment_nuclei_cmd_intensity_filter(tmp_path):
    """--max-intensity 0 must remove all nuclei: DAPI minimum in the test tile is 380."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("stardist")
    run_segment_nuclei_cmd(tmp_path, "stardist", ["--max-intensity", "0"])
    experiment = read_experiment(str(tmp_path / "test.zarr"))
    labels = experiment.labels["A1-102-nuclei"].squeeze().values
    assert labels.max() == 0, (
        f"Expected all labels removed with --max-intensity 0, got {labels.max()} max label"
    )


def run_segment_nuclei_cmd(tmp_path, segment_method, extra_args=None):
    if extra_args is None:
        extra_args = []
    tmp_path = str(tmp_path / "test.zarr")
    seg_args = [
        "scallops",
        "segment",
        "nuclei",
        "--method",
        segment_method,
        "--images",
        "scallops/tests/data/experimentC/input",
        "--groupby",
        "well",
        "tile",
        "--image-pattern",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--dapi-channel",
        "0",
        "--output=" + tmp_path,
        "--subset=A1-102",
    ]
    seg_args += extra_args
    subprocess.check_call(seg_args)


@pytest.mark.segmentation_watershed
def test_segment_cells_cmd(experiment_c, tmp_path):
    tmp_path = str(tmp_path / "test.zarr")
    image = experiment_c.images["A1-102"].isel(t=0, z=0)
    image.data = da.from_array(image.data)
    nuclei_labels = segment_nuclei_watershed(image)
    exp = Experiment()
    exp.labels["A1-102-nuclei"] = nuclei_labels
    exp.save(tmp_path)
    cell_labels, _ = segment_cells_watershed(image, nuclei=nuclei_labels)

    seg_args = [
        "scallops",
        "segment",
        "cell",
        "--nuclei-label",
        str(tmp_path),
        "--images",
        "scallops/tests/data/experimentC/input",
        "--groupby",
        "well",
        "tile",
        "--image-pattern",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--time",
        "0",
        "--output=" + tmp_path,
        "--subset=A1-102",
        "--method",
        "watershed",
    ]
    subprocess.check_call(seg_args)
    experiment = read_experiment(tmp_path)
    assert len(experiment.labels.keys()) == 3
    np.testing.assert_equal(cell_labels, experiment.labels["A1-102-cell"].values)


@pytest.mark.segmentation_watershed
def test_adaptive_threshold(image):
    labels = segment_nuclei_watershed(image)
    cell_labels, threshold = segment_cells_watershed(
        image,
        nuclei=labels,
        rm_small_std=3.5,
        threshold="quantile",
        at_least_nuclei=False,
        watershed_method="binary",
    )
    cell_labels = close_labels(cell_labels)
    if image.sizes["t"] == 1:
        assert (cell_labels == 0).sum() == 115347
    else:
        assert (cell_labels == 0).sum() == 129566


@pytest.mark.utils
@pytest.mark.parametrize("relabel", [True, False])
def test_remove_labels(relabel):
    labels = np.array([[1, 2, 3], [4, 5, 2]])
    filtered_labels = remove_labels(labels, [2, 4], relabel=relabel)
    expected = (
        np.array([[1, 0, 2], [0, 3, 0]])
        if relabel
        else np.array([[1, 0, 3], [0, 5, 0]])
    )
    np.testing.assert_array_equal(expected, filtered_labels)
    # input is not modified
    np.testing.assert_array_equal(np.array([[1, 2, 3], [4, 5, 2]]), labels)


@pytest.mark.utils
def test_filter_masked_regions():
    labels = np.array([[1, 2, 3], [4, 5, 2]])
    mask = np.array([[1, 0, 1], [1, 0, 1]])  # remove 2 and 5
    filtered_labels = remove_masked_regions(labels, mask)
    np.testing.assert_array_equal(np.array([[1, 0, 3], [4, 0, 0]]), filtered_labels)


def _random_disks(shape, n, max_radius, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=bool)
    for _ in range(n):
        center = [rng.integers(0, s) for s in shape]
        rr, cc = skimage.draw.disk(center, rng.integers(3, max_radius), shape=shape)
        image[rr, cc] = True
    return image


def _assert_same_objects(expected, labels):
    """Assert labels are sequential and in one-to-one correspondence with expected."""
    np.testing.assert_array_equal(np.unique(labels), np.arange(labels.max() + 1))
    np.testing.assert_array_equal(expected > 0, labels > 0)
    pairs = np.unique(np.stack([expected.ravel(), labels.ravel()], axis=1), axis=0)
    assert len(pairs) == expected.max() + 1 == labels.max() + 1


@pytest.mark.utils
@pytest.mark.parametrize("chunks", [(64, 70), (300, 317), (40, 100), 15])
def test_segment_blocks(chunks):
    image = _random_disks((300, 317), 60, 12)
    expected = skimage.measure.label(image)
    labels = segment_blocks(
        da.from_array(image, chunks=chunks), skimage.measure.label, depth=25
    )
    assert labels.shape == image.shape
    assert labels.dtype == np.int32
    result = labels.compute()
    assert result.dtype == np.int32
    _assert_same_objects(expected, result)


@pytest.mark.utils
def test_segment_blocks_3d():
    rng = np.random.default_rng(0)
    image = np.zeros((40, 60, 60), dtype=bool)
    for z, y, x in zip(*[rng.integers(3, s - 3, 20) for s in image.shape]):
        image[z - 2 : z + 3, y - 2 : y + 3, x - 2 : x + 3] = True
    expected = skimage.measure.label(image)
    labels = segment_blocks(
        da.from_array(image, chunks=20),
        skimage.measure.label,
        depth=10,
        spatial_dims=3,
    ).compute()
    _assert_same_objects(expected, labels)


@pytest.mark.utils
def test_segment_blocks_args_and_kwargs():
    image = _random_disks((200, 200), 40, 10, seed=1)
    mask = np.ones(image.shape, dtype=bool)
    mask[:, :100] = False
    expected = skimage.measure.label(image & mask)

    def seg_fn(block, mask_block, threshold):
        assert block.shape[0] == 2  # channel dimension is passed whole
        return skimage.measure.label((block[0] > threshold) & mask_block)

    labels = segment_blocks(
        da.from_array(np.stack([image, image]), chunks=(1, 50, 50)),
        seg_fn,
        da.from_array(mask, chunks=30),
        depth=20,
        threshold=0.5,
    )
    assert labels.shape == image.shape
    _assert_same_objects(expected, labels.compute())


@pytest.mark.utils
def test_segment_blocks_touching_cells_not_merged():
    # Two distinct cells touching exactly at the block border are not merged
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:30, 5:20] = 1
    image[10:30, 20:35] = 2
    labels = segment_blocks(
        da.from_array(image, chunks=20), lambda block: block, depth=10
    ).compute()
    np.testing.assert_array_equal(image, labels)


@pytest.mark.utils
def test_segment_blocks_empty():
    labels = segment_blocks(
        da.zeros((50, 50), chunks=20), skimage.measure.label, depth=5
    ).compute()
    np.testing.assert_array_equal(labels, 0)


@pytest.mark.utils
def test_segment_blocks_invalid_args():
    with pytest.raises(ValueError):
        segment_blocks(da.zeros((50, 50), chunks=20), skimage.measure.label, depth=(5,))
    with pytest.raises(ValueError):
        segment_blocks(
            da.zeros((50, 50), chunks=20),
            skimage.measure.label,
            da.zeros((40, 50), chunks=20),
        )


@pytest.mark.utils
def test_match_labels():
    a = np.array([[1, 1, 1, 1, 0, 2, 2, 2]])
    b = np.array([[3, 3, 3, 0, 0, 4, 5, 5]])
    # 1-3: IoU 3/4, 2-4: IoU 1/3, 2-5: IoU 2/3
    np.testing.assert_array_equal(_match_labels(a, b, 0.5), [[1, 3], [2, 5]])
    np.testing.assert_array_equal(_match_labels(a, b, 0.3), [[1, 3], [2, 4], [2, 5]])
    assert _match_labels(a, np.zeros_like(b), 0.5).shape == (0, 2)


@pytest.mark.utils
def test_segment_blocks_segments_each_block_once():
    ncalls = []

    def seg_fn(block):
        ncalls.append(1)
        return skimage.measure.label(block)

    image = da.from_array(_random_disks((200, 200), 20, 10), chunks=50)
    image = image.map_overlap(lambda b: b, depth=1, boundary="none")
    segment_blocks(image, seg_fn, depth=20).compute(scheduler="sync")
    assert len(ncalls) == 16
