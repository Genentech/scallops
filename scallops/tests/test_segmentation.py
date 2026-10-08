import subprocess

import dask.array as da
import numpy as np
import pytest
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential

from scallops.experiment.elements import Experiment
from scallops.io import read_experiment
from scallops.segmentation import remove_boundary_labels
from scallops.segmentation.util import (
    close_labels,
    dask_relabel_sequential,
    identify_tertiary_objects,
    remove_labels_by_intensity,
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
    assert area_chunked.max() == area.max(), f"{area_chunked.max()}, {area.max()}"
    assert area_chunked.min() == area.min(), f"{area_chunked.min()}, {area.min()}"
    overlap = jaccard(labels, labels_chunks)
    assert overlap >= 0.99, overlap


@pytest.mark.segmentation_cellpose
def test_segment_nuclei_cellpose(image):
    pytest.importorskip("cellpose")

    import scallops.segmentation.cellpose

    labels = scallops.segmentation.cellpose.segment_nuclei_cellpose(image, diameter=8)
    image_dask = image.copy(
        data=da.from_array(image.data, chunks=(1,) * (image.ndim - 2) + (256, 256))
    )
    labels_chunks = scallops.segmentation.cellpose.segment_nuclei_cellpose(
        image_dask,
        diameter=8,
    ).compute()
    assert len(np.unique(labels_chunks)) - 1 == labels_chunks.max()
    assert labels.max() > 2000
    assert np.abs(labels.max() - labels_chunks.max()) < 100
    area = regionprops_table(labels, properties=("area",))["area"]
    area_chunked = regionprops_table(labels_chunks, properties=("area",))["area"]

    assert area_chunked.max() == area.max(), f"{area_chunked.max()}, {area.max()}"
    assert area_chunked.min() == area.min(), f"{area_chunked.min()}, {area.min()}"
    overlap = jaccard(labels, labels_chunks)
    assert overlap >= 0.9, overlap


@pytest.mark.segmentation_cellpose
def test_segment_cells_cellpose(image):
    pytest.importorskip("cellpose")

    import scallops.segmentation.cellpose

    labels = scallops.segmentation.cellpose.segment_cells_cellpose(image)
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
def test_remove_labels_by_intensity(
    experiment_c_A1_102_nuclei, experiment_c_A1_102_pheno_aligned
):
    """Remove nuclei whose mean DAPI intensity falls outside a given range.

    Uses the pre-computed nuclei labels for tile A1-102 paired with the aligned
    phenotype image (channel 0 = DAPI).  The 90th-percentile mean intensity
    across all 2 643 labels is ~1 727 counts; filtering above that value should
    remove exactly 265 labels while keeping 2 378.
    """
    from skimage.measure import regionprops

    labels = experiment_c_A1_102_nuclei.squeeze().values.astype(np.int32)
    dapi = (
        experiment_c_A1_102_pheno_aligned.isel(c=0).squeeze().values.astype(np.float32)
    )

    regions = regionprops(labels, intensity_image=dapi)
    means = np.array([r.intensity_mean for r in regions])
    threshold = np.percentile(means, 90)  # ~1 727 counts

    # max_intensity: keep only dim nuclei, drop the bright ones (top 10 %)
    result = remove_labels_by_intensity(labels, dapi, intensity_max=threshold)
    n_kept = len(np.unique(result[result > 0]))
    n_removed = labels.max() - n_kept
    assert n_kept == 2378, f"Expected 2378 kept labels, got {n_kept}"
    assert n_removed == 265, f"Expected 265 removed labels, got {n_removed}"

    # min_intensity: keep only bright nuclei, drop the dim ones
    result_bright = remove_labels_by_intensity(labels, dapi, intensity_min=threshold)
    n_kept_bright = len(np.unique(result_bright[result_bright > 0]))
    assert n_kept_bright == 265, f"Expected 265 bright labels kept, got {n_kept_bright}"

    # no bounds: all labels survive
    result_all = remove_labels_by_intensity(labels, dapi)
    np.testing.assert_array_equal(result_all, labels)


@pytest.mark.utils
def test_filter_masked_regions():
    labels = np.array([[1, 2, 3], [4, 5, 2]])
    mask = np.array([[1, 0, 1], [1, 0, 1]])  # remove 2 and 5
    filtered_labels = remove_masked_regions(labels, mask)
    np.testing.assert_array_equal(np.array([[1, 0, 3], [4, 0, 0]]), filtered_labels)
