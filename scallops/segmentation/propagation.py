from typing import Literal, Sequence

import dask.array as da
import numpy as np
import xarray as xr
from centrosome.cpmorphology import fill_labeled_holes

from scallops.segmentation import _propagate
from scallops.segmentation.util import (
    close_labels,
    image2mask,
    remove_small_objects_std,
)


def _process_block(
    image: np.ndarray,
    nuclei: np.ndarray,
    mask: np.ndarray,
    weight: float,
    fill_holes: bool,
    closing_radius: int | None,
) -> np.ndarray:
    """Process a block of the input image for label propagation.

    This function applies label propagation using the provided nuclei, mask, and weight, followed by
    optional hole filling and label closing operations.

    :param image: Input image block as a NumPy array.
    :param nuclei: Nuclei labels as a NumPy array.
    :param mask: Binary mask indicating regions of interest.
    :param weight: Weight used for label propagation.
    :param fill_holes: Whether to fill holes in the labeled regions.
    :param closing_radius: Radius for the morphological closing operation (if None, closing is
        skipped).
    :return: Propagated labels as a NumPy array.
    """
    labels, _ = propagate(
        image=image,
        labels=nuclei,
        mask=mask,
        weight=weight,
    )

    if fill_holes:
        labels = fill_labeled_holes(labels, mask=labels == 0)

    if closing_radius is not None:
        labels = close_labels(labels, disk_radius=closing_radius)
        labels[nuclei > 0] = nuclei[nuclei > 0]
    return labels


def segment_cells_propagation(
    image: xr.DataArray,
    nuclei: np.ndarray,
    threshold: Literal["Li", "Otsu", "Local"] | float = "Li",
    cyto_channel: int | Sequence[int] | None = None,
    threshold_correction_factor: float = 1,
    nuclei_channel: int = 0,
    rm_small_std: float | None = None,
    rolling_ball: bool = False,
    sigma: float | None = None,
    closing_radius: int | None = None,
    t: list[int] | int | None = 0,
    regularization_factor: float = 0.05,
    fill_holes: bool = True,
    chunks: tuple[int, int] | None = None,
    depth: int = 30,
) -> tuple[np.ndarray, float | np.ndarray | None]:
    """Segment cells using propagation algorithm, matching cell labels to nuclei labels.

    :param image: Image with the dimensions (t, c, z, y, x)
    :param nuclei: Labeled segmentation mask of nuclei. Uses nuclei as seeds and matches cell labels to nuclei labels.
    :param cyto_channel: Index or list of indices for cyto channel(s). If not provided, all channels except for
        `nuclei_channel` are used.
    :param threshold:
        One of `Li`, `Otsu`,  or float value
        If `Li`, use Li’s iterative Minimum Cross Entropy method :func:`~skimage.filters.thresholding.threshold_li`.
        If `Otsu`, use Otsu’s method :func:`~skimage.filters.thresholding.threshold_otsu`.
    :param threshold_correction_factor: Factor to adjust the computed threshold by.
    :param threshold: Threshold to apply to cell mask.
    :param nuclei_channel: DAPI channel index
    :param rm_small_std: If specified, remove objects smaller than specified number of standard deviations of identified
        nuclei
    :param rolling_ball: If true, apply skimage.restoration.rolling_ball subtraction to mask prior to calculating
        threshold
    :param sigma: sigma (optional) gaussian filter sigma to smooth the image prior to computing threshold
    :param closing_radius: (optional) disk radius for closing cell labels
    :param t: Optional list of time indices (0-based) to use for computing cell mask
    :param regularization_factor: If method is propagation, takes two factors into account when deciding where to draw
        the dividing line between two touching secondary objects: the distance to the nearest primary object, and the
        intensity of the secondary object image
    :param fill_holes: Fill holes inside labels
    :param chunks: Chunk size to perform segmentation using non-overlapping chunks.
    :param depth: The number of elements that each block should share with its neighbors when using dask
    :return: Cell labels matching `nuclei` labels and thresholds if an auto threshold method is used.
    """
    if cyto_channel is None:
        cyto_channel = np.delete(np.arange(image.sizes["c"]), nuclei_channel)
    sel = dict(c=cyto_channel, z=0)
    if t is not None:
        sel["t"] = t
    image = image.isel(sel, missing_dims="ignore").squeeze()

    image, mask, threshold = image2mask(
        image=image,
        threshold=threshold,
        threshold_correction_factor=threshold_correction_factor,
        rolling_ball=rolling_ball,
        sigma=sigma,
        depth=depth,
    )
    if rm_small_std is not None:
        nuclei = remove_small_objects_std(nuclei, rm_small_std)

    if isinstance(image, da.Array):
        if chunks is None:
            chunks = image.chunksize

        image = image.rechunk(chunks)
        nuclei = da.from_array(nuclei, chunks=chunks)
        mask = mask.rechunk(chunks)
        if image.chunksize != image.shape:
            labels = da.map_overlap(
                _process_block,
                image,
                nuclei,
                mask,
                depth=depth,
                fill_holes=fill_holes,
                closing_radius=closing_radius,
                weight=regularization_factor,
                dtype=np.int32,
                meta=np.array((), dtype=np.int32),
            )
        else:
            labels = da.map_blocks(
                _process_block,
                image,
                nuclei,
                mask,
                fill_holes=fill_holes,
                closing_radius=closing_radius,
                weight=regularization_factor,
                dtype=np.int32,
                meta=np.array((), dtype=np.int32),
            )

    else:
        labels = _process_block(
            image=image,
            nuclei=nuclei,
            mask=mask,
            weight=regularization_factor,
            fill_holes=fill_holes,
            closing_radius=closing_radius,
        )
    return labels, threshold


def propagate(image, labels, mask, weight):
    """Propagate the labels to the nearest pixels.

    image - gives the Z height when computing distance
    labels - the labeled image pixels
    mask   - only label pixels within the mask
    weight - the weighting of x/y distance vs z distance
             high numbers favor x/y, low favor z

    returns a label matrix and the computed distances
    """
    if image.shape != labels.shape:
        raise ValueError(
            "Image shape %s != label shape %s" % (repr(image.shape), repr(labels.shape))
        )
    if image.shape != mask.shape:
        raise ValueError(
            "Image shape %s != mask shape %s" % (repr(image.shape), repr(mask.shape))
        )
    labels_out = np.zeros(labels.shape, np.int32)
    distances = -np.ones(labels.shape, np.float64)
    distances[labels > 0] = 0
    labels_and_mask = np.logical_and(labels != 0, mask)
    coords = np.argwhere(labels_and_mask)
    i1, i2 = _propagate.convert_to_ints(0.0)
    ncoords = coords.shape[0]
    pq = np.column_stack(
        (
            np.ones((ncoords,), int) * i1,
            np.ones((ncoords,), int) * i2,
            labels[labels_and_mask],
            coords,
        )
    )
    _propagate.propagate(
        np.ascontiguousarray(image, np.float64),
        np.ascontiguousarray(pq, np.int32),
        np.ascontiguousarray(mask, np.int8),
        labels_out,
        distances,
        float(weight),
    )
    labels_out[labels > 0] = labels[labels > 0]
    return labels_out, distances
