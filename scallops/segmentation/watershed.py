"""Watershed-based cell and nuclei segmentation utilities.

Provides functions for segmenting cells and nuclei using the watershed algorithm.

Watershed algorithm is applied to segment nuclei, using local mean filtering and intensity
thresholding. The resulting labeled nuclei are then used as seeds for the watershed algorithm
applied to the cytoplasmic channels for cell segmentation.

Authors:
    - The SCALLOPS development team
"""

from typing import Callable, Literal, Sequence

import dask.array as da
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.filters.rank import mean as skmean
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.util import img_as_ubyte

from scallops.segmentation.util import (
    close_labels,
    image2mask,
    relabel_sequential,
    remove_small_objects_std,
    threshold_quantile,
)


def _binarize(image: np.ndarray, radius: int, min_size: float) -> np.ndarray:
    """Apply local mean threshold to find outlines. Filter out background shapes. Otsu threshold on
    list of region mean intensities will remove a few dark cells. Could use shape to improve the
    filtering.

    :param image: Array with image data
    :param radius: expected radius of filter data
    :param min_size: Minimum size of relevant objects
    :return: masked array
    """
    dapi = img_as_ubyte(image)
    mean_filtered = skmean(dapi, disk(radius))
    mask = np.greater(dapi, mean_filtered)
    mask = remove_small_objects(mask, min_size=min_size)
    return mask


def _scoring(regions: regionprops, score_func: Callable, threshold: float) -> Sequence:
    scores = np.array([score_func(r) for r in regions])
    if all([s in (True, False) for s in scores]):
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        cut = [r.label for r, s in zip(regions, scores) if s < threshold]
    return cut


def _filter_by_region(
    labeled: np.ndarray,
    score: Callable,
    threshold: float,
    intensity_image: np.ndarray = None,
    relabel: bool = True,
) -> np.ndarray:
    """Apply a filter to label image. The `score` function takes a single region as input and
    returns a score. If scores are boolean, regions where the score is false are removed. Otherwise,
    the function `threshold` is applied to the list of scores to determine the minimum score at
    which a region is kept. If `relabel` is true, the regions are relabeled starting from 1.

    :param labeled: Labeled array, where all connected regions are assigned the same integer value
    :param score: Function to be applied to the region to score
    :param threshold: Foreground regions with mean DAPI intensity greater than `threshold` are
                      labeled as nuclei.
    :param intensity_image: Image with DAPI information
    :param relabel: Relabel region with integers
    :return: Array with labeled data
    """
    labeled = labeled.copy().astype(int).squeeze()
    if intensity_image is not None:
        intensity_image = intensity_image.squeeze()
    regions = regionprops(labeled, intensity_image=intensity_image)
    cut = _scoring(regions, score, threshold)
    labeled.flat[np.in1d(labeled.flat[:], cut)] = 0
    if relabel:
        labeled = relabel_sequential(labeled)
    return labeled


def segment_nuclei_watershed(
    image: xr.DataArray,
    nuclei_channel: int = 0,
    threshold: float = 200,
    area_min: float = 40,
    area_max: float = 400,
    smooth: float = 1.35,
    radius: int = 15,
) -> np.ndarray:
    """Segment nuclei using watershed algorithm.

    Uses local mean filtering to find cell foreground from aligned but unfiltered data, then filters
    identified regions by mean intensity threshold and area ranges.

    :param image:
        Image with the dimensions (t, c, z, y, x)
    :param nuclei_channel:
        Index of nuclei channel (typically DAPI)
    :param area_min:
        Minimum area to be considered a nuclei
    :param threshold:
        Foreground regions with mean DAPI intensity greater than `threshold` are labeled as nuclei.
    :param area_min, area_max:
        After individual nuclei are segmented from foreground using watershed algorithm, nuclei with
        `area_min` < area < `area_max` are retained.
    :param smooth:
        Size of gaussian kernel used to smooth the distance map to foreground prior to watershed.
    :param radius:
        Radius of disk used in local mean thresholding to identify foreground.
    :return:
        Nuclei labels
    """

    image_ = (
        image.isel(t=0, c=nuclei_channel, z=0, missing_dims="ignore").squeeze().values
    )
    mask = _binarize(image_, radius, area_min)
    labeled = label(mask)
    labeled = (
        _filter_by_region(
            labeled,
            lambda r: r.mean_intensity,
            threshold,
            intensity_image=image_,
        )
        > 0
    )
    # only fill holes below minimum area
    filled = ndi.binary_fill_holes(labeled)
    difference = label(filled != labeled)
    change = _filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]
    distance = ndi.distance_transform_edt(labeled)
    if smooth > 0:
        distance = gaussian(distance, sigma=smooth)

    coords = peak_local_max(distance, footprint=np.ones((3, 3)), exclude_border=False)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    markers = ndi.label(mask)[0]
    nuclei = watershed(-distance, markers, mask=labeled).astype(np.uint16)
    nuclei = _filter_by_region(
        nuclei, lambda r: area_min < r.area < area_max, threshold
    )
    return nuclei.astype(np.uint16)


def _process_block(mask, nuclei, image, closing_radius, at_least_nuclei, method):
    if method == "binary":
        labels = watershed(ndi.distance_transform_cdt(nuclei == 0), nuclei, mask=mask)
    else:
        watershed_mask = np.logical_or(mask, nuclei > 0)
        if method == "distance":
            labels = watershed(
                -ndi.distance_transform_cdt(watershed_mask), nuclei, mask=watershed_mask
            )
        else:  # intensity
            labels = watershed(np.max(image) - image, nuclei, mask=watershed_mask)
    if closing_radius:
        labels = close_labels(labels, disk_radius=closing_radius)
    if at_least_nuclei:
        labels[nuclei > 0] = nuclei[nuclei > 0]
    return labels


def segment_cells_watershed(
    image: xr.DataArray,
    nuclei: np.ndarray,
    threshold: Literal["Li", "quantile", "Otsu", "Local"] | float = "Li",
    cyto_channel: int | Sequence[int] | None = None,
    threshold_correction_factor: float = 1,
    quantile_threshold: float = 0.2,
    nuclei_channel: int = 0,
    rm_small_std: float | None = None,
    rolling_ball: bool = False,
    sigma: float | None = None,
    closing_radius: int | None = None,
    t: list[int] | int | None = 0,
    chunks: tuple[int, int] | None = None,
    depth: int = 30,
    watershed_method: Literal["binary", "distance", "intensity"] = "distance",
    at_least_nuclei: bool = True,
) -> tuple[np.ndarray, float | None]:
    """Segment cells using watershed algorithm, matching cell labels to nuclei labels.

    :param image: Image with the dimensions (t, c, z, y, x)
    :param nuclei: Labeled segmentation mask of nuclei. Uses nuclei as seeds and matches cell labels to nuclei labels.
    :param cyto_channel: Index or list of indices for cyto channel(s). If not provided, all channels except for `nuclei_channel` are used.
    :param threshold:
        One of `Li`, `Otsu`, `quantile`, or float value
        If `Li`, use Li’s iterative Minimum Cross Entropy method :func:`~skimage.filters.thresholding.threshold_li`.
        If `Otsu`, use Otsu’s method :func:`~skimage.filters.thresholding.threshold_otsu`.
        If `quantile`, use :func:`~scallops.segmentation.util.threshold_quantile`
    :param threshold_correction_factor: Factor to adjust the computed threshold by.
    :param quantile_threshold: Quantile when threshold is 'quantile`
    :param threshold: Threshold to apply to cell mask.
    :param nuclei_channel: DAPI channel index
    :param rm_small_std: If specified, remove objects smaller than specified number of standard deviations of identified nuclei
    :param rolling_ball: If true, apply skimage.restoration.rolling_ball subtraction to mask prior to calculating threshold
    :param sigma: sigma (optional) gaussian filter sigma to smooth the image prior to computing threshold
    :param closing_radius: (optional) disk radius for closing cell labels
    :param t: Optional list of time indices (0-based) to use for computing cell mask
    :param chunks: Chunk size to perform segmentation using non-overlapping chunks.
    :param depth: The number of elements that each block should share with its neighbors when using dask
    :param at_least_nuclei: Ensure cells occupy at least the same pixels as nuclei
    :param watershed_method: Watershed method:
        If `distance`, use distance transformation of the binarized mask.
        If `intensity`, use inverted cytoplasm intensity image (same as CellProfiler
        WATERSHED_I)
        If `binary`, use distance from nuclei as in publication `The phenotypic landscape
        of essential human genes <https://pubmed.ncbi.nlm.nih.gov/36347254/>.`
    :return: Cell labels matching `nuclei` labels and thresholds if an auto threshold method is used.
    """
    watershed_method = watershed_method.lower()
    assert watershed_method in ["binary", "distance", "intensity"]
    if threshold == "quantile":
        image, threshold = threshold_quantile(image=image, quantile=quantile_threshold)

    if cyto_channel is None:
        cyto_channel = np.delete(np.arange(image.sizes["c"]), nuclei_channel)
    sel = dict(c=cyto_channel, z=0)
    if t is not None:
        sel["t"] = t

    image = image.isel(sel, missing_dims="ignore").squeeze()
    if rm_small_std is not None:
        nuclei = remove_small_objects_std(nuclei, rm_small_std)

    image, mask, threshold = image2mask(
        image=image,
        threshold=threshold,
        threshold_correction_factor=threshold_correction_factor,
        rolling_ball=rolling_ball,
        sigma=sigma,
    )
    if watershed_method != "intensity":
        image = None

    if isinstance(image, da.Array):
        if chunks is None:
            chunks = mask.chunksize

        mask = mask.rechunk(chunks)
        nuclei = da.from_array(nuclei, chunks=chunks)
        if mask.chunksize != mask.shape:
            labels = da.map_overlap(
                _process_block,
                mask,
                nuclei,
                image,
                closing_radius=closing_radius,
                at_least_nuclei=at_least_nuclei,
                method=watershed_method,
                depth=depth,
                dtype=np.int32,
                meta=np.array((), dtype=np.int32),
            )
        else:
            labels = da.map_blocks(
                _process_block,
                mask,
                nuclei,
                image,
                closing_radius=closing_radius,
                at_least_nuclei=at_least_nuclei,
                method=watershed_method,
                dtype=np.int32,
                meta=np.array((), dtype=np.int32),
            )

    else:
        labels = _process_block(
            mask,
            nuclei,
            image,
            closing_radius,
            at_least_nuclei,
            method=watershed_method,
        )

    return labels, threshold
