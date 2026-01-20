"""Cell and nuclei segmentation utilities.

Authors:
    - The SCALLOPS development team
"""

import math
import os
from collections import Counter
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Callable, Literal

import dask.array as da
import dask.dataframe as dd
import fsspec
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr
from centrosome.outline import outline
from dask import delayed
from dask_image import ndfilters as dask_ndi
from scipy.sparse import issparse, sparray
from skimage.filters.thresholding import threshold_li, threshold_otsu
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import closing, disk, remove_small_objects
from skimage.restoration import rolling_ball as apply_rolling_ball
from skimage.util import map_array
from xarray import DataArray


def close_labels(labels: np.ndarray, disk_radius: int = 3) -> np.ndarray:
    """Close labels using skimage.morphology.closing.

    :param labels: Labels 2d array
    :param disk_radius: The radius of the disk-shaped footprint
    :return: Closed labels
    """
    return closing(labels, disk(disk_radius))


def threshold_quantile(
    image: xr.DataArray, quantile: float
) -> tuple[DataArray, np.ndarray]:
    """Compute threshold using specified quantile after subtraction of the minimum per channel.

    :param image: Image with the dimensions (t,c,z,y,x)
    :param quantile: Quantile (between 0 and 1 inclusive)
    :return: Tuple of image with minimum subtracted and computed threshold
    """
    image = image.where(image > 0, np.nan)
    mins = image.squeeze().min(dim=["y", "x"])
    image -= mins
    threshold = image.quantile(quantile).values
    image = image.fillna(0)
    return image, threshold


def image2mask(
    image: xr.DataArray,
    threshold: Literal["Li", "Otsu", "Local"] | float = "Li",
    threshold_correction_factor: float = 1,
    rolling_ball: bool = False,
    sigma: float | None = None,
    depth: int = 30,
) -> tuple[np.ndarray | da.Array, np.ndarray | da.Array, float | None]:
    """Convert an image to a mask.

    :param image: Image with the dimensions (t, c, y, x)
    :param threshold:
        One of `Li`, `Otsu`, `Local`
        If `Li`, use Li’s iterative Minimum Cross Entropy method :func:`~skimage.filters.thresholding.threshold_li`.
        If `Otsu`, use Otsu’s method :func:`~skimage.filters.thresholding.threshold_otsu`.
        If 'Local', compute mask using `image > smoothed * threshold_correction_factor`
    :param threshold_correction_factor: Factor to adjust the computed threshold by.
    :param threshold: Threshold to apply to mask.
    :param rolling_ball: If true, apply skimage.restoration.rolling_ball subtraction to mask prior to calculating threshold
    :param sigma: sigma (optional) gaussian filter sigma to smooth the image prior to computing threshold
    :param depth: The number of elements that each block should share with its neighbors when using dask
    :return: Image, mask, and threshold if threshold is `Li` or `Otsu`.
    """

    if isinstance(threshold, str):
        threshold = threshold.lower()
        assert threshold in ("li", "otsu", "local")
        if threshold == "local":
            assert sigma is not None, "Please provide sigma when threshold == `local`"

    image = cyto_channel_summary(image).data
    if rolling_ball:
        if isinstance(image, da.Array):

            def process_block(x):
                return x - apply_rolling_ball(image)

            image = da.map_overlap(
                process_block,
                image,
                depth=depth,
                boundary="reflect",
                meta=image._meta,
                dtype=image.dtype,
            )
        else:
            image = image - apply_rolling_ball(image)

    smoothed = image
    if sigma is not None:
        g = (
            ndi.gaussian_filter
            if not isinstance(image, da.Array)
            else dask_ndi.gaussian_filter
        )
        smoothed = g(image, sigma=sigma)  # usually larger than cells, 25 - 200
    mask = None
    threshold_val = threshold
    if isinstance(threshold, str):
        if threshold == "local":
            mask = image > smoothed * threshold_correction_factor  # usually 1.01 - 1.10
        elif threshold in ("li", "otsu"):
            if isinstance(smoothed, da.Array):

                def process_block(x, method, threshold_correction_factor):
                    threshold_val = (
                        threshold_otsu(x) if method == "otsu" else threshold_li(x)
                    )
                    threshold_val = threshold_val * threshold_correction_factor
                    return x > threshold_val

                mask = da.map_overlap(
                    process_block,
                    smoothed,
                    method=threshold,
                    threshold_correction_factor=threshold_correction_factor,
                    depth=depth,
                    boundary="none"
                    if smoothed.chunksize != smoothed.shape
                    else "reflect",
                    meta=np.array((), dtype=bool),
                    dtype=bool,
                )
            else:
                threshold_val = (
                    threshold_otsu(smoothed)
                    if threshold == "otsu"
                    else threshold_li(smoothed)
                )
                threshold_val = threshold_val * threshold_correction_factor
    if mask is None:
        mask = smoothed > threshold_val
    return smoothed, mask, threshold_val if isinstance(threshold_val, Number) else None


def remove_small_objects_std(labels: np.ndarray, rm_small_std: float) -> np.ndarray:
    """Removes small objects from labels.

    :param labels: Array of labels
    :param rm_small_std: Remove objects smaller than specified number of standard deviations of
        labels
    """

    counts = np.array(list(Counter(labels[labels > 0]).values()))
    min_size = (-rm_small_std * counts.std()) + counts.mean()
    rm_small = partial(remove_small_objects, min_size=min_size)
    return rm_small(labels)


def cyto_channel_summary(image: xr.DataArray) -> xr.DataArray:
    """Takes minimum intensity over cycles, followed by mean intensity over channels if both are
    present. If more than one channel and only one cycle (t) is present, takes median over channels.
    Note that if your image contains DAPI and SBS channels, you need to select non-DAPI channels
    first:

    >>> image.isel(
    ...     c=np.delete(np.arange(image.sizes["c"]), nuclei_channel)
    ... )  # doctest: +SKIP

    :param image: Image with dimensions (t, c, z, y, x)
    :return: The cell mask
    """
    if "c" not in image.dims:
        return image.isel(t=0, missing_dims="ignore").squeeze()
    if "t" in image.dims and image.sizes["t"] > 1:
        # min over cycles, mean over channels
        return image.min(dim="t").mean(dim="c")
    elif image.sizes["c"] > 1:
        # take median across channels
        return image.median(dim="c")
    return image.squeeze()


def remove_boundary_labels(labels: np.ndarray, relabel: bool = False) -> np.ndarray:
    """Remove labels at image boundaries.

    :param labels: An array of labels, which must be non-negative integers.
    :param relabel: Whether to relabel the labels.
    :return: Labels with boundaries removed
    """
    labels = labels.copy()
    cut = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    labels.flat[np.in1d(labels.flat[:], np.unique(cut))] = 0
    if relabel:
        labels = relabel_sequential(labels)
    return labels


def remove_labels_region_props(
    labels: np.ndarray,
    regions: list[RegionProperties],
    func: Callable[[RegionProperties], bool],
    relabel: bool = False,
) -> np.ndarray:
    """Filter labels using region props.

    :param labels: An array of labels, which must be non-negative integers.
    :param regions: List of region props from `skimage.measure.regionprops`.
    :param func: Function that returns `True` if label passes filter.
    :param relabel: Whether to relabel the labels
    :return: Filtered labels
    """
    cut = [r.label for r in regions if not func(r)]
    labels = labels.copy()
    labels.flat[np.in1d(labels.flat[:], cut)] = 0
    if relabel:
        labels = relabel_sequential(labels)
    return labels


def remove_masked_regions(
    labels: np.ndarray,
    mask: np.ndarray,
    relabel: bool = False,
) -> np.ndarray:
    """Remove labels in masked regions

    :param labels: An array of labels, which must be non-negative integers.
    :param mask: Binary mask where zeros indicate locations to remove.
    :param relabel: Whether to relabel the labels
    :return: Filtered labels
    """
    cut = np.unique(labels[mask == 0])
    cut = cut[cut > 0]
    labels = labels.copy()
    labels.flat[np.in1d(labels.flat[:], cut)] = 0
    if relabel:
        labels = relabel_sequential(labels)
    return labels


def identify_tertiary_objects(
    primary_labels: np.ndarray | da.Array,
    secondary_labels: np.ndarray | da.Array,
    shrink_primary: bool,
) -> np.ndarray | da.Array:
    """Identify tertiary objects by subtracting smaller objects from larger objects

    :param primary_labels: Primary labels (typically nuclei)
    :param secondary_labels: Secondary labels (typically cells)
    :param shrink_primary: Whether to shrink smaller objects prior to subtraction
    :return: The tertiary objects
    """

    if shrink_primary:
        if isinstance(primary_labels, da.Array):
            primary_labels = primary_labels.compute()
        # see https://github.com/CellProfiler/CellProfiler/blob/95b182e24246fa81d588676224572ce8780a1743/src/frontend/cellprofiler/modules/identifytertiaryobjects.py#L284C9-L290C51
        primary_mask = np.logical_or(primary_labels == 0, outline(primary_labels))
    else:
        primary_mask = primary_labels == 0
    #  tertiary_labels[primary_mask == False] = 0
    return (
        np.where(primary_mask, secondary_labels, 0)
        if not isinstance(secondary_labels, da.Array)
        else da.where(primary_mask, secondary_labels, 0)
    )


def remove_labels_by_area(
    labels: np.ndarray,
    area_min: float = -math.inf,
    area_max: float = math.inf,
    relabel: bool = False,
) -> np.ndarray:
    """Keep labels with `area_min` < area < `area_max`

    :param labels: An array of labels, which must be non-negative integers.
    :param area_min: Minimum area to include
    :param area_max: Maximum area to include
    :param relabel: Whether to relabel the labels
    :return: Filtered labels
    """
    if area_min is None:
        area_min = -math.inf
    if area_max is None:
        area_max = math.inf
    regions = regionprops(labels)

    def _area_filter(r):
        return area_min < r.area < area_max

    return remove_labels_region_props(
        labels=labels, regions=regions, func=_area_filter, relabel=relabel
    )


def _delete_lock_files():
    """Delete the lock files used for preventing errors when multiple processes load models using
    HDF5 simultaneously."""
    for file in [".scallops-stardist.lock", ".scallops-cellpose.lock"]:
        if os.path.exists(file):
            os.remove(file)


def _download_model(local_model_dir: Path, remote_model_file_name: str | list[str]):
    """Downloads model files from a remote directory to a local directory.

    This function checks the environment variable "SCALLOPS_MODEL_DIR" for the remote model directory.
    If specified, it downloads the specified model file(s) from the remote directory to the local directory,
    ensuring that the directory structure exists.

    :param local_model_dir: Local directory where the model files will be stored.
    :param remote_model_file_name: Name or list of names of the remote model file(s) to download.

    :example:

    .. code-block:: python

        from pathlib import Path

        # Single model file
        _download_model(Path("/local/models"), "model.h5")

        # Multiple model files
        _download_model(Path("/local/models"), ["model1.h5", "model2.h5"])
    """
    model_dir = os.environ.get("SCALLOPS_MODEL_DIR")
    if model_dir is not None and model_dir != "":
        if isinstance(remote_model_file_name, str):
            remote_model_file_name = [remote_model_file_name]
        for name in remote_model_file_name:
            remote_path = os.path.join(model_dir, name)
            local_model_file = local_model_dir / name
            if not local_model_file.exists():
                local_model_file.parent.mkdir(exist_ok=True, parents=True)
                fs, _ = fsspec.core.url_to_fs(remote_path)
                fs.get(remote_path, str(local_model_file))


def _area_overlap_chunk(x, y):
    x = x.flatten()
    y = y.flatten()
    x_labels = np.unique(x)
    x_labels = x_labels[x_labels != 0]
    results = []
    for x_label in x_labels:
        x_mask = x == x_label
        tmp_y = y[x_mask]
        x_area = len(tmp_y)
        y_labels = np.unique(tmp_y)
        y_labels = y_labels[y_labels != 0]
        for y_label in y_labels:
            x_y_overlap = np.sum(tmp_y == y_label)
            results.append((x_label, y_label, x_area, x_y_overlap))
    if len(results) == 0:
        return pd.DataFrame(
            {
                "x": pd.Series(dtype=x.dtype),
                "y": pd.Series(dtype=y.dtype),
                "area": pd.Series(dtype="int"),
                "overlap": pd.Series(dtype="int"),
            }
        )
    return pd.DataFrame(results, columns=["x", "y", "area", "overlap"])


def area_overlap(x: da.Array, y: da.Array) -> dd.DataFrame:
    """Find labels in y that overlap with labels in x.

    :param x: Reference labels, typically nuclei
    :param y: Query labels, typically cells
    :return: Dask dataframe with the columns "x", "y", "area", and "overlap"
    """
    y = y.rechunk(x.chunksize)
    results = []
    _overlap_chunk_delayed = delayed(_area_overlap_chunk)
    meta = dd.utils.make_meta(
        pd.DataFrame(
            {
                "x": pd.Series(dtype=x.dtype),
                "y": pd.Series(dtype=y.dtype),
                "area": pd.Series(dtype="int"),
                "overlap": pd.Series(dtype="int"),
            }
        )
    )
    for sl in da.core.slices_from_chunks(x.chunks):
        results.append(_overlap_chunk_delayed(x[sl], y[sl]))
    df = dd.from_delayed(results, meta=meta, verify_meta=False)
    df = df.groupby(["x", "y"], group_keys=False, sort=False, dropna=False).agg(
        {"area": "sum", "overlap": "sum"}
    )
    df["fraction_overlap"] = df["overlap"] / df["area"]
    return df


def overlap_to_iou(overlap: np.ndarray | sparray) -> np.ndarray | sparray:
    if issparse(overlap):  # no keepdims
        n_pixels_x = overlap.sum(axis=0)
        n_pixels_true = overlap.sum(axis=1)
        n_pixels_x.resize((1,) + n_pixels_x.shape)
        n_pixels_true.resize(n_pixels_true.shape + (1,))
    else:
        n_pixels_x = np.sum(overlap, axis=0, keepdims=True)
        n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / (n_pixels_x + n_pixels_true - overlap)


def _relabel_block(label_field, in_vals, output_type=np.uint32, offset=1):
    if in_vals[0] == 0:
        # always map 0 to 0
        out_vals = np.concatenate([[0], np.arange(offset, offset + len(in_vals) - 1)])
    else:
        out_vals = np.arange(offset, offset + len(in_vals))

    out_array = np.empty(label_field.shape, dtype=output_type)
    out_vals = out_vals.astype(output_type)
    return map_array(label_field, in_vals, out_vals, out=out_array)


def relabel_sequential(
    a: np.ndarray, unique_labels: np.ndarray | None = None
) -> np.ndarray:
    unique_labels = np.unique(a) if unique_labels is None else unique_labels
    return _relabel_block(a, unique_labels)


def dask_relabel_sequential(a: da.Array) -> da.Array:
    unique_labels = da.unique(a)
    return da.map_blocks(_relabel_block, a, unique_labels)
