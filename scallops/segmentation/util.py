"""Cell and nuclei segmentation utilities.

Authors:
    - The SCALLOPS development team
"""

import os
from collections import Counter
from collections.abc import Sequence
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Literal, cast

import dask
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
from numpy.typing import NDArray
from skimage.filters.thresholding import threshold_li, threshold_otsu
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import closing, disk, remove_small_objects
from skimage.restoration import rolling_ball as apply_rolling_ball
from skimage.util import map_array
from xarray import DataArray


def close_labels(labels: NDArray, disk_radius: int = 3) -> NDArray:
    """Close labels using skimage.morphology.closing.

    :param labels: Labels 2d array
    :param disk_radius: The radius of the disk-shaped footprint
    :return: Closed labels
    """
    return closing(labels, disk(disk_radius))


def threshold_quantile(
    image: xr.DataArray, quantile: float, sel: dict[str, Any] | None = None
) -> tuple[DataArray, NDArray]:
    """Compute threshold using specified quantile after subtraction of the minimum per channel.

    :param image: Image with the dimensions (t,c,z,y,x)
    :param quantile: Quantile (between 0 and 1 inclusive)
    :param sel: Optional `isel` selection (e.g. the channels used for the mask) restricting which
        pixels the quantile is computed from. The minimum is still subtracted from the whole image.
    :return: Tuple of image with minimum subtracted and computed threshold
    """
    image = image.where(image > 0, np.nan)
    mins = image.squeeze().min(dim=["y", "x"])
    image -= mins
    quantile_image = image if sel is None else image.isel(sel, missing_dims="ignore")
    threshold = quantile_image.quantile(quantile).values
    image = image.fillna(0)
    return image, threshold


def _summarize(
    image: xr.DataArray, rolling_ball: bool = False, depth: int = 150
) -> NDArray | da.Array:
    """Summarize channels/cycles, then optionally apply rolling ball subtraction.

    :param image: Image with the dimensions (t, c, y, x)
    :param rolling_ball: If true, apply skimage.restoration.rolling_ball subtraction
    :param depth: The number of elements that each block should share with its neighbors when using dask
    :return: Summarized image.
    """
    data = cyto_channel_summary(image).data
    if rolling_ball:
        if isinstance(data, da.Array):

            def process_block(x):
                return x - apply_rolling_ball(x)

            data = da.map_overlap(
                process_block,
                data,
                depth=depth,
                boundary="reflect",
                meta=data._meta,
                dtype=data.dtype,
            )
        else:
            data = data - apply_rolling_ball(data)
    return data


def _smooth(data: NDArray | da.Array, sigma: float | None) -> NDArray | da.Array:
    """Optionally apply gaussian smoothing.

    :param data: Image
    :param sigma: sigma (optional) gaussian filter sigma
    :return: Smoothed image, or `data` if `sigma` is None.
    """
    if sigma is None:
        return data
    g = (
        ndi.gaussian_filter
        if not isinstance(data, da.Array)
        else dask_ndi.gaussian_filter
    )
    return g(data, sigma=sigma)  # usually larger than cells, 25 - 200


def _threshold_method(
    threshold: Literal["Li", "Otsu", "Local"] | float | NDArray,
    sigma: float | None,
    sigma_name: str = "sigma",
) -> str | float | NDArray:
    """Validate threshold and return the lower-cased method name, or the numeric threshold."""
    if not isinstance(threshold, str):
        return threshold
    method = threshold.lower()
    if method not in ("li", "otsu", "local"):
        raise ValueError(f"Unknown threshold method: {threshold}")
    if method == "local" and sigma is None:
        raise ValueError(f"Please provide {sigma_name} when threshold == `local`")
    return method


def _threshold_mask(
    data: NDArray | da.Array,
    smoothed: NDArray | da.Array,
    threshold_method: str | float | NDArray,
    threshold_correction_factor: float = 1,
    depth: int = 30,
) -> tuple[NDArray | da.Array, float | None]:
    """Compute mask from summarized and smoothed images.

    `threshold_correction_factor` scales the computed `Li`/`Otsu` threshold and the smoothed
    image when `local`. A numeric (e.g. manually set or quantile) threshold is used as is.

    :return: Mask and threshold if threshold is `Li`, `Otsu`, or a number (including a 0-d
        array such as a computed quantile). None for `local`, and for `Li`/`Otsu` on a dask
        array, where the threshold is computed per block.
    """
    mask = None
    threshold_val: Any = threshold_method
    if isinstance(threshold_method, str):
        if threshold_method == "local":
            mask = data > smoothed * threshold_correction_factor  # usually 1.01 - 1.10
        elif threshold_method in ("li", "otsu"):
            if isinstance(smoothed, da.Array):

                def threshold_block(x, method, threshold_correction_factor):
                    threshold_val = (
                        threshold_otsu(x) if method == "otsu" else threshold_li(x)
                    )
                    threshold_val = threshold_val * threshold_correction_factor
                    return x > threshold_val

                mask = da.map_overlap(
                    threshold_block,
                    smoothed,
                    method=threshold_method,
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
                    if threshold_method == "otsu"
                    else threshold_li(smoothed)
                )
                threshold_val = threshold_val * threshold_correction_factor
    if mask is None:
        mask = smoothed > threshold_val
    if isinstance(threshold_val, np.ndarray) and threshold_val.ndim == 0:
        threshold_val = threshold_val.item()
    return (
        mask,
        float(cast(float, threshold_val))
        if isinstance(threshold_val, Number)
        else None,
    )


def image2mask(
    image: xr.DataArray,
    threshold: Literal["Li", "Otsu", "Local"] | float | NDArray = "Li",
    threshold_correction_factor: float = 1,
    rolling_ball: bool = False,
    sigma: float | None = None,
    depth: int = 150,
) -> tuple[NDArray | da.Array, NDArray | da.Array, float | None]:
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
    threshold_method = _threshold_method(threshold, sigma)
    data = _summarize(image, rolling_ball=rolling_ball, depth=depth)
    smoothed = _smooth(data, sigma)
    mask, threshold_val = _threshold_mask(
        data, smoothed, threshold_method, threshold_correction_factor, depth
    )
    return smoothed, mask, threshold_val


def _channel_selection(
    channel: int | Sequence[int] | NDArray, t: list[int] | int | None
) -> dict[str, Any]:
    """Return the `isel` selection for the given channel(s) and time indices at z=0.

    The channel dim is kept (even for a single channel) so that
    :func:`cyto_channel_summary` takes the minimum over cycles instead of only using
    the first cycle.
    """
    sel: dict[str, Any] = dict(c=np.atleast_1d(channel), z=0)
    if t is not None:
        sel["t"] = t
    return sel


def _resolve_cell_segmentation_sigmas(
    threshold: Literal["Li", "Otsu", "Local"] | float | NDArray,
    sigma: float | None,
    mask_sigma: float | None,
    names: tuple[str, str] = ("sigma", "mask_sigma"),
) -> tuple[str | float | NDArray, float | None, float | None]:
    """Resolve the split and mask sigmas (0 means no smoothing) and validate the threshold.

    :param threshold: Threshold method or value.
    :param sigma: Gaussian filter sigma for the split image (and for the mask if `mask_sigma` is None).
    :param mask_sigma: Gaussian filter sigma for the mask.
    :param names: Names of the `sigma` and `mask_sigma` parameters used in error messages
        (e.g. the CLI flags).
    :return: Lower-cased threshold method (or the numeric threshold), sigma, and mask sigma.
    """
    if sigma == 0:
        sigma = None
    sigma_name = names[0]
    if mask_sigma is None:
        mask_sigma = sigma
    elif mask_sigma == 0:
        mask_sigma = None
        sigma_name = f"a non-zero {names[1]}"
    else:
        sigma_name = names[1]
    return _threshold_method(threshold, mask_sigma, sigma_name), sigma, mask_sigma


def _prepare_cell_segmentation_images(
    image: xr.DataArray,
    cyto_channel: int | Sequence[int] | NDArray,
    threshold: Literal["Li", "Otsu", "Local"] | float | NDArray = "Li",
    threshold_correction_factor: float = 1,
    rolling_ball: bool = False,
    sigma: float | None = None,
    t: list[int] | int | None = None,
    mask_channel: int | Sequence[int] | NDArray | None = None,
    mask_sigma: float | None = None,
    depth: int = 30,
    return_split_image: bool = True,
) -> tuple[NDArray | da.Array | None, NDArray | da.Array, float | None]:
    """Compute the cell foreground mask and the image used to split touching cells.

    The mask (thresholding) step uses `mask_channel` and `mask_sigma`, while the split
    step (e.g. intensity watershed or propagation) uses `cyto_channel` and `sigma`.

    :param image: Image with the dimensions (t, c, z, y, x)
    :param cyto_channel: Index or list of indices of channel(s) used for the split image.
    :param threshold: Threshold method or value. See :func:`image2mask`.
    :param threshold_correction_factor: Factor to adjust the computed threshold by.
    :param rolling_ball: If true, apply rolling ball subtraction to both the mask and split images.
    :param sigma: Gaussian filter sigma for the split image (and for the mask if `mask_sigma` is None).
    :param t: Optional time indices (0-based) to use.
    :param mask_channel: Index or list of indices of channel(s) used for the mask. Defaults to
        `cyto_channel`.
    :param mask_sigma: Gaussian filter sigma for the mask. Defaults to `sigma`. Use 0 for no
        smoothing.
    :param depth: The number of elements that each block should share with its neighbors when using dask
    :param return_split_image: If False, skip computing the split image and return None in its place.
    :return: Split image, mask, and threshold if threshold is `Li` or `Otsu`.
    """

    def _select(channel):
        selected = image.isel(_channel_selection(channel, t), missing_dims="ignore")
        if "t" in selected.dims and selected.sizes["t"] == 1:
            selected = selected.squeeze("t")
        return selected

    threshold_method, sigma, mask_sigma = _resolve_cell_segmentation_sigmas(
        threshold, sigma, mask_sigma
    )

    # channel summary is order-insensitive, so compare sorted channel lists
    same_channels = mask_channel is None or np.array_equal(
        np.sort(np.atleast_1d(mask_channel)), np.sort(np.atleast_1d(cyto_channel))
    )
    mask_data = _summarize(
        _select(cyto_channel if mask_channel is None else mask_channel),
        rolling_ball=rolling_ball,
        depth=depth,
    )
    mask_smoothed = _smooth(mask_data, mask_sigma)
    mask, computed_threshold = _threshold_mask(
        mask_data, mask_smoothed, threshold_method, threshold_correction_factor, depth
    )
    if not return_split_image:
        return None, mask, computed_threshold
    if same_channels:
        # reuse the summarized (and rolling ball subtracted) image
        split_image = (
            mask_smoothed if mask_sigma == sigma else _smooth(mask_data, sigma)
        )
    else:
        split_image = _smooth(
            _summarize(_select(cyto_channel), rolling_ball=rolling_ball, depth=depth),
            sigma,
        )
    return split_image, mask, computed_threshold


def remove_small_objects_std(labels: NDArray, rm_small_std: float) -> NDArray:
    """Removes small objects from labels.

    :param labels: Array of labels
    :param rm_small_std: Remove objects smaller than specified number of standard deviations of
        labels
    :return: Labels with small objects removed
    """

    counts = np.array(list(Counter(labels[labels > 0]).values()))
    min_size = (-rm_small_std * counts.std()) + counts.mean()
    rm_small = partial(remove_small_objects, max_size=min_size - 1)
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


def remove_boundary_labels(labels: NDArray, relabel: bool = False) -> NDArray:
    """Remove labels at image boundaries.

    :param labels: An array of labels, which must be non-negative integers.
    :param relabel: Whether to relabel the labels.
    :return: Labels with boundaries removed
    """
    labels = labels.copy()
    cut = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    labels.flat[np.isin(labels.flat[:], np.unique(cut))] = 0
    if relabel:
        labels = relabel_sequential(labels)
    return labels


def remove_labels_regionprops(
    labels: NDArray,
    regions: list[RegionProperties],
    func: Callable[[RegionProperties], bool],
    relabel: bool = False,
) -> NDArray:
    """Filter labels using region properties.

    :param labels: An array of labels, which must be non-negative integers.
    :param regions: List of region props from `skimage.measure.regionprops`.
    :param func: Function that returns `True` if label passes filter.
    :param relabel: Whether to relabel the labels
    :return: Filtered labels
    """
    cut = [r.label for r in regions if not func(r)]
    return remove_labels(labels, cut, relabel)


def remove_masked_regions(
    labels: NDArray,
    mask: NDArray,
    relabel: bool = False,
) -> NDArray:
    """Remove labels in masked regions

    :param labels: An array of labels, which must be non-negative integers.
    :param mask: Binary mask where zeros indicate locations to remove.
    :param relabel: Whether to relabel the labels
    :return: Filtered labels
    """
    cut = np.unique(labels[mask == 0])
    cut = cut[cut > 0]
    return remove_labels(labels, cut, relabel)


def remove_labels(labels: NDArray, cut: Sequence[int], relabel: bool = False):
    """Remove labels in `cut`.

    :param labels: An array of labels, which must be non-negative integers.
    :param cut: List of labels to remove.
    :param relabel: Whether to relabel the labels
    :return: Filtered labels
    """

    labels = labels.copy()
    labels.flat[np.isin(labels.flat[:], cut)] = 0
    if relabel:
        labels = relabel_sequential(labels)
    return labels


def identify_tertiary_objects(
    primary_labels: NDArray | da.Array,
    secondary_labels: NDArray | da.Array,
    shrink_primary: bool,
) -> NDArray | da.Array:
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


def _label_pair_counts_chunk(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    for a in (x, y):
        if a.size == 0 or (
            np.issubdtype(a.dtype, np.unsignedinteger) and a.dtype.itemsize <= 4
        ):
            continue
        # signed labels up to 32 bits can only be out of range if negative; labels
        # wider than 32 bits are allowed as long as their values fit in 32 bits
        a_min = a.min()
        a_max = a.max() if a.dtype.itemsize > 4 else None
        if a_min < 0 or (a_max is not None and a_max > 0xFFFFFFFF):
            raise ValueError(
                "Label values must be between 0 and 2**32 - 1, "
                f"not [{a_min}, {a.max() if a_max is None else a_max}]"
            )
    # encode each pair of labels as a single 64-bit key, which is much faster to count
    # than grouping by two columns
    keys = (x.ravel().astype(np.uint64) << np.uint64(32)) | y.ravel().astype(np.uint64)
    # np.unique sorts, which releases the GIL, unlike hashing (e.g. pd.value_counts)
    return np.unique(keys, return_counts=True)


def _label_overlap_from_counts(
    chunk_counts: list[tuple[NDArray, NDArray]],
    dtype_1: np.dtype,
    dtype_2: np.dtype,
) -> pd.DataFrame:
    keys = np.concatenate([k for k, _ in chunk_counts])
    counts = np.concatenate([c for _, c in chunk_counts])
    # pairs that span chunk boundaries are counted in multiple chunks
    keys, inverse = np.unique(keys, return_inverse=True)
    overlap = np.bincount(inverse, weights=counts).astype(np.int64)
    label_1 = (keys >> np.uint64(32)).astype(dtype_1)
    label_2 = (keys & np.uint64(0xFFFFFFFF)).astype(dtype_2)

    # areas include pixels that overlap background in the other array. keys are sorted,
    # so label_1 is sorted
    unique_1, start_1 = np.unique(label_1, return_index=True)
    area_1 = np.add.reduceat(overlap, start_1)[np.searchsorted(unique_1, label_1)]
    unique_2, inverse_2 = np.unique(label_2, return_inverse=True)
    area_2 = np.bincount(inverse_2, weights=overlap)[inverse_2]

    pair = (label_1 != 0) & (label_2 != 0)
    # every pixel of a label that doesn't overlap any label overlaps background
    no_overlap_1 = np.setdiff1d(unique_1[unique_1 != 0], label_1[pair])
    no_overlap_2 = np.setdiff1d(unique_2[unique_2 != 0], label_2[pair])
    overlap_pair = overlap[pair]
    return pd.DataFrame(
        {
            "label_1": np.concatenate(
                [label_1[pair], np.zeros(len(no_overlap_2), dtype_1), no_overlap_1]
            ),
            "label_2": np.concatenate(
                [label_2[pair], no_overlap_2, np.zeros(len(no_overlap_1), dtype_2)]
            ),
            "overlap": np.concatenate(
                [
                    overlap_pair,
                    np.zeros(len(no_overlap_1) + len(no_overlap_2), np.int64),
                ]
            ),
            "iou": np.concatenate(
                [
                    overlap_pair / (area_1[pair] + area_2[pair] - overlap_pair),
                    np.zeros(len(no_overlap_1) + len(no_overlap_2)),
                ]
            ),
            "fraction_overlap": np.concatenate(
                [
                    overlap_pair / area_1[pair],
                    np.zeros(len(no_overlap_1) + len(no_overlap_2)),
                ]
            ),
        }
    )


def label_overlap(label_image_1: da.Array, label_image_2: da.Array) -> dd.DataFrame:
    """Compute the fraction overlap and intersection over union (IoU) for every pair of overlapping labels.

    :param label_image_1: Label array (typically nuclei)
    :param label_image_2: Label array with the same shape as `label_image_1` (typically cells)
    :return: Dask dataframe with the columns "label_1", "label_2", "overlap", "iou",
        and "fraction_overlap", with one row per pair of labels that overlap, excluding
        pairs where either label is background (0). "fraction_overlap" is the fraction
        of `label_1` contained within `label_2` (e.g. the fraction of a nucleus within a
        cell). Labels in `label_image_2` that do not overlap any label in
        `label_image_1` (e.g. cells without a nucleus) are included with "label_1" set
        to 0, and labels in `label_image_1` that do not overlap any label in
        `label_image_2` (e.g. nuclei outside of cells) are included with "label_2" set
        to 0. "overlap", "iou", and "fraction_overlap" are 0 for these rows.
    """
    assert label_image_1.shape == label_image_2.shape
    for a in (label_image_1, label_image_2):
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"Labels must be integers, not {a.dtype}")
    label_image_2 = label_image_2.rechunk(label_image_1.chunks)
    _chunk_delayed = delayed(_label_pair_counts_chunk, nout=2)
    chunk_counts = [
        _chunk_delayed(x_block, y_block)
        for x_block, y_block in zip(
            label_image_1.to_delayed().ravel(),
            label_image_2.to_delayed().ravel(),
            strict=True,
        )
    ]
    # one row per pair of overlapping labels, so small enough for a single partition
    df = delayed(_label_overlap_from_counts)(
        chunk_counts, label_image_1.dtype, label_image_2.dtype
    )
    meta = pd.DataFrame(
        {
            "label_1": pd.Series(dtype=label_image_1.dtype),
            "label_2": pd.Series(dtype=label_image_2.dtype),
            "overlap": pd.Series(dtype="int64"),
            "iou": pd.Series(dtype="float64"),
            "fraction_overlap": pd.Series(dtype="float64"),
        }
    )
    return dd.from_delayed([df], meta=meta, verify_meta=False)


def _drop_lower_priority_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("label_1", keep="first")


def assign_labels_by_overlap(
    overlap_df: pd.DataFrame | dd.DataFrame,
) -> pd.DataFrame | dd.DataFrame:
    """Assign each label in `label_1` (e.g. nuclei) to the label in `label_2` (e.g. cells)
    with the highest fraction overlap.

    Ties in fraction overlap are broken by IoU, then by the smallest `label_2`. Labels in
    `label_1` that do not overlap any label in `label_2` have `label_2` set to 0. Note
    that this gives each nucleus one cell, but one cell can have more than one nucleus.
    Labels in `label_2` that are not assigned to any label in `label_1` (e.g. cells with
    no nucleus, or cells whose only nucleus was assigned to another cell) are included
    with `label_1` set to 0 and "overlap", "iou", and "fraction_overlap" set to 0, so
    every label in `label_1` and `label_2` is included.

    :param overlap_df: Pandas or Dask dataframe returned by :func:`label_overlap`
    :return: Dataframe of the same type as `overlap_df` with the same columns as
        `overlap_df`, with one row per label in `label_1` and one row per unassigned
        label in `label_2`. For Dask input, the index is reset within each partition
        and therefore is not unique.
    """
    df = overlap_df[overlap_df["label_1"] != 0].sort_values(
        ["label_1", "fraction_overlap", "iou", "label_2"],
        ascending=[True, False, False, True],
    )
    if isinstance(df, dd.DataFrame):
        # sorting partitions on label_1, so all rows for a label_1 are in one partition
        df = df.map_partitions(_drop_lower_priority_overlaps, meta=df._meta)
        concat = dd.concat
    else:
        df = _drop_lower_priority_overlaps(df)
        concat = pd.concat
    labels_2 = overlap_df.loc[overlap_df["label_2"] != 0, ["label_2"]].drop_duplicates()
    unassigned = labels_2.merge(
        df[["label_2"]].drop_duplicates(), on="label_2", how="left", indicator=True
    )
    unassigned = unassigned.loc[unassigned["_merge"] == "left_only", ["label_2"]]
    unassigned = unassigned.assign(
        label_1=0, overlap=0, iou=0.0, fraction_overlap=0.0
    ).astype(overlap_df.dtypes.to_dict())[list(overlap_df.columns)]
    return concat([df, unassigned]).reset_index(drop=True)


# maximum label for which a lookup table is used to relabel (64 MB for uint32 labels)
_MAX_LUT_SIZE = 2**24


def _map_labels_block(
    block: NDArray, in_vals: NDArray, out_vals: NDArray, dtype: np.dtype
) -> NDArray:
    return map_array(block, in_vals, out_vals, out=np.empty(block.shape, dtype=dtype))


def _lut_block(block: NDArray, lut: NDArray) -> NDArray:
    return lut[block]


def relabel_by_assignment(
    labels: NDArray | da.Array,
    assignment_df: pd.DataFrame,
) -> NDArray | da.Array:
    """Relabel `label_2` (e.g. cells) to match the assigned label in `label_1` (e.g. nuclei).

    Each label in `labels` that is assigned to a label in `label_1` is renamed to that
    label. If multiple labels in `label_1` are assigned to the same label in `labels`
    (e.g. a cell containing two nuclei), the label in `label_1` with the largest
    fraction overlap is used, with ties broken by the smallest `label_1`. Labels that are
    not assigned to any label in `label_1` (rows with `label_1` set to 0 in
    :func:`assign_labels_by_overlap`) are given new sequential labels starting at
    `label_1` max value + 1. Background (0) is unchanged.

    :param labels: Label array corresponding to `label_2` in `assignment_df`. Every
        label in `labels` must be in `assignment_df`, otherwise incorrect labels are returned.
    :param assignment_df: Dataframe returned by :func:`assign_labels_by_overlap`
    :return: Relabeled array of the same type and shape as `labels`

    :example:

    >>> import numpy as np
    >>> import pandas as pd
    >>> cells = np.array([[0, 5, 5, 6], [7, 7, 0, 6]])
    >>> assignment_df = pd.DataFrame(
    ...     {
    ...         "label_1": [1, 2, 3, 4, 0],
    ...         "label_2": [5, 5, 6, 0, 7],
    ...         "fraction_overlap": [0.5, 1.0, 1.0, 0.0, 0.0],
    ...     }
    ... )
    >>> relabel_by_assignment(cells, assignment_df)
    array([[0, 2, 2, 3],
           [5, 5, 0, 3]])
    """
    offset = int(assignment_df["label_1"].max()) + 1 if len(assignment_df) > 0 else 1
    # assignment_df includes every label in labels, so labels doesn't need to be read
    unique_labels = assignment_df["label_2"].unique()
    assignment_df = (
        assignment_df[(assignment_df["label_1"] != 0) & (assignment_df["label_2"] != 0)]
        .sort_values(
            ["label_2", "fraction_overlap", "label_1"], ascending=[True, False, True]
        )
        .drop_duplicates("label_2", keep="first")
    )
    assigned_in = assignment_df["label_2"].values
    assigned_out = assignment_df["label_1"].values
    unassigned_in = np.setdiff1d(unique_labels[unique_labels != 0], assigned_in)
    unassigned_out = np.arange(offset, offset + len(unassigned_in))

    in_vals = np.concatenate([[0], assigned_in, unassigned_in]).astype(labels.dtype)
    # pick dtype before concatenating, since mixing uint64 and int64 promotes to float64
    max_out = max(
        int(assigned_out.max()) if len(assigned_out) > 0 else 0,
        offset + len(unassigned_in) - 1,
    )
    dtype = np.promote_types(labels.dtype, np.min_scalar_type(max_out))
    out_vals = np.concatenate(
        [
            np.zeros(1, dtype=dtype),
            assigned_out.astype(dtype),
            unassigned_out.astype(dtype),
        ]
    )
    max_label = int(in_vals.max())
    func: Callable[..., NDArray]
    args: tuple
    if (
        np.issubdtype(labels.dtype, np.integer)
        and in_vals.min() >= 0
        and max_label < _MAX_LUT_SIZE
    ):
        # indexing a lookup table is much faster than map_array
        lut = np.zeros(max_label + 1, dtype=dtype)
        lut[in_vals] = out_vals
        func, args = _lut_block, (lut,)
    else:
        func, args = _map_labels_block, (in_vals, out_vals, dtype)
    if isinstance(labels, da.Array):
        # wrap arrays in delayed so they're stored once in the graph, not in every task
        args = tuple(
            dask.delayed(arg, pure=True) if isinstance(arg, np.ndarray) else arg
            for arg in args
        )
        return da.map_blocks(
            func,
            labels,
            *args,
            dtype=dtype,
            meta=np.empty((0,) * labels.ndim, dtype=dtype),
        )
    return func(labels, *args)


def _relabel_block(label_field, in_vals, output_type=np.uint32, offset=1):
    if in_vals[0] == 0:
        # always map 0 to 0
        out_vals = np.concatenate([[0], np.arange(offset, offset + len(in_vals) - 1)])
    else:
        out_vals = np.arange(offset, offset + len(in_vals))

    out_array = np.empty(label_field.shape, dtype=output_type)
    out_vals = out_vals.astype(output_type)
    return map_array(label_field, in_vals, out_vals, out=out_array)


def relabel_sequential(a: NDArray, unique_labels: NDArray | None = None) -> NDArray:
    unique_labels = np.unique(a) if unique_labels is None else unique_labels
    return _relabel_block(a, unique_labels)


def dask_relabel_sequential(a: da.Array) -> da.Array:
    unique_labels = da.unique(a)
    return da.map_blocks(_relabel_block, a, unique_labels, dtype=np.uint32)
