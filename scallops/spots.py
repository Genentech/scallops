"""Spots Analysis Module.

This module provides functionalities for detecting and analyzing spots in imaging data.
It includes tools for identifying spots, calculating their properties, and performing related analyses.

Authors:
    - The SCALLOPS development team
"""

import itertools
import logging
import os
import random
import warnings
from collections.abc import Callable
from numbers import Number
from typing import Literal

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import skimage
import xarray as xr
from dask import delayed
from dask.array.core import slices_from_chunks
from dask_image import ndfilters as dask_ndi
from dask_image.ndfilters._gaussian import _get_border, _get_sigmas
from scipy import ndimage as ndi
from scipy.spatial import KDTree

from scallops.envir import SCALLOPS_IMAGE_SCALE
from scallops.features.spots import _download_ufish_model, _load_ufish_model
from scallops.reads import decode_max

logger = logging.getLogger("scallops")


def max_filter(data: xr.DataArray, width: int = 3) -> xr.DataArray:
    """Apply a maximum filter in a window of `width`. Conventionally operates on Laplacian-of-
    Gaussian filtered data, dilating channels to compensate for single-pixel alignment error.

    :param data: Image data, expected dimensions of (t,c,z,y,x)
    :param width: Neighborhood size for max filtering
    :return: Maxed `data` with preserved dimensions.
    """
    logger.debug(f"Performing maximum filter in a window of {width}")
    f = (
        dask_ndi.maximum_filter
        if isinstance(data.data, da.Array)
        else ndi.maximum_filter
    )

    maxed = f(
        data.data, size=np.concatenate((np.repeat(1, data.ndim - 2), (width, width)))
    )
    return data.copy(data=maxed)


def std(
    data: xr.DataArray, channel_agg_method: Literal["mean", "median", "sum"] = "mean"
) -> xr.DataArray:
    """Standard deviation over cycles, followed by mean across channels to estimate read locations.
    If only 1 cycle is present, takes standard deviation across channels.

    :param channel_agg_method: Method used to aggregate standard deviation across cycles
    :param data: LoG-ed image data, expected dimensions of (t, c, z, y, x) or (sigma, t, c, z, y,
        x).
    :return: Standard deviation score for each pixel, dimensions of (y,x).
    """
    assert channel_agg_method in ["mean", "median", "sum"]
    data = data.squeeze()

    if "t" in data.dims:
        x = data.std(dim="t")
        if "c" in data.dims:
            if channel_agg_method == "mean":
                return x.mean(dim="c")
            elif channel_agg_method == "median":
                return x.median(dim="c")
            else:
                return x.sum(dim="c")
        return x
    elif "c" in data.dims:
        return data.std(dim="c")
    else:
        raise ValueError("Unexpected input data.")


def _nunique_bases(x):
    return len(set(x))


def peak_thresholds_from_bases(
    bases_array: xr.DataArray,
    n_reads: int | None = 500000,
    seed: int = 239753,
    Q_cutoff: float = 60,
    remove_zero_entropy_barcodes: bool = True,
) -> pd.DataFrame:
    """Calculate recall, precision, f1, and accuracy for different cutoff value of peaks.

    :param bases_array: DataArray containing the bases data.
    :param n_reads: Number of reads to consider.
    :param seed: Seed for random number generator.
    :param Q_cutoff: Threshold for binarization of reads.
    :param remove_zero_entropy_barcodes: Whether to remove zero entropy barcodes.
    :return: Result data frame.
    """

    if n_reads is not None and n_reads < bases_array.sizes["read"]:
        rng = random.Random(seed)
        random_reads = rng.sample(range(0, bases_array.sizes["read"]), n_reads)
        random_reads = np.sort(random_reads)
        bases_array = bases_array.isel(read=random_reads)
    df_reads = decode_max(bases_array)
    apply_kw_args = dict()
    if remove_zero_entropy_barcodes:
        if isinstance(df_reads, dd.DataFrame):
            apply_kw_args["meta"] = int
        df_reads["nunique_bases"] = df_reads["barcode"].apply(
            _nunique_bases, **apply_kw_args
        )
        df_reads = df_reads.query("nunique_bases>1")
    if isinstance(df_reads, dd.DataFrame):
        df_reads = df_reads.compute()
    else:
        df_reads = df_reads.copy()

    return _peak_thresholds_from_reads(df_reads, Q_cutoff, False).sort_values(
        ["f0.5", "accuracy", "threshold"], ascending=False
    )


def peak_thresholds_from_reads(
    df_reads: pd.DataFrame,
    Q_cutoff: float = 60,
) -> pd.DataFrame:
    """Calculate recall, precision, f1, and accuracy for different cutoff value of peaks.

    :param df_reads: Data frame containing reads.
    :param Q_cutoff: Threshold for binarization of reads.
    :return: Result data frame.
    """
    return _peak_thresholds_from_reads(df_reads, Q_cutoff, True).sort_values(
        ["f1", "accuracy", "threshold"], ascending=False
    )


def _peak_thresholds_from_reads(
    df_reads: pd.DataFrame, Q_cutoff: float = 60, copy=True
) -> pd.DataFrame:
    """Calculate recall, precision, f1, and accuracy for different cutoff value of peaks.

    :param df_reads: Data frame containing reads.
    :param Q_cutoff: Threshold for binarization of reads.
    :param copy: Copy the dataframe
    :return: Result data frame.
    """
    if copy:
        df_reads = df_reads.copy()
    df_reads["read_type"] = (df_reads["Q_mean"] >= Q_cutoff).astype(int)
    q = np.arange(0.02, 1, 0.01)
    results = []
    quantiles = df_reads["peak"].quantile(q).values
    for i in range(len(quantiles)):
        threshold = quantiles[i]
        counts_over = df_reads.query(f"peak>{threshold}")["read_type"].value_counts()
        counts_under = df_reads.query(f"peak<={threshold}")["read_type"].value_counts()
        tp = counts_over.loc[1] if 1 in counts_over else 0
        fp = counts_over.loc[0] if 0 in counts_over else 0
        tn = counts_under.loc[0] if 0 in counts_under else 0
        fn = counts_under.loc[1] if 1 in counts_under else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (2 * precision * recall) / (precision + recall)

        results.append(
            [threshold, q[i], precision, recall, f1, accuracy, tp, fp, tn, fn]
        )
    df = pd.DataFrame(
        results,
        columns=[
            "threshold",
            "quantile",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "tp",
            "fp",
            "tn",
            "fn",
        ],
    )
    b2 = 0.5 * 0.5
    df["f0.5"] = ((1 + b2) * (df["precision"] * df["recall"])) / (
        b2 * df["precision"] + df["recall"]
    )
    return df


def _peaks_to_df(
    peaks: np.ndarray | da.Array, sigma_list: np.ndarray
) -> pd.DataFrame | dd.DataFrame:
    """Converts peak data into a pandas or dask DataFrame.

    :param peaks: Array containing the peak intensity values.
    :param sigma_list:Array containing sigma values corresponding to the peaks.
    :return: A DataFrame containing the peaks, sigma values, and their corresponding coordinates.

    :example:
    .. code-block:: python

        # Example usage of _peaks_to_df
        peaks_df = _peaks_to_df(peaks_array, sigma_array)
    """

    @dask.delayed
    def process_chunk(x, offset, sigma_list):
        indices = np.where(x > 0)
        y_index = 0 if len(indices) == 2 else 1
        x_index = 1 if len(indices) == 2 else 2
        return pd.DataFrame(
            {
                "peak": x[indices],
                "sigma": (
                    sigma_list[indices[0]]
                    if len(indices) > 2
                    else np.repeat(sigma_list, len(indices[0]))
                ),
                "y": indices[y_index] + offset[0],
                "x": indices[x_index] + offset[1],
            }
        )

    if isinstance(peaks, da.Array):
        output = [
            process_chunk(peaks[s], (s[0].start, s[1].start), sigma_list)
            for s in slices_from_chunks(peaks.chunks)
        ]
        meta = dd.utils.make_meta(
            [
                ("peak", peaks.dtype),
                ("sigma", sigma_list.dtype),
                ("y", np.int64),
                ("x", np.int64),
            ]
        )
        return dd.from_delayed(output, meta=meta).reset_index(drop=True)
    indices = np.where(peaks > 0)

    y_index = 0 if len(indices) == 2 else 1
    x_index = 1 if len(indices) == 2 else 2
    return pd.DataFrame(
        {
            "peak": peaks[indices],
            "sigma": (
                sigma_list[indices[0]]
                if len(indices) > 2
                else np.repeat(sigma_list, len(indices[0]))
            ),
            "y": indices[y_index],
            "x": indices[x_index],
        }
    )


def _find_peaks(
    std_array: xr.DataArray | np.ndarray | da.Array,
    n: int = 5,
) -> np.ndarray | da.Array:
    """Identifies peaks in a given standard deviation array using a neighborhood filter.

    :param std_array: Array containing standard deviation values.
    :param n: The size of the neighborhood for peak detection (default is 5).
    :return: Array indicating the identified peaks.

    :example:
    .. code-block:: python

        # Example usage of _find_peaks
        peaks = _find_peaks(std_array, n=5)
    """
    data = std_array.data if isinstance(std_array, xr.DataArray) else std_array
    is_dask = isinstance(data, da.Array)

    neighborhood_size = (1,) * (data.ndim - 2) + (n, n)

    maximum_filter = dask_ndi.maximum_filter if is_dask else ndi.maximum_filter
    minimum_filter = dask_ndi.minimum_filter if is_dask else ndi.minimum_filter

    data_max = maximum_filter(data, neighborhood_size)
    data_min = minimum_filter(data, neighborhood_size)
    peaks = data_max - data_min
    peaks[data != data_max] = 0
    # remove peaks close to edge
    border_width = (n,) * peaks.ndim
    for i, width in enumerate(border_width):
        peaks[(slice(None),) * i + (slice(None, width),)] = 0
        peaks[(slice(None),) * i + (slice(-width, None),)] = 0
    return peaks


def _merge_close_points_kdtree(points, prob, threshold=5):
    tree = KDTree(points)
    merged_points = []
    merged_prob = []
    processed_indices = np.zeros(len(points), dtype=bool)

    for i, p in enumerate(points):
        if i in processed_indices:
            continue

        indices_in_range = tree.query_ball_point(p, threshold)
        current_cluster_indices = []
        for idx in indices_in_range:
            if not processed_indices[idx]:
                current_cluster_indices.append(idx)
                processed_indices[idx] = True

        if current_cluster_indices:
            cluster_points = points[current_cluster_indices]
            cluster_probs = prob[current_cluster_indices]
            index = np.argmax(cluster_probs)
            merged_points.append(cluster_points[index])
            merged_prob.append(cluster_probs[index])

    return np.array(merged_points), merged_prob


def _spot_detect_chunk(
    image: np.ndarray,
    offset: tuple[int, int],
    n: int,
    load_model_func: Callable,
    predict_func: Callable,
    **predict_kwargs,
) -> pd.DataFrame:
    """Call spots for one chunk of an image

    :param image: Array with dimensions (t,c,y,x)
    :param offset: Image y,x, offset
    :param n: Minimum distance between spots
    :param prob_thresh: Probability threshold for spot detection
    :return: Spots dataframe
    """
    assert image.ndim == 4

    points = []
    prob = []
    model = load_model_func()
    for t in range(image.shape[0]):
        for c in range(image.shape[1]):
            points_, prob_ = predict_func(model, image[t, c], **predict_kwargs)
            if len(points_) > 0:
                points.append(points_)
                prob.append(prob_)
    if len(points) == 0:
        return pd.DataFrame(
            {
                "peak": pd.Series(dtype="float"),
                "y": pd.Series(dtype="int"),
                "x": pd.Series(dtype="int"),
            }
        )
    points = np.concatenate(points)  # (n_spots, 2)
    prob = np.concatenate(prob)
    merged_points, merged_prob = _merge_close_points_kdtree(points, prob, n)

    df = pd.DataFrame({"peak": merged_prob})
    df["y"] = merged_points[:, 0] + offset[0]
    df["x"] = merged_points[:, 1] + offset[1]
    return df


def _find_peaks_deep(
    iss_image: xr.DataArray,
    n: int = 5,
    method: Literal["spotiflow", "u-fish", "piscis"] = "spotiflow",
    **predict_kwargs,
) -> pd.DataFrame | dd.DataFrame:
    """Finds spots using spotiflow.

    :param iss_image: ISS image data with dimensions t,c,t,x.
    Note that the c dimension should only include the sequencing channels and the t
    dimension should only include timepoints to use.
    :param n: Minimum distance between spots
    :param method: Spot detection method
    :param predict_kwargs: Keyword arguments for predict
    :return: Data frame with peaks
    """

    def _load_model_spotiflow():
        from spotiflow.model import Spotiflow

        return Spotiflow.from_pretrained("general")

    def _load_model_piscis():
        import piscis

        return piscis.Piscis(model_name="20230905")

    def _predict_ufish(model, img, **predict_args):
        points, enhanced_img = model.predict(img, **predict_args)
        return points.values, np.ones(len(points))

    def _predict_spotiflow(model, img, **predict_args):
        points, details = model.predict(img, **predict_args)
        return points, details.prob

    def _predict_piscis(model, img, **predict_args):
        points = model.predict(img, **predict_args)
        return points.astype(int), np.ones(len(points))

    default_prediction_args = dict()

    if method == "u-fish":
        _download_ufish_model()
        _load_model_func = _load_ufish_model
        _predict_func = _predict_ufish
        default_prediction_args["axes"] = "yx"
    elif method == "spotiflow":
        _load_model_func = _load_model_spotiflow
        _predict_func = _predict_spotiflow
        default_prediction_args["subpix"] = False
        default_prediction_args["min_distance"] = n
        default_prediction_args["verbose"] = False
    elif method == "piscis":
        _load_model_func = _load_model_piscis
        _predict_func = _predict_piscis
        default_prediction_args["intermediates"] = False
        default_prediction_args["threshold"] = 1.0

    default_prediction_args.update(predict_kwargs)
    _load_model_func()
    iss_image = iss_image.data

    if isinstance(iss_image, da.Array):
        # no chunking in t or c
        chunksize = list(iss_image.chunksize)
        for i in range(2):
            if iss_image.chunksize[i] != iss_image.shape[i]:
                chunksize[i] = -1
        chunksize = tuple(chunksize)
        if chunksize != iss_image.chunksize:
            iss_image = iss_image.rechunk(chunksize)
        slices = da.core.slices_from_chunks(iss_image.chunks)
        _spot_detect_chunk_delayed = delayed(_spot_detect_chunk)
        results = []
        for sl in slices:
            iss_image_block = iss_image[sl]
            offset = []
            for i in range(2, 4):
                offset.append(sl[i].start)
            results.append(
                _spot_detect_chunk_delayed(
                    predict_func=_predict_func,
                    load_model_func=_load_model_func,
                    image=iss_image_block,
                    offset=offset,
                    n=n,
                    **default_prediction_args,
                )
            )
        columns = [("peak", np.float64), ("y", np.uint64), ("x", np.uint64)]
        meta = dd.utils.make_meta(columns)
        return dd.from_delayed(results, meta=meta, verify_meta=False)
    else:
        return _spot_detect_chunk(
            predict_func=_predict_func,
            load_model_func=_load_model_func,
            n=n,
            image=iss_image,
            offset=(0, 0),
            **default_prediction_args,
        )


def find_peaks(std_array: xr.DataArray, n: int = 5) -> pd.DataFrame | dd.DataFrame:
    """
    Finds local maxima. At a maximum, the value is max - min in a neighborhood of width `n`. Elsewhere, it is
    zero.

    :param std_array: Std image data
    :param n: width of the neighborhood

    :return: Data frame with peaks
    """
    peaks = _find_peaks(std_array, n)
    ds = xr.Dataset()
    ds["peak"] = xr.DataArray(peaks, dims=std_array.dims, coords=std_array.coords)
    x = (
        ds.to_dask_dataframe().query("peak>0")
        if isinstance(peaks, da.Array)
        else ds.to_dataframe().query("peak>0").reset_index()
    )
    return x


def chastity(
    x: np.ndarray | da.Array,
) -> tuple[np.ndarray | da.Array, np.ndarray | da.Array]:
    """Computes the chastity, which is defined as the ratio of the brightest base intensity divided
    by the sum of the brightest and second brightest base intensities after rescaling the intensity
    to between 0 and 1.

    This function is useful for identifying the purity of a signal by comparing the intensity of the
    brightest spot to the combined intensity of the two brightest spots. The result is a measure of
    how dominant the brightest spot is in the given data.

    :param x: Array with dimensions (t, c, y, x) representing the intensity data.
    :return: A tuple containing:
        - Chastity per cycle (np.ndarray | da.Array): The chastity score for each cycle.
        - Min chastity over all cycles (np.ndarray | da.Array): The minimum chastity score across all cycles.

    :example:

    .. code-block:: python

        # Example usage of chastity
        chastity_scores, min_chastity = chastity(intensity_data)
    """
    is_dask = isinstance(x, da.Array)
    if not is_dask:
        x = da.from_array(x)
    xmin = da.min(x)
    xmax = da.max(x)
    x = (x - xmin) / (xmax - xmin)
    top2 = da.topk(x, 2, axis=1)  # (t, c, y, x)
    total = da.sum(top2, axis=1)  # (t,y,x)
    p = da.max(top2, axis=1) / total
    p = da.nan_to_num(p)  # handle case where all values are 0
    if is_dask:
        min_p = da.min(p, axis=0)  # (t,y,x)
    else:
        p = p.compute()
        min_p = np.min(p, axis=0)
    return p, min_p


def transform_log(
    data: xr.DataArray, sigma: float | np.ndarray = 1, truncate: float = 4.0, **kwargs
) -> xr.DataArray:
    """Apply Laplacian-of-Gaussian filter.

    :param data: Image data with dimensions (t, c, y, x) or (c, y, x).
    :param sigma: Size or list of sizes of gaussian kernel used in Laplacian-of-Gaussian filter
    :param truncate: Truncate the filter at this many standard deviations
    :param kwargs: Keyword arguments to gaussian_laplace
    :return: LoG-ed data with dimensions (sigma, t, c, y, x).
    """

    assert data.dims == ("t", "c", "y", "x") or data.dims == ("c", "y", "x"), data.dims

    def process_chunk(
        x,
        s,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        scale_invariance=True,
        clip=False,
        **kwargs,
    ):
        # the peaks are negative so invert the signal
        # average s**2 provides scale invariance

        if x.ndim == 2:
            image_cube = -ndi.gaussian_laplace(
                x, s, mode=mode, cval=cval, truncate=truncate, **kwargs
            )
            if scale_invariance:
                image_cube = image_cube * np.mean(s) ** 2
        else:
            image_cube = np.empty(x.shape, dtype=x.dtype)
            for index in itertools.product(
                *[range(x.shape[i]) for i in range(x.ndim - 2)]
            ):
                value = -ndi.gaussian_laplace(
                    x[index], s, mode=mode, cval=cval, truncate=truncate, **kwargs
                )
                if scale_invariance:
                    value = value * np.mean(s) ** 2
                image_cube[index] = value

        if clip:
            image_cube = np.clip(image_cube, 0, 65535) / 65535
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image_cube = skimage.img_as_uint(image_cube)
        return image_cube

    def process_chunk_sigma_list(
        x,
        sigma_list,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        scale_invariance=True,
        clip=False,
        **kwargs,
    ):
        # the peaks are negative so invert the signal
        # average s**2 provides scale invariance
        image_cube = np.empty((len(sigma_list),) + x.shape, dtype=x.dtype)
        if x.ndim == 2:
            for sigma_index, s in enumerate(sigma_list):
                value = -ndi.gaussian_laplace(
                    x, s, mode=mode, cval=cval, truncate=truncate, **kwargs
                )
                if scale_invariance:
                    value = value * np.mean(s) ** 2
                image_cube[sigma_index] = value
        else:
            for index in itertools.product(
                *[range(x.shape[i]) for i in range(x.ndim - 2)]
            ):
                for sigma_index, s in enumerate(sigma_list):
                    value = -ndi.gaussian_laplace(
                        x[index], s, mode=mode, cval=cval, truncate=truncate, **kwargs
                    )
                    if scale_invariance:
                        value = value * np.mean(s) ** 2
                    image_cube[(sigma_index,) + index] = value
        if clip:
            image_cube = np.clip(image_cube, 0, 65535) / 65535
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image_cube = skimage.img_as_uint(image_cube)
        return image_cube

    if isinstance(sigma, Number):
        sigma = [sigma]
    sigma = np.array(sigma)
    match_blainey = os.environ.get(SCALLOPS_IMAGE_SCALE) == "1"
    data.data = data.data.astype(float)
    if isinstance(data.data, da.Array):
        dtype = data.data.dtype
        if match_blainey:
            dtype = np.uint16
        depth = list(
            _get_border(data.data, _get_sigmas(data.data, np.max(sigma)), truncate)
        )
        for i in range(data.data.ndim - 2):
            depth[i] = 0
        depth = tuple(depth)
        d = [
            da.map_overlap(
                process_chunk,
                data.data,
                depth=depth,
                boundary="none",
                dtype=dtype,
                meta=np.array((), dtype=dtype),
                s=sigma_,
                clip=match_blainey,
                truncate=truncate,
                **kwargs,
            )
            for sigma_ in sigma
        ]
        log_data = da.stack(d)
    else:
        log_data = process_chunk_sigma_list(
            data.data, sigma_list=sigma, truncate=truncate, clip=match_blainey, **kwargs
        )

    coords = data.coords.copy()
    coords["sigma"] = sigma
    return xr.DataArray(
        data=log_data,
        dims=("sigma",) + data.dims,
        coords=coords,
        attrs=dict(data.attrs),
    )


def normalize_base_intensities(
    image: np.ndarray | da.Array,
    qmin: float | None = 0.01,
    qmax: float | None = 0.98,
    eps: float = 1e-20,
    dtype: np.dtype = np.float32,
) -> da.Array | np.ndarray:
    """Normalize base intensities for every time and channel separately.

    :param image: Array with dimensions (t,c,y,x)
    :param qmin: Minimum quantile for normalization.
    :param qmax: Maximum quantile for normalization.
    :param eps: Small value added to the denominator for normalization.
    :param dtype: Data type of the output image.
    :return: Normalized base intensities
    """

    if qmax is None:
        qmax = 1
    if qmin is None:
        qmin = 0

    assert image.ndim == 4

    def _normalize(x, qmin, qmax, eps, result_type):
        # normalize every t+c separately
        mi, ma = np.quantile(x, q=[qmin, qmax], axis=[2, 3], keepdims=True)
        x = (x - mi) / (ma - mi + eps)
        return x.astype(result_type)

    eps = dtype(eps)
    if isinstance(image, da.Array):
        tmp = np.array([1, 2, 3], dtype=image.dtype) / np.array([0.5], dtype=dtype)

        return da.map_overlap(
            _normalize,
            image,
            qmin=qmin,
            qmax=qmax,
            eps=eps,
            result_type=dtype,
            dtype=tmp.dtype,
            meta=np.array((), dtype=tmp.dtype),
            depth=(0, 0, 30, 30),
        )
    return _normalize(image, qmin=qmin, qmax=qmax, eps=eps, result_type=dtype)
