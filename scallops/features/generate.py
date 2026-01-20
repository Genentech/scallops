"""Provides functions for generating features from image data and labeled regions.

Authors:
    - The SCALLOPS development team
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from inspect import Parameter, signature
from typing import Iterable, Literal

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import skimage.util
import zarr
from dask import delayed

from scallops.features.constants import _features, _features_rewrite
from scallops.features.spots import _download_ufish_model
from scallops.segmentation.util import relabel_sequential

logger = logging.getLogger("scallops")


def _img_from_zarrs(intensity_image: Sequence[zarr.Array], sl: tuple[slice]):
    images = []
    for image in intensity_image:
        image = image[..., sl[0], sl[1]]
        if image.ndim == 2:  # add leading c dim
            image = np.expand_dims(image, 0)
        elif image.ndim > 3:
            # unroll leading dimensions
            image = image.reshape((-1,) + image.shape[-2:])
        # change c,y,x to y,x,c
        image = np.transpose(image, (1, 2, 0))
        images.append(image)
    intensity_image = images[0] if len(images) == 1 else np.concatenate(images, axis=2)
    return intensity_image


def _feature_block(
    label_image: np.ndarray | zarr.Array,
    intensity_image: np.ndarray | Sequence[zarr.Array | np.ndarray] | None,
    funcs: list[Callable],
    labels: np.ndarray,
    channel_names: Sequence[str],
    sl: tuple[slice],
    normalize: bool = True,
    label_filter: np.ndarray | None = None,
):
    """Compute features for a block of labeled regions.

    :param label_image: Labeled image.
    :param intensity_image: Intensity image(s).
    :param funcs: List of feature extraction functions.
    :param labels: List of labels to extract features from.
    :param sl: Block slices.
    :param channel_names: Channel names.
    :param normalize: Whether to normalize image.
    :return: DataFrame containing extracted features.
    """

    if isinstance(label_image, zarr.Array):
        label_image = label_image[sl[0], sl[1]]
    if isinstance(intensity_image, Sequence):  # zarrs
        intensity_image = _img_from_zarrs(intensity_image, sl)

    if labels is None:
        labels = np.unique(label_image)
        labels = labels[labels > 0]
        labels = labels[np.isin(labels, label_filter)]

    label_image_original = label_image.copy()
    labels = np.sort(labels)
    # clear labels that this block is not responsible for
    label_image[~np.isin(label_image, labels)] = 0
    label_image = relabel_sequential(label_image)
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]

    if intensity_image is not None:
        if normalize:
            intensity_image = skimage.util.img_as_float(intensity_image)
        assert label_image.shape == intensity_image.shape[:2]
    offset = (0, 0) if sl is None else tuple([s.start for s in sl])
    results = {}
    for f in funcs:
        results.update(
            f(
                unique_labels=unique_labels,
                unique_labels_original=labels,
                channel_names=channel_names,
                label_image=label_image,
                label_image_original=label_image_original,
                intensity_image=intensity_image,
                offset=offset,
            )
        )

    return pd.DataFrame(results, index=labels)


def _create_dd_metadata(
    intensity_image_dtype, nchannels, normalize, channel_names_, funcs
):
    tmp = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0]])
    return _feature_block(
        label_image=tmp,
        intensity_image=np.stack([tmp] * nchannels, axis=2).astype(
            intensity_image_dtype
        )
        if nchannels > 0
        else None,
        labels=np.array([1, 2]),
        sl=None,
        normalize=normalize,
        channel_names=channel_names_,
        funcs=funcs,
    ).fillna(0)


def label_features(
    objects_df: pd.DataFrame,
    label_image: da.Array | zarr.Array,
    intensity_image: da.Array | zarr.Array | Sequence[zarr.Array] | None,
    features: Iterable[str],
    channel_names: dict[int | str, str] | None = None,
    normalize: bool = True,
    bounding_box_columns: Sequence[str] | None = None,
    overlap: int | None = None,
) -> dd.DataFrame | pd.DataFrame:
    """Extract features from labeled regions in the image.

    :param objects_df: Data frame containing labeled regions from `find_objects`.
    :param label_image: Labeled regions.
    :param intensity_image: Intensity image with dimensions (y, x, c) or zarr array(s)
        with dimensions with leading dimensions unrolled to channel dimension
    :param features: Features to extract.
    :param channel_names: Dictionary mapping channel index to channel names.
    :param normalize: Whether to normalize image.
    :param bounding_box_columns: Columns to use for BoundingBoxMinimum_Y,
        BoundingBoxMinimum_X, BoundingBoxMaximum_Y, and BoundingBoxMaximum_X
    :param overlap: Chunk overlap.
    :return: DataFrame with extracted features.
    """
    is_numpy = False
    if isinstance(label_image, np.ndarray):
        is_numpy = True
        label_image = da.from_array(label_image)
    label_shape = label_image.shape
    if intensity_image is not None:
        if isinstance(intensity_image, np.ndarray):
            intensity_image = da.from_array(intensity_image)
        if isinstance(intensity_image, da.Array):
            # y,x,c
            assert intensity_image.shape[:-1] == label_shape, (
                f"{intensity_image.shape} != {label_shape}"
            )
            label_image = label_image.rechunk(intensity_image.chunksize[:-1])

        else:
            if isinstance(intensity_image, zarr.Array):
                intensity_image = [intensity_image]
            for img in intensity_image:
                assert img.shape[-2:] == label_shape, (
                    f"{img.shape[-2:]} != {label_shape}"
                )
    features = normalize_features(features)
    if len(features) == 0:
        raise ValueError("No features to extract.")
    nchannels = 0
    intensity_image_dtype = np.uint16
    is_dask_array = isinstance(label_image, da.Array)

    if intensity_image is not None:
        if is_dask_array:
            nchannels = intensity_image.shape[-1]
        else:
            nchannels = 0
            for img in intensity_image:
                nchannels += np.prod(img.shape[:-2])
            nchannels = int(nchannels)
        intensity_image_dtype = (
            intensity_image.dtype if is_dask_array else intensity_image[0].dtype
        )
        logger.info(f"Number of channels: {nchannels}")
    channel_names_ = [f"Channel{i}" for i in range(nchannels)]
    if channel_names is not None:  # rename
        for channel_index in channel_names:
            channel_name = channel_names[channel_index]
            if isinstance(channel_index, str):
                try:
                    channel_index = int(channel_index)
                except ValueError:
                    logger.info(f"Unable to convert channel {channel_index} to int.")
                    continue
            if channel_index < 0 or channel_index >= nchannels:
                raise ValueError("Channel index out of range")
            channel_names_[channel_index] = channel_name
    funcs, requires_intensity_image = _create_funcs(
        features=features, n_channels=len(channel_names_)
    )
    if not requires_intensity_image:  # Don't pass intensity image if not needed
        intensity_image = None
    intensity_image_delayed = (
        delayed(intensity_image) if not isinstance(intensity_image, da.Array) else None
    )
    label_image_delayed = (
        delayed(label_image) if not isinstance(label_image, da.Array) else None
    )
    if bounding_box_columns is None:
        min_cols = objects_df.columns[
            objects_df.columns.str.contains("AreaShape_BoundingBoxMinimum_")
        ]

        max_cols = objects_df.columns[
            objects_df.columns.str.contains("AreaShape_BoundingBoxMaximum_")
        ]

        min_cols = sorted(min_cols, reverse=True)
        max_cols = sorted(max_cols, reverse=True)
    else:
        min_cols = bounding_box_columns[:2]
        max_cols = bounding_box_columns[2:]
    assert len(min_cols) == 2 and len(max_cols) == 2, (
        "Could not find bounding box columns"
    )
    if len(funcs) == 0:
        raise ValueError("No features to compute.")

    unique_func_names = {func.func.__name__ for func in funcs}
    DEFAULT_PADDING = {"intensity": 1, "neighbors": 100}
    padding = overlap if overlap is not None else 0
    for key in DEFAULT_PADDING.keys():
        if key in unique_func_names:
            padding = max(padding, DEFAULT_PADDING[key])
    if "spot_count" in unique_func_names:
        _download_ufish_model()
    uniform_chunk_df = None
    channel_names_delayed = delayed(channel_names_)
    UNIFORM_CHUNK_FUNC_NAMES = {"spot_count"}
    uniform_chunk_funcs = []
    funcs_ = []
    for func in funcs:
        if func.func.__name__ in UNIFORM_CHUNK_FUNC_NAMES:
            uniform_chunk_funcs.append(func)
        else:
            funcs_.append(func)
    funcs = funcs_
    chunk_slices = da.core.slices_from_chunks(
        label_image.chunks if is_dask_array else da.from_zarr(label_image).chunks
    )
    _feature_block_delayed = delayed(_feature_block)
    if len(uniform_chunk_funcs) > 0:
        results = []
        uniform_chunk_funcs_delayed = delayed(uniform_chunk_funcs)
        label_filter_delayed = delayed(objects_df.index.values)
        for sl in chunk_slices:
            if is_dask_array:
                label_block = label_image[sl]
                image_block = (
                    intensity_image[sl] if intensity_image is not None else None
                )
            results.append(
                _feature_block_delayed(
                    label_image=label_block if is_dask_array else label_image_delayed,
                    intensity_image=image_block
                    if is_dask_array
                    else intensity_image_delayed,
                    sl=sl,
                    normalize=normalize,
                    labels=None,
                    label_filter=label_filter_delayed,
                    channel_names=channel_names_delayed,
                    funcs=uniform_chunk_funcs_delayed,
                )
            )
        logger.info(f"Number of batches: {len(results):,}")
        uniform_chunk_df = dd.from_delayed(
            results,
            meta=_create_dd_metadata(
                intensity_image_dtype,
                nchannels,
                normalize,
                channel_names_,
                uniform_chunk_funcs,
            ),
            verify_meta=False,
        )
        if is_numpy:
            uniform_chunk_df = uniform_chunk_df.compute()

        uniform_chunk_df = uniform_chunk_df.groupby(
            uniform_chunk_df.index, group_keys=False, sort=False, dropna=False
        ).agg("sum")

    if len(funcs) > 0:
        results = []

        funcs_delayed = delayed(funcs)
        for sl in chunk_slices:
            array_start = [s.start for s in sl]
            array_end = [s.stop for s in sl]
            objects_df_slice = objects_df.query(
                f"{array_start[0]}<=`{min_cols[0]}`<{array_end[0]} & "
                f"{array_start[1]}<=`{min_cols[1]}`<{array_end[1]}"
            )
            if len(objects_df_slice) > 0:
                unique_labels = objects_df_slice.index.values

                sl = (
                    slice(
                        max(0, objects_df_slice[min_cols[0]].min() - padding),
                        min(
                            label_shape[0],
                            objects_df_slice[max_cols[0]].max() + padding,
                        ),
                    ),
                    slice(
                        max(0, objects_df_slice[min_cols[1]].min() - padding),
                        min(
                            label_shape[1],
                            objects_df_slice[max_cols[1]].max() + padding,
                        ),
                    ),
                )
                if is_dask_array:
                    label_block = label_image[sl]
                    image_block = (
                        intensity_image[sl] if intensity_image is not None else None
                    )

                results.append(
                    _feature_block_delayed(
                        label_image=label_block
                        if is_dask_array
                        else label_image_delayed,
                        intensity_image=image_block
                        if is_dask_array
                        else intensity_image_delayed,
                        sl=sl,
                        normalize=normalize,
                        labels=unique_labels,
                        channel_names=channel_names_delayed,
                        funcs=funcs_delayed,
                    )
                )
        logger.info(f"Number of batches: {len(results):,}")

        df = dd.from_delayed(
            results,
            meta=_create_dd_metadata(
                intensity_image_dtype, nchannels, normalize, channel_names_, funcs
            ),
            verify_meta=False,
        )
        if is_numpy:
            df = df.compute()

        return df.join(uniform_chunk_df) if uniform_chunk_df is not None else df
    if uniform_chunk_df is not None:
        return uniform_chunk_df

    raise ValueError("No features to compute.")


def normalize_features(features: Iterable[str]) -> set[str]:
    """Normalize feature names

    :param features: List of feature names.
    :return: Set of normalized feature names.
    """
    normalized_features = set()
    for feature in features:
        tokens = feature.lower().split("_")
        tokens[0] = tokens[0].replace("-", "")
        norm_feature = "_".join(tokens)
        normalized_features.add(norm_feature)
    return normalized_features


def _create_funcs(
    features: Iterable[str],
    n_channels: Sequence[str],
) -> tuple[list[Callable], bool]:
    """Create feature functionss.

    :param features: Iterable of feature names to be processed.
    :param n_channels: Number of channels in image
    :return: A tuple containing:
             - list of partial functions corresponding to the provided features.
             - boolean indicating whether any feature requires intensity image
    """

    funcs = []

    requires_intensity_image = False

    features_dict = _features.copy()
    func_tuple_to_params = defaultdict(lambda: [])
    # (func_name, non-channel parameter values)
    for feature in features:
        func_name, params = _get_params(
            features_dict=features_dict,
            name=feature,
            n_channels=n_channels,
        )
        key = [func_name]
        for param_name in params:
            if param_name not in ("c", "c1", "c2"):
                key.append(params[param_name])
        func_tuple_to_params[tuple(key)].append(params)

    for func_tuple, params_list in func_tuple_to_params.items():
        func_name = func_tuple[0]
        if "c" in params_list[0]:
            channels = set()
            for params in params_list:
                for val in params["c"]:
                    channels.add(val)
            requires_intensity_image = True
            new_params = dict(c=tuple(sorted(channels)))
            for p in params_list[0]:
                if p != "c":
                    new_params[p] = params_list[0][p]
            f = partial(features_dict[func_name], **new_params)
            funcs.append(f)
        elif "c1" in params_list[0]:
            requires_intensity_image = True
            seen_symmetric = set()  # c1 always < c2

            for params in params_list:
                c1 = params["c1"]
                c2 = params["c2"]
                for c1_ in c1:
                    for c2_ in c2:
                        if c1_ != c2_:
                            if c1_ < c2_:
                                seen_symmetric.add((c1_, c2_))
                            else:
                                seen_symmetric.add((c2_, c1_))
            additional_params = {}
            for p in params_list[0]:
                if p not in ("c1", "c2"):
                    additional_params[p] = params_list[0][p]
            rewrite_func = _features_rewrite.get(func_name)
            if rewrite_func is not None:
                new_params = dict(c=list(seen_symmetric))
                new_params.update(additional_params)
                f = partial(rewrite_func, **new_params)
                funcs.append(f)
            else:
                for c1, c2 in seen_symmetric:
                    new_params = dict(c1=c1, c2=c2)
                    new_params.update(additional_params)
                    f = partial(features_dict[func_name], **new_params)
                    funcs.append(f)
        else:
            for params in params_list:
                f = partial(features_dict[func_name], **params)
                funcs.append(f)

    return funcs, requires_intensity_image


def _get_params(
    features_dict: dict[str, Callable],
    name: str,
    n_channels: int,
    extra_features: dict[str, Callable] | None = None,
) -> tuple[str, dict]:
    """Create a function for the specified feature.

    :param features_dict: Dictionary that maps feature name to function
    :param name: feature name. For example, intensity_0
    :param n_channels: Number of channels in image
    :param extra_features: Dictionary with callable to be made part of the feature calling with
        custom function. Must take at least regionprops with name `r`
    :return: Tuple of name, and parameters
    """

    if extra_features is not None:
        for key, item in extra_features.items():
            assert "_" not in key, (
                f"Method {key} has an underscore in the name, please remove"
            )
            assert "_" not in item.__name__, (
                f"Function {item.__name__} has an underscore in the name, please remove"
            )
        features_dict.update(extra_features)
    tokens = name.split("_")
    method_name = tokens[0]
    f = features_dict.get(method_name)

    if f is None:
        raise ValueError(f"Method {method_name} (feature {name}) not found.")

    sig = signature(f)
    params = {}
    parameter_names = list(sig.parameters.keys())
    for key in sig.parameters:
        value = sig.parameters[key].default
        if value != Parameter.empty and value is not None:
            params[key] = value  # default value
    for i in range(1, len(tokens)):
        value = tokens[i]
        parameter_name = parameter_names[i - 1]
        annotation = sig.parameters[parameter_name].annotation
        if parameter_name in ("c", "c1", "c2"):
            values = []
            if value == "*":
                values = [i for i in range(n_channels)]
            else:
                for val in value.split(","):
                    val = int(val.strip())
                    if val < 0 or val >= n_channels:
                        raise ValueError(
                            "Channel must be between 0 and {}".format(n_channels - 1)
                        )
                    values.append(val)
            params[parameter_name] = values
        elif annotation in (int, float, bool, str, Literal):
            value = annotation(value)
            params[parameter_name] = value
        else:
            if annotation not in (np.ndarray, Sequence[str]):
                raise ValueError(
                    f"{parameter_name} not supported for {name} for type {annotation}"
                )
    return method_name, params
