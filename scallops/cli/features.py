"""Module for the Command-Line Interface (CLI) related to computing features.

This module provides functionality for computing features from labeled images through the CLI.

Authors:
    - The SCALLOPS development team
"""

import argparse
import json
import warnings
from collections.abc import Sequence
from itertools import zip_longest
from typing import Any, get_type_hints

import dask.array
import dask.array as da
import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
from dask.delayed import Delayed
from natsort import natsorted
from zarr import Group

from scallops.cli.find_objects import get_path
from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.features._plot import _plot_features
from scallops.features.constants import (
    _features,
    _features_multichannel,
    _features_single_channel,
    _label_name_to_prefix,
)
from scallops.features.find_objects import find_objects
from scallops.features.generate import label_features, normalize_features
from scallops.features.util import _get_names_from_pd_query
from scallops.io import (
    _images2fov,
    _set_up_experiment,
    _to_parquet,
    is_parquet_file,
    pluralize,
    read_anndata_zarr,
)

logger = _get_cli_logger()


def _read_merged_or_objects(
    paths: list[str],
    timepoint: str | None,
    label_name: str,
    image_key: str,
    label_filter: str | None,
):
    found_paths = []
    for path in paths:
        path_sep = fsspec.core.url_to_fs(path)[0].sep
        path = path.rstrip(path_sep)

        test_paths = [
            f"{path}{path_sep}{label_name}{path_sep}{image_key}.parquet",
            f"{path}{path_sep}{image_key}.zarr",
            f"{path}{path_sep}{image_key}.parquet",
            get_path(path, path_sep, label_name, image_key, timepoint),
        ]

        for test_path in test_paths:
            if fsspec.core.url_to_fs(path)[0].exists(test_path):
                found_paths.append(test_path)

    if len(found_paths) == 0:
        return None

    area_column = f"{_label_name_to_prefix[label_name]}_AreaShape_Area"
    merged_dfs = []
    for path in found_paths:
        if path.lower().endswith(".zarr"):
            data = read_anndata_zarr(path, dask=True)
            merged_df = data.obs
            columns = {area_column}
            assert area_column in data.var.index
            if label_filter is not None:
                query_columns = _get_names_from_pd_query(label_filter)
                columns.update(
                    c
                    for c in query_columns
                    if c not in merged_df.columns and c in data.var.index
                )
            columns = list(columns)
            values = data[:, columns].X.compute()
            for i in range(len(columns)):
                merged_df[columns[i]] = values[:, i]

        else:
            merged_df = pd.read_parquet(path)
        if "label" in merged_df.columns:
            merged_df = merged_df.set_index("label")
        merged_dfs.append(merged_df)
    return (
        merged_dfs[0]
        if len(merged_dfs) == 1
        else pd.concat(merged_dfs, axis=1, join="inner")
    )


def _get_feature_channel_indices(tokens):
    """Get indices of channel parameters in a feature name.

    This function identifies the positions of channel indices in the list of tokens representing a
    feature name. It uses the feature method name to look up parameter types and returns the indices
    of parameters that are channel indices.

    :param tokens: List of tokens obtained by splitting the feature name.
    :return: List of indices corresponding to channel parameters in the feature name.
    """
    method_name = tokens[0]
    f = _features.get(method_name)
    if f is None:
        return []
    parameters = get_type_hints(f)
    parameter_names = list(parameters)
    return [
        i
        for i in range(1, len(tokens))
        if (parameter_name := parameter_names[i - 1]) in ["c", "c1", "c2"]
        and parameters[parameter_name] in [int, Sequence[int]]
    ]


def _find_labels(
    label_paths: list[str],
    image_key: str,
    label_name: str,
    image_key_no_t: str | None,
    selected_timepoint: Any,
):
    timepoints = None
    g = None
    for label_path in label_paths:
        label_root = zarr.open(label_path, mode="r")
        labels_group = label_root.get("labels")
        if labels_group is not None:
            g = labels_group.get(f"{image_key}-{label_name}")
            if g is not None:
                timepoints = (
                    g.attrs["multiscales"][0]["metadata"]["t"]
                    if "t" in g.attrs["multiscales"][0]["metadata"]
                    else [None]
                )
                return g, timepoints
            if g is None and image_key_no_t is not None:
                g = labels_group.get(f"{image_key_no_t}-{label_name}")
                zarr_metadata = g.attrs["multiscales"][0]["metadata"]

                if "t" in zarr_metadata:
                    timepoints = zarr_metadata["t"]
                    if selected_timepoint in timepoints:
                        index = timepoints.index(selected_timepoint)
                        timepoints = [timepoints[index]]
                        return g, timepoints
                else:
                    timepoints = [None]
                    return g, timepoints
    return g, timepoints


def _stack_and_rename(image: xr.DataArray) -> xr.DataArray:
    image_dims = tuple([d for d in ["t", "c", "z"] if d in image.dims])
    with warnings.catch_warnings():
        # ignore UserWarning: rename 't_c_z' to 'c' does not create an index anymore.
        # Try using swap_dims instead or use set_index after rename to create an indexed coordinate.
        warnings.filterwarnings("ignore", "rename .*", UserWarning)
        return (
            image.stack(t_c_z=image_dims, create_index=False)
            .transpose(*("y", "x", "t_c_z"))
            .rename({"t_c_z": "c"})
            if len(image_dims) > 0
            else image.expand_dims("c", -1)
        )


def _image_key_without_time_and_selected_time(metadata):
    image_key_no_t = None
    selected_timepoint = None
    if "t" in metadata["group_metadata"]["group"]:
        image_key_no_t = []
        for key in metadata["group_metadata"]["group"]:
            if key != "t":
                image_key_no_t.append(str(metadata["group_metadata"]["group"][key]))
            else:
                selected_timepoint = metadata["group_metadata"]["group"][key]
        image_key_no_t = "-".join(image_key_no_t).replace("/", "-")

    return image_key_no_t, selected_timepoint


def single_feature(
    stacked_image_tuple: tuple[tuple[str, ...], list[str | Group], dict] | None,
    image_tuple: tuple[tuple[str, ...], list[str | Group], dict],
    label_paths: list[str],
    output_dir: str,
    merge_paths: list[str],
    label_name_to_features: dict[str, set[str]],
    label_name_to_min_max_area: dict[str, tuple[float | None, float | None]],
    features_plot: set[str],
    label_filter: str | None = None,
    channel_names: dict[str, str] = None,
    force: bool = False,
    no_version: bool = False,
    normalize: bool = True,
) -> list[Delayed]:
    """Compute features for a single image.

    This function processes a single image (and optionally a stacked image) to compute various
    features based on provided labels. The computed features are saved to the specified output
    directory. It reads the image data, applies the specified labels to extract regions of interest,
    and computes the requested features for each label.

    :param stacked_image_tuple: Optional tuple containing metadata and file paths for additional
                                stacked images. It includes:
                                - A tuple of metadata strings.
                                - A list of file paths or Zarr groups containing stacked image data.
                                - A dictionary with additional metadata.
                                If provided, the function ensures that the metadata ID matches
                                the primary image's metadata ID.
    :param image_tuple: Tuple containing metadata and file paths for the primary image. It includes:
                        - A tuple of metadata strings.
                        - A list of file paths or Zarr groups containing the primary image data.
                        - A dictionary with additional metadata.
    :param label_paths: Zarr paths containing labels used to identify regions of interest in the image
                         for feature computation.
    :param output_dir: Directory path where the computed feature files will be saved.
    :param merge_paths: Directory path containing output to merge
    :param label_name_to_features: Dictionary mapping label names (keys) to sets of
        feature names (values). Label names correspond to components in the labeled
        image (e.g. nuclei), and feature names specify the features to compute.
    :param label_name_to_min_max_area: Dictionary mapping label names (keys) to min/max
        area to keep
    :param features_plot: Features to plot.
    :param force: Boolean flag indicating whether to overwrite existing output files.
        If True, existing files with the same names will be overwritten. If False,
        skip computation for labels with existing output files.
    :param channel_names: Dict mapping channel index to channel name
    :param label_filter: Expression to filter labels (e.g.barcode_Q_mean_0/barcode_Q_mean > 0.5).
    :param no_version: Whether to skip version/CLI information in output.
    :param normalize: Normalize image.
    :returns: List of delayed results
    """
    _, file_list, metadata = image_tuple
    image_key = metadata["id"]

    if stacked_image_tuple is not None:
        _, stacked_file_list, stacked_metadata = stacked_image_tuple
        assert metadata["id"] == stacked_metadata["id"], (
            f"{metadata['id']} != {stacked_metadata['id']}"
        )

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_sep = output_fs.sep
    output_dir = output_dir.rstrip(output_fs.sep)
    image = _images2fov(file_list, metadata, dask=True)

    n_channels1 = None
    stacked_image = None
    if stacked_image_tuple is not None:
        stacked_image = _images2fov(stacked_file_list, stacked_metadata, dask=True)
        n_channels1 = image.sizes["c"]
    image_key_no_t, selected_timepoint = _image_key_without_time_and_selected_time(
        metadata
    )
    for label_name in label_name_to_features:
        label_prefix = _label_name_to_prefix[label_name]
        features = label_name_to_features[label_name]
        g, timepoints = _find_labels(
            label_paths=label_paths,
            image_key=image_key,
            label_name=label_name,
            image_key_no_t=image_key_no_t,
            selected_timepoint=selected_timepoint,
        )
        if g is None:
            logger.info(f"No labels found for {image_key}")
            continue
        labels_array = da.from_array(g[list(g.keys())[0]])
        for timepoint in timepoints:
            image_ = (
                image.sel(t=timepoint)
                if timepoint is not None and image.sizes.get("t", 0) > 1
                else image
            )
            image_ = _stack_and_rename(image_)
            if stacked_image is not None:
                stacked_image_ = _stack_and_rename(stacked_image)

            intensity_image = (
                xr.concat((image_, stacked_image_), dim="c", join="outer")
                if stacked_image is not None
                else image_
            )

            features_path = get_path(
                output_dir, output_sep, label_name, image_key, timepoint, ".parquet"
            )
            if not force and is_parquet_file(features_path):
                logger.info(
                    f"Skipping features for {image_key} {label_name}{' at t=' + timepoint if timepoint is not None else ''}."
                )
                continue

            merged_df = None
            if len(merge_paths) > 0:
                merged_df = _read_merged_or_objects(
                    paths=merge_paths,
                    timepoint=timepoint,
                    label_name=label_name,
                    image_key=image_key,
                    label_filter=label_filter,
                )
            if timepoint is not None and labels_array.ndim == 3:
                timepoint_index = timepoints.index(timepoint)
                label_image = labels_array[timepoint_index]
            else:
                label_image = labels_array

            if merged_df is None:
                logger.info(
                    f"Find {label_name} objects for {image_key}{' at t=' + timepoint if timepoint is not None else ''}."
                )
                merged_df = find_objects(label_image)
                objects_path = get_path(
                    output_dir, output_sep, label_name, image_key, timepoint
                )

                merged_df.index.name = "label"
                merged_df.columns = f"{label_prefix}_" + merged_df.columns
                _to_parquet(
                    merged_df,
                    objects_path,
                    write_index=True,
                    compute=True,
                    custom_metadata=dict(scallops=json.dumps(cli_metadata()))
                    if not no_version
                    else None,
                )
                merged_df = pd.read_parquet(objects_path)

            features = normalize_features(features)
            # strip nuclei_, etc. from features_plot
            features_plot_label = []
            for feature in features_plot:
                tokens = feature.lower().split("_")
                if tokens[0] == label_prefix:
                    features_plot_label.append("_".join(tokens[1:]))

            features_plot_label = normalize_features(features_plot_label)
            if stacked_image_tuple is not None:
                stacked_features = set()
                stacked_features_plot = set()
                for feature in features:  # add offset for image1
                    tokens = feature.lower().split("_")

                    if (
                        tokens[0] in _features_multichannel.keys()
                        or tokens[0] in _features_single_channel.keys()
                    ):
                        for token_index in range(
                            1,
                            3 if tokens[0] in _features_multichannel.keys() else 2,
                        ):
                            c = tokens[token_index]
                            if c[0] == "s":
                                if c in channel_names:
                                    channel_names[str(n_channels1 + int(c[1:]))] = (
                                        channel_names.pop(c)
                                    )
                                tokens[token_index] = str(n_channels1 + int(c[1:]))

                    new_feature = "_".join(tokens)
                    if feature in features_plot_label:
                        stacked_features_plot.add(new_feature)
                    stacked_features.add(new_feature)

                features = stacked_features
                features_plot_label = stacked_features_plot

            features = list(set(natsorted(features)))

            if label_filter is not None:
                merged_df = merged_df.query(label_filter)
            min_max_area = label_name_to_min_max_area.get(label_name)
            area_column = f"{label_prefix}_AreaShape_Area"
            n_labels = len(merged_df)
            prefix = ""
            if min_max_area[0] is not None or min_max_area[1] is not None:
                area_query = []
                if min_max_area[0] is not None:
                    area_query.append(f"{area_column}>={min_max_area[0]}")
                if min_max_area[1] is not None:
                    area_query.append(f"{area_column}<={min_max_area[1]}")
                merged_df = merged_df.query("&".join(area_query))
                n_labels_filtered = n_labels - len(merged_df)
                prefix = f"Removed {n_labels_filtered:,} out of "
            logger.info(
                f"{prefix}{n_labels:,} {pluralize('label', n_labels)}. "
                f"Area: {merged_df[area_column].min():,.0f} to {merged_df[area_column].max():,.0f}."
            )

            df = label_features(
                objects_df=merged_df,
                label_image=label_image,
                intensity_image=intensity_image,
                features=features,
                normalize=normalize,
                bounding_box_columns=[
                    f"{label_prefix}_AreaShape_BoundingBoxMinimum_Y",
                    f"{label_prefix}_AreaShape_BoundingBoxMinimum_X",
                    f"{label_prefix}_AreaShape_BoundingBoxMaximum_Y",
                    f"{label_prefix}_AreaShape_BoundingBoxMaximum_X",
                ],
                channel_names=channel_names,
            )
            # df will be None if only area and coordinates requested

            if df is not None:
                fs = fsspec.url_to_fs(features_path)[0]
                if fs.exists(features_path):
                    fs.rm(features_path, recursive=True)
                df.index.name = "label"
                df.columns = f"{label_prefix}_" + df.columns

                if isinstance(df, pd.DataFrame):
                    table = pa.Table.from_pandas(df, preserve_index=True)
                    if not no_version:
                        table = table.replace_schema_metadata(
                            {
                                "scallops".encode(): json.dumps(
                                    cli_metadata()
                                ).encode(),
                                **table.schema.metadata,
                            }
                        )
                    fs, output_file = fsspec.url_to_fs(features_path)
                    pq.write_table(
                        table,
                        features_path,
                        filesystem=fs,
                    )

                else:
                    _to_parquet(
                        df,
                        features_path,
                        write_index=True,
                        compute=True,
                        custom_metadata=dict(scallops=json.dumps(cli_metadata()))
                        if not no_version
                        else None,
                    )

            if len(features_plot_label) > 0:
                features_plot_label = [
                    label_prefix + "_" + feature for feature in features_plot_label
                ]
                df_features = pd.read_parquet(
                    features_path, columns=features_plot_label
                )
                centroid_columns = [
                    label_prefix + "_centroid-1",
                    label_name + "_centroid-0",
                ]
                df = merged_df[centroid_columns].join(df_features)
                pdf_path = get_path(
                    output_dir, output_sep, label_name, image_key, timepoint, ".pdf"
                )

                _plot_features(df, features_plot_label, pdf_path, centroid_columns)
    return []


def run_pipeline_compute_features(arguments: argparse.Namespace) -> None:
    """Run the pipeline to compute features for images.

    This function sets up the experiment, processes images, computes features based on the labels,
    and handles the execution of the pipeline.

    :param arguments: Parsed command-line arguments.
    """
    images_paths = arguments.images
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )

    image_patterns = arguments.image_pattern
    output_dir = arguments.output
    merge_paths = arguments.merge
    subset = arguments.subset
    force = arguments.force
    groupby = arguments.groupby
    channel_names = arguments.channel_rename
    stack_images = arguments.stack_images
    label_filter = arguments.label_filter
    label_paths = arguments.labels
    normalize = not arguments.no_normalize
    stack_image_pattern = arguments.stack_image_pattern
    cell_features = arguments.features_cell
    nuclei_features = arguments.features_nuclei
    cytosol_features = arguments.features_cytosol
    label_name_to_features = dict()
    if cell_features is not None:
        label_name_to_features["cell"] = set(cell_features)
    if nuclei_features is not None:
        label_name_to_features["nuclei"] = set(nuclei_features)
    if cytosol_features is not None:
        label_name_to_features["cytosol"] = set(cytosol_features)
    if len(label_name_to_features) == 0:
        raise ValueError("No features to compute")
    unique_features = set()
    for key in label_name_to_features:
        unique_features.update(label_name_to_features[key])

    features_plot = arguments.features_plot
    if features_plot is None:
        features_plot = []
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads(
            threads_per_worker=4 if "sizeshape" in unique_features else 1
        )

    label_name_to_min_max_area = dict(
        nuclei=[arguments.nuclei_min_area, arguments.nuclei_max_area],
        cytosol=[arguments.cytosol_min_area, arguments.cytosol_max_area],
        cell=[arguments.cell_min_area, arguments.cell_max_area],
    )

    assert len(label_name_to_features) > 0, "No features provided"
    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)
    for label in label_name_to_features:
        output_fs.makedirs(output_dir + output_fs.sep + label, exist_ok=True)

    no_version = arguments.no_version

    if channel_names is not None:
        # keys are strings in json
        try:
            channel_names = json.loads(channel_names)
            for key in channel_names.keys():
                channel_names[key] = channel_names[key].replace("_", "")
        except json.decoder.JSONDecodeError:
            raise ValueError("Unable to parse channel names")
    image_gen = _set_up_experiment(
        images_paths,
        files_pattern=image_patterns,
        subset=subset,
        group_by=groupby,
    )

    if stack_images is not None:
        stack_image_gen = _set_up_experiment(
            stack_images,
            files_pattern=stack_image_pattern,
            subset=subset,
            group_by=groupby,
        )
        image_gen = zip(stack_image_gen, image_gen)

    else:
        image_gen = zip_longest([None], image_gen)

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        delayed_objects = [
            single_feature(
                img_tuple[0],
                img_tuple[1],
                output_dir=output_dir,
                merge_paths=merge_paths,
                label_paths=label_paths,
                label_filter=label_filter,
                label_name_to_min_max_area=label_name_to_min_max_area,
                label_name_to_features=label_name_to_features,
                channel_names=channel_names,
                force=force,
                no_version=no_version,
                features_plot=features_plot,
                normalize=normalize,
            )
            for img_tuple in image_gen
        ]
        if len(delayed_objects) > 0:
            dask.compute(*delayed_objects)
