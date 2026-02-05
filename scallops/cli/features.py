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
from typing import get_type_hints

import dask.array
import dask.array as da
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
from dask.delayed import Delayed
from natsort import natsorted
from zarr import Group

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
from scallops.io import (
    _images2fov,
    _set_up_experiment,
    _to_parquet,
    is_parquet_file,
    pluralize,
)
from scallops.zarr_io import _read_ome_zarr_array

logger = _get_cli_logger()


def get_labels(labels_group: Group, name: str, suffix: str) -> zarr.Array | None:
    """Retrieve labels from a zarr group.

    :param labels_group: The zarr group containing labels.
    :param name: The identifier associated with the labels.
    :param suffix: The suffix used to identify the specific set of labels (e.g., 'nuclei').
    :return: The retrieved labels as a DataArray or None if the labels are not found.
    """
    try:
        return labels_group[f"{name}-{suffix}"]["0"]
    except KeyError as e:
        logger.warning(f'"{name}-{suffix}" not found in {labels_group}.')
        raise e


def _read_image(file_list: list[str], metadata: dict) -> xr.DataArray:
    """Read image files and preprocess them into a standardized format.

    This function reads image files specified in the file_list and processes them into an
    xarray.DataArray with dimensions adjusted as needed. It handles missing dimensions and stacks
    time and channel dimensions for further processing.

    :param file_list: List of file paths to the image files.
    :param metadata: Dictionary containing metadata associated with the images.
    :return: DataArray containing the processed image data.
    """
    image = _images2fov(file_list, metadata, dask=True)
    dims = tuple([d for d in ["t", "c", "z"] if d in image.dims])

    if len(dims) > 0:
        image = image.stack(t_c_z=dims, create_index=False).transpose(
            *("y", "x", "t_c_z")
        )
        with warnings.catch_warnings():
            # ignore UserWarning: rename 't_c_z' to 'c' does not create an index anymore.
            # Try using swap_dims instead or use set_index after rename to create an indexed coordinate.
            warnings.filterwarnings("ignore", "rename .*", UserWarning)
            image = image.rename({"t_c_z": "c"})
    else:
        # add trailing c dimension
        image = image.expand_dims("c", -1)
    return image


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


def single_feature(
    stacked_image_tuple: tuple[tuple[str, ...], list[str | Group], dict] | None,
    image_tuple: tuple[tuple[str, ...], list[str | Group], dict],
    labels_group: Group,
    output_dir: str,
    output_sep: str,
    objects_dir: str,
    objects_dir_sep: str,
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
    :param labels_group: Zarr group containing labels used to identify regions of interest in the image
                         for feature computation.
    :param output_dir: Directory path where the computed feature files will be saved.
    :param output_sep: Separator string used to construct the output file names. This helps in organizing
                       the output files systematically.
    :param objects_dir: Directory path containing find objects output.
    :param objects_dir_sep: File separator for `objects_dir`
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
    :param label_filter: Path to Parquet file containing `label` column. The path can
        contain expressions (e.g. s3://foo/{well}.parquet).
    :param no_version: Whether to skip version/CLI information in output.
    :param normalize: Normalize image.
    :returns: List of delayed results
    """
    _, file_list, metadata = image_tuple
    image_key = metadata["id"]

    if label_filter is not None:
        label_filter = label_filter.format(**metadata["file_metadata"][0])
        label_filter = pd.read_parquet(label_filter, columns=["label"])["label"]
    if stacked_image_tuple is not None:
        _, stacked_file_list, stacked_metadata = stacked_image_tuple
        assert metadata["id"] == stacked_metadata["id"], (
            f"{metadata['id']} != {stacked_metadata['id']}"
        )

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    join_df = False
    features_output_suffix = "" if join_df else "-features"
    zarr_inputs = False  # using zarr as input fails with zarr3 with credentials errors

    # for f in file_list:
    #     if not isinstance(f, (zarr.Group, zarr.Array)):
    #         zarr_inputs = False
    #         break

    if zarr_inputs and stacked_image_tuple is not None:
        for f in stacked_file_list:
            if not isinstance(f, (zarr.Group, zarr.Array)):
                zarr_inputs = False
                break
    if not zarr_inputs:
        image = _read_image(file_list, metadata)
    else:
        image = []
        for f in file_list:
            array, _, _, _ = _read_ome_zarr_array(f)
            image.append(array)
    n_channels1 = None
    if stacked_image_tuple is not None:
        if not zarr_inputs:
            stacked_image = _read_image(stacked_file_list, stacked_metadata)
            n_channels1 = image.sizes["c"]
            # clear coords to avoid issues with xr.concat
            for c in list(image.coords.keys()):
                del image.coords[c]
            for c in list(stacked_image.coords.keys()):
                del stacked_image.coords[c]
            image = xr.concat((image, stacked_image), dim="c")
        else:
            n_channels1 = 0
            for img in image:
                n_channels1 += np.prod(img.shape[:-2])
            n_channels1 = int(n_channels1)
            for f in stacked_file_list:
                array, _, _, _ = _read_ome_zarr_array(f)
                image.append(array)

    for label_name in label_name_to_features:
        features = label_name_to_features[label_name]
        output_parquet_path = f"{output_dir}{output_sep}{label_name}{output_sep}{image_key}{features_output_suffix}.parquet"
        objects_path = (
            f"{objects_dir}{objects_dir_sep}{label_name}{objects_dir_sep}{image_key}-objects.parquet"
            if objects_dir is not None
            else None
        )
        if not force and is_parquet_file(output_parquet_path):
            logger.info(f"Skipping features for {image_key} {label_name}")
            continue
        zarr_labels = get_labels(
            labels_group=labels_group,
            name=image_key,
            suffix=label_name,  # e.g. nuclei
        )

        if zarr_labels is None:
            logger.info(f"Unable to read {label_name} labels for {image_key}.")
            continue
        label_prefix = _label_name_to_prefix[label_name]
        if objects_path is None:
            logger.info(f"Find {label_name} objects for {image_key}.")
            objects_df = find_objects(zarr_labels)
            objects_path = f"{output_dir}{output_sep}{label_name}{output_sep}{image_key}-objects.parquet"
            objects_df.index.name = "label"
            objects_df.columns = f"{label_prefix}_" + objects_df.columns
            _to_parquet(
                objects_df,
                objects_path,
                write_index=True,
                compute=True,
                custom_metadata=dict(scallops=json.dumps(cli_metadata()))
                if not no_version
                else None,
            )
        else:
            logger.info(f"Loading objects from {objects_path}.")

        objects_df = pd.read_parquet(objects_path)

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
        logger.info(
            f"{image_key} {label_name} {len(features):,} {pluralize('feature', len(features))}: "
            f"{', '.join(features)}"
        )
        if label_filter is not None:
            objects_df = objects_df[objects_df.index.isin(label_filter)]
        min_max_area = label_name_to_min_max_area.get(label_name)
        area_column = f"{label_prefix}_AreaShape_Area"
        n_labels = len(objects_df)
        prefix = ""
        if min_max_area[0] is not None or min_max_area[1] is not None:
            area_query = []
            if min_max_area[0] is not None:
                area_query.append(f"{area_column}>={min_max_area[0]}")
            if min_max_area[1] is not None:
                area_query.append(f"{area_column}<={min_max_area[1]}")
            objects_df = objects_df.query("&".join(area_query))
            n_labels_filtered = n_labels - len(objects_df)
            prefix = f"Removed {n_labels_filtered:,} out of "
        logger.info(
            f"{prefix}{n_labels:,} {pluralize('label', n_labels)}. "
            f"Area: {objects_df[area_column].min():,.0f} to {objects_df[area_column].max():,.0f}."
        )

        df = label_features(
            objects_df=objects_df,
            label_image=zarr_labels if zarr_inputs else da.from_zarr(zarr_labels),
            intensity_image=image if zarr_inputs else image.data,
            features=features,
            normalize=normalize,
            channel_names=channel_names,
        )
        # df will be None if only area and coordinates requested

        if df is not None:
            fs = fsspec.url_to_fs(output_parquet_path)[0]
            if fs.exists(output_parquet_path):
                fs.rm(output_parquet_path, recursive=True)
            df.index.name = "label"
            df.columns = f"{label_prefix}_" + df.columns

            if isinstance(df, pd.DataFrame):
                table = pa.Table.from_pandas(df, preserve_index=True)
                if not no_version:
                    table = table.replace_schema_metadata(
                        {
                            "scallops".encode(): json.dumps(cli_metadata()).encode(),
                            **table.schema.metadata,
                        }
                    )
                fs, output_file = fsspec.url_to_fs(output_parquet_path)
                pq.write_table(
                    table,
                    output_parquet_path,
                    filesystem=fs,
                )

            else:
                _to_parquet(
                    df,
                    output_parquet_path,
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
                output_parquet_path, columns=features_plot_label
            )
            centroid_columns = [
                label_prefix + "_centroid-1",
                label_name + "_centroid-0",
            ]
            df = objects_df[centroid_columns].join(df_features)
            pdf_path = (
                f"{output_dir}{output_sep}{label_name}{output_sep}{image_key}.pdf"
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
    objects_dir = arguments.objects
    subset = arguments.subset
    force = arguments.force
    groupby = arguments.groupby
    channel_names = arguments.channel_rename
    stack_images = arguments.stack_images
    label_filter = arguments.label_filter
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

    objects_dir_sep = None
    if objects_dir is not None:
        objects_dir_sep = fsspec.core.url_to_fs(objects_dir)[0].sep
        objects_dir = objects_dir.rstrip(objects_dir_sep)
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
    labels_path = arguments.labels
    no_version = arguments.no_version
    assert labels_path is not None, "No labels provided"
    label_root = zarr.open(labels_path, mode="r")
    labels_group = label_root["labels"]
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
                output_sep=output_fs.sep,
                objects_dir=objects_dir,
                objects_dir_sep=objects_dir_sep,
                labels_group=labels_group,
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
