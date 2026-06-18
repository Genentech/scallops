"""Module for the Command-Line Interface (CLI) for finding objects.


Authors:
    - The SCALLOPS development team
"""

import argparse
import json

import dask.array as da
import fsspec
import zarr
from zarr import Group

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.features.constants import _label_name_to_prefix
from scallops.features.find_objects import find_objects
from scallops.io import _create_file_regex, _set_up_experiment, _to_parquet

logger = _get_cli_logger()


def get_path(
    output_dir: str,
    output_sep: str,
    label_name: str,
    image_key: str,
    timepoint: str | None = None,
    suffix="",
):
    return (
        (f"{output_dir}{output_sep}{label_name}{output_sep}{image_key}{suffix}")
        if timepoint is None
        else f"{output_dir}{output_sep}{label_name}{output_sep}t={timepoint}{output_sep}{image_key}{suffix}"
    )


def _execute(
    label_tuple: tuple[tuple[str, ...], list[str | Group], dict],
    timepoint: str | None,
    output_dir: str,
    output_sep: str,
    force: bool,
    no_version: bool,
):
    group, file_list, metadata = label_tuple
    assert len(file_list) == 1
    label_name = group[len(group) - 1]
    image_key = "-".join(group[:-1])  # exclude suffix from key
    path = get_path(
        output_dir,
        output_sep,
        label_name,
        image_key,
        timepoint,
        suffix="-objects.parquet",
    )
    fs = fsspec.url_to_fs(path)[0]
    if fs.exists(path):
        if force:
            fs.rm(path, recursive=True)
        else:
            logger.info(
                f"Skipping find objects for {metadata['id']}{' at t=' + timepoint if timepoint is not None else ''}."
            )
            return
    logger.info(
        f"Find objects for {metadata['id']}{' at t=' + timepoint if timepoint is not None else ''}."
    )
    g = file_list[0]
    array = da.from_zarr(g[list(g.keys())[0]])

    if timepoint is not None:
        timepoint_index = g.attrs["multiscales"][0]["metadata"]["t"].index(timepoint)
        array = array[timepoint_index]

    df = find_objects(array)

    df.index.name = "label"
    prefix = _label_name_to_prefix.get(label_name)
    if prefix is not None:
        df.columns = f"{prefix}_" + df.columns
    _to_parquet(
        df,
        path,
        write_index=True,
        compute=True,
        custom_metadata=dict(scallops=json.dumps(cli_metadata()))
        if not no_version
        else None,
    )
    logger.info(
        f"Saved objects for {metadata['id']}{' at t=' + timepoint if timepoint is not None else ''} to {path}."
    )


def run_pipeline_find_objects(arguments: argparse.Namespace) -> None:
    labels_paths = arguments.labels
    label_pattern = arguments.label_pattern
    label_suffix = arguments.label_suffix
    label_suffix = set(label_suffix) if label_suffix is not None else None
    # assume labels are named label_pattern-{label_type}
    label_pattern = label_pattern + "-{label_type}"
    no_version = arguments.no_version
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads(threads_per_worker=1)

    output_dir = arguments.output
    subset = arguments.subset
    if subset is not None:
        for i in range(len(subset)):
            subset[i] = subset[i] + "-*"
    force = arguments.force

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)
    _, _, keys = _create_file_regex(label_pattern)
    keys = list(keys)
    label_tuples = []
    timepoints = []

    for path in labels_paths:
        label_root = zarr.open(path, mode="r")
        labels_group = label_root.get("labels")
        if labels_group is not None:
            gen = _set_up_experiment(
                image_path=labels_group,
                files_pattern=label_pattern,
                group_by=keys,
                subset=subset,
            )
            for label_tuple in gen:
                label_key, file_list, metadata = label_tuple

                if (
                    label_suffix is None
                    or label_key[len(label_key) - 1] in label_suffix
                ):
                    assert len(file_list) == 1
                    g = file_list[0]
                    zarr_metadata = g.attrs["multiscales"][0]["metadata"]

                    if "t" not in zarr_metadata:
                        label_tuples.append(label_tuple)
                        timepoints.append(None)
                    else:
                        for timepoint_ in zarr_metadata["t"]:
                            label_tuples.append(label_tuple)
                            timepoints.append(timepoint_)

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        for i in range(len(label_tuples)):
            _execute(
                label_tuple=label_tuples[i],
                timepoint=timepoints[i],
                output_dir=output_dir,
                output_sep=output_fs.sep,
                force=force,
                no_version=no_version,
            )
