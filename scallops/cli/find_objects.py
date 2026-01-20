"""Module for the Command-Line Interface (CLI) for finding objects.


Authors:
    - The SCALLOPS development team
"""

import argparse
import json

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


def _execute(
    label_tuple: tuple[tuple[str, ...], list[str | Group], dict],
    output_dir: str,
    output_sep: str,
    force: bool = False,
    no_version: bool = False,
):
    group, file_list, metadata = label_tuple
    assert len(file_list) == 1

    label_name = group[len(group) - 1]
    image_key = "-".join(group[:-1])  # exclude suffix from key
    path = (
        f"{output_dir}{output_sep}{label_name}{output_sep}{image_key}-objects.parquet"
    )
    fs = fsspec.url_to_fs(path)[0]
    if fs.exists(path):
        if force:
            fs.rm(path, recursive=True)
        else:
            logger.info(f"Skipping finding objects for {metadata['id']}.")
            return
    logger.info(f"Finding objects for {metadata['id']}.")
    array = file_list[0]["0"]
    df = find_objects(array)
    df.index.name = "label"
    df.columns = f"{_label_name_to_prefix[label_name]}_" + df.columns
    _to_parquet(
        df,
        path,
        write_index=True,
        compute=True,
        custom_metadata=dict(scallops=json.dumps(cli_metadata()))
        if not no_version
        else None,
    )
    logger.info(f"Saved objects for {metadata['id']} to {path}.")


def run_pipeline_find_objects(arguments: argparse.Namespace) -> None:
    labels_path = arguments.labels
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

    label_root = zarr.open(labels_path, mode="r")
    labels_group = label_root["labels"]
    _, _, keys = _create_file_regex(label_pattern)
    gen = _set_up_experiment(
        image_path=labels_group,
        files_pattern=label_pattern,
        group_by=list(keys),
        subset=subset,
    )

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        [
            _execute(
                label_tuple=g,
                output_dir=output_dir,
                output_sep=output_fs.sep,
                force=force,
                no_version=no_version,
            )
            for g in gen
            if label_suffix is None or g[0][len(g[0]) - 1] in label_suffix
        ]
