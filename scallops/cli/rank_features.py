"""Module for the Command-Line Interface (CLI) related to ranking features.

Authors:
    - The SCALLOPS development team
"""

import argparse
import json
import os

import dask.dataframe as dd

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.features.rank import rank_features
from scallops.features.util import (
    _join_metadata,
    _query_anndata,
    _read_data,
    _slice_anndata,
)
from scallops.io import _to_parquet, is_parquet_file

logger = _get_cli_logger()


def run_pipeline_rank_features(arguments: argparse.Namespace):
    paths = arguments.input
    features = arguments.features
    method = arguments.rank_method
    label_filter = arguments.label_filter
    rank_output = arguments.output
    if rank_output is None:
        rank_output = os.path.splitext(os.path.basename(paths[0]))[0] + ".parquet"
        if len(paths) > 1:
            logger.info(f"Saving results to {rank_output}")
    rank_groups = arguments.by

    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")

    force = arguments.force
    no_version = arguments.no_version
    perturbation_column = arguments.perturbation
    min_labels = arguments.min_labels
    reference_value = arguments.reference
    iqr_multiplier = arguments.iqr_multiplier

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    if not rank_output.lower().endswith(
        ".parquet"
    ) and not rank_output.lower().endswith(".pq"):
        rank_output = rank_output + ".parquet"

    if not force and is_parquet_file(rank_output):
        logger.info(
            f"{rank_output} already exists, skipping. Use --force to overwrite."
        )
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(paths, features)
        # columns_needed = set()
        # columns_needed.add(perturbation_column)
        # if rank_groups is not None:
        #     columns_needed.update(rank_groups)
        # if label_filter is not None:
        #     columns_needed.update(_get_names_from_pd_query(label_filter))
        # if join_path is not None:
        #     columns_needed.update(join_fields)
        # _load_coords(data, list(columns_needed))

        if label_filter is None:
            label_filter = f"~`{perturbation_column}`.isna()"
        data = _slice_anndata(data, _query_anndata(data, label_filter).index)

        if join_path is not None:
            _join_metadata(
                data,
                dd.read_csv(join_path)
                if not join_path.lower().endswith(".parquet")
                or join_path.lower().endswith(".pq")
                else dd.read_parquet(join_path),
                join_fields,
            )
        rank_df = rank_features(
            data=data,
            rank_groups=rank_groups,
            perturbation_column=perturbation_column,
            reference_value=reference_value,
            method=method,
            min_labels=min_labels,
            iqr_multiplier=iqr_multiplier,
        )

        _to_parquet(
            rank_df,
            rank_output,
            write_index=False,
            compute=True,
            custom_metadata=dict(scallops=json.dumps(metadata)),
        )
