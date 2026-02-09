"""Module for the Command-Line Interface (CLI) related to normalizing features.

Authors:
    - The SCALLOPS development team
"""

import argparse
import json
import os

import dask.array as da
import dask.dataframe as dd
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.features.normalize import _convert_scale, normalize_features
from scallops.features.util import (
    _join_metadata,
    _query_anndata,
    _read_data,
    _slice_anndata,
)
from scallops.io import is_parquet_file
from scallops.utils import _fix_json
from scallops.zarr_io import is_anndata_zarr

logger = _get_cli_logger()


def run_pipeline_norm_features(arguments: argparse.Namespace):
    paths = arguments.input
    features = arguments.features
    reference = arguments.reference
    label_filter = arguments.label_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")

    norm_output = arguments.output
    force = arguments.force
    no_version = arguments.no_version
    normalize_groups = arguments.by
    normalize = arguments.method
    n_neighbors = arguments.neighbors
    mad_scale_factor = arguments.mad_scale_factor
    centering = not arguments.no_centering
    scaling = not arguments.no_scaling
    if mad_scale_factor.lower() == "normal":
        mad_scale_factor = _convert_scale(mad_scale_factor)
    else:
        mad_scale_factor = float(mad_scale_factor)

    robust = not arguments.no_robust
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()
    suffix = os.path.splitext(norm_output.lower())[1]
    if suffix not in {".zarr", ".parquet", ".pq"}:
        norm_output = norm_output + ".zarr"
    output_format = "zarr" if norm_output.lower().endswith("zarr") else "parquet"
    if not force:
        skip = False
        if output_format == "zarr" and is_anndata_zarr(norm_output):
            skip = True

        elif output_format == "parquet" and is_parquet_file(norm_output):
            skip = True
        if skip:
            logger.info(
                f"{norm_output} already exists, skipping. Use --force to overwrite."
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

        if label_filter is not None:
            data = _slice_anndata(data, _query_anndata(data, label_filter))
        if join_path is not None:
            _join_metadata(
                data,
                dd.read_csv(join_path)
                if not join_path.lower().endswith(".parquet")
                or join_path.lower().endswith(".pq")
                else dd.read_parquet(join_path),
                join_fields,
            )
        logger.info(f"# labels: {data.shape[0]:,}, # features: {data.shape[1]:,}")
        if centering or scaling:
            chunks = list(data.X.chunksize)
            feature_chunk_size = 10
            if chunks[1] != feature_chunk_size:
                chunks[1] = feature_chunk_size
                data.X = data.X.rechunk(tuple(chunks))
            data = normalize_features(
                data,
                reference,
                normalize=normalize,
                robust=robust,
                normalize_groups=normalize_groups,
                n_neighbors=n_neighbors,
                mad_scale=mad_scale_factor,
                centering=centering,
                scaling=scaling,
            )
        else:
            logger.info("No normalization")

        if output_format == "zarr":
            if not da.core._check_regular_chunks(data.X.chunks):
                # need uniform chunks to save to zarr
                chunks = list(data.X.chunksize)
                chunks[0] = "auto"
                data.X = data.X.rechunk(tuple(chunks))
            data.uns["scallops"] = _fix_json(metadata)
            data.write_zarr(norm_output, convert_strings_to_categoricals=False)

        else:
            data.X = data.X.compute()
            df = data.to_df().join(data.obs)
            table = pa.Table.from_pandas(df, preserve_index=True)
            table = table.replace_schema_metadata(
                {
                    "scallops".encode(): json.dumps(metadata).encode(),
                    **table.schema.metadata,
                }
            )
            fs, output_file = fsspec.url_to_fs(norm_output)
            pq.write_table(
                table,
                norm_output,
                filesystem=fs,
            )
