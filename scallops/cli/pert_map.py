"""Module for the Command-Line Interface (CLI) related to normalizing features.

Authors:
    - The SCALLOPS development team
"""

import argparse
import json
import os

import anndata
import dask.array as da
import dask.dataframe as dd
import fsspec
import pandas as pd
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
from scallops.features.agg import agg_features
from scallops.features.decomposition import pca
from scallops.features.map_eval import pairwise_similarities
from scallops.features.normalize import _convert_scale, normalize_features, typical_variation_normalization
from scallops.features.preprocessing import filter_data
from scallops.features.rank import rank_features
from scallops.features.util import (
    _join_metadata,
    _slice_anndata,
    pandas_to_anndata,
)
from scallops.io import _to_parquet, is_anndata, is_parquet_file, read_anndata
from scallops.utils import _fix_json

logger = _get_cli_logger()


def _read_data(
        data_paths: list[str],
        keys: list[str],
        feature_filter: str | pd.Index | None = None,
        label_filter: str | pd.Index | None = None,
) -> anndata.AnnData:
    results = []
    for data_path in data_paths:
        fs, data_path = fsspec.url_to_fs(data_path)
        if "*" in data_path:
            paths = fs.glob(data_path)
            if len(paths) == 0:
                raise ValueError(f"No files found at {data_path}.")

        else:
            paths = [data_path]
        for path in paths:
            path = fs.unstrip_protocol(path)
            if path.endswith(".parquet"):
                d = dd.read_parquet(path)
                d = pandas_to_anndata(d)
            elif path.endswith(".zarr") or path.endswith(".h5ad"):
                d = read_anndata(path, dask=True)
            else:
                raise ValueError(f"Unrecognized file type: {path}")
            assert not d.obs.index.has_duplicates, "Duplicate index detected."
            assert not d.var.index.has_duplicates, "Duplicate index detected."
            results.append(d)
    data = anndata.concat(results, keys=keys, index_unique="-")
    assert not data.obs.index.has_duplicates
    if isinstance(label_filter, str):
        label_filter = data.obs.query(label_filter).index
    if isinstance(feature_filter, str):
        feature_filter = data.var.query(feature_filter).index
    if label_filter is not None or feature_filter is not None:
        data = _slice_anndata(data, label_filter, feature_filter)
    return data


def rechunk_for_zarr(data: anndata.AnnData):
    if not da.core._check_regular_chunks(data.X.chunks):
        # need uniform chunks to save to zarr
        chunks = list(data.X.chunksize)
        chunks[0] = "auto"
        data.X = data.X.rechunk(tuple(chunks))
    return data


def rechunk(
        data: anndata.AnnData, rechunk_label_size: str, rechunk_feature_size: str
) -> anndata.AnnData:
    if rechunk_label_size is not None or rechunk_feature_size is not None:
        if rechunk_label_size is None:
            rechunk_label_size = "auto"
        if rechunk_feature_size is None:
            rechunk_feature_size = "auto"
        if rechunk_label_size.isdigit():
            rechunk_label_size = int(rechunk_label_size)
        if rechunk_feature_size.isdigit():
            rechunk_feature_size = int(rechunk_feature_size)
        data.X = data.X.rechunk((rechunk_label_size, rechunk_feature_size))
    return data


def run_similarity_matrix(arguments: argparse.Namespace):
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    by = arguments.by
    output = arguments.output

    if not force and is_anndata(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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
        data = anndata.AnnData(X=pairwise_similarities(data), obs=data.obs, var=data.obs)
        fs, output_basename = fsspec.url_to_fs(os.path.basename(output))
        fs.makedirs(output_basename, exist_ok=True)
        if output.lower().endswith(".zarr"):
            data = rechunk_for_zarr(data)
            data.uns["scallops"] = _fix_json(metadata)
            data.write_zarr(output, convert_strings_to_categoricals=False)
        else:
            data.uns["scallops"] = _fix_json(metadata)
            data.write_h5ad(output, convert_strings_to_categoricals=False)


def run_aggregate(arguments: argparse.Namespace):
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    by = arguments.by
    output = arguments.output

    if not force and is_anndata(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return
    center_reference_query = arguments.center_reference_query
    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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

        if center_reference_query is not None:
            data = normalize_features(data=data, normalize="zscore", scaling=False, robust=False,
                                      reference_query=center_reference_query)

        data = agg_features(
            data=data,
            by=by,
            agg_func="mean",
        )
        fs, output_basename = fsspec.url_to_fs(os.path.basename(output))
        fs.makedirs(output_basename, exist_ok=True)
        if output.lower().endswith(".zarr"):
            data = rechunk_for_zarr(data)
            data.uns["scallops"] = _fix_json(metadata)
            data.write_zarr(output, convert_strings_to_categoricals=False)
        else:
            data.uns["scallops"] = _fix_json(metadata)
            data.write_h5ad(output, convert_strings_to_categoricals=False)


def run_tvn(arguments: argparse.Namespace):
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    reference_query = arguments.reference_query
    by = arguments.by
    output = arguments.output

    if not force and is_anndata(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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
        data = typical_variation_normalization(
            data=data,
            reference_query=reference_query,
            by=by,
        )
        fs, output_basename = fsspec.url_to_fs(os.path.basename(output))
        fs.makedirs(output_basename, exist_ok=True)
        if output.lower().endswith(".zarr"):
            data = rechunk_for_zarr(data)
            data.uns["scallops"] = _fix_json(metadata)
            data.write_zarr(output, convert_strings_to_categoricals=False)
        else:
            data.uns["scallops"] = _fix_json(metadata)
            data.write_h5ad(output, convert_strings_to_categoricals=False)


def run_pca(arguments: argparse.Namespace):
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    n_components = arguments.components
    whiten = arguments.whiten
    output = arguments.output

    if not force and is_anndata(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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
        data = pca(
            data=data,
            n_components=n_components,
            min_std=None,
            standardize=False,
            standardize_by=None,
            whiten=whiten,
        )
        fs, output_basename = fsspec.url_to_fs(os.path.basename(output))
        fs.makedirs(output_basename, exist_ok=True)
        if output.lower().endswith(".zarr"):
            data = rechunk_for_zarr(data)
            data.uns["scallops"] = _fix_json(metadata)
            data.write_zarr(output, convert_strings_to_categoricals=False)
        else:
            data.uns["scallops"] = _fix_json(metadata)
            data.write_h5ad(output, convert_strings_to_categoricals=False)


def run_rank_features(arguments: argparse.Namespace):
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )

    method = arguments.rank_method

    rank_output = arguments.output
    if rank_output is None:
        rank_output = os.path.splitext(os.path.basename(data_paths[0]))[0] + ".parquet"
        if len(data_paths) > 1:
            logger.info(f"Saving results to {rank_output}")

    perturbation_column = arguments.perturbation
    min_labels = arguments.min_labels
    reference_value = arguments.reference
    iqr_multiplier = arguments.iqr_multiplier

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
        if label_filter is None:
            label_filter = f"~`{perturbation_column}`.isna()"
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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
        # columns_needed = set()
        # columns_needed.add(perturbation_column)
        # if by is not None:
        #     columns_needed.update(by)
        # if label_filter is not None:
        #     columns_needed.update(_get_names_from_pd_query(label_filter))
        # if join_path is not None:
        #     columns_needed.update(join_fields)
        # _load_coords(data, list(columns_needed))

        rank_df = rank_features(
            data=data,
            by=by,
            perturbation_column=perturbation_column,
            reference_value=reference_value,
            method=method,
            min_labels=min_labels,
            iqr_multiplier=iqr_multiplier,
        )
        fs, output_basename = fsspec.url_to_fs(os.path.basename(rank_output))
        fs.makedirs(output_basename, exist_ok=True)
        if isinstance(rank_df, dd.DataFrame):
            _to_parquet(
                rank_df,
                rank_output,
                write_index=False,
                custom_metadata=dict(scallops=json.dumps(metadata)),
            )
        else:
            table = pa.Table.from_pandas(rank_df, preserve_index=False)
            table = table.replace_schema_metadata(
                {
                    "scallops".encode(): json.dumps(metadata).encode(),
                    **table.schema.metadata,
                }
            )

            fs, rank_output = fsspec.url_to_fs(rank_output)
            pq.write_table(
                table,
                rank_output,
                filesystem=fs,
            )


def run_norm_features(arguments: argparse.Namespace):
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    reference = arguments.reference
    norm_output = arguments.output

    normalize = arguments.method
    n_neighbors = arguments.neighbors
    mad_scale_factor = arguments.mad_scale_factor
    centering = not arguments.no_centering
    scaling = not arguments.no_scaling
    if mad_scale_factor.lower() == "normal":
        mad_scale_factor = _convert_scale(mad_scale_factor)
    else:
        mad_scale_factor = float(mad_scale_factor)

    robust = arguments.robust

    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()
    suffix = os.path.splitext(norm_output.lower())[1]
    if suffix not in {".zarr", ".parquet", ".pq"}:
        norm_output = norm_output + ".zarr"
    output_ext = os.path.split(norm_output.lower())[1]
    if output_ext == ".zarr":
        output_format = "zarr"
    elif output_ext == ".h5ad":
        output_format = "h5ad"
    else:
        output_format = "parquet"
    if not force:
        skip = False
        if output_format in ("zarr", "h5ad") and is_anndata(norm_output):
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
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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
                by=by,
                n_neighbors=n_neighbors,
                mad_scale=mad_scale_factor,
                centering=centering,
                scaling=scaling,
            )
        else:
            logger.info("No normalization")
        fs, output_basename = fsspec.url_to_fs(os.path.basename(norm_output))
        fs.makedirs(output_basename, exist_ok=True)
        if output_format == "zarr":
            data = rechunk_for_zarr(data)
            data.uns["scallops"] = _fix_json(metadata)
            data.write_zarr(norm_output, convert_strings_to_categoricals=False)
        elif output_format == "h5ad":
            data.uns["scallops"] = _fix_json(metadata)
            data.write_h5ad(norm_output, convert_strings_to_categoricals=False)
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


def run_filter_data(arguments: argparse.Namespace) -> None:
    data_paths = arguments.dataset
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_col_size
    rechunk_feature_size = arguments.rechunk_row_size
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by
    keys = arguments.key
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    output_label_ids = arguments.output_label_ids
    output_feature_ids = arguments.output_feature_ids

    if (
            not force
            and is_parquet_file(output_label_ids)
            and is_parquet_file(output_feature_ids)
    ):
        logger.info(f"Skipping {output_label_ids}")
        return

    min_feature_variance = arguments.min_feature_variance
    max_feature_variance = arguments.max_feature_variance
    max_cell_fraction_not_finite = arguments.max_cell_fraction_not_finite
    if label_filter is not None and fsspec.url_to_fs(label_filter)[0].exists(
            label_filter
    ):
        label_filter = pd.read_parquet(label_filter).index
    if feature_filter is not None and fsspec.url_to_fs(feature_filter)[0].exists(
            feature_filter
    ):
        feature_filter = pd.read_parquet(feature_filter).index
    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, keys, feature_filter, label_filter)
        data = rechunk(data, rechunk_label_size, rechunk_feature_size)
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
        data = filter_data(
            data=data,
            max_fraction_not_finite=max_cell_fraction_not_finite,
            min_variance=min_feature_variance,
            max_variance=max_feature_variance,
            by=by,
        )
        # save indices only
        table = pa.Table.from_pandas(data.obs[[]], preserve_index=True)
        table = table.replace_schema_metadata(
            {
                "scallops".encode(): json.dumps(metadata).encode(),
                **table.schema.metadata,
            }
        )
        fs, output_label_ids = fsspec.url_to_fs(output_label_ids)
        fs.makedirs(os.path.basename(output_label_ids), exist_ok=True)
        pq.write_table(
            table,
            output_label_ids,
            filesystem=fs,
        )

        table = pa.Table.from_pandas(data.var[[]], preserve_index=True)
        table = table.replace_schema_metadata(
            {
                "scallops".encode(): json.dumps(metadata).encode(),
                **table.schema.metadata,
            }
        )
        fs, output_feature_ids = fsspec.url_to_fs(output_feature_ids)
        fs.makedirs(os.path.basename(output_feature_ids), exist_ok=True)
        pq.write_table(
            table,
            output_feature_ids,
            filesystem=fs,
        )
