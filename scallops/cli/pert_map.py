"""Module for the Command-Line Interface (CLI) related to normalizing features.

Authors:
    - The SCALLOPS development team
"""

import argparse
import json
import os

import anndata
import dask.dataframe as dd
import fsspec
import numpy as np
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
from scallops.features.map_eval import pairwise_similarities, read_corum, recall
from scallops.features.normalize import (
    _convert_scale,
    normalize_features,
    typical_variation_normalization,
)
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
    feature_filter: str | None = None,
    label_filter: str | None = None,
    use_dask: bool = True,
) -> anndata.AnnData:
    if label_filter is not None:
        if fsspec.url_to_fs(label_filter)[0].exists(label_filter):
            label_filter = pd.read_parquet(label_filter).index
        elif label_filter.endswith(".parquet"):
            logger.warning(f"{label_filter} path not found.")
    if feature_filter is not None:
        if fsspec.url_to_fs(feature_filter)[0].exists(feature_filter):
            feature_filter = pd.read_parquet(feature_filter).index
        elif feature_filter.endswith(".parquet"):
            logger.warning(f"{feature_filter} path not found.")
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
                d = dd.read_parquet(path) if use_dask else pd.read_parquet(path)
                d = pandas_to_anndata(d)
            elif path.endswith(".zarr") or path.endswith(".h5ad"):
                d = read_anndata(path, dask=use_dask)
            else:
                raise ValueError(f"Unrecognized file type: {path}")
            assert not d.obs.index.has_duplicates, "Duplicate obs index detected."
            assert not d.var.index.has_duplicates, "Duplicate var index detected."
            results.append(d)

    data = anndata.concat(results, index_unique="-")

    assert not data.obs.index.has_duplicates
    if isinstance(label_filter, str):
        label_filter = data.obs.query(label_filter).index
    if isinstance(feature_filter, str):
        feature_filter = data.var.query(feature_filter).index
    if label_filter is not None or feature_filter is not None:
        data = _slice_anndata(data, label_filter, feature_filter)
    return data


def rechunk(
    data: anndata.AnnData,
    rechunk_label_size: str | None,
    rechunk_feature_size: str | None,
) -> anndata.AnnData:
    if rechunk_label_size == "":
        rechunk_label_size = None
    if rechunk_feature_size == "":
        rechunk_feature_size = None
    if rechunk_label_size is not None or rechunk_feature_size is not None:
        if rechunk_label_size is not None and rechunk_label_size.isdigit():
            rechunk_label_size = int(rechunk_label_size)
        if rechunk_feature_size is not None and rechunk_feature_size.isdigit():
            rechunk_feature_size = int(rechunk_feature_size)
        if rechunk_label_size is None:
            rechunk_label_size = data.X.chunksize[0]
        if rechunk_feature_size is None:
            rechunk_feature_size = data.X.chunksize[1]
        data.X = data.X.rechunk((rechunk_label_size, rechunk_feature_size))
    return data


def run_recall(arguments: argparse.Namespace):
    data_paths = arguments.input
    force = arguments.force
    no_version = arguments.no_version
    dask_server_url = arguments.client
    ground_truth_paths = arguments.ground_truth_corum
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    output = arguments.output
    recall_thresholds = arguments.threshold
    if not force and is_parquet_file(output):
        logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
        return
    ground_truth = []
    for i in range(len(ground_truth_paths)):
        corum_df = read_corum(ground_truth_paths[i])
        corum_df = corum_df.set_index(corum_df["a"] + "-" + corum_df["b"])
        corum_name = os.path.basename(ground_truth_paths[i])
        ground_truth.append((corum_name, corum_df))
    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        similarity_data = _read_data(data_paths)
        similarity_data.X = similarity_data.X.compute()  # load into memory
        results = []
        gene_symbols = similarity_data.obs.index.values
        for ground_truth_name, ground_truth_df in ground_truth:
            indices_a = []
            indices_b = []

            for i in range(len(gene_symbols)):
                gene1 = gene_symbols[i]
                for j in range(i):
                    key = gene1 + "-" + gene_symbols[j]
                    if key in ground_truth_df.index:
                        indices_a.append(i)
                        indices_b.append(j)
            if len(indices_a) == 0:
                raise ValueError("No genes found in ground truth.")
            indices_a = np.array(indices_a)
            indices_b = np.array(indices_b)

            query_distribution = similarity_data.X[indices_a, indices_b]
            null_distribution = similarity_data.X[
                np.tril_indices(similarity_data.shape[0], k=-1)
            ]
            result = recall(
                query_distribution=query_distribution,
                null_distribution=null_distribution,
                recall_thresholds=recall_thresholds,
            )
            result["null"] = "all"
            result["reference"] = ground_truth_name
            results.append(result)

            in_corum = similarity_data.obs.index.isin(ground_truth_df["a"])
            similarity_data_corum = similarity_data.X[in_corum, in_corum]
            null_distribution_corum = similarity_data_corum[
                np.tril_indices(similarity_data_corum.shape[0], k=-1)
            ]
            result = recall(
                query_distribution=query_distribution,
                null_distribution=null_distribution_corum,
                recall_thresholds=recall_thresholds,
            )
            result["null"] = "corum"
            result["reference"] = ground_truth_name

        df = pd.concat(results)
        df["threshold"] = df["threshold"].astype(str)
        fs, output_dir = fsspec.url_to_fs(os.path.dirname(output))
        fs.makedirs(output_dir, exist_ok=True)
        _to_parquet(
            df,
            output,
            write_index=False,
            custom_metadata=dict(scallops=json.dumps(metadata)),
        )


def run_similarity_matrix(arguments: argparse.Namespace):
    data_paths = arguments.input
    force = arguments.force
    no_version = arguments.no_version

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
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
        data = _read_data(data_paths)
        logger.info(f"# labels: {data.shape[0]:,}, # features: {data.shape[1]:,}")
        data = anndata.AnnData(
            X=pairwise_similarities(data), obs=data.obs, var=data.obs
        )
        _write_anndata(data, output, metadata, None, None)


def run_aggregate(arguments: argparse.Namespace):
    data_paths = arguments.input
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_labels
    rechunk_feature_size = arguments.rechunk_features
    force = arguments.force
    no_version = arguments.no_version

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads(threads_per_worker=8)
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
        data = _read_data(data_paths, feature_filter, label_filter)
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
            data = normalize_features(
                data=data,
                normalize="zscore",
                scaling=False,
                robust=False,
                reference_query=center_reference_query,
            )

        data = agg_features(
            data=data,
            by=by,
            agg_func="mean",
        )
        fs, output_dir = fsspec.url_to_fs(os.path.dirname(output))
        fs.makedirs(output_dir, exist_ok=True)
        data.uns["scallops"] = _fix_json(metadata)
        if output.lower().endswith(".zarr"):
            data.write_zarr(output, convert_strings_to_categoricals=False)
        else:
            data.write_h5ad(output, convert_strings_to_categoricals=False)


def run_tvn(arguments: argparse.Namespace):
    data_paths = arguments.input
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_labels
    rechunk_feature_size = arguments.rechunk_features
    post_rechunk_label_size = arguments.post_rechunk_labels
    post_rechunk_feature_size = arguments.post_rechunk_features
    force = arguments.force
    no_version = arguments.no_version

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
        data = _read_data(data_paths, feature_filter, label_filter)
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

        _write_anndata(
            data, output, metadata, post_rechunk_label_size, post_rechunk_feature_size
        )


def run_pca(arguments: argparse.Namespace):
    data_paths = arguments.input
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    batch_size = arguments.batch_size
    if batch_size <= 0:
        batch_size = None
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_labels
    rechunk_feature_size = arguments.rechunk_features
    post_rechunk_label_size = arguments.post_rechunk_labels
    post_rechunk_feature_size = arguments.post_rechunk_features
    force = arguments.force
    no_version = arguments.no_version

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
    dask_config = {}
    if (
        batch_size is None
        and rechunk_label_size is not None
        or rechunk_feature_size is not None
    ):
        dask_config = {"array.rechunk.method": "tasks"}  # for dask PCA
    with (
        _create_default_dask_config(dask_config),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, feature_filter, label_filter)
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
            data=data, n_components=n_components, whiten=whiten, batch_size=batch_size
        )
        _write_anndata(
            data, output, metadata, post_rechunk_label_size, post_rechunk_feature_size
        )


def _write_anndata(data, output, metadata, rechunk_label_size, rechunk_feature_size):
    output = output.rstrip("/")
    fs, output_dir = fsspec.url_to_fs(os.path.dirname(output))
    fs.makedirs(output_dir, exist_ok=True)
    data.uns["scallops"] = _fix_json(metadata)
    data = rechunk(data, rechunk_label_size, rechunk_feature_size)
    if output.lower().endswith(".zarr"):
        data.write_zarr(output, convert_strings_to_categoricals=False)
    else:
        data.write_h5ad(output, convert_strings_to_categoricals=False)


def run_rank_features(arguments: argparse.Namespace):
    data_paths = arguments.input
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_labels
    rechunk_feature_size = arguments.rechunk_features

    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by

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
        data = _read_data(data_paths, feature_filter, label_filter)
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

        fs, output_dir = fsspec.url_to_fs(os.path.dirname(rank_output))
        fs.makedirs(output_dir, exist_ok=True)

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
    data_paths = arguments.input
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")

    rechunk_label_size = arguments.rechunk_labels
    rechunk_feature_size = arguments.rechunk_features
    post_rechunk_label_size = arguments.post_rechunk_labels
    post_rechunk_feature_size = arguments.post_rechunk_features
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    reference = arguments.reference
    output = arguments.output

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
    max_value = arguments.max_value
    batch_size = arguments.batch_size
    if batch_size < 0:
        batch_size = None
    centroid_column_names = arguments.centroid_columns
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads(threads_per_worker=8)

    output_ext = os.path.splitext(os.path.basename(output.lower()))[1]
    if output_ext == ".zarr":
        output_format = "zarr"
    elif output_ext == ".h5ad":
        output_format = "h5ad"
    else:
        output_format = "parquet"
    if not force:
        skip = False
        if output_format in ("zarr", "h5ad") and is_anndata(output):
            skip = True

        elif output_format == "parquet" and is_parquet_file(output):
            skip = True
        if skip:
            logger.info(f"{output} already exists, skipping. Use --force to overwrite.")
            return

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(
            {
                "distributed.scheduler.worker-saturation": 1.0,
                "optimization.fuse.active": False,
                "distributed.admin.large-graph-warning-threshold": "100MB",
                "distributed.worker.resources.process": 1,
            }
        ),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, feature_filter, label_filter)
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
        _log_data_shape(data)
        if centering or scaling:
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
                max_value=max_value,
                batch_size=batch_size,
                centroid_column_names=centroid_column_names,
            )
            _log_data_shape(data, "After normalization, ")
        else:
            logger.info("No normalization")
        fs, output_dir = fsspec.url_to_fs(os.path.dirname(output))
        fs.makedirs(output_dir, exist_ok=True)

        if output_format in ("zarr", "h5ad"):
            _write_anndata(
                data,
                output,
                metadata,
                post_rechunk_label_size,
                post_rechunk_feature_size,
            )
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
            fs, output_file = fsspec.url_to_fs(output)
            pq.write_table(
                table,
                output,
                filesystem=fs,
            )


def run_filter_data(arguments: argparse.Namespace) -> None:
    data_paths = arguments.input
    label_filter = arguments.label_filter
    feature_filter = arguments.feature_filter
    join_path = arguments.metadata
    join_fields = arguments.join
    if join_path is not None and join_fields is None:
        raise ValueError("Please specify join fields")
    rechunk_label_size = arguments.rechunk_labels
    rechunk_feature_size = arguments.rechunk_features
    post_rechunk_label_size = arguments.post_rechunk_labels
    post_rechunk_feature_size = arguments.post_rechunk_features
    force = arguments.force
    no_version = arguments.no_version
    by = arguments.by

    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    output = arguments.output

    if not force and is_anndata(output):
        logger.info(f"Skipping {output}")
        return

    min_feature_variance = arguments.min_feature_variance
    max_feature_variance = arguments.max_feature_variance
    max_cell_fraction_not_finite = arguments.max_cell_fraction_not_finite
    if min_feature_variance is not None and min_feature_variance < 0:
        min_feature_variance = None
    if max_feature_variance is not None and max_feature_variance < 0:
        max_feature_variance = None
    if max_cell_fraction_not_finite is not None and max_cell_fraction_not_finite < 0:
        max_cell_fraction_not_finite = None

    metadata = {}
    if not no_version:
        metadata.update(cli_metadata())
    with (
        _create_default_dask_config(
            {"distributed.scheduler.locks.lease-timeout": "inf"}
        ),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        data = _read_data(data_paths, feature_filter, label_filter)
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
        _log_data_shape(data)

        data = filter_data(
            data=data,
            max_fraction_not_finite=max_cell_fraction_not_finite,
            min_variance=min_feature_variance,
            max_variance=max_feature_variance,
            by=by,
        )

        _log_data_shape(data, "After filtering, ")
        _write_anndata(
            data, output, metadata, post_rechunk_label_size, post_rechunk_feature_size
        )


def _log_data_shape(data, prefix="", log_chunk_size=True):
    logger.info(f"{prefix}# labels: {data.shape[0]:,}, # features: {data.shape[1]:,}")
    if log_chunk_size:
        logger.info(f"Chunk size: {data.X.chunksize[0]:,}, {data.X.chunksize[1]:,}")
