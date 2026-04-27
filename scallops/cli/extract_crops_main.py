import argparse
from argparse import ArgumentParser

import fsspec
import zarr
from dask.bag import from_sequence

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.extract_crops import single_crop
from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    load_json,
    no_version_arg,
    output_dir_arg,
    subset_arg,
    verbose_arg,
)
from scallops.io import (
    _set_up_experiment,
)

logger = _get_cli_logger()


def run_pipeline_extract_crops(arguments: argparse.Namespace):
    images_paths = arguments.images
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )

    image_patterns = arguments.image_pattern
    output_dir = arguments.output
    merge_dir = arguments.merge
    subset = arguments.subset
    force = arguments.force
    groupby = arguments.groupby
    crop_size = arguments.crop_size
    crop_size = (crop_size, crop_size)
    label_filter = arguments.label_filter
    percentile_min = arguments.percentile_min
    percentile_max = arguments.percentile_max
    output_format = arguments.output_format
    local_percentile_normalize = arguments.local_percentile_normalize
    local_normalize_overlap = arguments.local_percentile_overlap
    percentile_normalize = None
    if percentile_min is not None or percentile_max is not None:
        if percentile_min is None:
            percentile_min = 0
        if percentile_max is None:
            percentile_max = 100
        percentile_normalize = (percentile_min, percentile_max)

    label_name = arguments.label_name  # cell, cytosol, nuclei
    chunks = arguments.chunks
    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = _dask_workers_threads()

    merge_dir_sep = None
    if merge_dir is not None:
        merge_dir_sep = fsspec.core.url_to_fs(merge_dir)[0].sep
        merge_dir = merge_dir.rstrip(merge_dir_sep)

    output_fs, _ = fsspec.core.url_to_fs(output_dir)
    output_dir = output_dir.rstrip(output_fs.sep)

    labels_path = arguments.labels
    no_version = arguments.no_version
    assert labels_path is not None, "No labels provided"
    label_root = zarr.open(labels_path, mode="r")
    labels_group = label_root["labels"]

    image_seq = from_sequence(
        _set_up_experiment(
            images_paths,
            files_pattern=image_patterns,
            subset=subset,
            group_by=groupby,
        )
    )

    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        image_seq.starmap(
            single_crop,
            output_dir=output_dir,
            output_sep=output_fs.sep,
            merge_dir=merge_dir,
            merge_dir_sep=merge_dir_sep,
            labels_group=labels_group,
            label_filter=label_filter,
            label_name=label_name,
            percentile_normalize=percentile_normalize,
            local_normalize_overlap=local_normalize_overlap,
            local_percentile_normalize=local_percentile_normalize,
            crop_size=crop_size,
            output_format=output_format,
            chunks=chunks,
            force=force,
            no_version=no_version,
        ).compute()


def main():
    parser = ArgumentParser()

    required = parser.add_argument_group("required arguments")
    images_arg(required)
    output_dir_arg(required)
    required.add_argument(
        "--labels",
        dest="labels",
        required=True,
        help="Path to zarr directory containing labels",
    )

    image_pattern_arg(parser)

    required.add_argument(
        "--label-name",
        help="Name of labels to use. For example `nuclei` or `cell`",
        default="cell",
    )
    required.add_argument(
        "--merge",
        required=False,
        help="Path to directory containing output from `merge`",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        help="Image crop size",
        default=224,
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        help="Percentile min for normalization",
        default=0.1,
    )
    parser.add_argument(
        "--percentile-max",
        help="Percentile max for normalization",
        type=float,
        default=99.9,
    )
    parser.add_argument(
        "--local-percentile-normalize",
        help="Perform percentile normalization locally",
        action="store_true",
    )
    parser.add_argument(
        "--local-percentile-overlap",
        type=int,
        help="Overlap for local normalization",
    )

    parser.add_argument(
        "--label-filter",
        help="Expression to filter labels (e.g. barcode_Q_mean_0/barcode_Q_mean > 0.5) or path to Parquet file containing labels to include.",
    )

    parser.add_argument(
        "--chunks",
        help="Chunk size for local normalization",
        type=int,
    )
    parser.add_argument(
        "--output-format",
        choices=["tiff", "npy"],
        default="tiff",
        help="Output image format",
    )

    groupby_arg(parser)
    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    verbose_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)

    parser.set_defaults(
        func=run_pipeline_extract_crops,
    )
    args = parser.parse_args()
    args.func(args)
