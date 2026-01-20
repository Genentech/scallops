import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    no_version_arg,
    output_dir_arg,
    verbose_arg,
)


def _run_pipeline_find_objects(arguments: argparse.Namespace):
    from scallops.cli.find_objects import run_pipeline_find_objects

    run_pipeline_find_objects(arguments)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    parser = subparsers.add_parser(
        "find-objects",
        help="Find objects in a labeled array.",
        description="Find objects in a labeled array and output Parquet file with label"
        " as index.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--labels",
        required=True,
        help="Path to zarr directory containing labels",
    )
    output_dir_arg(required)
    required.add_argument(
        "--label-pattern",
        required=True,
        help="Format string to extract metadata from labels (e.g. {well})",
    )
    parser.add_argument(
        "--label-suffix",
        nargs="*",
        default=["cell", "cytosol", "nuclei"],
        help="Label suffixes to include (e.g. nuclei, cell, cytosol)",
    )

    parser.add_argument(
        "-s", "--subset", nargs="*", help="Subset of labels to include."
    )
    force_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    verbose_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(
        func=_run_pipeline_find_objects,
    )
