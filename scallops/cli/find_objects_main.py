from __future__ import annotations

from scallops.cli.args import (
    SubParsers,
    add_command,
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    labels_arg,
    lazy_handler,
    no_version_arg,
    output_dir_arg,
)


def _create_parser(subparsers: SubParsers, default_help: bool) -> None:
    parser = add_command(
        subparsers,
        "find-objects",
        default_help,
        help="Find objects in a labeled array.",
        description="Find objects in a labeled array and output Parquet file with label"
        " as index.",
    )
    required = parser.add_argument_group("required arguments")
    labels_arg(required)

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

    no_version_arg(parser)
    parser.set_defaults(
        func=lazy_handler("scallops.cli.find_objects:run_pipeline_find_objects"),
    )
