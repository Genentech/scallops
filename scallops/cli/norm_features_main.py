import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    no_version_arg,
)


def _run_norm_features(arguments: argparse.Namespace):
    from scallops.cli.norm_features import run_pipeline_norm_features

    run_pipeline_norm_features(arguments)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    parser = subparsers.add_parser(
        "norm-features",
        help="Normalize features from output of `merge` command",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--input", help="Path to merged file(s)", required=True, nargs="+"
    )

    required.add_argument(
        "--output",
        help="Path to save normalized features in Zarr or Parquet format",
        required=True,
    )
    parser.add_argument(
        "--features",
        help="Features to include. If not specified, all features are used.",
        nargs="*",
    )

    parser.add_argument(
        "--label-filter",
        help="Expression to filter labels (e.g. barcode_Q_mean_0/barcode_Q_mean > 0.5)",
    )

    parser.add_argument(
        "--by",
        help="Stratify by groups when normalizing.",
        nargs="*",
    )

    parser.add_argument(
        "--reference",
        help="Reference expression to normalize to (e.g. gene_symbol=='NTC').",
    )

    parser.add_argument(
        "--no-robust",
        help="Do not use robust statistics for normalization.",
        action="store_true",
    )
    parser.add_argument(
        "--method",
        help="Normalization method",
        choices=["zscore", "local-zscore"],
        default="zscore",
    )
    parser.add_argument(
        "--neighbors",
        help="Number of neighbors for local z-score",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--no-centering",
        help="Do not center the data before scaling.",
        action="store_true",
    )
    parser.add_argument(
        "--no-scaling",
        help="Do not scale the data by dividing by standard deviation.",
        action="store_true",
    )

    parser.add_argument(
        "--metadata",
        help="Path to CSV or Parquet file containing metadata to join with merged data.",
    )
    parser.add_argument(
        "--join",
        help="Field(s) to join on",
        nargs="*",
    )

    parser.add_argument(
        "--mad-scale-factor",
        help="Numerical scale factor to divide median absolute deviation. "
        "The string “normal” is also accepted, and results in scale being the"
        " inverse of the standard normal quantile function at 0.75",
        default="normal",
        type=str,
    )
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_norm_features)
