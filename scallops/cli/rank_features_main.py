import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    no_version_arg,
)


def _run_rank_features(arguments: argparse.Namespace):
    from scallops.cli.rank_features import run_pipeline_rank_features

    run_pipeline_rank_features(arguments)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    parser = subparsers.add_parser(
        "rank-features",
        help="Rank features from output of `merge` command",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--input", help="Path to normalized file(s)", required=True, nargs="+"
    )

    required.add_argument(
        "--output",
        help="Path to Parquet file containing ranked features.",
    )

    parser.add_argument(
        "--features",
        help="Features to include. If not specified, all features are used.",
        nargs="*",
    )
    parser.add_argument(
        "--rank-method",
        help="Method to rank features",
        choices=["welch_t", "student_t", "mannwhitney"],
        default="welch_t",
    )
    parser.add_argument(
        "--label-filter",
        help="Expression to filter labels (e.g. barcode_Q_mean_0/barcode_Q_mean > 0.5)",
    )
    parser.add_argument(
        "--iqr-multiplier",
        help="Include values between Q25 - multiplier * IQR and Q75 - multiplier * IQR",
        type=float,
    )

    parser.add_argument(
        "--perturbation",
        help="Field name to group perturbations",
        default="gene_symbol",
    )
    parser.add_argument(
        "--reference",
        help="Reference value in `perturbation` to compare against.",
        required=True,
    )

    parser.add_argument(
        "--by",
        help="Stratify by groups when ranking.",
        nargs="*",
    )

    parser.add_argument(
        "--min-labels",
        help="Require at least `min-labels` to include perturbation",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--metadata",
        help="Path to CVS or Parquet file containing metadata to join with merged data.",
    )
    parser.add_argument(
        "--join",
        help="Field(s) to join on",
        nargs="*",
    )

    dask_client_arg(parser)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_rank_features)
