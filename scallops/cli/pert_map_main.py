import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    no_version_arg,
)


def _run_filter_data(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_filter_data

    run_filter_data(arguments)


def _run_pca(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_pca

    run_pca(arguments)


def _run_tvn(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_tvn

    run_tvn(arguments)


def _run_aggregate(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_aggregate

    run_aggregate(arguments)


def input_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Path to one or more zarr, h5ad, or Parquet files or a "
        "pattern to match files (e.g. s3://foo/*.zarr).",
    )


def common_args(
    parser: argparse.ArgumentParser,
    metadata: bool = True,
    pre_rechunk: bool = True,
    post_rechunk: bool = True,
    dask_client_value: str | None = None,
    rechunk_features: str | None = None,
    rechunk_labels: str | None = None,
):
    if metadata:
        metadata_args(parser)
    if pre_rechunk:
        pre_rechunk_args(
            parser, rechunk_features=rechunk_features, rechunk_labels=rechunk_labels
        )
    if post_rechunk:
        post_rechunk_args(
            parser, rechunk_features=rechunk_features, rechunk_labels=rechunk_labels
        )
    dask_client_arg(parser, dask_client_value)
    dask_cluster_arg(parser)
    force_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)


def pre_rechunk_args(
    parser: argparse.ArgumentParser,
    rechunk_features: str | None = None,
    rechunk_labels: str | None = None,
):
    parser.add_argument(
        "--pre-rechunk-labels",
        type=str,
        dest="rechunk_labels",
        default=rechunk_labels,
        help="Rechunk dataset labels before processing.",
    )

    parser.add_argument(
        "--pre-rechunk-features",
        type=str,
        dest="rechunk_features",
        default=rechunk_features,
        help="Rechunk dataset features before processing.",
    )


def post_rechunk_args(
    parser: argparse.ArgumentParser,
    rechunk_features: str | None = None,
    rechunk_labels: str | None = None,
):
    parser.add_argument(
        "--post-rechunk-labels",
        type=str,
        default=rechunk_labels,
        help="Rechunk dataset labels after processing.",
    )

    parser.add_argument(
        "--post-rechunk-features",
        type=str,
        default=rechunk_features,
        help="Rechunk dataset features after processing.",
    )


def _run_norm_features(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_norm_features

    run_norm_features(arguments)


def _run_similarity_matrix(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_similarity_matrix

    run_similarity_matrix(arguments)


def _run_recall(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_recall

    run_recall(arguments)


def filter_args(
    parser: argparse.ArgumentParser,
    label_filter: bool = True,
    feature_filter: bool = True,
):
    if label_filter:
        parser.add_argument(
            "--label-filter",
            type=str,
            help="Query string to filter dataset before processing (e.g. gene_symbol!='foo') or path to "
            "Parquet file containing label identifiers.",
        )
    if feature_filter:
        parser.add_argument(
            "--feature-filter",
            type=str,
            help="Query string to filter dataset before processing (e.g. gene_symbol!='foo') or path to "
            "Parquet file containing label identifiers.",
        )


def metadata_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--metadata",
        help="Path to CSV or Parquet file containing metadata to join with dataset.",
    )
    parser.add_argument(
        "--join",
        help="Field(s) in metadata to join on",
        nargs="*",
    )


def _create_similarity_matrix_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "similarity-matrix",
        help="Create pairwise similarity matrix",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to save result in zarr or h5ad format",
        required=True,
    )

    required.add_argument(
        "--by",
        help="Perturbation column(s) in dataset observations to aggregate by.",
        nargs="+",
    )

    common_args(parser=parser, metadata=False, pre_rechunk=False, post_rechunk=False)
    parser.set_defaults(func=_run_similarity_matrix)


def _create_aggregate_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "aggregate",
        help="Run aggregatation",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to save result in zarr or h5ad format",
        required=True,
    )

    required.add_argument(
        "--by",
        help="Perturbation column(s) in dataset observations to aggregate by.",
        nargs="+",
    )
    parser.add_argument(
        "--center-reference-query",
        help="Center the data to a reference before aggregating (e.g. gene_symbol=='NTC')",
    )
    filter_args(parser)

    common_args(parser, pre_rechunk=True, post_rechunk=True)
    parser.set_defaults(func=_run_aggregate)


def _create_tvn_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    parser = subparsers.add_parser(
        "tvn",
        help="Run TNV",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to save result in zarr or h5ad format",
        required=True,
    )
    required.add_argument(
        "--reference-query",
        help="Query to extract reference observations (e.g. gene_symbol=='NTC')",
    )
    parser.add_argument(
        "--by",
        help="Further align control and treatments in each group, using the covariance matrix of all negative "
        "(reference) controls as the target and the covariance matrix of each group of negative controls "
        "as the source.",
        nargs="*",
    )
    filter_args(parser)

    common_args(parser, pre_rechunk=True, post_rechunk=True, dask_client_value="none")
    parser.set_defaults(func=_run_tvn)


def _create_recall_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "recall",
        help="Run recall",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to save result in Parquet format",
        required=True,
    )
    required.add_argument(
        "--ground-truth-corum",
        help="Path(s) to ground truth datasets from CORUM",
        nargs="+",
    )
    required.add_argument(
        "--threshold",
        help="Recall threshold",
        nargs="+",
        type=float,
        default=[0.99, 0.95, 0.01, 0.05],
    )

    common_args(
        parser,
        metadata=False,
        pre_rechunk=False,
        post_rechunk=False,
        dask_client_value="none",
    )
    parser.set_defaults(func=_run_recall)


def _create_pca_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    parser = subparsers.add_parser(
        "pca",
        help="Run PCA",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to save result in zarr or h5ad format",
        required=True,
    )
    filter_args(parser)

    parser.add_argument(
        "--whiten",
        action="store_true",
        help="When True the components_ vectors are multiplied by the "
        "square root of n_samples and then divided by the singular "
        "values to ensure uncorrelated outputs with unit "
        "component-wise variances.",
    )
    parser.add_argument(
        "--components", type=int, default=128, help="Number of principal components"
    )
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    common_args(parser, pre_rechunk=True, post_rechunk=True)
    parser.set_defaults(func=_run_pca)


def _create_normalize_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "normalize",
        help="Normalize features",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to save normalized features in zarr, h5ad, or Parquet format",
        required=True,
    )
    filter_args(parser)

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
        "--robust",
        help="Use robust statistics for normalization.",
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
        "--mad-scale-factor",
        help="Numerical scale factor to divide median absolute deviation. "
        "The string “normal” is also accepted, and results in scale being the"
        " inverse of the standard normal quantile function at 0.75",
        default="normal",
        type=str,
    )
    parser.add_argument(
        "--max-value", help="Truncate to this value after scaling", type=float
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size to use for local z-score scaling to conserve memory",
        default=25000,
        type=int,
    )
    parser.add_argument(
        "--centroid-columns",
        help="Columns for y and x centroids to use for local zscore.",
        default=["Nuclei_AreaShape_Center_Y", "Nuclei_AreaShape_Center_X"],
        nargs=2,
    )

    common_args(parser, pre_rechunk=True, post_rechunk=True)
    parser.set_defaults(func=_run_norm_features)


def _create_filter_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "filter",
        help="Filter labels and features",
        description="Filter labels and features.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)
    required.add_argument(
        "--output",
        help="Path to save result in zarr or h5ad format",
        required=True,
    )
    filter_args(parser)
    parser.add_argument(
        "--min-feature-variance",
        default=0.1,
        type=float,
        help="Maximum median feature variance across `by` to retain a feature. "
        "Set to -1 to disable.",
    )
    parser.add_argument(
        "--max-feature-variance",
        type=float,
        help="Maximum median feature variance across `by` to retain a feature.",
    )
    parser.add_argument(
        "--max-cell-fraction-not-finite",
        default=0.25,
        type=float,
        help="Maximum fraction of non-finite values allowed per cell",
    )

    parser.add_argument(
        "--by",
        help="Metadata column(s) in dataset to stratify variance computation (e.g. plate well).",
        nargs="*",
    )
    common_args(
        parser,
        pre_rechunk=True,
        post_rechunk=True,
        rechunk_features="auto",
        rechunk_labels="auto",
    )
    parser.set_defaults(
        func=_run_filter_data,
    )


def _run_rank_features(arguments: argparse.Namespace):
    from scallops.cli.pert_map import run_rank_features

    run_rank_features(arguments)


def _create_rank_parser(
    subparsers: argparse.ArgumentParser, default_help: bool
) -> None:
    parser = subparsers.add_parser(
        "rank",
        help="Rank features from output of `merge` command",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    input_arg(required)

    required.add_argument(
        "--output",
        help="Path to Parquet file containing ranked features.",
    )
    filter_args(parser)
    parser.add_argument(
        "--rank-method",
        help="Method to rank features",
        choices=["welch_t", "student_t", "mannwhitney"],
        default="welch_t",
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

    common_args(parser, pre_rechunk=True, post_rechunk=False)
    parser.set_defaults(func=_run_rank_features)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool):
    parser = subparsers.add_parser(
        "pert-map",
        help="Perturbation map processing",
        description="Perturbation map processing.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    subparsers = parser.add_subparsers(help="Sub-command help.")
    _create_filter_parser(subparsers, default_help)
    _create_normalize_parser(subparsers, default_help)
    _create_rank_parser(subparsers, default_help)
    _create_pca_parser(subparsers, default_help)
    _create_tvn_parser(subparsers, default_help)
    _create_aggregate_parser(subparsers, default_help)
    _create_similarity_matrix_parser(subparsers, default_help)
    _create_recall_parser(subparsers, default_help)
