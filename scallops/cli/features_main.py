import argparse
from types import MethodType

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    no_version_arg,
    output_dir_arg,
    subset_arg,
    verbose_arg,
)


def _run_pipeline_compute_features(arguments: argparse.Namespace):
    """Executes the feature computation pipeline for the 'features' command.

    This internal function imports and runs the feature computation logic using the provided
    arguments.

    :param arguments: Parsed command-line arguments for the feature computation pipeline.
    """
    from scallops.cli.features import run_pipeline_compute_features

    run_pipeline_compute_features(arguments)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    """Sets up the argument parser for the 'features' command and its arguments.

    This internal function configures the command-line interface for computing features, adding
    various arguments related to feature extraction and image processing.

    :param subparsers: The subparsers object from the main parser to which the 'features' parser is
        added.
    :param default_help: Determines whether to include default values in help messages.
    """
    parser = subparsers.add_parser(
        "features",
        help="Compute features",
        description="Compute features and output Parquet files with label as index.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )

    parser.old_help = parser.format_help

    def new_format_help(x):
        """Custom help formatter that appends feature descriptions to the help output.

        :param x: The parser object.
        """
        print(x.old_help())
        from scallops.features.info import HELP

        print(HELP)

    # Override the default help formatter
    parser.format_help = MethodType(new_format_help, parser)
    required = parser.add_argument_group("required arguments")
    # Add common arguments
    images_arg(required)

    output_dir_arg(required)
    required.add_argument(
        "--labels",
        dest="labels",
        required=True,
        help="Path to zarr directory containing labels",
    )

    generic_features_help = (
        "A space-separated list of features to extract (e.g., 'area intensity_0 corr_0_1').\n"
        "Channels are 0-indexed. Use shortcuts for efficiency:\n"
        "  - For specific channels: 'intensity_0,1,2'\n"
        "  - For all channels (wildcard): 'intensity_*'\n"
        "  - For all channel pairs: 'colocalization_*_*'"
    )

    image_pattern_arg(parser)
    # Add feature-related arguments
    parser.add_argument(
        "--features-nuclei",
        nargs="+",
        type=str,
        help=generic_features_help,
    )

    parser.add_argument(
        "--features-cell",
        nargs="+",
        type=str,
        help=generic_features_help,
    )

    parser.add_argument(
        "--features-cytosol",
        nargs="+",
        type=str,
        help=generic_features_help,
    )

    parser.add_argument(
        "--objects",
        required=False,
        help="Path to directory containing output from `find-objects`",
    )
    parser.add_argument(
        "--stack-images",
        help="Path to additional images to stack with `images`. Add `s` prefix to refer"
        " to stack image channel index (e.g. corr_0_s0).",
        nargs="*",
    )
    parser.add_argument(
        "--label-filter",
        help="Path to Parquet containing labels to include.",
    )
    parser.add_argument(
        "--stack-image-pattern",
        help="Format string to extract metadata from the image file name.",
    )
    parser.add_argument(
        "--nuclei-min-area",
        type=float,
        default=2,
        help="Remove nuclei with area < `nuclei-area`",
    )
    parser.add_argument(
        "--nuclei-max-area",
        type=float,
        help="Remove nuclei with area > `nuclei-area`",
    )
    parser.add_argument(
        "--cell-min-area",
        type=float,
        default=2,
        help="Remove cell with area < `cell-area`",
    )
    parser.add_argument(
        "--cell-max-area",
        type=float,
        help="Remove cells with area > `cell-area`",
    )
    parser.add_argument(
        "--cytosol-min-area",
        type=float,
        default=2,
        help="Remove cytosolic labels with area < `cytosol-area`",
    )
    parser.add_argument(
        "--cytosol-max-area",
        type=float,
        help="Remove cytosolic labels with area > `cytosol-area`",
    )
    parser.add_argument(
        "--channel-rename",
        help="Inline JSON mapping channel index (0-based) to channel name for feature "
        'readability. Example \'{"0":"A", "2":"B"}\'',
    )
    parser.add_argument(
        "--features-plot",
        nargs="*",
        type=str,
        help="Optional feature names to plot",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=argparse.SUPPRESS,
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
        func=_run_pipeline_compute_features,
    )
