import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    expected_images_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    no_version_arg,
    subset_arg,
    verbose_arg,
    z_index_tile_arg,
)


def _run_illumination_correction_agg(arguments: argparse.Namespace):
    """Execute the mean/median-based illumination correction based on provided arguments.

    :param arguments: Parsed command-line arguments.
    """
    from scallops.cli.illumination_correction import run_illumination_correction_agg

    run_illumination_correction_agg(arguments)


def _create_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    """Create a parser for the illumination correction module.

    :param subparsers: Subparsers object for adding subcommands.
    :param default_help: Flag for default help formatter.
    :return: None
    """
    parser = subparsers.add_parser(
        "illum-corr",
        description="Calculate illumination correction",
        help="Calculate illumination correction",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )

    subparsers = parser.add_subparsers(help="sub-command help")
    _add_agg_parser(subparsers, default_help)


def _add_agg_parser(subparsers: argparse.ArgumentParser, default_help: bool) -> None:
    """Add a parser for calculating illumination correction by aggregating images by mean or median.

    :param subparsers: Subparsers object for adding subcommands.
    :return: None
    """
    parser = subparsers.add_parser(
        "agg",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
        description="Calculate illumination correction by aggregating images by mean, median or min, "
        "followed by median filter and rescaling. Outputs flat-field TIFF or Zarr image.",
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)

    required.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Path to output Zarr image or TIFF directory",
        required=True,
    )

    groupby_arg(parser)
    subset_arg(parser)
    image_pattern_arg(parser)
    parser.add_argument(
        "--smooth",
        type=int,
        help="The radius of the disk-shaped footprint for median filter. Default is "
        "sqrt((image_width * image_height) / (PI * 20)",
    )
    parser.add_argument(
        "--agg-method",
        choices=["mean", "median", "min"],
        default="mean",
        help="Method to aggregate images",
    )
    parser.add_argument(
        "--no-rescale",
        action="store_true",
        help="Do not use 2nd percentile for robust minimum",
    )
    parser.add_argument(
        "--output-image-format",
        dest="output_image_format",
        help="Output image format",
        default="tiff",
        choices=["tiff", "zarr"],
    )
    z_index_tile_arg(parser)
    parser.add_argument(
        "--channel",
        default=0,
        type=int,
        help="Channel index (0-based) to select best focus z index",
    )
    force_arg(parser)
    verbose_arg(parser)
    expected_images_arg(parser)
    no_version_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_illumination_correction_agg)
