from __future__ import annotations

from scallops.cli.args import (
    SubParsers,
    add_command,
    dask_client_arg,
    dask_cluster_arg,
    expected_images_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    lazy_handler,
    no_version_arg,
    subset_arg,
    t_index_arg,
    z_index_tile_arg,
)


def _create_parser(subparsers: SubParsers, default_help: bool) -> None:
    """Create a parser for the illumination correction module.

    :param subparsers: Subparsers object for adding subcommands.
    :param default_help: Flag for default help formatter.
    :return: None
    """
    parser = add_command(
        subparsers,
        "illum-corr",
        default_help,
        description="Calculate illumination correction",
        help="Calculate illumination correction",
    )

    subparsers = parser.add_subparsers(help="sub-command help")
    _add_agg_parser(subparsers, default_help)


def _add_agg_parser(subparsers: SubParsers, default_help: bool) -> None:
    """Add a parser for calculating illumination correction by aggregating images by mean or median.

    :param subparsers: Subparsers object for adding subcommands.
    :return: None
    """
    parser = add_command(
        subparsers,
        "agg",
        default_help,
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
    t_index_arg(parser)
    force_arg(parser)

    expected_images_arg(parser)
    no_version_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    parser.set_defaults(
        func=lazy_handler(
            "scallops.cli.illumination_correction:run_illumination_correction_agg"
        )
    )
