import argparse
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    HelpFormatter,
    Namespace,
)

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.register_main import image_spacing_arg
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
    z_index_tile_arg,
)


def _run_stitch(arguments: Namespace) -> None:
    """Execute the stitching pipeline."""
    from scallops.cli.stitch import run_stitch

    run_stitch(arguments)


def _run_stitch_preview(arguments: Namespace) -> None:
    """Execute the stitching preview pipeline."""
    from scallops.cli.stitch import run_stitch_preview

    run_stitch_preview(arguments)


def _create_stitch_preview_parser(
    subparsers: ArgumentParser, default_help: bool
) -> ArgumentParser:
    """Create the parser for the stitch preview command.

    :param subparsers: Subparsers object to which this parser will be added.
    :param default_help: Boolean indicating whether to use default argparse help formatting.
    :return: Configured argparse parser for the stitch preview command.
    """
    parser = subparsers.add_parser(
        "stitch-preview",
        help="Preview stitched images",
        description="Create a multi-tile image using image stage positions.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)
    required.add_argument("-o", "--output", required=True, help="Output directory.")
    image_pattern_arg(parser)
    groupby_arg(parser)
    subset_arg(parser)
    parser.add_argument(
        "-n", "--numbers", action="store_true", help="Display tile numbers."
    )
    parser.add_argument(
        "-b", "--bounds", action="store_true", help="Display tile bounds."
    )
    parser.add_argument(
        "--no-tiles", action="store_true", help="Do not display image tiles."
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        default=0,
        help="Channel index (0-based) to display.",
    )
    parser.add_argument(
        "-d",
        "--downsample",
        type=float,
        default=20,
        help="Downsample image resolution.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Log-transform pixel intensities to help visualize dim images.",
    )

    parser.add_argument(
        "--stage-positions",
        help=(
            "Optional CSV file containing stage positions. Use when image metadata "
            "is missing stage positions. Expected columns `name`, `y`, "
            "and `x`, where name is the full image path."
        ),
    )
    z_index_tile_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_stitch_preview)
    return parser


def _create_stitch_parser(
    subparsers: ArgumentParser, default_help: bool
) -> ArgumentParser:
    parser = subparsers.add_parser(
        "stitch",
        help="Stitch images",
        description="Stitch microscopy images",
        formatter_class=ArgumentDefaultsHelpFormatter
        if default_help
        else HelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)
    required.add_argument(
        "--report-output",
        required=True,
        help="Output directory for stitched positions and QC report.",
    )

    parser.add_argument(
        "--image-output",
        help="Output zarr directory for stitched images and masks.",
    )

    image_pattern_arg(parser)
    groupby_arg(parser)
    subset_arg(parser)
    parser.add_argument(
        "-c",
        "--align-channel",
        type=int,
        default=0,
        help="Channel index (0-based) to use for alignment.",
    )
    parser.add_argument(
        "--radial-correction-k",
        type=str,
        default="auto",
        help="K to correct for radial distortion. Use `auto` to automatically determine "
        "k and `none` to disable auto determination.",
    )

    parser.add_argument(
        "--stitch-alpha",
        type=float,
        default=0.001,
        help="Significance level for alignment error quantification.",
    )

    parser.add_argument(
        "--max-shift",
        type=float,
        default=[50, 100, 150],
        nargs="+",
        help="Maximum allowed per-tile shift in microns",
    )

    parser.add_argument(
        "--no-save-image",
        action="store_true",
        help="Do not save stitched image.",
    )
    parser.add_argument(
        "--no-save-labels",
        action="store_true",
        help="Do not save tile boundary label mask or tile source labels.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Do not evaluate stitching quality.",
    )
    parser.add_argument("--ffp", help="Path for flat-field correction profile image.")
    parser.add_argument("--dfp", help="Path for dark-field correction profile image.")

    parser.add_argument(
        "--blend",
        choices=["linear", "none"],
        default="none",
        help="Blending method for stitched images",
    )

    parser.add_argument(
        "--output-channels",
        nargs="*",
        type=int,
        help="Output channels to save in stitched image.",
    )

    parser.add_argument(
        "--crop",
        type=int,
        default=None,
        help="Crop tiles by `crop` pixels along each dimension when aligning tiles. Set"
        "automatically when radial correction is enabled.",
    )
    parser.add_argument(
        "--stage-positions",
        help=(
            "Optional CSV file containing stage positions. Use when image metadata "
            "is missing stage positions. Expected columns `name`, `y`, "
            "and `x`, where name is the full image path."
        ),
    )
    parser.add_argument(
        "--image-spacing",
        type=image_spacing_arg,
        help="Physical size y, x if image metadata does not contain this information",
    )
    parser.add_argument(
        "--min-overlap-fraction",
        type=float,
        help="Minimum tile overlap fraction to include edge in graph. Determined "
        "automatically if not provided.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=239753,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cross-correlation-upsample",
        type=int,
        default=1,
        help="Upsampling factor for registration precision.",
    )
    parser.add_argument(
        "--rename",
        help="CSV file mapping old image IDs to new IDs for output file names.",
    )
    parser.add_argument(
        "--flip-y-axis",
        type=int,
        choices=[1, 0],
        help="Whether to flip tile y axis. Determined automatically if not provided.",
    )
    parser.add_argument(
        "--flip-x-axis",
        type=int,
        choices=[1, 0],
        help="Whether to flip tile x axis. Determined automatically if not provided.",
    )
    parser.add_argument(
        "--swap-axes",
        type=int,
        choices=[1, 0],
        help="Whether to swap tile y and x axes. Determined automatically if not "
        "provided.",
    )

    z_index_tile_arg(parser)
    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    expected_images_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_stitch)
    return parser
