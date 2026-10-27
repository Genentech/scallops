from __future__ import annotations

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
    SubParsers,
    dask_client_arg,
    dask_cluster_arg,
    expected_images_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    no_version_arg,
    subset_arg,
    t_index_arg,
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
    subparsers: SubParsers, default_help: bool
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
    t_index_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_stitch_preview)
    return parser


def _create_stitch_parser(subparsers: SubParsers, default_help: bool) -> ArgumentParser:
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
    parser.add_argument(
        "--channel-name",
        nargs="*",
        help="Channel names to save in output image. If specified, must equal the number of channels in input tiles.",
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
        "--crop-y",
        type=int,
        default=None,
        help="Crop tiles by `crop` pixels along y dimension when aligning tiles. Set"
        "automatically when radial correction is enabled.",
    )
    parser.add_argument(
        "--crop-x",
        type=int,
        default=None,
        help="Crop tiles by `crop` pixels along x dimension when aligning tiles. Set"
        "automatically when radial correction is enabled.",
    )
    parser.add_argument(
        "--stage-positions",
        help=(
            "Optional CSV or Parquet file containing stage positions. Use when image metadata "
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
    parser.add_argument(
        "--channel-reference",
        type=int,
        help="If provided, perform alignment across channels within a tile using the specified reference channel (0-based).",
    )
    parser.add_argument(
        "--channel-cross-correlation-upsample",
        type=int,
        default=2,
        help="Upsampling factor for registration precision when aligning across channels.",
    )
    parser.add_argument(
        "--channel-window",
        type=int,
        default=2,
        help="Use a centered subset of a tile if greater than one when aligning across channels.",
    )

    parser.add_argument(
        "--channel-filter-min",
        type=float,
        default=0,
        help="Replace data outside of specified percentile range [p1, p2] with uniform noise when aligning across channels.",
    )
    parser.add_argument(
        "--channel-filter-max",
        type=float,
        default=90,
        help="Replace data outside of specified percentile range [p1, p2] with uniform noise when aligning across channels.",
    )
    parser.add_argument(
        "--alignment-correction-channels",
        nargs="+",
        help="Channel indices (0-based) to align to the reference channel (--align-channel) when fusing, or `all` "
        "(also `*` or `true`) for every channel except the reference. When given, every output channel is "
        "resampled once with a quintic B-spline (alignment composed with the radial correction) instead of the "
        "bilinear radial correction. Requires --alignment-correction-channel-order. Channels without values in "
        "--alignment-correction-params are estimated from the images of each well.",
    )
    parser.add_argument(
        "--alignment-correction-params",
        help="JSON file with each channel's alignment to the reference channel (magnification m about the tile "
        "centre and translation ty, tx in pixels), keyed by channel name (case-insensitive). Without it, or for "
        "channels it doesn't list, the alignment is estimated from each well's tiles.",
    )
    parser.add_argument(
        "--alignment-correction-channel-order",
        nargs="*",
        help="The --alignment-correction-params name of each input channel, in input order (e.g. DAPI FITC "
        "TRITC CY5 CY7); `-` for a channel with no entry. Required with --alignment-correction-channels.",
    )
    z_index_tile_arg(parser)
    t_index_arg(parser)
    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    expected_images_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_stitch)
    return parser
