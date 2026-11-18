from __future__ import annotations

from argparse import (
    ArgumentParser,
)

from scallops.cli.args import (
    SubParsers,
    add_command,
    bool_arg,
    dask_client_arg,
    dask_cluster_arg,
    expected_images_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    image_spacing_arg,
    images_arg,
    lazy_handler,
    no_version_arg,
    subset_arg,
    t_index_arg,
    z_index_tile_arg,
)


def _create_stitch_preview_parser(
    subparsers: SubParsers, default_help: bool
) -> ArgumentParser:
    """Create the parser for the stitch preview command.

    :param subparsers: Subparsers object to which this parser will be added.
    :param default_help: Boolean indicating whether to use default argparse help formatting.
    :return: Configured argparse parser for the stitch preview command.
    """
    parser = add_command(
        subparsers,
        "stitch-preview",
        default_help,
        help="Preview stitched images",
        description="Create a multi-tile image using image stage positions.",
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
        help="Downsampling factor for the preview image (e.g. 20 gives 1/20 of the "
        "full resolution).",
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
            "Optional CSV or Parquet file containing stage positions. Use when image "
            "metadata is missing stage positions. Expected columns `name`, `y`, "
            "and `x`, where name is the full image path and `y`, `x` are in microns."
        ),
    )
    z_index_tile_arg(parser)
    t_index_arg(parser)
    parser.set_defaults(func=lazy_handler("scallops.cli.stitch:run_stitch_preview"))
    return parser


def _create_stitch_parser(subparsers: SubParsers, default_help: bool) -> ArgumentParser:
    parser = add_command(
        subparsers,
        "stitch",
        default_help,
        help="Stitch images",
        description="Align image tiles starting from their stage positions, write the "
        "stitched positions and a QC report, and optionally write the fused image and "
        "tile label masks to OME-Zarr.",
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)
    required.add_argument(
        "--report-output",
        required=True,
        help="Output directory for the stitched tile positions, overlap evaluation "
        "table, and PDF QC report (and the channel alignment record when "
        "--alignment-correction-channels is given).",
    )

    parser.add_argument(
        "--image-output",
        help="Output Zarr directory for the stitched image and tile label masks. "
        "Required unless both --no-save-image and --no-save-labels are given.",
    )
    parser.add_argument(
        "--channel-name",
        nargs="*",
        help="Channel names for the output image, one per input channel (not per "
        "--output-channels entry); only the names of output channels are used. "
        "Defaults to the channel names in the image metadata.",
    )

    image_pattern_arg(parser)
    groupby_arg(parser)
    subset_arg(parser)
    parser.add_argument(
        "-c",
        "--align-channel",
        type=int,
        default=0,
        help="Channel index (0-based) used to align tiles. Also the reference channel "
        "for --alignment-correction-channels.",
    )
    parser.add_argument(
        "--radial-correction-k",
        type=str,
        default="auto",
        help="Radial distortion coefficient k: a number, `auto` to estimate k "
        "automatically, or `none` to disable radial correction.",
    )

    parser.add_argument(
        "--stitch-alpha",
        type=float,
        default=0.001,
        help="Significance level for accepting a pairwise tile alignment: its "
        "correlation is compared with a null distribution from non-overlapping tile "
        "pairs. Lower values are stricter. If no alignment is accepted, tiles are "
        "placed at their stage positions.",
    )

    parser.add_argument(
        "--max-shift",
        type=float,
        default=[50, 100, 150],
        nargs="+",
        help="Maximum allowed difference, in microns along y or x, between a pairwise "
        "tile alignment and the offset expected from the stage positions; alignments "
        "beyond it are rejected. If several values are given, they are tried in order "
        "until the stage layout (axis flips and swap) can be determined.",
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
        help="Do not compute the correlation (ZNCC) of each tile overlap after "
        "stitching, which is used to evaluate stitching quality.",
    )
    parser.add_argument(
        "--ffp",
        help="Path to the flat-field profile (FFP) image used for illumination "
        "correction.",
    )
    parser.add_argument(
        "--dfp",
        help="Path to the dark-field profile (DFP) image used for illumination "
        "correction.",
    )

    parser.add_argument(
        "--blend",
        choices=["linear", "none"],
        default="none",
        help="Blending of overlapping tiles: `none` uses the tile closest to the "
        "well center, `linear` blends overlapping tiles.",
    )

    parser.add_argument(
        "--output-channels",
        nargs="*",
        type=int,
        help="Channel indices (0-based) to save in the stitched image. All channels "
        "if not provided.",
    )

    parser.add_argument(
        "--crop-y",
        type=int,
        default=None,
        help="Pixels to crop from each edge of a tile along y. Applied when aligning "
        "tiles, and added to the crop applied when writing the stitched image. Set automatically when radial correction is enabled.",
    )
    parser.add_argument(
        "--crop-x",
        type=int,
        default=None,
        help="Pixels to crop from each edge of a tile along x. Applied when aligning "
        "tiles, and added to the crop applied when writing the stitched image. Set automatically when radial correction is enabled.",
    )
    parser.add_argument(
        "--stage-positions",
        help=(
            "Optional CSV or Parquet file containing stage positions. Use when image "
            "metadata is missing stage positions. Expected columns `name`, `y`, "
            "and `x`, where name is the full image path and `y`, `x` are in microns."
        ),
    )
    parser.add_argument(
        "--image-spacing",
        type=image_spacing_arg,
        help="Pixel size in microns as `Y,X` (e.g. `0.65,0.65`). Overrides the "
        "pixel size in the image metadata; required if the metadata lacks it.",
    )
    parser.add_argument(
        "--min-overlap-fraction",
        type=float,
        help="Minimum overlap, as a fraction of tile area, for two tiles to be aligned "
        "to each other. Defaults to one third of the overlap measured by aligning a "
        "few neighboring tiles while detecting the stage layout (0.033 if the layout "
        "can't be detected).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=239753,
        help="Random seed for sampling tile pairs (automatic radial correction k and "
        "the null distribution for --stitch-alpha) and tiles (channel alignment "
        "estimation).",
    )

    parser.add_argument(
        "--cross-correlation-upsample",
        type=int,
        default=1,
        help="Upsampling factor for subpixel tile alignment: shifts are estimated to "
        "1/N of a pixel. 1 aligns to the nearest whole pixel.",
    )
    parser.add_argument(
        "--rename",
        help="Headerless two-column CSV file mapping image keys (see --groupby; first "
        "column) to new keys (second column) for output file names.",
    )
    parser.add_argument(
        "--flip-y-axis",
        type=bool_arg,
        metavar="{1,0,true,false,yes,no}",
        help="Flip the tile y axis (1, true, or yes) or not (0, false, or no), "
        "overriding automatic detection. Determined automatically if not "
        "provided.",
    )
    parser.add_argument(
        "--flip-x-axis",
        type=bool_arg,
        metavar="{1,0,true,false,yes,no}",
        help="Flip the tile x axis (1, true, or yes) or not (0, false, or no), "
        "overriding automatic detection. Determined automatically if not "
        "provided.",
    )
    parser.add_argument(
        "--swap-axes",
        type=bool_arg,
        metavar="{1,0,true,false,yes,no}",
        help="Swap the tile y and x axes (1, true, or yes) or not (0, false, or no), "
        "overriding automatic detection. Determined automatically if not "
        "provided.",
    )
    channel_alignment_group = parser.add_argument_group(
        title="Channel alignment",
    )
    channel_alignment_group.add_argument(
        "--channel-reference",
        type=int,
        help="Reference channel index (0-based). If given, every other channel is "
        "aligned to it by a translation estimated separately in each tile by "
        "cross-correlation (tuned with --channel-cross-correlation-upsample, "
        "--channel-window, and --channel-filter-min/max). Channels are not "
        "aligned to each other if not provided. Use this when channel offsets "
        "vary from tile to tile; use --alignment-correction-channels instead to "
        "apply one magnification and translation per channel to all tiles in a "
        "well. The two cannot be combined.",
    )
    channel_alignment_group.add_argument(
        "--channel-cross-correlation-upsample",
        type=int,
        default=2,
        help="Upsampling factor for subpixel alignment across channels: shifts are "
        "estimated to 1/N of a pixel. 1 aligns to the nearest whole pixel.",
    )
    channel_alignment_group.add_argument(
        "--channel-window",
        type=int,
        default=2,
        help="When aligning across channels, use only the central 1/N of the tile "
        "along each axis (e.g. 2 uses the central half). 1 uses the whole tile.",
    )

    channel_alignment_group.add_argument(
        "--channel-filter-min",
        type=float,
        default=0,
        help="Lower percentile of the intensity range used when aligning across "
        "channels; values below it are replaced with uniform noise. Filtering is "
        "disabled when this is <= 0 and --channel-filter-max is >= 100.",
    )
    channel_alignment_group.add_argument(
        "--channel-filter-max",
        type=float,
        default=90,
        help="Upper percentile of the intensity range used when aligning across "
        "channels; values above it are replaced with uniform noise.",
    )
    channel_correction_group = parser.add_argument_group(
        title="Channel correction",
    )
    channel_correction_group.add_argument(
        "--alignment-correction-channels",
        nargs="+",
        help="Channel indices (0-based) to align to the reference channel "
        "(--align-channel) when fusing, or `all` (also `*` or `true`) for every "
        "channel in --output-channels except the reference. Each channel gets one "
        "magnification and translation, applied to all tiles of a stitched image "
        "(e.g. a well). Values come from --alignment-correction-params; channels "
        "it doesn't list are estimated from the image's tiles. Requires "
        "--alignment-correction-channel-order. When given, each output channel is "
        "resampled once with a quintic B-spline that combines this alignment with "
        "the radial correction, instead of the default bilinear radial "
        "correction. Cannot be combined with --channel-reference, which instead "
        "estimates a translation separately in each tile.",
    )
    channel_correction_group.add_argument(
        "--alignment-correction-params",
        help="JSON file with each channel's alignment to the reference channel "
        "(magnification m about the tile center and translation ty, tx in pixels), "
        "keyed by channel name (case-insensitive). Without it, or for channels it "
        "doesn't list, the alignment is estimated from each well's tiles.",
    )
    channel_correction_group.add_argument(
        "--alignment-correction-channel-order",
        nargs="*",
        help="The --alignment-correction-params name of each input channel, in input order (e.g. DAPI FITC "
        "TRITC CY5 CY7); `-` for a channel with no entry. Required with --alignment-correction-channels.",
    )
    z_index_tile_arg(parser)
    t_index_arg(parser)
    force_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    expected_images_arg(parser)
    no_version_arg(parser)
    parser.set_defaults(func=lazy_handler("scallops.cli.stitch:run_stitch"))
    return parser


def _create_parser(subparsers: SubParsers, default_help: bool) -> None:
    """Add the `stitch` and `stitch-preview` commands."""
    _create_stitch_parser(subparsers, default_help)
    _create_stitch_preview_parser(subparsers, default_help)
