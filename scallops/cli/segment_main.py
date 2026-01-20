from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    HelpFormatter,
    Namespace,
)

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    no_version_arg,
    subset_arg,
    verbose_arg,
    z_index_arg,
)


def _run_pipeline_segment_nuclei(arguments: Namespace) -> None:
    """Run the pipeline for nuclei segmentation.

    :param arguments: Parsed command-line arguments.
    """
    from scallops.cli.segment import run_pipeline_segment_nuclei

    run_pipeline_segment_nuclei(arguments)


def _run_pipeline_segment_cell(arguments: Namespace) -> None:
    """Run the pipeline for cell segmentation.

    :param arguments: Parsed command-line arguments.
    """
    from scallops.cli.segment import run_pipeline_segment_cell

    run_pipeline_segment_cell(arguments)


def _create_parser(subparsers: ArgumentParser, default_help: bool) -> None:
    """Create the main segmentation parser.

    :param subparsers: Subparsers object to which segmentation parsers will be added.
    :param default_help: Whether to use argparse's default help formatter.
    """
    parser = subparsers.add_parser(
        "segment",
        help="Nuclei and cell segmentation",
        description="Segmentation",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    _add_nuclei_parser(subparsers, default_help)
    _add_cell_parser(subparsers, default_help)


def _add_common_args(parser: ArgumentParser) -> None:
    """Add common segmentation arguments.

    :param parser: Argument parser to which the common arguments will be added.
    """

    parser.add_argument(
        "--dapi-channel",
        type=int,
        default=0,
        help="Channel index (0-based) where DAPI is found",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        dest="min_area",
        help="Filter labels with area < `min-area`",
    )

    parser.add_argument(
        "--max-area",
        type=float,
        dest="max_area",
        help="Filter labels with area > `-max-area`",
    )

    parser.add_argument(
        "--chunks",
        help="Chunk size to use to perform segmentation in chunks",
        type=int,
    )
    parser.add_argument(
        "--chunk-overlap",
        help="Chunk size overlap to use to perform segmentation using overlapping chunks",
        type=int,
    )
    z_index_arg(parser)
    no_version_arg(parser)


def _add_nuclei_parser(subparsers: ArgumentParser, default_help: bool = True) -> None:
    """Add the parser for nuclei segmentation.

    :param subparsers: Subparsers object to which the nuclei parser will be added.
    """
    parser = subparsers.add_parser(
        "nuclei",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
        description="Nuclei segmentation. Outputs a Zarr image containing nuclei labels.",
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)
    required.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Path to output zarr image directory",
        required=True,
    )
    parser.add_argument(
        "--method",
        help="Nuclei segmentation algorithm",
        default="stardist",
        choices=["cellpose", "stardist"],
    )
    image_pattern_arg(parser)
    groupby_arg(parser)
    _add_common_args(parser)
    parser.add_argument(
        "--stardist-clip",
        help="Whether to clip normalized image values to between 0 and 1",
        action="store_true",
    )
    parser.add_argument(
        "--stardist-pmin",
        help="Minimum percentile for image normalization. Default is 3.",
        type=float,
    )
    parser.add_argument(
        "--stardist-pmax",
        help="Maximum percentile for image normalization. Default is 99.8.",
        type=float,
    )

    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    verbose_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_pipeline_segment_nuclei)


def _add_cell_parser(subparsers: ArgumentParser, default_help: bool = True) -> None:
    """Add the parser for cell segmentation.

    :param subparsers: Subparsers object to which the cell parser will be added.
    """
    parser = subparsers.add_parser(
        "cell",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
        description="Cell segmentation. Outputs a Zarr image containing cell labels.",
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)
    required.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Path to output zarr image directory",
        required=True,
    )

    parser.add_argument(
        "--nuclei-label",
        help="Path to zarr directory containing nuclei labels for watershed or propagation segmentation",
    )
    parser.add_argument(
        "--method",
        help="Cell segmentation algorithm. Note that only `watershed` and `propagation`"
        " will output cells that match nuclei",
        default="propagation",
        choices=["cellpose", "propagation", "watershed", "watershed-intensity"],
    )
    parser.add_argument(
        "--threshold",
        help="Threshold for watershed or propagation methods. Either `Li`, `Otsu`, "
        "`Local`, or manually determined "
        "value",
        type=str,
        default="Li",
    )
    parser.add_argument(
        "--threshold-correction-factor",
        help="Factor to adjust the computed threshold by if `threshold` is not a "
        "manually determined value",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--cyto-channel",
        type=int,
        nargs="*",
        dest="cyto_channel",
        help="Channel index (0-based) to infer cell segmentation from. Default is all non-DAPI channels. If more than"
        " one channel specified, use minimum across time (cycles) then mean over channels, or if only one time "
        "point is present, use mean over channels.",
    )
    image_pattern_arg(parser)
    groupby_arg(parser)
    _add_common_args(parser)

    parser.add_argument(
        "--nuclei-min-area",
        type=float,
        help="Filter nuclei labels with area < `min-area`",
    )

    parser.add_argument(
        "--nuclei-max-area",
        type=float,
        help="Filter nuclei labels with area > `-max-area`",
    )

    parser.add_argument(
        "--rolling-ball",
        dest="cell_segmentation_rolling_ball",
        help="Apply rolling ball subtraction to cell mask prior to computing threshold",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--sigma",
        dest="cell_segmentation_sigma",
        help="Size of gaussian kernel used to smooth the cell mask prior to computing threshold",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--closing-radius",
        dest="closing_radius",
        help="Disk radius to use for binary closing cell labels post segmentation",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--time",
        dest="cell_segmentation_t",
        help="Time indices (0-based) to include when computing cell segmentation mask. Defaults to all time points.",
        type=int,
        action="append",
    )
    parser.add_argument(
        "--shrink-nuclei",
        help="Shrink nuclei prior to subtraction of nuclei from cells to identify the "
        "cytosol.",
        action="store_true",
    )

    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    verbose_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_pipeline_segment_cell)
