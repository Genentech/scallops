import argparse

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    barcodes_arg,
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
    z_index_arg,
)


def _merge_main(arguments: argparse.Namespace):
    """Entry point for merging SBS barcodes and phenotype data.

    :param arguments: Parsed command-line arguments containing the merge parameters.
    """
    from scallops.cli.pooled_if_sbs import merge_main

    merge_main(arguments)


def _reads_main(arguments: argparse.Namespace):
    """Entry point for running the read calling pipeline.

    :param arguments: Parsed command-line arguments containing read calling parameters.
    """
    from scallops.cli.pooled_if_sbs import reads_main

    reads_main(arguments)


def _spot_detect_main(arguments: argparse.Namespace):
    """Entry point for running the spot detection pipeline.

    :param arguments: Parsed command-line arguments containing spot detection parameters.
    """
    from scallops.cli.pooled_if_sbs import spot_detect_main

    spot_detect_main(arguments)


def _napari_pooled_if_sbs_main(arguments: argparse.Namespace):
    """Napari wrapper over the ISS pipeline.

    :param arguments: Parsed command-line arguments containing visualization parameters.
    """
    url = arguments.url
    from scallops.visualize.napari import pooled_iss

    _ = pooled_iss(url)
    import napari

    napari.run()


def _create_napari_parser(subparsers, default_help=False):
    """Creates the parser for the Napari command.

    :param subparsers: Subparser object to register the command.
    :param default_help: Boolean flag to enable default help formatting.
    """
    parser = subparsers.add_parser(
        "napari",
        help="Run Napari with results from `scallops pooled-sbs`",
        description="Run Napari with results from `scallops pooled-sbs`",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    _sort_groups(parser)
    parser.set_defaults(func=_napari_pooled_if_sbs_main)
    parser.add_argument("url", help="URL to pooled-sbs images.zarr directory")


def _create_merge_parser(subparsers, default_help):
    """Creates the parser for the merge command.

    :param subparsers: Subparser object to register the command.
    :param default_help: Boolean flag to enable default help formatting.
    """
    parser = subparsers.add_parser(
        "merge",
        description="Join in-situ barcodes with phenotype data and output as Parquet.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--sbs", required=True, help="Directory containing SBS parquet files."
    )
    required.add_argument(
        "--phenotype",
        required=True,
        nargs="+",
        help="Directories with phenotype parquet files.",
    )
    parser.add_argument(
        "--join-sbs", choices=["inner", "outer"], default="outer", help="SBS join type."
    )
    parser.add_argument(
        "--join-phenotype",
        choices=["inner", "outer"],
        default="outer",
        help="Phenotype join type.",
    )
    parser.add_argument(
        "--phenotype-suffix", nargs="*", help="Suffix for phenotype columns."
    )
    parser.add_argument(
        "--format",
        help="Output file format.",
        default="parquet",
        choices=["parquet", "zarr"],
    )

    barcodes_arg(parser)
    output_dir_arg(parser)
    subset_arg(parser)
    parser.add_argument(
        "--barcode-col",
        help="`Barcode` column in barcodes CSV",
        default="barcode",
    )
    force_arg(parser)
    dask_client_arg(parser, "none")
    dask_cluster_arg(parser)
    verbose_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_merge_main)


def _create_reads_parser(subparsers, default_help):
    """Creates the parser for the reads command.

    :param subparsers: Subparser object to register the command.
    :param default_help: Boolean flag to enable default help formatting.
    """
    parser = subparsers.add_parser(
        "reads",
        help="In-situ read calling",
        description="Run pooled in-situ sequencing read calling. "
        "Outputs reads, barcodes to labels assignments, crosstalk matrix, and table with corrected and uncorrected base intensities. ",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--spots",
        required=True,
        dest="spots",
        help="Zarr output from `scallops pooled-sbs spot-detect`",
    )
    required.add_argument(
        "--labels",
        required=True,
        dest="labels",
        help="Zarr output from `scallops segment` containing labels",
    )
    required.add_argument(
        "--label-name",
        help="Name of labels to use. For example `nuclei` or `cell`",
        required=True,
    )
    barcodes_arg(required)
    output_dir_arg(required)

    parser.add_argument(
        "--read-quality-filter",
        type=float,
        help="Filter reads before assigning reads to labels",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        help="Filter labels with area < `min-area`",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        help="Filter labels with area > `max-area`",
    )

    parser.add_argument(
        "--mismatches",
        dest="n_mismatches",
        type=int,
        help="Correct reads <= `mismatches` from closest match in `barcodes`",
    )
    parser.add_argument(
        "--expand-labels-distance",
        type=int,
        help="Expand labels by `expand-labels-distance` when matching reads to labels.",
    )

    parser.add_argument(
        "--threshold-peaks",
        default="auto",
        dest="threshold_peaks",
        type=str,
        help="Filter reads before assigning reads to labels. Use `auto` to automatically"
        " determine threshold.",
    )

    parser.add_argument(
        "--threshold-peaks-crosstalk",
        type=str,
        default="auto",
        help="Threshold for `peaks` for identifying sequencing reads used in "
        "crosstalk correction. Use `auto` to automatically determine threshold.",
    )

    parser.add_argument(
        "--crosstalk-correction-method",
        default="median",
        help="Method to correct channel crosstalk",
        choices=["li_and_speed", "median", "none"],
    )
    parser.add_argument(
        "--crosstalk-correction-by-t",
        help="Correct crosstalk separately for each cycle",
        action="store_true",
    )
    parser.add_argument(
        "--crosstalk-nreads",
        help="Number of reads to sample to compute crosstalk correction. Use -1"
        "to include all reads.",
        default=500000,
        type=int,
    )

    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="Call reads both in and outside labels.",
    )

    subset_arg(parser)

    parser.add_argument(
        "--bases",
        dest="bases",
        default="GTAC",
        help="ISS bases",
    )
    parser.add_argument(
        "--barcode-col",
        help="`Barcode` column in barcodes CSV",
        default="barcode",
    )
    parser.add_argument(
        "--save-bases",
        help="Save individual base intensities",
        action="store_true",
    )

    force_arg(parser)
    verbose_arg(parser)
    no_version_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_reads_main, save=[])


def _create_spot_detect_parser(subparsers, default_help):
    """Creates the parser for the spot-detect command.

    :param subparsers: Subparser object to register the command.
    :param default_help: Boolean flag to enable default help formatting.
    """
    parser = subparsers.add_parser(
        "spot-detect",
        help="In-situ spot detection",
        description="Run pooled in-situ sequencing spot detection. "
        "Outputs table of all candidate peaks and `max`, the input images with LoG and "
        "maximum filters applied. Optionally also outputs the `std` image, which "
        "contains the standard deviation over cycles, followed by the mean across "
        "channels to identify spot locations and the `LoG` image, which contains the "
        "LoG filtered image.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)

    required.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Path to output Zarr containing `peaks`, `max`, and optionally `std` "
        "and `log`",
        required=True,
    )
    required.add_argument(
        "-c",
        "--channel",
        nargs="+",
        dest="channels",
        type=int,
        required=True,
        help="Channel indices (0-based) to use for spot detection",
    )

    image_pattern_arg(parser)

    parser.add_argument(
        "--max-filter-width",
        dest="max_filter_width",
        help="Neighborhood size for max filtering on Laplacian-of-Gaussian filtered "
        "SBS data, dilating sequencing channels to compensate for single-pixel "
        "alignment error",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--sigma-log",
        dest="sigma_log",
        help="Size of gaussian kernel used in Laplacian-of-Gaussian filter",
        nargs="+",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--peak-neighborhood-size",
        dest="peak_neighborhood_size",
        help="Neighborhood size for peak detection",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--cycles",
        nargs="+",
        type=int,
        help="Optional subset of cycle indices (0-based) to include.",
    )
    parser.add_argument(
        "--chunks",
        help="Chunk size to use to perform parallel spot detection. If not specified, "
        "image chunk size is used",
        type=int,
    )
    z_index_arg(parser)
    groupby_arg(parser)
    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)

    parser.add_argument(
        "--save",
        help="Additional outputs to save",
        nargs="*",
        choices=["log", "std"],
    )
    parser.add_argument(
        "--expected-cycles",
        help="Validate that the specified number of cycles are provided.",
        type=int,
    )
    verbose_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.add_argument(
        "--spot-detection-method",
        help=argparse.SUPPRESS,
        default="log",
        choices=["log", "spotiflow", "u-fish", "piscis"],
    )
    parser.add_argument(
        "--spot-detection-n-cycles",
        help=argparse.SUPPRESS,
        type=int,
    )
    parser.set_defaults(
        func=_spot_detect_main,
    )


def _create_parser(subparsers, default_help):
    """Creates the top-level parser for the `pooled-sbs` command.

    :param subparsers: Subparser object to register the command.
    :param default_help: Boolean flag to enable default help formatting.
    """
    parser = subparsers.add_parser(
        "pooled-sbs",
        help="SBS image processing",
        description="SBS image processing pipeline.",
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
            if default_help
            else argparse.HelpFormatter
        ),
    )
    subparsers = parser.add_subparsers(help="Sub-command help.")
    _create_spot_detect_parser(subparsers, default_help)
    _create_reads_parser(subparsers, default_help)
    _create_merge_parser(subparsers, default_help)


#    _create_napari_parser(subparsers, default_help)
