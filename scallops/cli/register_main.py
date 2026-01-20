import argparse
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


def image_spacing_arg(s: str) -> tuple[float, float]:
    """Parse image spacing argument.

    This function takes a string containing physical size values (y, x)
    and returns a tuple of floats representing the physical size in the y
    and x dimensions.

    :param s: String with physical size values in the format "y,x".
    :return: Tuple containing two float values representing the physical size in y and x.

    .. example::

        >>> image_spacing_arg("1.34,1.34")
        (1.34, 1.34)
    """
    image_spacing = s.split(",")
    assert len(image_spacing) == 2, "Image spacing must contain 'y,x' values"
    return float(image_spacing[0]), float(image_spacing[1])


def _run_itk_registration(arguments: Namespace) -> None:
    """Run ITK registration."""
    from scallops.cli.register import run_itk_registration

    run_itk_registration(arguments)


def _run_itk_transform(arguments: Namespace) -> None:
    """Run ITK transformation."""
    from scallops.cli.register import run_itk_transform

    run_itk_transform(arguments)


def _run_cross_correlation_registration(arguments: Namespace) -> None:
    """Run cross-correlation registration."""
    from scallops.cli.register import run_cross_correlation_registration

    run_cross_correlation_registration(arguments)


def _create_parser(subparsers: ArgumentParser, default_help: bool) -> None:
    """Create the main parser for image registration."""
    parser = subparsers.add_parser(
        "registration",
        help="Image registration",
        description="Image registration",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    _create_elastix_parser(subparsers, default_help)
    _create_transform_parser(subparsers, default_help)
    _create_cross_correlation_parser(subparsers, default_help)


def _create_cross_correlation_parser(
    subparsers: ArgumentParser, default_help: bool
) -> None:
    """Create the parser for cross-correlation registration."""
    parser = subparsers.add_parser(
        "cross-correlation",
        description="Register image across and within cycles",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    images_arg(required)
    required.add_argument("-o", "--output", help="Zarr output directory", required=True)
    image_pattern_arg(parser)

    parser.add_argument(
        "--across-t-channel",
        dest="across_t_channel",
        type=int,
        help="Channel index (0-based) to use to register across cycles",
    )
    parser.add_argument(
        "--within-t-channel",
        nargs="*",
        type=int,
        dest="within_t_channel",
        help="Channel indices (0-based) to use to register within cycles",
    )
    parser.add_argument(
        "--within-t-filter-min",
        dest="registration_filter_min",
        type=float,
        default=0,
        help="Replace data outside of specified percentile range [p1, p2] with uniform noise when aligning within t",
    )
    parser.add_argument(
        "--within-t-filter-max",
        dest="registration_filter_max",
        type=float,
        default=90,
        help="Replace data outside of specified percentile range [p1, p2] with uniform noise when aligning within t",
    )

    groupby_arg(parser)
    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_cross_correlation_registration)


def _create_elastix_parser(subparsers: ArgumentParser, default_help: bool) -> None:
    """Create the parser for Elastix-based registration."""
    parser = subparsers.add_parser(
        "elastix",
        description="Register moving image to fixed image using ITK. "
        "If no fixed image is provided, registers moving image to a specified timepoint. "
        "Outputs are stored in Zarr format.",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--moving",
        nargs="+",
        required=True,
        help="Paths to directories containing nd2, tiff, zarr, or other Bio-Formats images",
    )
    parser.add_argument(
        "--fixed",
        nargs="+",
        help="Paths to directories containing nd2, tiff, zarr, or other Bio-Formats images",
    )
    parser.add_argument(
        "--moving-label",
        help="Path to Zarr directories containing labels to transform",
        nargs="*",
    )
    parser.add_argument(
        "--itk-parameters",
        nargs="+",
        default=["affine", "nl-100"],
        help="Paths to files containing ITK parameters or predefined parameter maps",
    )
    parser.add_argument(
        "--moving-image-pattern",
        dest="moving_image_pattern",
        help="Format string to extract metadata from the moving image file name",
    )
    parser.add_argument(
        "--fixed-image-pattern",
        dest="fixed_image_pattern",
        help="Format string to extract metadata from the fixed image file name",
    )
    parser.add_argument(
        "--moving-image-spacing",
        type=image_spacing_arg,
        help="Physical size y, x if image metadata does not contain this information",
    )
    parser.add_argument(
        "--fixed-image-spacing",
        type=image_spacing_arg,
        help="Physical size y, x if image metadata does not contain this information",
    )
    parser.add_argument(
        "--moving-channel",
        type=int,
        default=0,
        help="Moving channel index (0-based) to use for alignment",
    )
    parser.add_argument(
        "--fixed-channel",
        type=int,
        default=0,
        help="Fixed channel index (0-based) to use for alignment",
    )
    parser.add_argument(
        "--transform-output",
        dest="transform_output_dir",
        default="transforms",
        help="Path to output directory for transformations",
    )
    parser.add_argument(
        "--label-output",
        dest="label_output_dir",
        help="Path to save transformed moving labels",
    )
    parser.add_argument(
        "--moving-output",
        dest="moving_output_dir",
        help="Path to save transformed moving image",
    )

    parser.add_argument(
        "--time",
        "-t",
        default="0",
        help="Time index (0-based) or value for alignment across timepoints",
    )
    parser.add_argument(
        "--unroll-channels",
        action="store_true",
        help="Unroll channels (drop 't' dimension) in output image",
    )
    # parser.add_argument(
    #     "--register-within-t",
    #     action="store_true",
    #     help="Register within timepoints (across channels)",
    # )
    parser.add_argument(
        "--sort",
        nargs="+",
        help="Custom sort order. Example: 20231012_20x_6W_IF 20231010_20x_6W_FISH",
    )

    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Do not use landmarks to find corresponding regions between moving and "
        "fixed images to initialize the registration",
    )
    parser.add_argument(
        "--landmark-min-score",
        type=float,
        default=0.6,
        help="Minimum score to include matching region for landmark estimation",
    )
    parser.add_argument(
        "--landmark-step-size",
        type=float,
        default=1000,
        help="Grid step size for landmark estimation in physical units",
    )

    parser.add_argument(
        "--landmark-image-chunk-size",
        type=float,
        default=200,
        help="Image chunk size in physical units",
    )
    parser.add_argument(
        "--landmark-template-padding",
        type=float,
        default=[750, 1000, 1250, 2250],
        nargs="+",
        help="Template padding in physical units. Values are tried "
        " until `landmark-min-count` landmarks are found.",
    )
    parser.add_argument(
        "--landmark-initialization",
        choices=["com", "none"],
        default=["com", "none"],
        nargs="+",
        help="Initial alignment method for landmark estimation: com (center of mass) "
        "or none",
    )
    parser.add_argument(
        "--landmark-com-min-quantile",
        type=float,
        default=0.25,
        help="Include values >= specified quantile for center of mass computation.",
    )
    parser.add_argument(
        "--landmark-com-max-quantile",
        type=float,
        default=0.75,
        help="Include values <= specified quantile for center of mass computation.",
    )
    parser.add_argument(
        "--landmark-min-count",
        type=int,
        default=100,
        help="Ensure `landmark-min-count` landmarks are found.",
    )
    parser.add_argument(
        "--initial-transform",
        help=argparse.SUPPRESS,
        # help="Path to ITK parameters initialize the registration",
    )
    parser.add_argument(
        "--output-aligned-channels-only",
        action="store_true",
        help="Whether to output aligned channels only",
    )
    parser.add_argument(
        "--itk-channels",
        nargs="*",
        help="Paths to files containing ITK parameters or predefined parameter maps "
        "for registering across channels",
    )

    z_index_arg(parser)
    groupby_arg(parser)
    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    verbose_arg(parser)
    no_version_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_itk_registration, output="register")


def _create_transform_parser(subparsers: ArgumentParser, default_help: bool) -> None:
    """Create the parser for ITK-based transformations."""
    parser = subparsers.add_parser(
        "transformix",
        description="Transform moving image to fixed image using previously computed ITK transformations",
        formatter_class=(
            ArgumentDefaultsHelpFormatter if default_help else HelpFormatter
        ),
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--transform",
        dest="transform_dir",
        required=True,
        help="Path to directory containing transformations",
    )
    required.add_argument(
        "--output", required=True, help="Path to output Zarr directory"
    )

    required.add_argument(
        "--images", required=True, help="Path to Zarr directory to transform"
    )

    parser.add_argument(
        "--image-spacing",
        type=image_spacing_arg,
        help="Physical size y, x if metadata does not contain this information",
    )
    parser.add_argument(
        "--type",
        choices=["images", "labels"],
        default="labels",
        help="Whether to transform images or labels",
    )

    force_arg(parser)
    dask_client_arg(parser, value="none")
    dask_cluster_arg(parser)
    _sort_groups(parser)
    parser.set_defaults(func=_run_itk_transform)
