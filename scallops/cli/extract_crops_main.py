from __future__ import annotations

from scallops.cli.args import (
    SubParsers,
    _label_args_type,
    add_command,
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    lazy_handler,
    no_version_arg,
    output_dir_arg,
    subset_arg,
)


def _create_parser(subparsers: SubParsers, default_help: bool) -> None:
    parser = add_command(
        subparsers,
        "extract-crops",
        default_help,
        help="Extract crops from labeled images.",
    )

    required = parser.add_argument_group("required arguments")
    images_arg(required)
    output_dir_arg(required)

    image_pattern_arg(parser)

    required.add_argument(
        "--merge",
        required=True,
        help="Path to directory containing output from `merge`",
    )

    parser.add_argument(
        "--labels",
        help="One or more paths to zarr stores containing a 'labels' group. If the same "
        "label name appears in more than one store, the store listed last is used. Required when `mask` is true",
        nargs="+",
        type=_label_args_type,
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Set pixels not belonging to target label to zero",
    )
    parser.add_argument(
        "--label-name",
        help="Name of labels to use. For example `nuclei` or `cell`",
        default="cell",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        help="Image crop size",
        default=224,
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        help="Percentile min for normalization",
        default=0.1,
    )
    parser.add_argument(
        "--percentile-max",
        help="Percentile max for normalization",
        type=float,
        default=99.9,
    )
    parser.add_argument(
        "--local-percentile-normalize",
        help="Perform percentile normalization locally",
        action="store_true",
    )
    parser.add_argument(
        "--local-percentile-overlap",
        type=int,
        help="Overlap for local normalization",
    )

    parser.add_argument(
        "--label-filter",
        help="Expression to filter labels (e.g. barcode_Q_mean_0/barcode_Q_mean > 0.5 & Nuclei_Correlation_PearsonBox_ISS_PHENO>=0.9).",
    )

    parser.add_argument(
        "--chunks",
        help="Chunk size for local normalization",
        type=int,
    )
    parser.add_argument(
        "--output-format",
        choices=["tiff", "npy"],
        default="tiff",
        help="Output image format",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        help="Apply gaussian-smoothed mask to isolate target mask",
    )

    groupby_arg(parser)
    subset_arg(parser)
    force_arg(parser)
    dask_client_arg(parser)
    dask_cluster_arg(parser)

    no_version_arg(parser)

    parser.set_defaults(
        func=lazy_handler("scallops.cli.extract_crops:run_pipeline_extract_crops"),
    )
