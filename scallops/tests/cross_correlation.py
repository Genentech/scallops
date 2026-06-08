import argparse
from argparse import (
    Namespace,
)

import zarr
from dask.bag import from_sequence

from scallops.cli.arg_parser import _sort_groups
from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _get_cli_logger,
    dask_client_arg,
    dask_cluster_arg,
    force_arg,
    groupby_arg,
    image_pattern_arg,
    images_arg,
    load_json,
    subset_arg,
)
from scallops.io import (
    _add_suffix,
    _images2fov,
    _set_up_experiment,
)
from scallops.registration.crosscorrelation import align_image
from scallops.utils import _cpu_count
from scallops.zarr_io import (
    _write_zarr_image,
    open_ome_zarr,
)

logger = _get_cli_logger()


def single_cross_correlation(
    _,
    file_list: list[str],
    metadata: dict,
    across_t_channel: None | int,
    within_t_channel: None | list[int],
    filter_percentiles: tuple[float, float],
    output_dir: zarr.Group,
    force: bool = False,
) -> None:
    """Perform cross-correlation-based registration for a single image.

    This function aligns a single image across or within timepoints using cross-correlation. It
    allows filtering based on specified percentiles to refine the alignment and saves the aligned
    image to the provided Zarr output directory.

    :param _: Unused placeholder parameter required by the Dask starmap call.
    :param file_list: List of file paths for the images to process.
    :param metadata: Metadata dictionary containing information about the image.
    :param across_t_channel: Channel index used for alignment across timepoints. If None, no across-
        time alignment is performed.
    :param within_t_channel: List of channel indices to align within the same timepoint. If None, no
        within-time alignment is performed.
    :param filter_percentiles: Tuple containing the lower and upper percentiles for filtering pixel
        intensities (e.g., (0.1, 0.9)).
    :param output_dir: Zarr group where the aligned images will be saved.
    :param force: If True, forces the re-alignment even if the output already exists.
    """
    image_key = metadata["id"]
    if not force and output_dir.get(f"images/{image_key}") is not None:
        return logger.info(f"Skipping cross correlation for {image_key}")
    logger.info(f"Running cross correlation for {image_key}")
    image = _images2fov(file_list, metadata)
    image = align_image(
        image,
        align_within_time_channels=within_t_channel,
        align_between_time_channel=across_t_channel,
        filter_percentiles=filter_percentiles,
    )
    _write_zarr_image(name=image_key, root=output_dir, image=image)


def _run_cross_correlation_registration(arguments: Namespace) -> None:
    """Run cross-correlation registration."""
    from scallops.tests.cross_correlation import run_cross_correlation_registration

    run_cross_correlation_registration(arguments)


def run_cross_correlation_registration(arguments: argparse.Namespace) -> None:
    """Run image registration using cross-correlation.

    This function orchestrates the image registration process using cross-correlation, aligning a
    collection of images across and within cycles. It considers parameters such as filter
    percentiles, within-time channels, and output directories for storing the registered images.

    :param arguments: An argparse.Namespace object containing command-line arguments.
    """
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )
    if dask_server_url == "none" and dask_cluster_parameters:
        # If a JSON is passed to dask_cluster avoid null_context
        dask_server_url = None

    if dask_server_url is None and arguments.dask_cluster is None:
        dask_cluster_parameters = dict(threads_per_worker=1, n_workers=_cpu_count())

    filter_percentiles = (
        arguments.registration_filter_min,
        arguments.registration_filter_max,
    )
    within_t_channel = arguments.within_t_channel
    images = arguments.images
    image_pattern = arguments.image_pattern
    across_t_channel = arguments.across_t_channel
    group_by = arguments.groupby
    subset = arguments.subset
    force = arguments.force
    output_dir = arguments.output
    output_dir = _add_suffix(output_dir, ".zarr")

    output_dir = open_ome_zarr(output_dir, mode="a")

    moving_image_gen = _set_up_experiment(
        images, image_pattern, group_by, subset=subset
    )

    image_bag = from_sequence(moving_image_gen)
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        bag = image_bag.starmap(
            single_cross_correlation,
            output_dir=output_dir,
            filter_percentiles=filter_percentiles,
            within_t_channel=within_t_channel,
            across_t_channel=across_t_channel,
            force=force,
        )
        bag.compute()


def _create_cross_correlation_parser():
    """Create the parser for cross-correlation registration."""
    parser = argparse.ArgumentParser()
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
    return parser


if __name__ == "__main__":
    parser = _create_cross_correlation_parser()
    args = parser.parse_args()
    _run_cross_correlation_registration(args)
