"""
Module: scallops.cli.segment

This module provides a command-line interface (CLI) for performing nuclei and cell segmentation.



Authors:
    - The SCALLOPS development team

"""

import argparse
import importlib
from typing import Callable, Literal, Optional

import dask.array as da
import fsspec
import numpy as np
import zarr
from dask.bag import from_sequence
from zarr import Group

from scallops.cli.util import (
    _create_dask_client,
    _create_default_dask_config,
    _dask_workers_threads,
    _get_cli_logger,
    cli_metadata,
    load_json,
)
from scallops.io import _add_suffix, _images2fov, _set_up_experiment, get_image_spacing
from scallops.segmentation import remove_labels_by_area
from scallops.segmentation.util import _delete_lock_files, identify_tertiary_objects
from scallops.utils import _cpu_count
from scallops.xr import _z_projection
from scallops.zarr_io import (
    _write_zarr_labels,
    is_ome_zarr_array,
    open_ome_zarr,
    read_ome_zarr_array,
)

logger = _get_cli_logger()


def segment_nuclei(
    group: str,  # NOT USED
    file_list: list[str],
    metadata: dict,
    dapi_channel: int,
    method: Callable,
    root: Group,
    z_index: int | str,
    min_area: float | None = None,
    max_area: float | None = None,
    chunks: None | tuple[int, int] = None,
    chunk_overlap: None | int = None,
    clip: bool = False,
    force: bool = False,
    no_version: bool = False,
    pmin: float | None = None,
    pmax: float | None = None,
) -> Group:
    """Segment nuclei in images.

    :param group: Name of the group.
    :param file_list: List of file paths.
    :param metadata: Metadata for the images.
    :param dapi_channel: Index of the DAPI channel.
    :param method: Segmentation method.
    :param root: Zarr hierarchy root.
    :param z_index: Either 'max' or z-index
    :param min_area: Minimum area threshold for filtering labels.
    :param max_area: Maximum area threshold for filtering labels.
    :param chunks: Tuple specifying chunking size for Dask arrays.
    :param chunk_overlap: Overlap size for chunking.
    :param clip: Whether to clip intensity values during segmentation.
    :param pmin: Minimum percentile for image normalization.
    :param pmax: Maximum percentile for image normalization.
    :param force: Whether to overwrite existing output
    :param no_version: Whether to skip version/CLI information in output.
    :return: The root (for dask)
    """

    image_key = metadata["id"]
    if not force and is_ome_zarr_array(root.get(f"labels/{image_key}-nuclei")):
        logger.info(f"Skipping nuclei segmentation for {image_key}")
        return root
    logger.info(f"Running nuclei segmentation for {image_key}")
    image = _images2fov(file_list, metadata, dask=True).squeeze()
    image = _z_projection(image, z_index)

    nuclei_seg_args = {}
    if method.__name__ in ["segment_nuclei_stardist"]:
        nuclei_seg_args["clip"] = clip
        if pmin is not None:
            nuclei_seg_args["pmin"] = pmin
        if pmax is not None:
            nuclei_seg_args["pmax"] = pmax

    if method.__name__ in ["segment_nuclei_cellpose", "segment_nuclei_stardist"]:
        if chunks is not None:
            nuclei_seg_args["chunks"] = chunks
        if chunk_overlap is not None:
            nuclei_seg_args["depth"] = chunk_overlap

    nuclei = method(image=image, nuclei_channel=dapi_channel, **nuclei_seg_args)
    all_nuclei = nuclei

    if min_area is not None or max_area is not None:
        nuclei = remove_labels_by_area(nuclei, min_area, max_area)

    labels_dict = dict(nuclei=nuclei)
    if all_nuclei is not nuclei:
        labels_dict["nuclei.all"] = all_nuclei
    spacing = get_image_spacing(image.attrs)
    label_metadata = dict()
    if spacing is not None:
        for key in labels_dict.keys():
            label_metadata[key] = dict(physical_pixel_sizes=spacing)
    if not no_version:
        label_metadata.update(cli_metadata())
    for key, label_data in labels_dict.items():
        group_metadata = {
            "image-label": {"source": {"image": f"../../images/{image_key}"}}
        }
        additional_metadata = label_metadata.get(key) if label_metadata else None
        storage_options = None
        if isinstance(label_data, np.ndarray):
            storage_options = {"chunks": image.data.chunksize[-2:]}
        _write_zarr_labels(
            name=f"{image_key}-{key}",
            root=root,
            metadata=additional_metadata,
            group_metadata=group_metadata,
            labels=label_data,
            storage_options=storage_options,
        )

    return root


def segment_cells(
    group: str,
    file_list: list[str],
    metadata: dict,
    dapi_channel: int,
    cyto_channel: int | list[int] | None,
    method: Callable,
    root: Group,
    z_index: int | str,
    min_area: float | None = None,
    max_area: float | None = None,
    chunks: None | tuple[int, int] = None,
    chunk_overlap: None | int = None,
    nuclei_image_root: Group = None,
    cell_segmentation_threshold: str | float = "Li",
    threshold_correction_factor: float = 1,
    cell_segmentation_rolling_ball: bool = False,
    cell_segmentation_sigma: Optional[float] = None,
    closing_radius: Optional[int] = None,
    cell_segmentation_t: Optional[list[int]] = None,
    force: bool = False,
    shrink_primary: bool = False,
    no_version: bool = False,
    nuclei_min_area: float | None = None,
    nuclei_max_area: float | None = None,
    watershed_method: Literal["binary", "distance", "intensity"] = "distance",
) -> Group:
    """Segment cells in images.

    :param group: Name of the group.
    :param file_list: List of file paths.
    :param metadata: Metadata for the images.
    :param dapi_channel: Index of the DAPI channel.
    :param cyto_channel: Index or list of indices of the cytoplasmic channels.
    :param method: Segmentation method.
    :param root: Zarr hierarchy root.
    :param z_index: Either 'max' or z-index
    :param min_area: Minimum area threshold for filtering labels.
    :param max_area: Maximum area threshold for filtering labels.
    :param chunks: Tuple specifying chunking size for Dask arrays.
    :param chunk_overlap: Overlap size for chunking.
    :param nuclei_image_root: Zarr hierarchy root for nuclei images (if applicable).
    :param cell_segmentation_threshold: Threshold for cell segmentation.
    :param threshold_correction_factor: Factor to adjust threshold by.
    :param cell_segmentation_rolling_ball: Use rolling ball mask for cell segmentation.
    :param cell_segmentation_sigma: Standard deviation for smoothing in cell segmentation.
    :param closing_radius: Radius for closing operation in cell segmentation.
    :param cell_segmentation_t: List of timepoints to consider for cell segmentation.
    :param force: Whether to overwrite existing output
    :param shrink_primary: Whether to shrink primary labels.
    :param no_version: Whether to skip version/CLI information in output.
    """
    image_key = metadata["id"]
    if not force and is_ome_zarr_array(root.get(f"labels/{image_key}-cell")):
        logger.info(f"Skipping cell segmentation for {image_key}")
        return root
    image = _images2fov(file_list, metadata, dask=True).squeeze()
    image = _z_projection(image, z_index)
    if cyto_channel is None:
        cyto_channel = np.delete(np.arange(image.sizes["c"]), dapi_channel)
    if len(cyto_channel) == 0:
        cyto_channel = [dapi_channel]
    logger.info(f"Running cell segmentation for {image_key}")

    cell_seg_args = {"nuclei_channel": dapi_channel}
    nuclei = None
    if method.__name__ in ["segment_cells_watershed", "segment_cells_propagation"]:
        nuclei = read_ome_zarr_array(
            nuclei_image_root["labels"][image_key + "-nuclei"]
        ).values
        assert nuclei.shape == (
            image.sizes["y"],
            image.sizes["x"],
        ), "Size mismatch between nuclei and image"
        if nuclei_min_area is not None or nuclei_max_area is not None:
            nuclei = remove_labels_by_area(nuclei, nuclei_min_area, nuclei_max_area)
        cell_seg_args["nuclei"] = nuclei
        cell_seg_args["threshold_correction_factor"] = threshold_correction_factor
        cell_seg_args["threshold"] = cell_segmentation_threshold
        cell_seg_args["rolling_ball"] = cell_segmentation_rolling_ball
        cell_seg_args["sigma"] = cell_segmentation_sigma
        cell_seg_args["closing_radius"] = closing_radius
        cell_seg_args["t"] = cell_segmentation_t
    if method.__name__ == "segment_cells_watershed":
        cell_seg_args["watershed_method"] = watershed_method
    if chunks is not None:
        cell_seg_args["chunks"] = chunks
    if chunk_overlap is not None:
        cell_seg_args["depth"] = chunk_overlap
    result = method(image=image, cyto_channel=cyto_channel, **cell_seg_args)
    if method.__name__ in ["segment_cells_watershed", "segment_cells_propagation"]:
        cells = result[0]
        cell_threshold = result[1]
    else:
        cells = result
        cell_threshold = None

    all_cells = cells
    if min_area is not None or max_area is not None:
        if isinstance(cells, da.Array):
            cells = cells.compute()
        cells = remove_labels_by_area(cells, min_area, max_area, relabel=False)

    labels_dict = dict(cell=cells)
    if all_cells is not cells:
        labels_dict["cell.all"] = all_cells
    if nuclei is not None:
        cytosol = identify_tertiary_objects(nuclei, cells, shrink_primary)
        labels_dict["cytosol"] = cytosol
    label_metadata = dict()
    if cell_threshold is not None:
        label_metadata = dict(cell=dict(threshold=cell_threshold))

    if not no_version:
        label_metadata.update(cli_metadata())
    spacing = get_image_spacing(image.attrs)
    if spacing is not None:
        if label_metadata is None:
            label_metadata = dict()
        for key in labels_dict.keys():
            label_metadata[key] = dict(physical_pixel_sizes=spacing)

    for key, label_data in labels_dict.items():
        group_metadata = {
            "image-label": {"source": {"image": f"../../images/{image_key}"}}
        }
        additional_metadata = label_metadata.get(key) if label_metadata else None
        storage_options = None
        if isinstance(label_data, np.ndarray):
            storage_options = {"chunks": image.data.chunksize[-2:]}
        _write_zarr_labels(
            name=f"{image_key}-{key}",
            root=root,
            metadata=additional_metadata,
            group_metadata=group_metadata,
            labels=label_data,
            storage_options=storage_options,
        )

    return root


def run_pipeline(arguments: argparse.Namespace, nuclei: bool):
    """Run nuclei or cell segmentation pipeline.

    :param arguments: Command line arguments.
    :param nuclei: If True, perform nuclei segmentation; otherwise, perform cell segmentation.
    :return: None
    """
    method = arguments.method
    dask_server_url = arguments.client
    dask_cluster_parameters = (
        load_json(arguments.dask_cluster) if arguments.dask_cluster is not None else {}
    )

    if dask_server_url is None and arguments.dask_cluster is None:
        if method == "cellpose":
            threads_per_worker = 8
        else:
            threads_per_worker = _cpu_count()
        dask_cluster_parameters = _dask_workers_threads(
            threads_per_worker=threads_per_worker
        )

    data_path = arguments.images
    z_index = arguments.z_index

    if method == "cellpose" and dask_server_url == "none":
        logger.warning("Cellpose works best with distributed dask cluster.")
    image_pattern = arguments.image_pattern
    group_by = arguments.groupby
    dapi_channel = arguments.dapi_channel

    subset = arguments.subset
    min_area = arguments.min_area
    max_area = arguments.max_area
    no_version = arguments.no_version
    chunks = arguments.chunks
    chunk_overlap = arguments.chunk_overlap
    output = arguments.output.rstrip("/")

    force = arguments.force
    if chunks is not None:
        chunks = (chunks, chunks)
    output = _add_suffix(output, ".zarr")

    fs, _ = fsspec.core.url_to_fs(output)
    fs.makedirs(output, exist_ok=True)
    output_root = open_ome_zarr(output, mode="a")

    kwargs = dict()

    if not nuclei:
        kwargs["nuclei_min_area"] = arguments.nuclei_min_area
        kwargs["nuclei_max_area"] = arguments.nuclei_max_area
        kwargs["cyto_channel"] = arguments.cyto_channel
        kwargs["shrink_primary"] = arguments.shrink_nuclei
        threshold = arguments.threshold.lower()
        if threshold not in ["li", "otsu", "local"]:
            try:
                threshold = float(threshold)
            except ValueError:
                raise ValueError(
                    "Threshold must be either `Li`, `Otsu`, `Local`, or a valid number. Got {}".format(
                        threshold
                    )
                )
        elif threshold == "local":
            assert arguments.cell_segmentation_sigma is not None, (
                "Please provide sigma for `local` threshold"
            )

        kwargs["cell_segmentation_t"] = arguments.cell_segmentation_t
        if method in ["watershed", "watershed-intensity", "propagation"]:
            nuclei_label = arguments.nuclei_label
            if nuclei_label is None:
                raise ValueError(
                    f"Please provide nuclei labels for {method} segmentation"
                )
            if method == "watershed":
                kwargs["watershed_method"] = "distance"
            elif method == "watershed-intensity":
                method = "watershed"
                kwargs["watershed_method"] = "intensity"
            kwargs["closing_radius"] = arguments.closing_radius
            kwargs["nuclei_image_root"] = zarr.open(nuclei_label, mode="r")
            kwargs["cell_segmentation_threshold"] = threshold
            kwargs["threshold_correction_factor"] = (
                arguments.threshold_correction_factor
            )

            kwargs["cell_segmentation_rolling_ball"] = (
                arguments.cell_segmentation_rolling_ball
            )
            kwargs["cell_segmentation_sigma"] = arguments.cell_segmentation_sigma

    elif method in ["stardist"]:
        kwargs["clip"] = arguments.stardist_clip
        kwargs["pmin"] = arguments.stardist_pmin
        kwargs["pmax"] = arguments.stardist_pmax
    method = getattr(
        importlib.import_module("scallops.segmentation." + method),
        f"{'segment_nuclei_' if nuclei else 'segment_cells_'}{method}",
    )
    image_seq = from_sequence(
        _set_up_experiment(data_path, image_pattern, group_by, subset=subset)
    )
    with (
        _create_default_dask_config(),
        _create_dask_client(dask_server_url, **dask_cluster_parameters),
    ):
        image_seq.starmap(
            segment_nuclei if nuclei else segment_cells,
            dapi_channel=dapi_channel,
            method=method,
            root=output_root,
            min_area=min_area,
            max_area=max_area,
            chunks=chunks,
            chunk_overlap=chunk_overlap,
            z_index=z_index,
            force=force,
            no_version=no_version,
            **kwargs,
        ).compute()
    _delete_lock_files()


def run_pipeline_segment_nuclei(arguments: argparse.Namespace):
    """Run nuclei segmentation pipeline.

    :param arguments: Command line arguments.
    :return: None
    """
    run_pipeline(arguments, True)


def run_pipeline_segment_cell(arguments: argparse.Namespace):
    """Run cell segmentation pipeline.

    :param arguments: Command line arguments.
    """
    run_pipeline(arguments, False)
