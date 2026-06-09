"""Utility Module for scallops Command-Line Interface.

This module provides utility functions for the command-line interface (CLI) of the
scallops package.

Authors:     - The SCALLOPS development team
"""

import argparse
import functools
import json
import logging
import math
import os
import sys
import time
import types
from collections.abc import Sequence
from contextlib import nullcontext
from importlib.metadata import version
from typing import Any

import dask
import dask.array as da
import fsspec
import numpy as np
import xarray as xr
import zarr
from distributed import Client

from scallops.io import save_ome_tiff
from scallops.utils import _cpu_count
from scallops.zarr_io import _write_zarr_image


class ContextFilter(logging.Filter):
    """Filter to add process ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.pid = os.getpid()
        return True


def _create_dask_client(
    dask_scheduler_url: str | None, **kwargs
) -> Client | nullcontext:
    """Create a Dask client or return a null context if dask_scheduler_url is 'none'.

    :param dask_scheduler_url: URL of the Dask scheduler or 'none' for single-machine
        use.
    :param kwargs: Additional keyword arguments to pass to the Dask client.
    :return: Dask client or null context
    """

    if dask_scheduler_url != "none":
        if dask_scheduler_url == "localhost":
            dask_scheduler_url = None
        return Client(dask_scheduler_url, **kwargs)
    return nullcontext()


def _dask_workers_threads(
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    processes: bool = True,
) -> dict[str, int]:
    if n_workers is None and threads_per_worker is None:
        if processes:
            threads_per_worker = min(_cpu_count(), 4)
            n_workers = max(1, _cpu_count() // threads_per_worker)
        else:
            n_workers = 1
            threads_per_worker = _cpu_count()
    if n_workers is None and threads_per_worker is not None:
        n_workers = max(1, _cpu_count() // threads_per_worker) if processes else 1
    if n_workers and threads_per_worker is None:
        # Overcommit threads per worker, rather than undercommit
        threads_per_worker = max(1, int(math.ceil(_cpu_count() / n_workers)))

    return dict(threads_per_worker=threads_per_worker, n_workers=n_workers)


DEFAULT_DASK_CONFIG = {
    "dataframe.convert-string": False,
    "distributed.admin.large-graph-warning-threshold": "30MB",
    "distributed.admin.system-monitor.disk": False,
    "distributed.admin.system-monitor.gil.enabled": False,
    "distributed.admin.system-monitor.interval": "1 minute",
    "distributed.comm.timeouts.connect": "120s",
    "distributed.comm.timeouts.tcp": "60s",
    "distributed.scheduler.worker-ttl": "10 minutes",
    "logging.distributed": "error",
}


def _create_default_dask_config(config: dict | None = None) -> dask.config.set:
    if config is None:
        return dask.config.set(DEFAULT_DASK_CONFIG)

    conf_ = DEFAULT_DASK_CONFIG.copy()
    conf_.update(config)
    return dask.config.set(conf_)


def _log_function_call(func: types.FunctionType) -> types.FunctionType:
    """Decorator to log function calls and execution time.

    :param func: Function to wrap.
    :return: Wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(
            f"Exiting {func.__name__} with result={result} after {elapsed_time:.2f} seconds"
        )
        return result

    return wrapper


def _apply_logging_decorator(module: types.ModuleType) -> None:
    """Apply the logging decorator to all functions in a module.

    :param module: Module whose functions will be wrapped.
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, types.FunctionType):
            setattr(module, attr_name, _log_function_call(attr))


def _get_cli_logger() -> logging.Logger:
    """Configure and return the CLI logger.

    :return: Configured logger.
    """
    logger = logging.getLogger("scallops")
    log_level = os.environ.get("scallops_loglevel", "INFO").upper()
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            (
                "%(asctime)s - %(message)s"
                if log_level != "DEBUG"
                else "%(asctime)s - PID: %(process)d - %(funcName)s - %(levelname)s - %(message)s"
            ),
            datefmt="%m/%d/%Y %H:%M",
        )
        if log_level == "DEBUG":
            handler.addFilter(ContextFilter())
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_level == "DEBUG":
        _apply_logging_decorator(sys.modules[__name__])

    return logger


def _group_src_attrs(
    metadata: dict, metadata_fields: Sequence[str]
) -> tuple[list[Any], list[Any], list[Any]]:
    file_metadata = metadata["file_metadata"]
    src = metadata["group_metadata"]["src"]
    group_to_src = dict()
    group_to_attrs = dict()
    for i in range(len(file_metadata)):
        meta = file_metadata[i]
        group = tuple([meta[key] for key in meta.keys() if key not in metadata_fields])

        sources = group_to_src.get(group)
        if sources is None:
            sources = []
            attrs = dict(file_metadata=[])
            group_to_src[group] = sources
            group_to_attrs[group] = attrs
        sources.append(src[i])
        attrs["file_metadata"].append(meta)
    filelist = []
    attrs = []
    keys = []
    for key in group_to_src:
        filelist.append(group_to_src[key])
        attrs.append(group_to_attrs[key])
        keys.append(key)
    return keys, filelist, attrs


def _write_image(
    name: str,
    root: zarr.Group | str,
    image: np.ndarray | xr.DataArray | da.Array,
    output_format: str,
    file_separator: str,
    metadata: dict | None = None,
    compute: bool = True,
    **kwargs,
) -> None:
    """Write image data to Zarr or TIFF format.

    :param name: Name of the image.
    :param root: Zarr root or directory path for saving the image.
    :param image: Image data to be saved.
    :param output_format: Format for saving the image ('zarr' or 'tiff').
    :param file_separator: Separator used in file paths.
    :param metadata: Optional metadata for the image.
    :param compute: Whether to compute the Dask array before saving.
    """
    if output_format == "zarr":
        _write_zarr_image(
            name=name,
            root=root,
            image=image,
            metadata=metadata,
            compute=compute,
            **kwargs,
        )
    elif output_format == "tiff":
        image_path = f"{root}{file_separator}{name}.tif"
        if isinstance(image, xr.DataArray):
            save_ome_tiff(
                data=image.data,
                uri=image_path,
                channel_names=image.coords.get("c"),
                dim_order="".join(image.dims).upper(),
            )
        else:
            save_ome_tiff(data=image, uri=image_path)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def images_arg(parser: argparse.ArgumentParser) -> None:
    """Add the images argument to a parser."""
    parser.add_argument(
        "-i",
        "--images",
        nargs="+",
        required=True,
        help="Paths to input images or CVS/Parquet with `image` column "
        "containing full image path and additional columns "
        "containing metadata such as `plate`, `well`, `t`, `c`, or `z`. Note that "
        "image pattern is ignored when CSV/Parquet is used.",
    )


def _get_version() -> str:
    return version("scallops")


def _get_command() -> str:
    command = ["scallops"]
    for arg in sys.argv[1:]:
        if arg.find(" ") != -1:
            arg = f'"{arg}"'
        command.append(arg)

    return " ".join(command)


def cli_metadata():
    return dict(scallops_version=_get_version(), scallops_command=_get_command())


def cli_parquet_metadata():
    d = cli_metadata()
    d["scallops"] = json.dumps(d["scallops"])
    return d


def no_version_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --no-version argument to a parser."""
    parser.add_argument(
        "--no-version",
        action="store_true",
        help="Do not store command line arguments and scallops version in "
        "output metadata.",
    )


def image_pattern_arg(parser: argparse.ArgumentParser) -> None:
    """Add the image pattern argument to a parser."""
    parser.add_argument(
        "--image-pattern", help="Pattern to extract metadata from file names."
    )


def groupby_arg(parser: argparse.ArgumentParser) -> None:
    """Add the groupby argument to a parser."""
    parser.add_argument("-g", "--groupby", nargs="*", help="Keys to group images.")


def expected_images_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Validate that the specified number of images are provided.",
    )


def verbose_arg(parser):
    """Add the verbose argument to a parser."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run in verbose mode. Useful for debugging.",
    )


def subset_arg(parser: argparse.ArgumentParser) -> None:
    """Add the subset argument to a parser."""
    parser.add_argument(
        "-s", "--subset", nargs="*", help="Subset of images to include."
    )


def force_arg(parser: argparse.ArgumentParser) -> None:
    """Add the '--force' argument to the parser to enable overwriting existing output.

    :param parser: The argument parser to which the '--force' argument is added.
    """
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output",
    )


def _z_index_type(s: str) -> int | str:
    """Parse and validate the z-index argument.

    This function ensures that the input is a non-negative integer 'max'

    :param s: The input string representing a z-index.
    :return: The validated z-index as an integer or a string if == 'max'
    :raises ValueError: If the input is not a non-negative integer and != 'max'

    .. example::

       >>> _z_index_type("max")
       'max'
       >>> _z_index_type("3")
       3
       >>> _z_index_type("-1")
       Traceback (most recent call last):
           ...
       ValueError: z-index must be >=0
    """

    s = s.strip()
    if s == "max":
        return s
    try:
        value = int(s)
    except ValueError:
        raise ValueError("z-index must be either a non-negative integer or `max`")
    if value < 0:
        raise ValueError("z-index must be >=0")
    return value


def z_index_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--z-index",
        help="Either `max` or a z-index (0-based)",
        default="max",
        type=_z_index_type,
    )


def z_index_tile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--z-index",
        help="Either `max`, `focus`, z-index (0-based), or a path to a Parquet file "
        "containing columns `key` and `z_index`. `Focus` selects the best "
        "z-index using the slope of the image log-log power spectrum.",
        default="max",
        type=str,
    )


def output_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Add the output directory argument to a parser.

    :param parser: The argument parser to which the output directory argument is added.
    """
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="Path to the output directory where the results will be saved.",
    )


def dask_client_arg(parser: argparse.ArgumentParser, value=None) -> None:
    """Add the Dask client argument to a parser.

    :param parser: The argument parser to which the Dask client argument is added.
    :param value: Default value.
    """
    parser.add_argument(
        "--client",
        type=str,
        default=value,
        help="URL of the Dask scheduler. Use 'none' to disable distributed execution.",
    )


def dask_cluster_arg(parser: argparse.ArgumentParser) -> None:
    """Add the Dask cluster parameters.

    :param parser: The argument parser to which the Dask client argument is added.
    """
    parser.add_argument(
        "--dask-cluster",
        type=str,
        help="JSON URL or inline JSON containing dask cluster parameters.",
    )


def barcodes_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    """Add the barcodes file argument to a parser.

    :param parser: The argument parser to which the barcodes file argument is added.
    """
    parser.add_argument(
        "--barcodes",
        dest="barcodes",
        required=required,
        help="Path to the barcode CSV file containing a column named 'barcode'.",
    )


def load_json(path_or_str: str) -> dict:
    """Load a JSON file into a dictionary for inline JSON or URL.
    :param path_or_str: JSON file URL or inline JSON.
    :return: The dictionary loaded from JSON.
    """
    fs, _ = fsspec.url_to_fs(path_or_str)
    if fs.exists(path_or_str):
        with fs.open(path_or_str, "rt") as fp:
            return json.load(fp)
    return json.loads(path_or_str)


def _write_img_size(file_list: list[str]):
    from scallops.io import _images2fov, _localize_path

    local_file_list = []
    cleanup_file_list = []
    for path in file_list:
        local_path = _localize_path(path)
        if local_path is not None:
            cleanup_file_list.append(local_path)
            local_file_list.append(local_path)
        else:
            local_file_list.append(path)
    sizes = _images2fov(local_file_list, dask=True).sizes
    for path in cleanup_file_list:
        os.remove(path)
    with open("img_size.txt", "wt") as f:
        for dim in ["t", "c", "z", "y", "x"]:
            s = sizes[dim] if dim in sizes else 0
            f.write(f"{s}")
            f.write("\n")


def _write_group_size(metadata: dict):
    n_tiles = len(metadata["file_metadata"])
    metadata_fields = [v for v in ("c", "z") if v in metadata["file_metadata"][0]]
    if len(metadata_fields) > 0:
        from scallops.cli.util import _group_src_attrs

        keys, channel_sources, filepaths = _group_src_attrs(
            metadata=metadata, metadata_fields=tuple(metadata_fields)
        )
        n_tiles = len(filepaths)
    with open("group_size.txt", "wt") as f:
        f.write(f"{n_tiles}")
        f.write("\n")


def _list_images_wdl(
    image_pattern: str,
    urls: list[str],
    groupby: list[str],
    reference_time: str | None,
    subset: list[str] | None,
    batch_size_str: str | None,
    save_group_size: bool = False,
    expected_cycles_str: int | None = None,
):
    """Used by WDL workflow to output info about images"""
    from scallops.io import _set_up_experiment

    batch_size = 1
    expected_cycles = None
    if expected_cycles_str is not None and expected_cycles_str != "":
        expected_cycles = int(expected_cycles_str)
    if batch_size_str is not None and batch_size_str != "":
        batch_size = int(batch_size_str)
    if reference_time == "":
        reference_time = None

    if subset is not None and (
        len(subset) == 0 or (len(subset) == 1 and subset[0] == "")
    ):
        subset = None
    if image_pattern != "":
        groupby = [g for g in groupby if "{" + g + "}" in image_pattern]
    exp_gen = _set_up_experiment(
        image_path=urls, files_pattern=image_pattern, group_by=groupby, subset=subset
    )
    # "groups.txt": each line passed to --subset in cli
    # "groupby.txt": filtered groupby with values not in image_pattern removed
    groupby_t = "t" in groupby
    times = None

    if not save_group_size:
        with open("group_size.txt", "wt") as f:
            f.write("0\n")

    with (
        open("subsets.txt", "wt") as groups_out,
        open("subsets_with_t.txt", "wt") as groups_with_t_out,
    ):
        subset_ids = []
        subset_ids_with_reference_times = []
        first = True

        for g, file_list, metadata in exp_gen:
            times = None
            if first:
                first = False
                if save_group_size:
                    _write_group_size(metadata)
            if not groupby_t and "t" in metadata["file_metadata"][0]:
                times = [md["t"] for md in metadata["file_metadata"]]
                if expected_cycles is not None:
                    assert len(times) == expected_cycles
            t_suffix = ""
            if times is not None and len(times) > 0:
                t_suffix = (
                    f"-{times[0]}" if reference_time is None else f"-{reference_time}"
                )

            subset_ids.append('"' + metadata["id"] + '"')
            subset_ids_with_reference_times.append(
                '"' + metadata["id"] + t_suffix + '"'
            )
            if len(subset_ids) == batch_size:
                groups_out.write(" ".join(subset_ids))
                groups_out.write("\n")

                groups_with_t_out.write(" ".join(subset_ids_with_reference_times))
                groups_with_t_out.write("\n")

                subset_ids = []
                subset_ids_with_reference_times = []
        if len(subset_ids) > 0:
            groups_out.write(" ".join(subset_ids))
            groups_out.write("\n")

            groups_with_t_out.write(" ".join(subset_ids_with_reference_times))
            groups_with_t_out.write("\n")

    with open("groupby.txt", "wt") as f:
        for g in groupby:
            f.write(g)
            f.write("\n")
    groupby_with_t = list(groupby)

    if not groupby_t and times is not None:
        groupby_with_t.append("t")

    with open("groupby_with_t.txt", "wt") as f:
        for g in groupby_with_t:
            f.write(g)
            f.write("\n")

    with open("t.txt", "wt") as f:
        if times is not None:
            for val in times:
                f.write(str(val))
                f.write("\n")

    with open("groupby_pattern.txt", "wt") as f:
        first = True
        for g in groupby:
            if not first:
                f.write("-")
            first = False
            f.write("{")
            f.write(g)
            f.write("}")
    reference_time_suffix = ""
    if reference_time is not None:
        reference_time_suffix = f"-{reference_time}"
    elif times is not None and len(times) > 0:
        reference_time_suffix = f"-{times[0]}"

    with open("groupby_pattern_with_reference_t.txt", "wt") as f:
        first = True
        for g in groupby:
            if not first:
                f.write("-")
            first = False
            f.write("{")
            f.write(g)
            f.write("}")
            f.write(reference_time_suffix)
