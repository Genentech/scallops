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
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from importlib.metadata import version
from typing import Any, Literal, TypeAlias

import dask
import fsspec
import zarr
from distributed import Client

from scallops.utils import _cpu_count

SubParsers: TypeAlias = argparse._SubParsersAction
"""Type of the object returned by :meth:`argparse.ArgumentParser.add_subparsers`."""

ImageTuple: TypeAlias = tuple[tuple[str, ...], Sequence[str | zarr.Group], dict]
"""Group key, image sources, and metadata yielded by :func:`scallops.io._set_up_experiment`."""


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
) -> dict[str, int | None]:
    # adopted from LocalCluster
    if n_workers is None and threads_per_worker is None:
        if processes:
            threads_per_worker = min(_cpu_count(), 4)
            n_workers = max(1, _cpu_count() // threads_per_worker)
        # n_workers, threads_per_worker = nprocesses_nthreads(n=_cpu_count())
        else:
            n_workers = 1
            threads_per_worker = _cpu_count()
    if n_workers is None and threads_per_worker is not None:
        n_workers = max(1, _cpu_count() // threads_per_worker) if processes else 1
    if n_workers and threads_per_worker is None:
        # Overcommit threads per worker, rather than undercommit
        threads_per_worker = max(1, math.ceil(_cpu_count() / n_workers))

    return dict(threads_per_worker=threads_per_worker, n_workers=n_workers)


DEFAULT_DASK_CONFIG = {
    "dataframe.convert-string": False,
    "distributed.admin.large-graph-warning-threshold": "30MB",
    "distributed.admin.system-monitor.disk": False,
    "distributed.admin.system-monitor.gil.enabled": False,
    "distributed.admin.system-monitor.interval": "1 minute",
    "distributed.comm.timeouts.connect": "10 minutes",
    "distributed.comm.timeouts.tcp": "10 minutes",
    "distributed.scheduler.locks.lease-timeout": "10 minutes",
    "distributed.scheduler.worker-ttl": "10 minutes",
    "logging.distributed": "error",
}


def _create_default_dask_config(config: dict | None = None) -> dask.config.set:
    if config is None:
        return dask.config.set(DEFAULT_DASK_CONFIG)

    conf_ = DEFAULT_DASK_CONFIG.copy()
    conf_.update(config)
    return dask.config.set(conf_)


def _log_function_call(func: Callable) -> Callable:
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
    log_level = os.environ.get("SCALLOPS_LOGGING", "INFO").upper()
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            ("%(asctime)s - %(message)s"), datefmt="%m/%d/%Y %H:%M"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if os.environ.get("SCALLOPS_LOG_FUNCTION_CALLS", "0") == "1":
        _apply_logging_decorator(sys.modules[__name__])

    return logger


def _group_src_attrs(
    metadata: dict, metadata_fields: Sequence[str]
) -> tuple[list[Any], list[Any], list[Any]]:
    file_metadata = metadata["file_metadata"]
    src = metadata["group_metadata"]["src"]
    group_to_src: dict[tuple, list] = dict()
    group_to_attrs: dict[tuple, dict[str, list]] = dict()
    for i in range(len(file_metadata)):
        meta = file_metadata[i]
        group = tuple([meta[key] for key in meta.keys() if key not in metadata_fields])

        sources = group_to_src.get(group)
        if sources is None:
            sources = []
            attrs: dict[str, list] = dict(file_metadata=[])
            group_to_src[group] = sources
            group_to_attrs[group] = attrs
        else:
            attrs = group_to_attrs[group]
        sources.append(src[i])
        attrs["file_metadata"].append(meta)
    filelist = []
    attrs_list = []
    keys = []
    for key in group_to_src:
        filelist.append(group_to_src[key])
        attrs_list.append(group_to_attrs[key])
        keys.append(key)
    return keys, filelist, attrs_list


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


def _z_index_type(s: str) -> int | Literal["max"]:
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
        return "max"
    try:
        value = int(s)
    except ValueError:
        raise ValueError("z-index must be either a non-negative integer or `max`")
    if value < 0:
        raise ValueError("z-index must be >=0")
    return value


def _label_args_type(s: str) -> zarr.Group:
    # argparse only reports ArgumentTypeError/ValueError/TypeError as usage errors
    try:
        label_root = zarr.open(s, mode="r")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Unable to open {s}: {e}") from e
    labels_group = label_root.get("labels")
    if labels_group is None:
        raise argparse.ArgumentTypeError(f"Labels group not found for {s}")
    return labels_group


def find_label_group(label_groups: Sequence[zarr.Group], key: str) -> zarr.Group:
    """Find the label image `key` in one or more 'labels' groups.

    If `key` is present in more than one group, the group listed last wins. Entries
    that are not OME-Zarr images (e.g. left behind by an interrupted run) are not skipped.

    :param label_groups: The 'labels' groups to search, e.g. from :func:`labels_arg`.
    :param key: The label image name, e.g. `A1-102-nuclei`.
    :return: The OME-Zarr group for `key`.
    :raises ValueError: If `key` is not found in any group.
    """

    for group in reversed(label_groups):
        g = group.get(key)
        if g is not None:
            return g

    raise ValueError(f'Labels "{key}" not found.')


def last_label_match(matches: list) -> list:
    """Keep the last of a group's label matches, for `filter_matches` of
    :func:`scallops.io._set_up_experiment`.

    Matches are in `label_groups` order, so the store listed last wins.

    :param matches: The ``(Zarr node, fields)`` matches for one group.
    :return: The last match.
    """
    if len(matches) > 1:
        skipped = ", ".join(str(m[0]) for m in matches[:-1])
        logging.getLogger("scallops").info(f"Skipping duplicate labels at {skipped}.")
    return matches[-1:]


def z_index_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--z-index",
        help="Either `max` or a z-index (0-based)",
        default="max",
        type=_z_index_type,
    )


def _z_index_tile_type(s: str) -> int | str:
    """Parse and validate the per-tile z-index argument.

    :param s: The input string: 'max', 'focus', a z-index, or a Parquet path.
    :return: 'max', 'focus', the z-index as an integer, or the Parquet path.
    :raises argparse.ArgumentTypeError: If the input is a negative integer or an
        unrecognized keyword that does not look like a path.

    .. example::

       >>> _z_index_tile_type("Focus")
       'focus'
       >>> _z_index_tile_type("3")
       3
       >>> _z_index_tile_type("z.parquet")
       'z.parquet'
    """

    s = s.strip()
    if s.lower() in ("max", "focus"):
        return s.lower()

    try:
        value = int(s)
    except ValueError:
        # parquet path
        return s
    if value < 0:
        raise argparse.ArgumentTypeError("z-index must be >=0")
    return value


def z_index_tile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--z-index",
        help="Either `max`, `focus`, z-index (0-based), or a path to a Parquet file "
        "containing columns `key` and `z_index`. `focus` selects the best "
        "z-index using the slope of the image log-log power spectrum.",
        default="max",
        type=_z_index_tile_type,
    )


def t_index_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--t-index",
        dest="t_index",
        default=None,
        type=int,
        help="Time index (0-based) to select from images with a time dimension. "
        "Required when images have more than one timepoint (e.g. live imaging data).",
    )


def labels_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--labels",
        required=required,
        type=_label_args_type,
        nargs="+",
        help="One or more paths to zarr stores containing a 'labels' group. If the same "
        "label name appears in more than one store, the store listed last is used.",
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
