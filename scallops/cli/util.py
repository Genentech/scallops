"""Utility Module for scallops Command-Line Interface.

This module provides utility functions for the command-line interface (CLI) of the
scallops package.

Authors:     - The SCALLOPS development team
"""

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
from typing import Any, TypeAlias

import dask
import fsspec
import zarr
from distributed import Client

from scallops.utils import _cpu_count

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


def find_label_group(label_groups: Sequence[zarr.Group], key: str) -> zarr.Group:
    """Find the label image `key` in one or more 'labels' groups.

    If `key` is present in more than one group, the group listed last wins. Entries
    that are not OME-Zarr images (e.g. left behind by an interrupted run) are not skipped.

    :param label_groups: The 'labels' groups to search, e.g. from :func:`scallops.cli.args.labels_arg`.
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
