"""Argument parsing helpers for the scallops command-line interface.

This module is imported when the CLI starts, so it must only import the standard
library. Put helpers that need dask, zarr, or other heavy dependencies in
:mod:`scallops.cli.util`, or import them inside the function that uses them.

Authors:     - The SCALLOPS development team
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    import zarr

SubParsers: TypeAlias = argparse._SubParsersAction
"""Type of the object returned by :meth:`argparse.ArgumentParser.add_subparsers`."""

ArgContainer: TypeAlias = argparse._ActionsContainer
"""A parser or argument group, i.e. anything with ``add_argument``."""

_GROUP_ORDER = {
    "positional arguments": 0,
    "required arguments": 1,
    "optional arguments": 2,
}


class ScallopsArgumentParser(argparse.ArgumentParser):
    """Argument parser that titles the options group "optional arguments"."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for g in self._action_groups:
            if g.title == "options":
                g.title = "optional arguments"


def _sort_groups(parser: argparse.ArgumentParser) -> None:
    """List positional, then required, then optional arguments in help output."""
    parser._action_groups.sort(key=lambda x: _GROUP_ORDER.get(x.title or "", 3))


def _sort_groups_recursive(parser: argparse.ArgumentParser) -> None:
    """Apply :func:`_sort_groups` to `parser` and all of its subcommands."""
    _sort_groups(parser)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in set(action.choices.values()):
                _sort_groups_recursive(subparser)


def add_command(
    subparsers: SubParsers, name: str, default_help: bool, **kwargs: Any
) -> argparse.ArgumentParser:
    """Add a subcommand parser.

    :param subparsers: Object returned by
        :meth:`argparse.ArgumentParser.add_subparsers`.
    :param name: Subcommand name.
    :param default_help: Whether to show argument defaults in help output.
    :param kwargs: Passed to :meth:`argparse._SubParsersAction.add_parser`.
    :return: The new parser.
    """
    kwargs.setdefault(
        "formatter_class",
        argparse.ArgumentDefaultsHelpFormatter
        if default_help
        else argparse.HelpFormatter,
    )
    return subparsers.add_parser(name, **kwargs)


class _LazyHandler:
    """Callable that imports a command implementation on first call."""

    def __init__(self, target: str):
        self.target = target

    def __call__(self, arguments: argparse.Namespace) -> Any:
        module_name, _, func_name = self.target.partition(":")
        func = getattr(importlib.import_module(module_name), func_name)
        return func(arguments)

    def __repr__(self) -> str:
        return f"lazy_handler({self.target!r})"


def lazy_handler(target: str) -> Callable[[argparse.Namespace], Any]:
    """Return a command handler that imports its implementation when called.

    Use with ``parser.set_defaults(func=lazy_handler(...))`` so that heavy
    dependencies are not imported when the CLI starts.

    :param target: ``"module:function"``, where the function takes the parsed
        arguments.
    :return: Handler that imports and calls `target`.

    :example:

        .. code-block:: python

            parser.set_defaults(func=lazy_handler("scallops.cli.stitch:run_stitch"))
    """
    if ":" not in target:
        raise ValueError(f"Expected 'module:function', got {target!r}")
    return _LazyHandler(target)


def images_arg(parser: ArgContainer) -> None:
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


def no_version_arg(parser: ArgContainer) -> None:
    """Add the --no-version argument to a parser."""
    parser.add_argument(
        "--no-version",
        action="store_true",
        help="Do not store command line arguments and scallops version in "
        "output metadata.",
    )


def image_pattern_arg(parser: ArgContainer) -> None:
    """Add the image pattern argument to a parser."""
    parser.add_argument(
        "--image-pattern", help="Pattern to extract metadata from file names."
    )


def groupby_arg(parser: ArgContainer) -> None:
    """Add the groupby argument to a parser."""
    parser.add_argument("-g", "--groupby", nargs="*", help="Keys to group images.")


def expected_images_arg(parser: ArgContainer) -> None:
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Validate that the specified number of images are provided.",
    )


def subset_arg(parser: ArgContainer) -> None:
    """Add the subset argument to a parser."""
    parser.add_argument(
        "-s", "--subset", nargs="*", help="Subset of images to include."
    )


def force_arg(parser: ArgContainer) -> None:
    """Add the '--force' argument to the parser to enable overwriting existing output.

    :param parser: The argument parser to which the '--force' argument is added.
    """
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output",
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
    if len(image_spacing) != 2:
        raise argparse.ArgumentTypeError("Image spacing must contain 'y,x' values")
    return float(image_spacing[0]), float(image_spacing[1])


def _z_index_type(s: str) -> int | Literal["max"]:
    """Parse and validate the z-index argument.

    This function ensures that the input is a non-negative integer 'max'

    :param s: The input string representing a z-index.
    :return: The validated z-index as an integer or a string if == 'max'
    :raises argparse.ArgumentTypeError: If the input is not a non-negative integer and != 'max'

    .. example::

       >>> _z_index_type("max")
       'max'
       >>> _z_index_type("3")
       3
       >>> _z_index_type("-1")
       Traceback (most recent call last):
           ...
       argparse.ArgumentTypeError: z-index must be >=0
    """

    s = s.strip()
    if s == "max":
        return "max"
    try:
        value = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "z-index must be either a non-negative integer or `max`"
        )
    if value < 0:
        raise argparse.ArgumentTypeError("z-index must be >=0")
    return value


def _label_args_type(s: str) -> zarr.Group:
    import zarr

    # argparse only reports ArgumentTypeError/ValueError/TypeError as usage errors
    try:
        label_root = zarr.open(s, mode="r")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Unable to open {s}: {e}") from e
    labels_group = label_root.get("labels")
    if labels_group is None:
        raise argparse.ArgumentTypeError(f"Labels group not found for {s}")
    return labels_group


def z_index_arg(parser: ArgContainer) -> None:
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


def z_index_tile_arg(parser: ArgContainer) -> None:
    parser.add_argument(
        "--z-index",
        help="Either `max`, `focus`, z-index (0-based), or a path to a Parquet file "
        "containing columns `key` and `z_index`. `focus` selects the best "
        "z-index using the slope of the image log-log power spectrum.",
        default="max",
        type=_z_index_tile_type,
    )


def t_index_arg(parser: ArgContainer) -> None:
    parser.add_argument(
        "--t-index",
        dest="t_index",
        default=None,
        type=int,
        help="Time index (0-based) to select from images with a time dimension. "
        "Required when images have more than one timepoint (e.g. live imaging data).",
    )


def labels_arg(parser: ArgContainer, required: bool = True) -> None:
    parser.add_argument(
        "--labels",
        required=required,
        type=_label_args_type,
        nargs="+",
        help="One or more paths to zarr stores containing a 'labels' group. If the same "
        "label name appears in more than one store, the store listed last is used.",
    )


def output_dir_arg(parser: ArgContainer) -> None:
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


def dask_client_arg(parser: ArgContainer, value=None) -> None:
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


def dask_cluster_arg(parser: ArgContainer) -> None:
    """Add the Dask cluster parameters.

    :param parser: The argument parser to which the Dask client argument is added.
    """
    parser.add_argument(
        "--dask-cluster",
        type=str,
        help="JSON URL or inline JSON containing dask cluster parameters.",
    )


def barcodes_arg(parser: ArgContainer, required: bool = True) -> None:
    """Add the barcodes file argument to a parser.

    :param parser: The argument parser to which the barcodes file argument is added.
    """
    parser.add_argument(
        "--barcodes",
        dest="barcodes",
        required=required,
        help="Path to the barcode CSV file containing a column named 'barcode'.",
    )
