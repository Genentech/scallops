"""This module provides the entry point for the SCALLOPS command-line interface (CLI), allowing
users to run the end-to-end workflow of the SCALLOPS package.

It sets up argument parsing using the argparse library and defines subcommands for various tasks, including:

- Dialout
- Feature extraction
- Illumination correction
- Pooling and image filtering
- Image registration
- Image segmentation
- Image stitching

Each subcommand is handled by a specific function imported from the `scallops.cli` module. The `create_parsers` function
sets up the CLI argument parsing, while the `main` function processes the command-line arguments and executes the
appropriate subcommand.

Logging is configured to provide information on the processing status.

Usage:
    python __main__.py [subcommand] [options]

If installed as a package, the CLI can be invoked using the `scallops` command. Use help to see the available
subcommands:

    scallops --help

Authors:
    - The SCALLOPS development team
"""

import argparse
import os
import sys
import time
from importlib.metadata import version

from scallops.cli import (
    dialout_main,
    features_main,
    find_objects_main,
    illumination_correction_main,
    norm_features_main,
    pooled_if_sbs_main,
    rank_features_main,
    register_main,
    segment_main,
    stitch_main,
)
from scallops.cli.arg_parser import ScallopsArgumentParser
from scallops.cli.util import _get_cli_logger


def create_parsers(default_help: bool = False) -> argparse.ArgumentParser:
    root_parser = ScallopsArgumentParser(
        prog="scallops", description=f"Version {version('scallops')}"
    )
    subparsers = root_parser.add_subparsers(help="Command help")
    pooled_if_sbs_main._create_parser(subparsers, default_help)
    features_main._create_parser(subparsers, default_help)
    find_objects_main._create_parser(subparsers, default_help)
    segment_main._create_parser(subparsers, default_help)
    illumination_correction_main._create_parser(subparsers, default_help)
    dialout_main._create_parser(subparsers, default_help)
    norm_features_main._create_parser(subparsers, default_help)
    rank_features_main._create_parser(subparsers, default_help)
    register_main._create_parser(subparsers, default_help)
    stitch_main._create_stitch_parser(subparsers, default_help)
    stitch_main._create_stitch_preview_parser(subparsers, default_help)
    return root_parser


def main():
    root_parser = create_parsers(True)
    args = root_parser.parse_args()

    if "func" not in args:
        root_parser.print_help()
    else:
        if "verbose" in args and args.verbose:
            os.environ["scallops_loglevel"] = "DEBUG"
        logger = _get_cli_logger()
        logger.info(f"Scallops version: {version('scallops')}")
        cmd = ["scallops"] + sys.argv[1:]
        logger.info(f"Scallops command:\n{' '.join(cmd)}")
        start_time = time.time()
        args.func(args)
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            f"Processing done in {int(hours):02}H:{int(minutes):02}M:{int(seconds):02}S"
        )


if __name__ == "__main__":
    main()
