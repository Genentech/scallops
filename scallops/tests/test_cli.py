import argparse
import importlib

import pytest

from scallops.__main__ import create_parsers
from scallops.cli.args import _LazyHandler


def _commands(
    parser: argparse.ArgumentParser, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    """Return ``(command path, parser)`` for every command that has no subcommands."""
    subparser_actions = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ]
    if not subparser_actions:
        return [(path, parser)]
    commands = []
    for action in subparser_actions:
        for name, subparser in action.choices.items():
            commands += _commands(subparser, path + (name,))
    return commands


_COMMANDS = _commands(create_parsers())


@pytest.mark.parametrize(
    "parser", [p for _, p in _COMMANDS], ids=[" ".join(c) for c, _ in _COMMANDS]
)
def test_command_handler_resolves(parser):
    # lazy_handler targets are strings, so a typo or a renamed implementation function
    # would otherwise only fail when the command runs
    func = parser.get_default("func")
    assert callable(func), "command does not set a handler with set_defaults(func=...)"
    if isinstance(func, _LazyHandler):
        module_name, _, func_name = func.target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, func_name, None)), (
            f"{func.target} is not a function"
        )
