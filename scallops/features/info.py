"""Plain-text feature descriptions for ``scallops features --help``.

The descriptions live in ``features.rst`` (also included in the Sphinx docs); the
reStructuredText markup is stripped before printing.
"""

import re
from importlib.resources import files


def _rst_to_text(rst: str) -> str:
    """Strip the reStructuredText markup used in features.rst.

    Links keep only their text, inline literals and emphasis lose their markers,
    auto-numbered lists (``#.``) are numbered, reference targets are removed, and
    heading underlines are resized to their titles.

    >>> print(
    ...     _rst_to_text(
    ...         "Title\\n^^^^^^^^^^\\n.. _target:\\n\\n"
    ...         "* `intensity <https://example.org>`_\\n"
    ...         "    #. c: ``Channel`` index.\\n"
    ...         "    #. bins: *Number* of bins.\\n"
    ...         "* spots\\n"
    ...         "    Counts spots using U-FISH_.\\n"
    ...         "    #. c: Use `*` for all channels, e.g. `intensity_*`.\\n"
    ...         ".. _U-FISH: https://example.org"
    ...     )
    ... )
    Title
    ^^^^^
    <BLANKLINE>
    * intensity
        1. c: Channel index.
        2. bins: Number of bins.
    * spots
        Counts spots using U-FISH.
        1. c: Use * for all channels, e.g. intensity_*.
    """
    lines: list[str] = []
    item = 0
    for line in rst.splitlines():
        if line.startswith(".. _"):
            continue
        if re.fullmatch(r"([-=^~*])\1+", line) and lines and lines[-1].strip():
            line = line[0] * len(lines[-1])
        elif m := re.match(r"(\s*)#\. ", line):
            item += 1
            line = f"{m.group(1)}{item}. {line[m.end() :]}"
        elif line.strip() and not line[0].isspace():
            # a new bullet, heading, or paragraph ends the current numbered list
            item = 0
        line = re.sub(r"(?<![\w`*])\*([^\s*][^*]*?)\*(?![\w*])", r"\1", line)
        line = re.sub(r"`([^`<]+?)\s*<[^>]+>`_", r"\1", line)
        line = re.sub(r"``(.+?)``", r"\1", line)
        line = re.sub(r"`([^`]+)`_{0,2}", r"\1", line)
        # named references (U-FISH_); trailing "_" before whitespace or punctuation
        line = re.sub(
            r"(?<![\w`])([A-Za-z][\w.-]*?)_{1,2}(?=[\s.,;:!?)'\"]|$)", r"\1", line
        )
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def help_text() -> str:
    """Return the available feature descriptions as plain text.

    :return: Contents of features.rst with the markup stripped.
    """
    rst = files("scallops.features").joinpath("features.rst").read_text("utf-8")
    return _rst_to_text(rst)
