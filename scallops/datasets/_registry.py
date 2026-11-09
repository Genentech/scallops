"""Download scallops test/example data from GitHub release assets.

Each data file is uploaded as an individual asset of the ``test-data`` GitHub release.
``registry.txt`` lists one ``<sha256> <relative path>`` per line. The asset name is
derived from the hash and file name (see :func:`asset_name`), so updated files get new
asset names and older checkouts keep downloading the files they expect.

This module only uses the standard library so it can run before any optional
dependencies are installed.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

RELEASE_TAG = "test-data"
REPO = "Genentech/scallops"
REGISTRY_PATH = Path(__file__).with_name("registry.txt")
_CHUNK = 1 << 20


def base_url() -> str:
    """Base URL of release assets, overridable with ``SCALLOPS_DATA_URL``."""
    url = os.environ.get(
        "SCALLOPS_DATA_URL",
        f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}",
    )
    return url.rstrip("/")


def asset_name(path: str, sha256: str) -> str:
    """GitHub release asset name for a registry entry.

    Asset names are flat, so only the file name is kept, prefixed by part of the hash
    to make it unique. Characters GitHub would rewrite are replaced with ``_``.
    """
    name = re.sub(r"[^A-Za-z0-9._-]", "_", path.rsplit("/", 1)[-1])
    return f"{sha256[:16]}-{name}"


def read_registry(path: Path = REGISTRY_PATH) -> dict[str, str]:
    """Return mapping of relative path to sha256."""
    registry = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sha256, rel_path = line.split(maxsplit=1)
                registry[rel_path] = sha256
    return registry


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as r:
            while chunk := r.read(_CHUNK):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != sha256:
            raise OSError(
                f"Checksum mismatch for {url}: expected {sha256}, got {h.hexdigest()}"
            )
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates files readable only by owner
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch(
    dest_dir: str | Path | None = None,
    paths: Iterable[str] | None = None,
    max_workers: int = 8,
    verbose: bool = False,
) -> Path:
    """Download registry files into ``dest_dir``, skipping files that are up to date.

    :param dest_dir: Root directory; files are written to ``dest_dir/<relative path>``.
        Defaults to :func:`cache_dir`.
    :param paths: Relative paths to fetch. Defaults to every file in the registry.
    :param max_workers: Number of parallel downloads. Checksums of existing files are
        computed with the ``ThreadPoolExecutor`` default number of threads.
    :param verbose: Print progress to stderr.
    :return: ``dest_dir`` as a Path.
    """
    dest_dir = cache_dir() if dest_dir is None else Path(dest_dir)
    registry = read_registry()
    if paths is None:
        paths = list(registry)
    else:
        paths = list(paths)
        unknown = [p for p in paths if p not in registry]
        if unknown:
            raise KeyError(f"Not in data registry: {unknown}")

    def _needs_download(p: str) -> bool:
        dest = dest_dir / p
        return not dest.exists() or file_sha256(dest) != registry[p]

    with ThreadPoolExecutor() as pool:
        missing = [p for p, m in zip(paths, pool.map(_needs_download, paths)) if m]
    if not missing:
        return dest_dir
    if verbose:
        print(
            f"Downloading {len(missing)} data file(s) from {base_url()} to {dest_dir}",
            file=sys.stderr,
        )

    def _get(p: str) -> None:
        url = f"{base_url()}/{asset_name(p, registry[p])}"
        _download(url, dest_dir / p, registry[p])
        if verbose:
            print(f"  {p}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers) as pool:
        list(pool.map(_get, missing))
    return dest_dir


def cache_dir() -> Path:
    """User cache directory for scallops data (override with ``SCALLOPS_CACHE_DIR``)."""
    if "SCALLOPS_CACHE_DIR" in os.environ:
        return Path(os.environ["SCALLOPS_CACHE_DIR"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "scallops"
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(root) / "scallops" / "Cache"
    root = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(root) / "scallops"
