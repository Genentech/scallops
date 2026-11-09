# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

SCALLOPS (Scalable Library for Optical Pooled Screens) is a Python (>=3.12) package for analyzing
Optical Pooled Screens (OPS) imaging data at scale. It is built on xarray, dask, zarr (v3 only),
scikit-image, and anndata, and ships a `scallops` CLI plus WDL workflows for Terra/miniwdl/AWS HealthOmics/Google Cloud Batch, etc..

## Repository layout

- `scallops/` — package source
  - `io.py`, `zarr_io.py` — image/experiment reading and writing (OME-TIFF, OME-Zarr, ND2 via bioio)
  - `_anndata_io.py` — scallops' anndata writers, registered in place of anndata's on `import scallops`
  - `reads.py`, `codebook.py`, `spots.py`, `_demux.py` — in situ sequencing (SBS) base calling and barcode assignment
  - `features/` — per-cell feature extraction, normalization, aggregation, ranking
  - `segmentation/` — watershed, propagation (Cython: `_propagate.pyx`), stardist, cellpose
  - `registration/`, `stitch/`, `illumination_correction.py` — image alignment and preprocessing
  - `experiment/` — `Experiment` data structure
  - `cli/` — CLI subcommands; entry point is `scallops/__main__.py`
  - `tests/` — pytest suite; test data is downloaded from GitHub release assets (see Testing)
- `wdl/` — WDL workflows/tasks (`ops_*`, `stitch_*`, `pert_map_*`, `utils.wdl`)
- `docs/` — Sphinx docs (`api.rst` drives autosummary; `release_notes.rst`; notebooks are Git LFS)

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt -e ".[test,dev]"
pre-commit install
```

Test data is stored as assets of the `test-data` GitHub release, not in git. `conftest.py`
downloads missing or outdated files before tests run into the scallops cache directory
(`SCALLOPS_CACHE_DIR`, or the platform default such as `~/.cache/scallops` or
`~/Library/Caches/scallops`); `python scripts/test_data.py download` does the same by hand.
Tests reference data via `DATA_DIR = cache_dir()` (from `scallops.datasets._registry`), never
via paths inside the repo. `scallops/datasets/registry.txt`
lists each file's sha256; `scallops.datasets` downloads from the same registry.

To add or change test data, put the file in that directory, run
`python scripts/test_data.py upload` (needs an authenticated `gh`), and commit the updated
`registry.txt`. Don't delete release assets: older commits still reference them.

Notebooks (`*.ipynb`, `docs/notebooks/img/`) are still stored with **Git LFS**
(`git lfs install && git lfs pull`).

Installing in editable mode compiles the Cython extension `scallops.segmentation._propagate`. If you
edit `_propagate.pyx` or `heap.pxd`, reinstall (`uv pip install -e .`) to rebuild it.

## Testing

```bash
pytest -n auto                              # full suite (what CI runs)
pytest scallops/tests/test_reads.py         # one file
pytest -m "not cli_e2e"                     # skip slow end-to-end CLI/WDL tests
```

- pytest runs with `--doctest-modules --strict-markers`, so docstring examples run as tests and any
  new marker must be registered under `[tool.pytest.ini_options].markers` in `pyproject.toml`.
- `xfail_strict = true`: an xfail test that passes counts as a failure.
- Use the `client` fixture in `conftest.py` for tests that need a Dask cluster.
- WDL tests (`test_wdl.py`) need `miniwdl`; they run through the local container backend
  `miniwdl_local` (`scallops/_miniwdl_local.py`, registered as a miniwdl plugin entry point),
  which runs tasks without containers. It lives in the package, not `tests/`, because tests are
  excluded from the wheel and miniwdl loads every registered backend.
- CI (`.github/workflows/test.yml`) runs on Python 3.13 with `pytest -n auto`.

## Code style

- Ruff handles linting and formatting (line length 88, target py312; rules E4/E7/E9/F plus isort).
  Run `ruff check --fix . && ruff format .`, or just `pre-commit run --all-files`, which also runs mypy.
- Pre-commit blocks commits straight to `main`, so work on a branch.
- Docstrings use Sphinx/reST field lists (`:param x:`, `:return:`) and usually include an
  `:example:` with a `.. code-block:: python`. Type-hint public functions.
- Leading underscore = private helper (for example `_regionprops`, `_run_overlap_pipeline`).
- `scallops/__init__.py` runs on every `scallops` command, so it imports only anndata, zarr,
  `_anndata_io`, and `Experiment` (about 0.8 s in total). Don't add eager imports of other
  submodules there; `import scallops.io` etc. explicitly where needed. `_anndata_io.py` may only
  import anndata and modules anndata itself imports.
- Prefer lazy, chunked computation (dask/xarray) over loading full images into memory; the library
  targets very large datasets.

## CLI conventions

Each command is split across two modules in `scallops/cli/`:

- `<name>_main.py` defines the arguments. Its `_create_parser(subparsers, default_help)` adds the
  command with `add_command(...)`, reuses shared arguments from `cli/args.py`, and sets the
  handler with `parser.set_defaults(func=lazy_handler("scallops.cli.<name>:run_..."))`.
  The handler module is imported only when the command runs.
- `<name>.py` contains the implementation. Each handler takes the parsed `argparse.Namespace`.

`cli/args.py` may only import the standard library: it is loaded on every `scallops` call, and
`scallops --help` should take under a second. Runtime helpers (dask client, logging, output
metadata) live in `cli/util.py`. Don't call `_sort_groups`; `create_parsers` sorts every
parser's argument groups.

Some commands have their own subcommands (`segment nuclei|cell|overlap`, `dialout analysis|report`),
and `stitch_main.py` adds two commands (`stitch`, `stitch-preview`).

To add a command:
1. Create both modules.
2. Add the `_main` module to `_COMMANDS` in `scallops/__main__.py`.
3. Add a section to `docs/command_line.rst` with a `.. argparse::` directive.

The WDL tasks in `wdl/` call `scallops ...` directly, so renaming or removing a CLI option breaks
them. Update the matching task when you do either.

## Documentation

- Build: `sphinx-build -E -b html docs _public -v` (writes HTML to `_public/`).
- To publish a new public function, add it to the right `autosummary` block in `docs/api.rst`.
  The per-function `docs/scallops.*.rst` stubs are generated by autosummary; don't edit them by hand.
- Add user-facing changes to `docs/release_notes.rst`.

## Releases

Versions come from git tags via `setuptools_scm` (`vX.Y.Z`). Pushing a `v*` tag builds wheels with
cibuildwheel and publishes to PyPI; pushes to `main` and tags also build the Docker image on GHCR.
