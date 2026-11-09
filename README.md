
<h1 align="center">
<img src="https://raw.githubusercontent.com/Genentech/scallops/62db13112dee13dc228bd2a458ada5a7a973d2bf/docs/_static/scallopsLogo.png" width="150" alt="logo">
</h1><br>

# SCALLOPS

[![PyPI version](https://badge.fury.io/py/scallops.svg)](https://badge.fury.io/py/scallops)
[![Python Versions](https://img.shields.io/pypi/pyversions/scallops.svg)](https://pypi.org/project/scallops/)
[![License](https://img.shields.io/pypi/l/scallops.svg)](https://github.com/Genentech/scallops/blob/main/LICENSE)

SCALLOPS (Scalable Library for Optical Pooled Screens) is a comprehensive Python package designed to
streamline and scale the analysis of Optical Pooled Screens (OPS) for biological data. With a focus on
handling large-scale, high-throughput screening data, SCALLOPS provides tools for efficiently processing,
analyzing, and interpreting OPS data, leveraging modern distributed computing frameworks like Dask.

## Documentation
The full documentation, API reference, and tutorials can be found at: https://scallops.readthedocs.io

## Getting Started

### Prerequisites
SCALLOPS requires Python 3.12 or newer.

### 1. Environment Setup (Recommended)
We recommend using **uv** for high-performance Python environment management. You will need uv installed on your system. Installation instructions can be found here: https://docs.astral.sh/uv/

To set up a virtual environment:

```bash
# Create a virtual environment with a specific Python version
uv venv --python 3.12

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Installation and Usage Options

#### Option 1: Install from PyPI (Standard)
The easiest way to install the stable version is via pip (or uv pip):

```bash
uv pip install scallops
```

Optional features are available as extras:

| Extra | Adds |
|-------|------|
| `cellpose` | Cellpose segmentation |
| `dask-ml` | Distributed PCA with dask-ml |
| `dialout` | Dial-out PCR analysis (`pysam`) |
| `napari` | Viewing images in napari |
| `ufish` | U-FISH spot detection |

```bash
uv pip install "scallops[cellpose,napari]"
```

#### Option 2: Run via Docker (GitHub Container Registry)
SCALLOPS is available as a containerized image via the GitHub Container Registry (GHCR). This is the
best option for ensuring environment consistency.

```bash
# Pull the latest image
docker pull ghcr.io/genentech/scallops:latest

# Run the CLI directly
docker run --rm ghcr.io/genentech/scallops:latest scallops --help
```

#### Option 3: Install from Source (Development)
If you wish to contribute to the codebase or need the latest unreleased changes:

1. Clone the repository:
```bash
git clone https://github.com/Genentech/scallops.git
cd scallops
```

2. Install in editable mode with the test and development tools, and set up the pre-commit hooks:
```bash
uv pip install -r requirements.txt -e ".[test,dev]"
pre-commit install
```

3. Run the tests (test data is downloaded automatically on the first run):
```bash
pytest -n auto
```

## Main Focus Areas

* **High-Throughput Data Processing**: Designed to manage massive datasets typical of OPS experiments across multiple scales.
* **Scalability and Performance**: Optimized for both local and cloud-based distributed environments using Dask.
* **Modular Workflows**: Includes customizable WDL workflows that run locally with miniwdl or in the cloud on Terra, AWS HealthOmics, and Google Cloud Batch.

## Key Features

* **Efficient Data Handling**: Advanced memory management and lazy evaluation to minimize resource usage.
* **Command-Line Interface (CLI)**: Automates batch processing for seamless pipeline integration.
* **Standard Output Formats**: Writes images as OME-Zarr, tables (reads, per-cell features) as Parquet, and aggregated results as AnnData, alongside plots and summary statistics.
* **Notebook Examples**: [Example Jupyter notebooks](https://scallops.readthedocs.io/en/latest/example_notebooks.html) walk through real-world workflows.
* **Rich API**: A comprehensive API that allows for the creation of fully customized biological data pipelines.

## Typical Use Cases

* **Large-Scale Screening**: Handling the immense data loads of genome-wide OPS projects.
* **Biological Discovery**: Identifying and quantifying biological perturbations from high-throughput imaging.

## Contributing to SCALLOPS

We welcome all forms of contributions, including bug reports, documentation improvements, and feature enhancements.

* Report bugs and request features on the [issue tracker](https://github.com/Genentech/scallops/issues).
* To contribute code, follow Option 3 above to set up a development environment, create a branch,
  and open a pull request against `main`. The pre-commit hooks run Ruff and mypy; please make sure
  `pytest -n auto` passes and add an entry to `docs/release_notes.rst` for user-facing changes.
