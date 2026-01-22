<h1 align="center">
<img src="https://raw.githubusercontent.com/Genentech/scallops/62db13112dee13dc228bd2a458ada5a7a973d2bf/docs/_static/scallopsLogo.png" width="150" alt="logo">
</h1><br>

# SCALLOPS

[![PyPI version](https://badge.fury.io/py/scallops.svg)](https://badge.fury.io/py/scallops)
[![Python Versions](https://img.shields.io/pypi/pyversions/scallops.svg)](https://pypi.org/project/scallops/)
[![License](https://img.shields.io/pypi/l/scallops.svg)](https://raw.githubusercontent.com/Genentech/scallops/refs/heads/main/LICENSE)

## Description
SCALLOPS (Scalable Library for Optical Pooled Screens) is a comprehensive Python package designed to
streamline and scale the analysis of Optical Pooled Screens (OPS) for biological data. With a focus on
handling large-scale, high-throughput screening data, SCALLOPS provides tools for efficiently processing,
analyzing, and interpreting OPS data, leveraging modern distributed computing frameworks like Dask.

## Installation

### Option 1: Install from PyPI (Recommended)
For most users, the easiest way to install SCALLOPS is via pip. This will install the pre-compiled binary wheels for your operating system (Linux, Windows, or macOS).

```bash
pip install scallops

```

*Note: SCALLOPS requires Python 3.11 or newer.*

### Option 2: Install from Source (For Development)

If you wish to contribute to the codebase or need the latest unreleased changes:

1. Clone the repository and change to the scallops directory:
```bash
git clone [https://github.com/Genentech/scallops.git](https://github.com/Genentech/scallops.git)
cd scallops

```


2. Install SCALLOPS in editable mode with dependencies:
```bash
pip install -r requirements.txt -e .

```



## Main Focus Areas:

* **High-Throughput Data Processing**: SCALLOPS is built to manage massive datasets typical of OPS
experiments, allowing users to efficiently process and analyze data across multiple scales.
* **Scalability and Performance**: The package is optimized for both local and cloud-based distributed
environments, making it ideal for scaling to large datasets without compromising performance.

### Modular Workflows:

SCALLOPS provides an end-to-end WDL workflow that can be customized, allowing users to tailor analyses to
their specific experimental needs.

## Key Features:

* **Efficient Data Handling**: SCALLOPS utilizes advanced memory management and lazy evaluation
techniques, which minimize resource usage while handling large datasets.
* **Command-Line Interface (CLI)**: Automates batch processing and simplifies integration into larger
pipelines.
* **Customizable Outputs**: The package generates versatile outputs, including data visualizations and
summary statistics, which can be integrated into downstream analyses.
* **Notebook Examples**: SCALLOPS includes practical Jupyter notebooks that walk users through typical
workflows, making it easy to get started with real-world datasets.
* **Custom Features**: Advanced users can extend SCALLOPS with their own custom functions and workflows,
ensuring the package can grow with the complexity of the data.
* **Comprehensive API**: SCALLOPS provides a rich API that exposes all the package functionalities,
allowing users to integrate it directly into their own Python scripts and workflows. This makes
SCALLOPS highly adaptable, enabling users to build fully customized data pipelines and analyses
tailored to their unique experimental needs.

## Typical Use Cases:

* **Large-Scale Screening Projects**: SCALLOPS is designed for handling the immense data loads of
genome-wide OPS projects, helping users efficiently identify and quantify biological perturbations.
* **Data-Driven Insights**: SCALLOPS facilitates the discovery of patterns and trends in OPS data,
helping users extract and interpret complex biological systems' data.

## Contributing to SCALLOPS

We welcome all forms of contributions, including bug reports, documentation improvements, and feature
enhancements.
