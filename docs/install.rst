************
Installation
************


PyPI Release
=========================
Install SCALLOPS using your favorite environment manager::

    pip install scallops


Developer Instructions
========================


1. Clone the repository::

    git clone https://github.com/Genentech/scallops.git

2. Change to the scallops directory::

    cd scallops


4. Create a new environment using venv, mamba, conda or your preferred tool::

    mamba create --name scallops python=3.12


5. Activate the environment::

    mamba activate scallops


6. Install SCALLOPS::

    pip install -r requirements.txt -e .

Docker
======

Pre-built images are published to the GitHub Container Registry after every
push to ``main`` and on every release tag::

    docker pull ghcr.io/genentech/scallops:latest

Run a command::

    docker run --rm ghcr.io/genentech/scallops:latest scallops --help

Mount local data into the container::

    docker run --rm -v /path/to/data:/data ghcr.io/genentech/scallops:latest \
        scallops <command> ...

Building locally
----------------

A ``docker.mk`` file is provided that acts as the local equivalent of the CI
workflow.  It runs ``setuptools_scm`` as a preflight step to stamp the correct
version into the image, attaches OCI labels, and tags the image with the
current version::

    make -f docker.mk docker

GPU support
-----------

Both the CPU and GPU images use ``python:${PYTHON_VERSION}-slim-bookworm``
(default ``3.12``) as their base.  All CUDA libraries are installed from pip
wheels — no ``nvidia/cuda`` base image is needed.  The table below summarises
GPU availability per library:

.. list-table::
   :header-rows: 1
   :widths: 20 20 35 25

   * - Library
     - GPU in default (CPU) image
     - Affected commands
     - Notes
   * - PyTorch
     - No (CPU-only wheel)
     - ``segment`` (cellpose backend), ``dialout`` (U-FISH)
     - CPU image installs the ``cpu`` wheel explicitly; GPU image installs the
       CUDA wheel (``cu126`` by default)
   * - TensorFlow
     - No (CPU only)
     - ``segment`` (StarDist backend)
     - GPU image installs ``tensorflow[and-cuda]``, which bundles CUDA via pip
   * - RAPIDS
     - No (CPU only)
     - All commands using pandas / scikit-learn / dask internally
     - GPU image only; requires RAPIDS ≥ 25.12 for scikit-learn 1.9 compatibility

**GPU image** — installs ``tensorflow[and-cuda]``, RAPIDS (cuDF, cuML,
dask-cudf), and a CUDA-enabled PyTorch wheel.  ``RAPIDS_VERSION`` must be a
`current RAPIDS release`_ and ≥ 25.12::

    make -f docker.mk docker-gpu RAPIDS_VERSION=26.6.0

A single ``CUDA_VERSION`` knob (default ``12.6``) drives the RAPIDS package
suffix and the PyTorch wheel index.  Override it if your driver requires a
different minor version::

    make -f docker.mk docker-gpu RAPIDS_VERSION=26.6.0 CUDA_VERSION=12.4

The Python version is independently configurable for both images::

    make -f docker.mk docker PYTHON_VERSION=3.11

The build automatically sets the ``IS_GPU`` environment variable to ``1``
inside the container.  You can inspect it at runtime to confirm GPU support::

    docker run --rm <gpu-image> printenv IS_GPU

At runtime, pass ``--gpus all`` (Docker) or the equivalent Podman flag and
ensure the `NVIDIA Container Toolkit`_ is installed on the host::

    docker run --gpus all --rm <gpu-image> scallops segment ...

.. note::

   NVIDIA drivers must be installed on the host — the container image itself
   does not include drivers.  All CUDA runtime libraries (``nvidia-cuda-runtime``,
   ``nvidia-cudnn``, etc.) are installed as pip wheels inside the image and do
   not require a matching system CUDA toolkit.

.. _Mamba: https://mamba.readthedocs.io/en/latest/installation.html
.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
.. _RAPIDS: https://rapids.ai
.. _current RAPIDS release: https://docs.rapids.ai/install
