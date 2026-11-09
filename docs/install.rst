************
Installation
************


PyPI Release
=========================
SCALLOPS requires Python 3.12 or newer. Install it using your favorite environment manager::

    pip install scallops

Optional features are available as extras:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Adds
   * - ``cellpose``
     - Cellpose segmentation (``scallops segment ... --method cellpose``)
   * - ``dask-ml``
     - Distributed PCA with dask-ml
   * - ``dialout``
     - Dial-out PCR analysis (``scallops dialout``)
   * - ``napari``
     - Viewing images in napari
   * - ``ufish``
     - U-FISH spot detection (the ``spots`` feature in ``scallops features``)

For example::

    pip install "scallops[cellpose,napari]"


Developer Instructions
========================


1. Clone the repository::

    git clone https://github.com/Genentech/scallops.git

2. Change to the scallops directory::

    cd scallops

3. Create a new environment using venv, uv, mamba, conda or your preferred tool::

    mamba create --name scallops python=3.12

4. Activate the environment::

    mamba activate scallops

5. Install SCALLOPS in editable mode with the test and development tools::

    pip install -r requirements.txt -e ".[test,dev]"

6. Install the pre-commit hooks (Ruff and mypy)::

    pre-commit install

7. Run the tests. Test data is downloaded automatically on the first run::

    pytest -n auto

Installing in editable mode compiles the Cython extension
``scallops.segmentation._propagate``. If you change ``_propagate.pyx`` or ``heap.pxd``,
run ``pip install -e .`` again to rebuild it.

Docker
======

Pre-built images are published to the GitHub Container Registry:

* each release tag publishes ``:X.Y.Z``, ``:vX.Y.Z``, ``:X.Y``, ``:X`` and ``:latest``
* each push to ``main`` publishes ``:main``
* every image is also tagged with its full commit SHA (``:sha-<commit>``)

::

    docker pull ghcr.io/genentech/scallops:latest

Run a command::

    docker run --rm ghcr.io/genentech/scallops:latest scallops --help

Mount local data into the container::

    docker run --rm -v /path/to/data:/data ghcr.io/genentech/scallops:latest \
        scallops <command> ...

Building locally
----------------

Build the CPU image from the repository root::

    docker build --platform linux/amd64 -t scallops .

The version is computed from the git tags by ``setuptools_scm``, so build from a git
checkout (the ``.git`` directory must be in the build context).

The Dockerfile accepts these build arguments:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Argument
     - Default
     - Description
   * - ``BASE_IMAGE``
     - ``python:3.13-slim-bookworm``
     - Base image; must provide Python 3.12 or newer
   * - ``TORCH``
     - ``1``
     - ``0`` skips installing PyTorch (Cellpose and U-FISH still install it when enabled)
   * - ``TORCH_CUDA``
     - ``0``
     - PyTorch wheel index tag, for example ``cu126``; ``0`` installs the CPU wheel
   * - ``TF_CUDA``
     - ``0``
     - ``1`` installs ``tensorflow[and-cuda]``
   * - ``RAPIDS_VERSION``, ``RAPIDS_CUDA``
     - empty
     - Install RAPIDS (cuDF, cuML, dask-cudf), for example ``26.6.0`` and ``cu12``
   * - ``UFISH``
     - ``1``
     - ``0`` skips U-FISH
   * - ``CELLPOSE_VERSION``
     - ``3.1.1.2``
     - Cellpose version; empty skips Cellpose

GPU support
-----------

All CUDA libraries are installed from pip wheels, so no ``nvidia/cuda`` base image is
needed. The default image is CPU only:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Library
     - Used by
     - GPU build argument
   * - PyTorch
     - Cellpose segmentation, U-FISH spot detection
     - ``TORCH_CUDA=cu126``
   * - TensorFlow
     - StarDist segmentation
     - ``TF_CUDA=1``
   * - RAPIDS
     - GPU PCA (cuML)
     - ``RAPIDS_VERSION=26.6.0 RAPIDS_CUDA=cu12``; RAPIDS must be a
       `current RAPIDS release`_ and 25.12 or newer

Build a GPU image::

    docker build --platform linux/amd64 \
        --build-arg TORCH_CUDA=cu126 \
        --build-arg TF_CUDA=1 \
        --build-arg RAPIDS_VERSION=26.6.0 --build-arg RAPIDS_CUDA=cu12 \
        -t scallops-gpu .

At runtime, pass ``--gpus all`` (Docker) or the equivalent Podman flag and
ensure the `NVIDIA Container Toolkit`_ is installed on the host::

    docker run --gpus all --rm scallops-gpu scallops segment ...

.. note::

   NVIDIA drivers must be installed on the host — the container image itself
   does not include drivers.  All CUDA runtime libraries (``nvidia-cuda-runtime``,
   ``nvidia-cudnn``, etc.) are installed as pip wheels inside the image and do
   not require a matching system CUDA toolkit.

.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
.. _current RAPIDS release: https://docs.rapids.ai/install
