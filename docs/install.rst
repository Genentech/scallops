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

The default image uses ``tensorflow/tensorflow:2.21.0`` (CPU variant) as its
base.  The table below summarises which libraries use the GPU and which
commands are affected:

.. list-table::
   :header-rows: 1
   :widths: 20 20 35 25

   * - Library
     - GPU in default image
     - Affected commands
     - Notes
   * - PyTorch
     - **Yes** (auto-detected)
     - ``segment`` (cellpose backend), ``dialout`` (U-FISH)
     - PyPI wheels are CUDA-capable; GPU is used automatically when available
   * - TensorFlow
     - No (CPU only)
     - ``segment`` (StarDist backend)
     - Requires the GPU image (see below)
   * - RAPIDS
     - No (CPU only)
     - All commands using pandas / scikit-learn / dask internally
     - Requires the GPU image (see below)

**GPU image** — enables TensorFlow GPU and installs `RAPIDS`_ (cuDF, cuML,
dask-cudf), which accelerate the data-science operations used throughout the
pipeline.  ``RAPIDS_VERSION`` must match a `current RAPIDS release`_::

    make -f docker.mk docker-gpu RAPIDS_VERSION=25.06

The build automatically sets the ``IS_GPU`` environment variable to ``1``
inside the container.  You can inspect it at runtime to confirm GPU support
was compiled in::

    docker run --rm ghcr.io/genentech/scallops:latest printenv IS_GPU

At runtime, pass ``--gpus all`` (Docker) or the equivalent Podman flag and
ensure the `NVIDIA Container Toolkit`_ is installed on the host::

    docker run --gpus all --rm <gpu-image> scallops segment ...

.. note::

   NVIDIA drivers must be installed on the host — the container image itself
   does not include drivers.  PyTorch selects the GPU backend automatically;
   TensorFlow and RAPIDS require the GPU image built with
   ``TF_VERSION=2.21.0-gpu``.

.. _Mamba: https://mamba.readthedocs.io/en/latest/installation.html
.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
.. _RAPIDS: https://rapids.ai
.. _current RAPIDS release: https://docs.rapids.ai/install
