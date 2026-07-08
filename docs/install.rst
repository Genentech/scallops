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
base.  **PyTorch** (used by cellpose and U-FISH) is installed from PyPI and
ships CUDA-capable wheels — it will automatically use an NVIDIA GPU at runtime
if one is available.  **TensorFlow** runs on CPU only in the default image;
this affects the ``segment`` command when using the StarDist backend, which
relies on TensorFlow for model inference.

To enable GPU acceleration for TensorFlow as well, build with the GPU base
image::

    make -f docker.mk docker TF_VERSION=2.21.0-gpu

At runtime, pass ``--gpus all`` (Docker) or the equivalent Podman flag and
ensure the `NVIDIA Container Toolkit`_ is installed on the host::

    docker run --gpus all --rm ghcr.io/genentech/scallops:latest scallops ...

.. note::

   GPU containers require NVIDIA drivers on the host machine.  The image
   itself does not need to change — the same image works on CPU-only and
   GPU-equipped hosts; PyTorch selects the appropriate backend automatically.

.. _Mamba: https://mamba.readthedocs.io/en/latest/installation.html
.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
