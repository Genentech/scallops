# syntax=docker/dockerfile:1
# PYTHON_VERSION must match the Python inside tensorflow/tensorflow:${TF_VERSION}
# (currently 3.11); changing TF_VERSION may require updating PYTHON_VERSION.
ARG TF_VERSION=2.21.0
ARG PYTHON_VERSION=3.11
# Override at build time for GPU: --build-arg TORCH_COMPUTE=cu124 (or cu126, cu128, etc.)
ARG TORCH_COMPUTE=cpu

FROM --platform=linux/amd64 tensorflow/tensorflow:${TF_VERSION}
ARG PYTHON_VERSION
ARG TORCH_COMPUTE

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# build-essential: needed for mahotas/centrosome (requirements.txt) and the
# Cython extension; git: needed to clone ufish from its pinned tag
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq --no-install-recommends -y \
      build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR /build

# Each file is its own layer ordered most→least stable for cache efficiency.

# Install CPU-only PyTorch before ufish/cellpose to prevent the multi-GB NVIDIA
# CUDA stack from being pulled in as a transitive dep.
# --allow-insecure-host: download-r2.pytorch.org (PyTorch CDN) is intercepted
# by SSL inspection proxies; proper fix is to inject your proxy CA cert.
RUN uv pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/${TORCH_COMPUTE} \
      --allow-insecure-host download.pytorch.org \
      --allow-insecure-host download-r2.pytorch.org

# ufish: git-pinned tag, rarely bumped
COPY requirements.ufish.txt ./
RUN uv pip install -r requirements.ufish.txt

# cellpose 3.x: declares numpy<2.1 but is runtime-compatible with numpy 2.x;
# separate step so uv resolves against already-installed numpy
COPY requirements.cellpose.txt ./
RUN uv pip install -r requirements.cellpose.txt

# extra optional deps: only change when this Dockerfile changes
RUN uv pip install pysam napari napari_ome_zarr dask-ml miniwdl pytest pytest-xdist

# core deps: TF already in base image so strip it to avoid re-downloading
COPY requirements.txt ./
RUN grep -v '^tensorflow' requirements.txt | uv pip install -r /dev/stdin

# Compile Cython extension; --no-deps skips re-resolving all dependencies
# (already installed above) which would otherwise re-download tensorflow.
# || true on compileall: torch ships py312_intrinsics.py with PEP 695 syntax
# that Python 3.11 cannot parse — that file is never imported on 3.11.
# Stash _version.py to /tmp before the broad COPY so it survives rm -rf /build.
# Fail here with a clear Docker message if it hasn't been generated yet
# (run `python -m setuptools_scm` or `pip install -e .` first).
COPY scallops/_version.py /tmp/scm_version.py
COPY . .
# --no-deps: all deps already installed above; avoids re-downloading tensorflow.
# setuptools_scm falls back to 0.0.0 without .git, so after install we patch
# both the installed _version.py and the dist-info METADATA with the real
# version read from the pre-generated /tmp/scm_version.py.
# || true on compileall: torch ships py312_intrinsics.py (PEP 695 syntax) that
# Python 3.11 cannot parse — that file is never imported on 3.11.
RUN uv pip install --no-build-isolation --no-deps . && \
    SCM_VERSION=$(python3 -c "import re; print(re.search(r\"version = '([^']+)'\", open('/tmp/scm_version.py').read()).group(1))") && \
    DIST_PKGS=/usr/local/lib/python${PYTHON_VERSION}/dist-packages && \
    cp /tmp/scm_version.py ${DIST_PKGS}/scallops/_version.py && \
    sed -i "s/^Version: .*/Version: ${SCM_VERSION}/" ${DIST_PKGS}/scallops-*.dist-info/METADATA && \
    python -m compileall -q /usr/local/lib/python${PYTHON_VERSION} 2>&1 | \
      grep -v 'py312_intrinsics' || true && \
    rm -rf /build /tmp/scm_version.py

# fontconfig required by napari/vispy at import time (font rendering)
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq --no-install-recommends -y \
      libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1
