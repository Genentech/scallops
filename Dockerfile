# syntax=docker/dockerfile:1
# SCM_VERSION is passed by:
#   - CI:      .github/workflows/docker.yml  (resolved via python -m setuptools_scm)
#   - locally: docker.mk                     (make -f docker.mk docker / docker-gpu)
# Falls back to 0.0.0+unknown when building directly with docker/podman build .
#
# Base image is always python:${PYTHON_VERSION}-slim-bookworm for both CPU and GPU.
# On GPU builds, CUDA is installed entirely from pip wheels — no nvidia/cuda base needed:
#   - tensorflow[and-cuda] brings its own CUDA runtime via nvidia-*-cu12 pip packages
#   - torch is installed from PyTorch's CUDA-specific wheel index
#   - RAPIDS is installed from NVIDIA's PyPI with cuda-runtime pip dependencies
# A single CUDA_VERSION knob in docker.mk drives RAPIDS_CUDA and TORCH_CUDA_TAG.
#
# GPU build (TF[and-cuda] + RAPIDS + torch from pip):
#   make -f docker.mk docker-gpu RAPIDS_VERSION=25.06
# CPU build (default):
#   make -f docker.mk docker
# Custom Python version:
#   make -f docker.mk docker PYTHON_VERSION=3.11

# Only PYTHON_VERSION needs to precede FROM — it is the sole ARG used in the
# FROM instruction. All other build args are declared after FROM.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm

ARG PYTHON_VERSION
ARG SCM_VERSION=0.0.0+unknown
ARG TF_VERSION=2.21.0
# IS_GPU is set to 1 by docker.mk for GPU builds, 0 for CPU (default).
ARG IS_GPU=0
# RAPIDS: required when IS_GPU=1, ignored when IS_GPU=0.
# See https://docs.rapids.ai/install for current release versions.
ARG RAPIDS_VERSION
ARG RAPIDS_CUDA=cu12
# TORCH_CUDA_TAG: derived from CUDA_VERSION in docker.mk (e.g. 12.6 → cu126).
# Override via docker.mk if PyTorch doesn't ship a wheel for that exact minor.
ARG TORCH_CUDA_TAG=cu126
ENV IS_GPU=${IS_GPU}

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# build-essential: needed for mahotas/centrosome (requirements.txt) and Cython.
# git: needed to clone ufish from its pinned tag.
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq --no-install-recommends -y \
      build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

# SETUPTOOLS_SCM_PRETEND_VERSION set as Docker ENV so it propagates into the
# PEP 517 build subprocess that uv spawns (inline shell variables do not
# reliably cross that boundary). Tested: ENV propagation works, inline does not.
ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300 \
    SETUPTOOLS_SCM_PRETEND_VERSION=${SCM_VERSION}

WORKDIR /build

# Print the Python version in use so it's visible in build logs
RUN python3 --version

# Each file is its own layer ordered most→least stable for cache efficiency.

# Pre-install torch from the correct index before ufish/cellpose so their
# dependency resolution sees the right wheel and does not pull a different one.
# GPU: CUDA wheel from PyTorch's CUDA index (bundles its own CUDA libs, no conflict
#      with the TF+RAPIDS nvidia-* packages resolved later).
# CPU: explicit cpu-only wheel — default PyPI torch is now CUDA-enabled (+cu13x)
#      which is functional but adds ~1-2 GB unnecessarily to a CPU image.
RUN if [ "$IS_GPU" = "1" ]; then \
      uv pip install torch \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}; \
    else \
      uv pip install torch \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ufish: git-pinned tag, rarely bumped
COPY requirements.ufish.txt ./
RUN uv pip install -r requirements.ufish.txt

# cellpose 3.x: declares numpy<2.1 but is runtime-compatible with numpy 2.x;
# separate step so uv resolves against already-installed packages
COPY requirements.cellpose.txt ./
RUN uv pip install -r requirements.cellpose.txt

# extra optional deps: only change when this Dockerfile changes
RUN uv pip install pysam dask-ml miniwdl

# core deps: tensorflow is installed in the GPU/CPU step below, so strip it
# here to avoid a redundant re-download from requirements.txt
COPY requirements.txt ./
RUN grep -v '^tensorflow' requirements.txt | uv pip install -r /dev/stdin

# TF + RAPIDS: installed together in a single uv pip install so the resolver
# sees all nvidia-*-cu12 constraints simultaneously and finds a consistent
# solution. Separate calls would resolve independently and risk silent
# version conflicts on shared CUDA packages.
# --index-strategy unsafe-best-match: required for RAPIDS transitive deps
# (e.g. libucx-cu12) published by NVIDIA on both pypi.nvidia.com and PyPI at
# different patch versions. This mirrors pip's legacy --extra-index-url
# behaviour which already searched all indexes.
RUN if [ "$IS_GPU" = "1" ]; then \
      if [ -z "${RAPIDS_VERSION}" ]; then \
        echo "ERROR: RAPIDS_VERSION is required for GPU builds." \
             "Pass --build-arg RAPIDS_VERSION=<version>" \
             "(see https://docs.rapids.ai/install)" && exit 1; \
      fi && \
      uv pip install \
        tensorflow[and-cuda]==${TF_VERSION} \
        cudf-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        cuml-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        dask-cudf-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match; \
    else \
      uv pip install tensorflow==${TF_VERSION}; \
    fi

# --no-deps: all deps already installed above; avoids re-downloading tensorflow.
# SETUPTOOLS_SCM_PRETEND_VERSION (set via ENV above) ensures setuptools_scm
# writes the correct version into _version.py and the wheel metadata.
# PYTHON_VERSION is inferred dynamically from the interpreter so it always
# matches whatever Python the base image ships — no ARG to keep in sync.
COPY . .
RUN [ "${SCM_VERSION}" = "0.0.0+unknown" ] && \
      echo "WARNING: SCM_VERSION not set — version will be 0.0.0+unknown. Use 'make -f docker.mk docker' to stamp the correct version." || true && \
    uv pip install --no-deps . && \
    PYVER=$(python3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')") && \
    python3 -m compileall -q /usr/local/lib/python${PYVER} >/dev/null 2>&1 || true && \
    rm -rf /build

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1
