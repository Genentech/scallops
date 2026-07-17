# syntax=docker/dockerfile:1
# SCM_VERSION is passed by:
#   - CI:      .github/workflows/docker.yml  (resolved via python -m setuptools_scm)
#   - locally: docker.mk                     (make -f docker.mk docker / docker-gpu)
# Falls back to 0.0.0+unknown when building directly with docker/podman build .
#
# GPU build (TF-GPU + RAPIDS):
#   make -f docker.mk docker-gpu RAPIDS_VERSION=25.06
# CPU build (default):
#   make -f docker.mk docker
ARG TF_VERSION=2.21.0
ARG SCM_VERSION=0.0.0+unknown
# RAPIDS: required when IS_GPU=1, ignored when IS_GPU=0.
# See https://docs.rapids.ai/install for current release versions.
ARG RAPIDS_VERSION=
ARG RAPIDS_CUDA=cu12

FROM tensorflow/tensorflow:${TF_VERSION}
ARG SCM_VERSION
ARG RAPIDS_VERSION
ARG RAPIDS_CUDA
# IS_GPU is derived from TF_VERSION by docker.mk (1 when TF_VERSION contains
# -gpu, 0 otherwise). Docker ENV cannot evaluate shell expressions, so ARG is
# the necessary bridge: docker.mk computes it, ARG captures it, ENV persists it.
# Default 0 (CPU) when building directly without docker.mk.
ARG IS_GPU=0
ENV IS_GPU=${IS_GPU}

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# CPU builds: remove NVIDIA and deadsnakes apt repos — they fail SSL
# verification on restricted networks and are unused (CUDA comes from PyPI
# wheels, not apt).
# GPU builds: keep the NVIDIA repos (the GPU base image uses them) but still
# remove deadsnakes which is never needed.
# build-essential: needed for mahotas/centrosome (requirements.txt) and Cython.
# git: needed to clone ufish from its pinned tag.
RUN if [ "$IS_GPU" = "0" ]; then \
      rm -f /etc/apt/sources.list.d/cuda.list \
            /etc/apt/sources.list.d/nvidia-ml.list \
            /etc/apt/trusted.gpg.d/cuda-keyring.gpg; \
    fi && \
    rm -f /etc/apt/sources.list.d/deadsnakes*.list && \
    apt-get update -qq && \
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

# ufish: git-pinned tag, rarely bumped
COPY requirements.ufish.txt ./
RUN uv pip install -r requirements.ufish.txt

# cellpose 3.x: declares numpy<2.1 but is runtime-compatible with numpy 2.x;
# separate step so uv resolves against already-installed numpy
COPY requirements.cellpose.txt ./
RUN uv pip install -r requirements.cellpose.txt

# extra optional deps: only change when this Dockerfile changes
# napari/pytest excluded — napari is a GUI tool (useless headless) and test
# deps have no place in a production image
RUN uv pip install pysam dask-ml miniwdl

# core deps: TF already in base image so strip it to avoid re-downloading
COPY requirements.txt ./
RUN grep -v '^tensorflow' requirements.txt | uv pip install -r /dev/stdin

# RAPIDS: GPU-accelerated drop-in replacements for pandas (cuDF), scikit-learn
# (cuML), and dask (dask-cudf). Only installed when IS_GPU=1.
# For CPU builds this step is a no-op.
RUN if [ "$IS_GPU" = "1" ]; then \
      if [ -z "${RAPIDS_VERSION}" ]; then \
        echo "ERROR: RAPIDS_VERSION is required for GPU builds." \
             "Pass --build-arg RAPIDS_VERSION=<version>" \
             "(see https://docs.rapids.ai/install)" && exit 1; \
      fi && \
      pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        cudf-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        cuml-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        dask-cudf-${RAPIDS_CUDA}==${RAPIDS_VERSION}; \
    fi

# --no-deps: all deps already installed above; avoids re-downloading tensorflow.
# SETUPTOOLS_SCM_PRETEND_VERSION (set via ENV above) ensures setuptools_scm
# writes the correct version into _version.py and the wheel metadata.
# PYTHON_VERSION is inferred dynamically from the interpreter so it always
# matches whatever Python the TF base image ships — no ARG to keep in sync.
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
