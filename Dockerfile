# syntax=docker/dockerfile:1
# SCM_VERSION is passed by:
#   - CI:      .github/workflows/docker.yml  (resolved via python -m setuptools_scm)
#   - locally: docker.mk                     (make -f docker.mk docker)
# Falls back to 0.0.0+unknown when building directly with docker/podman build .
ARG TF_VERSION=2.21.0
ARG SCM_VERSION=0.0.0+unknown

FROM --platform=linux/amd64 tensorflow/tensorflow:${TF_VERSION}
ARG SCM_VERSION

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# build-essential: needed for mahotas/centrosome (requirements.txt) and the
# Cython extension; git: needed to clone ufish from its pinned tag
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

# --no-deps: all deps already installed above; avoids re-downloading tensorflow.
# SETUPTOOLS_SCM_PRETEND_VERSION (set via ENV above) ensures setuptools_scm
# writes the correct version into _version.py and the wheel metadata.
# PYTHON_VERSION is inferred dynamically from the interpreter so it always
# matches whatever Python the TF base image ships — no ARG to keep in sync.
# || true on compileall: torch ships py312_intrinsics.py (PEP 695 syntax) that
# Python 3.11 cannot parse — that file is never imported on 3.11.
COPY . .
RUN uv pip install --no-build-isolation --no-deps . && \
    PYVER=$(python3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')") && \
    python3 -m compileall -q /usr/local/lib/python${PYVER} 2>&1 | \
      grep -v 'py312_intrinsics' || true && \
    rm -rf /build

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1
