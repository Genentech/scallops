# Examples:
# CPU build (default):
#   docker build --arch amd64 -t scallops  .
# GPU build:
#   docker build --arch amd64 --build-arg RAPIDS_VERSION=26.6.0 --build-arg RAPIDS_CUDA=cu12 \
#   --build-arg TF_CUDA=1 --build-arg TORCH_CUDA=cu126 -t scallops-gpu  .
# Custom base image:
#   docker build --arch amd64 --build-arg BASE_IMAGE=python:3.13-slim-bookworm -t scallops-custom  .

ARG BASE_IMAGE=python:3.13-slim-bookworm

# numpy pin from requirements.txt for the cellpose override; copying only this line keeps
# the cellpose layer cached until the numpy pin changes. grep fails the build if no pin
FROM ${BASE_IMAGE} AS numpy-pin
COPY requirements.txt /tmp/
RUN grep '^numpy==' /tmp/requirements.txt > /numpy-override.txt

FROM ${BASE_IMAGE}
COPY --from=docker.io/astral/uv:0.11.30 /uv /uvx /bin/
# Delete the PEP 668 marker file
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq --no-install-recommends -y \
    build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR /build

# install torch, doesn't actually mean "no torch" while ufish or cellpose are enabled.
ARG TORCH="1"
# CUDA wheel tag of the PyTorch index (e.g. cu126), or 0 for CPU
ARG TORCH_CUDA="0"
RUN if [ "${TORCH}" = "1" ]; then \
      if [ "${TORCH_CUDA}" != "0" ]; then \
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}; \
      else \
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      fi \
    fi

# ordered from least to most frequently changed so pin bumps reuse the large layers above

ARG TF_VERSION=2.21.0
ARG TF_CUDA="0"
RUN if [ "${TF_CUDA}" = "1" ]; then \
      uv pip install tensorflow[and-cuda]==${TF_VERSION}; \
    else \
      uv pip install tensorflow==${TF_VERSION}; \
    fi

# 26.6.0
ARG RAPIDS_VERSION=""
# cu12
ARG RAPIDS_CUDA=""

RUN if [ "${RAPIDS_VERSION}" != "" ]; then \
      uv pip install \
        cudf-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        cuml-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        dask-cudf-${RAPIDS_CUDA}==${RAPIDS_VERSION} \
        --extra-index-url https://pypi.nvidia.com; \
    fi

# ufish
COPY requirements.ufish.txt ./
# install ufish?
ARG UFISH="1"
RUN if [ "${UFISH}" = "1" ]; then uv pip install -r requirements.ufish.txt; fi

# extra optional deps
ARG DASK_ML_VERSION="2025.1.0"
RUN uv pip install dask-ml==${DASK_ML_VERSION}

# cellpose declares numpy<2.1 but is runtime-compatible with numpy 2.x;
ARG CELLPOSE_VERSION="3.1.1.2"
COPY --from=numpy-pin /numpy-override.txt requirements.cellpose.override.txt
RUN if [ "${CELLPOSE_VERSION}" != "" ]; then uv pip install --overrides requirements.cellpose.override.txt cellpose==${CELLPOSE_VERSION}; fi

# core deps last so their pins win: each uv pip install can change already installed versions.
# tensorflow is installed in a prior step, so strip it from requirements.txt.

COPY requirements.txt ./
RUN grep -v '^tensorflow' requirements.txt > requirements.no-tf.txt && \
    uv pip install -r requirements.no-tf.txt

COPY . .

# requirements.txt pins every pyproject.toml dependency, so only build and install scallops
RUN uv pip install --no-deps . && rm -rf /build

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1
