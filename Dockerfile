# Examples:
# CPU build (default):
#   docker build --arch amd64 -t scallops  .
# GPU build:
#   docker build --arch amd64 --build-arg RAPIDS_VERSION=26.6.0 --build-arg RAPIDS_CUDA=cu12 \
#   --build-arg TF_CUDA=1 --build-arg TORCH_CUDA=cu126 -t scallops-gpu  .
# Custom base image:
#   docker build --arch amd64 --build-arg BASE_IMAGE=python:3.13-slim-bookworm -t scallops-custom  .

ARG BASE_IMAGE=python:3.12-slim-bookworm
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

# install torch?
ARG TORCH="1"
# cu126
ARG TORCH_CUDA="0"
RUN if [ "${TORCH}" = "1" ]; then \
      if [ "${TORCH_CUDA}" = "1" ]; then \
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}; \
      else \
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      fi \
    fi

# ufish: git-pinned tag, rarely bumped
COPY requirements.ufish.txt ./
# install ufish?
ARG UFISH="1"
RUN if [ "${UFISH}" = "1" ]; then uv pip install -r requirements.ufish.txt; fi

# cellpose 3.x: declares numpy<2.1 but is runtime-compatible with numpy 2.x;
ARG CELLPOSE_VERSION="3.1.1.2"
RUN if [ "${CELLPOSE_VERSION}" != "" ]]; then uv pip install cellpose==${CELLPOSE_VERSION}; fi

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

ARG TF_VERSION=2.21.0
ARG TF_CUDA="0"
RUN if [ "${TF_CUDA}" = "1" ]; then \
      uv pip install tensorflow[and-cuda]==${TF_VERSION}; \
    else \
      uv pip install tensorflow==${TF_VERSION}; \
    fi
# core deps: tensorflow is installed in prior step, so strip it from requirements.txt
COPY requirements.txt ./
RUN grep -v '^tensorflow' requirements.txt | uv pip install -r /dev/stdin

# extra optional deps
RUN uv pip install dask-ml

COPY . .

RUN uv pip install --no-cache-dir .
RUN  rm -rf /build

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1
