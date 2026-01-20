FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
SHELL ["/bin/bash", "-c"]

# Configure environment
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /software/scallops

# --- DEPENDENCY LAYER ---
COPY pyproject.toml setup.py requirements.txt requirements.cellpose.txt requirements.ufish.txt ./

ARG TF_PKG="tensorflow==2.19.0"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    set -eux; \
    # Install system dependencies
    apt-get update && \
    apt-get install --no-install-recommends -y \
      build-essential \
      git \
      ca-certificates && \
    # Create virtual environment
    uv venv /opt/venv && \
    # Filter out tensorflow from requirements
    grep -vE '^tensorflow(==|>=|<=|~=|!=)' requirements.txt > /tmp/requirements.no-tf.txt && \
    # Install Python dependencies
    uv pip install --no-cache-dir \
      -r /tmp/requirements.no-tf.txt \
      -r requirements.cellpose.txt \
      -r requirements.ufish.txt \
      "${TF_PKG}" && \
    # Cleanup
    apt-get remove -y build-essential git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/* /var/tmp/*

# --- APPLICATION LAYER ---
COPY scallops scallops/

# Install the application itself
RUN --mount=type=bind,source=.git,target=.git \
    apt-get update && \
    apt-get install --no-install-recommends -y build-essential git && \
    # Install app in editable mode into the venv
    uv pip install -e . && \
    apt-get remove -y build-essential git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
