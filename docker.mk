# docker.mk — local mirror of the CI/CD Docker workflow (.github/workflows/docker.yml)
#
# The CI workflow (docker/build-push-action) does the following that this file replicates:
#   1. Resolves the package version via setuptools_scm and passes it as SCM_VERSION build-arg
#   2. Attaches OCI standard labels (revision, source, version, created, …)
#   3. Tags the image with the semver version AND a short git-SHA tag
#   4. Uses docker buildx (not plain docker build) for consistency with CI
#
# What CI does that this file intentionally does NOT replicate:
#   - GitHub Actions layer cache (type=gha) — use a local registry cache if needed
#   - Automatic push to ghcr.io — use `make -f docker.mk docker-push` explicitly
#
# Base image is always python:$(PYTHON_VERSION)-slim-bookworm for both CPU and GPU.
# CUDA is installed entirely from pip wheels on GPU builds — no nvidia/cuda base needed.
# A single CUDA_VERSION knob drives the RAPIDS package suffix and PyTorch wheel index.
#
# Usage:
#   make -f docker.mk docker                                    # CPU image (default)
#   make -f docker.mk docker PYTHON_VERSION=3.11                # CPU image, Python 3.11
#   make -f docker.mk docker-gpu RAPIDS_VERSION=26.6.0           # GPU image + RAPIDS
#   make -f docker.mk docker-gpu RAPIDS_VERSION=26.6.0 \
#                               CUDA_VERSION=12.4               # GPU with specific CUDA
#   make -f docker.mk docker-push                               # build CPU + push to GHCR

# ── configurable knobs ────────────────────────────────────────────────────────
REGISTRY       ?= ghcr.io/genentech
IMAGE          ?= scallops
TF_VERSION     ?= 2.21.0
PYTHON_VERSION ?= 3.12
# RAPIDS_VERSION: required for GPU builds, ignored for CPU builds.
# See https://docs.rapids.ai/install for current release versions.
# Minimum compatible with scikit-learn 1.9 (requirements.txt): 25.12.
# Versions 25.06–25.10 use removed sklearn private APIs and fail to import cuml.
# All compatible versions resolve to the same CUDA runtime as TF 2.21 (cu12==12.9.79,
# cudnn==9.24); TF's pin is the most constrained and RAPIDS defers to it.
RAPIDS_VERSION ?=
# CUDA_VERSION: GPU only. Drives the RAPIDS package suffix and PyTorch wheel
# index URL. Does NOT need to exactly match the host driver — any CUDA 12.x
# pip wheels work with a CUDA 12.x driver (backward compatible within major).
# PyTorch ships wheels for cu121/cu124/cu126/cu128; if the derived TORCH_CUDA_TAG
# doesn't exist, override it: make docker-gpu CUDA_VERSION=12.5 TORCH_CUDA_TAG=cu124
CUDA_VERSION   ?= 12.6

# ── derived GPU flags ─────────────────────────────────────────────────────────
# IS_GPU is set to 1 by the docker-gpu target; default 0 (CPU).
# No longer derived from TF_VERSION — CUDA is installed from pip wheels, so
# TF_VERSION no longer carries a -gpu suffix.
IS_GPU         := 0

CUDA_MAJOR     := $(firstword $(subst ., ,$(CUDA_VERSION)))
CUDA_MINOR     := $(word 2,$(subst ., ,$(CUDA_VERSION)))
RAPIDS_CUDA    := cu$(CUDA_MAJOR)
# Override TORCH_CUDA_TAG if PyTorch doesn't ship a wheel for this exact minor.
TORCH_CUDA_TAG ?= cu$(CUDA_MAJOR)$(CUDA_MINOR)

FULL_IMAGE     := $(REGISTRY)/$(IMAGE)

# ── uv: prefer PATH, fall back to the default install location ────────────────
UV := $(shell which uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")

# ── values computed at parse time ─────────────────────────────────────────────
GIT_SHA        := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_SHA_SHORT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_SOURCE     := $(shell git remote get-url origin 2>/dev/null \
                    | sed 's|git@github.com:|https://github.com/|; s|\.git$$||')
BUILD_DATE     := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

# ── internal helpers ──────────────────────────────────────────────────────────

define _check_prereqs
	@"$(UV)" --version >/dev/null 2>&1 \
	    || (echo "[docker.mk] uv not found at $(UV) — installing via official installer..." \
	        && curl -LsSf https://astral.sh/uv/install.sh | sh \
	        && "$(UV)" --version >/dev/null 2>&1) \
	    || (echo "[docker.mk] ERROR: uv not found and installation failed." \
	            "Install manually: https://docs.astral.sh/uv/getting-started/installation/" \
	        && exit 1)
	@docker buildx version >/dev/null 2>&1 \
	    || (echo "[docker.mk] ERROR: docker buildx required (Docker 20.10+ or the buildx plugin)" \
	        && exit 1)
endef

# Verify that TF's bundled CUDA major matches CUDA_VERSION.
# Resolves tensorflow[and-cuda] metadata (no install) and inspects the
# nvidia-cuda-runtime-cuXX package name to extract the CUDA major.
# Warns if undetermined (e.g. transient network error), errors on a mismatch.
define _check_tf_cuda
	@_tmp=$$(mktemp) \
	 && printf 'tensorflow[and-cuda]==%s\n' '$(TF_VERSION)' > "$$_tmp" \
	 && _tf_cuda=$$("$(UV)" pip compile "$$_tmp" \
	      --python-version $(PYTHON_VERSION) \
	      --quiet \
	      2>/dev/null \
	    | grep -oE 'nvidia-cuda-runtime-cu[0-9]+' \
	    | sed 's/nvidia-cuda-runtime-cu//' \
	    | head -1) \
	 && rm -f "$$_tmp" \
	 && if [ -z "$$_tf_cuda" ]; then \
	      echo "[docker.mk] WARNING: Could not determine CUDA major for TF $(TF_VERSION) — skipping mismatch check."; \
	    elif [ "$$_tf_cuda" != "$(CUDA_MAJOR)" ]; then \
	      echo "[docker.mk] ERROR: TF $(TF_VERSION) requires CUDA cu$$_tf_cuda but CUDA_VERSION=$(CUDA_VERSION) (major: $(CUDA_MAJOR))."; \
	      echo "            Choose a TF_VERSION built for CUDA $(CUDA_MAJOR), or set CUDA_VERSION to a $$_tf_cuda.x value."; \
	      exit 1; \
	    else \
	      echo "[docker.mk] TF $(TF_VERSION) CUDA major (cu$$_tf_cuda) matches CUDA_VERSION=$(CUDA_VERSION). OK."; \
	    fi
endef

# Validate that tensorflow[and-cuda] and RAPIDS resolve to a consistent set of
# nvidia-*-cu12 packages before spending time on a multi-GB image build.
# Uses uv pip compile (resolve only, no install) to detect conflicts early.
# --index-strategy unsafe-best-match is required because some RAPIDS transitive
# deps (e.g. libucx-cu12) are published by NVIDIA on both pypi.nvidia.com and
# PyPI at different patch versions; uv's default first-index-only strategy
# cannot find a compatible version without cross-index search. pip's legacy
# --extra-index-url already had this behavior implicitly.
define _check_gpu_deps
	@_tmp=$$(mktemp) \
	 && printf 'tensorflow[and-cuda]==%s\ncudf-%s==%s\ncuml-%s==%s\ndask-cudf-%s==%s\n' \
	      '$(TF_VERSION)' \
	      '$(RAPIDS_CUDA)' '$(RAPIDS_VERSION)' \
	      '$(RAPIDS_CUDA)' '$(RAPIDS_VERSION)' \
	      '$(RAPIDS_CUDA)' '$(RAPIDS_VERSION)' > "$$_tmp" \
	 && echo "[docker.mk] Validating GPU dependency compatibility (TF $(TF_VERSION) + RAPIDS $(RAPIDS_VERSION))..." \
	 && { "$(UV)" pip compile "$$_tmp" \
	        --python-version $(PYTHON_VERSION) \
	        --extra-index-url https://pypi.nvidia.com \
	        --index-strategy unsafe-best-match \
	        --quiet \
	        > /dev/null \
	      && rm -f "$$_tmp" \
	      && echo "[docker.mk] GPU dependencies: OK."; \
	    } \
	    || { rm -f "$$_tmp"; \
	         echo "[docker.mk] ERROR: TF $(TF_VERSION) and RAPIDS $(RAPIDS_VERSION) have incompatible CUDA requirements."; \
	         echo "            Adjust TF_VERSION or RAPIDS_VERSION and retry."; \
	         exit 1; }
endef

define _build
	@SCM_VERSION=$$("$(UV)" run --with setuptools_scm --no-project --quiet \
	                  python3 -m setuptools_scm) \
	 && DOCKER_TAG=$$(echo "$$SCM_VERSION" | tr '+' '-') \
	 && echo "[docker.mk] Building $(FULL_IMAGE):$$DOCKER_TAG (IS_GPU=$(IS_GPU), Python $(PYTHON_VERSION), package version $$SCM_VERSION)" \
	 && docker buildx build \
	      --build-arg SCM_VERSION=$$SCM_VERSION \
	      --build-arg TF_VERSION=$(TF_VERSION) \
	      --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	      --build-arg IS_GPU=$(IS_GPU) \
	      --build-arg RAPIDS_VERSION=$(RAPIDS_VERSION) \
	      --build-arg RAPIDS_CUDA=$(RAPIDS_CUDA) \
	      --build-arg TORCH_CUDA_TAG=$(TORCH_CUDA_TAG) \
	      --label "org.opencontainers.image.created=$(BUILD_DATE)" \
	      --label "org.opencontainers.image.revision=$(GIT_SHA)" \
	      --label "org.opencontainers.image.source=$(GIT_SOURCE)" \
	      --label "org.opencontainers.image.version=$$SCM_VERSION" \
	      --label "org.opencontainers.image.title=$(IMAGE)" \
	      --label "org.opencontainers.image.url=$(GIT_SOURCE)" \
	      -t "$(FULL_IMAGE):$$DOCKER_TAG" \
	      -t "$(FULL_IMAGE):sha-$(GIT_SHA_SHORT)" \
	      -t "$(FULL_IMAGE):latest"
endef

.PHONY: docker docker-gpu docker-push

## Build the CPU image locally (default)
docker:
	$(call _check_prereqs)
	$(call _build) .

## Build the GPU image (CUDA from pip wheels: TF[and-cuda] + RAPIDS + torch).
## RAPIDS_VERSION is required. Example:
##   make -f docker.mk docker-gpu RAPIDS_VERSION=26.6.0
docker-gpu: IS_GPU := 1
docker-gpu:
	@[ -n "$(RAPIDS_VERSION)" ] \
	    || (echo "[docker.mk] ERROR: RAPIDS_VERSION is required for GPU builds." \
	            "Example: make -f docker.mk docker-gpu RAPIDS_VERSION=26.6.0" \
	            "(see https://docs.rapids.ai/install)" && exit 1)
	$(call _check_prereqs)
	$(call _check_tf_cuda)
	$(call _check_gpu_deps)
	$(call _build) .

## Build and push to GHCR (requires `docker login ghcr.io` first)
docker-push:
	$(call _check_prereqs)
	$(call _build) --push .
