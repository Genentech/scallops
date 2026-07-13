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
# Usage:
#   make -f docker.mk docker                                    # CPU image (default)
#   make -f docker.mk docker-gpu RAPIDS_VERSION=25.06           # GPU image + RAPIDS
#   make -f docker.mk docker-push                               # build CPU + push to GHCR
#   make -f docker.mk docker-push TF_VERSION=2.21.0-gpu \
#                                 RAPIDS_VERSION=25.06          # build GPU + push

# ── configurable knobs ────────────────────────────────────────────────────────
REGISTRY      ?= ghcr.io/genentech
IMAGE         ?= scallops
TF_VERSION    ?= 2.21.0
# RAPIDS_VERSION: required for GPU builds, ignored for CPU builds.
# See https://docs.rapids.ai/install for current release versions.
RAPIDS_VERSION ?=
RAPIDS_CUDA   ?= cu12

# ── derived flags ─────────────────────────────────────────────────────────────
# IS_GPU is computed from TF_VERSION so the Dockerfile never needs it passed
# explicitly by the user. Docker ENV cannot evaluate shell expressions, so
# docker.mk is the place to derive it.
IS_GPU := $(if $(findstring -gpu,$(TF_VERSION)),1,0)

FULL_IMAGE    := $(REGISTRY)/$(IMAGE)

# ── uv: prefer PATH, fall back to the default install location ────────────────
UV := $(shell which uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")

# ── values computed at parse time (none depend on setuptools_scm) ─────────────
GIT_SHA       := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_SHA_SHORT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_SOURCE    := $(shell git remote get-url origin 2>/dev/null \
                   | sed 's|git@github.com:|https://github.com/|; s|\.git$$||')
BUILD_DATE    := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

# ── internal helpers ──────────────────────────────────────────────────────────

# Installs missing prerequisites at shell execution time (not Make parse time).
# uv is used instead of pip to avoid PEP-668 "externally managed environment"
# errors on Homebrew/system Python installs.
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

# SCM_VERSION is a shell variable ($$) so it is evaluated at shell execution
# time, after _check_prereqs has ensured uv is present.
# `uv run --with setuptools_scm` creates a throwaway venv — no system Python
# pollution, no pip permission issues.
# IS_GPU and RAPIDS args are always passed; the Dockerfile ignores RAPIDS when
# IS_GPU=0 so there is no penalty for always forwarding them.
define _build
	@SCM_VERSION=$$("$(UV)" run --with setuptools_scm --no-project --quiet \
	                  python3 -m setuptools_scm) \
	 && DOCKER_TAG=$$(echo "$$SCM_VERSION" | tr '+' '-') \
	 && echo "[docker.mk] Building $(FULL_IMAGE):$$DOCKER_TAG (IS_GPU=$(IS_GPU), package version $$SCM_VERSION)" \
	 && docker buildx build \
	      --build-arg SCM_VERSION=$$SCM_VERSION \
	      --build-arg TF_VERSION=$(TF_VERSION) \
	      --build-arg IS_GPU=$(IS_GPU) \
	      --build-arg RAPIDS_VERSION=$(RAPIDS_VERSION) \
	      --build-arg RAPIDS_CUDA=$(RAPIDS_CUDA) \
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

## Build the GPU image (TF-GPU base + RAPIDS). RAPIDS_VERSION is required.
## Example: make -f docker.mk docker-gpu RAPIDS_VERSION=25.06
docker-gpu:
	@[ -n "$(RAPIDS_VERSION)" ] \
	    || (echo "[docker.mk] ERROR: RAPIDS_VERSION is required for GPU builds." \
	            "Example: make -f docker.mk docker-gpu RAPIDS_VERSION=25.06" \
	            "(see https://docs.rapids.ai/install)" && exit 1)
	$(call _check_prereqs)
	$(call _build) .

## Build and push to GHCR (requires `docker login ghcr.io` first)
docker-push:
	$(call _check_prereqs)
	$(call _build) --push .
