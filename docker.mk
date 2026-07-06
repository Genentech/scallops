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
#   make -f docker.mk docker            # CPU image (default)
#   make -f docker.mk docker-gpu TORCH_COMPUTE=cu124   # GPU image
#   make -f docker.mk docker-push       # build CPU + push to GHCR

# ── configurable knobs ────────────────────────────────────────────────────────
REGISTRY      ?= ghcr.io/genentech
IMAGE         ?= scallops
TF_VERSION    ?= 2.21.0
TORCH_COMPUTE ?= cpu

# ── derived values (match what docker/metadata-action computes in CI) ─────────
# NOTE: SCM_VERSION is intentionally left as a lazy variable (=) so that it is
# evaluated *after* the _check_deps recipe has installed setuptools_scm.
GIT_SHA       := $(shell git rev-parse HEAD)
GIT_SHA_SHORT := $(shell git rev-parse --short HEAD)
GIT_SOURCE    := $(shell git remote get-url origin 2>/dev/null \
                   | sed 's|git@github.com:|https://github.com/|; s|\.git$$||')
BUILD_DATE    := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
FULL_IMAGE    := $(REGISTRY)/$(IMAGE)

# ── prerequisite check ────────────────────────────────────────────────────────
# Installs missing Python deps then recomputes SCM_VERSION inside the recipe
# (top-level $(shell) runs before any target so we cannot rely on it here).
define check_prereqs
	@python -m setuptools_scm --version >/dev/null 2>&1 \
	    || (echo "[docker.mk] setuptools_scm not found — installing..." \
	        && python -m pip install -q setuptools_scm)
	@docker buildx version >/dev/null 2>&1 \
	    || (echo "[docker.mk] ERROR: docker buildx required (Docker 20.10+ or the buildx plugin)" \
	        && exit 1)
	$(eval SCM_VERSION := $(shell python -m setuptools_scm))
endef

# OCI labels — mirrors what docker/metadata-action generates in CI
define oci_labels
  --label org.opencontainers.image.created=$(BUILD_DATE) \
  --label org.opencontainers.image.revision=$(GIT_SHA) \
  --label org.opencontainers.image.source=$(GIT_SOURCE) \
  --label org.opencontainers.image.version=$(SCM_VERSION) \
  --label org.opencontainers.image.title=$(IMAGE) \
  --label org.opencontainers.image.url=$(GIT_SOURCE)
endef

define build_args
  --build-arg SCM_VERSION=$(SCM_VERSION) \
  --build-arg TF_VERSION=$(TF_VERSION) \
  --build-arg TORCH_COMPUTE=$(TORCH_COMPUTE)
endef

define image_tags
  -t $(FULL_IMAGE):$(SCM_VERSION) \
  -t $(FULL_IMAGE):sha-$(GIT_SHA_SHORT) \
  -t $(FULL_IMAGE):latest
endef

.PHONY: docker docker-gpu docker-push

## Build the CPU image locally (default)
docker:
	$(call check_prereqs)
	docker buildx build \
	  $(call build_args) \
	  $(call oci_labels) \
	  $(call image_tags) \
	  .

## Build a GPU image locally; override TORCH_COMPUTE if needed (e.g. cu126)
docker-gpu:
	$(MAKE) -f docker.mk docker TORCH_COMPUTE=$(or $(TORCH_COMPUTE),cu124)

## Build and push the CPU image to GHCR (requires `docker login ghcr.io` first)
docker-push:
	$(call check_prereqs)
	docker buildx build \
	  $(call build_args) \
	  $(call oci_labels) \
	  $(call image_tags) \
	  --push \
	  .
