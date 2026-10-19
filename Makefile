# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Aevom Makefile
# Orchestrates Docker builds and common development tasks

.PHONY: help check require-gpu-docker submodules build build-dev build-gpu up up-gpu dev dev-gpu dev-quick dev-gpu-quick down logs clean rebuild version test test-fast test-backend test-rdkit test-vina verify-gpu-conformers test-pyexec test-admet test-plip test-training check-eval eval eval-smoke eval-record eval-clean openshell-images up-openshell dev-openshell openshell-clean test-openshell

# Auto-detect Docker Compose command (plugin vs standalone)
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Version from git tags (single source of truth)
# Writes to .version file which is read by docker-compose
VERSION := $(shell git describe --tags --always 2>/dev/null || echo "0.0.0-dev")
export VERSION

# Files that trigger rebuilds (requirements and Dockerfiles)
BACKEND_DEPS := backend/requirements.txt backend/Dockerfile
FRONTEND_DEPS := frontend/package.json frontend/package-lock.json frontend/Dockerfile.dev
RDKIT_DEPS := services/rdkit/requirements.txt services/rdkit/Dockerfile
PYEXEC_DEPS := services/pyexec/requirements.txt services/pyexec/Dockerfile
# The vina image bakes in app/ and the GPU conformer worker (script, pins and
# environment build), so changes to any of them must rebuild it.
VINA_DEPS := services/vina/requirements.txt services/vina/Dockerfile \
	$(shell find services/vina/app services/vina/conformer_worker -type f -not -path '*/__pycache__/*' 2>/dev/null)
ADMET_DEPS := services/admet/requirements.txt services/admet/Dockerfile
PLIP_DEPS := services/plip/requirements.txt services/plip/Dockerfile
# Sandboxes run the code baked into the image rather than a live-reload mount,
# so any change under app/ must rebuild it.
PYEXEC_SANDBOX_DEPS := $(PYEXEC_DEPS) services/pyexec/Dockerfile.openshell \
	$(shell find services/pyexec/app -type f -not -path '*/__pycache__/*' 2>/dev/null)

# How python_execute is isolated in the local stacks. openshell (the default)
# runs it in one OpenShell sandbox per chat session; direct uses the shared
# pyexec service, e.g. `make dev PYEXEC_MODE=direct`. Training and the eval
# stack always use direct. See docs/openshell_sandbox.md.
PYEXEC_MODE ?= openshell
ifeq ($(PYEXEC_MODE),openshell)
OPENSHELL_OVERLAY := -f docker-compose.openshell.yml
PYEXEC_SERVICE := openshell-gateway
SANDBOX_IMAGE_STAMP := .build-stamps/pyexec-sandbox
else ifeq ($(PYEXEC_MODE),direct)
OPENSHELL_OVERLAY :=
PYEXEC_SERVICE := pyexec
SANDBOX_IMAGE_STAMP :=
else
$(error PYEXEC_MODE must be openshell or direct, not '$(PYEXEC_MODE)')
endif

# Microservices the dev stack waits for before starting the backend. admet is
# deliberately absent: its healthcheck allows 180s for the Chemprop ensemble to
# load, and blocking every `make dev` on that is worse than letting the first
# prediction pay for it.
WAIT_SERVICES := rdkit $(PYEXEC_SERVICE) vina plip

# Compose command shortcuts
PROD_COMPOSE = $(DOCKER_COMPOSE) -f docker-compose.yml $(OPENSHELL_OVERLAY)
PROD_GPU_COMPOSE = $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml $(OPENSHELL_OVERLAY)
DEV_COMPOSE = $(DOCKER_COMPOSE) -f docker-compose.dev.yml $(OPENSHELL_OVERLAY)
DEV_GPU_COMPOSE = $(DOCKER_COMPOSE) -f docker-compose.dev.yml -f docker-compose.gpu.yml $(OPENSHELL_OVERLAY)

# Wait for Docker containers to be healthy (120s timeout per service)
# Usage: $(call wait-healthy,service1 service2 ...)
#
# Containers are looked up through compose rather than by name: the project
# name, and so every container name, comes from the checkout's directory, which
# is not "aevom" for a worktree or an extracted source tarball.
define wait-healthy
	@for svc in $(1); do \
		printf "  ⏳ $$svc..."; \
		for i in $$(seq 1 60); do \
			cid=$$($(DEV_COMPOSE) ps -q $$svc 2>/dev/null | head -n1); \
			s=$$( [ -n "$$cid" ] && docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' $$cid 2>/dev/null || echo "waiting"); \
			case "$$s" in healthy|running) printf " ✓\n"; break;; esac; \
			if [ $$i -eq 60 ]; then printf " ✗ timeout\n"; exit 1; fi; \
			sleep 2; \
		done; \
	done
endef

# Default target
help:
	@echo "Aevom Development Commands"
	@echo ""
	@echo "  make check         - Verify system prerequisites (includes GPU check)"
	@echo "  make submodules    - Populate third_party/ (NIM skill content)"
	@echo "  make build         - Build all containers (production, CPU)"
	@echo "  make build-dev     - Build all containers (development, CPU)"
	@echo "  make build-gpu     - Build with Vina GPU acceleration"
	@echo "  make up            - Start production stack (CPU)"
	@echo "  make up-gpu        - Start production stack with Vina GPU"
	@echo "  make dev           - Start development stack with hot-reload (CPU)"
	@echo "  make dev-gpu       - Start development stack with Vina GPU"
	@echo ""
	@echo "Quick Start (skip rebuild if images exist):"
	@echo "  make dev-quick     - Quick start dev stack (no build)"
	@echo "  make dev-gpu-quick - Quick start dev stack with GPU (no build)"
	@echo ""
	@echo "python_execute runs in an OpenShell sandbox per chat session by default."
	@echo "Add PYEXEC_MODE=direct to any start target for the shared pyexec service:"
	@echo "    make dev PYEXEC_MODE=direct"
	@echo "  make openshell-clean - Remove the gateway and every sandbox it created"
	@echo "  make test-openshell  - Escape suite against a live sandbox"
	@echo ""
	@echo "  make down          - Stop all containers"
	@echo "  make logs          - Tail logs from all containers"
	@echo "  make clean         - Remove all containers and volumes"
	@echo "  make rebuild       - Clean rebuild of all containers"
	@echo ""
	@echo "Testing:"
	@echo "  make test-fast     - Run the backend suite natively in the devcontainer (~30s)"
	@echo "  make test          - Run the full Python test suite in containers"
	@echo "  make test-backend  - Run only the backend suite"
	@echo "  make test-rdkit    - Run only the rdkit suite (SAFE notation)"
	@echo "  make test-vina     - Run only the vina suite"
	@echo "  make test-pyexec   - Run only the pyexec suite"
	@echo "  make test-admet    - Run only the admet suite (property catalog)"
	@echo "  make test-plip     - Run only the plip suite (interaction profiling)"
	@echo "  make test-training - Run only the NeMo-Gym suite (needs aevom-training image)"
	@echo ""
	@echo "  For a single file or test, call pytest directly:"
	@echo "    pytest backend/api/tests/test_registry.py -k some_case"
	@echo ""
	@echo "Evals (end-to-end, drives a production-config stack via Harbor):"
	@echo "  make eval-smoke    - Cached tools, one attempt per task (the MR tier)"
	@echo "  make eval          - Live tools, every task (the nightly tier)"
	@echo "  make eval-record   - Re-record the tool cache fixtures"
	@echo "  make eval-clean    - Remove jobs/ and any leftover eval stack"
	@echo ""
	@echo "  Narrow to one task with TASK=, and pick a model with MODEL=:"
	@echo "    make eval-smoke TASK=structure-query-neighbors"
	@echo "    make eval MODEL=nemotron-3-super-rc-nim"
	@echo ""
	@echo "  Evals need API_KEY_NV. harbor $(HARBOR_VERSION) ships in the devcontainer"
	@echo "  image; rebuild the container if it is missing. See evals/README.md."
	@echo ""
	@echo "GPU commands require: NVIDIA driver, nvidia-container-toolkit"
	@echo "Run 'make check' to verify GPU support."
	@echo ""
	@echo "Training Mode (DPO Data Collection):"
	@echo "  After 'make dev', open these URLs in separate tabs:"
	@echo "    Oracle (good): http://localhost:5180/training?mode=oracle"
	@echo "    Weak (bad):    http://localhost:5180/training?mode=weak"
	@echo "  Export data:     python3 scripts/export_training_data.py"
	@echo ""
	@echo "Using: $(DOCKER_COMPOSE)"
	@echo ""

# Check prerequisites before building
check:
	@./scripts/check-prerequisites.sh

# Populate third_party/bionemo-agent-toolkit, which carries the NIM SKILL.md
# prose loaded by api/skills/binding.py. Safe to re-run; a no-op once checked out.
# Works without .git too (a source tarball), by downloading the pinned commit.
submodules:
	@./scripts/fetch-submodules.sh

# Generate .version file from git tags (single source of truth)
# This file is read by docker-compose for VERSION_NUMBER env var
.version:
	@echo "$(VERSION)" > .version
	@echo "Version: $(VERSION)"

# Show current version
version:
	@echo "$(VERSION)"

# Fast inner loop: the backend suite (~780 tests, ~30s) runs natively in the
# devcontainer, whose image ships pytest and the backend deps. Which suites run
# is configured by testpaths in pytest.ini. For a single file or test, call
# pytest directly -- `pytest backend/api/tests/test_registry.py -k some_case`.
#
# Guarded because this runs against the current shell's interpreter: on the host
# there is no pytest, so the preflight points at `make test` rather than letting
# it fail with a confusing ModuleNotFoundError.
#
# `make test` below remains the full gate: it also covers the rdkit, vina,
# pyexec and training suites, which need their container images.
test-fast:
	@./scripts/check-devcontainer.sh
	@pytest -q

# Run the full Python test suite. Each suite runs inside the container image
# that ships its dependencies (backend, rdkit, vina, pyexec, and aevom-training
# for the NeMo-Gym suite). Use the per-suite targets below to run just one.
test:
	@./scripts/run-tests.sh

# Per-suite targets (delegate to the runner's suite filtering)
test-backend:
	@./scripts/run-tests.sh backend

test-rdkit:
	@./scripts/run-tests.sh rdkit

test-vina:
	@./scripts/run-tests.sh vina

# Required GPU gate for the conformer worker: rebuilds the GPU vina image if
# needed, then fails unless nvMolKit ran in the isolated worker (no skips, no
# CPU fallback) and the process-separation checks pass.
verify-gpu-conformers: .build-stamps/vina-gpu
	@./scripts/verify-gpu-conformers.sh aevom-vina-gpu:latest

test-pyexec:
	@./scripts/run-tests.sh pyexec

test-admet:
	@./scripts/run-tests.sh admet

test-plip:
	@./scripts/run-tests.sh plip

test-training:
	@./scripts/run-tests.sh training

# Build production containers in correct order
# rdkit must be built first as pyexec depends on aevom-rdkit:latest image
build: check .version
	@echo "Building rdkit service (this may take 20-30 minutes on first run)..."
	$(DOCKER_COMPOSE) build rdkit
	@echo "Building remaining services..."
	$(DOCKER_COMPOSE) build

# Build development containers in correct order
build-dev: check .version
	@echo "Building rdkit service (this may take 20-30 minutes on first run)..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build rdkit
	@echo "Building remaining services..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build

# Start production stack
up: build $(SANDBOX_IMAGE_STAMP)
	$(PROD_COMPOSE) up -d
	@echo ""
	@echo "Aevom is running at http://localhost:5180"
	@echo "Phoenix tracing UI at http://localhost:6080"
	@echo "python_execute:    $(PYEXEC_MODE)"

# Start development stack with hot-reload
# Uses smart incremental builds - only rebuilds services whose dependencies changed
# Staged startup ensures each layer is healthy before starting the next
dev: check .version .build-stamps/backend .build-stamps/vite .build-stamps/rdkit .build-stamps/pyexec $(SANDBOX_IMAGE_STAMP) .build-stamps/vina-cpu .build-stamps/admet .build-stamps/plip
	@echo ""
	@echo "Starting infrastructure (Redis, PostgreSQL, Phoenix)..."
	@$(DEV_COMPOSE) up -d redis postgres phoenix
	$(call wait-healthy,redis postgres)
	@echo "Starting microservices (RDKit, PyExec, Vina, ADMET, PLIP)..."
	@$(DEV_COMPOSE) up -d rdkit $(PYEXEC_SERVICE) vina admet plip
	$(call wait-healthy,$(WAIT_SERVICES))
	@echo "Starting backend..."
	@$(DEV_COMPOSE) up -d backend
	$(call wait-healthy,backend)
	@echo "Starting frontend..."
	@$(DEV_COMPOSE) up -d vite
	@echo ""
	@echo "Development server running:"
	@echo "  Frontend (Vite):   http://localhost:5180"
	@echo "  Backend (FastAPI): http://localhost:8080"
	@echo "  Phoenix tracing:   http://localhost:6080"
	@echo "  python_execute:    $(PYEXEC_MODE)"
	@echo ""
	@echo "Training Mode (DPO data collection):"
	@echo "  Oracle (good):     http://localhost:5180/training?mode=oracle"
	@echo "  Weak (bad):        http://localhost:5180/training?mode=weak"

# Quick start development stack (skip build if images exist)
# Use this for fast iteration when you haven't changed dependencies
dev-quick: .version
	@echo "Starting development stack (quick mode - no rebuild)..."
	@$(DEV_COMPOSE) up -d redis postgres phoenix
	$(call wait-healthy,redis postgres)
	@$(DEV_COMPOSE) up -d rdkit $(PYEXEC_SERVICE) vina admet plip
	$(call wait-healthy,$(WAIT_SERVICES))
	@$(DEV_COMPOSE) up -d backend
	$(call wait-healthy,backend)
	@$(DEV_COMPOSE) up -d vite
	@echo ""
	@echo "Development server running:"
	@echo "  Frontend (Vite):   http://localhost:5180"
	@echo "  Backend (FastAPI): http://localhost:8080"
	@echo "  Phoenix tracing:   http://localhost:6080"
	@echo "  python_execute:    $(PYEXEC_MODE)"
	@echo ""
	@echo "Note: Run 'make dev' if you changed requirements.txt or package.json"

# Quick start development stack with GPU (skip build if images exist)
dev-gpu-quick: require-gpu-docker .version
	@echo "Starting development stack with Vina GPU (quick mode - no rebuild)..."
	@$(DEV_GPU_COMPOSE) up -d redis postgres phoenix
	$(call wait-healthy,redis postgres)
	@$(DEV_GPU_COMPOSE) up -d rdkit $(PYEXEC_SERVICE) vina admet plip
	$(call wait-healthy,$(WAIT_SERVICES))
	@$(DEV_GPU_COMPOSE) up -d backend
	$(call wait-healthy,backend)
	@$(DEV_GPU_COMPOSE) up -d vite
	@echo ""
	@echo "Development server running with Vina GPU:"
	@echo "  Frontend (Vite):   http://localhost:5180"
	@echo "  Backend (FastAPI): http://localhost:8080"
	@echo "  Phoenix tracing:   http://localhost:6080"
	@echo "  Vina GPU:          Enabled"
	@echo "  python_execute:    $(PYEXEC_MODE)"
	@echo ""
	@echo "Note: Run 'make dev-gpu' if you changed requirements.txt or package.json"

# python_execute under OpenShell: one governed sandbox per chat session, created
# by an OpenShell gateway through the host Docker socket. See
# docs/openshell_sandbox.md. The sandbox image layers a non-root user onto
# aevom-pyexec:latest, which only the production compose file tags, so both are
# built through it even for the dev stack.
OPENSHELL_COMPOSE = $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.openshell.yml

.build-stamps/pyexec-sandbox: $(PYEXEC_SANDBOX_DEPS) .build-stamps/rdkit | .build-stamps
	DOCKER_BUILDKIT=1 $(OPENSHELL_COMPOSE) build pyexec
	DOCKER_BUILDKIT=1 $(OPENSHELL_COMPOSE) build pyexec-sandbox
	@touch $@

openshell-images: .version
	@rm -f .build-stamps/pyexec-sandbox
	@$(MAKE) .build-stamps/pyexec-sandbox

# Kept for existing scripts and habits; OpenShell is now the default.
up-openshell:
	@$(MAKE) up PYEXEC_MODE=openshell

dev-openshell:
	@$(MAKE) dev PYEXEC_MODE=openshell

# Sandboxes are containers the gateway created, not compose services, so
# `down` does not remove them.
openshell-clean:
	-docker rm -f $$(docker ps -aq --filter label=openshell.ai/managed-by=openshell) 2>/dev/null
	-$(OPENSHELL_COMPOSE) rm -sf openshell-gateway openshell-keys 2>/dev/null

# Escape suite: raw Python inside a live sandbox, bypassing RestrictedPython.
test-openshell:
	@./scripts/test-openshell.sh

# Stop all containers. The openshell overlay moves pyexec into the direct-pyexec
# profile, and compose down skips services whose profile is inactive.
down:
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml -f docker-compose.openshell.yml --profile direct-pyexec down 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.openshell.yml --profile direct-pyexec down 2>/dev/null || true

# Tail logs
logs:
	$(DEV_COMPOSE) logs -f 2>/dev/null || $(PROD_COMPOSE) logs -f

# Clean up everything, including sandboxes and the gateway's key volume
clean: down openshell-clean
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml -f docker-compose.openshell.yml --profile direct-pyexec down -v --rmi local 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.openshell.yml --profile direct-pyexec down -v --rmi local 2>/dev/null || true
	docker rmi aevom-rdkit:latest 2>/dev/null || true
	rm -f .version
	rm -rf .build-stamps
	@echo "Cleaned up containers, volumes, images, and build stamps"

# Force rebuild everything from scratch
rebuild: clean build

# The GPU vina image builds for x86_64 and arm64 engines (arm64: DGX Spark,
# GH200). Docker Desktop on an Apple Silicon Mac also reports aarch64 but has no
# NVIDIA GPU, so arm64 Docker Desktop is refused. Failing here takes seconds; the
# alternative is a long build, or a stack that cannot see a GPU at all.
require-gpu-docker:
	@arch=$$(docker info --format '{{.Architecture}}' 2>/dev/null); \
	os=$$(docker info --format '{{.OperatingSystem}}' 2>/dev/null); \
	case "$$arch:$$os" in \
		aarch64:*"Docker Desktop"*|arm64:*"Docker Desktop"*) \
		   echo "GPU targets need an NVIDIA GPU; Docker Desktop on Apple Silicon has none."; \
		   echo "Use the CPU targets instead: make up, make dev, make dev-quick."; \
		   exit 1 ;; \
	esac; \
	case "$$arch" in \
		x86_64|amd64|aarch64|arm64) ;; \
		*) echo "GPU targets need an x86_64 or arm64 Docker engine with an NVIDIA GPU; this one is '$${arch:-unknown}'."; \
		   echo "Use the CPU targets instead: make up, make dev, make dev-quick."; \
		   exit 1 ;; \
	esac

# Build with Vina GPU support (requires nvidia-container-toolkit)
build-gpu: require-gpu-docker check .version
	@echo "Building with Vina GPU acceleration..."
	@echo "Note: Requires nvidia-container-toolkit and CUDA-capable GPU"
	@echo "Building rdkit service (this may take 20-30 minutes on first run)..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml build rdkit
	@echo "Building remaining services..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml build

# Start production stack with Vina GPU
up-gpu: build-gpu $(SANDBOX_IMAGE_STAMP)
	$(PROD_GPU_COMPOSE) up -d
	@echo ""
	@echo "Aevom is running with Vina GPU:"
	@echo "  App:             http://localhost:5180"
	@echo "  Phoenix tracing: http://localhost:6080"
	@echo "  Vina GPU:        Enabled"
	@echo "  python_execute:  $(PYEXEC_MODE)"

# Stamp files to track when services were last built
# These are updated after successful builds and compared against dependency files
.build-stamps:
	@mkdir -p .build-stamps

# Smart build: only rebuild services whose dependencies changed
.build-stamps/backend: $(BACKEND_DEPS) | .build-stamps
	@echo "Backend dependencies changed, rebuilding..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build backend
	@touch $@

.build-stamps/vite: $(FRONTEND_DEPS) | .build-stamps
	@echo "Frontend dependencies changed, rebuilding..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build vite
	@touch $@

.build-stamps/rdkit: $(RDKIT_DEPS) | .build-stamps
	@echo "RDKit dependencies changed, rebuilding..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build rdkit
	@touch $@

.build-stamps/pyexec: $(PYEXEC_DEPS) .build-stamps/rdkit | .build-stamps
	@echo "PyExec dependencies changed, rebuilding..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build pyexec
	@touch $@

.build-stamps/vina-gpu: $(VINA_DEPS) | require-gpu-docker .build-stamps
	@echo "Vina dependencies changed, rebuilding GPU version..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml -f docker-compose.gpu.yml build vina
	@touch $@

.build-stamps/vina-cpu: $(VINA_DEPS) | .build-stamps
	@echo "Vina dependencies changed, rebuilding CPU version..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build vina
	@touch $@

.build-stamps/admet: $(ADMET_DEPS) | .build-stamps
	@echo "ADMET dependencies changed, rebuilding..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build admet
	@touch $@

.build-stamps/plip: $(PLIP_DEPS) | .build-stamps
	@echo "PLIP dependencies changed, rebuilding..."
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f docker-compose.dev.yml build plip
	@touch $@

# Start development stack with Vina GPU support
# Uses smart incremental builds - only rebuilds services whose dependencies changed
# Staged startup ensures each layer is healthy before starting the next
dev-gpu: require-gpu-docker check .version .build-stamps/backend .build-stamps/vite .build-stamps/rdkit .build-stamps/pyexec $(SANDBOX_IMAGE_STAMP) .build-stamps/vina-gpu .build-stamps/admet .build-stamps/plip
	@echo ""
	@echo "Starting infrastructure (Redis, PostgreSQL, Phoenix)..."
	@$(DEV_GPU_COMPOSE) up -d redis postgres phoenix
	$(call wait-healthy,redis postgres)
	@echo "Starting microservices (RDKit, PyExec, Vina, ADMET, PLIP)..."
	@$(DEV_GPU_COMPOSE) up -d rdkit $(PYEXEC_SERVICE) vina admet plip
	$(call wait-healthy,$(WAIT_SERVICES))
	@echo "Starting backend..."
	@$(DEV_GPU_COMPOSE) up -d backend
	$(call wait-healthy,backend)
	@echo "Starting frontend..."
	@$(DEV_GPU_COMPOSE) up -d vite
	@echo ""
	@echo "Development server running with Vina GPU:"
	@echo "  Frontend (Vite):   http://localhost:5180"
	@echo "  Backend (FastAPI): http://localhost:8080"
	@echo "  Phoenix tracing:   http://localhost:6080"
	@echo "  Vina GPU:          Enabled"
	@echo ""
	@echo "Training Mode (DPO data collection):"
	@echo "  Oracle (good):     http://localhost:5180/training?mode=oracle"
	@echo "  Weak (bad):        http://localhost:5180/training?mode=weak"

# ---------------------------------------------------------------------------
# Evals
#
# End-to-end runs against a stack built from docker-compose.eval.yml, which
# carries the production ConfigMap values rather than the dev stack's hot-reload
# mounts. Harbor owns the stack lifecycle per task, so these do not disturb a
# running `make dev`.
#
# All of these shell out to evals/scripts/, which is also what CI runs, so each
# tier has one definition rather than a Makefile copy that drifts from it.
# ---------------------------------------------------------------------------

# Pinned for the same reason .gitlab-ci.yml pins it: evals/scripts/gate.py parses
# harbor's JobResult schema, and harbor is pre-1.0.
HARBOR_VERSION := 0.20.0

# TASK= narrows to one task directory; MODEL= overrides AEVOM_MODEL and
# ChatRequest.model_id.
EVAL_TASKS := $(if $(TASK),evals/tasks/$(TASK),evals/tasks)
EVAL_ENV := AEVOM_EVAL_TASKS=$(EVAL_TASKS) $(if $(MODEL),AEVOM_EVAL_MODEL=$(MODEL),)
# Narrowing to one task makes the suite-wide trial floor wrong.
EVAL_TRIALS := $(if $(TASK),AEVOM_EVAL_MIN_TRIALS=1,)

# The eval stack pins images and never builds them, so build first. Unchanged
# services are cache hits.
define eval-build
	@echo "Building the images the eval stack pins..."
	@DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) build rdkit
	@DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) build
endef

# Runs before the image builds, not after: the failure this catches is a missing
# harbor, and discovering that at the end of a multi-minute build is the whole
# problem. eval-record is deliberately not gated on it -- recording drives the
# stack directly and never invokes harbor.
check-eval:
	@./scripts/check-eval-prerequisites.sh

# The merge-request tier: cached tool responses, one attempt per task.
eval-smoke: check check-eval .version
	$(call eval-build)
	@$(EVAL_ENV) $(EVAL_TRIALS) AEVOM_EVAL_MODE=mixed \
		AEVOM_EVAL_MIN_MEAN=0.8 AEVOM_EVAL_MIN_TASK=0.5 \
		bash evals/scripts/run_suite.sh

# The nightly tier: every tool call live, three attempts so a single bad sample
# reads as variance rather than as a regression.
eval: check check-eval .version
	$(call eval-build)
	@$(EVAL_ENV) $(EVAL_TRIALS) AEVOM_EVAL_MODE=off AEVOM_EVAL_ATTEMPTS=3 \
		AEVOM_EVAL_MIN_MEAN=0.7 AEVOM_EVAL_MIN_TASK=0.34 \
		bash evals/scripts/run_suite.sh

# Refresh evals/fixtures/tool_cache from live tools. Deliberately not through
# Harbor: one cache shared across all tasks is the point, and Harbor tears the
# stack down between them. Read evals/fixtures/tool_cache/README.md first.
eval-record: check .version
	$(call eval-build)
	@evals/scripts/record_fixtures.py $(TASK) $(if $(MODEL),--model $(MODEL),)

# Harbor deletes its stacks on success; an interrupted run can leave one behind.
eval-clean:
	@rm -rf jobs evals/results
	@$(DOCKER_COMPOSE) -f docker-compose.eval.yml -p aevom-eval-record down -v 2>/dev/null || true
	@echo "Removed jobs/, evals/results/ and any leftover recording stack."