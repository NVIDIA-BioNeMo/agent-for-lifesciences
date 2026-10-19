#!/usr/bin/env bash
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

#
# Aevom Test Runner
# Runs every Python pytest suite inside the container image that ships its deps.
#
#   backend  -> backend image  (api/tests, api/tools/tests, api/training/tests)
#   rdkit    -> rdkit image    (services/rdkit/tests)
#   vina     -> vina image     (services/vina/tests)
#   pyexec   -> pyexec image   (services/pyexec/tests)
#   admet    -> admet image   (services/admet/tests)
#   plip     -> plip image    (services/plip/tests)
#   training -> aevom-training image (training/tests, NeMo-Gym suite)
#
# The training suite runs only when aevom-training:latest is available locally
# (or pullable with --pull); otherwise it is reported as skipped. That image is
# built separately via training/slurm/container/build_from_source.sh.
#
# Usage:
#   scripts/run-tests.sh                 # run all available suites
#   scripts/run-tests.sh backend         # run a single suite
#   scripts/run-tests.sh vina pyexec     # run a subset
#   scripts/run-tests.sh --pull          # docker pull aevom-training before training suite
#   scripts/run-tests.sh --pull training # pull + run only the training suite

set -uo pipefail

# Colors (disabled when not a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="docker-compose.test.yml"
TRAINING_IMAGE="aevom-training:latest"

# Auto-detect Docker Compose command (plugin vs standalone), mirroring the Makefile.
if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    echo -e "${RED}ERROR: Docker Compose is not available.${NC}" >&2
    exit 1
fi
DC=("${COMPOSE[@]}" -f "$COMPOSE_FILE")

# Version for the backend build arg (single source of truth: git tags).
export VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null || echo "0.0.0-dev")}"
# Bind-mount sources must resolve on the host daemon (docker-outside-of-docker).
export HOST_PROJECT_PATH="${HOST_PROJECT_PATH:-$REPO_ROOT}"

PULL_TRAINING=false
SUITES=()
for arg in "$@"; do
    case "$arg" in
        --pull) PULL_TRAINING=true ;;
        backend|rdkit|vina|pyexec|admet|plip|training) SUITES+=("$arg") ;;
        -h|--help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}" >&2
            echo "Valid suites: backend rdkit vina pyexec admet plip training (flags: --pull)" >&2
            exit 2
            ;;
    esac
done

# Default: run every suite.
if [ "${#SUITES[@]}" -eq 0 ]; then
    SUITES=(backend rdkit vina pyexec admet plip training)
fi

want() {
    local target="$1"
    for s in "${SUITES[@]}"; do
        [ "$s" = "$target" ] && return 0
    done
    return 1
}

# True when the training image's conformer worker was built from the repo's
# current pins. The label is set by the build scripts; images built without it
# are checked against the build-info.json inside the worker prefix.
training_worker_current() {
    local want got cid
    want="$(services/vina/conformer_worker/pins_sha.sh)"
    got="$(docker image inspect --format '{{ index .Config.Labels "aevom.conformer_worker.pins_sha" }}' "$TRAINING_IMAGE" 2>/dev/null)"
    if [ -z "$got" ] || [ "$got" = "unset" ] || [ "$got" = "<no value>" ]; then
        cid="$(docker create "$TRAINING_IMAGE" 2>/dev/null)" || return 1
        got="$(docker cp "$cid:/opt/aevom-conformer-worker/share/aevom-conformer-worker/build-info.json" - 2>/dev/null \
            | tar -xO 2>/dev/null \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["pins_sha256"])' 2>/dev/null)"
        docker rm "$cid" >/dev/null 2>&1
    fi
    [ "$got" = "$want" ]
}

# Results tracking
declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
record() { RESULT_NAMES+=("$1"); RESULT_STATUS+=("$2"); }

run_suite() {
    local name="$1" service="$2"
    echo ""
    echo -e "${BLUE}=== ${name} tests ===${NC}"
    if "${DC[@]}" run --rm --no-deps "$service"; then
        record "$name" "PASS"
    else
        record "$name" "FAIL"
    fi
}

# ---------------------------------------------------------------------------
# Build lightweight test images once (rdkit first -- pyexec builds FROM it).
# Cache is reused from `make build-dev`; a fresh devcontainer builds from scratch.
# ---------------------------------------------------------------------------
BUILD_SERVICES=()
want backend && BUILD_SERVICES+=("backend-tests")
want vina && BUILD_SERVICES+=("vina-tests")
want pyexec && BUILD_SERVICES+=("pyexec-tests")
want admet && BUILD_SERVICES+=("admet-tests")
want plip && BUILD_SERVICES+=("plip-tests")

if want pyexec || want rdkit; then
    echo -e "${BLUE}Building rdkit base image (aevom-rdkit:latest)...${NC}"
    if ! "${DC[@]}" build rdkit; then
        echo -e "${RED}ERROR: failed to build rdkit base image.${NC}" >&2
        exit 1
    fi
fi

if [ "${#BUILD_SERVICES[@]}" -gt 0 ]; then
    echo -e "${BLUE}Building test images: ${BUILD_SERVICES[*]}...${NC}"
    if ! "${DC[@]}" build "${BUILD_SERVICES[@]}"; then
        echo -e "${RED}ERROR: failed to build test images.${NC}" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Run suites
# ---------------------------------------------------------------------------
want backend && run_suite "backend" "backend-tests"
want rdkit && run_suite "rdkit" "rdkit-tests"
want vina && run_suite "vina" "vina-tests"
want pyexec && run_suite "pyexec" "pyexec-tests"
want admet && run_suite "admet" "admet-tests"
want plip && run_suite "plip" "plip-tests"

if want training; then
    if [ "$PULL_TRAINING" = true ]; then
        echo -e "${BLUE}Pulling ${TRAINING_IMAGE}...${NC}"
        docker pull "$TRAINING_IMAGE" >/dev/null 2>&1 || \
            echo -e "${YELLOW}WARNING: could not pull ${TRAINING_IMAGE}.${NC}" >&2
    fi
    if docker image inspect "$TRAINING_IMAGE" >/dev/null 2>&1; then
        if ! training_worker_current; then
            echo ""
            echo -e "${RED}=== training tests ===${NC}"
            echo -e "${RED}STALE: ${TRAINING_IMAGE} has a conformer worker built from different pins than${NC}"
            echo -e "${RED}  services/vina/conformer_worker/ (or none). Rebuild it with${NC}"
            echo -e "${RED}  training/slurm/container/build_from_source.sh.${NC}"
            record "training" "FAIL"
        else
            run_suite "training" "training-tests"
        fi
    else
        echo ""
        echo -e "${YELLOW}=== training tests ===${NC}"
        echo -e "${YELLOW}SKIPPED: ${TRAINING_IMAGE} not available.${NC}"
        echo -e "${YELLOW}  Build it with training/slurm/container/build_from_source.sh,${NC}"
        echo -e "${YELLOW}  or re-run with --pull to fetch it from the registry.${NC}"
        record "training" "SKIP"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BLUE}========== Test Summary ==========${NC}"
FAILED=0
for i in "${!RESULT_NAMES[@]}"; do
    name="${RESULT_NAMES[$i]}"; status="${RESULT_STATUS[$i]}"
    case "$status" in
        PASS) echo -e "  ${GREEN}PASS${NC}  ${name}" ;;
        FAIL) echo -e "  ${RED}FAIL${NC}  ${name}"; FAILED=1 ;;
        SKIP) echo -e "  ${YELLOW}SKIP${NC}  ${name}" ;;
    esac
done
echo -e "${BLUE}==================================${NC}"

if [ "$FAILED" -ne 0 ]; then
    echo -e "${RED}Some test suites failed.${NC}"
    exit 1
fi
echo -e "${GREEN}All run test suites passed.${NC}"
exit 0
