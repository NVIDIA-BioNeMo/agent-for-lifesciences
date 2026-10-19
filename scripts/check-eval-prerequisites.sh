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
# Aevom Eval Preflight
#
# Guards `make eval`, `make eval-smoke` and `make eval-record`, which drive the
# suite through the harbor CLI. The devcontainer image ships harbor in a venv of
# its own at /opt/harbor-venv (see .devcontainer/Dockerfile); on the host there
# is usually no harbor at all, and the run would die partway through
# evals/scripts/run_suite.sh after already having built images.
#
# Stays silent on success. Exits 1 with remediation instructions otherwise.
#
# Passing on the host is deliberate: a venv on PATH is a supported way to run the
# suite, and someone who set one up should not be told to stop. The check is for
# harbor, not for the devcontainer.

set -uo pipefail

if [ -t 1 ]; then
    RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
    RED=''; YELLOW=''; NC=''
fi

# Kept equal to HARBOR_VERSION in the Makefile, .gitlab-ci.yml and the
# devcontainer Dockerfile; evals/tests/test_ci_wiring.py fails if they disagree.
HARBOR_VERSION="0.20.0"

fail() {
    echo -e "${RED}✗ The evals cannot run here: harbor is not on PATH.${NC}" >&2
    echo "" >&2
    echo -e "$1" >&2
    echo "" >&2
    exit 1
}

command -v harbor >/dev/null 2>&1 && exit 0

# CI installs harbor in the job's before_script rather than baking it into an
# image, so neither remediation below applies there.
if [ -n "${CI:-}" ]; then
    fail "The eval jobs install harbor in .gitlab-ci.yml's .eval-base
before_script and put it on PATH. If it is missing here, that step did not run
or the venv it creates is not on PATH for this shell."
fi

# Are we in a container at all? Env vars are not usable for this: VS Code only
# sets REMOTE_CONTAINERS for processes it spawns itself, so a shell attached any
# other way (docker exec, an agent session) has it unset and would look like the
# host. These marker files are created by the runtime, so they always agree.
if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    fail "You are in a container, but it has no harbor. Either the devcontainer
image predates the eval toolchain, or this is not the devcontainer.

  ${YELLOW}Do NOT pip install harbor here.${NC} It pulls litellm and supabase,
  which would be resolved against the backend dependencies \`make test-fast\`
  runs on. It belongs in the image, in a venv of its own, so it survives
  rebuilds.

  ${YELLOW}Rebuild the devcontainer${NC} (VS Code / Cursor:
  \"Dev Containers: Rebuild Container\"), then re-run your \`make eval\` command."
fi

fail "You are on the host, not in the devcontainer, and harbor is not installed.

  ${YELLOW}Reopen the repo in the devcontainer${NC} (VS Code / Cursor:
  \"Dev Containers: Reopen in Container\"). It ships harbor ${HARBOR_VERSION}
  and everything else the suite needs, and the make targets then work as-is.

  ${YELLOW}Or install harbor on the host, in a venv of its own${NC} (it needs
  Python 3.12 or newer):

    python3 -m venv ~/.harbor-venv
    ~/.harbor-venv/bin/pip install harbor==${HARBOR_VERSION}
    export PATH=\"\$HOME/.harbor-venv/bin:\$PATH\"

  Then re-run your \`make eval\` command. Keep it out of your system Python:
  harbor pulls litellm and supabase, whose pins on openai, httpx and tiktoken
  collide with Aevom's."
