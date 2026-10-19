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
# Aevom DevContainer Preflight
#
# Guards `make test-fast`, which runs the backend suite against the interpreter
# in the current shell. That only works inside the devcontainer, whose image
# ships pytest and the backend dependencies (see .devcontainer/Dockerfile).
# On the host there is no pytest and no backend deps, so the run would fail with
# a confusing ModuleNotFoundError instead of pointing at `make test`.
#
# Stays silent on success -- this sits on a hot path and should not add noise.
# Exits 1 with remediation instructions otherwise.

set -uo pipefail

if [ -t 1 ]; then
    RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
    RED=''; YELLOW=''; NC=''
fi

fail() {
    echo -e "${RED}✗ make test-fast cannot run here.${NC}" >&2
    echo "" >&2
    echo -e "$1" >&2
    echo "" >&2
    exit 1
}

# Are we in a container at all? Env vars are not usable for this: VS Code only
# sets REMOTE_CONTAINERS for processes it spawns itself, so a shell attached any
# other way (docker exec, an agent session) has it unset and would look like the
# host. These marker files are created by the runtime, so they always agree.
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    fail "You are on the host, not in the devcontainer. The backend deps and
pytest live in the devcontainer image, not on your machine.

  ${YELLOW}Run the containerized suite instead:${NC}
    make test

  ${YELLOW}Or reopen the repo in the devcontainer${NC} (VS Code / Cursor:
  \"Dev Containers: Reopen in Container\") and then \`make test-fast\` works."
fi

# In a container, but is it one that can run the suite? Catches both a stale
# devcontainer image built before the test toolchain was added, and a shell
# exec'd into some other Aevom container.
if ! command -v pytest >/dev/null 2>&1; then
    fail "You are in a container, but it has no pytest on PATH. Either the
devcontainer image predates the test toolchain, or this is not the devcontainer.

  ${YELLOW}Do NOT pip install pytest.${NC} It belongs in the image, via
  requirements-dev.txt, so it survives rebuilds.

  ${YELLOW}Rebuild the devcontainer${NC} (VS Code / Cursor:
  \"Dev Containers: Rebuild Container\").

  ${YELLOW}Or run the containerized suite, which needs nothing installed:${NC}
    make test"
fi
