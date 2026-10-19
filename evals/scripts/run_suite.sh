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

# Run the Aevom eval suite through Harbor and gate on the result.
#
# Both the Makefile targets and the GitLab jobs call this, so the harbor
# invocation exists once. Everything is configured by environment variable
# because that is what a CI job can set without re-quoting a command line.
#
#   AEVOM_EVAL_MODE          off | mixed | replay   (default mixed)
#   AEVOM_EVAL_TASKS         task path or directory (default evals/tasks)
#   AEVOM_EVAL_ATTEMPTS      harbor --n-attempts    (default 1)
#   AEVOM_EVAL_CONCURRENCY   harbor --n-concurrent  (default 1)
#   AEVOM_EVAL_MODEL         harbor --model, becomes ChatRequest.model_id
#   AEVOM_EVAL_JOBS_DIR      where harbor writes    (default jobs)
#   AEVOM_EVAL_MIN_MEAN      gate threshold on the mean       (default 0.8)
#   AEVOM_EVAL_MIN_TASK      gate threshold per task          (default unset)
#   AEVOM_EVAL_MIN_TRIALS    fail if fewer trials graded      (default 1)
#   AEVOM_API_IMAGE etc.     pin stack images; see docker-compose.eval.yml
#
# Concurrency defaults to 1 deliberately. Each trial starts a full Aevom stack
# -- postgres, redis, rdkit, pyexec, vina, phoenix, molstar-headless -- and two
# of those at once will exhaust a CI runner before they finish.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${AEVOM_EVAL_MODE:-mixed}"
TASKS="${AEVOM_EVAL_TASKS:-evals/tasks}"
ATTEMPTS="${AEVOM_EVAL_ATTEMPTS:-1}"
CONCURRENCY="${AEVOM_EVAL_CONCURRENCY:-1}"
JOBS_DIR="${AEVOM_EVAL_JOBS_DIR:-jobs}"
MIN_MEAN="${AEVOM_EVAL_MIN_MEAN:-0.8}"
MIN_TRIALS="${AEVOM_EVAL_MIN_TRIALS:-1}"

# Bind-mount sources must resolve on the Docker daemon's filesystem, which is
# the host's under docker-outside-of-docker. Same indirection as scripts/run-tests.sh.
export HOST_PROJECT_PATH="${HOST_PROJECT_PATH:-$REPO_ROOT}"

# The agent and verifier are imported by harbor in this process.
export PYTHONPATH="$REPO_ROOT/evals${PYTHONPATH:+:$PYTHONPATH}"

# `make eval` already ran this. Repeated here because CI calls this script
# directly, and because one script owning the message means one message.
./scripts/check-eval-prerequisites.sh || exit 1

# The stack reads its keys from mounted files, and docker-compose.eval.yml sets
# create_host_path:false so a missing one fails at container start rather than
# surfacing as an unexplained model error. In CI the values arrive as masked
# variables instead of files.
#
# There are two, because models are split across two endpoints in
# backend/api/llm/config.py. Which one a run needs depends on AEVOM_MODEL, and
# docker-compose.eval.yml pins it to nemotron-3-ultra-nvinf -- the production
# model -- which reads the *NVINF* one from inference-api.nvidia.com. Both files
# must exist for the mount either way.
#
# API_KEY_NV has a second job that has nothing to do with models: the stack mounts
# it at /run/secrets/nvidia_api_key, which is where NvidiaApiTool reads the key for
# health.api.nvidia.com. The NIM-backed tasks need it even though the pinned model
# does not, so it is checked separately below.
write_key_file() {
    local path="$1" variable="$2" value="$3"
    if [ -f "$path" ]; then
        return 0
    fi
    if [ -n "$value" ]; then
        printf '%s' "$value" > "$path"
        echo "Wrote $path from \$$variable."
    else
        # An empty file satisfies the mount for a run that does not need this
        # endpoint. Announced, not silent: an empty key reaches the model as a
        # 401 several minutes later, pointing nowhere near this line.
        : > "$path"
        echo "NOTE: $path is empty (\$$variable is unset). A model on that" >&2
        echo "      endpoint will fail to authenticate." >&2
    fi
}

write_key_file API_KEY_NV NVIDIA_API_KEY "${NVIDIA_API_KEY:-}"
write_key_file API_KEY_NVINF NVIDIA_API_KEY_NVINF "${NVIDIA_API_KEY_NVINF:-}"
write_key_file API_KEY_ANTHROPIC ANTHROPIC_API_KEY "${ANTHROPIC_API_KEY:-}"
write_key_file API_KEY_OPENAI OPENAI_API_KEY "${OPENAI_API_KEY:-}"

if [ ! -s API_KEY_NV ] && [ ! -s API_KEY_NVINF ] && [ ! -s API_KEY_ANTHROPIC ] \
    && [ ! -s API_KEY_OPENAI ]; then
    cat >&2 <<'EOF'
ERROR: no API key is available, so no model can be called.

The model runs live in every cache mode -- only tool responses are ever replayed
-- so the suite cannot run without one. Provide the key for the endpoint your
AEVOM_MODEL uses, as a file at the repo root or as an environment variable:

    API_KEY_NV     / $NVIDIA_API_KEY        integrate.api.nvidia.com
    API_KEY_NVINF  / $NVIDIA_API_KEY_NVINF  inference-api.nvidia.com
                                            (nemotron-3-ultra-nvinf, what the
                                             eval stack pins)
    API_KEY_ANTHROPIC / $ANTHROPIC_API_KEY  api.anthropic.com
    API_KEY_OPENAI    / $OPENAI_API_KEY     api.openai.com
EOF
    exit 1
fi

# Warned rather than fatal: most tasks need no NVIDIA tool key, so a run without
# one is still worth doing. Said up front because the alternative is three tasks
# scoring zero several minutes in, each reporting only "API key file not found:
# /run/secrets/nvidia_api_key", which reads like a stack misconfiguration.
if [ ! -s API_KEY_NV ]; then
    echo "" >&2
    echo "NOTE: API_KEY_NV is empty, so the tools that call health.api.nvidia.com" >&2
    echo "      have no key. The tasks built on the NIM endpoints -- msa-alignment-" >&2
    echo "      coverage, binder-backbone-design, de-novo-sequence-design -- will" >&2
    echo "      score zero. Set \$NVIDIA_API_KEY to grade them." >&2
    echo "" >&2
fi

export AEVOM_TOOL_CACHE_MODE="$MODE"
CACHE_SOURCE=""
if [ "$MODE" != "off" ]; then
    # A scratch copy, because `mixed` records on a miss and the committed
    # fixtures are not the run's scratch space. See evals/fixtures/tool_cache/README.md.
    CACHE_SOURCE="$REPO_ROOT/$JOBS_DIR/tool_cache"
    rm -rf "$CACHE_SOURCE"
    mkdir -p "$(dirname "$CACHE_SOURCE")"
    cp -r evals/fixtures/tool_cache "$CACHE_SOURCE"
    echo "Tool cache: $MODE, from a copy of evals/fixtures/tool_cache"
else
    echo "Tool cache: off -- every tool call is live"
fi

# --- Make the daemon able to see what the stack mounts ----------------------
#
# The stack bind-mounts both key files, prompts.md and the tool cache. Those
# paths are resolved by the Docker *daemon*, in its own filesystem. Under
# docker-outside-of-docker -- a devcontainer, or a CI runner sharing the host
# socket -- the checkout path inside this container need not exist on the host.
# docker-compose.eval.yml sets create_host_path:false so that fails loudly, but
# it fails as "Docker compose command failed" once per trial, which says nothing
# about paths.
#
# The probe mounts the directory and looks for a regular file inside it. Probing
# a file directly does not work: the daemon creates a missing bind source as a
# directory, and `test -s` passes on a directory (size 4096), so the check this
# script originally shipped with green-lit a path the daemon could not see.
daemon_can_see() {
    docker run --rm -v "$1:/probe:ro" --entrypoint /bin/sh alpine:3 \
        -c 'test -f /probe/prompts.md' >/dev/null 2>&1
}

STAGING=""
cleanup_staging() {
    # The staged copy holds API keys, so it does not outlive the run. Removed
    # through a container because the host filesystem is, again, not ours.
    [ -n "$STAGING" ] || return 0
    docker run --rm -v "$(dirname "$STAGING"):/parent" --entrypoint /bin/sh alpine:3 \
        -c "rm -rf '/parent/$(basename "$STAGING")'" >/dev/null 2>&1 || true
}
trap cleanup_staging EXIT INT TERM

if ! daemon_can_see "$HOST_PROJECT_PATH"; then
    # Rather than telling the operator to reconfigure their runner, copy the few
    # files the stack needs somewhere the daemon does have: a path under /tmp,
    # which the daemon creates for us. The copy streams over stdin through a
    # container, since a shared filesystem is precisely what is missing.
    STAGING="/tmp/aevom-eval-${CI_JOB_ID:-$$}"
    echo "The Docker daemon cannot see $HOST_PROJECT_PATH."
    echo "Staging the stack's mounts to $STAGING on the Docker host instead."
    cleanup_staging
    tar -c prompts.md API_KEY_NV API_KEY_NVINF API_KEY_ANTHROPIC API_KEY_OPENAI \
        | docker run -i --rm -v "$STAGING:/dst" alpine:3 tar -x -C /dst
    if [ -n "$CACHE_SOURCE" ]; then
        tar -c -C "$(dirname "$CACHE_SOURCE")" "$(basename "$CACHE_SOURCE")" \
            | docker run -i --rm -v "$STAGING:/dst" alpine:3 tar -x -C /dst
    fi
    export HOST_PROJECT_PATH="$STAGING"
    if ! daemon_can_see "$HOST_PROJECT_PATH"; then
        cat >&2 <<EOF
ERROR: staging to $STAGING did not work either, so the stack has no way to read
its API key or prompts.md.

That means the daemon could not be written to through a container, which is
unusual. Set HOST_PROJECT_PATH to the repository's path *on the Docker host* --
on a CI runner that shares the host socket, the build directory is often a named
volume with no host path, so CI_BUILDS_DIR has to point at a host bind mount.
EOF
        exit 1
    fi
fi

if [ -n "$CACHE_SOURCE" ]; then
    if [ -n "$STAGING" ]; then
        export AEVOM_TOOL_CACHE_HOST_DIR="$STAGING/$(basename "$CACHE_SOURCE")"
    else
        export AEVOM_TOOL_CACHE_HOST_DIR="$HOST_PROJECT_PATH/$JOBS_DIR/tool_cache"
    fi
fi

harbor_args=(
    run
    --path "$TASKS"
    --agent aevom_eval.agent:AevomAgent
    --verifier aevom_eval.verifier:AevomVerifier
    --jobs-dir "$JOBS_DIR"
    --n-attempts "$ATTEMPTS"
    --n-concurrent "$CONCURRENCY"
    --yes
)
[ -n "${AEVOM_EVAL_MODEL:-}" ] && harbor_args+=(--model "$AEVOM_EVAL_MODEL")
# Recorded on every trial result, so a number can be traced to the images that
# produced it.
[ -n "${AEVOM_API_IMAGE:-}" ] && harbor_args+=(--ae "AEVOM_IMAGE_TAG=$AEVOM_API_IMAGE")

echo "+ harbor ${harbor_args[*]}"
# Not `set -e`-fatal: harbor exits non-zero for a failed *trial*, and a trial
# that scored badly is a result to gate on, not a broken run. The gate decides.
harbor "${harbor_args[@]}" || echo "harbor exited non-zero; grading what it produced"

gate_args=(
    --jobs-dir "$JOBS_DIR"
    --min-mean-reward "$MIN_MEAN"
    --min-trials "$MIN_TRIALS"
    --json-out "$JOBS_DIR/gate.json"
)
[ -n "${AEVOM_EVAL_MIN_TASK:-}" ] && gate_args+=(--min-task-reward "$AEVOM_EVAL_MIN_TASK")

# Not exec'd. exec replaces this shell, and the EXIT trap goes with it -- which
# left the staged API keys sitting on the Docker host after every run.
gate_status=0
evals/scripts/gate.py "${gate_args[@]}" || gate_status=$?
exit "$gate_status"
