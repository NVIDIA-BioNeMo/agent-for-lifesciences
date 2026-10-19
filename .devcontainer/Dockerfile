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

FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    curl \
    wget \
    jq \
    git \
    make \
    ca-certificates \
    gnupg \
    openssh-client \
    rsync \
    python3-full \
    python3-pip \
    python3-venv \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with passwordless sudo (handle if already exists)
RUN (getent group $USERNAME || groupadd --gid $USER_GID $USERNAME) \
    && (id -u $USERNAME > /dev/null 2>&1 || useradd --uid $USER_UID --gid $USER_GID -m $USERNAME) \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Rancher CLI (no devcontainer feature available)
RUN curl -fsSL https://releases.rancher.com/cli2/v2.9.0/rancher-linux-amd64-v2.9.0.tar.gz \
    | tar -xz -C /tmp \
    && mv /tmp/rancher-v2.9.0/rancher /usr/local/bin/ \
    && rm -rf /tmp/rancher-v2.9.0 \
    && chmod +x /usr/local/bin/rancher

# Install the GitLab CLI (glab), for reading and commenting on merge requests,
# pipelines and issues on the project's GitLab instance from the shell.
#
# Pinned and checksum-verified. The sums are from the release's checksums.txt:
#   https://gitlab.com/gitlab-org/cli/-/releases/v<GLAB_VERSION>/downloads/checksums.txt
# Bump all three ARGs together. The deb depends only on git, installed above.
#
# Authentication is deliberately not baked into the image. After a rebuild, run
#   glab auth login --hostname <gitlab host>
# (a token needs the `api` scope). It is stored in ~/.config/glab-cli, which is
# in the container's writable layer, so it has to be redone after each rebuild.
ARG GLAB_VERSION=1.122.0
ARG GLAB_SHA256_AMD64=ab853b5e4ab83a42f804c1394b108a698601ae0f37411ba8db0a68c2725c69f6
ARG GLAB_SHA256_ARM64=31e171f9fe69b9adfffec9d6e56e1e51a9ca0580c3dbb8fa263f4dcf8dab6772
RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) sha="$GLAB_SHA256_AMD64" ;; \
         arm64) sha="$GLAB_SHA256_ARM64" ;; \
         *) echo "No glab release for architecture: $arch" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/glab.deb \
        "https://gitlab.com/gitlab-org/cli/-/releases/v${GLAB_VERSION}/downloads/glab_${GLAB_VERSION}_linux_${arch}.deb" \
    && echo "${sha}  /tmp/glab.deb" | sha256sum -c - \
    && dpkg -i /tmp/glab.deb \
    && rm -f /tmp/glab.deb \
    && glab --version

# Install Python packages globally (for the ubuntu user)
# arize-phoenix is pinned again by backend/requirements.txt at the end of this
# file, which additionally caps arize-phoenix-evals<3. Without that cap `import
# phoenix` raises ModuleNotFoundError: No module named 'phoenix.evals.models'.
# Left in place here (rather than deleted) so this layer's cache still hits.
RUN pip3 install --break-system-packages jinja2 streamlit pandas numpy tiktoken arize-phoenix==13.0.3

# Install NeMo RL and NeMo-Gym training dependencies
# These are required for running GRPO training with NeMo-Gym integration
# 
# Core ML: torch, transformers, accelerate, datasets
# Distributed: ray[default]
# LLM serving: vllm
# Logging: wandb, tensorboard, mlflow, nvidia-ml-py, swanlab
# Config: omegaconf, hydra-core
# Server (NeMo-Gym): pydantic, fastapi, uvicorn, httpx, aiohttp
# Build tools: uv (for NeMo RL venv management)
# Utilities: tqdm, devtools, openai, rich, colored, tiktoken, etc.
RUN pip3 install --break-system-packages --timeout 300 \
    torch \
    transformers \
    accelerate \
    datasets \
    "ray[default]" \
    vllm \
    wandb \
    tensorboard \
    mlflow \
    nvidia-ml-py \
    swanlab \
    nvidia-resiliency-ext \
    omegaconf \
    hydra-core \
    pydantic \
    fastapi \
    uvicorn \
    httpx \
    aiohttp \
    tqdm \
    devtools \
    openai \
    rich \
    colored \
    tiktoken \
    blobfile \
    matplotlib \
    plotly \
    sympy \
    pyzmq \
    torchdata \
    uv \
    pre-commit

# Backend runtime deps plus the test toolchain, so the backend suite runs
# natively in the devcontainer (see `make test-fast`) instead of needing a
# container build. backend/requirements.txt is the single source of truth --
# it carries the arize-phoenix-evals<3 cap that a bare `arize-phoenix==13.0.3`
# install silently violates, which breaks `import phoenix`.
#
# Installed as root so the console scripts (pytest, ruff) land in /usr/local/bin,
# which is already on PATH. A --break-system-packages install run as the ubuntu
# user instead lands in ~/.local/bin.
COPY backend/requirements.txt /tmp/reqs/backend-requirements.txt
COPY requirements-dev.txt /tmp/reqs/requirements-dev.txt
RUN pip3 install --break-system-packages --timeout 300 \
        -r /tmp/reqs/backend-requirements.txt \
        -r /tmp/reqs/requirements-dev.txt \
    && rm -rf /tmp/reqs

# Harbor, the eval runner behind `make eval` (see evals/README.md).
#
# In a venv of its own, deliberately. Harbor pulls litellm and supabase, and the
# system interpreter above already balances torch, vllm, transformers, ray,
# arize-phoenix and the whole backend runtime -- litellm's transitive pins on
# openai, httpx and tiktoken land squarely on packages `make test-fast` depends
# on. Nothing needs them to share an interpreter: harbor is a CLI that imports
# evals/aevom_eval off PYTHONPATH, and the driver it uploads is stdlib-only.
#
# Keep this pin equal to HARBOR_VERSION in the Makefile and .gitlab-ci.yml;
# evals/tests/test_ci_wiring.py fails if the three disagree.
ARG HARBOR_VERSION=0.20.0
RUN python3 -m venv /opt/harbor-venv \
    && /opt/harbor-venv/bin/pip install --no-cache-dir "harbor==${HARBOR_VERSION}" \
    && ln -s /opt/harbor-venv/bin/harbor /usr/local/bin/harbor \
    && test "$(harbor --version)" = "${HARBOR_VERSION}"

# `pip3 install --break-system-packages` run as the ubuntu user installs console
# scripts here. Without this they are invisible, which is how a phantom pytest
# install went unnoticed for a while.
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Package-cache directories, pre-created and owned by ubuntu. devcontainer.json
# mounts named volumes over these three paths so the ~8GB of wheels and
# uv-managed interpreters survive a rebuild.
#
# The ownership is the point. Docker seeds a fresh named volume from whatever the
# image has at the mount point, permissions included; when the path is absent
# from the image it creates the volume root-owned instead, and uv -- which runs
# as ubuntu -- then cannot write its own cache. Creating them here means a
# volume created on a clean machine starts out uid 1000.
RUN mkdir -p /home/ubuntu/.cache/uv \
             /home/ubuntu/.cache/pip \
             /home/ubuntu/.local/share/uv/python \
    && chown -R ubuntu:ubuntu /home/ubuntu/.cache /home/ubuntu/.local

# Ensure /tmp has correct permissions (1777 sticky bit) for devcontainer features
# Features need to create temp files during apt-get operations
RUN chmod 1777 /tmp

# NOTE: Do NOT set USER here - devcontainer features need root access to install.
# The "remoteUser" in devcontainer.json handles running as ubuntu.
