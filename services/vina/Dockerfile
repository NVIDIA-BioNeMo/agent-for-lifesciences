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

# Autodock Vina Service Dockerfile
# Supports both CPU-only and GPU-accelerated builds
#
# Build options:
#   CPU-only: docker build --target cpu-final -t aevom-vina .
#   GPU-enabled: docker build --target gpu-final -t aevom-vina-gpu .
#
# gpu-final builds for linux/amd64 and linux/arm64 (Grace: DGX Spark, GH200).
# The GPU stages are chosen by TARGETARCH, which BuildKit sets per platform.
ARG TARGETARCH

# =============================================================================
# GPU conformer worker environment (builder only; gpu-final copies the result)
#
# nvMolKit runs in a separate process with its own interpreter and packages,
# never inside the Vina service. See conformer_worker/build_env.sh.
# =============================================================================
FROM ubuntu:22.04 AS conformer-worker-env

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bzip2 binutils \
    && rm -rf /var/lib/apt/lists/*

COPY conformer_worker/pins.env conformer_worker/build_env.sh conformer_worker/inventory.py /build/conformer_worker/
RUN bash /build/conformer_worker/build_env.sh

# =============================================================================
# CPU-only stage: Uses pre-built Vina binary
# =============================================================================
FROM python:3.11-slim AS cpu-final

# Set by BuildKit to the platform being built (amd64 on Linux hosts, arm64 on
# Apple Silicon).
ARG TARGETARCH

WORKDIR /app

# Install system dependencies including OpenBabel runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Download the pre-built Vina binary for this architecture from the official
# releases, and run it once: the service's health check never invokes it, so a
# binary for the wrong architecture would otherwise surface only at the first dock.
RUN case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      amd64) VINA_ARCH=x86_64 ;; \
      arm64) VINA_ARCH=aarch64 ;; \
      *) echo "No Vina release binary for architecture ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    wget -q "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_${VINA_ARCH}" \
    -O /usr/local/bin/vina && \
    chmod +x /usr/local/bin/vina && \
    /usr/local/bin/vina --version

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (meeko for preparation, rdkit for molecule handling)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code. The worker script ships for parity, but this image has
# no worker environment: VINA_GPU=true here falls back to RDKit with a logged reason.
COPY app/ ./app/
COPY conformer_worker/ ./conformer_worker/

# Set environment variables for CPU mode
ENV VINA_GPU=false
ENV VINA_BINARY=/usr/local/bin/vina

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# GPU base image per architecture, and Uni-Dock built from source on top of it
#
# amd64 keeps CUDA 12.6 (CUDA 13 drops Volta). arm64 uses CUDA 12.9: the oldest
# nvcc that targets GB10 (sm_121) natively. Uni-Dock publishes no 1.2.0
# binaries, so both architectures compile it; 1.1.0 hangs on GB10 as soon as
# one GPU batch holds two or more ligands (rigid-ensemble docking).
# =============================================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS gpu-base-amd64
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS gpu-base-arm64

FROM gpu-base-${TARGETARCH} AS unidock
ARG TARGETARCH
ARG UNIDOCK_REF=1.2.0
ARG UNIDOCK_COMMIT=95e409172b15dec0989aea70b0f2328e8ca52025
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git cmake make g++ \
      libboost-system-dev libboost-thread-dev libboost-serialization-dev \
      libboost-filesystem-dev libboost-program-options-dev libboost-timer-dev \
    && rm -rf /var/lib/apt/lists/*
# Each architecture entry also embeds PTX, so newer GPUs can JIT-compile it.
# amd64: the same set as the 1.1.0 release binary (Volta, Ampere, Hopper).
# arm64: GH200 (sm_90), GB200 (sm_100), GB10 (sm_121, which also runs sm_120).
# Boost is linked statically, so gpu-final needs no Boost runtime. The binary is
# checked with ldd rather than run: it needs libcuda.so.1, which only the
# NVIDIA container runtime provides.
RUN case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      amd64) cuda_archs="70;80;90" ;; \
      arm64) cuda_archs="90;100;120;121" ;; \
      *) echo "No Uni-Dock build for architecture ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && git clone -q --depth 1 --branch "$UNIDOCK_REF" https://github.com/dptech-corp/Uni-Dock.git /tmp/unidock \
    && [ "$(git -C /tmp/unidock rev-parse HEAD)" = "$UNIDOCK_COMMIT" ] \
    && cmake -S /tmp/unidock/unidock -B /tmp/unidock/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="$cuda_archs" \
         -DBoost_USE_STATIC_LIBS=ON \
    && cmake --build /tmp/unidock/build -j"$(nproc)" --target unidock \
    && install -m 0755 /tmp/unidock/build/unidock /usr/local/bin/unidock \
    && rm -rf /tmp/unidock \
    && missing="$(ldd /usr/local/bin/unidock | grep 'not found' | grep -v 'libcuda\.so\.1' || true)" \
    && if [ -n "$missing" ]; then echo "unidock has unresolved libraries: $missing" >&2; exit 1; fi \
    && if ldd /usr/local/bin/unidock | grep -q libboost; then echo "unidock links Boost dynamically" >&2; exit 1; fi

# =============================================================================
# GPU stage: Uses Uni-Dock (CUDA-native) for GPU-accelerated docking
# =============================================================================
FROM gpu-base-${TARGETARCH} AS gpu-final

WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and system dependencies
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    wget \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip

# Download pre-built CPU Vina binary (used by Python API fallback in production)
ARG TARGETARCH
RUN case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      amd64) VINA_ARCH=x86_64 ;; \
      arm64) VINA_ARCH=aarch64 ;; \
      *) echo "No Vina release binary for architecture ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    wget -q "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_${VINA_ARCH}" \
    -O /usr/local/bin/vina && \
    chmod +x /usr/local/bin/vina && \
    /usr/local/bin/vina --version

# Uni-Dock (CUDA-native GPU docking), built in the unidock stage above
COPY --from=unidock /usr/local/bin/unidock /usr/local/bin/unidock
RUN missing="$(ldd /usr/local/bin/unidock | grep 'not found' | grep -v 'libcuda\.so\.1' || true)" \
    && if [ -n "$missing" ]; then echo "unidock has unresolved libraries: $missing" >&2; exit 1; fi

# Python requirements for the service process (pip RDKit, Meeko, Open Babel).
# nvMolKit, torch and CUDA libraries are NOT installed here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU conformer worker: separate interpreter and packages, started on demand by
# app/conformer_client.py. Nothing in it is on the service's import or library path.
COPY --from=conformer-worker-env /opt/aevom-conformer-worker /opt/aevom-conformer-worker

# Copy application code
COPY app/ ./app/
COPY conformer_worker/ ./conformer_worker/

# Set environment variables for GPU mode
ENV VINA_GPU=true
ENV VINA_BINARY=/usr/local/bin/vina
ENV UNIDOCK_BINARY=/usr/local/bin/unidock
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CONFORMER_WORKER_PYTHON=/opt/aevom-conformer-worker/bin/python
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CONFORMER_COUNT=50
ENV ROT_BOND_THRESHOLD=15
ENV UNIDOCK_SEARCH_MODE=balance

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
