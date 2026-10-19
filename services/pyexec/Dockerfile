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

# =============================================================================
# PyExec Service - Sandboxed Python Execution for Molecular Analysis
# 
# Builds on the RDKit service image to provide Python execution with
# scientific libraries pre-installed (numpy, scipy, rdkit).
#
# NOTE: Requires rdkit image to be built first. Use `make build` to ensure
# correct build order.
#
# Build args:
#   RDKIT_IMAGE - Base image to use (default: aevom-rdkit:latest for local dev)
# =============================================================================

ARG RDKIT_IMAGE=aevom-rdkit:latest

# Wheels for every requirement, built against the same interpreter and glibc as
# the final stage. Some dependencies publish no wheel for every architecture --
# biotraj, which biotite pulls in, has none for linux/aarch64 -- so pip compiles
# them here, and the compiler never reaches the image that runs untrusted code.
FROM ${RDKIT_IMAGE} AS wheels

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

FROM ${RDKIT_IMAGE}

WORKDIR /app

# OpenBabel's wheel links against these at import time; the RDKit base image
# does not carry them. Same pair the vina image installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies for pyexec
COPY requirements.txt .
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
