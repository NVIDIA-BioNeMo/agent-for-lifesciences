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

# Backend Dockerfile - Builds React frontend and runs FastAPI backend
# Uses repo root as build context to access both frontend/ and backend/
#
# Build: docker build -f backend/Dockerfile -t aevom-api .
# Run:   docker run -p 8000:8000 aevom-api

# Build argument for version (from git tag via `git describe --tags --always`)
ARG VERSION=0.0.0-dev

# Stage 1: Build React frontend
FROM node:20-alpine AS builder

# Pass the version to frontend build
ARG VERSION

WORKDIR /app

# Copy package files and install dependencies
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy source and build with version injected
COPY frontend/ .
ENV VERSION=${VERSION}
RUN npm run build

# Stage 2: Runtime with FastAPI
FROM python:3.11-slim

# Accept the version arg again for runtime
ARG VERSION

WORKDIR /app

# Set environment variable for backend access
ENV VERSION_NUMBER=${VERSION}

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application
COPY backend/api/ ./api/

# The OpenShell sandbox policy for python_execute (PYEXEC_MODE=openshell),
# sent with every sandbox the api creates.
COPY deploy/openshell/pyexec-policy.yaml ./deploy/openshell/pyexec-policy.yaml

# Curated skills, loaded at startup from /app/skills (see skills/registry.py).
COPY backend/skills/ ./skills/

# Prose for the libraries installed in the python_execute sandbox. First-party
# today, but loaded through the same root-plus-allowlist path as the NIM skills
# so that swapping in an upstream submodule later touches only
# library_skills_root() in skills/binding.py.
COPY backend/library-skills/ ./library-skills/

# NIM skill prose from the pinned bionemo-agent-toolkit submodule, flattened to
# /app/nim-skills (see skills/binding.py for the resolution order). The whole
# tree is copied, but only the skills allowlisted in binding.py are loaded.
# Requires the submodule to be checked out at build time: an empty
# third_party/ makes this COPY fail rather than silently shipping no skills.
COPY third_party/bionemo-agent-toolkit/nim-skills/ ./nim-skills/

# Third-party attributions. The nim-skills content above is CC-BY-4.0, whose
# attribution has to travel with every copy, including this image.
COPY NOTICE ./NOTICE

# Copy built frontend from builder stage
COPY --from=builder /app/dist ./static

# The system prompt. api/prompts.py reads /prompts.md at import time, so the
# image has to carry a copy or the server cannot start. Deployments that want a
# different prompt without a rebuild mount over this path -- Compose with a bind
# mount, Kubernetes with the aevom-prompts ConfigMap. Last so a prompt edit only
# invalidates this layer.
COPY prompts.md /prompts.md

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
