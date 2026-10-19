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

# Protein-ligand interaction profiling service.
#
# Small and CPU-only: profiling one complex is about a second of work, and the
# heaviest dependency is OpenBabel, which arrives as a prebuilt wheel. Building
# OpenBabel from source would add half an hour to this image and change nothing
# about the answers.
FROM python:3.12-slim-bookworm

# libxrender and libxext are OpenBabel's transitive native dependencies; the
# slim base omits them and the failure is an unhelpful missing-.so at import.
RUN apt-get update \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
      curl \
      libgomp1 \
      libxext6 \
      libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prove the stack works at build time rather than on the first request. PLIP
# needs OpenBabel's Python bindings for protonation and atom typing, and a
# broken binding presents as an import error deep inside analyze().
RUN python3 -c "\
from openbabel import openbabel, pybel; \
from plip.structure.preparation import PDBComplex; \
from plip.exchange.report import BindingSiteReport; \
import gemmi; \
print('PLIP stack present: openbabel', openbabel.OBReleaseVersion(), '| gemmi', gemmi.__version__)"

COPY app/ ./app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
