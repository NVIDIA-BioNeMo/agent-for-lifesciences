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

# ADMET prediction service.
#
# ADMET-AI depends on torch and Chemprop. The default PyPI torch wheel carries
# the bundled CUDA runtime -- several gigabytes of it -- and this service runs
# happily on CPU: a batch of 30 molecules through two small multitask Chemprop
# ensembles is seconds of work. The CPU index below is what keeps the image from being
# mostly unused CUDA libraries. Point ADMET_MODELS_DIR at a mounted checkpoint
# directory if a GPU deployment ever wants different weights.
FROM python:3.12-slim-bookworm

# Three of these are transitive native dependencies that the slim base image
# omits, and each one fails at import with nothing but a missing-.so message:
# libxrender and libxext are RDKit's drawing libraries, which ADMET-AI pulls in
# via rdkit.Chem.Draw, and libexpat is needed by cuik_molmaker, the featurizer
# extension Chemprop imports unconditionally.
RUN apt-get update \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
      curl \
      libexpat1 \
      libgomp1 \
      libxext6 \
      libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# PyPI's Linux torch wheels, on aarch64 as well as x86_64, depend on roughly 2 GB
# of nvidia-* CUDA wheels; the +cpu build from the index above does not. Fail
# here if the resolver ever prefers the PyPI wheel.
RUN python3 -c "\
import importlib.metadata as md, torch; \
assert torch.__version__.endswith('+cpu'), f'expected a +cpu torch build, got {torch.__version__}'; \
cuda = sorted(d.metadata['Name'] for d in md.distributions() if d.metadata['Name'].lower().startswith('nvidia-')); \
assert not cuda, f'CUDA wheels installed: {cuda}'; \
print('torch', torch.__version__, '(CPU-only, no nvidia-* wheels)')"

# Fail the build rather than the first request if the packaged weights are not
# where ADMET-AI expects them: a runtime discovery of this costs a debugging
# session, a build-time one costs a red build.
RUN python3 -c "\
from admet_ai.constants import DEFAULT_MODELS_DIR, DEFAULT_DRUGBANK_PATH; \
import pathlib; \
models = pathlib.Path(str(DEFAULT_MODELS_DIR)); \
assert models.is_dir(), f'ADMET-AI models directory missing: {models}'; \
ensembles = sorted(p for p in models.iterdir() if p.is_dir()); \
assert ensembles, f'No model ensembles in {models}'; \
assert pathlib.Path(str(DEFAULT_DRUGBANK_PATH)).is_file(), 'DrugBank reference set missing'; \
print(f'ADMET-AI weights present: {len(ensembles)} ensembles')"

COPY app/ ./app/

EXPOSE 8000

# Longer start period than the other services: reading the ten Chemprop
# checkpoints off disk takes about five seconds on top of the usual Python
# start, and /health answers "degraded" rather than refusing connections while
# that happens.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
