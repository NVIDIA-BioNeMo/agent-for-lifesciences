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
# Stage 1: Build Stage
# =============================================================================
FROM python:3.12-slim-bookworm AS build

# Install build dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
      autoconf \
      binutils-dev \
      bison \
      bzip2 \
      ca-certificates \
      check \
      file \
      flex \
      g++ \
      gcc \
      gfortran \
      git \
      gnupg \
      hwloc \
      libatlas-base-dev \
      libblas-dev \
      libbz2-dev \
      libcairo2-dev \
      libeigen3-dev \
      libevent-dev \
      libfreetype-dev \
      libgoogle-glog-dev \
      libjpeg-dev \
      liblapack-dev \
      liblcms2-dev \
      libnetcdf-dev \
      libnuma-dev \
      libopenjp2-7-dev \
      libpng-dev \
      libssl-dev \
      libsqlite3-dev \
      libsubunit0 \
      libsubunit-dev \
      libtiff-dev \
      libtool \
      libwebp-dev \
      make \
      mesa-common-dev \
      openssh-client \
      perl \
      python3-dev \
      python3-pip \
      tar \
      wget \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir numpy pillow

# Parallelism for the C++ builds below. Empty means "derive it from cores and
# memory" (see build-jobs.sh); set it to cap a build by hand.
ARG BUILD_JOBS=""
ENV BUILD_JOBS=${BUILD_JOBS}
COPY build-jobs.sh /usr/local/bin/build-jobs
RUN chmod +x /usr/local/bin/build-jobs

# Build CMake from source
WORKDIR /tmp/cmake-build
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1.tar.gz \
 && tar -xf cmake-3.28.1.tar.gz \
 && cd cmake-3.28.1 \
 && JOBS="$(build-jobs)" \
 && ./bootstrap --parallel="$JOBS" \
 && make -j"$JOBS" \
 && make install \
 && rm -rf /tmp/cmake-build

# Build Boost (bintray is dead, using official archives.boost.io)
WORKDIR /tmp/boost-build
RUN wget --quiet https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.gz \
 && tar -xf boost_1_90_0.tar.gz \
 && cd boost_1_90_0 \
 && ./bootstrap.sh \
      --with-python=python3 \
      --with-libraries=python,serialization,system,iostreams \
      --prefix=/usr/local/boost \
 && ./b2 install -j"$(build-jobs)" \
 && rm -rf /tmp/boost-build

# Build RDKit
ENV RDKit_release="Release_2025_09_3"
ENV RDBASE="/opt/rdkit"
WORKDIR /tmp/rdkit-build
RUN wget --quiet https://github.com/rdkit/rdkit/archive/${RDKit_release}.tar.gz \
 && tar -xf ${RDKit_release}.tar.gz \
 && cd rdkit-${RDKit_release} \
 && mkdir build \
 && cd build \
 && cmake \
      -DCMAKE_INSTALL_PREFIX=/opt/rdkit \
      -DBOOST_ROOT=/usr/local/boost \
      -DRDK_BUILD_COORDGEN_SUPPORT=OFF \
      -DRDK_INSTALL_INTREE=OFF \
      -DRDK_BUILD_INCHI_SUPPORT=ON \
      -DRDK_BUILD_CAIRO_SUPPORT=ON \
      -DRDK_BUILD_CPP_TESTS=OFF \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DPython_EXECUTABLE=$(which python3) \
      .. \
 && make -j"$(build-jobs)" \
 && make install \
 && rm -rf /tmp/rdkit-build

# Collect all shared library dependencies for the runtime stage
# This creates a manifest of required system libraries
RUN mkdir -p /opt/runtime-libs \
 && for lib in /opt/rdkit/lib/*.so* /usr/local/boost/lib/*.so*; do \
      if [ -f "$lib" ]; then \
        ldd "$lib" 2>/dev/null | grep "=> /" | awk '{print $3}' >> /tmp/deps.txt; \
      fi; \
    done \
 && sort -u /tmp/deps.txt > /opt/deps-manifest.txt

# =============================================================================
# Stage 2: Runtime Stage
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

# Install only runtime dependencies (no -dev packages).
#
# Debian builds libquadmath0 only for x86, so requesting it on arm64 fails the
# whole install.
RUN ARCH_PKGS="" \
 && if [ "$(dpkg --print-architecture)" = "amd64" ]; then ARCH_PKGS="libquadmath0"; fi \
 && apt-get update \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends \
      ${ARCH_PKGS} \
      curl \
      libbz2-1.0 \
      libcairo2 \
      libfreetype6 \
      libgoogle-glog0v6 \
      libgomp1 \
      libjpeg62-turbo \
      liblapack3 \
      libblas3 \
      liblcms2-2 \
      libnetcdf19 \
      libnuma1 \
      libopenjp2-7 \
      libpng16-16 \
      libsqlite3-0 \
      libssl3 \
      libtiff6 \
      libwebp7 \
      zlib1g \
      libevent-2.1-7 \
      libevent-pthreads-2.1-7 \
      libgfortran5 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir numpy pillow

# Copy Boost shared libraries
COPY --from=build /usr/local/boost/lib/ /usr/local/boost/lib/

# Copy RDKit installation (includes libs and Python bindings)
COPY --from=build /opt/rdkit/ /opt/rdkit/

# Copy any additional shared libraries that were installed to /usr/local/lib
COPY --from=build /usr/local/lib/ /usr/local/lib/

# Set environment variables
ENV RDBASE="/opt/rdkit"
ENV LD_LIBRARY_PATH="/opt/rdkit/lib:/usr/local/boost/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/rdkit/lib/python3.12/site-packages:${PYTHONPATH}"

# Create symlinks for RDKit data directories (installed to share/RDKit/ but expected at $RDBASE/)
# This enables Fragment descriptors, PAINS filters, and other data-dependent features
RUN ln -s /opt/rdkit/share/RDKit/Data /opt/rdkit/Data \
 && ln -s /opt/rdkit/share/RDKit/Contrib /opt/rdkit/Contrib

# Update library cache
RUN ldconfig

# Verify RDKit installation (including Cairo drawing support)
RUN python3 -c "\
from rdkit import Chem; \
from rdkit.Chem import Draw; \
from rdkit.Chem.Draw import rdMolDraw2D; \
mol = Chem.MolFromSmiles('CCO'); \
print('RDKit import successful:', mol.GetNumAtoms(), 'atoms in ethanol'); \
drawer = rdMolDraw2D.MolDraw2DCairo(300, 300); \
drawer.DrawMolecule(mol); \
drawer.FinishDrawing(); \
print('Cairo drawing support: OK'); \
from PIL import Image; \
print('Pillow import: OK')"

# Tell pip that RDKit is already installed.
#
# RDKit is built from source above into /opt/rdkit and exposed via PYTHONPATH,
# so it carries no pip metadata. Without this stub, any dependency that requires
# `rdkit` (datamol does) makes pip install the PyPI wheel on top -- a second,
# different RDKit shadowed by PYTHONPATH, for ~150MB of dead weight. The
# verification step immediately above is what makes the claim honest.
RUN RDKIT_VER="$(python3 -c 'import rdkit; print(rdkit.__version__)')" \
 && SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
 && DIST="${SITE}/rdkit-${RDKIT_VER}.dist-info" \
 && mkdir -p "${DIST}" \
 && printf 'Metadata-Version: 2.1\nName: rdkit\nVersion: %s\n' "${RDKIT_VER}" > "${DIST}/METADATA" \
 && echo 'aevom-source-build' > "${DIST}/INSTALLER" \
 && : > "${DIST}/RECORD" \
 && python3 -c "import importlib.metadata as m; print('rdkit visible to pip as', m.version('rdkit'))"

WORKDIR /app

# Install Python dependencies for FastAPI app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install safe-mol (SAFE notation) without its declared dependencies.
#
# safe-mol requires torch, transformers, datasets, accelerate, evaluate and
# wandb, and its __init__ imports all of them eagerly -- multiple GB for an
# image that only needs safe.encode/safe.decode. The converter path's real
# dependencies are in requirements.txt above, and vendor/safe_init_lite.py
# replaces the __init__ with a lazy one. Pinned: the shim tracks this version.
COPY vendor/safe_init_lite.py /tmp/safe_init_lite.py
RUN pip install --no-cache-dir --no-deps safe-mol==0.1.14 \
 && SAFE_DIR="$(python3 -c 'import os, site; print(os.path.join(site.getsitepackages()[0], "safe"))')" \
 && test -f "${SAFE_DIR}/converter.py" \
 && cp "${SAFE_DIR}/__init__.py" "${SAFE_DIR}/_init_full.py" \
 && cp /tmp/safe_init_lite.py "${SAFE_DIR}/__init__.py" \
 && rm /tmp/safe_init_lite.py

# Verify SAFE conversion works and that nothing pulled torch back in
RUN python3 -c "\
import sys; \
import safe as sf; \
safe_str = sf.encode('CC(Cc1ccc(cc1)C(C(=O)O)C)C'); \
assert '.' in safe_str, safe_str; \
assert sf.decode(safe_str, canonical=True); \
assert 'torch' not in sys.modules, 'lite safe/__init__.py is not torch-free'; \
print('SAFE import successful (torch-free):', safe_str)"

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
