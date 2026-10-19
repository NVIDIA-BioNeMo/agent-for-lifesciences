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

# Build the isolated GPU conformer worker environment.
#
# Produces $WORKER_PREFIX (/opt/aevom-conformer-worker): a self-contained conda
# prefix with its own Python, RDKit, CPU torch, the CUDA runtime and nvMolKit,
# and nothing from the Vina parent's preparation stack. Used by the builder
# stage of services/vina/Dockerfile and both training Dockerfiles; only
# $WORKER_PREFIX is copied into the final images.
#
# Two prefixes keep pruning dependency-aware:
#   1. $BUILD_PREFIX holds compilers, headers and the CUDA toolkit. nvMolKit is
#      built there into a wheel.
#   2. $WORKER_PREFIX is created fresh by the solver from runtime packages only,
#      pinned to the exact builds the extension was linked against. Nothing is
#      force-removed.
#
# Requires: bash, curl, git, ca-certificates, bzip2, sha256sum (builder stage).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pins.env
source "$HERE/pins.env"

MACHINE="$(uname -m)"
case "$MACHINE" in
    x86_64)
        MICROMAMBA_ASSET=micromamba-linux-64
        CONDA_COMPILER_SUFFIX=linux-64
        SYSTEM_LIB_DIR=x86_64-linux-gnu
        ;;
    aarch64)
        MICROMAMBA_ASSET=micromamba-linux-aarch64
        MICROMAMBA_SHA256="$MICROMAMBA_SHA256_AARCH64"
        CONDA_COMPILER_SUFFIX=linux-aarch64
        SYSTEM_LIB_DIR=aarch64-linux-gnu
        CUDA_VERSION="$CUDA_VERSION_AARCH64"
        ;;
    *) echo "[build_env] ERROR: unsupported architecture $MACHINE" >&2; exit 1 ;;
esac

JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
MM=/usr/local/bin/micromamba
export MAMBA_ROOT_PREFIX=/opt/micromamba-root
CHANNELS=(-c conda-forge --override-channels)
SRC=/tmp/nvmolkit-src
WHEELS=/tmp/nvmolkit-wheels
INFO_DIR="$WORKER_PREFIX/share/aevom-conformer-worker"

log() { echo "[build_env] $*" >&2; }
die() { echo "[build_env] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "micromamba $MICROMAMBA_VERSION"
curl -fsSL -o "$MM" \
    "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/${MICROMAMBA_ASSET}"
echo "${MICROMAMBA_SHA256}  ${MM}" | sha256sum -c - >/dev/null || die "micromamba checksum mismatch"
chmod +x "$MM"

# ---------------------------------------------------------------------------
log "build prefix $BUILD_PREFIX"
"$MM" create -y -p "$BUILD_PREFIX" "${CHANNELS[@]}" \
    "python=${WORKER_PYTHON_VERSION}" \
    "rdkit=${RDKIT_VERSION}" "librdkit-dev=${RDKIT_VERSION}" \
    libboost-devel libboost-python-devel libboost-headers \
    eigen expat numpy \
    "cuda-version=${CUDA_VERSION}" cuda-nvcc cuda-cudart-dev cuda-nvtx-dev cuda-cccl libcurand-dev \
    "gcc_${CONDA_COMPILER_SUFFIX}=${GCC_VERSION}" "gxx_${CONDA_COMPILER_SUFFIX}=${GCC_VERSION}" \
    cmake make "scikit-build>=0.18" setuptools wheel pip patchelf
# RDKit's CMake config names a static expat that conda-forge does not ship.
[ -e "$BUILD_PREFIX/lib/libexpat.a" ] || ar rcs "$BUILD_PREFIX/lib/libexpat.a"

# ---------------------------------------------------------------------------
log "nvMolKit $NVMOLKIT_REF"
rm -rf "$SRC"
git init -q "$SRC"
git -C "$SRC" fetch -q --depth 1 "$NVMOLKIT_REPO" "$NVMOLKIT_REF"
git -C "$SRC" checkout -q FETCH_HEAD
[ "$(git -C "$SRC" rev-parse HEAD)" = "$NVMOLKIT_REF" ] || die "checked-out commit is not $NVMOLKIT_REF"
grep -qx "version = \"${NVMOLKIT_VERSION}\"" "$SRC/pyproject.toml" \
    || die "pyproject.toml at $NVMOLKIT_REF does not declare version $NVMOLKIT_VERSION"

rm -rf "$WHEELS"
"$MM" run -p "$BUILD_PREFIX" bash -euo pipefail -c "
    cd '$SRC'
    export CONDA_PREFIX='$BUILD_PREFIX'
    export CUDAToolkit_ROOT='$BUILD_PREFIX'
    export CMAKE_BUILD_PARALLEL_LEVEL='$JOBS'
    export NVMOLKIT_CUDA_TARGET_MODE='$NVMOLKIT_CUDA_TARGET_MODE'
    python -m pip wheel . --no-deps --no-build-isolation -w '$WHEELS' -v
"
WHEEL=$(ls "$WHEELS"/nvmolkit-"$NVMOLKIT_VERSION"-*.whl 2>/dev/null || ls "$WHEELS"/nvMolKit-"$NVMOLKIT_VERSION"-*.whl)
[ -f "$WHEEL" ] || die "no nvMolKit $NVMOLKIT_VERSION wheel produced"
log "built $(basename "$WHEEL")"

# The RDKit version compiled into nvMolKit must be the runtime one.
VERSIONS_H=$(find "$SRC/_skbuild" -name versions.h | head -1)
[ -n "$VERSIONS_H" ] || die "generated versions.h not found"
compiled=$(awk '/#define RDKIT_VERSION_(MAJOR|MINOR|PATCH) /{printf "%s.", $3}' "$VERSIONS_H")
compiled="${compiled%.}"
IFS=. read -r want_major want_minor want_patch <<<"$RDKIT_VERSION"
[ "$compiled" = "${want_major}.$((10#$want_minor)).$((10#$want_patch))" ] \
    || die "nvMolKit compiled against RDKit $compiled, expected $RDKIT_VERSION"

# ---------------------------------------------------------------------------
# Exact builds of everything the extension links against, from the build
# prefix: owner package of every library ldd resolves inside $BUILD_PREFIX.
log "resolving linked packages"
EXTRACT=/tmp/nvmolkit-wheel-extract
rm -rf "$EXTRACT" && mkdir -p "$EXTRACT"
"$BUILD_PREFIX/bin/python" -m zipfile -e "$WHEEL" "$EXTRACT"
LINKED_PKGS=$("$BUILD_PREFIX/bin/python" - "$BUILD_PREFIX" "$EXTRACT" <<'PY'
import json, subprocess, sys
from pathlib import Path
prefix, extract = Path(sys.argv[1]), Path(sys.argv[2])
owners = {}
for record in (prefix / "conda-meta").glob("*.json"):
    data = json.loads(record.read_text())
    for f in data.get("files", []):
        owners[str(prefix / f)] = (data["name"], data["version"], data["build"])
env = {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": str(prefix / "lib")}
specs = {}
for so in extract.rglob("*.so*"):
    out = subprocess.run(["ldd", str(so)], capture_output=True, text=True, env=env).stdout
    for line in out.splitlines():
        if "=>" in line:
            target = line.split("=>")[1].split("(")[0].strip()
            if target.startswith(str(prefix)):
                real = str(Path(target).resolve())
                owner = owners.get(target) or owners.get(real)
                if owner:
                    specs[owner[0]] = owner
for name in ("python", "rdkit", "librdkit", "numpy", "cuda-version"):
    for record in (prefix / "conda-meta").glob(f"{name}-[0-9]*.json"):
        data = json.loads(record.read_text())
        if data["name"] == name:
            specs[name] = (data["name"], data["version"], data["build"])
print(" ".join(f"{n}=={v}={b}" for n, v, b in sorted(specs.values())))
PY
)
log "runtime pins: $LINKED_PKGS"

# ---------------------------------------------------------------------------
log "runtime prefix $WORKER_PREFIX"
rm -rf "$WORKER_PREFIX"
# shellcheck disable=SC2086
"$MM" create -y -p "$WORKER_PREFIX" "${CHANNELS[@]}" $LINKED_PKGS cuda-cudart pip
"$WORKER_PREFIX/bin/python" -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"
# nvMolKit's pip requirements (rdkit, nvidia-cuda-runtime-cu12) are satisfied by
# conda's rdkit and cuda-cudart; installing them from PyPI would load a second
# RDKit and a second CUDA runtime.
"$WORKER_PREFIX/bin/python" -m pip install --no-cache-dir --no-deps "$WHEEL"

SITE=$("$WORKER_PREFIX/bin/python" -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
# Point the extension at the runtime prefix rather than the build prefix.
find "$SITE/nvmolkit" -name '*.so*' -type f -print0 | while IFS= read -r -d '' so; do
    "$BUILD_PREFIX/bin/patchelf" --set-rpath "\$ORIGIN/../../.." "$so"
done

# ---------------------------------------------------------------------------
log "native compatibility gate"
for spec in $LINKED_PKGS; do
    name="${spec%%==*}"; want="${spec#*==}"
    got=$("$MM" list -p "$WORKER_PREFIX" --json | "$BUILD_PREFIX/bin/python" -c \
        "import json,sys; print(next((p['version']+'='+p['build_string'] for p in json.load(sys.stdin) if p['name']=='$name'), 'MISSING'))")
    [ "$got" = "$want" ] || die "runtime $name is $got, extension was linked against $want"
done

log "post-prune checks"
check_out=$("$WORKER_PREFIX/bin/python" -I -m pip check 2>&1 || true)
unexpected=$(echo "$check_out" | grep -v -E "^No broken requirements|^nvmolkit [0-9.]+ requires (rdkit|nvidia-cuda-runtime-cu12), which is not installed\.$" || true)
[ -z "$unexpected" ] || die "pip check: $unexpected"

LDD_REPORT=/tmp/aevom-worker-ldd.txt
: >"$LDD_REPORT"
ldd_fail=0
while IFS= read -r -d '' so; do
    out=$(env -u LD_LIBRARY_PATH ldd "$so" 2>&1 || true)
    printf '== %s\n%s\n' "${so#$WORKER_PREFIX/}" "$out" >>"$LDD_REPORT"
    while IFS= read -r line; do
        case "$line" in
            *"libcuda.so.1 => not found"*) ;;  # driver library, injected at runtime
            *"not found"*) echo "unresolved in $so: $line" >&2; ldd_fail=1 ;;
            *"=> /"*)
                target=$(echo "$line" | sed -E 's/.*=> ([^ ]+).*/\1/')
                case "$target" in
                    "$WORKER_PREFIX"/*|/lib/"$SYSTEM_LIB_DIR"/*|/lib64/*|/usr/lib/"$SYSTEM_LIB_DIR"/libc.so*|/usr/lib/"$SYSTEM_LIB_DIR"/libm.so*) ;;
                    *) echo "$so resolves $target outside the worker prefix" >&2; ldd_fail=1 ;;
                esac ;;
        esac
    done <<<"$out"
done < <(find "$SITE/nvmolkit" "$SITE/rdkit" -name '*.so*' -type f -print0)
[ "$ldd_fail" = 0 ] || die "ldd check failed (see above)"

env -i PATH=/usr/bin:/bin "$WORKER_PREFIX/bin/python" -I -c "
import importlib.metadata as m, rdkit, numpy, torch, nvmolkit
from rdkit import rdBase
assert m.version('nvMolKit') == '$NVMOLKIT_VERSION', m.version('nvMolKit')
assert rdBase.rdkitVersion == '$RDKIT_VERSION', rdBase.rdkitVersion
print('imports ok:', rdBase.rdkitVersion, torch.__version__, m.version('nvMolKit'))
"

# ---------------------------------------------------------------------------
log "records"
mkdir -p "$INFO_DIR"
cp "$HERE/pins.env" "$INFO_DIR/pins.env"
cp "$LDD_REPORT" "$INFO_DIR/ldd-report.txt"
"$MM" env export -p "$WORKER_PREFIX" --explicit --md5 >"$INFO_DIR/conda-explicit.txt"
"$WORKER_PREFIX/bin/python" -I -m pip freeze --all >"$INFO_DIR/pip-freeze.txt"
"$WORKER_PREFIX/bin/python" -I "$HERE/inventory.py" "$WORKER_PREFIX" "$INFO_DIR/licenses.json"
PINS_SHA=$(cat "$HERE/pins.env" "$HERE/build_env.sh" "$HERE/inventory.py" | sha256sum | cut -d' ' -f1)
ABI=$("$WORKER_PREFIX/bin/python" -I -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))")
RDKIT_BUILD=$("$MM" list -p "$WORKER_PREFIX" --json | "$WORKER_PREFIX/bin/python" -I -c \
    "import json,sys; print(next(p['build_string'] for p in json.load(sys.stdin) if p['name']=='rdkit'))")
# nvMolKit's CMake names its Boost.Python modules plain "_x.so", so the ABI is
# pinned by the wheel tag and by which libboost_python each module links.
PY_TAG=$("$WORKER_PREFIX/bin/python" -I -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
case "$(basename "$WHEEL")" in *"-$PY_TAG-$PY_TAG-"*) ;; *) die "wheel $(basename "$WHEEL") is not tagged $PY_TAG" ;; esac
BOOST_PY_LINKED=0
for so in $(find "$SITE/nvmolkit" -name '*.so' -type f); do
    needed=$(readelf -d "$so" | sed -n 's/.*NEEDED.*\[\(libboost_python[0-9]*\)\..*/\1/p')
    for lib in $needed; do
        [ "$lib" = "libboost_python${PY_TAG#cp}" ] || die "$so links $lib, not libboost_python${PY_TAG#cp}"
        BOOST_PY_LINKED=$((BOOST_PY_LINKED + 1))
    done
done
[ "$BOOST_PY_LINKED" -gt 0 ] || die "no nvMolKit module links libboost_python${PY_TAG#cp}"
cat >"$INFO_DIR/build-info.json" <<JSON
{
  "pins_sha256": "$PINS_SHA",
  "nvmolkit_ref": "$NVMOLKIT_REF",
  "nvmolkit_version": "$NVMOLKIT_VERSION",
  "nvmolkit_cuda_target_mode": "$NVMOLKIT_CUDA_TARGET_MODE",
  "rdkit_version": "$RDKIT_VERSION",
  "rdkit_build": "$RDKIT_BUILD",
  "python_soabi": "$ABI",
  "machine": "$MACHINE",
  "cuda_version": "$CUDA_VERSION",
  "torch_version": "$TORCH_VERSION",
  "runtime_pins": "$LINKED_PKGS"
}
JSON
cat "$INFO_DIR/build-info.json" >&2

"$MM" clean -a -y >/dev/null
rm -rf "$SRC" "$WHEELS" "$EXTRACT" "$LDD_REPORT"
log "done: $WORKER_PREFIX"
