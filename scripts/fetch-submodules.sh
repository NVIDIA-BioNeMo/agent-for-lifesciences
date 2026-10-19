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
# Populate third_party/bionemo-agent-toolkit.
#
# In a git clone this is `git submodule update`. A source tarball has no .git,
# so there is no gitlink to read the pinned commit from; the pin is also kept in
# third_party/bionemo-agent-toolkit.commit (test_submodule_pin.py holds the two
# equal), and the matching GitHub archive is downloaded instead.
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$PROJECT_ROOT/third_party/bionemo-agent-toolkit"
PIN_FILE="$PROJECT_ROOT/third_party/bionemo-agent-toolkit.commit"
REPO_URL="https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit"

apply_toolkit_patches() {
    local patch_file="$PROJECT_ROOT/third_party/patches/bionemo-agent-toolkit-prune-orphans-copyright.patch"
    if ! command -v patch >/dev/null 2>&1; then
        echo "The patch command is required for the toolkit copyright correction." >&2
        exit 1
    fi
    if patch --dry-run --forward --silent -p1 -d "$SUBMODULE_DIR" < "$patch_file" >/dev/null 2>&1; then
        patch --forward --silent -p1 -d "$SUBMODULE_DIR" < "$patch_file"
        echo "Applied BioNeMo toolkit copyright correction."
    elif patch --dry-run --reverse --silent -p1 -d "$SUBMODULE_DIR" < "$patch_file" >/dev/null 2>&1; then
        echo "BioNeMo toolkit copyright correction is already applied."
    else
        echo "Toolkit copyright patch does not match this checkout; review its header before continuing." >&2
        exit 1
    fi
}

# This checkout's own .git (a directory, or a file in a worktree) -- not
# `git rev-parse`, which also succeeds for a tarball unpacked inside some other
# repository.
if [ -e "$PROJECT_ROOT/.git" ]; then
    git -C "$PROJECT_ROOT" submodule update --init --recursive
    apply_toolkit_patches
    echo "Submodules up to date."
    exit 0
fi

if [ -d "$SUBMODULE_DIR/nim-skills" ]; then
    apply_toolkit_patches
    echo "third_party/bionemo-agent-toolkit is already populated."
    exit 0
fi

commit="$(tr -d '[:space:]' < "$PIN_FILE")"
case "$commit" in
    *[!0-9a-f]*|"")
        echo "Unreadable pin in $PIN_FILE: '$commit'" >&2
        exit 1
        ;;
esac

echo "No git checkout found; downloading bionemo-agent-toolkit at $commit..."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! curl -fsSL --retry 3 "$REPO_URL/archive/$commit.tar.gz" -o "$tmp/toolkit.tar.gz"; then
    echo "Download failed: $REPO_URL/archive/$commit.tar.gz" >&2
    echo "Check your network connection (or proxy) and re-run 'make submodules'." >&2
    exit 1
fi

mkdir -p "$tmp/extract"
tar -xzf "$tmp/toolkit.tar.gz" -C "$tmp/extract"
src="$(find "$tmp/extract" -mindepth 1 -maxdepth 1 -type d | head -n1)"
if [ ! -d "$src/nim-skills" ]; then
    echo "The downloaded archive has no nim-skills/ directory; is the pin correct?" >&2
    exit 1
fi

mkdir -p "$SUBMODULE_DIR"
cp -R "$src/." "$SUBMODULE_DIR/"
apply_toolkit_patches
echo "Populated third_party/bionemo-agent-toolkit at $commit."
