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

"""The tarball pin file must name the same commit as the submodule gitlink.

A source tarball has no .git, so scripts/fetch-submodules.sh downloads the
commit named in third_party/bionemo-agent-toolkit.commit. If that file drifts
from the gitlink, tarball users silently get different skill prose than a git
clone does.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN_FILE = REPO_ROOT / "third_party" / "bionemo-agent-toolkit.commit"


def _gitlink_commit() -> str:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "HEAD", "third_party/bionemo-agent-toolkit"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    # "160000 commit <sha>\tthird_party/bionemo-agent-toolkit"
    fields = result.stdout.split()
    assert len(fields) >= 3 and fields[1] == "commit", f"no gitlink found: {result.stdout!r}"
    return fields[2]


def test_pin_file_is_a_full_sha():
    pin = PIN_FILE.read_text().strip()
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin), pin


def test_pin_file_matches_gitlink():
    assert PIN_FILE.read_text().strip() == _gitlink_commit()
