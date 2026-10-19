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

"""Unscoped storage is constructed only inside the storage package.

``DataStorageService`` and ``DataArtifactRepository`` read and delete any
session's artifacts. Everything outside ``api/storage`` gets storage through
``scoped_storage(db_session, session_id)`` (or the ``get_scoped_storage`` route
dependency), which confines it to one session. Every route handler used to
build its own unscoped service, which is how a reference alone came to be
enough to read an artifact. This test keeps that from coming back.
"""

import ast
import re
from pathlib import Path

import pytest

# Located from this file, not the repo root: the backend test container mounts
# the package at /app/api with no repo around it.
BACKEND_API = Path(__file__).resolve().parents[1]
REPO = BACKEND_API.parents[1]
UNSCOPED = {"DataStorageService", "DataArtifactRepository"}

# Where constructing them is the point.
ALLOWED_DIRS = [BACKEND_API / "storage"]

# Documented exceptions, as file -> number of constructions. A new one must be
# argued for here, not slipped in.
EXCEPTIONS = {
    # One shared in-memory repo backs every rollout; the service over it is
    # wrapped in SessionScopedStorage(read_shared=True) on the next line.
    REPO / "training" / "nemo_gym_wrapper" / "app.py": 1,
}


def _is_test(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_") or path.name == "conftest.py"


def _scanned_files() -> list[Path]:
    files = [
        path
        for path in BACKEND_API.rglob("*.py")
        if not _is_test(path)
        and not any(path.is_relative_to(allowed) for allowed in ALLOWED_DIRS)
    ]
    return sorted(files + [path for path in EXCEPTIONS if path.exists()])


_CALL_RE = re.compile(r"\b(?:%s)\s*\(" % "|".join(sorted(UNSCOPED)))


def _constructions(path: Path) -> list[int]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # The training wrapper targets a newer Python than this suite runs on
        # (it uses 3.12 f-string syntax). Scan its text instead: that can only
        # overcount, never miss a construction.
        return [
            lineno
            for lineno, line in enumerate(source.splitlines(), start=1)
            if _CALL_RE.search(line)
        ]
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in UNSCOPED:
            lines.append(node.lineno)
    return lines


def test_the_scan_covers_the_route_modules():
    scanned = {
        path.relative_to(BACKEND_API).as_posix()
        for path in _scanned_files()
        if path.is_relative_to(BACKEND_API)
    }
    for module in ["main.py", "artifacts/routes.py", "sandbox/routes.py", "skills/routes.py"]:
        assert module in scanned


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_unscoped_storage_outside_the_storage_package(path):
    found = _constructions(path)
    allowed = EXCEPTIONS.get(path, 0)
    assert len(found) <= allowed, (
        f"{path.relative_to(REPO).as_posix()} constructs unscoped storage at lines {found}. "
        "Use api.storage.scoped_storage(db_session, session_id) instead."
    )


def test_the_guard_notices_a_construction(tmp_path):
    bad = tmp_path / "routes.py"
    bad.write_text(
        "from api.storage import DataStorageService\n"
        "import api.storage.repository as r\n"
        "def handler(session):\n"
        "    DataStorageService(session)\n"
        "    r.DataArtifactRepository(session)\n"
    )
    assert _constructions(bad) == [4, 5]


def test_a_file_this_python_cannot_parse_is_still_scanned(tmp_path):
    unparseable = tmp_path / "app.py"
    unparseable.write_text(
        "def handler(session):\n"
        "    storage = DataStorageService(session)\n"
        "    return (\n"
    )
    assert _constructions(unparseable) == [2]
