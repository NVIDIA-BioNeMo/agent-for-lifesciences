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

"""Which storage endpoints AEVOM_FEAT_STORAGE_ACCESS_API is allowed to switch off.

The flag exists to gate the endpoints that hand back artifact *content*. It must
not gate the metadata listing, because that is what the storage strip hydrates
from on load and after every response. The k8s deployment ships the flag off, so
while the listing was gated the strip stayed permanently empty there -- artifacts
stored fine, the browser just got a 404 that the frontend swallows.

The endpoint bodies are read rather than called: importing api.main pulls in the
whole app, which no test in this suite does.
"""

import ast
from pathlib import Path

import pytest

from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

SESSION = "sess_strip"
OTHER_SESSION = "sess_other"
GUARD = "_require_storage_api"


def functions() -> dict[str, ast.AST]:
    """Map every function in main.py to its syntax tree."""
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }


def calls_the_guard(function: ast.AST) -> bool:
    """Whether the function actually invokes the flag check.

    A substring search would match the docstring that explains the choice.
    """
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == GUARD
        for node in ast.walk(function)
    )


@pytest.fixture
def endpoints():
    return functions()


class TestTheListingIsNotGated:
    def test_the_strip_can_hydrate_with_the_flag_off(self, endpoints):
        assert not calls_the_guard(endpoints["list_storage_artifacts"])

    def test_the_listing_is_still_scoped_to_one_session(self, endpoints):
        """Ungating must not widen what the endpoint returns."""
        assert "Depends(get_scoped_storage)" in ast.unparse(
            endpoints["list_storage_artifacts"]
        )


class TestContentStaysGated:
    def test_single_artifact_content_is_gated(self, endpoints):
        assert calls_the_guard(endpoints["get_artifact_content"])

    def test_export_is_gated(self, endpoints):
        assert calls_the_guard(endpoints["export_artifacts"])

    def test_the_guard_still_refuses_when_disabled(self, endpoints):
        guard = ast.unparse(endpoints[GUARD])
        assert "STORAGE_ACCESS_API_ENABLED" in guard
        assert "404" in guard


@pytest.mark.asyncio
class TestTheListingCarriesOnlyWhatTheStripNeeds:
    """The listing is safe to ungate because of what list_artifacts returns."""

    async def test_it_returns_no_artifact_content(self):
        service = DataStorageService(repository=InMemoryDataArtifactRepository())
        scoped = SessionScopedStorage(service, SESSION)
        await scoped.store("pdb", b"ATOM      1  N   MET A   1", description="EGFR")

        [artifact] = await service.list_artifacts(limit=50, session_id=SESSION)

        assert "content" not in artifact

    async def test_it_carries_the_description_the_strip_renders(self):
        service = DataStorageService(repository=InMemoryDataArtifactRepository())
        scoped = SessionScopedStorage(service, SESSION)
        await scoped.store("sdf", b"aspirin", description="PubChem CID 2244")

        [artifact] = await service.list_artifacts(limit=50, session_id=SESSION)

        assert artifact["description"] == "PubChem CID 2244"

    async def test_another_sessions_artifacts_are_not_listed(self):
        service = DataStorageService(repository=InMemoryDataArtifactRepository())
        await SessionScopedStorage(service, OTHER_SESSION).store("sdf", b"theirs")

        assert await service.list_artifacts(limit=50, session_id=SESSION) == []
