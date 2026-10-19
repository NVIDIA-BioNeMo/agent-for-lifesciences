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

"""``POST /api/sandbox/selection``: residue selections built for sandbox code.

Python run in the sandbox has coordinates but no registry and no viewer. This
route is how it makes a selection other tools can use, so what matters is that
the artifact is the same one a user pick or ``select_residues`` produces.
"""

import json

import pytest
import pytest_asyncio
from fastapi import HTTPException

from api.sandbox import routes
from api.sandbox.routes import SelectionSandboxRequest
from api.sandbox.session import ReferenceNotAllowedError
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService
from api.selections.registry import get_available_selections
from api.tests.test_selection_ops import CIF
from api.tests.test_user_selections import FakeRedis


class FakeSandboxSession:
    client_session_id = "sess_1"


class FakeSessionManager:
    def __init__(self):
        self.allowed: list[str] = []
        self.deny = False

    async def validate_access(self, token, reference):
        if self.deny:
            raise ReferenceNotAllowedError(reference)
        return FakeSandboxSession()

    async def add_allowed_ref(self, token, reference):
        self.allowed.append(reference)


class FakeDb:
    commits = 0

    async def commit(self):
        self.commits += 1


@pytest.fixture
def redis():
    return FakeRedis()


@pytest_asyncio.fixture
async def wired(monkeypatch, redis):
    manager = FakeSessionManager()
    storage = DataStorageService(repository=InMemoryDataArtifactRepository())
    monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        routes,
        "scoped_storage",
        lambda _session, session_id: SessionScopedStorage(storage, session_id),
    )
    monkeypatch.setattr(routes, "_redis_or_none", lambda: redis)
    structure_ref = await storage.store("mmcif", CIF.encode(), {"session_id": "sess_1"})
    return manager, storage, structure_ref


def request(structure_ref, **overrides):
    body = {"structure_ref": structure_ref, "chain": "A", "residues": [56, "115"]}
    body.update(overrides)
    return SelectionSandboxRequest(**body)


async def call(req):
    return await routes.sandbox_store_selection(req, "token", FakeDb())


@pytest.mark.asyncio
class TestSandboxSelection:
    async def test_builds_and_stores_a_selection(self, wired):
        manager, storage, structure_ref = wired
        response = await call(request(structure_ref, name="epitope"))
        assert response.success
        assert (response.residue_count, response.missing) == (2, [])
        data = json.loads(await storage.retrieve(response.reference))
        assert data["provenance"] == "agent"
        assert data["name"] == "epitope"
        assert {a["residue"]["authSeqNumber"] for a in data["atoms"]} == {56, 115}

    async def test_per_residue_atoms_make_an_atom_selection(self, wired):
        _, storage, structure_ref = wired
        response = await call(request(structure_ref, residues=["56:CA,CG", 115]))
        assert response.success
        data = json.loads(await storage.retrieve(response.reference))
        assert data["scope"] == "atom"
        assert sorted(a["name"] for a in data["atoms"] if a["residue"]["authSeqNumber"] == 56) == ["CA", "CG"]

    async def test_a_keyword_is_accepted(self, wired):
        _, storage, structure_ref = wired
        response = await call(request(structure_ref, atoms="backbone"))
        data = json.loads(await storage.retrieve(response.reference))
        assert {a["name"] for a in data["atoms"]} <= {"N", "CA", "C", "O"}

    async def test_whole_residues_stay_residue_scoped(self, wired):
        _, storage, structure_ref = wired
        response = await call(request(structure_ref))
        assert json.loads(await storage.retrieve(response.reference))["scope"] == "residue"

    async def test_an_atom_that_is_not_there_is_reported(self, wired):
        _, _, structure_ref = wired
        response = await call(request(structure_ref, residues=["56:CA,NZ"]))
        assert response.success
        assert response.missing == ["56:NZ"]

    async def test_is_listed_in_the_sessions_selection_state(self, wired, redis):
        _, _, structure_ref = wired
        response = await call(request(structure_ref, name="epitope"))
        (info,) = await get_available_selections(redis, "sess_1")
        assert (info.reference, info.provenance, info.name) == (
            response.reference,
            "agent",
            "epitope",
        )

    async def test_the_session_may_use_the_new_reference(self, wired):
        manager, _, structure_ref = wired
        response = await call(request(structure_ref))
        assert manager.allowed == [response.reference]

    async def test_missing_residues_are_reported(self, wired):
        _, _, structure_ref = wired
        response = await call(request(structure_ref, residues=[56, 9999]))
        assert response.success
        assert response.missing == ["9999"]

    async def test_nothing_found_is_a_failure_not_an_empty_selection(self, wired):
        manager, _, structure_ref = wired
        response = await call(request(structure_ref, residues=[9999]))
        assert not response.success
        assert "9999" in response.error
        assert manager.allowed == []

    async def test_bad_input_is_reported_in_the_response(self, wired):
        _, _, structure_ref = wired
        response = await call(request(structure_ref, chain="Z"))
        assert not response.success
        assert "Chain 'Z'" in response.error

    async def test_a_reference_the_session_may_not_read_is_refused(self, wired):
        manager, _, structure_ref = wired
        manager.deny = True
        with pytest.raises(HTTPException) as caught:
            await call(request(structure_ref))
        assert caught.value.status_code == 403

    async def test_an_unknown_reference_is_a_404(self, wired):
        with pytest.raises(HTTPException) as caught:
            await call(request("mmcif_nope0000"))
        assert caught.value.status_code in (400, 404)
