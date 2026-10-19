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

"""Tests for the sandbox store/retrieve endpoints' binary content path.

The store endpoint is JSON, so `array` artifacts (compressed .npz) cannot travel
through the plain `content` field. These tests cover the `content_base64` path
that carries them, and check that the bytes come back byte-identical -- an array
that silently round-trips through UTF-8 would be corrupt but not obviously so.
"""

import base64
import io

import numpy as np
import pytest

from api.sandbox import routes
from api.sandbox.routes import StoreRequest, get_content_type
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

from fastapi import HTTPException


class FakeSandboxSession:
    def __init__(self, client_session_id="sess_1"):
        self.client_session_id = client_session_id


class FakeSessionManager:
    def __init__(self):
        self.session = FakeSandboxSession()
        self.allowed: list[str] = []

    async def get_session(self, token):
        return self.session

    async def validate_access(self, token, reference):
        return self.session

    async def add_allowed_ref(self, token, reference):
        self.allowed.append(reference)


@pytest.fixture
def wired(monkeypatch):
    """Point the routes at in-memory storage and a permissive session manager."""
    manager = FakeSessionManager()
    repo = InMemoryDataArtifactRepository()
    storage = DataStorageService(repository=repo)

    monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        routes,
        "scoped_storage",
        lambda _session, session_id: SessionScopedStorage(storage, session_id),
    )
    return storage


def npz_bytes(array):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.asarray(array))
    return buffer.getvalue()


@pytest.mark.asyncio
class TestStoreBinaryContent:
    async def test_stores_base64_content_as_exact_bytes(self, wired):
        payload = npz_bytes([[1.0, 2.0], [3.0, 4.0]])
        response = await routes.store_sandbox_data(
            StoreRequest(
                data_type="array",
                content_base64=base64.b64encode(payload).decode(),
            ),
            x_sandbox_session="tok",
            session=None,
        )

        assert response.success
        assert response.reference.startswith("array_")
        assert await wired.retrieve(response.reference) == payload

    async def test_the_stored_array_is_still_loadable_by_numpy(self, wired):
        matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
        response = await routes.store_sandbox_data(
            StoreRequest(
                data_type="array",
                content_base64=base64.b64encode(npz_bytes(matrix)).decode(),
            ),
            x_sandbox_session="tok",
            session=None,
        )
        content = await wired.retrieve(response.reference)
        with np.load(io.BytesIO(content)) as archive:
            assert np.allclose(archive["data"], matrix)

    async def test_text_content_still_works(self, wired):
        response = await routes.store_sandbox_data(
            StoreRequest(data_type="smiles", content="CCO"),
            x_sandbox_session="tok",
            session=None,
        )
        assert await wired.retrieve(response.reference) == b"CCO"

    async def test_malformed_base64_is_a_400(self, wired):
        with pytest.raises(HTTPException) as exc:
            await routes.store_sandbox_data(
                StoreRequest(data_type="array", content_base64="not base64!!"),
                x_sandbox_session="tok",
                session=None,
            )
        assert exc.value.status_code == 400

    async def test_neither_field_is_a_400(self, wired):
        with pytest.raises(HTTPException) as exc:
            await routes.store_sandbox_data(
                StoreRequest(data_type="array"),
                x_sandbox_session="tok",
                session=None,
            )
        assert exc.value.status_code == 400

    async def test_base64_wins_when_both_are_supplied(self, wired):
        payload = npz_bytes([1, 2, 3])
        response = await routes.store_sandbox_data(
            StoreRequest(
                data_type="array",
                content="ignored",
                content_base64=base64.b64encode(payload).decode(),
            ),
            x_sandbox_session="tok",
            session=None,
        )
        assert await wired.retrieve(response.reference) == payload

    async def test_the_new_reference_becomes_readable(self, wired):
        # Sandbox code stores an array then loads it back in the same run.
        response = await routes.store_sandbox_data(
            StoreRequest(
                data_type="array",
                content_base64=base64.b64encode(npz_bytes([1])).decode(),
            ),
            x_sandbox_session="tok",
            session=None,
        )
        manager = routes.get_session_manager()
        assert response.reference in manager.allowed


@pytest.mark.asyncio
class TestStorageStaysInTheClientSession:
    async def test_a_store_is_tagged_with_the_client_session(self, wired):
        response = await routes.store_sandbox_data(
            StoreRequest(data_type="smiles", content="CCO", metadata={"session_id": "x"}),
            x_sandbox_session="tok",
            session=None,
        )
        _, metadata = await wired.retrieve_with_metadata(response.reference)
        assert metadata["session_id"] == "sess_1"

    async def test_a_token_without_a_client_session_gets_no_storage(self, wired):
        routes.get_session_manager().session = FakeSandboxSession(client_session_id=None)
        with pytest.raises(HTTPException) as exc:
            await routes.store_sandbox_data(
                StoreRequest(data_type="smiles", content="CCO"),
                x_sandbox_session="tok",
                session=None,
            )
        assert exc.value.status_code == 403

    async def test_an_allowed_ref_from_another_session_is_not_found(self, wired):
        # allowed_refs names what sandbox code may ask for; scoping makes sure
        # the answer still comes from the issuing session.
        foreign = await wired.store("smiles", b"CCO", {"session_id": "sess_2"})
        with pytest.raises(HTTPException) as exc:
            await routes.get_sandbox_data(foreign, x_sandbox_session="tok", session=None)
        assert exc.value.status_code == 404


class TestContentType:
    def test_array_is_served_as_binary(self):
        assert get_content_type("array") == "application/octet-stream"

    def test_structures_keep_their_chemical_mime(self):
        assert get_content_type("pdb") == "chemical/x-pdb"
        assert get_content_type("mmcif") == "chemical/x-mmcif"

    def test_unknown_types_fall_back_to_binary(self):
        assert get_content_type("protein_blob") == "application/octet-stream"
