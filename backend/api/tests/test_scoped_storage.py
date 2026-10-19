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

"""Tests for SessionScopedStorage.

Every operation is confined to one session: stores are tagged with it, lists
and searches filter by it, and reads, deletes and existence checks only see
artifacts it owns. Another session's reference behaves like a missing one --
references are short enough to guess, so holding one must not be access.
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from api.storage import ArtifactNotFoundError
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

MINE = "sess_mine"
THEIRS = "sess_theirs"


@pytest.fixture
def base():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def mine(base):
    return SessionScopedStorage(base, MINE)


@pytest.fixture
def theirs(base):
    return SessionScopedStorage(base, THEIRS)


class TestConstruction:
    @pytest.mark.parametrize("session_id", [None, ""])
    def test_a_scope_needs_a_session(self, session_id):
        with pytest.raises(ValueError):
            SessionScopedStorage(MagicMock(), session_id)

    def test_session_id_property(self):
        assert SessionScopedStorage(MagicMock(), MINE).session_id == MINE


class TestStore:
    @pytest.mark.asyncio
    async def test_store_injects_session_id(self, mock_storage):
        scoped = SessionScopedStorage(mock_storage, session_id="sess_abc")
        await scoped.store("sdf", b"data", metadata={"name": "mol1"})
        mock_storage.store.assert_called_once_with(
            "sdf", b"data", {"name": "mol1", "session_id": "sess_abc"}, None, None
        )

    @pytest.mark.asyncio
    async def test_store_creates_metadata_if_none(self, mock_storage):
        scoped = SessionScopedStorage(mock_storage, session_id="sess_abc")
        await scoped.store("pdb", b"pdb data")
        mock_storage.store.assert_called_once_with(
            "pdb", b"pdb data", {"session_id": "sess_abc"}, None, None
        )

    @pytest.mark.asyncio
    async def test_metadata_cannot_claim_another_session(self, mine, theirs):
        ref = await mine.store("sdf", b"data", metadata={"session_id": THEIRS})
        assert await mine.retrieve(ref) == b"data"
        with pytest.raises(ArtifactNotFoundError):
            await theirs.retrieve(ref)

    @pytest.mark.asyncio
    async def test_store_does_not_mutate_the_callers_metadata(self, mine):
        metadata = {"name": "mol1"}
        await mine.store("sdf", b"data", metadata=metadata)
        assert metadata == {"name": "mol1"}


class TestOwnership:
    @pytest.mark.asyncio
    async def test_the_owner_can_read_its_artifact(self, mine):
        ref = await mine.store("sdf", b"mine")
        assert await mine.retrieve(ref) == b"mine"
        content, metadata = await mine.retrieve_with_metadata(ref)
        assert content == b"mine"
        assert metadata["session_id"] == MINE
        assert await mine.exists(ref)

    @pytest.mark.asyncio
    async def test_another_session_cannot_retrieve(self, mine, theirs):
        ref = await mine.store("sdf", b"mine")
        with pytest.raises(ArtifactNotFoundError):
            await theirs.retrieve(ref)
        with pytest.raises(ArtifactNotFoundError):
            await theirs.retrieve_with_metadata(ref)

    @pytest.mark.asyncio
    async def test_another_session_sees_it_as_missing(self, mine, theirs):
        ref = await mine.store("sdf", b"mine")
        assert not await theirs.exists(ref)

    @pytest.mark.asyncio
    async def test_another_session_cannot_delete(self, mine, theirs):
        ref = await mine.store("sdf", b"mine")
        assert await theirs.delete(ref) is False
        assert await mine.retrieve(ref) == b"mine"
        assert await mine.delete(ref) is True
        assert not await mine.exists(ref)

    @pytest.mark.asyncio
    async def test_lists_and_searches_see_only_the_owner(self, mine, theirs):
        await mine.store("sdf", b"mine", description="aspirin")
        await theirs.store("sdf", b"theirs", description="aspirin")
        assert [a["metadata"]["session_id"] for a in await mine.list_artifacts()] == [MINE]
        found = await mine.search_artifacts(query="aspirin")
        assert [a["metadata"]["session_id"] for a in found] == [MINE]
        exported = await mine.export_artifacts()
        assert [a["content"] for a in exported] == ["mine"]


class TestSharedRows:
    """Rows with no session belong to nobody, so a plain scope cannot read them."""

    @pytest_asyncio.fixture
    async def shared_ref(self, base):
        ref = "pdb_shared01"
        await base._repository.create(artifact_id=ref, data_type="pdb", content=b"shared")
        return ref

    @pytest.mark.asyncio
    async def test_a_plain_scope_cannot_read_them(self, mine, shared_ref):
        with pytest.raises(ArtifactNotFoundError):
            await mine.retrieve(shared_ref)
        assert not await mine.exists(shared_ref)

    @pytest.mark.asyncio
    async def test_read_shared_admits_them_for_reading_only(self, base, shared_ref):
        training = SessionScopedStorage(base, MINE, read_shared=True)
        assert await training.retrieve(shared_ref) == b"shared"
        assert await training.exists(shared_ref)
        assert await training.delete(shared_ref) is False

    @pytest.mark.asyncio
    async def test_read_shared_still_refuses_other_sessions(self, base, theirs):
        ref = await theirs.store("sdf", b"theirs")
        training = SessionScopedStorage(base, MINE, read_shared=True)
        with pytest.raises(ArtifactNotFoundError):
            await training.retrieve(ref)


class TestDelegation:
    @pytest.mark.asyncio
    async def test_list_delegates_session_id(self, mock_storage):
        scoped = SessionScopedStorage(mock_storage, session_id="sess_abc")
        artifacts = await scoped.list_artifacts()
        mock_storage.list_artifacts.assert_called_once_with(50, session_id="sess_abc")
        assert artifacts[0]["reference"] == "sdf_test1234"

    @pytest.mark.asyncio
    async def test_list_custom_limit(self, mock_storage):
        scoped = SessionScopedStorage(mock_storage, session_id="sess_abc")
        await scoped.list_artifacts(limit=10)
        mock_storage.list_artifacts.assert_called_once_with(10, session_id="sess_abc")

    @pytest.mark.asyncio
    async def test_reads_pass_the_session(self, mock_storage):
        scoped = SessionScopedStorage(mock_storage, session_id="sess_abc")
        await scoped.retrieve("sdf_test1234")
        mock_storage.retrieve.assert_called_once_with(
            "sdf_test1234", session_id="sess_abc", include_shared=False
        )
        await scoped.delete("sdf_test1234")
        mock_storage.delete.assert_called_once_with("sdf_test1234", session_id="sess_abc")

    def test_get_data_type_delegates(self, mock_storage):
        scoped = SessionScopedStorage(mock_storage, session_id="sess_abc")
        assert scoped.get_data_type("sdf_test1234") == "sdf"
        mock_storage.get_data_type.assert_called_once_with("sdf_test1234")
