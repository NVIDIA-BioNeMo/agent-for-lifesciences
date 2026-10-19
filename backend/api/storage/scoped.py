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

"""Session-scoped storage: the only storage request handlers and tools hold.

Every operation is confined to one session. Stores are tagged with it, lists and
searches are filtered by it, and reads, deletes and existence checks only see
artifacts it owns. A reference another session owns behaves exactly like one
that does not exist, so holding (or guessing) a reference is not access.

This is used by both the production chat path (main.py) and the gym-style
training environment to isolate artifacts across browser sessions and
parallel training trajectories.

Example:
    from api.storage.scoped import SessionScopedStorage

    base_storage = DataStorageService(db_session)
    storage = SessionScopedStorage(base_storage, session_id="browser_abc123")

    ref = await storage.store("sdf", sdf_bytes)
    await storage.retrieve(ref)            # ok
    await other_session_storage.retrieve(ref)  # ArtifactNotFoundError
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from api.storage.service import DataStorageService


def scoped_storage(db_session: "AsyncSession", session_id: str) -> "SessionScopedStorage":
    """Storage over ``db_session`` confined to ``session_id``.

    How code outside the storage package gets storage: constructing
    ``DataStorageService`` directly is rejected by
    tests/test_storage_construction_guard.py.
    """
    from api.storage.service import DataStorageService

    return SessionScopedStorage(DataStorageService(db_session), session_id)


async def artifact_owner(db_session: "AsyncSession", reference: str) -> Optional[str]:
    """The session that owns ``reference``, or None. Reads no content."""
    from api.storage.service import DataStorageService

    return await DataStorageService(db_session).owner_of(reference)


class SessionScopedStorage:
    """DataStorageService confined to the artifacts one session owns."""

    def __init__(
        self,
        storage: "DataStorageService",
        session_id: str,
        *,
        read_shared: bool = False,
    ):
        """Bind ``storage`` to ``session_id``.

        Args:
            storage: The unscoped service to delegate to.
            session_id: The owning session. Required: a scope with no session
                would be no scope at all.
            read_shared: Also let reads see artifacts with no session. Only
                training sets this, because the inputs it preloads are shared
                by every rollout of a prompt. Writes are never shared.

        Raises:
            ValueError: If ``session_id`` is empty.
        """
        if not session_id:
            raise ValueError("SessionScopedStorage requires a session_id")
        self._storage = storage
        self._session_id = session_id
        self._read_shared = read_shared

    @property
    def session_id(self) -> str:
        """Get the client session ID for this storage scope."""
        return self._session_id

    async def store(
        self,
        data_type: str,
        content: bytes,
        metadata: dict | None = None,
        ttl_hours: int | None = None,
        description: str | None = None,
    ) -> str:
        """Store data tagged with this session, whatever the metadata says."""
        meta = dict(metadata or {})
        meta["session_id"] = self._session_id
        return await self._storage.store(
            data_type, content, meta, ttl_hours, description
        )

    async def retrieve(self, reference: str) -> bytes:
        """Retrieve data by reference, if this session owns it."""
        return await self._storage.retrieve(
            reference, session_id=self._session_id, include_shared=self._read_shared
        )

    async def retrieve_with_metadata(self, reference: str) -> tuple[bytes, dict]:
        """Retrieve data and metadata by reference, if this session owns it."""
        return await self._storage.retrieve_with_metadata(
            reference, session_id=self._session_id, include_shared=self._read_shared
        )

    async def delete(self, reference: str) -> bool:
        """Delete an artifact, if this session owns it."""
        return await self._storage.delete(reference, session_id=self._session_id)

    async def exists(self, reference: str) -> bool:
        """Check whether an artifact exists and this session can read it."""
        return await self._storage.exists(
            reference, session_id=self._session_id, include_shared=self._read_shared
        )

    def get_data_type(self, reference: str) -> str:
        """Get data type from reference."""
        return self._storage.get_data_type(reference)

    async def commit(self) -> None:
        """Commit pending storage operations to the database."""
        await self._storage.commit()

    async def list_artifacts(self, limit: int = 50) -> list[dict]:
        """List this session's artifacts."""
        return await self._storage.list_artifacts(limit, session_id=self._session_id)

    async def search_artifacts(
        self,
        query: str | None = None,
        data_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search this session's artifacts."""
        return await self._storage.search_artifacts(
            query=query,
            data_type=data_type,
            limit=limit,
            session_id=self._session_id,
        )

    async def export_artifacts(
        self, data_type: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Export this session's artifacts with content."""
        return await self._storage.export_artifacts(
            session_id=self._session_id, data_type=data_type, limit=limit
        )
