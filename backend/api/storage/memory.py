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

"""In-memory implementation of the data artifact repository.

This module provides an in-memory backend for DataStorageService,
enabling the real production storage code to run without PostgreSQL.
Used in the training pipeline where artifacts are ephemeral (per-episode)
and don't need persistence.

Example:
    from api.storage.memory import InMemoryDataArtifactRepository
    from api.storage.service import DataStorageService

    repo = InMemoryDataArtifactRepository()
    storage = DataStorageService(repository=repo)

    ref = await storage.store("sdf", sdf_bytes, {"smiles": "CCO"})
    content = await storage.retrieve(ref)
"""

from datetime import datetime
from typing import Any, Optional

from api.db.models import DataArtifact
from .envelope import extract_envelope_fields
from .repository import AbstractDataArtifactRepository


class InMemoryDataArtifactRepository(AbstractDataArtifactRepository):
    """In-memory implementation of the artifact repository.

    Stores artifacts in a plain dict, suitable for training and testing
    where persistence across process restarts is not needed.

    Thread-safety note: This implementation is NOT thread-safe.
    In training, each resource server process is single-threaded
    (async but single-process), so this is fine.
    """

    def __init__(self):
        """Initialize with empty storage."""
        self._artifacts: dict[str, DataArtifact] = {}

    async def create(
        self,
        artifact_id: str,
        data_type: str,
        content: bytes,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        session_id: Optional[str] = None,
    ) -> DataArtifact:
        envelope = extract_envelope_fields(metadata)
        artifact = DataArtifact(
            id=artifact_id,
            data_type=data_type,
            content=content,
            description=description,
            metadata_json=metadata or {},
            session_id=session_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            provenance_json=envelope.get("provenance"),
            units=envelope.get("units"),
            uncertainty=envelope.get("uncertainty"),
            validation_json=envelope.get("validation"),
        )
        self._artifacts[artifact_id] = artifact
        return artifact

    async def get(self, artifact_id: str) -> Optional[DataArtifact]:
        """Get a data artifact by ID.

        Args:
            artifact_id: The artifact reference ID.

        Returns:
            DataArtifact instance or None if not found.
        """
        return self._artifacts.get(artifact_id)

    async def get_for_session(
        self, artifact_id: str, session_id: str, include_shared: bool = False
    ) -> Optional[DataArtifact]:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return None
        owner = getattr(artifact, "session_id", None)
        if owner == session_id or (include_shared and owner is None):
            return artifact
        return None

    async def delete_for_session(self, artifact_id: str, session_id: str) -> bool:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None or getattr(artifact, "session_id", None) != session_id:
            return False
        del self._artifacts[artifact_id]
        return True

    async def delete(self, artifact_id: str) -> bool:
        """Delete a data artifact by ID.

        Args:
            artifact_id: The artifact reference ID.

        Returns:
            True if artifact was deleted, False if not found.
        """
        if artifact_id in self._artifacts:
            del self._artifacts[artifact_id]
            return True
        return False

    async def delete_expired(self) -> int:
        """Delete all expired artifacts.

        Returns:
            Number of artifacts deleted.
        """
        now = datetime.utcnow()
        expired_ids = [
            aid
            for aid, artifact in self._artifacts.items()
            if artifact.expires_at is not None and artifact.expires_at < now
        ]
        for aid in expired_ids:
            del self._artifacts[aid]
        return len(expired_ids)

    async def list_all(self, limit: int = 50, session_id: str | None = None) -> list[DataArtifact]:
        now = datetime.utcnow()
        active = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.expires_at is None or artifact.expires_at > now
        ]
        if session_id is not None:
            active = [a for a in active if getattr(a, "session_id", None) == session_id]
        active.sort(key=lambda a: a.created_at or now, reverse=True)
        return active[:limit]

    async def list_with_content(
        self, session_id: str, data_type: str | None = None, limit: int = 100
    ) -> list[DataArtifact]:
        now = datetime.utcnow()
        active = [
            a for a in self._artifacts.values()
            if (a.expires_at is None or a.expires_at > now)
            and getattr(a, "session_id", None) == session_id
        ]
        if data_type is not None:
            active = [a for a in active if a.data_type == data_type]
        active.sort(key=lambda a: a.created_at or now, reverse=True)
        return active[:limit]

    async def search(
        self,
        query: str | None = None,
        data_type: str | None = None,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[DataArtifact]:
        now = datetime.utcnow()
        active = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.expires_at is None or artifact.expires_at > now
        ]
        if session_id is not None:
            active = [a for a in active if getattr(a, "session_id", None) == session_id]
        if data_type is not None:
            active = [a for a in active if a.data_type == data_type]

        if query:
            needle = query.lower()
            active = [
                a
                for a in active
                if needle in (a.description or "").lower()
                or needle in (a.id or "").lower()
                or needle in (a.data_type or "").lower()
            ]

        active.sort(key=lambda a: a.created_at or now, reverse=True)
        return active[:limit]

    def clear(self) -> None:
        """Remove all artifacts. Useful for cleanup between episodes."""
        self._artifacts.clear()

    def __len__(self) -> int:
        """Get number of stored artifacts."""
        return len(self._artifacts)

    def __contains__(self, artifact_id: str) -> bool:
        """Check if an artifact exists."""
        return artifact_id in self._artifacts
