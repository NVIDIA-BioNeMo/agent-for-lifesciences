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

"""Repository for data artifact storage operations.

This module provides database access for storing and retrieving
tool data artifacts using the repository pattern.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import DataArtifact
from .envelope import extract_envelope_fields


class AbstractDataArtifactRepository(ABC):
    """Abstract interface for data artifact repository operations."""
    
    @abstractmethod
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
        """Create a new data artifact."""
        pass
    
    @abstractmethod
    async def get(self, artifact_id: str) -> Optional[DataArtifact]:
        """Get a data artifact by ID, whichever session owns it."""
        pass

    @abstractmethod
    async def get_for_session(
        self, artifact_id: str, session_id: str, include_shared: bool = False
    ) -> Optional[DataArtifact]:
        """Get a data artifact by ID only if ``session_id`` owns it.

        ``include_shared`` also admits rows with no session, which only
        training uses: its preloaded inputs are shared by every rollout.
        """
        pass

    @abstractmethod
    async def delete(self, artifact_id: str) -> bool:
        """Delete a data artifact by ID, whichever session owns it."""
        pass

    @abstractmethod
    async def delete_for_session(self, artifact_id: str, session_id: str) -> bool:
        """Delete a data artifact by ID only if ``session_id`` owns it."""
        pass
    
    @abstractmethod
    async def delete_expired(self) -> int:
        """Delete all expired artifacts. Returns count deleted."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: int = 50, session_id: str | None = None) -> list[DataArtifact]:
        """List all non-expired artifacts, optionally filtered by session_id."""
        pass

    @abstractmethod
    async def list_with_content(
        self, session_id: str, data_type: str | None = None, limit: int = 100
    ) -> list[DataArtifact]:
        """List artifacts with full content, filtered by session_id."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str | None = None,
        data_type: str | None = None,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[DataArtifact]:
        """Find artifacts by description, reference, or type (metadata only)."""
        pass


class DataArtifactRepository(AbstractDataArtifactRepository):
    """SQLAlchemy implementation of data artifact repository.
    
    Provides CRUD operations for data artifacts stored in PostgreSQL.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session.
        
        Args:
            session: SQLAlchemy async session.
        """
        self._session = session
    
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
            expires_at=expires_at,
            provenance_json=envelope.get("provenance"),
            units=envelope.get("units"),
            uncertainty=envelope.get("uncertainty"),
            validation_json=envelope.get("validation"),
        )
        self._session.add(artifact)
        await self._session.flush()
        await self._session.refresh(artifact)
        return artifact
    
    async def get(self, artifact_id: str) -> Optional[DataArtifact]:
        """Get a data artifact by ID.
        
        Args:
            artifact_id: The artifact reference ID.
            
        Returns:
            DataArtifact instance or None if not found.
        """
        result = await self._session.execute(
            select(DataArtifact).where(DataArtifact.id == artifact_id)
        )
        return result.scalar_one_or_none()

    async def get_for_session(
        self, artifact_id: str, session_id: str, include_shared: bool = False
    ) -> Optional[DataArtifact]:
        owner = DataArtifact.session_id == session_id
        if include_shared:
            owner = or_(owner, DataArtifact.session_id.is_(None))
        result = await self._session.execute(
            select(DataArtifact).where(DataArtifact.id == artifact_id, owner)
        )
        return result.scalar_one_or_none()

    async def delete(self, artifact_id: str) -> bool:
        """Delete a data artifact by ID.
        
        Args:
            artifact_id: The artifact reference ID.
            
        Returns:
            True if artifact was deleted, False if not found.
        """
        result = await self._session.execute(
            delete(DataArtifact).where(DataArtifact.id == artifact_id)
        )
        await self._session.flush()
        return result.rowcount > 0

    async def delete_for_session(self, artifact_id: str, session_id: str) -> bool:
        result = await self._session.execute(
            delete(DataArtifact).where(
                DataArtifact.id == artifact_id, DataArtifact.session_id == session_id
            )
        )
        await self._session.flush()
        return result.rowcount > 0
    
    async def delete_expired(self) -> int:
        """Delete all expired artifacts.
        
        Returns:
            Number of artifacts deleted.
        """
        result = await self._session.execute(
            delete(DataArtifact).where(
                DataArtifact.expires_at.isnot(None),
                DataArtifact.expires_at < datetime.utcnow(),
            )
        )
        await self._session.flush()
        return result.rowcount
    
    async def list_all(self, limit: int = 50, session_id: str | None = None) -> list[DataArtifact]:
        now = datetime.utcnow()
        query = select(DataArtifact).where(
            or_(DataArtifact.expires_at.is_(None), DataArtifact.expires_at > now)
        )
        if session_id is not None:
            query = query.where(DataArtifact.session_id == session_id)
        query = query.order_by(DataArtifact.created_at.desc()).limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def list_with_content(
        self, session_id: str, data_type: str | None = None, limit: int = 100
    ) -> list[DataArtifact]:
        """Return full DataArtifact objects (including content) in a single query."""
        now = datetime.utcnow()
        query = select(DataArtifact).where(
            DataArtifact.session_id == session_id,
            or_(DataArtifact.expires_at.is_(None), DataArtifact.expires_at > now),
        )
        if data_type is not None:
            query = query.where(DataArtifact.data_type == data_type)
        query = query.order_by(DataArtifact.created_at.desc()).limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def search(
        self,
        query: str | None = None,
        data_type: str | None = None,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[DataArtifact]:
        """Find artifacts by description, reference, or type.

        Content is loaded along with the row -- the column is on the same
        table -- but callers are expected to read metadata only. Pulling
        payloads into the prompt is the thing the reference pattern exists to
        avoid.
        """
        now = datetime.utcnow()
        statement = select(DataArtifact).where(
            or_(DataArtifact.expires_at.is_(None), DataArtifact.expires_at > now)
        )
        if session_id is not None:
            statement = statement.where(DataArtifact.session_id == session_id)
        if data_type is not None:
            statement = statement.where(DataArtifact.data_type == data_type)

        if query:
            pattern = f"%{query}%"
            statement = statement.where(
                or_(
                    DataArtifact.description.ilike(pattern),
                    DataArtifact.id.ilike(pattern),
                    DataArtifact.data_type.ilike(pattern),
                )
            )

        statement = statement.order_by(DataArtifact.created_at.desc()).limit(limit)
        result = await self._session.execute(statement)
        return list(result.scalars().all())



