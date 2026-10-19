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

"""Data storage service for tool artifacts.

This service provides the primary interface for tools to store and retrieve
data artifacts. It generates typed reference IDs and manages the lifecycle
of stored data.

Example:
    # Storing data from a tool
    storage = DataStorageService(session)
    ref = await storage.store("sdf", sdf_content.encode(), {"smiles": "CCO"})
    # Returns: "sdf_a1b2c3d4"

    # Retrieving data in another tool
    content = await storage.retrieve(ref)
    # Returns: b"..."
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .datatypes import DataType, is_text_type
from .repository import DataArtifactRepository

if TYPE_CHECKING:
    from .repository import AbstractDataArtifactRepository

logger = logging.getLogger(__name__)

REFERENCE_ID_LENGTH = 8


def generate_reference_id(
    data_type: "str | DataType", length: int = REFERENCE_ID_LENGTH
) -> str:
    """Generate a unique typed reference ID in production format.

    Used by both production storage (DataStorageService.store) and the
    training dataset adapter to ensure consistent ref naming.

    Args:
        data_type: Type prefix for the reference (e.g., "sdf", "pdb"). A
            ``DataType`` member is accepted as well as its bare string value.
        length: Length of the random hex portion (default 8).

    Returns:
        Reference ID in format "{data_type}_{random_hex}",
        e.g. "sdf_a1b2c3d4".
    """
    # `DataType` is a str enum, but from Python 3.11 on an f-string formats a
    # member as "DataType.PNG" rather than "png". Every caller today passes a
    # bare string, so this only guards the next one -- but it guards it against a
    # reference that parses and stores and then cannot be served, since
    # /api/images and the reference detector both match on the type prefix.
    prefix = data_type.value if isinstance(data_type, DataType) else data_type
    random_part = secrets.token_hex(length // 2)
    return f"{prefix}_{random_part}"


_UNDERSCORED_TYPE_NAMES: tuple[str, ...] = tuple(
    sorted((dt.value for dt in DataType if "_" in dt.value), key=len, reverse=True)
)


class DataStorageError(Exception):
    """Base exception for data storage errors."""

    pass


class ArtifactNotFoundError(DataStorageError):
    """Raised when a requested artifact is not found."""

    pass


class InvalidReferenceError(DataStorageError):
    """Raised when a reference ID is malformed or invalid."""

    pass


class DataStorageService:
    """Service for storing and retrieving tool data artifacts.

    This service manages the lifecycle of data artifacts that are passed
    between tools via references. Instead of passing raw data through the
    LLM, tools store data and pass lightweight reference IDs.

    Reference IDs are typed and formatted as: "{data_type}_{random_hex}"
    For example: "sdf_a1b2c3d4", "pdb_e5f6g7h8"

    Attributes:
        REFERENCE_ID_LENGTH: Length of the random portion of reference IDs.
        DEFAULT_TTL_HOURS: Default time-to-live for artifacts (None = no expiry).
    """

    REFERENCE_ID_LENGTH = 8
    DEFAULT_TTL_HOURS: Optional[int] = None  # No expiry by default

    def __init__(
        self,
        session: AsyncSession = None,
        *,
        repository: "AbstractDataArtifactRepository | None" = None,
    ):
        """Initialize the storage service.

        Accepts either a SQLAlchemy session (production, creates a
        DataArtifactRepository) or an explicit repository instance
        (training/testing, e.g. InMemoryDataArtifactRepository).

        Args:
            session: SQLAlchemy async session for database operations.
            repository: Pre-built repository instance (takes precedence).

        Raises:
            ValueError: If neither session nor repository is provided.
        """
        if repository is not None:
            self._repository = repository
        elif session is not None:
            self._repository = DataArtifactRepository(session)
        else:
            raise ValueError("Either session or repository must be provided")

    def _generate_reference_id(self, data_type: str) -> str:
        """Generate a unique typed reference ID.

        Args:
            data_type: Type of data (e.g., "sdf", "pdb").

        Returns:
            Reference ID in format "{data_type}_{random_hex}".
        """
        return generate_reference_id(data_type, self.REFERENCE_ID_LENGTH)

    def _parse_reference_id(self, reference: str) -> tuple[str, str]:
        """Parse a reference ID into its components.

        Args:
            reference: Reference ID to parse.

        Returns:
            Tuple of (data_type, random_part).

        Raises:
            InvalidReferenceError: If reference format is invalid.
        """
        # Type names may contain underscores ("annotated_cif") and so may the
        # id part of hand-written refs ("sdf_winner_001"), so neither the first
        # nor the last underscore is the boundary in general.
        for type_name in _UNDERSCORED_TYPE_NAMES:
            prefix = f"{type_name}_"
            if reference.startswith(prefix) and len(reference) > len(prefix):
                return type_name, reference[len(prefix):]

        parts = reference.split("_", 1)
        if len(parts) != 2:
            raise InvalidReferenceError(
                f"Invalid reference format: '{reference}'. Expected 'type_id' format."
            )
        return parts[0], parts[1]

    async def commit(self) -> None:
        """Commit the underlying database session.

        Called between graph iterations to make stored artifacts visible
        to other DB sessions (sandbox HTTP, frontend polling).
        """
        if hasattr(self._repository, "_session"):
            await self._repository._session.commit()

    async def store(
        self,
        data_type: "str | DataType",
        content: bytes,
        metadata: Optional[dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store data and return a reference ID.

        Args:
            data_type: Type of data being stored (e.g., "sdf", "pdb", "mol2").
                A ``DataType`` member is accepted as well as its bare value.
            content: Binary content to store.
            metadata: Optional metadata about the artifact.
            ttl_hours: Hours until expiration (None = use default, 0 = no expiry).
            description: Optional short description of the artifact (LLM-provided).

        Returns:
            Reference ID that can be used to retrieve the data.

        Example:
            ref = await storage.store("sdf", sdf_bytes, {"smiles": "CCO"}, description="Aspirin molecule")
            # Returns: "sdf_a1b2c3d4"
        """
        data_type = data_type.value if isinstance(data_type, DataType) else data_type
        reference_id = self._generate_reference_id(data_type)

        # Calculate expiration time
        expires_at = None
        effective_ttl = ttl_hours if ttl_hours is not None else self.DEFAULT_TTL_HOURS
        if effective_ttl and effective_ttl > 0:
            expires_at = datetime.utcnow() + timedelta(hours=effective_ttl)

        session_id = (metadata or {}).get("session_id")
        await self._repository.create(
            artifact_id=reference_id,
            data_type=data_type,
            content=content,
            metadata=metadata,
            description=description,
            expires_at=expires_at,
            session_id=session_id,
        )

        logger.info(
            f"Stored {data_type} artifact: {reference_id} "
            f"({len(content)} bytes, expires: {expires_at}, description: {description})"
        )

        return reference_id

    async def _lookup(
        self, reference: str, session_id: Optional[str], include_shared: bool
    ) -> Any:
        """Load an artifact, restricted to ``session_id`` when one is given.

        A row another session owns is indistinguishable from a missing one, so
        a caller cannot probe for references that exist elsewhere.
        """
        if session_id is None:
            return await self._repository.get(reference)
        return await self._repository.get_for_session(reference, session_id, include_shared)

    async def _delete(self, reference: str, session_id: Optional[str]) -> bool:
        if session_id is None:
            return await self._repository.delete(reference)
        return await self._repository.delete_for_session(reference, session_id)

    async def _live_artifact(
        self, reference: str, session_id: Optional[str], include_shared: bool
    ) -> Any:
        artifact = await self._lookup(reference, session_id, include_shared)

        if artifact is None:
            raise ArtifactNotFoundError(f"Artifact not found: {reference}")

        if artifact.expires_at and artifact.expires_at < datetime.utcnow():
            await self._delete(reference, artifact.session_id)
            raise ArtifactNotFoundError(f"Artifact expired: {reference}")

        return artifact

    async def owner_of(self, reference: str) -> Optional[str]:
        """The session that owns ``reference``, without reading its content.

        None for a missing, expired or shared artifact. This is the one lookup
        that ignores scope: it lets a route that receives a bare reference
        (``/api/images``) find whose ownership to check before reading.
        """
        try:
            artifact = await self._live_artifact(reference, None, False)
        except ArtifactNotFoundError:
            return None
        return artifact.session_id

    async def retrieve(
        self,
        reference: str,
        *,
        session_id: Optional[str] = None,
        include_shared: bool = False,
    ) -> bytes:
        """Retrieve data by reference ID.

        Request handlers and tools never call this directly: they hold a
        :class:`~api.storage.scoped.SessionScopedStorage`, which always passes
        its ``session_id``. Without one the lookup spans every session.

        Args:
            reference: Reference ID from a previous store operation.
            session_id: Only return the artifact if this session owns it.
            include_shared: With ``session_id``, also admit rows with no session.

        Returns:
            Binary content of the stored artifact.

        Raises:
            ArtifactNotFoundError: If the artifact is not found (or not owned).
            InvalidReferenceError: If the reference format is invalid.

        Example:
            content = await storage.retrieve("sdf_a1b2c3d4", session_id=sid)
        """
        data_type, _ = self._parse_reference_id(reference)
        artifact = await self._live_artifact(reference, session_id, include_shared)
        logger.info(f"Retrieved {data_type} artifact: {reference}")
        return artifact.content

    async def retrieve_with_metadata(
        self,
        reference: str,
        *,
        session_id: Optional[str] = None,
        include_shared: bool = False,
    ) -> tuple[bytes, dict[str, Any]]:
        """Retrieve data and metadata by reference ID.

        Scoping is as for :meth:`retrieve`.

        Returns:
            Tuple of (content, metadata).

        Raises:
            ArtifactNotFoundError: If the artifact is not found (or not owned).
            InvalidReferenceError: If the reference format is invalid.
        """
        data_type, _ = self._parse_reference_id(reference)
        artifact = await self._live_artifact(reference, session_id, include_shared)
        logger.info(f"Retrieved {data_type} artifact with metadata: {reference}")
        return artifact.content, artifact.metadata_json

    async def delete(self, reference: str, *, session_id: Optional[str] = None) -> bool:
        """Delete an artifact by reference ID.

        Args:
            reference: Reference ID of the artifact to delete.
            session_id: Only delete the artifact if this session owns it.

        Returns:
            True if artifact was deleted, False if not found (or not owned).
        """
        self._parse_reference_id(reference)

        deleted = await self._delete(reference, session_id)

        if deleted:
            logger.info(f"Deleted artifact: {reference}")

        return deleted

    async def exists(
        self,
        reference: str,
        *,
        session_id: Optional[str] = None,
        include_shared: bool = False,
    ) -> bool:
        """Check if an artifact exists.

        Args:
            reference: Reference ID to check.
            session_id: Only count the artifact if this session owns it.
            include_shared: With ``session_id``, also admit rows with no session.

        Returns:
            True if artifact exists, is not expired, and (if scoped) is owned.
        """
        try:
            self._parse_reference_id(reference)
        except InvalidReferenceError:
            return False

        artifact = await self._lookup(reference, session_id, include_shared)

        if artifact is None:
            return False

        if artifact.expires_at and artifact.expires_at < datetime.utcnow():
            return False

        return True

    async def cleanup_expired(self) -> int:
        """Delete all expired artifacts, across every session.

        Internal maintenance; never reachable from a request.

        Returns:
            Number of artifacts deleted.
        """
        count = await self._repository.delete_expired()
        if count > 0:
            logger.info(f"Cleaned up {count} expired artifacts")
        return count

    def get_data_type(self, reference: str) -> str:
        """Get the data type from a reference ID.

        Args:
            reference: Reference ID to parse.

        Returns:
            The data type portion of the reference.

        Raises:
            InvalidReferenceError: If the reference format is invalid.
        """
        data_type, _ = self._parse_reference_id(reference)
        return data_type

    async def list_artifacts(
        self, limit: int = 50, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List stored artifacts (metadata only, no content).

        Args:
            limit: Maximum number of artifacts to return.
            session_id: If provided, only return artifacts for this session.
                ``None`` lists every session, which only internal callers do.
        """
        artifacts = await self._repository.list_all(limit=limit, session_id=session_id)

        return [
            {
                "reference": artifact.id,
                "data_type": artifact.data_type,
                "description": artifact.description,
                "metadata": artifact.metadata_json,
                "created_at": artifact.created_at.isoformat()
                if artifact.created_at
                else None,
            }
            for artifact in artifacts
        ]

    async def search_artifacts(
        self,
        query: str | None = None,
        data_type: str | None = None,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search stored artifacts by description, reference, or type.

        Args:
            query: Free text matched against description, reference, and type.
            data_type: Restrict to one data type.
            limit: Maximum artifacts to return.
            session_id: If provided, only search this session's artifacts.

        Returns:
            Artifact metadata, newest first. Never content.
        """
        artifacts = await self._repository.search(
            query=query, data_type=data_type, limit=limit, session_id=session_id
        )

        return [
            {
                "reference": artifact.id,
                "data_type": artifact.data_type,
                "description": artifact.description,
                "metadata": artifact.metadata_json,
                "created_at": artifact.created_at.isoformat()
                if artifact.created_at
                else None,
            }
            for artifact in artifacts
        ]

    async def export_artifacts(
        self, session_id: str, data_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Export artifacts with content in a single query.

        Used by the export endpoint to avoid N+1 retrieval loops.
        """
        import base64

        artifacts = await self._repository.list_with_content(
            session_id=session_id, data_type=data_type, limit=limit
        )
        results = []
        for artifact in artifacts:
            dtype = artifact.data_type
            if is_text_type(dtype):
                content_str = artifact.content.decode("utf-8")
            else:
                content_str = base64.b64encode(artifact.content).decode("utf-8")
            results.append({
                "reference": artifact.id,
                "data_type": dtype,
                "content": content_str,
                "metadata": artifact.metadata_json,
                "description": artifact.description,
            })
        return results


def format_storage_message(
    data_type: str,
    reference: str,
    source_description: str,
    accepting_tools: Optional[list[str]] = None,
) -> str:
    """Format a helpful message for the LLM about stored data.

    Args:
        data_type: Type of data stored (e.g., "sdf", "pdb").
        reference: The generated reference ID.
        source_description: Description of what generated this data.
        accepting_tools: List of tool names that accept this data type.

    Returns:
        Human-readable message guiding the LLM on how to use the reference.

    Example:
        msg = format_storage_message(
            "sdf", "sdf_abc123", "CCO (ethanol)",
            ["visualize_3d", "dock_ligand"]
        )
        # Returns: "SDF data for CCO (ethanol) has been stored as 'sdf_abc123'.
        #           You can pass this reference to: visualize_3d, dock_ligand"
    """
    type_upper = data_type.upper()

    message = (
        f"{type_upper} data for {source_description} has been stored as '{reference}'."
    )

    if accepting_tools:
        tools_list = ", ".join(accepting_tools)
        message += f" You can pass this reference to tools that accept {type_upper} data, such as: {tools_list}."
    else:
        message += (
            f" You can pass this reference to any tool that accepts {type_upper} data."
        )

    return message
