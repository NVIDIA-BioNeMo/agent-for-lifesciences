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

"""Storage state module for tracking stored data artifacts.

This module provides context about stored artifacts to the LLM via prompt injection,
following the same pattern as viewer_state.py. The storage state is refreshed
at the start of each chat request and injected into the system prompt as
`[STORAGE STATE: ...]`.

This allows the LLM to know what data artifacts are available for use in tools
without making a separate tool call to query storage.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ArtifactInfo:
    """Information about a single stored artifact."""
    
    reference: str  # e.g., "sdf_abc123"
    data_type: str  # e.g., "sdf", "pdb"
    description: Optional[str] = None  # LLM-provided description
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def get_display_name(self) -> str:
        """Get a human-readable display name for the artifact.
        
        Prefers the LLM-provided description, then falls back to
        extracting a name from metadata, then to the reference ID.
        
        Returns:
            Display name like "Aspirin docked to COX-2" or "sdf_abc123".
        """
        # Prefer the LLM-provided description
        if self.description:
            # Truncate if too long
            if len(self.description) > 50:
                return self.description[:47] + "..."
            return self.description
        
        # Try common metadata keys for a human-readable name
        for key in ["name", "label", "gene_name", "molecule_name", "smiles"]:
            if key in self.metadata and self.metadata[key]:
                value = str(self.metadata[key])
                # Truncate long values (e.g., SMILES strings)
                if len(value) > 30:
                    return value[:27] + "..."
                return value
        
        # Fall back to reference ID
        return self.reference
    
    def to_brief(self) -> str:
        """Generate a brief one-line description.
        
        Returns:
            Brief string like "sdf_abc123 - Aspirin molecule" or "sdf_abc123 (SDF)".
        """
        type_upper = self.data_type.upper()
        
        # If we have a description, use a cleaner format
        if self.description:
            desc = self.description
            if len(desc) > 40:
                desc = desc[:37] + "..."
            return f"{self.reference} - {desc}"
        
        # Fall back to old format with display name
        display_name = self.get_display_name()
        if display_name != self.reference:
            return f"{self.reference} ({display_name}, {type_upper})"
        return f"{self.reference} ({type_upper})"


@dataclass
class StorageState:
    """Complete state of stored data artifacts.
    
    This is refreshed from the database at the start of each chat request.
    """
    
    is_empty: bool = True
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    
    def to_description(self) -> str:
        """Generate a human-readable description of the storage state.
        
        Returns:
            Multi-line description of all stored artifacts.
        """
        if self.is_empty:
            return "No data artifacts are currently stored."
        
        lines = [f"**{len(self.artifacts)} stored artifact(s):**"]
        
        for artifact in self.artifacts:
            display_name = artifact.get_display_name()
            type_upper = artifact.data_type.upper()
            
            line = f"- `{artifact.reference}`: {display_name} ({type_upper})"
            
            if artifact.created_at:
                line += f" - created {artifact.created_at.strftime('%Y-%m-%d %H:%M')}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for observation serialization.
        
        Returns:
            Dict representation suitable for RL training.
        """
        return {
            "is_empty": self.is_empty,
            "artifacts": [
                {
                    "reference": a.reference,
                    "data_type": a.data_type,
                    "description": a.description,
                    "metadata": a.metadata,
                }
                for a in self.artifacts
            ],
        }
    
    def to_condensed(self) -> str:
        """Generate a condensed one-line summary for system prompt injection.
        
        Returns a compact representation like:
        - Empty: "[STORAGE STATE: Empty - no stored artifacts]"
        - One SDF: "[STORAGE STATE: sdf_abc123 (aspirin, SDF)]"
        - Multiple: "[STORAGE STATE: sdf_abc123 (aspirin, SDF); pdb_xyz789 (CDK2, PDB)]"
        """
        if self.is_empty:
            return "[STORAGE STATE: Empty - no stored artifacts]"
        
        # Limit to first 5 artifacts to keep prompt concise
        parts = [artifact.to_brief() for artifact in self.artifacts[:5]]
        
        result = f"[STORAGE STATE: {'; '.join(parts)}"
        
        if len(self.artifacts) > 5:
            result += f"; ... and {len(self.artifacts) - 5} more"
        
        result += "]"
        return result


import json

# Redis key TTL for storage state (5 minutes)
STORAGE_STATE_TTL = 300


async def refresh_storage_state(redis_client, storage: "DataStorageService", session_id: str) -> None:
    """Refresh the storage state from the database and store in Redis.
    
    Called at the start of each chat request to get current artifact state.
    The state is stored in Redis keyed by session_id so it can be retrieved
    later without re-querying the database.
    
    Args:
        redis_client: Redis client instance.
        storage: DataStorageService instance for querying artifacts.
        session_id: Browser session ID for Redis key scoping.
    """
    try:
        artifacts_data = await storage.list_artifacts(limit=50)
        
        artifacts = []
        for data in artifacts_data:
            # Parse created_at if it's a string
            created_at = None
            if data.get("created_at"):
                if isinstance(data["created_at"], str):
                    try:
                        created_at = datetime.fromisoformat(data["created_at"])
                    except ValueError:
                        pass
                elif isinstance(data["created_at"], datetime):
                    created_at = data["created_at"]
            
            artifacts.append(ArtifactInfo(
                reference=data["reference"],
                data_type=data["data_type"],
                description=data.get("description"),
                metadata=data.get("metadata", {}),
                created_at=created_at,
            ))
        
        state = StorageState(
            is_empty=len(artifacts) == 0,
            artifacts=artifacts,
        )
        
        # Store in Redis
        key = f"storage:{session_id}"
        state_data = {
            "is_empty": state.is_empty,
            "artifacts": [
                {
                    "reference": a.reference,
                    "data_type": a.data_type,
                    "description": a.description,
                    "metadata": a.metadata,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in state.artifacts
            ],
        }
        await redis_client.set(key, json.dumps(state_data), ex=STORAGE_STATE_TTL)
        
        logger.info(f"[refresh_storage_state] Found {len(artifacts)} artifacts for session {session_id}")
        for a in artifacts[:5]:  # Log first 5
            logger.debug(f"[refresh_storage_state]   - {a.reference} ({a.data_type})")
        
    except Exception as e:
        logger.warning(f"Failed to refresh storage state: {e}")


async def get_current_storage_state(redis_client, session_id: str) -> StorageState:
    """Get the current storage state from Redis.
    
    Args:
        redis_client: Redis client instance.
        session_id: Browser session ID for Redis key scoping.
    
    Returns:
        StorageState from Redis, or empty state if not found.
    """
    key = f"storage:{session_id}"
    raw = await redis_client.get(key)
    if raw is None:
        return StorageState(is_empty=True)
    
    try:
        data = json.loads(raw)
        artifacts = []
        for a in data.get("artifacts", []):
            created_at = None
            if a.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(a["created_at"])
                except (ValueError, TypeError):
                    pass
            artifacts.append(ArtifactInfo(
                reference=a["reference"],
                data_type=a["data_type"],
                description=a.get("description"),
                metadata=a.get("metadata", {}),
                created_at=created_at,
            ))
        return StorageState(
            is_empty=data.get("is_empty", True),
            artifacts=artifacts,
        )
    except Exception as e:
        logger.warning(f"Failed to deserialize storage state for {session_id}: {e}")
        return StorageState(is_empty=True)


# Type hint import at module level causes circular import, so use string annotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.storage import DataStorageService
