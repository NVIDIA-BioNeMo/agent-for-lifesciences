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

"""Sandbox session management backed by Redis.

This module manages ephemeral sessions for sandboxed Python execution,
tracking which data references each session is allowed to access.
Sessions are stored in Redis so they work across multiple API pods.
"""

import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Default TTL for sandbox sessions in Redis (10 minutes)
DEFAULT_SESSION_TTL_SECONDS = 600


class SandboxSessionError(Exception):
    """Base exception for sandbox session errors."""
    pass


class SessionNotFoundError(SandboxSessionError):
    """Raised when a session token is invalid or expired."""
    pass


class ReferenceNotAllowedError(SandboxSessionError):
    """Raised when attempting to access a reference not allowed for this session."""
    pass


@dataclass
class SandboxSession:
    """Represents an active sandbox execution session.
    
    Each python_execute call creates a session that tracks:
    - Which data references the code is allowed to access
    - Session expiration time
    - Access statistics for rate limiting
    - Visualization events queued during execution
    - Client session ID for artifact scoping
    """
    id: str
    allowed_refs: set[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    max_accesses: Optional[int] = None
    visualization_events: list = field(default_factory=list)
    client_session_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_ref_allowed(self, reference: str) -> bool:
        """Check if a reference is allowed for this session."""
        return reference in self.allowed_refs
    
    def can_access(self) -> bool:
        """Check if the session can make more data access requests."""
        if self.max_accesses is None:
            return True
        return self.access_count < self.max_accesses
    
    def record_access(self) -> None:
        """Record a data access request."""
        self.access_count += 1
    
    def add_visualization_event(self, event: dict) -> None:
        """Add a visualization event to the queue."""
        self.visualization_events.append(event)
    
    def get_and_clear_visualization_events(self) -> list:
        """Get all queued visualization events and clear the queue."""
        events = self.visualization_events.copy()
        self.visualization_events.clear()
        return events
    
    def has_pending_visualizations(self) -> bool:
        """Check if there are pending visualization events."""
        return len(self.visualization_events) > 0

    def to_dict(self) -> dict:
        """Serialize to dict for Redis storage."""
        return {
            "id": self.id,
            "allowed_refs": list(self.allowed_refs),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "max_accesses": self.max_accesses,
            "visualization_events": self.visualization_events,
            "client_session_id": self.client_session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SandboxSession":
        """Deserialize from dict."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        expires_at = data.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return cls(
            id=data["id"],
            allowed_refs=set(data.get("allowed_refs", [])),
            created_at=created_at,
            expires_at=expires_at,
            access_count=data.get("access_count", 0),
            max_accesses=data.get("max_accesses"),
            visualization_events=data.get("visualization_events", []),
            client_session_id=data.get("client_session_id"),
        )


def _redis_key(token: str) -> str:
    """Build Redis key for a sandbox session."""
    return f"sandbox:{token}"


class SandboxSessionManager:
    """Manages sandbox sessions backed by Redis.
    
    This manager handles:
    - Session creation with scoped reference access
    - Session validation and expiration
    - Rate limiting per session
    
    All state is stored in Redis, making this manager stateless
    and safe for use across multiple API pods.
    """
    
    TOKEN_LENGTH = 32  # 256 bits of entropy
    
    def __init__(self):
        """Initialize the session manager."""
        pass
    
    async def _get_redis(self):
        """Get Redis client."""
        from api.redis import get_redis
        return await get_redis()
    
    async def create_session(
        self,
        allowed_refs: set[str],
        ttl_minutes: Optional[int] = None,
        max_accesses: Optional[int] = None,
        client_session_id: Optional[str] = None,
    ) -> SandboxSession:
        """Create a new sandbox session.
        
        Args:
            allowed_refs: Set of reference IDs this session can access.
            ttl_minutes: Session TTL in minutes (None = default 10 min).
            max_accesses: Max data access requests (None = unlimited).
            client_session_id: Optional client session ID for artifact scoping.
            
        Returns:
            Created SandboxSession with unique token.
        """
        redis = await self._get_redis()
        
        # Generate secure token
        token = secrets.token_urlsafe(self.TOKEN_LENGTH)
        
        # Calculate expiration
        expires_at = None
        if ttl_minutes is not None and ttl_minutes > 0:
            expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        
        session = SandboxSession(
            id=token,
            allowed_refs=allowed_refs,
            expires_at=expires_at,
            max_accesses=max_accesses,
            client_session_id=client_session_id,
        )
        
        # Store in Redis with TTL
        redis_ttl = (ttl_minutes * 60) if ttl_minutes and ttl_minutes > 0 else DEFAULT_SESSION_TTL_SECONDS
        await redis.set(
            _redis_key(token),
            json.dumps(session.to_dict()),
            ex=redis_ttl,
        )
        
        logger.info(
            f"Created sandbox session {token[:8]}... "
            f"with {len(allowed_refs)} allowed refs, "
            f"expires: {expires_at}, max_accesses: {max_accesses}, "
            f"client_session: {client_session_id[:8] if client_session_id else None}..."
        )
        
        return session
    
    async def _load_session(self, token: str) -> Optional[SandboxSession]:
        """Load a session from Redis.
        
        Args:
            token: Session token.
            
        Returns:
            SandboxSession if found, None if not found or expired.
        """
        redis = await self._get_redis()
        raw = await redis.get(_redis_key(token))
        if raw is None:
            return None
        
        try:
            data = json.loads(raw)
            session = SandboxSession.from_dict(data)
            
            # Check application-level expiration
            if session.is_expired():
                await redis.delete(_redis_key(token))
                return None
            
            return session
        except Exception as e:
            logger.warning(f"Failed to deserialize sandbox session {token[:8]}...: {e}")
            return None
    
    async def _save_session(self, session: SandboxSession) -> None:
        """Save session state back to Redis.
        
        This is called after mutations (access_count, visualization_events, allowed_refs).
        
        Args:
            session: Session to save.
        """
        redis = await self._get_redis()
        
        # Preserve remaining TTL
        ttl = await redis.ttl(_redis_key(session.id))
        if ttl <= 0:
            ttl = DEFAULT_SESSION_TTL_SECONDS
        
        await redis.set(
            _redis_key(session.id),
            json.dumps(session.to_dict()),
            ex=ttl,
        )
    
    async def get_session(self, token: str) -> SandboxSession:
        """Get a session by token.
        
        Args:
            token: Session token.
            
        Returns:
            SandboxSession if found and valid.
            
        Raises:
            SessionNotFoundError: If session not found or expired.
        """
        session = await self._load_session(token)
        
        if session is None:
            raise SessionNotFoundError(f"Session not found: {token[:8]}...")
        
        return session
    
    async def validate_access(self, token: str, reference: str) -> SandboxSession:
        """Validate that a session can access a specific reference.
        
        Args:
            token: Session token.
            reference: Reference ID to access.
            
        Returns:
            SandboxSession if access is allowed.
            
        Raises:
            SessionNotFoundError: If session not found or expired.
            ReferenceNotAllowedError: If reference not in allowed set.
            SandboxSessionError: If rate limit exceeded.
        """
        session = await self.get_session(token)
        
        # Check rate limit
        if not session.can_access():
            raise SandboxSessionError(
                f"Session {token[:8]}... exceeded max access limit "
                f"({session.max_accesses})"
            )
        
        # Check reference is allowed
        if not session.is_ref_allowed(reference):
            logger.warning(
                f"Session {token[:8]}... attempted to access "
                f"non-allowed reference: {reference}"
            )
            raise ReferenceNotAllowedError(
                f"Reference '{reference}' is not allowed for this session. "
                f"Allowed refs: {session.allowed_refs}"
            )
        
        # Record access and save
        session.record_access()
        await self._save_session(session)
        
        logger.debug(
            f"Session {token[:8]}... accessed {reference} "
            f"(access {session.access_count}/{session.max_accesses or 'unlimited'})"
        )
        
        return session
    
    async def delete_session(self, token: str) -> bool:
        """Delete a session.
        
        Args:
            token: Session token.
            
        Returns:
            True if session was deleted, False if not found.
        """
        redis = await self._get_redis()
        result = await redis.delete(_redis_key(token))
        if result:
            logger.info(f"Deleted sandbox session {token[:8]}...")
            return True
        return False
    
    async def add_visualization_event(self, token: str, event: dict) -> None:
        """Add a visualization event to a session.
        
        Args:
            token: Session token.
            event: Visualization event dict.
        """
        session = await self.get_session(token)
        session.add_visualization_event(event)
        await self._save_session(session)
    
    async def get_and_clear_visualization_events(self, token: str) -> list:
        """Get and clear visualization events for a session.
        
        Args:
            token: Session token.
            
        Returns:
            List of visualization events.
        """
        session = await self.get_session(token)
        events = session.get_and_clear_visualization_events()
        await self._save_session(session)
        return events
    
    async def add_allowed_ref(self, token: str, reference: str) -> None:
        """Add a reference to a session's allowed refs.
        
        Args:
            token: Session token.
            reference: Reference to add.
        """
        session = await self.get_session(token)
        session.allowed_refs.add(reference)
        await self._save_session(session)
    
    @property
    def active_session_count(self) -> int:
        """Get count of active sessions.
        
        Note: This is approximate since we can't efficiently scan Redis.
        Returns 0 for now -- health check doesn't depend on exact count.
        """
        return 0


# Module-level singleton (stateless facade, safe as singleton)
_session_manager: Optional[SandboxSessionManager] = None


def get_session_manager() -> SandboxSessionManager:
    """Get the sandbox session manager.
    
    Returns:
        SandboxSessionManager instance.
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SandboxSessionManager()
    return _session_manager
