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

"""Per-session API-key secret storage backed by Redis.

Architecture (mirrors ``api.mcp.session_store``):
- **Redis** (``secret:{session_id}``, 4h TTL) is the source of truth for a
  session's stored secrets. One JSON string per session keeps API pods
  stateless and horizontally scalable.
- Secret *values* are stored in Redis in plaintext, exactly like the existing
  MCP ``api_key``. This is not a hardware vault and secrets are ephemeral
  (they disappear on TTL expiry / new session). Values are never serialized
  outward by the public API -- only the built-in ``http_request`` tool reads
  them (server-side) via :meth:`SecretSessionStore.get_value`.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default session TTL: 4 hours (matches MCP / skills sessions).
DEFAULT_SESSION_TTL = 4 * 60 * 60


@dataclass
class SecretMeta:
    """A stored secret. ``value`` is sensitive and never leaves the backend."""

    id: str
    name: str
    value: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Full serialization for Redis storage (includes the value)."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecretMeta":
        return cls(
            id=data["id"],
            name=data["name"],
            value=data.get("value", ""),
            created_at=data.get("created_at", time.time()),
        )

    def to_public_dict(self) -> dict[str, Any]:
        """API-facing view. Deliberately omits ``value``."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
        }


def _redis_key(session_id: str) -> str:
    """Redis key for a session's secrets."""
    return f"secret:{session_id}"


class SecretSessionStore:
    """Manages a session's API-key secrets, backed by Redis.

    Secrets are keyed by a generated id, but looked up by ``name`` (the handle
    the agent references via the ``http_request`` tool). Adding a secret whose
    name already exists replaces the previous one.
    """

    def __init__(self, session_ttl: float = DEFAULT_SESSION_TTL):
        self._session_ttl = session_ttl

    async def _get_redis(self):
        from api.redis import get_redis

        return await get_redis()

    async def _load(self, session_id: str) -> dict[str, SecretMeta]:
        redis = await self._get_redis()
        raw = await redis.get(_redis_key(session_id))
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return {sid: SecretMeta.from_dict(meta) for sid, meta in data.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load secrets for %s: %s", session_id, exc)
            return {}

    async def _save(self, session_id: str, secrets: dict[str, SecretMeta]) -> None:
        redis = await self._get_redis()
        data = {sid: meta.to_dict() for sid, meta in secrets.items()}
        await redis.set(
            _redis_key(session_id),
            json.dumps(data),
            ex=int(self._session_ttl),
        )

    async def add_secret(self, session_id: str, name: str, value: str) -> SecretMeta:
        """Store a secret for a session, replacing any existing one by name."""
        secrets = await self._load(session_id)

        # Replace any existing secret with the same name (name is the lookup key).
        for existing_id, meta in list(secrets.items()):
            if meta.name == name:
                del secrets[existing_id]

        secret = SecretMeta(
            id=f"sk_{uuid.uuid4().hex[:12]}",
            name=name,
            value=value,
        )
        secrets[secret.id] = secret
        await self._save(session_id, secrets)
        logger.info(
            "Stored secret %s (name=%s) for session %s",
            secret.id,
            name,
            session_id,
        )
        return secret

    async def list_secrets(self, session_id: str) -> list[SecretMeta]:
        secrets = await self._load(session_id)
        return sorted(secrets.values(), key=lambda m: m.created_at)

    async def get_value(self, session_id: str, name: str) -> Optional[str]:
        """Resolve a secret value by name. Internal use only (the tool)."""
        if not session_id or not name:
            return None
        secrets = await self._load(session_id)
        for meta in secrets.values():
            if meta.name == name:
                return meta.value
        return None

    async def remove_secret(self, session_id: str, secret_id: str) -> bool:
        secrets = await self._load(session_id)
        if secret_id not in secrets:
            return False
        del secrets[secret_id]
        await self._save(session_id, secrets)
        logger.info("Removed secret %s for session %s", secret_id, session_id)
        return True


_secret_session_store: Optional[SecretSessionStore] = None


def get_secret_session_store() -> SecretSessionStore:
    """Get the secret session store singleton."""
    global _secret_session_store
    if _secret_session_store is None:
        _secret_session_store = SecretSessionStore()
        logger.info("Created secret session store")
    return _secret_session_store
