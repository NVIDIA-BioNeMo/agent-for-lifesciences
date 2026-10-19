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

"""Which owner token each session belongs to.

Stored in Redis as ``aevom:session_owner:{session_id}`` -> sha256(token), with a
TTL refreshed on every successful check. A session whose binding has expired is
unowned, and so unreachable: the check fails closed.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any, Optional

from .principal import hash_token

logger = logging.getLogger(__name__)

KEY_PREFIX = "aevom:session_owner:"
DEFAULT_TTL_SECONDS = 86400


def owner_ttl_seconds() -> int:
    """How long a session stays bound without use (read per-call for tests)."""
    raw = os.environ.get("AEVOM_SESSION_OWNER_TTL_S", "")
    try:
        ttl = int(raw)
    except ValueError:
        return DEFAULT_TTL_SECONDS
    return ttl if ttl > 0 else DEFAULT_TTL_SECONDS


class SessionAlreadyBoundError(Exception):
    """A session id that already has an owner cannot be claimed again."""


class SessionOwnership:
    """Bind sessions to owner tokens and check the binding."""

    def __init__(self, redis: Any):
        self._redis = redis

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{KEY_PREFIX}{session_id}"

    async def bind(self, session_id: str, token: str) -> None:
        """Make ``token`` the owner of a new session.

        Raises:
            SessionAlreadyBoundError: If the session already has an owner.
        """
        claimed = await self._redis.set(
            self._key(session_id), hash_token(token), ex=owner_ttl_seconds(), nx=True
        )
        if not claimed:
            raise SessionAlreadyBoundError(session_id)

    async def owns(self, session_id: Optional[str], token: Optional[str]) -> bool:
        """Whether ``token`` owns ``session_id``. Refreshes the TTL when it does."""
        if not session_id or not token:
            return False
        key = self._key(session_id)
        stored = await self._redis.get(key)
        if stored is None:
            return False
        if isinstance(stored, bytes):
            stored = stored.decode("ascii", errors="replace")
        if not hmac.compare_digest(stored, hash_token(token)):
            return False
        await self._redis.expire(key, owner_ttl_seconds())
        return True
