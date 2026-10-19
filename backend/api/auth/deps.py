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

"""FastAPI dependencies that turn a ``session_id`` into an owned one.

A request with no owner token gets 401. A token that does not own the session
gets 404, the same as a session that does not exist, so ids cannot be probed.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Path, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.config import get_async_session
from api.storage.scoped import SessionScopedStorage, scoped_storage

from .ownership import SessionOwnership
from .principal import owner_token_from

logger = logging.getLogger(__name__)

MISSING_TOKEN_DETAIL = (
    "Missing owner token. Create a session with POST /api/session/create and "
    "send the owner_token it returns as 'Authorization: Bearer <token>'."
)


def get_session_ownership() -> SessionOwnership:
    from api.redis import get_redis

    try:
        redis = get_redis()
    except RuntimeError as e:
        logger.error("Session ownership unavailable: %s", e)
        raise HTTPException(status_code=503, detail="Session ownership store unavailable")
    return SessionOwnership(redis)


async def assert_owned_session(
    request: Request, session_id: str, ownership: SessionOwnership
) -> None:
    """Raise unless the request's owner token owns ``session_id``."""
    token = owner_token_from(request)
    if token is None:
        raise HTTPException(status_code=401, detail=MISSING_TOKEN_DETAIL)
    if not await ownership.owns(session_id, token):
        raise HTTPException(status_code=404, detail="Session not found")


async def require_owned_session(
    request: Request,
    session_id: Optional[str] = Query(None),
    ownership: SessionOwnership = Depends(get_session_ownership),
) -> str:
    """``?session_id=`` that the caller owns. Missing is a 400."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    await assert_owned_session(request, session_id, ownership)
    return session_id


async def optional_owned_session(
    request: Request,
    session_id: Optional[str] = Query(None),
    ownership: SessionOwnership = Depends(get_session_ownership),
) -> Optional[str]:
    """``?session_id=`` that the caller owns, or None when none was given.

    For list endpoints, which return nothing without a session.
    """
    if not session_id:
        return None
    await assert_owned_session(request, session_id, ownership)
    return session_id


async def require_owned_path_session(
    request: Request,
    session_id: str = Path(...),
    ownership: SessionOwnership = Depends(get_session_ownership),
) -> str:
    """``/{session_id}/`` in the path, which the caller owns."""
    await assert_owned_session(request, session_id, ownership)
    return session_id


async def get_scoped_storage(
    session_id: str = Depends(require_owned_session),
    db_session: AsyncSession = Depends(get_async_session),
) -> SessionScopedStorage:
    """Storage confined to the owned ``?session_id=``."""
    return scoped_storage(db_session, session_id)
