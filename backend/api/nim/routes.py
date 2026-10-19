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

"""API routes for registering self-hosted NIMs, mirroring the MCP routes."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.deps import optional_owned_session, require_owned_session

from .client import NIMAuthError, NIMConnectionError
from .session_store import get_nim_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nim", tags=["nim"])


class AddNIMServerRequest(BaseModel):
    """Request body for registering a NIM."""

    name: str
    url: str
    api_key: Optional[str] = None


class NIMServersListResponse(BaseModel):
    """Response for listing a session's NIMs."""

    servers: list[dict]
    count: int


@router.post("/servers")
async def add_nim_server(
    request: AddNIMServerRequest,
    session_id: str = Depends(require_owned_session),
) -> dict[str, Any]:
    """Probe a NIM and register it for this session.

    The NIM's tools are *not* enabled by this call. They stay withheld until
    ``POST /servers/{id}/confirm``, so the user can see what registering it
    would add before any of it is reachable.
    """
    store = get_nim_session_store()

    try:
        config = await store.add_server(
            session_id=session_id,
            name=request.name,
            url=request.url,
            api_key=request.api_key,
        )
    except NIMAuthError as exc:
        logger.warning("NIM auth error: %s", exc)
        raise HTTPException(status_code=401, detail=str(exc))
    except NIMConnectionError as exc:
        logger.warning("NIM connection error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface, do not 500 silently
        logger.exception("Unexpected error registering NIM")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    return config.to_response()


@router.get("/servers", response_model=NIMServersListResponse)
async def list_nim_servers(
    session_id: Optional[str] = Depends(optional_owned_session),
) -> NIMServersListResponse:
    """List the NIMs registered for this session."""
    if not session_id:
        return NIMServersListResponse(servers=[], count=0)

    servers = await get_nim_session_store().get_servers(session_id)
    return NIMServersListResponse(
        servers=[server.to_response() for server in servers],
        count=len(servers),
    )


@router.delete("/servers/{server_id}")
async def remove_nim_server(
    server_id: str,
    session_id: str = Depends(require_owned_session),
) -> dict[str, str]:
    """Remove a NIM registration from this session."""
    if not await get_nim_session_store().remove_server(session_id, server_id):
        raise HTTPException(status_code=404, detail=f"NIM {server_id} not found")
    return {"status": "removed", "server_id": server_id}


@router.post("/servers/{server_id}/confirm")
async def confirm_nim_server(
    server_id: str,
    session_id: str = Depends(require_owned_session),
) -> dict[str, str]:
    """Enable a registered NIM's tools after a human has reviewed them.

    Idempotent: confirming an already-confirmed NIM succeeds.
    """
    if not await get_nim_session_store().confirm_server(session_id, server_id):
        raise HTTPException(status_code=404, detail=f"NIM {server_id} not found")
    return {"status": "confirmed", "server_id": server_id}
