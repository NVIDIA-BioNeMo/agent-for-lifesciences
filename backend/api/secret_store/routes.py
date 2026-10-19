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

"""API routes for session-scoped secret (API key) management.

Mirrors ``api.mcp.routes``. Secrets are stored per browser session. The secret
*value* is write-only from the client's perspective: it is accepted on create
but never returned by any endpoint.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.deps import optional_owned_session, require_owned_session

from .session_store import get_secret_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/secrets", tags=["secrets"])


class AddSecretRequest(BaseModel):
    """Request body for adding a secret."""

    name: str
    value: str


class SecretResponse(BaseModel):
    """Public view of a stored secret (never includes the value)."""

    id: str
    name: str
    created_at: float


class SecretsListResponse(BaseModel):
    """Response for listing a session's secrets."""

    secrets: list[SecretResponse]
    count: int


@router.post("/", response_model=SecretResponse)
async def add_secret(
    request: AddSecretRequest,
    session_id: str = Depends(require_owned_session),
):
    """Store an API key for a session."""

    name = request.name.strip()
    value = request.value.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not value:
        raise HTTPException(status_code=400, detail="value is required")

    store = get_secret_session_store()
    secret = await store.add_secret(session_id, name, value)
    return SecretResponse(**secret.to_public_dict())


@router.get("/", response_model=SecretsListResponse)
async def list_secrets(
    session_id: Optional[str] = Depends(optional_owned_session),
):
    """List a session's stored secrets (names only, never values)."""
    if not session_id:
        return SecretsListResponse(secrets=[], count=0)

    store = get_secret_session_store()
    secrets = await store.list_secrets(session_id)
    public = [SecretResponse(**s.to_public_dict()) for s in secrets]
    return SecretsListResponse(secrets=public, count=len(public))


@router.delete("/{secret_id}")
async def remove_secret(
    secret_id: str,
    session_id: str = Depends(require_owned_session),
):
    """Remove a stored secret from a session."""

    store = get_secret_session_store()
    removed = await store.remove_secret(session_id, secret_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Secret {secret_id} not found")

    return {"status": "removed", "secret_id": secret_id}
