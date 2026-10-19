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

"""Client for the Aevom HTTP API, as a browser would use it.

Only endpoints the frontend itself calls are used, so the eval exercises the
same surface production does. Where the browser holds an owner cookie, the
client holds the same token and sends it as ``Authorization: Bearer``.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterator

from .http import get_json, post_json, stream_sse, wait_for_health


class AevomClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.owner_token: str | None = None

    def wait_until_healthy(self, timeout_sec: float = 300.0) -> dict:
        body = wait_for_health(f"{self.base_url}/api/health", timeout_sec)
        if body.get("status") != "healthy":
            raise RuntimeError(f"api reported unhealthy: {body}")
        return body

    def info(self) -> dict:
        """Active model and feature flags -- recorded so a run is self-describing."""
        return get_json(f"{self.base_url}/api/info", timeout=30.0)

    def create_session(self) -> str:
        body = post_json(
            f"{self.base_url}/api/session/create",
            {},
            timeout=30.0,
            token=self.owner_token,
        )
        session_id = (body or {}).get("session_id")
        owner_token = (body or {}).get("owner_token")
        if not session_id or not owner_token:
            raise RuntimeError(
                f"/api/session/create returned no session_id or owner_token: {body}"
            )
        self.owner_token = owner_token
        return session_id

    def chat(
        self,
        message: str,
        session_id: str,
        *,
        model_id: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
        timeout: float = 1800.0,
    ) -> Iterator[dict]:
        """Stream one turn. Mirrors the frontend's ChatRequest.

        Only ``message`` is sent, never ``messages``: history comes from the
        backend's own record. Posting a transcript would take the legacy path
        and let the harness silently reshape the trajectory.
        """
        payload: dict[str, Any] = {
            "message": message,
            "session_id": session_id,
            "max_tokens": max_tokens,
        }
        if model_id:
            payload["model_id"] = model_id
        if temperature is not None:
            payload["temperature"] = temperature
        return stream_sse(
            f"{self.base_url}/api/chat", payload, timeout=timeout, token=self.owner_token
        )

    def conversation(self, session_id: str) -> dict:
        return get_json(
            f"{self.base_url}/api/conversation/{session_id}",
            timeout=60.0,
            token=self.owner_token,
        )

    def artifacts(self, session_id: str) -> list[dict]:
        body = get_json(
            f"{self.base_url}/api/storage/artifacts?session_id={session_id}",
            timeout=60.0,
            token=self.owner_token,
        )
        if isinstance(body, dict):
            return body.get("artifacts") or []
        return body or []

    def selections(self, session_id: str) -> Any:
        return get_json(
            f"{self.base_url}/api/selections?session_id={session_id}",
            timeout=60.0,
            token=self.owner_token,
        )
