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

"""Built-in HTTP request tool for calling authenticated external APIs.

The agent structures an HTTP request and references a stored API key **by name**
(``secret_name``). The value is resolved server-side from the session's secret
store and injected into the outgoing auth header; it never appears in the tool
arguments, the response, logs, or the chat transcript.

This is the generic escape hatch. When a skill declares (or we extract) a
data-type contract, ``api.skills.endpoint_tool`` synthesizes a typed tool for
that endpoint that takes storage references instead of raw molecular data; the
skill prose steers the model there. This tool stays available for endpoints no
contract covers.

The security rules -- host allowlist, SSRF guard, no redirects, bounded timeout,
capped response, secret scrubbing -- live in ``http_common`` and are shared with
the synthesized tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .base import BaseTool, DataIOCategory, ToolResult, ToolType
from .http_common import (
    ALLOWED_METHODS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_INLINE_BYTES,
    CappedResponse,
    HttpToolError,
    clamp_timeout,
    load_allowed_hosts,
    resolve_auth_header,
    send_request,
    validate_target,
)

logger = logging.getLogger(__name__)


class HttpRequestTool(BaseTool):
    """Make an authenticated HTTP request to an allowlisted external API.

    The model passes ``secret_name`` (the handle of a stored API key), never the
    key itself. The tool injects the key as a Bearer token server-side.
    """

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def summary_template(self) -> str:
        return "Calling {url}..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        # OUTPUT_STORES so the registry injects a storage service, letting us
        # persist large response bodies as artifacts.
        return DataIOCategory.OUTPUT_STORES

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        allowed = ", ".join(sorted(load_allowed_hosts()))
        return {
            "type": "function",
            "function": {
                "name": "http_request",
                "description": (
                    "Make an HTTP request to an external API a skill needs (e.g. a "
                    "NIM endpoint).\n\n"
                    "PREFER A SKILL'S OWN TOOL: if a skill provides a dedicated tool "
                    "(named `skill_*`), use that instead. It takes data references "
                    "directly, handles the request body, and stores each output as a "
                    "typed reference. Use http_request only for endpoints no skill "
                    "tool covers.\n\n"
                    "AUTHENTICATION (read carefully):\n"
                    "- To send an API key, set `secret_name` to the NAME of a key the "
                    'user stored via the Add Key panel (e.g. secret_name="NVIDIA_API_KEY"). '
                    "The backend looks up the key and adds the auth header for you "
                    "(default `Authorization: Bearer <key>`).\n"
                    "- NEVER put the key — or a placeholder like `${NGC_API_KEY}`, "
                    "`$NGC_API_KEY`, or `Bearer <key>` — in `headers` yourself. "
                    "Placeholders are sent literally and the request will 401. Just "
                    "set `secret_name`.\n"
                    "- Use the EXACT stored key name. Skill docs often reference an "
                    "env-var name (e.g. NGC_API_KEY) that may DIFFER from the stored "
                    "name. If the name is wrong, the error lists the available key "
                    "names — retry with one of those. If no suitable key is stored, "
                    "tell the user to add it in the Add Key panel.\n\n"
                    "Example: http_request(url=\"https://host/generate\", method=\"POST\", "
                    'secret_name="NVIDIA_API_KEY", json_body={...}).\n\n'
                    f"Only allowlisted hosts are permitted (currently: {allowed}). "
                    "Large responses are stored as an artifact and returned as a "
                    "reference you can load with python_execute."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full request URL (https). Host must be allowlisted.",
                        },
                        "method": {
                            "type": "string",
                            "enum": sorted(ALLOWED_METHODS),
                            "description": "HTTP method. Defaults to POST.",
                        },
                        "secret_name": {
                            "type": "string",
                            "description": (
                                "Name of a stored API key to send as a Bearer token. "
                                "Do not pass the key value itself."
                            ),
                        },
                        "headers": {
                            "type": "object",
                            "description": "Optional additional request headers.",
                            "additionalProperties": {"type": "string"},
                        },
                        "json_body": {
                            "type": "object",
                            "description": "Optional JSON request body.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional URL query parameters.",
                            "additionalProperties": {"type": "string"},
                        },
                        "auth_header": {
                            "type": "string",
                            "description": "Header name for the key. Defaults to 'Authorization'.",
                        },
                        "auth_prefix": {
                            "type": "string",
                            "description": "Prefix before the key. Defaults to 'Bearer '.",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        url = arguments.get("url")
        headers = {str(k): str(v) for k, v in (arguments.get("headers") or {}).items()}
        secret_value: Optional[str] = None

        try:
            method, host = validate_target(url, arguments.get("method") or "POST")
            secret_value = await resolve_auth_header(
                self._session_id,
                arguments.get("secret_name"),
                headers,
                auth_header=arguments.get("auth_header"),
                auth_prefix=arguments.get("auth_prefix"),
            )
            response = await send_request(
                method,
                url,
                headers=headers,
                params=arguments.get("params") or None,
                json_body=arguments.get("json_body"),
                timeout=clamp_timeout(
                    arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
                ),
                secret_value=secret_value,
            )
        except HttpToolError as exc:
            return ToolResult.error(str(exc))

        return await self._handle_response(response, url, host)

    async def _handle_response(
        self,
        response: CappedResponse,
        url: str,
        host: str,
    ) -> ToolResult:
        status = response.status_code
        body_bytes = response.content
        truncated = response.truncated

        # Small responses: return inline (parse JSON when possible).
        if len(body_bytes) <= MAX_INLINE_BYTES:
            text = body_bytes.decode("utf-8", errors="replace")
            body: Any = text
            try:
                body = json.loads(text)
            except (ValueError, TypeError):
                pass
            return ToolResult.json_data(
                {"status_code": status, "truncated": truncated, "body": body}
            )

        # Large responses: persist as an artifact, return a reference.
        storage = self._storage
        if storage is not None:
            try:
                ref = await storage.store(
                    "json",
                    body_bytes,
                    metadata={
                        "source": "http_request",
                        "url": url,
                        "host": host,
                        "status_code": status,
                    },
                    description=f"HTTP {status} response from {host}"[:256],
                )
                message = (
                    f"HTTP {status} response ({len(body_bytes)} bytes) from {host} stored as "
                    f"'{ref}'. Load it with python_execute (aevom.load('{ref}')) to process."
                )
                return ToolResult.data_reference(
                    ref, "json", message, metadata={"status_code": status}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to store http_request response: %s", exc)

        # Fallback: return a truncated inline preview.
        preview = body_bytes[:MAX_INLINE_BYTES].decode("utf-8", errors="replace")
        return ToolResult.json_data(
            {"status_code": status, "truncated": True, "body_preview": preview}
        )
