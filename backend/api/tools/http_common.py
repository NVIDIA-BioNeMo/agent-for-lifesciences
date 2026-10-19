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

"""Shared security surface for outbound HTTP from tools.

Two tools make outbound requests on the model's behalf: the generic
``http_request`` escape hatch and the per-skill tools synthesized from a skill
contract (``api.skills.endpoint_tool``). They must enforce the *same* rules --
host allowlist, SSRF guard, no redirects, bounded timeout, capped response,
secrets resolved by name and scrubbed from output. Keeping that logic here means
there is one place to audit and no way for the newer path to quietly diverge.

Callers signal refusal by catching :class:`HttpToolError`, whose message is
written for the model to read and act on.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_HOSTS = "health.api.nvidia.com"

# Responses at or below this size are returned inline to the model; larger
# responses are persisted as an artifact and returned as a reference.
MAX_INLINE_BYTES = 8 * 1024
# Absolute cap on how much of a response body we retain.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 180

ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

# Matches shell-style variable placeholders the model sometimes copies from skill
# docs into a manual header (e.g. "Bearer ${NGC_API_KEY}" or "$NGC_API_KEY").
_PLACEHOLDER_RE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")


class HttpToolError(Exception):
    """A refused or failed request, with a message intended for the model."""


def has_placeholder(value: str) -> bool:
    """Whether a header value contains an uninterpolated ``${VAR}``/``$VAR``."""
    return bool(_PLACEHOLDER_RE.search(value or ""))


def http_request_tool_enabled() -> bool:
    """Whether the generic ``http_request`` tool is offered (read per-call for tests).

    Off by default: it lets the model send arbitrary requests to any allowlisted
    host. Skill endpoint tools, which are confirmed per endpoint, do not depend
    on it.
    """
    return os.environ.get("AEVOM_FEAT_HTTP_REQUEST_TOOL", "false").lower() in ("true", "1")


def load_allowed_hosts() -> set[str]:
    """Load the host allowlist from the environment (read per-call for tests)."""
    raw = os.environ.get("HTTP_TOOL_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS)
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def is_blocked_ip(ip: str) -> bool:
    """Whether an IP is in a range we refuse to send credentials to.

    An address that cannot be parsed is blocked: a check that cannot decide
    must refuse.
    """
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return True
    # ``::ffff:10.0.0.1`` reaches 10.0.0.1, but the IPv6 range predicates do
    # not all see through the mapping.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def check_ssrf(host: str) -> None:
    """Raise :class:`HttpToolError` if the host is (or resolves to) a non-public IP."""
    # IP literal: validate directly.
    try:
        ipaddress.ip_address(host)
        if is_blocked_ip(host):
            raise HttpToolError(
                f"Host '{host}' resolves to a non-public address and is blocked."
            )
        return
    except ValueError:
        pass  # Not an IP literal; resolve the hostname below.

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        logger.warning("Could not resolve host %s for SSRF check: %s", host, exc)
        raise HttpToolError(
            f"Could not resolve host '{host}', so no request was made."
        ) from exc
    if not infos:
        raise HttpToolError(
            f"Host '{host}' resolved to no addresses, so no request was made."
        )

    for info in infos:
        ip = info[4][0]
        if is_blocked_ip(ip):
            raise HttpToolError(
                f"Host '{host}' resolves to a non-public address ({ip}) and is blocked."
            )


def scrub(text: str, secret: Optional[str]) -> str:
    """Remove a secret value from a string before it is logged or returned."""
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def validate_target(url: str, method: str = "POST") -> tuple[str, str]:
    """Validate a URL and method against the allowlist and SSRF guard.

    Returns:
        The normalized ``(method, host)``.

    Raises:
        HttpToolError: If the method, scheme, or host is not permitted. Nothing
            is sent and no secret is resolved when this raises.
    """
    if not url or not isinstance(url, str):
        raise HttpToolError("url is required")

    normalized_method = str(method or "POST").upper()
    if normalized_method not in ALLOWED_METHODS:
        raise HttpToolError(
            f"Unsupported method '{normalized_method}'. Allowed: {sorted(ALLOWED_METHODS)}"
        )

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HttpToolError("url must be an http(s) URL")
    host = parsed.hostname
    if not host:
        raise HttpToolError("url is missing a host")

    allowlist = load_allowed_hosts()
    if host.lower() not in allowlist:
        raise HttpToolError(
            f"Host '{host}' is not allowlisted, so no request was made. "
            f"Allowed hosts: {sorted(allowlist)}. "
            "An administrator can permit it via HTTP_TOOL_ALLOWED_HOSTS."
        )

    check_ssrf(host)
    return normalized_method, host


async def resolve_auth_header(
    session_id: Optional[str],
    secret_name: Optional[str],
    headers: dict[str, str],
    *,
    auth_header: Optional[str] = None,
    auth_prefix: Optional[str] = None,
) -> Optional[str]:
    """Resolve ``secret_name`` from the session store and inject it into ``headers``.

    Mutates ``headers`` in place and returns the resolved secret value so the
    caller can scrub it from any error text. The value is never accepted as an
    argument, so it cannot appear in the tool call the model produced.

    Raises:
        HttpToolError: If the named key is not stored, or if the caller supplied
            an uninterpolated placeholder header instead of using ``secret_name``.
    """
    from api.secret_store import get_secret_session_store

    store = get_secret_session_store()

    # Guard the common mistake: interpolating a secret into a manual header
    # (e.g. "Authorization": "Bearer ${NGC_API_KEY}") instead of using
    # secret_name. Steer the model to secret_name and surface valid names.
    if not secret_name:
        placeholder_headers = [k for k, v in headers.items() if has_placeholder(v)]
        if placeholder_headers:
            available = [s.name for s in await store.list_secrets(session_id)]
            raise HttpToolError(
                "Do not put a secret or a placeholder like '${...}'/'$VAR' in "
                f"headers ({placeholder_headers}). Instead set `secret_name` to a "
                f"stored key name. Available key names: "
                f"{available or '(none stored — ask the user to add one)'}."
            )
        return None

    secret_value = await store.get_value(session_id, secret_name)
    if not secret_value:
        available = [s.name for s in await store.list_secrets(session_id)]
        raise HttpToolError(
            f"No stored API key named '{secret_name}' for this session. "
            f"Available key names: {available or '(none)'}. "
            "Retry with `secret_name` set to one of those, or ask the user to "
            "add the key via the Add Key panel."
        )

    header_name = auth_header or "Authorization"
    prefix = "Bearer " if auth_prefix is None else auth_prefix
    headers[header_name] = f"{prefix}{secret_value}"
    return secret_value


def clamp_timeout(value: Any) -> float:
    """Coerce a requested timeout into the permitted range."""
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = float(DEFAULT_TIMEOUT_SECONDS)
    return max(1.0, min(timeout, float(MAX_TIMEOUT_SECONDS)))


@dataclass
class CappedResponse:
    """An HTTP response whose body has been truncated to ``MAX_RESPONSE_BYTES``."""

    status_code: int
    content: bytes
    truncated: bool


async def send_request(
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    timeout: float = float(DEFAULT_TIMEOUT_SECONDS),
    secret_value: Optional[str] = None,
) -> CappedResponse:
    """Send a request with redirects disabled and the response body capped.

    Redirects are never followed: a redirect could bounce an allowlisted host to
    an arbitrary one, carrying the injected credential with it.

    Raises:
        HttpToolError: On any transport failure, with the secret scrubbed.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as an error
        message = scrub(str(exc), secret_value)
        logger.warning("HTTP request to %s failed: %s", url, message)
        raise HttpToolError(f"HTTP request failed: {message}") from exc

    return cap_response(response)


def cap_response(response: Any) -> CappedResponse:
    """Truncate a response body to ``MAX_RESPONSE_BYTES``."""
    body = response.content or b""
    truncated = False
    if len(body) > MAX_RESPONSE_BYTES:
        body = body[:MAX_RESPONSE_BYTES]
        truncated = True
    return CappedResponse(
        status_code=response.status_code, content=body, truncated=truncated
    )
