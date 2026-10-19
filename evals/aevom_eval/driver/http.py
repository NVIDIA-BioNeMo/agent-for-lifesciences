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

"""Minimal HTTP helpers over urllib, for use inside the Harbor environment.

``main`` is a bare ``python:3.12-slim``, so no requests/httpx. The one thing
worth care here is that ``/api/chat`` is an SSE stream: it must be read
incrementally, because the harness has to react to a ``structure_query`` event
while the request is still open. ``urllib.request.urlopen`` returns a file-like
object that supports exactly that.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any


class HttpError(RuntimeError):
    """A non-2xx response, carrying the status and body for diagnosis."""

    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:500]}")
        self.status = status
        self.url = url
        self.body = body


def _request(
    method: str,
    url: str,
    payload: Any | None = None,
    timeout: float = 60.0,
    token: str | None = None,
) -> urllib.request.addinfourl:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, url, e.read().decode("utf-8", "replace")) from e


def get_json(url: str, timeout: float = 60.0, token: str | None = None) -> Any:
    with _request("GET", url, timeout=timeout, token=token) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def post_json(
    url: str, payload: Any, timeout: float = 60.0, token: str | None = None
) -> Any:
    with _request("POST", url, payload, timeout=timeout, token=token) as response:
        body = response.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else None


def stream_sse(
    url: str, payload: Any, timeout: float = 1800.0, token: str | None = None
) -> Iterator[dict]:
    """Yield each decoded ``data:`` payload from an SSE response as it arrives.

    Non-JSON frames and SSE comments are skipped rather than raising: a
    malformed frame mid-stream should not discard the turn recorded so far.
    """
    response = _request("POST", url, payload, timeout=timeout, token=token)
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if not body:
                continue
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event
    finally:
        response.close()


def wait_for_health(
    url: str,
    timeout_sec: float,
    interval_sec: float = 2.0,
) -> dict:
    """Poll ``url`` until it answers, returning the parsed body.

    Compose healthchecks already gate ``depends_on``, but the agent may still
    arrive during the api's own startup (database migrations, tool registry
    construction), so this is a second gate with a clear error.
    """
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return get_json(url, timeout=10.0)
        except Exception as e:
            last_error = e
            time.sleep(interval_sec)
    raise TimeoutError(
        f"{url} did not become healthy within {timeout_sec}s; last error: {last_error}"
    )
