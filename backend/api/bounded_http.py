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

"""HTTP reads with a ceiling on how much of the body is held in memory.

MCP servers and self-hosted NIMs are reached at URLs a user registers, so the
size of what comes back is theirs to decide. ``httpx`` will happily buffer a
body of any size; this reads it as a stream and stops once it passes
``MAX_RESPONSE_BYTES``.
"""

from typing import Any, Optional

import httpx

# Largest decoded response body accepted from an MCP server or NIM. Looked up
# on every call rather than bound as a default argument, so tests can lower it.
MAX_RESPONSE_BYTES = 1024 ** 3

# These describe the body as it was on the wire. The rebuilt response carries
# the decoded body, so keeping them would make httpx decode it a second time.
_WIRE_FRAMING_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class ResponseTooLargeError(Exception):
    """A response body exceeded ``MAX_RESPONSE_BYTES`` and was discarded."""

    def __init__(self, url: str, limit: int, declared: Optional[int] = None):
        self.url = url
        self.limit = limit
        self.declared = declared
        detail = f" (Content-Length {declared} bytes)" if declared is not None else ""
        super().__init__(
            f"Response from {url} exceeded the {format_limit(limit)} limit{detail}"
        )


def format_limit(limit: Optional[int] = None) -> str:
    """Render a byte count the way an error message should show it, e.g. "1 GiB"."""
    value = float(MAX_RESPONSE_BYTES if limit is None else limit)
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            break
        value /= 1024
    return f"{value:g} {unit}"


async def request_bounded(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """Send a request and return the response with its body fully read.

    The limit applies to the decoded body, so a compressed response that
    inflates past it is stopped too. The returned response is an ordinary,
    already-read ``httpx.Response``: ``.json()``, ``.text`` and ``.headers``
    work as they would after ``client.request``.

    Raises:
        ResponseTooLargeError: The body, declared or actual, exceeded
            ``MAX_RESPONSE_BYTES``.
        httpx.HTTPError: Transport failures, exactly as ``client.request``
            would raise them.
    """
    limit = MAX_RESPONSE_BYTES
    async with client.stream(method, url, **kwargs) as response:
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise ResponseTooLargeError(url, limit, declared=int(declared))

        body = bytearray()
        async for chunk in response.aiter_bytes():
            body += chunk
            if len(body) > limit:
                raise ResponseTooLargeError(url, limit)

    headers = [
        (name, value)
        for name, value in response.headers.multi_items()
        if name.lower() not in _WIRE_FRAMING_HEADERS
    ]
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=bytes(body),
        request=response.request,
    )
