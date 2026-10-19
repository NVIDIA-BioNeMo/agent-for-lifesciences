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

"""Size-capped reads from user-registered endpoints."""

import gzip
import json

import httpx
import pytest

from api import bounded_http
from api.bounded_http import ResponseTooLargeError, format_limit, request_bounded


class _ChunkedStream(httpx.AsyncByteStream):
    """A body with no Content-Length, delivered in pieces."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.sent = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.sent += 1
            yield chunk


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(bounded_http, "MAX_RESPONSE_BYTES", 100)
    return 100


@pytest.mark.asyncio
class TestRequestBounded:
    async def test_a_body_under_the_limit_reads_like_a_normal_response(self, small_limit):
        def handler(request):
            return httpx.Response(200, json={"ok": True}, headers={"x-extra": "kept"})

        async with _client(handler) as client:
            response = await request_bounded(client, "GET", "http://nim/v1/models")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert response.headers["x-extra"] == "kept"

    async def test_a_body_exactly_at_the_limit_is_accepted(self, small_limit):
        async with _client(lambda r: httpx.Response(200, content=b"x" * small_limit)) as client:
            response = await request_bounded(client, "GET", "http://nim/")

        assert len(response.content) == small_limit

    async def test_an_oversized_content_length_is_refused_before_reading(self, small_limit):
        stream = _ChunkedStream([b"x" * 200])

        def handler(request):
            return httpx.Response(200, headers={"content-length": "200"}, stream=stream)

        async with _client(handler) as client:
            with pytest.raises(ResponseTooLargeError) as exc_info:
                await request_bounded(client, "GET", "http://nim/")

        assert exc_info.value.declared == 200
        assert stream.sent == 0

    async def test_an_oversized_chunked_body_is_stopped_mid_stream(self, small_limit):
        stream = _ChunkedStream([b"x" * 60] * 10)

        async with _client(lambda r: httpx.Response(200, stream=stream)) as client:
            with pytest.raises(ResponseTooLargeError) as exc_info:
                await request_bounded(client, "GET", "http://nim/")

        assert exc_info.value.declared is None
        assert stream.sent == 2

    async def test_the_limit_applies_to_the_decoded_body(self, small_limit):
        # 10 KB of zeros compresses to a few dozen bytes on the wire.
        compressed = gzip.compress(b"0" * 10_000)
        assert len(compressed) < small_limit

        def handler(request):
            return httpx.Response(
                200, headers={"content-encoding": "gzip"}, content=compressed
            )

        async with _client(handler) as client:
            with pytest.raises(ResponseTooLargeError):
                await request_bounded(client, "GET", "http://nim/")

    async def test_a_compressed_body_under_the_limit_is_decoded_once(self, small_limit):
        body = json.dumps({"id": "model"}).encode()

        def handler(request):
            return httpx.Response(
                200, headers={"content-encoding": "gzip"}, content=gzip.compress(body)
            )

        async with _client(handler) as client:
            response = await request_bounded(client, "GET", "http://nim/")

        assert response.json() == {"id": "model"}
        assert "content-encoding" not in response.headers

    async def test_redirects_are_returned_not_followed(self, small_limit):
        def handler(request):
            return httpx.Response(302, headers={"location": "http://elsewhere/"})

        async with _client(handler) as client:
            response = await request_bounded(client, "GET", "http://nim/")

        assert response.is_redirect
        assert response.headers["location"] == "http://elsewhere/"


class TestFormatLimit:
    @pytest.mark.parametrize(
        ("limit", "expected"),
        [
            (100, "100 bytes"),
            (1024, "1 KiB"),
            (5 * 1024 ** 2, "5 MiB"),
            (1024 ** 3, "1 GiB"),
            (1536 * 1024 ** 3, "1536 GiB"),
        ],
    )
    def test_renders_human_units(self, limit, expected):
        assert format_limit(limit) == expected

    def test_defaults_to_the_module_limit(self):
        assert format_limit() == "1 GiB"
