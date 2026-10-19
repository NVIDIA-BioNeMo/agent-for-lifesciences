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

"""MCPClient against a user-registered URL: redirects and oversized responses.

Only the wire is stubbed, so the JSON-RPC framing, the endpoint-path fallback
in ``connect()`` and the error translation stay under test.
"""

import json

import httpx
import pytest

from api.mcp.client import (
    MCPClient,
    MCPConnectionError,
    MCPResponseTooLargeError,
    MCPToolSchema,
)
from api.mcp.wrapper import MCPToolWrapper
from api.tools.base import ContentType

INITIALIZE_RESULT = {
    "protocolVersion": "2024-11-05",
    "serverInfo": {"name": "test-server"},
}


class FakeMCPServer:
    """Answers JSON-RPC by method; records every URL it was sent."""

    def __init__(self, by_method=None, by_path=None):
        self.by_method = by_method or {}
        self.by_path = by_path or {}
        self.seen_urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen_urls.append(str(request.url))
        if request.url.path in self.by_path:
            return self.by_path[request.url.path]
        body = json.loads(request.content)
        answer = self.by_method.get(body["method"])
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": answer or {}})


@pytest.fixture
def fake_server(monkeypatch):
    real_client = httpx.AsyncClient

    def _install(server: FakeMCPServer) -> FakeMCPServer:
        def factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return real_client(*args, transport=httpx.MockTransport(server), **kwargs)

        monkeypatch.setattr("api.mcp.client.httpx.AsyncClient", factory)
        return server

    return _install


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr("api.bounded_http.MAX_RESPONSE_BYTES", 512)


@pytest.mark.asyncio
class TestOversizedResponses:
    async def test_an_oversized_tool_result_reaches_the_agent_as_a_clear_error(
        self, fake_server, small_limit
    ):
        fake_server(FakeMCPServer(by_method={
            "initialize": INITIALIZE_RESULT,
            "tools/call": {"content": [{"type": "text", "text": "x" * 2000}]},
        }))
        client = MCPClient("http://mcp.internal")
        await client.connect()
        tool = MCPToolWrapper(
            mcp_client=client,
            schema=MCPToolSchema(name="search", description="Search"),
            server_id="srv1",
            server_name="Internal",
        )

        result = await tool.execute({"query": "everything"})

        assert result.content_type == ContentType.ERROR
        assert "larger than Aevom's 512 bytes limit" in result.data
        assert "less data" in result.data

    async def test_an_oversized_initialize_is_not_retried_on_other_paths(
        self, fake_server, small_limit
    ):
        server = fake_server(FakeMCPServer(by_method={
            "initialize": {**INITIALIZE_RESULT, "padding": "x" * 2000},
        }))

        with pytest.raises(MCPResponseTooLargeError):
            await MCPClient("http://mcp.internal").connect()

        assert server.seen_urls == ["http://mcp.internal"]


@pytest.mark.asyncio
class TestRedirects:
    async def test_a_redirect_is_not_followed_and_the_target_is_reported(self, fake_server):
        redirect = httpx.Response(307, headers={"location": "http://mcp.internal/mcp/"})
        server = fake_server(FakeMCPServer(by_path={
            "": redirect,
            "/": redirect,
            "/mcp": httpx.Response(404),
            "/sse": httpx.Response(404),
            "/api/mcp": httpx.Response(404),
        }))

        with pytest.raises(MCPConnectionError) as exc_info:
            await MCPClient("http://mcp.internal").connect()

        message = str(exc_info.value)
        assert "http://mcp.internal/mcp/" in message
        assert "does not follow redirects" in message
        assert not any(url.endswith("/mcp/") for url in server.seen_urls)
