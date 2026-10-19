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

"""An MCP server's tools stay withheld until a human authorises them.

Aevom held two extension paths to very different standards. A skill that wants
to call a single HTTP endpoint yields no tool at all until its contract is
confirmed, and every request it then makes is checked against an allowlist and
an SSRF guard. An MCP server -- which contributes a whole toolset of tools
Aevom has never seen, at a URL the user supplies -- was bound into the agent on
one unreviewed POST. The weaker path guarded the smaller capability.

These tests pin the gate that closes that gap, at the chokepoint that matters:
not "are unconfirmed tools hidden from the model", but "are they unreachable".
Hiding alone would leave them dispatchable by name.
"""

import pytest

from api.mcp.session_store import MCPServerConfig, MCPServerConnection, MCPSessionStore


class FakeClient:
    async def disconnect(self):
        return None


class FakeTool:
    """Enough of an MCP tool wrapper for the store to enumerate it."""

    def __init__(self, name: str):
        self.name = name
        self.original_name = name

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": "", "parameters": {}},
        }


def connection(server_id: str = "srv1", confirmed: bool = False) -> MCPServerConnection:
    return MCPServerConnection(
        config=MCPServerConfig(
            id=server_id, name="Example", url="https://mcp.example.com",
            confirmed=confirmed,
        ),
        client=FakeClient(),
        tools=[FakeTool("mcp_example_search")],
    )


@pytest.fixture
def store(monkeypatch):
    """A store whose Redis round-trip is stubbed out to a plain dict."""
    s = MCPSessionStore()
    saved: dict[str, dict[str, MCPServerConfig]] = {}

    async def _load(session_id):
        return dict(saved.get(session_id, {}))

    async def _save(session_id, configs):
        saved[session_id] = dict(configs)

    monkeypatch.setattr(s, "_load_configs", _load)
    monkeypatch.setattr(s, "_save_configs", _save)
    s._saved = saved  # test handle
    return s


def bind(store, conn: MCPServerConnection, session_id: str = "s1") -> None:
    """Put a connection in the local cache and its config in the fake Redis."""
    store._local_cache.setdefault(session_id, {})[conn.id] = conn
    store._saved.setdefault(session_id, {})[conn.id] = conn.config


@pytest.mark.asyncio
class TestUnconfirmedServersAreUnreachable:
    async def test_the_model_is_not_offered_the_tools(self, store, monkeypatch):
        conn = connection(confirmed=False)
        bind(store, conn)
        monkeypatch.setattr(store, "get_servers", lambda sid: _servers([conn]))

        assert await store.get_tool_schemas("s1") == []

    async def test_the_tools_cannot_be_dispatched_by_name_either(
        self, store, monkeypatch
    ):
        # The half that hiding alone would miss: the registry resolves a tool
        # by name before executing it, so withholding must cover that path too.
        conn = connection(confirmed=False)
        bind(store, conn)
        monkeypatch.setattr(store, "get_servers", lambda sid: _servers([conn]))

        assert await store.get_tool_by_name("s1", "mcp_example_search") is None

    async def test_a_confirmed_server_is_fully_available(self, store, monkeypatch):
        conn = connection(confirmed=True)
        bind(store, conn)
        monkeypatch.setattr(store, "get_servers", lambda sid: _servers([conn]))

        assert len(await store.get_tool_schemas("s1")) == 1
        assert await store.get_tool_by_name("s1", "mcp_example_search") is not None


@pytest.mark.asyncio
class TestConfirming:
    async def test_confirming_makes_the_tools_reachable(self, store, monkeypatch):
        conn = connection(confirmed=False)
        bind(store, conn)
        monkeypatch.setattr(store, "get_servers", lambda sid: _servers([conn]))

        assert await store.get_tool_schemas("s1") == []
        assert await store.confirm_server("s1", "srv1") is True
        assert len(await store.get_tool_schemas("s1")) == 1

    async def test_confirmation_reaches_the_live_connection_not_just_redis(
        self, store
    ):
        # The in-process connection holds its own config object; if only Redis
        # were updated the running process would keep withholding.
        conn = connection(confirmed=False)
        bind(store, conn)

        await store.confirm_server("s1", "srv1")

        assert conn.config.confirmed is True

    async def test_confirmation_is_persisted(self, store):
        conn = connection(confirmed=False)
        bind(store, conn)

        await store.confirm_server("s1", "srv1")

        assert store._saved["s1"]["srv1"].confirmed is True

    async def test_confirming_twice_is_a_no_op(self, store):
        bind(store, connection(confirmed=False))

        assert await store.confirm_server("s1", "srv1") is True
        assert await store.confirm_server("s1", "srv1") is True

    async def test_an_unknown_server_is_reported_not_silently_accepted(self, store):
        assert await store.confirm_server("s1", "does-not-exist") is False


class TestTheStoredFlag:
    def test_defaults_to_unconfirmed(self):
        config = MCPServerConfig(id="a", name="n", url="u")
        assert config.confirmed is False

    def test_survives_a_round_trip_through_redis(self):
        config = MCPServerConfig(id="a", name="n", url="u", confirmed=True)
        assert MCPServerConfig.from_dict(config.to_dict()).confirmed is True

    def test_a_config_stored_before_this_shipped_is_treated_as_unconfirmed(self):
        # Fail closed: servers already in Redis must not be grandfathered in.
        legacy = {"id": "a", "name": "n", "url": "u", "connected_at": 0.0}
        assert MCPServerConfig.from_dict(legacy).confirmed is False

    def test_clients_can_tell_offered_tools_from_reachable_ones(self):
        payload = connection(confirmed=False).to_dict()
        assert payload["tools_count"] == 1
        assert payload["confirmed"] is False


async def _servers(conns):
    return conns
