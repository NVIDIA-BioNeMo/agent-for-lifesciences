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

"""Graph nodes must actually receive the request's ``config``.

Services that cannot be serialised into graph state -- the storage service
above all -- travel in ``config["configurable"]``. LangGraph only injects that
config into nodes whose ``config`` parameter it recognises by annotation: since
1.x an annotation it does not accept is skipped with a warning, and the node
runs with ``config=None`` instead. Nothing fails loudly when that happens. The
refresh node simply reports every session as having no artifacts, which wiped
the storage block from the model's context and stopped the storage_update SSE
event the frontend's storage strip listens for.
"""

import json
import sys
import types
import warnings
from importlib import import_module

import pytest

from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

SESSION = "sess_graph"
REQUEST = "req_graph"

CONFIG_WARNING = "config' parameter should be typed"


class FakeRedis:
    """The subset of the async redis client the refresh path touches."""

    def __init__(self):
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value


class StubLLM:
    """Enough of a chat model for the graph builders to bind tools to."""

    def bind_tools(self, tools):
        return self


@pytest.fixture
def refresh_backends(monkeypatch):
    """Run the real refresh path against in-memory storage and a fake Redis.

    Only the viewer and selection lookups are stubbed; storage state is left
    real, because it is the part under test.
    """
    pytest.importorskip("langgraph")
    redis = FakeRedis()

    async def no_viewer_state(_request_id):
        return None

    async def no_selections(_redis_client, _session_id):
        return []

    # api.redis is replaced wholesale rather than patched. Importing it pulls in
    # the third-party redis package, and pytest puts backend/api on sys.path
    # (api/tests is a package, api is not), so "redis" resolves to api/redis.py
    # unless something else imported the real one first. Standing in a stub
    # keeps this file passing however it is invoked.
    stub = types.ModuleType("api.redis")
    stub.get_redis = lambda: redis
    monkeypatch.setitem(sys.modules, "api.redis", stub)

    # These resolve through sys.modules at call time, so patch the module object
    # itself; import_module avoids relying on the parent package's attributes.
    monkeypatch.setattr(
        import_module("api.tools.viewer_state"),
        "get_current_viewer_state",
        no_viewer_state,
    )
    monkeypatch.setattr(
        import_module("api.selections"), "get_available_selections", no_selections
    )
    return redis


@pytest.fixture
def stored_sdf():
    """A session with one artifact in it, reachable only through config."""
    service = DataStorageService(repository=InMemoryDataArtifactRepository())
    return SessionScopedStorage(service, SESSION)


def empty_state():
    return {
        "messages": [],
        "viewer_state": {},
        "storage_state": {},
        "selections": [],
        "findings": [],
        "request_id": REQUEST,
        "session_id": SESSION,
        "total_episode_input_tokens": 0,
        "total_episode_output_tokens": 0,
        "iteration_count": 0,
    }


async def one_node_graph():
    """Compile the real refresh node on its own so nothing else can supply it."""
    from langgraph.graph import END, START, StateGraph

    from api.graph.agent import _refresh_node
    from api.graph.state import AevomState

    graph = StateGraph(AevomState)
    graph.add_node("refresh_state", _refresh_node)
    graph.add_edge(START, "refresh_state")
    graph.add_edge("refresh_state", END)
    return graph.compile()


@pytest.mark.asyncio
class TestTheRefreshNodeSeesTheConfig:
    async def test_it_reports_the_sessions_artifacts(
        self, refresh_backends, stored_sdf
    ):
        reference = await stored_sdf.store("sdf", b"aspirin", description="aspirin")
        compiled = await one_node_graph()

        result = await compiled.ainvoke(
            empty_state(), config={"configurable": {"storage": stored_sdf}}
        )

        storage_state = result["storage_state"]
        assert storage_state["is_empty"] is False
        assert [a["reference"] for a in storage_state["artifacts"]] == [reference]

    async def test_it_publishes_them_for_the_next_turn_to_read(
        self, refresh_backends, stored_sdf
    ):
        # The state block the model sees is rebuilt from Redis, not from the
        # node's return value.
        await stored_sdf.store("pdb", b"ATOM", description="EGFR")
        compiled = await one_node_graph()

        await compiled.ainvoke(
            empty_state(), config={"configurable": {"storage": stored_sdf}}
        )

        published = json.loads(refresh_backends.values[f"storage:{SESSION}"])
        assert published["is_empty"] is False
        assert published["artifacts"][0]["description"] == "EGFR"

    async def test_an_empty_session_is_still_reported_as_empty(
        self, refresh_backends, stored_sdf
    ):
        compiled = await one_node_graph()

        result = await compiled.ainvoke(
            empty_state(), config={"configurable": {"storage": stored_sdf}}
        )

        assert result["storage_state"]["is_empty"] is True


class TestNoNodeSignatureIsSkipped:
    """Covers the workflow subgraphs too, whose nodes are local closures."""

    def test_building_the_agent_graph_warns_about_no_config_parameter(self):
        pytest.importorskip("langgraph")
        from api.graph.agent import build_agent_graph

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_agent_graph(StubLLM(), [], "system prompt")

        skipped = [w for w in caught if CONFIG_WARNING in str(w.message)]
        assert skipped == [], [str(w.message) for w in skipped]
