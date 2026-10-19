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

"""The first LLM call of a turn must see the same envelope as every later one.

The graph runs ``START -> agent``, so the turn's *planning* call -- the one that
decides what the turn will do -- reads whatever the streaming layer seeded. That
seed used to be hand-rolled, and it omitted candidates, exclusions and tool
budgets that ``refresh_state_from_backends`` supplies from the second iteration
onward. The workflow subgraphs were worse: seeded with a literal
``{"is_empty": True}`` storage state, the first call inside a workflow was told
``[STORAGE STATE: Empty]`` while the parent session held artifacts. Missing is
bad; false is worse.

Both seeds now go through the refresh function, so these tests assert on the
seeded state and on the prompt block it produces.
"""

import json
import sys
import types

import pytest

from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

SESSION = "sess_planning"
REQUEST = "req_planning"

CANDIDATES = [
    {
        "smiles": "Oc1ccc(O)cc1",
        "passed": False,
        "gates": [
            {
                "name": "pains_alerts",
                "value": 1,
                "bound": 0,
                "passed": False,
                "hits": ["catechol_A"],
            }
        ],
    }
]

FINDINGS = [
    {
        "turn_seq": 1,
        "iteration": 2,
        "summary": "vina_dock(lig=sdf_a1b2c3d4) -> -9.50 kcal/mol",
    }
]


class FakeRedis:
    """The subset of the async redis client the refresh path touches."""

    def __init__(self):
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value


class FakeRecorder:
    """Stands in for the request's TurnRecorder.

    Both ledger loaders fall back to the recorder when their config carries no
    session factory, which is how the subgraphs read them and how a caller
    without a database reads them.
    """

    def __init__(self, findings=(), candidates=()):
        self._findings = list(findings)
        self._candidates = list(candidates)

    async def get_findings(self):
        return list(self._findings)

    async def get_candidates(self):
        return list(self._candidates)


@pytest.fixture
def backends(monkeypatch):
    """Real refresh path against in-memory storage and a fake Redis.

    Mirrors test_graph_config_injection: api.redis is replaced wholesale
    because importing it resolves to the third-party package or to api/redis.py
    depending on what ran first.
    """
    pytest.importorskip("langgraph")
    redis = FakeRedis()

    async def no_viewer_state(_request_id):
        return None

    async def set_no_viewer_state(_request_id, _state):
        return None

    async def no_selections(_redis_client, _session_id):
        return []

    stub = types.ModuleType("api.redis")
    stub.get_redis = lambda: redis
    monkeypatch.setitem(sys.modules, "api.redis", stub)

    from importlib import import_module

    viewer_state = import_module("api.tools.viewer_state")
    monkeypatch.setattr(viewer_state, "get_current_viewer_state", no_viewer_state)
    monkeypatch.setattr(viewer_state, "set_viewer_state", set_no_viewer_state)
    monkeypatch.setattr(
        import_module("api.selections"), "get_available_selections", no_selections
    )
    return redis


@pytest.fixture
def storage():
    return SessionScopedStorage(
        DataStorageService(repository=InMemoryDataArtifactRepository()), SESSION
    )


@pytest.fixture
def budget():
    """One budgeted tool, so the [TOOL BUDGET] block has something to render."""
    from api.tools import budget as budget_module

    before = budget_module.active_budgets()
    budget_module.set_budgets({"vina_dock": 3})
    yield
    budget_module.set_budgets(before)


@pytest.fixture
def recorder():
    return FakeRecorder(findings=FINDINGS, candidates=CANDIDATES)


class CapturingGraph:
    """Records the state it was invoked with and yields no events."""

    def __init__(self):
        self.initial_state = None
        self.config = None

    def astream_events(self, initial_state, **kwargs):
        self.initial_state = initial_state
        self.config = kwargs.get("config")

        async def _empty():
            return
            yield  # pragma: no cover - unreachable, makes this a generator

        return _empty()


async def seeded_state(storage, recorder=None) -> dict:
    """Drive stream_graph far enough to capture the state it seeds the graph with."""
    from api.graph.streaming import stream_graph

    graph = CapturingGraph()
    async for _ in stream_graph(
        graph,
        system_prompt="BASE",
        conversation_history=[{"role": "user", "content": "dock this"}],
        viewer_state=None,
        session_id=SESSION,
        storage=storage,
        request_id=REQUEST,
        # No factory: the ledger loaders fall back to the recorder, which keeps
        # this test off a database without changing the path under test.
        db_session_factory=None,
        recorder=recorder,
    ):
        pass
    return graph.initial_state


@pytest.mark.asyncio
class TestThePlanningCallSeesTheSession:
    async def test_it_sees_the_sessions_artifacts(self, backends, storage):
        reference = await storage.store("sdf", b"aspirin", description="aspirin")

        state = await seeded_state(storage)

        assert state["storage_state"]["is_empty"] is False
        assert [a["reference"] for a in state["storage_state"]["artifacts"]] == [
            reference
        ]

    async def test_it_sees_the_candidate_ledger(self, backends, storage, recorder):
        state = await seeded_state(storage, recorder)

        assert [row["smiles"] for row in state["candidates"]] == ["Oc1ccc(O)cc1"]

    async def test_it_sees_the_exclusions_those_candidates_imply(
        self, backends, storage, recorder
    ):
        state = await seeded_state(storage, recorder)

        assert [rule["feature"] for rule in state["exclusions"]] == ["catechol_A"]

    async def test_it_sees_the_session_log(self, backends, storage, recorder):
        state = await seeded_state(storage, recorder)

        assert [row["summary"] for row in state["findings"]] == [
            FINDINGS[0]["summary"]
        ]

    async def test_it_sees_the_remaining_tool_budgets(
        self, backends, storage, budget
    ):
        state = await seeded_state(storage)

        assert [entry["tool_name"] for entry in state["tool_budgets"]] == [
            "vina_dock"
        ]

    async def test_no_state_key_goes_missing_when_a_ledger_is_unreadable(
        self, backends, storage
    ):
        """With no factory and no recorder both ledgers read as unknown."""
        state = await seeded_state(storage)

        for key in ("findings", "candidates", "exclusions", "tool_budgets"):
            assert state[key] == [], key

    async def test_the_run_uses_the_config_the_seed_was_built_from(
        self, backends, storage
    ):
        from api.graph.agent import GRAPH_RECURSION_LIMIT
        from api.graph.streaming import stream_graph

        graph = CapturingGraph()
        async for _ in stream_graph(
            graph,
            system_prompt="BASE",
            conversation_history=[{"role": "user", "content": "hi"}],
            viewer_state=None,
            session_id=SESSION,
            storage=storage,
            request_id=REQUEST,
            db_session_factory=None,
        ):
            pass

        assert graph.config["configurable"]["storage"] is storage
        assert graph.config["recursion_limit"] == GRAPH_RECURSION_LIMIT


@pytest.mark.asyncio
class TestThePlanningPromptCarriesTheBlocks:
    """The seeded dict is the mechanism; the prompt block is the symptom."""

    async def planning_prompt(self, storage, recorder=None) -> str:
        from api.context.assembler import ContextAssembler

        state = await seeded_state(storage, recorder)
        return ContextAssembler("BASE")._build_system_message(state).content

    async def test_candidates_and_exclusions_reach_the_prompt(
        self, backends, storage, recorder
    ):
        prompt = await self.planning_prompt(storage, recorder)

        assert "[CANDIDATES:" in prompt
        assert "[EXCLUSIONS:" in prompt
        assert "catechol_A" in prompt

    async def test_the_budget_reaches_the_prompt(self, backends, storage, budget):
        prompt = await self.planning_prompt(storage)

        assert "[TOOL BUDGET:" in prompt
        assert "vina_dock" in prompt

    async def test_the_prompt_does_not_claim_an_empty_session(
        self, backends, storage
    ):
        await storage.store("pdb", b"ATOM", description="EGFR")

        prompt = await self.planning_prompt(storage)

        assert "[STORAGE STATE: Empty" not in prompt
        assert "EGFR" in prompt


@pytest.mark.asyncio
class TestWorkflowSubgraphsSeeTheParentsSession:
    """The audit's false block: a workflow told the session was empty."""

    @pytest.fixture
    def captured(self, monkeypatch, backends):
        """Workflow tools whose subgraphs record the state they are handed."""
        graph = CapturingGraph()

        for module, name in (
            ("small_molecule", "build_small_molecule_graph"),
            ("structure_prediction", "build_structure_prediction_graph"),
            ("docking", "build_docking_graph"),
            ("protein_ligand", "build_protein_ligand_graph"),
        ):
            imported = __import__(f"api.graph.workflows.{module}", fromlist=[name])
            monkeypatch.setattr(imported, name, lambda *a, **k: graph)

        from api.graph import workflows

        tools = {
            tool.name: tool for tool in workflows.build_workflow_tools(object(), [])
        }
        return graph, tools

    @pytest.fixture
    def parent_context(self, storage, recorder):
        """The ContextVars the parent graph leaves for the workflow tool."""
        from api.conversations import reset_recorder, set_recorder
        from api.graph import tools as graph_tools

        storage_token = graph_tools._graph_storage_ctx.set(storage)
        session_token = graph_tools._graph_session_id_ctx.set(SESSION)
        request_token = graph_tools._graph_request_id_ctx.set(REQUEST)
        recorder_token = set_recorder(recorder)
        yield
        reset_recorder(recorder_token)
        graph_tools._graph_request_id_ctx.reset(request_token)
        graph_tools._graph_session_id_ctx.reset(session_token)
        graph_tools._graph_storage_ctx.reset(storage_token)

    async def test_the_child_sees_the_parents_artifacts(
        self, captured, parent_context, storage
    ):
        reference = await storage.store("pdb", b"ATOM", description="docked complex")
        graph, tools = captured

        await tools["run_docking_workflow"].coroutine(query="dock it")

        seeded = graph.initial_state
        assert seeded["storage_state"]["is_empty"] is False
        assert [a["reference"] for a in seeded["storage_state"]["artifacts"]] == [
            reference
        ]

    async def test_every_workflow_is_seeded_the_same_way(
        self, captured, parent_context, storage
    ):
        await storage.store("pdb", b"ATOM", description="docked complex")
        graph, tools = captured

        for name, tool in tools.items():
            graph.initial_state = None
            await tool.coroutine(query="go")
            assert graph.initial_state["storage_state"]["is_empty"] is False, name

    async def test_the_child_sees_the_candidate_ledger(
        self, captured, parent_context, storage
    ):
        graph, tools = captured

        await tools["run_small_molecule_analysis"].coroutine(query="go")

        seeded = graph.initial_state
        assert [row["smiles"] for row in seeded["candidates"]] == ["Oc1ccc(O)cc1"]
        assert [rule["feature"] for rule in seeded["exclusions"]] == ["catechol_A"]

    async def test_the_child_still_gets_the_session_log(
        self, captured, parent_context, storage
    ):
        graph, tools = captured

        await tools["run_structure_prediction"].coroutine(query="fold it")

        assert [row["summary"] for row in graph.initial_state["findings"]] == [
            FINDINGS[0]["summary"]
        ]

    async def test_the_query_is_the_childs_only_message(
        self, captured, parent_context, storage
    ):
        """State crosses the boundary; the parent's trajectory does not."""
        graph, tools = captured

        await tools["run_docking_workflow"].coroutine(query="dock it")

        messages = graph.initial_state["messages"]
        assert len(messages) == 1
        assert messages[0].content == "dock it"


@pytest.mark.asyncio
class TestAnUnreachableBackendDoesNotEndTheTurn:
    """A subgraph that cannot read the session runs with an empty envelope.

    Refusing to start the workflow would be the worse failure: the seed is
    context, not a precondition.
    """

    @pytest.fixture
    def broken_redis(self, monkeypatch):
        pytest.importorskip("langgraph")

        def unreachable():
            raise RuntimeError("REDIS_URL environment variable is not set")

        stub = types.ModuleType("api.redis")
        stub.get_redis = unreachable
        monkeypatch.setitem(sys.modules, "api.redis", stub)

    async def test_the_workflow_still_runs(self, broken_redis, monkeypatch):
        graph = CapturingGraph()
        for module, name in (
            ("small_molecule", "build_small_molecule_graph"),
            ("structure_prediction", "build_structure_prediction_graph"),
            ("docking", "build_docking_graph"),
            ("protein_ligand", "build_protein_ligand_graph"),
        ):
            imported = __import__(f"api.graph.workflows.{module}", fromlist=[name])
            monkeypatch.setattr(imported, name, lambda *a, **k: graph)

        from api.graph import workflows

        tools = {
            tool.name: tool for tool in workflows.build_workflow_tools(object(), [])
        }

        await tools["run_docking_workflow"].coroutine(query="dock it")

        assert graph.initial_state["storage_state"] == {
            "is_empty": True,
            "artifacts": [],
        }


class TestNoSeedRestatesTheEnvelopeByHand:
    """Both seeds must keep going through the one function that builds state.

    Source-level, in the style of TestWorkflowSubgraphWiring: a future hand-
    rolled seed is exactly how this defect arrived.
    """

    def _source(self, *parts) -> str:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        return path.read_text(encoding="utf-8")

    def test_streaming_seeds_through_the_refresh_function(self):
        assert "refresh_state_from_backends(initial_state" in self._source(
            "graph", "streaming.py"
        )

    def test_the_subgraph_seeds_through_the_refresh_function(self):
        assert "refresh_state_from_backends(initial_state" in self._source(
            "graph", "workflows", "__init__.py"
        )

    def test_no_subgraph_seed_asserts_an_empty_storage_state(self):
        source = self._source("graph", "workflows", "__init__.py")
        assert '"storage_state": {"is_empty": True}' not in source


class TestTheStateShapeStaysCovered:
    """Every field the assembler reads must exist in the seed.

    A field added to AevomState and to the refresh function but not to the
    seeds reintroduces the defect for that field alone, silently.
    """

    def test_the_seed_names_every_envelope_field(self):
        pytest.importorskip("langgraph")
        from pathlib import Path

        from api.graph.state import AevomState

        envelope_fields = {
            "viewer_state",
            "storage_state",
            "selections",
            "findings",
            "tool_budgets",
            "candidates",
            "exclusions",
        }
        assert envelope_fields <= set(AevomState.__annotations__)

        for parts in (("graph", "streaming.py"), ("graph", "workflows", "__init__.py")):
            source = Path(__file__).resolve().parents[1].joinpath(*parts)
            text = source.read_text(encoding="utf-8")
            for field in envelope_fields:
                assert f'"{field}":' in text, f"{parts[-1]} does not seed {field}"


def test_the_published_storage_state_survives_the_seed(monkeypatch):
    """The seed publishes to Redis, which is where the next turn reads it."""
    import asyncio

    async def run():
        redis = FakeRedis()
        stub = types.ModuleType("api.redis")
        stub.get_redis = lambda: redis
        monkeypatch.setitem(sys.modules, "api.redis", stub)

        from importlib import import_module

        viewer_state = import_module("api.tools.viewer_state")

        async def none_viewer(_request_id):
            return None

        async def set_none(_request_id, _state):
            return None

        async def no_selections(_redis_client, _session_id):
            return []

        monkeypatch.setattr(viewer_state, "get_current_viewer_state", none_viewer)
        monkeypatch.setattr(viewer_state, "set_viewer_state", set_none)
        monkeypatch.setattr(
            import_module("api.selections"), "get_available_selections", no_selections
        )

        scoped = SessionScopedStorage(
            DataStorageService(repository=InMemoryDataArtifactRepository()), SESSION
        )
        await scoped.store("pdb", b"ATOM", description="EGFR")
        await seeded_state(scoped)

        published = json.loads(redis.values[f"storage:{SESSION}"])
        assert published["artifacts"][0]["description"] == "EGFR"

    pytest.importorskip("langgraph")
    asyncio.run(run())
