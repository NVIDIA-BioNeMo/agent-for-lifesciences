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

"""The graph's super-step budget must scale with the agent iteration cap.

One agent iteration walks four nodes, so a fixed recursion_limit silently
becomes the binding constraint as soon as MAX_ITERATIONS is raised: the run
dies with GraphRecursionError partway to the cap instead of stopping cleanly
through force_respond. These tests pin that relationship.
"""

from pathlib import Path

import pytest


def _limits():
    """Import the constants, skipping where langgraph is not installed.

    api.graph.agent pulls in langgraph, which is present in the runtime image
    but not necessarily in a bare local venv.
    """
    pytest.importorskip("langgraph")
    from api.graph.agent import (
        GRAPH_RECURSION_LIMIT,
        MAX_ITERATIONS,
        NODES_PER_ITERATION,
    )

    return MAX_ITERATIONS, NODES_PER_ITERATION, GRAPH_RECURSION_LIMIT


class TestRecursionBudgetCoversTheCap:
    def test_budget_exceeds_the_nodes_a_full_run_needs(self):
        """Every iteration must be reachable before LangGraph hard-stops."""
        cap, per_iteration, budget = _limits()
        assert budget > cap * per_iteration

    def test_budget_leaves_room_for_the_closing_nodes(self):
        """START, the final agent turn, and force_respond all cost steps too."""
        cap, per_iteration, budget = _limits()
        assert budget - (cap * per_iteration) >= 10

    def test_nodes_per_iteration_matches_the_graph(self):
        """agent -> tools -> await_pending_ops -> refresh_state."""
        _, per_iteration, _ = _limits()
        assert per_iteration == 4

    def test_cap_is_positive(self):
        cap, _, _ = _limits()
        assert cap > 0


class TestStreamingWiring:
    """Needs no imports, so it guards the wiring even in a bare venv."""

    def test_streaming_uses_the_derived_budget(self):
        source = Path(__file__).resolve().parents[1] / "graph" / "streaming.py"
        text = source.read_text(encoding="utf-8")
        assert '"recursion_limit": GRAPH_RECURSION_LIMIT' in text

    def test_streaming_has_no_hardcoded_recursion_limit(self):
        source = Path(__file__).resolve().parents[1] / "graph" / "streaming.py"
        text = source.read_text(encoding="utf-8")
        assert '"recursion_limit": 100' not in text


def _workflows_source() -> str:
    path = Path(__file__).resolve().parents[1] / "graph" / "workflows" / "__init__.py"
    return path.read_text(encoding="utf-8")


class TestWorkflowSubgraphWiring:
    """No subgraph routes to force_respond, so its budget is its only stop."""

    def test_workflow_budget_derives_from_the_node_cycle(self):
        assert "WORKFLOW_MAX_ITERATIONS * NODES_PER_ITERATION" in _workflows_source()

    def test_workflow_has_no_hardcoded_recursion_limit(self):
        assert '"recursion_limit": 100' not in _workflows_source()


class TestExhaustedWorkflowIsNotFatal:
    """A subgraph that runs out of budget must not end the user's turn.

    Raised out of the workflow tool, GraphRecursionError leaves through
    LangGraph's tool error handler and kills the whole chat request, discarding
    everything the child measured.
    """

    @pytest.fixture
    def workflow_tools(self, monkeypatch):
        pytest.importorskip("langgraph")
        from langgraph.errors import GraphRecursionError

        from api.graph import workflows

        class _ExhaustedGraph:
            async def astream_events(self, *args, **kwargs):
                raise GraphRecursionError(
                    f"Recursion limit of {kwargs['config']['recursion_limit']} reached"
                )
                yield  # pragma: no cover - unreachable, makes this a generator

        for module, name in (
            ("small_molecule", "build_small_molecule_graph"),
            ("structure_prediction", "build_structure_prediction_graph"),
            ("docking", "build_docking_graph"),
            ("protein_ligand", "build_protein_ligand_graph"),
        ):
            imported = __import__(
                f"api.graph.workflows.{module}", fromlist=[name]
            )
            monkeypatch.setattr(imported, name, lambda *a, **k: _ExhaustedGraph())

        return {
            tool.name: tool
            for tool in workflows.build_workflow_tools(object(), [])
        }

    @pytest.mark.asyncio
    async def test_returns_partial_result_instead_of_raising(self, workflow_tools):
        from api.graph.workflows import WORKFLOW_MAX_ITERATIONS

        result = await workflow_tools["run_protein_ligand_analysis"].coroutine(
            query="analyze the binding site"
        )

        assert str(WORKFLOW_MAX_ITERATIONS) in result
        assert "without reaching a conclusion" in result

    @pytest.mark.asyncio
    async def test_every_workflow_degrades_the_same_way(self, workflow_tools):
        for name, tool in workflow_tools.items():
            result = await tool.coroutine(query="go")
            assert "without reaching a conclusion" in result, name
