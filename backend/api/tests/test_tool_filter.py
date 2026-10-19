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

"""Tests for narrowing the exposed tool surface.

A deployment that has to match another agent's tool set, or that must not be
able to look up a target it was given anonymously, needs the model to see a
smaller registry. The requirement is not "the model is told not to" -- it is
that the tool is absent from the schemas, absent from the graph's bindings, and
unresolvable at execution time. So these tests care that one filter decides all
three, and that a deployment which configured nothing sees no change at all.
"""

import pytest

from api.tools.base import BaseTool, ToolResult, ToolType
from api.tools.registry import ToolFilter, ToolRegistry


class StubTool(BaseTool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self._name, "description": "", "parameters": {}},
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.text("done")


def registry_with(names, allowlist=None, denylist=None) -> ToolRegistry:
    registry = ToolRegistry(
        ToolFilter(
            allowlist=frozenset(allowlist or ()),
            denylist=frozenset(denylist or ()),
        )
    )
    for name in names:
        registry.register(StubTool(name))
    return registry


def test_an_unconfigured_deployment_keeps_every_tool():
    registry = registry_with(["a", "b", "c"])

    assert registry.tool_names == ["a", "b", "c"]
    assert registry.withheld == []


def test_an_allowlist_withholds_everything_it_does_not_name():
    registry = registry_with(["a", "b", "c"], allowlist=["a", "c"])

    assert registry.tool_names == ["a", "c"]
    assert registry.withheld == ["b"]


def test_a_denylist_removes_only_what_it_names():
    registry = registry_with(["a", "b", "c"], denylist=["b"])

    assert registry.tool_names == ["a", "c"]


def test_the_denylist_wins_over_the_allowlist():
    """So widening the allowlist later cannot resurrect a tool that was denied."""
    registry = registry_with(["a", "b"], allowlist=["a", "b"], denylist=["b"])

    assert registry.tool_names == ["a"]


def test_a_withheld_tool_is_unresolvable_not_merely_undocumented():
    registry = registry_with(["a", "b"], denylist=["b"])

    assert registry.get("b") is None
    assert "b" not in registry
    assert len(registry) == 1


@pytest.mark.asyncio
async def test_withheld_tools_are_absent_from_the_schemas():
    registry = registry_with(["a", "b"], denylist=["b"])

    schemas = await registry.get_all_schemas()

    assert [schema["function"]["name"] for schema in schemas] == ["a"]


@pytest.mark.asyncio
async def test_session_scoped_tools_cannot_slip_past_an_allowlist(monkeypatch):
    """MCP and skill-contract tools are resolved per session, not registered.

    They would otherwise be the one route around the filter, which for a
    benchmark run is the difference between a valid number and an invalid one.
    """

    class Store:
        async def get_tool_schemas(self, session_id):
            return [
                {"type": "function", "function": {"name": "skill__lookup", "parameters": {}}},
                {"type": "function", "function": {"name": "a", "parameters": {}}},
            ]

    monkeypatch.setattr("api.skills.get_skill_session_store", lambda: Store())
    registry = registry_with(["a"], allowlist=["a"])

    schemas = await registry.get_all_schemas()

    assert [schema["function"]["name"] for schema in schemas] == ["a", "a"]


def test_the_filter_reads_the_environment_by_default(monkeypatch):
    monkeypatch.setenv("AEVOM_TOOL_ALLOWLIST", "python_execute, boltz2 ,submit_artifact")
    monkeypatch.setenv("AEVOM_TOOL_DENYLIST", "search_pubchem")

    tool_filter = ToolFilter()

    assert tool_filter.active
    assert tool_filter.permits("boltz2")
    assert not tool_filter.permits("search_pubchem")
    assert not tool_filter.permits("vina_dock")


def test_an_empty_environment_means_production(monkeypatch):
    monkeypatch.delenv("AEVOM_TOOL_ALLOWLIST", raising=False)
    monkeypatch.setenv("AEVOM_TOOL_DENYLIST", "  ")

    tool_filter = ToolFilter()

    assert not tool_filter.active
    assert tool_filter.permits("anything")


class TestWorkflowMetaTools:
    """The workflow subgraphs are the one capability no other agent has."""

    def test_they_are_on_unless_switched_off(self, monkeypatch):
        from api.graph.agent import workflows_enabled

        monkeypatch.delenv("AEVOM_INCLUDE_WORKFLOWS", raising=False)
        assert workflows_enabled() is True

        for value in ("false", "FALSE", "0", "no"):
            monkeypatch.setenv("AEVOM_INCLUDE_WORKFLOWS", value)
            assert workflows_enabled() is False, value

        monkeypatch.setenv("AEVOM_INCLUDE_WORKFLOWS", "true")
        assert workflows_enabled() is True

    def test_the_graph_binds_none_of_them_when_switched_off(self, monkeypatch):
        pytest.importorskip("langgraph")
        from api.graph.agent import build_agent_graph

        bound: list[list] = []

        class RecordingLLM:
            def bind_tools(self, tools):
                bound.append(list(tools))
                return self

        monkeypatch.setenv("AEVOM_INCLUDE_WORKFLOWS", "false")
        build_agent_graph(RecordingLLM(), [], "system prompt")

        assert bound == [[]]
