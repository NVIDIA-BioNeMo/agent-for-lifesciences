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

"""A MolQL expression that fails to run is reported as such, not as "0 atoms".

Regression: the agent sent ``(in atom.resno [29, 30, ...])``. Mol* rejects that
as a parse error, but the model was told "Selection query matched 0 atoms" with
chain suggestions, which reads as a fact about the structure. It guessed at the
cause for several turns. The error text now reaches the model.
"""

import sys
import types

import pytest
from langchain_core.messages import ToolMessage

from api.selections.format import (
    MOLQL_ERROR_PREFIX,
    format_selection_empty_with_hints,
    format_selection_query_error,
    is_molql_error_summary,
    molql_error_summary,
)
from api.selections.registry import SelectionInfo

PARSE_ERROR = "expected EOF at line 1 column 1, got '(sel.atom.re...'"


class TestSummaryMarker:
    def test_round_trips(self):
        assert is_molql_error_summary(molql_error_summary(PARSE_ERROR))

    def test_prefix_is_the_one_the_frontend_posts(self):
        # frontend/src/lib/molstar/molqlErrors.ts and the eval bridge hard-code
        # this string; changing it here without them silently disables the path.
        assert MOLQL_ERROR_PREFIX == "MolQL error:"

    @pytest.mark.parametrize(
        "summary",
        ["", None, "Selected 0 atoms", "No atoms matched the query", "Selected 345 atoms in 19 residues"],
    )
    def test_ordinary_summaries_are_not_errors(self, summary):
        assert not is_molql_error_summary(summary)

    def test_headless_service_prefix_is_not_doubled(self):
        summary = molql_error_summary("MolQL query failed: Symbol 'in' is not implemented.")
        assert summary == "MolQL error: Symbol 'in' is not implemented."

    def test_empty_detail_still_says_something(self):
        assert molql_error_summary("").startswith(MOLQL_ERROR_PREFIX)
        assert molql_error_summary("") != MOLQL_ERROR_PREFIX


class TestQueryErrorMessage:
    def test_carries_the_engine_error(self):
        msg = format_selection_query_error(molql_error_summary(PARSE_ERROR))
        assert PARSE_ERROR in msg

    def test_says_the_query_did_not_run(self):
        msg = format_selection_query_error(molql_error_summary(PARSE_ERROR))
        assert "failed to execute" in msg
        assert "matched 0 atoms" not in msg

    def test_points_at_set_has_for_discrete_residues(self):
        msg = format_selection_query_error(molql_error_summary(PARSE_ERROR))
        assert "set.has" in msg and "atom.resno" in msg
        assert "in-range" in msg

    def test_differs_from_the_empty_result_message(self):
        empty = format_selection_empty_with_hints([{"label": "x", "chain_ids": ["A"]}])
        assert format_selection_query_error(molql_error_summary(PARSE_ERROR)) != empty


class TestToolSchemaTeachesSetHas:
    def test_schema_documents_discrete_residue_selection(self):
        from api.tools.molstar_queries import MolstarQueryTool

        desc = MolstarQueryTool().get_schema()["function"]["description"]
        assert "(set.has (set 29 30 31" in desc
        assert "NO `in` function" in desc


def _tool_message(query_id: str) -> ToolMessage:
    return ToolMessage(
        content=f"Selection pending [PENDING:{query_id}]",
        tool_call_id="call_1",
        id="msg_1",
    )


def _selection(summary: str, atom_count: int = 0) -> SelectionInfo:
    return SelectionInfo(
        reference="selection_x",
        molql="(sel.atom.res ...)",
        action="select",
        atom_count=atom_count,
        residue_count=0,
        summary=summary,
        query_id="q1",
    )


@pytest.fixture
def pending(monkeypatch):
    """Stub the registry, Redis and pending-op plumbing around the node."""

    def install(sel: SelectionInfo | None):
        import api.graph.tools as graph_tools
        import api.selections.registry as registry

        # api.redis needs a client library the native suite lacks, so it is
        # replaced wholesale, as the other graph tests do.
        redis_stub = types.ModuleType("api.redis")
        redis_stub.get_redis = lambda: object()
        monkeypatch.setitem(sys.modules, "api.redis", redis_stub)

        async def drain(timeout=10.0):
            return set()

        async def lookup(_redis, _session, _query_id):
            return sel

        monkeypatch.setattr(graph_tools, "drain_pending_ops", drain)
        monkeypatch.setattr(
            graph_tools,
            "get_and_clear_pending_structure_queries",
            lambda: [("q1", "s1", None)],
        )
        monkeypatch.setattr(registry, "get_selection_by_query_id", lookup)

    return install


def _content(result: dict) -> str:
    (msg,) = result["messages"]
    return msg.content


class TestMainGraphNode:
    @pytest.mark.asyncio
    async def test_parse_error_reaches_the_model(self, pending):
        from api.graph.agent import _await_pending_ops_node

        pending(_selection(molql_error_summary(PARSE_ERROR)))
        out = await _await_pending_ops_node({"messages": [_tool_message("q1")]})
        content = _content(out)
        assert PARSE_ERROR in content
        assert "matched 0 atoms" not in content

    @pytest.mark.asyncio
    async def test_a_real_empty_result_keeps_the_zero_atoms_message(self, pending):
        from api.graph.agent import _await_pending_ops_node

        pending(_selection("No atoms matched the query"))
        out = await _await_pending_ops_node({"messages": [_tool_message("q1")]})
        assert "matched 0 atoms" in _content(out)

    @pytest.mark.asyncio
    async def test_success_is_unchanged(self, pending):
        from api.graph.agent import _await_pending_ops_node

        pending(_selection("Selected 345 atoms in 19 residues", atom_count=345))
        out = await _await_pending_ops_node({"messages": [_tool_message("q1")]})
        assert "Selection active: 345 atoms" in _content(out)
