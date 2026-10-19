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

"""The canonical invocation event must exist on every path, and reach the wire.

`ToolInvocationEvent` (ADR 0001) shipped with no readers at all: the registry
built it, and `graph/tools.py` then discarded the `ToolResult` in order to
return the string LangGraph's ToolNode requires. Nothing downstream -- SSE,
eval, training -- could see it, and no test noticed, because the tests only
asserted the event was *constructed*.

So these tests assert the two things that were actually missing.

First, that the event is present on *every* terminal branch of the registry,
including the failure ones. A contract honoured on the happy path only is not a
contract: a consumer cannot distinguish "no event for this call" from "this
branch forgot", so an absent event would have to be read as "unknown"
everywhere and the field would carry no information at all.

Second, that it survives the trip to the `tool_call_end` payload -- the actual
boundary where it was being lost.
"""

import sys
import types

import pytest

from api.storage import InvalidReferenceError
from api.tools.base import BaseTool, ContentType, DataIOCategory, ToolResult, ToolType
from api.tools.policy import PolicyEngine, PolicyResult
from api.tools.registry import ToolRegistry
from api.tools import budget


class StubTool(BaseTool):
    """Minimal tool whose outcome each test dictates."""

    def __init__(
        self,
        name: str = "stub_tool",
        raises: Exception | None = None,
        category: DataIOCategory = DataIOCategory.TEXT_ONLY,
    ):
        self._name = name
        self._raises = raises
        self._category = category

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return self._category

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self._name, "description": "", "parameters": {}},
        }

    async def execute(self, arguments: dict) -> ToolResult:
        if self._raises is not None:
            raise self._raises
        return ToolResult.text("done")


class DenyEverything:
    """A policy that refuses, to exercise the deny branch."""

    name = "deny_everything"
    active = True

    async def evaluate(self, tool_name, arguments, session_id, request_id):
        return PolicyResult.deny("denied for the test")


class FakeRedis:
    """The subset of the async client the budget module uses."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def __await__(self):
        async def _self():
            return self

        return _self().__await__()

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture(autouse=True)
def no_tool_spans():
    from api.telemetry import set_skip_tool_spans

    set_skip_tool_spans(True)
    yield
    set_skip_tool_spans(False)


@pytest.fixture(autouse=True)
def clean_budgets():
    original = budget.active_budgets()
    yield
    budget.set_budgets(original)


@pytest.fixture(autouse=True)
def redis(monkeypatch):
    client = FakeRedis()
    module = types.ModuleType("api.redis")
    module.get_redis = lambda: client
    monkeypatch.setitem(sys.modules, "api.redis", module)
    return client


def registry_with(tool: BaseTool, policy_engine=None) -> ToolRegistry:
    registry = ToolRegistry(policy_engine=policy_engine)
    registry.register(tool)
    return registry


@pytest.mark.asyncio
class TestEveryTerminalBranchCarriesAnEvent:
    """One test per branch. Adding a seventh branch without an event fails here."""

    async def test_success(self):
        result = await registry_with(StubTool()).execute("stub_tool", {}, session_id="s1")

        assert result.invocation_event is not None
        assert result.invocation_event.tool_name == "stub_tool"
        assert result.invocation_event.success is True
        assert result.invocation_event.policy_decision == "allow"

    async def test_text_only_tools_are_not_excluded(self):
        # The storage-injection branch is gated on data_io_category; the event
        # must not be, or every text tool would be invisible to eval.
        tool = StubTool(category=DataIOCategory.TEXT_ONLY)
        result = await registry_with(tool).execute("stub_tool", {}, session_id="s1")

        assert result.invocation_event is not None

    async def test_unknown_tool(self):
        result = await registry_with(StubTool()).execute("no_such_tool", {}, session_id="s1")

        assert result.content_type is ContentType.ERROR
        assert result.invocation_event is not None
        assert result.invocation_event.success is False
        # Policy never ran: the name failed to resolve first.
        assert result.invocation_event.policy_decision is None

    async def test_policy_denied(self):
        registry = registry_with(
            StubTool(), policy_engine=PolicyEngine([DenyEverything()])
        )

        result = await registry.execute("stub_tool", {}, session_id="s1")

        assert result.invocation_event is not None
        assert result.invocation_event.policy_decision == "deny"
        assert result.invocation_event.policy_reason == "denied for the test"

    async def test_budget_exhausted(self):
        budget.set_budgets({"stub_tool": 0})

        result = await registry_with(StubTool()).execute("stub_tool", {}, session_id="s1")

        event = result.invocation_event
        assert event is not None
        assert event.success is False
        # A spend limit is not a safety refusal; the trajectory must not
        # conflate them.
        assert event.policy_decision == "allow"
        assert event.budget_state == {"limit": 0, "used": 0, "remaining": 0}

    async def test_invalid_reference(self):
        tool = StubTool(raises=InvalidReferenceError("bad ref"))

        result = await registry_with(tool).execute("stub_tool", {}, session_id="s1")

        assert result.invocation_event is not None
        assert result.invocation_event.success is False

    async def test_unexpected_exception(self):
        tool = StubTool(raises=RuntimeError("boom"))

        result = await registry_with(tool).execute("stub_tool", {}, session_id="s1")

        assert result.invocation_event is not None
        assert result.invocation_event.success is False
        assert "boom" in result.invocation_event.error


@pytest.mark.asyncio
class TestTheEventSurvivesToTheWire:
    """The specific hop that was missing: registry -> wrapper -> tool_call_end."""

    async def test_fields_reach_a_tool_call_end_payload(self):
        from api.graph.tools import _record_invocation_event, set_tool_context, reset_tool_context
        from api.graph.streaming import _invocation_fields
        from api.telemetry_contract import ToolInvocationEvent

        tokens, *_ = set_tool_context(None, "s1", "r1")
        try:
            _record_invocation_event(
                "vina_dock",
                ToolInvocationEvent(
                    tool_name="vina_dock",
                    arguments={},
                    job_id="job-123",
                    policy_decision="allow",
                    artifact_refs=["pdb_abc123"],
                ),
            )

            fields = _invocation_fields("vina_dock")
        finally:
            reset_tool_context(tokens)

        assert fields["job_id"] == "job-123"
        assert fields["policy_decision"] == "allow"
        assert fields["artifact_refs"] == ["pdb_abc123"]
        # result_ref was on the payload all along but hardcoded None.
        assert fields["result_ref"] == "pdb_abc123"

    async def test_absence_is_reported_as_absence_not_as_a_default(self):
        # A tool that ran outside a bound context (workflow subgraph, test)
        # must yield no keys at all, so consumers read "unknown" rather than
        # "no job" / "policy allowed".
        from api.graph.streaming import _invocation_fields

        assert _invocation_fields("never_ran") == {}

    async def test_repeated_calls_to_one_tool_are_consumed_in_order(self):
        from api.graph.tools import _record_invocation_event, set_tool_context, reset_tool_context
        from api.graph.streaming import _invocation_fields
        from api.telemetry_contract import ToolInvocationEvent

        tokens, *_ = set_tool_context(None, "s1", "r1")
        try:
            for job in ("first", "second"):
                _record_invocation_event(
                    "vina_dock",
                    ToolInvocationEvent(tool_name="vina_dock", arguments={}, job_id=job),
                )

            assert _invocation_fields("vina_dock")["job_id"] == "first"
            assert _invocation_fields("vina_dock")["job_id"] == "second"
            assert _invocation_fields("vina_dock") == {}
        finally:
            reset_tool_context(tokens)
