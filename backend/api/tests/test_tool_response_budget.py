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

"""Tests for the tool response budget.

Tool results were the last unbounded text path into the model's context: the
storage pattern keeps SDF/PDB payloads out, but a tool that *returns text* was
exempt. The budget closes that hole in ``ToolRegistry.execute``, which is the
one path shared by the production graph and the NeMo-Gym resource server -- so
these tests are as much about parity as about truncation.
"""

import pytest

from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.service import DataStorageService
from api.tools.base import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_RESPONSE_TOKENS,
    BaseTool,
    ContentType,
    ToolResult,
    ToolType,
    build_capped_response,
)
from api.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    """Returns whatever text it is handed, at whatever length."""

    def __init__(self, payload: str, max_tokens: int | None = None):
        self._payload = payload
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "echo"

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def max_response_tokens(self) -> int:
        if self._max_tokens is None:
            return super().max_response_tokens
        return self._max_tokens

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": "echo", "description": "", "parameters": {}},
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.text(self._payload)


@pytest.fixture(autouse=True)
def no_tool_spans():
    """Take the span-free path, which is also how training executes tools.

    Without this every ``registry.execute`` pays a multi-second OTel export
    timeout against a collector that is not running in the test environment.
    ``TestSpanPath`` covers the other branch explicitly.
    """
    from api.telemetry import set_skip_tool_spans

    set_skip_tool_spans(True)
    yield
    set_skip_tool_spans(False)


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


def registry_with(tool: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


class TestBuildCappedResponse:
    def test_keeps_head_and_tail(self):
        text = "HEAD" + ("x" * 1000) + "TAIL"
        capped = build_capped_response(text, 100, "text_abcd1234")
        assert capped.startswith("HEAD")
        assert capped.endswith("TAIL")

    def test_names_the_overflow_reference_and_how_to_read_it(self):
        capped = build_capped_response("y" * 500, 100, "text_abcd1234")
        assert "text_abcd1234" in capped
        assert "aevom.load_text('text_abcd1234')" in capped

    def test_says_so_when_the_middle_is_unrecoverable(self):
        # Without storage there is no ref to point at, and claiming otherwise
        # would send the model chasing an artifact that does not exist.
        capped = build_capped_response("y" * 500, 100, None)
        assert "not recoverable" in capped
        assert "text_" not in capped

    def test_reports_how_much_was_elided(self):
        capped = build_capped_response("y" * 500, 100, None)
        assert "400 characters elided" in capped


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_short_output_is_untouched(self, storage):
        registry = registry_with(EchoTool("all done"))
        result = await registry.execute("echo", {}, storage=storage)
        assert result.to_tool_response() == "all done"
        assert result.response_override is None

    @pytest.mark.asyncio
    async def test_long_output_is_capped(self, storage):
        payload = "z" * 100_000
        registry = registry_with(EchoTool(payload, max_tokens=100))
        result = await registry.execute("echo", {}, storage=storage)

        response = result.to_tool_response()
        assert len(response) < 1000
        assert len(response) < len(payload)

    @pytest.mark.asyncio
    async def test_the_full_output_is_recoverable_from_the_overflow_artifact(
        self, storage
    ):
        payload = "line-one\n" + ("z" * 100_000) + "\nline-last"
        registry = registry_with(EchoTool(payload, max_tokens=100))
        result = await registry.execute("echo", {}, storage=storage)

        refs = [
            a["reference"]
            for a in await storage.list_artifacts()
            if a["data_type"] == "text"
        ]
        assert len(refs) == 1
        assert (await storage.retrieve(refs[0])).decode() == payload
        assert refs[0] in result.to_tool_response()

    @pytest.mark.asyncio
    async def test_data_is_left_intact_for_non_model_consumers(self, storage):
        # Visualization events and the training wrapper's pose-ref tracking
        # read result.data directly; capping must not reach them.
        payload = "z" * 100_000
        registry = registry_with(EchoTool(payload, max_tokens=100))
        result = await registry.execute("echo", {}, storage=storage)
        assert result.data == payload

    @pytest.mark.asyncio
    async def test_capping_still_happens_without_storage(self):
        payload = "z" * 100_000
        registry = registry_with(EchoTool(payload, max_tokens=100))
        result = await registry.execute("echo", {}, storage=None)

        response = result.to_tool_response()
        assert len(response) < 1000
        assert "not recoverable" in response

    @pytest.mark.asyncio
    async def test_a_failing_store_does_not_fail_the_tool_call(self, storage):
        async def boom(*args, **kwargs):
            raise RuntimeError("database is down")

        storage.store = boom
        payload = "z" * 100_000
        registry = registry_with(EchoTool(payload, max_tokens=100))
        result = await registry.execute("echo", {}, storage=storage)

        assert result.content_type is ContentType.TEXT
        assert "not recoverable" in result.to_tool_response()

    @pytest.mark.asyncio
    async def test_per_tool_budget_is_honoured(self, storage):
        payload = "z" * 100_000
        tight = await registry_with(EchoTool(payload, max_tokens=100)).execute(
            "echo", {}, storage=storage
        )
        loose = await registry_with(EchoTool(payload, max_tokens=1000)).execute(
            "echo", {}, storage=storage
        )
        assert len(tight.to_tool_response()) < len(loose.to_tool_response())

    @pytest.mark.asyncio
    async def test_error_results_are_capped_too(self, storage):
        class Exploding(EchoTool):
            async def execute(self, arguments: dict) -> ToolResult:
                return ToolResult.error(self._payload)

        registry = registry_with(Exploding("z" * 100_000, max_tokens=100))
        result = await registry.execute("echo", {}, storage=storage)
        assert result.content_type is ContentType.ERROR
        assert len(result.to_tool_response()) < 1000


class TestSpanPath:
    @pytest.mark.asyncio
    async def test_capping_survives_the_instrumented_path(self, storage):
        # The span branch records the uncapped output as tool.raw.output and
        # the capped one as the span's I/O; both must coexist without error.
        from api.telemetry import set_skip_tool_spans

        set_skip_tool_spans(False)
        registry = registry_with(EchoTool("z" * 100_000, max_tokens=100))
        result = await registry.execute("echo", {}, storage=storage)
        assert len(result.to_tool_response()) < 1000


class TestToolBudgets:
    def test_the_default_applies_to_tools_that_do_not_opt_in(self):
        assert EchoTool("").max_response_tokens == DEFAULT_MAX_RESPONSE_TOKENS

    def test_unbounded_output_tools_are_tighter_than_the_default(self):
        from api.skills.tools import LoadSkillResourceTool, LoadSkillTool
        from api.tools.python_exec import PythonExecuteTool

        for tool in (PythonExecuteTool(), LoadSkillTool(), LoadSkillResourceTool()):
            assert tool.max_response_tokens < DEFAULT_MAX_RESPONSE_TOKENS

    def test_the_char_estimate_is_shared_with_the_context_budget(self):
        # The cap and the prompt budget must agree on what a token costs,
        # otherwise a "2000 token" tool result is not 2000 tokens of budget.
        from api.context.budget import CHARS_PER_TOKEN as budget_chars_per_token

        assert CHARS_PER_TOKEN == budget_chars_per_token
