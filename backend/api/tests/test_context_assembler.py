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

"""The assembler must shrink a prompt without ever orphaning a tool result."""

from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from api.context.assembler import (
    LEVEL_COLLAPSED,
    LEVEL_PREVIEW,
    LEVEL_SUMMARY,
    LEVEL_VERBATIM,
    ContextAssembler,
    split_into_turns,
)


@dataclass
class FakeConfig:
    context_window: int = 8000
    max_tokens: int = 1000


def _call(call_id: str, name: str = "vina_dock") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": {"ligand": "sdf_aaaaaaaa"}}],
    )


def _result(call_id: str, content: str, name: str = "vina_dock") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def _turn(index: int, payload_chars: int) -> list:
    """A user question, one tool round trip, and an answer."""
    call_id = f"call_{index}"
    return [
        HumanMessage(content=f"Question {index}"),
        _call(call_id),
        _result(
            call_id,
            f"Docking complete. Best affinity -9.{index} kcal/mol\n"
            + ("x" * payload_chars)
            + f"\nPose stored as sdf_{index}{index}{index}{index}{index}{index}{index}{index}",
        ),
        AIMessage(content=f"The best pose scored -9.{index} kcal/mol."),
    ]


def _state(messages: list) -> dict:
    return {
        "messages": messages,
        "viewer_state": {},
        "storage_state": {},
        "selections": [],
        "findings": [],
    }


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_a_call_and_its_results_group_together():
    turns = split_into_turns(
        [
            HumanMessage(content="dock it"),
            _call("a"),
            _result("a", "done"),
            AIMessage(content="finished"),
        ]
    )

    assert len(turns) == 1
    assert len(turns[0].exchanges) == 1
    assert turns[0].exchanges[0].results[0].tool_call_id == "a"
    assert turns[0].trailing[0].content == "finished"


def test_a_tool_result_with_no_call_is_discarded():
    turns = split_into_turns(
        [HumanMessage(content="hi"), _result("orphan", "nobody asked for this")]
    )

    assert turns[0].exchanges == []
    assert turns[0].trailing == []


def test_parallel_calls_keep_all_their_results():
    assistant = AIMessage(
        content="",
        tool_calls=[
            {"id": "a", "name": "vina_dock", "args": {}},
            {"id": "b", "name": "vina_dock", "args": {}},
        ],
    )
    turns = split_into_turns(
        [HumanMessage(content="dock both"), assistant, _result("a", "x"), _result("b", "y")]
    )

    assert [r.tool_call_id for r in turns[0].exchanges[0].results] == ["a", "b"]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_a_short_conversation_is_left_alone():
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig())
    messages = assembler.assemble(_state(_turn(1, 100)))

    assert assembler.last_report.level == LEVEL_VERBATIM
    assert assembler.last_report.degraded_messages == 0
    assert any(isinstance(m, ToolMessage) for m in messages)


def test_no_model_config_means_no_degradation():
    """Without a declared window there is no budget, so nothing is cut."""
    assembler = ContextAssembler("You are Aevom.", config=None)
    history = [m for i in range(1, 8) for m in _turn(i, 20_000)]

    assembler.assemble(_state(history))

    assert assembler.last_report.level == LEVEL_VERBATIM


def test_mild_pressure_previews_tool_results_and_keeps_their_refs():
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=5000))
    history = [m for i in range(1, 5) for m in _turn(i, 2_000)]

    messages = assembler.assemble(_state(history))

    assert assembler.last_report.level >= LEVEL_PREVIEW
    previewed = [
        m for m in messages if isinstance(m, ToolMessage) and "truncated" in str(m.content)
    ]
    assert previewed, "expected at least one previewed tool result"
    assert "sdf_11111111" in str(previewed[0].content)


def test_heavy_pressure_collapses_but_never_orphans():
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=3000))
    history = [m for i in range(1, 10) for m in _turn(i, 4_000)]

    messages = assembler.assemble(_state(history))

    announced = {
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
    }
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert announced == answered
    assert assembler.last_report.level >= LEVEL_COLLAPSED


def test_extreme_pressure_summarizes_old_turns_and_drops_their_calls():
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=2200))
    history = [m for i in range(1, 12) for m in _turn(i, 6_000)]

    messages = assembler.assemble(_state(history))

    assert assembler.last_report.level == LEVEL_SUMMARY
    condensed = [m for m in messages if "[EARLIER TURN, CONDENSED]" in str(m.content)]
    assert condensed
    announced = {
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
    }
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert announced == answered


def test_the_turn_in_progress_survives_degradation_of_everything_older():
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=3000))
    history = [m for i in range(1, 9) for m in _turn(i, 4_000)]
    history.append(HumanMessage(content="Now compare them"))

    messages = assembler.assemble(_state(history))

    assert messages[-1].content == "Now compare them"


def test_a_single_oversized_message_is_reported_rather_than_hidden():
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=1200))
    history = [HumanMessage(content="x" * 200_000)]

    assembler.assemble(_state(history))

    assert assembler.last_report.still_over_budget is True


def test_exhausting_the_ladder_is_reported_even_when_nothing_degraded():
    """Running out of ladder is the signal that compaction is now needed."""
    from api.context.assembler import build_context_assembler

    recorded = {}

    class Span:
        def set_attribute(self, key, value):
            recorded[key] = value

    assemble = build_context_assembler("p", config=FakeConfig(context_window=1200))
    import api.telemetry as telemetry

    original = telemetry.get_current_chat_span
    telemetry.get_current_chat_span = lambda: Span()
    try:
        assemble(_state([HumanMessage(content="x" * 200_000)]))
    finally:
        telemetry.get_current_chat_span = original

    assert recorded["context.still_over_budget"] is True
    assert recorded["context.degradation_level"] == LEVEL_VERBATIM


def test_the_system_prompt_is_always_first_and_never_degraded():
    assembler = ContextAssembler("FLOOR MARKER", config=FakeConfig(context_window=2000))
    history = [m for i in range(1, 10) for m in _turn(i, 5_000)]

    messages = assembler.assemble(_state(history))

    assert isinstance(messages[0], SystemMessage)
    assert "FLOOR MARKER" in messages[0].content
    assert sum(isinstance(m, SystemMessage) for m in messages) == 1


def test_the_session_log_stays_in_the_floor_while_the_trajectory_degrades():
    """The numbers survive precisely because they live above the ladder."""
    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=2500))
    state = _state([m for i in range(1, 10) for m in _turn(i, 5_000)])
    state["findings"] = [
        {
            "turn_seq": 1,
            "iteration": 1,
            "tool_name": "vina_dock",
            "summary": "vina_dock(ligand=sdf_aaaaaaaa) -> -9.5 kcal/mol",
        }
    ]

    messages = assembler.assemble(state)

    assert "-9.5 kcal/mol" in messages[0].content


def test_bound_tool_schemas_are_charged_against_the_budget():
    schemas = [{"type": "function", "function": {"name": "t", "description": "d" * 4000}}]
    with_tools = ContextAssembler("p", config=FakeConfig(), tool_schemas=schemas)
    without = ContextAssembler("p", config=FakeConfig())

    assert with_tools._available_tokens() < without._available_tokens()


def test_a_long_session_assembles_without_quadratic_rework():
    """Assembly sits on the request path, so it cannot re-render per step."""
    import time

    assembler = ContextAssembler("You are Aevom.", config=FakeConfig(context_window=8000))
    history = [m for i in range(1, 61) for m in _turn(i, 40_000)]
    state = _state(history)

    started = time.monotonic()
    assembler.assemble(state)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"assembly of 60 turns took {elapsed:.2f}s"


@pytest.mark.parametrize("level", [LEVEL_PREVIEW, LEVEL_COLLAPSED])
def test_every_degradation_level_preserves_call_ids(level):
    turns = split_into_turns([HumanMessage(content="q"), _call("z"), _result("z", "y" * 5000)])
    rendered = turns[0].exchanges[0].render(level)

    assert [m.tool_call_id for m in rendered if isinstance(m, ToolMessage)] == ["z"]
