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

"""Tests for rebuilding LangChain messages from the durable record.

The failure this guards against is a provider 400 from an orphaned tool
message, which surfaces as a generic bad-request and is considerably harder to
diagnose than the missing history it would replace.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from api.context import rehydrate_messages


def _assistant(content="", tool_calls=None, reasoning=None, extra=None):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "reasoning": reasoning,
        "extra": extra,
    }


def _tool(content, tool_call_id, name="vina_dock"):
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
    }


def _call(call_id, name="vina_dock", args=None):
    return {"id": call_id, "name": name, "args": args or {}, "type": "tool_call"}


class TestPlainMessages:
    def test_roles_map_to_their_message_types(self):
        out = rehydrate_messages([
            {"role": "system", "content": "You are Aevom."},
            {"role": "user", "content": "dock aspirin"},
            _assistant(content="Done."),
        ])
        assert [type(m) for m in out] == [SystemMessage, HumanMessage, AIMessage]

    def test_empty_transcript_yields_nothing(self):
        assert rehydrate_messages([]) == []

    def test_unknown_roles_are_skipped(self):
        out = rehydrate_messages([{"role": "developer", "content": "x"}])
        assert out == []


class TestToolCallPairing:
    def test_a_complete_pair_is_restored(self):
        out = rehydrate_messages([
            {"role": "user", "content": "dock it"},
            _assistant(tool_calls=[_call("c1")]),
            _tool("-9.5 kcal/mol", "c1"),
            _assistant(content="It scored -9.5."),
        ])
        assert [type(m) for m in out] == [
            HumanMessage,
            AIMessage,
            ToolMessage,
            AIMessage,
        ]
        assert out[1].tool_calls[0]["id"] == "c1"
        assert out[2].tool_call_id == "c1"

    def test_parallel_calls_are_restored_together(self):
        out = rehydrate_messages([
            _assistant(tool_calls=[_call("c1"), _call("c2")]),
            _tool("-9.5", "c1"),
            _tool("-6.1", "c2"),
        ])
        assert len(out) == 3
        assert {m.tool_call_id for m in out[1:]} == {"c1", "c2"}

    def test_an_unanswered_call_drops_the_whole_block(self):
        """An interrupted turn leaves a call with no result; sending it 400s."""
        out = rehydrate_messages([
            {"role": "user", "content": "dock it"},
            _assistant(tool_calls=[_call("c1")]),
        ])
        assert [type(m) for m in out] == [HumanMessage]

    def test_a_partially_answered_block_is_dropped_whole(self):
        out = rehydrate_messages([
            {"role": "user", "content": "dock both"},
            _assistant(tool_calls=[_call("c1"), _call("c2")]),
            _tool("-9.5", "c1"),
        ])
        assert [type(m) for m in out] == [HumanMessage]

    def test_an_orphaned_result_is_dropped(self):
        out = rehydrate_messages([
            {"role": "user", "content": "dock it"},
            _tool("-9.5", "c1"),
            _assistant(content="Done."),
        ])
        assert [type(m) for m in out] == [HumanMessage, AIMessage]

    def test_a_later_turn_survives_an_earlier_broken_one(self):
        """One interrupted turn must not cost the user the rest of the session."""
        out = rehydrate_messages([
            {"role": "user", "content": "first"},
            _assistant(tool_calls=[_call("c1")]),
            {"role": "user", "content": "second"},
            _assistant(tool_calls=[_call("c2")]),
            _tool("-6.1", "c2"),
            _assistant(content="Done."),
        ])
        assert [type(m) for m in out] == [
            HumanMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
            AIMessage,
        ]


class TestReasoningRestoration:
    def test_reasoning_returns_where_the_model_emitted_it(self):
        """retains_reasoning reads additional_kwargs["reasoning"]."""
        out = rehydrate_messages([_assistant(content="ok", reasoning="I should dock.")])
        assert out[0].additional_kwargs["reasoning"] == "I should dock."

    def test_absent_reasoning_leaves_no_empty_key(self):
        out = rehydrate_messages([_assistant(content="ok")])
        assert "reasoning" not in out[0].additional_kwargs

    def test_provider_specific_fields_are_carried_back(self):
        """Gemini 3 rejects a turn whose thought signatures went missing."""
        out = rehydrate_messages([
            _assistant(content="ok", extra={"thought_signature": "sig_abc"})
        ])
        assert out[0].additional_kwargs["thought_signature"] == "sig_abc"


class TestLegacyClients:
    def test_flat_role_content_pairs_still_work(self):
        """Scripts posting a whole transcript predate the record."""
        out = rehydrate_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert [m.content for m in out] == ["hi", "hello"]
