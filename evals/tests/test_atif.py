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

"""Tests for the ATIF projection of a driver trajectory.

These assertions are a contract with Harbor's viewer, not with a spec on paper.
The trial page reads ``trajectory.agent.model_name`` through an unguarded
intermediate, so a document without an ``agent`` object does not degrade -- it
throws, and the page renders "Oops! An unexpected error occurred." That is the
failure these tests exist to keep out.

Harbor-free, like the rest of evals/tests: this runs in the plain devcontainer.
"""

from __future__ import annotations

from aevom_eval.atif import EXTRA_KEY, SCHEMA_VERSION, to_atif, unwrap


def _events() -> list[dict]:
    """One loop: the model thinks, calls a tool, sees it fail, then answers."""
    return [
        {"span_id": "abc123"},
        {"reasoning": "I should look up the structure."},
        {
            "tool_call_start": {
                "id": "call_1",
                "name": "structure_query",
                "args": {"pdb_id": "1ABC"},
                "summary": "query 1ABC",
            },
            "iteration": 1,
        },
        {"tool_call_end": {"id": "call_1", "name": "structure_query", "success": False,
                           "output_preview": "not found"}, "iteration": 1},
        {"content": "I could not find that structure."},
        {"usage": {"input_tokens": 1200, "output_tokens": 340, "total_tokens": 1540}},
    ]


def _trajectory(**overrides) -> dict:
    trajectory = {
        "instruction": "Find 1ABC.",
        "session_id": "sess-1",
        # As the driver records it: the stack's /api/model block, verbatim.
        "info": {
            "model_id": "nemotron-3-ultra-nvinf",
            "model_name": "nvidia/nvidia/nemotron-3-ultra",
            "pretty_name": "Nemotron 3 Ultra (inference-api)",
        },
        "final_content": "I could not find that structure.",
        "reasoning": ["I should look up the structure."],
        "usage": {"input_tokens": 1200, "output_tokens": 340, "total_tokens": 1540},
        "raw_events": _events(),
    }
    trajectory.update(overrides)
    return trajectory


def _convert(trajectory: dict) -> dict:
    return to_atif(trajectory, agent_name="aevom", agent_version="sha-abc123")


class TestTheFieldsTheViewerReads:
    def test_the_agent_object_is_present(self):
        """The one field whose absence crashes the trial page."""
        document = _convert(_trajectory())
        assert document["agent"]["model_name"] == "nemotron-3-ultra-nvinf"

    def test_the_agent_is_named_and_versioned(self):
        document = _convert(_trajectory())
        assert document["agent"]["name"] == "aevom"
        assert document["agent"]["version"] == "sha-abc123"

    def test_it_declares_the_schema_it_conforms_to(self):
        assert _convert(_trajectory())["schema_version"] == SCHEMA_VERSION

    def test_steps_are_a_list_even_when_nothing_was_recorded(self):
        """The page renders `steps.length`, so the key cannot be absent."""
        document = _convert({"instruction": "", "raw_events": []})
        assert document["steps"] == []

    def test_the_model_is_the_one_that_answered_not_the_one_requested(self):
        """The nightly requests none and the stack chooses, so `info` decides."""
        document = _convert(_trajectory(model_id="requested-model"))
        assert document["agent"]["model_name"] == "nemotron-3-ultra-nvinf"

    def test_the_requested_model_stands_in_when_the_stack_reported_none(self):
        """`info` is empty when the turn died before the stack answered."""
        document = _convert(_trajectory(info={}, model_id="requested-model"))
        assert document["agent"]["model_name"] == "requested-model"

    def test_the_model_falls_back_to_the_one_harbor_asked_for(self):
        document = to_atif(
            _trajectory(info={}),
            agent_name="aevom",
            agent_version="local",
            model_name="from-harbor",
        )
        assert document["agent"]["model_name"] == "from-harbor"


class TestSteps:
    def test_the_instruction_opens_the_conversation(self):
        steps = _convert(_trajectory())["steps"]
        assert steps[0]["source"] == "user"
        assert steps[0]["message"] == "Find 1ABC."

    def test_the_answer_is_carried_as_a_message(self):
        """The content the user came to read."""
        steps = _convert(_trajectory())["steps"]
        assert steps[-1]["message"] == "I could not find that structure."

    def test_speaking_after_a_tool_result_starts_a_new_step(self):
        """Otherwise a whole multi-tool run collapses into a single turn."""
        steps = _convert(_trajectory())["steps"]
        agent_steps = [step for step in steps if step["source"] == "agent"]
        assert len(agent_steps) == 2
        assert agent_steps[0]["reasoning_content"] == "I should look up the structure."
        assert agent_steps[1]["message"] == "I could not find that structure."

    def test_step_ids_are_ordinal_from_one(self):
        steps = _convert(_trajectory())["steps"]
        assert [step["step_id"] for step in steps] == list(range(1, len(steps) + 1))

    def test_an_answer_survives_when_the_events_did_not(self):
        """Fixtures and salvaged runs carry the digest without the raw stream."""
        steps = _convert(_trajectory(raw_events=[]))["steps"]
        assert steps[-1]["source"] == "agent"
        assert steps[-1]["message"] == "I could not find that structure."


class TestToolCalls:
    def test_the_call_and_its_result_are_paired(self):
        agent_step = _convert(_trajectory())["steps"][1]
        call = agent_step["tool_calls"][0]
        assert call["function_name"] == "structure_query"
        assert call["arguments"] == {"pdb_id": "1ABC"}
        result = agent_step["observation"]["results"][0]
        assert result["source_call_id"] == call["tool_call_id"]
        assert result["content"] == "not found"

    def test_a_failure_is_recorded_as_one(self):
        """Whether a tool failed is the question most trials are opened to answer."""
        agent_step = _convert(_trajectory())["steps"][1]
        assert agent_step["observation"]["results"][0]["extra"]["success"] is False

    def test_arguments_are_always_an_object(self):
        """The schema types them as one, so a bare value is kept, not dropped."""
        trajectory = _trajectory(
            raw_events=[{"tool_call_start": {"id": "c1", "name": "t", "args": "1ABC"}}]
        )
        call = _convert(trajectory)["steps"][1]["tool_calls"][0]
        assert call["arguments"] == {"value": "1ABC"}


class TestTheRecordSurvives:
    def test_the_driver_record_is_kept_verbatim(self):
        trajectory = _trajectory()
        assert _convert(trajectory)["extra"][EXTRA_KEY] == trajectory

    def test_unwrap_returns_what_the_verifier_used_to_read(self):
        trajectory = _trajectory()
        assert unwrap(_convert(trajectory)) == trajectory

    def test_unwrap_passes_a_bare_record_through(self):
        """Every run before this change, and every recorded fixture, is bare."""
        trajectory = _trajectory()
        assert unwrap(trajectory) is trajectory

    def test_unwrap_tolerates_nothing_at_all(self):
        assert unwrap(None) is None


class TestFinalMetrics:
    def test_the_turn_totals_cross_over(self):
        metrics = _convert(_trajectory())["final_metrics"]
        assert metrics["total_prompt_tokens"] == 1200
        assert metrics["total_completion_tokens"] == 340

    def test_a_turn_that_reported_no_usage_claims_none(self):
        metrics = _convert(_trajectory(usage=None))["final_metrics"]
        assert metrics["total_prompt_tokens"] is None
        assert metrics["total_completion_tokens"] is None
