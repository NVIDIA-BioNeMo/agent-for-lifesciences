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

"""Tests for carrying model reasoning back into the next turn.

Training already does this: the Responses API loop appends the whole output
array, reasoning items included, so the trained policy sees its own chain of
thought. Production dropped it, which meant the deployed policy saw strictly
less than the trained one. These tests pin the round-trip and the per-model
gate that decides when it happens.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from api.graph.llm import _langchain_to_openai_messages
from api.llm.config import MODELS, ModelConfig


RETAINING = ModelConfig(id="r", name="r", retains_reasoning=True)
NOT_RETAINING = ModelConfig(id="n", name="n", retains_reasoning=False)
THINK_TAGS = ModelConfig(
    id="t", name="t", retains_reasoning=True, content_has_think_tags=True
)


def thinking_turn(reasoning: str = "The pose looks strained.", content: str = "Done.") -> AIMessage:
    return AIMessage(content=content, additional_kwargs={"reasoning": reasoning})


class TestRetention:
    def test_reasoning_is_sent_back_when_the_model_retains_it(self):
        converted = _langchain_to_openai_messages([thinking_turn()], RETAINING)
        assert converted[0]["reasoning_content"] == "The pose looks strained."

    def test_reasoning_is_dropped_when_the_model_does_not_retain_it(self):
        converted = _langchain_to_openai_messages([thinking_turn()], NOT_RETAINING)
        assert "reasoning_content" not in converted[0]

    def test_reasoning_is_dropped_without_a_config(self):
        # The pre-existing behaviour; callers that do not know the model must
        # not start sending a field the provider may reject.
        converted = _langchain_to_openai_messages([thinking_turn()])
        assert "reasoning_content" not in converted[0]

    def test_content_is_untouched_by_retention(self):
        converted = _langchain_to_openai_messages([thinking_turn()], RETAINING)
        assert converted[0]["content"] == "Done."

    def test_a_turn_without_reasoning_gains_no_field(self):
        converted = _langchain_to_openai_messages([AIMessage(content="hi")], RETAINING)
        assert "reasoning_content" not in converted[0]

    def test_reasoning_survives_alongside_tool_calls(self):
        # The case that matters most: an assistant turn whose only output is a
        # tool call, where reasoning is the sole record of why.
        message = AIMessage(
            content="",
            additional_kwargs={"reasoning": "Dock it against 1ABC."},
            tool_calls=[{"name": "vina_dock", "args": {"x": 1}, "id": "c1"}],
        )
        converted = _langchain_to_openai_messages([message], RETAINING)
        assert converted[0]["reasoning_content"] == "Dock it against 1ABC."
        assert converted[0]["tool_calls"][0]["function"]["name"] == "vina_dock"


class TestThinkTagModels:
    def test_reasoning_is_inlined_with_the_delimiter_restored(self):
        # LLMClient consumed </think> when splitting reasoning out of the
        # content stream, so this is the only place that can put it back.
        converted = _langchain_to_openai_messages(
            [thinking_turn("thinking hard", "the answer")], THINK_TAGS
        )
        assert converted[0]["content"] == "thinking hard</think>the answer"
        assert "reasoning_content" not in converted[0]

    def test_the_delimiter_is_not_emitted_twice(self):
        message = AIMessage(
            content="already</think>split",
            additional_kwargs={"reasoning": "thinking hard"},
        )
        converted = _langchain_to_openai_messages([message], THINK_TAGS)
        assert converted[0]["content"].count("</think>") == 1

    def test_existing_reasoning_in_content_is_left_alone(self):
        message = AIMessage(
            content="already</think>split",
            additional_kwargs={"reasoning": "thinking hard"},
        )
        converted = _langchain_to_openai_messages([message], THINK_TAGS)
        assert converted[0]["content"] == "already</think>split"

    def test_empty_content_still_gets_the_delimiter(self):
        message = AIMessage(content="", additional_kwargs={"reasoning": "why"})
        converted = _langchain_to_openai_messages([message], THINK_TAGS)
        assert converted[0]["content"] == "why</think>"


class TestConversionIsOtherwiseUnchanged:
    def test_other_roles_round_trip(self):
        converted = _langchain_to_openai_messages(
            [
                SystemMessage(content="sys"),
                HumanMessage(content="hi"),
                ToolMessage(content="-9.5", tool_call_id="c1", name="vina_dock"),
            ],
            RETAINING,
        )
        assert [m["role"] for m in converted] == ["system", "user", "tool"]
        assert converted[2]["tool_call_id"] == "c1"

    def test_thought_signatures_are_still_preserved(self):
        # Gemini 3's strict validation depends on these surviving; retention
        # must not have displaced them.
        message = AIMessage(
            content="x",
            additional_kwargs={"provider_specific_fields": {"thought_signatures": ["s"]}},
        )
        converted = _langchain_to_openai_messages([message], NOT_RETAINING)
        assert converted[0]["provider_specific_fields"]["thought_signatures"] == ["s"]


class TestModelGating:
    def test_the_nemotron_reasoning_models_retain(self):
        for model_id in (
            "nemotron-3-super-120b",
            "nemotron-3-nano-omni",
            "nemotron-3-5-super-vl",
            "nemotron-3-5-lightning",
            "nemotron-3-super-dynamo",
            "nemotron-3-super-rc-nim",
            "nemotron-3-ultra",
            "nemotron-3-ultra-nvinf",
        ):
            assert MODELS[model_id].retains_reasoning, model_id

    def test_the_other_build_nvidia_reasoning_models_retain(self):
        # Each was sent its own reasoning_content back alongside a tool call
        # and answered normally, so the round trip is known to be clean.
        for model_id in ("muse-glimmer", "gpt-oss-20b", "glm-5-3", "glm-5-3-flash"):
            assert MODELS[model_id].retains_reasoning, model_id

    def test_gemini_3_does_not_retain(self):
        # Its reasoning is encoded in thought signatures, and perturbing that
        # path risks the strict-validation 400s the client logs defensively.
        assert not MODELS["gemini-3-pro"].retains_reasoning

    def test_no_model_retains_reasoning_it_cannot_produce(self):
        for model_id, config in MODELS.items():
            if config.retains_reasoning:
                assert config.supports_reasoning, model_id
