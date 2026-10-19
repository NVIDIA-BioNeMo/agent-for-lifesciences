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

"""Tests for prompt token budgeting.

Before this existed, context overflow surfaced only as a provider 400 whose
sole diagnostic was a message count. These tests pin the two things that make
the budget trustworthy: that it counts the parts of a prompt that actually
grow (tool schemas, tool results, reasoning), and that it reports its own
error against the provider's token count rather than assuming it is right.
"""

import logging
import math

import pytest

from api.context.budget import (
    CHARS_PER_TOKEN,
    CONTEXT_SAFETY_MARGIN_TOKENS,
    BudgetReport,
    check_context_budget,
    estimate_messages_tokens,
    estimate_tokens,
    record_budget_on_span,
    record_estimate_error,
)
from api.llm.config import MODELS, ModelConfig


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def messages_of(char_count: int, count: int = 1) -> list[dict]:
    return [{"role": "user", "content": "x" * char_count} for _ in range(count)]


class TestEstimation:
    def test_empty_text_costs_nothing(self):
        assert estimate_tokens("") == 0

    def test_estimate_scales_with_length(self):
        assert estimate_tokens("x" * 400) == math.ceil(400 / CHARS_PER_TOKEN)

    def test_the_estimator_errs_toward_over_counting(self):
        # Measured against Nemotron 3 Super, 4.0 chars/token under-counted by
        # ~11% -- a budget that under-counts fails to warn precisely when the
        # prompt is largest. This constant must stay on the safe side.
        assert CHARS_PER_TOKEN < 4

    def test_tool_schemas_are_counted(self):
        # Schemas are resent on every call and are the largest fixed floor in
        # this system; ignoring them makes every estimate optimistic.
        tools = [
            {
                "type": "function",
                "function": {"name": "t", "description": "d" * 2000},
            }
        ]
        without = estimate_messages_tokens(messages_of(100))
        with_tools = estimate_messages_tokens(messages_of(100), tools)
        assert with_tools > without + 400

    def test_tool_calls_on_an_assistant_message_are_counted(self):
        plain = [{"role": "assistant", "content": ""}]
        calling = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "dock", "arguments": "a" * 1000},
                    }
                ],
            }
        ]
        assert estimate_messages_tokens(calling) > estimate_messages_tokens(plain)

    def test_reasoning_content_is_counted(self):
        # Reasoning is re-sent for the Nemotron family, so it costs budget.
        plain = [{"role": "assistant", "content": "ok"}]
        thinking = [
            {"role": "assistant", "content": "ok", "reasoning_content": "r" * 1000}
        ]
        assert estimate_messages_tokens(thinking) > estimate_messages_tokens(plain)

    def test_non_string_content_does_not_raise(self):
        multimodal = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
        assert estimate_messages_tokens(multimodal) > 0

    def test_unserializable_content_does_not_raise(self):
        assert estimate_messages_tokens([{"role": "user", "content": object()}]) > 0

    def test_each_message_carries_framing_overhead(self):
        # Many tiny messages are not free; framing is what makes a 40-turn
        # trajectory expensive even when each message is short.
        one = estimate_messages_tokens(messages_of(4, count=1))
        ten = estimate_messages_tokens(messages_of(4, count=10))
        assert ten > one * 5


class TestBudgetArithmetic:
    def test_available_is_the_window_less_generation_and_margin(self):
        config = ModelConfig(id="t", name="t", context_window=10_000, max_tokens=1_000)
        report = check_context_budget(config, messages_of(4))
        assert report.available_tokens == 10_000 - 1_000 - CONTEXT_SAFETY_MARGIN_TOKENS

    def test_a_per_call_generation_limit_overrides_the_model_default(self):
        config = ModelConfig(id="t", name="t", context_window=10_000, max_tokens=1_000)
        report = check_context_budget(
            config, messages_of(4), max_output_tokens=5_000
        )
        assert report.reserved_output_tokens == 5_000

    def test_a_small_prompt_is_within_budget(self):
        config = ModelConfig(id="t", name="t", context_window=100_000, max_tokens=1_000)
        assert not check_context_budget(config, messages_of(100)).over_budget

    def test_a_large_prompt_breaches_the_budget(self):
        config = ModelConfig(id="t", name="t", context_window=10_000, max_tokens=1_000)
        report = check_context_budget(config, messages_of(200_000))
        assert report.over_budget
        assert report.headroom_tokens < 0

    def test_a_breach_is_logged_loudly(self, caplog):
        config = ModelConfig(id="t", name="t", context_window=10_000, max_tokens=1_000)
        with caplog.at_level(logging.WARNING, logger="api.context.budget"):
            check_context_budget(config, messages_of(200_000))
        assert "AEVOM-CONTEXT-BUDGET" in caplog.text

    def test_a_model_with_no_declared_window_yields_no_budget(self):
        # Rather than silently pretending a prompt fits.
        config = ModelConfig(id="t", name="t", context_window=0, max_tokens=1_000)
        report = check_context_budget(config, messages_of(100))
        assert report.available_tokens == 0
        assert report.utilization == 0.0


class TestSpanRecording:
    def test_the_budget_is_recorded(self):
        span = RecordingSpan()
        config = ModelConfig(id="t", name="t", context_window=10_000, max_tokens=1_000)
        record_budget_on_span(span, check_context_budget(config, messages_of(400)))

        assert span.attributes["context.tokens.budget"] == 6952
        assert span.attributes["context.tokens.window"] == 10_000
        assert span.attributes["context.over_budget"] is False
        assert span.attributes["context.tokens.used"] > 0

    def test_the_estimate_error_is_recorded_against_the_provider_count(self):
        span = RecordingSpan()
        record_estimate_error(span, estimated=1_100, actual=1_000)
        assert span.attributes["context.tokens.actual"] == 1_000
        assert span.attributes["context.tokens.estimate_error"] == 100
        assert span.attributes["context.tokens.estimate_ratio"] == 1.1

    def test_a_missing_provider_count_records_nothing(self):
        # Some endpoints omit usage; a zero would poison the calibration.
        span = RecordingSpan()
        record_estimate_error(span, estimated=1_100, actual=0)
        assert span.attributes == {}

    def test_recording_without_a_span_is_a_no_op(self):
        config = ModelConfig(id="t", name="t")
        record_budget_on_span(None, check_context_budget(config, messages_of(4)))
        record_estimate_error(None, 1, 1)

    def test_a_failing_span_does_not_propagate(self):
        class Broken:
            def set_attribute(self, key, value):
                raise RuntimeError("exporter is down")

        config = ModelConfig(id="t", name="t")
        record_budget_on_span(Broken(), check_context_budget(config, messages_of(4)))


class TestModelRegistry:
    def test_every_model_declares_a_context_window(self):
        for model_id, config in MODELS.items():
            assert config.context_window > 0, model_id

    def test_every_model_can_generate_within_its_window(self):
        # A generation limit at or above the window would make the available
        # budget zero or negative and every prompt look like a breach.
        for model_id, config in MODELS.items():
            assert config.max_tokens < config.context_window, model_id

    def test_every_model_leaves_room_for_a_real_prompt(self):
        for model_id, config in MODELS.items():
            report = BudgetReport(
                estimated_tokens=0,
                available_tokens=config.context_window
                - config.max_tokens
                - CONTEXT_SAFETY_MARGIN_TOKENS,
                context_window=config.context_window,
                reserved_output_tokens=config.max_tokens,
                message_count=0,
            )
            assert report.available_tokens > 8_000, model_id


class TestClientIntegration:
    def test_the_streaming_response_accepts_the_estimate(self):
        # The streaming path only learns the true prompt size at the end, so
        # the estimate has to be carried through to the usage chunk.
        from api.llm.client import StreamingResponse

        response = StreamingResponse(
            iter([]), supports_reasoning=False, estimated_prompt_tokens=1234
        )
        assert response._estimated_prompt_tokens == 1234
