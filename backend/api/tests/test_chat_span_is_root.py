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

"""A user turn is a trace of its own, rooted at chat_request.

Whatever span happens to be active when the turn starts must not adopt it.
FastAPI's native tracing is the case that bit: it made every chat_request a
child of a ``POST /api/chat`` server span, so the Phoenix trace list showed
HTTP requests instead of user turns.
"""

from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from api.telemetry import ChatSpanContext, IterationSpanContext, trace_tool_execution
from api.training.telemetry import EpisodeSpan


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with patch("api.telemetry.get_tracer", return_value=tracer), patch(
        "api.training.telemetry.get_tracer", return_value=tracer
    ):
        yield tracer, exporter


def by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


class TestChatRequestIsTheRoot:
    def test_an_active_http_span_does_not_parent_the_turn(self, tracing):
        tracer, exporter = tracing

        with tracer.start_as_current_span("POST /api/chat"):
            with ChatSpanContext("hello"):
                pass

        assert by_name(exporter)["chat_request"].parent is None

    def test_iterations_and_tools_still_nest_under_it(self, tracing):
        tracer, exporter = tracing

        with tracer.start_as_current_span("POST /api/chat"):
            with ChatSpanContext("hello"):
                with IterationSpanContext(1):
                    trace_tool_execution("molecule_info", {"smiles": "C"}).end()

        spans = by_name(exporter)
        chat = spans["chat_request"]
        iteration = spans["reasoning_iteration.1"]
        tool = spans["tool.molecule_info"]
        assert iteration.parent.span_id == chat.context.span_id
        assert tool.parent.span_id == iteration.context.span_id
        assert tool.context.trace_id == chat.context.trace_id

    def test_the_http_span_is_on_a_different_trace(self, tracing):
        tracer, exporter = tracing

        with tracer.start_as_current_span("POST /api/chat"):
            with ChatSpanContext("hello"):
                pass

        spans = by_name(exporter)
        assert (
            spans["chat_request"].context.trace_id
            != spans["POST /api/chat"].context.trace_id
        )


class TestTheTrainingEpisodeIsTheRoot:
    def test_an_active_seed_session_span_does_not_parent_the_episode(self, tracing):
        tracer, exporter = tracing

        with tracer.start_as_current_span("POST /seed_session"):
            episode = EpisodeSpan(task_prompt="dock it", episode_id="ep-1")
        episode.span.end()

        assert by_name(exporter)["chat_request"].parent is None
