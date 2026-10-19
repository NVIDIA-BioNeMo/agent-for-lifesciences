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

"""The chat span has to say which session it belongs to.

A trace is only useful if it can be joined back to whatever drove it -- a
benchmark task, an eval case, a bug report. Without the session id the only
handle is the user message text, so tooling matches on a prefix of it, which is
fragile and becomes ambiguous the moment two runs send the same prompt. Both
are true of a benchmark harness, where every task of a type sends a near
identical message.
"""

from unittest.mock import patch

from api.telemetry import ChatSpanContext


class RecordingSpan:
    def __init__(self):
        self.attributes = {}
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def is_recording(self):
        return True

    def set_status(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def end(self):
        self.ended = True


class RecordingTracer:
    def __init__(self, span):
        self.span = span

    def start_span(self, name, context=None):
        self.span.name = name
        return self.span


def attributes_for(**kwargs) -> dict:
    span = RecordingSpan()
    with patch("api.telemetry.get_tracer", return_value=RecordingTracer(span)):
        with ChatSpanContext(**kwargs):
            pass
    return span.attributes


class TestTheJoinKey:
    def test_the_session_is_recorded_on_the_span(self):
        attributes = attributes_for(user_message="hello", session_id="sess-abc123")

        assert attributes["chat.session_id"] == "sess-abc123"

    def test_the_message_is_still_recorded_in_full(self):
        attributes = attributes_for(user_message="hello", session_id="s")

        assert attributes["chat.user_message"] == "hello"

    def test_no_session_records_no_key_rather_than_an_empty_one(self):
        """An empty attribute would join every sessionless trace to every other."""
        attributes = attributes_for(user_message="hello")

        assert "chat.session_id" not in attributes

    def test_an_empty_session_is_treated_as_absent(self):
        attributes = attributes_for(user_message="hello", session_id="")

        assert "chat.session_id" not in attributes

    def test_the_model_still_rides_along(self):
        attributes = attributes_for(
            user_message="hello", model_name="nemotron", session_id="s"
        )

        assert attributes["llm.model"] == "nemotron"
