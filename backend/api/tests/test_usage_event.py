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

"""Tests for the token totals the chat stream reports as it closes.

Aevom counted tokens long before it told anyone: the numbers reached Phoenix as
span attributes and stopped there, so no HTTP client could see what a turn cost.
This frame is what carries them out, and the eval harness turns it into Harbor's
token column -- which makes the key names below a contract with
``evals/aevom_eval/driver/trajectory.py``, not decoration.
"""

from api.graph.streaming import _usage_event
from api.telemetry import TokenAccumulator


def _spent(input_tokens: int, output_tokens: int) -> TokenAccumulator:
    accumulator = TokenAccumulator()
    accumulator.add(input_tokens, output_tokens)
    return accumulator


class TestUsageEvent:
    def test_it_reports_what_the_turn_spent(self):
        event = _usage_event(_spent(1200, 340))
        assert event["usage"]["input_tokens"] == 1200
        assert event["usage"]["output_tokens"] == 340

    def test_the_total_is_the_two_added_up(self):
        """Sent rather than left to the reader, because every consumer wants it."""
        assert _usage_event(_spent(1200, 340))["usage"]["total_tokens"] == 1540

    def test_a_turn_of_many_calls_arrives_as_one_total(self):
        """A turn spans an LLM call per agent iteration. Harbor counts trials,
        not calls, so what crosses is the sum the accumulator already holds."""
        accumulator = _spent(500, 100)
        accumulator.add(700, 240)
        event = _usage_event(accumulator)
        assert event["usage"]["input_tokens"] == 1200
        assert event["usage"]["output_tokens"] == 340

    def test_a_turn_that_called_no_model_reports_zero(self):
        """Zero is a real answer here. The absent case is a stream that died
        before this frame, which never reaches a client at all."""
        event = _usage_event(TokenAccumulator())
        assert event["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def test_the_keys_are_the_ones_the_eval_driver_reads(self):
        """Renaming any of these silently empties Harbor's token column, since
        the driver keys off them by name. Change both sides or neither."""
        event = _usage_event(_spent(1, 2))
        assert set(event) == {"usage"}
        assert set(event["usage"]) == {
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }
