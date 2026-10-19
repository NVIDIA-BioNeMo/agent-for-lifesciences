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

"""Tests that the driver keeps the token totals the stream reports.

The turn's cost arrives as one SSE frame at the end of the stream and is never
repeated, so if the trajectory drops it there is nothing to fall back on and
Harbor's token column stays empty. The frame's shape is produced by
``backend/api/graph/streaming.py::_usage_event``; the tests there pin the names,
these pin that the driver keeps them.

Harbor-free, like the rest of evals/tests: this runs in the plain devcontainer.
"""

from __future__ import annotations

from aevom_eval.driver.trajectory import Trajectory

USAGE_FRAME = {"usage": {"input_tokens": 1200, "output_tokens": 340, "total_tokens": 1540}}


def _trajectory() -> Trajectory:
    return Trajectory(instruction="dock it", session_id="s1", model_id="m1")


class TestUsageCapture:
    def test_the_frame_is_kept(self):
        trajectory = _trajectory()
        trajectory.record(USAGE_FRAME)
        assert trajectory.usage == USAGE_FRAME["usage"]

    def test_it_survives_into_the_written_record(self):
        """The verifier and the Harbor agent both read the file, not the object."""
        trajectory = _trajectory()
        trajectory.record(USAGE_FRAME)
        assert trajectory.to_dict()["usage"]["total_tokens"] == 1540

    def test_a_stream_that_sent_none_reports_none(self):
        """Not zero. A turn whose stream died spent tokens nobody counted, and
        recording that as 0 would tell Harbor the trial was free."""
        trajectory = _trajectory()
        trajectory.record({"content": "hello"})
        assert trajectory.usage is None
        assert trajectory.to_dict()["usage"] is None

    def test_the_rest_of_the_turn_is_undisturbed(self):
        """The frame shares the stream with everything else, so it must not be
        mistaken for content and must not swallow what came before it."""
        trajectory = _trajectory()
        trajectory.record({"content": "the answer"})
        trajectory.record(USAGE_FRAME)
        assert trajectory.final_content == "the answer"
        assert trajectory.usage is not None

    def test_it_stays_in_the_raw_events(self):
        """Every frame is kept verbatim so a later scoring change can reread the
        run rather than re-run it."""
        trajectory = _trajectory()
        trajectory.record(USAGE_FRAME)
        assert USAGE_FRAME in trajectory.to_dict()["raw_events"]
