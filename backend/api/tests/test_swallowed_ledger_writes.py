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

"""A silently-dropped ledger write must at least be countable.

`TurnRecorder` swallows its own failures deliberately: a findings row is worth
less than the docking run it describes, so a failed write must never take down
a turn. That judgement is right and these tests do not challenge it.

What they pin is the consequence of the judgement. A real bug -- every second
write of a molecule raising `MissingGreenlet`, silently dropping whole ADMET
panels out of the candidate ledger that `best_candidate` ranks on -- survived a
full 20-task live run because a WARNING log line was the only evidence it left.
Swallowing without counting is how a data-loss bug becomes invisible.
"""

import pytest

from api.conversations.recorder import (
    TurnRecorder,
    reset_swallowed_writes,
    swallowed_writes,
)


class ExplodingSessionFactory:
    """A session factory whose sessions fail on use, to force the swallow."""

    def __call__(self):
        return self

    async def __aenter__(self):
        raise RuntimeError("database is gone")

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def clean_counters():
    reset_swallowed_writes()
    yield
    reset_swallowed_writes()


def recorder() -> TurnRecorder:
    import uuid

    return TurnRecorder(
        ExplodingSessionFactory(), "s1", uuid.uuid4(), uuid.uuid4(), 1
    )


@pytest.mark.asyncio
class TestFailuresAreStillSwallowed:
    """The existing contract: a failed write does not raise."""

    async def test_recording_a_message_does_not_raise(self):
        await recorder().record_user("dock aspirin into 1ABC")

    async def test_capturing_candidates_does_not_raise(self):
        from api.tools.base import ToolResult

        await recorder().capture_candidates("admet_predict", ToolResult.text("x"))


@pytest.mark.asyncio
class TestFailuresAreCounted:
    """The new contract: silence is measurable."""

    async def test_a_healthy_process_reports_nothing(self):
        assert swallowed_writes() == {}

    async def test_a_dropped_message_is_counted_under_its_operation(self):
        await recorder().record_user("hello")

        assert swallowed_writes() == {"append_message": 1}

    async def test_repeated_failures_accumulate(self):
        rec = recorder()
        await rec.record_user("one")
        await rec.record_user("two")

        assert swallowed_writes()["append_message"] == 2

    async def test_operations_are_counted_separately(self):
        rec = recorder()
        await rec.record_user("hello")
        await rec.update_tool("call-1", "patched")

        assert swallowed_writes() == {
            "append_message": 1,
            "update_tool_message": 1,
        }

    async def test_the_counter_is_a_copy_not_the_live_dict(self):
        await recorder().record_user("hello")

        snapshot = swallowed_writes()
        snapshot["append_message"] = 999

        assert swallowed_writes()["append_message"] == 1
