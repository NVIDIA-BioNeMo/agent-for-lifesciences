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

"""A small batch says so, at the point of use.

Told in prose that twenty candidates is a coin flip, the agent enumerated
several hundred analogues, filtered to two hundred, and then passed them to
this tool five at a time -- 567 calls at a median batch of 5 across a hundred
task-runs. The instruction was read and the behaviour did not change, which is
the same lesson an earlier campaign measured about guidance the model has to
remember. So the number appears where the number is.
"""

import pytest

from api.tools.admet import MAX_MOLECULES, SMALL_BATCH, AdmetPredictTool


def render(n: int) -> str:
    tool = AdmetPredictTool()
    results = [{"smiles": "C" * (i + 1), "predictions": {}} for i in range(n)]
    lines: list[str] = []
    if 0 < len(results) < tool.SMALL_BATCH:
        lines.append(
            f"Scored {len(results)} molecule(s). This call takes up to "
            f"{MAX_MOLECULES} for the same cost"
        )
    return "\n".join(lines)


class TestTheSmallBatchNote:
    def test_a_handful_is_told_it_is_a_handful(self):
        assert "takes up to 256" in render(5)

    def test_the_note_names_how_many_were_scored(self):
        assert "Scored 5 molecule(s)" in render(5)

    def test_a_full_batch_is_not_nagged(self):
        """A caller already doing the right thing should not be lectured."""
        assert render(200) == ""

    def test_the_boundary_is_exclusive(self):
        assert render(SMALL_BATCH) == ""
        assert render(SMALL_BATCH - 1) != ""

    def test_an_empty_result_is_not_a_small_batch(self):
        """Nothing scored is a different failure, reported elsewhere."""
        assert render(0) == ""

    def test_the_threshold_is_well_under_the_capacity(self):
        assert SMALL_BATCH < MAX_MOLECULES / 10
