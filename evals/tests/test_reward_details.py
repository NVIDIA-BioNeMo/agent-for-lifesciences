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

"""Tests for the projection into Harbor's viewer reward schema.

``reward.json`` and ``reward-details.json`` exist only so the viewer's verifier
panel renders our dimensions instead of leaving a reader to open
``aevom_verifier.json`` by hand. The viewer validates both files and falls back to
a raw JSON dump when a field is missing or the wrong type -- with no error
anywhere -- so the failure mode is a panel that quietly stops working after an
unrelated change to scoring.

The predicates below are transcribed from the validators in Harbor 0.20.0's
viewer bundle. Re-check them on a Harbor bump; the pin lives in .gitlab-ci.yml.
Harbor-free on purpose, like test_scoring.py.
"""

from __future__ import annotations

import json

from aevom_eval.scoring import (
    ANSWER,
    NO_ERRORS,
    OVERALL,
    TOOL_SELECTION,
    reward_details,
    score_trajectory,
)


def is_reward_json(value) -> bool:
    """The viewer's check on reward.json: an object of finite numbers."""
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(number, (int, float))
        and not isinstance(number, bool)
        and number == number  # noqa: PLR0124 - excludes NaN, as Number.isFinite does
        and number not in (float("inf"), float("-inf"))
        for number in value.values()
    )


def is_criterion(value) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("value"), (int, float))
        and not isinstance(value.get("value"), bool)
        and isinstance(value.get("weight"), (int, float))
        and not isinstance(value.get("weight"), bool)
    )


def is_reward_entry(value) -> bool:
    if not isinstance(value, dict):
        return False
    warnings = value.get("warnings")
    judge_output = value.get("judge_output")
    return (
        isinstance(value.get("kind"), str)
        and isinstance(value.get("score"), (int, float))
        and not isinstance(value.get("score"), bool)
        and isinstance(value.get("criteria"), list)
        and all(is_criterion(criterion) for criterion in value["criteria"])
        and (
            warnings is None
            or (
                isinstance(warnings, list)
                and all(isinstance(item, str) for item in warnings)
            )
        )
        and (judge_output is None or isinstance(judge_output, str))
    )


def is_reward_details_json(value) -> bool:
    """The viewer's check on reward-details.json."""
    if not isinstance(value, dict):
        return False
    return all(
        all(is_reward_entry(item) for item in entry)
        if isinstance(entry, list)
        else is_reward_entry(entry)
        for entry in value.values()
    )


def trajectory() -> dict:
    return {
        "session_id": "s1",
        "final_content": "The molecular weight of caffeine is 194.19 g/mol.",
        "tool_calls": [
            {"name": "search_pubchem", "success": True},
            {"name": "get_pubchem_properties", "success": False},
        ],
        "artifacts": [{"data_type": "sdf"}],
        "bridge_events": [],
        "errors": [],
        "hit_iteration_limit": False,
    }


EXPECT = {
    "required_tools": ["search_pubchem", "dock_ligand"],
    "answer_contains": ["194.19"],
}


class TestSchema:
    def test_rewards_satisfy_the_viewer(self):
        score = score_trajectory(trajectory(), EXPECT)
        assert is_reward_json(score.rewards)

    def test_details_satisfy_the_viewer(self):
        details = reward_details(score_trajectory(trajectory(), EXPECT))
        assert is_reward_details_json(details)

    def test_both_survive_a_json_round_trip(self):
        # What the viewer parses is the file, not the dict, and json.dumps is where
        # a stray non-serialisable value would surface.
        score = score_trajectory(trajectory(), EXPECT)
        assert is_reward_json(json.loads(json.dumps(score.rewards)))
        assert is_reward_details_json(json.loads(json.dumps(reward_details(score))))

    def test_a_missing_trajectory_still_projects(self):
        # The zero-score path builds its rewards separately from the scoring path,
        # so it is the one most likely to drift out of the schema.
        score = score_trajectory(None, EXPECT)
        assert is_reward_json(score.rewards)
        assert is_reward_details_json(reward_details(score))


class TestContent:
    def test_every_dimension_appears(self):
        score = score_trajectory(trajectory(), EXPECT)
        assert set(reward_details(score)) == set(score.rewards)

    def test_a_dimension_score_matches_its_reward(self):
        score = score_trajectory(trajectory(), EXPECT)
        details = reward_details(score)
        for dimension, reward in score.rewards.items():
            assert details[dimension]["score"] == reward

    def test_criteria_carry_the_failure_detail(self):
        # The detail is the whole reason to read this panel: it names the observed
        # value, not just the verdict.
        details = reward_details(score_trajectory(trajectory(), EXPECT))
        failed = [
            criterion
            for criterion in details[TOOL_SELECTION]["criteria"]
            if criterion["value"] == 0.0
        ]
        assert failed, "dock_ligand was never called, so a criterion should fail"
        assert "dock_ligand" in failed[0]["description"]

    def test_overall_is_composed_of_the_other_dimensions(self):
        score = score_trajectory(trajectory(), EXPECT)
        criteria = reward_details(score)[OVERALL]["criteria"]
        assert {criterion["name"] for criterion in criteria} == set(score.rewards) - {
            OVERALL
        }
        # Only declared dimensions are scored, so this EXPECT yields exactly these
        # two. no_errors is absent rather than a free 1.0, even though the
        # trajectory has a failed tool call -- which is the property that makes the
        # overall reward a mean over what the task actually asked for.
        assert {criterion["name"] for criterion in criteria} == {TOOL_SELECTION, ANSWER}
        assert NO_ERRORS not in reward_details(score)
