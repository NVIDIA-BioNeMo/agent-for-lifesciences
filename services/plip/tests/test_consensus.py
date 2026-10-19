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

"""Tests for the consensus fold across structures.

The fold itself is arithmetic, so these tests are about the judgment calls: that
a structure with no interactions still counts in the denominator, that duplicate
contacts within one pose are one vote rather than two, and that what fell below
the threshold is reported rather than discarded -- an interaction in 4 of 10 poses
is a real finding about the sampler.
"""

import pytest

from app.consensus import DEFAULT_THRESHOLD, MAX_MINORITY_REPORTED, fold

TRP84 = "A:84:TRP:pistacking"
PHE288 = "A:288:PHE:hbond"
TYR121 = "A:121:TYR:waterbridge"


class TestFold:
    def test_an_interaction_in_every_structure_is_conserved(self):
        result = fold({"a": [TRP84], "b": [TRP84], "c": [TRP84]})
        assert [entry.key for entry in result.conserved] == [TRP84]
        assert result.conserved[0].fraction == 1.0
        assert result.conserved[0].sources == ["a", "b", "c"]

    def test_an_interaction_in_one_of_four_is_variable(self):
        result = fold({"a": [TRP84], "b": [], "c": [], "d": []})
        assert result.conserved == []
        assert [entry.key for entry in result.variable] == [TRP84]
        assert result.variable[0].fraction == 0.25

    def test_the_threshold_is_inclusive(self):
        # Exactly half the poses passes at the default. An interaction the
        # better-scoring half agrees on should not be dropped on a tie.
        result = fold({"a": [TRP84], "b": [TRP84], "c": [], "d": []})
        assert [entry.key for entry in result.conserved] == [TRP84]

    def test_a_stricter_threshold_moves_it_to_variable(self):
        result = fold({"a": [TRP84], "b": [TRP84], "c": [], "d": []}, threshold=0.75)
        assert result.conserved == []
        assert [entry.key for entry in result.variable] == [TRP84]

    def test_structures_with_no_interactions_still_count(self):
        # Dropping them would inflate every fraction: a pose that binds nothing
        # is evidence against the interaction, not absence of evidence.
        result = fold({"a": [TRP84], "b": [], "c": []})
        assert result.total == 3
        assert result.variable[0].fraction == pytest.approx(0.333, abs=0.001)

    def test_duplicates_within_one_structure_are_one_vote(self):
        # Three contacts in one pose out of two poses is half, not three halves.
        result = fold({"a": [TRP84, TRP84, TRP84], "b": []})
        assert result.conserved[0].count == 1
        assert result.conserved[0].fraction == 0.5

    def test_orders_by_frequency_then_residue(self):
        result = fold(
            {
                "a": [TRP84, PHE288, TYR121],
                "b": [TRP84, PHE288],
                "c": [TRP84],
            },
            threshold=0.0,
        )
        assert [entry.key for entry in result.conserved] == [TRP84, PHE288, TYR121]

    def test_parses_the_residue_out_of_the_key(self):
        entry = fold({"a": [TRP84]}).conserved[0]
        assert (entry.chain, entry.residue_number, entry.residue_name, entry.family) == (
            "A",
            84,
            "TRP",
            "pistacking",
        )

    def test_a_malformed_key_is_skipped(self):
        result = fold({"a": [TRP84, "nonsense", "A:notanumber:TRP:hbond"]})
        assert [entry.key for entry in result.conserved] == [TRP84]

    def test_an_empty_ensemble_folds_to_nothing(self):
        result = fold({})
        assert result.total == 0
        assert result.conserved == []
        assert result.variable == []

    def test_the_default_threshold_is_half(self):
        assert fold({"a": [TRP84]}).threshold == DEFAULT_THRESHOLD


class TestReporting:
    def test_groups_conserved_interactions_by_residue(self):
        result = fold(
            {
                "a": [TRP84, "A:84:TRP:hydrophobic", PHE288],
                "b": [TRP84, "A:84:TRP:hydrophobic", PHE288],
            }
        )
        residues = result.residues()
        assert residues[0]["residue_number"] == 84
        assert residues[0]["families"] == ["hydrophobic", "pistacking"]

    def test_residue_grouping_keeps_the_weakest_support(self):
        # A residue is only as well-supported as its least frequent interaction,
        # and rounding that up would overstate the evidence.
        result = fold(
            {
                "a": [TRP84, "A:84:TRP:hydrophobic"],
                "b": [TRP84],
            },
            threshold=0.5,
        )
        assert result.residues()[0]["min_fraction"] == 0.5

    def test_the_variable_list_is_capped_but_counted(self):
        many = {
            "a": [f"A:{index}:ALA:hydrophobic" for index in range(200)],
            "b": [],
            "c": [],
        }
        payload = fold(many).to_dict()
        assert len(payload["variable"]) == MAX_MINORITY_REPORTED
        assert payload["num_variable"] == 200

    def test_the_payload_carries_the_denominator(self):
        payload = fold({"a": [TRP84], "b": [TRP84]}).to_dict()
        assert payload["total_structures"] == 2
        assert payload["conserved"][0]["count"] == 2
