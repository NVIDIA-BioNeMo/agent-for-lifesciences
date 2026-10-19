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

"""Tests for the ADMET property catalog.

The catalog is the part of this service that encodes judgement rather than
plumbing, so it is what is worth testing without a GPU or a 4 GB image. Two
claims matter: every alias a caller might plausibly type resolves, and the
direction of each property is right -- a catalog that says higher hERG is better
would turn the gate panel into a machine for selecting cardiotoxins.
"""

import pytest

from app.properties import (
    BY_NAME,
    CORE_PANEL,
    PROPERTIES,
    Direction,
    TaskType,
    describe,
    resolve,
    resolve_names,
)


class TestCatalogIntegrity:
    def test_canonical_names_are_unique(self):
        names = [prop.name for prop in PROPERTIES]
        assert len(names) == len(set(names))

    def test_no_alias_collides_with_another_property(self):
        seen: dict[str, str] = {}
        for prop in PROPERTIES:
            for alias in prop.aliases:
                assert alias not in seen, (
                    f"alias '{alias}' claimed by both {seen.get(alias)} and "
                    f"{prop.name}"
                )
                seen[alias] = prop.name

    def test_every_core_panel_property_is_in_the_catalog(self):
        for name in CORE_PANEL:
            assert name in BY_NAME, name

    def test_the_core_panel_is_the_eight_smdd_properties(self):
        assert len(CORE_PANEL) == 8

    def test_abbreviations_are_unique_and_short(self):
        # The tool layer uses these as table headers; a collision would put two
        # different properties in one column.
        seen: dict[str, str] = {}
        for prop in PROPERTIES:
            assert prop.abbrev, prop.name
            assert len(prop.abbrev) <= 8, (prop.name, prop.abbrev)
            assert prop.abbrev not in seen, (
                f"abbrev '{prop.abbrev}' claimed by both {seen.get(prop.abbrev)} "
                f"and {prop.name}"
            )
            seen[prop.abbrev] = prop.name

    def test_regression_properties_declare_units(self):
        # A bare number with no units is exactly the failure mode this catalog
        # exists to prevent. QED is the one genuinely dimensionless score.
        for prop in PROPERTIES:
            if prop.task_type is TaskType.REGRESSION and prop.name != "QED":
                assert prop.units, prop.name

    def test_classification_properties_declare_no_units(self):
        for prop in PROPERTIES:
            if prop.task_type is TaskType.CLASSIFICATION:
                assert prop.units == "", prop.name


class TestDirections:
    @pytest.mark.parametrize(
        "name",
        ["hERG", "AMES", "CYP3A4_Veith", "Clearance_Hepatocyte_AZ", "DILI"],
    )
    def test_liabilities_are_lower_is_better(self, name):
        assert BY_NAME[name].direction is Direction.LOWER

    @pytest.mark.parametrize("name", ["Caco2_Wang", "Solubility_AqSolDB", "QED"])
    def test_desirable_properties_are_higher_is_better(self, name):
        assert BY_NAME[name].direction is Direction.HIGHER

    @pytest.mark.parametrize("name", ["BBB_Martins", "PPBR_AZ", "VDss_Lombardo"])
    def test_programme_dependent_properties_are_context(self, name):
        assert BY_NAME[name].direction is Direction.CONTEXT

    def test_ld50_direction_is_explained(self):
        # log(1/(mol/kg)) inverts the intuition, so the note has to say so or
        # the direction reads as a bug.
        prop = BY_NAME["LD50_Zhu"]
        assert prop.direction is Direction.LOWER
        assert "lower" in prop.note.lower()

    def test_context_properties_that_could_mislead_carry_a_note(self):
        for name in ("BBB_Martins", "PPBR_AZ"):
            assert BY_NAME[name].note


class TestResolution:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("herg", "hERG"),
            ("hERG", "hERG"),
            ("HERG", "hERG"),
            ("bbb", "BBB_Martins"),
            ("solubility", "Solubility_AqSolDB"),
            ("cyp3a4", "CYP3A4_Veith"),
            ("clearance", "Clearance_Hepatocyte_AZ"),
            ("caco2", "Caco2_Wang"),
            ("caco_2", "Caco2_Wang"),
            ("ppbr", "PPBR_AZ"),
            ("ames", "AMES"),
            ("mw", "molecular_weight"),
            ("logp", "logP"),
        ],
    )
    def test_aliases_resolve_to_canonical_names(self, alias, expected):
        assert resolve(alias).name == expected

    def test_whitespace_is_tolerated(self):
        assert resolve("  herg  ").name == "hERG"

    def test_unknown_names_resolve_to_none(self):
        assert resolve("bioluminescence") is None
        assert resolve("") is None
        assert resolve(None) is None

    def test_resolve_names_separates_known_from_unknown(self):
        known, unknown = resolve_names(["herg", "frobnicity", "ames"])
        assert known == ["hERG", "AMES"]
        assert unknown == ["frobnicity"]

    def test_resolve_names_deduplicates_aliases_of_one_property(self):
        known, _ = resolve_names(["herg", "hERG", "herg_inhibition"])
        assert known == ["hERG"]

    def test_resolve_names_preserves_request_order(self):
        known, _ = resolve_names(["ames", "caco2", "herg"])
        assert known == ["AMES", "Caco2_Wang", "hERG"]


class TestDescribe:
    def test_describes_a_known_property(self):
        described = describe("herg")
        assert described["name"] == "hERG"
        assert described["known"] is True
        assert described["direction"] == "lower"
        assert described["task_type"] == "classification"
        assert described["abbrev"] == "hERG"

    def test_marks_an_unknown_property(self):
        described = describe("frobnicity")
        assert described == {"name": "frobnicity", "known": False}
