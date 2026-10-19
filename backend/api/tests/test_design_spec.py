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

"""The design spec model and the checks run on a hotspot before design.

The thresholds under test come from how RFD4-Proteina is trained, so the tests
pin the behaviour at the edges of that distribution rather than at round
numbers.
"""

import json

import pytest

from api.selections.design_checks import (
    HIGH_HOTSPOT_ATOMS,
    MAX_EPITOPE_SPAN,
    VERY_HIGH_HOTSPOT_ATOMS,
    hotspot_warnings,
    selector_atoms,
)
from api.selections.design_spec import (
    DesignSpec,
    SpecError,
    SpecLayer,
    parse_design_spec,
    parse_layer,
    wrap_hotspot,
)
from api.selections.structure import StructureAtom


def residue_atoms(chain, seq, n_atoms=8, x=0.0, comp="ALA"):
    """A residue with ``n_atoms`` atoms, the first of them its CA."""
    names = ["CA"] + [f"X{i}" for i in range(n_atoms - 1)]
    return [
        StructureAtom(
            name=name, element="C", id=seq * 100 + i, x=x, y=0.0, z=0.0,
            label_chain=chain, label_seq=seq, auth_chain=chain, auth_seq=seq,
            ins_code="", comp_id=comp, entity_type="polymer",
        )
        for i, name in enumerate(names)
    ]


def protein(n_residues, chain="A", spacing=3.8, n_atoms=8):
    atoms = []
    for seq in range(1, n_residues + 1):
        atoms.extend(residue_atoms(chain, seq, n_atoms, x=seq * spacing))
    return atoms


class TestSpecModel:
    def test_round_trips(self):
        spec = DesignSpec("mmcif_t", [SpecLayer("hotspot", "selection_a")], name="n", parent="design_spec_p")
        assert parse_design_spec(spec.to_json()) == spec

    def test_a_bare_hotspot_wraps_into_a_one_layer_spec(self):
        spec = wrap_hotspot("mmcif_t", "selection_a")
        assert [(l.condition, l.selection_ref) for l in spec.layers] == [("hotspot", "selection_a")]

    def test_the_rfd4_condition_name_is_accepted(self):
        assert parse_layer({"condition": "C_HOT", "selection_ref": "selection_a"}).condition == "hotspot"

    def test_unsupported_conditions_say_what_is_supported(self):
        with pytest.raises(SpecError, match="hotspot"):
            parse_layer({"condition": "centroid", "selection_ref": "selection_a"})

    def test_a_layer_needs_a_selection(self):
        with pytest.raises(SpecError, match="selection_ref"):
            parse_layer({"condition": "hotspot"})

    def test_a_repeated_condition_is_refused(self):
        raw = {
            "target_ref": "mmcif_t",
            "layers": [
                {"condition": "hotspot", "selection_ref": "selection_a"},
                {"condition": "hotspot", "selection_ref": "selection_b"},
            ],
        }
        with pytest.raises(SpecError, match="more than once"):
            parse_design_spec(raw)

    @pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"layers": []}', b'{"target_ref": "x"}'])
    def test_something_that_is_not_a_spec_is_refused(self, raw):
        with pytest.raises(SpecError, match="not a design spec"):
            parse_design_spec(raw)

    def test_a_newer_version_is_refused_rather_than_misread(self):
        raw = json.dumps({"version": 99, "target_ref": "t", "layers": []})
        with pytest.raises(SpecError, match="version"):
            parse_design_spec(raw)

    def test_with_layers_replaces_only_the_named_condition(self):
        spec = DesignSpec("t", [SpecLayer("hotspot", "selection_a")])
        edited = spec.with_layers([SpecLayer("hotspot", "selection_b")])
        assert [l.selection_ref for l in edited.layers] == ["selection_b"]
        assert [l.selection_ref for l in spec.layers] == ["selection_a"]


class TestSelectorReading:
    def test_named_atoms_are_counted_by_name(self):
        assert selector_atoms("sel('A/*/40/[CG,OH]')", []) == ("A", 40, 2)
        assert selector_atoms("sel('A/*/40/CG')", []) == ("A", 40, 1)

    def test_a_whole_residue_counts_its_atoms_in_the_target(self):
        assert selector_atoms("sel('A/*/3')", protein(5, n_atoms=7)) == ("A", 3, 7)

    def test_something_else_is_ignored(self):
        assert selector_atoms("sel('A')", []) is None


class TestHotspotWarnings:
    def whole(self, residues, chain="A"):
        return [f"sel('{chain}/*/{n}')" for n in residues]

    def test_a_typical_epitope_has_no_warnings(self):
        target = protein(200)
        assert hotspot_warnings(["sel('A/*/10/CB')", "sel('A/*/11/CB')", "sel('A/*/12/CB')"], target) == []

    def test_more_than_usual_is_a_mild_note(self):
        target = protein(400)
        warnings = hotspot_warnings(self.whole(range(10, 12)), target)  # 16 atoms
        assert HIGH_HOTSPOT_ATOMS < 16 <= VERY_HIGH_HOTSPOT_ATOMS
        assert len(warnings) == 1
        assert "more than the ~5" in warnings[0]

    def test_far_outside_training_is_a_strong_warning_with_a_remedy(self):
        target = protein(400)
        (warning,) = hotspot_warnings(self.whole(range(10, 13)), target)  # 24 atoms
        assert "outside what it learned" in warning
        assert "atoms=" in warning

    def test_the_training_cap_is_reported_for_a_small_target(self):
        target = protein(10, n_atoms=8)  # 80 atoms; 24 is 30%
        warnings = hotspot_warnings(self.whole(range(1, 4)), target)
        assert any("capped hotspots at 10%" in w for w in warnings)

    def test_most_of_a_chain_is_not_an_epitope(self):
        target = protein(10)
        warnings = hotspot_warnings(self.whole(range(1, 8)), target)
        assert any("most of the chain" in w for w in warnings)

    def test_distant_residues_cannot_share_a_binder(self):
        target = protein(100, spacing=3.8)
        far = hotspot_warnings(["sel('A/*/5/CA')", "sel('A/*/60/CA')"], target)
        assert any("A apart" in w for w in far)
        assert 3.8 * 55 > MAX_EPITOPE_SPAN

    def test_a_compact_patch_is_not_flagged_for_span(self):
        target = protein(100, spacing=3.8)
        assert hotspot_warnings(["sel('A/*/5/CA')", "sel('A/*/8/CA')"], target) == []

    def test_no_selectors_no_warnings(self):
        assert hotspot_warnings([], protein(5)) == []
