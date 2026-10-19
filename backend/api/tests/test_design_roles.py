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

"""Design roles: the catalog, how a spec carries them, and how they compile.

The compile tests pin what each engine is *told*, because that is where a
role either takes effect or silently does not.
"""

import json

import pytest
from pydantic import ValidationError

from api.selections.design_checks import role_findings
from api.selections.design_spec import (
    DesignSpec,
    Layout,
    SpecError,
    SpecLayer,
    parse_design_spec,
    parse_layer,
    parse_layout,
)
from api.selections.models import SelectionData
from api.selections.registry import SelectionInfo, active_role_layers
from api.selections.role_compile import (
    MODE_BINDER,
    MODE_MOTIF,
    LayerInput,
    compile_rfd4,
    compile_rfdiffusion1,
    merge_ranges,
    residue_selector,
    resolve_selection,
)
from api.selections.roles import (
    ENGINE_RFD4,
    ENGINE_RFDIFFUSION1,
    ROLE_ORDER,
    ROLES,
    SUPPORT_NONE,
    apply_atom_rule,
    get_role,
    registry_payload,
)
from api.storage.mmcif_numbering import build_numbering_map

# --- A target and selections on it -----------------------------------------

AUTH_OFFSET = 16  # author numbering runs 16 ahead of label numbering
N_RESIDUES = 40
SEQUENCE = {5: "PHE", 6: "TYR", 7: "TRP", 8: "SER", 9: "ASP", 10: "LYS", 11: "ARG", 12: "LEU"}


# Residues sit 3.8 A apart; the carbonyl C reaches 2.5 A toward the next residue,
# so C-N across a peptide bond is a physical 1.3 A and the chain is unbroken.
OFFSET_X = {"N": 0.0, "CA": 1.0, "C": 2.5, "O": 2.5, "CB": 1.0}


def make_target(n: int = N_RESIDUES, gap_after: tuple[int, ...] = ()) -> str:
    header = """\
data_TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""
    rows = []
    atom_id = 1
    shift = 0.0
    for seq in range(1, n + 1):
        comp = SEQUENCE.get(seq, "ALA")
        for name in ("N", "CA", "C", "O", "CB"):
            rows.append(
                f"ATOM {atom_id} {name[0]} {name} . {comp} A 1 {seq} ? "
                f"{seq * 3.8 + shift + OFFSET_X[name]:.3f} 0.000 0.000 1.00 0.00 {seq + AUTH_OFFSET} {comp} A {name} 1"
            )
            atom_id += 1
        if seq in gap_after:
            shift += 8.0  # unmodelled residues: the chain jumps, the ids do not
    return header + "\n".join(rows) + "\n#\n"


TARGET = make_target()
NUMBERING = build_numbering_map(TARGET)


def atoms_of(seq: int, names=("N", "CA", "C", "O", "CB"), comp=None, chain="A"):
    """The atoms the viewer posts for one residue: label ids plus author ids."""
    comp = comp or SEQUENCE.get(seq, "ALA")
    return [
        {
            "name": name,
            "element": name[0],
            "id": seq * 100 + i,
            "coords": [0.0, 0.0, 0.0],
            "residue": {
                "name": comp,
                "seqNumber": seq,
                "chainId": chain,
                "insCode": "",
                "entityType": "polymer",
                "authSeqNumber": seq + AUTH_OFFSET,
                "authChainId": chain,
            },
        }
        for i, name in enumerate(names)
    ]


def selection_of(*residues) -> dict:
    """A stored selection from residue numbers or ``(seq, names)`` pairs."""
    atoms = []
    for item in residues:
        if isinstance(item, tuple):
            atoms.extend(atoms_of(item[0], names=item[1]))
        else:
            atoms.extend(atoms_of(item))
    return {"atoms": atoms}


def layer(condition, selection, value=None, ref=None) -> LayerInput:
    ref = ref or f"selection_{condition}_{value or 'x'}"
    return LayerInput(layer=SpecLayer(condition, ref, value), selection=selection)


def span(first: int, last: int) -> dict:
    return selection_of(*range(first, last + 1))


# --- The catalog -----------------------------------------------------------


class TestRegistry:
    def test_every_role_has_a_plain_label_and_an_rfd4_annotation(self):
        for role in ROLES.values():
            assert role.label and role.annotation
            assert role.annotation in role.tooltip, "the RFD4 term belongs in the tooltip"

    def test_the_five_primary_roles_are_named_with_plain_verbs(self):
        primary = [r.label for r in ROLES.values() if r.ui_group == "primary"]
        assert primary == ["Hotspot", "Keep", "Fix position", "Fix sequence", "Float"]

    def test_a_role_is_found_by_key_or_by_rfd4_annotation(self):
        assert get_role("hotspot") is get_role("C_HOT")
        assert get_role("c_crd").key == "fix_position"
        assert get_role("nonsense") is None

    def test_enum_roles_list_their_variants_and_only_those_the_nim_accepts(self):
        # The live NIM rejects STRAND and COIL: its enum is HELIX, SHEET, LOOP.
        assert [v.value for v in ROLES["secondary_structure"].variants] == ["HELIX", "SHEET", "LOOP"]
        assert [v.value for v in ROLES["rasa"].variants] == ["BURIED", "INTERMEDIATE", "EXPOSED"]

    def test_the_payload_is_json_and_names_both_engines(self):
        payload = json.loads(json.dumps(registry_payload()))
        assert [r["key"] for r in payload["roles"]] == list(ROLE_ORDER)
        assert {e["key"] for e in payload["engines"]} == {ENGINE_RFD4, ENGINE_RFDIFFUSION1}

    def test_rfdiffusion_cannot_take_float_or_chemistry(self):
        for key in ("float", "secondary_structure", "hbond", "pistack", "salt"):
            assert ROLES[key].support(ENGINE_RFDIFFUSION1)[0] == SUPPORT_NONE

    def test_colours_stay_clear_of_the_chain_greens_and_slates(self):
        greens = {"#22c55e", "#16a34a", "#4ade80", "#10b981"}
        assert not greens & {r.colour for r in ROLES.values()}


class TestAtomRules:
    def test_a_whole_aromatic_residue_narrows_to_its_ring(self):
        ring = apply_atom_rule(ROLES["pistack"].atom_rule, "PHE", ["N", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
        assert ring == ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]

    def test_half_a_ring_is_not_a_ring(self):
        assert apply_atom_rule(ROLES["pistack"].atom_rule, "PHE", ["CG", "CD1", "CD2"]) is None

    def test_a_residue_without_a_ring_cannot_pi_stack(self):
        assert apply_atom_rule(ROLES["pistack"].atom_rule, "ALA", ["N", "CA", "CB"]) is None

    def test_tryptophan_offers_either_ring(self):
        six = ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
        assert apply_atom_rule(ROLES["pistack"].atom_rule, "TRP", six) == six

    def test_donor_atoms_are_narrowed_from_a_whole_serine(self):
        donor = ROLES["hbond"].variant("DONOR").atom_rule
        assert apply_atom_rule(donor, "SER", ["N", "CA", "C", "O", "CB", "OG"]) == ["N", "OG"]

    def test_proline_has_no_amide_donor(self):
        donor = ROLES["hbond"].variant("DONOR").atom_rule
        assert apply_atom_rule(donor, "PRO", ["N", "CA", "CB"]) is None

    def test_an_aspartate_is_a_negative_charge_not_a_positive_one(self):
        positive = ROLES["salt"].variant("POSITIVE").atom_rule
        negative = ROLES["salt"].variant("NEGATIVE").atom_rule
        atoms = ["CG", "OD1", "OD2"]
        assert apply_atom_rule(positive, "ASP", atoms) is None
        assert apply_atom_rule(negative, "ASP", atoms) == ["OD1", "OD2"]

    def test_no_rule_accepts_the_pick_as_it_is(self):
        assert apply_atom_rule(None, "ALA", ["CA"]) == ["CA"]


# --- Roles on a selection --------------------------------------------------


class TestSelectionRole:
    def payload(self, **extra):
        return SelectionData(atoms=[], atom_count=0, residue_count=0, provenance="user", **extra)

    def test_a_role_may_be_named_by_its_annotation(self):
        assert self.payload(role="C_HOT").role == "hotspot"

    def test_an_unknown_role_is_rejected(self):
        with pytest.raises(ValidationError):
            self.payload(role="make_it_good")

    def test_a_plain_selection_has_no_role(self):
        data = self.payload()
        assert data.role is None and data.value is None

    def test_the_value_may_be_a_layout(self):
        assert self.payload(role="keep", value={"linkers": [[5, 10]]}).value == {"linkers": [[5, 10]]}


def info(ref, role, value=None, atoms=5):
    return SelectionInfo(
        reference=ref,
        molql="",
        action="select",
        atom_count=atoms,
        residue_count=1,
        summary="",
        provenance="user",
        role=role,
        role_value=value,
    )


class TestActiveRoleLayers:
    def test_the_newest_selection_for_a_role_wins(self):
        layers = active_role_layers([info("selection_a", "hotspot"), info("selection_b", "hotspot")])
        assert [l.reference for l in layers] == ["selection_b"]

    def test_committing_a_role_empty_removes_it(self):
        layers = active_role_layers(
            [info("selection_a", "hotspot"), info("selection_b", "hotspot", atoms=0)]
        )
        assert layers == []

    def test_float_groups_and_enum_values_are_separate_layers(self):
        layers = active_role_layers(
            [
                info("selection_a", "float", "motif"),
                info("selection_b", "float", "g2"),
                info("selection_c", "rasa", "BURIED"),
                info("selection_d", "rasa", "EXPOSED"),
            ]
        )
        assert len(layers) == 4

    def test_roleless_selections_are_not_layers(self):
        assert active_role_layers([info("selection_a", None)]) == []

    def test_layers_come_back_in_role_order(self):
        layers = active_role_layers([info("selection_k", "keep"), info("selection_h", "hotspot")])
        assert [l.role for l in layers] == ["hotspot", "keep"]


# --- The spec --------------------------------------------------------------


class TestSpecLayers:
    def test_float_may_appear_once_per_group(self):
        spec = DesignSpec(
            "t",
            [SpecLayer("float", "s1", "motif"), SpecLayer("float", "s2", "g2")],
        )
        assert parse_design_spec(spec.to_json()).layers_for("float")[1].value == "g2"

    def test_the_same_float_group_twice_is_refused(self):
        raw = {
            "target_ref": "t",
            "layers": [
                {"condition": "float", "selection_ref": "a", "value": "g"},
                {"condition": "float", "selection_ref": "b", "value": "g"},
            ],
        }
        with pytest.raises(SpecError, match="more than once"):
            parse_design_spec(raw)

    def test_float_defaults_to_the_motif_group(self):
        assert parse_layer({"condition": "float", "selection_ref": "s"}).value == "motif"

    def test_an_enum_role_needs_a_member_of_its_enum(self):
        with pytest.raises(SpecError, match="HELIX"):
            parse_layer({"condition": "secondary_structure", "selection_ref": "s", "value": "STRAND"})

    def test_an_enum_value_is_normalised(self):
        layer = parse_layer({"condition": "C_RAS", "selection_ref": "s", "value": "buried"})
        assert (layer.condition, layer.value) == ("rasa", "BURIED")

    def test_an_unknown_condition_lists_the_roles(self):
        with pytest.raises(SpecError, match="hotspot"):
            parse_layer({"condition": "vibes", "selection_ref": "s"})

    def test_role_is_accepted_as_an_alias_for_condition(self):
        layer = parse_layer({"role": "keep", "selection_ref": "s"})
        assert (layer.condition, layer.selection_ref) == ("keep", "s")

    def test_role_alias_takes_the_same_value_as_condition(self):
        layer = parse_layer({"role": "rasa", "selection_ref": "s", "value": "buried"})
        assert (layer.condition, layer.value) == ("rasa", "BURIED")

    def test_condition_wins_when_both_are_given(self):
        layer = parse_layer({"condition": "hotspot", "role": "keep", "selection_ref": "s"})
        assert layer.condition == "hotspot"

    def test_an_unknown_role_alias_lists_the_roles(self):
        with pytest.raises(SpecError, match="Unknown condition 'vibes'.*hotspot"):
            parse_layer({"role": "vibes", "selection_ref": "s"})

    def test_a_layer_with_neither_name_says_what_to_send(self):
        with pytest.raises(SpecError, match="'condition' \\(also accepted as 'role'\\)"):
            parse_layer({"selection_ref": "s"})

    def test_with_layers_replaces_by_condition_and_parameter(self):
        spec = DesignSpec("t", [SpecLayer("float", "s1", "motif"), SpecLayer("float", "s2", "g2")])
        edited = spec.with_layers([SpecLayer("float", "s9", "g2")])
        assert {l.value: l.selection_ref for l in edited.layers} == {"motif": "s1", "g2": "s9"}

    def test_the_engine_round_trips_and_is_checked(self):
        spec = DesignSpec("t", [SpecLayer("hotspot", "s")], engine="rfdiffusion1")
        assert parse_design_spec(spec.to_json()).engine == "rfdiffusion1"
        with pytest.raises(SpecError, match="engine"):
            parse_design_spec({"target_ref": "t", "layers": [], "engine": "magic"})


class TestLayout:
    def test_the_layout_rides_on_the_keep_layer(self):
        spec = DesignSpec("t", [SpecLayer("keep", "s", {"n_term": [5, 10], "linkers": [[8, 12]]})])
        assert spec.layout.n_term == (5, 10)
        assert spec.layout.linker(0) == (8, 12)

    def test_a_missing_linker_takes_the_default(self):
        assert Layout().linker(3) == (10, 30)

    def test_nonsense_lengths_are_refused(self):
        with pytest.raises(SpecError):
            parse_layout({"n_term": [20, 5]})
        with pytest.raises(SpecError):
            parse_layout({"linkers": [[1, 9999]]})
        with pytest.raises(SpecError):
            parse_layout("long")

    def test_a_single_number_is_a_fixed_length(self):
        assert parse_layout({"c_term": 12}).c_term == (12, 12)


class TestRoleFindings:
    def test_a_hotspot_outside_keep_is_an_error(self):
        found = role_findings({"keep": {1, 2, 3}, "hotspot": {3, 4}})
        assert any("Hotspot" in e and "outside Keep" in e for e in found.errors)

    def test_fixing_or_floating_outside_keep_is_an_error(self):
        found = role_findings({"keep": {1}, "fix_position": {2}, "float": {3}, "fix_sequence": {4}})
        assert len(found.errors) == 3

    def test_without_keep_nothing_is_outside_it(self):
        assert role_findings({"hotspot": {1}, "float": {2}}).errors == []

    def test_float_outside_fix_position_is_only_a_warning(self):
        found = role_findings({"keep": {1, 2}, "fix_position": {1}, "float": {1, 2}})
        assert found.errors == [] and len(found.warnings) == 1

    def test_messages_use_the_names_given(self):
        found = role_findings({"keep": {1}, "hotspot": {2}}, names={2: "A18"})
        assert "A18" in found.errors[0]


# --- Compiling for RFD4 ----------------------------------------------------


def compiled(*layers, **kwargs):
    return compile_rfd4(list(layers), NUMBERING, **kwargs)


# A chain with unmodelled residues after label 20. The label ids stay 1..40
# (a PDB converted to mmCIF is renumbered over the modelled residues), but the
# coordinates jump, which is what the model rejects if the contig bonds across.
BROKEN = make_target(gap_after=(20,))
BROKEN_NUMBERING = build_numbering_map(BROKEN)


class TestChainBreaks:
    def test_a_physical_chain_has_no_breaks(self):
        assert NUMBERING.breaks == {}

    def test_a_jump_in_coordinates_is_a_break_even_when_the_ids_are_consecutive(self):
        assert list(BROKEN_NUMBERING.breaks) == [("A", 20)]
        assert BROKEN_NUMBERING.breaks[("A", 20)] > 2.0

    def test_spans_are_cut_at_the_break(self):
        assert BROKEN_NUMBERING.split_span("A", 1, 40) == [(1, 20), (21, 40)]
        assert BROKEN_NUMBERING.split_span("A", 1, 15) == [(1, 15)]
        assert BROKEN_NUMBERING.split_span("A", 21, 40) == [(21, 40)]

    def test_a_break_is_described_in_the_numbers_the_user_sees(self):
        (note,) = BROKEN_NUMBERING.break_notes("A", 1, 40)
        assert "A36 and A37" in note and "unmodelled" in note

    def test_a_binder_uses_the_side_of_the_break_that_holds_the_hotspot(self):
        plan = compile_rfd4(
            [layer("hotspot", selection_of((5, ("CB",))))],
            BROKEN_NUMBERING,
            binder_min=20,
            binder_max=30,
        )
        assert plan.ok, plan.errors
        assert plan.contig == "20-30, /0, A1-20"
        # What is held matches what the contig carries, not the whole chain.
        assert plan.conditions["C_SEQ"]["true"] == [{"select": "sel('A/*/[1-20]')"}]
        # The other side is dropped out loud, in the numbers the user sees.
        (warning,) = [w for w in plan.warnings if "broken" in w]
        assert "A37-A56" in warning and "Add a Keep" in warning

    def test_the_far_side_is_used_when_the_hotspot_is_there(self):
        plan = compile_rfd4(
            [layer("hotspot", selection_of((30, ("CB",))))],
            BROKEN_NUMBERING,
            binder_min=20,
            binder_max=30,
        )
        assert plan.contig == "20-30, /0, A21-40"

    def test_a_hotspot_on_both_sides_of_the_break_cannot_run(self):
        plan = compile_rfd4(
            [layer("hotspot", selection_of((5, ("CB",)), (30, ("CB",))))],
            BROKEN_NUMBERING,
            binder_min=20,
            binder_max=30,
        )
        assert not plan.ok
        assert any("both sides of a break" in e and "A36" in e for e in plan.errors)

    def test_a_keep_across_the_break_cannot_run(self):
        plan = compile_rfd4(
            [layer("hotspot", selection_of((5, ("CB",)))), layer("keep", span(1, 40))],
            BROKEN_NUMBERING,
            binder_min=20,
            binder_max=30,
        )
        assert not plan.ok
        assert any("both sides of a break" in e for e in plan.errors)

    def test_a_motif_across_the_break_is_joined_by_a_linker(self):
        plan = compile_rfd4(
            [layer("keep", span(15, 25))],
            BROKEN_NUMBERING,
            mode="motif",
            layout=Layout(linkers=((6, 8),)),
        )
        assert plan.contig == "A15-20, 6-8, A21-25"
        assert any("broken between" in n for n in plan.notes)

    def test_a_keep_on_one_side_is_not_cut(self):
        plan = compile_rfd4(
            [layer("hotspot", selection_of((5, ("CB",)))), layer("keep", span(1, 15))],
            BROKEN_NUMBERING,
            binder_min=20,
            binder_max=30,
        )
        assert plan.contig == "20-30, /0, A1-15"
        assert not any("broken" in n for n in plan.notes)

    def test_the_original_binder_path_never_emits_two_segments_of_one_chain(self):
        from api.tools.rfd4_proteina import build_contig

        assert build_contig(BROKEN_NUMBERING, ["A"], 20, 30) == "20-30, /0, A1-20"
        assert build_contig(NUMBERING, ["A"], 20, 30) == "20-30, /0, A1-40"


class TestRanges:
    def test_adjacent_numbers_merge(self):
        assert merge_ranges([5, 3, 4, 9]) == [(3, 5), (9, 9)]

    def test_selectors_name_ranges(self):
        assert residue_selector("A", [10, 11, 12, 20]) == "sel('A/*/[10-12, 20]')"
        assert residue_selector("A", [10]) == "sel('A/*/10')"


class TestResolve:
    def test_author_numbers_are_translated_to_label_numbers(self):
        resolved = resolve_selection(selection_of(5), NUMBERING)
        residue = resolved.residues[0]
        assert (residue.author, residue.label) == ("A21", "A5")

    def test_what_cannot_be_placed_is_reported_not_dropped_quietly(self):
        selection = selection_of(5)
        selection["atoms"][0]["residue"]["authSeqNumber"] = 999
        assert resolve_selection(selection, NUMBERING).unmapped == {"A999"}

    def test_a_whole_residue_is_recognised(self):
        resolved = resolve_selection(selection_of(8), NUMBERING)
        assert not resolved.residues[0].is_whole  # five atoms of a serine's six


class TestBinderCompile:
    def test_a_hotspot_alone_matches_the_original_request(self):
        plan = compiled(layer("hotspot", selection_of((5, ("CB",)))), binder_min=60, binder_max=120)
        assert plan.ok
        assert plan.contig == "60-120, /0, A1-40"
        assert plan.conditions["C_CRD"] == {"true": [{"select": "(sel('A')) and is_protein_backbone"}]}
        assert plan.conditions["C_SEQ"] == {"true": [{"select": "sel('A')"}]}
        assert plan.conditions["C_HOT"] == {"true": [{"select": "sel('A/*/5/CB')"}]}

    def test_a_binder_needs_a_hotspot(self):
        plan = compiled(layer("keep", span(5, 12)))
        assert any("needs a Hotspot" in e for e in plan.errors)

    def test_keep_crops_the_target(self):
        plan = compiled(layer("keep", span(5, 12)), layer("hotspot", selection_of(6, 7)))
        assert plan.contig == "60-120, /0, A5-12"
        assert plan.conditions["C_SEQ"]["true"][0]["select"] == "sel('A/*/[5-12]')"
        assert "sel('A/*/[5-12]')" in plan.conditions["C_CRD"]["true"][0]["select"]

    def test_stretches_that_are_not_adjacent_are_carried_as_one_stretch(self):
        # The model refuses two segments of one chain ("two separate output
        # chains both named 'A'"), so a binder's target is one stretch per
        # chain, with the residues between the kept ones carried too.
        plan = compiled(layer("keep", selection_of(5, 6, 20, 21)), layer("hotspot", selection_of(5)))
        assert plan.contig == "60-120, /0, A5-21"
        assert plan.conditions["C_SEQ"]["true"] == [{"select": "sel('A/*/[5-21]')"}]
        assert any("between the kept stretches" in n for n in plan.notes)

    def test_a_hotspot_outside_keep_cannot_run(self):
        plan = compiled(layer("keep", span(5, 8)), layer("hotspot", selection_of(30)))
        assert not plan.ok
        assert any("outside Keep" in e for e in plan.errors)

    def test_fix_position_on_a_side_chain_adds_to_the_defaults_and_fixes_the_sequence(self):
        plan = compiled(
            layer("keep", span(5, 12)),
            layer("hotspot", selection_of(6)),
            layer("fix_position", selection_of((6, ("CB", "CG", "OH")))),
        )
        selects = [e["select"] for e in plan.conditions["C_CRD"]["true"]]
        assert len(selects) == 2 and "sel('A/*/6/[CB,CG,OH]')" in selects
        assert any("sequence is fixed too" in n for n in plan.notes)

    def test_the_translation_is_reported_with_the_old_wording(self):
        plan = compiled(layer("hotspot", selection_of(5)))
        assert any("Residue A21 in the viewer is A5" in n for n in plan.notes)


class TestMotifCompile:
    def test_pieces_are_joined_by_linkers(self):
        plan = compiled(layer("keep", selection_of(5, 6, 7, 20, 21)), mode=MODE_MOTIF)
        assert plan.contig == "A5-7, 10-30, A20-21"

    def test_the_layout_sets_the_linker_and_the_termini(self):
        layout = Layout(n_term=(15, 30), c_term=(5, 5), linkers=((8, 12),))
        plan = compiled(layer("keep", selection_of(5, 6, 20, 21)), mode=MODE_MOTIF, layout=layout)
        assert plan.contig == "15-30, A5-6, 8-12, A20-21, 5"

    def test_two_chains_make_two_runs(self):
        other = selection_of(30)
        for atom in other["atoms"]:
            atom["residue"]["chainId"] = atom["residue"]["authChainId"] = "A"
        plan = compiled(layer("keep", span(5, 6)), mode=MODE_MOTIF)
        assert "/0" not in plan.contig

    def test_a_motif_needs_a_keep(self):
        plan = compiled(layer("fix_position", selection_of(5)), mode=MODE_MOTIF)
        assert any("needs a Keep" in e for e in plan.errors)

    def test_keep_alone_holds_the_backbone_and_sequence_of_what_it_keeps(self):
        plan = compiled(layer("keep", span(5, 8)), mode=MODE_MOTIF)
        assert plan.conditions["C_SEQ"]["true"] == [{"select": "sel('A/*/[5-8]')"}]
        assert plan.conditions["C_CRD"]["true"][0]["select"].endswith("is_protein_backbone")

    def test_an_explicit_fix_position_replaces_the_default_in_a_motif(self):
        plan = compiled(
            layer("keep", span(20, 24)),
            layer("fix_position", selection_of(20, 21)),
            mode=MODE_MOTIF,
        )
        assert plan.conditions["C_CRD"]["true"] == [
            {"select": "sel('A/*/20')"},
            {"select": "sel('A/*/21')"},
        ]
        assert any("replaces the default" in n for n in plan.notes)

    def test_an_explicit_fix_sequence_replaces_the_default_too(self):
        plan = compiled(
            layer("keep", span(5, 8)), layer("fix_sequence", selection_of(5)), mode=MODE_MOTIF
        )
        assert plan.conditions["C_SEQ"]["true"] == [{"select": "sel('A/*/5')"}]

    def test_float_groups_become_unindexed_groups(self):
        plan = compiled(
            layer("keep", span(5, 8)),
            layer("float", selection_of(5, 6), "motif"),
            layer("float", selection_of(8), "tail"),
            mode=MODE_MOTIF,
        )
        assert plan.conditions["C_IDX"]["false"] == [
            {"select": "sel('A/*/[5-6]')", "group": "motif"},
            {"select": "sel('A/*/8')", "group": "tail"},
        ]


class TestChemistryCompile:
    def test_a_whole_aromatic_residue_is_sent_as_its_ring(self):
        phe = (5, ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"))
        plan = compiled(layer("hotspot", selection_of(8)), layer("pistack", selection_of(phe)))
        assert plan.ok, plan.errors
        assert plan.conditions["C_PISTACK"]["true"] == [
            {"select": "sel('A/*/5/[CG,CD1,CD2,CE1,CE2,CZ]')"}
        ]

    def test_a_pick_that_cannot_play_the_role_is_an_error_naming_the_residue(self):
        plan = compiled(layer("hotspot", selection_of(8)), layer("pistack", selection_of(20)))
        assert any("A36" in e and "cannot play this role" in e for e in plan.errors)

    def test_donor_and_acceptor_on_the_same_atoms_are_sent_as_both(self):
        serine = selection_of((8, ("OG",)))
        plan = compiled(
            layer("hotspot", selection_of(5)),
            layer("hbond", serine, "DONOR"),
            layer("hbond", serine, "ACCEPTOR"),
        )
        assert plan.conditions["C_HB"]["BOTH"] == [{"select": "sel('A/*/8/OG')"}]
        assert "DONOR" not in plan.conditions["C_HB"]
        assert any("BOTH" in n for n in plan.notes)

    def test_distinct_enum_values_stay_distinct(self):
        plan = compiled(
            layer("hotspot", selection_of(5)),
            layer("rasa", selection_of(6, 7), "BURIED"),
            layer("rasa", selection_of(20), "EXPOSED"),
        )
        assert plan.conditions["C_RAS"]["BURIED"] == [{"select": "sel('A/*/[6-7]')"}]
        assert plan.conditions["C_RAS"]["EXPOSED"] == [{"select": "sel('A/*/20')"}]

    def test_secondary_structure_uses_the_nims_enum_names(self):
        plan = compiled(layer("hotspot", selection_of(5)), layer("secondary_structure", span(10, 14), "HELIX"))
        assert "HELIX" in plan.conditions["C_SEC"]

    def test_hydrophobic_contacts_are_narrowed_to_side_chain_carbons(self):
        plan = compiled(layer("hotspot", selection_of(5)), layer("hydrophobic", selection_of((12, ("N", "CB", "CG")))))
        assert plan.conditions["C_HPHOB"]["true"] == [{"select": "sel('A/*/12/[CB,CG]')"}]

    def test_a_donor_pick_is_narrowed_to_the_atoms_that_donate(self):
        serine = selection_of((8, ("N", "CA", "OG")))
        plan = compiled(layer("hotspot", selection_of(5)), layer("hbond", serine, "DONOR"))
        assert plan.ok, plan.errors
        assert plan.conditions["C_HB"]["DONOR"] == [{"select": "sel('A/*/8/[N,OG]')"}]

    def test_a_salt_bridge_is_sent_with_its_charge(self):
        plan = compiled(
            layer("hotspot", selection_of(5)),
            layer("salt", selection_of((9, ("CG", "OD1", "OD2"))), "NEGATIVE"),
            layer("salt", selection_of((10, ("NZ",))), "POSITIVE"),
        )
        assert plan.ok, plan.errors
        assert plan.conditions["C_SALT"]["NEGATIVE"] == [{"select": "sel('A/*/9/[OD1,OD2]')"}]
        assert plan.conditions["C_SALT"]["POSITIVE"] == [{"select": "sel('A/*/10/NZ')"}]

    def test_a_charge_the_residue_does_not_have_is_an_error(self):
        plan = compiled(
            layer("hotspot", selection_of(5)),
            layer("salt", selection_of((9, ("CG", "OD1", "OD2"))), "POSITIVE"),
        )
        assert any("A25" in e for e in plan.errors)

    def test_a_cation_pi_contact_is_sent_with_its_side(self):
        plan = compiled(
            layer("hotspot", selection_of(5)),
            layer("pication", selection_of((10, ("NZ",))), "CATION"),
        )
        assert plan.ok, plan.errors
        assert plan.conditions["C_PICATION"]["CATION"] == [{"select": "sel('A/*/10/NZ')"}]

    def test_every_chemistry_role_sits_beside_the_hotspot_without_replacing_it(self):
        plan = compiled(
            layer("hotspot", selection_of(5)),
            layer("rasa", selection_of(6), "BURIED"),
            layer("secondary_structure", span(10, 12), "LOOP"),
        )
        assert "C_HOT" in plan.conditions and "C_RAS" in plan.conditions and "C_SEC" in plan.conditions


# --- Compiling for RFdiffusion v1 ------------------------------------------


class TestRFdiffusion1Compile:
    def test_a_binder_contig_uses_author_numbering(self):
        plan = compile_rfdiffusion1(
            [layer("keep", span(5, 12)), layer("hotspot", selection_of(6, 7))],
            binder_min=50,
            binder_max=80,
        )
        assert plan.ok
        assert plan.contigs == "A21-28/0 50-80"
        assert plan.hotspot_res == ["A22", "A23"]

    def test_hotspot_atoms_are_widened_and_said_so(self):
        plan = compile_rfdiffusion1(
            [layer("keep", span(5, 12)), layer("hotspot", selection_of((6, ("CB",))))]
        )
        assert plan.hotspot_res == ["A22"]
        assert any("widened to whole residues" in n for n in plan.notes)

    def test_a_motif_contig_is_slash_joined(self):
        layout = Layout(n_term=(10, 20), c_term=(10, 20), linkers=((5, 15),))
        plan = compile_rfdiffusion1(
            [layer("keep", selection_of(5, 6, 20, 21))], mode=MODE_MOTIF, layout=layout
        )
        assert plan.contigs == "10-20/A21-22/5-15/A36-37/10-20"

    def test_roles_it_cannot_take_are_reported_dropped(self):
        plan = compile_rfdiffusion1(
            [
                layer("keep", span(5, 12)),
                layer("hotspot", selection_of(6)),
                layer("float", selection_of(7)),
                layer("pistack", selection_of(5)),
            ]
        )
        assert plan.ok
        dropped = " ".join(plan.warnings)
        assert "Float was dropped" in dropped and "Pi-stack was dropped" in dropped

    def test_fix_sequence_becomes_proteinmpnn_fixed_positions(self):
        plan = compile_rfdiffusion1(
            [
                layer("keep", span(5, 12)),
                layer("hotspot", selection_of(6)),
                layer("fix_sequence", selection_of(5, 6, 7, 10)),
            ]
        )
        assert plan.fixed_positions == ["A:21-23", "A:26"]

    def test_it_needs_a_keep_because_its_contig_names_residues(self):
        plan = compile_rfdiffusion1([layer("hotspot", selection_of(6))])
        assert any("needs a Keep" in e for e in plan.errors)


class TestFrontendFixture:
    """The viewer's tests run against a copy of the catalog; keep it honest."""

    def test_the_fixture_matches_the_registry(self):
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[3] / "frontend/tests/design-roles.fixture.json"
        if not path.exists():
            pytest.skip("frontend sources are not in this container")
        assert json.loads(path.read_text()) == json.loads(json.dumps(registry_payload())), (
            "regenerate with: python3 -c \"import json,sys; sys.path.insert(0,'backend'); "
            "from api.selections.roles import registry_payload; "
            "print(json.dumps(registry_payload(), indent=2, sort_keys=True))\" "
            "> frontend/tests/design-roles.fixture.json"
        )
