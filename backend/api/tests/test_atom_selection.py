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

"""Atom-level selections: per-residue atoms, keywords, atom-level set algebra.

The viewer lets a user pick one atom of a residue, so the backend has to say
which atoms a selection holds, build such selections by number, and combine
them without rounding them back up to whole residues. Whole-residue behaviour
must not change.
"""

import json

import pytest

from api.selections.ops import (
    SelectionError,
    combine,
    parse_residue_spec,
    select_residues,
)
from api.selections.residues import (
    STANDARD_HEAVY_ATOMS,
    ResidueRef,
    format_residue_listing,
    residues_from_atoms,
)
from api.selections.structure import parse_structure_atoms
from api.tools.rfd4_proteina import build_hotspot_spec
from api.storage.mmcif_numbering import build_numbering_map

# Heavy atoms of the residues used below, in PDB order.
ATOMS = {
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "GLY": ["N", "CA", "C", "O"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}


def make_cif(residues):
    """mmCIF for chain A: ``residues`` is [(comp, label_seq)], author = label + 16."""
    rows = []
    n = 0
    for comp, seq in residues:
        for name in ATOMS[comp]:
            n += 1
            rows.append(
                f"ATOM {n} {name[0]} {name} . {comp} A 1 {seq} ? {n}.0 0.0 0.0 {seq + 16} {comp} A {name}"
            )
    return (
        "data_T\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n_atom_site.label_entity_id\n_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "_atom_site.auth_seq_id\n_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n" + "\n".join(rows) + "\n#\n"
    )


CIF = make_cif([("TYR", 40), ("MET", 99), ("GLY", 100), ("TRP", 101)])


@pytest.fixture
def structure():
    return parse_structure_atoms(CIF)


def names_by_residue(atoms):
    out = {}
    for a in atoms:
        out.setdefault(a.auth_seq, []).append(a.name)
    return out


class TestStandardTable:
    def test_matches_the_fixture_residues(self):
        for comp, names in ATOMS.items():
            assert STANDARD_HEAVY_ATOMS[comp] == len(names)


class TestPartialListing:
    def payload(self, structure, **kw):
        found = select_residues(structure, "A", parse_residue_spec(kw.pop("items")), **kw)
        return [a.to_payload() for a in found.atoms]

    def test_a_whole_residue_has_no_atom_names(self, structure):
        (ref,) = residues_from_atoms(self.payload(structure, items=[56]))
        assert not ref.partial
        assert ref.token == "Y56"

    def test_chosen_atoms_are_named(self, structure):
        (ref,) = residues_from_atoms(self.payload(structure, items=["56:OH,CZ"]))
        assert ref.partial
        assert ref.atoms == ("CZ", "OH")  # structure order, not request order
        assert ref.token == "Y56[CZ,OH]"

    def test_a_long_atom_list_is_counted(self, structure):
        (ref,) = residues_from_atoms(self.payload(structure, items=["56:sidechain"]))
        assert ref.token == "Y56[CB,CG,CD1,CD2,+4]"

    def test_a_partial_residue_breaks_a_run(self, structure):
        # 56 whole, 115 partial, 116 whole (Gly), 117 whole (Trp): the partial
        # one stands alone and the whole neighbours still form a run.
        atoms = self.payload(structure, items=[56, "115:SD", 116, 117])
        listing = format_residue_listing(residues_from_atoms(atoms))
        assert listing == "A: Y56 M115[SD] G116-W117 (2 res)"

    def test_survives_a_dict_round_trip(self, structure):
        (ref,) = residues_from_atoms(self.payload(structure, items=["56:OH"]))
        assert ResidueRef.from_dict(ref.to_dict()) == ref

    def test_a_non_standard_residue_reads_as_whole(self):
        ref = ResidueRef("A", None, "A", 301, comp_id="HEM", entity_type="non-polymer", atoms=("FE",))
        assert not ref.partial


class TestSpecParsing:
    def test_atoms_follow_a_colon(self):
        (span,) = parse_residue_spec(["56:OH,CZ"])
        assert (span.start, span.end, span.atoms) == (56, 56, ("OH", "CZ"))

    def test_a_range_can_take_a_keyword(self):
        (span,) = parse_residue_spec(["100-110:sidechain"])
        assert (span.start, span.end, span.atoms) == (100, 110, ("SIDECHAIN",))

    def test_insertion_code_and_atoms(self):
        (span,) = parse_residue_spec(["52A:CA"])
        assert (span.start, span.ins_code, span.atoms) == (52, "A", ("CA",))

    def test_plain_numbers_are_unchanged(self):
        assert parse_residue_spec([56])[0].atoms is None

    def test_a_colon_with_no_atoms_is_an_error(self):
        with pytest.raises(SelectionError, match="names no atoms"):
            parse_residue_spec(["56:"])

    def test_text_round_trips(self):
        assert str(parse_residue_spec(["56:OH,CZ"])[0]) == "56:OH,CZ"


class TestSelectingAtoms:
    def pick(self, structure, items, **kw):
        return select_residues(structure, "A", parse_residue_spec(items), **kw)

    def test_per_residue_atoms(self, structure):
        found = self.pick(structure, ["56:OH,CZ", "115:SD"])
        assert names_by_residue(found.atoms) == {56: ["CZ", "OH"], 115: ["SD"]}
        assert found.residue_count == 2
        assert found.atom_level

    def test_whole_residues_are_not_atom_level(self, structure):
        found = self.pick(structure, [56, "115"])
        assert not found.atom_level
        assert len(found.atoms) == 12 + 8

    def test_backbone_keyword(self, structure):
        found = self.pick(structure, [56, 115], atom_names="backbone")
        assert names_by_residue(found.atoms) == {56: ["N", "CA", "C", "O"], 115: ["N", "CA", "C", "O"]}
        assert found.atom_level

    def test_sidechain_keyword_and_glycine_has_none(self, structure):
        found = self.pick(structure, [115, 116], atom_names="sidechain")
        assert names_by_residue(found.atoms) == {115: ["CB", "CG", "SD", "CE"]}

    def test_a_list_of_names_still_works(self, structure):
        found = self.pick(structure, [56, 115], atom_names=["CA"])
        assert names_by_residue(found.atoms) == {56: ["CA"], 115: ["CA"]}

    def test_a_residues_own_atoms_beat_the_call_wide_ones(self, structure):
        found = self.pick(structure, ["56:OH", 115], atom_names=["CA"])
        assert names_by_residue(found.atoms) == {56: ["OH"], 115: ["CA"]}

    def test_an_atom_the_residue_lacks_is_reported(self, structure):
        found = self.pick(structure, ["56:OH,NZ"])
        assert names_by_residue(found.atoms) == {56: ["OH"]}
        assert found.missing == ["56:NZ"]

    def test_only_missing_atoms_leaves_nothing(self, structure):
        found = self.pick(structure, ["56:NZ"])
        assert found.atoms == []
        assert found.missing == ["56:NZ"]

    def test_label_numbering(self, structure):
        found = self.pick(structure, ["40:OH"], numbering="label")
        assert names_by_residue(found.atoms) == {56: ["OH"]}


class TestAtomLevelAlgebra:
    def sel(self, structure, items):
        found = select_residues(structure, "A", parse_residue_spec(items))
        return [a.to_payload() for a in found.atoms]

    def test_subtract_removes_only_the_named_atoms(self, structure):
        whole = self.sel(structure, [56])
        oh = self.sel(structure, ["56:OH"])
        rest = combine("subtract", whole, oh, atom_level=True)
        assert len(rest) == 11
        assert "OH" not in {a["name"] for a in rest}

    def test_subtract_by_residue_drops_the_residue(self, structure):
        whole = self.sel(structure, [56, 115])
        oh = self.sel(structure, ["56:OH"])
        assert {a["residue"]["authSeqNumber"] for a in combine("subtract", whole, oh)} == {115}

    def test_intersect_keeps_the_shared_atoms(self, structure):
        first = self.sel(structure, ["56:OH,CZ"])
        second = self.sel(structure, ["56:CZ,CE1"])
        shared = combine("intersect", first, second, atom_level=True)
        assert [a["name"] for a in shared] == ["CZ"]

    def test_intersect_by_residue_keeps_all_of_the_firsts_atoms(self, structure):
        first = self.sel(structure, [56])
        second = self.sel(structure, ["56:CZ"])
        assert len(combine("intersect", first, second)) == 12

    def test_union_dedups_atoms(self, structure):
        a = self.sel(structure, ["56:OH,CZ"])
        b = self.sel(structure, ["56:CZ,CE1"])
        merged = combine("union", a, b, atom_level=True)
        assert sorted(x["name"] for x in merged) == ["CE1", "CZ", "OH"]


class TestHotspotFromAtoms:
    def spec(self, items, structure=None):
        atoms = parse_structure_atoms(CIF)
        found = select_residues(atoms, "A", parse_residue_spec(items))
        selection = {"atoms": [a.to_payload() for a in found.atoms]}
        return build_hotspot_spec(selection, build_numbering_map(CIF))

    def test_chosen_atoms_are_named_in_label_numbering(self):
        spec = self.spec(["56:CZ,OH", "115:SD"])
        assert spec.selectors == ["sel('A/*/40/[CZ,OH]')", "sel('A/*/99/SD')"]
        assert spec.notes[0].startswith("Residue A56")  # translated, not dropped

    def test_a_whole_small_residue_lists_every_atom(self):
        spec = self.spec([116])
        assert spec.selectors == ["sel('A/*/100/[N,CA,C,O]')"]

    def test_more_than_eight_chosen_atoms_becomes_the_whole_residue_and_says_so(self):
        spec = self.spec(["117:N,CA,C,O,CB,CG,CD1,CD2,NE1"])
        assert spec.selectors == ["sel('A/*/101')"]
        assert any("9 atoms were chosen" in n and "whole residue" in n for n in spec.notes)

    def test_a_truly_whole_big_residue_gets_no_such_note(self):
        spec = self.spec([117])
        assert spec.selectors == ["sel('A/*/101')"]
        assert not any("atoms were chosen" in n for n in spec.notes)
