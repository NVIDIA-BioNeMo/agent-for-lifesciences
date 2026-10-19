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

"""Building selections from a stored structure, with no viewer.

The fixture has author numbering running 16 ahead of label numbering on chain
A (as in 5O45), a second chain whose author id differs from its label id, a
water, and a trailing loop whose rows are as wide as the atom table. That last one once read as atoms.
"""

import pytest

from api.selections.ops import (
    SelectionError,
    combine,
    expand,
    parse_residue_spec,
    select_residues,
)
from api.selections.residues import residues_from_atoms
from api.selections.structure import parse_structure_atoms

CIF = """\
data_T
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
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
ATOM   1 N N  . ASN A 1 1   ? 0.000 0.000 0.000 17  ASN A N
ATOM   2 C CA . ASN A 1 1   ? 1.000 0.000 0.000 17  ASN A CA
ATOM   3 N N  . TYR A 1 40  ? 10.000 0.000 0.000 56  TYR A N
ATOM   4 C CA . TYR A 1 40  ? 11.000 0.000 0.000 56  TYR A CA
ATOM   5 C CG . TYR A 1 40  ? 12.000 0.000 0.000 56  TYR A CG
ATOM   6 C CA . MET A 1 99  ? 20.000 0.000 0.000 115 MET A CA
ATOM   7 C CA . SER A 1 100 ? 30.000 0.000 0.000 116 SER A CA
ATOM   8 C CA . THR A 1 101 ? 31.000 0.000 0.000 117 THR A CA
ATOM   9 C CA . GLY B 2 1   ? 11.500 4.000 0.000 5   GLY C CA
HETATM 10 O O . HOH A 3 .   ? 11.000 1.000 0.000 301 HOH A O
#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.pdbx_ptnr1_PDB_ins_code
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.pdbx_ptnr2_PDB_ins_code
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_atom_id
disulf1 disulf A CYS 5 ? 9 A CYS 1 ? 4
#
"""


@pytest.fixture(scope="module")
def structure():
    return parse_structure_atoms(CIF)


def payloads(result):
    return [a.to_payload() for a in result.atoms]


class TestParsing:
    def test_only_the_atom_table_is_read(self, structure):
        # The _struct_conn row has the same width as the atom table.
        assert len(structure) == 10
        assert all(a.name for a in structure)

    def test_both_numberings_survive(self, structure):
        tyr = next(a for a in structure if a.comp_id == "TYR" and a.name == "CA")
        assert (tyr.label_seq, tyr.auth_seq) == (40, 56)
        assert (tyr.label_chain, tyr.auth_chain) == ("A", "A")

    def test_chain_ids_can_differ_between_systems(self, structure):
        gly = next(a for a in structure if a.comp_id == "GLY")
        assert (gly.label_chain, gly.auth_chain) == ("B", "C")

    def test_water_is_typed(self, structure):
        assert [a.entity_type for a in structure].count("water") == 1

    def test_pdb_input_is_read_too(self):
        pdb = (
            "ATOM      1  CA  TYR A  56      11.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  MET A 115      20.000   0.000   0.000  1.00  0.00           C\n"
        )
        atoms = parse_structure_atoms(pdb)
        assert [(a.auth_chain, a.auth_seq) for a in atoms] == [("A", 56), ("A", 115)]

    def test_garbage_yields_nothing_rather_than_raising(self):
        assert parse_structure_atoms("not a structure") == []


class TestResidueSpec:
    def test_numbers_ranges_and_insertion_codes(self):
        spans = parse_residue_spec([56, "100-120", "52A", " 7 "])
        assert [str(s) for s in spans] == ["56", "100-120", "52A", "7"]

    @pytest.mark.parametrize("bad", ["abc", "5-", "1..3", "", None, True])
    def test_unreadable_residue_is_named(self, bad):
        with pytest.raises(SelectionError):
            parse_residue_spec([bad])

    def test_a_backwards_range_is_refused(self):
        with pytest.raises(SelectionError, match="backwards"):
            parse_residue_spec(["120-100"])

    def test_empty_is_refused(self):
        with pytest.raises(SelectionError):
            parse_residue_spec([])


class TestSelectResidues:
    def test_author_numbering_is_the_default_meaning_of_a_number(self, structure):
        found = select_residues(structure, "A", parse_residue_spec([56]))
        assert {(a.auth_seq, a.label_seq) for a in found.atoms} == {(56, 40)}
        assert found.residue_count == 1
        assert found.missing == []

    def test_label_numbering_reaches_the_same_residue(self, structure):
        found = select_residues(structure, "A", parse_residue_spec([40]), numbering="label")
        assert {a.auth_seq for a in found.atoms} == {56}

    def test_the_same_number_in_the_wrong_system_is_a_different_residue(self, structure):
        found = select_residues(structure, "A", parse_residue_spec([40]))
        assert found.atoms == []
        assert found.missing == ["40"]

    def test_a_range_takes_every_residue_in_it(self, structure):
        found = select_residues(structure, "A", parse_residue_spec(["115-117"]))
        assert found.residue_count == 3

    def test_atom_names_filter_each_residue(self, structure):
        found = select_residues(structure, "A", parse_residue_spec([56]), atom_names=["ca"])
        assert [a.name for a in found.atoms] == ["CA"]

    def test_what_was_not_found_is_reported_alongside_what_was(self, structure):
        found = select_residues(structure, "A", parse_residue_spec([56, 9999]))
        assert found.residue_count == 1
        assert found.missing == ["9999"]

    def test_chain_ids_follow_the_numbering_system(self, structure):
        assert select_residues(structure, "C", parse_residue_spec([5])).residue_count == 1
        assert select_residues(
            structure, "B", parse_residue_spec([1]), numbering="label"
        ).residue_count == 1

    def test_an_unknown_chain_lists_the_known_ones(self, structure):
        with pytest.raises(SelectionError, match="A, C"):
            select_residues(structure, "Z", parse_residue_spec([1]))

    def test_water_is_never_selected_by_number(self, structure):
        assert select_residues(structure, "A", parse_residue_spec([301])).atoms == []

    def test_unknown_numbering_is_refused(self, structure):
        with pytest.raises(SelectionError):
            select_residues(structure, "A", parse_residue_spec([1]), numbering="pdb")


class TestSetAlgebra:
    @pytest.fixture
    def two(self, structure):
        first = payloads(select_residues(structure, "A", parse_residue_spec([56, 115])))
        second = payloads(select_residues(structure, "A", parse_residue_spec([115, 117])))
        return first, second

    @staticmethod
    def numbers(atoms):
        return sorted({r.auth_seq for r in residues_from_atoms(atoms)})

    def test_union(self, two):
        assert self.numbers(combine("union", *two)) == [56, 115, 117]

    def test_union_does_not_duplicate_atoms(self, two):
        merged = combine("union", *two)
        keys = [(a["residue"]["seqNumber"], a["name"]) for a in merged]
        assert len(keys) == len(set(keys))

    def test_intersect(self, two):
        assert self.numbers(combine("intersect", *two)) == [115]

    def test_subtract(self, two):
        assert self.numbers(combine("subtract", *two)) == [56]

    def test_an_unknown_operation_is_refused(self, two):
        with pytest.raises(SelectionError):
            combine("xor", *two)

    def test_subtracting_a_whole_atom_filter_still_removes_the_residue(self, structure):
        whole = payloads(select_residues(structure, "A", parse_residue_spec([56])))
        ca_only = payloads(
            select_residues(structure, "A", parse_residue_spec([56]), atom_names=["CA"])
        )
        assert combine("subtract", whole, ca_only) == []


class TestExpand:
    def seed(self, structure):
        return payloads(select_residues(structure, "A", parse_residue_spec([56])))

    def test_takes_whole_residues_with_any_atom_in_reach(self, structure):
        grown = expand(self.seed(structure), structure, 5.0)
        # Residue 17 sits 10 A away; chain C's glycine is ~3.5 A from the Tyr CA.
        assert sorted({(r.auth_chain, r.auth_seq) for r in residues_from_atoms(grown)}) == [
            ("A", 56),
            ("C", 5),
        ]

    def test_whole_residues_pull_in_atoms_beyond_the_radius(self, structure):
        grown = expand(self.seed(structure), structure, 11.0)
        seqs = {r.auth_seq for r in residues_from_atoms(grown)}
        assert {17, 56}.issubset(seqs)
        assert len([a for a in grown if a["residue"]["authSeqNumber"] == 17]) == 2

    def test_atom_precision_when_whole_residues_are_off(self, structure):
        grown = expand(self.seed(structure), structure, 9.5, whole_residues=False)
        assert len([a for a in grown if a["residue"]["authSeqNumber"] == 17]) == 1

    def test_water_is_never_pulled_in(self, structure):
        grown = expand(self.seed(structure), structure, 30.0)
        assert all(a["residue"]["name"] != "HOH" for a in grown)

    def test_the_original_can_be_left_out(self, structure):
        grown = expand(self.seed(structure), structure, 5.0, include_original=False)
        assert {r.auth_seq for r in residues_from_atoms(grown)} == {5}

    @pytest.mark.parametrize("radius", [0, -1, 31])
    def test_radius_is_bounded(self, structure, radius):
        with pytest.raises(SelectionError):
            expand(self.seed(structure), structure, radius)

    def test_an_empty_selection_has_nothing_to_measure_from(self, structure):
        with pytest.raises(SelectionError):
            expand([], structure, 5.0)
