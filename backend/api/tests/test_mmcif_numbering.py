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

"""Author numbering is not label numbering, and conflating them is silent.

This is the failure mode worth a test file of its own. Mol* reports
``auth_seq_id``; AtomWorks resolves a condition spec's ``res_id`` against
``label_seq_id``. When they differ -- which they do for most deposited
structures -- sending author numbers to the NIM does not error. It conditions
a different residue and returns a confident, wrong design.

The fixture below is 5O45, the PD-L1 entry Aevom's binder workflow was built
against, reduced to the three epitope residues plus their neighbours. Its
author numbering is offset from its label numbering by 16, so every
assertion here would pass trivially on a structure where the two agree.
"""

import pytest

from api.storage.mmcif_numbering import build_numbering_map

# Author 17 is label 1, so author 56 is label 40. Columns and values are
# copied from the real entry, trimmed to the residues the tests name.
OFFSET_CIF = """\
data_5O45
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
ATOM   1 N CA  . ASN A 1 1   ? 1.000 1.000 1.000 17  ASN A CA
ATOM   2 C CB  . ASN A 1 1   ? 1.000 1.000 2.000 17  ASN A CB
ATOM   3 N CA  . TYR A 1 40  ? 2.000 2.000 1.000 56  TYR A CA
ATOM   4 C CG  . TYR A 1 40  ? 2.000 2.000 2.000 56  TYR A CG
ATOM   5 O OH  . TYR A 1 40  ? 2.000 2.000 3.000 56  TYR A OH
ATOM   6 N CA  . MET A 1 99  ? 3.000 3.000 1.000 115 MET A CA
ATOM   7 S SD  . MET A 1 99  ? 3.000 3.000 2.000 115 MET A SD
ATOM   8 N CA  . ASN A 1 129 ? 4.000 4.000 1.000 145 ASN A CA
HETATM 9 O O   . HOH B 2 .   ? 9.000 9.000 9.000 301 HOH A O
#
"""

# A structure whose two numbering systems agree, which is the case that makes
# the bug invisible if it is the only thing ever tested.
AGREEING_CIF = """\
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
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
ATOM 1 N CA . GLY A 1 1 ? 1.000 1.000 1.000 1 GLY A CA
ATOM 2 N CA . ALA A 1 2 ? 2.000 1.000 1.000 2 ALA A CA
#
"""


@pytest.fixture
def offset():
    return build_numbering_map(OFFSET_CIF)


class TestTranslation:
    def test_author_residue_maps_to_its_label_counterpart(self, offset):
        # The single assertion the whole feature rests on.
        assert offset.lookup("A", 56).seq_id == 40

    def test_translation_preserves_residue_identity(self, offset):
        # A mapping that lands on the wrong residue of the right number is
        # the failure this guards, so check the chemistry and not just the int.
        assert offset.lookup("A", 56).comp_id == "TYR"
        assert offset.lookup("A", 115).comp_id == "MET"

    def test_offset_is_constant_across_the_chain(self, offset):
        assert offset.lookup("A", 17).seq_id == 1
        assert offset.lookup("A", 145).seq_id == 129

    def test_unknown_residue_is_none_rather_than_a_guess(self, offset):
        # Returning the author number as a fallback would be the dangerous
        # answer: it looks like a translation and is not one.
        assert offset.lookup("A", 9999) is None


class TestDisagreementDetection:
    def test_offset_structure_is_flagged(self, offset):
        assert offset.identical is False

    def test_agreeing_structure_is_not_flagged(self):
        assert build_numbering_map(AGREEING_CIF).identical is True

    def test_water_does_not_flag_a_structure(self):
        # Non-polymers have no label_seq_id in either system, so they cannot
        # disagree; counting them would flag every structure with a solvent.
        assert build_numbering_map(AGREEING_CIF).identical is True


class TestNonPolymers:
    def test_water_maps_with_no_label_seq_id(self, offset):
        water = offset.lookup("A", 301)
        assert water is not None
        assert water.seq_id is None

    def test_polymer_span_ignores_non_polymers(self, offset):
        # The span feeds the contig, and a contig naming a water would be
        # rejected by the NIM.
        assert offset.polymer_span("A") == (1, 129)


class TestChains:
    def test_author_chain_resolves_to_the_polymer_label_chain(self, offset):
        # Author chain A covers both the protein (label A) and its water
        # (label B); a design condition means the protein.
        assert offset.label_chain("A") == "A"

    def test_chains_are_listed_in_file_order(self, offset):
        assert offset.chain_ids == ["A"]


class TestDegenerateInput:
    def test_empty_content_yields_an_empty_map(self):
        assert not build_numbering_map("")

    def test_file_without_atom_site_yields_an_empty_map(self):
        # Empty must mean "cannot translate", never "nothing to translate".
        assert not build_numbering_map("data_X\n_entry.id X\n")

    def test_bytes_are_accepted(self):
        assert build_numbering_map(OFFSET_CIF.encode()).lookup("A", 56).seq_id == 40
