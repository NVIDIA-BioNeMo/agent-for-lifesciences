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

"""Tests for the curated linker library.

The spans are measured rather than asserted, so the tests that matter are the
ones where geometry is known independently: para-phenylene holds its anchors
1.39 x 2 A across the ring plus two 1.5 A bonds, which is 5.8 A and nothing else,
and a bicyclo[1.1.1]pentane's 1.85 A bridgeheads put it near 4.9. If the
measurement machinery breaks, those numbers move.
"""

import pytest
from rdkit import Chem

from app import linker_library as lib


class TestLibraryDefinition:
    def test_every_entry_parses_and_carries_its_attachment_points(self):
        for entry in lib.LINKER_LIBRARY:
            mol = Chem.MolFromSmiles(entry.smiles)
            assert mol is not None, f"{entry.name} does not parse"
            dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
            assert len(dummies) == 2, f"{entry.name} needs exactly two attachment points"
            assert {atom.GetAtomMapNum() for atom in dummies} == {
                lib.ATTACH_A,
                lib.ATTACH_B,
            }, f"{entry.name} has the wrong attachment numbering"

    def test_names_are_unique(self):
        names = [entry.name for entry in lib.LINKER_LIBRARY]
        assert len(names) == len(set(names))

    def test_only_the_zero_atom_entry_is_a_direct_bond(self):
        direct = [entry.name for entry in lib.LINKER_LIBRARY if entry.is_direct_bond]
        assert direct == ["direct_bond"]


class TestMeasurement:
    def test_every_entry_measures(self):
        geometries = lib.measure_all()
        missing = [
            entry.name for entry in lib.LINKER_LIBRARY if entry.name not in geometries
        ]
        assert missing == [], f"unmeasurable entries: {missing}"

    @pytest.mark.parametrize(
        "name,expected,tolerance",
        [
            # Two 1.5 A bonds plus the para distance across a benzene ring (2.79 A).
            ("phenyl_1_4", 5.79, 0.15),
            # Same construction across a meta pair (2.42 A).
            ("phenyl_1_3", 5.01, 0.15),
            # Bicyclo[1.1.1]pentane bridgeheads sit 1.85 A apart.
            ("bicyclopentane_1_3", 4.99, 0.2),
            # Two 1.43 A C-C bonds and a 1.20 A triple bond, collinear.
            ("alkyne", 4.06, 0.15),
            ("direct_bond", 1.47, 0.1),
        ],
    )
    def test_rigid_spans_match_hand_geometry(self, name, expected, tolerance):
        geometry = lib.measure(lib.LINKERS_BY_NAME[name])
        assert geometry is not None
        assert abs(geometry.span_min - expected) < tolerance
        # Rigid means the range does not open up across conformers.
        assert geometry.span_max - geometry.span_min < 0.5

    def test_flexible_entries_report_a_range(self):
        geometry = lib.measure(lib.LINKERS_BY_NAME["butylene"])
        assert geometry is not None
        assert geometry.span_max - geometry.span_min > 1.0

    def test_a_rigid_rod_spans_further_than_a_chain_of_the_same_size(self):
        alkyne = lib.measure(lib.LINKERS_BY_NAME["alkyne"])
        ethylene = lib.measure(lib.LINKERS_BY_NAME["ethylene"])
        assert alkyne.num_linker_atoms == ethylene.num_linker_atoms
        # The whole argument for rigid bridges, in two numbers: further reach at
        # no rotatable-bond cost.
        assert alkyne.rb_cost == 0
        assert ethylene.rb_cost > alkyne.rb_cost
        assert alkyne.span_min > ethylene.span_max

    def test_ring_bearing_entries_carry_more_atoms_than_they_span(self):
        phenylene = lib.measure(lib.LINKERS_BY_NAME["phenyl_1_4"])
        assert phenylene.path_atoms == 4
        assert phenylene.num_linker_atoms == 6
        chain = lib.measure(lib.LINKERS_BY_NAME["propylene"])
        assert chain.path_atoms == chain.num_linker_atoms

    def test_measurement_is_cached(self):
        entry = lib.LINKERS_BY_NAME["amide"]
        first = lib.measure(entry)
        assert lib.measure(entry) is first


class TestMatching:
    def test_matches_bracket_the_requested_distance(self):
        for distance in (2.5, 4.0, 6.0, 8.0, 10.0):
            matches = lib.matching_linkers(distance)
            assert matches, f"nothing spans {distance} A"
            for geometry in matches:
                assert geometry.span_min - 0.8 <= distance <= geometry.span_max + 0.8

    def test_rotatable_bond_headroom_excludes_expensive_bridges(self):
        with_headroom = lib.matching_linkers(6.0, rb_headroom=2)
        assert with_headroom
        assert all(geometry.rb_cost <= 2 for geometry in with_headroom)

    def test_rigid_only_drops_the_chains(self):
        rigid = {geometry.name for geometry in lib.matching_linkers(6.0, rigid_only=True)}
        assert "butylene" not in rigid
        assert "peg2" not in rigid

    def test_results_are_ordered_by_cost(self):
        costs = [geometry.rb_cost for geometry in lib.matching_linkers(6.0, one_per_family=False)]
        assert costs == sorted(costs)

    def test_nothing_spans_an_impossible_gap(self):
        assert lib.matching_linkers(30.0) == []


class TestAssembly:
    PHENOL = Chem.MolFromSmiles("Oc1ccccc1")
    PYRIDINE = Chem.MolFromSmiles("c1ccncc1")

    def test_joins_two_fragments_through_a_linker(self):
        smiles, error = lib.assemble(self.PHENOL, 4, self.PYRIDINE, 0, "[*:1]C(=O)N[*:2]")
        assert error is None
        assert smiles == Chem.CanonSmiles("O=C(Nc1ccncc1)c1ccc(O)cc1")

    def test_direct_bond_joins_the_fragments_to_each_other(self):
        smiles, error = lib.assemble(
            self.PHENOL, 4, self.PYRIDINE, 0, lib.LINKERS_BY_NAME["direct_bond"].smiles
        )
        assert error is None
        assert smiles == Chem.CanonSmiles("Oc1ccc(-c2ccncc2)cc1")

    def test_explicit_hydrogens_behave_the_same_as_implicit(self):
        with_hs, error_a = lib.assemble(
            Chem.AddHs(self.PHENOL), 4, self.PYRIDINE, 0, "[*:1]C#C[*:2]"
        )
        without_hs, error_b = lib.assemble(self.PHENOL, 4, self.PYRIDINE, 0, "[*:1]C#C[*:2]")
        assert error_a is None and error_b is None
        assert with_hs == without_hs

    def test_every_library_entry_assembles(self):
        for entry in lib.LINKER_LIBRARY:
            smiles, error = lib.assemble(self.PHENOL, 4, self.PYRIDINE, 0, entry.smiles)
            assert error is None, f"{entry.name}: {error}"
            assert Chem.MolFromSmiles(smiles) is not None

    def test_assembled_molecule_keeps_both_fragments_as_substructures(self):
        # This is SMDD-Bench's gate 2, checked at the point of construction.
        smiles, _ = lib.assemble(self.PHENOL, 4, self.PYRIDINE, 0, "[*:1]c1ccc([*:2])cc1")
        product = Chem.MolFromSmiles(smiles)
        assert product.HasSubstructMatch(self.PHENOL)
        assert product.HasSubstructMatch(self.PYRIDINE)

    def test_out_of_range_site_is_reported_not_raised(self):
        smiles, error = lib.assemble(self.PHENOL, 99, self.PYRIDINE, 0, "[*:1]C[*:2]")
        assert smiles is None
        assert "out of range" in error

    def test_site_without_a_hydrogen_is_rejected(self):
        # The 1-position of phenol carries the hydroxyl, not a hydrogen.
        smiles, error = lib.assemble(self.PHENOL, 1, self.PYRIDINE, 0, "[*:1]C[*:2]")
        assert smiles is None
        assert "hydrogen" in error

    def test_unpaired_attachment_point_is_rejected(self):
        smiles, error = lib.assemble(self.PHENOL, 4, None, None, "[*:1]C(=O)N[*:2]")
        assert smiles is None
        assert "unpaired" in error

    def test_unparseable_linker_is_reported(self):
        smiles, error = lib.assemble(self.PHENOL, 4, self.PYRIDINE, 0, "not a smiles")
        assert smiles is None
        assert "unparseable" in error


class TestMarkAttachment:
    def test_marks_an_implicit_hydrogen_position(self):
        marked, error = lib.mark_attachment(Chem.MolFromSmiles("c1ccccc1"), 0, 1)
        assert error is None
        assert Chem.MolToSmiles(marked) == Chem.MolToSmiles(Chem.MolFromSmiles("[*:1]c1ccccc1"))

    def test_marking_drops_the_conformer(self):
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        marked, error = lib.mark_attachment(mol, 0, 1)
        assert error is None
        assert marked.GetNumConformers() == 0

    def test_a_hydrogen_is_not_a_site(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        hydrogen = next(
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1
        )
        marked, error = lib.mark_attachment(mol, hydrogen, 1)
        assert marked is None
        assert "not a substitutable heavy atom" in error
