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

"""Tests for the SMILES diagnosis.

RDKit answers "is this a molecule" with None, which is correct and useless for
the next question. Agents write SMILES by hand and then drop the failures with
`if mol is None: continue`, so the loss is silent: measured over 2759 structures
written into this sandbox by hand, 318 would not parse and the code that wrote
them never learned that, or why.

The five classes tested here are not a taxonomy anyone invented. They are every
failure in that sample, in proportion: an NH written as an atom (137), a ring
that cannot be kekulized (89), a ring closure left open (46), an atom asked for
one bond too many (42), an aromatic atom outside a ring (4).
"""

import pytest

from app.aevom_client import AevomClient, AevomError, diagnose_smiles

pytest.importorskip("rdkit", reason="RDKit lives in the base image")


@pytest.fixture
def client():
    return AevomClient(session_token="tok", backend_url="http://backend:8000")


class TestTheFiveWaysItActuallyBreaks:
    def test_a_hydrogen_written_as_an_atom(self):
        # The single largest class. A chemist says "the sulfonamide NH"; SMILES
        # reads N bonded to a hydrogen atom, and there is no bond left for S.
        message = diagnose_smiles("Cc1ccc(NHS(=O)(=O)c2ccccc2)cc1")

        assert "NH" in message
        assert "not an atom" in message
        assert "Write N" in message

    def test_a_ring_closure_left_open(self):
        message = diagnose_smiles("C1CCC(=O)Nc1ccc(F)cc1")

        assert "odd number of times" in message
        assert "opened and never closed" in message

    def test_a_ring_that_cannot_be_kekulized(self):
        # One atom short of the ring that was intended -- what editing a
        # substituent on an aromatic ring does when a character goes missing.
        message = diagnose_smiles("Cc1ccc(NS(=O)(=O)c2ccc(F)c2F)cc1")

        assert "kekulize" in message.lower()
        assert "[nH]" in message

    def test_an_atom_asked_for_one_bond_too_many(self):
        # CF3 written in the direction it is spoken rather than the direction it
        # bonds, so the carbon is asked for a fifth bond.
        message = diagnose_smiles("C(F)(F)Fc1ccccc1")

        assert "valence" in message.lower()
        assert "FC(F)(F)c1" in message

    def test_an_aromatic_atom_outside_a_ring(self):
        message = diagnose_smiles("Brcc1ccccc1")

        assert "non-ring atom" in message

    def test_a_name_is_not_a_structure(self):
        # Outside square brackets SMILES admits only the organic subset, so a
        # letter from outside it proves the string was never SMILES.
        message = diagnose_smiles("methylcarboxamide")

        assert "cannot appear in SMILES" in message

    def test_a_placeholder_atom_is_named(self):
        message = diagnose_smiles("N3CCC[C@H]3C(X)Y")

        assert "placeholder atom is [*]" in message

    def test_a_smarts_pattern_is_named(self):
        message = diagnose_smiles("[NX3][CX3](=O)[#6]")

        assert "SMARTS" in message
        assert "MolFromSmarts" in message

    def test_a_protein_sequence_is_named(self):
        message = diagnose_smiles("DDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAVKML")

        assert "protein sequence" in message

    def test_a_group_formula_is_named_as_such(self):
        message = diagnose_smiles("Cc1ccc(CF3)cc1")

        assert "group formula" in message
        assert "C(F)(F)F" in message


class TestItDoesNotCryWolf:
    """A diagnosis that fires on valid chemistry is worse than none."""

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",
            "CN",                      # methylamine: an N and a C, not "CN" the group
            "c1ccncc1",
            "Cc1cc[nH]c1",             # a legitimate bracketed [nH]
            "OC(F)(F)F",
            "N[C@@H](C)C(=O)O",
            "C[N+](=O)[O-]",          # nitro, bonded the way it actually bonds
            "C(=O)O",
        ],
    )
    def test_valid_smiles_still_parses(self, client, smiles):
        assert client.mol_from_smiles(smiles) is not None


class TestTheErrorCarriesTheDiagnosis:
    def test_mol_from_smiles_says_why(self, client):
        with pytest.raises(AevomError) as caught:
            client.mol_from_smiles("Cc1ccc(NHS(=O)(=O)c2ccccc2)cc1")

        message = str(caught.value)
        assert "Failed to parse SMILES" in message
        assert "not an atom" in message

    def test_a_substituent_that_will_not_attach_is_refused(self, client):
        # The one mistake enumerate_analogs still permits. Left unchecked it
        # returns a shorter list and no reason for it.
        with pytest.raises(AevomError) as caught:
            client.enumerate_analogs("Cc1ccccc1", r_groups=["C#N", "NHS(=O)(=O)"])

        message = str(caught.value)
        assert "NHS(=O)(=O)" in message
        assert "will not attach" in message

    def test_a_good_substituent_list_is_left_alone(self, client):
        analogs = client.enumerate_analogs(
            "Cc1ccccc1", r_groups=["C#C", "OC(F)(F)F"], decorate=False
        )

        assert analogs
        assert all(isinstance(s, str) for s in analogs)
