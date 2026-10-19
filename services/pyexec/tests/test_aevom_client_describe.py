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

"""Tests for the structural readout.

A SMILES string is not legible, and the failure this exists for is a real one
from the traces: an agent wrote `OC(F)(F)c1ccccc1` while describing it as "O-CF2H
attached to phenyl (oxygen linked to phenyl)". It is the other way round -- the
ring is on the carbon, so the string is a gem-fluorohydrin, the hydrate of an
acyl fluoride. It parses, it clears all twelve drug-likeness gates, and it was
shipped as a final answer in two of six runs on that task.

The point of these tests is that the readout disagrees with the misreading.
"""

import pytest

from app.aevom_client import AevomClient, AevomError, describe_molecule

pytest.importorskip("rdkit", reason="RDKit lives in the base image")


@pytest.fixture
def client():
    return AevomClient(session_token="tok", backend_url="http://backend:8000")


class TestTheCaseThatMotivatedIt:
    """The two strings the agent confused, told apart."""

    def test_the_one_it_wrote_is_a_halohydrin(self):
        described = describe_molecule("OC(F)(F)c1ccccc1")

        assert "gem-halohydrin" in described["groups"]

    def test_the_one_it_meant_is_an_ether(self):
        described = describe_molecule("C(F)(F)Oc1ccccc1")

        assert "gem-halohydrin" not in described["groups"]
        assert "aryl ether" in described["groups"]

    def test_they_have_the_same_formula(self):
        # Which is why nothing else caught it.
        assert (
            describe_molecule("OC(F)(F)c1ccccc1")["formula"]
            == describe_molecule("C(F)(F)Oc1ccccc1")["formula"]
        )


class TestWhatItReports:
    def test_a_sulfonamide_is_named_before_the_amine_in_it(self):
        groups = describe_molecule("NS(=O)(=O)c1ccccc1")["groups"]

        assert "sulfonamide" in groups

    def test_an_acid_is_not_reported_as_an_alcohol(self):
        groups = describe_molecule("OC(=O)c1ccccc1")["groups"]

        assert "carboxylic acid" in groups
        assert "alcohol" not in groups

    def test_rings_are_described(self):
        rings = describe_molecule("c1ccc2[nH]ccc2c1")["rings"]

        assert [r["size"] for r in rings] == [6, 5]
        assert all(r["aromatic"] for r in rings)
        assert rings[1]["heteroatoms"] == ["N"]

    def test_the_formula_is_the_graph_s_own(self):
        assert describe_molecule("CCO")["formula"] == "C2H6O"

    def test_an_impossible_group_is_flagged_as_well_as_named(self):
        described = describe_molecule("FOc1ccccc1")

        assert "aryl hypohalite (O-halogen)" in described["groups"]
        assert described["unstable"] is not None

    def test_an_ordinary_drug_is_not_over_reported(self):
        groups = describe_molecule("CC(=O)Oc1ccccc1C(=O)O")["groups"]

        assert "ester" in groups
        assert "carboxylic acid" in groups
        assert not [g for g in groups if "halide" in g or "halamine" in g]

    def test_a_bad_smiles_carries_the_diagnosis(self, client):
        with pytest.raises(AevomError) as caught:
            describe_molecule("Cc1ccc(NHS(=O)(=O)c2ccccc2)cc1")

        assert "not an atom" in str(caught.value)


class TestTheFluorineAxis:
    """CH3 -> CH2F -> CHF2 -> CF3 is not four equally inert points.

    The model applies "fluorine blocks oxidative metabolism" without asking what
    the carbon bearing it is attached to, and picks CH2F when CF3 reads as too
    lipophilic. At a benzylic or alpha-heteroatom position that is an alkylating
    agent, not a metabolic block. 151 such structures were typed by hand across
    72 of 120 benchmark runs, at the same rate in every arm -- so the written
    guidance did not touch it, and naming the group is what is left.
    """

    def test_a_benzylic_fluoride_is_named_as_such(self):
        groups = describe_molecule("FCc1ccccc1")["groups"]

        assert "benzylic halide" in groups

    def test_an_alpha_halo_ether_is_named_as_such(self):
        groups = describe_molecule("FCOc1ccccc1")["groups"]

        assert "alpha-halo ether" in groups

    def test_a_trifluoromethyl_is_not_confused_with_them(self):
        groups = describe_molecule("FC(F)(F)c1ccccc1")["groups"]

        assert "trifluoromethyl" in groups
        assert "benzylic halide" not in groups

    def test_an_aryl_fluoride_is_neither(self):
        groups = describe_molecule("Fc1ccccc1")["groups"]

        assert "aryl fluoride" in groups
        assert "benzylic halide" not in groups
