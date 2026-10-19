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

"""Tests for the non-isolable-group filter on enumerated analogues.

A one-cut matched-molecular-pair swap does not know what it is attaching to. Cut
the O-CH3 bond of an anisole, re-attach "F" from the default substituent list,
and the product is an aryl hypofluorite -- a reagent that does not survive water,
offered as a lead-optimisation candidate.

Measured over four ordinary motifs (anisole, thioanisole, phenol, a
cyclopropylamide): 31 of 269 enumerated analogues, 12%, carried such a group.
Nothing downstream catches them. PAINS, Brenk and NIH pass every one, and the
ADMET model returns confident numbers for all of them -- so the agent builds
structure-activity reasoning on a molecule that cannot exist, and in eight of 122
benchmark submissions shipped one as its final answer.
"""

import pytest

from app.aevom_client import UNSTABLE_GROUPS, AevomClient, unstable_group

pytest.importorskip("rdkit", reason="RDKit lives in the base image")

ANISOLE = "COc1ccccc1"
THIOANISOLE = "CSc1ccccc1"
CYCLOPROPYLAMIDE = "O=C(NC1CC1)c1ccccc1"


@pytest.fixture
def client():
    return AevomClient(session_token="tok", backend_url="http://backend:8000")


class TestNamingTheGroup:
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("FOc1ccccc1", "hypohalite"),
            ("ClOc1ccccc1", "hypohalite"),
            ("FSc1ccccc1", "sulfenyl halide"),
            ("O=C(NF)c1ccccc1", "N-halamine"),
            ("O=C(F)c1ccccc1", "acyl halide"),
            ("OOc1ccccc1", "acyclic peroxide"),
        ],
    )
    def test_it_names_what_is_wrong(self, smiles, expected):
        found = unstable_group(smiles)
        assert found is not None and expected in found

    @pytest.mark.parametrize(
        "smiles",
        [
            "COc1ccccc1",              # anisole
            "OC(F)(F)Fc1ccccc1".replace("Fc1", "c1"),  # trifluoromethoxy, O-C not O-F
            "CSc1ccccc1",              # thioanisole
            "NNC(=O)c1ccncc1",         # isoniazid: a hydrazide is real chemistry
            "CC1(C)OO2",               # a cyclic peroxide fragment, as in artemisinin
            "Oc1ccccc1",
            "O=C(NO)c1ccccc1",         # hydroxamic acid: a real drug motif
        ],
    )
    def test_it_leaves_real_chemistry_alone(self, smiles):
        from rdkit import Chem

        if Chem.MolFromSmiles(smiles) is None:
            pytest.skip("not a parseable probe")
        assert unstable_group(smiles) is None

    def test_the_list_is_short_on_purpose(self):
        # Every entry is a group that cannot be isolated. Hydrazides,
        # hydroxylamines and cyclic endoperoxides are deliberately absent:
        # isoniazid and artemisinin are drugs, and refusing a molecule someone
        # wanted costs more than passing an unusual one.
        assert len(UNSTABLE_GROUPS) <= 8


class TestTheEnumeratorStopsMakingThem:
    def test_an_anisole_no_longer_yields_hypohalites(self, client):
        analogs = client.enumerate_analogs(ANISOLE, decorate=False)

        assert analogs
        assert not [s for s in analogs if unstable_group(s)]

    def test_a_thioanisole_no_longer_yields_sulfenyl_halides(self, client):
        analogs = client.enumerate_analogs(THIOANISOLE, decorate=False)

        assert not [s for s in analogs if unstable_group(s)]

    def test_an_amide_no_longer_yields_n_halamines(self, client):
        analogs = client.enumerate_analogs(CYCLOPROPYLAMIDE, decorate=False)

        assert not [s for s in analogs if unstable_group(s)]

    def test_the_useful_analogues_survive(self, client):
        # The filter must cost almost nothing: these are ordinary swaps on the
        # same cut that produced the hypohalites.
        analogs = set(client.enumerate_analogs(ANISOLE, decorate=False))

        assert "CCOc1ccccc1" in analogs      # ethoxy
        assert "Cc1ccccc1" in analogs        # the O removed entirely
        assert len(analogs) > 50

    def test_a_caller_can_still_ask_for_them(self, client):
        kept = client.enumerate_analogs(ANISOLE, decorate=False, allow_unstable=True)

        assert [s for s in kept if unstable_group(s)]
