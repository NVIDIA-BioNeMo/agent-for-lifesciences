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

"""Tests for the analogue enumerator.

Lead optimisation asks for a small change to a molecule you already have,
inside a similarity bound. Nothing in the sandbox did that: the agent either
hand-wrote a handful of SMILES or reached for a generative model, which cannot
be held inside the bound. So the properties worth pinning are that it produces
real, distinct, connected molecules, that the similarity bound is honoured
rather than advertised, and that it is callable from sandboxed code.
"""

import pytest

from app.aevom_client import AevomClient, AevomError
from app.sandbox import SandboxContext, execute_code

pytest.importorskip("rdkit", reason="RDKit lives in the base image")

# The reference ligand of smdd_004_ABL1_6, a real lead-optimisation starting point.
LEAD = "Cc1ncc(F)cc1-c1ccc2cc(NC(=O)C3CC3)ncc2c1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def client():
    return AevomClient(session_token="tok", backend_url="http://backend:8000")


class TestWhatComesOut:
    def test_a_lead_yields_hundreds_of_analogues(self, client):
        analogs = client.enumerate_analogs(LEAD)

        assert len(analogs) > 100

    def test_every_analogue_is_a_real_single_molecule(self, client):
        from rdkit import Chem

        for smiles in client.enumerate_analogs(ASPIRIN, max_analogs=60):
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, smiles
            # A cut that re-attached nothing would leave two fragments.
            assert "." not in smiles, smiles

    def test_they_are_distinct_and_exclude_the_input(self, client):
        from rdkit import Chem

        analogs = client.enumerate_analogs(LEAD, max_analogs=200)

        assert len(analogs) == len(set(analogs))
        assert Chem.MolToSmiles(Chem.MolFromSmiles(LEAD)) not in analogs

    def test_a_mol_and_its_smiles_give_the_same_answer(self, client):
        from rdkit import Chem

        assert client.enumerate_analogs(ASPIRIN, max_analogs=40) == client.enumerate_analogs(
            Chem.MolFromSmiles(ASPIRIN), max_analogs=40
        )


class TestTheSimilarityBound:
    def test_the_bound_is_enforced_not_advertised(self, client):
        """The bound is the requirement in a lead-optimisation brief, so a
        single analogue outside it is a wrong answer, not a near miss."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        lead = Chem.MolFromSmiles(LEAD)
        reference_fp = AllChem.GetMorganFingerprintAsBitVect(lead, 2, 2048)

        for smiles in client.enumerate_analogs(LEAD, similarity_min=0.7):
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)
            assert DataStructs.TanimotoSimilarity(reference_fp, fp) >= 0.7, smiles

    def test_a_tighter_bound_keeps_fewer(self, client):
        loose = client.enumerate_analogs(LEAD, similarity_min=0.5)
        tight = client.enumerate_analogs(LEAD, similarity_min=0.9)

        assert len(tight) < len(loose)

    def test_an_impossible_bound_returns_nothing_rather_than_raising(self, client):
        assert client.enumerate_analogs(ASPIRIN, similarity_min=1.0) == []


class TestLimitsAndRefusals:
    def test_the_cap_is_respected(self, client):
        assert len(client.enumerate_analogs(LEAD, max_analogs=25)) == 25

    def test_decoration_can_be_turned_off(self, client):
        """Decorating an aromatic CH adds heavy atoms; swapping need not."""
        with_decoration = client.enumerate_analogs(LEAD, max_analogs=5000)
        without = client.enumerate_analogs(LEAD, max_analogs=5000, decorate=False)

        assert len(without) < len(with_decoration)

    def test_a_custom_substituent_list_is_used(self, client):
        analogs = client.enumerate_analogs(ASPIRIN, r_groups=["Cl"], max_analogs=50)

        assert analogs
        assert all("Cl" in smiles for smiles in analogs)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"r_groups": []}, "No substituents"),
            ({"max_analogs": 0}, "must be positive"),
            ({"similarity_min": 1.5}, "between 0 and 1"),
        ],
    )
    def test_a_nonsensical_request_says_what_is_wrong(self, client, kwargs, message):
        with pytest.raises(AevomError, match=message):
            client.enumerate_analogs(ASPIRIN, **kwargs)

    def test_an_invalid_molecule_is_refused(self, client):
        with pytest.raises(AevomError, match="Invalid SMILES"):
            client.enumerate_analogs("not a molecule")

    def test_none_is_refused_rather_than_returning_nothing(self, client):
        with pytest.raises(AevomError, match="None"):
            client.enumerate_analogs(None)


def test_it_is_callable_from_sandboxed_code():
    """RestrictedPython is the environment that matters; a helper that works
    in a test and not in the sandbox is no helper at all."""
    context = SandboxContext(
        session_token="tok", backend_url="http://backend:8000", allowed_refs=set()
    )
    result = execute_code(
        f"analogs = aevom.enumerate_analogs('{ASPIRIN}', max_analogs=12)\n"
        "result = len(analogs)\n",
        sandbox_context=context,
    )

    assert result["success"] is True, result["error"]
    assert result["result"] == 12
