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

"""Tests for the aevom client's synthetic accessibility helper.

The helper exists because the scorer ships in RDKit's Contrib tree rather than
the package, so reaching it needs `sys.path`, and `sys` is not importable in the
sandbox. That makes "callable from sandboxed code" the property worth pinning,
alongside the numbers themselves -- the score is only useful if it agrees with
what `molecule_gates` reports for the same molecule.
"""

import pytest

from app.aevom_client import AevomClient, AevomError, _load_sascorer
from app.sandbox import SandboxContext, execute_code

pytest.importorskip("rdkit", reason="RDKit lives in the base image")

pytestmark = pytest.mark.skipif(
    _load_sascorer() is None,
    reason="RDKit's Contrib tree is absent from this environment",
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
# Morphine: fused rings and four stereocentres, so it scores far above aspirin.
MORPHINE = "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5"


@pytest.fixture
def client():
    return AevomClient(session_token="tok", backend_url="http://backend:8000")


class TestScoring:
    def test_scores_a_smiles_string(self, client):
        assert 1.0 <= client.sa_score(ASPIRIN) <= 10.0

    def test_scores_an_rdkit_mol(self, client):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ASPIRIN)
        assert client.sa_score(mol) == pytest.approx(client.sa_score(ASPIRIN))

    def test_a_simple_drug_scores_low(self, client):
        assert client.sa_score(ASPIRIN) < 3.0

    def test_a_complex_natural_product_scores_higher(self, client):
        assert client.sa_score(MORPHINE) > client.sa_score(ASPIRIN)

    def test_it_agrees_with_the_gates_service(self, client):
        # molecule_gates applies a threshold to this same score, so the two must
        # not disagree -- a molecule cannot pass one and fail the other.
        from rdkit import Chem

        scorer = _load_sascorer()
        expected = scorer.calculateScore(Chem.MolFromSmiles(ASPIRIN))
        assert client.sa_score(ASPIRIN) == pytest.approx(expected)


class TestFailures:
    def test_invalid_smiles_is_rejected(self, client):
        with pytest.raises(AevomError, match="Invalid SMILES"):
            client.sa_score("not a molecule")

    def test_none_is_rejected(self, client):
        with pytest.raises(AevomError, match="None"):
            client.sa_score(None)

    def test_a_missing_contrib_tree_is_reported_clearly(self, client, monkeypatch):
        # An image built without RDKit's Contrib tree should say so, rather than
        # failing with an AttributeError on None.
        monkeypatch.setattr("app.aevom_client._load_sascorer", lambda: None)
        with pytest.raises(AevomError, match="Contrib"):
            client.sa_score(ASPIRIN)


class TestReachableFromTheSandbox:
    def test_sandboxed_code_can_call_it(self, monkeypatch):
        # The point of the helper: this is the call the sandbox cannot make for
        # itself, because importing sys to extend the path is blocked.
        monkeypatch.setattr(
            "app.aevom_client.AevomClient._fetch_data",
            lambda self, ref: b"",
        )
        context = SandboxContext(
            session_token="tok", backend_url="http://backend:8000", allowed_refs=set()
        )
        result = execute_code(
            f'result = round(aevom.sa_score("{ASPIRIN}"), 3)',
            sandbox_context=context,
        )
        assert result["success"] is True, result["error"]
        assert 1.0 <= result["result"] <= 10.0

    def test_the_manual_route_is_still_blocked(self):
        # If this ever starts passing, the helper is no longer necessary.
        result = execute_code(
            "import sys\nfrom rdkit.Chem import RDConfig\nresult = 1"
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]
