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

"""Tests for the PLIP service HTTP surface.

The behaviour worth pinning is partial failure. A request carries an ensemble,
and one unreadable diffusion sample out of ten must not discard the other nine --
so a bad structure lands in `failed` while the rest are profiled and folded.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_STRUCTURES, app

FIXTURE = pathlib.Path(__file__).parent / "1eve_site.pdb"
DONEPEZIL = "E20:A:2001"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def structure() -> str:
    return FIXTURE.read_text()


def post(client, **payload):
    response = client.post("/interactions", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


class TestService:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["service"] == "plip"

    def test_publishes_the_families_it_detects(self, client):
        body = client.get("/families").json()
        names = [entry["name"] for entry in body["families"]]
        assert "pistacking" in names
        assert len(names) == 8
        assert body["max_structures"] == MAX_STRUCTURES


class TestSingleStructure:
    def test_profiles_one_structure(self, client, structure):
        body = post(client, structures=[{"name": "pose", "content": structure}])
        assert body["success"]
        assert body["structures"][0]["designated"] == DONEPEZIL
        assert len(body["structures"][0]["fingerprint"]) == 9

    def test_a_single_structure_folds_to_full_consensus(self, client, structure):
        body = post(client, structures=[{"name": "pose", "content": structure}])
        assert body["consensus"]["total_structures"] == 1
        assert len(body["consensus"]["conserved"]) == 9
        assert body["consensus"]["num_variable"] == 0

    def test_reports_consensus_residues(self, client, structure):
        body = post(client, structures=[{"name": "pose", "content": structure}])
        residues = body["consensus_residues"]
        assert {entry["residue_number"] for entry in residues} == {
            84,
            121,
            279,
            288,
            330,
            331,
            334,
        }

    def test_carries_the_family_legend(self, client, structure):
        body = post(client, structures=[{"name": "pose", "content": structure}])
        assert body["families"]["pistacking"] == "pi-stacking"


class TestEnsemble:
    def test_identical_poses_are_fully_conserved(self, client, structure):
        body = post(
            client,
            structures=[
                {"name": f"pose{index}", "content": structure} for index in range(3)
            ],
        )
        assert body["consensus"]["total_structures"] == 3
        assert all(
            entry["fraction"] == 1.0 for entry in body["consensus"]["conserved"]
        )

    def test_the_threshold_is_honoured(self, client, structure):
        protein_only = "\n".join(
            line
            for line in structure.splitlines()
            if not line.startswith(("HETATM", "CONECT", "HET   ", "LINK  "))
        )
        # One pose binds, one does not: every interaction sits at exactly half.
        body = post(
            client,
            structures=[
                {"name": "bound", "content": structure},
                {"name": "empty", "content": protein_only},
            ],
            consensus_threshold=0.9,
        )
        assert body["consensus"]["conserved"] == []
        assert body["consensus"]["num_variable"] == 9


class TestPartialFailure:
    def test_one_bad_structure_does_not_discard_the_others(self, client, structure):
        body = post(
            client,
            structures=[
                {"name": "good", "content": structure},
                {"name": "bad", "content": "not a structure at all"},
            ],
        )
        assert body["success"]
        assert [entry["name"] for entry in body["structures"]] == ["good"]
        assert [entry["name"] for entry in body["failed"]] == ["bad"]
        # The denominator is the structures that were profiled, not the ones
        # that were sent -- otherwise a parse failure looks like a pose that
        # simply did not make the interaction.
        assert body["consensus"]["total_structures"] == 1

    def test_all_bad_is_a_failed_request(self, client):
        body = post(client, structures=[{"name": "bad", "content": "nope"}])
        assert not body["success"]
        assert "None of the structures" in body["error"]
        assert body["failed"][0]["name"] == "bad"

    def test_the_error_names_a_cause(self, client):
        body = post(client, structures=[{"name": "bad", "content": "nope"}])
        assert "not recognized" in body["failed"][0]["error"]


class TestConcurrency:
    def test_two_requests_do_not_interleave_an_analysis(self, structure):
        # PLIP writes the protonated structure to a fixed path, so overlapping
        # analyses would read each other's file. This drives two requests at once
        # through a real event loop and checks both answers are still right --
        # a corrupted read shows up as a different fingerprint or a crash.
        import asyncio

        from app.main import InteractionsRequest, Structure, interactions

        async def drive():
            request = InteractionsRequest(
                structures=[Structure(name="pose", content=structure)]
            )
            return await asyncio.gather(
                interactions(request),
                interactions(request),
                interactions(request),
            )

        results = asyncio.run(drive())
        fingerprints = [
            tuple(response.structures[0]["fingerprint"]) for response in results
        ]
        assert len(set(fingerprints)) == 1
        assert len(fingerprints[0]) == 9

    def test_a_failed_structure_releases_the_lock(self, client, structure):
        # The lock is taken inside the loop, so an exception must not leave it
        # held -- if it did, this second request would hang rather than answer.
        post(client, structures=[{"name": "bad", "content": "nope"}])
        body = post(client, structures=[{"name": "good", "content": structure}])
        assert body["success"]


class TestRequestValidation:
    def test_rejects_an_empty_request(self, client):
        assert client.post("/interactions", json={"structures": []}).status_code == 422

    def test_rejects_too_many_structures(self, client, structure):
        response = client.post(
            "/interactions",
            json={
                "structures": [
                    {"name": str(index), "content": "x"}
                    for index in range(MAX_STRUCTURES + 1)
                ]
            },
        )
        assert response.status_code == 422

    def test_rejects_an_out_of_range_threshold(self, client, structure):
        response = client.post(
            "/interactions",
            json={
                "structures": [{"name": "pose", "content": structure}],
                "consensus_threshold": 1.5,
            },
        )
        assert response.status_code == 422

    def test_a_requested_ligand_is_passed_through(self, client, structure):
        body = post(
            client,
            structures=[{"name": "pose", "content": structure}],
            ligand="NAG",
        )
        assert body["structures"][0]["designated"].startswith("NAG")
