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

"""Tests for the fragment_link tool.

Every service is stubbed. Whether a molecule really can hold two fragments is the
rdkit suite's question (``services/rdkit/tests/test_constrained_pose.py``, and
``test_pdbbind_validation.py`` for whether the thresholds are the right ones).
What is tested here is the orchestration: that candidates are found when none are
given, that verification is chunked, that the property panel and the Vina rescore
attach to the right molecules, that survivors are ranked and stored, and that a
rejected candidate comes back with the reason rather than silently missing.
"""

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.fragment_link import FragmentLinkTool

FRAGMENT_SDF = """benzene
  RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    1.0000    0.0000    0.0000 C   0  0
M  END
"""

POSE_SDF = """pose
  RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
   10.0000    2.0000    3.0000 C   0  0
   12.0000    4.0000    5.0000 C   0  0
  1  2  1  0
M  END
"""


def verified(smiles: str, **overrides) -> dict:
    result = {
        "smiles": smiles,
        "verified": True,
        "failed_gate": None,
        "reason": None,
        "matches": [],
        "strain": 8.2,
        "max_fragment_rmsd": 0.24,
        "min_protein_distance": 2.7,
        "num_clashes": 0,
        "num_linker_atoms": 6,
        "embed_mode": "restrained",
        "pose_sdf": POSE_SDF,
    }
    result.update(overrides)
    return result


def rejected(smiles: str, gate: str = "gate5c_pose", reason: str = "too short") -> dict:
    return {
        "smiles": smiles,
        "verified": False,
        "failed_gate": gate,
        "reason": reason,
        "matches": [],
        "strain": None,
        "max_fragment_rmsd": None,
        "min_protein_distance": None,
        "num_clashes": 0,
        "pose_sdf": None,
    }


GEOMETRY = {
    "success": True,
    "fragments": [{"index": 0, "smiles": "c1ccccc1", "num_sites": 6, "sites": []}],
    "pairs": [
        {
            "pair_id": "p0",
            "atom_a": 1,
            "atom_b": 3,
            "fragment_a": 0,
            "fragment_b": 1,
            "anchor_distance": 8.4,
            "exit_vector_angle": 20.0,
            "tube_clearance": 3.5,
            "linker_atoms_min": 5,
            "linker_atoms_max": 9,
            "saturated_chain_rb_cost": 6,
            "requires_rigid_linker": False,
            "templates": {"safe": "c1ccccc1.[*{6-13}].c1ccncc1", "marker": "..."},
            "linkers": [],
            "warnings": [],
        }
    ],
    "candidates": [
        {
            "smiles": "c1ccc(-c2ccncc2)cc1",
            "pair_id": "p0",
            "linker": "phenyl",
            "linker_atoms": 6,
            "linker_rb_cost": 2,
            "anchor_distance": 8.4,
            "atom_a": 1,
            "atom_b": 3,
        }
    ],
    "rb_headroom": 8,
    "rb_max": 10,
    "pocket_atoms": 871,
    "warnings": [],
}

GATES = {
    "success": True,
    "results": [
        {
            "smiles": "c1ccc(-c2ccncc2)cc1",
            "passed": True,
            "properties": {"mw": 155.2, "rotatable_bonds": 1},
            "failures": [],
        }
    ],
}


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = FragmentLinkTool()
    instance._storage = storage
    # The ledger is a session-scoped side effect with no session in a unit test.
    instance._record = _no_record
    return instance


async def _no_record(survivors, provenance):
    return None


def route(tool, handlers: dict, calls: list | None = None):
    """Dispatch stubbed service calls by endpoint."""

    async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        if calls is not None:
            calls.append((endpoint, json_data))
        payload = handlers[endpoint]
        if isinstance(payload, Exception):
            raise payload
        if callable(payload):
            return payload(json_data)
        return payload

    tool._call_service = _call


async def refs(storage) -> tuple[list[str], str]:
    fragments = [await storage.store("sdf", FRAGMENT_SDF.encode()) for _ in range(2)]
    pocket = await storage.store("pdb", b"ATOM      1  N   ALA A   1\nEND\n")
    return fragments, pocket


class TestInputHandling:
    @pytest.mark.asyncio
    async def test_requires_fragments(self, tool):
        result = await tool.execute({"candidate_smiles": ["CCO"]})
        assert result.content_type == ContentType.ERROR
        assert "fragment_sdf_refs" in result.data

    @pytest.mark.asyncio
    async def test_a_bad_fragment_reference_is_named(self, tool):
        result = await tool.execute({"fragment_sdf_refs": ["nonsense"]})
        assert result.content_type == ContentType.ERROR
        assert "nonsense" in result.data

    @pytest.mark.asyncio
    async def test_candidates_can_come_from_a_csv(self, tool, storage):
        fragments, pocket = await refs(storage)
        csv_ref = await storage.store(
            "csv", b"smiles,score\nc1ccc(-c2ccncc2)cc1,0.9\nCCO,0.4\n"
        )
        calls: list = []
        route(
            tool,
            {
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [verified(s) for s in body["smiles"]],
                },
                "/molecule-gates": GATES,
            },
            calls,
        )

        await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "pocket_pdb_ref": pocket,
                "candidate_csv_ref": csv_ref,
            }
        )

        verify = next(body for endpoint, body in calls if endpoint == "/constrained-pose")
        assert verify["smiles"] == ["c1ccc(-c2ccncc2)cc1", "CCO"]

    @pytest.mark.asyncio
    async def test_duplicate_candidates_are_verified_once(self, tool, storage):
        fragments, _pocket = await refs(storage)
        calls: list = []
        route(
            tool,
            {
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [verified(s) for s in body["smiles"]],
                },
                "/molecule-gates": GATES,
            },
            calls,
        )
        await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO", "CCO", "CCC"]}
        )
        verify = next(body for endpoint, body in calls if endpoint == "/constrained-pose")
        assert verify["smiles"] == ["CCO", "CCC"]


class TestEnumeration:
    @pytest.mark.asyncio
    async def test_with_no_candidates_it_builds_its_own(self, tool, storage):
        fragments, pocket = await refs(storage)
        calls: list = []
        route(
            tool,
            {
                "/link-geometry": GEOMETRY,
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [verified(s) for s in body["smiles"]],
                },
                "/molecule-gates": GATES,
            },
            calls,
        )

        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "pocket_pdb_ref": pocket}
        )

        endpoints = [endpoint for endpoint, _ in calls]
        assert endpoints[0] == "/link-geometry"
        assert "/constrained-pose" in endpoints
        assert result.metadata["num_verified"] == 1
        assert "assembled from the linker library" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_the_library_finding_nothing_is_explained(self, tool, storage):
        fragments, pocket = await refs(storage)
        empty = {**GEOMETRY, "candidates": []}
        empty["pairs"] = [
            {**GEOMETRY["pairs"][0], "warnings": ["No library entry spans 14.2 A"]}
        ]
        route(tool, {"/link-geometry": empty})

        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "pocket_pdb_ref": pocket}
        )

        assert result.content_type == ContentType.ERROR
        assert "No library entry spans" in result.data
        assert "generate=true" in result.data

    @pytest.mark.asyncio
    async def test_genmol_molecules_join_the_library_ones(self, tool, storage, monkeypatch):
        fragments, pocket = await refs(storage)
        seen: dict = {}

        async def fake_generate(safe_string, num_molecules=60, **kwargs):
            seen["safe"] = safe_string
            seen["count"] = num_molecules
            return [("c1ccc(C#Cc2ccncc2)cc1", 0.8)], None

        monkeypatch.setattr(
            "api.tools.fragment_link.generate_from_safe", fake_generate
        )
        calls: list = []
        route(
            tool,
            {
                "/link-geometry": GEOMETRY,
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [verified(s) for s in body["smiles"]],
                },
                "/molecule-gates": {"success": True, "results": []},
            },
            calls,
        )

        result = await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "pocket_pdb_ref": pocket,
                "generate": True,
                "num_generate": 12,
            }
        )

        assert seen["safe"] == "c1ccc(-c2ccncc2)cc1" or seen["safe"].startswith("c1ccccc1")
        assert seen["count"] == 12
        verify = next(body for endpoint, body in calls if endpoint == "/constrained-pose")
        assert "c1ccc(C#Cc2ccncc2)cc1" in verify["smiles"]
        assert "GenMol returned 1 molecule" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_genmol_failing_does_not_lose_the_library_candidates(
        self, tool, storage, monkeypatch
    ):
        fragments, pocket = await refs(storage)

        async def fake_generate(safe_string, num_molecules=60, **kwargs):
            return [], "GenMol failed: the NIM is unreachable"

        monkeypatch.setattr(
            "api.tools.fragment_link.generate_from_safe", fake_generate
        )
        route(
            tool,
            {
                "/link-geometry": GEOMETRY,
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [verified(s) for s in body["smiles"]],
                },
                "/molecule-gates": GATES,
            },
        )

        result = await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "pocket_pdb_ref": pocket,
                "generate": True,
            }
        )

        assert result.metadata["num_verified"] == 1
        assert "the NIM is unreachable" in result.metadata["message"]


class TestVerification:
    @pytest.mark.asyncio
    async def test_large_batches_are_chunked(self, tool, storage):
        fragments, _pocket = await refs(storage)
        calls: list = []
        route(
            tool,
            {
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [rejected(s) for s in body["smiles"]],
                },
            },
            calls,
        )

        smiles = [f"C{'C' * n}O" for n in range(70)]
        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": smiles}
        )

        chunks = [body for endpoint, body in calls if endpoint == "/constrained-pose"]
        assert len(chunks) == 3
        assert sum(len(chunk["smiles"]) for chunk in chunks) == 70
        assert len(result.metadata["results"]) == 70

    @pytest.mark.asyncio
    async def test_rejections_are_reported_with_their_reason(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(
            tool,
            {
                "/constrained-pose": {
                    "success": True,
                    "results": [
                        rejected("CCO", "gate2_substructure", "fragment 0 is missing"),
                        rejected("CCC", "gate5c_pose", "the bridge is too short"),
                    ],
                }
            },
        )

        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO", "CCC"]}
        )

        assert result.content_type == ContentType.TEXT
        assert "gate2_substructure" in result.data
        assert "fragment 0 is missing" in result.data
        assert "the bridge is too short" in result.data

    @pytest.mark.asyncio
    async def test_thresholds_are_passed_through(self, tool, storage):
        fragments, _pocket = await refs(storage)
        calls: list = []
        route(
            tool,
            {"/constrained-pose": {"success": True, "results": [rejected("CCO")]}},
            calls,
        )
        await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "candidate_smiles": ["CCO"],
                "strain_max": 40.0,
                "rmsd_max": 1.5,
            }
        )
        body = calls[0][1]
        assert body["strain_max"] == 40.0
        assert body["rmsd_max"] == 1.5

    @pytest.mark.asyncio
    async def test_a_verification_failure_is_reported_not_raised(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(tool, {"/constrained-pose": RuntimeError("rdkit is down")})
        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO"]}
        )
        assert result.content_type == ContentType.ERROR
        assert "rdkit is down" in result.data


class TestGatesAndRanking:
    @pytest.mark.asyncio
    async def test_a_property_failure_is_kept_but_flagged(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(
            tool,
            {
                "/constrained-pose": {"success": True, "results": [verified("CCO")]},
                "/molecule-gates": {
                    "success": True,
                    "results": [
                        {
                            "smiles": "CCO",
                            "passed": False,
                            "properties": {"mw": 700.0},
                            "failures": ["MW 700.0 violates <= 600"],
                        }
                    ],
                },
            },
        )

        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO"]}
        )

        assert "MW 700.0 violates" in result.metadata["message"]
        assert "change what the linker is made of" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_gate_passers_outrank_gate_failers(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(
            tool,
            {
                "/constrained-pose": {
                    "success": True,
                    "results": [verified("CCO", strain=1.0), verified("CCC", strain=9.0)],
                },
                "/molecule-gates": {
                    "success": True,
                    "results": [
                        {"smiles": "CCO", "passed": False, "failures": ["MW"], "properties": {}},
                        {"smiles": "CCC", "passed": True, "failures": [], "properties": {}},
                    ],
                },
            },
        )

        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO", "CCC"]}
        )

        order = [row["smiles"] for row in result.metadata["results"] if row["verified"]]
        assert order == ["CCO", "CCC"]
        assert list(result.metadata["pose_refs"]) == ["CCC", "CCO"]

    @pytest.mark.asyncio
    async def test_a_gate_service_failure_does_not_lose_the_survivors(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(
            tool,
            {
                "/constrained-pose": {"success": True, "results": [verified("CCO")]},
                "/molecule-gates": RuntimeError("gates are down"),
            },
        )
        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO"]}
        )
        assert result.metadata["num_verified"] == 1


class TestPosesAndRescoring:
    @pytest.mark.asyncio
    async def test_poses_are_stored_as_references(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(
            tool,
            {
                "/constrained-pose": {"success": True, "results": [verified("CCO")]},
                "/molecule-gates": {"success": True, "results": []},
            },
        )

        result = await tool.execute(
            {"fragment_sdf_refs": fragments, "candidate_smiles": ["CCO"]}
        )

        assert result.content_type == ContentType.DATA_REFERENCE
        reference = result.metadata["pose_refs"]["CCO"]
        assert (await storage.retrieve(reference)).decode().startswith("pose")

    @pytest.mark.asyncio
    async def test_only_the_best_few_poses_are_stored(self, tool, storage):
        fragments, _pocket = await refs(storage)
        route(
            tool,
            {
                "/constrained-pose": {
                    "success": True,
                    "results": [verified(f"C{'C' * n}O") for n in range(8)],
                },
                "/molecule-gates": {"success": True, "results": []},
            },
        )
        result = await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "candidate_smiles": [f"C{'C' * n}O" for n in range(8)],
                "max_survivors": 2,
            }
        )
        assert len(result.metadata["pose_refs"]) == 2
        assert result.metadata["num_verified"] == 8

    @pytest.mark.asyncio
    async def test_rescoring_uses_local_mode_and_boxes_the_pose(self, tool, storage):
        fragments, pocket = await refs(storage)
        docked: list = []

        async def fake_dock(base_url, endpoint, json_data, timeout):
            docked.append((base_url, endpoint, json_data))
            return {"success": True, "best_affinity": -9.4}

        tool._call_service_at = fake_dock
        route(
            tool,
            {
                "/constrained-pose": {"success": True, "results": [verified("CCO")]},
                "/molecule-gates": {"success": True, "results": []},
            },
        )

        result = await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "pocket_pdb_ref": pocket,
                "candidate_smiles": ["CCO"],
                "rescore_with_vina": True,
            }
        )

        _url, endpoint, body = docked[0]
        assert endpoint == "/dock-auto"
        assert body["score_mode"] == "local"
        # The box is centred on the pose, not on the origin.
        assert body["center_x"] == pytest.approx(11.0)
        assert body["center_y"] == pytest.approx(3.0)
        assert body["size_x"] >= 12.0
        assert "Vina -9.40 kcal/mol" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_a_rescoring_failure_leaves_the_survivor_standing(self, tool, storage):
        fragments, pocket = await refs(storage)

        async def failing_dock(base_url, endpoint, json_data, timeout):
            raise RuntimeError("vina is down")

        tool._call_service_at = failing_dock
        route(
            tool,
            {
                "/constrained-pose": {"success": True, "results": [verified("CCO")]},
                "/molecule-gates": {"success": True, "results": []},
            },
        )

        result = await tool.execute(
            {
                "fragment_sdf_refs": fragments,
                "pocket_pdb_ref": pocket,
                "candidate_smiles": ["CCO"],
                "rescore_with_vina": True,
            }
        )

        assert result.metadata["num_verified"] == 1
        assert result.metadata["results"][0].get("vina_error")


class TestLedger:
    @pytest.mark.asyncio
    async def test_every_survivor_is_recorded_with_its_provenance(self, storage, monkeypatch):
        rows: list = []

        async def fake_store(**fields):
            rows.append(fields)
            return True

        monkeypatch.setattr("api.tools.fragment_link.store_candidate", fake_store)
        tool = FragmentLinkTool()
        tool._storage = storage
        fragments, pocket = await refs(storage)
        route(
            tool,
            {
                "/link-geometry": GEOMETRY,
                "/constrained-pose": lambda body: {
                    "success": True,
                    "results": [verified(s) for s in body["smiles"]],
                },
                "/molecule-gates": GATES,
            },
        )

        await tool.execute({"fragment_sdf_refs": fragments, "pocket_pdb_ref": pocket})

        assert len(rows) == 1
        assert rows[0]["smiles"] == "c1ccc(-c2ccncc2)cc1"
        assert rows[0]["origin"] == "fragment_link:library:phenyl"
        assert rows[0]["scores"]["strain_kcal_mol"] == 8.2
        assert rows[0]["scores"]["pose_fit_rmsd_angstrom"] == 0.24


class TestSchema:
    def test_references_are_annotated(self, tool):
        properties = tool.get_schema()["function"]["parameters"]["properties"]
        assert properties["fragment_sdf_refs"]["x-aevom-reference"] == {"data_type": "sdf"}
        assert properties["pocket_pdb_ref"]["x-aevom-reference"] == {"data_type": "pdb"}
        assert properties["candidate_csv_ref"]["x-aevom-reference"] == {"data_type": "csv"}

    def test_the_name_matches_the_schema(self, tool):
        assert tool.get_schema()["function"]["name"] == tool.name

    def test_the_description_says_rmsd_is_not_the_signal(self, tool):
        description = tool.get_schema()["function"]["description"].lower()
        assert "by construction" in description
