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

"""Tests for the fragment_exit_vectors tool.

The geometry itself belongs to the rdkit suite
(``services/rdkit/tests/test_fragment_link.py``), which checks it against PDBBind
cases with known attachment points. What matters here is the handoff: that
fragment SDFs are fetched by reference and sent as content, that the SAFE
templates come back as references genmol_generate_molecules can take, and that
the inline text says enough to choose a pair without re-reading the geometry.
"""

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.fragment_exit_vectors import FragmentExitVectorsTool

FRAGMENT_SDF = """benzene
  RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    1.0000    0.0000    0.0000 C   0  0
    0.5000    0.8660    0.0000 C   0  0
   -0.5000    0.8660    0.0000 C   0  0
   -1.0000    0.0000    0.0000 C   0  0
   -0.5000   -0.8660    0.0000 C   0  0
    0.5000   -0.8660    0.0000 C   0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
"""


def pair(pair_id: str = "p0", safe: str | None = "c1ccccc1.[*{3-6}].c1ccccc1") -> dict:
    return {
        "pair_id": pair_id,
        "site_a": "f0a1",
        "site_b": "f1a3",
        "fragment_a": 0,
        "fragment_b": 1,
        "atom_a": 1,
        "atom_b": 3,
        "anchor_distance": 8.4,
        "tip_distance": 6.1,
        "exit_vector_angle": 24.0,
        "alignment_a": 0.91,
        "alignment_b": 0.88,
        "tube_clearance": 3.4,
        "linker_atoms_min": 5,
        "linker_atoms_max": 9,
        "rb_headroom": 6,
        "saturated_chain_rb_cost": 6,
        "requires_rigid_linker": False,
        "score": 0.82,
        "templates": {
            "safe": safe,
            "safe_error": None if safe else "could not encode",
            "marker": "c1ccccc1(*).* c1ccccc1",
            "marker_error": None,
            "mask_min": 6,
            "mask_max": 13,
        },
        "linkers": [
            {"name": "phenyl", "rb_cost": 2, "span_min": 5.8, "span_max": 6.4},
            {"name": "alkyne", "rb_cost": 0, "span_min": 6.2, "span_max": 6.6},
        ],
        "warnings": [],
    }


def response(pairs: list[dict] | None = None, **overrides) -> dict:
    payload = {
        "success": True,
        "fragments": [
            {"index": 0, "smiles": "c1ccccc1", "num_sites": 6, "sites": []},
            {"index": 1, "smiles": "c1ccncc1", "num_sites": 5, "sites": []},
        ],
        "pairs": pairs if pairs is not None else [pair()],
        "candidates": [],
        "growth": None,
        "fragment_separation": [
            {"fragment_a": 0, "fragment_b": 1, "closest_heavy_distance": 6.9}
        ],
        "rotatable_bonds_used": 0,
        "rb_headroom": 8,
        "rb_max": 10,
        "pocket_atoms": 871,
        "warnings": [],
        "error": None,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = FragmentExitVectorsTool()
    instance._storage = storage
    return instance


def stub(tool, payload, calls: list | None = None):
    async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        if calls is not None:
            calls.append(json_data)
        if isinstance(payload, Exception):
            raise payload
        return payload

    tool._call_service = _call


async def fragment_refs(storage, count: int = 2) -> list[str]:
    return [
        await storage.store("sdf", FRAGMENT_SDF.encode()) for _ in range(count)
    ]


class TestInputHandling:
    @pytest.mark.asyncio
    async def test_requires_fragments(self, tool):
        result = await tool.execute({})
        assert result.content_type == ContentType.ERROR
        assert "fragment_sdf_refs" in result.data

    @pytest.mark.asyncio
    async def test_a_bad_reference_is_named(self, tool):
        result = await tool.execute({"fragment_sdf_refs": ["not-a-reference"]})
        assert result.content_type == ContentType.ERROR
        assert "not-a-reference" in result.data

    @pytest.mark.asyncio
    async def test_a_missing_artifact_is_distinguished_from_a_bad_reference(self, tool):
        result = await tool.execute({"fragment_sdf_refs": ["sdf_deadbeef"]})
        assert result.content_type == ContentType.ERROR
        assert "well formed" in result.data

    @pytest.mark.asyncio
    async def test_the_service_gets_content_not_references(self, tool, storage):
        refs = await fragment_refs(storage)
        pocket = await storage.store("pdb", b"ATOM      1  N   ALA A   1\nEND\n")
        calls: list = []
        stub(tool, response(), calls)

        await tool.execute({"fragment_sdf_refs": refs, "pocket_pdb_ref": pocket})

        sent = calls[0]
        assert sent["fragment_sdfs"][0].startswith("benzene")
        assert "ATOM" in sent["pocket_pdb"]
        assert refs[0] not in str(sent)

    @pytest.mark.asyncio
    async def test_hotspots_are_passed_through_as_floats(self, tool, storage):
        refs = await fragment_refs(storage, 1)
        calls: list = []
        stub(tool, response(pairs=[]), calls)

        await tool.execute(
            {"fragment_sdf_refs": refs, "target_points": [[1, 2, 3]]}
        )
        assert calls[0]["target_points"] == [[1.0, 2.0, 3.0]]

    @pytest.mark.asyncio
    async def test_a_service_failure_is_reported_not_raised(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, RuntimeError("rdkit is down"))
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert result.content_type == ContentType.ERROR
        assert "rdkit is down" in result.data

    @pytest.mark.asyncio
    async def test_a_service_level_error_is_surfaced(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, {"success": False, "error": "fragment 1: no 3D coordinates"})
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert result.content_type == ContentType.ERROR
        assert "no 3D coordinates" in result.data


class TestTemplates:
    @pytest.mark.asyncio
    async def test_the_template_is_stored_as_a_safe_reference(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, response())

        result = await tool.execute({"fragment_sdf_refs": refs})

        assert result.content_type == ContentType.DATA_REFERENCE
        stored = result.metadata["template_refs"]
        assert list(stored) == ["p0"]
        content = await storage.retrieve(stored["p0"])
        assert content.decode() == "c1ccccc1.[*{3-6}].c1ccccc1"

    @pytest.mark.asyncio
    async def test_the_message_points_at_genmol(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, response())
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert "genmol_generate_molecules" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_a_pair_without_a_safe_form_stores_nothing(self, tool, storage):
        """The marker form still travels inline; only SAFE is a reference."""
        refs = await fragment_refs(storage)
        stub(tool, response(pairs=[pair(safe=None)]))

        result = await tool.execute({"fragment_sdf_refs": refs})

        assert result.content_type == ContentType.TEXT
        assert result.metadata["template_refs"] == {}
        assert "marker form" in result.data

    @pytest.mark.asyncio
    async def test_only_the_best_few_pairs_get_references(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, response(pairs=[pair(f"p{i}") for i in range(6)]))
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert len(result.metadata["template_refs"]) == 3


class TestMessage:
    @pytest.mark.asyncio
    async def test_the_geometry_is_reported_inline(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, response())
        result = await tool.execute({"fragment_sdf_refs": refs})
        message = result.metadata["message"]
        assert "8.4 A apart" in message
        assert "24.0 deg apart" in message
        assert "5-9 atoms" in message
        assert "phenyl (2 RB)" in message

    @pytest.mark.asyncio
    async def test_a_rigid_bridge_requirement_is_spelled_out(self, tool, storage):
        refs = await fragment_refs(storage)
        rigid = pair()
        rigid["requires_rigid_linker"] = True
        stub(tool, response(pairs=[rigid]))
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert "has to be rigid" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_growth_vectors_are_reported_for_one_fragment(self, tool, storage):
        refs = await fragment_refs(storage, 1)
        stub(
            tool,
            response(
                pairs=[],
                growth={
                    "rb_headroom": 8,
                    "vectors": [
                        {
                            "site_id": "f0a1",
                            "atom_index": 1,
                            "free_depth": 7.5,
                            "blocked": False,
                            "score": 0.7,
                            "linker_atoms_min": 4,
                            "linker_atoms_max": 8,
                            "mask_min": 5,
                            "mask_max": 12,
                            "templates": {"safe": None, "marker": "c1ccccc1(*)"},
                        }
                    ],
                },
            ),
        )
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert "Growth vectors" in result.data
        assert "7.5 A of open space" in result.data

    @pytest.mark.asyncio
    async def test_assembled_candidates_come_with_a_warning_to_verify(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(
            tool,
            response(
                candidates=[
                    {
                        "smiles": "c1ccc(-c2ccccc2)cc1",
                        "pair_id": "p0",
                        "linker": "phenyl",
                        "linker_atoms": 6,
                        "linker_rb_cost": 2,
                        "anchor_distance": 8.4,
                        "atom_a": 1,
                        "atom_b": 3,
                    }
                ]
            ),
        )
        result = await tool.execute(
            {"fragment_sdf_refs": refs, "assemble_candidates": True}
        )
        assert "Verify these with fragment_link" in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_a_missing_pocket_warning_reaches_the_model(self, tool, storage):
        refs = await fragment_refs(storage)
        stub(tool, response(warnings=["No pocket supplied, so free depth is unknown."]))
        result = await tool.execute({"fragment_sdf_refs": refs})
        assert "No pocket supplied" in result.metadata["message"]


class TestSchema:
    def test_references_are_annotated_for_the_reference_resolver(self, tool):
        properties = tool.get_schema()["function"]["parameters"]["properties"]
        assert properties["fragment_sdf_refs"]["x-aevom-reference"] == {"data_type": "sdf"}
        assert properties["pocket_pdb_ref"]["x-aevom-reference"] == {"data_type": "pdb"}

    def test_the_name_matches_the_schema(self, tool):
        assert tool.get_schema()["function"]["name"] == tool.name
