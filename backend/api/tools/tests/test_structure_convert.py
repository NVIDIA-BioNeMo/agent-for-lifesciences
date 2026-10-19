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

"""Tests for the convert_structure tool.

The interesting cases are the refusals. mmCIF to PDB is a lossy direction, and
the failure mode worth preventing is a file that parses but is wrong -- chains
silently renamed, atoms silently dropped -- because nothing downstream would
notice. So each PDB limit is checked here for a refusal that names the limit and
tells the model the original mmCIF reference still works.
"""

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.structure_convert import (
    MAX_PDB_RESIDUE_NUMBER,
    ConvertStructureTool,
)

PDB = "\n".join(
    [
        "ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00  0.00           N",
        "ATOM      2  CA  GLY A   1       2.000   1.000   1.000  1.00  0.00           C",
        "ATOM      3  C   GLY A   1       3.000   1.000   1.000  1.00  0.00           C",
        "ATOM      4  O   GLY A   1       4.000   1.000   1.000  1.00  0.00           O",
        "ATOM      5  N   ALA A   2       5.000   1.000   1.000  1.00  0.00           N",
        "ATOM      6  CA  ALA A   2       6.000   1.000   1.000  1.00  0.00           C",
        "END",
        "",
    ]
)

_MMCIF_HEADER = """data_boltz2
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
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def mmcif(
    chain: str = "A",
    residue_start: int = 1,
    residue_count: int = 4,
    atoms_per_residue: int = 4,
) -> str:
    """Build a small mmCIF with controllable chain id and residue numbering."""
    atom_names = ["N", "CA", "C", "O"]
    rows = []
    serial = 1
    for offset in range(residue_count):
        residue = residue_start + offset
        for index in range(atoms_per_residue):
            name = atom_names[index % len(atom_names)]
            rows.append(
                f"ATOM {serial} {name[0]} {name} . GLY {chain} 1 {residue} ? "
                f"{serial}.000 1.000 1.000 1.00 0.00 {residue} {chain} 1"
            )
            serial += 1
    return _MMCIF_HEADER + "\n".join(rows) + "\n#\n"


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = ConvertStructureTool()
    instance._storage = storage
    return instance


@pytest.mark.asyncio
class TestConversion:
    async def test_mmcif_becomes_a_pdb_reference(self, tool, storage):
        source = await storage.store(
            data_type="mmcif",
            content=mmcif().encode(),
            metadata={"affinity_pic50": 6.4, "confidence_score": 0.9},
            description="Boltz-2 complex",
        )

        result = await tool.execute(
            {"structure_ref": source, "target_format": "pdb"}
        )

        assert result.content_type == ContentType.DATA_REFERENCE
        assert result.data.startswith("pdb_")
        written = (await storage.retrieve(result.data)).decode()
        assert written.startswith("ATOM")
        assert "END" in written
        assert result.metadata["atom_count"] == 16
        assert result.metadata["chain_ids"] == ["A"]

    async def test_pdb_becomes_an_mmcif_reference(self, tool, storage):
        source = await storage.store(data_type="pdb", content=PDB.encode())

        result = await tool.execute(
            {"structure_ref": source, "target_format": "mmcif"}
        )

        assert result.data.startswith("mmcif_")
        written = (await storage.retrieve(result.data)).decode()
        assert written.startswith("data_")
        assert "_atom_site." in written

    async def test_the_affinity_numbers_survive_the_conversion(self, tool, storage):
        # Converting to dock a Boltz-2 complex should not lose the numbers that
        # made it worth docking; the findings ledger reads these off the metadata.
        source = await storage.store(
            data_type="mmcif",
            content=mmcif().encode(),
            metadata={"affinity_pic50": 6.4, "source": "boltz2"},
        )

        result = await tool.execute(
            {"structure_ref": source, "target_format": "pdb"}
        )

        _, metadata = await storage.retrieve_with_metadata(result.data)
        assert metadata["affinity_pic50"] == 6.4
        assert metadata["source_ref"] == source

    async def test_the_result_names_the_tools_the_new_format_unlocks(
        self, tool, storage
    ):
        source = await storage.store(data_type="mmcif", content=mmcif().encode())

        result = await tool.execute(
            {"structure_ref": source, "target_format": "pdb"}
        )

        message = result.metadata["message"]
        assert "vina_dock" in message
        assert "proteinmpnn_design_sequence" in message

    async def test_mmcif_stored_under_a_pdb_prefix_is_still_read_as_mmcif(
        self, tool, storage
    ):
        # openfold3 has historically stored CIF text under a `pdb` ref. Trusting
        # the prefix would hand mmCIF to the PDB parser, which skips unknown
        # lines and yields an empty structure rather than an error.
        source = await storage.store(data_type="pdb", content=mmcif().encode())

        result = await tool.execute(
            {"structure_ref": source, "target_format": "pdb"}
        )

        assert result.content_type == ContentType.DATA_REFERENCE
        assert result.metadata["source_format"] == "mmcif"


@pytest.mark.asyncio
class TestRefusals:
    async def _error(self, tool, arguments) -> str:
        result = await tool.execute(arguments)
        assert result.content_type == ContentType.ERROR
        return result.data

    async def test_a_long_chain_id_cannot_be_written_as_pdb(self, tool, storage):
        # Boltz-2 permits 4-character chain ids; PDB has one column for them.
        source = await storage.store(
            data_type="mmcif", content=mmcif(chain="LIG1").encode()
        )

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "pdb"}
        )

        assert "single-character chain ids" in message
        assert "LIG1" in message
        assert source in message
        assert "visualize_pdb" in message

    async def test_an_out_of_range_residue_number_is_refused(self, tool, storage):
        source = await storage.store(
            data_type="mmcif",
            content=mmcif(residue_start=MAX_PDB_RESIDUE_NUMBER + 1).encode(),
        )

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "pdb"}
        )

        assert str(MAX_PDB_RESIDUE_NUMBER) in message
        assert source in message

    async def test_the_atom_limit_is_reported_rather_than_truncated(
        self, tool, storage, monkeypatch
    ):
        # The real limit needs 100k atoms to reach, so the constant is lowered
        # instead of building a structure that size.
        monkeypatch.setattr(
            "api.tools.structure_convert.MAX_PDB_ATOMS", 4, raising=True
        )
        source = await storage.store(data_type="mmcif", content=mmcif().encode())

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "pdb"}
        )

        assert "16 atoms" in message
        assert "PDB cannot represent it" in message

    async def test_converting_to_the_format_it_already_is_is_refused(
        self, tool, storage
    ):
        source = await storage.store(data_type="mmcif", content=mmcif().encode())

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "mmcif"}
        )

        assert "already MMCIF" in message

    async def test_an_unparseable_structure_is_a_clean_error(self, tool, storage):
        source = await storage.store(
            data_type="mmcif", content=b"data_x\n_atom_site.id\n"
        )

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "pdb"}
        )

        assert "Could not parse" in message
        assert source in message

    async def test_a_structure_with_no_coordinates_is_a_clean_error(
        self, tool, storage
    ):
        # A parseable mmCIF whose atom loop is empty. The PDB parser tolerates
        # this and yields nothing, so an explicit check is what turns it into a
        # message instead of an empty file.
        source = await storage.store(
            data_type="mmcif", content=(_MMCIF_HEADER + "#\n").encode()
        )

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "pdb"}
        )

        assert "no atoms" in message

    async def test_an_empty_artifact_is_a_clean_error(self, tool, storage):
        source = await storage.store(data_type="mmcif", content=b"   \n")

        message = await self._error(
            tool, {"structure_ref": source, "target_format": "pdb"}
        )

        assert "is empty" in message

    async def test_a_missing_structure_ref_is_rejected(self, tool):
        message = await self._error(tool, {"target_format": "pdb"})
        assert "structure_ref" in message

    async def test_an_unknown_target_format_is_rejected(self, tool, storage):
        source = await storage.store(data_type="mmcif", content=mmcif().encode())
        message = await self._error(
            tool, {"structure_ref": source, "target_format": "sdf"}
        )
        assert "'pdb' or 'mmcif'" in message

    async def test_an_unknown_reference_points_at_the_producers(self, tool):
        message = await self._error(
            tool, {"structure_ref": "mmcif_missing1", "target_format": "pdb"}
        )
        assert "boltz2" in message


class TestSchema:
    def test_the_description_says_when_conversion_is_unnecessary(self):
        # Converting purely to display a structure wastes a turn and an artifact.
        description = ConvertStructureTool().get_schema()["function"]["description"]
        assert "visualize_pdb" in description
        assert "structure_query" in description

    def test_target_format_is_constrained_to_the_two_real_options(self):
        properties = ConvertStructureTool().get_schema()["function"]["parameters"][
            "properties"
        ]
        assert properties["target_format"]["enum"] == ["pdb", "mmcif"]
