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

"""Reading the data type back out of a reference ID.

A reference is ``{data_type}_{hex}``, but some type names contain an
underscore. Splitting on the first one read ``annotated_cif_…`` as type
``annotated``, so ``visualize_pdb`` sent RFD4-Proteina designs to Mol* as PDB,
which parses mmCIF text as zero atoms and shows an empty viewer.
"""

import pytest
from api.storage.datatypes import DataType
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.service import (
    DataStorageService,
    InvalidReferenceError,
    generate_reference_id,
)
from api.tools.visualization import VisualizePdbTool

ANNOTATED_CIF = "data_DESIGN\n#\nloop_\n_atom_site.id\n_atom_site.Cartn_x\n1 0.0\n"


@pytest.fixture
def storage() -> DataStorageService:
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.mark.parametrize("data_type", [dt.value for dt in DataType])
def test_every_data_type_round_trips_through_its_reference(storage, data_type):
    assert storage.get_data_type(generate_reference_id(data_type)) == data_type


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("annotated_cif_1ff60a8d", "annotated_cif"),
        ("skill_file_a1b2c3d4", "skill_file"),
        ("sdf_winner_001", "sdf"),
        ("pdb_abc123", "pdb"),
    ],
)
def test_known_references(storage, reference, expected):
    assert storage.get_data_type(reference) == expected


def test_reference_without_id_is_rejected(storage):
    with pytest.raises(InvalidReferenceError):
        storage.get_data_type("pdb")


@pytest.mark.asyncio
async def test_visualize_pdb_sends_annotated_cif_as_mmcif(storage):
    ref = await storage.store("annotated_cif", ANNOTATED_CIF.encode())
    tool = VisualizePdbTool()
    tool.set_storage(storage)

    result = await tool.execute({"pdb_ref": ref})

    assert result.metadata["format"] == "mmcif"


@pytest.mark.asyncio
async def test_visualize_pdb_reports_a_designs_chain_roles(storage):
    # The viewer colours a design by what each chain is, not by its letter.
    ref = await storage.store(
        "annotated_cif",
        ANNOTATED_CIF.encode(),
        metadata={"binder_chains": ["B"], "target_chains": ["A"]},
    )
    tool = VisualizePdbTool()
    tool.set_storage(storage)

    result = await tool.execute({"pdb_ref": ref})

    assert result.metadata["roles"] == {"target": ["A"], "binder": ["B"]}


@pytest.mark.asyncio
async def test_visualize_pdb_has_no_roles_for_an_ordinary_structure(storage):
    ref = await storage.store("annotated_cif", ANNOTATED_CIF.encode())
    tool = VisualizePdbTool()
    tool.set_storage(storage)

    result = await tool.execute({"pdb_ref": ref})

    assert "roles" not in result.metadata
