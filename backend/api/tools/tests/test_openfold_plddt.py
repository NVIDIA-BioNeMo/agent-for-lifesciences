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

"""Tests for OpenFold3's per-residue pLDDT offload.

OpenFold3's scores carry only a mean pLDDT; the per-residue profile arrives in
the predicted PDB's CA B-factor column. The tool offloads it as an ``array``
artifact so the model calls ``aevom.load_array(ref)`` instead of re-parsing the
structure (which trips biotite's default parse that drops B-factors).
"""

import io

import numpy as np
import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.openfold import OpenFold3Tool


@pytest.fixture
def tool():
    t = OpenFold3Tool()
    t._storage = DataStorageService(repository=InMemoryDataArtifactRepository())
    return t


def _pdb_line(serial, name, resi, b):
    line = "ATOM  " + f"{serial:>5} " + f"{name:<4}" + " " + "ALA" + " A"
    line += f"{resi:>4}" + "    " + f"{11.0:8.3f}{12.0:8.3f}{13.0:8.3f}"
    line += f"{1.0:6.2f}{b:6.2f}" + " " * 10 + f"{name[0]:>2}"
    return line


def _pred_pdb(plddts):
    """One residue per pLDDT, each with a CA (carrying the value) and an N."""
    lines = []
    serial = 1
    for i, b in enumerate(plddts, 1):
        lines.append(_pdb_line(serial, "CA", i, b))
        serial += 1
        lines.append(_pdb_line(serial, "N", i, b))  # non-CA, must be ignored
        serial += 1
    return "\n".join(lines) + "\nEND\n"


@pytest.mark.asyncio
async def test_offloads_one_plddt_per_residue(tool):
    ref = await tool._store_per_residue_plddt(
        _pred_pdb([55.0, 82.0, 90.5]), "pred1", "pdb_abc12345", "ubiquitin"
    )
    assert ref and ref.startswith("array_")
    data = await tool._storage.retrieve(ref)
    with np.load(io.BytesIO(data)) as archive:
        arr = archive["data"]
    # One value per residue (CA only), in order -- not one per atom.
    assert [round(float(x), 1) for x in arr] == [55.0, 82.0, 90.5]


@pytest.mark.asyncio
async def test_no_ca_atoms_returns_none(tool):
    assert await tool._store_per_residue_plddt(
        "HEADER    X\nEND\n", "pred1", "pdb_abc12345", None
    ) is None


@pytest.mark.asyncio
async def test_missing_storage_returns_none():
    t = OpenFold3Tool()
    t._storage = None
    assert await t._store_per_residue_plddt(
        _pred_pdb([70.0]), "p", "r", None
    ) is None
