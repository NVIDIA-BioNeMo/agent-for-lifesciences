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

"""Tests for getting a structure into the format PLIP reads.

The case that matters is Boltz-2: it emits mmCIF only, and it emits ten diffusion
samples per prediction. If the mmCIF path is broken, interaction profiling is
unavailable for exactly the tool that most needs it, so the round trip is checked
by profiling both formats and comparing the fingerprints rather than by
inspecting the converted text.
"""

import pathlib

import pytest

from app.formats import ConversionError, sniff, to_pdb
from app.interactions import profile

FIXTURE = pathlib.Path(__file__).parent / "1eve_site.pdb"


@pytest.fixture(scope="module")
def pdb_text() -> str:
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def mmcif_text(pdb_text) -> str:
    """The same structure as mmCIF, written by gemmi."""
    import gemmi

    structure = gemmi.read_pdb_string(pdb_text)
    structure.setup_entities()
    return structure.make_mmcif_document().as_string()


class TestSniffing:
    def test_recognizes_pdb(self, pdb_text):
        assert sniff(pdb_text) == "pdb"

    def test_recognizes_mmcif(self, mmcif_text):
        assert sniff(mmcif_text) == "mmcif"

    def test_skips_leading_comments_and_blanks(self):
        assert sniff("# a comment\n\ndata_XYZ\n_cell.length_a 1\n") == "mmcif"

    def test_defaults_to_pdb_on_empty_text(self):
        assert sniff("") == "pdb"


class TestConversion:
    def test_pdb_passes_through_untouched(self, pdb_text):
        assert to_pdb(pdb_text) == pdb_text

    def test_mmcif_becomes_pdb(self, mmcif_text):
        converted = to_pdb(mmcif_text)
        assert sniff(converted) == "pdb"
        assert "HETATM" in converted

    def test_the_round_trip_preserves_the_interactions(self, pdb_text, mmcif_text):
        # The real assertion: a Boltz-2 style mmCIF must profile identically to
        # the PDB it came from, or the conversion is losing the ligand's bonds,
        # its HETATM flag, or the waters that mediate a bridge.
        from_pdb = profile(pdb_text).site()
        from_mmcif = profile(to_pdb(mmcif_text)).site()
        assert from_mmcif is not None
        assert set(from_mmcif.fingerprint) == set(from_pdb.fingerprint)

    def test_an_empty_structure_is_refused(self):
        with pytest.raises(ConversionError) as excinfo:
            to_pdb("   \n\n")
        assert "empty" in str(excinfo.value)

    def test_unparseable_mmcif_is_refused_with_a_reason(self):
        # gemmi returns a model-less structure here rather than raising, and
        # calling shorten_chain_names() on that segfaults. If this test ever
        # crashes the interpreter instead of failing, the guard is gone.
        with pytest.raises(ConversionError) as excinfo:
            to_pdb("data_broken\nloop_\n_atom_site.nonsense\n???\n")
        assert "no models" in str(excinfo.value)

    def test_an_mmcif_header_with_no_atoms_is_refused(self):
        with pytest.raises(ConversionError):
            to_pdb("data_empty\n_cell.length_a   1.000\n")
