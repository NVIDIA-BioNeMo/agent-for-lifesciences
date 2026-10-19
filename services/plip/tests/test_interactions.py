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

"""Tests for PLIP interaction profiling.

The fixture is the donepezil binding site of 1EVE, cut to residues within 12 A of
the ligand. That trim was verified to reproduce the full structure's nine
interactions for E20 exactly, so the numbers asserted here are the ones PLIP
gives for the deposited complex -- a well-characterized site with hydrophobic
contacts, a hydrogen bond, two pi-stackings against Trp84 and Trp279, a pi-cation
to Phe330 and a water bridge. That mixture is why it makes a good fixture: it
exercises six of the eight families in one structure.

The glycans in the fixture lost their binding partners in the trim, which is
useful rather than accidental: they give a multi-ligand structure where the
ligand anyone cares about is not the first one PLIP reports.
"""

import pathlib
import tempfile

import pytest

from app.interactions import (
    FAMILIES,
    Interaction,
    ProfilingError,
    biolip_artifacts,
    configure,
    profile,
)

FIXTURE = pathlib.Path(__file__).parent / "1eve_site.pdb"
DONEPEZIL = "E20:A:2001"

# The nine coarsened interactions PLIP reports for donepezil in 1EVE.
EXPECTED_FINGERPRINT = {
    "A:84:TRP:hydrophobic",
    "A:84:TRP:pistacking",
    "A:121:TYR:waterbridge",
    "A:279:TRP:pistacking",
    "A:288:PHE:hbond",
    "A:330:PHE:hydrophobic",
    "A:330:PHE:pication",
    "A:331:PHE:hydrophobic",
    "A:334:TYR:hydrophobic",
}


@pytest.fixture(scope="module")
def structure() -> str:
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def result(structure):
    return profile(structure)


class TestProfiling:
    def test_finds_the_expected_interactions(self, result):
        site = result.site()
        assert set(site.fingerprint) == EXPECTED_FINGERPRINT

    def test_designates_the_real_ligand_over_the_glycans(self, result):
        # The glycans come back as ligands too. Profiling one of those instead
        # would answer confidently about the wrong molecule.
        assert result.designated == DONEPEZIL
        assert "most interactions" in result.designation_reason

    def test_reports_every_ligand_it_found(self, result):
        identifiers = {ligand.identifier for ligand in result.ligands}
        assert DONEPEZIL in identifiers
        assert len(identifiers) > 1

    def test_orders_ligands_by_interaction_count(self, result):
        counts = [len(ligand.interactions) for ligand in result.ligands]
        assert counts == sorted(counts, reverse=True)

    def test_recovers_the_ligand_identity(self, result):
        site = result.site()
        assert site.hetid == "E20"
        assert site.chain == "A"
        assert site.position == 2001
        assert site.num_heavy_atoms == 28
        assert site.num_rings == 2
        # PLIP perceives the ligand's bonds, so the SMILES is a real check that
        # the right het group was picked up rather than a fragment of it.
        assert "OC" in site.smiles.upper()

    def test_counts_interactions_per_family(self, result):
        counts = result.site().counts()
        assert counts["hydrophobic"] == 5
        assert counts["pistacking"] == 2
        assert counts["hbond"] == 1
        assert counts["pication"] == 1
        assert counts["waterbridge"] == 1
        # Families with nothing to report are absent rather than zero.
        assert "halogen" not in counts

    def test_groups_interactions_by_residue(self, result):
        residues = {
            entry["residue_number"]: entry for entry in result.site().residues()
        }
        assert residues[84]["residue_name"] == "TRP"
        assert residues[84]["families"] == ["hydrophobic", "pistacking"]
        assert list(residues) == sorted(residues)

    def test_no_warnings_on_a_clean_structure(self, result):
        assert result.warnings == []


class TestCoarsening:
    def test_the_key_is_pose_independent(self):
        # Atom indices differ between poses of the same ligand; chain, residue
        # and family do not. That is the whole basis of the consensus fold.
        interaction = Interaction(
            family="hbond", chain="A", residue_number=288, residue_name="PHE"
        )
        assert interaction.key == "A:288:PHE:hbond"

    def test_duplicate_contacts_to_one_residue_collapse(self, result):
        site = result.site()
        # Five hydrophobic contacts, but only four distinct residues: two of
        # them are to the same residue and are one fact about the pose.
        hydrophobic = [i for i in site.interactions if i.family == "hydrophobic"]
        assert len(hydrophobic) == 5
        assert len({i.key for i in hydrophobic}) == 4

    def test_detail_keeps_the_geometry_and_drops_the_atom_indices(self, result):
        hbond = next(i for i in result.site().interactions if i.family == "hbond")
        assert isinstance(hbond.detail["dist_d-a"], float)
        assert hbond.detail["don_angle"] > 90
        assert not any("idx" in key for key in hbond.detail)
        assert "ligcoo" not in hbond.detail

    def test_distances_are_numbers_not_strings(self, result):
        # PLIP reports them as strings, which would silently break any
        # downstream comparison.
        for interaction in result.site().interactions:
            for key, value in interaction.detail.items():
                if key.startswith("dist"):
                    assert isinstance(value, float), (key, value)

    def test_every_family_has_a_label(self):
        from app.interactions import FAMILY_LABELS

        assert set(FAMILY_LABELS) == set(FAMILIES)


class TestLigandSelection:
    def test_honours_a_requested_hetid(self, structure):
        result = profile(structure, ligand="NAG")
        assert result.site().hetid == "NAG"
        assert "requested" in result.designation_reason

    def test_honours_a_full_identifier(self, structure):
        result = profile(structure, ligand=DONEPEZIL)
        assert result.designated == DONEPEZIL

    def test_a_missing_ligand_falls_back_and_says_so(self, structure):
        # Silently profiling a different ligand is the failure worth avoiding:
        # the answer would look authoritative and be about the wrong molecule.
        result = profile(structure, ligand="XYZ")
        assert result.designated == DONEPEZIL
        assert result.warnings
        assert "XYZ" in result.warnings[0]

    def test_plip_supplies_the_irrelevance_list(self):
        artifacts = biolip_artifacts()
        assert "NAG" in artifacts
        assert "E20" not in artifacts

    def test_docked_ligand_placeholders_are_not_demoted(self):
        # Vina writes its pose as UNL, which PLIP's list calls irrelevant --
        # true of a crystal structure, false of a complex we just docked. Left
        # in, a crystallographic cofactor would outrank the computed pose.
        artifacts = biolip_artifacts()
        assert "UNL" not in artifacts
        assert "LIG" not in artifacts

    def test_a_docked_pose_named_unl_is_still_the_designated_ligand(self, structure):
        # End to end for the case above: rename the ligand to what vina_dock
        # writes and it must still win over the glycans in the same file.
        as_unl = structure.replace("E20 A", "UNL A").replace("E20 ", "UNL ")
        result = profile(as_unl)
        assert result.designated is not None
        assert result.designated.startswith("UNL")
        assert len(result.site().fingerprint) == len(EXPECTED_FINGERPRINT)


class TestConfiguration:
    def test_the_temp_file_leak_is_switched_off(self, result):
        # PLIP writes a randomly named "fixed" copy of every structure it reads
        # and never deletes it. Left on, a long-running server leaks one file per
        # request. `result` has already run an analysis, so this also proves the
        # setting is applied before the first one rather than after.
        from plip.basic import config

        assert config.NOFIXFILE is True

    def test_profiling_leaves_no_randomly_named_files_behind(self, structure):
        temp = pathlib.Path(tempfile.gettempdir())
        before = set(temp.glob("plipfixed.*"))
        profile(structure)
        assert set(temp.glob("plipfixed.*")) == before

    def test_configure_is_idempotent(self):
        configure()
        configure()
        from plip.basic import config

        assert config.NOFIXFILE is True


class TestFailures:
    def test_garbage_input_is_an_error_not_a_crash(self):
        # PLIP calls sys.exit() here, which would take the server down.
        with pytest.raises(ProfilingError) as excinfo:
            profile("this is not a structure")
        assert "not recognized" in str(excinfo.value)

    def test_the_message_is_not_just_an_exit_code(self):
        with pytest.raises(ProfilingError) as excinfo:
            profile("this is not a structure")
        assert str(excinfo.value).strip() != "PLIP could not read the structure: 1"

    def test_a_protein_with_no_ligand_profiles_as_empty(self, structure):
        # Dropping every HETATM leaves a valid protein with nothing bound.
        protein_only = "\n".join(
            line
            for line in structure.splitlines()
            if not line.startswith(("HETATM", "CONECT", "HET   ", "LINK  "))
        )
        result = profile(protein_only)
        assert result.ligands == []
        assert result.designated is None
        assert result.warnings
        assert "HETATM" in result.warnings[0]
