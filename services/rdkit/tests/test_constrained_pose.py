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

"""Tests for constrained-pose verification.

The synthetic cases build their own ground truth: take a molecule, embed it once,
carve two rings out of that conformer, and hand them back as posed fragments. The
molecule they came from must then verify, and molecules that cannot hold those two
rings where they were must not.

The committed PDBBind cases carry the rest. ``1uwh`` is sorafenib cut into
chlorophenyl and methylamide-pyridine with the urea-phenoxy bridge hidden -- a 9.4
A span, the regime SMDD's two-fragment instances live in. The wider sweep behind
the thresholds these tests assert lives in ``test_pdbbind_validation.py``.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from app import constrained_pose as cp
from app.main import app
from tests.pdbbind_cases import fixture_ids, load_fixture

client = TestClient(app)


def carve(mol: Chem.Mol, keep: tuple[int, ...]) -> Chem.Mol:
    """The sub-molecule on ``keep``, carrying its coordinates from ``mol``."""
    editable = Chem.RWMol(mol)
    for index in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
        editable.RemoveAtom(index)
    carved = editable.GetMol()
    Chem.SanitizeMol(carved)
    return carved


def posed_case(smiles: str, seed: int = 0xBEEF) -> tuple[Chem.Mol, list[Chem.Mol]]:
    """A molecule with a 3D pose, and its two rings as separately posed fragments."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) >= 2, smiles
    return mol, [carve(mol, ring) for ring in rings[:2]]


def sdf(mol: Chem.Mol) -> str:
    return Chem.MolToMolBlock(mol)


# A pair of benzenes 9 A apart along x, in a molecule that cannot possibly join
# them, for the failure paths that do not need a real case.
FAR_APART = "c1ccccc1"


@pytest.fixture(scope="module")
def diphenylpropane():
    """1,3-diphenylpropane: two rings, a three-carbon bridge, no ambiguity."""
    return posed_case("c1ccc(CCCc2ccccc2)cc1")


@pytest.fixture(scope="module")
def sorafenib():
    return load_fixture("1uwh")


class TestAssignment:
    def test_a_missing_fragment_is_gate_two(self, diphenylpropane):
        _, fragments = diphenylpropane
        result = cp.verify_one("CCCCCC", fragments, np.zeros((0, 3)))
        assert result.failed_gate == "gate2_substructure"
        assert "not a substructure" in result.reason

    def test_fused_fragments_are_gate_three(self):
        # Naphthalene contains two six-rings, but they share two atoms, so no
        # assignment of two separate benzenes onto it can be disjoint.
        mol, fragments = posed_case("c1ccc2ccccc2c1")
        result = cp.verify_one("c1ccc2ccccc2c1", fragments, np.zeros((0, 3)))
        assert result.failed_gate == "gate3_disjoint"
        assert "share atoms" in result.reason

    def test_a_disconnected_candidate_is_gate_three(self, diphenylpropane):
        _, fragments = diphenylpropane
        result = cp.verify_one("c1ccccc1.c1ccccc1", fragments, np.zeros((0, 3)))
        assert result.failed_gate == "gate3_disjoint"
        assert "covalently" in result.reason

    def test_unparseable_smiles_is_gate_one(self, diphenylpropane):
        _, fragments = diphenylpropane
        result = cp.verify_one("not a molecule", fragments, np.zeros((0, 3)))
        assert result.failed_gate == "gate1_smiles"

    def test_the_assignment_is_disjoint_and_complete(self, diphenylpropane):
        mol, fragments = diphenylpropane
        matches, gate, reason = cp.assign_fragments(
            Chem.MolFromSmiles(Chem.MolToSmiles(mol)), fragments
        )
        assert gate is None, reason
        assert len(matches) == 2
        first, second = (set(match.atom_indices) for match in matches)
        assert not first & second

    def test_the_geometrically_possible_assignment_wins(self, diphenylpropane):
        """Two identical rings means several disjoint assignments; only one fits."""
        mol, fragments = diphenylpropane
        candidate = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        matches, gate, _ = cp.assign_fragments(candidate, fragments)
        assert gate is None
        graph = Chem.GetDistanceMatrix(candidate)
        deficit = cp._reach_deficit(
            candidate, graph, cp._coord_map(matches, fragments)
        )
        assert deficit == 0.0

    def test_a_bridge_that_cannot_reach_is_named_as_such(self, sorafenib):
        fragments = [
            Chem.RemoveHs(Chem.MolFromMolBlock(block)) for block in sorafenib.fragment_sdfs
        ]
        # A single bond between two fragments whose anchors are 9.4 A apart.
        joined = f"{sorafenib.fragment_smiles[0]}.{sorafenib.fragment_smiles[1]}"
        direct = Chem.MolToSmiles(Chem.MolFromSmiles(joined)).replace(".", "")
        result = cp.verify_one(direct, fragments, np.zeros((0, 3)))
        assert result.failed_gate is not None
        if result.failed_gate == "gate5c_pose":
            assert "short" in result.reason or "cannot span" in result.reason


class TestPose:
    def test_the_molecule_its_fragments_came_from_verifies(self, diphenylpropane):
        mol, fragments = diphenylpropane
        result = cp.verify_one(Chem.MolToSmiles(mol), fragments, np.zeros((0, 3)))
        assert result.verified, result.reason
        assert result.max_fragment_rmsd < cp.DEFAULT_RMSD_MAX
        assert result.num_linker_atoms == 3
        assert result.force_field == "MMFF94"

    def test_the_pose_comes_back_in_the_pocket_frame(self, diphenylpropane):
        """Regression: coordMap constrains distances, not position.

        Without the superposition step the pose lands near the origin, tens of
        Angstroms from the fragments it is supposed to hold, and every distance
        measured against the protein afterwards is meaningless.
        """
        mol, fragments = diphenylpropane
        result = cp.verify_one(Chem.MolToSmiles(mol), fragments, np.zeros((0, 3)))
        posed = Chem.MolFromMolBlock(result.pose_sdf)
        expected = np.vstack(
            [fragment.GetConformer().GetPositions() for fragment in fragments]
        ).mean(axis=0)
        assert np.linalg.norm(posed.GetConformer().GetPositions().mean(axis=0) - expected) < 2.0

    def test_a_bridge_one_atom_short_holds_the_poses_worse(self, diphenylpropane):
        mol, fragments = diphenylpropane
        short = cp.verify_one("c1ccc(CCc2ccccc2)cc1", fragments, np.zeros((0, 3)))
        true = cp.verify_one(Chem.MolToSmiles(mol), fragments, np.zeros((0, 3)))
        assert short.max_fragment_rmsd > true.max_fragment_rmsd

    def test_protein_contact_is_reported_and_judged(self, diphenylpropane):
        mol, fragments = diphenylpropane
        # A wall of atoms through the middle of the bridge.
        middle = np.vstack(
            [fragment.GetConformer().GetPositions() for fragment in fragments]
        ).mean(axis=0)
        protein = np.array([middle + offset for offset in np.eye(3) * 0.1])
        result = cp.verify_one(Chem.MolToSmiles(mol), fragments, protein)
        assert result.min_protein_distance < cp.POSE_CLASH_DISTANCE
        assert result.num_clashes >= 1
        assert result.failed_gate == "gate5c_pose"
        assert "protein" in result.reason

    def test_without_a_pocket_no_contact_is_measured(self, diphenylpropane):
        mol, fragments = diphenylpropane
        result = cp.verify_one(Chem.MolToSmiles(mol), fragments, np.zeros((0, 3)))
        assert result.min_protein_distance is None
        assert result.num_clashes == 0

    def test_strain_can_be_reported_without_being_judged(self, diphenylpropane):
        mol, fragments = diphenylpropane
        result = cp.verify_one(
            Chem.MolToSmiles(mol), fragments, np.zeros((0, 3)), strain_max=None
        )
        assert result.strain is not None
        assert result.energy_restrained is not None
        assert result.energy_free is not None
        assert result.verified

    def test_the_pose_is_a_parseable_sdf_of_the_candidate(self, diphenylpropane):
        mol, fragments = diphenylpropane
        result = cp.verify_one(Chem.MolToSmiles(mol), fragments, np.zeros((0, 3)))
        posed = Chem.MolFromMolBlock(result.pose_sdf)
        assert posed is not None
        assert posed.GetNumHeavyAtoms() == mol.GetNumHeavyAtoms()
        assert posed.GetConformer().Is3D()


class TestCommittedCases:
    @pytest.mark.parametrize("pdb_id", fixture_ids())
    def test_the_known_answer_holds_its_own_fragments(self, pdb_id):
        case = load_fixture(pdb_id)
        results, error = cp.verify(
            [case.ligand_smiles], case.fragment_sdfs, case.pocket_pdb
        )
        assert error is None, error
        result = results[0]
        assert result.failed_gate != "gate2_substructure"
        assert result.failed_gate != "gate3_disjoint"
        assert result.max_fragment_rmsd < cp.DEFAULT_RMSD_MAX, result.reason

    @pytest.mark.parametrize("pdb_id", fixture_ids())
    def test_the_constrained_pose_recovers_the_crystal_pose(self, pdb_id):
        """Criterion V4, on the cases where the verifier is trusted at all."""
        case = load_fixture(pdb_id)
        results, _ = cp.verify([case.ligand_smiles], case.fragment_sdfs, case.pocket_pdb)
        result = results[0]
        if result.strain is not None and result.strain > cp.DEFAULT_STRAIN_MAX:
            pytest.skip(f"{pdb_id}: MMFF disagrees with this crystal geometry")
        crystal = Chem.RemoveHs(Chem.MolFromMolBlock(case.ligand_sdf))
        posed = Chem.RemoveHs(Chem.MolFromMolBlock(result.pose_sdf))
        assert rdMolAlign.CalcRMS(posed, crystal) < 1.5

    def test_a_linker_four_atoms_too_long_folds_into_the_protein(self):
        from app import linker_library as lib

        case = load_fixture("1uwh")
        fragments = [
            Chem.RemoveHs(Chem.MolFromMolBlock(block)) for block in case.fragment_sdfs
        ]
        overlong, error = lib.assemble(
            fragments[0],
            case.true_anchors[0],
            fragments[1],
            case.true_anchors[1],
            "[*:1]" + "C" * (case.true_linker_atoms + 4) + "[*:2]",
        )
        assert error is None, error
        results, _ = cp.verify([overlong], case.fragment_sdfs, case.pocket_pdb)
        assert not results[0].verified
        assert results[0].failed_gate == "gate5c_pose"


class TestBatch:
    def test_every_candidate_gets_a_result_in_order(self, diphenylpropane):
        mol, fragments = diphenylpropane
        blocks = [sdf(fragment) for fragment in fragments]
        smiles = [Chem.MolToSmiles(mol), "nonsense", "CCCCCC"]
        results, error = cp.verify(smiles, blocks)
        assert error is None
        assert [result.smiles for result in results] == smiles
        assert results[0].verified
        assert results[1].failed_gate == "gate1_smiles"
        assert results[2].failed_gate == "gate2_substructure"

    def test_no_candidates_is_an_input_error(self, diphenylpropane):
        _, fragments = diphenylpropane
        results, error = cp.verify([], [sdf(fragments[0])])
        assert results is None
        assert "no candidate" in error

    def test_no_fragments_is_an_input_error(self):
        results, error = cp.verify(["CCO"], [])
        assert results is None
        assert "no fragments" in error

    def test_an_unusable_fragment_names_its_index(self, diphenylpropane):
        _, fragments = diphenylpropane
        results, error = cp.verify(["CCO"], [sdf(fragments[0]), "garbage"])
        assert results is None
        assert error.startswith("fragment 1")


class TestEndpoint:
    def test_verification_round_trips(self, diphenylpropane):
        mol, fragments = diphenylpropane
        response = client.post(
            "/constrained-pose",
            json={
                "smiles": [Chem.MolToSmiles(mol)],
                "fragment_sdfs": [sdf(fragment) for fragment in fragments],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["num_verified"] == 1
        assert body["results"][0]["pose_sdf"]
        assert body["thresholds"]["rmsd_max"] == cp.DEFAULT_RMSD_MAX

    def test_poses_can_be_left_out_of_the_response(self, diphenylpropane):
        mol, fragments = diphenylpropane
        response = client.post(
            "/constrained-pose",
            json={
                "smiles": [Chem.MolToSmiles(mol)],
                "fragment_sdfs": [sdf(fragment) for fragment in fragments],
                "include_poses": False,
            },
        )
        assert "pose_sdf" not in response.json()["results"][0]

    def test_a_bad_fragment_is_reported_not_raised(self):
        response = client.post(
            "/constrained-pose", json={"smiles": ["CCO"], "fragment_sdfs": ["nope"]}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert "fragment 0" in body["error"]

    def test_an_empty_candidate_list_is_rejected_by_the_schema(self):
        response = client.post(
            "/constrained-pose", json={"smiles": [], "fragment_sdfs": ["x"]}
        )
        assert response.status_code == 422
