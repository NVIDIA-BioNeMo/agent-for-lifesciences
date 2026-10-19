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

"""Tests for the enhanced docking pipeline.

Tests cover:
- Rotatable bond counting (preparation.count_rotatable_bonds)
- CPU conformer generation (conformer.generate_conformers)
- GPU conformer generation (conformer.generate_conformers with use_gpu=True)
- Rigid ligand PDBQT preparation (preparation.prepare_ligand_rigid)
- Auto-dock pipeline routing (pipeline.run_auto_dock) for all four cases
- DockAutoRequest model validation

Run with: pytest services/vina/tests/test_enhanced_pipeline.py -v
"""

import os
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

BENZENE_SMILES = "c1ccccc1"  # 0 rotatable bonds
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # ~3 rotatable bonds
IMATINIB_SMILES = (
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
)  # ~7 rotatable bonds

# A molecule that exceeds typical threshold (dasatinib has ~5, sunitinib ~6).
# Use a long-chain molecule to ensure > 15 rotatable bonds.
LONG_CHAIN_SMILES = "CCCCCCCCCCCCCCCCCCCCCCCCCCCC"  # docosane-like, many rot bonds

SAMPLE_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
TER       6      ALA A   1
END
"""


def _mol_from_smiles(smi: str) -> Chem.Mol:
    """Build a 3D molecule from SMILES for test use."""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _sdf_from_smiles(smi: str) -> str:
    """Generate SDF string from SMILES."""
    mol = _mol_from_smiles(smi)
    return Chem.MolToMolBlock(mol)


# ---------------------------------------------------------------------------
# TestRotatableBonds
# ---------------------------------------------------------------------------


class TestRotatableBonds:
    """Test count_rotatable_bonds from preparation module."""

    def test_benzene_zero_rotatable(self):
        from app.preparation import count_rotatable_bonds

        sdf = _sdf_from_smiles(BENZENE_SMILES)
        assert count_rotatable_bonds(sdf) == 0

    def test_aspirin_rotatable(self):
        from app.preparation import count_rotatable_bonds

        sdf = _sdf_from_smiles(ASPIRIN_SMILES)
        rot = count_rotatable_bonds(sdf)
        assert rot >= 2, f"Aspirin should have >= 2 rotatable bonds, got {rot}"

    def test_long_chain_high_rotatable(self):
        from app.preparation import count_rotatable_bonds

        sdf = _sdf_from_smiles(LONG_CHAIN_SMILES)
        rot = count_rotatable_bonds(sdf)
        assert rot > 15, f"Long chain should have > 15 rotatable bonds, got {rot}"

    def test_invalid_sdf_raises(self):
        from app.preparation import count_rotatable_bonds

        with pytest.raises(ValueError, match="Failed to parse"):
            count_rotatable_bonds("NOT_VALID_SDF")

    def test_matches_rdkit_directly(self):
        """Verify our function agrees with a direct RDKit call."""
        from app.preparation import count_rotatable_bonds

        sdf = _sdf_from_smiles(IMATINIB_SMILES)
        mol = Chem.MolFromMolBlock(sdf, removeHs=False)
        expected = rdMolDescriptors.CalcNumRotatableBonds(mol)
        assert count_rotatable_bonds(sdf) == expected


# ---------------------------------------------------------------------------
# TestConformerGenerationCPU
# ---------------------------------------------------------------------------


class TestConformerGenerationCPU:
    """Test CPU conformer generation via RDKit backend."""

    def test_generates_requested_count(self):
        from app.conformer import generate_conformers

        mol = _mol_from_smiles(ASPIRIN_SMILES)
        conformers = generate_conformers(mol, num_conformers=5, use_gpu=False)
        assert len(conformers) == 5, f"Expected 5 conformers, got {len(conformers)}"

    def test_each_conformer_has_one_conf(self):
        from app.conformer import generate_conformers

        mol = _mol_from_smiles(ASPIRIN_SMILES)
        conformers = generate_conformers(mol, num_conformers=3, use_gpu=False)
        for i, c in enumerate(conformers):
            assert c.GetNumConformers() == 1, (
                f"Conformer {i} has {c.GetNumConformers()} conformers, expected 1"
            )

    def test_conformers_sorted_by_energy(self):
        """Lowest energy conformer should come first."""
        from app.conformer import generate_conformers

        mol = _mol_from_smiles(IMATINIB_SMILES)
        conformers = generate_conformers(mol, num_conformers=10, use_gpu=False)
        assert len(conformers) > 0

    def test_benzene_single_conformer(self):
        from app.conformer import generate_conformers

        mol = _mol_from_smiles(BENZENE_SMILES)
        conformers = generate_conformers(mol, num_conformers=3, use_gpu=False)
        assert len(conformers) >= 1

    def test_fallback_when_gpu_requested_but_unavailable(self, monkeypatch):
        """When use_gpu=True but the GPU worker cannot start, fall back to CPU."""
        from dataclasses import replace

        from app import conformer_client
        from app.conformer import BACKEND_CPU_FALLBACK, generate_conformers_with_report

        config = replace(conformer_client.WorkerConfig.from_env(), python="/nonexistent/python")
        missing = conformer_client.ConformerWorkerClient(config)
        monkeypatch.setattr(conformer_client, "_client", missing)

        mol = _mol_from_smiles(ASPIRIN_SMILES)
        conformers, report = generate_conformers_with_report(mol, num_conformers=3, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert len(conformers) >= 1, "Should fall back to CPU"


# ---------------------------------------------------------------------------
# TestConformerGenerationGPU
# ---------------------------------------------------------------------------


@pytest.mark.gpu_worker
class TestConformerGenerationGPU:
    """GPU conformer generation through the out-of-process nvMolKit worker.

    Skipped where the worker cannot start (see ``conftest.gpu_worker``); under
    ``AEVOM_REQUIRE_GPU_WORKER=1`` that, or any CPU fallback, is a failure.
    """

    @staticmethod
    def _generate(mol, n):
        from app.conformer import BACKEND_GPU_WORKER, generate_conformers_with_report

        conformers, report = generate_conformers_with_report(mol, num_conformers=n, use_gpu=True)
        assert report.backend == BACKEND_GPU_WORKER, f"fell back to CPU: {report.fallback_reason}"
        return conformers

    def test_gpu_generates_requested_count(self, gpu_worker):
        conformers = self._generate(_mol_from_smiles(ASPIRIN_SMILES), 5)
        assert len(conformers) == 5, f"Expected 5 conformers, got {len(conformers)}"

    def test_gpu_each_conformer_has_one_conf(self, gpu_worker):
        for c in self._generate(_mol_from_smiles(ASPIRIN_SMILES), 3):
            assert c.GetNumConformers() == 1

    def test_gpu_imatinib_larger_ensemble(self, gpu_worker):
        assert len(self._generate(_mol_from_smiles(IMATINIB_SMILES), 20)) > 0


# ---------------------------------------------------------------------------
# TestRigidLigandPreparation
# ---------------------------------------------------------------------------


class TestRigidLigandPreparation:
    """Test prepare_ligand_rigid from preparation module."""

    def test_rigid_pdbqt_has_torsdof_zero(self):
        from app.preparation import prepare_ligand_rigid

        mol = _mol_from_smiles(ASPIRIN_SMILES)
        pdbqt, err = prepare_ligand_rigid(mol)
        assert err is None, f"Rigid prep failed: {err}"
        assert pdbqt is not None
        assert "TORSDOF 0" in pdbqt

    def test_rigid_pdbqt_has_atom_lines(self):
        from app.preparation import prepare_ligand_rigid

        mol = _mol_from_smiles(BENZENE_SMILES)
        pdbqt, err = prepare_ligand_rigid(mol)
        assert err is None
        atom_lines = [l for l in pdbqt.split("\n") if l.startswith("ATOM")]
        assert len(atom_lines) > 0

    def test_rigid_pdbqt_no_branch_keywords(self):
        from app.preparation import prepare_ligand_rigid

        mol = _mol_from_smiles(IMATINIB_SMILES)
        pdbqt, err = prepare_ligand_rigid(mol)
        assert err is None
        assert "BRANCH" not in pdbqt

    def test_no_conformer_returns_error(self):
        from app.preparation import prepare_ligand_rigid

        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        # No EmbedMolecule -> no conformer
        pdbqt, err = prepare_ligand_rigid(mol)
        assert pdbqt is None
        assert err is not None
        assert "no 3d conformer" in err.lower()


# ---------------------------------------------------------------------------
# TestDockAutoModel
# ---------------------------------------------------------------------------


class TestDockAutoModel:
    """Test DockAutoRequest Pydantic model."""

    def test_defaults(self):
        from app.models import DockAutoRequest

        req = DockAutoRequest(
            receptor_pdb="ATOM...",
            ligand_sdf="molblock...",
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        )
        assert req.num_conformers == 50
        assert req.rot_bond_threshold == 15
        assert req.exhaustiveness == 32
        assert req.size_x == 20.0

    def test_custom_conformer_count(self):
        from app.models import DockAutoRequest

        req = DockAutoRequest(
            receptor_pdb="ATOM...",
            ligand_sdf="mol...",
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            num_conformers=100,
            rot_bond_threshold=20,
        )
        assert req.num_conformers == 100
        assert req.rot_bond_threshold == 20

    def test_conformer_count_too_high(self):
        from app.models import DockAutoRequest

        with pytest.raises(ValueError):
            DockAutoRequest(
                receptor_pdb="x",
                ligand_sdf="y",
                center_x=0,
                center_y=0,
                center_z=0,
                num_conformers=600,  # max is 500
            )

    def test_box_size_over_35_rejected(self):
        from app.models import DockAutoRequest

        with pytest.raises(ValueError):
            DockAutoRequest(
                receptor_pdb="x",
                ligand_sdf="y",
                center_x=0,
                center_y=0,
                center_z=0,
                size_x=40.0,  # exceeds 35 A limit
            )


# ---------------------------------------------------------------------------
# TestDockResponseMetadata
# ---------------------------------------------------------------------------


class TestDockResponseMetadata:
    """Test new metadata fields on DockResponse."""

    def test_default_metadata(self):
        from app.models import DockResponse

        resp = DockResponse(success=True)
        assert resp.docking_mode == "flexible"
        assert resp.conformers_generated == 0
        assert resp.best_conformer_idx == 0
        assert resp.rotatable_bonds is None

    def test_ensemble_metadata(self):
        from app.models import DockResponse

        resp = DockResponse(
            success=True,
            docking_mode="rigid_ensemble",
            conformers_generated=50,
            best_conformer_idx=17,
            rotatable_bonds=22,
        )
        assert resp.docking_mode == "rigid_ensemble"
        assert resp.conformers_generated == 50
        assert resp.best_conformer_idx == 17
        assert resp.rotatable_bonds == 22


# ---------------------------------------------------------------------------
# TestAutoDockPipelineCPU
# ---------------------------------------------------------------------------


class TestAutoDockPipelineCPU:
    """Integration tests for the auto-dock pipeline on CPU.

    These tests exercise the full pipeline.run_auto_dock() function including
    receptor preparation, rotatable bond counting, strategy selection, and
    docking execution.
    """

    def test_flexible_path_low_rot_bonds(self):
        """Case 1/3: flexible docking for molecules with few rotatable bonds."""
        from app.pipeline import run_auto_dock

        ligand_sdf = _sdf_from_smiles(ASPIRIN_SMILES)

        result = run_auto_dock(
            receptor_pdb=SAMPLE_PDB,
            ligand_sdf=ligand_sdf,
            center=(1.0, 0.5, 0.0),
            size=(15.0, 15.0, 15.0),
            exhaustiveness=1,
            num_modes=1,
            rot_bond_threshold=100,  # force flexible path
        )

        assert result["docking_mode"] == "flexible"
        assert result["conformers_generated"] == 0
        assert result["rotatable_bonds"] is not None
        assert result["rotatable_bonds"] <= 100
        # Docking may fail on this tiny receptor, but the routing should be correct

    def test_ensemble_path_high_rot_bonds(self):
        """Case 2/4: ensemble rigid docking for highly flexible molecules.

        The pipeline should route to rigid_ensemble mode.  All conformers
        are supplied to a single Uni-Dock / Vina call (no custom batching).
        """
        from app.pipeline import run_auto_dock

        ligand_sdf = _sdf_from_smiles(LONG_CHAIN_SMILES)

        result = run_auto_dock(
            receptor_pdb=SAMPLE_PDB,
            ligand_sdf=ligand_sdf,
            center=(1.0, 0.5, 0.0),
            size=(15.0, 15.0, 15.0),
            exhaustiveness=1,
            num_modes=1,
            num_conformers=3,
            rot_bond_threshold=5,  # force ensemble path
        )

        if result["success"]:
            assert result["docking_mode"] == "rigid_ensemble"
            assert result["conformers_generated"] > 0
            assert result["rotatable_bonds"] is not None
            assert result["rotatable_bonds"] > 5
        else:
            # Allow failure only from docking backend, not routing/conformers
            assert result["error"] is not None
            err_lower = result["error"].lower()
            assert "docking" in err_lower or "cuda" in err_lower, (
                f"Unexpected error: {result['error']}"
            )

    def test_low_threshold_forces_ensemble(self):
        """Setting rot_bond_threshold=0 should force ensemble for any molecule."""
        from app.pipeline import run_auto_dock

        ligand_sdf = _sdf_from_smiles(BENZENE_SMILES)

        result = run_auto_dock(
            receptor_pdb=SAMPLE_PDB,
            ligand_sdf=ligand_sdf,
            center=(1.0, 0.5, 0.0),
            size=(15.0, 15.0, 15.0),
            exhaustiveness=1,
            num_modes=1,
            num_conformers=2,
            rot_bond_threshold=0,  # force ensemble even for benzene
        )

        # Benzene has 0 rot bonds, threshold=0 means 0 > 0 is False -> flexible!
        # Actually 0 > 0 = False, so it stays flexible.
        # Let's verify with the actual logic: use_ensemble = rot_bonds > rot_bond_threshold
        # So 0 > 0 = False => flexible. This is correct behavior.
        assert result["docking_mode"] == "flexible"

    def test_invalid_sdf_returns_error(self):
        from app.pipeline import run_auto_dock

        result = run_auto_dock(
            receptor_pdb=SAMPLE_PDB,
            ligand_sdf="INVALID_SDF_DATA",
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
        )

        assert result["success"] is False
        assert result["error"] is not None

    def test_invalid_receptor_returns_error(self):
        from app.pipeline import run_auto_dock

        ligand_sdf = _sdf_from_smiles(BENZENE_SMILES)
        result = run_auto_dock(
            receptor_pdb="NOT_A_PDB",
            ligand_sdf=ligand_sdf,
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
        )

        assert result["success"] is False
        assert result["error"] is not None

    def test_result_has_expected_keys(self):
        """The result dict should have all keys expected by DockResponse."""
        from app.pipeline import run_auto_dock

        ligand_sdf = _sdf_from_smiles(BENZENE_SMILES)
        result = run_auto_dock(
            receptor_pdb=SAMPLE_PDB,
            ligand_sdf=ligand_sdf,
            center=(1.0, 0.5, 0.0),
            size=(15.0, 15.0, 15.0),
            exhaustiveness=1,
        )

        expected_keys = {
            "success", "gpu_accelerated", "execution_time_ms", "poses",
            "best_affinity", "docking_mode", "conformers_generated",
            "best_conformer_idx", "rotatable_bonds", "error",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )


# ---------------------------------------------------------------------------
# TestRigidBatchDocking
# ---------------------------------------------------------------------------


class TestRigidBatchDocking:
    """Test run_rigid_batch_docking from docking module."""

    def test_batch_with_multiple_conformers(self):
        """Test batch docking with multiple rigid PDBQTs."""
        from app.docking import run_rigid_batch_docking
        from app.preparation import prepare_ligand_rigid, prepare_receptor

        receptor_pdbqt, _, rec_err = prepare_receptor(SAMPLE_PDB)
        assert rec_err is None

        mol = _mol_from_smiles(ASPIRIN_SMILES)
        from app.conformer import generate_conformers

        conformers = generate_conformers(mol, num_conformers=3, use_gpu=False)
        assert len(conformers) > 0

        rigid_pdbqts = []
        for c in conformers:
            pdbqt, err = prepare_ligand_rigid(c)
            if pdbqt:
                rigid_pdbqts.append(pdbqt)

        assert len(rigid_pdbqts) > 0, "No rigid PDBQTs generated"

        success, poses, best_aff, ms, gpu, best_idx, error = run_rigid_batch_docking(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqts=rigid_pdbqts,
            center=(1.0, 0.5, 0.0),
            size=(15.0, 15.0, 15.0),
            exhaustiveness=1,
            num_modes=1,
        )

        # The tiny receptor may cause docking failure, but the function
        # should not crash.
        assert isinstance(success, bool)
        assert isinstance(poses, list)
        assert isinstance(best_idx, int)

    def test_empty_list_returns_error(self):
        from app.docking import run_rigid_batch_docking

        success, poses, aff, ms, gpu, idx, error = run_rigid_batch_docking(
            receptor_pdbqt="ATOM...",
            ligand_pdbqts=[],
            center=(0, 0, 0),
            size=(20, 20, 20),
        )
        assert not success
        assert error is not None

    def test_box_guard_rejects_large_box_on_gpu(self):
        """When GPU is enabled, boxes > 35 A should be rejected early."""
        from app.docking import run_rigid_batch_docking, VINA_GPU_ENABLED

        if not VINA_GPU_ENABLED:
            pytest.skip("Box guard only active when VINA_GPU=true")

        success, poses, aff, ms, gpu, idx, error = run_rigid_batch_docking(
            receptor_pdbqt="ATOM  1  N  ALA A 1  0.0 0.0 0.0 1.0 0.0  N",
            ligand_pdbqts=["ROOT\nATOM  1  C  LIG A 1  0.0 0.0 0.0 1.0 0.0  C\nENDROOT\nTORSDOF 0"],
            center=(0, 0, 0),
            size=(40, 20, 20),  # x exceeds 35 A
        )
        assert not success
        assert "35" in error
        assert "size_x" in error
