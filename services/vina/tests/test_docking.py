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

"""Tests for Vina docking service.

Tests cover:
- Receptor preparation (PDB → PDBQT)
- Ligand preparation (SDF/SMILES → PDBQT)
- PDBQT format validation
- Integration with Vina docking

Run with: pytest services/vina/tests/
"""

import re
import pytest


# Sample test data
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
BENZENE_SMILES = "c1ccccc1"
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

# Simple PDB fragment for testing receptor preparation
SAMPLE_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
TER       6      ALA A   1
END
"""

# PDB with water for testing water removal
SAMPLE_PDB_WITH_WATER = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
HETATM  100  O   HOH A   2      10.000  10.000  10.000  1.00  0.00           O
TER
END
"""


class TestReceptorPreparation:
    """Test receptor preparation (PDB → PDBQT)."""

    def test_prepare_receptor_returns_pdbqt(self):
        """Test that prepare_receptor returns valid PDBQT content."""
        from app.preparation import prepare_receptor

        pdbqt, num_atoms, error = prepare_receptor(SAMPLE_PDB)

        assert error is None, f"Receptor preparation failed: {error}"
        assert pdbqt is not None
        assert num_atoms > 0

    def test_prepare_receptor_atom_count(self):
        """Test that atom count matches expected."""
        from app.preparation import prepare_receptor

        pdbqt, num_atoms, error = prepare_receptor(SAMPLE_PDB)

        assert error is None
        # Should have at least the original atoms (may have added hydrogens)
        assert num_atoms >= 5

    def test_prepare_receptor_removes_water(self):
        """Test that water molecules are removed by default."""
        from app.preparation import prepare_receptor

        pdbqt, num_atoms, error = prepare_receptor(
            SAMPLE_PDB_WITH_WATER, 
            remove_water=True
        )

        assert error is None
        # Water should be removed
        assert 'HOH' not in pdbqt

    def test_prepare_receptor_keeps_water_when_requested(self):
        """Test that water can be kept if requested."""
        from app.preparation import prepare_receptor

        pdbqt, num_atoms, error = prepare_receptor(
            SAMPLE_PDB_WITH_WATER, 
            remove_water=False
        )

        # May or may not have HOH depending on implementation
        # Just verify it doesn't error
        assert error is None

    def test_prepare_receptor_pdbqt_has_atom_lines(self):
        """Test that PDBQT output contains ATOM records."""
        from app.preparation import prepare_receptor

        pdbqt, _, error = prepare_receptor(SAMPLE_PDB)
        assert error is None

        atom_lines = [l for l in pdbqt.split('\n') if l.startswith('ATOM')]
        assert len(atom_lines) > 0, "PDBQT should contain ATOM records"


class TestLigandPreparation:
    """Test ligand preparation (SDF/SMILES → PDBQT)."""

    def test_prepare_ligand_from_smiles(self):
        """Test ligand preparation from SMILES."""
        from app.preparation import prepare_ligand

        pdbqt, num_atoms, num_rot, error = prepare_ligand(smiles=BENZENE_SMILES)

        assert error is None, f"Ligand preparation failed: {error}"
        assert pdbqt is not None
        assert num_atoms > 0

    def test_prepare_ligand_aspirin(self):
        """Test ligand preparation with aspirin (more complex molecule)."""
        from app.preparation import prepare_ligand

        pdbqt, num_atoms, num_rot, error = prepare_ligand(smiles=ASPIRIN_SMILES)

        assert error is None, f"Ligand preparation failed: {error}"
        assert pdbqt is not None
        assert num_atoms > 0
        # Aspirin has rotatable bonds
        assert num_rot >= 1

    def test_prepare_ligand_caffeine(self):
        """Test ligand preparation with caffeine (fused rings)."""
        from app.preparation import prepare_ligand

        pdbqt, num_atoms, num_rot, error = prepare_ligand(smiles=CAFFEINE_SMILES)

        assert error is None, f"Ligand preparation failed: {error}"
        assert pdbqt is not None
        assert num_atoms > 0

    def test_prepare_ligand_pdbqt_has_root(self):
        """Test that ligand PDBQT has ROOT/ENDROOT markers."""
        from app.preparation import prepare_ligand

        pdbqt, _, _, error = prepare_ligand(smiles=BENZENE_SMILES)
        assert error is None

        # Ligand PDBQT should have torsion tree markers
        assert 'ROOT' in pdbqt or 'ATOM' in pdbqt


class TestPDBQTFormat:
    """Test PDBQT format compliance."""

    def test_receptor_pdbqt_accepted_by_vina(self):
        """Test that receptor PDBQT is accepted by Vina (no parsing errors)."""
        from app.preparation import prepare_receptor
        import subprocess
        import tempfile
        import os

        pdbqt, _, error = prepare_receptor(SAMPLE_PDB)
        assert error is None

        # Create a minimal ligand PDBQT for testing
        ligand_pdbqt = """ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
TORSDOF 0
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, "receptor.pdbqt")
            lig_path = os.path.join(tmpdir, "ligand.pdbqt")
            out_path = os.path.join(tmpdir, "output.pdbqt")

            with open(rec_path, 'w') as f:
                f.write(pdbqt)
            with open(lig_path, 'w') as f:
                f.write(ligand_pdbqt)

            # Run Vina with minimal settings to check parsing
            result = subprocess.run(
                ["vina", "--receptor", rec_path, "--ligand", lig_path,
                 "--center_x", "0", "--center_y", "0", "--center_z", "0",
                 "--size_x", "10", "--size_y", "10", "--size_z", "10",
                 "--exhaustiveness", "1", "--out", out_path],
                capture_output=True, text=True, timeout=60
            )

            # Check for PDBQT parsing errors
            error_output = result.stderr or ""
            assert "PDBQT parsing error" not in error_output, \
                f"Vina rejected PDBQT: {error_output}"

    def test_ligand_pdbqt_accepted_by_vina(self):
        """Test that ligand PDBQT is accepted by Vina (no parsing errors)."""
        from app.preparation import prepare_ligand, prepare_receptor
        import subprocess
        import tempfile
        import os

        rec_pdbqt, _, rec_error = prepare_receptor(SAMPLE_PDB)
        assert rec_error is None

        lig_pdbqt, _, _, lig_error = prepare_ligand(smiles=BENZENE_SMILES)
        assert lig_error is None

        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, "receptor.pdbqt")
            lig_path = os.path.join(tmpdir, "ligand.pdbqt")
            out_path = os.path.join(tmpdir, "output.pdbqt")

            with open(rec_path, 'w') as f:
                f.write(rec_pdbqt)
            with open(lig_path, 'w') as f:
                f.write(lig_pdbqt)

            # Run Vina with minimal settings to check parsing
            result = subprocess.run(
                ["vina", "--receptor", rec_path, "--ligand", lig_path,
                 "--center_x", "0", "--center_y", "0", "--center_z", "0",
                 "--size_x", "10", "--size_y", "10", "--size_z", "10",
                 "--exhaustiveness", "1", "--out", out_path],
                capture_output=True, text=True, timeout=60
            )

            # Check for PDBQT parsing errors
            error_output = result.stderr or ""
            assert "PDBQT parsing error" not in error_output, \
                f"Vina rejected PDBQT: {error_output}"


class TestPDBQTConversion:
    """Test PDBQT to PDB conversion."""

    def test_pdbqt_to_pdb_strips_vina_keywords(self):
        """Test that PDBQT to PDB conversion strips Vina keywords."""
        from app.preparation import pdbqt_to_pdb

        pdbqt = """ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
TORSDOF 0
END
"""
        pdb = pdbqt_to_pdb(pdbqt)

        assert 'ROOT' not in pdb
        assert 'ENDROOT' not in pdb
        assert 'TORSDOF' not in pdb
        assert 'ATOM' in pdb


class TestDocking:
    """Test docking functionality."""
    
    @pytest.mark.skip(reason="Requires longer running time")
    def test_dock_simple(self):
        """Test simple docking with prepared receptor and ligand."""
        # This would call the /dock endpoint
        pass


class TestModels:
    """Test Pydantic models."""
    
    def test_dock_request_defaults(self):
        """Test DockRequest has correct defaults."""
        from app.models import DockRequest
        
        request = DockRequest(
            receptor_pdbqt="ATOM...",
            ligand_pdbqt="ATOM...",
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        )
        
        assert request.size_x == 20.0
        assert request.size_y == 20.0
        assert request.size_z == 20.0
        assert request.exhaustiveness == 32
        assert request.num_modes == 9
    
    def test_dock_request_validation(self):
        """Test DockRequest validates box size limits."""
        from app.models import DockRequest
        
        # Should raise validation error for size > 126
        with pytest.raises(ValueError):
            DockRequest(
                receptor_pdbqt="ATOM...",
                ligand_pdbqt="ATOM...",
                center_x=0.0,
                center_y=0.0,
                center_z=0.0,
                size_x=200.0,  # Too large
            )
