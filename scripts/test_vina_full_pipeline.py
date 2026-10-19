#!/usr/bin/env python3
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

"""Test the full Vina docking pipeline to verify atom counts are preserved.

This tests the end-to-end flow including the _combine_receptor_ligand_pdb step.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, '/app')

from Bio.PDB import PDBParser


def test_pdbqt_to_pdb():
    """Test that the fixed _pdbqt_to_pdb generates valid PDB that Biopython can parse."""
    
    # Import the actual function from the backend
    from api.tools.vina import VinaDockTool
    
    tool = VinaDockTool()
    
    # Sample PDBQT output from Meeko (aspirin)
    pdbqt_content = """REMARK SMILES CC(=O)Oc1ccccc1C(=O)O
ROOT
ATOM      1  C   UNL     1      -0.235   0.460   0.418  1.00  0.00     0.145 A 
ATOM      2  C   UNL     1      -0.698   1.777   0.513  1.00  0.00     0.048 A 
ATOM      3  C   UNL     1       0.164   2.833   0.227  1.00  0.00     0.004 A 
ATOM      4  C   UNL     1       1.485   2.575  -0.140  1.00  0.00     0.001 A 
ATOM      5  C   UNL     1       1.950   1.258  -0.215  1.00  0.00     0.019 A 
ATOM      6  C   UNL     1       1.095   0.182   0.070  1.00  0.00     0.102 A 
ENDROOT
BRANCH   1   7
ATOM      7  O   UNL     1      -1.142  -0.551   0.746  1.00  0.00    -0.426 OA
BRANCH   7   8
ATOM      8  C   UNL     1      -1.719  -1.136  -0.376  1.00  0.00     0.308 C 
ATOM      9  O   UNL     1      -1.569  -0.789  -1.540  1.00  0.00    -0.252 OA
ATOM     10  C   UNL     1      -2.596  -2.270   0.065  1.00  0.00     0.139 C 
ENDBRANCH   7   8
ENDBRANCH   1   7
BRANCH   6  11
ATOM     11  C   UNL     1       1.590  -1.219   0.020  1.00  0.00     0.339 C 
ATOM     12  O   UNL     1       0.977  -2.226   0.321  1.00  0.00    -0.246 OA
BRANCH  11  13
ATOM     13  O   UNL     1       2.864  -1.301  -0.411  1.00  0.00    -0.478 OA
ATOM     14  H   UNL     1       3.049  -2.263  -0.410  1.00  0.00     0.297 HD
ENDBRANCH  11  13
ENDBRANCH   6  11
TORSDOF 4
"""
    
    # Convert PDBQT to PDB
    pdb_content = tool._pdbqt_to_pdb(pdbqt_content)
    
    print("=" * 60)
    print("TEST: _pdbqt_to_pdb()")
    print("=" * 60)
    print("\nGenerated PDB:")
    print(pdb_content)
    
    # Count HETATM/ATOM lines
    pdb_atom_count = sum(1 for line in pdb_content.split('\n') 
                         if line.startswith(('ATOM', 'HETATM')))
    
    print(f"\nPDB atom count: {pdb_atom_count}")
    
    # Parse with Biopython
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_path = f.name
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", temp_path)
        
        biopython_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    biopython_count += len(residue)
        
        print(f"Biopython atom count: {biopython_count}")
        
        if pdb_atom_count == biopython_count:
            print("\n✅ SUCCESS: All atoms preserved through Biopython parsing!")
        else:
            print(f"\n❌ FAILURE: Lost {pdb_atom_count - biopython_count} atoms in Biopython parsing")
            return False
            
    finally:
        os.unlink(temp_path)
    
    return True


def test_combine_receptor_ligand():
    """Test that _combine_receptor_ligand_pdb produces valid output."""
    
    from api.tools.vina import VinaDockTool
    
    tool = VinaDockTool()
    
    # Minimal receptor PDB (just a few atoms from chain A)
    receptor_pdb = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.245   2.390   0.000  1.00  0.00           O
TER       5      ALA A   1
END
"""
    
    # Minimal ligand PDB (from our fixed _pdbqt_to_pdb)
    ligand_pdb = """HETATM    1  C1  UNL L   1      10.000  10.000  10.000  1.00  0.00           C
HETATM    2  C2  UNL L   2      11.000  10.000  10.000  1.00  0.00           C
HETATM    3  O1  UNL L   3      12.000  10.000  10.000  1.00  0.00           O
"""
    
    print("\n" + "=" * 60)
    print("TEST: _combine_receptor_ligand_pdb()")
    print("=" * 60)
    
    combined_pdb = tool._combine_receptor_ligand_pdb(receptor_pdb, ligand_pdb, ligand_chain_id="Z")
    
    print("\nCombined PDB (first 20 lines):")
    for i, line in enumerate(combined_pdb.split('\n')[:20]):
        print(line)
    
    # Count atoms in combined PDB
    receptor_atoms = sum(1 for line in receptor_pdb.split('\n') 
                        if line.startswith('ATOM'))
    ligand_atoms = sum(1 for line in ligand_pdb.split('\n') 
                      if line.startswith('HETATM'))
    combined_atoms = sum(1 for line in combined_pdb.split('\n') 
                        if line.startswith(('ATOM', 'HETATM')))
    
    print(f"\nReceptor atoms: {receptor_atoms}")
    print(f"Ligand atoms: {ligand_atoms}")
    print(f"Combined atoms: {combined_atoms}")
    print(f"Expected: {receptor_atoms + ligand_atoms}")
    
    # Parse with Biopython
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(combined_pdb)
        temp_path = f.name
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", temp_path)
        
        biopython_count = 0
        chains_found = set()
        for model in structure:
            for chain in model:
                chains_found.add(chain.id)
                for residue in chain:
                    biopython_count += len(residue)
        
        print(f"Biopython atom count: {biopython_count}")
        print(f"Chains found: {chains_found}")
        
        if biopython_count >= receptor_atoms + ligand_atoms:
            print("\n✅ SUCCESS: All atoms preserved in combined structure!")
        else:
            print(f"\n❌ FAILURE: Lost atoms in combination")
            return False
            
    finally:
        os.unlink(temp_path)
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("VINA PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    success1 = test_pdbqt_to_pdb()
    success2 = test_combine_receptor_ligand()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"_pdbqt_to_pdb: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"_combine_receptor_ligand_pdb: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    sys.exit(0 if success1 and success2 else 1)
