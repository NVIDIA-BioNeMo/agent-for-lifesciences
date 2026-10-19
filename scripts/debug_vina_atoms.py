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

"""Debug script to track atom counts through the Vina docking pipeline.

Tests each stage:
1. SDF generation (RDKit)
2. Ligand preparation (Meeko/OpenBabel -> PDBQT)
3. PDBQT to PDB conversion
4. Biopython PDB parsing
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def get_service_url(service_name: str) -> str:
    """Get the URL for a Docker service.
    
    When running inside the Docker network, use service names.
    When running outside, use container IPs.
    """
    import os
    
    # Check if we're running inside Docker (via env var or by checking /proc/1/cgroup)
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
    
    if in_docker:
        # Inside Docker, use service names directly
        return service_name
    else:
        # Outside Docker, get container IP
        result = subprocess.run(
            ["docker", "inspect", f"aevom-{service_name}-1", "--format", 
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"],
            capture_output=True, text=True
        )
        return result.stdout.strip()


def call_service(container_ip: str, endpoint: str, data: dict) -> dict:
    """Call a service endpoint via curl."""
    import requests
    url = f"http://{container_ip}:8000{endpoint}"
    response = requests.post(url, json=data, timeout=60)
    return response.json()


def count_sdf_atoms(sdf_content: str) -> int:
    """Count atoms in an SDF/MOL block.
    
    The atom count is in the counts line (4th line), first number.
    Format: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    Where aaa = number of atoms (columns 1-3)
    
    Note: The counts line may have numbers without spaces, so we need to
    parse positionally or look for the V2000/V3000 marker.
    """
    lines = sdf_content.split('\n')
    if len(lines) < 4:
        return 0
    
    counts_line = lines[3]  # 4th line (0-indexed: 3)
    
    # V2000 format: First 3 chars are atom count (may have leading spaces)
    # The line ends with "V2000" or "V3000"
    if 'V2000' in counts_line or 'V3000' in counts_line:
        # Split by whitespace and get the first number
        parts = counts_line.split()
        if len(parts) >= 1:
            try:
                return int(parts[0])
            except ValueError:
                pass
    
    # Fallback: try to parse first number from space-separated
    parts = counts_line.split()
    if len(parts) >= 2:
        try:
            return int(parts[0])
        except ValueError:
            pass
    return 0


def count_pdbqt_atoms(pdbqt_content: str) -> int:
    """Count ATOM/HETATM lines in PDBQT content."""
    count = 0
    for line in pdbqt_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            count += 1
    return count


def count_pdb_atoms(pdb_content: str) -> int:
    """Count ATOM/HETATM lines in PDB content."""
    count = 0
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            count += 1
    return count


def vina_type_to_element(vina_type: str) -> str:
    """Convert Vina atom type to element symbol."""
    vina_type = vina_type.strip().upper()
    
    type_map = {
        'A': 'C', 'C': 'C', 'NA': 'N', 'N': 'N', 'OA': 'O', 'O': 'O',
        'SA': 'S', 'S': 'S', 'HD': 'H', 'H': 'H', 'F': 'F', 'CL': 'CL',
        'BR': 'BR', 'I': 'I', 'P': 'P', 'FE': 'FE', 'ZN': 'ZN',
        'MG': 'MG', 'CA': 'CA', 'MN': 'MN',
    }
    
    if vina_type in type_map:
        return type_map[vina_type]
    if vina_type:
        return vina_type[:2] if len(vina_type) >= 2 and vina_type[1].isalpha() else vina_type[0]
    return 'C'


def pdbqt_to_pdb(pdbqt_content: str) -> str:
    """Convert PDBQT to PDB format (FIXED version).
    
    This version generates unique atom names to fix Biopython parsing.
    """
    pdb_lines = []
    skip_keywords = {'ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF'}
    element_counts: dict = {}
    
    for line in pdbqt_content.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        
        parts = stripped.split()
        if not parts:
            continue
        
        keyword = parts[0]
        
        if keyword in skip_keywords:
            continue
        
        if keyword in ('ATOM', 'HETATM'):
            try:
                serial = int(line[6:11].strip())
            except (ValueError, IndexError):
                serial = 1
            
            # Get Vina atom type (columns 77-79)
            vina_type = ""
            if len(line) > 77:
                vina_type = line[77:].split()[0] if line[77:].strip() else ""
            
            element = vina_type_to_element(vina_type)
            
            # Generate unique atom name
            element_counts[element] = element_counts.get(element, 0) + 1
            count = element_counts[element]
            
            if len(element) == 1:
                atom_name = f" {element}{count:<2d}"
            else:
                atom_name = f"{element}{count:<2d}"
            atom_name = atom_name[:4]
            
            res_name = line[17:20] if len(line) > 20 else "LIG"
            
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except (ValueError, IndexError):
                continue
            
            pdb_line = (
                f"HETATM{serial:5d} {atom_name:4s} {res_name:3s} L{serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
            )
            
            pdb_lines.append(pdb_line)
        elif keyword in ('TER', 'END'):
            pdb_lines.append(line)
    
    return '\n'.join(pdb_lines)


def parse_with_biopython(pdb_content: str) -> int:
    """Parse PDB content with Biopython and count atoms."""
    from Bio.PDB import PDBParser
    
    parser = PDBParser(QUIET=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_path = f.name
    
    try:
        structure = parser.get_structure("test", temp_path)
        atom_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    atom_count += len(residue)
        return atom_count
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    # Test molecule: Aspirin
    # Aspirin has 21 atoms total (13 heavy atoms + 8 hydrogens)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    molecule_name = "Aspirin"
    
    print("=" * 70)
    print(f"VINA PIPELINE ATOM COUNT DEBUG")
    print(f"Test molecule: {molecule_name}")
    print(f"SMILES: {smiles}")
    print("=" * 70)
    print()
    
    # Get service URLs
    rdkit_host = get_service_url("rdkit")
    vina_host = get_service_url("vina")
    
    print(f"RDKit service: {rdkit_host}:8000")
    print(f"Vina service: {vina_host}:8000")
    print()
    
    # Stage 1: Generate SDF
    print("-" * 70)
    print("STAGE 1: Generate SDF (RDKit)")
    print("-" * 70)
    
    sdf_response = call_service(rdkit_host, "/smiles-to-sdf", {
        "smiles": smiles,
        "optimize": True
    })
    
    if not sdf_response.get("success"):
        print(f"ERROR: SDF generation failed: {sdf_response.get('error')}")
        sys.exit(1)
    
    sdf_content = sdf_response["sdf"]
    sdf_atom_count = count_sdf_atoms(sdf_content)
    
    print(f"SDF generated successfully")
    print(f"Atom count in SDF: {sdf_atom_count}")
    print()
    print("SDF content (first 30 lines):")
    print("-" * 40)
    for i, line in enumerate(sdf_content.split('\n')[:30]):
        print(f"{i+1:3d}: {line}")
    print("-" * 40)
    print()
    
    # Stage 2: Prepare ligand (SDF -> PDBQT)
    print("-" * 70)
    print("STAGE 2: Prepare Ligand (Meeko/OpenBabel -> PDBQT)")
    print("-" * 70)
    
    ligand_response = call_service(vina_host, "/prepare-ligand", {
        "sdf_content": sdf_content,
        "add_hydrogens": True
    })
    
    if not ligand_response.get("success"):
        print(f"ERROR: Ligand preparation failed: {ligand_response.get('error')}")
        sys.exit(1)
    
    pdbqt_content = ligand_response["pdbqt"]
    pdbqt_atom_count = count_pdbqt_atoms(pdbqt_content)
    
    print(f"PDBQT generated successfully")
    print(f"Atom count in PDBQT: {pdbqt_atom_count}")
    print(f"Reported atoms: {ligand_response.get('num_atoms')}")
    print(f"Rotatable bonds: {ligand_response.get('num_rotatable_bonds')}")
    print()
    print("PDBQT content:")
    print("-" * 40)
    for line in pdbqt_content.split('\n'):
        print(line)
    print("-" * 40)
    print()
    
    # Stage 3: Convert PDBQT to PDB
    print("-" * 70)
    print("STAGE 3: Convert PDBQT to PDB (_pdbqt_to_pdb)")
    print("-" * 70)
    
    pdb_content = pdbqt_to_pdb(pdbqt_content)
    pdb_atom_count = count_pdb_atoms(pdb_content)
    
    print(f"PDB conversion complete")
    print(f"Atom count in PDB: {pdb_atom_count}")
    print()
    print("PDB content:")
    print("-" * 40)
    for line in pdb_content.split('\n'):
        print(line)
    print("-" * 40)
    print()
    
    # Stage 4: Parse with Biopython
    print("-" * 70)
    print("STAGE 4: Parse PDB with Biopython")
    print("-" * 70)
    
    try:
        biopython_atom_count = parse_with_biopython(pdb_content)
        print(f"Biopython parsed successfully")
        print(f"Atom count from Biopython: {biopython_atom_count}")
    except Exception as e:
        print(f"ERROR: Biopython parsing failed: {e}")
        biopython_atom_count = "ERROR"
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: Atom Counts Through Pipeline")
    print("=" * 70)
    print(f"Stage 1 - SDF (RDKit):           {sdf_atom_count} atoms")
    print(f"Stage 2 - PDBQT (Meeko):         {pdbqt_atom_count} atoms")
    print(f"Stage 3 - PDB (_pdbqt_to_pdb):   {pdb_atom_count} atoms")
    print(f"Stage 4 - Biopython parsing:     {biopython_atom_count} atoms")
    print()
    
    # Identify where atoms are lost
    if sdf_atom_count != pdbqt_atom_count:
        print(f"⚠️  ATOM LOSS at Stage 2: SDF->PDBQT ({sdf_atom_count} -> {pdbqt_atom_count})")
    
    if pdbqt_atom_count != pdb_atom_count:
        print(f"⚠️  ATOM LOSS at Stage 3: PDBQT->PDB ({pdbqt_atom_count} -> {pdb_atom_count})")
    
    if isinstance(biopython_atom_count, int) and pdb_atom_count != biopython_atom_count:
        print(f"⚠️  ATOM LOSS at Stage 4: PDB->Biopython ({pdb_atom_count} -> {biopython_atom_count})")
    
    if (sdf_atom_count == pdbqt_atom_count == pdb_atom_count and 
        isinstance(biopython_atom_count, int) and pdb_atom_count == biopython_atom_count):
        print("✓ No atom loss detected in pipeline stages 1-4")
        print("  Issue may be in the receptor-ligand combination step or visualization")


if __name__ == "__main__":
    main()
