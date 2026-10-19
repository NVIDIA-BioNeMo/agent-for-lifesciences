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

"""Test vina service handling of pathological PDB structures.

This script tests various problematic structures that might come from
OpenFold3 or other structure prediction tools:
1. Missing element symbols
2. Extreme coordinates
3. Very large structures
4. Duplicate atoms
5. Invalid coordinates (NaN-like patterns)

Run: python scripts/test_pathological_structures.py
"""

import httpx
import subprocess
import time
import json
import os

VINA_URL = os.environ.get("VINA_URL", "http://172.19.0.2:8000")
DEBUG_LOG = "/tmp/vina_test_debug.log"  # Use /tmp to avoid permission issues

def get_mem():
    """Get vina container memory usage."""
    try:
        r = subprocess.run(
            ['docker', 'stats', 'aevom-vina-1', '--no-stream', '--format', '{{.MemUsage}}'],
            capture_output=True, text=True, timeout=2
        )
        return r.stdout.strip()
    except:
        return "unknown"


def log_test(test_name: str, result: dict):
    """Log test result."""
    entry = {
        "test": test_name,
        "timestamp": int(time.time() * 1000),
        **result
    }
    with open(DEBUG_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_normal_pdb(num_residues: int = 100) -> str:
    """Generate a normal PDB structure for baseline."""
    lines = []
    atom_num = 1
    for res_num in range(1, num_residues + 1):
        # Simple alanine residue
        x, y, z = res_num * 3.8, 0, 0  # Extended chain
        for atom_name, element, dx, dy, dz in [
            (" N  ", "N", 0, 0, 0),
            (" CA ", "C", 1.47, 0, 0),
            (" C  ", "C", 2.47, 1.0, 0),
            (" O  ", "O", 2.47, 2.2, 0),
            (" CB ", "C", 1.47, -1.0, 1.0),
        ]:
            lines.append(
                f"ATOM  {atom_num:5d} {atom_name} ALA A{res_num:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z+dz:8.3f}  1.00  0.00          {element:>2s}"
            )
            atom_num += 1
    lines.append("END")
    return "\n".join(lines)


def generate_missing_elements_pdb(num_residues: int = 100) -> str:
    """Generate PDB with missing element symbols (common OpenFold3 issue)."""
    lines = []
    atom_num = 1
    for res_num in range(1, num_residues + 1):
        x, y, z = res_num * 3.8, 0, 0
        for atom_name, dx, dy, dz in [
            (" N  ", 0, 0, 0),
            (" CA ", 1.47, 0, 0),
            (" C  ", 2.47, 1.0, 0),
            (" O  ", 2.47, 2.2, 0),
        ]:
            # Missing element symbol in columns 77-78
            lines.append(
                f"ATOM  {atom_num:5d} {atom_name} ALA A{res_num:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z+dz:8.3f}  1.00  0.00"
            )
            atom_num += 1
    lines.append("END")
    return "\n".join(lines)


def generate_extreme_coords_pdb(num_residues: int = 100) -> str:
    """Generate PDB with extreme coordinate values."""
    lines = []
    atom_num = 1
    for res_num in range(1, num_residues + 1):
        # Some atoms with extreme coordinates
        x = 99999.0 if res_num % 10 == 0 else res_num * 3.8
        y, z = 0, 0
        for atom_name, element, dx, dy, dz in [
            (" N  ", "N", 0, 0, 0),
            (" CA ", "C", 1.47, 0, 0),
            (" C  ", "C", 2.47, 1.0, 0),
            (" O  ", "O", 2.47, 2.2, 0),
        ]:
            lines.append(
                f"ATOM  {atom_num:5d} {atom_name} ALA A{res_num:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z+dz:8.3f}  1.00  0.00          {element:>2s}"
            )
            atom_num += 1
    lines.append("END")
    return "\n".join(lines)


def generate_large_pdb(num_residues: int = 3000) -> str:
    """Generate a very large PDB structure."""
    return generate_normal_pdb(num_residues)


def generate_duplicate_atoms_pdb(num_residues: int = 100) -> str:
    """Generate PDB with many duplicate atom positions."""
    lines = []
    atom_num = 1
    for res_num in range(1, num_residues + 1):
        x, y, z = res_num * 3.8, 0, 0
        for atom_name, element, dx, dy, dz in [
            (" N  ", "N", 0, 0, 0),
            (" CA ", "C", 1.47, 0, 0),
            (" C  ", "C", 2.47, 1.0, 0),
            (" O  ", "O", 2.47, 2.2, 0),
        ]:
            lines.append(
                f"ATOM  {atom_num:5d} {atom_name} ALA A{res_num:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z+dz:8.3f}  1.00  0.00          {element:>2s}"
            )
            atom_num += 1
            # Add duplicate at same position
            lines.append(
                f"ATOM  {atom_num:5d} {atom_name} ALA A{res_num:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z+dz:8.3f}  1.00  0.00          {element:>2s}"
            )
            atom_num += 1
    lines.append("END")
    return "\n".join(lines)


def test_receptor_prep(name: str, pdb_content: str) -> dict:
    """Test receptor preparation with given PDB."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    atom_count = pdb_content.count('\nATOM') + pdb_content.count('\nHETATM')
    print(f"  Input atoms: {atom_count}")
    print(f"  Input size: {len(pdb_content)/1024:.1f} KB")
    print(f"  Memory before: {get_mem()}")
    
    start = time.time()
    try:
        resp = httpx.post(
            f"{VINA_URL}/prepare-receptor",
            json={
                "pdb_content": pdb_content,
                "remove_water": True,
                "remove_heteroatoms": False,
            },
            timeout=120.0
        )
        elapsed = time.time() - start
        data = resp.json()
        
        result = {
            "success": data.get("success", False),
            "input_atoms": atom_count,
            "output_atoms": data.get("num_atoms"),
            "error": data.get("error"),
            "time_s": round(elapsed, 2),
            "memory_after": get_mem()
        }
        
        if data.get("success"):
            print(f"  ✓ SUCCESS: {data.get('num_atoms')} atoms, {elapsed:.2f}s")
        else:
            print(f"  ✗ FAILED: {data.get('error')}")
            
    except Exception as e:
        elapsed = time.time() - start
        result = {
            "success": False,
            "input_atoms": atom_count,
            "error": str(e),
            "time_s": round(elapsed, 2),
            "memory_after": get_mem()
        }
        print(f"  ✗ EXCEPTION after {elapsed:.2f}s: {e}")
    
    print(f"  Memory after: {result['memory_after']}")
    log_test(name, result)
    return result


def main():
    print("Pathological Structure Test Suite")
    print(f"Vina URL: {VINA_URL}")
    print(f"Debug log: {DEBUG_LOG}")
    
    # Clear debug log
    os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
    if os.path.exists(DEBUG_LOG):
        os.remove(DEBUG_LOG)
    
    # Check vina service health
    print("\nChecking vina service health...")
    try:
        resp = httpx.get(f"{VINA_URL}/health", timeout=10)
        print(f"  Health: {resp.json()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Make sure vina service is running!")
        return
    
    results = []
    
    # Test 1: Normal structure (baseline)
    results.append(test_receptor_prep(
        "Normal PDB (100 residues)",
        generate_normal_pdb(100)
    ))
    
    # Test 2: Missing element symbols
    results.append(test_receptor_prep(
        "Missing Elements (100 residues)",
        generate_missing_elements_pdb(100)
    ))
    
    # Test 3: Extreme coordinates
    results.append(test_receptor_prep(
        "Extreme Coordinates (100 residues)",
        generate_extreme_coords_pdb(100)
    ))
    
    # Test 4: Large structure
    results.append(test_receptor_prep(
        "Large Structure (500 residues)",
        generate_large_pdb(500)
    ))
    
    # Test 5: Very large structure
    results.append(test_receptor_prep(
        "Very Large Structure (1000 residues)",
        generate_large_pdb(1000)
    ))
    
    # Test 6: Duplicate atoms
    results.append(test_receptor_prep(
        "Duplicate Atoms (100 residues, 2x atoms)",
        generate_duplicate_atoms_pdb(100)
    ))
    
    # Test 7: RCSB PGK1 for comparison
    print("\n" + "="*60)
    print("Downloading RCSB 3PGK for comparison...")
    try:
        pdb = httpx.get("https://files.rcsb.org/download/3PGK.pdb", timeout=30).text
        results.append(test_receptor_prep("RCSB 3PGK (real protein)", pdb))
    except Exception as e:
        print(f"  Failed to download: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✓ PASS" if r.get("success") else "✗ FAIL"
        atoms = f"{r.get('input_atoms', '?')} -> {r.get('output_atoms', '?')}"
        time_s = r.get("time_s", "?")
        print(f"  {status}: {atoms} atoms, {time_s}s")
        if not r.get("success") and r.get("error"):
            print(f"         Error: {r['error'][:80]}...")
    
    print(f"\nDebug log: {DEBUG_LOG}")
    
    # Now test docking with complex ligands
    print("\n" + "="*60)
    print("DOCKING STRESS TESTS")
    print("="*60)
    
    # Use the RCSB 3PGK as receptor
    try:
        pdb = httpx.get("https://files.rcsb.org/download/3PGK.pdb", timeout=30).text
        test_docking_stress(pdb)
    except Exception as e:
        print(f"Failed to download PGK: {e}")


def test_docking_stress(receptor_pdb: str):
    """Test docking with increasingly complex ligands."""
    
    # Prepare receptor once
    print("\nPreparing receptor...")
    resp = httpx.post(
        f"{VINA_URL}/prepare-receptor",
        json={"pdb_content": receptor_pdb, "remove_water": True},
        timeout=120.0
    )
    if not resp.json().get("success"):
        print(f"  Receptor prep failed: {resp.json().get('error')}")
        return
    receptor_pdbqt = resp.json()["pdbqt"]
    print(f"  Receptor ready: {resp.json()['num_atoms']} atoms")
    
    # Test ligands of increasing complexity
    test_ligands = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "small drug"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C", "small molecule"),
        ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "medium drug - production failure case"),
        ("Taxol_fragment", "CC(=O)OC1C(=O)C2(C)C(O)CC3OC3(C(C)=C1C)C2(C)C", "complex fused rings"),
    ]
    
    for name, smiles, desc in test_ligands:
        print(f"\n--- Testing: {name} ({desc}) ---")
        print(f"  SMILES: {smiles[:50]}...")
        
        # Prepare ligand
        resp = httpx.post(
            f"{VINA_URL}/prepare-ligand",
            json={"smiles": smiles},
            timeout=60.0
        )
        if not resp.json().get("success"):
            print(f"  ✗ Ligand prep failed: {resp.json().get('error')}")
            continue
            
        ligand_pdbqt = resp.json()["pdbqt"]
        num_atoms = resp.json().get("num_atoms", "?")
        num_rotatable = resp.json().get("num_rotatable_bonds", "?")
        print(f"  Ligand: {num_atoms} atoms, {num_rotatable} rotatable bonds")
        
        # Test docking with different exhaustiveness
        for exhaustiveness in [8, 16, 32]:
            print(f"  Memory before (exhaust={exhaustiveness}): {get_mem()}")
            start = time.time()
            try:
                resp = httpx.post(
                    f"{VINA_URL}/dock",
                    json={
                        "receptor_pdbqt": receptor_pdbqt,
                        "ligand_pdbqt": ligand_pdbqt,
                        "center_x": 0, "center_y": 0, "center_z": 0,
                        "size_x": 20, "size_y": 20, "size_z": 20,
                        "exhaustiveness": exhaustiveness,
                        "num_modes": 5
                    },
                    timeout=300.0
                )
                elapsed = time.time() - start
                data = resp.json()
                
                if data.get("success"):
                    print(f"  ✓ exhaust={exhaustiveness}: {data.get('best_affinity'):.2f} kcal/mol, {elapsed:.1f}s")
                else:
                    print(f"  ✗ exhaust={exhaustiveness}: {data.get('error')}")
            except Exception as e:
                elapsed = time.time() - start
                print(f"  ✗ exhaust={exhaustiveness}: EXCEPTION after {elapsed:.1f}s: {e}")
            
            print(f"  Memory after (exhaust={exhaustiveness}): {get_mem()}")


if __name__ == "__main__":
    main()
