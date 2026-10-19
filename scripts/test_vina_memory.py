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

"""Test script to investigate Vina memory usage with large proteins.

This script:
1. Downloads PGK1 (417 residues, ~6500 atoms with H) - the problematic protein
2. Prepares it as a receptor
3. Docks a small ligand
4. Monitors memory usage at each step

Run: python scripts/test_vina_memory.py
"""

import json
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

import httpx

# Vina service URL (local container internal IP)
VINA_URL = os.environ.get("VINA_URL", "http://172.19.0.2:8000")

# PDB IDs to test (increasingly larger)
TEST_PROTEINS = [
    ("1AKI", "Lysozyme - small (~1000 atoms)"),
    ("1HSG", "HIV Protease - medium (~3000 atoms)"),
    ("3PGK", "PGK1 - large (~6500 atoms)"),  # The problematic one
]

# Simple aspirin SMILES for docking
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# Debug log path
DEBUG_LOG = "/workspaces/aevom/.cursor/debug.log"


def get_container_memory() -> dict:
    """Get vina container memory usage."""
    try:
        result = subprocess.run(
            ["docker", "stats", "aevom-vina-1", "--no-stream", "--format", 
             "{{.MemUsage}},{{.MemPerc}}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "container_mem_usage": parts[0] if parts else "unknown",
                "container_mem_percent": parts[1] if len(parts) > 1 else "unknown"
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


def log_step(step: str, data: dict):
    """Log a test step."""
    entry = {
        "hypothesisId": "TEST",
        "location": "test_vina_memory.py",
        "message": step,
        "data": {**data, **get_container_memory()},
        "timestamp": int(time.time() * 1000)
    }
    with open(DEBUG_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{step}] {json.dumps(data, indent=2)}")
    print(f"  Container memory: {get_container_memory()}")


def download_pdb(pdb_id: str) -> Optional[str]:
    """Download PDB from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = httpx.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")
        return None


def prepare_receptor(pdb_content: str) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    """Call vina service to prepare receptor."""
    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(
                f"{VINA_URL}/prepare-receptor",
                json={
                    "pdb_content": pdb_content,
                    "remove_water": True,
                    "remove_heteroatoms": False,
                    "ph": 7.4
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["success"], data.get("pdbqt"), data.get("num_atoms"), data.get("error")
    except Exception as e:
        return False, None, None, str(e)


def prepare_ligand(smiles: str) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    """Call vina service to prepare ligand."""
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{VINA_URL}/prepare-ligand",
                json={
                    "smiles": smiles,
                    "add_hydrogens": True,
                    "ph": 7.4
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["success"], data.get("pdbqt"), data.get("num_atoms"), data.get("error")
    except Exception as e:
        return False, None, None, str(e)


def dock(receptor_pdbqt: str, ligand_pdbqt: str, exhaustiveness: int = 16) -> Tuple[bool, dict, Optional[str]]:
    """Call vina service to dock."""
    try:
        # Use a generic center (will be adjusted based on receptor)
        with httpx.Client(timeout=600) as client:
            response = client.post(
                f"{VINA_URL}/dock",
                json={
                    "receptor_pdbqt": receptor_pdbqt,
                    "ligand_pdbqt": ligand_pdbqt,
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "center_z": 0.0,
                    "size_x": 20.0,
                    "size_y": 20.0,
                    "size_z": 20.0,
                    "exhaustiveness": exhaustiveness,
                    "num_modes": 5,
                    "energy_range": 3.0
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["success"], data, data.get("error")
    except Exception as e:
        return False, {}, str(e)


def test_protein(pdb_id: str, description: str, exhaustiveness: int = 16):
    """Test vina with a specific protein."""
    print(f"\n{'='*60}")
    print(f"Testing: {pdb_id} - {description}")
    print(f"Exhaustiveness: {exhaustiveness}")
    print(f"{'='*60}")
    
    log_step(f"START_{pdb_id}", {"pdb_id": pdb_id, "exhaustiveness": exhaustiveness})
    
    # Step 1: Download PDB
    print(f"\n1. Downloading {pdb_id}...")
    pdb_content = download_pdb(pdb_id)
    if not pdb_content:
        log_step(f"DOWNLOAD_FAILED_{pdb_id}", {"error": "Failed to download"})
        return False
    
    pdb_lines = len(pdb_content.split('\n'))
    atom_lines = pdb_content.count('\nATOM')
    log_step(f"DOWNLOADED_{pdb_id}", {"pdb_lines": pdb_lines, "atom_lines": atom_lines})
    
    # Step 2: Prepare receptor
    print(f"\n2. Preparing receptor...")
    start = time.time()
    success, receptor_pdbqt, num_atoms, error = prepare_receptor(pdb_content)
    prep_time = time.time() - start
    
    if not success:
        log_step(f"RECEPTOR_PREP_FAILED_{pdb_id}", {"error": error, "time_s": prep_time})
        print(f"   FAILED: {error}")
        return False
    
    log_step(f"RECEPTOR_PREPARED_{pdb_id}", {
        "num_atoms": num_atoms,
        "pdbqt_size": len(receptor_pdbqt),
        "time_s": round(prep_time, 2)
    })
    print(f"   Success: {num_atoms} atoms, {prep_time:.2f}s")
    
    # Step 3: Prepare ligand
    print(f"\n3. Preparing ligand (aspirin)...")
    start = time.time()
    success, ligand_pdbqt, lig_atoms, error = prepare_ligand(ASPIRIN_SMILES)
    lig_time = time.time() - start
    
    if not success:
        log_step(f"LIGAND_PREP_FAILED_{pdb_id}", {"error": error})
        print(f"   FAILED: {error}")
        return False
    
    log_step(f"LIGAND_PREPARED_{pdb_id}", {"num_atoms": lig_atoms, "time_s": round(lig_time, 2)})
    print(f"   Success: {lig_atoms} atoms, {lig_time:.2f}s")
    
    # Step 4: Dock
    print(f"\n4. Docking (exhaustiveness={exhaustiveness})...")
    log_step(f"DOCKING_START_{pdb_id}", {"exhaustiveness": exhaustiveness})
    
    start = time.time()
    success, result, error = dock(receptor_pdbqt, ligand_pdbqt, exhaustiveness)
    dock_time = time.time() - start
    
    if not success:
        log_step(f"DOCKING_FAILED_{pdb_id}", {
            "error": error,
            "time_s": round(dock_time, 2),
            "exhaustiveness": exhaustiveness
        })
        print(f"   FAILED after {dock_time:.2f}s: {error}")
        return False
    
    log_step(f"DOCKING_SUCCESS_{pdb_id}", {
        "best_affinity": result.get("best_affinity"),
        "num_poses": len(result.get("poses", [])),
        "execution_time_ms": result.get("execution_time_ms"),
        "time_s": round(dock_time, 2),
        "exhaustiveness": exhaustiveness
    })
    print(f"   Success: {result.get('best_affinity'):.2f} kcal/mol, {dock_time:.2f}s")
    
    return True


def main():
    print("Vina Memory Test Script")
    print(f"Vina URL: {VINA_URL}")
    print(f"Debug log: {DEBUG_LOG}")
    
    # Clear debug log
    os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
    if os.path.exists(DEBUG_LOG):
        os.remove(DEBUG_LOG)
    
    # Check vina service health
    print("\nChecking vina service health...")
    try:
        response = httpx.get(f"{VINA_URL}/health", timeout=10)
        print(f"  Health: {response.json()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Make sure vina service is running!")
        sys.exit(1)
    
    # Test with different proteins and exhaustiveness levels
    results = []
    
    for pdb_id, description in TEST_PROTEINS:
        for exhaustiveness in [8, 16, 32]:
            success = test_protein(pdb_id, description, exhaustiveness)
            results.append((pdb_id, exhaustiveness, success))
            
            # Give container time to recover between tests
            time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for pdb_id, exhaustiveness, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {pdb_id} (exhaust={exhaustiveness}): {status}")
    
    print(f"\nDebug log written to: {DEBUG_LOG}")


if __name__ == "__main__":
    main()
