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

"""Molecular operations using RDKit.

This module provides functions for working with molecules:
- SMILES parsing and validation
- 2D structure generation
- Image rendering
"""

import io
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw


# Default image settings
DEFAULT_IMAGE_SIZE = (400, 400)


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """Validate a SMILES string.
    
    Args:
        smiles: SMILES string to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not smiles or not smiles.strip():
        return False, "SMILES string is empty"
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, f"Invalid SMILES string: '{smiles}'"
    
    return True, None


def smiles_to_image_bytes(
    smiles: str,
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[Optional[bytes], Optional[str]]:
    """Convert SMILES string to PNG image bytes.
    
    Returning bytes rather than writing a file is what lets this service scale
    horizontally: it runs behind an HPA, so a PNG written to pod-local disk is
    unreadable by every other replica.
    
    Args:
        smiles: SMILES string of the molecule.
        size: Image dimensions as (width, height).
        
    Returns:
        Tuple of (image_bytes, error_message).
    """
    # Validate SMILES
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        return None, error
    
    # Parse molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Generate image to bytes
    try:
        img = Draw.MolToImage(mol, size=size)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue(), None
    except Exception as e:
        return None, f"Failed to generate image: {str(e)}"


def get_molecule_info(smiles: str) -> Tuple[Optional[dict], Optional[str]]:
    """Get basic information about a molecule.
    
    Args:
        smiles: SMILES string of the molecule.
        
    Returns:
        Tuple of (info_dict, error_message).
    """
    # Validate SMILES
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        return None, error
    
    mol = Chem.MolFromSmiles(smiles)
    
    from rdkit.Chem import Descriptors, rdMolDescriptors
    
    try:
        info = {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_hbd": rdMolDescriptors.CalcNumHBD(mol),  # H-bond donors
            "num_hba": rdMolDescriptors.CalcNumHBA(mol),  # H-bond acceptors
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),  # Topological polar surface area
            "logp": round(Descriptors.MolLogP(mol), 2),  # Octanol-water partition coefficient
        }
        return info, None
    except Exception as e:
        return None, f"Failed to calculate molecule info: {str(e)}"


def smiles_to_sdf(
    smiles: str,
    optimize: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Convert SMILES string to SDF/MOL format with 3D coordinates.
    
    Generates a 3D conformer using RDKit's ETKDG algorithm and returns
    the molecule in SDF format suitable for 3D visualization.
    
    Args:
        smiles: SMILES string of the molecule.
        optimize: Whether to optimize the 3D geometry with force field (default True).
        
    Returns:
        Tuple of (sdf_string, error_message).
    """
    # Validate SMILES
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        return None, error
    
    mol = Chem.MolFromSmiles(smiles)
    
    try:
        # Add hydrogens for proper 3D geometry
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer using ETKDG (Experimental-Torsion Distance Geometry)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # For reproducibility
        
        result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            # Fallback to random coordinates if ETKDG fails
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        if optimize:
            # Optimize geometry with MMFF force field
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                # Fall back to UFF if MMFF fails
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    pass  # Use unoptimized geometry
        
        # Convert to SDF/MOL block
        sdf_string = Chem.MolToMolBlock(mol)
        
        return sdf_string, None
        
    except Exception as e:
        return None, f"Failed to generate 3D structure: {str(e)}"

