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

"""Molecular preparation for Autodock Vina using established libraries.

This module provides functions to convert PDB (receptor) and SDF (ligand)
formats to PDBQT format required by Autodock Vina.

Uses:
- OpenBabel for receptor PDB → PDBQT conversion (robust, handles charges and atom types)
- Meeko for ligand SDF → PDBQT conversion (proper torsion tree for flexible docking)
"""

import logging
import tempfile
import json
import time
import resource
from pathlib import Path
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

logger = logging.getLogger(__name__)

# #region agent log
import os as _os

DEBUG_LOG_PATH = _os.environ.get("DEBUG_LOG_PATH", "/app/debug_logs/debug.log")


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    """Write debug log entry to NDJSON file."""
    try:
        mem_mb = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )  # KB to MB on Linux
        entry = {
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": {**data, "memory_mb": round(mem_mb, 2)},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# #endregion

# =============================================================================
# PDBQT Format Detection
# =============================================================================

# AutoDock/Vina atom types found in column 77-79 of PDBQT files
_VINA_ATOM_TYPES = frozenset(
    {
        "A",
        "C",
        "N",
        "NA",
        "OA",
        "O",
        "SA",
        "S",
        "HD",
        "H",
        "F",
        "Cl",
        "Br",
        "I",
        "P",
        "Fe",
        "Zn",
        "Mg",
        "Ca",
        "Mn",
    }
)


# Record types that appear only in raw PDB files, never in a prepared PDBQT
# receptor. Their presence is a definitive "this is raw PDB, prepare it"
# signal that overrides the ambiguous column-77 atom-type heuristic below.
_RAW_PDB_ONLY_RECORDS = (
    "HEADER", "TITLE", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR",
    "REVDAT", "JRNL", "SEQRES", "HELIX", "SHEET", "CRYST1", "ORIGX", "SCALE",
    "MTRIX", "MODEL", "ENDMDL", "DBREF", "SEQADV", "MODRES", "FORMUL", "HETNAM",
)


def is_pdbqt(content: str) -> bool:
    """Detect whether *content* is already in PDBQT format.

    Heuristic: look at the first 50 ATOM/HETATM lines.  If the majority
    have a recognised Vina atom-type token in columns 77-79 **or** the
    file contains the Meeko/AutoDockTools ``REMARK  Name =`` header, we
    classify it as PDBQT.

    The column-77 test alone is ambiguous: a bare element symbol (``C``,
    ``N``, ``O``, ``S``) sits in the same columns of a raw PDB ATOM line and is
    *also* a valid Vina atom type, so a plain RCSB structure -- especially a
    biological assembly, which RSCB wraps in ``MODEL``/``ENDMDL`` and ships with
    ``COMPND`` headers -- was being misread as already-prepared PDBQT. The
    preparation step was then skipped and the raw, multi-model PDB handed
    straight to Vina, which rejected it ("Unexpected multi-MODEL tag" /
    "inappropriate tag ... COMPND"). A prepared PDBQT never contains any of the
    raw-PDB-only records below, so seeing one is a decisive vote for raw PDB.
    """
    has_header = False
    vina_type_hits = 0
    atom_lines_checked = 0
    max_check = 50

    for line in content.split("\n"):
        # A record that only exists in raw PDBs settles it immediately. These
        # records precede the first ATOM line, so they are always seen before
        # the atom-line budget below is exhausted.
        if line.startswith(_RAW_PDB_ONLY_RECORDS):
            return False

        if line.startswith("REMARK") and "active torsions:" in line:
            has_header = True

        if atom_lines_checked >= max_check:
            break

        if line.startswith(("ATOM", "HETATM")) and len(line) >= 78:
            atom_lines_checked += 1
            token = line[77:].split()[0] if line[77:].strip() else ""
            if token in _VINA_ATOM_TYPES:
                vina_type_hits += 1

    if has_header:
        return True
    if atom_lines_checked > 0 and vina_type_hits / atom_lines_checked > 0.5:
        return True
    return False


class PDBQTValidationError(Exception):
    """Raised when a pre-prepared PDBQT fails lightweight validation."""

    pass


# Upper bound for pre-prepared PDBQT receptors (generous -- the 10 MB PDB
# limit does not apply because PDBQT includes hydrogens and charge columns).
MAX_PDBQT_SIZE_MB = 150
MAX_PDBQT_ATOMS = 100_000


def validate_pdbqt_receptor(pdbqt_content: str) -> dict:
    """Lightweight validation for an already-prepared PDBQT receptor.

    Checks coordinate sanity and atom count but does **not** re-prepare
    the file or enforce the strict PDB size limit.

    Returns:
        Dict with ``total_atoms``, ``file_size_kb``, ``chains``.

    Raises:
        PDBQTValidationError: on clearly broken files.
    """
    size_kb = len(pdbqt_content) / 1024
    if size_kb > MAX_PDBQT_SIZE_MB * 1024:
        raise PDBQTValidationError(
            f"PDBQT file too large: {size_kb:.1f} KB > {MAX_PDBQT_SIZE_MB} MB limit"
        )

    atom_count = 0
    invalid_coords = 0
    chains: set[str] = set()

    for line in pdbqt_content.split("\n"):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_count += 1

        if len(line) > 21:
            chains.add(line[21])

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            import math

            if any(math.isnan(c) or math.isinf(c) for c in (x, y, z)):
                invalid_coords += 1
        except (ValueError, IndexError):
            invalid_coords += 1

    if atom_count == 0:
        raise PDBQTValidationError("PDBQT file contains no ATOM/HETATM records")
    if atom_count > MAX_PDBQT_ATOMS:
        raise PDBQTValidationError(
            f"PDBQT has {atom_count} atoms (limit {MAX_PDBQT_ATOMS})"
        )
    if invalid_coords > atom_count * 0.1:
        raise PDBQTValidationError(
            f"Too many invalid coordinates: {invalid_coords}/{atom_count}"
        )

    logger.info(
        f"PDBQT validation passed: {atom_count} atoms, "
        f"{len(chains)} chains, {size_kb:.0f} KB"
    )
    return {
        "total_atoms": atom_count,
        "file_size_kb": size_kb,
        "chains": list(chains),
    }


# =============================================================================
# PDB Validation for Pathological Structures
# =============================================================================

# Safety limits to prevent OOM from pathological structures
MAX_INPUT_ATOMS = 15000  # Max atoms in input PDB (before H added)
MAX_OUTPUT_ATOMS = 50000  # Max atoms after hydrogen addition
MAX_PDB_SIZE_MB = 10  # Max PDB file size in MB
COORD_LIMIT = 9999.0  # Max reasonable coordinate value


class PDBValidationError(Exception):
    """Raised when PDB structure fails validation checks."""

    pass


def validate_pdb_structure(pdb_content: str) -> dict:
    """Validate PDB structure and return statistics.

    Checks for pathological conditions that could cause OOM or crashes:
    - Excessive atom count
    - Invalid coordinates (NaN, Inf, extremely large)
    - Missing element symbols
    - Duplicate atoms at same position
    - File size limits

    Args:
        pdb_content: PDB format string.

    Returns:
        Dict with validation results and statistics.

    Raises:
        PDBValidationError: If structure fails critical validation.
    """
    stats = {
        "file_size_kb": len(pdb_content) / 1024,
        "total_lines": pdb_content.count("\n"),
        "atom_count": 0,
        "hetatm_count": 0,
        "missing_elements": 0,
        "invalid_coords": 0,
        "coord_warnings": [],
        "duplicate_positions": 0,
        "residue_count": 0,
        "chains": set(),
    }

    # Check file size
    if stats["file_size_kb"] > MAX_PDB_SIZE_MB * 1024:
        raise PDBValidationError(
            f"PDB file too large: {stats['file_size_kb']:.1f}KB > {MAX_PDB_SIZE_MB}MB limit"
        )

    seen_positions = set()
    residues = set()

    for line in pdb_content.split("\n"):
        record_type = line[:6].strip() if len(line) >= 6 else ""

        if record_type == "ATOM":
            stats["atom_count"] += 1
        elif record_type == "HETATM":
            stats["hetatm_count"] += 1
        else:
            continue

        # Check element symbol (columns 77-78)
        if len(line) < 78 or not line[76:78].strip():
            stats["missing_elements"] += 1

        # Extract coordinates (columns 31-54)
        try:
            x = float(line[30:38].strip()) if len(line) > 38 else 0
            y = float(line[38:46].strip()) if len(line) > 46 else 0
            z = float(line[46:54].strip()) if len(line) > 54 else 0

            # Check for invalid values
            import math

            if any(math.isnan(c) or math.isinf(c) for c in [x, y, z]):
                stats["invalid_coords"] += 1
                stats["coord_warnings"].append(
                    f"NaN/Inf at atom {stats['atom_count'] + stats['hetatm_count']}"
                )
            elif any(abs(c) > COORD_LIMIT for c in [x, y, z]):
                stats["invalid_coords"] += 1
                stats["coord_warnings"].append(
                    f"Extreme coord at atom {stats['atom_count'] + stats['hetatm_count']}: ({x:.1f}, {y:.1f}, {z:.1f})"
                )

            # Check for duplicate positions (rounded to 0.1 Å)
            pos_key = (round(x, 1), round(y, 1), round(z, 1))
            if pos_key in seen_positions:
                stats["duplicate_positions"] += 1
            seen_positions.add(pos_key)

        except (ValueError, IndexError):
            stats["invalid_coords"] += 1

        # Track residues and chains
        if len(line) > 26:
            chain = line[21] if len(line) > 21 else " "
            res_num = line[22:26].strip() if len(line) > 26 else ""
            res_name = line[17:20].strip() if len(line) > 20 else ""
            residues.add((chain, res_num, res_name))
            stats["chains"].add(chain)

    stats["residue_count"] = len(residues)
    stats["chains"] = list(stats["chains"])
    stats["total_atoms"] = stats["atom_count"] + stats["hetatm_count"]

    # Critical validation checks
    if stats["total_atoms"] > MAX_INPUT_ATOMS:
        raise PDBValidationError(
            f"Too many atoms: {stats['total_atoms']} > {MAX_INPUT_ATOMS} limit. "
            "This structure is too large for safe docking."
        )

    if stats["invalid_coords"] > stats["total_atoms"] * 0.1:  # >10% invalid
        raise PDBValidationError(
            f"Too many invalid coordinates: {stats['invalid_coords']} of {stats['total_atoms']} atoms. "
            "This structure appears corrupted."
        )

    # #region agent log
    _debug_log(
        "F",
        "preparation.py:validate_pdb_structure",
        "PDB validation complete",
        {
            "total_atoms": stats["total_atoms"],
            "residue_count": stats["residue_count"],
            "missing_elements": stats["missing_elements"],
            "invalid_coords": stats["invalid_coords"],
            "duplicate_positions": stats["duplicate_positions"],
            "file_size_kb": round(stats["file_size_kb"], 2),
        },
    )
    # #endregion

    return stats


def prepare_receptor(
    pdb_content: str,
    remove_water: bool = True,
    remove_heteroatoms: bool = False,
    ph: float = 7.4,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Convert PDB to PDBQT format for receptor using OpenBabel.

    OpenBabel handles proper PDBQT formatting including:
    - Correct column alignment
    - Gasteiger partial charge assignment
    - AutoDock atom type assignment

    Args:
        pdb_content: PDB format string of the receptor.
        remove_water: Remove water molecules (HOH, WAT).
        remove_heteroatoms: Remove all heteroatoms (ligands, cofactors).
        ph: pH for protonation state (used by OpenBabel).

    Returns:
        Tuple of (pdbqt_string, num_atoms, error_message).
        On success: (pdbqt, atom_count, None)
        On failure: (None, None, error_message)
    """
    try:
        # Validate PDB structure before processing
        try:
            validation = validate_pdb_structure(pdb_content)
            logger.info(
                f"PDB validation passed: {validation['total_atoms']} atoms, "
                f"{validation['residue_count']} residues, "
                f"{validation['missing_elements']} missing elements"
            )

            # Warn about potential issues
            if validation["missing_elements"] > 100:
                logger.warning(
                    f"PDB has {validation['missing_elements']} atoms with missing element symbols. "
                    "This may indicate a non-standard structure (e.g., from OpenFold3)."
                )
            if validation["duplicate_positions"] > 50:
                logger.warning(
                    f"PDB has {validation['duplicate_positions']} atoms at duplicate positions. "
                    "This may indicate structural issues."
                )

        except PDBValidationError as e:
            logger.error(f"PDB validation failed: {e}")
            return None, None, str(e)

        # Try OpenBabel first (preferred method)
        try:
            from openbabel import openbabel

            return _prepare_receptor_openbabel(
                pdb_content, remove_water, remove_heteroatoms, ph
            )
        except ImportError:
            logger.warning(
                "OpenBabel not available, using fallback receptor preparation"
            )
            return _prepare_receptor_fallback(
                pdb_content, remove_water, remove_heteroatoms, ph
            )
    except Exception as e:
        logger.exception("Failed to prepare receptor")
        return None, None, f"Receptor preparation failed: {str(e)}"


def _prepare_receptor_openbabel(
    pdb_content: str,
    remove_water: bool = True,
    remove_heteroatoms: bool = False,
    ph: float = 7.4,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Prepare receptor using OpenBabel (preferred method).

    OpenBabel properly handles PDBQT format with correct:
    - Column alignment and spacing
    - Gasteiger charge assignment
    - AutoDock atom typing
    """
    from openbabel import openbabel

    # #region agent log
    pdb_lines = len(pdb_content.split("\n"))
    _debug_log(
        "B",
        "preparation.py:_prepare_receptor_openbabel:entry",
        "Starting receptor preparation",
        {"pdb_lines": pdb_lines},
    )
    # #endregion

    # Pre-process PDB to remove water/heteroatoms if requested
    if remove_water or remove_heteroatoms:
        pdb_content = _filter_pdb_content(pdb_content, remove_water, remove_heteroatoms)

    # Create OpenBabel conversion objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")

    # Set options for PDBQT output
    # -xr: Output as rigid receptor (no flexible residues)
    # -xc: Combine all residues (don't split by residue)
    obConversion.AddOption("r", obConversion.OUTOPTIONS)  # rigid receptor

    # Create molecule and read PDB
    mol = openbabel.OBMol()

    # Read from string
    success = obConversion.ReadString(mol, pdb_content)
    if not success:
        return None, None, "OpenBabel failed to parse PDB content"

    # #region agent log
    atoms_before_h = mol.NumAtoms()
    _debug_log(
        "B",
        "preparation.py:_prepare_receptor_openbabel:before_add_hydrogens",
        "Before AddHydrogens",
        {"num_atoms_before": atoms_before_h},
    )
    # #endregion

    # Add hydrogens at specified pH
    mol.AddHydrogens(False, True, ph)  # polarOnly=False, correctForPH=True

    atoms_after_h = mol.NumAtoms()
    hydrogen_ratio = atoms_after_h / max(atoms_before_h, 1)

    # #region agent log
    _debug_log(
        "B",
        "preparation.py:_prepare_receptor_openbabel:after_add_hydrogens",
        "After AddHydrogens",
        {
            "num_atoms_before": atoms_before_h,
            "num_atoms_after": atoms_after_h,
            "hydrogen_ratio": round(hydrogen_ratio, 2),
        },
    )
    # #endregion

    # CRITICAL: Check for pathological hydrogen explosion
    if atoms_after_h > MAX_OUTPUT_ATOMS:
        error_msg = (
            f"Structure too large after hydrogen addition: {atoms_after_h} atoms "
            f"(limit: {MAX_OUTPUT_ATOMS}). This may indicate a corrupted or pathological structure."
        )
        logger.error(error_msg)
        # #region agent log
        _debug_log(
            "F",
            "preparation.py:_prepare_receptor_openbabel:hydrogen_explosion",
            "PATHOLOGICAL: Atom count exploded after AddHydrogens",
            {
                "atoms_before": atoms_before_h,
                "atoms_after": atoms_after_h,
                "hydrogen_ratio": round(hydrogen_ratio, 2),
                "limit": MAX_OUTPUT_ATOMS,
            },
        )
        # #endregion
        return None, None, error_msg

    # Warn if hydrogen ratio is abnormally high (typical is ~2x)
    if hydrogen_ratio > 3.0:
        logger.warning(
            f"Abnormal hydrogen ratio: {hydrogen_ratio:.1f}x "
            f"({atoms_before_h} -> {atoms_after_h} atoms). "
            "Structure may have unusual chemistry."
        )

    # #region agent log
    _debug_log(
        "C",
        "preparation.py:_prepare_receptor_openbabel:before_charges",
        "Before ComputeCharges",
        {"num_atoms": mol.NumAtoms()},
    )
    # #endregion

    # Assign Gasteiger partial charges
    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    if charge_model:
        charge_model.ComputeCharges(mol)

    # #region agent log
    _debug_log(
        "C",
        "preparation.py:_prepare_receptor_openbabel:after_charges",
        "After ComputeCharges",
        {"num_atoms": mol.NumAtoms()},
    )
    # #endregion

    # Convert to PDBQT string
    pdbqt_string = obConversion.WriteString(mol)

    if not pdbqt_string or not pdbqt_string.strip():
        return None, None, "OpenBabel produced empty PDBQT output"

    # Count atoms (ATOM/HETATM lines)
    atom_count = sum(
        1 for line in pdbqt_string.split("\n") if line.startswith(("ATOM", "HETATM"))
    )

    if atom_count == 0:
        return None, None, "No atoms in PDBQT output"

    logger.info(f"Prepared receptor with OpenBabel: {atom_count} atoms")
    return pdbqt_string.strip(), atom_count, None


def _filter_pdb_content(
    pdb_content: str,
    remove_water: bool = True,
    remove_heteroatoms: bool = False,
) -> str:
    """Filter PDB content to remove water and/or heteroatoms."""
    water_residues = {"HOH", "WAT", "H2O", "TIP", "TIP3", "SPC"}
    output_lines = []

    for line in pdb_content.split("\n"):
        record_type = line[:6].strip() if len(line) >= 6 else ""

        # Always keep non-coordinate records
        if record_type not in ("ATOM", "HETATM"):
            output_lines.append(line)
            continue

        # Parse residue name (columns 18-20, 0-indexed 17:20)
        residue_name = line[17:20].strip() if len(line) >= 20 else ""

        # Skip water if requested
        if remove_water and residue_name in water_residues:
            continue

        # Skip heteroatoms if requested
        if remove_heteroatoms and record_type == "HETATM":
            continue

        output_lines.append(line)

    return "\n".join(output_lines)


def _prepare_receptor_fallback(
    pdb_content: str,
    remove_water: bool = True,
    remove_heteroatoms: bool = False,
    ph: float = 7.4,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Fallback receptor preparation when OpenBabel is not available.

    Uses RDKit for basic conversion. Less robust than OpenBabel but
    provides a working fallback.
    """
    # Filter PDB content
    if remove_water or remove_heteroatoms:
        pdb_content = _filter_pdb_content(pdb_content, remove_water, remove_heteroatoms)

    # Use RDKit to parse and re-export
    # Write to temp file for RDKit
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content)
        temp_pdb = f.name

    try:
        mol = Chem.MolFromPDBFile(temp_pdb, removeHs=False)
        if mol is None:
            return None, None, "RDKit failed to parse PDB content"

        # Add hydrogens
        mol = Chem.AddHs(mol, addCoords=True)

        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)

        # Generate PDBQT manually with proper formatting
        pdbqt_lines = []
        conf = mol.GetConformer()

        atom_type_map = {
            6: "C",
            7: "N",
            8: "O",
            16: "S",
            1: "H",
            15: "P",
            9: "F",
            17: "Cl",
            35: "Br",
            53: "I",
            26: "Fe",
            30: "Zn",
            12: "Mg",
            20: "Ca",
        }

        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atomic_num = atom.GetAtomicNum()

            # Get atom type
            atom_type = atom_type_map.get(atomic_num, "C")

            # Refine atom type for H-bonding
            if atomic_num == 8:
                atom_type = "OA"  # Oxygen acceptor
            elif atomic_num == 7:
                if atom.GetTotalNumHs() < 3:
                    atom_type = "NA"  # Nitrogen acceptor
            elif atomic_num == 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() in (7, 8):
                        atom_type = "HD"  # Hydrogen donor
                        break
            elif atomic_num == 16:
                atom_type = "SA"  # Sulfur acceptor

            # Get charge
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if not (-10 < charge < 10):
                    charge = 0.0
            except (KeyError, ValueError):
                charge = 0.0

            # Get atom info from RDKit
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info:
                atom_name = pdb_info.GetName()
                res_name = pdb_info.GetResidueName()
                chain_id = pdb_info.GetChainId()
                res_num = pdb_info.GetResidueNumber()
            else:
                atom_name = atom.GetSymbol()
                res_name = "UNK"
                chain_id = "A"
                res_num = 1

            # Format PDBQT line - must follow exact column spec
            # OpenBabel format: charge is right-justified, no trailing space before atom type
            line = (
                f"ATOM  {i + 1:5d} {atom_name:4s} {res_name:3s} {chain_id:1s}{res_num:4d}    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:+.3f} {atom_type:>2s}"
            )
            pdbqt_lines.append(line)

        pdbqt_lines.append("END")
        pdbqt_string = "\n".join(pdbqt_lines)

        logger.info(
            f"Prepared receptor with RDKit fallback: {len(pdbqt_lines) - 1} atoms"
        )
        return pdbqt_string, len(pdbqt_lines) - 1, None

    finally:
        Path(temp_pdb).unlink(missing_ok=True)


def prepare_ligand(
    sdf_content: Optional[str] = None,
    smiles: Optional[str] = None,
    add_hydrogens: bool = True,
    ph: float = 7.4,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """Convert SDF/SMILES to PDBQT format for ligand using Meeko.

    Meeko is the preferred tool for ligand preparation as it handles:
    - Torsion tree generation for flexible bonds
    - Proper AutoDock atom typing
    - Gasteiger partial charges

    Args:
        sdf_content: SDF/MOL format string of the ligand.
        smiles: SMILES string (alternative, will generate 3D coords).
        add_hydrogens: Add hydrogens to the molecule.
        ph: pH for protonation state.

    Returns:
        Tuple of (pdbqt_string, num_atoms, num_rotatable_bonds, error_message).
        On success: (pdbqt, atom_count, rot_bonds, None)
        On failure: (None, None, None, error_message)
    """
    try:
        # Get molecule from input
        mol = None

        if sdf_content:
            mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
            if mol is None:
                return None, None, None, "Failed to parse SDF content"
        elif smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None, None, f"Failed to parse SMILES: {smiles}"

            # Generate 3D coordinates
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result == -1:
                # Try with random coords
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result == -1:
                    return None, None, None, "Failed to generate 3D coordinates"

            # Optimize geometry
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass  # Continue with unoptimized geometry
        else:
            return None, None, None, "Either sdf_content or smiles must be provided"

        # Add hydrogens if requested and not already present
        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=True)

        # Calculate properties
        num_atoms = mol.GetNumAtoms()
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Try Meeko first (preferred method)
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)

            # Get first setup (usually there's only one)
            if mol_setups:
                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
                    mol_setups[0]
                )

                if not is_ok:
                    logger.warning(f"Meeko warning: {error_msg}")

                if pdbqt_string:
                    logger.info(
                        f"Prepared ligand with Meeko: {num_atoms} atoms, {num_rotatable} rotatable bonds"
                    )
                    return pdbqt_string, num_atoms, num_rotatable, None

            logger.warning("Meeko returned no setups, using fallback")

        except ImportError:
            logger.warning("Meeko not available, using fallback PDBQT generation")
        except Exception as e:
            logger.warning(f"Meeko preparation failed, using fallback: {e}")

        # Try OpenBabel as secondary option
        try:
            pdbqt_string = _prepare_ligand_openbabel(mol)
            if pdbqt_string:
                logger.info(
                    f"Prepared ligand with OpenBabel: {num_atoms} atoms, {num_rotatable} rotatable bonds"
                )
                return pdbqt_string, num_atoms, num_rotatable, None
        except Exception as e:
            logger.warning(f"OpenBabel ligand prep failed: {e}")

        # Last resort: simple PDBQT generation
        pdbqt_string = _generate_simple_pdbqt(mol)

        if pdbqt_string:
            logger.info(
                f"Prepared ligand with fallback: {num_atoms} atoms, {num_rotatable} rotatable bonds"
            )
            return pdbqt_string, num_atoms, num_rotatable, None
        else:
            return None, None, None, "Failed to generate PDBQT"

    except Exception as e:
        logger.exception("Failed to prepare ligand")
        return None, None, None, f"Ligand preparation failed: {str(e)}"


def count_rotatable_bonds(sdf_content: str) -> int:
    """Count rotatable bonds in a molecule from its SDF content.

    Uses RDKit's ``CalcNumRotatableBonds`` which implements the standard
    Lipinski definition (single, non-ring, non-terminal bonds connecting
    non-hydrogen atoms).

    Args:
        sdf_content: SDF/MOL block string.

    Returns:
        Number of rotatable bonds.

    Raises:
        ValueError: If the SDF content cannot be parsed.
    """
    mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
    if mol is None:
        raise ValueError("Failed to parse SDF content for rotatable bond count")
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def prepare_ligand_rigid(mol) -> Tuple[Optional[str], Optional[str]]:
    """Prepare a rigid PDBQT for a single-conformer RDKit molecule.

    Uses Meeko (preferred) or OpenBabel for proper AutoDock atom typing
    and Gasteiger charge assignment, then flattens the torsion tree to
    produce a rigid PDBQT with ``TORSDOF 0`` and zero active torsions.

    Args:
        mol: RDKit Mol with 3D coordinates and explicit hydrogens.

    Returns:
        Tuple of (pdbqt_string, error_message).
        On success: (pdbqt, None).
        On failure: (None, error_message).
    """
    try:
        if mol.GetNumConformers() == 0:
            return None, "Molecule has no 3D conformer"

        flexible_pdbqt = None

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if mol_setups:
                pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
                if pdbqt_str:
                    flexible_pdbqt = pdbqt_str
                    logger.debug("Rigid prep: atom types via Meeko")
        except ImportError:
            logger.debug("Rigid prep: Meeko unavailable, trying OpenBabel")
        except Exception as exc:
            logger.debug(f"Rigid prep: Meeko failed ({exc}), trying OpenBabel")

        if flexible_pdbqt is None:
            try:
                flexible_pdbqt = _prepare_ligand_openbabel(mol)
                if flexible_pdbqt:
                    logger.debug("Rigid prep: atom types via OpenBabel")
            except Exception as exc:
                logger.debug(f"Rigid prep: OpenBabel failed ({exc})")

        if flexible_pdbqt is None:
            return None, "Neither Meeko nor OpenBabel could produce PDBQT"

        rigid_pdbqt = _flatten_pdbqt_to_rigid(flexible_pdbqt)
        if rigid_pdbqt:
            return rigid_pdbqt, None
        return None, "Failed to flatten PDBQT to rigid format"
    except Exception as e:
        logger.exception("Rigid ligand preparation failed")
        return None, f"Rigid ligand preparation failed: {str(e)}"


def _flatten_pdbqt_to_rigid(pdbqt: str) -> Optional[str]:
    """Flatten a flexible PDBQT into a rigid one (zero active torsions).

    Strips BRANCH/ENDBRANCH/ROOT/ENDROOT keywords, keeps only
    ATOM/HETATM lines, and wraps them in a single ROOT block with
    ``TORSDOF 0``.
    """
    atom_lines = [
        line for line in pdbqt.split("\n") if line.startswith(("ATOM", "HETATM"))
    ]
    if not atom_lines:
        return None
    return "\n".join(["ROOT"] + atom_lines + ["ENDROOT", "TORSDOF 0"])


def _prepare_ligand_openbabel(mol) -> Optional[str]:
    """Prepare ligand using OpenBabel (secondary option after Meeko)."""
    try:
        from openbabel import openbabel

        # Export RDKit mol to SDF string
        sdf_string = Chem.MolToMolBlock(mol)

        # Convert using OpenBabel
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdbqt")

        ob_mol = openbabel.OBMol()
        success = obConversion.ReadString(ob_mol, sdf_string)
        if not success:
            return None

        # Assign charges
        charge_model = openbabel.OBChargeModel.FindType("gasteiger")
        if charge_model:
            charge_model.ComputeCharges(ob_mol)

        pdbqt_string = obConversion.WriteString(ob_mol)
        return pdbqt_string.strip() if pdbqt_string else None

    except ImportError:
        return None


def _generate_simple_pdbqt(mol) -> Optional[str]:
    """Generate simple PDBQT without torsion tree (last resort fallback).

    This is a simplified conversion that doesn't handle flexible bonds
    properly, but works for rigid docking.

    Args:
        mol: RDKit molecule with 3D coordinates.

    Returns:
        PDBQT format string or None on failure.
    """
    try:
        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)

        lines = ["ROOT"]

        conf = mol.GetConformer()

        # Atom type mapping
        atom_type_map = {
            6: "C",  # Carbon
            7: "N",  # Nitrogen (can be NA for H-bond acceptor)
            8: "O",  # Oxygen (can be OA for H-bond acceptor)
            16: "S",  # Sulfur (can be SA for H-bond acceptor)
            1: "H",  # Hydrogen (HD for polar H)
            9: "F",  # Fluorine
            17: "Cl",  # Chlorine
            35: "Br",  # Bromine
            53: "I",  # Iodine
            15: "P",  # Phosphorus
        }

        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)

            # Get element and determine atom type
            atomic_num = atom.GetAtomicNum()
            base_type = atom_type_map.get(atomic_num, "C")

            # Refine atom type based on context
            atom_type = base_type

            # Oxygen: OA if can accept H-bond
            if atomic_num == 8:
                atom_type = "OA"
            # Nitrogen: NA if can accept H-bond, N otherwise
            elif atomic_num == 7:
                # Check if nitrogen has lone pairs (acceptor)
                if atom.GetTotalNumHs() < 3:
                    atom_type = "NA"
            # Hydrogen: HD if attached to N or O (donor)
            elif atomic_num == 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() in (7, 8):
                        atom_type = "HD"
                        break
            # Sulfur: SA if can accept
            elif atomic_num == 16:
                atom_type = "SA"

            # Get partial charge
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if not (-10 < charge < 10):  # Sanity check
                    charge = 0.0
            except (KeyError, ValueError):
                charge = 0.0

            # Format PDBQT ATOM line - use OpenBabel-compatible format
            # The charge field should be formatted to match what Vina expects
            atom_name = atom.GetSymbol()
            line = (
                f"ATOM  {i + 1:5d} {atom_name:4s} LIG A   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"  1.00  0.00    {charge:+.3f} {atom_type:>2s}"
            )
            lines.append(line)

        lines.append("ENDROOT")
        lines.append("TORSDOF 0")  # No torsions in simple mode

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Failed to generate simple PDBQT")
        return None


def pdbqt_to_pdb(pdbqt_content: str) -> str:
    """Convert PDBQT back to PDB format.

    Strips Vina-specific columns (charge, atom type) and keywords
    (ROOT, ENDROOT, BRANCH, ENDBRANCH, TORSDOF).

    Args:
        pdbqt_content: PDBQT format string.

    Returns:
        PDB format string.
    """
    pdb_lines = []

    skip_keywords = {"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"}

    for line in pdbqt_content.split("\n"):
        # Skip Vina-specific keywords
        parts = line.split()
        if parts and parts[0] in skip_keywords:
            continue

        # Convert ATOM/HETATM lines
        if line.startswith(("ATOM", "HETATM")):
            # Keep only first 66 characters (standard PDB)
            pdb_line = line[:66] if len(line) >= 66 else line

            # Add element symbol at columns 77-78 if we can extract it
            if len(line) > 77:
                element = line[77:79].strip()
                if element:
                    pdb_line = pdb_line.ljust(76) + f"  {element:>2s}"

            pdb_lines.append(pdb_line)
        elif line.startswith(("TER", "END", "MODEL", "ENDMDL", "REMARK")):
            pdb_lines.append(line)

    return "\n".join(pdb_lines)
