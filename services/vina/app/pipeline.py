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

"""Auto-routing docking pipeline.

Implements the ``/dock-auto`` logic: given raw PDB/PDBQT + SDF,
automatically selects the optimal docking strategy based on GPU
availability and ligand flexibility (rotatable bond count).

If the receptor is already in PDBQT format (e.g. pre-prepared training
data), the preparation step is skipped entirely to avoid double-
preparation that can corrupt the file and trigger false size-limit
rejections.

Four docking cases:
    1. CPU + rot_bonds <= threshold  ->  input pose + Meeko torsion tree + Vina flexible
    2. CPU + rot_bonds >  threshold  ->  RDKit conformer ensemble + Vina rigid batch
    3. GPU + rot_bonds <= threshold  ->  input pose + Meeko torsion tree + Uni-Dock flexible
    4. GPU + rot_bonds >  threshold  ->  nvMolKit conformer ensemble + Uni-Dock rigid batch

Only case 4 generates conformers on the GPU, and it does so in a separate
worker process (see ``conformer_client``); the flexible cases dock the pose in
the SDF (embedded with RDKit only when it arrives as SMILES).

Result dicts may carry a ``_diagnostics`` key for internal verification. It is
not part of ``DockResponse`` and is never returned to API clients.
"""

import logging
import os
import time
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from .conformer import generate_conformers_with_report
from .docking import (
    SCORE_MODES,
    VINA_GPU_ENABLED,
    run_docking,
    run_rigid_batch_docking,
    validate_docking_params,
    DockingValidationError,
)
from .preparation import (
    count_rotatable_bonds,
    is_pdbqt,
    prepare_ligand,
    prepare_ligand_rigid,
    prepare_receptor,
    validate_pdbqt_receptor,
    PDBQTValidationError,
)

logger = logging.getLogger(__name__)

DEFAULT_ROT_BOND_THRESHOLD = int(os.environ.get("ROT_BOND_THRESHOLD", "15"))
DEFAULT_CONFORMER_COUNT = int(os.environ.get("CONFORMER_COUNT", "50"))


def run_auto_dock(
    receptor_pdb: str,
    ligand_sdf: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 32,
    num_modes: int = 9,
    energy_range: float = 3.0,
    num_conformers: int = DEFAULT_CONFORMER_COUNT,
    rot_bond_threshold: int = DEFAULT_ROT_BOND_THRESHOLD,
    score_mode: str = "search",
) -> dict:
    """Run the full docking pipeline with automatic strategy selection.

    Accepts either raw PDB **or** pre-prepared PDBQT as the receptor.
    When the input is already PDBQT (detected via :func:`is_pdbqt`),
    preparation is skipped to avoid double-conversion and false
    size-limit rejections.

    Args:
        receptor_pdb: Receptor structure content (PDB *or* PDBQT format).
        ligand_sdf: Raw SDF content for the ligand.
        center: (x, y, z) search box centre in Angstroms.
        size: (x, y, z) search box dimensions in Angstroms.
        exhaustiveness: Search thoroughness (1-64).
        num_modes: Max poses to return.
        energy_range: Energy range for poses (kcal/mol).
        num_conformers: Number of conformers for ensemble mode.
        rot_bond_threshold: Rotatable-bond threshold for ensemble mode.
        score_mode: ``search`` (default), ``local`` or ``score``. The latter two
            keep the pose in ``ligand_sdf`` instead of searching for a new one,
            and therefore never take the ensemble branch.

    Returns:
        Dict matching ``DockResponse`` schema (success, poses, best_affinity,
        execution_time_ms, gpu_accelerated, docking_mode, conformers_generated,
        best_conformer_idx, error).
    """
    start_time = time.time()

    # ------------------------------------------------------------------ #
    # 1. Prepare receptor (skip if already PDBQT)
    # ------------------------------------------------------------------ #
    if is_pdbqt(receptor_pdb):
        logger.info("Receptor detected as PDBQT — skipping preparation")
        try:
            validate_pdbqt_receptor(receptor_pdb)
        except PDBQTValidationError as e:
            return _error_response(
                f"Pre-prepared PDBQT validation failed: {e}",
                start_time,
            )
        receptor_pdbqt = receptor_pdb
    else:
        receptor_pdbqt, _n_atoms, receptor_error = prepare_receptor(
            receptor_pdb,
            remove_water=True,
            remove_heteroatoms=True,
        )
        if receptor_error:
            return _error_response(
                f"Receptor preparation failed: {receptor_error}",
                start_time,
            )
        if not receptor_pdbqt:
            return _error_response(
                "Receptor preparation returned empty PDBQT",
                start_time,
            )

    # ------------------------------------------------------------------ #
    # 2. Parse ligand and count rotatable bonds
    # ------------------------------------------------------------------ #
    try:
        rot_bonds = count_rotatable_bonds(ligand_sdf)
    except ValueError as e:
        return _error_response(str(e), start_time)

    if score_mode not in SCORE_MODES:
        return _error_response(
            f"Unknown score_mode {score_mode!r}; expected one of {', '.join(SCORE_MODES)}",
            start_time,
        )

    # The ensemble branch regenerates the ligand's conformers and docks those,
    # which discards the pose it was given. That is the right thing when the
    # question is "where does this bind"; it is the wrong thing, and silently so,
    # when the question is "how good is this pose".
    use_ensemble = rot_bonds > rot_bond_threshold and score_mode == "search"
    use_gpu = VINA_GPU_ENABLED and score_mode == "search"

    docking_mode = "rigid_ensemble" if use_ensemble else "flexible"
    logger.info(
        f"Auto-dock: rot_bonds={rot_bonds}, threshold={rot_bond_threshold}, "
        f"gpu={use_gpu}, mode={docking_mode}, score_mode={score_mode}"
    )

    # ------------------------------------------------------------------ #
    # 3. Branch: flexible vs rigid-ensemble
    # ------------------------------------------------------------------ #
    if use_ensemble:
        return _run_ensemble_path(
            receptor_pdbqt=receptor_pdbqt,
            ligand_sdf=ligand_sdf,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            energy_range=energy_range,
            num_conformers=num_conformers,
            use_gpu=use_gpu,
            rot_bonds=rot_bonds,
            start_time=start_time,
        )
    else:
        return _run_flexible_path(
            receptor_pdbqt=receptor_pdbqt,
            ligand_sdf=ligand_sdf,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            energy_range=energy_range,
            use_gpu=use_gpu,
            rot_bonds=rot_bonds,
            start_time=start_time,
            score_mode=score_mode,
        )


# ---------------------------------------------------------------------- #
# Flexible docking (cases 1 & 3)
# ---------------------------------------------------------------------- #


def _run_flexible_path(
    *,
    receptor_pdbqt: str,
    ligand_sdf: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    use_gpu: bool,
    rot_bonds: int,
    start_time: float,
    score_mode: str = "search",
) -> dict:
    """Standard flexible docking via Meeko torsion tree + Vina/Uni-Dock.

    Also the path for ``local`` and ``score``: Meeko writes the torsion tree
    around the coordinates it is given, so the pose in ``ligand_sdf`` survives
    preparation and is what Vina starts from.
    """

    # Prepare ligand (Meeko produces torsion tree)
    ligand_pdbqt, n_atoms, n_rot, lig_error = prepare_ligand(
        sdf_content=ligand_sdf,
        add_hydrogens=True,
    )
    if lig_error:
        return _error_response(f"Ligand preparation failed: {lig_error}", start_time)
    if not ligand_pdbqt:
        return _error_response("Ligand preparation returned empty PDBQT", start_time)

    # Validate complexity
    try:
        validate_docking_params(receptor_pdbqt, ligand_pdbqt, size, exhaustiveness)
    except DockingValidationError as e:
        return _error_response(str(e), start_time)

    # Dock
    success, poses, best_affinity, exec_ms, gpu_used, error = run_docking(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        size=size,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range,
        score_mode=score_mode,
    )

    total_ms = int((time.time() - start_time) * 1000)

    if not success:
        return _error_response(
            f"Docking failed: {error}", start_time, gpu_used=gpu_used
        )

    return {
        "success": True,
        "gpu_accelerated": gpu_used,
        "execution_time_ms": total_ms,
        "poses": [
            {
                "rank": p.rank,
                "affinity": p.affinity,
                "rmsd_lb": p.rmsd_lb,
                "rmsd_ub": p.rmsd_ub,
                "pdbqt": p.pdbqt,
            }
            for p in poses
        ],
        "best_affinity": best_affinity,
        "docking_mode": "flexible" if score_mode == "search" else score_mode,
        "conformers_generated": 0,
        "best_conformer_idx": 0,
        "rotatable_bonds": rot_bonds,
        "error": None,
    }


# ---------------------------------------------------------------------- #
# Rigid ensemble docking (cases 2 & 4)
# ---------------------------------------------------------------------- #


def _run_ensemble_path(
    *,
    receptor_pdbqt: str,
    ligand_sdf: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    num_conformers: int,
    use_gpu: bool,
    rot_bonds: int,
    start_time: float,
) -> dict:
    """Conformer ensemble generation + rigid batch docking."""

    # Parse the SDF into an RDKit mol
    mol = Chem.MolFromMolBlock(ligand_sdf, removeHs=False)
    if mol is None:
        return _error_response(
            "Failed to parse ligand SDF for conformer generation", start_time
        )

    # Generate conformer ensemble (nvMolKit worker on GPU, RDKit on CPU)
    conformer_mols, conformer_report = generate_conformers_with_report(
        mol,
        num_conformers=num_conformers,
        use_gpu=use_gpu,
    )
    diagnostics = {"conformer": conformer_report.as_dict()}

    if not conformer_mols:
        return {
            **_error_response("Conformer generation failed — no conformers produced", start_time),
            "_diagnostics": diagnostics,
        }

    logger.info(f"Preparing {len(conformer_mols)} rigid PDBQTs for batch docking")

    # Convert each conformer to a rigid PDBQT (TORSDOF 0)
    rigid_pdbqts: list[str] = []
    for i, conf_mol in enumerate(conformer_mols):
        pdbqt, err = prepare_ligand_rigid(conf_mol)
        if err or not pdbqt:
            logger.warning(f"Skipping conformer {i}: {err}")
            continue
        rigid_pdbqts.append(pdbqt)

    if not rigid_pdbqts:
        return {
            **_error_response("All conformers failed rigid PDBQT preparation", start_time),
            "_diagnostics": diagnostics,
        }

    # Batch dock all conformers rigidly
    (
        success,
        poses,
        best_affinity,
        exec_ms,
        gpu_used,
        best_conf_idx,
        error,
    ) = run_rigid_batch_docking(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqts=rigid_pdbqts,
        center=center,
        size=size,
        exhaustiveness=exhaustiveness,
        num_modes=1,  # one pose per rigid conformer
        energy_range=energy_range,
    )

    total_ms = int((time.time() - start_time) * 1000)

    if not success:
        return {
            **_error_response(f"Rigid ensemble docking failed: {error}", start_time, gpu_used=gpu_used),
            "_diagnostics": diagnostics,
        }

    return {
        "success": True,
        "gpu_accelerated": gpu_used,
        "execution_time_ms": total_ms,
        "poses": [
            {
                "rank": p.rank,
                "affinity": p.affinity,
                "rmsd_lb": p.rmsd_lb,
                "rmsd_ub": p.rmsd_ub,
                "pdbqt": p.pdbqt,
            }
            for p in poses
        ],
        "best_affinity": best_affinity,
        "docking_mode": "rigid_ensemble",
        "conformers_generated": len(rigid_pdbqts),
        "best_conformer_idx": best_conf_idx,
        "rotatable_bonds": rot_bonds,
        "error": None,
        "_diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _error_response(
    message: str,
    start_time: float,
    gpu_used: bool = False,
) -> dict:
    total_ms = int((time.time() - start_time) * 1000)
    logger.error(f"Auto-dock error: {message}")
    return {
        "success": False,
        "gpu_accelerated": gpu_used,
        "execution_time_ms": total_ms,
        "poses": [],
        "best_affinity": None,
        "docking_mode": "unknown",
        "conformers_generated": 0,
        "best_conformer_idx": 0,
        "rotatable_bonds": None,
        "error": message,
    }
