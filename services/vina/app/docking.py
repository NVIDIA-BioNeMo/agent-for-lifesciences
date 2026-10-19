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

"""Autodock Vina execution wrapper.

This module provides functions to execute Vina docking and parse results.
Supports both CPU (vina) and GPU (Uni-Dock) execution modes.
"""

import logging
import os
import re
import subprocess
import tempfile
import time
import json
import resource
from pathlib import Path
from typing import List, Optional, Tuple

from .models import DockingPose

logger = logging.getLogger(__name__)

# #region agent log
DEBUG_LOG_PATH = os.environ.get("DEBUG_LOG_PATH", "/app/debug_logs/debug.log")


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
# Docking Safety Limits
# =============================================================================

# Limit docking complexity to prevent OOM
# Based on testing: Imatinib (68 atoms, 14 torsions) + PGK1 (3854 atoms) at exhaust=32
# completes in 61s with ~100MB memory. This is complexity score ~1.7M.
# Set limits above tested values but below catastrophic levels.
#
# NOTE: Meeko assigns more torsions than RDKit's CalcNumRotatableBonds
# (e.g. amide bonds, ring-substituent bonds).  Uni-Dock handles up to
# ~40 torsions without issues; allow 40 here to avoid false rejections
# for training-data ligands.
MAX_RECEPTOR_ATOMS = 100_000  # Max atoms in receptor PDBQT (large assemblies)
MAX_LIGAND_ATOMS = 200  # Max atoms in ligand PDBQT
MAX_LIGAND_TORSIONS = 40  # Max torsions (BRANCH keywords, not same as rotatable bonds)
MAX_SEARCH_VOLUME = 125000  # Max search box volume (50^3)
MAX_COMPLEXITY_SCORE = 15_000_000  # receptor_atoms * ligand_torsions * exhaustiveness


class DockingValidationError(Exception):
    """Raised when docking parameters exceed safe limits."""

    pass


def validate_docking_params(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    size: Tuple[float, float, float],
    exhaustiveness: int,
) -> dict:
    """Validate docking parameters and calculate complexity score.

    Args:
        receptor_pdbqt: Receptor in PDBQT format.
        ligand_pdbqt: Ligand in PDBQT format.
        size: Search box dimensions.
        exhaustiveness: Search thoroughness.

    Returns:
        Dict with validation results.

    Raises:
        DockingValidationError: If parameters exceed safe limits.
    """
    receptor_atoms = receptor_pdbqt.count("\nATOM") + receptor_pdbqt.count("\nHETATM")
    ligand_atoms = ligand_pdbqt.count("\nATOM") + ligand_pdbqt.count("\nHETATM")
    ligand_torsions = ligand_pdbqt.count("BRANCH")  # Each BRANCH = one torsion
    search_volume = size[0] * size[1] * size[2]

    # Complexity score: higher = more memory/time intensive
    complexity_score = receptor_atoms * max(ligand_torsions, 1) * exhaustiveness

    validation = {
        "receptor_atoms": receptor_atoms,
        "ligand_atoms": ligand_atoms,
        "ligand_torsions": ligand_torsions,
        "search_volume": search_volume,
        "exhaustiveness": exhaustiveness,
        "complexity_score": complexity_score,
        "warnings": [],
    }

    # Check limits
    if receptor_atoms > MAX_RECEPTOR_ATOMS:
        raise DockingValidationError(
            f"Receptor too large: {receptor_atoms} atoms > {MAX_RECEPTOR_ATOMS} limit. "
            "Consider using a smaller binding pocket or removing distant residues."
        )

    if ligand_atoms > MAX_LIGAND_ATOMS:
        raise DockingValidationError(
            f"Ligand too large: {ligand_atoms} atoms > {MAX_LIGAND_ATOMS} limit. "
            "This ligand is too complex for safe docking."
        )

    if ligand_torsions > MAX_LIGAND_TORSIONS:
        raise DockingValidationError(
            f"Ligand has too many rotatable bonds: {ligand_torsions} > {MAX_LIGAND_TORSIONS} limit. "
            "Consider using a more rigid ligand or reducing flexibility."
        )

    if search_volume > MAX_SEARCH_VOLUME:
        raise DockingValidationError(
            f"Search box too large: {search_volume:.0f} Å³ > {MAX_SEARCH_VOLUME} limit. "
            "Consider using a smaller search box focused on the binding site."
        )

    if complexity_score > MAX_COMPLEXITY_SCORE:
        raise DockingValidationError(
            f"Docking too complex: score={complexity_score} > {MAX_COMPLEXITY_SCORE} limit. "
            f"(receptor={receptor_atoms}, torsions={ligand_torsions}, exhaust={exhaustiveness}). "
            "Try reducing exhaustiveness or using a simpler ligand."
        )

    # Warnings for borderline cases
    if receptor_atoms > 7500:
        validation["warnings"].append(
            f"Large receptor ({receptor_atoms} atoms) may be slow"
        )
    if ligand_torsions > 10:
        validation["warnings"].append(
            f"Flexible ligand ({ligand_torsions} torsions) may be slow"
        )
    if complexity_score > 300000:
        validation["warnings"].append(
            f"High complexity ({complexity_score}) may take several minutes"
        )

    # #region agent log
    _debug_log(
        "F",
        "docking.py:validate_docking_params",
        "Docking validation complete",
        {
            "receptor_atoms": receptor_atoms,
            "ligand_atoms": ligand_atoms,
            "ligand_torsions": ligand_torsions,
            "search_volume": search_volume,
            "complexity_score": complexity_score,
            "exhaustiveness": exhaustiveness,
            "warnings": validation["warnings"],
        },
    )
    # #endregion

    return validation


# Environment variables for Vina configuration
VINA_GPU_ENABLED = os.environ.get("VINA_GPU", "false").lower() == "true"
VINA_BINARY = os.environ.get("VINA_BINARY", "vina")

# What to do with the pose that was handed in.
#
# ``search``  global search, ignoring the input coordinates -- the docking
#             everything in this service did before.
# ``local``   ``--local_only``: minimise from the input pose and keep it.
# ``score``   ``--score_only``: score the input pose, changing nothing.
#
# The two pose-preserving modes exist for candidates that already have a
# meaningful pose -- a constrained-embedding result, a crystal ligand, a
# scaffold hop off a known binder -- where a global search would throw away the
# very thing being asked about.
SCORE_MODES = ("search", "local", "score")

# Uni-Dock configuration (used when VINA_GPU=true)
UNIDOCK_BINARY = os.environ.get("UNIDOCK_BINARY", "/usr/local/bin/unidock")
UNIDOCK_SEARCH_MODE = os.environ.get("UNIDOCK_SEARCH_MODE", "balance")


def check_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available via nvidia-smi.

    Uni-Dock is CUDA-native, so nvidia-smi is a sufficient check
    (unlike OpenCL-based Vina-GPU 2.1 which also needed ICD loader
    verification).

    Returns:
        True if nvidia-smi succeeds and a GPU is detected.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_vina_version() -> Tuple[str, bool]:
    """Get Vina/Uni-Dock version and check if GPU is available.

    Returns:
        Tuple of (version_string, gpu_available).
    """
    try:
        # Use the Uni-Dock binary for version check when GPU mode is enabled
        binary = UNIDOCK_BINARY if VINA_GPU_ENABLED else VINA_BINARY
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        version = "unknown"
        if result.returncode == 0:
            # Parse version from output
            output = result.stdout + result.stderr
            version_match = re.search(r"(\d+\.\d+\.?\d*)", output)
            if version_match:
                version = version_match.group(1)

        # Check if GPU docking will actually work
        gpu_available = False
        if VINA_GPU_ENABLED:
            gpu_available = check_gpu_available()

        return version, gpu_available

    except Exception as e:
        logger.error(f"Failed to get Vina version: {e}")
        return "unknown", False


def _mode_arguments(
    score_mode: str,
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    output_path: Path,
) -> List[str]:
    """The arguments that differ between the three modes.

    ``--score_only`` refuses ``--out``: it does not move the ligand, so there is
    nothing to write, and Vina treats being asked as a usage error.
    """
    if score_mode == "score":
        return ["--score_only"]
    if score_mode == "local":
        return ["--local_only", "--out", str(output_path)]
    return [
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--energy_range",
        str(energy_range),
        "--out",
        str(output_path),
    ]


# "Estimated Free Energy of Binding   : -9.371 (kcal/mol) [=(1)+(2)+(3)-(4)]"
BINDING_ENERGY_PATTERN = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*([-\d.]+)", re.IGNORECASE
)


def parse_scored_pose(stdout: str, pose_pdbqt: str) -> List[DockingPose]:
    """Build the single pose that ``--local_only`` or ``--score_only`` produces.

    Neither mode writes the ``REMARK VINA RESULT`` line that
    :func:`parse_vina_output` reads -- ``--local_only``'s output file carries only
    the SMILES remarks and the coordinates, and ``--score_only`` writes no file at
    all. Both print the energy to stdout, so that is where it is read from.

    The RMSD columns are zero by construction: they measure the distance to the
    best mode of a search, and there is no search here.
    """
    match = BINDING_ENERGY_PATTERN.search(stdout)
    if match is None:
        return []

    lines = [
        line
        for line in pose_pdbqt.splitlines()
        if line.startswith(
            ("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "TER")
        )
    ]
    if not lines:
        return []

    return [
        DockingPose(
            rank=1,
            affinity=float(match.group(1)),
            rmsd_lb=0.0,
            rmsd_ub=0.0,
            pdbqt="\n".join(lines),
        )
    ]


def run_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 32,
    num_modes: int = 9,
    energy_range: float = 3.0,
    score_mode: str = "search",
) -> Tuple[bool, List[DockingPose], Optional[float], int, bool, Optional[str]]:
    """Execute Vina docking and return results.

    Args:
        receptor_pdbqt: Receptor structure in PDBQT format.
        ligand_pdbqt: Ligand structure in PDBQT format.
        center: (x, y, z) coordinates of search box center.
        size: (x, y, z) dimensions of search box.
        exhaustiveness: Search thoroughness (1-64).
        num_modes: Maximum number of poses to return.
        energy_range: Energy range for poses (kcal/mol).
        score_mode: One of :data:`SCORE_MODES`. ``local`` and ``score`` keep the
            input pose and always run on the CPU binary -- Uni-Dock's
            single-ligand mode implements neither.

    Returns:
        Tuple of:
            - success: bool
            - poses: List[DockingPose]
            - best_affinity: Optional[float]
            - execution_time_ms: int
            - gpu_used: bool
            - error: Optional[str]
    """
    start_time = time.time()

    if score_mode not in SCORE_MODES:
        return (
            False,
            [],
            None,
            0,
            False,
            f"Unknown score_mode {score_mode!r}; expected one of {', '.join(SCORE_MODES)}",
        )
    # No search means no search space to be thorough about, so the complexity
    # bound -- which is a bound on search cost -- is scored at the floor.
    use_gpu = VINA_GPU_ENABLED and score_mode == "search"

    # Validate docking parameters before proceeding
    try:
        validation = validate_docking_params(
            receptor_pdbqt,
            ligand_pdbqt,
            size,
            exhaustiveness if score_mode == "search" else 1,
        )
        for warning in validation.get("warnings", []):
            logger.warning(f"Docking warning: {warning}")
    except DockingValidationError as e:
        logger.error(f"Docking validation failed: {e}")
        return False, [], None, 0, use_gpu, str(e)

    # #region agent log
    receptor_atoms = validation["receptor_atoms"]
    ligand_atoms = validation["ligand_atoms"]
    ligand_torsions = validation["ligand_torsions"]
    receptor_size_kb = len(receptor_pdbqt) / 1024
    ligand_size_kb = len(ligand_pdbqt) / 1024
    search_volume = validation["search_volume"]
    complexity_score = validation["complexity_score"]
    _debug_log(
        "A",
        "docking.py:run_docking:entry",
        "Starting docking",
        {
            "receptor_atoms": receptor_atoms,
            "receptor_size_kb": round(receptor_size_kb, 2),
            "ligand_atoms": ligand_atoms,
            "ligand_size_kb": round(ligand_size_kb, 2),
            "ligand_torsions": ligand_torsions,
            "search_volume": search_volume,
            "complexity_score": complexity_score,
            "exhaustiveness": exhaustiveness,
            "center": list(center),
            "size": list(size),
        },
    )
    # #endregion

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write input files
            receptor_path = tmpdir_path / "receptor.pdbqt"
            ligand_path = tmpdir_path / "ligand.pdbqt"
            output_path = tmpdir_path / "output.pdbqt"

            receptor_path.write_text(receptor_pdbqt)
            ligand_path.write_text(ligand_pdbqt)

            # Build command -- branch on GPU (Uni-Dock) vs CPU (Vina) mode
            # Uni-Dock supports all standard Vina CLI args for single-ligand
            # mode, so the command is nearly identical. Only differences:
            # binary name and explicit --scoring vina.
            box = [
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
            ]
            if use_gpu:
                cmd = (
                    [
                        UNIDOCK_BINARY,
                        "--receptor",
                        str(receptor_path),
                        "--ligand",
                        str(ligand_path),
                    ]
                    + box
                    + [
                        "--scoring",
                        "vina",
                        "--exhaustiveness",
                        str(exhaustiveness),
                        "--num_modes",
                        str(num_modes),
                        "--energy_range",
                        str(energy_range),
                        "--out",
                        str(output_path),
                    ]
                )
            else:
                cmd = (
                    [
                        VINA_BINARY,
                        "--receptor",
                        str(receptor_path),
                        "--ligand",
                        str(ligand_path),
                    ]
                    + box
                    + _mode_arguments(
                        score_mode, exhaustiveness, num_modes, energy_range, output_path
                    )
                )

            backend = "Uni-Dock (GPU)" if use_gpu else "Vina (CPU)"
            logger.info(f"Running {backend} in {score_mode} mode: {' '.join(cmd)}")

            # #region agent log
            _debug_log(
                "A",
                "docking.py:run_docking:before_subprocess",
                "Before docking subprocess",
                {
                    "cmd": cmd,
                    "gpu_mode": VINA_GPU_ENABLED,
                    "receptor_file_size": receptor_path.stat().st_size,
                    "ligand_file_size": ligand_path.stat().st_size,
                },
            )
            # #endregion

            # Execute docking
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for complex docking
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # #region agent log
            _debug_log(
                "A",
                "docking.py:run_docking:after_subprocess",
                "After Vina subprocess",
                {
                    "returncode": result.returncode,
                    "execution_time_ms": execution_time_ms,
                    "stdout_lines": len(result.stdout.split("\n"))
                    if result.stdout
                    else 0,
                    "stderr_lines": len(result.stderr.split("\n"))
                    if result.stderr
                    else 0,
                },
            )
            # #endregion

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown Vina error"
                logger.error(f"Vina failed: {error_msg}")
                # #region agent log
                _debug_log(
                    "A",
                    "docking.py:run_docking:error",
                    "Vina error",
                    {
                        "returncode": result.returncode,
                        "stderr": result.stderr[:500] if result.stderr else None,
                        "stdout": result.stdout[:500] if result.stdout else None,
                    },
                )
                # #endregion
                return False, [], None, execution_time_ms, use_gpu, error_msg

            if score_mode == "search":
                if not output_path.exists():
                    return (
                        False,
                        [],
                        None,
                        execution_time_ms,
                        use_gpu,
                        "No output file generated",
                    )
                poses = parse_vina_output(output_path.read_text())
                failure = "No poses found in output"
            else:
                # --score_only writes nothing, so the pose that comes back is the
                # one that went in; --local_only writes the minimised pose but no
                # REMARK VINA RESULT, so the affinity comes off stdout either way.
                moved = output_path.read_text() if output_path.exists() else ligand_pdbqt
                poses = parse_scored_pose(result.stdout, moved)
                failure = (
                    "Vina reported no binding energy; the pose may lie outside the box"
                )

            if not poses:
                return False, [], None, execution_time_ms, use_gpu, failure

            best_affinity = poses[0].affinity

            logger.info(
                f"Docking complete ({score_mode}): {len(poses)} poses, "
                f"best affinity: {best_affinity:.2f} kcal/mol, "
                f"time: {execution_time_ms}ms"
            )

            return True, poses, best_affinity, execution_time_ms, use_gpu, None

    except subprocess.TimeoutExpired:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return (
            False,
            [],
            None,
            execution_time_ms,
            use_gpu,
            "Docking timed out (>10 minutes)",
        )
    except FileNotFoundError:
        execution_time_ms = int((time.time() - start_time) * 1000)
        binary = UNIDOCK_BINARY if use_gpu else VINA_BINARY
        return (
            False,
            [],
            None,
            execution_time_ms,
            False,
            f"Docking binary not found: {binary}",
        )
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.exception("Docking failed")
        return False, [], None, execution_time_ms, use_gpu, str(e)


def parse_vina_output(pdbqt_content: str) -> List[DockingPose]:
    """Parse Vina output PDBQT file containing multiple poses.

    Vina outputs multiple MODELs, each with a REMARK containing:
    - REMARK VINA RESULT:    -7.2      0.000      0.000

    Args:
        pdbqt_content: Content of Vina output PDBQT file.

    Returns:
        List of DockingPose objects sorted by affinity.
    """
    poses = []

    # Split by MODEL
    models = re.split(r"\nMODEL\s+\d+\s*\n", "\n" + pdbqt_content)

    for i, model in enumerate(models):
        if not model.strip():
            continue

        # Look for REMARK VINA RESULT line
        result_match = re.search(
            r"REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", model
        )

        if result_match:
            affinity = float(result_match.group(1))
            rmsd_lb = float(result_match.group(2))
            rmsd_ub = float(result_match.group(3))

            # Extract the pose coordinates (everything except MODEL/ENDMDL/REMARK lines)
            pose_lines = []
            for line in model.split("\n"):
                if line.startswith(
                    (
                        "ATOM",
                        "HETATM",
                        "ROOT",
                        "ENDROOT",
                        "BRANCH",
                        "ENDBRANCH",
                        "TORSDOF",
                        "TER",
                    )
                ):
                    pose_lines.append(line)

            pose_pdbqt = "\n".join(pose_lines)

            if pose_pdbqt.strip():
                poses.append(
                    DockingPose(
                        rank=len(poses) + 1,
                        affinity=affinity,
                        rmsd_lb=rmsd_lb,
                        rmsd_ub=rmsd_ub,
                        pdbqt=pose_pdbqt,
                    )
                )

    # If no MODEL delimiters, try parsing as single pose
    if not poses:
        result_match = re.search(
            r"REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", pdbqt_content
        )

        if result_match:
            affinity = float(result_match.group(1))
            rmsd_lb = float(result_match.group(2))
            rmsd_ub = float(result_match.group(3))

            pose_lines = []
            for line in pdbqt_content.split("\n"):
                if line.startswith(
                    (
                        "ATOM",
                        "HETATM",
                        "ROOT",
                        "ENDROOT",
                        "BRANCH",
                        "ENDBRANCH",
                        "TORSDOF",
                        "TER",
                    )
                ):
                    pose_lines.append(line)

            pose_pdbqt = "\n".join(pose_lines)

            if pose_pdbqt.strip():
                poses.append(
                    DockingPose(
                        rank=1,
                        affinity=affinity,
                        rmsd_lb=rmsd_lb,
                        rmsd_ub=rmsd_ub,
                        pdbqt=pose_pdbqt,
                    )
                )

    # Sort by affinity (most negative first)
    poses.sort(key=lambda p: p.affinity)

    # Update ranks after sorting
    for i, pose in enumerate(poses):
        pose.rank = i + 1

    return poses


def run_docking_python_api(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 32,
    num_modes: int = 9,
    energy_range: float = 3.0,
) -> Tuple[bool, List[DockingPose], Optional[float], int, bool, Optional[str]]:
    """Execute Vina docking using Python API (alternative to CLI).

    This uses the vina Python package directly instead of calling the binary.
    May be more reliable in some containerized environments.

    Args:
        Same as run_docking()

    Returns:
        Same as run_docking()
    """
    start_time = time.time()

    try:
        from vina import Vina

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write input files
            receptor_path = tmpdir_path / "receptor.pdbqt"
            ligand_path = tmpdir_path / "ligand.pdbqt"
            output_path = tmpdir_path / "output.pdbqt"

            receptor_path.write_text(receptor_pdbqt)
            ligand_path.write_text(ligand_pdbqt)

            # Initialize Vina
            v = Vina(sf_name="vina")

            # Set receptor
            v.set_receptor(str(receptor_path))

            # Set ligand
            v.set_ligand_from_file(str(ligand_path))

            # Set search box
            v.compute_vina_maps(
                center=list(center),
                box_size=list(size),
            )

            # Run docking
            v.dock(
                exhaustiveness=exhaustiveness,
                n_poses=num_modes,
            )

            # Write output
            v.write_poses(str(output_path), n_poses=num_modes)

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Parse output
            output_content = output_path.read_text()
            poses = parse_vina_output(output_content)

            if not poses:
                return False, [], None, execution_time_ms, False, "No poses found"

            best_affinity = poses[0].affinity if poses else None

            logger.info(
                f"Docking (Python API) complete: {len(poses)} poses, "
                f"best affinity: {best_affinity:.2f} kcal/mol"
            )

            return True, poses, best_affinity, execution_time_ms, False, None

    except ImportError:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return (
            False,
            [],
            None,
            execution_time_ms,
            False,
            "Vina Python package not available",
        )
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.exception("Docking (Python API) failed")
        return False, [], None, execution_time_ms, False, str(e)


UNIDOCK_MAX_BOX_DIMENSION = 35.0  # Angstroms; larger boxes trigger CUDA errors


def run_rigid_batch_docking(
    receptor_pdbqt: str,
    ligand_pdbqts: List[str],
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 32,
    num_modes: int = 1,
    energy_range: float = 3.0,
) -> Tuple[bool, List[DockingPose], Optional[float], int, bool, int, Optional[str]]:
    """Dock an ensemble of rigid conformers and return the best result.

    Each ligand PDBQT should have ``TORSDOF 0`` (no rotatable bonds).

    GPU path (Uni-Dock): writes all conformers to a ``conformers/``
    directory, creates a single ``ligand_index.txt``, and makes one
    ``unidock`` call with ``--ligand_index`` + ``--dir``.  Uni-Dock
    handles its own internal dynamic batching.

    CPU path (Vina): uses ``--batch`` with the conformer files listed
    directly on the command line.

    Args:
        receptor_pdbqt: Receptor in PDBQT format.
        ligand_pdbqts: List of rigid ligand PDBQTs (one per conformer).
        center: (x, y, z) search box center.
        size: (x, y, z) search box dimensions.
        exhaustiveness: Search thoroughness.
        num_modes: Poses per conformer (typically 1 for rigid ensemble).
        energy_range: Energy range for poses.

    Returns:
        Tuple of (success, poses, best_affinity, exec_time_ms, gpu_used,
        best_conformer_idx, error).
    """
    start_time = time.time()

    if not ligand_pdbqts:
        return False, [], None, 0, VINA_GPU_ENABLED, 0, "No conformer PDBQTs provided"

    if VINA_GPU_ENABLED:
        for dim_name, dim_val in zip(("x", "y", "z"), size):
            if dim_val > UNIDOCK_MAX_BOX_DIMENSION:
                return (
                    False,
                    [],
                    None,
                    0,
                    True,
                    0,
                    f"Uni-Dock search box size_{dim_name}={dim_val:.1f} A exceeds "
                    f"{UNIDOCK_MAX_BOX_DIMENSION:.0f} A limit. Use a smaller box "
                    "focused on the binding site.",
                )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            receptor_path = tmpdir_path / "receptor.pdbqt"
            receptor_path.write_text(receptor_pdbqt)

            conformer_dir = tmpdir_path / "conformers"
            conformer_dir.mkdir()
            output_dir = tmpdir_path / "output_poses"
            output_dir.mkdir()

            for i, pdbqt in enumerate(ligand_pdbqts):
                (conformer_dir / f"molecule_conf_{i + 1:03d}.pdbqt").write_text(pdbqt)

            conformer_files = sorted(conformer_dir.glob("*.pdbqt"))

            if VINA_GPU_ENABLED:
                index_path = tmpdir_path / "ligand_index.txt"
                index_path.write_text("\n".join(str(f) for f in conformer_files) + "\n")
                cmd = [
                    UNIDOCK_BINARY,
                    "--receptor",
                    str(receptor_path),
                    "--ligand_index",
                    str(index_path),
                    "--center_x",
                    str(center[0]),
                    "--center_y",
                    str(center[1]),
                    "--center_z",
                    str(center[2]),
                    "--size_x",
                    str(size[0]),
                    "--size_y",
                    str(size[1]),
                    "--size_z",
                    str(size[2]),
                    "--dir",
                    str(output_dir),
                    "--search_mode",
                    UNIDOCK_SEARCH_MODE,
                    "--scoring",
                    "vina",
                    "--num_modes",
                    str(num_modes),
                ]
            else:
                cmd = (
                    [
                        VINA_BINARY,
                        "--receptor",
                        str(receptor_path),
                        "--batch",
                    ]
                    + [str(f) for f in conformer_files]
                    + [
                        "--center_x",
                        str(center[0]),
                        "--center_y",
                        str(center[1]),
                        "--center_z",
                        str(center[2]),
                        "--size_x",
                        str(size[0]),
                        "--size_y",
                        str(size[1]),
                        "--size_z",
                        str(size[2]),
                        "--exhaustiveness",
                        str(exhaustiveness),
                        "--num_modes",
                        str(num_modes),
                        "--energy_range",
                        str(energy_range),
                        "--dir",
                        str(output_dir),
                    ]
                )

            logger.info(
                f"Running rigid ensemble docking: {len(ligand_pdbqts)} conformers, "
                f"backend={'Uni-Dock' if VINA_GPU_ENABLED else 'Vina'}"
            )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown docking error"
                logger.error(f"Ensemble docking failed: {error_msg}")
                return (
                    False,
                    [],
                    None,
                    execution_time_ms,
                    VINA_GPU_ENABLED,
                    0,
                    error_msg,
                )

            best_affinity: Optional[float] = None
            best_pose: Optional[DockingPose] = None
            best_conf_idx = 0

            for i, conf_file in enumerate(conformer_files):
                out_path = output_dir / (conf_file.stem + "_out.pdbqt")
                if not out_path.exists():
                    continue

                poses = parse_vina_output(out_path.read_text())
                if poses and (
                    best_affinity is None or poses[0].affinity < best_affinity
                ):
                    best_affinity = poses[0].affinity
                    best_pose = poses[0]
                    best_conf_idx = i

            if best_pose is None:
                return (
                    False,
                    [],
                    None,
                    execution_time_ms,
                    VINA_GPU_ENABLED,
                    0,
                    "No poses found across conformer ensemble",
                )

            best_pose.rank = 1

            logger.info(
                f"Rigid ensemble docking complete: best affinity {best_affinity:.2f} kcal/mol "
                f"from conformer {best_conf_idx + 1}/{len(ligand_pdbqts)}, "
                f"time={execution_time_ms}ms"
            )

            return (
                True,
                [best_pose],
                best_affinity,
                execution_time_ms,
                VINA_GPU_ENABLED,
                best_conf_idx,
                None,
            )

    except subprocess.TimeoutExpired:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return (
            False,
            [],
            None,
            execution_time_ms,
            VINA_GPU_ENABLED,
            0,
            "Ensemble docking timed out (>10 minutes)",
        )
    except FileNotFoundError:
        execution_time_ms = int((time.time() - start_time) * 1000)
        binary = UNIDOCK_BINARY if VINA_GPU_ENABLED else VINA_BINARY
        return (
            False,
            [],
            None,
            execution_time_ms,
            False,
            0,
            f"Docking binary not found: {binary}",
        )
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.exception("Rigid ensemble docking failed")
        return False, [], None, execution_time_ms, VINA_GPU_ENABLED, 0, str(e)
