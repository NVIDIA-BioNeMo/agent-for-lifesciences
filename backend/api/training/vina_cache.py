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

"""Training-only vina_dock result cache.

During RL training rollouts, the agent repeatedly docks the same winner/loser
ligand pairs against the same receptor. This module turns those calls into
O(ms) disk reads instead of O(s) Vina service invocations by reusing the
pre-computed docked poses and affinity scores shipped with the
aevom-data-gen dataset.

Cache layout (per complex):

    <dataset_base>/complex_NNNN/
        metadata.json              # winner + loser vina_score values
        receptor.pdb               # protein (retrieved from session storage)
        docked.pdbqt               # winner's pre-docked pose
        winner.sdf                 # winner ligand (preload_refs source)
        losers/
            loser_NN_<sab>.sdf     # loser ligand (preload_refs source)
            loser_NN_docked.pdbqt  # loser's pre-docked pose

The cache is built once per training session in ``build_vina_cache`` by
inspecting the ``preload_refs`` dict. At call time,
``build_cached_vina_result`` combines the cached pose with the receptor into
a fresh complex PDB, stores it in session storage, and returns a
``ToolResult`` that is shape-identical to a real ``VinaDockTool`` response.

On any failure (missing file, unreadable PDBQT, retrieval error) the caller
is expected to fall back to the real Vina service.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from api.storage import format_storage_message
from api.tools.base import ToolResult
from api.tools.vina import VinaDockTool

logger = logging.getLogger(__name__)


_LOSER_FILENAME_RE = re.compile(r"^loser_(\d+)_")


@dataclass
class VinaCacheEntry:
    """A single cached docking result.

    Attributes:
        cached_affinity: Binding affinity in kcal/mol (from metadata.json).
        docked_pdbqt_path: Absolute path to the pre-computed docked PDBQT file.
        source_sdf_path: Absolute path to the ligand SDF that this entry covers
            (for debugging / telemetry).
    """

    cached_affinity: float
    docked_pdbqt_path: Path
    source_sdf_path: Path


def _resolve_abs(path_str: str, dataset_base_path: Optional[Path]) -> Path:
    """Resolve a preload file path to an absolute path on disk.

    Mirrors the logic in ``AevomResourcesServer._resolve_preload_path``: if
    the path is absolute and exists use it as-is; otherwise join against the
    dataset base path.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    if dataset_base_path is not None:
        return dataset_base_path / p
    return p


def build_vina_cache(
    preload_refs: dict[str, str],
    dataset_base_path: Optional[str | Path] = None,
) -> dict[str, VinaCacheEntry]:
    """Build a map of ligand_sdf_ref -> cached docking result.

    Scans ``preload_refs`` for ``.sdf`` entries that live inside a
    ``complex_NNNN`` directory alongside a ``metadata.json`` and the
    expected ``docked.pdbqt`` file. Entries that don't match the expected
    layout, lack an affinity score, or have a missing docked pose file
    are silently skipped -- the caller will fall back to real Vina for
    those refs.

    Args:
        preload_refs: The ``preload_refs`` dict from the training task
            (ref_name -> file_path). Paths may be absolute or relative
            to ``dataset_base_path``.
        dataset_base_path: Base path used to resolve relative entries.
            When ``None``, only absolute paths are resolvable.

    Returns:
        Dict keyed by ligand sdf ref name (e.g. ``"sdf_45876ea8"``) with
        ``VinaCacheEntry`` values. Refs whose docked pose is missing
        are omitted.
    """
    base = Path(dataset_base_path) if dataset_base_path else None
    cache: dict[str, VinaCacheEntry] = {}

    for ref_name, rel_path in preload_refs.items():
        if not rel_path.lower().endswith(".sdf"):
            continue

        sdf_abs = _resolve_abs(rel_path, base)
        if not sdf_abs.exists():
            logger.debug(
                "[vina_cache] Skip %s: SDF not found at %s", ref_name, sdf_abs
            )
            continue

        # Locate sibling metadata.json by walking up until we find it
        # (handles both winner.sdf and losers/loser_*.sdf layouts).
        complex_dir: Optional[Path] = None
        for candidate in (sdf_abs.parent, sdf_abs.parent.parent):
            if (candidate / "metadata.json").exists():
                complex_dir = candidate
                break

        if complex_dir is None:
            logger.debug(
                "[vina_cache] Skip %s: no metadata.json near %s", ref_name, sdf_abs
            )
            continue

        try:
            metadata = json.loads((complex_dir / "metadata.json").read_text())
        except Exception as e:
            logger.warning(
                "[vina_cache] Failed to parse metadata.json in %s: %s",
                complex_dir,
                e,
            )
            continue

        affinity: Optional[float] = None
        docked_path: Optional[Path] = None

        filename = sdf_abs.name
        if filename == "winner.sdf":
            winner_metrics = metadata.get("winner_metrics", {}) or {}
            raw_score = winner_metrics.get("vina_score")
            if raw_score is not None:
                affinity = float(raw_score)
            docked_path = complex_dir / "docked.pdbqt"
        else:
            # Match by loser_filename (relative to complex_dir)
            match = _LOSER_FILENAME_RE.match(filename)
            if match is None:
                logger.debug(
                    "[vina_cache] Skip %s: filename %s doesn't match loser pattern",
                    ref_name,
                    filename,
                )
                continue
            loser_idx = match.group(1)

            try:
                rel_to_complex = sdf_abs.relative_to(complex_dir).as_posix()
            except ValueError:
                rel_to_complex = filename

            for pair in metadata.get("pairs", []):
                if pair.get("loser_filename") == rel_to_complex:
                    raw_score = (pair.get("metrics") or {}).get("vina_score")
                    if raw_score is not None:
                        affinity = float(raw_score)
                    break

            docked_path = complex_dir / "losers" / f"loser_{loser_idx}_docked.pdbqt"

        if affinity is None:
            logger.debug(
                "[vina_cache] Skip %s: no affinity in metadata for %s",
                ref_name,
                filename,
            )
            continue

        if docked_path is None or not docked_path.exists():
            logger.debug(
                "[vina_cache] Skip %s: docked pose missing at %s",
                ref_name,
                docked_path,
            )
            continue

        cache[ref_name] = VinaCacheEntry(
            cached_affinity=affinity,
            docked_pdbqt_path=docked_path,
            source_sdf_path=sdf_abs,
        )

    logger.info(
        "[vina_cache] Built cache: %d/%d sdf refs eligible",
        len(cache),
        sum(1 for p in preload_refs.values() if p.lower().endswith(".sdf")),
    )
    return cache


# Shared helper instance -- ``_parse_pdbqt_ligand`` and
# ``_combine_receptor_ligand_pdb`` are effectively static Biopython utilities
# on VinaDockTool (they only self-reference for ``_vina_type_to_element``,
# which is also state-free). Reusing them guarantees byte-level parity with
# production output.
_VINA_HELPERS = VinaDockTool()


async def build_cached_vina_result(
    entry: VinaCacheEntry,
    arguments: dict[str, Any],
    storage: Any,
) -> ToolResult:
    """Build a ``ToolResult`` that matches a real ``vina_dock`` response.

    Reads the cached PDBQT pose, combines it with the receptor retrieved
    from session storage, stores the resulting complex, and emits the same
    message + metadata shape that ``VinaDockTool.execute`` would produce on
    a real docking run.

    Args:
        entry: Cache entry for the ligand being docked.
        arguments: Raw tool arguments from the agent (used for
            ``receptor_pdb_ref``, ``center_*``, ``size_*``, ``description``).
        storage: Session-scoped storage (``SessionScopedStorage``) used to
            retrieve the receptor and store the output complex.

    Returns:
        ``ToolResult`` of type ``DATA_REFERENCE`` whose ``.data`` is the
        stored complex ref and whose ``.metadata["message"]`` is formatted
        identically to real Vina output.

    Raises:
        Exception: If the receptor can't be retrieved or the PDBQT can't be
            parsed. Callers should catch and fall back to real Vina.
    """
    t0 = time.time()

    receptor_pdb_ref = arguments.get("receptor_pdb_ref", "")
    ligand_sdf_ref = arguments.get("ligand_sdf_ref", "")
    center_x = arguments.get("center_x")
    center_y = arguments.get("center_y")
    center_z = arguments.get("center_z")
    size_x = arguments.get("size_x", 20.0)
    size_y = arguments.get("size_y", 20.0)
    size_z = arguments.get("size_z", 20.0)
    exhaustiveness = arguments.get("exhaustiveness", 16)
    description = arguments.get("description")

    receptor_bytes = await storage.retrieve(receptor_pdb_ref)
    receptor_pdb_content = receptor_bytes.decode("utf-8")

    pdbqt_content = entry.docked_pdbqt_path.read_text()

    ligand_residue = _VINA_HELPERS._parse_pdbqt_ligand(pdbqt_content)
    complex_pdb = _VINA_HELPERS._combine_receptor_ligand_pdb(
        receptor_pdb=receptor_pdb_content,
        ligand_residue=ligand_residue,
        ligand_chain_id="Z",
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    pose_metadata: dict[str, Any] = {
        "source": "vina_dock",
        "affinity_kcal_mol": entry.cached_affinity,
        "num_poses": 1,
        "gpu_accelerated": False,
        "execution_time_ms": elapsed_ms,
        "receptor_ref": receptor_pdb_ref,
        "ligand_ref": ligand_sdf_ref,
        "ligand_chain": "Z",
        "center": [center_x, center_y, center_z],
        "size": [size_x, size_y, size_z],
        "exhaustiveness": exhaustiveness,
        "is_complex": True,
        "docking_mode": "cached",
        "conformers_generated": 0,
        "rotatable_bonds": None,
        "cached": True,
        "cached_pose_path": str(entry.docked_pdbqt_path),
    }

    pose_ref = await storage.store(
        data_type="pdb",
        content=complex_pdb.encode("utf-8"),
        metadata=pose_metadata,
        description=description,
    )

    message = format_storage_message(
        data_type="pdb",
        reference=pose_ref,
        source_description="receptor-ligand docked complex",
        accepting_tools=["visualize_pdb"],
    )

    lig_resname = ligand_residue.resname
    best_affinity = entry.cached_affinity
    message += (
        f"\n\nDocking Results:\n"
        f"- Best binding affinity: {best_affinity:.2f} kcal/mol\n"
        f"- Number of poses found: 1\n"
        f"- Execution time: {elapsed_ms}ms\n"
        f"- Complex structure: Receptor + Ligand (chain Z)\n"
        f"- Ligand residue name in complex: {lig_resname} (chain Z, residue 1)\n"
        f"- Docking mode: cached\n"
    )

    if best_affinity > -6:
        message += "\nInterpretation: Weak binding (> -6 kcal/mol)"
    elif best_affinity > -9:
        message += (
            "\nInterpretation: Moderate binding "
            "(-6 to -9 kcal/mol, typical drug range)"
        )
    else:
        message += "\nInterpretation: Strong binding (< -9 kcal/mol)"

    return ToolResult.data_reference(
        reference=pose_ref,
        data_type="pdb",
        message=message,
        metadata=pose_metadata,
    )
