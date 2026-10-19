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

"""FastAPI service for Autodock Vina molecular docking.

This service provides endpoints for:
- Receptor preparation (PDB to PDBQT)
- Ligand preparation (SDF/SMILES to PDBQT)
- Molecular docking with configurable search box
"""

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from . import conformer_client
from .docking import get_vina_version, run_docking, run_docking_python_api
from .models import (
    DockAutoRequest,
    DockRequest,
    DockResponse,
    DockingPose,
    HealthResponse,
    PrepareLigandRequest,
    PrepareLigandResponse,
    PrepareReceptorRequest,
    PrepareReceptorResponse,
    ProbeGridRequest,
    ProbeGridResponse,
    VinaInfoResponse,
)
from .pipeline import run_auto_dock
from .preparation import prepare_ligand, prepare_receptor
from .probe_grid import run_probe_grid

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


class HealthCheckFilter(logging.Filter):
    """Filter to suppress health check access logs at INFO level."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "/health" in message and "GET" in message:
            return LOG_LEVEL == "DEBUG"
        return True


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Apply health check filter to uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckFilter())

# Track initialization
_health_initialized = False

# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await asyncio.to_thread(conformer_client.shutdown)


app = FastAPI(
    title="Autodock Vina Service",
    description="Molecular docking service using Autodock Vina with Meeko preparation",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global _health_initialized

    if not _health_initialized:
        logger.info("Vina service is online and healthy")
        _health_initialized = True
    else:
        logger.debug("Health check heartbeat")

    return HealthResponse(status="healthy", service="vina")


@app.get("/info", response_model=VinaInfoResponse)
async def info():
    """Get Vina service information including version and GPU status."""
    version, gpu_available = get_vina_version()
    gpu_enabled = os.environ.get("VINA_GPU", "false").lower() == "true"

    if gpu_enabled and gpu_available:
        gpu_backend = "unidock"
    elif not gpu_enabled:
        gpu_backend = "cpu_vina"
    else:
        gpu_backend = "none"

    return VinaInfoResponse(
        vina_version=version,
        gpu_available=gpu_available,
        gpu_enabled=gpu_enabled,
        gpu_backend=gpu_backend,
    )


# =============================================================================
# Preparation Endpoints
# =============================================================================


@app.post("/prepare-receptor", response_model=PrepareReceptorResponse)
async def prepare_receptor_endpoint(request: PrepareReceptorRequest):
    """Convert PDB to PDBQT format for docking.

    Prepares a protein receptor structure for Autodock Vina by:
    - Optionally removing water molecules
    - Optionally removing heteroatoms (ligands, cofactors)
    - Assigning atom types and partial charges
    - Converting to PDBQT format

    Args:
        request: PrepareReceptorRequest with PDB content and options.

    Returns:
        PrepareReceptorResponse with PDBQT content or error.
    """
    logger.info(
        f"Preparing receptor: remove_water={request.remove_water}, "
        f"remove_heteroatoms={request.remove_heteroatoms}"
    )

    if not request.pdb_content or not request.pdb_content.strip():
        return PrepareReceptorResponse(success=False, error="PDB content is empty")

    pdbqt, num_atoms, error = prepare_receptor(
        pdb_content=request.pdb_content,
        remove_water=request.remove_water,
        remove_heteroatoms=request.remove_heteroatoms,
        ph=request.ph,
    )

    if error:
        logger.error(f"Receptor preparation failed: {error}")
        return PrepareReceptorResponse(success=False, error=error)

    logger.info(f"Receptor prepared successfully: {num_atoms} atoms")

    return PrepareReceptorResponse(
        success=True,
        pdbqt=pdbqt,
        num_atoms=num_atoms,
    )


@app.post("/prepare-ligand", response_model=PrepareLigandResponse)
async def prepare_ligand_endpoint(request: PrepareLigandRequest):
    """Convert SDF/SMILES to PDBQT format for docking.

    Prepares a small molecule ligand for Autodock Vina by:
    - Parsing SDF or generating 3D from SMILES
    - Adding hydrogens
    - Assigning Gasteiger partial charges
    - Identifying rotatable bonds for flexible docking
    - Converting to PDBQT format with torsion tree

    Args:
        request: PrepareLigandRequest with SDF content or SMILES.

    Returns:
        PrepareLigandResponse with PDBQT content or error.
    """
    logger.info(
        f"Preparing ligand: sdf={'yes' if request.sdf_content else 'no'}, "
        f"smiles={'yes' if request.smiles else 'no'}"
    )

    if not request.sdf_content and not request.smiles:
        return PrepareLigandResponse(
            success=False, error="Either sdf_content or smiles must be provided"
        )

    pdbqt, num_atoms, num_rotatable, error = prepare_ligand(
        sdf_content=request.sdf_content,
        smiles=request.smiles,
        add_hydrogens=request.add_hydrogens,
        ph=request.ph,
    )

    if error:
        logger.error(f"Ligand preparation failed: {error}")
        return PrepareLigandResponse(success=False, error=error)

    logger.info(
        f"Ligand prepared successfully: {num_atoms} atoms, "
        f"{num_rotatable} rotatable bonds"
    )

    return PrepareLigandResponse(
        success=True,
        pdbqt=pdbqt,
        num_atoms=num_atoms,
        num_rotatable_bonds=num_rotatable,
    )


# =============================================================================
# Docking Endpoint
# =============================================================================


@app.post("/dock", response_model=DockResponse)
async def dock_endpoint(request: DockRequest):
    """Execute molecular docking with Autodock Vina.

    Docks a ligand to a receptor within a specified search box and returns
    ranked binding poses with affinity scores.

    The search box is defined by center coordinates and dimensions in Angstroms.
    Larger boxes increase search space but may reduce accuracy.

    ``score_mode`` decides whether the pose in the ligand PDBQT is a starting
    point or an answer: ``search`` discards it, ``local`` minimises from it, and
    ``score`` leaves it exactly as it is. Both pose-preserving modes return one
    pose and no ranking, since there is nothing to rank.

    Args:
        request: DockRequest with receptor PDBQT, ligand PDBQT, and box params.

    Returns:
        DockResponse with poses, affinities, and execution metadata.
    """
    logger.info(
        f"Starting docking: center=({request.center_x}, {request.center_y}, {request.center_z}), "
        f"size=({request.size_x}, {request.size_y}, {request.size_z}), "
        f"exhaustiveness={request.exhaustiveness}, score_mode={request.score_mode}"
    )

    # Validate inputs
    if not request.receptor_pdbqt or not request.receptor_pdbqt.strip():
        return DockResponse(success=False, error="Receptor PDBQT is empty")

    if not request.ligand_pdbqt or not request.ligand_pdbqt.strip():
        return DockResponse(success=False, error="Ligand PDBQT is empty")

    # Try command-line docking first
    # Run in thread pool to avoid blocking the async event loop (health checks)
    success, poses, best_affinity, exec_time, gpu_used, error = await asyncio.to_thread(
        run_docking,
        receptor_pdbqt=request.receptor_pdbqt,
        ligand_pdbqt=request.ligand_pdbqt,
        center=(request.center_x, request.center_y, request.center_z),
        size=(request.size_x, request.size_y, request.size_z),
        exhaustiveness=request.exhaustiveness,
        num_modes=request.num_modes,
        energy_range=request.energy_range,
        score_mode=request.score_mode,
    )

    # If CLI fails, try Python API as fallback. Search mode only: the Python
    # bindings expose optimize() and score() rather than the CLI's flags, and a
    # fallback that silently re-searched would return a pose the caller did not
    # ask for.
    if not success and request.score_mode == "search" and "not found" in (error or "").lower():
        logger.info("CLI docking failed, trying Python API...")
        (
            success,
            poses,
            best_affinity,
            exec_time,
            gpu_used,
            error,
        ) = await asyncio.to_thread(
            run_docking_python_api,
            receptor_pdbqt=request.receptor_pdbqt,
            ligand_pdbqt=request.ligand_pdbqt,
            center=(request.center_x, request.center_y, request.center_z),
            size=(request.size_x, request.size_y, request.size_z),
            exhaustiveness=request.exhaustiveness,
            num_modes=request.num_modes,
            energy_range=request.energy_range,
        )

    if not success:
        logger.error(f"Docking failed: {error}")
        return DockResponse(
            success=False,
            gpu_accelerated=gpu_used,
            execution_time_ms=exec_time,
            error=error,
        )

    logger.info(
        f"Docking complete: {len(poses)} poses, "
        f"best={best_affinity:.2f} kcal/mol, "
        f"time={exec_time}ms, gpu={gpu_used}"
    )

    return DockResponse(
        success=True,
        gpu_accelerated=gpu_used,
        execution_time_ms=exec_time,
        poses=poses,
        best_affinity=best_affinity,
        docking_mode="flexible" if request.score_mode == "search" else request.score_mode,
    )


# =============================================================================
# Auto-Dock Endpoint (combined preparation + routing + docking)
# =============================================================================


@app.post("/dock-auto", response_model=DockResponse)
async def dock_auto_endpoint(request: DockAutoRequest):
    """Automatically prepare and dock a ligand with optimal strategy.

    Accepts PDB **or** pre-prepared PDBQT for the receptor, plus raw SDF
    for the ligand.  When the receptor is already in PDBQT format the
    preparation step is skipped entirely, avoiding double-conversion and
    false size-limit rejections.

    Pipeline steps:
    1. Detect receptor format and prepare if needed (PDB -> PDBQT)
    2. Count rotatable bonds in the ligand
    3. Select docking strategy:
       - Flexible (Meeko torsion tree) if rot_bonds <= threshold
       - Rigid ensemble (conformer generation + batch docking) if above
    4. Use GPU (Uni-Dock, plus nvMolKit conformers from the out-of-process
       worker for the rigid ensemble) when available, CPU (RDKit + Vina)
       otherwise

    A ``score_mode`` other than ``search`` overrides steps 3 and 4: the pose in
    the SDF is kept, so conformers are not regenerated and the CPU binary is
    used.
    """
    logger.info(
        f"Auto-dock: center=({request.center_x}, {request.center_y}, {request.center_z}), "
        f"conformers={request.num_conformers}, rot_threshold={request.rot_bond_threshold}, "
        f"score_mode={request.score_mode}"
    )

    if not request.receptor_pdb or not request.receptor_pdb.strip():
        return DockResponse(success=False, error="Receptor PDB content is empty")

    if not request.ligand_sdf or not request.ligand_sdf.strip():
        return DockResponse(success=False, error="Ligand SDF content is empty")

    result = await asyncio.to_thread(
        run_auto_dock,
        receptor_pdb=request.receptor_pdb,
        ligand_sdf=request.ligand_sdf,
        center=(request.center_x, request.center_y, request.center_z),
        size=(request.size_x, request.size_y, request.size_z),
        exhaustiveness=request.exhaustiveness,
        num_modes=request.num_modes,
        energy_range=request.energy_range,
        num_conformers=request.num_conformers,
        rot_bond_threshold=request.rot_bond_threshold,
        score_mode=request.score_mode,
    )

    poses = [DockingPose(**p) for p in result.get("poses", [])]

    return DockResponse(
        success=result["success"],
        gpu_accelerated=result.get("gpu_accelerated", False),
        execution_time_ms=result.get("execution_time_ms", 0),
        poses=poses,
        best_affinity=result.get("best_affinity"),
        error=result.get("error"),
        docking_mode=result.get("docking_mode", "unknown"),
        conformers_generated=result.get("conformers_generated", 0),
        best_conformer_idx=result.get("best_conformer_idx", 0),
        rotatable_bonds=result.get("rotatable_bonds"),
    )


# =============================================================================
# Probe Grid Endpoint
# =============================================================================


def _probe_grid_payload(request: ProbeGridRequest) -> dict:
    """Prepare the receptor, compute the grids, and pack them for transport.

    Runs on a worker thread: the receptor preparation shells out to OpenBabel
    and the grid itself is seconds of numpy, neither of which may block the
    event loop.
    """
    pdbqt, _num_atoms, error = prepare_receptor(
        pdb_content=request.pdb_content,
        remove_water=request.remove_water,
        remove_heteroatoms=request.remove_heteroatoms,
        ph=request.ph,
    )
    if error:
        return {"error": f"Receptor preparation failed: {error}"}

    try:
        result = run_probe_grid(
            pdbqt_text=pdbqt,
            center=tuple(request.center),
            size=tuple(request.size),
            spacing=request.spacing,
            probe_names=tuple(request.probes),
            use_templates=request.atom_typing == "template",
            max_minima=request.max_minima,
            min_separation=request.min_separation,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    # float32 halves the payload at a precision far finer than the ~1e-3
    # agreement the grid itself has with Vina's reference maps. The archive key
    # must be "data": that is what the sandbox's array loader reads.
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=result.grids.astype(np.float32))

    return {
        "npz_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "probes": list(result.probes),
        "origin": [float(v) for v in result.origin],
        "spacing": result.spacing,
        "shape": list(result.shape),
        "minima": [m.as_dict() for m in result.minima],
        "num_receptor_atoms": result.num_receptor_atoms,
        "num_template_typed": result.num_template_typed,
    }


@app.post("/probe-grid", response_model=ProbeGridResponse)
async def probe_grid_endpoint(request: ProbeGridRequest):
    """Map where each pharmacophore type would most want to sit in a pocket.

    Places a probe atom of a given type at every point of a box and scores it
    against the receptor with Vina's inter-molecular function. A minimum in the
    acceptor-probe grid is a position where a ligand acceptor would be
    well-satisfied -- that is, next to a receptor donor.

    Unlike docking, nothing is searched and no ligand is required: the result
    depends only on the receptor, which is what makes it usable before any
    molecule exists.
    """
    logger.info(
        f"Probe grid: center={request.center}, size={request.size}, "
        f"spacing={request.spacing}, probes={request.probes}, "
        f"typing={request.atom_typing}"
    )

    if not request.pdb_content or not request.pdb_content.strip():
        return ProbeGridResponse(success=False, error="PDB content is empty")

    payload = await asyncio.to_thread(_probe_grid_payload, request)

    if "error" in payload:
        logger.error(f"Probe grid failed: {payload['error']}")
        return ProbeGridResponse(success=False, error=payload["error"])

    logger.info(
        f"Probe grid done: shape={payload['shape']}, "
        f"{len(payload['minima'])} minima, best={payload['minima'][0]['energy'] if payload['minima'] else None}"
    )

    return ProbeGridResponse(success=True, **payload)
