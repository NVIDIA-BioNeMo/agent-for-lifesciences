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

"""FastAPI service for RDKit molecular operations.

This service provides endpoints for:
- Converting SMILES to molecule images
- Validating SMILES strings
- Getting molecule information
"""

import base64
import logging
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .constrained_pose import (
    DEFAULT_RMSD_MAX,
    DEFAULT_SEEDS,
    DEFAULT_STRAIN_MAX,
    POSE_CLASH_DISTANCE,
    verify as verify_constrained_pose,
)
from .fragment_link import (
    DEFAULT_RB_MAX,
    analyze as analyze_link_geometry,
    parse_pocket_coords,
)
from .gates import (
    DEFAULT_THRESHOLDS,
    MAX_MOLECULES,
    THRESHOLD_NAMES,
    evaluate as evaluate_gates,
)
from .linker_library import LINKER_LIBRARY, measure_all as measure_linkers
from .molecule import (
    get_molecule_info,
    smiles_to_image_bytes,
    smiles_to_sdf,
    validate_smiles,
)
from .safe_notation import (
    DEFAULT_SLICER,
    SUPPORTED_SLICERS,
    apply_mask,
    safe_to_smiles,
    smiles_to_safe,
)

# Get log level from environment, default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


class HealthCheckFilter(logging.Filter):
    """Filter to suppress health check access logs at INFO level.
    
    Health check heartbeats are only logged at DEBUG level to reduce noise.
    The initial "service online" message is always logged at INFO.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Check if this is a uvicorn access log for /health endpoint
        message = record.getMessage()
        if "/health" in message and "GET" in message:
            # Only show health check access logs at DEBUG level
            return LOG_LEVEL == "DEBUG"
        return True


# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Apply health check filter to uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckFilter())

# Track if initial health check has been logged
_health_initialized = False


app = FastAPI(
    title="RDKit Service",
    description="Molecular operations service using RDKit",
    version="1.0.0",
)


# SMDD-Bench Fragment Assembly gives one or two fragments. The bound is higher so
# a caller exploring a multi-fragment merge is not blocked, and finite so a
# thousand-fragment request cannot turn into a quadratic pair enumeration.
MAX_FRAGMENTS = 8

# Candidates per constrained-pose request. Each one costs a restrained embedding
# and two force-field minimisations -- tens of milliseconds to a second -- so this
# is a latency bound, and a caller with hundreds of candidates is expected to
# chunk. The sandbox helper does.
MAX_CANDIDATES = 64


class SmilesToImageRequest(BaseModel):
    """Request model for SMILES to image conversion."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    width: int = Field(default=400, ge=100, le=2000, description="Image width in pixels")
    height: int = Field(default=400, ge=100, le=2000, description="Image height in pixels")


class SmilesToImageResponse(BaseModel):
    """Response model for SMILES to image conversion.

    The PNG travels inline as base64 rather than as a URL to a file this service
    wrote. The caller is expected to store the decoded bytes as a `png` data
    artifact, which is what makes the depiction readable from any api replica.
    """
    success: bool
    image_base64: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None


class ValidateSmilesRequest(BaseModel):
    """Request model for SMILES validation."""
    smiles: str = Field(..., description="SMILES string to validate")


class ValidateSmilesResponse(BaseModel):
    """Response model for SMILES validation."""
    valid: bool
    error: Optional[str] = None


class MoleculeInfoRequest(BaseModel):
    """Request model for molecule info."""
    smiles: str = Field(..., description="SMILES string of the molecule")


class MoleculeInfoResponse(BaseModel):
    """Response model for molecule info."""
    success: bool
    info: Optional[dict] = None
    error: Optional[str] = None


class SmilesToSdfRequest(BaseModel):
    """Request model for SMILES to SDF conversion."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    optimize: bool = Field(default=True, description="Optimize 3D geometry with force field")


class SmilesToSdfResponse(BaseModel):
    """Response model for SMILES to SDF conversion."""
    success: bool
    sdf: Optional[str] = None
    error: Optional[str] = None


class SmilesToSafeRequest(BaseModel):
    """Request model for SMILES to SAFE conversion."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    slicer: str = Field(
        default=DEFAULT_SLICER,
        description=f"Fragmentation algorithm: {', '.join(SUPPORTED_SLICERS)}",
    )
    mask_min: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "With mask_max, replace the terminal fragment with a GenMol "
            "generation mask [*{min-max}]"
        ),
    )
    mask_max: Optional[int] = Field(
        default=None, ge=1, description="Upper bound of the generation mask"
    )


class SmilesToSafeResponse(BaseModel):
    """Response model for SMILES to SAFE conversion."""
    success: bool
    safe: Optional[str] = None
    num_fragments: Optional[int] = None
    fragmented: Optional[bool] = None
    masked: bool = False
    slicer_used: Optional[str] = None
    replaced_fragment: Optional[str] = None
    error: Optional[str] = None


class SafeToSmilesRequest(BaseModel):
    """Request model for SAFE to SMILES conversion."""
    safe: str = Field(..., description="SAFE fragment notation string")


class SafeToSmilesResponse(BaseModel):
    """Response model for SAFE to SMILES conversion."""
    success: bool
    smiles: Optional[str] = None
    error: Optional[str] = None


class MoleculeGatesRequest(BaseModel):
    """Request model for drug-likeness gate evaluation."""
    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_MOLECULES,
        description="SMILES strings to score",
    )
    thresholds: Optional[dict] = Field(
        default=None,
        description=(
            "Overrides for the default threshold panel. A null value disables "
            f"that gate. Recognized names: {', '.join(THRESHOLD_NAMES)}"
        ),
    )
    reference_smiles: str = Field(
        default="",
        description="Molecule that the similarity and scaffold gates measure against",
    )


class MoleculeGatesResponse(BaseModel):
    """Response model for drug-likeness gate evaluation."""
    success: bool
    results: Optional[list[dict]] = None
    invalid_smiles: Optional[list[str]] = None
    thresholds: Optional[dict] = None
    unknown_thresholds: Optional[list[str]] = None
    reference_smiles: Optional[str] = None
    reference_scaffold: Optional[str] = None
    error: Optional[str] = None


class LinkGeometryRequest(BaseModel):
    """Request model for fragment exit-vector geometry."""

    fragment_sdfs: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_FRAGMENTS,
        description=(
            "One SDF block per fragment, each carrying the pose the fragment has "
            "to keep. Two fragments is a linking task; one is a growth task"
        ),
    )
    pocket_pdb: str = Field(
        default="",
        description=(
            "Receptor PDB. Without it, free depth and tube clearance come back "
            "null and the geometry is topological only"
        ),
    )
    target_points: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            "Hotspot coordinates to aim growth vectors at, typically probe_grid "
            "minima. On a two-fragment task the partner fragment is used instead"
        ),
    )
    max_sites_per_fragment: int = Field(
        default=12, ge=1, le=64, description="Substitutable sites reported per fragment"
    )
    max_pairs: int = Field(
        default=10, ge=1, le=100, description="Candidate site pairs reported, best first"
    )
    rb_max: int = Field(
        default=DEFAULT_RB_MAX,
        ge=1,
        le=50,
        description="Rotatable-bond ceiling the assembled molecule must respect",
    )
    assemble_candidates: bool = Field(
        default=False,
        description=(
            "Also return complete candidate SMILES built from the measured linker "
            "library, which needs no generative model"
        ),
    )
    max_candidates_per_pair: int = Field(
        default=6, ge=1, le=32, description="Library entries assembled per pair"
    )
    rigid_linkers_only: bool = Field(
        default=False,
        description="Restrict library matches to the rigid entries",
    )


class LinkGeometryResponse(BaseModel):
    """Response model for fragment exit-vector geometry."""

    success: bool
    fragments: Optional[list[dict]] = None
    pairs: Optional[list[dict]] = None
    candidates: Optional[list[dict]] = None
    growth: Optional[dict] = None
    fragment_separation: Optional[list[dict]] = None
    rotatable_bonds_used: Optional[int] = None
    rb_headroom: Optional[int] = None
    rb_max: Optional[int] = None
    pocket_atoms: Optional[int] = None
    warnings: Optional[list[str]] = None
    error: Optional[str] = None


class ConstrainedPoseRequest(BaseModel):
    """Request model for constrained-pose verification."""

    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_CANDIDATES,
        description="Candidate molecules to verify against the fragment poses",
    )
    fragment_sdfs: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_FRAGMENTS,
        description="One SDF block per fragment, carrying the pose that must be held",
    )
    pocket_pdb: str = Field(
        default="",
        description=(
            "Receptor PDB. Without it no clash check runs, and a candidate "
            "threading the protein wall will verify"
        ),
    )
    num_seeds: int = Field(
        default=DEFAULT_SEEDS,
        ge=1,
        le=16,
        description="Independent restrained-embedding attempts per candidate",
    )
    strain_max: Optional[float] = Field(
        default=DEFAULT_STRAIN_MAX,
        ge=0.0,
        description=(
            "Strain ceiling in kcal/mol relative to the free minimum, or null to "
            "report strain without judging it"
        ),
    )
    rmsd_max: float = Field(
        default=DEFAULT_RMSD_MAX,
        gt=0.0,
        description="Per-fragment deviation ceiling in Angstroms",
    )
    clash_distance: float = Field(
        default=POSE_CLASH_DISTANCE,
        gt=0.0,
        description="Protein contact distance below which an atom counts as clashing",
    )
    include_poses: bool = Field(
        default=True,
        description="Return the minimised pose SDF per candidate",
    )


class ConstrainedPoseResponse(BaseModel):
    """Response model for constrained-pose verification."""

    success: bool
    results: Optional[list[dict]] = None
    num_verified: Optional[int] = None
    pocket_atoms: Optional[int] = None
    thresholds: Optional[dict] = None
    error: Optional[str] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    global _health_initialized
    
    if not _health_initialized:
        logger.info("RDKit service is online and healthy")
        _health_initialized = True
    else:
        logger.debug("Health check heartbeat")
    
    return {"status": "healthy", "service": "rdkit"}


@app.post("/smiles-to-image", response_model=SmilesToImageResponse)
async def convert_smiles_to_image(request: SmilesToImageRequest):
    """Convert a SMILES string to a PNG image.
    
    The image is returned inline as base64; nothing is written to disk.
    
    Args:
        request: Contains SMILES string and optional image dimensions.
        
    Returns:
        Response with base64-encoded PNG bytes or an error message.
    """
    logger.info(f"Converting SMILES to image: {request.smiles}")
    
    image_bytes, error = smiles_to_image_bytes(
        smiles=request.smiles,
        size=(request.width, request.height),
    )
    
    if error:
        logger.error(f"SMILES to image failed: {error}")
        return SmilesToImageResponse(
            success=False,
            error=error,
        )
    
    logger.info(f"Image generated successfully: {len(image_bytes)} bytes")
    
    return SmilesToImageResponse(
        success=True,
        image_base64=base64.b64encode(image_bytes).decode("ascii"),
        content_type="image/png",
    )


@app.post("/validate", response_model=ValidateSmilesResponse)
async def validate_smiles_endpoint(request: ValidateSmilesRequest):
    """Validate a SMILES string.
    
    Args:
        request: Contains SMILES string to validate.
        
    Returns:
        Response indicating if SMILES is valid.
    """
    is_valid, error = validate_smiles(request.smiles)
    return ValidateSmilesResponse(valid=is_valid, error=error)


@app.post("/molecule-info", response_model=MoleculeInfoResponse)
async def get_molecule_info_endpoint(request: MoleculeInfoRequest):
    """Get information about a molecule from its SMILES.
    
    Returns molecular properties like:
    - Molecular weight
    - Formula
    - Number of atoms, bonds, rings
    - LogP, TPSA, etc.
    
    Args:
        request: Contains SMILES string.
        
    Returns:
        Response with molecule information or error.
    """
    info, error = get_molecule_info(request.smiles)
    
    if error:
        return MoleculeInfoResponse(success=False, error=error)
    
    return MoleculeInfoResponse(success=True, info=info)


@app.get("/gate-defaults")
async def gate_defaults():
    """Report the default threshold panel and every overridable name.

    Exposed so a caller can show the bounds it is about to apply without
    duplicating them, and so drift between the service and its callers is
    visible rather than silent.
    """
    return {
        "defaults": DEFAULT_THRESHOLDS,
        "threshold_names": list(THRESHOLD_NAMES),
        "max_molecules": MAX_MOLECULES,
    }


@app.post("/molecule-gates", response_model=MoleculeGatesResponse)
async def molecule_gates(request: MoleculeGatesRequest):
    """Score molecules against the drug-likeness panel.

    Returns the physical property panel, synthetic accessibility, structural
    alerts, similarity to an optional reference, and a per-gate pass/fail
    verdict for each molecule.

    Args:
        request: SMILES list, optional threshold overrides, optional reference.

    Returns:
        Response with one result per parseable molecule, or an error.
    """
    logger.info(
        f"Evaluating gates for {len(request.smiles)} molecule(s) "
        f"(reference={'yes' if request.reference_smiles else 'no'})"
    )

    report, error = evaluate_gates(
        smiles_list=request.smiles,
        thresholds=request.thresholds,
        reference_smiles=request.reference_smiles,
    )

    if error or report is None:
        logger.error(f"Gate evaluation failed: {error}")
        return MoleculeGatesResponse(success=False, error=error)

    passed = sum(1 for result in report.results if result["passed"])
    logger.info(
        f"Gate evaluation complete: {passed}/{len(report.results)} passed, "
        f"{len(report.invalid_smiles)} unparseable"
    )

    return MoleculeGatesResponse(
        success=True,
        results=report.results,
        invalid_smiles=report.invalid_smiles,
        thresholds=report.thresholds,
        unknown_thresholds=report.unknown_thresholds,
        reference_smiles=report.reference_smiles,
        reference_scaffold=report.reference_scaffold,
    )


@app.get("/linker-library")
async def linker_library():
    """The curated linker library, with every span and cost measured on request.

    Exposed so a caller can see the same numbers the pair matcher uses, and so a
    model writing its own candidates in the sandbox can pick a bridge by span
    without guessing. The first call pays for the conformer embeddings; every
    call after it is served from the process cache.
    """
    geometries = measure_linkers()
    return {
        "linkers": [
            {
                **geometries[entry.name].as_dict(),
                "smiles": entry.smiles,
                "family": entry.family,
                "rigid": entry.rigid,
                "note": entry.note,
            }
            for entry in LINKER_LIBRARY
            if entry.name in geometries
        ],
        "unmeasurable": [
            entry.name for entry in LINKER_LIBRARY if entry.name not in geometries
        ],
    }


@app.post("/link-geometry", response_model=LinkGeometryResponse)
async def link_geometry(request: LinkGeometryRequest):
    """Find where posed fragments can be substituted, and what could bridge them.

    Reports substitutable sites with their outward vectors and the free depth in
    front of each, ranked pairs with the anchor separation, exit-vector angle and
    tube clearance between them, the linker atom-count window that spans the gap,
    the rotatable-bond headroom left under the gate, GenMol templates in both
    accepted encodings, and -- on request -- assembled candidate SMILES from the
    measured linker library.

    Args:
        request: Fragment SDFs, optional pocket, and reporting bounds.

    Returns:
        Response with the geometry report, or an error.
    """
    logger.info(
        f"Link geometry for {len(request.fragment_sdfs)} fragment(s) "
        f"(pocket={'yes' if request.pocket_pdb else 'no'}, "
        f"assemble={request.assemble_candidates})"
    )

    report, error = analyze_link_geometry(
        fragment_sdfs=request.fragment_sdfs,
        pocket_pdb=request.pocket_pdb,
        target_points=request.target_points,
        max_sites_per_fragment=request.max_sites_per_fragment,
        max_pairs=request.max_pairs,
        rb_max=request.rb_max,
        assemble_candidates=request.assemble_candidates,
        max_candidates_per_pair=request.max_candidates_per_pair,
        rigid_linkers_only=request.rigid_linkers_only,
    )

    if error or report is None:
        logger.error(f"Link geometry failed: {error}")
        return LinkGeometryResponse(success=False, error=error)

    logger.info(
        f"Link geometry complete: {len(report['pairs'])} pair(s), "
        f"{len(report['candidates'])} assembled candidate(s), "
        f"rb_headroom={report['rb_headroom']}"
    )
    return LinkGeometryResponse(success=True, **report)


@app.post("/constrained-pose", response_model=ConstrainedPoseResponse)
async def constrained_pose(request: ConstrainedPoseRequest):
    """Check whether candidates can hold their fragments where they were posed.

    Per candidate: the disjoint substructure assignment SMDD-Bench's gates 2 and 3
    ask for, a restrained embedding that pins every fragment atom to its input
    coordinate, a force-field minimisation with those atoms fixed, the strain that
    arrangement costs against the molecule's own free minimum, the closest
    approach to the protein, the per-fragment deviation, and the resulting pose.

    A candidate that cannot be embedded at all is the cleanest rejection there is:
    its topology cannot bridge those poses.

    Args:
        request: Candidate SMILES, fragment SDFs, optional pocket, thresholds.

    Returns:
        Response with one result per candidate, or an error if the input is unusable.
    """
    logger.info(
        f"Constrained pose for {len(request.smiles)} candidate(s) against "
        f"{len(request.fragment_sdfs)} fragment(s) "
        f"(pocket={'yes' if request.pocket_pdb else 'no'})"
    )

    results, error = verify_constrained_pose(
        smiles=request.smiles,
        fragment_sdfs=request.fragment_sdfs,
        pocket_pdb=request.pocket_pdb,
        num_seeds=request.num_seeds,
        strain_max=request.strain_max,
        rmsd_max=request.rmsd_max,
        clash_distance=request.clash_distance,
    )

    if error or results is None:
        logger.error(f"Constrained pose failed: {error}")
        return ConstrainedPoseResponse(success=False, error=error)

    verified = sum(1 for result in results if result.verified)
    logger.info(f"Constrained pose complete: {verified}/{len(results)} verified")

    return ConstrainedPoseResponse(
        success=True,
        results=[result.as_dict(include_pose=request.include_poses) for result in results],
        num_verified=verified,
        pocket_atoms=len(parse_pocket_coords(request.pocket_pdb)) if request.pocket_pdb else 0,
        thresholds={
            "strain_max": request.strain_max,
            "rmsd_max": request.rmsd_max,
            "clash_distance": request.clash_distance,
            "num_seeds": request.num_seeds,
        },
    )


@app.post("/smiles-to-sdf", response_model=SmilesToSdfResponse)
async def convert_smiles_to_sdf(request: SmilesToSdfRequest):
    """Convert a SMILES string to SDF format with 3D coordinates.
    
    Generates a 3D conformer suitable for molecular visualization
    in tools like Molstar.
    
    Args:
        request: Contains SMILES string and optimization options.
        
    Returns:
        Response with SDF string or error message.
    """
    logger.info(f"Converting SMILES to SDF: {request.smiles}")
    
    sdf, error = smiles_to_sdf(
        smiles=request.smiles,
        optimize=request.optimize,
    )
    
    if error:
        logger.error(f"SMILES to SDF failed: {error}")
        return SmilesToSdfResponse(success=False, error=error)
    
    logger.info("SDF generated successfully")
    return SmilesToSdfResponse(success=True, sdf=sdf)


@app.post("/smiles-to-safe", response_model=SmilesToSafeResponse)
async def convert_smiles_to_safe(request: SmilesToSafeRequest):
    """Encode a SMILES string as SAFE fragment notation.

    When both mask bounds are given, the terminal fragment is replaced by a
    GenMol generation mask, producing a scaffold-decoration input in one call.

    Args:
        request: Contains SMILES string, slicer choice and optional mask bounds.

    Returns:
        Response with the SAFE string or error message.
    """
    logger.info(f"Encoding SMILES as SAFE (slicer={request.slicer}): {request.smiles}")

    safe_str, error, fragmented, slicer_used = smiles_to_safe(
        smiles=request.smiles,
        slicer=request.slicer,
    )

    if error:
        logger.error(f"SAFE encoding failed: {error}")
        return SmilesToSafeResponse(success=False, error=error)

    # A mask needs both bounds; one alone is a caller mistake worth reporting
    # rather than silently ignoring.
    if (request.mask_min is None) != (request.mask_max is None):
        return SmilesToSafeResponse(
            success=False,
            error="mask_min and mask_max must be provided together",
        )

    masked = False
    replaced_fragment = None
    if request.mask_min is not None and request.mask_max is not None:
        masked_str, replaced_fragment, mask_error = apply_mask(
            safe_str, request.mask_min, request.mask_max
        )
        if mask_error:
            logger.error(f"SAFE masking failed: {mask_error}")
            return SmilesToSafeResponse(success=False, error=mask_error)
        safe_str = masked_str
        masked = True

    # Fragment count excludes the mask token, which is a placeholder rather
    # than a fragment of the input molecule.
    num_fragments = len(safe_str.split(".")) - (1 if masked else 0)

    logger.info(
        f"SAFE encoded successfully ({num_fragments} fragments, "
        f"slicer={slicer_used}, masked={masked})"
    )
    return SmilesToSafeResponse(
        success=True,
        safe=safe_str,
        num_fragments=num_fragments,
        fragmented=fragmented,
        masked=masked,
        slicer_used=slicer_used,
        replaced_fragment=replaced_fragment,
    )


@app.post("/safe-to-smiles", response_model=SafeToSmilesResponse)
async def convert_safe_to_smiles(request: SafeToSmilesRequest):
    """Decode SAFE fragment notation back to canonical SMILES.

    Args:
        request: Contains the SAFE string.

    Returns:
        Response with the canonical SMILES or error message.
    """
    logger.info(f"Decoding SAFE to SMILES: {request.safe}")

    smiles, error = safe_to_smiles(request.safe)

    if error:
        logger.error(f"SAFE decoding failed: {error}")
        return SafeToSmilesResponse(success=False, error=error)

    logger.info("SAFE decoded successfully")
    return SafeToSmilesResponse(success=True, smiles=smiles)


