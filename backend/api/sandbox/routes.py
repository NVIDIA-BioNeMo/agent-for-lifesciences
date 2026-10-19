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

"""API routes for sandbox data access.

These endpoints are called by the pyexec service to retrieve
stored data artifacts during code execution.

Endpoints:
- GET /api/sandbox/data/{reference} - Retrieve stored data
- POST /api/sandbox/store - Store new data from sandbox code
- POST /api/sandbox/visualize - Trigger MolStar visualization from sandbox code
- POST /api/sandbox/selection - Build a residue selection from a structure ref
- POST /api/sandbox/probe-grid - Compute a probe energy grid, return a reference
- POST /api/sandbox/constrained-pose - Verify candidates against posed fragments
"""

import base64
import binascii
import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.config import get_async_session
from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    SessionScopedStorage,
    scoped_storage,
)
from api.storage.datatypes import content_type_for

from .session import (
    get_session_manager,
    SandboxSessionManager,
    SessionNotFoundError,
    ReferenceNotAllowedError,
    SandboxSessionError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class StoreRequest(BaseModel):
    """Request model for storing data from sandbox.

    Exactly one of ``content`` (text) or ``content_base64`` (binary) is used.
    Binary support exists for ``array`` artifacts, which are ``.npz`` payloads
    written by ``aevom.store_array()`` and by skill endpoint tools.
    """

    data_type: str = Field(
        ...,
        description="Type of data (e.g., 'sdf', 'pdb', 'json', 'array')",
        examples=["sdf", "pdb", "array"],
    )
    content: Optional[str] = Field(
        default=None,
        description="String content to store (UTF-8 text types).",
    )
    content_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded content, for binary types such as 'array'.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short description of the data for reference tracking",
    )


class StoreResponse(BaseModel):
    """Response model for store operation."""

    success: bool
    reference: Optional[str] = None
    error: Optional[str] = None


class VisualizeRequest(BaseModel):
    """Request model for triggering visualization."""

    sdf: Optional[str] = Field(
        default=None,
        description="SDF content for small molecule visualization",
    )
    pdb: Optional[str] = Field(
        default=None,
        description="PDB content for protein structure visualization",
    )
    label: str = Field(
        default="Molecule",
        description="Display label for the structure",
    )


class SelectionSandboxRequest(BaseModel):
    """Request model for a residue selection built on behalf of sandbox code."""

    structure_ref: str = Field(..., description="PDB or mmCIF reference")
    chain: str = Field(..., description="Chain id, in the given numbering system")
    residues: list[int | str] = Field(
        ..., min_length=1, description="Residue numbers and ranges: [56, '100-120']"
    )
    numbering: str = Field(default="author", description="'author' or 'label'")
    atoms: Optional[list[str] | str] = Field(
        default=None,
        description="Atom names to keep, or 'backbone' / 'sidechain'",
    )
    name: Optional[str] = Field(default=None, max_length=80)
    parent: Optional[str] = Field(default=None, description="Selection this derives from")


class SelectionSandboxResponse(BaseModel):
    """Response carrying the stored selection's reference and what was found.

    ``missing`` lists residues (``"56"``) and atoms of residues that are not
    there (``"56:NZ"``).
    """

    success: bool
    reference: Optional[str] = None
    atom_count: int = 0
    residue_count: int = 0
    missing: list[str] = Field(default_factory=list)
    error: Optional[str] = None


DEFAULT_SANDBOX_PROBES = ("C_H", "N_D", "O_A")


class ProbeGridSandboxRequest(BaseModel):
    """Request model for a probe grid computed on behalf of sandbox code."""

    receptor_ref: str = Field(..., description="PDB reference for the receptor")
    center: list[float] = Field(
        ..., min_length=3, max_length=3, description="Box centre [x, y, z] in angstrom"
    )
    size: list[float] = Field(
        default=[20.0, 20.0, 20.0],
        min_length=3,
        max_length=3,
        description="Box dimensions [x, y, z] in angstrom",
    )
    spacing: float = Field(default=0.5, ge=0.2, le=1.0)
    probes: Optional[list[str]] = Field(default=None)
    max_minima: int = Field(default=5, ge=1, le=50)


class ProbeGridSandboxResponse(BaseModel):
    """Response model carrying a stored grid reference, never the grid itself."""

    success: bool
    reference: Optional[str] = None
    probes: Optional[list[str]] = None
    origin: Optional[list[float]] = None
    spacing: Optional[float] = None
    shape: Optional[list[int]] = None
    minima: Optional[list[dict[str, Any]]] = None
    error: Optional[str] = None


MAX_SANDBOX_CANDIDATES = 200


class ConstrainedPoseSandboxRequest(BaseModel):
    """Request model for constrained-pose verification on behalf of sandbox code."""

    fragment_refs: list[str] = Field(
        ..., min_length=1, max_length=4, description="SDF references, one per fragment"
    )
    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_SANDBOX_CANDIDATES,
        description="Candidate molecules to verify",
    )
    pocket_ref: Optional[str] = Field(
        default=None, description="PDB reference for the pocket, for the clash check"
    )
    num_seeds: int = Field(default=4, ge=1, le=16)
    strain_max: Optional[float] = Field(default=None, ge=0.0)
    rmsd_max: Optional[float] = Field(default=None, gt=0.0)
    store_poses: bool = Field(
        default=False,
        description=(
            "Store each verified pose as an sdf reference. Off by default: a "
            "scoring loop over hundreds of candidates wants the numbers, and "
            "storing every pose would fill the artifact store with rejects."
        ),
    )


class ConstrainedPoseSandboxResponse(BaseModel):
    """Response model carrying per-candidate measurements, poses only as refs."""

    success: bool
    results: Optional[list[dict[str, Any]]] = None
    num_verified: Optional[int] = None
    pocket_atoms: Optional[int] = None
    thresholds: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class VisualizeResponse(BaseModel):
    """Response model for visualization request."""

    success: bool
    queued: bool = False
    error: Optional[str] = None


router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


def get_content_type(data_type: str) -> str:
    """Get appropriate content type for a data type."""
    return content_type_for(data_type)


def _client_storage(session: AsyncSession, sandbox_session: Any) -> SessionScopedStorage:
    """Storage confined to the chat session the sandbox token was issued for.

    ``allowed_refs`` says which references sandbox code may name; this makes
    the reads and writes behind them stay inside the issuing session too. A
    token not bound to a client session gets no storage at all.
    """
    client_session_id = getattr(sandbox_session, "client_session_id", None)
    if not client_session_id:
        raise HTTPException(
            status_code=403, detail="Sandbox session is not bound to a client session"
        )
    return scoped_storage(session, client_session_id)


@router.get("/data/{reference}")
async def get_sandbox_data(
    reference: str,
    x_sandbox_session: Annotated[str, Header()],
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Retrieve stored data for sandbox execution.

    This endpoint is called by the pyexec aevom client to load
    data artifacts by reference. Access is controlled by sandbox
    session tokens that specify which references are allowed.
    """
    manager = get_session_manager()

    # Validate session and reference access
    try:
        sandbox_session = await manager.validate_access(x_sandbox_session, reference)
    except SessionNotFoundError as e:
        logger.warning(f"Sandbox data request with invalid session: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except ReferenceNotAllowedError as e:
        logger.warning(f"Sandbox data request for non-allowed ref: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except SandboxSessionError as e:
        # Rate limit exceeded
        logger.warning(f"Sandbox session error: {e}")
        raise HTTPException(status_code=429, detail=str(e))

    # Retrieve data from storage
    storage = _client_storage(session, sandbox_session)

    try:
        content, metadata = await storage.retrieve_with_metadata(reference)
    except InvalidReferenceError:
        logger.warning(f"Sandbox requested invalid ref format: {reference}")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid reference format: '{reference}'. "
                "References use the format 'type_id' (e.g., 'pdb_abc12345', 'sdf_abc12345'). "
                "Use fetch_and_clean_pdb or fetch_protein_sequence to obtain valid references."
            ),
        )
    except ArtifactNotFoundError:
        logger.warning(f"Sandbox requested non-existent ref: {reference}")
        raise HTTPException(status_code=404, detail=f"Reference not found: {reference}")

    # Determine content type from reference prefix
    data_type = storage.get_data_type(reference)
    content_type = get_content_type(data_type)

    logger.info(
        f"Sandbox session {x_sandbox_session[:8]}... retrieved {reference} "
        f"({len(content)} bytes, type: {data_type})"
    )

    return Response(
        content=content,
        media_type=content_type,
        headers={
            "X-Data-Type": data_type,
            "X-Reference": reference,
        },
    )


@router.get("/health")
async def sandbox_health():
    """Health check for sandbox endpoints."""
    manager = get_session_manager()
    return {
        "status": "healthy",
        "active_sessions": manager.active_session_count,
    }


@router.post("/store", response_model=StoreResponse)
async def store_sandbox_data(
    request: StoreRequest,
    x_sandbox_session: Annotated[str, Header()],
    session: AsyncSession = Depends(get_async_session),
) -> StoreResponse:
    """Store data from sandbox execution.

    This endpoint allows sandbox code (via aevom.store()) to persist
    molecular data back to the data store.
    """
    manager = get_session_manager()

    # Validate session exists (don't require specific ref access for storing)
    try:
        sandbox_session = await manager.get_session(x_sandbox_session)
    except SessionNotFoundError as e:
        logger.warning(f"Store request with invalid session: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    # Validate request
    if not request.data_type:
        raise HTTPException(status_code=400, detail="data_type is required")
    if request.content_base64:
        try:
            content_bytes = base64.b64decode(request.content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=400, detail=f"content_base64 is not valid base64: {exc}"
            )
    elif request.content:
        content_bytes = request.content.encode("utf-8")
    else:
        raise HTTPException(
            status_code=400, detail="content or content_base64 is required"
        )

    # Store data
    storage = _client_storage(session, sandbox_session)

    # Merge user-provided metadata with client session_id for artifact scoping
    metadata = dict(request.metadata) if request.metadata else {}
    if sandbox_session.client_session_id:
        metadata["session_id"] = sandbox_session.client_session_id

    try:
        reference = await storage.store(
            data_type=request.data_type,
            content=content_bytes,
            metadata=metadata,
            description=request.description,
        )

        # Add the new reference to allowed refs for this session
        await manager.add_allowed_ref(x_sandbox_session, reference)

        logger.info(
            f"Sandbox session {x_sandbox_session[:8]}... stored {request.data_type} "
            f"({len(content_bytes)} bytes) as {reference} "
            f"(client_session: {sandbox_session.client_session_id[:8] if sandbox_session.client_session_id else None}...)"
        )

        return StoreResponse(success=True, reference=reference)

    except Exception as e:
        logger.exception(f"Failed to store sandbox data: {e}")
        return StoreResponse(success=False, error=str(e))


@router.post("/visualize", response_model=VisualizeResponse)
async def visualize_sandbox_molecule(
    request: VisualizeRequest,
    x_sandbox_session: Annotated[str, Header()],
) -> VisualizeResponse:
    """Queue a visualization request from sandbox execution.

    This endpoint allows sandbox code (via aevom.visualize()) to trigger
    MolStar visualization. The visualization request is queued on the
    session and will be processed after code execution completes.
    """
    manager = get_session_manager()

    # Validate session
    try:
        sandbox_session = await manager.get_session(x_sandbox_session)
    except SessionNotFoundError as e:
        logger.warning(f"Visualize request with invalid session: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    # Validate request
    if not request.sdf and not request.pdb:
        raise HTTPException(
            status_code=400, detail="Either 'sdf' or 'pdb' content is required"
        )

    # Queue the visualization event on the session
    viz_event = {
        "label": request.label,
    }
    if request.sdf:
        viz_event["type"] = "visualize_3d"
        viz_event["sdf"] = request.sdf
    elif request.pdb:
        viz_event["type"] = "visualize_pdb"
        viz_event["pdb"] = request.pdb

    await manager.add_visualization_event(x_sandbox_session, viz_event)

    logger.info(
        f"Sandbox session {x_sandbox_session[:8]}... queued visualization "
        f"({viz_event['type']}, label: {request.label})"
    )

    return VisualizeResponse(success=True, queued=True)


def _redis_or_none() -> Optional[Any]:
    """The shared Redis client, or None when this process has none configured."""
    try:
        from api.redis import get_redis

        return get_redis()
    except (RuntimeError, ImportError):
        return None


@router.post("/selection", response_model=SelectionSandboxResponse)
async def sandbox_store_selection(
    request: SelectionSandboxRequest,
    x_sandbox_session: Annotated[str, Header()],
    session: AsyncSession = Depends(get_async_session),
) -> SelectionSandboxResponse:
    """Build a residue selection from a stored structure on behalf of sandbox code.

    The sandbox has coordinates but no registry and no viewer. This builds the
    same kind of ``selection`` artifact a user pick or the ``select_residues``
    tool produces, registers it so it appears in the next turn's selection
    state, and returns the reference -- so analysis code can hand a computed
    set of residues straight to a design tool or show it to the user.
    """
    from api.selections.ops import SelectionError, parse_residue_spec, select_residues
    from api.selections.service import store_built_selection
    from api.selections.structure import parse_structure_atoms

    manager = get_session_manager()
    try:
        sandbox_session = await manager.validate_access(
            x_sandbox_session, request.structure_ref
        )
    except SessionNotFoundError as e:
        logger.warning(f"Selection request with invalid session: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except ReferenceNotAllowedError as e:
        logger.warning(f"Selection request for non-allowed ref: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except SandboxSessionError as e:
        logger.warning(f"Sandbox session error: {e}")
        raise HTTPException(status_code=429, detail=str(e))

    storage = _client_storage(session, sandbox_session)
    try:
        raw = await storage.retrieve(request.structure_ref)
    except InvalidReferenceError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reference format: '{request.structure_ref}'",
        )
    except ArtifactNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Reference not found: {request.structure_ref}"
        )

    try:
        structure = parse_structure_atoms(raw.decode("utf-8", errors="replace"))
        if not structure:
            raise SelectionError(
                f"Could not read any atoms from '{request.structure_ref}'."
            )
        found = select_residues(
            structure,
            request.chain,
            parse_residue_spec(request.residues),
            numbering=request.numbering,
            atom_names=request.atoms,
        )
    except SelectionError as e:
        return SelectionSandboxResponse(success=False, error=str(e))
    if not found.atoms:
        return SelectionSandboxResponse(
            success=False,
            missing=found.missing,
            error=(
                f"None of those residues exist in chain {request.chain}: "
                f"{', '.join(found.missing)}."
            ),
        )

    redis_client = _redis_or_none()

    client_session = sandbox_session.client_session_id or ""
    stored = await store_built_selection(
        [a.to_payload() for a in found.atoms],
        storage=storage,
        redis_client=redis_client,
        session_id=client_session,
        provenance="agent",
        scope="atom" if found.atom_level else "residue",
        name=request.name,
        parent=request.parent,
        commit=session.commit,
    )
    await manager.add_allowed_ref(x_sandbox_session, stored["reference"])
    return SelectionSandboxResponse(
        success=True,
        reference=stored["reference"],
        atom_count=stored["atom_count"],
        residue_count=stored["residue_count"],
        missing=found.missing,
    )


@router.post("/probe-grid", response_model=ProbeGridSandboxResponse)
async def sandbox_probe_grid(
    request: ProbeGridSandboxRequest,
    x_sandbox_session: Annotated[str, Header()],
    session: AsyncSession = Depends(get_async_session),
) -> ProbeGridSandboxResponse:
    """Compute a probe interaction-energy grid on behalf of sandbox code.

    The sandbox cannot reach the Vina service directly, and should not: the
    backend is deliberately its only outbound dependency. So the backend
    resolves the receptor reference, calls the service, stores the grid, and
    hands back a reference that ``aevom.probe_grid()`` turns into a numpy array
    through the ordinary data route. A 166,000-point grid therefore never passes
    through the model's context.

    The same service endpoint backs the ``probe_grid`` tool; this route exists
    so Python code can get the array rather than the summary.
    """
    manager = get_session_manager()

    try:
        sandbox_session = await manager.validate_access(
            x_sandbox_session, request.receptor_ref
        )
    except SessionNotFoundError as e:
        logger.warning(f"Probe grid request with invalid session: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except ReferenceNotAllowedError as e:
        logger.warning(f"Probe grid request for non-allowed ref: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except SandboxSessionError as e:
        logger.warning(f"Sandbox session error: {e}")
        raise HTTPException(status_code=429, detail=str(e))

    storage = _client_storage(session, sandbox_session)
    try:
        receptor_bytes = await storage.retrieve(request.receptor_ref)
    except InvalidReferenceError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reference format: '{request.receptor_ref}'",
        )
    except ArtifactNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Reference not found: {request.receptor_ref}"
        )

    from api.tools.probe_grid import PROBE_TYPES, REQUEST_TIMEOUT, ProbeGridTool

    probes = request.probes or list(DEFAULT_SANDBOX_PROBES)
    unknown = [p for p in probes if p not in PROBE_TYPES]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown probe type(s): {', '.join(unknown)}. "
                f"Available: {', '.join(PROBE_TYPES)}"
            ),
        )

    tool = ProbeGridTool()
    try:
        service_response = await tool._call_service(
            endpoint="/probe-grid",
            method="POST",
            json_data={
                "pdb_content": receptor_bytes.decode("utf-8"),
                "center": [float(v) for v in request.center],
                "size": [float(v) for v in request.size],
                "spacing": float(request.spacing),
                "probes": probes,
                "max_minima": int(request.max_minima),
            },
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - reported to sandbox code as an error
        logger.exception("Sandbox probe grid request failed")
        return ProbeGridSandboxResponse(success=False, error=str(exc))

    if not service_response.get("success"):
        return ProbeGridSandboxResponse(
            success=False, error=service_response.get("error", "unknown error")
        )

    try:
        npz_bytes = base64.b64decode(service_response["npz_base64"], validate=True)
    except (KeyError, ValueError, binascii.Error) as exc:
        return ProbeGridSandboxResponse(
            success=False, error=f"Unreadable grid payload: {exc}"
        )

    metadata = {
        "source": "aevom.probe_grid",
        "receptor_ref": request.receptor_ref,
        "probes": service_response.get("probes"),
        "origin": service_response.get("origin"),
        "spacing": service_response.get("spacing"),
        "shape": service_response.get("shape"),
        "center": [float(v) for v in request.center],
        "size": [float(v) for v in request.size],
    }
    if sandbox_session.client_session_id:
        metadata["session_id"] = sandbox_session.client_session_id

    reference = await storage.store(
        data_type="array",
        content=npz_bytes,
        metadata=metadata,
        description=f"Probe energy grid for {request.receptor_ref}",
    )
    # Without this the helper's own load_array() would be refused by the same
    # session that just asked for the grid.
    await manager.add_allowed_ref(x_sandbox_session, reference)

    logger.info(
        f"Sandbox session {x_sandbox_session[:8]}... computed probe grid "
        f"{service_response.get('shape')} for {len(probes)} probe(s) as {reference}"
    )

    return ProbeGridSandboxResponse(
        success=True,
        reference=reference,
        probes=service_response.get("probes"),
        origin=service_response.get("origin"),
        spacing=service_response.get("spacing"),
        shape=service_response.get("shape"),
        minima=service_response.get("minima"),
    )


@router.post("/constrained-pose", response_model=ConstrainedPoseSandboxResponse)
async def sandbox_constrained_pose(
    request: ConstrainedPoseSandboxRequest,
    x_sandbox_session: Annotated[str, Header()],
    session: AsyncSession = Depends(get_async_session),
) -> ConstrainedPoseSandboxResponse:
    """Verify candidate molecules against posed fragments for sandbox code.

    Same service call the ``fragment_link`` tool makes, exposed so a
    model-written loop can score its own enumeration -- hundreds of candidates,
    filtered and ranked in Python -- without the poses or the rejected rows
    reaching the model's context. Only what the code prints does.

    Poses come back as references rather than SDF blocks, and only when asked
    for: the measurements are the point, and storing a pose per reject would
    fill the artifact store with molecules nobody wants.
    """
    manager = get_session_manager()

    references = list(request.fragment_refs) + (
        [request.pocket_ref] if request.pocket_ref else []
    )
    try:
        for reference in references:
            sandbox_session = await manager.validate_access(
                x_sandbox_session, reference
            )
    except SessionNotFoundError as e:
        logger.warning(f"Constrained pose request with invalid session: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except ReferenceNotAllowedError as e:
        logger.warning(f"Constrained pose request for non-allowed ref: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except SandboxSessionError as e:
        logger.warning(f"Sandbox session error: {e}")
        raise HTTPException(status_code=429, detail=str(e))

    storage = _client_storage(session, sandbox_session)
    contents: list[str] = []
    for reference in references:
        try:
            contents.append((await storage.retrieve(reference)).decode("utf-8"))
        except InvalidReferenceError:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference format: '{reference}'"
            )
        except ArtifactNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Reference not found: {reference}"
            )

    fragment_sdfs = contents[: len(request.fragment_refs)]
    pocket_pdb = contents[len(request.fragment_refs) :]

    from api.tools.fragment_link import VERIFY_TIMEOUT, FragmentLinkTool

    payload: dict[str, Any] = {
        "smiles": [str(s) for s in request.smiles],
        "fragment_sdfs": fragment_sdfs,
        "pocket_pdb": pocket_pdb[0] if pocket_pdb else "",
        "num_seeds": int(request.num_seeds),
        "include_poses": True,
    }
    if request.strain_max is not None:
        payload["strain_max"] = float(request.strain_max)
    if request.rmsd_max is not None:
        payload["rmsd_max"] = float(request.rmsd_max)

    tool = FragmentLinkTool()
    try:
        service_response = await tool._call_service(
            endpoint="/constrained-pose",
            method="POST",
            json_data=payload,
            timeout=VERIFY_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - reported to sandbox code as an error
        logger.exception("Sandbox constrained pose request failed")
        return ConstrainedPoseSandboxResponse(success=False, error=str(exc))

    if not service_response.get("success"):
        return ConstrainedPoseSandboxResponse(
            success=False, error=service_response.get("error", "unknown error")
        )

    results = service_response.get("results") or []
    for result in results:
        pose_sdf = result.pop("pose_sdf", None)
        if not (pose_sdf and request.store_poses and result.get("verified")):
            continue
        metadata = {
            "source": "aevom.constrained_pose",
            "smiles": result.get("smiles"),
            "fragment_refs": list(request.fragment_refs),
            "strain": result.get("strain"),
            "min_protein_distance": result.get("min_protein_distance"),
        }
        if sandbox_session.client_session_id:
            metadata["session_id"] = sandbox_session.client_session_id
        reference = await storage.store(
            data_type="sdf",
            content=pose_sdf.encode("utf-8"),
            metadata=metadata,
            description=f"Constrained pose for {result.get('smiles')}",
        )
        # Without this the sandbox could not load back the pose it just made.
        await manager.add_allowed_ref(x_sandbox_session, reference)
        result["pose_ref"] = reference

    logger.info(
        f"Sandbox session {x_sandbox_session[:8]}... verified "
        f"{len(request.smiles)} candidate(s), "
        f"{service_response.get('num_verified')} held their poses"
    )

    return ConstrainedPoseSandboxResponse(
        success=True,
        results=results,
        num_verified=service_response.get("num_verified"),
        pocket_atoms=service_response.get("pocket_atoms"),
        thresholds=service_response.get("thresholds"),
    )


@router.get("/visualizations/{session_token}")
async def get_pending_visualizations(
    session_token: str,
    x_sandbox_session: Annotated[str, Header()],
) -> list[dict]:
    """Get and clear pending visualization events for a session.

    This is called by python_execute after code execution to retrieve
    any visualization events that were queued during execution.
    """
    # Verify caller matches the session
    if session_token != x_sandbox_session:
        raise HTTPException(status_code=403, detail="Session token mismatch")

    manager = get_session_manager()

    try:
        events = await manager.get_and_clear_visualization_events(session_token)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if events:
        logger.info(
            f"Session {session_token[:8]}... returned {len(events)} visualization events"
        )

    return events
