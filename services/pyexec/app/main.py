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

"""PyExec Service - FastAPI application for sandboxed Python execution.

This service provides a secure environment for executing Python code
with access to scientific computing libraries (numpy, scipy, rdkit).

Supports the Aevom data store integration, allowing code to load stored
molecular data (PDB, SDF, selections) via the pre-loaded `aevom` client.
"""

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .sandbox import execute_code, SandboxContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---- Backpressure / saturation telemetry -----------------------------------
#
# Each uvicorn worker process is single-threaded async, and ``execute_code``
# is fully synchronous (SIGALRM-based timeout requires the main thread of
# the worker), so per-worker concurrency is structurally capped at 1
# in-flight ``/execute`` request. The orchestrator (``services.yaml``)
# determines pool size via ``--workers N``.
#
# When offered load exceeds N workers, requests pile up in the kernel
# accept queue / asyncio task queue. From the worker's perspective, an
# inbound request that has been queue-waiting for many seconds is
# indistinguishable from one that just arrived -- but the *client* knows
# when it sent the request. We therefore have the client (the aevom
# tool registry, ``backend/api/tools/base.py``) inject an
# ``x-aevom-client-send-ns`` header at request-issue time, and the
# server here measures ``wire_wait_s = receive_time - client_send_time``.
# A non-zero wire-wait is the canonical backpressure signal: the request
# spent that long sitting in a queue between client and server. We emit
# a loud ``[AEVOM-PYEXEC-BACKPRESSURE]`` warning when wire_wait_s exceeds
# ``BACKPRESSURE_WIRE_WAIT_WARN_S`` (default 5s).
#
# We *deliberately do not* add a semaphore to cap admitted requests --
# the correct response to backpressure is to scale the service, not to
# silently mask the saturation by throttling. The whole point of this
# warning is to make the saturation impossible to miss.
#
# In-flight counter is per-worker-process (each uvicorn worker gets its
# own counter); aggregating across workers requires reading every
# worker's logs, which is fine for post-mortem, and is the right answer
# anyway because each worker is its own scheduling unit.

# The hard ceiling on a single execution. The caller (``python_execute``) sets
# its own lower defaults; this only has to be no lower than the longest budget
# the caller will ever ask for, which is the 180s used during training runs.
# When it was 60s, those requests were rejected outright with a 422.
MAX_TIMEOUT_SECONDS = 180
DEFAULT_TIMEOUT_SECONDS = 30

BACKPRESSURE_WIRE_WAIT_WARN_S = float(
    os.environ.get("AEVOM_PYEXEC_WIRE_WAIT_WARN_S", "5.0")
)
BACKPRESSURE_LIFETIME_WARN_S = float(
    os.environ.get("AEVOM_PYEXEC_LIFETIME_WARN_S", "30.0")
)

# Counter is process-local; uvicorn worker processes don't share state.
# Mutated only from the event loop, so no lock needed.
_in_flight_count: int = 0
_in_flight_max_seen: int = 0
_backpressure_warn_count: int = 0

# Libraries whose first import costs seconds. Imported before uvicorn binds when
# PYEXEC_WARMUP=1, so a healthy /health already implies a warm interpreter: the
# backend's sandbox pool treats that as ready.
_WARMUP_MODULES = (
    "numpy",
    "scipy",
    "scipy.spatial",
    "sklearn",
    "rdkit.Chem",
    "rdkit.Chem.AllChem",
    "rdkit.Chem.Descriptors",
    "Bio.PDB",
    "biotite.structure",
    "pubchempy",
    "safe",
)


def _warm_imports() -> None:
    import importlib

    started = time.monotonic()
    for name in _WARMUP_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # a missing optional lib must not block startup
            logger.warning("warmup import of %s failed: %s", name, exc)
    logger.info("warmup imports done in %.2fs", time.monotonic() - started)


if os.environ.get("PYEXEC_WARMUP") == "1":
    _warm_imports()

app = FastAPI(
    title="PyExec Service",
    description="Sandboxed Python execution for molecular analysis",
    version="0.1.0",
)

# Behind the OpenShell gateway, a connection uvicorn closes while idle is not
# reported back through the relay: the gateway keeps it pooled, and the next
# request sent down it never gets an answer. Closing after every response keeps
# the gateway from reusing a connection at all.
CONNECTION_CLOSE = os.environ.get("PYEXEC_CONNECTION_CLOSE") == "1"


@app.middleware("http")
async def _connection_close(request: Request, call_next):
    response = await call_next(request)
    if CONNECTION_CLOSE:
        response.headers["Connection"] = "close"
    return response


class SandboxContextRequest(BaseModel):
    """Request model for sandbox context (data store access)."""
    
    session_token: str = Field(
        ...,
        description="Session token for authenticating with backend data store",
    )
    backend_url: str = Field(
        ...,
        description="Base URL of the backend API (e.g., 'http://backend:8000')",
    )
    allowed_refs: list[str] = Field(
        default_factory=list,
        description="List of reference IDs this session can access",
    )


class ExecuteRequest(BaseModel):
    """Request model for code execution."""
    
    code: str = Field(
        ...,
        description="Python code to execute",
        examples=["import numpy as np\nprint(np.array([1,2,3]).sum())"],
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Variables to inject into the execution context (e.g., selection data)",
    )
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=MAX_TIMEOUT_SECONDS,
        description="Maximum execution time in seconds",
    )
    sandbox_context: SandboxContextRequest | None = Field(
        default=None,
        description="Sandbox context for data store access. When provided, the `aevom` client "
                    "will be pre-loaded, allowing code to call `aevom.load('pdb_abc123')` etc.",
    )


class ExecuteResponse(BaseModel):
    """Response model for code execution."""
    
    success: bool = Field(
        ...,
        description="Whether the code executed successfully",
    )
    output: str = Field(
        default="",
        description="Captured stdout from code execution",
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )
    result: Any = Field(
        default=None,
        description="Return value from the 'result' variable if set",
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "pyexec",
        "in_flight": _in_flight_count,
        "in_flight_max_seen": _in_flight_max_seen,
        "backpressure_warn_count": _backpressure_warn_count,
        "pid": os.getpid(),
    }


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest, http_request: Request) -> ExecuteResponse:
    """Execute Python code in a sandboxed environment.
    
    The code runs with access to:
    - numpy (as np)
    - math
    - scipy (via import)
    - rdkit (via import)
    - aevom (if sandbox_context provided) - for loading stored data
    
    Any variables in `context` are injected into the execution namespace.
    
    When `sandbox_context` is provided, the `aevom` client is pre-loaded:
    ```python
    # Load stored molecular data
    pdb_data = aevom.load("pdb_abc123")
    coords = aevom.load_coords("pdb_abc123")
    mol = aevom.load_rdkit_mol("sdf_xyz789")
    ```
    
    To return a result, assign it to a variable named `result`:
    ```python
    import numpy as np
    result = np.array([1, 2, 3]).mean()
    ```
    
    Security:
    - Code is compiled with RestrictedPython
    - Dangerous operations (file I/O, subprocess, etc.) are blocked
    - Execution has a configurable timeout
    - Data store access is scoped to allowed_refs only
    """
    global _in_flight_count, _in_flight_max_seen, _backpressure_warn_count

    t_received_ns = time.time_ns()
    t_perf_received = time.perf_counter()

    # Wire-wait: how long this request sat in the kernel/uvicorn queue
    # between the client's send and our handler dispatch. The canonical
    # backpressure signal -- non-zero wire-wait means the worker pool
    # was saturated when this request arrived. (Header is optional;
    # legacy clients without instrumentation produce wire_wait=0.)
    client_send_hdr = http_request.headers.get("x-aevom-client-send-ns")
    wire_wait_s: float = 0.0
    if client_send_hdr:
        try:
            wire_wait_s = max(0.0, (t_received_ns - int(client_send_hdr)) / 1e9)
        except (TypeError, ValueError):
            wire_wait_s = 0.0

    _in_flight_count += 1
    if _in_flight_count > _in_flight_max_seen:
        _in_flight_max_seen = _in_flight_count
    in_flight_at_entry = _in_flight_count

    try:
        logger.info(
            f"Executing code ({len(request.code)} chars), "
            f"timeout={request.timeout_seconds}s, "
            f"in_flight={in_flight_at_entry}, "
            f"wire_wait_s={wire_wait_s:.2f}"
        )

        if wire_wait_s >= BACKPRESSURE_WIRE_WAIT_WARN_S:
            _backpressure_warn_count += 1
            # Loud, structured, greppable. Log volume is naturally rate-limited
            # because we only emit once per overloaded request, not per second.
            logger.warning(
                "[AEVOM-PYEXEC-BACKPRESSURE] worker is saturated -- this "
                "request waited %.2fs in the kernel/uvicorn queue before "
                "being dispatched (warn threshold %.2fs). "
                "in_flight_at_entry=%d max_in_flight_seen=%d "
                "code_len=%d timeout=%ds pid=%d. "
                "Cause: offered load exceeds the per-worker pool size; "
                "scale workers in services.yaml or add full-stack service "
                "nodes (NUM_FULL_STACK_SERVICE_NODES). Do NOT mask this "
                "with a client-side semaphore -- backpressure must be "
                "visible to drive the right scaling decision.",
                wire_wait_s,
                BACKPRESSURE_WIRE_WAIT_WARN_S,
                in_flight_at_entry,
                _in_flight_max_seen,
                len(request.code),
                request.timeout_seconds,
                os.getpid(),
            )

        if len(request.code) > 50000:
            raise HTTPException(
                status_code=400,
                detail="Code too long (max 50,000 characters)",
            )

        sandbox_ctx = None
        if request.sandbox_context:
            sandbox_ctx = SandboxContext(
                session_token=request.sandbox_context.session_token,
                backend_url=request.sandbox_context.backend_url,
                allowed_refs=set(request.sandbox_context.allowed_refs),
            )
            logger.info(
                f"Sandbox context enabled with {len(sandbox_ctx.allowed_refs)} allowed refs"
            )

        result = execute_code(
            code=request.code,
            context=request.context,
            timeout_seconds=request.timeout_seconds,
            sandbox_context=sandbox_ctx,
        )

        if result["success"]:
            logger.info(f"Code executed successfully, output length: {len(result['output'])}")
        else:
            logger.warning(f"Code execution failed: {result['error']}")

        try:
            response = ExecuteResponse(
                success=result["success"],
                output=result["output"],
                error=result["error"],
                result=result["result"],
            )
            # Belt-and-suspenders: force serialization to happen *here*,
            # inside the try. FastAPI/Pydantic normally serialize the
            # returned model after the handler returns, which is outside any
            # try/except in this function -- a serialization failure there
            # (e.g. execute_code()'s sanitizer missing a case) would still
            # surface as a bare 500 despite this handler "succeeding".
            # Returning a pre-serialized Response bypasses that automatic
            # step, so this except clause is actually reachable.
            return Response(
                content=response.model_dump_json(),
                media_type="application/json",
            )
        except Exception as exc:
            # execute_code() already sanitizes `result` (see
            # sandbox._json_safe), so this should be unreachable -- but a
            # response-construction/serialization failure must never surface
            # as a bare 500 after the code ran successfully; that strips the
            # success/error context the caller
            # (backend/api/tools/python_exec.py) relies on to explain what
            # happened. Degrade to a clear, structured error instead.
            logger.error(f"Failed to build response for successful execution: {exc}")
            fallback = ExecuteResponse(
                success=False,
                output=result["output"],
                error=(
                    f"Code executed successfully, but its result could not be "
                    f"returned ({type(exc).__name__}: {exc}). Avoid assigning "
                    "an object that isn't JSON-serializable (a plain number, "
                    "string, list, or dict) to `result`."
                ),
                result=None,
            )
            return Response(
                content=fallback.model_dump_json(),
                media_type="application/json",
            )
    finally:
        lifetime_s = time.perf_counter() - t_perf_received
        _in_flight_count -= 1

        # Second backpressure signal: total handler lifetime well in
        # excess of the request's own ``timeout_seconds`` budget. This
        # catches the case where a client did NOT send the header (so
        # wire_wait_s reads 0) but we still spent a long time on the
        # request. Emitting this independently of the wire-wait branch
        # ensures we don't miss saturation just because of header
        # propagation gaps.
        if (
            wire_wait_s < BACKPRESSURE_WIRE_WAIT_WARN_S
            and lifetime_s > max(
                BACKPRESSURE_LIFETIME_WARN_S,
                request.timeout_seconds + 10.0,
            )
        ):
            _backpressure_warn_count += 1
            logger.warning(
                "[AEVOM-PYEXEC-BACKPRESSURE] handler lifetime %.2fs exceeded "
                "expected fast-path (timeout_seconds=%ds + 10s slack). "
                "wire_wait_s=%.2f (header may be missing) "
                "in_flight_at_entry=%d max_in_flight_seen=%d pid=%d. "
                "Likely cause: queue wait beyond what wire-wait could "
                "measure, or genuinely slow user code. Inspect the next "
                "few warnings: if many fire concurrently across workers, "
                "the pool is saturated and needs scaling.",
                lifetime_s,
                request.timeout_seconds,
                wire_wait_s,
                in_flight_at_entry,
                _in_flight_max_seen,
                os.getpid(),
            )


@app.get("/info")
async def info():
    """Get information about available libraries and capabilities."""
    libraries = {}
    
    # Check numpy
    try:
        import numpy as np
        libraries["numpy"] = np.__version__
    except ImportError:
        libraries["numpy"] = None
    
    # Check scipy
    try:
        import scipy
        libraries["scipy"] = scipy.__version__
    except ImportError:
        libraries["scipy"] = None
    
    # Check scikit-learn
    try:
        import sklearn
        libraries["scikit-learn"] = sklearn.__version__
    except ImportError:
        libraries["scikit-learn"] = None
    
    # Check rdkit
    try:
        import rdkit
        libraries["rdkit"] = rdkit.__version__
    except ImportError:
        libraries["rdkit"] = None
    
    # Check biopython
    try:
        import Bio
        libraries["biopython"] = Bio.__version__
    except ImportError:
        libraries["biopython"] = None

    # Check openbabel
    try:
        from openbabel import openbabel as _ob
        libraries["openbabel"] = _ob.OBReleaseVersion()
    except Exception:
        libraries["openbabel"] = None

    # Check biotite
    try:
        import biotite
        libraries["biotite"] = biotite.__version__
    except ImportError:
        libraries["biotite"] = None

    # Check pubchempy
    try:
        import pubchempy
        libraries["pubchempy"] = pubchempy.__version__
    except ImportError:
        libraries["pubchempy"] = None

    # Check safe-mol (SAFE fragment notation)
    try:
        import importlib.metadata

        libraries["safe-mol"] = importlib.metadata.version("safe-mol")
    except Exception:
        libraries["safe-mol"] = None
    
    return {
        "service": "pyexec",
        "libraries": libraries,
        "capabilities": [
            "Scientific computing with numpy and scipy",
            "Clustering and regression with scikit-learn",
            "Molecular analysis with RDKit",
            "Structure parsing with BioPython and biotite",
            "File-format conversion and protonation with OpenBabel",
            "SAFE fragment notation encoding/decoding (safe-mol)",
            "Selection data analysis (coords, distances, etc.)",
            "Aevom data store integration (load/store PDB, SDF, sequences by reference)",
            "MolStar visualization from Python code",
            "PubChem molecule search and download",
        ],
        "limits": {
            "max_code_length": 50000,
            "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
            "default_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        },
        "aevom_functions": [
            "aevom.load(ref) - Load any stored data by reference",
            "aevom.load_pdb(ref) - Load PDB as string",
            "aevom.load_sdf(ref) - Load SDF as string",
            "aevom.load_coords(ref) - Extract coordinates as numpy array",
            "aevom.load_rdkit_mol(ref) - Load SDF as RDKit Mol object",
            "aevom.load_pdb_structure(ref) - Load PDB with full atom info",
            "aevom.load_selection(ref) - Load a selection dict; .residues() lists residues with author and label numbering",
            "aevom.store_selection(structure_ref, residues, chain, name) - Make a selection from residue numbers, return its reference",
            "aevom.load_safe(ref) - Load SAFE fragment notation as string",
            "aevom.store(type, content, metadata) - Store data, return reference",
            "aevom.store_sdf(content, metadata) - Store SDF data",
            "aevom.store_pdb(content, metadata) - Store PDB data",
            "aevom.store_smiles(smiles, metadata) - Store SMILES string",
            "aevom.store_safe(safe_str, metadata) - Store SAFE fragment notation",
            "aevom.safe_from_smiles(smiles, slicer) - Encode SMILES as SAFE",
            "aevom.smiles_from_safe(safe_str) - Decode SAFE back to SMILES",
            "aevom.scaffold_to_safe(smiles, min, max) - SAFE scaffold + GenMol mask",
            "aevom.visualize(mol_or_sdf, label) - Display in MolStar viewer",
            "aevom.mol_to_sdf(mol) - Convert RDKit Mol to SDF string",
            "aevom.mol_from_sdf(sdf) - Parse SDF to RDKit Mol",
            "aevom.mol_from_smiles(smiles) - Create RDKit Mol from SMILES",
            "aevom.canonical_smiles(mol) - Get canonical SMILES",
            "aevom.embed_mol_3d(mol) - Generate 3D coordinates",
            "aevom.sa_score(mol_or_smiles) - Synthetic accessibility, 1 (easy) to 10",
            "aevom.lookup_ccd_code(name) - Look up PDB CCD code for molecule (e.g., 'imatinib' -> 'STI')",
            "aevom.search_pubchem(name) - Search PubChem by name, returns {cid, isomeric_smiles, inchi, ...}",
            "aevom.get_pubchem_properties(cid) - Get MW, TPSA, h_bond_donor_count, etc.",
            "aevom.download_pubchem_sdf(cid) - Download 3D structure from PubChem as SDF string",
        ],
    }



