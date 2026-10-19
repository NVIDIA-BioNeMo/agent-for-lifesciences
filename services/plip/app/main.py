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

"""FastAPI service for protein-ligand interaction profiling.

Endpoints:
- ``/health`` -- liveness
- ``/interactions`` -- profile one or more structures and fold a consensus

One request carries the whole ensemble rather than one structure, because the
question worth asking of ten diffusion samples is what they agree on, and that is
not answerable one HTTP call at a time.
"""

import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .consensus import DEFAULT_THRESHOLD, fold
from .formats import ConversionError, to_pdb
from .interactions import FAMILIES, FAMILY_LABELS, ProfilingError, profile

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# PLIP writes the protonated structure to a fixed path when reading from a
# string, so two analyses running at once would race on that file. Hence one at a
# time. The lock is taken per structure rather than per request: a 25-structure
# ensemble would otherwise hold it for half a minute and stall every other
# session, and interleaving costs nothing since each analysis is independent.
_analysis_lock = asyncio.Lock()

# Bounds on one request. Twenty-five covers Boltz-2's ten diffusion samples and
# Vina's nine modes with room to spare.
MAX_STRUCTURES = 25
MAX_STRUCTURE_BYTES = 40 * 1024 * 1024


class HealthCheckFilter(logging.Filter):
    """Suppress health check access logs unless LOG_LEVEL is DEBUG."""

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
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

_health_initialized = False

app = FastAPI(
    title="PLIP Service",
    description="Protein-ligand interaction profiling using PLIP",
    version="1.0.0",
)


class Structure(BaseModel):
    """One structure to profile."""

    name: str = Field(..., description="Label for this structure, used in the consensus")
    content: str = Field(..., description="Structure text, PDB or mmCIF")


class InteractionsRequest(BaseModel):
    """Request model for interaction profiling."""

    structures: list[Structure] = Field(
        ...,
        min_length=1,
        max_length=MAX_STRUCTURES,
        description="Structures to profile. Several means a consensus is folded.",
    )
    ligand: str = Field(
        default="",
        description=(
            "Optional hetid ('E20') or full identifier ('E20:A:2001') to "
            "profile. Defaults to the largest interacting ligand."
        ),
    )
    consensus_threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Fraction of structures an interaction must appear in",
    )


class InteractionsResponse(BaseModel):
    """Response model for interaction profiling."""

    success: bool
    structures: Optional[list[dict]] = None
    consensus: Optional[dict] = None
    consensus_residues: Optional[list[dict]] = None
    failed: Optional[list[dict]] = None
    families: Optional[dict] = None
    error: Optional[str] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    global _health_initialized

    if not _health_initialized:
        logger.info("PLIP service is online and healthy")
        _health_initialized = True
    else:
        logger.debug("Health check heartbeat")

    return {"status": "healthy", "service": "plip"}


@app.get("/families")
async def families():
    """Report the interaction families this service detects."""
    return {
        "families": [
            {"name": name, "label": FAMILY_LABELS[name]} for name in FAMILIES
        ],
        "max_structures": MAX_STRUCTURES,
    }


# Elements that are a cofactor rather than part of the protein or the ligand.
# Reported by absence as much as by presence: a metalloenzyme co-folded without
# its metal has an apo site, and every interaction PLIP finds there is a
# substitute for the one that actually anchors the ligand.
_COFACTOR_ELEMENTS = frozenset(
    "ZN MG MN FE CU CA NI CO CD NA K".split()
)


def _cofactors(pdb_text: str) -> list[str]:
    """Metal ions and heme present as HETATM, by residue name."""
    found: set[str] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        residue = line[17:20].strip().upper()
        element = line[76:78].strip().upper()
        if residue in ("HEM", "HEC", "HEA", "HEB", "SF4", "FES", "NAP", "FAD",
                       "FMN", "PLP", "SAM", "COA", "NAD"):
            found.add(residue)
        elif element in _COFACTOR_ELEMENTS and residue == element:
            found.add(residue)
    return sorted(found)


def _profile_one(name: str, content: str, ligand: str) -> dict:
    """Convert, profile, and package one structure. Runs in a worker thread."""
    pdb_text = to_pdb(content)
    result = profile(pdb_text, ligand=ligand)
    payload = result.to_dict()
    payload["name"] = name
    site = result.site()
    payload["fingerprint"] = site.fingerprint if site else []
    payload["cofactors"] = _cofactors(pdb_text)
    return payload


@app.post("/interactions", response_model=InteractionsResponse)
async def interactions(request: InteractionsRequest):
    """Profile protein-ligand interactions and fold a consensus across structures.

    Each structure is profiled independently; a structure that cannot be read is
    reported in ``failed`` rather than failing the whole request, so one bad
    diffusion sample does not discard the other nine.

    Args:
        request: Structures, optional ligand designation, consensus threshold.

    Returns:
        Per-structure interactions, the consensus fold, and any failures.
    """
    logger.info(
        f"Profiling {len(request.structures)} structure(s) "
        f"(ligand={request.ligand or 'auto'}, "
        f"threshold={request.consensus_threshold})"
    )

    oversized = [
        structure.name
        for structure in request.structures
        if len(structure.content.encode("utf-8")) > MAX_STRUCTURE_BYTES
    ]
    if oversized:
        return InteractionsResponse(
            success=False,
            error=(
                f"These structures exceed the "
                f"{MAX_STRUCTURE_BYTES // (1024 * 1024)}MB per-structure limit: "
                f"{', '.join(oversized)}"
            ),
        )

    profiled: list[dict] = []
    failed: list[dict] = []

    for structure in request.structures:
        try:
            async with _analysis_lock:
                payload = await asyncio.to_thread(
                    _profile_one, structure.name, structure.content, request.ligand
                )
        except (ConversionError, ProfilingError) as exc:
            logger.warning("Could not profile %s: %s", structure.name, exc)
            failed.append({"name": structure.name, "error": str(exc)})
            continue
        # One bad input must not fail the batch.
        except (Exception, SystemExit) as exc:
            logger.exception("Unexpected failure profiling %s", structure.name)
            failed.append({"name": structure.name, "error": str(exc)})
            continue
        profiled.append(payload)

    if not profiled:
        return InteractionsResponse(
            success=False,
            failed=failed,
            error=(
                "None of the structures could be profiled. "
                + "; ".join(entry["error"] for entry in failed[:3])
            ),
        )

    consensus = fold(
        {payload["name"]: payload["fingerprint"] for payload in profiled},
        threshold=request.consensus_threshold,
    )

    logger.info(
        f"Profiled {len(profiled)} structure(s), {len(failed)} failed; "
        f"{len(consensus.conserved)} conserved interactions"
    )

    return InteractionsResponse(
        success=True,
        structures=profiled,
        consensus=consensus.to_dict(),
        consensus_residues=consensus.residues(),
        failed=failed,
        families={name: FAMILY_LABELS[name] for name in FAMILIES},
    )
