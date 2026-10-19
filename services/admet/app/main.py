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

"""FastAPI service for ADMET property prediction.

Wraps the ADMET-AI ensemble (Chemprop models trained on the Therapeutics Data
Commons ADMET datasets) behind the same request/response shape as
``services/rdkit``: JSON in, JSON out, ``success`` plus ``error`` rather than
HTTP status codes for domain failures, so a caller reads one field to know what
happened.

Endpoints:
- GET  /health     - liveness: the process is up, and whether the model loaded
- GET  /ready      - readiness: 503 until the ensemble is usable
- GET  /properties - the property catalog, with units and directions
- POST /predict    - properties for a batch of SMILES

``/health`` and ``/ready`` are deliberately different answers. During the seconds
the ensemble takes to load, the process is perfectly alive and must not be
restarted, but must not be sent traffic either -- a container orchestrator needs
both facts separately.
"""

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .model import ModelUnavailableError, get_predictor
from .properties import CORE_PANEL, PROPERTIES, describe, resolve_names

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Every request runs the whole batch through all ten Chemprop checkpoints, so
# large batches amortize the fixed cost nicely -- but an unbounded one turns a
# single request into a long stall behind which every other caller queues.
MAX_BATCH = int(os.environ.get("ADMET_MAX_BATCH", 256))


class HealthCheckFilter(logging.Filter):
    """Keep health check heartbeats out of the log unless debugging."""

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

# Whether to build the ensemble during startup. On by default: the first request
# then pays nothing, and a bad weights mount is visible in the container log at
# boot rather than on whichever request happens to be first.
EAGER_LOAD = os.environ.get("ADMET_EAGER_LOAD", "true").lower() in (
    "true",
    "1",
    "yes",
)

app = FastAPI(
    title="ADMET Service",
    description="ADMET property prediction using the ADMET-AI ensemble",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Request model for ADMET prediction."""

    smiles: list[str] = Field(
        ..., description="SMILES strings to score. Order is preserved."
    )
    properties: Optional[list[str]] = Field(
        default=None,
        description="Property names or aliases to return. Omit for the "
        "eight-property core panel; pass ['all'] for everything the "
        "ensemble produces.",
    )


class MoleculeResult(BaseModel):
    """Predictions for one molecule."""

    smiles: str
    properties: dict[str, float]
    # Where this molecule sits among approved drugs, per property, as a
    # percentile. Present only when the DrugBank reference set is loaded.
    percentiles: dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    """Response model for ADMET prediction."""

    success: bool
    results: list[MoleculeResult] = Field(default_factory=list)
    # SMILES that RDKit could not parse. Reported rather than silently dropped:
    # a caller scoring 30 candidates needs to know which one never ran.
    invalid_smiles: list[str] = Field(default_factory=list)
    # Metadata for every property present in the results, so a caller can
    # render units and directions without its own copy of the catalog.
    property_info: list[dict[str, Any]] = Field(default_factory=list)
    unknown_properties: list[str] = Field(default_factory=list)
    error: Optional[str] = None


@app.on_event("startup")
async def warm_model() -> None:
    """Load the ensemble at boot unless told otherwise."""
    if not EAGER_LOAD:
        logger.info("ADMET_EAGER_LOAD is off; the ensemble loads on first use")
        return
    try:
        get_predictor().ensure_loaded()
    except ModelUnavailableError as exc:
        # Deliberately not fatal: /health reports the reason, which is more use
        # than a container that exits before anyone can ask it anything.
        logger.error("Startup model load failed: %s", exc)


@app.get("/health")
async def health():
    """Health check, including model load state."""
    predictor = get_predictor()
    return {
        "status": "healthy" if predictor.loaded else "degraded",
        "service": "admet",
        "model_loaded": predictor.loaded,
        "error": predictor.load_error,
    }


@app.get("/ready")
async def ready():
    """Readiness check: 503 while the ensemble is still loading or unusable."""
    predictor = get_predictor()
    if predictor.loaded:
        return {"status": "ready", "service": "admet"}
    return JSONResponse(
        status_code=503,
        content={
            "status": "loading" if predictor.load_error is None else "unavailable",
            "service": "admet",
            "error": predictor.load_error,
        },
    )


@app.get("/properties")
async def list_properties():
    """Return the property catalog and which of it forms the core panel."""
    return {
        "core_panel": list(CORE_PANEL),
        "properties": [describe(prop.name) for prop in PROPERTIES],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict ADMET properties for a batch of SMILES."""
    smiles = [entry.strip() for entry in request.smiles if entry and entry.strip()]
    if not smiles:
        return PredictResponse(
            success=False, error="No SMILES strings were provided."
        )
    if len(smiles) > MAX_BATCH:
        return PredictResponse(
            success=False,
            error=(
                f"{len(smiles)} molecules exceeds the {MAX_BATCH} per request "
                "limit. Split the batch."
            ),
        )

    requested = request.properties
    unknown: list[str] = []
    if requested is None:
        keep: Optional[list[str]] = None
    elif any(name.strip().lower() == "all" for name in requested):
        keep = [prop.name for prop in PROPERTIES]
    else:
        keep, unknown = resolve_names(requested)
        if not keep:
            return PredictResponse(
                success=False,
                unknown_properties=unknown,
                error=(
                    "None of the requested properties are known: "
                    + ", ".join(unknown)
                    + ". Call GET /properties for the catalog."
                ),
            )

    logger.info(
        "Predicting ADMET for %d molecules (%s properties)",
        len(smiles),
        len(keep) if keep is not None else "core panel",
    )

    try:
        results, invalid = get_predictor().predict(smiles, keep)
    except ModelUnavailableError as exc:
        return PredictResponse(success=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - the caller gets the reason
        logger.exception("ADMET prediction failed")
        return PredictResponse(
            success=False, error=f"{type(exc).__name__}: {exc}"
        )

    present: list[str] = []
    for result in results:
        for name in result["properties"]:
            if name not in present:
                present.append(name)

    logger.info(
        "Scored %d molecules (%d invalid) on %d properties",
        len(results),
        len(invalid),
        len(present),
    )

    return PredictResponse(
        success=True,
        results=[MoleculeResult(**result) for result in results],
        invalid_smiles=invalid,
        property_info=[describe(name) for name in present],
        unknown_properties=unknown,
    )
