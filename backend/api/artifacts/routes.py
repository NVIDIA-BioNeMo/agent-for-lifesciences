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

"""Put files into a session, and read back what the agent submitted.

Until now the only ways into artifact storage were a tool producing something
and the token-scoped ``/api/sandbox/store`` that sandbox code uses. Neither
helps a caller that simply *has* a file: an evaluation harness holding a task's
receptor and reference ligand, or a chemist who wants to hand Aevom a structure
from their own machine. ``POST /api/session/{id}/artifacts`` is that path.

Reads are the mirror image. ``GET /api/session/{id}/submissions`` returns what
``submit_artifact`` designated as the deliverable, so a caller does not have to
infer the answer from the newest artifact or from the wording of the final
assistant message.

Both routes need the session's owner token (see api/auth): the session must
have come from ``POST /api/session/create``, and the caller must hold the token
it was bound to.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.deps import assert_owned_session, get_session_ownership
from api.auth.ownership import SessionOwnership
from api.db.config import get_async_session
from api.storage import scoped_storage
from api.storage.datatypes import DATA_TYPE_INFO, is_text_type

from .infer import infer_data_type
from .submissions import is_submission

logger = logging.getLogger(__name__)

# Ownership is checked separately (require_owned_valid_session). This only
# keeps a session id from carrying anything surprising into a log line or a
# metadata field.
# Deliberately wider than a UUID: training rollouts mint their own ids.
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")

# Ceilings, not tuning knobs. A single PDB entry with its waters is a few MB and
# a Boltz-2 ensemble a few tens; past that a caller is uploading a dataset,
# which belongs on disk with references into it, not in one request body.
MAX_ARTIFACT_BYTES = int(os.getenv("AEVOM_ARTIFACT_MAX_BYTES", 32 * 1024 * 1024))
MAX_REQUEST_BYTES = int(
    os.getenv("AEVOM_ARTIFACT_REQUEST_MAX_BYTES", 128 * 1024 * 1024)
)
MAX_ARTIFACTS_PER_REQUEST = int(os.getenv("AEVOM_ARTIFACT_MAX_COUNT", 64))

_SUBMISSION_SCAN_LIMIT = 200


class ArtifactUpload(BaseModel):
    """One file to place into a session's storage."""

    content: Optional[str] = Field(
        default=None,
        description="UTF-8 text content. Use content_base64 for binary types.",
    )
    content_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded content, required for binary types "
        "such as 'array' and 'png'.",
    )
    data_type: Optional[str] = Field(
        default=None,
        description="Aevom storage type ('sdf', 'pdb', 'mmcif', ...). "
        "Inferred from the filename or the content when omitted.",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Original filename. Used to infer data_type and returned "
        "in the response so a caller can match references to its own files.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short description shown in the storage strip and to the "
        "model when it searches session history.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata to attach to the artifact.",
    )


class SeedArtifactsRequest(BaseModel):
    """A batch of files to place into one session."""

    artifacts: list[ArtifactUpload] = Field(
        ..., description="Files to store. Stored in the order given."
    )


class StoredArtifact(BaseModel):
    """What one stored file became."""

    reference: str
    data_type: str
    size_bytes: int
    filename: Optional[str] = None
    description: Optional[str] = None
    # How data_type was decided: 'declared', 'filename' or 'content'. Surfaced
    # so a caller can tell when it is relying on inference.
    data_type_source: str


class SeedArtifactsResponse(BaseModel):
    """References for everything stored, plus a filename index."""

    session_id: str
    artifacts: list[StoredArtifact]
    # filename -> reference, for callers that track their inputs by name. Only
    # includes uploads that carried a filename.
    references: dict[str, str] = Field(default_factory=dict)


class Submission(BaseModel):
    """A deliverable the agent designated with ``submit_artifact``."""

    artifact_ref: str
    data_type: str
    label: Optional[str] = None
    notes: Optional[str] = None
    submitted_at: Optional[str] = None
    # The submission record itself, distinct from the artifact it points at.
    record_ref: str


class SubmissionsResponse(BaseModel):
    """Submissions for a session, most recent first."""

    session_id: str
    submissions: list[Submission]


router = APIRouter(prefix="/api/session", tags=["artifacts"])


def require_valid_session_id(session_id: str) -> str:
    """Reject a session id that could not have come from this API."""
    if not _SESSION_ID_PATTERN.match(session_id or ""):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid session_id. Expected 8-128 characters of letters, "
                "digits, '.', '_', ':' or '-', such as the UUID returned by "
                "POST /api/session/create."
            ),
        )
    return session_id


async def require_owned_valid_session(
    request: Request,
    session_id: str = Depends(require_valid_session_id),
    ownership: SessionOwnership = Depends(get_session_ownership),
) -> str:
    await assert_owned_session(request, session_id, ownership)
    return session_id


def _decode_payload(upload: ArtifactUpload, index: int) -> bytes:
    """Turn one upload's content field into bytes.

    ``content_base64`` wins when both are present, matching
    ``/api/sandbox/store``: a caller that went to the trouble of encoding meant
    the encoded payload.
    """
    if upload.content_base64 is not None:
        try:
            return base64.b64decode(upload.content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"artifacts[{index}].content_base64 is not valid base64: {exc}",
            )
    if upload.content is not None:
        return upload.content.encode("utf-8")
    raise HTTPException(
        status_code=400,
        detail=(
            f"artifacts[{index}] has neither 'content' nor 'content_base64'. "
            "Provide one."
        ),
    )


def _resolve_type(upload: ArtifactUpload, content: bytes, index: int) -> tuple[str, str]:
    """Decide the storage type for one upload, or explain why we cannot."""
    data_type, source = infer_data_type(upload.data_type, upload.filename, content)

    if data_type is None and source == "declared":
        raise HTTPException(
            status_code=400,
            detail=(
                f"artifacts[{index}].data_type '{upload.data_type}' is not a "
                "known Aevom data type. Valid types: "
                + ", ".join(sorted(dt.value for dt in DATA_TYPE_INFO))
            ),
        )
    if data_type is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not determine a data type for artifacts[{index}]"
                + (f" ('{upload.filename}')" if upload.filename else "")
                + ". Give it a recognized file extension or set data_type "
                "explicitly."
            ),
        )

    # A binary type sent through the text field would have been mangled by the
    # caller's own JSON encoder before it ever reached us, so refuse rather
    # than store bytes that cannot be what the caller meant.
    if not is_text_type(data_type.value) and upload.content_base64 is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"artifacts[{index}] is type '{data_type.value}', which is "
                "binary. Send it in 'content_base64' rather than 'content'."
            ),
        )

    return data_type.value, source


def _check_sizes(payloads: list[bytes]) -> None:
    """Enforce the per-artifact and per-request ceilings."""
    for index, payload in enumerate(payloads):
        if not payload:
            raise HTTPException(
                status_code=400, detail=f"artifacts[{index}] is empty."
            )
        if len(payload) > MAX_ARTIFACT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"artifacts[{index}] is {len(payload)} bytes, over the "
                    f"{MAX_ARTIFACT_BYTES} byte per-artifact limit."
                ),
            )
    total = sum(len(payload) for payload in payloads)
    if total > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The request carries {total} bytes, over the "
                f"{MAX_REQUEST_BYTES} byte limit. Split it across several "
                "requests."
            ),
        )


@router.post("/{session_id}/artifacts", response_model=SeedArtifactsResponse)
async def create_session_artifacts(
    request: SeedArtifactsRequest,
    session_id: str = Depends(require_owned_valid_session),
    session: AsyncSession = Depends(get_async_session),
) -> SeedArtifactsResponse:
    """Store files into a session and return their references.

    The whole batch is validated before anything is written, so a request with
    one bad entry stores nothing and the caller does not have to reason about a
    partially seeded session.
    """
    uploads = request.artifacts
    if not uploads:
        raise HTTPException(
            status_code=400, detail="'artifacts' must contain at least one entry."
        )
    if len(uploads) > MAX_ARTIFACTS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{len(uploads)} artifacts exceeds the "
                f"{MAX_ARTIFACTS_PER_REQUEST} per request limit."
            ),
        )

    payloads = [_decode_payload(upload, index) for index, upload in enumerate(uploads)]
    _check_sizes(payloads)
    resolved = [
        _resolve_type(upload, payload, index)
        for index, (upload, payload) in enumerate(zip(uploads, payloads))
    ]

    storage = scoped_storage(session, session_id)
    stored: list[StoredArtifact] = []

    for upload, payload, (data_type, source) in zip(uploads, payloads, resolved):
        metadata = dict(upload.metadata)
        metadata.setdefault("source", "session_artifacts_api")
        if upload.filename:
            metadata.setdefault("filename", upload.filename)

        description = upload.description or (
            f"Uploaded {upload.filename}" if upload.filename else None
        )

        reference = await storage.store(
            data_type=data_type,
            content=payload,
            metadata=metadata,
            description=description,
        )
        stored.append(
            StoredArtifact(
                reference=reference,
                data_type=data_type,
                size_bytes=len(payload),
                filename=upload.filename,
                description=description,
                data_type_source=source,
            )
        )

    await session.commit()

    logger.info(
        "Seeded %d artifacts into session %s: %s",
        len(stored),
        session_id,
        ", ".join(f"{item.filename or '?'}->{item.reference}" for item in stored),
    )

    return SeedArtifactsResponse(
        session_id=session_id,
        artifacts=stored,
        references={
            item.filename: item.reference for item in stored if item.filename
        },
    )


@router.get("/{session_id}/submissions", response_model=SubmissionsResponse)
async def list_session_submissions(
    session_id: str = Depends(require_owned_valid_session),
    session: AsyncSession = Depends(get_async_session),
) -> SubmissionsResponse:
    """Return the artifacts the agent designated as deliverables.

    Newest first, because a task that was answered twice was answered the
    second time. Content is not inlined: fetch it with
    ``GET /api/storage/artifact/{reference}``.
    """
    storage = scoped_storage(session, session_id)
    artifacts = await storage.list_artifacts(limit=_SUBMISSION_SCAN_LIMIT)

    submissions: list[Submission] = []
    for artifact in artifacts:
        metadata = artifact.get("metadata") or {}
        if not is_submission(metadata):
            continue
        artifact_ref = metadata.get("artifact_ref")
        if not artifact_ref:
            continue
        submissions.append(
            Submission(
                artifact_ref=artifact_ref,
                data_type=metadata.get("artifact_data_type") or "",
                label=metadata.get("label") or artifact.get("description"),
                notes=metadata.get("notes"),
                submitted_at=artifact.get("created_at"),
                record_ref=artifact["reference"],
            )
        )

    return SubmissionsResponse(session_id=session_id, submissions=submissions)
