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

"""The record a ``submit_artifact`` call leaves behind.

A submission is a claim about an artifact ("this molecule is my answer"), not a
new artifact, so it is stored as a small JSON artifact pointing at the real one.
That keeps it inside the session-scoped store every other artifact lives in --
no new table, no new lifecycle -- and means it survives for exactly as long as
the thing it points at.

The metadata flag is what makes the claim findable. Both the tool that writes
submissions and the endpoint that reads them import it from here so they cannot
drift apart.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

# Marks an artifact as a submission record rather than a payload. Namespaced
# because artifact metadata is otherwise open to tools and to sandbox code.
SUBMISSION_FLAG = "aevom_submission"


def build_submission_record(
    artifact_ref: str,
    artifact_data_type: str,
    label: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Build the JSON body of a submission record.

    Args:
        artifact_ref: The artifact being designated as the deliverable.
        artifact_data_type: Its storage type, copied so a reader knows the
            format without parsing the reference.
        label: Short human-facing name for the deliverable.
        notes: The agent's own account of why this is the answer.

    Returns:
        A JSON-serializable dict.
    """
    return {
        "artifact_ref": artifact_ref,
        "data_type": artifact_data_type,
        "label": label or "",
        "notes": notes or "",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


def build_submission_metadata(
    artifact_ref: str,
    artifact_data_type: str,
    label: Optional[str] = None,
    notes: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the artifact metadata that makes a submission record findable.

    Label and notes are duplicated here rather than left in the record body so
    that listing submissions is a metadata-only query -- the same reason
    ``list_artifacts`` never loads content.
    """
    metadata: dict[str, Any] = {
        "source": "submit_artifact",
        SUBMISSION_FLAG: True,
        "artifact_ref": artifact_ref,
        "artifact_data_type": artifact_data_type,
    }
    if label:
        metadata["label"] = label
    if notes:
        metadata["notes"] = notes
    if session_id:
        metadata["session_id"] = session_id
    return metadata


def is_submission(metadata: Optional[dict[str, Any]]) -> bool:
    """Whether an artifact's metadata marks it as a submission record."""
    return bool((metadata or {}).get(SUBMISSION_FLAG))
