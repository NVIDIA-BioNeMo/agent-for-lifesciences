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

"""Scientific artifact envelope: structured provenance for stored artifacts.

Part of the P0 architecture-contracts work (see ``internal_docs/docs/0001-architecture-boundaries.md``).
Storage references (`DataArtifact`) already solve location and size for large
scientific outputs; this module adds the minimal structured fields the audit
calls out as missing meaning: what produced an artifact, from what inputs, and
under what units/uncertainty/validation status.

This is deliberately a "core" type: no FastAPI/LangGraph/SQLAlchemy imports.
Producer tools build an `ArtifactProvenance` and pass it to
`DataStorageService.store(..., metadata={"provenance": provenance.to_dict()})`
-- the existing `metadata` parameter, so no tool call site is *required* to
change. Envelope fields are optional and additive; artifacts stored without
them behave exactly as they do today.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Reserved top-level keys inside a `store(metadata=...)` dict that carry
# structured envelope data. A repository implementation lifts these into the
# dedicated `DataArtifact` columns (see `storage/repository.py`) while leaving
# the full metadata dict untouched, so nothing that reads `metadata_json`
# today needs to change.
ENVELOPE_METADATA_KEYS = ("provenance", "units", "uncertainty", "validation")


@dataclass
class ArtifactProvenance:
    """Lineage for a stored scientific artifact.

    All fields are optional -- populate what a given tool actually knows.
    """

    source_tool: Optional[str] = None
    """Name of the tool that produced this artifact (e.g. "convert_structure")."""

    tool_version: Optional[str] = None
    """Version/identifier of the tool or underlying model/container, if known."""

    input_refs: list[str] = field(default_factory=list)
    """Storage references this artifact was derived from, if any."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Tool parameters/arguments material to reproducing this artifact."""

    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize, dropping empty/unset fields to keep JSONB compact."""
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


def extract_envelope_fields(
    metadata: Optional[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pull reserved envelope keys out of a `store(metadata=...)` dict.

    Returns a dict of ``{"provenance": ..., "units": ..., "uncertainty": ...,
    "validation": ...}`` containing only the keys that were actually present
    and non-empty. Used by the storage repositories to populate the dedicated
    `DataArtifact` envelope columns without requiring `DataStorageService.store()`
    to change signature.
    """
    if not metadata:
        return {}
    return {
        key: metadata[key]
        for key in ENVELOPE_METADATA_KEYS
        if metadata.get(key)
    }
