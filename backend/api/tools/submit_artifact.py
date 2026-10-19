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

"""Let the agent say which artifact is its answer.

Without this, anything reading a finished session has to guess the deliverable
-- take the newest artifact of a plausible type, or parse it out of the final
assistant message. Both guesses are wrong in the cases that matter most: a
session that enumerated forty analogues and picked one, or a session whose
closing paragraph discusses three candidates and commits to the second.

The claim is stored as a small JSON artifact in the same session-scoped store
as everything else (see :mod:`api.artifacts.submissions`), so it is readable
through ``GET /api/session/{id}/submissions``, through the ordinary artifact
list, and by the training resource server, which shares this storage layer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from api.artifacts.submissions import (
    build_submission_metadata,
    build_submission_record,
)
from api.artifacts.infer import infer_data_type
from api.storage import InvalidReferenceError

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# An answer written inline is a SMARTS pattern, a residue list or a short JSON
# object. Anything larger is already an artifact, or should be produced by a
# tool that stores one.
MAX_INLINE_CHARS = 100_000


class SubmitArtifactTool(BaseTool):
    """Designate a stored artifact -- or a short inline answer -- as the deliverable.

    Data Flow:
        Input: an existing reference, or literal content to store first
        Output: Stores a submission record, returns confirmation text
    """

    @property
    def name(self) -> str:
        return "submit_artifact"

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    @property
    def summary_template(self) -> str:
        return "Submitting the final answer..."

    def format_summary(self, arguments: dict) -> str:
        ref = arguments.get("artifact_ref")
        if ref:
            return f"Submitting {ref} as the final answer..."
        return "Submitting the final answer..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit_artifact",
                "description": (
                    "Mark one artifact as your final answer to the current "
                    "task. Call this once you have settled on a deliverable -- "
                    "the chosen molecule, the predicted structure, the answer "
                    "document -- so that whoever reads this session knows "
                    "which of the artifacts you produced is the answer, rather "
                    "than having to guess from the most recent one.\n\n"
                    "Pass 'artifact_ref' for something already stored (a "
                    "molecule from generate_3d_coordinates, a complex from "
                    "boltz2). For a short answer that is not yet an artifact "
                    "-- a SMARTS pattern, a list of interaction residues, a "
                    "small JSON object -- pass 'content' instead and it will "
                    "be stored for you.\n\n"
                    "Submitting again replaces the answer, so a later call "
                    "wins if you change your mind. Explain your choice in "
                    "'notes': it is recorded alongside the answer.\n\n"
                    "Before you call this, check the answer you are about to "
                    "submit against every requirement the task stated -- each "
                    "numeric bound, each structural filter, each property that "
                    "had to move or hold. Check the exact thing being "
                    "submitted, not the one you evaluated three steps ago. Use "
                    "molecule_gates if you have it, or RDKit in python_execute "
                    "if you do not: Descriptors and FilterCatalog cover "
                    "molecular properties and the PAINS/Brenk alerts, and "
                    "DataStructs.TanimotoSimilarity over Morgan fingerprints "
                    "covers similarity to a parent. If something fails, fix it "
                    "and submit the fixed molecule. A near-miss on one stated "
                    "bound is the most common way a good answer is rejected, "
                    "and it is the cheapest thing to catch.\n\n"
                    "If the task is judged on something you can predict -- a "
                    "binding affinity, a docked pose, a property -- run that "
                    "prediction on the answer itself. Spending a whole "
                    "prediction budget on candidates and then submitting an "
                    "untested one leaves you with no evidence about the only "
                    "molecule that is graded. Keep a call in reserve for the "
                    "answer, and if the budget is gone, submit the best "
                    "molecule you actually measured rather than a better-looking "
                    "one you did not."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the artifact that is your answer, "
                                "e.g. 'sdf_a1b2c3d4'. Omit only when passing "
                                "'content' instead."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Literal answer text, for answers that are not "
                                "already stored. Ignored when 'artifact_ref' "
                                "is given."
                            ),
                        },
                        "data_type": {
                            "type": "string",
                            "description": (
                                "Storage type for inline 'content' ('json', "
                                "'smiles', 'text', ...). Inferred from the "
                                "content when omitted."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": (
                                "Short name for the deliverable, e.g. "
                                "'optimized analogue 12'."
                            ),
                        },
                        "notes": {
                            "type": "string",
                            "description": (
                                "Why this is the answer -- which constraints it "
                                "satisfies, what it was chosen over."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def _store_inline(
        self, content: str, declared_type: str | None, label: str | None
    ) -> tuple[str, str]:
        """Store an inline answer, returning its reference and type."""
        payload = content.encode("utf-8")
        data_type, _ = infer_data_type(declared_type, None, payload)
        # An inline answer with no recognizable format is prose, and prose is
        # exactly what the 'text' type is for -- unlike the upload endpoint,
        # there is no file here whose type we might be discarding.
        type_value = data_type.value if data_type is not None else "text"

        reference = await self._storage.store(
            data_type=type_value,
            content=payload,
            metadata={"source": "submit_artifact", "inline": True},
            description=label or "Submitted answer",
        )
        return reference, type_value

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Record a submission for the current session."""
        artifact_ref = (arguments.get("artifact_ref") or "").strip()
        content = arguments.get("content")
        label = (arguments.get("label") or "").strip() or None
        notes = (arguments.get("notes") or "").strip() or None

        if self._storage is None:
            return ToolResult.error(
                "Storage service not available, so a submission cannot be "
                "recorded. Report your answer in your reply instead."
            )

        if not artifact_ref and not content:
            return ToolResult.error(
                "Nothing to submit. Pass 'artifact_ref' for a stored artifact, "
                "or 'content' for a short answer written inline."
            )

        if artifact_ref:
            try:
                exists = await self._storage.exists(artifact_ref)
            except InvalidReferenceError:
                exists = False
            if not exists:
                return ToolResult.error(
                    f"No artifact found for '{artifact_ref}', so it cannot be "
                    "submitted. Check the reference with "
                    "search_session_history, or pass the answer as 'content'."
                )
            data_type = self._storage.get_data_type(artifact_ref)
        else:
            if len(content) > MAX_INLINE_CHARS:
                return ToolResult.error(
                    f"Inline content is {len(content)} characters, over the "
                    f"{MAX_INLINE_CHARS} limit. Store it with python_execute "
                    "(aevom.store) and submit the reference instead."
                )
            artifact_ref, data_type = await self._store_inline(
                content, arguments.get("data_type"), label
            )

        record = build_submission_record(artifact_ref, data_type, label, notes)
        record_ref = await self._storage.store(
            data_type="json",
            content=json.dumps(record, indent=2).encode("utf-8"),
            metadata=build_submission_metadata(
                artifact_ref,
                data_type,
                label=label,
                notes=notes,
                session_id=self._session_id,
            ),
            description=f"Submitted answer: {label or artifact_ref}",
        )

        logger.info(
            "Session %s submitted %s (%s) as %s",
            self._session_id,
            artifact_ref,
            data_type,
            record_ref,
        )

        message = (
            f"Submitted '{artifact_ref}' ({data_type}) as the answer"
            + (f": {label}." if label else ".")
        )
        if notes:
            message += f" Recorded rationale: {notes}"

        return ToolResult.text(
            message,
            metadata={
                "submitted_ref": artifact_ref,
                "submitted_data_type": data_type,
                "submission_record_ref": record_ref,
                "label": label,
            },
        )
