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

"""Canonical tool-invocation event shape shared across production, training,
and eval.

Part of the P0 architecture-contracts work (see
``internal_docs/docs/0001-architecture-boundaries.md``). Before this module,
``ToolRegistry._execute_impl`` set OTel span attributes (``tool.name``,
``tool.result.content_type``, ``tool.budget.*``, and now ``tool.job_id`` /
``tool.policy.decision`` from the P0 job/policy work) ad hoc, and nothing
outside of Phoenix could read them without re-deriving the same subset. This
module names that shape once so production, training, and eval can agree on
field names without duplicating the derivation logic.

``ToolRegistry._execute_impl`` builds one of these per call and attaches it to
``ToolResult.invocation_event`` (see ``api.tools.base.ToolResult``) in
addition to -- not instead of -- setting the OTel span attributes it already
set. This is additive: nothing that reads ``ToolResult`` today is affected by
the new field, and span attribute names/values are unchanged.

Known P0 gap (documented, not fixed here): this event is available on the
``ToolResult`` returned by ``ToolRegistry.execute()`` and on the OTel span
when one exists, but is not yet threaded through the SSE ``tool_call_end``
event that ``evals/aevom_eval/driver/trajectory.py`` records, so
eval/training trajectory JSON does not carry ``job_id``/``policy_decision``
yet. The underlying data is durably queryable in the meantime via the
``jobs`` table (job_id) and Phoenix spans (policy_decision) directly. See
ADR 0001's "Future phases".
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolInvocationEvent:
    """The canonical shape of one tool call, for production/training/eval."""

    tool_name: str
    arguments: dict[str, Any]
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    content_type: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    artifact_refs: list[str] = field(default_factory=list)
    job_id: Optional[str] = None
    # Terminal state of the tracked Job, when this call ran under a JobManager
    # (e.g. "completed", "failed", "queued_upstream"). The id alone says a job
    # existed; the status is what an eval or a training reward can assert on.
    job_status: Optional[str] = None
    policy_decision: Optional[str] = None
    policy_reason: Optional[str] = None
    budget_state: Optional[dict[str, Any]] = None
    raw_output_ref: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "content_type": self.content_type,
            "success": self.success,
            "error": self.error,
            "artifact_refs": self.artifact_refs,
            "job_id": self.job_id,
            "job_status": self.job_status,
            "policy_decision": self.policy_decision,
            "policy_reason": self.policy_reason,
            "budget_state": self.budget_state,
            "raw_output_ref": self.raw_output_ref,
        }


def build_invocation_event(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    session_id: Optional[str],
    request_id: Optional[str],
    result: Any,
    job_id: Optional[str],
    job_status: Optional[str] = None,
    policy_decision: Optional[str],
    policy_reason: Optional[str],
    budget_state: Optional[dict[str, Any]] = None,
    raw_output_ref: Optional[str] = None,
) -> ToolInvocationEvent:
    """Build the canonical event for one completed tool call.

    ``result`` is a ``ToolResult`` (typed loosely here to avoid importing
    ``api.tools.base`` and creating a cycle, since ``base.py`` is imported by
    nearly everything under ``api.tools``).
    """
    content_type = getattr(getattr(result, "content_type", None), "value", None)
    success = content_type != "error" if content_type is not None else None
    error = str(result.data) if success is False else None

    artifact_refs: list[str] = []
    if content_type == "data_reference" and getattr(result, "data", None):
        artifact_refs.append(str(result.data))

    return ToolInvocationEvent(
        tool_name=tool_name,
        arguments=arguments,
        session_id=session_id,
        request_id=request_id,
        content_type=content_type,
        success=success,
        error=error,
        artifact_refs=artifact_refs,
        job_id=job_id,
        job_status=job_status,
        policy_decision=policy_decision,
        policy_reason=policy_reason,
        budget_state=budget_state,
        raw_output_ref=raw_output_ref,
    )
