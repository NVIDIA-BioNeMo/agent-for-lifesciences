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

"""Trajectory record for one eval turn.

Both the container-side driver and the host-side verifier read this, so it stays
stdlib-only and JSON-serializable.

Every raw SSE event is kept alongside the digested view. Eval runs are expensive
and the interesting question usually arrives after the run ("did it retry the
failing tool, or give up?"), so a scoring change should not require a re-run.

The SSE vocabulary mirrored here is ``api/graph/streaming.py``: each frame is a
dict keyed by exactly one of ``span_id``, ``reasoning``, ``content``,
``tool_call_start``, ``tool_call_end``, ``tool_call_update``, ``tool_event``,
``storage_update``, ``rate_limit_status``, ``hit_iteration_limit`` or ``usage``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Trajectory:
    def __init__(self, *, instruction: str, session_id: str, model_id: str | None):
        self.instruction = instruction
        self.session_id = session_id
        self.model_id = model_id
        self.info: dict[str, Any] = {}
        self.raw_events: list[dict] = []
        self.tool_calls: list[dict] = []
        self.reasoning: list[str] = []
        self.content_chunks: list[str] = []
        self.storage_updates: list[dict] = []
        self.rate_limit_events: list[Any] = []
        self.bridge_events: list[dict] = []
        self.errors: list[str] = []
        self.hit_iteration_limit = False
        self.span_id: str | None = None
        self.duration_sec: float | None = None
        # Token totals for the turn, sent once as the stream closes. None when
        # the stream died first, which is not the same as a turn that spent
        # nothing.
        self.usage: dict | None = None
        # Artifacts and selections as the API reports them at end of turn.
        self.artifacts: list[dict] = []
        self.selections: Any = None
        self.conversation: dict | None = None

    # -- ingest ------------------------------------------------------------

    def record(self, event: dict) -> None:
        self.raw_events.append(event)

        if "span_id" in event:
            self.span_id = event["span_id"]
        if "reasoning" in event:
            self.reasoning.append(str(event["reasoning"]))
        if "content" in event:
            self.content_chunks.append(str(event["content"]))
        if "hit_iteration_limit" in event:
            self.hit_iteration_limit = True
        if "rate_limit_status" in event:
            self.rate_limit_events.append(event["rate_limit_status"])
        if "storage_update" in event:
            self.storage_updates.append(event["storage_update"])
        if "usage" in event:
            self.usage = event["usage"]

        start = event.get("tool_call_start")
        if start:
            self.tool_calls.append(
                {
                    "id": start.get("id"),
                    "name": start.get("name"),
                    "args": start.get("args"),
                    "summary": start.get("summary"),
                    "iteration": event.get("iteration"),
                    "success": None,
                    "output_preview": None,
                }
            )

        end = event.get("tool_call_end")
        if end:
            call = self._find_call(end.get("id"), end.get("name"))
            if call is None:
                # An end with no start still matters -- it is evidence of a tool
                # that ran, so it is recorded rather than dropped.
                call = {
                    "id": end.get("id"),
                    "name": end.get("name"),
                    "args": None,
                    "summary": None,
                    "iteration": event.get("iteration"),
                }
                self.tool_calls.append(call)
            call["success"] = end.get("success")
            call["output_preview"] = end.get("output_preview")
            call["result_ref"] = end.get("result_ref")
            # Canonical invocation-event fields (ADR 0001). Copied only when the
            # backend actually sent them: a key absent here means the harness
            # could not observe it, which scoring must distinguish from an
            # observed-but-wrong value. See scoring._check_job_and_policy.
            for key in ("job_id", "job_status", "policy_decision", "artifact_refs"):
                if key in end:
                    call[key] = end[key]

        update = event.get("tool_call_update")
        if update:
            call = self._find_call(update.get("id"), None)
            if call is not None:
                # The graph patches structure_query output after the pending op
                # resolves, so this later value is the one the model saw.
                call["output_preview"] = update.get("output_preview")

    def _find_call(self, call_id: str | None, name: str | None) -> dict | None:
        for call in reversed(self.tool_calls):
            if call_id and call.get("id") == call_id:
                return call
        if name:
            for call in reversed(self.tool_calls):
                if call.get("name") == name and call.get("success") is None:
                    return call
        return None

    # -- views -------------------------------------------------------------

    @property
    def final_content(self) -> str:
        return "".join(self.content_chunks)

    @property
    def tool_names(self) -> list[str]:
        return [call["name"] for call in self.tool_calls if call.get("name")]

    @property
    def failed_tool_calls(self) -> list[dict]:
        return [call for call in self.tool_calls if call.get("success") is False]

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "info": self.info,
            "span_id": self.span_id,
            "duration_sec": self.duration_sec,
            "final_content": self.final_content,
            "usage": self.usage,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "hit_iteration_limit": self.hit_iteration_limit,
            "rate_limit_events": self.rate_limit_events,
            "storage_updates": self.storage_updates,
            "bridge_events": self.bridge_events,
            "artifacts": self.artifacts,
            "selections": self.selections,
            "conversation": self.conversation,
            "errors": self.errors,
            "raw_events": self.raw_events,
        }

    def write(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @staticmethod
    def load(path: Path | str) -> dict:
        return json.loads(Path(path).read_text())
