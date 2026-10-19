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

"""Project the driver's record into Harbor's ATIF trajectory document.

ATIF is not an export format the harness chose. Harbor's viewer reads
``<trial>/agent/trajectory.json`` and assumes ATIF, so it is the contract of the
path the agent already writes to. Writing anything else there breaks the trial
page outright: the viewer reads ``trajectory.agent.model_name`` without guarding
the intermediate, so a document with no ``agent`` key throws a TypeError, while a
*missing* file would have rendered its "No ATIF trajectory found" empty state.

The driver's own record survives verbatim under ATIF's ``extra``, which is where
the schema puts producer-specific data -- every other object in it is
``extra: forbid``. :func:`unwrap` is the inverse, so the verifier and the tasks'
liveness scripts keep reading exactly the record they read before.

Stdlib only, like the driver it reads: this is imported by tests that run in the
plain devcontainer, with no Harbor installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

SCHEMA_VERSION = "ATIF-v1.7"

# The key under `extra` that holds the driver's record. Named rather than bare so
# a reader can tell whose data it is without inferring it from the shape.
EXTRA_KEY = "aevom"


def to_atif(
    trajectory: dict,
    *,
    agent_name: str,
    agent_version: str,
    model_name: str | None = None,
) -> dict:
    """Render one driver trajectory as an ATIF document."""
    model = model_of(trajectory) or model_name
    steps = _steps(trajectory, model)
    usage = trajectory.get("usage") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": trajectory.get("session_id"),
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": usage.get("input_tokens"),
            "total_completion_tokens": usage.get("output_tokens"),
            "total_steps": len(steps),
        },
        "extra": {EXTRA_KEY: trajectory},
    }


def model_of(trajectory: dict) -> str | None:
    """The model that served the turn, as the id ``--model`` would take.

    The driver copies the stack's ``/api/model`` block into ``info``, so
    ``info.model_id`` is what actually answered -- which is the interesting one
    when, as in the nightly, no model was requested and the stack chose.
    """
    info = trajectory.get("info") or {}
    return info.get("model_id") or trajectory.get("model_id")


def unwrap(document: dict | None) -> dict | None:
    """The driver's record, from either an ATIF document or a bare record.

    Bare records are still accepted because they are what every run before this
    conversion produced, and what the recorded fixtures hold.
    """
    if not isinstance(document, dict):
        return None
    nested = (document.get("extra") or {}).get(EXTRA_KEY)
    return nested if isinstance(nested, dict) else document


def _steps(trajectory: dict, model: str | None) -> list[dict]:
    steps: list[dict] = []
    instruction = trajectory.get("instruction")
    if instruction:
        steps.append({"step_id": 1, "source": "user", "message": str(instruction)})

    for turn in _turns(trajectory.get("raw_events") or []):
        steps.append(_agent_step(turn, len(steps) + 1, model))

    # A run whose events were lost still has its answer, and a trajectory with no
    # agent step at all would render as a conversation the model never joined.
    if not any(step["source"] == "agent" for step in steps):
        answer = trajectory.get("final_content") or ""
        reasoning = "".join(trajectory.get("reasoning") or [])
        if answer or reasoning:
            steps.append(
                _agent_step(
                    {"text": [answer], "reasoning": [reasoning], "calls": [], "results": []},
                    len(steps) + 1,
                    model,
                )
            )
    return steps


def _turns(events: list[dict]) -> Iterator[dict]:
    """Split the SSE stream into one turn per LLM call.

    The stream carries no turn delimiter, but the agent loop leaves one: the
    model speaks, calls tools, sees their results, then speaks again. So text
    arriving after a tool result belongs to the next call, not the one that
    asked for it.
    """
    turn = _blank_turn()
    for event in events:
        speaks = "reasoning" in event or "content" in event
        if speaks and turn["results"]:
            yield turn
            turn = _blank_turn()
        if "reasoning" in event:
            turn["reasoning"].append(str(event["reasoning"]))
        if "content" in event:
            turn["text"].append(str(event["content"]))

        start = event.get("tool_call_start")
        if start:
            turn["calls"].append(start)
        end = event.get("tool_call_end")
        if end:
            turn["results"].append(end)

    if any(turn[field] for field in turn):
        yield turn


def _blank_turn() -> dict[str, list]:
    return {"text": [], "reasoning": [], "calls": [], "results": []}


def _agent_step(turn: dict, step_id: int, model: str | None) -> dict:
    reasoning = "".join(turn["reasoning"])
    calls = [_tool_call(call, index) for index, call in enumerate(turn["calls"])]
    results = [_observation_result(result) for result in turn["results"]]
    return {
        "step_id": step_id,
        "source": "agent",
        "model_name": model,
        "message": "".join(turn["text"]),
        "reasoning_content": reasoning or None,
        "tool_calls": calls or None,
        "observation": {"results": results} if results else None,
        "llm_call_count": 1,
    }


def _tool_call(call: dict, index: int) -> dict:
    arguments = call.get("args")
    if not isinstance(arguments, dict):
        # The schema requires an object. A tool that took a bare value keeps it
        # rather than losing it to an empty dict.
        arguments = {} if arguments is None else {"value": arguments}
    return {
        "tool_call_id": str(call.get("id") or f"call-{index + 1}"),
        "function_name": str(call.get("name") or "unknown"),
        "arguments": arguments,
        "extra": {"summary": call["summary"]} if call.get("summary") else None,
    }


def _observation_result(result: dict) -> dict:
    extra: dict[str, Any] = {}
    if result.get("success") is not None:
        extra["success"] = result["success"]
    if result.get("result_ref"):
        extra["result_ref"] = result["result_ref"]
    return {
        "source_call_id": str(result["id"]) if result.get("id") else None,
        "content": result.get("output_preview") or "",
        "extra": extra or None,
    }
