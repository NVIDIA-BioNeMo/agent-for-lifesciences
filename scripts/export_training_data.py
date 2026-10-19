#!/usr/bin/env python3
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

"""Export DPO training data from Phoenix traces.

This script queries Phoenix for training spans, matches them with feedback
annotations, and exports paired (chosen/rejected) responses in a model-agnostic
messages format suitable for NeMo RL training.

The output format uses a "messages" array instead of pre-templated prompts,
allowing the same training data to be used for any model by applying chat
templates at training time.

Usage:
    # Export all complete pairs
    python3 scripts/export_training_data.py

    # Export to specific file
    python3 scripts/export_training_data.py --output training_data.jsonl

    # Use production Phoenix (via port-forward)
    python3 scripts/export_training_data.py --prod

    # Preview without writing (dry run)
    python3 scripts/export_training_data.py --dry-run

    # Filter by minimum training version
    python3 scripts/export_training_data.py --min-version 0.1.0
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Phoenix configuration
AEVOM_PROJECT_ID = "UHJvamVjdDoy"


def parse_version(version_str: str) -> tuple:
    """Parse a semantic version string into a comparable tuple.

    Handles versions like "0.1.0", "v0.1.0", "1.2.3-dev", etc.
    Returns a tuple of integers for comparison (e.g., (0, 1, 0)).

    Args:
        version_str: Version string to parse.

    Returns:
        Tuple of version components for comparison.
    """
    if not version_str:
        return (0, 0, 0)

    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")

    # Remove any suffix after hyphen (e.g., "0.0.0-dev" -> "0.0.0")
    version_str = version_str.split("-")[0]

    # Split by dots and convert to integers
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)

    # Pad to at least 3 components
    while len(parts) < 3:
        parts.append(0)

    return tuple(parts)


def version_gte(version: str, min_version: str) -> bool:
    """Check if version is greater than or equal to min_version.

    Args:
        version: Version to check.
        min_version: Minimum required version.

    Returns:
        True if version >= min_version.
    """
    return parse_version(version) >= parse_version(min_version)


@dataclass
class ToolCall:
    """A tool call with its result."""

    name: str
    arguments: dict  # Parsed for convenience
    raw_arguments: str  # Original JSON string from LLM
    result: str
    result_type: str  # text, data_reference, pdb_3d, etc.


@dataclass
class ReasoningStep:
    """A single reasoning step (LLM call + optional tool call)."""

    reasoning_content: Optional[str]  # Model's thinking/reasoning
    tool_call: Optional[ToolCall]  # Tool called, if any
    response_content: Optional[str]  # Final response (only on last step)


@dataclass
class TrainingResponse:
    """A single training response from a span with full reasoning trace."""

    span_id: str
    query_id: str
    query_text: str
    mode: str  # oracle, weak, production
    system_prompt: str  # The full effective system prompt
    messages_json: Optional[
        str
    ]  # Raw messages array from chat.messages_json (model-agnostic)
    reasoning_trace: list[ReasoningStep]  # Full chain of reasoning and tool calls
    final_response: str  # The final assistant response
    feedback: Optional[str]  # thumbs_up, thumbs_down, or None
    timestamp: str


@dataclass
class TrainingPair:
    """A complete training pair with chosen and rejected responses."""

    query_id: str
    query_text: str
    chosen: TrainingResponse
    rejected: TrainingResponse


def get_phoenix_url(use_prod: bool = False) -> str:
    """Get Phoenix GraphQL endpoint URL."""
    if use_prod:
        return "http://localhost:6006/graphql"

    # Check for explicit URL
    env_url = os.environ.get("PHOENIX_URL")
    if env_url:
        return f"{env_url.rstrip('/')}/graphql"

    # Try Docker container discovery
    try:
        import subprocess

        result = subprocess.run(
            [
                "docker",
                "inspect",
                "aevom-phoenix-1",
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"http://{result.stdout.strip()}:6006/graphql"
    except Exception:
        pass

    return "http://localhost:6006/graphql"


def query_phoenix(query: str, url: str) -> dict:
    """Execute a GraphQL query against Phoenix."""
    payload = json.dumps({"query": query}).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error connecting to Phoenix at {url}: {e}", file=sys.stderr)
        return {"data": None, "error": str(e)}


def get_training_spans(url: str) -> list[dict]:
    """Get all spans with training metadata from Phoenix.

    Fetches chat_request spans that have training metadata (query_id),
    including their descendants to extract the full reasoning trace.
    """
    print("Fetching training spans from Phoenix...")

    # Query spans with descendants in a single request
    # Include metadata in spanAnnotations to get version_number
    query = f'''{{
        node(id: "{AEVOM_PROJECT_ID}") {{
            ... on Project {{
                spans(
                    first: 5000,
                    sort: {{col: startTime, dir: desc}}
                ) {{
                    edges {{
                        node {{
                            spanId
                            name
                            startTime
                            attributes
                            spanAnnotations {{
                                name
                                label
                                metadata
                            }}
                            descendants(first: 100) {{
                                edges {{
                                    node {{
                                        spanId
                                        name
                                        attributes
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}'''

    result = query_phoenix(query, url)

    if not result.get("data") or not result["data"].get("node"):
        print("No data returned from Phoenix", file=sys.stderr)
        return []

    spans = []
    for edge in result["data"]["node"]["spans"]["edges"]:
        span = edge["node"]

        # Parse attributes
        attrs = {}
        if span.get("attributes"):
            try:
                attrs = (
                    json.loads(span["attributes"])
                    if isinstance(span["attributes"], str)
                    else span["attributes"]
                )
            except json.JSONDecodeError:
                continue

        # Filter for chat_request spans with training metadata
        if span["name"] != "chat_request":
            continue

        training = attrs.get("training", {})
        if not training.get("query_id"):
            continue

        # Get feedback annotation if present (with version_number from metadata)
        feedback = None
        version_number = None
        for ann in span.get("spanAnnotations", []):
            if ann.get("name") == "user_feedback":
                feedback = ann.get("label")
                # Extract version from annotation metadata
                # Try version_number first, fall back to training_version for old data
                ann_metadata = ann.get("metadata")
                if ann_metadata:
                    if isinstance(ann_metadata, str):
                        try:
                            ann_metadata = json.loads(ann_metadata)
                        except json.JSONDecodeError:
                            ann_metadata = {}
                    version_number = ann_metadata.get(
                        "version_number"
                    ) or ann_metadata.get("training_version")
                break

        # Get chat and prompt attributes
        chat = attrs.get("chat", {})
        prompt = attrs.get("prompt", {})

        # Extract full reasoning trace from descendants
        descendants = span.get("descendants", {}).get("edges", [])
        reasoning_trace, final_response, has_raw_data = extract_reasoning_trace(
            descendants
        )

        spans.append(
            {
                "span_id": span["spanId"],
                "query_id": training.get("query_id"),
                "query_text": chat.get("user_message", "")
                or chat.get("user_message_preview", ""),  # Prefer full message
                "mode": training.get("mode", "production"),
                "prompt_type": prompt.get("type", "production"),
                "system_prompt": prompt.get(
                    "effective", ""
                ),  # Full effective system prompt
                "messages_json": chat.get(
                    "messages_json"
                ),  # Raw messages array (model-agnostic)
                "expected_tool": training.get("expected_tool"),
                "timestamp": span["startTime"],
                "feedback": feedback,
                "version_number": version_number,  # Version from feedback annotation
                "reasoning_trace": reasoning_trace,
                "final_response": final_response,
                "has_raw_data": has_raw_data,  # True only if all spans have raw attributes
            }
        )

    print(f"Found {len(spans)} training spans")
    return spans


def extract_reasoning_trace(
    descendants: list[dict],
) -> tuple[list[ReasoningStep], str, bool]:
    """Extract the full reasoning trace from descendant spans.

    Reconstructs the chain of reasoning steps including:
    - Model reasoning/thinking
    - Tool calls and their results
    - Final response

    NOTE (2026-05-27): the live agent-loop span tree introduced by
    `agent_loop_spans` does NOT carry `llm.raw.reasoning` /
    `llm.raw.tool_calls` -- the synthetic `llm.complete` spans this
    extractor depends on are no longer emitted. Runs from after that
    refactor will yield empty reasoning_steps here unless DPO-export
    full-migration to the live tree is done as separate follow-up work
    (see plans/profile-agent-loop-3n; the plan explicitly defers this).
    The chat.* root attributes that the rest of the export pipeline
    consumes (chat.user_message, chat.final_response, training.reward,
    etc.) are unchanged, so basic SFT-style export still works on the
    new runs; DPO pair generation that needs per-step reasoning will
    not.

    Args:
        descendants: List of descendant span edges from GraphQL query.

    Returns:
        Tuple of (reasoning_steps, final_response, has_raw_data)
        has_raw_data is True only if ALL llm/tool spans have raw attributes.
    """
    # Parse all descendants into a more usable format
    parsed_spans = []
    for edge in descendants:
        child = edge["node"]
        attrs = {}
        if child.get("attributes"):
            try:
                attrs = (
                    json.loads(child["attributes"])
                    if isinstance(child["attributes"], str)
                    else child["attributes"]
                )
            except json.JSONDecodeError:
                continue
        parsed_spans.append(
            {
                "name": child["name"],
                "span_id": child["spanId"],
                "attrs": attrs,
            }
        )

    # Live-tree detection: warn once per call if the new span tree is
    # present without legacy llm.complete children so downstream callers
    # know reasoning_steps will be empty.
    if any(s["name"] == "policy_llm.responses" for s in parsed_spans) and not any(
        s["name"] == "llm.complete" for s in parsed_spans
    ):
        # Best-effort -- not a hard error; the pair-extraction guards
        # already skip rollouts with empty reasoning_steps.
        pass

    # Group by iteration and build reasoning steps
    # The pattern is: llm.complete (reasoning + tool call) -> tool.* (result) -> repeat -> final llm.complete (response)
    reasoning_steps = []
    current_step = None
    final_response = ""

    # Track raw tool calls from LLM spans to match with tool results
    pending_raw_tool_calls = {}  # tool_name -> raw_arguments JSON string

    # Track whether we have raw data for all spans
    llm_spans_total = 0
    llm_spans_with_raw = 0
    tool_spans_total = 0
    tool_spans_with_raw = 0

    for span in parsed_spans:
        name = span["name"]
        attrs = span["attrs"]

        if name.startswith("llm."):
            llm_spans_total += 1
            llm = attrs.get("llm", {})
            raw = llm.get("raw", {})

            # Check if this span has raw data
            if (
                raw.get("content") is not None
                or raw.get("reasoning") is not None
                or raw.get("tool_calls") is not None
            ):
                llm_spans_with_raw += 1

            # Use raw attributes (primary source, no duplicates stored)
            # Note: llm.reasoning_content and llm.response_content were removed as duplicates
            reasoning = raw.get("reasoning", "")
            response = raw.get("content", "")
            raw_tool_calls_json = raw.get("tool_calls")  # JSON string of tool calls
            tool_calls_names = llm.get(
                "tool_calls", ""
            )  # Comma-separated names for quick lookup

            # Parse raw tool calls if available
            if raw_tool_calls_json:
                try:
                    raw_tool_calls_list = json.loads(raw_tool_calls_json)
                    for tc in raw_tool_calls_list:
                        tc_name = tc.get("function", {}).get("name", "")
                        tc_args_str = tc.get("function", {}).get("arguments", "{}")
                        if tc_name:
                            pending_raw_tool_calls[tc_name] = tc_args_str
                except json.JSONDecodeError:
                    pass

            # If this LLM call has a tool call, start a new reasoning step
            has_tool_calls = raw_tool_calls_json or tool_calls_names
            if has_tool_calls:
                current_step = ReasoningStep(
                    reasoning_content=reasoning if reasoning else None,
                    tool_call=None,  # Will be filled when we see the tool span
                    response_content=None,
                )
                # Store the tool name for matching
                if raw_tool_calls_json:
                    try:
                        raw_list = json.loads(raw_tool_calls_json)
                        if raw_list:
                            current_step._pending_tool = (
                                raw_list[0].get("function", {}).get("name", "")
                            )
                    except json.JSONDecodeError:
                        current_step._pending_tool = (
                            tool_calls_names.split(",")[0]
                            if "," in tool_calls_names
                            else tool_calls_names
                        )
                else:
                    current_step._pending_tool = (
                        tool_calls_names.split(",")[0]
                        if "," in tool_calls_names
                        else tool_calls_names
                    )
                reasoning_steps.append(current_step)
            elif response:
                # Final response (no tool call)
                final_response = response
                # Add a final step if there was reasoning
                if reasoning:
                    reasoning_steps.append(
                        ReasoningStep(
                            reasoning_content=reasoning,
                            tool_call=None,
                            response_content=response,
                        )
                    )
            elif reasoning and not has_tool_calls:
                # Reasoning without tool call or response (intermediate thinking)
                reasoning_steps.append(
                    ReasoningStep(
                        reasoning_content=reasoning,
                        tool_call=None,
                        response_content=None,
                    )
                )

        elif name.startswith("tool."):
            tool_spans_total += 1
            tool_attrs = attrs.get("tool", {})
            raw = tool_attrs.get("raw", {})

            # Check if this span has raw data
            if raw.get("output") is not None or raw.get("arguments") is not None:
                tool_spans_with_raw += 1

            tool_name = tool_attrs.get("name", name.replace("tool.", ""))

            # Use raw attributes (primary source, no duplicates stored)
            # Note: tool.result.output was removed as a duplicate of tool.raw.output
            result_output = raw.get("output", "")
            result_type = tool_attrs.get("result", {}).get("content_type", "text")

            # Get raw arguments - prefer from raw attr, then from pending LLM tool calls
            raw_args_str = raw.get("arguments") or pending_raw_tool_calls.get(
                tool_name, "{}"
            )
            try:
                tool_args = json.loads(raw_args_str) if raw_args_str else {}
            except json.JSONDecodeError:
                tool_args = tool_attrs.get("arg", {})
                raw_args_str = json.dumps(tool_args)

            # Find the matching reasoning step and add the tool call
            for step in reversed(reasoning_steps):
                if hasattr(step, "_pending_tool") and step._pending_tool == tool_name:
                    step.tool_call = ToolCall(
                        name=tool_name,
                        arguments=tool_args,
                        raw_arguments=raw_args_str,
                        result=result_output,
                        result_type=result_type,
                    )
                    delattr(step, "_pending_tool")
                    break
            else:
                # No matching step found, create a new one for orphan tool
                reasoning_steps.append(
                    ReasoningStep(
                        reasoning_content=None,
                        tool_call=ToolCall(
                            name=tool_name,
                            arguments=tool_args,
                            raw_arguments=raw_args_str,
                            result=result_output,
                            result_type=result_type,
                        ),
                        response_content=None,
                    )
                )

    # Clean up any remaining _pending_tool attributes
    for step in reasoning_steps:
        if hasattr(step, "_pending_tool"):
            delattr(step, "_pending_tool")

    # Determine if we have complete raw data
    # Raw data is required if there are any llm/tool spans
    has_raw_data = True
    if llm_spans_total > 0 and llm_spans_with_raw < llm_spans_total:
        has_raw_data = False
    if tool_spans_total > 0 and tool_spans_with_raw < tool_spans_total:
        has_raw_data = False
    # If no descendant spans at all, we can't verify raw data
    if llm_spans_total == 0:
        has_raw_data = False

    return reasoning_steps, final_response, has_raw_data


def group_by_query(spans: list[dict]) -> dict[str, list[dict]]:
    """Group spans by query_id."""
    groups = defaultdict(list)
    for span in spans:
        groups[span["query_id"]].append(span)
    return dict(groups)


def find_pairs(
    spans_by_query: dict[str, list[dict]],
    require_raw_data: bool = True,
    data_augmentation: bool = False,
    min_version: Optional[str] = None,
) -> list[TrainingPair]:
    """Find complete training pairs (one chosen, one rejected).

    A complete pair requires:
    - At least one span with thumbs_up feedback (chosen) that has raw data
    - At least one span with thumbs_down feedback (rejected) that has raw data

    The system prompt from the CHOSEN response is used as the training prompt.

    Args:
        spans_by_query: Spans grouped by query_id.
        require_raw_data: If True (default), only include pairs where both
            chosen and rejected have complete raw OTEL data (llm.raw.*, tool.raw.*).
        data_augmentation: If True, produce all unique (chosen, rejected) pairs
            via Cartesian product. If False (default), only use the first of each.
        min_version: If provided, only include spans with version_number >= min_version.
    """
    pairs = []

    for query_id, spans in spans_by_query.items():
        # Filter for spans with raw data if required
        if require_raw_data:
            chosen_spans = [
                s
                for s in spans
                if s["feedback"] == "thumbs_up" and s.get("has_raw_data", False)
            ]
            rejected_spans = [
                s
                for s in spans
                if s["feedback"] == "thumbs_down" and s.get("has_raw_data", False)
            ]
        else:
            chosen_spans = [s for s in spans if s["feedback"] == "thumbs_up"]
            rejected_spans = [s for s in spans if s["feedback"] == "thumbs_down"]

        # Filter by minimum version if specified
        if min_version:
            chosen_spans = [
                s
                for s in chosen_spans
                if s.get("version_number")
                and version_gte(s["version_number"], min_version)
            ]
            rejected_spans = [
                s
                for s in rejected_spans
                if s.get("version_number")
                and version_gte(s["version_number"], min_version)
            ]

        if not chosen_spans or not rejected_spans:
            continue

        # Sort by timestamp for deterministic ordering
        chosen_spans.sort(key=lambda s: s["timestamp"])
        rejected_spans.sort(key=lambda s: s["timestamp"])

        if data_augmentation:
            # Cartesian product: all unique (chosen, rejected) combinations
            pairs_to_add = [
                (chosen, rejected)
                for chosen in chosen_spans
                for rejected in rejected_spans
            ]
        else:
            # Default: only use the first of each
            pairs_to_add = [(chosen_spans[0], rejected_spans[0])]

        for chosen, rejected in pairs_to_add:
            # Get query text from chosen span (preferred)
            query_text = chosen.get("query_text") or rejected.get("query_text") or ""

            pairs.append(
                TrainingPair(
                    query_id=query_id,
                    query_text=query_text,
                    chosen=TrainingResponse(
                        span_id=chosen["span_id"],
                        query_id=query_id,
                        query_text=query_text,
                        mode=chosen["mode"],
                        system_prompt=chosen.get("system_prompt", ""),
                        messages_json=chosen.get("messages_json"),
                        reasoning_trace=chosen.get("reasoning_trace", []),
                        final_response=chosen.get("final_response", ""),
                        feedback="thumbs_up",
                        timestamp=chosen["timestamp"],
                    ),
                    rejected=TrainingResponse(
                        span_id=rejected["span_id"],
                        query_id=query_id,
                        query_text=query_text,
                        mode=rejected["mode"],
                        system_prompt=rejected.get("system_prompt", ""),
                        messages_json=rejected.get("messages_json"),
                        reasoning_trace=rejected.get("reasoning_trace", []),
                        final_response=rejected.get("final_response", ""),
                        feedback="thumbs_down",
                        timestamp=rejected["timestamp"],
                    ),
                )
            )

    return pairs


def format_reasoning_trace(response: TrainingResponse) -> str:
    """Format the full reasoning trace including tool calls.

    Outputs the raw OpenAI function calling format exactly as received from the LLM:
    - <think>reasoning</think> (model's reasoning/thinking)
    - {"function_call": {"name": "...", "arguments": "..."}} (raw tool call)
    - [Tool Result] result text
    - Final response

    Args:
        response: TrainingResponse with reasoning trace.

    Returns:
        Formatted string of the full response including reasoning and tool calls.
    """
    parts = []

    for step in response.reasoning_trace:
        # Add reasoning/thinking if present
        if step.reasoning_content:
            parts.append(f"<think>\n{step.reasoning_content}\n</think>")

        # Add tool call if present - output in raw OpenAI function calling format
        if step.tool_call:
            tc = step.tool_call
            # Output the raw format as-is (preserves original LLM output)
            function_call = {
                "function_call": {
                    "name": tc.name,
                    "arguments": tc.raw_arguments,  # Original JSON string from LLM
                }
            }
            parts.append(json.dumps(function_call))

            # Add tool result
            parts.append(f"[Tool Result]\n{tc.result}")

        # Add response content if this step has it (usually the final step)
        if step.response_content:
            parts.append(step.response_content)

    # If we have a final response that wasn't captured in steps
    if response.final_response and not any(
        s.response_content for s in response.reasoning_trace
    ):
        parts.append(response.final_response)

    return "\n\n".join(parts) if parts else response.final_response


def export_pairs(
    pairs: list[TrainingPair],
    output_path: Path,
    dry_run: bool = False,
    include_system_prompt: bool = False,
) -> None:
    """Export training pairs to JSONL file in model-agnostic messages format.

    Uses the messages array from the CHOSEN response (stored as chat.messages_json).
    Falls back to reconstructing from system_prompt + query_text for legacy data.

    Args:
        pairs: List of training pairs.
        output_path: Output file path.
        dry_run: If True, only print what would be written.
        include_system_prompt: If True, include system messages. Default False
            to reduce sequence length for training.
    """
    entries = []
    for pair in pairs:
        # Get raw messages from span (stored as chat.messages_json)
        # This is model-agnostic - chat templates are applied at training time
        messages_json = pair.chosen.messages_json
        if messages_json:
            try:
                messages = json.loads(messages_json)
            except json.JSONDecodeError:
                messages = None
        else:
            messages = None

        # Fallback for legacy spans without messages_json
        if not messages:
            if include_system_prompt:
                system_prompt = pair.chosen.system_prompt
                if not system_prompt:
                    system_prompt = "You are Aevom, an AI Computational Chemist. Use available tools to answer queries accurately."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pair.query_text},
                ]
            else:
                messages = [
                    {"role": "user", "content": pair.query_text},
                ]
        elif not include_system_prompt:
            # Filter out system messages from the messages array
            messages = [m for m in messages if m.get("role") != "system"]

        # Format full reasoning traces for both responses
        chosen_response = format_reasoning_trace(pair.chosen)
        rejected_response = format_reasoning_trace(pair.rejected)

        entry = {
            "messages": messages,  # Model-agnostic messages array
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
            # Metadata for debugging/filtering
            "metadata": {
                "query_id": pair.query_id,
                "chosen_mode": pair.chosen.mode,
                "rejected_mode": pair.rejected.mode,
                "chosen_span_id": pair.chosen.span_id,
                "rejected_span_id": pair.rejected.span_id,
                "chosen_steps": len(pair.chosen.reasoning_trace),
                "rejected_steps": len(pair.rejected.reasoning_trace),
            },
        }
        entries.append(entry)

    if dry_run:
        print(f"\nDry run - would write {len(entries)} entries to {output_path}")
        if entries:
            print("\nSample entry:")
            print(json.dumps(entries[0], indent=2))
        return

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nExported {len(entries)} training pairs to {output_path}")


def print_summary(
    spans_by_query: dict[str, list[dict]],
    pairs: list[TrainingPair],
    data_augmentation: bool = False,
) -> None:
    """Print a summary of the training data."""
    total_queries = len(spans_by_query)
    total_spans = sum(len(s) for s in spans_by_query.values())

    # Count spans by raw data status
    spans_with_raw = sum(
        1
        for spans in spans_by_query.values()
        for s in spans
        if s.get("has_raw_data", False)
    )
    spans_without_raw = total_spans - spans_with_raw

    # Count by feedback status (ignoring raw data)
    complete_any = 0  # Has both feedback types (any raw status)
    chosen_only = 0
    rejected_only = 0
    no_feedback = 0

    # Count by feedback status (with raw data requirement)
    complete_with_raw = 0  # Has both feedback types with raw data

    # Calculate augmentation potential (Cartesian product counts)
    augmented_pairs_any = 0  # Total pairs with data augmentation (any raw status)
    augmented_pairs_raw = 0  # Total pairs with data augmentation (raw data only)

    for query_id, spans in spans_by_query.items():
        chosen_any = [s for s in spans if s["feedback"] == "thumbs_up"]
        rejected_any = [s for s in spans if s["feedback"] == "thumbs_down"]
        chosen_raw = [
            s
            for s in spans
            if s["feedback"] == "thumbs_up" and s.get("has_raw_data", False)
        ]
        rejected_raw = [
            s
            for s in spans
            if s["feedback"] == "thumbs_down" and s.get("has_raw_data", False)
        ]

        has_chosen = len(chosen_any) > 0
        has_rejected = len(rejected_any) > 0
        has_chosen_raw = len(chosen_raw) > 0
        has_rejected_raw = len(rejected_raw) > 0

        if has_chosen and has_rejected:
            complete_any += 1
            augmented_pairs_any += len(chosen_any) * len(rejected_any)
            if has_chosen_raw and has_rejected_raw:
                complete_with_raw += 1
                augmented_pairs_raw += len(chosen_raw) * len(rejected_raw)
        elif has_chosen:
            chosen_only += 1
        elif has_rejected:
            rejected_only += 1
        else:
            no_feedback += 1

    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"Total queries:           {total_queries}")
    print(f"Total responses:         {total_spans}")
    print(f"  - with raw OTEL data:  {spans_with_raw}")
    print(f"  - without raw data:    {spans_without_raw}")
    print("-" * 60)
    print("FEEDBACK STATUS (any raw data status):")
    print(f"  Complete pairs:        {complete_any}")
    print(f"  Chosen only:           {chosen_only}")
    print(f"  Rejected only:         {rejected_only}")
    print(f"  No feedback:           {no_feedback}")
    print("-" * 60)
    print("EXPORTABLE (requires raw data):")
    print(f"  Unique queries:        {complete_with_raw}")
    if data_augmentation:
        print(f"  Augmented pairs:       {augmented_pairs_raw} (Cartesian product)")
    else:
        print(f"  Pairs (first only):    {complete_with_raw}")
        if augmented_pairs_raw > complete_with_raw:
            print(f"  Potential w/ --data-augmentation: {augmented_pairs_raw} pairs")
    print("=" * 60)

    if pairs:
        print(f"\nExporting {len(pairs)} pairs by mode combination:")
        mode_combos = defaultdict(int)
        for pair in pairs:
            combo = f"{pair.chosen.mode} (chosen) vs {pair.rejected.mode} (rejected)"
            mode_combos[combo] += 1
        for combo, count in sorted(mode_combos.items()):
            print(f"  {combo}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Export DPO training data from Phoenix traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("training_data.jsonl"),
        help="Output file path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Use production Phoenix via port-forward (localhost:6006)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing output file"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Only print summary, don't export"
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include pairs without raw OTEL data (llm.raw.*, tool.raw.*). "
        "Use only for testing - exported data quality will be degraded.",
    )
    parser.add_argument(
        "--data-augmentation",
        action="store_true",
        help="Generate all unique (chosen, rejected) pairs via Cartesian product. "
        "Deterministic ordering by timestamp. Default: only first pair per query.",
    )
    parser.add_argument(
        "--min-version",
        type=str,
        default=None,
        help="Only export data with version_number >= this value. "
        "Semantic versioning format (e.g., '0.1.0', 'v1.0.0'). "
        "Data without a version will be excluded when this flag is set.",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Include system prompt in the messages array. "
        "Default behavior omits system prompts to reduce sequence length.",
    )

    args = parser.parse_args()

    # Get Phoenix URL
    url = get_phoenix_url(args.prod)
    print(f"Connecting to Phoenix: {url}")

    # Fetch spans
    spans = get_training_spans(url)
    if not spans:
        print("No training spans found. Make sure you have:")
        print("  1. Sent chat requests with prompt_mode set")
        print("  2. Submitted feedback (thumbs up/down)")
        sys.exit(1)

    # Group and find pairs (require raw data by default)
    spans_by_query = group_by_query(spans)
    require_raw = not args.include_legacy
    pairs = find_pairs(
        spans_by_query,
        require_raw_data=require_raw,
        data_augmentation=args.data_augmentation,
        min_version=args.min_version,
    )

    # Print summary
    print_summary(spans_by_query, pairs, data_augmentation=args.data_augmentation)

    if args.include_legacy:
        print(
            "\n⚠️  WARNING: --include-legacy is set. Pairs without raw OTEL data are included."
        )
        print("   Data quality may be degraded. Only use for testing.")

    if args.data_augmentation:
        print(
            "\n📊 Data augmentation enabled: generating all (chosen × rejected) combinations."
        )

    if args.min_version:
        print(
            f"\n🔖 Version filter enabled: only exporting data with version_number >= {args.min_version}"
        )

    if args.include_system_prompt:
        print("\n📝 System prompts will be included in messages.")
    else:
        print("\n📝 System prompts omitted (use --include-system-prompt to include).")

    if args.summary_only:
        return

    if not pairs:
        print("\nNo exportable pairs found.")
        if require_raw:
            print("A complete pair requires:")
            print("  1. Both thumbs_up AND thumbs_down feedback for the same query")
            print("  2. BOTH responses must have raw OTEL data (llm.raw.*, tool.raw.*)")
            print("\nTo include legacy pairs without raw data, use --include-legacy")
        else:
            print(
                "A complete pair requires both thumbs_up and thumbs_down feedback for the same query."
            )
        sys.exit(0)

    # Export
    export_pairs(pairs, args.output, args.dry_run, args.include_system_prompt)


if __name__ == "__main__":
    main()
