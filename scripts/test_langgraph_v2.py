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

"""Integration test for the LangGraph v2 chat endpoint.

Tests all three Chat.tsx example prompts against /api/chat/v2 and
compares the SSE event stream with the expected format.

Usage:
    # Start dev stack first
    make dev

    # Run all tests
    python3 scripts/test_langgraph_v2.py

    # Run a single test
    python3 scripts/test_langgraph_v2.py --test small_molecule
"""

import argparse
import json
import sys
import time
from typing import Optional

import httpx

BASE_URL = "http://localhost:8000"

EXAMPLE_PROMPTS = {
    "small_molecule": {
        "prompt": "What is the 3D structure of tovorafenib?",
        "expected_events": ["content", "tool_call_start", "tool_call_end"],
        "expected_tools": ["python_execute"],
    },
    "structure_prediction": {
        "prompt": (
            "Predict the 3d structure of the PGK1 docked to imatinib. "
            "Show me the predicted complex in the viewer and describe any "
            "interesting features you see."
        ),
        "expected_events": ["content", "tool_call_start", "tool_call_end"],
        "expected_tools": [
            "fetch_protein_sequence", "openfold3", "visualize_pdb",
        ],
    },
    "protein_ligand": {
        "prompt": (
            "Analyze the the crystal structure of Gefitinib bound to the "
            "Epidermal Growth Factor Receptor kinase domain (EGFR). "
            "Show me all molecular interactions within 5A of the drug. "
            "Report the distance and type of each molecular interaction."
        ),
        "expected_events": [
            "content", "tool_call_start", "tool_call_end",
            "storage_update",
        ],
        "expected_tools": [
            "search_rcsb", "fetch_and_clean_pdb", "visualize_pdb",
            "structure_query", "python_execute",
        ],
    },
}


def test_endpoint(
    name: str,
    prompt: str,
    expected_events: list[str],
    expected_tools: list[str],
    endpoint: str = "/api/chat/v2",
) -> dict:
    """Test a single prompt against the v2 endpoint.

    Returns a dict with test results.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Endpoint: {endpoint}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'='*60}")

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "session_id": f"test_{name}_{int(time.time())}",
    }

    events_seen: dict[str, int] = {}
    tools_called: list[str] = []
    content_chunks: list[str] = []
    errors: list[str] = []

    start = time.time()

    try:
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{BASE_URL}{endpoint}",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    return {
                        "name": name,
                        "passed": False,
                        "error": f"HTTP {response.status_code}",
                        "duration": time.time() - start,
                    }

                for line in response.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Categorise the event
                    if "content" in event:
                        events_seen["content"] = events_seen.get("content", 0) + 1
                        content_chunks.append(event["content"])
                    if "reasoning" in event:
                        events_seen["reasoning"] = events_seen.get("reasoning", 0) + 1
                    if "tool_call_start" in event:
                        events_seen["tool_call_start"] = events_seen.get("tool_call_start", 0) + 1
                        tool_name = event["tool_call_start"].get("name", "")
                        tools_called.append(tool_name)
                        print(f"  [tool_call_start] {tool_name}")
                    if "tool_call_end" in event:
                        events_seen["tool_call_end"] = events_seen.get("tool_call_end", 0) + 1
                    if "storage_update" in event:
                        events_seen["storage_update"] = events_seen.get("storage_update", 0) + 1
                    if "span_id" in event:
                        events_seen["span_id"] = events_seen.get("span_id", 0) + 1
                    if "Error:" in str(event.get("content", "")):
                        errors.append(event["content"])

    except httpx.ConnectError:
        return {
            "name": name,
            "passed": False,
            "error": "Connection refused — is the dev stack running?",
            "duration": time.time() - start,
        }
    except Exception as e:
        return {
            "name": name,
            "passed": False,
            "error": str(e),
            "duration": time.time() - start,
        }

    duration = time.time() - start

    # Check expectations
    missing_events = [e for e in expected_events if e not in events_seen]
    missing_tools = [t for t in expected_tools if t not in tools_called]

    passed = not missing_events and not missing_tools and not errors
    content = "".join(content_chunks)

    print(f"\n  Duration: {duration:.1f}s")
    print(f"  Events: {events_seen}")
    print(f"  Tools called: {tools_called}")
    print(f"  Content length: {len(content)} chars")
    if missing_events:
        print(f"  MISSING events: {missing_events}")
    if missing_tools:
        print(f"  MISSING tools: {missing_tools}")
    if errors:
        print(f"  ERRORS: {errors}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "name": name,
        "passed": passed,
        "duration": duration,
        "events": events_seen,
        "tools_called": tools_called,
        "content_length": len(content),
        "missing_events": missing_events,
        "missing_tools": missing_tools,
        "errors": errors,
    }


def compare_v1_v2(name: str, prompt: str) -> None:
    """Run the same prompt against v1 and v2 and compare."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {name}")
    print(f"{'='*60}")

    v1_result = test_endpoint(
        f"{name}_v1", prompt, [], [], endpoint="/api/chat",
    )
    v2_result = test_endpoint(
        f"{name}_v2", prompt, [], [], endpoint="/api/chat/v2",
    )

    print(f"\n  V1 events: {v1_result.get('events', {})}")
    print(f"  V2 events: {v2_result.get('events', {})}")
    print(f"  V1 tools: {v1_result.get('tools_called', [])}")
    print(f"  V2 tools: {v2_result.get('tools_called', [])}")


def main():
    parser = argparse.ArgumentParser(description="Test LangGraph v2 endpoint")
    parser.add_argument(
        "--test",
        choices=list(EXAMPLE_PROMPTS.keys()) + ["all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare v1 and v2 endpoints side by side",
    )
    args = parser.parse_args()

    tests = (
        list(EXAMPLE_PROMPTS.keys())
        if args.test == "all"
        else [args.test]
    )

    results = []
    for test_name in tests:
        config = EXAMPLE_PROMPTS[test_name]
        if args.compare:
            compare_v1_v2(test_name, config["prompt"])
        else:
            result = test_endpoint(
                test_name,
                config["prompt"],
                config["expected_events"],
                config["expected_tools"],
            )
            results.append(result)

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {r['name']}: {status} ({r['duration']:.1f}s)")

        all_passed = all(r["passed"] for r in results)
        print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
