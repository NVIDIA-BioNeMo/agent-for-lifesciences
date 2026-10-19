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

"""Phoenix GraphQL query helper for Aevom trace debugging.

This script provides convenient functions for querying OpenTelemetry traces
from the Arize Phoenix collector. It's designed to be used during development
and debugging to inspect LLM reasoning, tool calls, and execution flow.

Supports both local Docker development and Kubernetes deployments.

Usage:
    # Get 10 most recent spans (auto-detects local Docker Phoenix)
    python3 scripts/phoenix_query.py recent 10

    # Query K8s production Phoenix (requires port-forward first)
    kubectl -n aevom port-forward svc/phoenix 6006:6006 &
    python3 scripts/phoenix_query.py --prod recent 10

    # Force local Docker Phoenix
    python3 scripts/phoenix_query.py --local recent 10

    # Get recent spans filtered by name pattern
    python3 scripts/phoenix_query.py recent 20 --filter "tool."

    # Get recent spans and show full reasoning content
    python3 scripts/phoenix_query.py recent 5 --show-reasoning

Examples:
    # After running a test, see what happened
    python3 scripts/phoenix_query.py recent 15

    # Debug tool execution
    python3 scripts/phoenix_query.py recent 20 --filter "tool."

    # See full LLM reasoning
    python3 scripts/phoenix_query.py recent 10 --filter "reasoning_iteration" --show-reasoning

    # Debug K8s production
    python3 scripts/phoenix_query.py --prod errors 10

    # Show full tool args (up to 10k chars)
    python3 scripts/phoenix_query.py recent 5 --filter "tool." --full

    # Show absolutely everything with no truncation
    python3 scripts/phoenix_query.py recent 2 --filter "tool.python_execute" --no-limit
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Optional

# Phoenix configuration.
#
# The project is resolved by NAME at runtime rather than pinned as a base64
# node id. The old pin ("UHJvamVjdDoy", i.e. Project:2) is only correct for a
# Phoenix whose second project happens to be this one -- it breaks on a fresh
# volume, and it cannot reach a second deployment at all. The bench stack is
# exactly that second deployment: its own Phoenix, its own project named
# aevom-smdd, on its own container.
DEFAULT_PROJECT_NAME = "aevom"
BENCH_PROJECT_NAME = "aevom-smdd"
DEFAULT_PHOENIX_CONTAINER = "aevom-phoenix-1"
BENCH_PHOENIX_CONTAINER = "aevom-smdd-phoenix-1"

_project_id_cache: dict[str, str] = {}


def project_name() -> str:
    """Which Phoenix project to read, by name."""
    import os

    return os.environ.get("PHOENIX_PROJECT", DEFAULT_PROJECT_NAME)


def get_project_id() -> str:
    """The node id of the configured project, looked up by name.

    Raises:
        SystemExit: if no project of that name exists, which is a clearer
            failure than an empty result set from a wrong id.
    """
    name = project_name()
    if name in _project_id_cache:
        return _project_id_cache[name]

    result = graphql_query(
        "{ projects { edges { node { id name } } } }", quiet=True
    )
    edges = ((result.get("data") or {}).get("projects") or {}).get("edges") or []
    available = {e["node"]["name"]: e["node"]["id"] for e in edges if e.get("node")}
    if name not in available:
        raise SystemExit(
            f"no Phoenix project named {name!r}. Available: "
            f"{', '.join(sorted(available)) or '(none)'}"
        )
    _project_id_cache[name] = available[name]
    return available[name]


def get_phoenix_url() -> tuple[str, str]:
    """Get Phoenix URL - supports multiple deployment modes.
    
    Discovery order:
    1. PHOENIX_URL environment variable (explicit override)
    2. Docker container inspection (local development)
    3. localhost:6006 (K8s port-forward or local Phoenix)
    
    Returns:
        Tuple of (Phoenix GraphQL endpoint URL, source description)
    """
    import os
    import subprocess
    
    # 1. Check for explicit URL (K8s port-forward, remote Phoenix, etc.)
    env_url = os.environ.get("PHOENIX_URL")
    if env_url:
        # Ensure it ends with /graphql
        if not env_url.endswith("/graphql"):
            env_url = env_url.rstrip("/") + "/graphql"
        # Provide descriptive source based on URL
        if "localhost:6006" in env_url:
            source = "K8s Phoenix (localhost:6006 via port-forward)"
        elif "localhost:6080" in env_url:
            source = "Local Docker Phoenix (localhost:6080)"
        else:
            source = f"PHOENIX_URL: {env_url}"
        return env_url, source
    
    # 2. Try Docker container discovery (local development)
    container = os.environ.get("PHOENIX_CONTAINER", DEFAULT_PHOENIX_CONTAINER)
    try:
        result = subprocess.run(
            ["docker", "inspect", container, "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
            return f"http://{ip}:6006/graphql", f"Docker container {container}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 3. Fallback to localhost (K8s port-forward or direct access)
    return "http://localhost:6006/graphql", "localhost:6006 (K8s port-forward?)"


_phoenix_url_cache: Optional[tuple[str, str]] = None

def graphql_query(query: str, variables: Optional[dict] = None, quiet: bool = False) -> dict:
    """Execute a GraphQL query against Phoenix.
    
    Args:
        query: GraphQL query string
        variables: Optional query variables
        quiet: If True, don't print connection info
        
    Returns:
        JSON response from Phoenix
        
    Raises:
        Exception: If query fails
    """
    global _phoenix_url_cache
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    
    # Get Phoenix URL dynamically (cache for subsequent calls)
    if _phoenix_url_cache is None:
        _phoenix_url_cache = get_phoenix_url()
        if not quiet:
            url, source = _phoenix_url_cache
            print(f"🔗 Connecting to Phoenix: {source}", file=sys.stderr)
    
    url, _ = _phoenix_url_cache
    last_error = None
    
    for attempt in range(2):  # Retry once
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                if "errors" in result:
                    print(f"GraphQL errors: {result['errors']}", file=sys.stderr)
                return result
        except urllib.error.URLError as e:
            last_error = e
            continue
    
    raise Exception(f"Failed to connect to Phoenix at {url}: {last_error}")


def get_recent_spans(
    limit: int = 10,
    name_filter: Optional[str] = None,
    hours_back: int = 24
) -> list[dict]:
    """Get recent spans from the aevom project.
    
    Args:
        limit: Maximum number of spans to return
        name_filter: Optional string to filter span names (substring match)
        hours_back: How many hours back to look (currently ignored, fetches most recent)
        
    Returns:
        List of span dictionaries
    """
    
    query = """
    query GetSpans($projectId: GlobalID!, $first: Int!, $startTime: DateTime!, $endTime: DateTime!) {
        node(id: $projectId) {
            ... on Project {
                name
                spans(
                    first: $first,
                    sort: {col: startTime, dir: desc},
                    timeRange: {start: $startTime, end: $endTime}
                ) {
                    edges {
                        node {
                            id
                            name
                            spanId
                            statusCode
                            startTime
                            endTime
                            latencyMs
                            parentId
                            attributes
                        }
                    }
                }
            }
        }
    }
    """
    
    # Note: Phoenix uses a simpler query format without explicit variables
    # Build the query inline - skip timeRange as it can cause parsing issues
    simple_query = f'''{{
        node(id: "{get_project_id()}") {{
            ... on Project {{
                name
                spans(
                    first: {limit * 3 if name_filter else limit},
                    sort: {{col: startTime, dir: desc}}
                ) {{
                    edges {{
                        node {{
                            id
                            name
                            spanId
                            statusCode
                            startTime
                            endTime
                            latencyMs
                            parentId
                            attributes
                        }}
                    }}
                }}
            }}
        }}
    }}'''
    
    result = graphql_query(simple_query)
    
    if not result.get("data") or not result["data"].get("node"):
        return []
    
    spans = []
    for edge in result["data"]["node"]["spans"]["edges"]:
        span = edge["node"]
        # Parse attributes from JSON string
        if span.get("attributes"):
            try:
                span["attributes"] = json.loads(span["attributes"])
            except json.JSONDecodeError:
                pass
        spans.append(span)
    
    # Apply name filter if specified
    if name_filter:
        spans = [s for s in spans if name_filter.lower() in s["name"].lower()]
        spans = spans[:limit]
    
    return spans


def format_span(span: dict, show_reasoning: bool = False, show_full: bool = False, no_limit: bool = False) -> str:
    """Format a span for display.
    
    Args:
        span: Span dictionary
        show_reasoning: Whether to show full reasoning content
        show_full: Whether to show full content without truncation (10k limit)
        no_limit: Whether to show absolutely everything (no truncation at all)
        
    Returns:
        Formatted string representation
    """
    lines = []
    if no_limit:
        max_len = float('inf')
    else:
        max_len = 10000 if show_full else 200
    
    # Header with name, status, and timing
    status_emoji = "✓" if span["statusCode"] == "OK" else "✗"
    latency = span.get("latencyMs", 0)
    lines.append(f"{status_emoji} {span['name']} ({latency:.0f}ms)")
    
    # Start time
    lines.append(f"   Time: {span['startTime']}")
    
    # Parse and display key attributes
    attrs = span.get("attributes", {})
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            attrs = {}
    
    # Tool-specific attributes
    if span["name"].startswith("tool."):
        tool_attrs = attrs.get("tool", {})
        
        # Use raw (untruncated) values when no_limit is set
        if no_limit and tool_attrs.get("raw", {}).get("arguments"):
            arg_str = tool_attrs["raw"]["arguments"]
            # Parse and pretty-print if possible
            try:
                arg_obj = json.loads(arg_str)
                arg_str = json.dumps(arg_obj, indent=2)
            except json.JSONDecodeError:
                pass
            lines.append(f"   Args (RAW): {arg_str}")
        elif tool_attrs.get("arg"):
            arg_str = json.dumps(tool_attrs['arg'], indent=2 if show_full else None)
            if len(arg_str) > max_len:
                arg_str = arg_str[:max_len] + "..."
            lines.append(f"   Args: {arg_str}")
        
        if tool_attrs.get("result"):
            result = tool_attrs["result"]
            lines.append(f"   Result: {result.get('content_type', 'unknown')} (success={result.get('success', '?')})")
            
            # Use raw output when no_limit is set
            if no_limit and tool_attrs.get("raw", {}).get("output"):
                output = tool_attrs["raw"]["output"]
                lines.append(f"   Output (RAW): {output}")
            elif result.get("output"):
                output = result["output"]
                if len(output) > max_len:
                    output = output[:max_len] + "..."
                lines.append(f"   Output: {output}")
        if tool_attrs.get("error"):
            lines.append(f"   ERROR: {tool_attrs['error']}")
    
    # LLM-specific attributes
    elif span["name"].startswith("llm."):
        llm_attrs = attrs.get("llm", {})
        if llm_attrs.get("model"):
            lines.append(f"   Model: {llm_attrs['model']}")
        if llm_attrs.get("tool_calls"):
            lines.append(f"   Tool calls: {llm_attrs['tool_calls']}")
        # Read reasoning from raw attributes (no duplicates)
        raw_attrs = llm_attrs.get("raw", {})
        if show_reasoning and raw_attrs.get("reasoning"):
            reasoning = raw_attrs["reasoning"]
            lines.append(f"   Reasoning ({len(reasoning)} chars):")
            # Indent reasoning content
            for rline in reasoning.split("\n")[:20]:  # Limit lines
                lines.append(f"      {rline[:100]}")
            if len(reasoning.split("\n")) > 20:
                lines.append("      ... (truncated)")
    
    # Iteration-specific attributes
    elif span["name"].startswith("reasoning_iteration"):
        iter_attrs = attrs.get("iteration", {})
        if iter_attrs.get("tool_calls_count") is not None:
            lines.append(f"   Tool calls: {iter_attrs['tool_calls_count']}")
        if iter_attrs.get("exit_reason"):
            lines.append(f"   Exit reason: {iter_attrs['exit_reason']}")
        # Note: reasoning content is now only in child llm.complete spans (llm.raw.reasoning)
        if iter_attrs.get("reasoning_length"):
            lines.append(f"   Reasoning length: {iter_attrs['reasoning_length']} chars (see child llm.complete span)")
    
    # Chat request attributes
    elif span["name"] == "chat_request":
        chat_attrs = attrs.get("chat", {})
        if chat_attrs.get("user_message"):
            user_msg = chat_attrs['user_message']
            preview = user_msg[:100] + "..." if len(user_msg) > 100 else user_msg
            lines.append(f"   User: {preview}")
        elif chat_attrs.get("user_message_preview"):
            lines.append(f"   User: {chat_attrs['user_message_preview'][:100]}")
        if chat_attrs.get("total_iterations"):
            lines.append(f"   Iterations: {chat_attrs['total_iterations']}")
        if chat_attrs.get("total_tool_calls"):
            lines.append(f"   Total tool calls: {chat_attrs['total_tool_calls']}")
        if chat_attrs.get("hit_iteration_limit"):
            lines.append("   ⚠ Hit iteration limit!")
        # Show final response (truncated by default, full with --no-limit)
        if chat_attrs.get("final_response"):
            response = chat_attrs["final_response"]
            if no_limit:
                lines.append(f"   Final response ({len(response)} chars):")
                for rline in response.split("\n"):
                    lines.append(f"      {rline}")
            else:
                truncated = response[:500] + "..." if len(response) > 500 else response
                lines.append(f"   Final response: {truncated}")
    
    return "\n".join(lines)


def print_spans(spans: list[dict], show_reasoning: bool = False, show_full: bool = False, no_limit: bool = False) -> None:
    """Print formatted spans to stdout.
    
    Args:
        spans: List of span dictionaries
        show_reasoning: Whether to show full reasoning content
        show_full: Whether to show full content without truncation (10k limit)
        no_limit: Whether to show absolutely everything (no truncation at all)
    """
    if not spans:
        print("No spans found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(spans)} spans")
    print(f"{'='*60}\n")
    
    for i, span in enumerate(spans):
        print(format_span(span, show_reasoning, show_full, no_limit))
        if i < len(spans) - 1:
            print("-" * 40)
    
    print()


def get_errors(limit: int = 10, hours_back: int = 24) -> list[dict]:
    """Get recent error spans.
    
    Args:
        limit: Maximum number of errors to return
        hours_back: How many hours back to look
        
    Returns:
        List of error span dictionaries with extracted error messages
    """
    spans = get_recent_spans(limit=limit * 5, hours_back=hours_back)
    errors = []
    
    for span in spans:
        if span.get("statusCode") == "ERROR":
            attrs = span.get("attributes", {})
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except json.JSONDecodeError:
                    attrs = {}
            
            tool_attrs = attrs.get("tool", {})
            error_msg = tool_attrs.get("error") or tool_attrs.get("result", {}).get("output") or "Unknown error"
            
            errors.append({
                "name": span["name"],
                "error": error_msg,
                "latencyMs": span.get("latencyMs", 0),
                "time": span.get("startTime", "")
            })
            
            if len(errors) >= limit:
                break
    
    return errors


def get_project_stats() -> dict:
    """Get basic project statistics.
    
    Returns:
        Dictionary with project stats
    """
    query = f'''{{
        node(id: "{get_project_id()}") {{
            ... on Project {{
                name
                traceCount
                recordCount
            }}
        }}
    }}'''
    
    result = graphql_query(query)
    if result.get("data") and result["data"].get("node"):
        return result["data"]["node"]
    return {}


def main():
    """Main entry point."""
    import os
    
    parser = argparse.ArgumentParser(
        description="Query Phoenix traces for Aevom debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global flags
    parser.add_argument("--prod", action="store_true",
                       help="Connect to K8s Phoenix via port-forward (localhost:6006)")
    parser.add_argument("--local", action="store_true",
                       help="Force local Docker Phoenix connection")
    parser.add_argument("--bench", action="store_true",
                       help="Read the SMDD-Bench stack's own Phoenix instead of the dev stack's")
    parser.add_argument("--project",
                       help=f"Phoenix project name (default: {DEFAULT_PROJECT_NAME})")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # recent command
    recent_parser = subparsers.add_parser("recent", help="Get recent spans")
    recent_parser.add_argument("limit", type=int, nargs="?", default=10,
                               help="Number of spans to fetch (default: 10)")
    recent_parser.add_argument("--filter", "-f", type=str, default=None,
                               help="Filter spans by name substring")
    recent_parser.add_argument("--show-reasoning", "-r", action="store_true",
                               help="Show full reasoning content")
    recent_parser.add_argument("--full", action="store_true",
                               help="Show full content without truncation (up to 10k chars)")
    recent_parser.add_argument("--no-limit", action="store_true",
                               help="Show absolutely everything (no truncation at all)")
    recent_parser.add_argument("--hours", type=int, default=24,
                               help="Hours back to search (default: 24)")
    
    # stats command
    subparsers.add_parser("stats", help="Show project statistics")
    
    # errors command
    errors_parser = subparsers.add_parser("errors", help="Get recent errors")
    errors_parser.add_argument("limit", type=int, nargs="?", default=10,
                               help="Number of errors to fetch (default: 10)")
    errors_parser.add_argument("--hours", type=int, default=24,
                               help="Hours back to search (default: 24)")
    
    args = parser.parse_args()
    
    # Handle --prod and --local flags by setting environment variable
    if args.prod:
        os.environ["PHOENIX_URL"] = "http://localhost:6006"
    elif args.local:
        os.environ["PHOENIX_URL"] = "http://localhost:6080"

    # The bench stack publishes no ports, so it is reached by container IP --
    # the same trick smdd_aevom/stack.py uses for the rest of that stack.
    if getattr(args, "bench", False):
        os.environ.setdefault("PHOENIX_CONTAINER", BENCH_PHOENIX_CONTAINER)
        os.environ.setdefault("PHOENIX_PROJECT", BENCH_PROJECT_NAME)
    if getattr(args, "project", None):
        os.environ["PHOENIX_PROJECT"] = args.project
    
    if args.command == "recent":
        spans = get_recent_spans(
            limit=args.limit,
            name_filter=args.filter,
            hours_back=args.hours
        )
        no_limit = getattr(args, 'no_limit', False)
        print_spans(spans, show_reasoning=args.show_reasoning, show_full=args.full, no_limit=no_limit)
    
    elif args.command == "stats":
        stats = get_project_stats()
        print(f"\nProject: {stats.get('name', 'unknown')}")
        print(f"Total traces: {stats.get('traceCount', 0)}")
        print(f"Total records: {stats.get('recordCount', 0)}")
        print()
    
    elif args.command == "errors":
        errors = get_errors(limit=args.limit, hours_back=args.hours)
        if not errors:
            print("\nNo errors found.")
        else:
            print(f"\n{'='*60}")
            print(f"Found {len(errors)} errors")
            print(f"{'='*60}\n")
            for err in errors:
                print(f"✗ {err['name']} ({err['latencyMs']:.0f}ms)")
                print(f"   {err['error'][:200]}")
                print("-" * 40)
        print()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
