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

"""Training episode trace query tool for Aevom GRPO training.

Analogous to phoenix_query.py but for training episode traces stored in JSONL
files written by the resource server during GRPO training.

Episode traces are written to /results/episode_traces/episodes.jsonl (on cluster)
or a local path after scp-ing the file from the cluster.

Usage:
    # View recent episodes
    python3 scripts/training_query.py recent 10 --file episodes.jsonl

    # View recent episodes with full tool output
    python3 scripts/training_query.py recent 5 --file episodes.jsonl --full

    # Filter to episodes with specific tool usage
    python3 scripts/training_query.py recent 20 --file episodes.jsonl --filter vina_dock

    # View only error episodes
    python3 scripts/training_query.py errors 10 --file episodes.jsonl

    # Show statistics
    python3 scripts/training_query.py stats --file episodes.jsonl

    # View a single episode in full detail
    python3 scripts/training_query.py detail 0 --file episodes.jsonl

    # View recent episodes with reasoning traces (thinking + tool calls)
    python3 scripts/training_query.py recent 5 --file episodes.jsonl --show-reasoning

Examples:
    # After copying traces from cluster
    scp cluster:~/runs/<RUN>/*/episode_traces/episodes.jsonl ./episodes.jsonl
    python3 scripts/training_query.py stats --file episodes.jsonl
    python3 scripts/training_query.py recent 5 --file episodes.jsonl --show-reasoning
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


def load_episodes(filepath: str) -> list[dict]:
    """Load episodes from a JSONL file.

    Args:
        filepath: Path to the episodes.jsonl file

    Returns:
        List of episode dictionaries (most recent last)
    """
    episodes = []
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping malformed line {line_num}: {e}", file=sys.stderr
                )
    return episodes


def format_episode_summary(
    ep: dict, index: int, full: bool = False, show_reasoning: bool = False
) -> str:
    """Format an episode for summary display.

    Args:
        ep: Episode dictionary
        index: Episode index
        full: Whether to show full output (no truncation)
        show_reasoning: Whether to show reasoning trace content

    Returns:
        Formatted string
    """
    lines = []

    reward = ep.get("reward", 0)
    num_turns = ep.get("num_turns", 0)
    num_tool_calls = ep.get("num_tool_calls", 0)
    num_errors = ep.get("num_tool_errors", 0)
    timestamp = ep.get("timestamp", "")
    env_id = ep.get("env_id", "")

    status = "+" if reward >= 0.5 else "~" if reward > 0 else "-"
    lines.append(
        f"[{status}] Episode #{index}  reward={reward:.2f}  "
        f"turns={num_turns}  tools={num_tool_calls}  errors={num_errors}"
    )
    if timestamp:
        lines.append(f"   Time: {timestamp}")
    if env_id:
        lines.append(f"   Env: {env_id}")

    # Reward breakdown
    breakdown = ep.get("reward_breakdown", {})
    if breakdown:
        parts = [f"{k}={v:.2f}" for k, v in breakdown.items()]
        lines.append(f"   Rewards: {', '.join(parts)}")

    # Ground truth
    gt = ep.get("ground_truth_summary", {})
    if gt.get("sabotage_type"):
        lines.append(f"   Sabotage: {gt['sabotage_type']}")

    # Task prompt (truncated in summary)
    task = ep.get("task_prompt", "")
    if task:
        preview = task[:150].replace("\n", " ")
        if len(task) > 150:
            preview += "..."
        lines.append(f"   Task: {preview}")

    # Tool calls summary
    trace = ep.get("episode_trace", [])
    if trace:
        tool_names = [tc.get("tool_name", "?") for tc in trace]
        lines.append(f"   Tools: {' -> '.join(tool_names)}")

    # Show reasoning trace (interleaved thinking + tool calls)
    reasoning_trace = ep.get("reasoning_trace", [])
    if show_reasoning and reasoning_trace:
        lines.append("   --- Reasoning Trace ---")
        for rt in reasoning_trace:
            turn = rt.get("turn", 0)
            reasoning = rt.get("reasoning", "")
            tool_name = rt.get("tool_name", "")
            tool_args = rt.get("tool_args", "")
            tool_output = rt.get("tool_output", "")
            tool_success = rt.get("tool_success", True)
            elapsed = rt.get("elapsed_ms", 0)

            lines.append(f"   [Turn {turn}]")
            if reasoning:
                lines.append(f"     Thinking:")
                for rline in reasoning.split("\n"):
                    lines.append(f"       {rline}")
            if tool_name:
                status_icon = "ok" if tool_success else "ERR"
                lines.append(f"     Tool: {tool_name} [{status_icon}] ({elapsed:.0f}ms)")
                if tool_args:
                    lines.append(f"       Args: {tool_args}")
                if tool_output:
                    lines.append(f"       Output: {tool_output}")

    # Show tool details if full mode (legacy view without reasoning)
    elif full and trace:
        lines.append("   --- Tool Call Details ---")
        for tc in trace:
            name = tc.get("tool_name", "?")
            success = tc.get("success", True)
            elapsed = tc.get("elapsed_ms", 0)
            status_icon = "ok" if success else "ERR"
            lines.append(f"   [{status_icon}] {name} ({elapsed:.0f}ms)")

            args = tc.get("arguments", {})
            if args:
                arg_str = json.dumps(args, indent=2)
                for arg_line in arg_str.split("\n"):
                    lines.append(f"       Args: {arg_line}")

            output = tc.get("output", "")
            if output:
                lines.append(f"       Output: {output}")

    # Final response (truncated in summary unless full or show_reasoning)
    response = ep.get("final_response", "")
    if response:
        if full or show_reasoning:
            lines.append(f"   Response: {response}")
        else:
            preview = response[:200].replace("\n", " ")
            if len(response) > 200:
                preview += "..."
            lines.append(f"   Response: {preview}")

    return "\n".join(lines)


def format_episode_detail(ep: dict, index: int) -> str:
    """Format a single episode with full detail (no truncation).

    Args:
        ep: Episode dictionary
        index: Episode index

    Returns:
        Formatted string with complete episode data
    """
    lines = []
    reward = ep.get("reward", 0)
    lines.append(f"{'=' * 70}")
    lines.append(f"EPISODE #{index}  |  reward={reward:.2f}")
    lines.append(f"{'=' * 70}")

    # Metadata
    lines.append(f"  Timestamp:    {ep.get('timestamp', '')}")
    lines.append(f"  Env ID:       {ep.get('env_id', '')}")
    lines.append(f"  Turns:        {ep.get('num_turns', 0)}")
    lines.append(f"  Tool calls:   {ep.get('num_tool_calls', 0)}")
    lines.append(f"  Tool errors:  {ep.get('num_tool_errors', 0)}")

    # Reward breakdown
    breakdown = ep.get("reward_breakdown", {})
    if breakdown:
        lines.append("")
        lines.append("  REWARD BREAKDOWN:")
        for k, v in breakdown.items():
            lines.append(f"    {k}: {v:.2f}")

    # Ground truth
    gt = ep.get("ground_truth_summary", {})
    if gt:
        lines.append("")
        lines.append("  GROUND TRUTH:")
        for k, v in gt.items():
            if v:
                lines.append(f"    {k}: {v}")

    # Task prompt
    task = ep.get("task_prompt", "")
    if task:
        lines.append("")
        lines.append("  TASK PROMPT:")
        for tline in task.split("\n"):
            lines.append(f"    {tline}")

    # Reasoning trace (preferred view -- shows thinking + tool calls together)
    reasoning_trace = ep.get("reasoning_trace", [])
    if reasoning_trace:
        lines.append("")
        lines.append(f"  REASONING TRACE ({len(reasoning_trace)} turns):")
        lines.append(f"  {'=' * 60}")
        for rt in reasoning_trace:
            turn = rt.get("turn", 0)
            reasoning = rt.get("reasoning", "")
            tool_name = rt.get("tool_name", "")
            tool_args = rt.get("tool_args", "")
            tool_output = rt.get("tool_output", "")
            tool_success = rt.get("tool_success", True)
            elapsed = rt.get("elapsed_ms", 0)

            lines.append(f"  --- Turn {turn} ---")

            if reasoning:
                lines.append("  THINKING:")
                for rline in reasoning.split("\n"):
                    lines.append(f"    {rline}")

            if tool_name:
                status_icon = "ok" if tool_success else "FAIL"
                lines.append(
                    f"  TOOL: {tool_name}  [{status_icon}]  {elapsed:.0f}ms"
                )
                if tool_args:
                    lines.append("    Arguments:")
                    try:
                        parsed = json.loads(tool_args)
                        arg_str = json.dumps(parsed, indent=2)
                    except (json.JSONDecodeError, TypeError):
                        arg_str = tool_args
                    for arg_line in arg_str.split("\n"):
                        lines.append(f"      {arg_line}")
                if tool_output:
                    lines.append(f"    Output ({len(tool_output)} chars):")
                    for out_line in tool_output.split("\n"):
                        lines.append(f"      {out_line}")

            lines.append(f"  {'-' * 60}")

    # Fall back to episode_trace if no reasoning_trace
    else:
        trace = ep.get("episode_trace", [])
        if trace:
            lines.append("")
            lines.append(f"  EPISODE TRACE ({len(trace)} tool calls):")
            lines.append(f"  {'-' * 60}")
            for i, tc in enumerate(trace, 1):
                name = tc.get("tool_name", "?")
                success = tc.get("success", True)
                elapsed = tc.get("elapsed_ms", 0)
                turn = tc.get("turn", "?")
                status_icon = "ok" if success else "FAIL"

                lines.append(
                    f"  [{i}] {name}  turn={turn}  {elapsed:.0f}ms  [{status_icon}]"
                )

                args = tc.get("arguments", {})
                if args:
                    lines.append("      Arguments:")
                    arg_str = json.dumps(args, indent=2)
                    for arg_line in arg_str.split("\n"):
                        lines.append(f"        {arg_line}")

                output = tc.get("output", "")
                if output:
                    lines.append(f"      Output ({len(output)} chars):")
                    for out_line in output.split("\n"):
                        lines.append(f"        {out_line}")

                lines.append(f"  {'-' * 60}")

    # Final response
    response = ep.get("final_response", "")
    if response:
        lines.append("")
        lines.append("  FINAL RESPONSE:")
        for rline in response.split("\n"):
            lines.append(f"    {rline}")

    return "\n".join(lines)


def cmd_recent(
    episodes: list[dict],
    limit: int,
    filter_str: Optional[str],
    full: bool,
    show_reasoning: bool = False,
) -> None:
    """Show recent episodes."""
    # Episodes are in chronological order; reverse for most-recent-first
    recent = list(reversed(episodes))

    if filter_str:
        filtered = []
        for ep in recent:
            trace = ep.get("episode_trace", [])
            tool_names = [tc.get("tool_name", "") for tc in trace]
            # Match filter against tool names, sabotage type, or env_id
            searchable = " ".join(tool_names).lower()
            searchable += (
                " "
                + ep.get("ground_truth_summary", {}).get("sabotage_type", "").lower()
            )
            searchable += " " + ep.get("env_id", "").lower()
            if filter_str.lower() in searchable:
                filtered.append(ep)
        recent = filtered

    recent = recent[:limit]

    if not recent:
        print("\nNo episodes found.")
        return

    print(f"\n{'=' * 60}")
    print(f"Showing {len(recent)} of {len(episodes)} episodes")
    print(f"{'=' * 60}\n")

    for i, ep in enumerate(recent):
        # Compute the original index (from end of file)
        orig_idx = len(episodes) - 1 - episodes[::-1].index(ep)
        print(
            format_episode_summary(
                ep, orig_idx, full=full, show_reasoning=show_reasoning
            )
        )
        if i < len(recent) - 1:
            print("-" * 50)
    print()


def cmd_errors(episodes: list[dict], limit: int) -> None:
    """Show episodes with tool errors."""
    error_eps = [ep for ep in reversed(episodes) if ep.get("num_tool_errors", 0) > 0]
    error_eps = error_eps[:limit]

    if not error_eps:
        print("\nNo episodes with errors found.")
        return

    print(f"\n{'=' * 60}")
    print(f"Found {len(error_eps)} episodes with errors")
    print(f"{'=' * 60}\n")

    for i, ep in enumerate(error_eps):
        orig_idx = len(episodes) - 1 - episodes[::-1].index(ep)
        print(format_episode_summary(ep, orig_idx, full=True))
        if i < len(error_eps) - 1:
            print("-" * 50)
    print()


def cmd_detail(episodes: list[dict], index: int) -> None:
    """Show full detail for a single episode."""
    if index < 0 or index >= len(episodes):
        print(
            f"Error: Episode index {index} out of range (0-{len(episodes) - 1})",
            file=sys.stderr,
        )
        return

    ep = episodes[index]
    print(format_episode_detail(ep, index))
    print()


def cmd_stats(episodes: list[dict]) -> None:
    """Show aggregate statistics."""
    if not episodes:
        print("\nNo episodes found.")
        return

    rewards = [ep.get("reward", 0) for ep in episodes]
    tool_counts = [ep.get("num_tool_calls", 0) for ep in episodes]
    turn_counts = [ep.get("num_turns", 0) for ep in episodes]
    error_counts = [ep.get("num_tool_errors", 0) for ep in episodes]

    # Tool usage frequency
    tool_freq: dict[str, int] = {}
    for ep in episodes:
        for tc in ep.get("episode_trace", []):
            name = tc.get("tool_name", "unknown")
            tool_freq[name] = tool_freq.get(name, 0) + 1

    # Sabotage type distribution
    sabotage_dist: dict[str, int] = {}
    for ep in episodes:
        st = ep.get("ground_truth_summary", {}).get("sabotage_type", "unknown")
        sabotage_dist[st] = sabotage_dist.get(st, 0) + 1

    # Reward distribution
    reward_buckets = {"0.0": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.8": 0, "0.8+": 0}
    for r in rewards:
        if r == 0:
            reward_buckets["0.0"] += 1
        elif r <= 0.3:
            reward_buckets["0.1-0.3"] += 1
        elif r <= 0.5:
            reward_buckets["0.3-0.5"] += 1
        elif r <= 0.8:
            reward_buckets["0.5-0.8"] += 1
        else:
            reward_buckets["0.8+"] += 1

    print(f"\n{'=' * 60}")
    print(f"Training Episode Statistics  ({len(episodes)} episodes)")
    print(f"{'=' * 60}")

    print(f"\n  Rewards:")
    print(f"    Mean:  {sum(rewards) / len(rewards):.3f}")
    print(f"    Min:   {min(rewards):.3f}")
    print(f"    Max:   {max(rewards):.3f}")
    print(f"    Distribution:")
    for bucket, count in reward_buckets.items():
        bar = "#" * int(count / len(episodes) * 40)
        print(
            f"      {bucket:>7}: {count:>4} ({count / len(episodes) * 100:.0f}%) {bar}"
        )

    print(f"\n  Tool Calls:")
    print(f"    Mean per episode: {sum(tool_counts) / len(tool_counts):.1f}")
    print(f"    Max per episode:  {max(tool_counts)}")
    print(f"    Total errors:     {sum(error_counts)}")

    print(f"\n  Turns:")
    print(f"    Mean per episode: {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"    Max per episode:  {max(turn_counts)}")

    if tool_freq:
        print(f"\n  Tool Usage Frequency:")
        for name, count in sorted(tool_freq.items(), key=lambda x: -x[1]):
            print(f"    {name:>25}: {count}")

    if sabotage_dist:
        print(f"\n  Sabotage Type Distribution:")
        for st, count in sorted(sabotage_dist.items(), key=lambda x: -x[1]):
            print(f"    {st:>25}: {count}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query Aevom training episode traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="episodes.jsonl",
        help="Path to episodes.jsonl file (default: episodes.jsonl)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # recent command
    recent_parser = subparsers.add_parser("recent", help="Show recent episodes")
    recent_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=10,
        help="Number of episodes to show (default: 10)",
    )
    recent_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter by tool name, sabotage type, or env_id",
    )
    recent_parser.add_argument(
        "--full",
        action="store_true",
        help="Show full tool call details",
    )
    recent_parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show reasoning trace (thinking + tool calls) for each turn",
    )

    # errors command
    errors_parser = subparsers.add_parser(
        "errors", help="Show episodes with tool errors"
    )
    errors_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=10,
        help="Number of error episodes to show (default: 10)",
    )

    # detail command
    detail_parser = subparsers.add_parser(
        "detail", help="Show full detail for a single episode"
    )
    detail_parser.add_argument(
        "index",
        type=int,
        help="Episode index (0-based)",
    )

    # stats command
    subparsers.add_parser("stats", help="Show aggregate statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    episodes = load_episodes(args.file)

    if args.command == "recent":
        cmd_recent(
            episodes,
            args.limit,
            args.filter,
            args.full,
            show_reasoning=args.show_reasoning,
        )
    elif args.command == "errors":
        cmd_errors(episodes, args.limit)
    elif args.command == "detail":
        cmd_detail(episodes, args.index)
    elif args.command == "stats":
        cmd_stats(episodes)


if __name__ == "__main__":
    main()
