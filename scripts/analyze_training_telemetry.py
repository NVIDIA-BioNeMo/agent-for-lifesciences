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

"""Analyze a Phoenix SQLite DB from an Aevom training run.

Checks that telemetry is complete and correct:
  - System prompt and tool schemas recorded on chat_request spans
  - total_effective_reward annotation on root spans
  - Annotation sums match span attribute rewards
  - Tool-calling stats (zero-tool %, error rates, hallucinated names)
  - Episode outcome distribution

Usage:
    python3 scripts/analyze_training_telemetry.py /path/to/phoenix.db

If the DB uses WAL mode, ensure the -wal and -shm files are alongside
the .db file (copy all three together).
"""

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_analysis(db_path: str) -> dict:
    conn = connect(db_path)
    results: dict = {}

    # ----------------------------------------------------------------
    # Basic stats
    # ----------------------------------------------------------------
    section("Basic Stats")
    total_spans = conn.execute("SELECT count(*) FROM spans").fetchone()[0]
    total_traces = conn.execute(
        "SELECT count(DISTINCT trace_rowid) FROM spans"
    ).fetchone()[0]
    print(f"  Total spans: {total_spans}")
    print(f"  Total traces: {total_traces}")
    results["total_spans"] = total_spans
    results["total_traces"] = total_traces

    # ----------------------------------------------------------------
    # System prompt and tool schema presence
    # ----------------------------------------------------------------
    section("System Prompt & Tool Schema Presence")
    chat_rows = conn.execute(
        "SELECT attributes FROM spans WHERE name = 'chat_request'"
    ).fetchall()
    total_episodes = len(chat_rows)
    print(f"  Total chat_request spans: {total_episodes}")

    has_system_prompt = 0
    has_tool_schemas = 0
    has_user_message = 0
    has_final_response = 0
    has_episode_outcome = 0

    for r in chat_rows:
        attrs = json.loads(r["attributes"]) if r["attributes"] else {}
        chat = attrs.get("chat", {})
        if chat.get("system_prompt"):
            has_system_prompt += 1
        if chat.get("tool_schemas"):
            has_tool_schemas += 1
        if chat.get("user_message"):
            has_user_message += 1
        if chat.get("final_response"):
            has_final_response += 1
        if chat.get("episode_outcome"):
            has_episode_outcome += 1

    def pct(n: int) -> str:
        return (
            f"{n}/{total_episodes} ({100 * n / total_episodes:.0f}%)"
            if total_episodes
            else "0"
        )

    print(f"  Has system_prompt:   {pct(has_system_prompt)}")
    print(f"  Has tool_schemas:    {pct(has_tool_schemas)}")
    print(f"  Has user_message:    {pct(has_user_message)}")
    print(f"  Has final_response:  {pct(has_final_response)}")
    print(f"  Has episode_outcome: {pct(has_episode_outcome)}")

    if has_system_prompt == 0:
        print("  ** WARNING: No system prompts recorded. Deploy Step 1 fix.")
    if has_tool_schemas == 0:
        print("  ** WARNING: No tool schemas recorded. Deploy Step 1 fix.")

    results["has_system_prompt"] = has_system_prompt
    results["has_tool_schemas"] = has_tool_schemas

    # ----------------------------------------------------------------
    # Episode outcome distribution
    # ----------------------------------------------------------------
    section("Episode Outcome Distribution")
    outcome_dist: Counter = Counter()
    for r in chat_rows:
        attrs = json.loads(r["attributes"]) if r["attributes"] else {}
        outcome = attrs.get("chat", {}).get("episode_outcome", "unknown")
        outcome_dist[outcome] += 1
    for outcome, count in sorted(outcome_dist.items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {count}")
    results["outcome_dist"] = dict(outcome_dist)

    # ----------------------------------------------------------------
    # Reward attribute vs annotation verification
    # ----------------------------------------------------------------
    section("Reward Verification")

    total_eff_count = conn.execute(
        "SELECT count(*) FROM span_annotations WHERE label = 'total_effective_reward'"
    ).fetchone()[0]
    print(
        f"  Has total_effective_reward annotation: {total_eff_count}/{total_episodes}"
    )
    if total_eff_count == 0:
        print("  ** WARNING: No total_effective_reward annotations. Deploy Step 2 fix.")
    elif total_eff_count < total_episodes:
        print(
            f"  ** NOTE: Only {total_eff_count}/{total_episodes} episodes have annotations."
            f" Likely a Phoenix ingestion race condition. Check retry config."
        )

    # Verify annotation matches span attribute where both exist
    reward_mismatch = 0
    chat_spans_with_rowid = conn.execute(
        "SELECT id, attributes FROM spans WHERE name = 'chat_request'"
    ).fetchall()
    for s in chat_spans_with_rowid:
        attrs = json.loads(s["attributes"]) if s["attributes"] else {}
        span_reward = attrs.get("training", {}).get("reward")
        ann = conn.execute(
            "SELECT score FROM span_annotations WHERE span_rowid = ? AND label = 'total_effective_reward'",
            (s["id"],),
        ).fetchone()
        if ann and span_reward is not None:
            if abs(ann["score"] - span_reward) > 0.01:
                reward_mismatch += 1

    if reward_mismatch > 0:
        print(f"  ** WARNING: {reward_mismatch} episodes where annotation != attribute")

    # Check per-reward annotation duplication
    label_counts = conn.execute(
        "SELECT label, count(*) as cnt FROM span_annotations GROUP BY label ORDER BY cnt DESC"
    ).fetchall()
    print("\n  Annotation label distribution:")
    for lc in label_counts:
        print(f"    {lc['label']}: {lc['cnt']}")

    results["has_total_effective_annotation"] = total_eff_count
    results["reward_mismatches"] = reward_mismatch

    # ----------------------------------------------------------------
    # Tool-calling stats
    # ----------------------------------------------------------------
    section("Tool-Calling Stats")
    tc_dist: Counter = Counter()
    zero_tools = 0
    for r in chat_rows:
        attrs = json.loads(r["attributes"]) if r["attributes"] else {}
        tc = attrs.get("chat", {}).get("total_tool_calls", 0)
        tc_dist[tc] += 1
        if tc == 0:
            zero_tools += 1

    print(f"  Episodes with 0 tool calls: {pct(zero_tools)}")
    print(f"  Tool call distribution:")
    for k in sorted(tc_dist.keys()):
        print(f"    {k:3d} calls: {tc_dist[k]:4d} episodes")
    results["zero_tool_episodes"] = zero_tools

    # Tool name distribution. Exclude tool_service.http (a synthetic
    # child span for per-attempt HTTP wall introduced 2026-05-27 -- it
    # is not a model-emitted tool name).
    tool_rows = conn.execute(
        "SELECT name, count(*) as cnt FROM spans "
        "WHERE name LIKE 'tool.%' AND name != 'tool_service.http' "
        "GROUP BY name ORDER BY cnt DESC"
    ).fetchall()
    print(f"\n  Tool name distribution:")
    known_tools = {"tool.vina_dock", "tool.python_execute", "tool.molecule_info"}
    hallucinated = []
    for t in tool_rows:
        marker = "" if t["name"] in known_tools else " ** HALLUCINATED"
        print(f"    {t['name']}: {t['cnt']}{marker}")
        if t["name"] not in known_tools:
            hallucinated.append(t["name"])
    results["hallucinated_tools"] = hallucinated

    # Telemetry tree generation: live vs synthetic (2026-05-27 marker)
    live_spans = conn.execute(
        "SELECT count(*) FROM spans WHERE name = 'policy_llm.responses'"
    ).fetchone()[0]
    synthetic_iter_spans = conn.execute(
        "SELECT count(*) FROM spans WHERE name LIKE 'reasoning_iteration.%'"
    ).fetchone()[0]
    print(
        f"\n  Telemetry tree: live policy_llm.responses={live_spans}, "
        f"synthetic reasoning_iteration.*={synthetic_iter_spans}"
    )
    results["live_policy_llm_spans"] = live_spans
    results["synthetic_iter_spans"] = synthetic_iter_spans

    # Error rates
    tool_errors = conn.execute(
        "SELECT name, count(*) FROM spans WHERE name LIKE 'tool.%' AND status_code = 'ERROR' GROUP BY name"
    ).fetchall()
    tool_totals = {t["name"]: t["cnt"] for t in tool_rows}
    print(f"\n  Tool error rates:")
    for t in tool_errors:
        total = tool_totals.get(t["name"], 1)
        rate = t[1] / total * 100
        print(f"    {t['name']}: {t[1]}/{total} ({rate:.1f}%)")
    results["tool_error_rates"] = {
        t["name"]: t[1] / tool_totals.get(t["name"], 1) for t in tool_errors
    }

    # ----------------------------------------------------------------
    # Tool-service host distribution (live tool_service.http only)
    # ----------------------------------------------------------------
    section("Tool-Service Host Distribution")
    host_rows = conn.execute(
        "SELECT attributes FROM spans WHERE name = 'tool_service.http' LIMIT 50000"
    ).fetchall()
    host_counts: Counter = Counter()
    for r in host_rows:
        attrs = json.loads(r["attributes"]) if r["attributes"] else {}
        tool_attrs = attrs.get("tool", {}) if isinstance(attrs, dict) else {}
        host = tool_attrs.get("service_host", "?")
        tool = tool_attrs.get("name", "?")
        host_counts[(tool, host)] += 1
    if host_counts:
        for (tool, host), cnt in sorted(
            host_counts.items(), key=lambda kv: (kv[0][0], -kv[1])
        ):
            print(f"    {tool}@{host}: {cnt}")
    else:
        print("  (no tool_service.http spans -- pre-2026-05-27 run)")
    results["tool_service_host_distribution"] = {
        f"{t}@{h}": c for (t, h), c in host_counts.items()
    }

    # ----------------------------------------------------------------
    # Span duration check
    # ----------------------------------------------------------------
    section("Span Duration Check")
    chat_spans = conn.execute(
        "SELECT start_time, end_time FROM spans WHERE name = 'chat_request' LIMIT 10"
    ).fetchall()
    proper = 0
    for s in chat_spans:
        try:
            st = datetime.fromisoformat(str(s["start_time"]).replace("+00:00", ""))
            et = datetime.fromisoformat(str(s["end_time"]).replace("+00:00", ""))
            dur = (et - st).total_seconds()
            if dur > 0:
                proper += 1
        except (ValueError, TypeError):
            pass
    print(f"  Spans with proper duration (of first 10): {proper}/10")

    # ----------------------------------------------------------------
    # Reward distribution
    # ----------------------------------------------------------------
    section("Reward Distribution")
    reward_vals: Counter = Counter()
    for r in chat_rows:
        attrs = json.loads(r["attributes"]) if r["attributes"] else {}
        reward = attrs.get("training", {}).get("reward")
        if reward is not None:
            reward_vals[round(reward, 2)] += 1
    for rv in sorted(reward_vals.keys()):
        print(f"  reward={rv}: {reward_vals[rv]} episodes")

    conn.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    section("SUMMARY")
    issues = []
    if has_system_prompt == 0:
        issues.append("System prompt not recorded (Step 1)")
    if has_tool_schemas == 0:
        issues.append("Tool schemas not recorded (Step 1)")
    if results.get("has_total_effective_annotation", 0) == 0:
        issues.append("No total_effective_reward annotation (Step 2)")
    if has_episode_outcome == 0:
        issues.append("No episode_outcome attribute (Step 3)")
    if zero_tools > 0:
        issues.append(f"{zero_tools} episodes with zero tool calls (Step 4)")
    if hallucinated:
        issues.append(f"Hallucinated tool names: {hallucinated}")

    if issues:
        print("  Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  All checks passed!")

    results["issues"] = issues
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <phoenix.db>")
        sys.exit(1)

    db = sys.argv[1]
    if not Path(db).exists():
        print(f"Error: {db} not found")
        sys.exit(1)

    run_analysis(db)
