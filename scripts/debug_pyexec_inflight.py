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

"""Debug-mode diagnostic for v0529d pyexec max-in-flight = 112.

Loads the v0529d phoenix.db snapshot and emits NDJSON evidence to the
debug log path so we can evaluate the following hypotheses:

  H1: 112 is "real" client-perspective in-flight = HTTP queueing + long-tail
      execution; long-tail spans (p90=132s) accumulate over time.
  H2: profile_run.py over-counts due to span duplication (multiple span
      names lumped together, or duplicate rows).
  H3: Zombie spans -- end_time set far after actual completion.
  H4: At max-tick, the 112 spans are a mix of long-running spans started
      minutes earlier + a fresh burst.
  H5: Clock skew between resource server and simple_agent.

Usage:
    python3 scripts/debug_pyexec_inflight.py

Writes one NDJSON line per evidence record to:
    /workspaces/aevom/.cursor/debug-64c20c.log
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = "/tmp/aevom_profile/snap_phoenix_1780172502.db"
LOG_PATH = "/workspaces/aevom/.cursor/debug-64c20c.log"
SESSION_ID = "64c20c"

# v0529d steady-state window (1h:2h relative; this is the absolute window the
# profile_run.py invocation used, taken from the report header):
WIN_START_UTC = "2026-05-30T17:22:44.745994+00:00"
WIN_END_UTC = "2026-05-30T18:22:44.745994+00:00"

# Pyexec bucket: matches scripts/profile_run.py:TOOL_BUCKETS
PYEXEC_TOOL_SUFFIXES = {"python_execute", "stateful_python_code_exec", "python_exexecute"}

CONCURRENCY_TICK_S = 5


def log(hypothesis: str, message: str, data: dict) -> None:
    rec = {
        "sessionId": SESSION_ID,
        "runId": "v0529d-pyexec-investigation",
        "hypothesisId": hypothesis,
        "location": "debug_pyexec_inflight.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def parse_iso_to_unix(iso: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def ts_to_unix_expr(col: str) -> str:
    # Phoenix stores timestamps as ISO strings. Convert to unix float.
    return (
        f"CAST(strftime('%s', {col}) AS REAL) + "
        f"CAST(substr({col}, 21, 6) AS REAL) / 1e6"
    )


def main() -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    win_start = parse_iso_to_unix(WIN_START_UTC)
    win_end = parse_iso_to_unix(WIN_END_UTC)

    log("setup", "diagnostic startup", {
        "db_path": DB_PATH,
        "win_start_unix": win_start,
        "win_end_unix": win_end,
        "win_duration_s": win_end - win_start,
    })

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # --- 1. List the schema / table to confirm we're querying the right thing
    schema = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    log("setup", "phoenix.db tables", {"tables": [r["name"] for r in schema]})

    # --- 2. Hypothesis H2: span-name distribution within the "pyexec" bucket
    #
    # profile_run.py groups any span with name LIKE 'tool.%' AND name !=
    # 'tool_service.http' whose tool suffix maps to tool_pyexec.
    sql = f"""
        SELECT name,
               COUNT(*) AS n_spans,
               COUNT(DISTINCT span_id) AS n_unique_span_ids,
               COUNT(DISTINCT trace_rowid) AS n_unique_traces,
               MIN({ts_to_unix_expr('start_time')}) AS s_min,
               MAX({ts_to_unix_expr('start_time')}) AS s_max
          FROM spans
         WHERE end_time IS NOT NULL
           AND name LIKE 'tool.%'
           AND name != 'tool_service.http'
           AND ({ts_to_unix_expr('start_time')}) < ?
           AND ({ts_to_unix_expr('end_time')}) > ?
         GROUP BY name
         ORDER BY n_spans DESC
    """
    rows = conn.execute(sql, (win_end, win_start)).fetchall()
    name_dist = []
    for r in rows:
        name = r["name"]
        is_pyexec = name.startswith("tool.") and name[5:] in PYEXEC_TOOL_SUFFIXES
        name_dist.append({
            "name": name,
            "n_spans": r["n_spans"],
            "n_unique_span_ids": r["n_unique_span_ids"],
            "n_unique_traces": r["n_unique_traces"],
            "is_pyexec_bucket": is_pyexec,
        })
    log("H2", "span-name distribution for tool.* (window)", {
        "all_tool_names": name_dist,
        "pyexec_total_spans": sum(d["n_spans"] for d in name_dist if d["is_pyexec_bucket"]),
        "pyexec_unique_span_ids": sum(d["n_unique_span_ids"] for d in name_dist if d["is_pyexec_bucket"]),
    })

    # --- 3. Load the actual pyexec span intervals
    sql_pyexec = f"""
        SELECT span_id,
               trace_rowid,
               name,
               {ts_to_unix_expr('start_time')} AS s,
               {ts_to_unix_expr('end_time')}   AS e,
               parent_id
          FROM spans
         WHERE end_time IS NOT NULL
           AND name IN ('tool.python_execute', 'tool.stateful_python_code_exec', 'tool.python_exexecute')
           AND ({ts_to_unix_expr('start_time')}) < ?
           AND ({ts_to_unix_expr('end_time')}) > ?
    """
    pyexec_rows = conn.execute(sql_pyexec, (win_end, win_start)).fetchall()
    pyexec_spans = []
    for r in pyexec_rows:
        s = float(r["s"])
        e = float(r["e"])
        pyexec_spans.append({
            "span_id": r["span_id"],
            "trace_rowid": r["trace_rowid"],
            "name": r["name"],
            "s": s,
            "e": e,
            "dur": e - s,
            "parent_id": r["parent_id"],
        })
    log("setup", "pyexec spans loaded", {
        "n_pyexec_spans": len(pyexec_spans),
    })

    # --- 4. Hypothesis H2: duplicate span detection
    span_id_counter = Counter(r["span_id"] for r in pyexec_spans)
    dup_span_ids = {sid: n for sid, n in span_id_counter.items() if n > 1}
    log("H2", "duplicate span_id detection within pyexec bucket", {
        "n_total": len(pyexec_spans),
        "n_unique_span_ids": len(span_id_counter),
        "n_duplicates": len(dup_span_ids),
        "dup_sample": list(dup_span_ids.items())[:5],
    })

    # --- 5. Hypothesis H3: duration distribution
    durations = sorted([sp["dur"] for sp in pyexec_spans])
    n = len(durations)
    def pct(p):
        return durations[min(n - 1, int(n * p))] if n else 0
    bucket_count = Counter()
    bucket_edges = [(0, 1), (1, 5), (5, 30), (30, 60), (60, 120), (120, 300), (300, 1800)]
    for d in durations:
        for lo, hi in bucket_edges:
            if lo <= d < hi:
                bucket_count[f"[{lo},{hi})s"] += 1
                break
        else:
            bucket_count[">=1800s"] += 1
    log("H1", "pyexec span duration distribution", {
        "n": n,
        "min": durations[0] if n else None,
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": durations[-1] if n else None,
        "mean": sum(durations) / n if n else 0,
        "buckets": dict(bucket_count),
    })

    # --- 6. Compute in-flight series at 5s ticks (mirror profile_run.py)
    n_ticks = max(1, int((win_end - win_start) / CONCURRENCY_TICK_S) + 1)
    delta = [0] * (n_ticks + 1)
    for sp in pyexec_spans:
        s = max(sp["s"], win_start)
        e = min(sp["e"], win_end)
        if s >= e:
            continue
        si = int((s - win_start) / CONCURRENCY_TICK_S)
        ei = int((e - win_start) / CONCURRENCY_TICK_S) + 1
        if 0 <= si < n_ticks:
            delta[si] += 1
        if 0 <= ei <= n_ticks:
            delta[ei] -= 1
    series = []
    running = 0
    for i in range(n_ticks):
        running += delta[i]
        series.append(running)
    max_inflight = max(series) if series else 0
    max_ticks = [i for i, v in enumerate(series) if v == max_inflight]
    log("setup", "reproduced max_inflight from raw spans", {
        "max_inflight": max_inflight,
        "ticks_at_max": max_ticks[:5],
        "n_ticks_at_max": len(max_ticks),
        "tick_size_s": CONCURRENCY_TICK_S,
    })

    # --- 7. Hypothesis H4: at the max-in-flight tick, what are the spans?
    if max_ticks:
        peak_tick = max_ticks[0]
        peak_unix = win_start + peak_tick * CONCURRENCY_TICK_S
        # A span is "open at peak tick" if s <= peak_unix < e + tick_s
        # (matches the delta-array semantics in profile_run.py)
        peak_window_start = peak_unix
        peak_window_end = peak_unix + CONCURRENCY_TICK_S
        open_at_peak = [
            sp for sp in pyexec_spans
            if sp["s"] < peak_window_end and sp["e"] + CONCURRENCY_TICK_S > peak_window_start
        ]
        # Bucket: how long had each span been running BEFORE the peak tick?
        age_buckets = Counter()
        future_buckets = Counter()  # how long after peak does each span persist?
        for sp in open_at_peak:
            age = peak_unix - sp["s"]  # negative if span starts in future of tick
            future = sp["e"] - peak_unix
            age_buckets[
                "<0s" if age < 0 else
                "[0,5)s" if age < 5 else
                "[5,30)s" if age < 30 else
                "[30,60)s" if age < 60 else
                "[60,120)s" if age < 120 else
                "[120,300)s" if age < 300 else
                ">=300s"
            ] += 1
            future_buckets[
                "[0,5)s" if future < 5 else
                "[5,30)s" if future < 30 else
                "[30,60)s" if future < 60 else
                "[60,120)s" if future < 120 else
                "[120,300)s" if future < 300 else
                ">=300s"
            ] += 1
        log("H4", "spans alive at peak tick (max_inflight)", {
            "peak_tick_index": peak_tick,
            "peak_tick_unix": peak_unix,
            "n_open_at_peak": len(open_at_peak),
            "age_before_peak_buckets": dict(age_buckets),
            "remaining_after_peak_buckets": dict(future_buckets),
            "sample_first_5": [
                {"sid": sp["span_id"][:16], "s_before_peak_s": peak_unix - sp["s"],
                 "e_after_peak_s": sp["e"] - peak_unix, "dur_s": sp["dur"]}
                for sp in sorted(open_at_peak, key=lambda x: x["s"])[:5]
            ],
        })

    # --- 8. Hypothesis H1 component: in-flight without long-tail spans
    # Recompute series excluding spans with duration >= 30s
    delta_short = [0] * (n_ticks + 1)
    delta_long = [0] * (n_ticks + 1)
    for sp in pyexec_spans:
        s = max(sp["s"], win_start)
        e = min(sp["e"], win_end)
        if s >= e:
            continue
        si = int((s - win_start) / CONCURRENCY_TICK_S)
        ei = int((e - win_start) / CONCURRENCY_TICK_S) + 1
        target = delta_short if sp["dur"] < 30 else delta_long
        if 0 <= si < n_ticks:
            target[si] += 1
        if 0 <= ei <= n_ticks:
            target[ei] -= 1
    s_short, r_short = [], 0
    s_long, r_long = [], 0
    for i in range(n_ticks):
        r_short += delta_short[i]
        r_long += delta_long[i]
        s_short.append(r_short)
        s_long.append(r_long)
    log("H1", "in-flight decomposed by span duration",
        {
            "max_inflight_total": max_inflight,
            "max_inflight_short_only_lt30s": max(s_short),
            "max_inflight_long_only_ge30s": max(s_long),
            "mean_inflight_short": sum(s_short) / len(s_short),
            "mean_inflight_long": sum(s_long) / len(s_long),
            "n_short_spans": sum(1 for sp in pyexec_spans if sp["dur"] < 30),
            "n_long_spans": sum(1 for sp in pyexec_spans if sp["dur"] >= 30),
        })

    # --- 9. Hypothesis H3: cross-check pyexec span vs its tool_service.http child
    # tool_service.http should be a child of tool.python_execute (or sibling).
    # If pyexec span end_time >> tool_service.http end_time, the pyexec span
    # might be "zombie" -- end_time set artificially late.
    sql_http = f"""
        SELECT span_id, trace_rowid, parent_id,
               {ts_to_unix_expr('start_time')} AS s,
               {ts_to_unix_expr('end_time')} AS e
          FROM spans
         WHERE name = 'tool_service.http'
           AND end_time IS NOT NULL
           AND ({ts_to_unix_expr('start_time')}) < ?
           AND ({ts_to_unix_expr('end_time')}) > ?
    """
    http_rows = conn.execute(sql_http, (win_end, win_start)).fetchall()
    # group http spans by parent
    http_by_parent: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in http_rows:
        if r["parent_id"]:
            http_by_parent[r["parent_id"]].append((float(r["s"]), float(r["e"])))

    pyexec_spans_with_http = 0
    pyexec_dur_minus_http_dur = []  # tool.python_execute dur - max(tool_service.http) dur
    py_end_after_http_end = []  # pyexec.end - max(http.end)
    for sp in pyexec_spans:
        children = http_by_parent.get(sp["span_id"], [])
        if children:
            pyexec_spans_with_http += 1
            http_total = sum(e - s for s, e in children)
            http_end_max = max(e for _, e in children)
            pyexec_dur_minus_http_dur.append(sp["dur"] - http_total)
            py_end_after_http_end.append(sp["e"] - http_end_max)
    log("H3", "pyexec span vs tool_service.http child timing", {
        "n_pyexec_spans": len(pyexec_spans),
        "n_pyexec_with_http_child": pyexec_spans_with_http,
        "n_pyexec_without_http_child": len(pyexec_spans) - pyexec_spans_with_http,
        "pyexec_dur_minus_http_total_p50": sorted(pyexec_dur_minus_http_dur)[len(pyexec_dur_minus_http_dur)//2] if pyexec_dur_minus_http_dur else None,
        "pyexec_dur_minus_http_total_p99": sorted(pyexec_dur_minus_http_dur)[int(len(pyexec_dur_minus_http_dur)*0.99)] if pyexec_dur_minus_http_dur else None,
        "pyexec_dur_minus_http_total_max": max(pyexec_dur_minus_http_dur) if pyexec_dur_minus_http_dur else None,
        "pyexec_end_after_http_end_p50": sorted(py_end_after_http_end)[len(py_end_after_http_end)//2] if py_end_after_http_end else None,
        "pyexec_end_after_http_end_p99": sorted(py_end_after_http_end)[int(len(py_end_after_http_end)*0.99)] if py_end_after_http_end else None,
        "pyexec_end_after_http_end_max": max(py_end_after_http_end) if py_end_after_http_end else None,
    })

    # --- 10. H5: clock skew check
    # Check whether pyexec span end_time is before its parent's start_time
    # (impossible without clock skew). And whether tool_service.http
    # start_time is before its parent tool.python_execute start_time.
    sql_parent_of_pyexec = f"""
        SELECT s.span_id AS child_id,
               s.parent_id AS parent_id,
               {ts_to_unix_expr('s.start_time')} AS child_s,
               {ts_to_unix_expr('s.end_time')}   AS child_e,
               {ts_to_unix_expr('p.start_time')} AS parent_s,
               {ts_to_unix_expr('p.end_time')}   AS parent_e,
               p.name AS parent_name
          FROM spans s
          LEFT JOIN spans p ON s.parent_id = p.span_id
         WHERE s.name IN ('tool.python_execute','tool.stateful_python_code_exec','tool.python_exexecute')
           AND s.end_time IS NOT NULL
           AND ({ts_to_unix_expr('s.start_time')}) < ?
           AND ({ts_to_unix_expr('s.end_time')}) > ?
    """
    parent_rows = conn.execute(sql_parent_of_pyexec, (win_end, win_start)).fetchall()
    skew_evidence = {
        "n_pyexec_total": len(parent_rows),
        "n_pyexec_with_parent": 0,
        "n_pyexec_orphan": 0,
        "n_child_starts_before_parent": 0,
        "n_child_ends_after_parent": 0,
        "parent_name_distribution": Counter(),
        "max_skew_child_before_parent_s": 0.0,
        "max_skew_child_after_parent_s": 0.0,
    }
    for r in parent_rows:
        if r["parent_id"] and r["parent_s"] is not None:
            skew_evidence["n_pyexec_with_parent"] += 1
            skew_evidence["parent_name_distribution"][r["parent_name"] or "<unknown>"] += 1
            if r["child_s"] < r["parent_s"]:
                skew_evidence["n_child_starts_before_parent"] += 1
                skew_evidence["max_skew_child_before_parent_s"] = max(
                    skew_evidence["max_skew_child_before_parent_s"],
                    r["parent_s"] - r["child_s"],
                )
            if r["parent_e"] is not None and r["child_e"] > r["parent_e"]:
                skew_evidence["n_child_ends_after_parent"] += 1
                skew_evidence["max_skew_child_after_parent_s"] = max(
                    skew_evidence["max_skew_child_after_parent_s"],
                    r["child_e"] - r["parent_e"],
                )
        else:
            skew_evidence["n_pyexec_orphan"] += 1
    skew_evidence["parent_name_distribution"] = dict(skew_evidence["parent_name_distribution"])
    log("H5", "pyexec parent/child clock skew check", skew_evidence)

    # --- 11. Burst-arrival test: arrival count per 5s tick around the peak
    # If the user's intuition is right, the peak isn't from a sudden burst
    # of 100+ arrivals; it's from long-tail accumulation.
    arrivals_per_tick = Counter()
    for sp in pyexec_spans:
        if win_start <= sp["s"] < win_end:
            ti = int((sp["s"] - win_start) / CONCURRENCY_TICK_S)
            arrivals_per_tick[ti] += 1
    if max_ticks:
        peak_tick = max_ticks[0]
        # Sample 12 ticks (60s) centered on the peak
        window = [(t, arrivals_per_tick.get(t, 0)) for t in range(max(0, peak_tick - 6), peak_tick + 6)]
        log("H1", "arrival rate around peak tick", {
            "peak_tick": peak_tick,
            "arrivals_per_tick_around_peak": window,
            "max_arrivals_in_any_tick": max(arrivals_per_tick.values()) if arrivals_per_tick else 0,
            "mean_arrivals_per_tick": (
                sum(arrivals_per_tick.values()) / len(arrivals_per_tick)
                if arrivals_per_tick else 0
            ),
        })

    # --- 12. Drill into the long-tail spans: examine actual examples
    # Pull 10 long-tail pyexec spans + their tool_service.http children +
    # parent agent.turn, see if their durations match HTTP children timing.
    sql_long_tail = f"""
        SELECT s.span_id, s.trace_rowid, s.parent_id, s.name AS sname,
               s.status_code, s.status_message,
               {ts_to_unix_expr('s.start_time')} AS pstart,
               {ts_to_unix_expr('s.end_time')}   AS pend,
               (CAST(strftime('%s', s.end_time) AS REAL) - CAST(strftime('%s', s.start_time) AS REAL))
                 + (CAST(substr(s.end_time, 21, 6) AS REAL) - CAST(substr(s.start_time, 21, 6) AS REAL))/1e6 AS pdur
          FROM spans s
         WHERE s.name = 'tool.python_execute'
           AND s.end_time IS NOT NULL
           AND ({ts_to_unix_expr('s.start_time')}) < ?
           AND ({ts_to_unix_expr('s.end_time')}) > ?
         ORDER BY pdur DESC
         LIMIT 10
    """
    long_rows = conn.execute(sql_long_tail, (win_end, win_start)).fetchall()
    examples = []
    for r in long_rows:
        sid = r["span_id"]
        # get tool_service.http children
        kids = conn.execute(
            f"""SELECT name, status_code,
                       {ts_to_unix_expr('start_time')} AS s,
                       {ts_to_unix_expr('end_time')} AS e,
                       attributes
                  FROM spans WHERE parent_id = ?
                 ORDER BY start_time""",
            (sid,),
        ).fetchall()
        kid_list = []
        for k in kids:
            try:
                kattrs = json.loads(k["attributes"]) if k["attributes"] else {}
            except Exception:
                kattrs = {}
            kid_list.append({
                "name": k["name"],
                "status": k["status_code"],
                "s_rel_to_parent": float(k["s"]) - float(r["pstart"]),
                "e_rel_to_parent": float(k["e"]) - float(r["pstart"]),
                "dur": float(k["e"]) - float(k["s"]),
                "attempt": (kattrs.get("tool", {}) or {}).get("attempt"),
                "status_code": (kattrs.get("tool", {}) or {}).get("http", {}).get("status_code")
                    if isinstance((kattrs.get("tool", {}) or {}).get("http"), dict) else None,
            })
        # check gap: from parent_start to first child_start; last_child_end to parent_end
        if kid_list:
            pre = kid_list[0]["s_rel_to_parent"]
            post = r["pdur"] - kid_list[-1]["e_rel_to_parent"]
            between = []
            for i in range(1, len(kid_list)):
                between.append(kid_list[i]["s_rel_to_parent"] - kid_list[i-1]["e_rel_to_parent"])
        else:
            pre, post, between = None, None, []
        examples.append({
            "span_id": sid[:16],
            "parent_id": (r["parent_id"] or "")[:16],
            "duration": r["pdur"],
            "status": r["status_code"],
            "n_http_children": len(kid_list),
            "pre_first_child_s": pre,
            "between_children_s": between,
            "post_last_child_s": post,
            "http_children": kid_list,
        })
    log("H3", "10 longest pyexec spans -- full lifecycle", {"spans": examples})

    # --- 13. End-time clustering test
    # If the 217 long-tail spans (>120s) end uniformly, their end_times
    # would scatter across all 720 5s ticks. Bucket their end_times by
    # 5s tick and see if there's concentration.
    long_tail_spans = [sp for sp in pyexec_spans if sp["dur"] >= 120]
    end_tick_count = Counter()
    for sp in long_tail_spans:
        if win_start <= sp["e"] < win_end:
            ti = int((sp["e"] - win_start) / CONCURRENCY_TICK_S)
            end_tick_count[ti] += 1
    # Top 10 most-populated end ticks
    top_end_ticks = end_tick_count.most_common(10)
    log("H1", "long-tail (>=120s) end-time clustering", {
        "n_long_tail_total": len(long_tail_spans),
        "n_long_tail_in_window_end": sum(end_tick_count.values()),
        "n_distinct_end_ticks": len(end_tick_count),
        "top_10_end_ticks_count": top_end_ticks,
        "max_long_tail_endings_in_one_5s_tick": top_end_ticks[0][1] if top_end_ticks else 0,
    })

    # --- 14. Arrival burstiness across the whole window
    # How concentrated are arrivals? If "bursty refit" pattern, most ticks
    # should have 0 arrivals and a few ticks should have many.
    arrival_counts = list(arrivals_per_tick.values())
    n_ticks_with_arrivals = len(arrival_counts)
    log("H1", "arrival burstiness across whole window", {
        "n_5s_ticks_total": n_ticks,
        "n_5s_ticks_with_any_arrival": n_ticks_with_arrivals,
        "pct_ticks_with_arrivals": n_ticks_with_arrivals / n_ticks * 100,
        "mean_arrivals_when_present": sum(arrival_counts) / max(1, n_ticks_with_arrivals),
        "max_arrivals_in_any_5s_tick": max(arrival_counts) if arrival_counts else 0,
        "p90_arrivals_when_present": sorted(arrival_counts)[int(len(arrival_counts) * 0.9)] if arrival_counts else 0,
        "p99_arrivals_when_present": sorted(arrival_counts)[int(len(arrival_counts) * 0.99)] if arrival_counts else 0,
    })

    # --- 15. Same-trace concurrent pyexec calls
    # If multiple pyexec spans share the same chat_request trace AND overlap
    # in time, that's a "single rollout firing multiple parallel tool calls"
    # case. Each rollout normally does ONE tool call per turn (serial). If
    # 100 rollouts each have 1 active pyexec call, that's 100 in-flight.
    # If 50 rollouts each have 2 parallel pyexec calls, that's 100 in-flight
    # but only 50 actual rollouts.
    spans_by_trace = defaultdict(list)
    for sp in pyexec_spans:
        spans_by_trace[sp["trace_rowid"]].append(sp)
    overlap_per_trace = Counter()
    for tid, spans in spans_by_trace.items():
        n = len(spans)
        # count max concurrent spans within this trace
        events = []
        for sp in spans:
            events.append((sp["s"], +1))
            events.append((sp["e"], -1))
        events.sort()
        running = 0
        max_concurrent = 0
        for _, d in events:
            running += d
            max_concurrent = max(max_concurrent, running)
        overlap_per_trace[max_concurrent] += 1
    log("H1", "max concurrent pyexec spans within a single trace", {
        "n_traces_with_pyexec": len(spans_by_trace),
        "distribution_max_concurrent_pyexec_per_trace": dict(overlap_per_trace),
        "n_traces_with_parallel_pyexec": sum(c for k, c in overlap_per_trace.items() if k >= 2),
    })

    log("setup", "diagnostic done", {"n_lines_written": "see file"})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "sessionId": SESSION_ID,
                "runId": "v0529d-pyexec-investigation",
                "hypothesisId": "ERROR",
                "location": "debug_pyexec_inflight.py",
                "message": f"diagnostic crashed: {e}",
                "data": {"traceback": traceback.format_exc()},
                "timestamp": int(time.time() * 1000),
            }) + "\n")
        sys.exit(1)
