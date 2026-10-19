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

"""Profile an Aevom training run from its Phoenix telemetry DB.

Reads `phoenix.db` (local or remote via `host:path`), optionally slices by
a wall-clock window, and emits a markdown report (and optional JSON) that
identifies the run's primary bottleneck by combining:

  * Component wall share         (where is chat-request time going?)
  * Worker-busy / queue-pressure (which components are back-pressured?)
  * Wall inflation               (do client spans include queue wait?)
  * Per-rollout phase breakdown  (where is "unaccounted" time hiding?)
  * Throughput summary           (inference EPH, trainer EPH, efficiency)

Spans are client-side, so `in_flight` = `workers_busy + queue_depth`.
Capacity (`est_workers`) is taken from `--workers <name>=N,...` when
provided; otherwise estimated as the mode of in-flight on high-load ticks.

Usage:

    # Full run, single command, profile + bottleneck identification
    python scripts/profile_run.py \\
        cluster:~/runs/v0528c-pool-and-parser/5345125-logs/phoenix_db/phoenix.db \\
        --logs-dir cluster:~/runs/v0528c-pool-and-parser/5345125-logs

    # Just the first 30 minutes
    python scripts/profile_run.py /tmp/phoenix.db --window 0s:30m

    # Pin known worker capacities for accurate back-pressure metrics
    python scripts/profile_run.py /tmp/phoenix.db \\
        --workers vllm=416,judge=64,pyexec=128,vina=32
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Component taxonomy
# --------------------------------------------------------------------------

# Map of `tool.{name}` -> component bucket. Anything not listed -> "tool_other".
TOOL_BUCKETS: dict[str, str] = {
    "python_execute": "tool_pyexec",
    "stateful_python_code_exec": "tool_pyexec",
    "python_exexecute": "tool_pyexec",  # typo seen in v0528c data
    "vina_dock": "tool_vina",
    "dock": "tool_vina",
    "vina": "tool_vina",
    "molecule_info": "tool_rdkit",
    "structure_query": "tool_rdkit",
    "struture_query": "tool_rdkit",  # typo seen in v0528c data
    "smiles_to_image": "tool_rdkit",
    "rdkit_ops": "tool_rdkit",
    "visualize_pdb": "tool_molstar",
    "visualize": "tool_molstar",
    "visualize_3d": "tool_molstar",
    "aevom.visualize": "tool_molstar",
    "aevom.load": "tool_molstar",
}

# Ordered list of components the bottleneck table reports.
ALL_COMPONENTS = [
    "vllm",
    "judge",
    "tool_pyexec",
    "tool_vina",
    "tool_rdkit",
    "tool_molstar",
    "tool_other",
]

# Tick resolutions
CONCURRENCY_TICK_S = 5
TIMESERIES_TICK_S = 30
IDLE_GAP_MIN_S = 30


def tool_to_component(tool_name: str) -> str:
    return TOOL_BUCKETS.get(tool_name, "tool_other")


# --------------------------------------------------------------------------
# DBLocation: parse <host>:<path> and fetch + snapshot
# --------------------------------------------------------------------------


@dataclass
class DBLocation:
    raw: str
    host: Optional[str]
    remote_path: Optional[str]
    local_path: Path  # original DB on disk (post-fetch for remote)


def parse_remote_spec(spec: str) -> tuple[Optional[str], str]:
    """Parse `host:path` into (host, path). Returns (None, path) for local."""
    if ":" not in spec:
        return None, spec
    head, sep, tail = spec.partition(":")
    # Treat single-letter Windows drive prefix as local. We're on linux so
    # this is rare but cheap to guard against.
    if len(head) == 1 and head.isalpha():
        return None, spec
    return head, tail


def _ssh_fetch(remote: str, local: Path) -> None:
    """scp remote -> local. Errors propagate."""
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] fetching {remote} -> {local}", file=sys.stderr)
    subprocess.check_call(["scp", "-q", remote, str(local)])


def resolve_db(spec: str, cache_dir: Path) -> DBLocation:
    """Resolve a db_location (local path or host:path) -> local snapshot.

    For remote: scp the db file (and sibling .db-wal / .db-shm if they
    exist, to handle live DBs). For both: take a sqlite snapshot in the
    cache dir so we never read while writers are mutating the file.
    """
    host, path = parse_remote_spec(spec)

    if host is None:
        original = Path(os.path.expanduser(path)).resolve()
        if not original.exists():
            raise FileNotFoundError(f"phoenix DB not found: {original}")
    else:
        fetch_dir = cache_dir / f"fetch_{host}_{abs(hash(path)) % 10**8}"
        fetch_dir.mkdir(parents=True, exist_ok=True)
        local = fetch_dir / Path(path).name
        _ssh_fetch(f"{host}:{path}", local)
        # Best-effort fetch the WAL + SHM sidecars (live DBs).
        for ext in (".db-wal", ".db-shm"):
            sib_remote = path[:-3] + ext if path.endswith(".db") else path + ext
            sib_local = fetch_dir / (local.name[:-3] + ext)
            try:
                _ssh_fetch(f"{host}:{sib_remote}", sib_local)
            except subprocess.CalledProcessError:
                # WAL/SHM may not exist (post-checkpoint). Ignore.
                pass
        original = local

    # Snapshot via sqlite3.backup so we get a consistent point-in-time view.
    snap = cache_dir / f"snap_{original.stem}_{int(time.time())}.db"
    snap.parent.mkdir(parents=True, exist_ok=True)
    if snap.exists():
        snap.unlink()
    src = sqlite3.connect(f"file:{original}?mode=ro", uri=True)
    dst = sqlite3.connect(str(snap))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()

    return DBLocation(
        raw=spec, host=host,
        remote_path=path if host else None, local_path=snap,
    )


# --------------------------------------------------------------------------
# Window: parse "30m:1h" / "ISO:ISO" / open-ended
# --------------------------------------------------------------------------


@dataclass
class Window:
    start_unix: float
    end_unix: float
    label: str

    @property
    def duration_s(self) -> float:
        return self.end_unix - self.start_unix

    @property
    def duration_h(self) -> float:
        return self.duration_s / 3600

    def start_iso(self) -> str:
        return datetime.fromtimestamp(self.start_unix, tz=timezone.utc).isoformat()

    def end_iso(self) -> str:
        return datetime.fromtimestamp(self.end_unix, tz=timezone.utc).isoformat()


_REL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh])\s*$", re.I)


def _parse_rel(s: str) -> Optional[float]:
    """Parse "30m", "1.5h", "120s" -> seconds. Returns None if not relative."""
    m = _REL_RE.match(s)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2).lower()
    return n * {"s": 1, "m": 60, "h": 3600}[unit]


def _parse_iso(s: str) -> Optional[float]:
    """Parse ISO 8601 -> unix seconds. Returns None if not ISO."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def resolve_window(spec: Optional[str], run_start_unix: float,
                   run_end_unix: float) -> Window:
    """Resolve a --window spec to (start, end) unix seconds.

    Formats supported:
      None or "" / "full"            -> full run
      "30m" / "1h"                   -> first 30m / first hour from run start
      ":30m" / ":1h"                 -> from run start to relative offset
      "30m:" / "1h:"                 -> from offset to run end
      "30m:1h"                       -> relative slice
      "2026-05-28T20:00Z:..."        -> absolute slice
    """
    if not spec or spec.lower() == "full":
        return Window(run_start_unix, run_end_unix, "full run")

    # Split on the last colon — but only if both sides could be absolute ISO.
    # ISO timestamps contain colons in HH:MM:SS, so split is tricky. Strategy:
    # try splitting on every colon; the parse function rejects bad halves.
    if ":" not in spec:
        # bare "30m" / "1h" / single absolute -> open-ended start window
        spec = ":" + spec  # treat as "from start to X"

    # Find split point: try parsing prefix/suffix. We split at the FIRST
    # colon that yields two parseable halves (preferring later splits for
    # absolute ISO).
    candidates = [i for i, c in enumerate(spec) if c == ":"]
    chosen: Optional[tuple[str, str]] = None
    for i in candidates:
        left = spec[:i]
        right = spec[i + 1:]
        left_ok = left == "" or _parse_rel(left) is not None or _parse_iso(left) is not None
        right_ok = right == "" or _parse_rel(right) is not None or _parse_iso(right) is not None
        if left_ok and right_ok:
            chosen = (left, right)
    if chosen is None:
        raise ValueError(f"--window: cannot parse {spec!r}")

    left, right = chosen

    def _resolve_endpoint(s: str, default: float) -> float:
        if s == "":
            return default
        rel = _parse_rel(s)
        if rel is not None:
            return run_start_unix + rel
        abs_ = _parse_iso(s)
        if abs_ is not None:
            return abs_
        raise ValueError(f"--window: cannot parse endpoint {s!r}")

    start = _resolve_endpoint(left, run_start_unix)
    end = _resolve_endpoint(right, run_end_unix)
    if start >= end:
        raise ValueError(f"--window: start must precede end (got {start} >= {end})")
    return Window(start, end, spec)


# --------------------------------------------------------------------------
# Worker spec parsing
# --------------------------------------------------------------------------


def parse_workers_spec(spec: Optional[str]) -> dict[str, int]:
    """Parse "--workers vllm=416,judge=64,pyexec=128,vina=32" -> dict.

    Component aliases accepted: pyexec -> tool_pyexec, vina -> tool_vina,
    rdkit -> tool_rdkit, molstar -> tool_molstar.
    """
    if not spec:
        return {}
    alias = {
        "pyexec": "tool_pyexec",
        "vina": "tool_vina",
        "rdkit": "tool_rdkit",
        "molstar": "tool_molstar",
    }
    out: dict[str, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--workers: bad pair {pair!r}, expected name=N")
        name, _, val = pair.partition("=")
        name = name.strip()
        name = alias.get(name, name)
        try:
            out[name] = int(val.strip())
        except ValueError:
            raise ValueError(f"--workers: bad capacity for {name}: {val!r}")
    return out


# --------------------------------------------------------------------------
# Span loading
# --------------------------------------------------------------------------


@dataclass
class IntervalSet:
    """A set of (start_unix, end_unix) pairs for one component."""
    name: str
    intervals: list[tuple[float, float]] = field(default_factory=list)
    walls: list[float] = field(default_factory=list)  # end - start, seconds

    def add(self, s: float, e: float) -> None:
        if e < s:
            return
        self.intervals.append((s, e))
        self.walls.append(e - s)


def _ts_to_unix_sql() -> str:
    """SQL expression converting a timestamp column to unix seconds (float)."""
    return "(julianday({col}) - 2440587.5) * 86400"


def fetch_run_bounds(conn: sqlite3.Connection) -> tuple[float, float]:
    """Return (min_chat_start_unix, max_chat_end_unix)."""
    r = conn.execute(f"""
        SELECT MIN({_ts_to_unix_sql().format(col='start_time')}) AS s,
               MAX({_ts_to_unix_sql().format(col='end_time')})   AS e
          FROM spans WHERE name = 'chat_request' AND end_time IS NOT NULL
    """).fetchone()
    if r is None or r["s"] is None:
        raise RuntimeError("phoenix.db has no completed chat_request spans")
    return float(r["s"]), float(r["e"])


def load_component_intervals(conn: sqlite3.Connection,
                             win: Window) -> dict[str, IntervalSet]:
    """Load (start, end, wall) for each component over the window.

    Components:
      vllm           <- policy_llm.responses
      judge          <- judge_evaluation
      tool_pyexec    <- tool.python_execute & aliases
      tool_vina      <- tool.vina_dock & aliases
      tool_rdkit     <- tool.molecule_info / tool.structure_query / etc.
      tool_molstar   <- tool.visualize* / tool.aevom.*
      tool_other     <- everything else under tool.* (excl. tool_service.http)
    """
    sets = {c: IntervalSet(c) for c in ALL_COMPONENTS}
    chat = IntervalSet("chat_request")

    # Single query, partition in Python.
    sql = f"""
        SELECT name,
               {_ts_to_unix_sql().format(col='start_time')} AS s,
               {_ts_to_unix_sql().format(col='end_time')}   AS e
          FROM spans
         WHERE end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
           AND ({_ts_to_unix_sql().format(col='end_time')})   > ?
           AND (
                name = 'chat_request'
             OR name = 'policy_llm.responses'
             OR name = 'judge_evaluation'
             OR (name LIKE 'tool.%' AND name != 'tool_service.http')
           )
    """
    rows = conn.execute(sql, (win.end_unix, win.start_unix)).fetchall()
    for r in rows:
        s, e = float(r["s"]), float(r["e"])
        name = r["name"]
        if name == "chat_request":
            chat.add(s, e)
            continue
        if name == "policy_llm.responses":
            sets["vllm"].add(s, e)
            continue
        if name == "judge_evaluation":
            sets["judge"].add(s, e)
            continue
        if name.startswith("tool."):
            tool = name[5:]
            comp = tool_to_component(tool)
            sets[comp].add(s, e)
    return {"chat": chat, **sets}


def load_chat_rollout_records(conn: sqlite3.Connection,
                              win: Window) -> list[dict]:
    """For each chat_request in window, return a dict with metadata used for
    top-N slowest table + completion-rate + phase breakdown."""
    sql = f"""
        SELECT trace_rowid,
               span_id,
               {_ts_to_unix_sql().format(col='start_time')} AS s,
               {_ts_to_unix_sql().format(col='end_time')}   AS e,
               attributes
          FROM spans
         WHERE name = 'chat_request' AND end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='end_time')}) >= ?
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
    """
    rows = conn.execute(sql, (win.start_unix, win.end_unix)).fetchall()
    out: list[dict] = []
    for r in rows:
        attrs = {}
        if r["attributes"]:
            try:
                attrs = json.loads(r["attributes"])
            except Exception:
                pass
        chat_attrs = attrs.get("chat", {}) or {}
        out.append({
            "trace_rowid": r["trace_rowid"],
            "span_id": r["span_id"],
            "start_unix": float(r["s"]),
            "end_unix": float(r["e"]),
            "wall_s": float(r["e"]) - float(r["s"]),
            "iterations": chat_attrs.get("total_iterations"),
            "tool_calls": chat_attrs.get("total_tool_calls"),
            "outcome": chat_attrs.get("episode_outcome"),
            "hit_limit": chat_attrs.get("hit_iteration_limit"),
        })
    return out


# Orchestration spans added 2026-05-31 (dark-span instrumentation). These
# light up the phases that profile_run.py historically reported as gaps:
#   episode.seed          (server)  -> front_pad
#   agent.fetch_state     (agent)   -> front_pad (turn 1) + interturn_gap
#   agent.tool_dispatch   (agent)   -> within-turn tool round-trip wall
#   verify.reward_tiers   (server)  -> post_last_turn -> judge_start
#   verify.edit_detection (server)  -> post_last_turn -> judge_start
#   verify.finalize       (server)  -> judge_end -> chat_end
# Runs predating the instrumentation simply produce empty buckets (the
# Orchestration breakdown section then renders a "no spans" hint).
ORCH_SPAN_NAMES = [
    "episode.seed",
    "agent.fetch_state",
    "agent.tool_dispatch",
    "verify.reward_tiers",
    "verify.edit_detection",
    "verify.finalize",
]


def load_phase_data(conn: sqlite3.Connection, win: Window) -> dict[int, dict]:
    """Per chat_request trace, gather start of first agent.turn, end of last
    agent.turn, judge start/end, plus the orchestration spans. Used to derive
    the phase breakdown and the orchestration breakdown."""
    orch_in = ",".join(f"'{n}'" for n in ORCH_SPAN_NAMES)
    sql = f"""
        SELECT trace_rowid, name,
               {_ts_to_unix_sql().format(col='start_time')} AS s,
               {_ts_to_unix_sql().format(col='end_time')}   AS e
          FROM spans
         WHERE end_time IS NOT NULL
           AND (name = 'chat_request'
                OR name LIKE 'agent.turn.%'
                OR name = 'judge_evaluation'
                OR name IN ({orch_in}))
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
           AND ({_ts_to_unix_sql().format(col='end_time')})   > ?
    """
    rows = conn.execute(sql, (win.end_unix, win.start_unix)).fetchall()
    by_trace: dict[int, dict] = defaultdict(lambda: {
        "chat": None, "turns": [], "judges": [],
        "orch": defaultdict(list),
    })
    for r in rows:
        tid = r["trace_rowid"]
        s, e = float(r["s"]), float(r["e"])
        name = r["name"]
        if name == "chat_request":
            by_trace[tid]["chat"] = (s, e)
        elif name.startswith("agent.turn."):
            by_trace[tid]["turns"].append((s, e))
        elif name == "judge_evaluation":
            by_trace[tid]["judges"].append((s, e))
        else:  # orchestration span
            by_trace[tid]["orch"][name].append((s, e))
    return by_trace


def load_judge_per_endpoint(conn: sqlite3.Connection,
                            win: Window) -> dict[str, list[float]]:
    """For each judge endpoint, return list of llm.complete wall (seconds)."""
    sql = f"""
        SELECT json_extract(attributes, '$.judge.endpoint.name') AS ep,
               ({_ts_to_unix_sql().format(col='end_time')}
                - {_ts_to_unix_sql().format(col='start_time')})  AS wall
          FROM spans
         WHERE name = 'llm.complete' AND end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
           AND ({_ts_to_unix_sql().format(col='end_time')})   > ?
    """
    rows = conn.execute(sql, (win.end_unix, win.start_unix)).fetchall()
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ep = r["ep"] or "(no_endpoint)"
        out[ep].append(float(r["wall"]))
    return out


def load_per_tool_latencies(conn: sqlite3.Connection,
                            win: Window) -> dict[str, list[float]]:
    """Per `tool.{name}` latency list. Excludes `tool_service.http`."""
    sql = f"""
        SELECT name,
               ({_ts_to_unix_sql().format(col='end_time')}
                - {_ts_to_unix_sql().format(col='start_time')}) AS wall
          FROM spans
         WHERE name LIKE 'tool.%' AND name != 'tool_service.http'
           AND end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
           AND ({_ts_to_unix_sql().format(col='end_time')})   > ?
    """
    rows = conn.execute(sql, (win.end_unix, win.start_unix)).fetchall()
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r["name"]].append(float(r["wall"]))
    return out


def load_tool_service_attempts(conn: sqlite3.Connection,
                               win: Window) -> list[dict]:
    """Per tool_service.http call: attempt #, tool name, status code, wall."""
    sql = f"""
        SELECT json_extract(attributes, '$.tool.name')             AS tool,
               json_extract(attributes, '$.tool.attempt')          AS attempt,
               json_extract(attributes, '$.tool.http.status_code') AS status,
               ({_ts_to_unix_sql().format(col='end_time')}
                - {_ts_to_unix_sql().format(col='start_time')})    AS wall
          FROM spans
         WHERE name = 'tool_service.http' AND end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
           AND ({_ts_to_unix_sql().format(col='end_time')})   > ?
    """
    rows = conn.execute(sql, (win.end_unix, win.start_unix)).fetchall()
    return [
        {"tool": r["tool"], "attempt": r["attempt"],
         "status": r["status"], "wall": float(r["wall"])}
        for r in rows
    ]


def load_per_turn_dispatch(conn: sqlite3.Connection,
                           win: Window) -> list[dict]:
    """Per `policy_llm.responses` span, pull the dispatch-behavior attrs added
    to the live agent loop on 2026-05-30:

      * ``llm.tool_calls_count``  -- N function calls the policy chose this turn
      * ``llm.has_reasoning``     -- did this turn carry a reasoning item?
      * ``llm.functions``         -- list of function names called this turn

    Returns one dict per turn (missing attrs surface as None / [])."""
    sql = f"""
        SELECT json_extract(attributes, '$.llm.tool_calls_count') AS n_calls,
               json_extract(attributes, '$.llm.has_reasoning')    AS has_reason,
               json_extract(attributes, '$.llm.functions')        AS fns
          FROM spans
         WHERE name = 'policy_llm.responses' AND end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='start_time')}) < ?
           AND ({_ts_to_unix_sql().format(col='end_time')})   > ?
    """
    rows = conn.execute(sql, (win.end_unix, win.start_unix)).fetchall()
    out: list[dict] = []
    for r in rows:
        # json_extract returns array-valued attributes as a JSON text blob.
        fns: list = []
        if r["fns"] is not None:
            raw = r["fns"]
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        fns = [str(x) for x in parsed]
                except Exception:
                    pass
            elif isinstance(raw, list):
                fns = [str(x) for x in raw]
        n = r["n_calls"]
        hr = r["has_reason"]
        out.append({
            "n_calls": int(n) if n is not None else None,
            "has_reasoning": bool(hr) if hr is not None else None,
            "functions": fns,
        })
    return out


def load_chat_completions_per_min(conn: sqlite3.Connection,
                                  win: Window) -> list[tuple[int, int]]:
    sql = f"""
        SELECT CAST({_ts_to_unix_sql().format(col='end_time')} / 60 AS INTEGER) * 60 AS bucket_s,
               COUNT(*) AS n
          FROM spans
         WHERE name = 'chat_request' AND end_time IS NOT NULL
           AND ({_ts_to_unix_sql().format(col='end_time')}) >= ?
           AND ({_ts_to_unix_sql().format(col='end_time')}) <  ?
         GROUP BY bucket_s
         ORDER BY bucket_s
    """
    rows = conn.execute(sql, (win.start_unix, win.end_unix)).fetchall()
    return [(int(r["bucket_s"]), int(r["n"])) for r in rows]


# --------------------------------------------------------------------------
# In-flight series + concurrency stats (delta-array, O(N+T))
# --------------------------------------------------------------------------


def in_flight_series(intervals: list[tuple[float, float]],
                     tick_s: int, win_start: float, win_end: float) -> list[int]:
    """Return [count_at_tick] for ticks [win_start, win_start+tick_s, ...].

    Uses delta-array prefix sum, O(N + T).
    """
    n_ticks = max(1, int((win_end - win_start) / tick_s) + 1)
    delta = [0] * (n_ticks + 1)
    for s, e in intervals:
        s = max(s, win_start)
        e = min(e, win_end)
        if s >= e:
            continue
        si = int((s - win_start) / tick_s)
        ei = int((e - win_start) / tick_s) + 1
        if 0 <= si < n_ticks:
            delta[si] += 1
        if 0 <= ei <= n_ticks:
            delta[ei] -= 1
    series = []
    running = 0
    for i in range(n_ticks):
        running += delta[i]
        series.append(running)
    return series


def percentiles(xs: list[float], qs=(0.10, 0.50, 0.75, 0.90, 0.95, 0.99)
                ) -> dict[str, float]:
    if not xs:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    s = sorted(xs)
    n = len(s)
    return {f"p{int(q * 100)}": s[min(n - 1, int(n * q))] for q in qs}


def percentiles_int(xs: list[int], qs=(0.50, 0.75, 0.90, 0.95)
                    ) -> dict[str, int]:
    if not xs:
        return {f"p{int(q * 100)}": 0 for q in qs}
    s = sorted(xs)
    n = len(s)
    return {f"p{int(q * 100)}": s[min(n - 1, int(n * q))] for q in qs}


# --------------------------------------------------------------------------
# Capacity estimation
# --------------------------------------------------------------------------


@dataclass
class CapacityEstimate:
    component: str
    workers: int
    source: str  # "provided" | "est_mode" | "underutilized"
    mean_inflight: float
    max_inflight: int
    p95_inflight: int


def estimate_capacity(component: str, series: list[int],
                      provided: Optional[int]) -> CapacityEstimate:
    """Estimate worker capacity (`est_workers`).

    The goal is an estimate of how many concurrent in-flight requests the
    component's worker pool can serve simultaneously. We never see the pool
    size directly; we only see client-side in_flight = workers_busy + queue.

    Heuristic, when not provided:

      * If max(in_flight) < 3: load was tiny -> capacity genuinely unknown.
      * Otherwise: `est_workers = p90(in_flight | in_flight > 0)`.
        Among ticks where the component had ANY load, p90 is a robust upper
        estimate of the typical "high-load" value. It is insensitive to
        outliers in the queueing tail (which would push max up) and to
        long idle stretches (which would pull the mode down to 1).

    This is still just an estimate; pct_workers_busy / pct_queueing using it
    should be read as "directional". The robust back-pressure signal is
    `wall_inflation_p50`, which doesn't depend on `est_workers`.
    """
    if not series:
        return CapacityEstimate(component, provided or 0,
                                "provided" if provided else "underutilized",
                                0.0, 0, 0)
    if provided is not None:
        return CapacityEstimate(
            component, provided, "provided",
            mean_inflight=sum(series) / len(series),
            max_inflight=max(series),
            p95_inflight=percentiles_int(series)["p95"],
        )
    p_stats = percentiles_int(series)
    max_inflight = max(series)
    mean_inflight = sum(series) / len(series)
    if max_inflight < 3:
        return CapacityEstimate(
            component, max_inflight, "underutilized",
            mean_inflight=mean_inflight,
            max_inflight=max_inflight,
            p95_inflight=p_stats["p95"],
        )
    non_idle = [x for x in series if x > 0]
    if not non_idle:
        return CapacityEstimate(
            component, max_inflight, "underutilized",
            mean_inflight=mean_inflight,
            max_inflight=max_inflight,
            p95_inflight=p_stats["p95"],
        )
    sorted_non_idle = sorted(non_idle)
    p90_non_idle = sorted_non_idle[min(len(sorted_non_idle) - 1,
                                        int(len(sorted_non_idle) * 0.90))]
    return CapacityEstimate(
        component, int(p90_non_idle), "est_p90",
        mean_inflight=mean_inflight,
        max_inflight=max_inflight,
        p95_inflight=p_stats["p95"],
    )


# --------------------------------------------------------------------------
# Back-pressure signals
# --------------------------------------------------------------------------


@dataclass
class BackPressure:
    component: str
    n_calls: int
    capacity: CapacityEstimate
    pct_workers_busy: float
    pct_queueing: float
    wall_inflation_p50: float  # p50(wall) / p10(wall), 1.0 means no inflation
    p50_wall_s: float
    p99_wall_s: float
    label: str  # low | medium | HIGH
    back_pressure_fraction: float


def compute_back_pressure(component: str,
                          iset: IntervalSet,
                          series: list[int],
                          capacity: CapacityEstimate) -> BackPressure:
    if not series or capacity.workers <= 0:
        return BackPressure(component, len(iset.intervals), capacity,
                            0.0, 0.0, 1.0, 0.0, 0.0, "low", 0.0)
    cap = capacity.workers
    n_ticks = len(series)
    n_busy = sum(1 for x in series if x >= cap)
    n_queue = sum(1 for x in series if x > cap)
    pct_busy = n_busy / n_ticks
    pct_queue = n_queue / n_ticks
    p = percentiles(iset.walls)
    p10 = p["p10"]
    p50 = p["p50"]
    inflation = (p50 / p10) if p10 > 0 and p50 == p50 else 1.0
    bp_fraction = max(
        pct_queue * 2.0,
        pct_busy,
        max(0.0, (inflation - 1.0)) * 0.5,
    )
    if bp_fraction > 0.5:
        label = "HIGH"
    elif bp_fraction > 0.2:
        label = "medium"
    else:
        label = "low"
    return BackPressure(
        component=component,
        n_calls=len(iset.intervals),
        capacity=capacity,
        pct_workers_busy=pct_busy,
        pct_queueing=pct_queue,
        wall_inflation_p50=inflation,
        p50_wall_s=p["p50"],
        p99_wall_s=p["p99"],
        label=label,
        back_pressure_fraction=bp_fraction,
    )


# --------------------------------------------------------------------------
# Idle gap detection (vLLM)
# --------------------------------------------------------------------------


def find_idle_gaps(series: list[int], tick_s: int,
                   min_gap_s: int = IDLE_GAP_MIN_S) -> list[tuple[int, int]]:
    """Return list of (tick_index, gap_s) for runs of in_flight=0 lasting
    >= min_gap_s."""
    gaps: list[tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for i, c in enumerate(series):
        if c == 0 and not in_gap:
            in_gap = True
            gap_start = i
        elif c > 0 and in_gap:
            in_gap = False
            length = (i - gap_start) * tick_s
            if length >= min_gap_s:
                gaps.append((gap_start, length))
    if in_gap:
        length = (len(series) - gap_start) * tick_s
        if length >= min_gap_s:
            gaps.append((gap_start, length))
    return gaps


# --------------------------------------------------------------------------
# Phase breakdown per rollout
# --------------------------------------------------------------------------


@dataclass
class PhaseBreakdown:
    front_pad_s: list[float] = field(default_factory=list)
    interturn_gap_s: list[float] = field(default_factory=list)
    post_last_turn_to_judge_s: list[float] = field(default_factory=list)
    judge_end_to_chat_end_s: list[float] = field(default_factory=list)


def compute_phase_breakdown(phase_data: dict[int, dict]) -> PhaseBreakdown:
    pb = PhaseBreakdown()
    for tid, d in phase_data.items():
        if d["chat"] is None or not d["turns"]:
            continue
        cs, ce = d["chat"]
        turns = sorted(d["turns"], key=lambda x: x[0])
        first_turn_s = turns[0][0]
        last_turn_e = turns[-1][1]
        pb.front_pad_s.append(max(0.0, first_turn_s - cs))
        for i in range(len(turns) - 1):
            pb.interturn_gap_s.append(max(0.0, turns[i + 1][0] - turns[i][1]))
        if d["judges"]:
            judges = sorted(d["judges"], key=lambda x: x[0])
            pb.post_last_turn_to_judge_s.append(
                max(0.0, judges[0][0] - last_turn_e))
            pb.judge_end_to_chat_end_s.append(
                max(0.0, ce - judges[-1][1]))
    return pb


# --------------------------------------------------------------------------
# Orchestration breakdown (2026-05-31 dark-span instrumentation)
# --------------------------------------------------------------------------


@dataclass
class OrchestrationBreakdown:
    # Per orchestration-span-name: list of per-occurrence walls (seconds).
    # (A trace can have many agent.fetch_state / agent.tool_dispatch spans.)
    walls_by_span: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list))
    # Per dark phase: (gap_total_s, explained_total_s) accumulated across
    # rollouts. "explained" = orchestration-span wall whose start falls inside
    # that phase's window. Lets us report how much of each historically-dark
    # phase is now lit.
    phase_gap_s: dict[str, float] = field(
        default_factory=lambda: defaultdict(float))
    phase_explained_s: dict[str, float] = field(
        default_factory=lambda: defaultdict(float))
    # Tool-dispatch wall that lands inside a turn (not a gap) -- reported
    # separately so it is visible without being double-counted into a phase.
    within_turn_dispatch_s: list[float] = field(default_factory=list)
    n_traces_with_orch: int = 0


def _classify_phase(s: float, first_turn_s: float, last_turn_e: float,
                    judge_start: Optional[float], judge_end: Optional[float],
                    turns: list[tuple]) -> str:
    """Bucket an orchestration span (by its start time) into the dark phase
    it lives in. Spans that start inside a turn interval are 'within_turn'."""
    for ts, te in turns:
        if ts <= s < te:
            return "within_turn"
    if s < first_turn_s:
        return "front_pad"
    if judge_start is not None and last_turn_e <= s < judge_start:
        return "post_last_turn_to_judge"
    if judge_end is not None and s >= judge_end:
        return "judge_end_to_chat_end"
    if judge_start is None and s >= last_turn_e:
        # No judge in this trace: everything after the last turn collapses
        # into the post-last-turn window up to chat end.
        return "post_last_turn_to_judge"
    return "interturn_gap"


def compute_orchestration_breakdown(
        phase_data: dict[int, dict]) -> OrchestrationBreakdown:
    ob = OrchestrationBreakdown()
    for tid, d in phase_data.items():
        if d["chat"] is None or not d["turns"]:
            continue
        orch = d.get("orch") or {}
        cs, ce = d["chat"]
        turns = sorted(d["turns"], key=lambda x: x[0])
        first_turn_s = turns[0][0]
        last_turn_e = turns[-1][1]
        judges = sorted(d["judges"], key=lambda x: x[0]) if d["judges"] else []
        judge_start = judges[0][0] if judges else None
        judge_end = judges[-1][1] if judges else None

        # Per-rollout phase gap totals (mirror compute_phase_breakdown so the
        # explained fraction shares a denominator).
        ob.phase_gap_s["front_pad"] += max(0.0, first_turn_s - cs)
        for i in range(len(turns) - 1):
            ob.phase_gap_s["interturn_gap"] += max(
                0.0, turns[i + 1][0] - turns[i][1])
        if judges:
            ob.phase_gap_s["post_last_turn_to_judge"] += max(
                0.0, judge_start - last_turn_e)
            ob.phase_gap_s["judge_end_to_chat_end"] += max(
                0.0, ce - judge_end)

        has_any = False
        for name, spans in orch.items():
            for (s, e) in spans:
                wall = max(0.0, e - s)
                ob.walls_by_span[name].append(wall)
                has_any = True
                phase = _classify_phase(
                    s, first_turn_s, last_turn_e, judge_start, judge_end, turns)
                if phase == "within_turn":
                    if name == "agent.tool_dispatch":
                        ob.within_turn_dispatch_s.append(wall)
                    continue
                ob.phase_explained_s[phase] += wall
        if has_any:
            ob.n_traces_with_orch += 1
    return ob


# --------------------------------------------------------------------------
# Bottleneck ranker
# --------------------------------------------------------------------------


@dataclass
class ComponentReport:
    component: str
    wall_total_s: float
    wall_share_pct: float
    back_pressure: BackPressure
    score: float


def rank_components(bps: dict[str, BackPressure],
                    chat_total_wall: float) -> list[ComponentReport]:
    out: list[ComponentReport] = []
    for comp in ALL_COMPONENTS:
        bp = bps[comp]
        if bp.capacity.workers == 0 and bp.n_calls == 0:
            continue
        wall_total = sum(IntervalSet(comp).walls)  # placeholder, overwritten
        out.append(ComponentReport(
            component=comp,
            wall_total_s=0.0,
            wall_share_pct=0.0,
            back_pressure=bp,
            score=0.0,
        ))
    return out


def build_component_reports(intervals: dict[str, IntervalSet],
                            bps: dict[str, BackPressure],
                            chat_total_wall_s: float) -> list[ComponentReport]:
    out: list[ComponentReport] = []
    for comp in ALL_COMPONENTS:
        iset = intervals[comp]
        bp = bps[comp]
        wall_total = sum(iset.walls)
        share = (wall_total / chat_total_wall_s * 100) if chat_total_wall_s > 0 else 0.0
        score = share * (1 + bp.back_pressure_fraction)
        out.append(ComponentReport(
            component=comp,
            wall_total_s=wall_total,
            wall_share_pct=share,
            back_pressure=bp,
            score=score,
        ))
    out.sort(key=lambda r: -r.score)
    return out


# --------------------------------------------------------------------------
# Performance summary (trainer EPH + episode efficiency)
# --------------------------------------------------------------------------


@dataclass
class StepFileMeta:
    step: int
    mtime_unix: float
    line_count: int


def fetch_step_files(logs_dir: Optional[str]) -> Optional[list[StepFileMeta]]:
    """Enumerate <logs_dir>/exp_001/train_data_step*.jsonl with (mtime, lines).
    For remote logs dirs, runs one ssh that emits "mtime lines basename" per file.
    Returns None if logs_dir is unset; raises on any other failure.
    """
    if not logs_dir:
        return None
    host, path = parse_remote_spec(logs_dir)
    exp_dir = f"{path.rstrip('/')}/exp_001"

    if host is None:
        local = Path(os.path.expanduser(exp_dir))
        if not local.exists():
            raise FileNotFoundError(f"logs-dir exp_001 not found: {local}")
        meta: list[StepFileMeta] = []
        for p in sorted(local.glob("train_data_step*.jsonl")):
            m = re.search(r"train_data_step(\d+)\.jsonl$", p.name)
            if not m:
                continue
            with p.open("rb") as fh:
                lines = sum(1 for _ in fh)
            meta.append(StepFileMeta(
                step=int(m.group(1)),
                mtime_unix=p.stat().st_mtime,
                line_count=lines,
            ))
        return meta

    # Remote: one shell roundtrip emits a stable triple per line.
    cmd = (
        f"cd {exp_dir} && "
        "for f in train_data_step*.jsonl; do "
        '  [ -f "$f" ] && echo "$(stat -c %Y \"$f\") $(wc -l < \"$f\") $f"; '
        "done"
    )
    print(f"[info] enumerating step files: ssh {host} ...", file=sys.stderr)
    r = subprocess.run(
        ["ssh", host, cmd], capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"remote step-file enumeration failed: {r.stderr.strip()}")
    meta: list[StepFileMeta] = []
    for line in r.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        m = re.search(r"train_data_step(\d+)\.jsonl$", parts[2])
        if not m:
            continue
        try:
            meta.append(StepFileMeta(
                step=int(m.group(1)),
                mtime_unix=float(parts[0]),
                line_count=int(parts[1]),
            ))
        except ValueError:
            continue
    meta.sort(key=lambda x: x.step)
    return meta


@dataclass
class PerfSummary:
    window_h: float
    inference_episodes: int
    trainer_episodes: Optional[int]
    inference_eph: float
    trainer_eph: Optional[float]
    efficiency_pct: Optional[float]
    n_train_steps: Optional[int]
    consumption_ratio: Optional[float]


def compute_perf_summary(chat_records: list[dict], win: Window,
                         step_files: Optional[list[StepFileMeta]]) -> PerfSummary:
    inf_count = sum(1 for c in chat_records
                    if win.start_unix <= c["end_unix"] < win.end_unix)
    inf_eph = inf_count / win.duration_h if win.duration_h > 0 else 0.0
    if step_files is None:
        return PerfSummary(
            window_h=win.duration_h, inference_episodes=inf_count,
            trainer_episodes=None, inference_eph=inf_eph,
            trainer_eph=None, efficiency_pct=None,
            n_train_steps=None, consumption_ratio=None,
        )
    in_window = [
        s for s in step_files
        if win.start_unix <= s.mtime_unix < win.end_unix
    ]
    trainer_count = sum(s.line_count for s in in_window)
    trainer_eph = trainer_count / win.duration_h if win.duration_h > 0 else 0.0
    eff_pct = (trainer_count / inf_count * 100) if inf_count > 0 else None
    cons_ratio = (trainer_eph / inf_eph) if inf_eph > 0 else None
    return PerfSummary(
        window_h=win.duration_h, inference_episodes=inf_count,
        trainer_episodes=trainer_count, inference_eph=inf_eph,
        trainer_eph=trainer_eph, efficiency_pct=eff_pct,
        n_train_steps=len(in_window), consumption_ratio=cons_ratio,
    )


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _fmt_s(s: float) -> str:
    if s != s:  # NaN
        return "n/a"
    sign = "-" if s < 0 else ""
    a = abs(s)
    if a >= 60:
        return f"{sign}{a/60:.1f}m"
    if a >= 1:
        return f"{sign}{a:.1f}s"
    return f"{sign}{a*1000:.0f}ms"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"


def _fmt_int(x: Optional[int]) -> str:
    return "n/a" if x is None else f"{x:,}"


def _fmt_float(x: Optional[float], suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:,.1f}{suffix}"


def render_header(loc: DBLocation, win: Window, label: str,
                  n_rollouts: int, chat_total_wall: float) -> str:
    lines = [f"# Profile: {label}", ""]
    lines.append(f"- **DB**: `{loc.raw}`")
    if loc.host:
        lines.append(f"- **Fetched to**: `{loc.local_path}`")
    lines.append(f"- **Window**: {win.label}")
    lines.append(f"- **Window UTC**: {win.start_iso()} -> {win.end_iso()}")
    lines.append(f"- **Window wall**: {win.duration_h:.2f}h ({win.duration_s:,.0f}s)")
    lines.append(f"- **Rollouts in window**: {n_rollouts:,}")
    lines.append(f"- **Aggregate chat_request wall**: {_fmt_s(chat_total_wall)}")
    lines.append("")
    return "\n".join(lines)


def render_perf_summary(perf: PerfSummary, has_logs_dir: bool) -> str:
    lines = ["## Performance summary", ""]
    rows = [
        ("Window", f"{perf.window_h:.2f}h"),
        ("Inference produced", _fmt_int(perf.inference_episodes)),
        ("Inference EPH (rollouts/h)", _fmt_float(perf.inference_eph)),
    ]
    if perf.trainer_episodes is not None:
        rows.extend([
            ("Trainer consumed", _fmt_int(perf.trainer_episodes)),
            ("Trainer EPH (episodes/h)", _fmt_float(perf.trainer_eph)),
            ("Episode efficiency (consumed / produced)",
             "n/a" if perf.efficiency_pct is None else f"{perf.efficiency_pct:.1f}%"),
            ("Trainer steps in window", _fmt_int(perf.n_train_steps)),
            ("Consumption ratio (trainer / inference EPH)",
             "n/a" if perf.consumption_ratio is None else f"{perf.consumption_ratio:.2f}"),
        ])
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if not has_logs_dir:
        lines.append("> _Trainer-side metrics: pass `--logs-dir <path-or-host:path>` to compute._")
        lines.append("")
    else:
        # Interpret consumption_ratio
        cr = perf.consumption_ratio
        if cr is not None:
            if cr < 0.9:
                msg = ("Trainer is consuming < 90% of inference output -> "
                       "inference is over-producing OR rollouts are aging out / being filtered.")
            elif cr <= 1.1:
                msg = ("Trainer ~ inference -> data-bound: trainer would go "
                       "faster with more inference capacity.")
            else:
                msg = ("Trainer > inference -> consuming aged-out / replayed data "
                       "(red flag, check buffer config).")
            lines.append(f"> _{msg}_")
            lines.append("")
    return "\n".join(lines)


def render_bottleneck_summary(reports: list[ComponentReport]) -> str:
    lines = ["## Bottleneck summary", ""]
    top = reports[0] if reports else None
    if top and top.wall_share_pct > 0:
        bp = top.back_pressure
        cap = bp.capacity
        lines.append(
            f"> **Bottleneck: `{top.component}`** -- {top.wall_share_pct:.0f}% of chat wall, "
            f"workers-busy {_fmt_pct(bp.pct_workers_busy)}, "
            f"queueing {_fmt_pct(bp.pct_queueing)}, "
            f"wall-inflation {bp.wall_inflation_p50:.1f}x "
            f"(back-pressure **{bp.label}**)."
        )
        lines.append(
            f"> _Capacity ({cap.source}) = {cap.workers} workers; "
            f"mean in-flight {cap.mean_inflight:.0f}, max {cap.max_inflight}._"
        )
        lines.append("")
    lines.append(
        "| component | n_calls | est_workers | mean_inflight | max_inflight "
        "| pct_busy | pct_queueing | wall_inflation_p50 | back_pressure "
        "| wall_share | score |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in reports:
        bp = r.back_pressure
        cap = bp.capacity
        cap_str = f"{cap.workers} ({cap.source})"
        lines.append(
            f"| `{r.component}` | {bp.n_calls:,} | {cap_str} | "
            f"{cap.mean_inflight:.1f} | {cap.max_inflight} | "
            f"{_fmt_pct(bp.pct_workers_busy)} | {_fmt_pct(bp.pct_queueing)} | "
            f"{bp.wall_inflation_p50:.2f}x | **{bp.label}** | "
            f"{r.wall_share_pct:.1f}% | {r.score:.1f} |"
        )
    lines.append("")
    lines.append(
        "_`pct_busy` = % of 5s ticks where in_flight >= est_workers. "
        "`pct_queueing` = % of ticks where in_flight > est_workers. "
        "`wall_inflation_p50` = p50(span_wall) / p10(span_wall) -- "
        "values >> 1 indicate clients waited for workers. "
        "**`wall_inflation` is the most robust back-pressure signal** -- it "
        "does NOT depend on the `est_workers` heuristic. Pass `--workers "
        "<comp>=<n>,...` to pin known capacities for accurate `pct_busy` / "
        "`pct_queueing`._"
    )
    lines.append("")
    return "\n".join(lines)


def render_rollout_budget(intervals: dict[str, IntervalSet],
                          chat_total_wall: float, n_rollouts: int) -> str:
    lines = ["## Per-rollout budget", ""]
    if n_rollouts == 0:
        lines.append("_No rollouts in window._")
        lines.append("")
        return "\n".join(lines)
    mean_chat = chat_total_wall / n_rollouts
    lines.append(f"- Rollouts in window: **{n_rollouts:,}**")
    lines.append(f"- Mean chat_request wall: **{_fmt_s(mean_chat)}**")
    lines.append("")
    lines.append("| component | total_wall | per_rollout_mean | % of chat wall |")
    lines.append("|---|---|---|---|")
    accounted = 0.0
    for comp in ALL_COMPONENTS:
        total = sum(intervals[comp].walls)
        accounted += total
        per_rol = total / n_rollouts
        share = total / chat_total_wall * 100 if chat_total_wall > 0 else 0
        lines.append(
            f"| `{comp}` | {_fmt_s(total)} | {_fmt_s(per_rol)} | {share:.1f}% |"
        )
    unaccounted = max(0.0, chat_total_wall - accounted)
    lines.append(
        f"| _unaccounted_ | {_fmt_s(unaccounted)} | "
        f"{_fmt_s(unaccounted / n_rollouts)} | "
        f"{unaccounted / chat_total_wall * 100:.1f}% |"
    )
    lines.append("")
    return "\n".join(lines)


def render_phase_breakdown(pb: PhaseBreakdown) -> str:
    lines = ["## Phase breakdown within rollout", ""]
    lines.append(
        "Where does the per-rollout wall go that ISN'T inside a tool / vLLM / judge span? "
        "This breaks `chat_request` into four time segments per trace and reports percentiles."
    )
    lines.append("")
    lines.append("| phase | n | mean | p50 | p90 | p99 | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, xs in [
        ("front_pad (chat_start -> first turn)", pb.front_pad_s),
        ("interturn_gap (between turns)", pb.interturn_gap_s),
        ("post_last_turn -> judge_start", pb.post_last_turn_to_judge_s),
        ("judge_end -> chat_end", pb.judge_end_to_chat_end_s),
    ]:
        if not xs:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        p = percentiles(xs)
        mean = sum(xs) / len(xs)
        lines.append(
            f"| {name} | {len(xs):,} | {_fmt_s(mean)} | {_fmt_s(p['p50'])} | "
            f"{_fmt_s(p['p90'])} | {_fmt_s(p['p99'])} | {_fmt_s(max(xs))} |"
        )
    lines.append("")
    return "\n".join(lines)


# Map each dark phase to the orchestration spans expected to explain it.
_PHASE_LABELS = [
    ("front_pad", "front_pad (chat_start -> first turn)",
     "episode.seed + first agent.fetch_state"),
    ("interturn_gap", "interturn_gap (between turns)",
     "agent.fetch_state"),
    ("post_last_turn_to_judge", "post_last_turn -> judge_start",
     "verify.reward_tiers + verify.edit_detection"),
    ("judge_end_to_chat_end", "judge_end -> chat_end",
     "verify.finalize"),
]


def render_orchestration_breakdown(ob: OrchestrationBreakdown) -> str:
    """Attribute the previously-dark phase time to the 2026-05-31
    orchestration spans (resource-server seed/verify + agent fetch/dispatch).
    Shows per-span latency and, per dark phase, how much is now explained vs
    still residual."""
    lines = ["## Orchestration breakdown (dark-span attribution)", ""]
    total_occurrences = sum(len(v) for v in ob.walls_by_span.values())
    if total_occurrences == 0:
        lines.append(
            "_No orchestration spans in window. This run predates the "
            "2026-05-31 dark-span instrumentation (episode.seed / "
            "agent.fetch_state / agent.tool_dispatch / verify.reward_tiers / "
            "verify.edit_detection / verify.finalize), or telemetry was a "
            "no-op. The Phase breakdown above still shows the raw gaps._"
        )
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"Traces with orchestration spans: **{ob.n_traces_with_orch:,}**. "
        "Per-span latency:"
    )
    lines.append("")
    lines.append("| span | n | mean | p50 | p90 | p99 | max | total_wall |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ORCH_SPAN_NAMES:
        xs = ob.walls_by_span.get(name, [])
        if not xs:
            lines.append(f"| `{name}` | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        p = percentiles(xs)
        mean = sum(xs) / len(xs)
        lines.append(
            f"| `{name}` | {len(xs):,} | {_fmt_s(mean)} | {_fmt_s(p['p50'])} | "
            f"{_fmt_s(p['p90'])} | {_fmt_s(p['p99'])} | {_fmt_s(max(xs))} | "
            f"{_fmt_s(sum(xs))} |"
        )
    lines.append("")

    # Phase coverage: how much of each dark phase is now explained.
    lines.append(
        "Dark-phase coverage (explained = orchestration-span wall starting "
        "inside that phase window; residual = still unaccounted):"
    )
    lines.append("")
    lines.append("| phase | gap_total | explained | residual | explained % | lit by |")
    lines.append("|---|---|---|---|---|---|")
    for key, label, lit_by in _PHASE_LABELS:
        gap = ob.phase_gap_s.get(key, 0.0)
        explained = min(ob.phase_explained_s.get(key, 0.0), gap) if gap > 0 else \
            ob.phase_explained_s.get(key, 0.0)
        residual = max(0.0, gap - explained)
        pct = (explained / gap * 100) if gap > 0 else 0.0
        lines.append(
            f"| {label} | {_fmt_s(gap)} | {_fmt_s(explained)} | "
            f"{_fmt_s(residual)} | {pct:.0f}% | {lit_by} |"
        )
    lines.append("")

    if ob.within_turn_dispatch_s:
        xs = ob.within_turn_dispatch_s
        p = percentiles(xs)
        lines.append(
            f"`agent.tool_dispatch` inside turns (not a gap): "
            f"n={len(xs):,}, p50={_fmt_s(p['p50'])}, p90={_fmt_s(p['p90'])}, "
            f"total={_fmt_s(sum(xs))}. The gap between this client wall and "
            f"the server-side `tool.{{name}}` span is the resource-server "
            f"queue wait."
        )
        lines.append("")
    return "\n".join(lines)


def render_vllm_section(iset: IntervalSet, series: list[int],
                        bp: BackPressure, tick_s: int) -> str:
    lines = ["## vLLM analysis (`policy_llm.responses`)", ""]
    if not iset.intervals:
        lines.append("_No vLLM spans in window._")
        lines.append("")
        return "\n".join(lines)
    p = percentiles(iset.walls)
    lines.append(f"- Calls: **{len(iset.intervals):,}**")
    lines.append(
        f"- Latency: p50={_fmt_s(p['p50'])} p90={_fmt_s(p['p90'])} "
        f"p99={_fmt_s(p['p99'])} max={_fmt_s(max(iset.walls))}"
    )
    lines.append(f"- Total wall: {_fmt_s(sum(iset.walls))}")
    lines.append("")
    # Idle gaps
    gaps = find_idle_gaps(series, tick_s)
    if gaps:
        gap_lens = sorted([g[1] for g in gaps], reverse=True)
        lines.append("### Idle gaps (in_flight = 0 for >= 30s)")
        lines.append("")
        lines.append(
            f"- Total gaps: **{len(gaps)}**, total idle time: "
            f"**{_fmt_s(sum(gap_lens))}** ({sum(gap_lens)/(len(series)*tick_s)*100:.1f}% of window)"
        )
        lines.append(f"- Longest gaps (top 5): " + ", ".join(_fmt_s(g) for g in gap_lens[:5]))
        lines.append("")
    else:
        lines.append("_No vLLM idle gaps >= 30s in window._")
        lines.append("")
    return "\n".join(lines)


def render_judge_section(iset: IntervalSet, series: list[int], bp: BackPressure,
                         per_endpoint: dict[str, list[float]]) -> str:
    lines = ["## Judge analysis (`judge_evaluation` + child `llm.complete`)", ""]
    if not iset.intervals:
        lines.append("_No judge spans in window._")
        lines.append("")
        return "\n".join(lines)
    p = percentiles(iset.walls)
    lines.append(f"- Calls: **{len(iset.intervals):,}**")
    lines.append(
        f"- Latency: p50={_fmt_s(p['p50'])} p90={_fmt_s(p['p90'])} "
        f"p99={_fmt_s(p['p99'])} max={_fmt_s(max(iset.walls))}"
    )
    lines.append(f"- Total wall: {_fmt_s(sum(iset.walls))}")
    lines.append("")
    if not per_endpoint:
        lines.append("_No `judge.endpoint.name` attribute on `llm.complete` spans._")
        lines.append("")
        return "\n".join(lines)
    lines.append("### Per-endpoint latency")
    lines.append("")
    lines.append("| endpoint | n | mean | p50 | p90 | p99 | max | total_wall | wall_share |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    rows = sorted(per_endpoint.items(), key=lambda kv: -sum(kv[1]))
    total = sum(sum(v) for v in per_endpoint.values()) or 1.0
    for ep, walls in rows:
        pe = percentiles(walls)
        lines.append(
            f"| `{ep}` | {len(walls):,} | {_fmt_s(sum(walls)/len(walls))} | "
            f"{_fmt_s(pe['p50'])} | {_fmt_s(pe['p90'])} | {_fmt_s(pe['p99'])} | "
            f"{_fmt_s(max(walls))} | {_fmt_s(sum(walls))} | "
            f"{sum(walls)/total*100:.1f}% |"
        )
    lines.append("")
    # Drop-simulation: if we removed the slowest endpoint, what's the saving?
    if len(rows) >= 2:
        lines.append("### Drop simulation (re-roll one endpoint's calls uniformly across survivors)")
        lines.append("")
        lines.append("| drop | n | savings_s | savings_pct |")
        lines.append("|---|---|---|---|")
        survivors_total_walls = {ep: sum(walls) for ep, walls in rows}
        survivors_total_counts = {ep: len(walls) for ep, walls in rows}
        old_total = sum(survivors_total_walls.values())
        sims = []
        for victim_ep, walls in rows:
            survivors_n = sum(v for k, v in survivors_total_counts.items() if k != victim_ep)
            if survivors_n == 0:
                continue
            survivor_mean = sum(
                (sum(per_endpoint[k]) / len(per_endpoint[k])) * len(per_endpoint[k])
                for k in survivors_total_counts if k != victim_ep
            ) / survivors_n
            victim_n = len(walls)
            victim_total = sum(walls)
            new_total = (old_total - victim_total) + survivor_mean * victim_n
            saving = old_total - new_total
            sims.append((victim_ep, victim_n, saving, saving / old_total * 100))
        for ep, n, saving, pct in sorted(sims, key=lambda x: -x[2]):
            lines.append(f"| `{ep}` | {n:,} | {_fmt_s(saving)} | {pct:.1f}% |")
        lines.append("")
    return "\n".join(lines)


def render_tool_section(per_tool: dict[str, list[float]],
                        chat_total_wall: float,
                        tool_service_rows: list[dict]) -> str:
    lines = ["## Per-tool latency", ""]
    if not per_tool:
        lines.append("_No tool spans in window._")
        lines.append("")
        return "\n".join(lines)
    lines.append("| tool | bucket | n | mean | p50 | p90 | p99 | max | total | wall_share |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = sorted(per_tool.items(), key=lambda kv: -sum(kv[1]))
    for name, walls in rows:
        if not walls:
            continue
        p = percentiles(walls)
        tot = sum(walls)
        bucket = tool_to_component(name[5:]) if name.startswith("tool.") else "?"
        lines.append(
            f"| `{name}` | `{bucket}` | {len(walls):,} | "
            f"{_fmt_s(tot/len(walls))} | {_fmt_s(p['p50'])} | "
            f"{_fmt_s(p['p90'])} | {_fmt_s(p['p99'])} | {_fmt_s(max(walls))} | "
            f"{_fmt_s(tot)} | "
            f"{tot/chat_total_wall*100 if chat_total_wall else 0:.1f}% |"
        )
    lines.append("")
    # tool_service.http retry distribution
    if tool_service_rows:
        lines.append("### `tool_service.http` retry distribution")
        lines.append("")
        per_tool_attempt: dict[tuple, list[float]] = defaultdict(list)
        for r in tool_service_rows:
            per_tool_attempt[(r["tool"] or "?", r["attempt"])].append(r["wall"])
        lines.append("| tool | attempt | n | mean | max |")
        lines.append("|---|---|---|---|---|")
        for (tool, attempt), walls in sorted(per_tool_attempt.items(),
                                             key=lambda kv: (-len(kv[1]), str(kv[0]))):
            lines.append(
                f"| `{tool}` | {attempt} | {len(walls):,} | "
                f"{_fmt_s(sum(walls)/len(walls))} | {_fmt_s(max(walls))} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_dispatch_section(records: list[dict]) -> str:
    """Render per-turn LLM dispatch behavior (tools-per-turn distribution,
    reasoning prevalence, top function-name frequencies)."""
    lines = ["## Per-turn LLM dispatch behavior (`policy_llm.responses`)", ""]
    if not records:
        lines.append("_No `policy_llm.responses` spans in window._")
        lines.append("")
        return "\n".join(lines)

    # Distinguish "attr genuinely absent" (older runs that pre-date the
    # 2026-05-30 dispatch attrs) from "attr present and zero". If the entire
    # window has no records carrying these attrs, the agent overlay is on an
    # older revision; tell the reader instead of printing misleading zeros.
    have_counts = [r for r in records if r["n_calls"] is not None]
    have_reasoning_flag = [r for r in records if r["has_reasoning"] is not None]
    have_fn_list = [r for r in records if r["functions"]]

    if not (have_counts or have_reasoning_flag or have_fn_list):
        lines.append(
            f"_Found {len(records):,} turns but none carry `llm.tool_calls_count` / "
            "`llm.has_reasoning` / `llm.functions` attrs. Agent overlay predates "
            "the 2026-05-30 dispatch-attrs change._"
        )
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- Total turns: **{len(records):,}**")

    if have_reasoning_flag:
        with_reasoning = sum(1 for r in have_reasoning_flag if r["has_reasoning"])
        denom = len(have_reasoning_flag)
        lines.append(
            f"- Turns with reasoning: **{with_reasoning:,} / {denom:,}** "
            f"({with_reasoning / denom * 100:.1f}%)"
        )

    if have_counts:
        counts = [r["n_calls"] for r in have_counts]
        p = percentiles_int(counts, qs=(0.50, 0.90, 0.95))
        n_zero = sum(1 for c in counts if c == 0)
        mean = sum(counts) / len(counts)
        lines.append(
            f"- Tool calls per turn: mean={mean:.2f} p50={p['p50']} "
            f"p90={p['p90']} p95={p['p95']} max={max(counts)} "
            f"(zero-tool turns: {n_zero:,}, {n_zero / len(counts) * 100:.1f}%)"
        )

    lines.append("")

    if have_fn_list:
        freq: Counter = Counter()
        turns_with_fn: Counter = Counter()
        for r in have_fn_list:
            seen_in_turn: set[str] = set()
            for fn in r["functions"]:
                freq[fn] += 1
                seen_in_turn.add(fn)
            for fn in seen_in_turn:
                turns_with_fn[fn] += 1
        total_calls = sum(freq.values())
        lines.append("### Function-name frequency (top 20)")
        lines.append("")
        lines.append("| function | calls | call_share | turns_invoking | turn_share |")
        lines.append("|---|---|---|---|---|")
        n_turns = len(records)
        for fn, n in freq.most_common(20):
            call_share = n / total_calls * 100 if total_calls else 0.0
            turn_share = turns_with_fn[fn] / n_turns * 100 if n_turns else 0.0
            lines.append(
                f"| `{fn}` | {n:,} | {call_share:.1f}% | "
                f"{turns_with_fn[fn]:,} | {turn_share:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def render_time_series(per_component_series: dict[str, list[int]],
                       tick_s: int, win: Window) -> str:
    lines = ["## In-flight time series", ""]
    if not per_component_series:
        return "\n".join(lines) + "\n"
    # Take the chat_request series first to size buckets; resample everything
    # to TIMESERIES_TICK_S buckets (max of bucket).
    ts_tick = TIMESERIES_TICK_S
    sub = max(1, ts_tick // tick_s)
    components_to_show = [c for c in ["chat", "vllm", "judge", "tool_pyexec", "tool_vina"]
                          if c in per_component_series and per_component_series[c]]
    n_ticks = min(len(per_component_series[c]) for c in components_to_show) if components_to_show else 0
    n_buckets = n_ticks // sub
    if n_buckets == 0:
        lines.append("_Window too short for time series._")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"_Tick = {ts_tick}s, max in-flight within each bucket._")
    lines.append("")
    header = "| t_min | " + " | ".join(components_to_show) + " |"
    sep = "|---|" + "|".join(["---"] * len(components_to_show)) + "|"
    lines.append(header)
    lines.append(sep)
    # Cap rows to 50 to keep the report compact; use floor sub-sampling.
    max_rows = 50
    stride = max(1, n_buckets // max_rows)
    for i in range(0, n_buckets, stride):
        row_vals = []
        for c in components_to_show:
            s = per_component_series[c]
            chunk = s[i*sub:(i+1)*sub]
            row_vals.append(max(chunk) if chunk else 0)
        t_rel_min = i * ts_tick / 60
        lines.append(f"| {t_rel_min:.1f} | " + " | ".join(str(v) for v in row_vals) + " |")
    lines.append("")
    return "\n".join(lines)


def render_top_slowest(chat_records: list[dict], top_n: int,
                       phase_data: dict[int, dict]) -> str:
    lines = [f"## Top-{top_n} slowest rollouts", ""]
    if not chat_records:
        lines.append("_No rollouts in window._")
        lines.append("")
        return "\n".join(lines)
    top = sorted(chat_records, key=lambda r: -r["wall_s"])[:top_n]
    lines.append("| # | wall | iters | tool_calls | outcome | hit_limit |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(top, 1):
        lines.append(
            f"| {i} | {_fmt_s(r['wall_s'])} | {r['iterations']} | "
            f"{r['tool_calls']} | `{r['outcome']}` | {r['hit_limit']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_completion_rate(per_min: list[tuple[int, int]], win: Window) -> str:
    lines = ["## Completion rate", ""]
    if not per_min:
        lines.append("_No completions in window._")
        lines.append("")
        return "\n".join(lines)
    counts = [n for _, n in per_min]
    p = percentiles_int(counts, qs=(0.10, 0.50, 0.90, 0.95))
    mean = sum(counts) / len(counts)
    lines.append(
        f"- Completions/min: mean={mean:.1f}  p10={p['p10']} p50={p['p50']} "
        f"p90={p['p90']} p95={p['p95']}  max={max(counts)}"
    )
    lines.append(f"- Buckets reporting: {len(counts)}, total completions: {sum(counts):,}")
    # Detect stalls: how many minutes had completions < 50% of median
    floor = max(1, p["p50"] // 2)
    stalls = sum(1 for n in counts if n < floor)
    if stalls:
        lines.append(
            f"- **Stalls**: {stalls} minute-buckets had completions < {floor} "
            f"(< 50% of median). Possible refit / queue / backpressure events."
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# JSON renderer
# --------------------------------------------------------------------------


def _dispatch_json(records: list[dict]) -> dict:
    """JSON-serialisable summary of per-turn LLM dispatch behavior."""
    if not records:
        return {"n_turns": 0}
    have_counts = [r["n_calls"] for r in records if r["n_calls"] is not None]
    have_flag = [r["has_reasoning"] for r in records if r["has_reasoning"] is not None]
    freq: Counter = Counter()
    for r in records:
        for fn in r["functions"]:
            freq[fn] += 1
    out: dict = {"n_turns": len(records)}
    if have_counts:
        out["tool_calls_per_turn"] = {
            "mean": sum(have_counts) / len(have_counts),
            **{k: v for k, v in percentiles_int(have_counts).items()},
            "max": max(have_counts),
            "n_zero": sum(1 for c in have_counts if c == 0),
            "n_with_attr": len(have_counts),
        }
    if have_flag:
        out["reasoning"] = {
            "n_with_attr": len(have_flag),
            "n_with_reasoning": sum(1 for v in have_flag if v),
        }
    if freq:
        out["function_frequency"] = [
            {"name": name, "calls": n} for name, n in freq.most_common(50)
        ]
    return out


def render_json(loc: DBLocation, win: Window, label: str,
                perf: PerfSummary, reports: list[ComponentReport],
                intervals: dict[str, IntervalSet], pb: PhaseBreakdown,
                idle_gaps: list[tuple[int, int]],
                per_endpoint: dict[str, list[float]],
                per_tool: dict[str, list[float]],
                chat_records: list[dict], top_n: int,
                completions_per_min: list[tuple[int, int]],
                dispatch_records: list[dict],
                ob: Optional["OrchestrationBreakdown"] = None) -> dict:
    out = {
        "schema_version": 1,
        "db": loc.raw,
        "label": label,
        "window": {
            "label": win.label,
            "start_unix": win.start_unix,
            "end_unix": win.end_unix,
            "duration_h": win.duration_h,
        },
        "n_rollouts": len(chat_records),
        "performance": {
            "window_h": perf.window_h,
            "inference_episodes": perf.inference_episodes,
            "trainer_episodes": perf.trainer_episodes,
            "inference_eph": perf.inference_eph,
            "trainer_eph": perf.trainer_eph,
            "efficiency_pct": perf.efficiency_pct,
            "n_train_steps": perf.n_train_steps,
            "consumption_ratio": perf.consumption_ratio,
        },
        "bottleneck": [
            {
                "component": r.component,
                "score": r.score,
                "wall_total_s": r.wall_total_s,
                "wall_share_pct": r.wall_share_pct,
                "n_calls": r.back_pressure.n_calls,
                "capacity": {
                    "workers": r.back_pressure.capacity.workers,
                    "source": r.back_pressure.capacity.source,
                    "mean_inflight": r.back_pressure.capacity.mean_inflight,
                    "max_inflight": r.back_pressure.capacity.max_inflight,
                    "p95_inflight": r.back_pressure.capacity.p95_inflight,
                },
                "pct_workers_busy": r.back_pressure.pct_workers_busy,
                "pct_queueing": r.back_pressure.pct_queueing,
                "wall_inflation_p50": r.back_pressure.wall_inflation_p50,
                "p50_wall_s": r.back_pressure.p50_wall_s,
                "p99_wall_s": r.back_pressure.p99_wall_s,
                "back_pressure_label": r.back_pressure.label,
                "back_pressure_fraction": r.back_pressure.back_pressure_fraction,
            }
            for r in reports
        ],
        "phase_breakdown": {
            "front_pad_s": percentiles(pb.front_pad_s),
            "interturn_gap_s": percentiles(pb.interturn_gap_s),
            "post_last_turn_to_judge_s": percentiles(pb.post_last_turn_to_judge_s),
            "judge_end_to_chat_end_s": percentiles(pb.judge_end_to_chat_end_s),
        },
        "orchestration": (
            {
                "n_traces_with_orch": ob.n_traces_with_orch,
                "spans": {
                    name: {
                        **percentiles(ob.walls_by_span.get(name, [])),
                        "n": len(ob.walls_by_span.get(name, [])),
                        "total_wall_s": sum(ob.walls_by_span.get(name, [])),
                    }
                    for name in ORCH_SPAN_NAMES
                },
                "phase_coverage": {
                    key: {
                        "gap_total_s": ob.phase_gap_s.get(key, 0.0),
                        "explained_s": ob.phase_explained_s.get(key, 0.0),
                    }
                    for key, _label, _lit in _PHASE_LABELS
                },
                "within_turn_dispatch_total_s": sum(ob.within_turn_dispatch_s),
            }
            if ob is not None
            else None
        ),
        "idle_gaps_vllm": [
            {"tick_index": ti, "gap_s": g} for ti, g in idle_gaps[:50]
        ],
        "judge_per_endpoint": {
            ep: {**percentiles(walls), "n": len(walls), "total_wall_s": sum(walls)}
            for ep, walls in per_endpoint.items()
        },
        "per_tool": {
            name: {**percentiles(walls), "n": len(walls), "total_wall_s": sum(walls)}
            for name, walls in per_tool.items()
        },
        "top_slowest": sorted(chat_records, key=lambda r: -r["wall_s"])[:top_n],
        "completions_per_min": [
            {"bucket_unix_s": b, "n": n} for b, n in completions_per_min
        ],
        "per_turn_dispatch": _dispatch_json(dispatch_records),
    }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Profile an Aevom training run from its Phoenix DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("db_location",
                    help="Local path OR host:path to phoenix.db")
    ap.add_argument("--window", default=None,
                    help="Wall-clock window, e.g. '0s:30m', '1h:', '2026-05-28T22:00Z:...'")
    ap.add_argument("--logs-dir", default=None,
                    help="Run logs dir (local or host:path) for trainer EPH + efficiency. "
                         "Expects <logs-dir>/exp_001/train_data_step*.jsonl.")
    ap.add_argument("--workers", default=None,
                    help="Comma-separated <comp>=<n> capacities, e.g. "
                         "vllm=416,judge=64,pyexec=128,vina=32")
    ap.add_argument("--top", type=int, default=10,
                    help="Top-N slowest rollouts (default 10)")
    ap.add_argument("--output", default=None,
                    help="Write markdown report to FILE (default: stdout)")
    ap.add_argument("--json", default=None,
                    help="Additionally write JSON profile to FILE")
    ap.add_argument("--cache-dir", default="/tmp/aevom_profile",
                    help="Cache dir for remote fetches + sqlite snapshots")
    ap.add_argument("--label", default=None,
                    help="Header label (default: derive from db path)")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        loc = resolve_db(args.db_location, cache_dir)
        workers_spec = parse_workers_spec(args.workers)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    label = args.label or Path(loc.remote_path or loc.local_path.name).parent.parent.name

    conn = sqlite3.connect(f"file:{loc.local_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        run_start, run_end = fetch_run_bounds(conn)
        win = resolve_window(args.window, run_start, run_end)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    intervals = load_component_intervals(conn, win)
    chat_iset = intervals.pop("chat")
    chat_records = load_chat_rollout_records(conn, win)
    chat_total_wall = sum(chat_iset.walls)

    # In-flight series per component (5s ticks)
    series: dict[str, list[int]] = {}
    series["chat"] = in_flight_series(chat_iset.intervals, CONCURRENCY_TICK_S,
                                       win.start_unix, win.end_unix)
    for comp in ALL_COMPONENTS:
        series[comp] = in_flight_series(intervals[comp].intervals,
                                        CONCURRENCY_TICK_S,
                                        win.start_unix, win.end_unix)

    # Capacity + back-pressure per component
    caps: dict[str, CapacityEstimate] = {}
    bps: dict[str, BackPressure] = {}
    for comp in ALL_COMPONENTS:
        caps[comp] = estimate_capacity(comp, series[comp], workers_spec.get(comp))
        bps[comp] = compute_back_pressure(comp, intervals[comp],
                                          series[comp], caps[comp])

    reports = build_component_reports(intervals, bps, chat_total_wall)

    # Phase breakdown + orchestration (dark-span) attribution
    phase_data = load_phase_data(conn, win)
    pb = compute_phase_breakdown(phase_data)
    ob = compute_orchestration_breakdown(phase_data)

    # vLLM idle gaps
    idle_gaps = find_idle_gaps(series["vllm"], CONCURRENCY_TICK_S)

    # Judge per-endpoint
    per_endpoint = load_judge_per_endpoint(conn, win)

    # Per-tool latencies + tool_service.http
    per_tool = load_per_tool_latencies(conn, win)
    tool_service_rows = load_tool_service_attempts(conn, win)

    # Per-turn dispatch behavior (uses 2026-05-30 policy_llm.responses attrs)
    dispatch_records = load_per_turn_dispatch(conn, win)

    # Completion rate
    completions_per_min = load_chat_completions_per_min(conn, win)

    # Performance summary (trainer-side)
    try:
        step_files = fetch_step_files(args.logs_dir) if args.logs_dir else None
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"warning: --logs-dir failed ({e}); trainer metrics disabled",
              file=sys.stderr)
        step_files = None
    perf = compute_perf_summary(chat_records, win, step_files)

    # ---- Render ----
    parts = []
    parts.append(render_header(loc, win, label,
                               n_rollouts=len(chat_records),
                               chat_total_wall=chat_total_wall))
    parts.append(render_perf_summary(perf, has_logs_dir=bool(args.logs_dir)))
    parts.append(render_bottleneck_summary(reports))
    parts.append(render_rollout_budget(intervals, chat_total_wall,
                                       len(chat_records)))
    parts.append(render_phase_breakdown(pb))
    parts.append(render_orchestration_breakdown(ob))
    parts.append(render_vllm_section(intervals["vllm"], series["vllm"],
                                     bps["vllm"], CONCURRENCY_TICK_S))
    parts.append(render_judge_section(intervals["judge"], series["judge"],
                                      bps["judge"], per_endpoint))
    parts.append(render_tool_section(per_tool, chat_total_wall,
                                     tool_service_rows))
    parts.append(render_dispatch_section(dispatch_records))
    parts.append(render_time_series(
        {"chat": series["chat"], **{c: series[c] for c in ALL_COMPONENTS}},
        CONCURRENCY_TICK_S, win,
    ))
    parts.append(render_top_slowest(chat_records, args.top, phase_data))
    parts.append(render_completion_rate(completions_per_min, win))

    md = "\n".join(parts).rstrip() + "\n"

    if args.output:
        Path(args.output).write_text(md)
        print(f"[info] wrote markdown -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(md)

    if args.json:
        js = render_json(loc, win, label, perf, reports, intervals, pb,
                         idle_gaps, per_endpoint, per_tool, chat_records,
                         args.top, completions_per_min, dispatch_records,
                         ob=ob)
        Path(args.json).write_text(json.dumps(js, indent=2, default=str))
        print(f"[info] wrote json -> {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
