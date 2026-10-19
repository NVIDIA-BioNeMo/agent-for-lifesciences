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

"""Rollout timing report: per-step straggler analysis from episodes.jsonl.

Parses an episodes.jsonl produced by a NeMo-Gym GRPO run and prints a markdown
report covering:

- Step cadence (wall time between rollout-batch boundaries)
- Episode duration distribution per step (min/p50/p75/p90/p99/max of
  sum(reasoning_trace[].elapsed_ms))
- Straggler contribution: step_wall - p75_episode = wall time spent waiting on tail
- Turn count / tool-call histograms
- Finish-reason breakdown (from ray-driver.log, if provided)

Usage:
    python scripts/rollout_timing_report.py \
        --episodes /tmp/episodes.jsonl \
        --ray-driver /tmp/ray-driver.log \
        --label "v0422a 6n" \
        --num-prompts-per-step 8 \
        --num-generations-per-prompt 16

The episodes-file argument also accepts an `host:/absolute/path` spec, which is
resolved via ssh (cat-over-ssh) into a local tempfile.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _pcts(xs: list[float], pcts=(50, 75, 90, 99)) -> dict[int, float]:
    if not xs:
        return {p: float("nan") for p in pcts}
    s = sorted(xs)
    n = len(s)
    out: dict[int, float] = {}
    for p in pcts:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        out[p] = s[idx]
    return out


def _fmt_ms(v: float) -> str:
    if v != v:  # NaN
        return "n/a"
    if v >= 60_000:
        return f"{v/1000:.1f}s"
    if v >= 1_000:
        return f"{v/1000:.2f}s"
    return f"{v:.0f}ms"


def _resolve_path(spec: str) -> str:
    """Accept either a local path or `host:/abs/path`; if remote, cat over ssh."""
    if ":" in spec and not os.path.exists(spec):
        host, remote = spec.split(":", 1)
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"{host}_", suffix=".jsonl", delete=False
        )
        tmp.close()
        print(f"[info] fetching {spec} -> {tmp.name}", file=sys.stderr)
        subprocess.check_call(
            ["scp", "-q", spec, tmp.name],
        )
        return tmp.name
    return spec


@dataclass
class Episode:
    line_idx: int
    timestamp: datetime
    num_turns: int
    num_tool_calls: int
    elapsed_ms_total: float
    reward: float
    reasoning_trace: list[dict]


def _parse_timestamp(ts: str) -> datetime:
    # e.g. 2026-04-22T06:29:54-0700
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")


def load_episodes(path: str) -> list[Episode]:
    eps: list[Episode] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            rt = d.get("reasoning_trace") or []
            elapsed = sum(float(t.get("elapsed_ms", 0) or 0) for t in rt)
            try:
                ts = _parse_timestamp(d["timestamp"])
            except Exception:
                continue
            eps.append(
                Episode(
                    line_idx=i,
                    timestamp=ts,
                    num_turns=int(d.get("num_turns", 0) or 0),
                    num_tool_calls=int(d.get("num_tool_calls", 0) or 0),
                    elapsed_ms_total=elapsed,
                    reward=float(d.get("reward", 0.0) or 0.0),
                    reasoning_trace=rt,
                )
            )
    return eps


def bucket_into_steps(eps: list[Episode], step_size: int) -> list[list[Episode]]:
    """Bucket sequential episodes into fixed-size batches (`num_prompts_per_step *
    num_generations_per_prompt`). Episodes are in file order; NeMo-Gym appends
    them roughly in completion order within a step."""
    out: list[list[Episode]] = []
    cur: list[Episode] = []
    for e in eps:
        cur.append(e)
        if len(cur) >= step_size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def parse_finish_reasons_per_step(ray_driver_log: str) -> list[Counter]:
    """Split the ray-driver.log at each 'Collecting rollouts: 100%' boundary and
    count `finish_reason=X` tokens between boundaries. Returns list of Counters,
    one per completed step."""
    per_step: list[Counter] = []
    current: Counter = Counter()
    boundary_re = re.compile(r"Collecting rollouts: 100%")
    fr_re = re.compile(r"finish_reason=([a-z_]+)")
    with open(ray_driver_log, "r", errors="ignore") as f:
        for line in f:
            for m in fr_re.finditer(line):
                current[m.group(1)] += 1
            if boundary_re.search(line):
                per_step.append(current)
                current = Counter()
    # Trailing (potentially in-progress) step not included; add only if non-empty
    if sum(current.values()) > 0:
        per_step.append(current)
    return per_step


def render(
    eps: list[Episode],
    label: str,
    step_size: int,
    ray_driver_log: Optional[str],
) -> str:
    steps = bucket_into_steps(eps, step_size)
    lines: list[str] = []
    push = lines.append

    push(f"# Rollout timing report: {label}")
    push("")
    push(f"- Source episodes: {len(eps)}")
    push(f"- Step size (num_prompts * num_generations): **{step_size}**")
    push(f"- Number of (complete) steps detected: **{len(steps)}**")
    push("")

    # Overall distributions
    push("## Episode duration (sum of reasoning_trace[].elapsed_ms)")
    push("")
    durations = [e.elapsed_ms_total for e in eps]
    pct = _pcts(durations, (50, 75, 90, 99))
    push(
        "| p50 | p75 | p90 | p99 | max |"
    )
    push("|---|---|---|---|---|")
    push(
        f"| {_fmt_ms(pct[50])} | {_fmt_ms(pct[75])} | {_fmt_ms(pct[90])} | "
        f"{_fmt_ms(pct[99])} | {_fmt_ms(max(durations) if durations else float('nan'))} |"
    )
    push("")
    zero_dur = sum(1 for d in durations if d == 0)
    push(
        f"- Episodes with zero elapsed_ms (likely LLM finish_reason=stop/length with no tools): "
        f"**{zero_dur}** / {len(eps)} ({100*zero_dur/max(1,len(eps)):.1f}%)"
    )
    push("")

    # Turn / tool-call histograms
    push("## Turn & tool-call distribution")
    push("")
    nt_hist = Counter(e.num_turns for e in eps)
    ntc_hist = Counter(e.num_tool_calls for e in eps)
    push(f"- num_turns histogram (sorted): {dict(sorted(nt_hist.items()))}")
    push(f"- num_tool_calls histogram (sorted): {dict(sorted(ntc_hist.items()))}")
    push("")

    # Per-step table
    push("## Per-step durations and straggler contribution")
    push("")
    push(
        "Column defs: `wall` is the elapsed wall-clock between the first and last "
        "`timestamp` in the bucket (approximate, since episode timestamps are "
        "appended roughly in completion order). `straggler` = p99 - p75, and "
        "`straggler_frac` = (max - p75) / max, i.e. the tail's share of the step. "
        "`cadence` is wall-clock between this step's last episode and the next "
        "step's last episode."
    )
    push("")
    push(
        "| step | n | wall | p50 | p75 | p90 | p99 | max | straggler (p99-p75) | straggler_frac | cadence |"
    )
    push("|---|---|---|---|---|---|---|---|---|---|---|")

    step_sums: list[tuple[float, float, float]] = []  # (wall, p75, max)
    last_step_last_ts: Optional[datetime] = None
    cadences: list[float] = []
    for s_i, bucket in enumerate(steps):
        bucket_durs = [e.elapsed_ms_total for e in bucket]
        pct = _pcts(bucket_durs, (50, 75, 90, 99))
        mx = max(bucket_durs) if bucket_durs else float("nan")
        t_first = bucket[0].timestamp
        t_last = bucket[-1].timestamp
        wall_ms = (t_last - t_first).total_seconds() * 1000.0
        cadence_ms = float("nan")
        if last_step_last_ts is not None:
            cadence_ms = (t_last - last_step_last_ts).total_seconds() * 1000.0
            cadences.append(cadence_ms)
        last_step_last_ts = t_last
        straggler_ms = pct[99] - pct[75]
        straggler_frac = (mx - pct[75]) / mx if mx and mx == mx else float("nan")
        push(
            f"| {s_i} | {len(bucket)} | {_fmt_ms(wall_ms)} | {_fmt_ms(pct[50])} | "
            f"{_fmt_ms(pct[75])} | {_fmt_ms(pct[90])} | {_fmt_ms(pct[99])} | "
            f"{_fmt_ms(mx)} | {_fmt_ms(straggler_ms)} | "
            f"{straggler_frac*100:.1f}% | {_fmt_ms(cadence_ms)} |"
        )
        step_sums.append((wall_ms, pct[75], mx))

    push("")
    if cadences:
        c_pct = _pcts(cadences, (50, 75, 90, 99))
        mean_cadence = sum(cadences) / len(cadences)
        push(
            f"- Step-to-step cadence: mean={_fmt_ms(mean_cadence)} "
            f"p50={_fmt_ms(c_pct[50])} p90={_fmt_ms(c_pct[90])} max={_fmt_ms(max(cadences))}"
        )
    if step_sums:
        avg_wait = sum((mx - p75) for _, p75, mx in step_sums) / len(step_sums)
        avg_wall = sum(w for w, _, _ in step_sums) / len(step_sums)
        push(
            f"- Mean per-step tail wait (max-p75): **{_fmt_ms(avg_wait)}** "
            f"out of mean step wall={_fmt_ms(avg_wall)} "
            f"(**{avg_wait/max(1e-9,avg_wall)*100:.1f}%** of step spent on tail)"
        )
    push("")

    # Finish reasons
    if ray_driver_log and os.path.exists(ray_driver_log):
        push("## Finish-reason breakdown per step (from ray-driver.log)")
        push("")
        per_step = parse_finish_reasons_per_step(ray_driver_log)
        push(
            "| step | tool_calls | length | stop | length % |"
        )
        push("|---|---|---|---|---|")
        for i, c in enumerate(per_step):
            tc = c.get("tool_calls", 0)
            ln = c.get("length", 0)
            st = c.get("stop", 0)
            tot = tc + ln + st
            push(
                f"| {i} | {tc} | {ln} | {st} | "
                f"{(100*ln/tot) if tot else 0:.1f}% |"
            )
        push("")
        totals = Counter()
        for c in per_step:
            totals.update(c)
        tot = sum(totals.values())
        if tot:
            push(
                f"- Overall: tool_calls={totals['tool_calls']} "
                f"length={totals['length']} stop={totals['stop']} "
                f"(length rate = {100*totals['length']/tot:.1f}%)"
            )
        push("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        required=True,
        help="Path (or host:/abs/path) to episodes.jsonl",
    )
    ap.add_argument(
        "--ray-driver",
        help="Optional local path to ray-driver.log for finish-reason breakdown",
    )
    ap.add_argument("--label", required=True, help="Label for the report header")
    ap.add_argument(
        "--num-prompts-per-step",
        type=int,
        default=8,
        help="num_prompts_per_step (from GRPO config; default 8)",
    )
    ap.add_argument(
        "--num-generations-per-prompt",
        type=int,
        default=16,
        help="num_generations_per_prompt (default 16)",
    )
    args = ap.parse_args()

    ep_path = _resolve_path(args.episodes)
    rd_path = _resolve_path(args.ray_driver) if args.ray_driver else None

    eps = load_episodes(ep_path)
    step_size = args.num_prompts_per_step * args.num_generations_per_prompt
    print(render(eps, args.label, step_size, rd_path))


if __name__ == "__main__":
    main()
