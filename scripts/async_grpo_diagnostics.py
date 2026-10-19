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

"""Async-GRPO bottleneck diagnostics.

Diagnostics that complement `rollout_timing_report.py` and
`phoenix_timing_report.py` for runs using `async_grpo.enabled=true` +
non-colocated vLLM (the v0508c+ topology):

1. **Replay-buffer starvation**: parses ray-driver.log for
   `Step coordination: training_step=N, max_age=M, buffer_size=K` lines and
   reports the fraction of poll cycles where `buffer_size=0` (= training is
   idle waiting for fresh trajectories) per step. High starvation rate is
   strong evidence that the run is **inference-bound** and adding more vLLM
   workers will help; low starvation indicates the bottleneck is elsewhere.

1c. **Inference over-production ratio**: counts how many chat_request
   completions occurred per trajectory the trainer drained. A
   sustained ratio > 1.0 with non-zero `⏸️ Pausing collection` backpressure
   would *normally* be the trainer-bound signal -- but this number is
   ALSO inflated by short runs that were killed before the trainer could
   consume the in-flight rollouts. We only report a real
   `steady_state_aged_out_pct` for runs long enough to have reached
   steady state (>= 20 training steps OR >= 3h of trainer wall, the
   threshold below which the killed-run-in-flight wedge dominates).
   See [internal_docs/docs/training_scaling_curve.md](../internal_docs/docs/training_scaling_curve.md)
   "ep/h was an inference-completion metric" post-mortem for why the
   old aged-out formula over-counted.

2. **vLLM aggregate utilization**: estimates inference-GPU utilization as
   `aggregate_LLM_wall / (num_inference_GPUs * elapsed_seconds)` using the
   per-rollout LLM-wall estimate from the Phoenix DB.

3. **Tool-service load**: parses `services-*.log` for pyexec activity (start
   and end of each `Executing code` block) and reports request volume,
   in-flight concurrency over time (peak + mean), failure rate, and total
   active wall time. Vina utilization is reported indirectly via the cache
   hit-rate on `tool.vina_dock` spans.

Usage:
    python scripts/async_grpo_diagnostics.py \
        --ray-driver /tmp/v0508c/ray-driver.log \
        --phoenix-db /tmp/v0508c/phoenix.db \
        --services /tmp/v0508c/services-0.log \
        --num-inference-gpus 32 \
        --label "v0508c 12n async (4t+4i+4s)"
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Diagnostic 1: replay-buffer starvation
# ---------------------------------------------------------------------------

_BUFFER_RE = re.compile(
    r"training_step=(\d+),\s*max_age=(\d+),\s*buffer_size=(\d+)"
)


def parse_buffer_lines(ray_driver_log: Path) -> dict[int, list[int]]:
    """Return {training_step: [buffer_size, ...observations...]}."""
    per_step: dict[int, list[int]] = defaultdict(list)
    with open(ray_driver_log, errors="ignore") as f:
        for line in f:
            m = _BUFFER_RE.search(line)
            if m:
                per_step[int(m.group(1))].append(int(m.group(3)))
    return per_step


def buffer_starvation_report(per_step: dict[int, list[int]]) -> str:
    if not per_step:
        return "## Replay-buffer starvation\n\nNo `Step coordination:` lines found in ray-driver.log.\n"

    lines = ["## Replay-buffer starvation (async-GRPO inference-bound signal)", ""]
    lines.append(
        "When training is async-GRPO with `async_grpo.enabled=true`, the policy "
        "trainer polls a shared `ReplayBuffer` for fresh trajectories. Each poll "
        "logs `Step coordination: training_step=N, max_age=M, buffer_size=K`. "
        "`buffer_size=0` means training is **idle**, waiting for inference to "
        "produce more rollouts. A high starvation rate is a smoking gun for "
        "inference being the bottleneck."
    )
    lines.append("")

    total = sum(len(v) for v in per_step.values())
    zeros = sum(1 for v in per_step.values() for b in v if b == 0)
    lines.append(
        f"- Total poll cycles observed: **{total}**  "
        f"(steps {min(per_step)}-{max(per_step)})"
    )
    lines.append(
        f"- Aggregate starvation rate (buffer_size=0): **{zeros}/{total} = "
        f"{100*zeros/max(1,total):.1f}%**"
    )
    lines.append("")
    lines.append("| step | starvation% | mean buf | max buf | n polls |")
    lines.append("|---|---|---|---|---|")
    for ts in sorted(per_step):
        bs = per_step[ts]
        n = len(bs)
        z = sum(1 for b in bs if b == 0)
        mean_b = sum(bs) / n
        max_b = max(bs)
        # Skip "1-poll" rows (transition lines printed once at boundary)
        if n <= 1:
            continue
        lines.append(
            f"| {ts} | {100*z/n:.1f}% | {mean_b:.1f} | {max_b} | {n} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostic 1b: refit pause windows + collector backpressure
# ---------------------------------------------------------------------------
#
# `nemo_rl/algorithms/async_utils.py` already prints structured markers on
# every refit cycle. We pair them up here:
#
#   🔄 Preparing for refit: pausing new generations...      <- refit window opens
#   ✅ Ready for refit (took {prep:.2f}s)                   <- prep done
#   🔄 Resuming generation starts after refit                <- refit window closes
#
# and we count "already-exist-in-buffer" backpressure events:
#
#   ⏸️ Pausing collection: all target weights [...] for weight version N
#       already exist in buffer. Waiting for weight update...
#
# That last line fires whenever the collector has accumulated trajectories
# targeting every reachable weight version in [v ... v+max_age-1] and is
# being throttled because training hasn't drained the buffer yet. A high
# rate of these means the run is **training-bound**, not inference-bound.

_PREPARING_RE = re.compile(r"🔄 Preparing for refit")
_READY_RE = re.compile(r"✅ Ready for refit \(took ([0-9.]+)s\)")
_RESUMING_RE = re.compile(r"🔄 Resuming generation starts after refit")
_BACKPRESSURE_RE = re.compile(
    r"⏸️ Pausing collection: all target weights .* already exist in buffer"
)
_PENDING_THREADS_RE = re.compile(
    r"⏸️ Waiting for refit to complete before starting new generation \((\d+) threads"
)
_PAUSING_REFIT_RE = re.compile(r"⏸️ Pausing collection for refit")


def parse_refit_markers(ray_driver_log: Path) -> dict:
    """Count refit markers and backpressure events.

    Ray-emitted prints in ray-driver.log do not carry inline timestamps
    (only `[36m(AsyncTrajectoryCollector pid=...)[0m` prefixes), so we
    cannot derive refit-window walls from this file. The `weight_sync`
    timer in the trainer (already exported to tfevents) measures the
    actual refit prep + transfer wall, so we focus here on cycle counts
    and backpressure incidence.
    """
    n_preparing = 0
    n_ready = 0
    n_resuming = 0
    n_pausing_for_refit = 0
    prep_times: list[float] = []
    pending_threads_samples: list[int] = []
    backpressure_count = 0

    with open(ray_driver_log, errors="ignore") as f:
        for line in f:
            if _BACKPRESSURE_RE.search(line):
                backpressure_count += 1
                continue
            if _PREPARING_RE.search(line):
                n_preparing += 1
                continue
            m = _READY_RE.search(line)
            if m:
                n_ready += 1
                prep_times.append(float(m.group(1)))
                continue
            if _RESUMING_RE.search(line):
                n_resuming += 1
                continue
            if _PAUSING_REFIT_RE.search(line):
                n_pausing_for_refit += 1
                continue
            m = _PENDING_THREADS_RE.search(line)
            if m:
                pending_threads_samples.append(int(m.group(1)))

    return {
        "n_preparing": n_preparing,
        "n_ready": n_ready,
        "n_resuming": n_resuming,
        "n_pausing_for_refit": n_pausing_for_refit,
        "prep_times": prep_times,
        "pending_threads_samples": pending_threads_samples,
        "backpressure_count": backpressure_count,
    }


def refit_marker_report(parsed: dict) -> str:
    lines = ["## Refit cycles + collector backpressure (ray-driver.log markers)", ""]
    lines.append(
        "Counts the structured markers emitted by "
        "`nemo_rl/algorithms/async_utils.py` in async-GRPO. The trainer-"
        "side `weight_sync` and `exposed_generation` timers in tfevents "
        "give the actual wall costs; this section attributes their "
        "**incidence** and surfaces the **backpressure signal** (the "
        "smoking gun for training-bound vs inference-bound)."
    )
    lines.append("")

    if parsed["n_preparing"] == 0 and parsed["backpressure_count"] == 0:
        lines.append("No async-GRPO markers found.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- 🔄 Preparing for refit cycles: **{parsed['n_preparing']}**")
    lines.append(f"- ✅ Ready for refit completions: **{parsed['n_ready']}**")
    lines.append(f"- 🔄 Resuming generation cycles: **{parsed['n_resuming']}**")
    lines.append(
        f"- ⏸️ Pausing collection for refit (collector-paused-mid-step): "
        f"**{parsed['n_pausing_for_refit']}**"
    )

    pt = parsed["prep_times"]
    if pt:
        s = sorted(pt)
        n = len(s)
        lines.append(
            f"- `Ready for refit (took Xs)` prep wait: "
            f"n={n} mean={sum(pt)/n:.3f}s median={s[n//2]:.3f}s "
            f"p90={s[min(n-1, int(0.9*(n-1)))]:.3f}s max={max(pt):.3f}s "
            f"sum={sum(pt):.1f}s"
        )
        lines.append(
            "  - Sub-second prep means the collector had no in-flight "
            "rollouts to drain at refit time (rare in practice for "
            "long-rollout runs)."
        )

    pts = parsed["pending_threads_samples"]
    if pts:
        lines.append(
            f"- `⏸️ Waiting for refit to complete before starting new generation` "
            f"events: **{len(pts)}** (pending threads "
            f"min={min(pts)} mean={sum(pts)/len(pts):.1f} max={max(pts)})"
        )
    else:
        lines.append(
            "- No `⏸️ Waiting for refit to complete before starting new generation` "
            "events; refits never landed in the middle of an active rollout "
            "batch."
        )

    bp = parsed["backpressure_count"]
    lines.append("")
    lines.append(
        f"- **`⏸️ Pausing collection: ... already exist in buffer` "
        f"backpressure events: {bp}**"
    )
    lines.append(
        "  - Each event = the collector was throttled because the buffer "
        "already held trajectories at every reachable weight version in "
        "`[v ... v+max_age-1]`. High counts here are the smoking gun for "
        "the run being **training-bound** (the trainer is the slow leg), "
        "not inference-bound."
    )
    if parsed["n_resuming"] > 0:
        ratio = bp / parsed["n_resuming"]
        lines.append(
            f"  - Backpressure events per refit cycle: **{ratio:.1f}**"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostic 1c: inference over-production + (long-run only) aged-out%
# ---------------------------------------------------------------------------
#
# Original framing (deprecated 2026-05-17):
#
#     aged_out_pct = (N_completed - N_consumed - final_buffer_size) / N_completed
#
# Failure mode: for runs killed mid-bootstrap (e.g. v0517a's 87 min, 3 steps),
# this counts every rollout that *would have been* consumed by a future
# step as "aged out", massively overstating wastage. The 65% number we
# reported for v0517a was actually almost entirely in-flight-at-cancel,
# not aged out.
#
# New framing:
#
#  (1) inference_over_production_ratio = N_completed / N_consumed
#      Always reported. >1.0 = inference produced more than trainer drained
#      during the run window. Could be waste (steady-state aging) OR could
#      be in-flight WIP that would have been consumed in a longer run.
#
#  (2) steady_state_aged_out_pct = 1 - 1/over_production_ratio
#      Reported ONLY for "long" runs (>= 20 training steps OR >= 3h of
#      trainer wall window from tfevents). Below this threshold the
#      killed-run-in-flight wedge dominates and the number is not
#      interpretable as "wasted inference work".
#
# A rollout is "actually aged out" when its chat_request.end fired but the
# trajectory was older than `max_age` weight versions by the time the
# trainer tried to sample it. Distinguishing this from "still in flight at
# termination" requires either (a) timestamp-aligned consumption events
# (which we don't have -- the Step coordination lines have no inline ts) or
# (b) a long enough steady-state window that the in-flight wedge becomes
# negligible relative to the cumulative production-consumption gap.

def parse_step_coordination_summary(
    per_step: dict[int, list[int]],
) -> dict:
    """Summarise per-step buffer state into the inputs the aged-out
    estimator needs.
    """
    if not per_step:
        return {
            "n_steps": 0,
            "max_step": 0,
            "min_step": 0,
            "final_buffer_size": 0,
            "max_buffer_size": 0,
        }
    max_step = max(per_step)
    min_step = min(per_step)
    final_buffer_size = per_step[max_step][-1]
    max_buffer_size = max(b for v in per_step.values() for b in v)
    return {
        "n_steps": max_step - min_step + 1,
        "max_step": max_step,
        "min_step": min_step,
        "final_buffer_size": final_buffer_size,
        "max_buffer_size": max_buffer_size,
    }


def aged_out_report(
    phoenix_db: Path,
    per_step: dict[int, list[int]],
    train_global_batch_size: int,
    trainer_wall_window_s: Optional[float] = None,
) -> str:
    """Report inference over-production and (for long runs) aged-out%.

    `trainer_wall_window_s`, if provided, is the tfevents-derived wall window
    spanned by all training step events. Used to decide whether the run is
    long enough to interpret over-production as steady-state aging.
    """
    summary = parse_step_coordination_summary(per_step)

    conn = sqlite3.connect(str(phoenix_db))
    n_completed_row = conn.execute(
        "SELECT COUNT(*) FROM spans "
        "WHERE name='chat_request' AND end_time IS NOT NULL"
    ).fetchone()
    n_completed = int(n_completed_row[0]) if n_completed_row else 0

    lines = [
        "## Inference over-production + aged-out% (long-run steady-state only)",
        "",
    ]
    lines.append(
        "Inference over-production ratio = chat_request completions per "
        "trajectory the trainer drained during the observed step window. "
        "Above 1.0 means inference produced more than the trainer consumed; "
        "the deficit is **either** trajectories that aged out of the "
        "`max_age` window (real waste) **or** trajectories that were "
        "in-flight at the moment the run was cancelled (would have been "
        "consumed in a longer run, not real waste). The two cannot be "
        "distinguished from log files alone; we only report a "
        "steady-state aged-out% estimate for runs long enough that the "
        "in-flight wedge becomes negligible (>= 20 training steps OR "
        ">= 3h of trainer wall window). See "
        "[internal_docs/docs/training_scaling_curve.md](../internal_docs/docs/training_scaling_curve.md) "
        "\"ep/h was an inference-completion metric\" post-mortem for why "
        "the old aged-out formula was misleading for short probes."
    )
    lines.append("")

    if n_completed == 0 or summary["n_steps"] == 0:
        lines.append(
            "No chat_request spans or no Step coordination lines found; "
            "cannot compute over-production ratio."
        )
        lines.append("")
        return "\n".join(lines)

    n_steps_window = max(1, summary["max_step"] - summary["min_step"] + 1)
    n_consumed_window = n_steps_window * train_global_batch_size
    production_ratio = n_completed / max(1, n_consumed_window)

    lines.append(
        f"- Completed rollouts (chat_request.end fired): **{n_completed}**"
    )
    lines.append(
        f"- Step range observed: **{summary['min_step']} .. "
        f"{summary['max_step']}** ({n_steps_window} steps)"
    )
    lines.append(
        f"- train_global_batch_size: **{train_global_batch_size}**  "
        f"(pass `--train-global-batch-size` to override; should be "
        f"`num_prompts_per_step * num_generations_per_prompt`)"
    )
    lines.append(
        f"- Trajectories consumed by trainer during window: "
        f"**{n_consumed_window}** = {n_steps_window} steps * "
        f"{train_global_batch_size}"
    )
    lines.append(
        f"- Final buffer residual (still consumable at termination): "
        f"**{summary['final_buffer_size']}**  "
        f"(prompts; multiply by num_generations_per_prompt for trajectories)"
    )
    lines.append(
        f"- Max buffer residual seen across the run: "
        f"**{summary['max_buffer_size']}**"
    )
    lines.append("")
    lines.append(
        f"- **Inference over-production ratio: {production_ratio:.2f}**  "
        f"(N_completed / N_consumed_window)"
    )

    # Decide if this run is "long enough" to interpret over-production as
    # aging. Either the step count or the trainer wall window can qualify.
    long_run_steps = n_steps_window >= 20
    long_run_wall = (
        trainer_wall_window_s is not None and trainer_wall_window_s >= 3 * 3600
    )
    if long_run_steps or long_run_wall:
        # In steady state, over-production above 1.0 is mostly aging because
        # the in-flight wedge is bounded by the buffer ceiling and amortises
        # away as the run gets long.
        excess = max(0.0, production_ratio - 1.0)
        aged_out_pct = 100.0 * excess / production_ratio
        wall_str = (
            _fmt_h_local(trainer_wall_window_s)
            if trainer_wall_window_s
            else "n/a"
        )
        lines.append(
            f"- **Steady-state aged-out%: {aged_out_pct:.1f}%**  "
            f"(qualifies as 'long run': "
            f"steps={n_steps_window}, trainer wall={wall_str})"
        )
        lines.append("")
        if aged_out_pct >= 25.0:
            lines.append(
                "  > Verdict: inference over-producing in steady state. "
                "Real aged-out waste; the trainer is the binding constraint. "
                "Pair with the `⏸️ Pausing collection` backpressure count: "
                "if events/refit-cycle is also >= 1.0, scaling the trainer "
                "(more train nodes, larger batch) is the right next lever."
            )
        elif aged_out_pct >= 10.0:
            lines.append(
                "  > Verdict: mild steady-state aging. Trainer is nearly "
                "keeping up with inference; modest trainer scale-up would "
                "close the gap."
            )
        else:
            lines.append(
                "  > Verdict: aging negligible. Trainer is draining "
                "inference output; bottleneck is elsewhere (judge throttle, "
                "per-rollout wall, buffer width, etc)."
            )
    else:
        wall_str = (
            _fmt_h_local(trainer_wall_window_s)
            if trainer_wall_window_s
            else "n/a"
        )
        lines.append(
            f"- Steady-state aged-out%: **not computed** (run too short: "
            f"only {n_steps_window} steps observed and "
            f"trainer wall={wall_str}; need >= 20 steps OR >= 3h trainer "
            f"wall to separate aging from in-flight-at-cancel)"
        )
        lines.append("")
        lines.append(
            "  > Interpretation guidance: ratio above is the **upper bound** "
            "on aging. The true aged-out% is somewhere between 0 (all "
            "excess was in-flight, would have been consumed in a longer "
            "run) and the upper-bound estimate. Re-run for >= 3h trainer "
            "wall to read the steady-state number."
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_h_local(seconds: Optional[float]) -> str:
    if not seconds:
        return "n/a"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


# ---------------------------------------------------------------------------
# Diagnostic 2: vLLM aggregate utilization
# ---------------------------------------------------------------------------


def _parse_phoenix_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s.split("+")[0].split(".")[0], "%Y-%m-%d %H:%M:%S")


def vllm_utilization_report(
    phoenix_db: Path, num_inference_gpus: int
) -> str:
    """Estimate aggregate vLLM utilization from Phoenix spans.

    Live tree (post-2026-05-27): sum of ``policy_llm.responses`` span
    walls divided by ``num_inference_gpus * span_window_elapsed``.
    These walls are the real vLLM HTTP wall time recorded by the
    ``aevom_simple_agent`` overlay -- no longer an estimate.

    Synthetic fallback (older runs): per-iteration LLM time is estimated
    as ``sum(reasoning_iteration.wall - tool_children.wall)``.
    """
    conn = sqlite3.connect(str(phoenix_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT span_id, parent_id, trace_rowid, name, start_time, end_time "
        "FROM spans"
    ).fetchall()

    spans_by_trace: dict[int, list] = defaultdict(list)
    span_by_id: dict[str, dict] = {}
    children: dict[str, list[dict]] = defaultdict(list)

    has_live = False
    for r in rows:
        if r["name"] == "policy_llm.responses":
            has_live = True
        s = {
            "span_id": r["span_id"],
            "parent_id": r["parent_id"],
            "trace_rowid": r["trace_rowid"],
            "name": r["name"],
            "start": _parse_phoenix_ts(r["start_time"]),
            "end": _parse_phoenix_ts(r["end_time"]),
        }
        spans_by_trace[r["trace_rowid"]].append(s)
        span_by_id[r["span_id"]] = s
        if r["parent_id"]:
            children[r["parent_id"]].append(s)

    total_llm_s = 0.0
    n_iters = 0
    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None

    if has_live:
        for spans in spans_by_trace.values():
            for s in spans:
                if s["name"] != "policy_llm.responses":
                    continue
                wall = (s["end"] - s["start"]).total_seconds()
                total_llm_s += max(0.0, wall)
                n_iters += 1
                if earliest is None or s["start"] < earliest:
                    earliest = s["start"]
                if latest is None or s["end"] > latest:
                    latest = s["end"]
        source = "live `policy_llm.responses`"
    else:
        for spans in spans_by_trace.values():
            for s in spans:
                if not s["name"].startswith("reasoning_iteration."):
                    continue
                iter_wall = (s["end"] - s["start"]).total_seconds()
                tool_wall = sum(
                    (c["end"] - c["start"]).total_seconds()
                    for c in children.get(s["span_id"], [])
                    if c["name"].startswith("tool.")
                    and c["name"] != "tool_service.http"
                )
                total_llm_s += max(0.0, iter_wall - tool_wall)
                n_iters += 1
                if earliest is None or s["start"] < earliest:
                    earliest = s["start"]
                if latest is None or s["end"] > latest:
                    latest = s["end"]
        source = "estimated `reasoning_iteration.wall - tool_children.wall`"

    if earliest is None:
        return (
            "## vLLM aggregate utilization\n\n"
            "No policy_llm.responses or reasoning_iteration spans found.\n"
        )

    elapsed_s = (latest - earliest).total_seconds()
    util = total_llm_s / (elapsed_s * num_inference_gpus)

    lines = ["## vLLM aggregate utilization", ""]
    lines.append(
        f"Inference-GPU utilization sourced from **{source}**, divided by "
        f"`num_inference_gpus * span_window_elapsed`."
    )
    lines.append("")
    lines.append(
        f"- LLM-call samples: **{n_iters}**"
    )
    lines.append(
        f"- Aggregate LLM wall: **{total_llm_s/3600:.2f}h** "
        f"({total_llm_s:.0f}s)"
    )
    lines.append(
        f"- Span window elapsed: **{elapsed_s/3600:.2f}h** "
        f"({elapsed_s:.0f}s)"
    )
    lines.append(
        f"- Inference GPUs available:  **{num_inference_gpus}**"
    )
    lines.append(
        f"- Aggregate utilization:  **{100*util:.1f}%**"
    )
    lines.append("")
    lines.append(
        "Interpretation: <50% with high replay-buffer starvation = inference is "
        "starved by per-rollout queue depth (more vLLM workers will help even "
        "though average GPU util is low). >70% with high starvation = vLLM is "
        "compute-bound, more inference nodes give roughly linear gains. "
        "Anything with low starvation = inference is over-provisioned."
    )
    lines.append("")
    return "\n".join(lines)


def tool_service_host_distribution_report(phoenix_db: Path) -> str:
    """Per-service-host distribution of container tool HTTP attempts.

    Sourced from the live ``tool_service.http`` spans (post-2026-05-27)'s
    ``tool.service_host`` attribute. Reveals whether load is balanced
    across the service node pool, e.g., the v0520a-multijudge open
    question about vina-only services 1 & 2 going idle.
    """
    conn = sqlite3.connect(str(phoenix_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT attributes, start_time, end_time "
        "FROM spans WHERE name='tool_service.http' AND end_time IS NOT NULL"
    ).fetchall()

    if not rows:
        return (
            "## Tool-service host distribution\n\n"
            "No `tool_service.http` spans found (pre-2026-05-27 run "
            "or telemetry off).\n"
        )

    from collections import Counter

    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        attrs = r["attributes"]
        if isinstance(attrs, (bytes, str)) and attrs:
            try:
                attrs = json.loads(attrs)
            except Exception:
                attrs = {}
        flat: dict = {}

        def _walk(prefix: str, val) -> None:
            if isinstance(val, dict):
                for k, v in val.items():
                    key = f"{prefix}.{k}" if prefix else k
                    _walk(key, v)
            else:
                flat[prefix] = val

        if isinstance(attrs, dict):
            _walk("", attrs)
        tool = str(flat.get("tool.name", "?"))
        host = str(flat.get("tool.service_host", "?"))
        wall = (
            _parse_phoenix_ts(r["end_time"]) - _parse_phoenix_ts(r["start_time"])
        ).total_seconds() * 1000.0
        by_pair[(tool, host)].append(wall)

    lines = ["## Tool-service host distribution", ""]
    lines.append("| tool | service_host | n | mean_ms | p90_ms | total_s |")
    lines.append("|---|---|---|---|---|---|")
    items = sorted(by_pair.items(), key=lambda kv: (kv[0][0], -len(kv[1])))
    for (tool, host), xs in items:
        xs_sorted = sorted(xs)
        p90 = xs_sorted[max(0, int(0.9 * (len(xs_sorted) - 1)))]
        lines.append(
            f"| `{tool}` | `{host}` | {len(xs)} | "
            f"{sum(xs)/len(xs):.0f} | {p90:.0f} | {sum(xs)/1000:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostic 3: tool-service load (pyexec)
# ---------------------------------------------------------------------------

_PYEXEC_START_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s.*INFO - Executing code"
)
_PYEXEC_OK_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s.*INFO - Code executed successfully"
)
_PYEXEC_FAIL_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s.*WARNING - Code execution failed"
)


def _parse_svc_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def pyexec_load_report(services_logs: Iterable[Path]) -> str:
    starts: list[datetime] = []
    ends: list[datetime] = []
    failures = 0
    for log in services_logs:
        if not Path(log).exists():
            continue
        with open(log, errors="ignore") as f:
            for line in f:
                m = _PYEXEC_START_RE.match(line)
                if m:
                    starts.append(_parse_svc_ts(m.group(1)))
                    continue
                m = _PYEXEC_OK_RE.match(line)
                if m:
                    ends.append(_parse_svc_ts(m.group(1)))
                    continue
                m = _PYEXEC_FAIL_RE.match(line)
                if m:
                    ends.append(_parse_svc_ts(m.group(1)))
                    failures += 1

    lines = ["## Tool-service load: pyexec", ""]
    if not starts:
        lines.append("No pyexec activity found.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "pyexec runs only on the **first** service node (the full-stack node) "
        "as a single uvicorn process. If concurrency approaches the worker "
        "count or service wall approaches run wall, scaling rollouts will "
        "saturate this service and we'll need to spread pyexec across more "
        "service nodes."
    )
    lines.append("")

    # Concurrency over time
    events = [(t, 1) for t in starts] + [(t, -1) for t in ends]
    events.sort(key=lambda x: x[0])
    in_flight = 0
    peak = 0
    sample_total = 0
    sample_n = 0
    for _ts, delta in events:
        in_flight += delta
        if in_flight > peak:
            peak = in_flight
        sample_total += in_flight
        sample_n += 1
    mean_in_flight = sample_total / max(1, sample_n)

    active_wall_s = (max(ends) - min(starts)).total_seconds() if ends else 0.0

    lines.append(f"- Executions started:  **{len(starts)}**")
    lines.append(f"- Executions ended:    **{len(ends)}**")
    lines.append(
        f"- Failure rate (sandbox/syntax/runtime errors): **{failures}/{len(ends)} = "
        f"{100*failures/max(1,len(ends)):.1f}%**"
    )
    lines.append(f"- Peak concurrent in-flight: **{peak}**")
    lines.append(
        f"- Mean concurrent in-flight (across event ticks): **{mean_in_flight:.2f}**"
    )
    lines.append(
        f"- Active wall:  **{active_wall_s/60:.1f} min**  "
        f"({active_wall_s:.0f}s)"
    )
    lines.append(
        f"- Throughput:  **{60*len(starts)/max(1,active_wall_s):.1f} executions/min**"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bonus: vina cache hit rate (from Phoenix)
# ---------------------------------------------------------------------------


def vina_cache_report(phoenix_db: Path) -> str:
    """Report cache hit-rate on tool.vina_dock spans by latency bucket. The
    cache decision is reflected in latency: <1s = cache hit, >1s = real dock."""
    conn = sqlite3.connect(str(phoenix_db))
    rows = conn.execute(
        "SELECT (julianday(end_time) - julianday(start_time)) * 86400 AS lat "
        "FROM spans WHERE name = 'tool.vina_dock'"
    ).fetchall()
    total = len(rows)
    if total == 0:
        return ""
    cached = sum(1 for r in rows if r[0] < 1.0)
    lines = ["## Tool-service load: vina cache", ""]
    lines.append(
        f"- vina_dock spans: **{total}**"
    )
    lines.append(
        f"- Cache hits (latency < 1s):  **{cached}/{total} = "
        f"{100*cached/total:.1f}%**"
    )
    lines.append(
        f"- Real docks executed: **{total - cached}**  (the only ones that "
        f"actually use the 32 vina GPU instances)"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _trainer_wall_window_from_tfevents(tb_dir: Optional[Path]) -> Optional[float]:
    """Return the wall window (seconds) between the first and last
    `timing/train/total_step_time` event in the given tensorboard dir.

    Used by the aged-out% diagnostic to decide whether the run is
    long-enough to interpret over-production as steady-state aging.
    Quietly returns None when tfevents is missing or unparseable.
    """
    if tb_dir is None or not tb_dir.is_dir():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return None
    if "timing/train/total_step_time" not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars("timing/train/total_step_time")
    if len(events) < 2:
        return None
    return events[-1].wall_time - events[0].wall_time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ray-driver", required=True, type=Path)
    ap.add_argument("--phoenix-db", required=True, type=Path)
    ap.add_argument(
        "--services",
        nargs="+",
        type=Path,
        required=True,
        help="One or more services-*.log files (full-stack node has pyexec).",
    )
    ap.add_argument("--num-inference-gpus", type=int, required=True)
    ap.add_argument(
        "--train-global-batch-size",
        type=int,
        default=128,
        help=(
            "Number of trajectories consumed per training step. Used by the "
            "over-production / aged-out%% estimator. Set to "
            "`num_prompts_per_step * num_generations_per_prompt` for the run "
            "being analysed (v0509c baseline: 128; 1.5x scale-up probe: 192)."
        ),
    )
    ap.add_argument(
        "--tfevents-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to the run's `exp_*/tensorboard` directory. Used "
            "to read the trainer wall window so the aged-out%% diagnostic "
            "can decide whether the run qualifies as 'long enough' for a "
            "steady-state estimate (>= 20 steps OR >= 3h trainer wall)."
        ),
    )
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    print(f"# Async-GRPO diagnostics: {args.label}\n")
    per_step = parse_buffer_lines(args.ray_driver)
    print(buffer_starvation_report(per_step))
    trainer_wall_s = _trainer_wall_window_from_tfevents(args.tfevents_dir)
    print(aged_out_report(
        args.phoenix_db, per_step, args.train_global_batch_size,
        trainer_wall_window_s=trainer_wall_s,
    ))
    refit = parse_refit_markers(args.ray_driver)
    print(refit_marker_report(refit))
    print(vllm_utilization_report(args.phoenix_db, args.num_inference_gpus))
    print(tool_service_host_distribution_report(args.phoenix_db))
    print(pyexec_load_report(args.services))
    print(vina_cache_report(args.phoenix_db))


if __name__ == "__main__":
    main()
