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

"""Per-training-step timing decomposition from tfevents.

Reads the tensorboard event file written by NeMo-RL's `Timer` instrumentation
and produces a per-step breakdown of where wall-clock time goes. Optionally
compares two or more runs side-by-side.

The async-GRPO loop in `nemo_rl/algorithms/grpo.py` wraps each phase in
`timer.time(...)` and exports under `timing/train/`. We surface the most
informative tags:

  * total_step_time              - wall of the whole step (numerator)
  * exposed_generation           - synchronous wait on rollouts (= "training is
                                   blocked on inference for fresh trajectories")
  * weight_sync                  - vLLM weight-refit pause cost
  * policy_training              - Megatron backprop wall
  * policy_and_reference_logprobs- forward passes for advantages
  * reward_calculation           - reward server roundtrip
  * data_processing, training_prep, advantage_calculation - bookkeeping

We also pull from `train/timing/rollout/`:

  * total                        - per-step rollout-side wall
  * run_rollouts / await_results - sub-decomposition

And related throughput / buffer signals from `train/`:

  * buffer_size, avg_trajectory_age
  * natural_termination_rate, truncation_rate
  * mean_gen_tokens_per_sample, turns_per_sample/mean
  * global_valid_seqs, total_reward/mean, num_tool_calls/mean

Usage:
    python scripts/training_step_timings.py /tmp/tfevents/v0509c
    python scripts/training_step_timings.py /tmp/tfevents/v0508c \\
        /tmp/tfevents/v0509c /tmp/tfevents/v0510a /tmp/tfevents/v0510b \\
        --compare
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Tags we care about, grouped by section.
TIMING_TRAIN_TAGS = [
    "timing/train/total_step_time",
    "timing/train/exposed_generation",
    "timing/train/weight_sync",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
    "timing/train/reward_calculation",
    "timing/train/training_prep",
    "timing/train/data_processing",
    "timing/train/advantage_calculation",
    "timing/train/logprob_inference_prep",
    "timing/train/checkpointing",
]

ROLLOUT_TAGS = [
    "train/timing/rollout/total",
    "train/timing/rollout/run_rollouts",
    "train/timing/rollout/await_results",
    "train/timing/rollout/postprocess_results",
    "train/timing/rollout/aggregate_metrics",
]

PIPELINE_TAGS = [
    "train/buffer_size",
    "train/avg_trajectory_age",
    "train/natural_termination_rate",
    "train/truncation_rate",
    "train/global_valid_seqs",
    "train/mean_gen_tokens_per_sample",
    "train/turns_per_sample/mean",
    "train/aevom_simple_agent/num_tool_calls/mean",
    "train/aevom_simple_agent/num_tool_errors/mean",
    "train/total_reward/mean",
    "train/aevom_simple_agent/judge_reasoning_reward/mean",
]


def _load(run_dir: Path) -> EventAccumulator:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _series(ea: EventAccumulator, tag: str) -> list[tuple[int, float]]:
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "max": float("nan"), "sum": 0.0}
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "mean": sum(xs) / n,
        "median": s[n // 2],
        "p90": s[min(n - 1, int(0.9 * (n - 1)))],
        "max": s[-1],
        "sum": sum(xs),
    }


def _fmt_s(v: float) -> str:
    if v != v:
        return "n/a"
    if v < 0.1:
        return f"{v*1000:.0f}ms"
    if v < 60:
        return f"{v:.1f}s"
    if v < 3600:
        return f"{v/60:.1f}m"
    return f"{v/3600:.2f}h"


def _fmt_n(v: float, digits: int = 1) -> str:
    if v != v:
        return "n/a"
    return f"{v:.{digits}f}"


# ----------------------------------------------------------------------
# Single-run report
# ----------------------------------------------------------------------


def report_single(run_dir: Path, label: str) -> str:
    ea = _load(run_dir)
    tags = ea.Tags()["scalars"]
    lines: list[str] = []
    push = lines.append

    push(f"# Step-timing decomposition: {label}")
    push("")
    push(f"- tfevents: `{run_dir}`")
    push(f"- Total scalar tags: {len(tags)}")

    # Determine number of steps from total_step_time
    total = _series(ea, "timing/train/total_step_time")
    if not total:
        push("")
        push("No `timing/train/total_step_time` tag found - was the run async-GRPO?")
        return "\n".join(lines) + "\n"
    push(f"- Steps observed: **{len(total)}**")
    last_step = total[-1][0]
    first_step = total[0][0]
    push(f"- Step range: {first_step} -> {last_step}")
    push("")

    # ---------------- Train-loop timing ----------------
    push("## Train-loop wall (timing/train/*) - mean per step")
    push("")
    push("`exposed_generation` = synchronous wait on inference (training "
         "blocked on rollouts). `weight_sync` = vLLM weight-refit pause. "
         "`policy_training` = Megatron backprop. The rest are bookkeeping.")
    push("")
    push("| phase | n | mean | median | p90 | max | sum | %total |")
    push("|---|---|---|---|---|---|---|---|")
    total_sum = sum(v for _, v in total)
    rows = []
    for tag in TIMING_TRAIN_TAGS:
        vs = [v for _, v in _series(ea, tag)]
        if not vs:
            continue
        st = _stats(vs)
        pct = 100 * st["sum"] / total_sum if total_sum else float("nan")
        short = tag.replace("timing/train/", "")
        rows.append((short, st, pct))
    # Sort by sum descending (skip total_step_time being top)
    rows.sort(key=lambda r: -r[1]["sum"])
    for short, st, pct in rows:
        push(
            f"| `{short}` | {st['n']} | {_fmt_s(st['mean'])} | "
            f"{_fmt_s(st['median'])} | {_fmt_s(st['p90'])} | "
            f"{_fmt_s(st['max'])} | {_fmt_s(st['sum'])} | {pct:.1f}% |"
        )
    push("")

    # exposed_generation as a fraction of total_step_time per step (telling
    # us how much of training wall is just waiting for rollouts).
    expgen = dict(_series(ea, "timing/train/exposed_generation"))
    poltr = dict(_series(ea, "timing/train/policy_training"))
    wts = dict(_series(ea, "timing/train/weight_sync"))
    if expgen:
        ratios = []
        for step, t_total in total:
            if t_total > 0 and step in expgen:
                ratios.append(expgen[step] / t_total)
        if ratios:
            push(
                f"- `exposed_generation / total_step_time`: "
                f"mean={100*statistics.mean(ratios):.1f}%  "
                f"median={100*statistics.median(ratios):.1f}%  "
                f"max={100*max(ratios):.1f}%"
            )
    if poltr:
        ratios = []
        for step, t_total in total:
            if t_total > 0 and step in poltr:
                ratios.append(poltr[step] / t_total)
        if ratios:
            push(
                f"- `policy_training / total_step_time`: "
                f"mean={100*statistics.mean(ratios):.1f}%  "
                f"median={100*statistics.median(ratios):.1f}%  "
                f"max={100*max(ratios):.1f}%"
            )
    if wts:
        ws_vals = list(wts.values())
        push(
            f"- `weight_sync` per step: mean={_fmt_s(statistics.mean(ws_vals))}  "
            f"max={_fmt_s(max(ws_vals))}  "
            f"sum={_fmt_s(sum(ws_vals))} "
            f"({100*sum(ws_vals)/total_sum:.1f}% of all step time)"
        )
    push("")

    # ---------------- Rollout-side timing ----------------
    push("## Rollout-side wall (train/timing/rollout/*) - mean per step")
    push("")
    push("This is the rollout-collector view: how long it took to assemble a "
         "training batch from the replay buffer, including waiting for "
         "results to land.")
    push("")
    push("| phase | n | mean | median | p90 | max |")
    push("|---|---|---|---|---|---|")
    for tag in ROLLOUT_TAGS:
        vs = [v for _, v in _series(ea, tag)]
        if not vs:
            continue
        st = _stats(vs)
        short = tag.replace("train/timing/rollout/", "")
        push(
            f"| `{short}` | {st['n']} | {_fmt_s(st['mean'])} | "
            f"{_fmt_s(st['median'])} | {_fmt_s(st['p90'])} | "
            f"{_fmt_s(st['max'])} |"
        )
    push("")

    # ---------------- Pipeline / quality signals ----------------
    push("## Pipeline-state signals (train/* per step)")
    push("")
    push("| signal | mean | median | min | max |")
    push("|---|---|---|---|---|")
    for tag in PIPELINE_TAGS:
        vs = [v for _, v in _series(ea, tag)]
        if not vs:
            continue
        short = tag.replace("train/", "")
        st = _stats(vs)
        mn = min(vs)
        push(
            f"| `{short}` | {_fmt_n(st['mean'], 2)} | "
            f"{_fmt_n(st['median'], 2)} | {_fmt_n(mn, 2)} | "
            f"{_fmt_n(st['max'], 2)} |"
        )
    push("")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Multi-run side-by-side
# ----------------------------------------------------------------------


def report_compare(run_dirs: list[Path], labels: list[str]) -> str:
    eas = [_load(d) for d in run_dirs]
    lines: list[str] = []
    push = lines.append

    push("# Step-timing comparison")
    push("")
    push("| run | tfevents |")
    push("|---|---|")
    for label, d in zip(labels, run_dirs):
        push(f"| {label} | `{d}` |")
    push("")

    # Step counts
    push("## Steps observed")
    push("")
    push("| run | n_steps | first | last |")
    push("|---|---|---|---|")
    for label, ea in zip(labels, eas):
        s = _series(ea, "timing/train/total_step_time")
        if not s:
            push(f"| {label} | 0 | - | - |")
            continue
        push(f"| {label} | {len(s)} | {s[0][0]} | {s[-1][0]} |")
    push("")

    # Per-phase mean times
    def _per_run_means(tag: str) -> list[Optional[float]]:
        out: list[Optional[float]] = []
        for ea in eas:
            vs = [v for _, v in _series(ea, tag)]
            out.append(statistics.mean(vs) if vs else None)
        return out

    def _per_run_sums(tag: str) -> list[Optional[float]]:
        out: list[Optional[float]] = []
        for ea in eas:
            vs = [v for _, v in _series(ea, tag)]
            out.append(sum(vs) if vs else None)
        return out

    def _row(label: str, vals: list[Optional[float]], formatter) -> str:
        cells = " | ".join(formatter(v) if v is not None else "n/a" for v in vals)
        return f"| {label} | {cells} |"

    push("## Mean per-step timing (s)")
    push("")
    push("| phase | " + " | ".join(labels) + " |")
    push("|---|" + "|".join("---" for _ in labels) + "|")
    for tag in TIMING_TRAIN_TAGS:
        vs = _per_run_means(tag)
        if all(v is None for v in vs):
            continue
        short = tag.replace("timing/train/", "")
        push(_row(f"`{short}`", vs, _fmt_s))
    push("")

    push("## Aggregate sum per phase (s)")
    push("")
    push("| phase | " + " | ".join(labels) + " |")
    push("|---|" + "|".join("---" for _ in labels) + "|")
    for tag in TIMING_TRAIN_TAGS:
        vs = _per_run_sums(tag)
        if all(v is None for v in vs):
            continue
        short = tag.replace("timing/train/", "")
        push(_row(f"`{short}`", vs, _fmt_s))
    push("")

    push("## Phase as % of total_step_time (sum / sum)")
    push("")
    push("| phase | " + " | ".join(labels) + " |")
    push("|---|" + "|".join("---" for _ in labels) + "|")
    totals = _per_run_sums("timing/train/total_step_time")
    for tag in TIMING_TRAIN_TAGS:
        if tag == "timing/train/total_step_time":
            continue
        sums = _per_run_sums(tag)
        cells: list[str] = []
        for s_phase, s_total in zip(sums, totals):
            if s_phase is None or s_total is None or not s_total:
                cells.append("n/a")
            else:
                cells.append(f"{100*s_phase/s_total:.1f}%")
        short = tag.replace("timing/train/", "")
        push(f"| `{short}` | " + " | ".join(cells) + " |")
    push("")

    # Pipeline signals
    push("## Pipeline-state signals (mean across all steps)")
    push("")
    push("| signal | " + " | ".join(labels) + " |")
    push("|---|" + "|".join("---" for _ in labels) + "|")
    for tag in PIPELINE_TAGS:
        vs = _per_run_means(tag)
        if all(v is None for v in vs):
            continue
        short = tag.replace("train/", "")
        push(_row(f"`{short}`", vs, lambda v: _fmt_n(v, 2)))
    push("")

    # Throughput: episodes/h estimate from total wall and global_valid_seqs
    push("## Throughput")
    push("")
    push("| run | wall (sum total_step_time) | global_valid_seqs (sum) | "
         "ep/h estimate |")
    push("|---|---|---|---|")
    for label, ea in zip(labels, eas):
        wall = sum(v for _, v in _series(ea, "timing/train/total_step_time"))
        seqs = sum(v for _, v in _series(ea, "train/global_valid_seqs"))
        ep_per_h = seqs / (wall / 3600) if wall > 0 else 0
        push(f"| {label} | {_fmt_s(wall)} | {seqs:.0f} | {ep_per_h:.0f} |")
    push("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="One or more directories containing tfevents files")
    ap.add_argument("--labels", nargs="+",
                    help="Optional labels (default: dir basenames)")
    ap.add_argument("--compare", action="store_true",
                    help="Emit side-by-side comparison instead of per-run reports")
    args = ap.parse_args()

    labels = args.labels or [d.name for d in args.dirs]
    if len(labels) != len(args.dirs):
        ap.error("--labels count must match number of dirs")

    if args.compare or len(args.dirs) > 1:
        print(report_compare(args.dirs, labels))
    else:
        print(report_single(args.dirs[0], labels[0]))


if __name__ == "__main__":
    main()
