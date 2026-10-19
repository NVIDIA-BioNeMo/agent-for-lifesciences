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

"""Efficiency report: standardized one-block summary for an Aevom GRPO run.

Reads a run directory (locally synced from
`cluster:~/runs/<RUN_NAME>/`) and emits a fixed-format summary block intended to
be appended to `internal_docs/docs/training_scaling_curve.md` after every run, so that
12n / 18n / 24n / future configurations are directly comparable. Pure
read-only, no network access.

Inputs (per run dir):
- `<jobid>-logs/episode_traces/episodes.jsonl`  -> inference rollout count,
  reward, finish-reason proxy via num_turns histogram, lifetime wall window
- `<jobid>-logs/ray-driver.log`                 -> training step count,
  async-GRPO buffer starvation (if applicable)
- `<jobid>-logs/exp_001/tensorboard/`            -> tfevents with per-step
  trainer timing (`timing/train/total_step_time`) and consumed-batch counts
  (`train/global_valid_seqs`) -- used to compute trainer_eph (the primary
  throughput metric; see "Throughput metrics" below)
- `*<jobid>.out`                                 -> elapsed wall, topology
  banner ("Nodes: N (P policy + S service); Policy: T train + I inference")
- `grpo_30b_nanov3*.sbatch`                      -> nominal topology +
  `AEVOM_EXTRA_ARGS` overrides for train_global_batch_size etc.
- `configs/aevom_grpo_30b_nanov3*.yaml`          -> max_new_tokens, max_steps,
  num_prompts/num_gens, default train_global_batch_size

Throughput metrics (post-2026-05-17 redefinition):
- `trainer_eph_steady` = (n_steps - 1) * train_global_batch_size / wall_steady_h
  where wall_steady_h is the tfevents wall window from first step event to
  last step event (excludes bootstrap). Primary metric -- what the trainer
  achieves once warm.
- `trainer_eph_valid_steady` = sum(train/global_valid_seqs across the same
  window of steps) / wall_steady_h. Same denominator, accounts for the
  ~80% valid-sequence filter rate.
- `trainer_eph_lifetime` = n_steps * train_global_batch_size / wall_lifetime_h
  where wall_lifetime_h spans the full inference window (episodes.jsonl
  first->last ts). Includes bootstrap; the "honest" number for short runs.
- `inference_eph` = legacy episodes.jsonl rate, renamed. Measures rollout
  *production*, not trainer consumption.
- `consumption_ratio` = trainer_eph_lifetime / inference_eph. Below 1.0
  means inference is over-producing; the deficit is aged-out or in-flight
  at run end.

Usage:
    python scripts/efficiency_report.py /path/to/v0508c-explicit-pdb-ref
    python scripts/efficiency_report.py cluster:~/runs/v0508c-explicit-pdb-ref \
        --no-fetch                # if already synced to /tmp/<run-name>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ssh_fetch(remote: str, dest: Path) -> Path:
    """If `remote` is host:path, scp it to dest/. Returns the local dir."""
    if ":" not in remote:
        return Path(remote)
    host, _, rpath = remote.partition(":")
    name = Path(rpath).name
    local = dest / name
    local.mkdir(parents=True, exist_ok=True)
    print(f"[info] fetching {remote} -> {local}", file=sys.stderr)
    subprocess.check_call(
        ["rsync", "-a", "--info=progress2", f"{remote}/", f"{local}/"]
    )
    return local


def _fmt_h(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


@dataclass
class Topology:
    total: Optional[int]
    train: Optional[int]
    inference: Optional[int]
    service: Optional[int]

    def __str__(self) -> str:
        parts = []
        if self.total is not None:
            parts.append(f"{self.total}n")
        if self.train is not None and self.inference is not None and self.service is not None:
            parts.append(f"({self.train}t + {self.inference}i + {self.service}s)")
        return " ".join(parts) if parts else "?"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def find_log_dir(run_dir: Path) -> Optional[Path]:
    candidates = [p for p in run_dir.iterdir() if p.is_dir() and p.name.endswith("-logs")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_topology_from_out(out_file: Path) -> Topology:
    """Banner format from the modified sbatch:

        Nodes:     12 (8 policy + 4 service)
          Policy:  4 train + 8 inference

    Falls back gracefully on the older banner format (12n runs before the
    Phase 1 sbatch changes) which only printed `Nodes: N (P policy + S service)`.
    """
    total = train = inference = service = None
    if not out_file.exists():
        return Topology(total, train, inference, service)
    txt = out_file.read_text(errors="ignore")
    # Optionally tolerate `+ N judge` suffix introduced by the v0514a-era banner.
    m = re.search(
        r"Nodes:\s+(\d+)\s+\((\d+)\s+policy\s+\+\s+(\d+)\s+service(?:\s+\+\s+\d+\s+judge)?\)",
        txt,
    )
    if m:
        total = int(m.group(1))
        policy = int(m.group(2))
        service = int(m.group(3))
        # Older runs had policy = train + inference colocated, no split.
        # New banner adds the "Policy: X train + Y inference" line.
        m2 = re.search(r"Policy:\s+(\d+)\s+train\s+\+\s+(\d+)\s+inference", txt)
        if m2:
            train = int(m2.group(1))
            inference = int(m2.group(2))
        else:
            # Pre-async runs were colocated; split = (policy, 0).
            train = policy
            inference = 0
    return Topology(total, train, inference, service)


def parse_topology_from_sbatch(sbatch_file: Path) -> Topology:
    """Pre-out-file fallback: parse POLICY_NODES / INFERENCE_NODES /
    NUM_SERVICE_NODES + #SBATCH --nodes from the run's sbatch."""
    if not sbatch_file.exists():
        return Topology(None, None, None, None)
    txt = sbatch_file.read_text(errors="ignore")
    total = None
    m = re.search(r"^#SBATCH\s+--nodes=(\d+)", txt, re.MULTILINE)
    if m:
        total = int(m.group(1))
    pn = inf = svc = None
    m = re.search(r'^POLICY_NODES="?\$?\{?POLICY_NODES:?-?(\d+)\}?"?', txt, re.MULTILINE)
    if not m:
        m = re.search(r"^POLICY_NODES=(\d+)", txt, re.MULTILINE)
    if m:
        pn = int(m.group(1))
    m = re.search(r'^INFERENCE_NODES="?\$?\{?INFERENCE_NODES:?-?(\d+)\}?"?', txt, re.MULTILINE)
    if not m:
        m = re.search(r"^INFERENCE_NODES=(\d+)", txt, re.MULTILINE)
    if m:
        inf = int(m.group(1))
    m = re.search(r'^NUM_SERVICE_NODES="?\$?\{?NUM_SERVICE_NODES:?-?(\d+)\}?"?', txt, re.MULTILINE)
    if not m:
        m = re.search(r"^NUM_SERVICE_NODES=(\d+)", txt, re.MULTILINE)
    if m:
        svc = int(m.group(1))
    train = (pn - inf) if (pn is not None and inf is not None) else None
    return Topology(total, train, inf, svc)


def parse_config(yaml_file: Path) -> dict:
    """Cheap key:value scrape; we don't need full YAML parsing for these few
    fields, and avoiding pyyaml keeps the script dependency-free."""
    out: dict[str, str] = {}
    if not yaml_file.exists():
        return out
    keys = {
        "num_prompts_per_step": r"num_prompts_per_step:\s*(\S+)",
        "num_generations_per_prompt": r"num_generations_per_prompt:\s*(\S+)",
        "max_total_sequence_length": r"max_total_sequence_length:\s*(\S+)",
        "max_new_tokens": r"max_new_tokens:\s*(\S+)",
        "max_rollout_turns": r"max_rollout_turns:\s*(\S+)",
        "agent_max_steps": r"^\s*max_steps:\s*(\S+)",
        "cluster_num_nodes": r"^cluster:\s*\n(?:[^\n]*\n)*?\s*num_nodes:\s*(\d+)",
        "train_global_batch_size": r"train_global_batch_size:\s*(\S+)",
    }
    raw = yaml_file.read_text(errors="ignore")
    txt = "\n".join(
        re.sub(r"(?<!\\)#.*$", "", line) for line in raw.splitlines()
    )
    for k, pat in keys.items():
        m = re.search(pat, txt, re.MULTILINE)
        if m:
            out[k] = m.group(1)
    # colocated.resources.num_nodes lives inside the policy.generation block.
    # Use a 2-pass scan: find `colocated:` then the subsequent `num_nodes:`.
    m = re.search(r"colocated:\s*\n((?:.{0,120}\n){1,10})", txt)
    if m:
        m2 = re.search(r"num_nodes:\s*(\d+)", m.group(1))
        if m2:
            out["colocated_num_nodes"] = m2.group(1)
    return out


_TS_RE = re.compile(r"^\s*\"timestamp\":\s*\"([^\"]+)\"")


def episode_stats(eps_file: Path) -> dict:
    """Stream episodes.jsonl once and compute count, time window, num_turns
    histogram, reward stats. Avoids loading all 138MB into memory."""
    if not eps_file.exists():
        return {}
    n = 0
    rewards: list[float] = []
    turns_hist: dict[int, int] = {}
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    tool_errors_total = 0
    with open(eps_file, "r", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            ts = d.get("timestamp")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            r = d.get("reward")
            if r is not None:
                try:
                    rewards.append(float(r))
                except (TypeError, ValueError):
                    pass
            t = d.get("num_turns")
            if t is not None:
                try:
                    turns_hist[int(t)] = turns_hist.get(int(t), 0) + 1
                except (TypeError, ValueError):
                    pass
            te = d.get("num_tool_errors")
            if te:
                try:
                    tool_errors_total += int(te)
                except (TypeError, ValueError):
                    pass
    return {
        "count": n,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "rewards": rewards,
        "turns_hist": turns_hist,
        "tool_errors_total": tool_errors_total,
    }


def _iso_to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")


_STEP_RE = re.compile(r"training_step=(\d+),\s*max_age=\d+,\s*buffer_size=(\d+)")
_STEP_LINE_RE = re.compile(r"\bstep[\s_=:]?(\d+)\b", re.IGNORECASE)


def find_phoenix_db(log_dir: Path) -> Optional[Path]:
    """Phoenix DB lives at one of two paths depending on whether the run
    artifacts were synced flat (`<log_dir>/phoenix.db`) or with the remote
    layout preserved (`<log_dir>/phoenix_db/phoenix.db`)."""
    for p in (log_dir / "phoenix.db", log_dir / "phoenix_db" / "phoenix.db"):
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _phoenix_ts_to_dt(s: str) -> datetime:
    """Phoenix stores timestamps as 'YYYY-MM-DD HH:MM:SS.ffffff' (no tz)."""
    return datetime.fromisoformat(s)


def phoenix_stats(phoenix_db: Path) -> dict:
    """Read per-call walls from the Phoenix span DB.

    Policy-LLM walls come from the live ``policy_llm.responses`` spans
    introduced 2026-05-27 (real vLLM HTTP wall recorded by the
    aevom_simple_agent overlay). For pre-refactor runs we fall back to
    the synthetic ``llm.complete`` child of ``reasoning_iteration.*``
    (those record 0ms but the older estimator was still useful for
    judge-vs-policy attribution).

    Judge walls continue to come from ``llm.complete`` under
    ``judge_evaluation`` -- the judge path is unchanged.

    Also computes peak concurrent in-flight judge calls per 5-minute
    bucket -- the key signal for endpoint contention.

    Returns ``{}`` on a missing/empty DB so the caller can degrade
    gracefully on older runs without traces."""
    out: dict = {}
    if not phoenix_db.exists() or phoenix_db.stat().st_size == 0:
        return out
    try:
        conn = sqlite3.connect(f"file:{phoenix_db}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        conn.execute("SELECT 1 FROM spans LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        conn.close()
        return out

    def _walls(rows: list[tuple[str, str]]) -> list[float]:
        return [
            (_phoenix_ts_to_dt(e) - _phoenix_ts_to_dt(s)).total_seconds()
            for s, e in rows
        ]

    judge_rows = conn.execute(
        """
        SELECT s.start_time, s.end_time
        FROM spans s JOIN spans p ON s.parent_id = p.span_id
        WHERE s.name='llm.complete'
          AND p.name='judge_evaluation'
          AND s.end_time IS NOT NULL
        """
    ).fetchall()

    # Prefer live policy_llm.responses spans; fall back to the
    # pre-2026-05-27 synthetic reasoning_iteration child if the live
    # spans are absent (so we can still re-run this script against an
    # older run's phoenix.db for comparison).
    policy_rows = conn.execute(
        """
        SELECT start_time, end_time
        FROM spans
        WHERE name='policy_llm.responses'
          AND end_time IS NOT NULL
        """
    ).fetchall()
    policy_source = "live"
    if not policy_rows:
        policy_rows = conn.execute(
            """
            SELECT s.start_time, s.end_time
            FROM spans s JOIN spans p ON s.parent_id = p.span_id
            WHERE s.name='llm.complete'
              AND p.name LIKE 'reasoning_iteration.%'
              AND s.end_time IS NOT NULL
            """
        ).fetchall()
        policy_source = "synthetic"

    rollout_rows = conn.execute(
        "SELECT start_time, end_time FROM spans "
        "WHERE name='chat_request' AND end_time IS NOT NULL"
    ).fetchall()

    out["judge_walls"] = _walls(judge_rows)
    out["policy_walls"] = _walls(policy_rows)
    out["policy_walls_source"] = policy_source
    out["rollout_walls"] = _walls(rollout_rows)

    # Per-5-min-bucket peak concurrent in-flight judge requests.
    # The judge runs against an external (NVIDIA-hosted) endpoint, so this
    # is the main observable for endpoint queueing. We use a discrete event
    # sweep over (start, +1), (end, -1) ticks.
    judge_events_rows = conn.execute(
        "SELECT start_time, end_time FROM spans "
        "WHERE name='judge_evaluation' AND end_time IS NOT NULL "
        "ORDER BY start_time"
    ).fetchall()
    peaks: list[int] = []
    if judge_events_rows:
        ticks: list[tuple[datetime, int]] = []
        for s, e in judge_events_rows:
            ticks.append((_phoenix_ts_to_dt(s), +1))
            ticks.append((_phoenix_ts_to_dt(e), -1))
        ticks.sort()
        t0 = ticks[0][0]
        bucket_peaks: dict[int, int] = {}
        in_flight = 0
        for ts, delta in ticks:
            in_flight += delta
            bucket = int((ts - t0).total_seconds() // 300)
            bucket_peaks[bucket] = max(bucket_peaks.get(bucket, 0), in_flight)
        peaks = [bucket_peaks[b] for b in sorted(bucket_peaks)]
    out["judge_concurrency_peaks_5min"] = peaks

    conn.close()
    return out


def driver_stats(driver_log: Path) -> dict:
    if not driver_log.exists():
        return {}
    max_train_step = -1
    starvation_total = 0
    starvation_zero = 0
    with open(driver_log, "r", errors="ignore") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                buf = int(m.group(2))
                if step > max_train_step:
                    max_train_step = step
                starvation_total += 1
                if buf == 0:
                    starvation_zero += 1
    return {
        "max_train_step": max_train_step if max_train_step >= 0 else None,
        "starvation_polls_total": starvation_total,
        "starvation_polls_zero": starvation_zero,
        "starvation_pct": (
            100 * starvation_zero / starvation_total if starvation_total else None
        ),
    }


def parse_sbatch_overrides(sbatch_files: list[Path]) -> dict:
    """Extract hydra overrides from sbatch ``AEVOM_EXTRA_ARGS`` exports.

    A run dir typically contains the base sbatch *and* a thin wrapper that
    sets ``AEVOM_EXTRA_ARGS=...`` (e.g.
    ``grpo_30b_nanov3_10n_pps32_localjudge.sbatch``). We merge overrides from
    all files, so the wrapper's values win where the base doesn't override.

    Returns ``{policy.train_global_batch_size: '192', grpo.num_prompts_per_step:
    '12', ...}`` so the trainer-eph math uses the values actually passed to
    the run, not just the YAML defaults. Quietly returns ``{}`` for older
    sbatches that didn't use AEVOM_EXTRA_ARGS.
    """
    out: dict[str, str] = {}
    for sbatch_file in sbatch_files:
        if not sbatch_file or not sbatch_file.exists():
            continue
        txt = sbatch_file.read_text(errors="ignore")
        for m_block in re.finditer(
            r'AEVOM_EXTRA_ARGS=(?:"\$\{AEVOM_EXTRA_ARGS:-\}\s*)?"([^"]*)"',
            txt,
            re.DOTALL,
        ):
            for m in re.finditer(r"([\w.]+)=([^\s\\]+)", m_block.group(1)):
                out[m.group(1)] = m.group(2)
    return out


def find_tfevents_dir(log_dir: Path) -> Optional[Path]:
    """Look for the tensorboard event files under <log_dir>/exp_*/tensorboard."""
    for cand in sorted(log_dir.glob("exp_*/tensorboard")):
        if cand.is_dir() and any(cand.glob("events.out.tfevents.*")):
            return cand
    # Older runs may also drop tfevents directly into a flat dir.
    flat = log_dir / "tensorboard"
    if flat.is_dir() and any(flat.glob("events.out.tfevents.*")):
        return flat
    return None


def tfevents_stats(tb_dir: Optional[Path]) -> dict:
    """Read minimal trainer-side step stats from tfevents.

    Returns:
        n_steps           -- number of `timing/train/total_step_time` events
        step_first/last   -- first and last training_step indices observed
        wall_first/last   -- Unix wall_time at the first and last step events
        wall_window_s     -- wall_last - wall_first (steady-state window;
                             excludes bootstrap before the first step and
                             trailing time after the last step)
        sum_step_time_s   -- sum of `timing/train/total_step_time`
        sum_valid_seqs    -- sum of `train/global_valid_seqs`
        valid_seqs_after_first -- sum_valid_seqs minus the first step's value;
                                  divided by wall_window_s gives the
                                  steady-state valid-seq ep/h estimate

    Returns ``{}`` when tfevents is missing or `total_step_time` is absent
    (legacy sync-GRPO runs). Tensorboard imports are lazy so the script
    doesn't pay the import cost for runs without tfevents.
    """
    if tb_dir is None:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return {}
    tags = ea.Tags().get("scalars", [])
    if "timing/train/total_step_time" not in tags:
        return {}
    events = ea.Scalars("timing/train/total_step_time")
    if not events:
        return {}
    valid = (
        ea.Scalars("train/global_valid_seqs")
        if "train/global_valid_seqs" in tags
        else []
    )
    sum_valid = sum(e.value for e in valid)
    valid_after_first = sum_valid - (valid[0].value if valid else 0.0)
    return {
        "n_steps": len(events),
        "step_first": events[0].step,
        "step_last": events[-1].step,
        "wall_first": events[0].wall_time,
        "wall_last": events[-1].wall_time,
        "wall_window_s": events[-1].wall_time - events[0].wall_time,
        "sum_step_time_s": sum(e.value for e in events),
        "sum_valid_seqs": sum_valid,
        "valid_seqs_after_first": valid_after_first,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def render_summary(
    run_name: str,
    jobid: Optional[str],
    topo: Topology,
    config: dict,
    eps: dict,
    drv: dict,
    phoenix: Optional[dict] = None,
    tfe: Optional[dict] = None,
    sbatch_overrides: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    push = lines.append
    push(f"=== {run_name} ===")
    push(f"  jobid: {jobid or '?'}")
    push(f"  topology: {topo}")
    if topo.train is not None and topo.inference is not None and topo.service is not None:
        gpus_train = topo.train * 8
        gpus_inf = topo.inference * 8
        gpus_svc = topo.service * 8
        push(
            f"            GPUs: {gpus_train} train + {gpus_inf} inference + "
            f"{gpus_svc} service"
        )
    push(
        "  config: "
        f"max_new_tokens={config.get('max_new_tokens','?')}, "
        f"max_steps={config.get('agent_max_steps','?')}, "
        f"prompts/step={config.get('num_prompts_per_step','?')}, "
        f"gens/prompt={config.get('num_generations_per_prompt','?')}, "
        f"max_seq={config.get('max_total_sequence_length','?')}"
    )

    # Wall (lifetime) -- inference window from first to last rollout completion
    elapsed_s = None
    if eps.get("first_ts") and eps.get("last_ts"):
        elapsed_s = (_iso_to_dt(eps["last_ts"]) - _iso_to_dt(eps["first_ts"])).total_seconds()
    elapsed_str = _fmt_h(elapsed_s) if elapsed_s else "?"
    push(f"  elapsed (lifetime, first->last episode ts): {elapsed_str}")

    n = eps.get("count", 0)
    steps = drv.get("max_train_step")

    # Resolve effective train_global_batch_size: sbatch override (e.g.
    # `policy.train_global_batch_size=192`) wins over the YAML default.
    overrides = sbatch_overrides or {}
    batch_str = (
        overrides.get("policy.train_global_batch_size")
        or config.get("train_global_batch_size")
    )
    try:
        train_global_batch_size = int(batch_str) if batch_str is not None else None
    except (TypeError, ValueError):
        train_global_batch_size = None

    push(
        f"  episodes (inference rollouts): {n}   training steps reached: {steps or '?'}"
        + (
            f"   train_global_batch_size: {train_global_batch_size}"
            if train_global_batch_size
            else ""
        )
    )
    eps_per_step_cfg = None
    if config.get("num_prompts_per_step") and config.get("num_generations_per_prompt"):
        try:
            eps_per_step_cfg = (
                int(config["num_prompts_per_step"])
                * int(config["num_generations_per_prompt"])
            )
        except ValueError:
            pass
    if steps and steps > 0 and eps_per_step_cfg:
        observed_per_step = n / max(1, steps + 1)
        push(
            f"  inference rollouts/step (lifetime avg): observed {observed_per_step:.1f} "
            f"(configured {eps_per_step_cfg})"
        )

    # --- Trainer throughput (primary) and inference throughput (legacy) ---
    # See module docstring "Throughput metrics" for definitions.
    inference_eph: Optional[float] = None
    if elapsed_s and elapsed_s > 0:
        inference_eph = n * 3600.0 / elapsed_s

    trainer_eph_steady: Optional[float] = None
    trainer_eph_valid_steady: Optional[float] = None
    trainer_eph_lifetime: Optional[float] = None
    tfe = tfe or {}
    if tfe and train_global_batch_size:
        n_steps = tfe.get("n_steps", 0)
        window_s = tfe.get("wall_window_s", 0.0) or 0.0
        if n_steps >= 2 and window_s > 0:
            # (n_steps - 1) intervals are spanned between first and last event
            trainer_eph_steady = (
                (n_steps - 1) * train_global_batch_size * 3600.0 / window_s
            )
            valid_after_first = tfe.get("valid_seqs_after_first", 0.0) or 0.0
            if valid_after_first > 0:
                trainer_eph_valid_steady = valid_after_first * 3600.0 / window_s
        if n_steps >= 1 and elapsed_s and elapsed_s > 0:
            trainer_eph_lifetime = n_steps * train_global_batch_size * 3600.0 / elapsed_s

    # Primary line: the metric we want eyes on.
    if trainer_eph_steady is not None:
        window_s = tfe.get("wall_window_s", 0) or 0
        # < 10 minutes of trainer wall is almost certainly a back-to-back
        # burst that hasn't seen a single refill cycle -- mark it so we
        # don't read it as a long-run steady state.
        burst_flag = "  **[BURST -- window too short to be steady-state]**" if window_s < 600 else ""
        push(
            f"  trainer ep/h (steady, post-bootstrap): "
            f"**{trainer_eph_steady:.0f}**  "
            f"[{tfe.get('n_steps','?')} steps over "
            f"{_fmt_h(window_s)} "
            f"of trainer wall window]{burst_flag}"
        )
        if trainer_eph_valid_steady is not None:
            push(
                f"  trainer ep/h (valid-seqs, steady): "
                f"{trainer_eph_valid_steady:.0f}  "
                f"[sum global_valid_seqs across post-first steps / window]"
            )
    else:
        reason = []
        if not tfe:
            reason.append("no tfevents")
        else:
            if tfe.get("n_steps", 0) < 2:
                reason.append(
                    f"only {tfe.get('n_steps', 0)} step event(s) (need >= 2)"
                )
            if (tfe.get("wall_window_s") or 0) <= 0:
                reason.append("zero wall window")
        if train_global_batch_size is None:
            reason.append("train_global_batch_size unknown (config + sbatch not found)")
        push(f"  trainer ep/h: n/a ({'; '.join(reason) or 'unknown reason'})")

    if trainer_eph_lifetime is not None:
        push(
            f"  trainer ep/h (lifetime, includes bootstrap): "
            f"{trainer_eph_lifetime:.0f}  "
            f"[n_steps x batch / lifetime wall]"
        )

    if inference_eph is not None:
        push(
            f"  inference ep/h (rollout production rate, legacy 'ep/h'): "
            f"{inference_eph:.1f}"
        )
        if trainer_eph_lifetime is not None and inference_eph > 0:
            ratio = trainer_eph_lifetime / inference_eph
            push(
                f"  consumption ratio (trainer_lifetime / inference): "
                f"{ratio:.2f}  "
                f"[<1.0 = inference over-producing; deficit = aged-out + in-flight at end]"
            )

    # Rewards
    rewards = eps.get("rewards") or []
    if rewards:
        push(
            f"  reward: mean={statistics.mean(rewards):.3f}  "
            f"non-zero={sum(1 for r in rewards if r > 0)/len(rewards)*100:.1f}%  "
            f"p90={_percentile(rewards, 90):.3f}  "
            f"max={max(rewards):.3f}"
        )

    # Turn histogram + max-step saturation rate
    th = eps.get("turns_hist") or {}
    if th:
        max_steps = None
        try:
            max_steps = int(config.get("agent_max_steps") or 0) or None
        except ValueError:
            pass
        zero_pct = 100 * th.get(0, 0) / max(1, sum(th.values()))
        push(f"  num_turns: 0={zero_pct:.1f}% (rollouts with no tool calls)")
        if max_steps:
            sat = 100 * th.get(max_steps, 0) / max(1, sum(th.values()))
            push(
                f"  max_steps saturation (={max_steps}): {sat:.1f}% "
                f"(rollouts capped by step budget)"
            )

    if eps.get("tool_errors_total") is not None and n:
        push(
            f"  tool_errors: {eps['tool_errors_total']} total = "
            f"{eps['tool_errors_total']/max(1,n):.2f} per episode"
        )

    # Async-GRPO buffer starvation
    if drv.get("starvation_polls_total"):
        push(
            f"  async-buffer starvation (training idle waiting for rollouts): "
            f"{drv['starvation_pct']:.1f}% of polls "
            f"({drv['starvation_polls_zero']}/{drv['starvation_polls_total']})"
        )

    # Per-LLM-call wall split (judge endpoint vs our policy vLLM) and
    # per-rollout wall budget breakdown. Sourced from the Phoenix span DB
    # so older runs without traces degrade silently.
    if phoenix:
        judge = phoenix.get("judge_walls") or []
        policy = phoenix.get("policy_walls") or []
        rollout = phoenix.get("rollout_walls") or []
        peaks = phoenix.get("judge_concurrency_peaks_5min") or []
        if judge:
            push(
                f"  judge wall (NVIDIA endpoint):  "
                f"mean={statistics.mean(judge):6.1f}s  "
                f"p50={_percentile(judge,50):6.1f}s  "
                f"p90={_percentile(judge,90):6.1f}s  "
                f"p99={_percentile(judge,99):6.1f}s  "
                f"n={len(judge)}"
            )
        if policy:
            policy_src = phoenix.get("policy_walls_source", "?")
            push(
                f"  policy wall (our vLLM, {policy_src}): "
                f"mean={statistics.mean(policy):6.1f}s  "
                f"p50={_percentile(policy,50):6.1f}s  "
                f"p90={_percentile(policy,90):6.1f}s  "
                f"p99={_percentile(policy,99):6.1f}s  "
                f"n={len(policy)}"
            )
        if peaks:
            push(
                f"  judge concurrency (per 5-min bucket): "
                f"median peak={statistics.median(peaks):.0f}, "
                f"max peak={max(peaks)}, n_buckets={len(peaks)}"
            )
        if rollout:
            n_roll = len(rollout)
            roll_mean = statistics.mean(rollout)
            per_judge = (sum(judge) / n_roll) if judge else 0.0
            per_policy = (sum(policy) / n_roll) if policy else 0.0
            per_other = max(0.0, roll_mean - per_judge - per_policy)
            push(
                f"  per-rollout wall budget (mean {roll_mean:.0f}s = "
                f"judge {per_judge:.0f}s [{100*per_judge/roll_mean:.0f}%] + "
                f"policy {per_policy:.0f}s [{100*per_policy/roll_mean:.0f}%] + "
                f"other {per_other:.0f}s [{100*per_other/roll_mean:.0f}%])"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def report_for_run(run_path: Path) -> str:
    log_dir = find_log_dir(run_path)
    if log_dir is None:
        return f"=== {run_path.name} ===\n  (no <jobid>-logs directory found)\n"
    jobid = log_dir.name.split("-")[0]
    out_files = sorted(run_path.glob(f"*{jobid}*.out"))
    out_file = out_files[0] if out_files else None
    sbatch_files = sorted(run_path.glob("grpo*nanov3*.sbatch"))
    # Prefer the wrapper sbatch if it's a thin wrapper (smaller); banner has the
    # truth at runtime so we only need topology fallback.
    sbatch_file = sbatch_files[0] if sbatch_files else None
    # Glob the GRPO yaml rather than hardcoding a name: the canonical config is
    # aevom_grpo_30b_nanov3_pps32_2train.yaml, but older run dirs used other
    # nano-v3 variants. Pick the first match.
    config_globs = sorted((run_path / "configs").glob("aevom_grpo_30b_nanov3*.yaml"))
    config_file = config_globs[0] if config_globs else (
        run_path / "configs" / "aevom_grpo_30b_nanov3_pps32_2train.yaml"
    )

    topo = (
        parse_topology_from_out(out_file)
        if out_file
        else Topology(None, None, None, None)
    )
    if topo.total is None and sbatch_file:
        # Fallback to sbatch when the .out file doesn't have the new banner.
        topo_fallback = parse_topology_from_sbatch(sbatch_file)
        if topo_fallback.total:
            topo = topo_fallback

    config = parse_config(config_file)
    # Backward-compat: pre-Phase-1 .out files don't include the
    # "Policy: X train + Y inference" line. For runs whose YAML uses
    # non-colocated inference (the v0508c+ topology), reconstruct the split
    # using cluster.num_nodes (= train + inference total) and
    # colocated.resources.num_nodes (= inference reservation).
    if (
        topo.train is not None
        and topo.inference == 0
        and "colocated_num_nodes" in config
    ):
        cluster_n = int(config.get("cluster_num_nodes", topo.train))
        inf_n = int(config["colocated_num_nodes"])
        if 0 < inf_n < cluster_n:
            topo.train = cluster_n - inf_n
            topo.inference = inf_n
    eps = episode_stats(log_dir / "episode_traces" / "episodes.jsonl")
    drv = driver_stats(log_dir / "ray-driver.log")
    phoenix_db = find_phoenix_db(log_dir)
    phoenix = phoenix_stats(phoenix_db) if phoenix_db else {}
    tfe = tfevents_stats(find_tfevents_dir(log_dir))
    sb_over = parse_sbatch_overrides(sbatch_files)
    return render_summary(
        run_path.name, jobid, topo, config, eps, drv, phoenix,
        tfe=tfe, sbatch_overrides=sb_over,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories (local paths or host:/path). "
        "Multiple dirs render multiple comparable summary blocks.",
    )
    ap.add_argument(
        "--fetch-into",
        default="/tmp/aevom_runs",
        help="Where to rsync remote run dirs (default /tmp/aevom_runs).",
    )
    args = ap.parse_args()

    fetch_root = Path(args.fetch_into)
    fetch_root.mkdir(parents=True, exist_ok=True)
    for spec in args.run_dirs:
        local = _ssh_fetch(spec, fetch_root) if ":" in spec else Path(spec)
        print(report_for_run(local))


if __name__ == "__main__":
    main()
