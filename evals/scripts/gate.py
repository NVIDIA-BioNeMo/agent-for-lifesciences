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

"""Turn a Harbor job result into a CI pass or fail.

Harbor has no ``--min-pass-rate``: ``harbor run`` exits 0 as long as the
machinery worked, whatever the rewards were. So the gate lives here, reading a
job directory and exiting non-zero when the suite regressed.

    evals/scripts/gate.py                                # newest job under jobs/
    evals/scripts/gate.py --job-dir jobs/2026-08-06__18-28-37
    evals/scripts/gate.py --min-mean-reward 0.9 --min-task-reward 0.5

Rewards come from the per-trial ``<job>/<trial>/result.json`` files, not from the
job's own ``result.json``. The job-level file is an aggregate: it carries
``stats.evals[...].metrics`` with no task attribution beyond the trial names
embedded in ``reward_stats``, so per-task thresholds cannot be computed from it.
It is still read, for ``n_total_trials`` -- that is how a trial which produced no
result file at all becomes visible.

Four things fail the gate besides a low score, each of which has a way of looking
like success if you only read the exit code:

* A trial that raised. Harbor records the exception and moves on, leaving no
  verifier_result; counting it as absent rather than as zero would quietly raise
  the mean of whatever did run.
* A trial with no result.json, which is what a cancelled or never-started trial
  leaves behind. Counted against the job rather than ignored.
* A job that ran fewer trials than asked for. An empty suite has a mean reward of
  nothing, and ``--min-trials`` is what makes that a failure instead of a pass.
* Rewards that contain ``harness_ok`` but no ``reward``. That means the task's
  stub ``tests/test.sh`` graded the run instead of AevomVerifier -- the job forgot
  ``--verifier``, and it measured almost nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_DIR = REPO_ROOT / "jobs"

# Written by evals/tasks/*/tests/test.sh, which cannot grade correctness.
HARNESS_ONLY_KEY = "harness_ok"

# Stands in for trials Harbor counted but left no result.json for. A name rather
# than a separate tally so it shows up in the per-task table, which is where
# someone reading a failure will look.
MISSING_TRIAL_TASK = "<trial with no result.json>"


class GateError(Exception):
    """Something about the job makes it ungradeable."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="Defaults to the most recently modified job under --jobs-dir.",
    )
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument(
        "--reward-key",
        default="reward",
        help="The overall dimension AevomVerifier reports; others are informational.",
    )
    parser.add_argument(
        "--min-mean-reward",
        type=float,
        default=0.8,
        help="Threshold on the mean over all trials.",
    )
    parser.add_argument(
        "--min-task-reward",
        type=float,
        default=None,
        help=(
            "Optional floor on each task's own mean, so one task collapsing "
            "cannot hide behind the others."
        ),
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=1,
        help="Fail if the job graded fewer trials than this.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the computed summary here, for a CI artifact.",
    )
    return parser.parse_args(argv)


def newest_job(jobs_dir: Path) -> Path:
    if not jobs_dir.is_dir():
        raise GateError(f"no jobs directory at {jobs_dir}; did harbor run at all?")
    candidates = [path for path in jobs_dir.iterdir() if (path / "result.json").is_file()]
    if not candidates:
        raise GateError(
            f"no job under {jobs_dir} has a result.json; the run did not finish"
        )
    return max(candidates, key=lambda path: (path / "result.json").stat().st_mtime)


def load_result(job_dir: Path) -> dict:
    path = job_dir / "result.json"
    if not path.is_file():
        raise GateError(f"{path} does not exist")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise GateError(f"{path} is not valid JSON: {e}") from None


def condense(message: object, limit: int = 600) -> str:
    """One line, keeping both ends of a long message.

    Truncating the head alone loses the reason: a failed `docker compose` puts
    the whole command first and the actual error last, so a 200-character head
    read "Docker compose command failed ... Command: docker compose --project-name"
    and stopped, which is the part nobody needed.
    """
    text = " ".join(str(message).split())
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]} [...] {text[-(limit - head):]}"


def load_trials(job_dir: Path) -> tuple[list[dict], list[str]]:
    """Every trial's own result.json, one directory below the job's."""
    problems: list[str] = []
    trials: list[dict] = []
    for path in sorted(job_dir.glob("*/result.json")):
        try:
            trials.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"{path.parent.name}: result.json is unreadable: {e}")
            trials.append({"trial_name": path.parent.name})
    return trials, problems


def collect(
    result: dict, trials: list[dict], reward_key: str
) -> tuple[dict[str, list[float]], list[str]]:
    """Per-task reward lists, plus a problem note for every trial that has one.

    Errored and ungraded trials contribute 0.0 rather than being skipped: a
    crashed trial is a failed trial, and dropping it would raise the mean.
    """
    per_task: dict[str, list[float]] = {}
    problems: list[str] = []

    if not trials:
        raise GateError(
            f"no <trial>/result.json under the job directory; harbor started "
            f"{result.get('n_total_trials', 0)} trial(s) and none finished"
        )

    # Harbor's own count of what it set out to run. A trial that was cancelled, or
    # died before writing anything, exists only here.
    expected = result.get("n_total_trials") or 0
    if expected > len(trials):
        shortfall = expected - len(trials)
        problems.append(
            f"{shortfall} of {expected} trial(s) left no result.json; counted as 0.0"
        )
        per_task[MISSING_TRIAL_TASK] = [0.0] * shortfall

    for trial in trials:
        task = trial.get("task_name") or trial.get("trial_name") or "<unnamed>"
        rewards = (trial.get("verifier_result") or {}).get("rewards")

        exception = trial.get("exception_info")
        if exception:
            problems.append(
                f"{task}: {exception.get('exception_type')}: "
                f"{condense(exception.get('exception_message'))}"
            )
            per_task.setdefault(task, []).append(0.0)
            continue

        if not rewards:
            problems.append(f"{task}: no rewards recorded; the verifier did not report")
            per_task.setdefault(task, []).append(0.0)
            continue

        if reward_key not in rewards:
            if HARNESS_ONLY_KEY in rewards:
                problems.append(
                    f"{task}: graded by tests/test.sh, not AevomVerifier -- rewards "
                    f"are {sorted(rewards)}. Pass "
                    "--verifier aevom_eval.verifier:AevomVerifier."
                )
            else:
                problems.append(
                    f"{task}: no {reward_key!r} in rewards {sorted(rewards)}"
                )
            per_task.setdefault(task, []).append(0.0)
            continue

        per_task.setdefault(task, []).append(float(rewards[reward_key]))

    return per_task, problems


def report(per_task: dict[str, list[float]], reward_key: str) -> dict:
    task_means = {
        task: round(statistics.fmean(values), 4) for task, values in per_task.items()
    }
    all_values = [value for values in per_task.values() for value in values]
    return {
        "reward_key": reward_key,
        "n_trials": len(all_values),
        "mean_reward": round(statistics.fmean(all_values), 4) if all_values else 0.0,
        "task_means": task_means,
        "task_trials": {task: len(values) for task, values in per_task.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        job_dir = args.job_dir or newest_job(args.jobs_dir)
        result = load_result(job_dir)
        trials, unreadable = load_trials(job_dir)
        per_task, problems = collect(result, trials, args.reward_key)
    except GateError as e:
        print(f"eval gate: {e}", file=sys.stderr)
        return 2

    summary = report(per_task, args.reward_key)
    summary["job_dir"] = str(job_dir)
    summary["problems"] = unreadable + problems
    # Untruncated, because the console line is condensed and a compose failure's
    # detail is the one thing worth having in the CI artifact.
    summary["exceptions"] = [
        {
            "task": trial.get("task_name") or trial.get("trial_name"),
            "type": (trial.get("exception_info") or {}).get("exception_type"),
            "message": (trial.get("exception_info") or {}).get("exception_message"),
        }
        for trial in trials
        if trial.get("exception_info")
    ]

    print(f"job: {job_dir}")
    width = max(len(task) for task in summary["task_means"])
    for task, mean in sorted(summary["task_means"].items()):
        attempts = summary["task_trials"][task]
        print(f"  {task:<{width}}  {mean:.3f}  over {attempts} trial(s)")
    print(
        f"mean {args.reward_key}: {summary['mean_reward']:.3f} "
        f"over {summary['n_trials']} trial(s)"
    )

    for problem in summary["problems"]:
        print(f"  problem: {problem}", file=sys.stderr)

    failures = []
    if summary["n_trials"] < args.min_trials:
        failures.append(
            f"only {summary['n_trials']} trial(s) graded, expected at least "
            f"{args.min_trials}"
        )
    if summary["mean_reward"] < args.min_mean_reward:
        failures.append(
            f"mean {args.reward_key} {summary['mean_reward']:.3f} is below "
            f"{args.min_mean_reward}"
        )
    if args.min_task_reward is not None:
        below = {
            task: mean
            for task, mean in summary["task_means"].items()
            if mean < args.min_task_reward
        }
        if below:
            failures.append(
                f"task(s) below {args.min_task_reward}: "
                + ", ".join(f"{task} {mean:.3f}" for task, mean in sorted(below.items()))
            )

    summary["passed"] = not failures
    summary["failures"] = failures
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")

    if failures:
        print("\neval gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\neval gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
