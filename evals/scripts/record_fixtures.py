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

"""Record tool-cache fixtures by running the seed tasks against live tools.

The merge-request eval tier replays cached tool responses so it does not depend
on PubChem, the RCSB, or a Vina run finishing in CI. Those caches have to come
from somewhere: this script brings up the eval stack with
``AEVOM_TOOL_CACHE_MODE=record``, drives each task through the same driver Harbor
uses, and leaves the recordings under ``evals/fixtures/tool_cache/`` to be
committed.

It deliberately does not go through Harbor. Harbor owns a task's lifecycle and
would tear the stack down between tasks, and one cache shared across all tasks is
the point -- a later task that fetches 1CRN should hit the entry an earlier one
recorded.

    evals/scripts/record_fixtures.py                     # all tasks
    evals/scripts/record_fixtures.py structure-query-neighbors
    evals/scripts/record_fixtures.py --fresh             # discard existing cache
    evals/scripts/record_fixtures.py --mode replay       # check they replay

Recording calls the real model and the real tools, so it needs API_KEY_NV and
takes a while -- the docking task alone is minutes. Run it locally, then re-run
with ``--mode replay`` before committing: a recording nobody replayed is a
recording that does not work, and cache keys that never match turn the
merge-request tier into a suite of cache misses.

Scores come from the same ``[metadata.expect]`` block the Harbor verifier reads,
so the output is comparable to a real job's.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evals"))

from aevom_eval import scoring  # noqa: E402  -- needs the path above

COMPOSE_FILE = REPO_ROOT / "docker-compose.eval.yml"
TASKS_DIR = REPO_ROOT / "evals" / "tasks"
DEFAULT_CACHE_DIR = REPO_ROOT / "evals" / "fixtures" / "tool_cache"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evals" / "results" / "record"

# A project name distinct from both the dev stack (`aevom`) and anything Harbor
# generates, so recording never disturbs a stack the developer is using.
PROJECT = "aevom-eval-record"

# Same image the tasks declare in [environment].docker_image, so the driver runs
# on the interpreter it will run on under Harbor.
HARNESS_IMAGE = "python:3.12-slim"


def host_path(path: Path) -> str:
    """Translate a repo path into one the Docker daemon can resolve.

    Inside the devcontainer the repo is at /workspaces/aevom, but the daemon is
    the host's, where that path does not exist. HOST_PROJECT_PATH carries the
    host-side root -- the same indirection docker-compose.yml uses for secrets.
    """
    root = os.environ.get("HOST_PROJECT_PATH")
    if not root:
        return str(path)
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        raise SystemExit(
            f"{path} is outside {REPO_ROOT}, so it cannot be translated to a host "
            "path for the Docker daemon. Keep --cache-dir and --results-dir inside "
            "the repo."
        ) from None
    return str(Path(root) / relative)


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(argv)}", flush=True)
    return subprocess.run(argv, check=True, **kwargs)


def compose(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, *args],
        cwd=REPO_ROOT,
        env=env,
    )


def task_names() -> list[str]:
    return sorted(
        path.name for path in TASKS_DIR.iterdir() if (path / "task.toml").exists()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Task directory names; default is every task under evals/tasks.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Where recordings land. Must be inside the repo.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--mode",
        choices=["record", "replay", "mixed"],
        default="record",
        help="AEVOM_TOOL_CACHE_MODE for the stack. `replay` verifies a recording.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing recordings first, rather than adding to them.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Overrides AEVOM_MODEL for the stack and ChatRequest.model_id.",
    )
    parser.add_argument(
        "--keep-up",
        action="store_true",
        help="Leave the stack running afterwards, to inspect logs or Phoenix.",
    )
    parser.add_argument("--turn-timeout", type=float, default=2400.0)
    args = parser.parse_args(argv)
    # Both become bind-mount sources, which have to be absolute.
    args.cache_dir = args.cache_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    return args


def stack_cache_dir(args: argparse.Namespace) -> Path:
    """Where the stack writes. Only `record` is allowed to touch the fixtures.

    `mixed` records on a miss, so pointing it at the committed directory would
    leave the working tree dirty after every smoke run and quietly grow the
    fixtures with whatever the model happened to try. It gets a scratch copy.
    """
    if args.mode == "record":
        return args.cache_dir

    scratch = args.results_dir / "tool_cache"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(args.cache_dir, scratch)
    return scratch


def stack_env(args: argparse.Namespace, cache_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["AEVOM_TOOL_CACHE_MODE"] = args.mode
    env["AEVOM_TOOL_CACHE_HOST_DIR"] = host_path(cache_dir)
    if args.model:
        env["AEVOM_MODEL"] = args.model
    return env


def expectations(task: str) -> dict:
    config = tomllib.loads((TASKS_DIR / task / "task.toml").read_text())
    return (config.get("metadata") or {}).get("expect") or {}


def record_one(task: str, args: argparse.Namespace) -> dict:
    """Drive one task and return the summary the driver printed."""
    results = args.results_dir / task
    results.mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        f"{PROJECT}_default",
        "-v",
        f"{host_path(REPO_ROOT / 'evals')}:/harness:ro",
        "-v",
        f"{host_path(results)}:/out",
        "-e",
        "PYTHONPATH=/harness",
        HARNESS_IMAGE,
        "python",
        "-m",
        "aevom_eval.driver.run_task",
        "--instruction-file",
        f"/harness/tasks/{task}/instruction.md",
        "--output",
        "/out/trajectory.json",
        "--turn-timeout",
        str(args.turn_timeout),
    ]
    if args.model:
        command += ["--model", args.model]

    started = time.monotonic()
    # Not check=True: a task that fails still leaves cache entries for the tools
    # it did reach, and one bad task should not abandon the whole recording.
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = round(time.monotonic() - started, 1)

    (results / "driver.log").write_text(completed.stderr)
    summary = _last_json_line(completed.stdout)

    trajectory_path = results / "trajectory.json"
    trajectory = (
        json.loads(trajectory_path.read_text()) if trajectory_path.exists() else None
    )
    score = scoring.score_trajectory(trajectory, expectations(task))

    status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
    print(f"  {task}: {status} in {elapsed}s, reward {score.rewards.get('reward')}")
    if summary:
        print(f"    tools: {summary.get('tools')}")
        if summary.get("failed_tools"):
            print(f"    failed: {summary['failed_tools']}")
        if summary.get("errors"):
            print(f"    errors: {summary['errors']}")
    else:
        print(f"    no summary; see {results / 'driver.log'}")
    for check in score.failures:
        print(f"    FAIL {check.dimension}/{check.name}: {check.detail}")

    (results / "score.json").write_text(json.dumps(score.to_dict(), indent=2) + "\n")

    return {
        "task": task,
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "summary": summary,
        "rewards": score.rewards,
    }


def _last_json_line(text: str) -> dict | None:
    """The driver prints its summary as the final JSON line on stdout."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not (REPO_ROOT / "API_KEY_NV").exists():
        print(
            "API_KEY_NV is missing. The model runs live in every mode -- only tool "
            "responses are cached -- so there is nothing to record or replay "
            "without it.",
            file=sys.stderr,
        )
        return 2

    tasks = args.tasks or task_names()
    unknown = [task for task in tasks if not (TASKS_DIR / task / "task.toml").exists()]
    if unknown:
        print(f"unknown task(s): {unknown}; have {task_names()}", file=sys.stderr)
        return 2

    if args.fresh and args.mode != "record":
        print("--fresh only makes sense with --mode record", file=sys.stderr)
        return 2
    if args.fresh and args.cache_dir.exists():
        print(f"discarding existing recordings in {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = stack_cache_dir(args)
    if cache_dir != args.cache_dir:
        print(f"{args.mode} mode: stack writes to a scratch copy at {cache_dir}")
    env = stack_env(args, cache_dir)
    results: list[dict] = []
    try:
        # --wait blocks on the healthchecks, so the driver never races startup.
        compose("up", "-d", "--wait", env=env)
        for task in tasks:
            results.append(record_one(task, args))
    finally:
        if args.keep_up:
            print(f"stack left running as project {PROJECT}")
        else:
            # -v drops the postgres/redis volumes: the recording is the cache
            # directory, and stale session state would only confuse a rerun.
            compose("down", "-v", env=env)

    entries = sorted(cache_dir.rglob("*.json"))
    rewards = [entry["rewards"].get("reward", 0.0) for entry in results]
    mean = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    print(f"\n{len(entries)} cache file(s) under {cache_dir}")
    print(f"mean reward over {len(results)} task(s): {mean}")
    (args.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "tasks": results,
                "cache_files": len(entries),
                "mean_reward": mean,
            },
            indent=2,
        )
        + "\n"
    )

    if args.mode == "record":
        print(
            "Re-run with --mode replay before committing; an unreplayed recording "
            "has not been shown to work."
        )

    failed = [entry["task"] for entry in results if entry["returncode"] != 0]
    if failed:
        print(f"tasks whose driver exited non-zero: {failed}", file=sys.stderr)
        print(
            "Their partial recordings were kept. Inspect the trajectories under "
            f"{args.results_dir} before committing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
