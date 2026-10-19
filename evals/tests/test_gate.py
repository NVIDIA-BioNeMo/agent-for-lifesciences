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

"""Tests for the CI gate.

The gate is the only thing standing between a broken eval run and a green
pipeline, and most of its work is refusing to be fooled by a job that looks
finished. Those refusals are what these tests cover.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gate  # noqa: E402  -- needs the path above


REAL_JOB = Path(__file__).resolve().parent / "data" / "real_job"


def write_job(
    tmp_path: Path,
    trials: list[dict],
    name: str = "job-1",
    n_total_trials: int | None = None,
) -> Path:
    """A job directory laid out the way harbor lays one out.

    The job's own result.json is an aggregate with no per-trial detail; rewards
    live in a result.json inside each trial's directory. Fabricating the inlined
    ``trial_results`` key an earlier version of the gate expected is how the gate
    came to pass its tests while failing on a real job.
    """
    job_dir = tmp_path / name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-0000-0000-000000000000",
                "started_at": "2026-08-06T00:00:00Z",
                "n_total_trials": (
                    len(trials) if n_total_trials is None else n_total_trials
                ),
                "stats": {"n_completed_trials": len(trials), "n_errored_trials": 0},
            }
        )
    )
    for index, record in enumerate(trials):
        # Harbor suffixes the trial directory, so repeated attempts of one task
        # do not collide.
        trial_dir = job_dir / f"{record.get('task_name', 'unnamed')}__{index}"
        trial_dir.mkdir()
        (trial_dir / "result.json").write_text(json.dumps(record))
    return job_dir


def trial(task: str, reward: float | None = None, **extra) -> dict:
    record: dict = {"task_name": task, "trial_name": f"{task}.1"}
    if reward is not None:
        record["verifier_result"] = {"rewards": {"reward": reward}}
    record.update(extra)
    return record


def test_a_passing_suite_exits_zero(tmp_path, capsys):
    job = write_job(tmp_path, [trial("a", 1.0), trial("b", 0.9)])
    assert gate.main(["--job-dir", str(job)]) == 0
    assert "eval gate passed" in capsys.readouterr().out


def test_a_low_mean_fails(tmp_path, capsys):
    job = write_job(tmp_path, [trial("a", 1.0), trial("b", 0.2)])
    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.8"]) == 1
    assert "is below 0.8" in capsys.readouterr().err


def test_one_collapsed_task_can_be_caught_behind_a_good_mean(tmp_path, capsys):
    """Three perfect tasks and one dead one still average 0.75."""
    trials = [trial("a", 1.0), trial("b", 1.0), trial("c", 1.0), trial("d", 0.0)]
    job = write_job(tmp_path, trials)

    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.7"]) == 0
    assert (
        gate.main(
            [
                "--job-dir",
                str(job),
                "--min-mean-reward",
                "0.7",
                "--min-task-reward",
                "0.5",
            ]
        )
        == 1
    )
    assert "task(s) below 0.5: d 0.000" in capsys.readouterr().err


def test_an_errored_trial_counts_as_zero(tmp_path, capsys):
    """Skipping it would raise the mean of whatever else ran."""
    job = write_job(
        tmp_path,
        [
            trial("a", 1.0),
            trial(
                "b",
                exception_info={
                    "exception_type": "TimeoutError",
                    "exception_message": "agent timed out",
                    "exception_traceback": "...",
                    "occurred_at": "2026-08-06T00:00:00Z",
                },
            ),
        ],
    )
    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.8"]) == 1

    captured = capsys.readouterr()
    assert "mean reward: 0.500" in captured.out
    assert "TimeoutError" in captured.err


def test_a_long_exception_keeps_the_part_that_says_why(tmp_path, capsys):
    """Compose prints the command first and the reason last.

    A head-only truncation reported "Docker compose command failed ... Command:
    docker compose --project-name" over and over, which named the failure and
    hid its cause.
    """
    message = (
        "Docker compose command failed for environment docking-affinity. "
        "Command: docker compose --project-name x__env --project-directory "
        + "/very/long/path" * 40
        + " up -d. Error: bind source path does not exist: /builds/x/API_KEY_NV"
    )
    job = write_job(
        tmp_path,
        [
            trial(
                "a",
                exception_info={
                    "exception_type": "RuntimeError",
                    "exception_message": message,
                },
            )
        ],
    )
    out = tmp_path / "gate.json"
    assert gate.main(["--job-dir", str(job), "--json-out", str(out)]) == 1

    printed = capsys.readouterr().err
    assert "bind source path does not exist" in printed
    assert "Docker compose command failed" in printed
    # The artifact keeps the whole thing, since the console line is condensed.
    assert json.loads(out.read_text())["exceptions"][0]["message"] == message


def test_the_harness_liveness_check_cannot_satisfy_the_gate(tmp_path, capsys):
    """A job that forgot --verifier must not pass on `harness_ok` alone."""
    job = write_job(
        tmp_path,
        [{"task_name": "a", "verifier_result": {"rewards": {"harness_ok": 1.0}}}],
    )
    assert gate.main(["--job-dir", str(job)]) == 1

    captured = capsys.readouterr()
    assert "not AevomVerifier" in captured.err
    assert "--verifier" in captured.err


def test_missing_rewards_count_as_zero(tmp_path, capsys):
    job = write_job(tmp_path, [trial("a", 1.0), trial("b")])
    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.8"]) == 1
    assert "no rewards recorded" in capsys.readouterr().err


def test_too_few_trials_fails_even_at_full_reward(tmp_path, capsys):
    """An empty-ish suite scores 1.0 on the one thing it managed to run."""
    job = write_job(tmp_path, [trial("a", 1.0)])
    assert gate.main(["--job-dir", str(job), "--min-trials", "4"]) == 1
    assert "only 1 trial(s) graded" in capsys.readouterr().err


def test_repeated_attempts_average_per_task(tmp_path, capsys):
    """--n-attempts 3 on the live tier means three trials for one task."""
    job = write_job(
        tmp_path, [trial("a", 1.0), trial("a", 0.0), trial("a", 0.5)]
    )
    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.4"]) == 0
    assert "0.500  over 3 trial(s)" in capsys.readouterr().out


def test_a_job_with_no_trials_is_an_error_not_a_pass(tmp_path, capsys):
    job = write_job(tmp_path, [])
    assert gate.main(["--job-dir", str(job)]) == 2
    assert "none finished" in capsys.readouterr().err


def test_a_trial_that_left_no_result_json_counts_as_zero(tmp_path, capsys):
    """What a cancelled trial leaves behind: a count and nothing else."""
    job = write_job(tmp_path, [trial("a", 1.0)], n_total_trials=2)
    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.8"]) == 1

    captured = capsys.readouterr()
    assert "1 of 2 trial(s) left no result.json" in captured.err
    assert "mean reward: 0.500" in captured.out


def test_an_unreadable_trial_result_counts_as_zero(tmp_path, capsys):
    job = write_job(tmp_path, [trial("a", 1.0)], n_total_trials=2)
    truncated = job / "b__1"
    truncated.mkdir()
    (truncated / "result.json").write_text("{oops")

    assert gate.main(["--job-dir", str(job), "--min-mean-reward", "0.8"]) == 1
    captured = capsys.readouterr()
    assert "unreadable" in captured.err
    assert "mean reward: 0.500" in captured.out


def test_a_real_harbor_job_grades(capsys):
    """Recorded from harbor 0.20.0, so the fabricated fixtures cannot drift alone.

    The gate originally read a ``trial_results`` key off the job's result.json.
    Every fabricated test passed; the first real job failed with "contains no
    trial_results" after a full run had already scored 1.0.
    """
    assert gate.main(["--job-dir", str(REAL_JOB), "--min-mean-reward", "0.9"]) == 0

    out = capsys.readouterr().out
    assert "aevom/small-molecule-properties" in out
    assert "mean reward: 1.000" in out


def test_a_missing_result_json_is_an_error(tmp_path, capsys):
    (tmp_path / "half-finished").mkdir()
    assert gate.main(["--jobs-dir", str(tmp_path)]) == 2
    assert "did not finish" in capsys.readouterr().err


def test_a_missing_jobs_dir_is_an_error(tmp_path, capsys):
    assert gate.main(["--jobs-dir", str(tmp_path / "nope")]) == 2
    assert "no jobs directory" in capsys.readouterr().err


def test_the_newest_job_is_chosen(tmp_path):
    """Iterating a rerun leaves several job directories behind."""
    old = write_job(tmp_path, [trial("a", 0.0)], name="old")
    new = write_job(tmp_path, [trial("a", 1.0)], name="new")
    import os

    os.utime(old / "result.json", (1, 1))
    os.utime(new / "result.json", (2, 2))

    assert gate.newest_job(tmp_path) == new


def test_summary_is_written_for_the_artifact(tmp_path):
    job = write_job(tmp_path, [trial("a", 1.0), trial("b", 0.5)])
    out = tmp_path / "out" / "gate.json"
    gate.main(
        ["--job-dir", str(job), "--min-mean-reward", "0.7", "--json-out", str(out)]
    )

    summary = json.loads(out.read_text())
    assert summary["passed"] is True
    assert summary["mean_reward"] == 0.75
    assert summary["task_means"] == {"a": 1.0, "b": 0.5}


def test_malformed_json_is_an_error(tmp_path, capsys):
    job = tmp_path / "bad"
    job.mkdir()
    (job / "result.json").write_text("{not json")
    assert gate.main(["--job-dir", str(job)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


@pytest.mark.parametrize("reward_key", ["reward", "answer"])
def test_the_graded_dimension_is_selectable(tmp_path, reward_key):
    """The live tier may want to gate on one dimension alone."""
    job = write_job(
        tmp_path,
        [{"task_name": "a", "verifier_result": {"rewards": {reward_key: 1.0}}}],
    )
    assert gate.main(["--job-dir", str(job), "--reward-key", reward_key]) == 0
