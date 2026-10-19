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

"""Harbor verifier for Aevom tasks.

Grades the trajectory ``AevomAgent`` left in the trial's agent log directory
against the ``[metadata.expect]`` block of the task's ``task.toml``. All the
judgement lives in ``scoring.py``; this class is the Harbor adapter around it.

It reads files rather than re-querying the stack. By verification time Harbor may
already have torn the environment down, and the trajectory already carries the
end-of-turn ``/api/storage/artifacts``, ``/api/selections`` and
``/api/conversation`` responses that the driver collected while the stack was up.

Note the one residual gap from running the production config: the production
ConfigMap sets ``AEVOM_FEAT_STORAGE_ACCESS_API=false``, which disables
``/api/storage/artifact/{ref}``. Artifacts are therefore graded on metadata --
type, count, description -- and not on content. Grading content would require
diverging from production config, which would cost more than it buys.
"""

from __future__ import annotations

import json
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from aevom_eval.atif import unwrap
from aevom_eval.scoring import OVERALL, reward_details, score_trajectory

TRAJECTORY_FILENAME = "trajectory.json"


class AevomVerifier(BaseVerifier):
    async def verify(self) -> VerifierResult:
        expect = self._expectations()
        trajectory = self._load_trajectory()
        score = score_trajectory(trajectory, expect)

        self._write_report(score, expect, trajectory)

        for check in score.failures:
            self.logger.warning(
                "FAIL [%s] %s: %s", check.dimension, check.name, check.detail
            )
        self.logger.info("Rewards: %s", score.rewards)

        return VerifierResult(rewards=dict(score.rewards))

    def _expectations(self) -> dict:
        metadata = self.task.config.metadata or {}
        expect = metadata.get("expect")
        if not expect:
            # A task with no expectations would silently score 0.0 on nothing,
            # which reads as a passing run of an empty test.
            raise ValueError(
                f"Task {self.task.name!r} has no [metadata.expect] block in "
                "task.toml, so there is nothing to verify."
            )
        return expect

    def _load_trajectory(self) -> dict | None:
        path = self._trajectory_path()
        if path is None:
            self.logger.error(
                "No %s under %s; the agent did not produce a trajectory",
                TRAJECTORY_FILENAME,
                self.trial_paths.agent_dir,
            )
            return None
        try:
            # The file is an ATIF document for the viewer's sake; the record this
            # grades rides inside it.
            return unwrap(json.loads(path.read_text()))
        except Exception as e:
            self.logger.error("Trajectory at %s is unreadable: %s", path, e)
            return None

    def _trajectory_path(self) -> Path | None:
        agent_dir = Path(self.trial_paths.agent_dir)
        direct = agent_dir / TRAJECTORY_FILENAME
        if direct.exists():
            return direct
        # Multi-step trials relocate agent logs under steps/<name>/agent.
        return next(iter(sorted(agent_dir.rglob(TRAJECTORY_FILENAME))), None)

    def _write_report(self, score, expect: dict, trajectory: dict | None) -> None:
        """Leave a readable verdict next to the trial, for triage without re-running."""
        report = {
            "task": self.task.name,
            "rewards": score.rewards,
            "expect": expect,
            "checks": score.to_dict()["checks"],
            "observed": _observed(trajectory),
        }
        destination = Path(self.trial_paths.verifier_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "aevom_verifier.json").write_text(
            json.dumps(report, indent=2, default=str)
        )

        lines = [f"task: {self.task.name}", ""]
        for key, value in score.rewards.items():
            marker = "" if key != OVERALL else "  <- overall"
            lines.append(f"{key}: {value}{marker}")
        lines.append("")
        for check in score.to_dict()["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(f"[{status}] {check['dimension']}/{check['name']}: {check['detail']}")
        (destination / "aevom_verifier.txt").write_text("\n".join(lines) + "\n")

        # The same verdict again under the two names Harbor's viewer looks for, so
        # its verifier panel renders the dimensions and their checks instead of
        # leaving a reader to open aevom_verifier.json from the file browser. The
        # files above stay the source of truth; these are a projection of them
        # into the viewer's schema.
        (destination / "reward.json").write_text(json.dumps(score.rewards, indent=2))
        (destination / "reward-details.json").write_text(
            json.dumps(reward_details(score), indent=2)
        )


def _observed(trajectory: dict | None) -> dict:
    if not trajectory:
        return {}
    calls = trajectory.get("tool_calls") or []
    return {
        "session_id": trajectory.get("session_id"),
        "model": (trajectory.get("info") or {}).get("model_id"),
        "duration_sec": trajectory.get("duration_sec"),
        "tools_called": [call.get("name") for call in calls],
        "failed_tools": [
            call.get("name") for call in calls if call.get("success") is False
        ],
        "artifact_types": sorted(
            {
                artifact.get("data_type")
                for artifact in (trajectory.get("artifacts") or [])
                if artifact.get("data_type")
            }
        ),
        "structure_queries": [
            {
                "molql": event.get("molql"),
                "atom_count": event.get("atom_count"),
                "residue_count": event.get("residue_count"),
                "error": event.get("error"),
            }
            for event in (trajectory.get("bridge_events") or [])
            if event.get("kind") == "structure_query"
        ],
        "hit_iteration_limit": trajectory.get("hit_iteration_limit"),
        "driver_errors": trajectory.get("errors") or [],
        "answer": trajectory.get("final_content"),
    }
