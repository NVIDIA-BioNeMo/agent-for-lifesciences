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

"""Aevom as a Harbor agent.

Harbor's "agent" is the system under evaluation, so the roles invert relative to
Aevom's own vocabulary: Aevom is the agent, the Aevom Compose stack is the
environment, and this class is the client that drives it.

It does almost nothing itself. The work happens in ``driver/``, which runs inside
the environment's ``main`` container, because the Aevom stack publishes no host
ports -- fixed ports would collide the moment ``--n-concurrent`` exceeds one.
This class uploads the driver, execs it, and brings the trajectory back.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from aevom_eval.atif import model_of, to_atif

# Where the driver and its outputs live inside `main`.
HARNESS_DIR = "/harness"
TRAJECTORY_PATH = f"{HARNESS_DIR}/trajectory.json"
INSTRUCTION_PATH = f"{HARNESS_DIR}/instruction.md"

_PACKAGE_ROOT = Path(__file__).resolve().parent

# Compose service names from docker-compose.eval.yml, resolved on the Harbor
# network from inside `main`.
DEFAULT_API_BASE = "http://api:8000"
DEFAULT_MOLSTAR_BASE = "http://molstar-headless:8020"


class AevomAgent(BaseAgent):
    """Drives one Aevom turn per task and records everything it observed."""

    @staticmethod
    def name() -> str:
        return "aevom"

    def version(self) -> str | None:
        # Tracks the stack under test, not the harness: the images are what a
        # result is attributable to.
        return self.extra_env.get("AEVOM_IMAGE_TAG") or "local"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        env = self.extra_env
        self._api_base = env.get("AEVOM_API_BASE", DEFAULT_API_BASE)
        self._molstar_base = env.get("AEVOM_MOLSTAR_BASE", DEFAULT_MOLSTAR_BASE)
        self._max_tokens = int(env.get("AEVOM_MAX_TOKENS", "8192"))
        self._temperature = env.get("AEVOM_TEMPERATURE")
        self._health_timeout = float(env.get("AEVOM_HEALTH_TIMEOUT", "300"))
        self._turn_timeout = float(env.get("AEVOM_TURN_TIMEOUT", "1800"))

    async def setup(self, environment: BaseEnvironment) -> None:
        """Place the driver in the environment and confirm the stack answers.

        Health is checked here rather than in ``run()`` so an unhealthy stack
        fails during setup, where Harbor reports it as an infrastructure problem
        instead of an agent that scored zero.
        """
        await environment.ensure_dirs([HARNESS_DIR])
        await environment.upload_dir(
            source_dir=_PACKAGE_ROOT / "driver",
            target_dir=f"{HARNESS_DIR}/aevom_eval/driver",
        )
        # `python -m aevom_eval.driver.run_task` needs the parent package to be
        # importable, and the uploaded tree has no __init__.py above `driver`.
        await environment.exec(f"touch {HARNESS_DIR}/aevom_eval/__init__.py")

        # Reuses the driver's own wait rather than restating it, so there is one
        # definition of "healthy" and it is the one under test.
        probe = (
            "from aevom_eval.driver.aevom_client import AevomClient;"
            f"print(AevomClient({self._api_base!r})"
            f".wait_until_healthy({self._health_timeout!r}))"
        )
        result = await environment.exec(
            f"python -c {shlex.quote(probe)}",
            cwd=HARNESS_DIR,
            timeout_sec=int(self._health_timeout) + 60,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Aevom stack did not become healthy during agent setup:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        self.logger.info("Aevom stack healthy: %s", (result.stdout or "").strip())

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await _write_instruction(environment, instruction)

        command = self._driver_command()
        self.logger.info("Running Aevom driver: %s", command)
        result = await environment.exec(
            command,
            cwd=HARNESS_DIR,
            timeout_sec=int(self._turn_timeout) + 120,
        )

        (self.logs_dir / "driver.stdout.txt").write_text(result.stdout or "")
        (self.logs_dir / "driver.stderr.txt").write_text(result.stderr or "")

        # The trajectory is fetched whatever the exit code: a driver that failed
        # partway still recorded what the stack did, and that is the evidence the
        # verifier needs to distinguish a broken harness from a poor answer.
        trajectory = await self._collect_trajectory(environment)

        context.metadata = {
            "driver_return_code": result.return_code,
            "api_base": self._api_base,
            "model_name": self.model_name,
            **_trajectory_summary(trajectory),
        }

        # Harbor sums these across a job's trials to fill its token column.
        # Left None when the turn reported no usage, so a stream that died is
        # absent from the total rather than counted as a free trial.
        # n_cache_tokens stays unset: the stack records prompt and completion
        # tokens only, and it has no cached figure to report.
        usage = (trajectory or {}).get("usage") or {}
        context.n_input_tokens = usage.get("input_tokens")
        context.n_output_tokens = usage.get("output_tokens")

        if result.return_code != 0 and trajectory is None:
            raise RuntimeError(
                "Aevom driver failed and produced no trajectory:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def _driver_command(self) -> str:
        args = [
            "python",
            "-m",
            "aevom_eval.driver.run_task",
            "--instruction-file",
            INSTRUCTION_PATH,
            "--output",
            TRAJECTORY_PATH,
            "--api-base",
            self._api_base,
            "--molstar-base",
            self._molstar_base,
            "--max-tokens",
            str(self._max_tokens),
            "--health-timeout",
            str(self._health_timeout),
            "--turn-timeout",
            str(self._turn_timeout),
        ]
        # Harbor's --model arrives as self.model_name and becomes
        # ChatRequest.model_id, so the eval selects a model the same way the
        # frontend's model picker does. Left unset, the stack uses AEVOM_MODEL
        # from the production ConfigMap.
        if self.model_name:
            args += ["--model", self.model_name]
        if self._temperature:
            args += ["--temperature", str(self._temperature)]
        return " ".join(shlex.quote(arg) for arg in args)

    async def _collect_trajectory(self, environment: BaseEnvironment) -> dict | None:
        local_path = self.logs_dir / "trajectory.json"
        try:
            await environment.download_file(
                source_path=TRAJECTORY_PATH, target_path=local_path
            )
        except Exception as e:
            self.logger.warning("Could not download trajectory: %s", e)
            return None

        try:
            trajectory = json.loads(local_path.read_text())
        except Exception as e:
            self.logger.warning("Trajectory at %s is unreadable: %s", local_path, e)
            return None

        self._rewrite_as_atif(local_path, trajectory)
        return trajectory

    def _rewrite_as_atif(self, local_path: Path, trajectory: dict) -> None:
        """Restate the trajectory in the format the viewer's page assumes.

        This path is Harbor's, not ours: its viewer serves whatever sits here as
        an ATIF document. The driver's record survives inside it, so a failure
        here costs the rendered page but never the evidence.
        """
        try:
            document = to_atif(
                trajectory,
                agent_name=self.name(),
                agent_version=self.version() or "local",
                model_name=self.model_name,
            )
            local_path.write_text(json.dumps(document, indent=2, default=str))
        except Exception as e:
            self.logger.warning("Could not write the ATIF trajectory: %s", e)


async def _write_instruction(environment: BaseEnvironment, instruction: str) -> None:
    """Put the instruction in a file rather than on the command line.

    Instructions are multi-line prose with quotes and parentheses -- MolQL
    expressions especially -- and shell-quoting them into a heredoc is a
    needless way to corrupt a task.
    """
    await environment.exec(
        "python -c "
        + shlex.quote(
            "import base64,pathlib,sys;"
            f"p=pathlib.Path({INSTRUCTION_PATH!r});"
            "p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_bytes(base64.b64decode(sys.argv[1]))"
        )
        + " "
        + shlex.quote(_b64(instruction))
    )


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode()).decode("ascii")


def _trajectory_summary(trajectory: dict | None) -> dict:
    """A compact view for Harbor's result JSON, so triage needs no file digging."""
    if not trajectory:
        return {"trajectory": "missing"}
    tool_calls = trajectory.get("tool_calls") or []
    return {
        "session_id": trajectory.get("session_id"),
        "aevom_model": model_of(trajectory),
        "duration_sec": trajectory.get("duration_sec"),
        "n_tool_calls": len(tool_calls),
        "tools_used": sorted({c.get("name") for c in tool_calls if c.get("name")}),
        "n_failed_tool_calls": sum(
            1 for call in tool_calls if call.get("success") is False
        ),
        "n_artifacts": len(trajectory.get("artifacts") or []),
        "answer_chars": len(trajectory.get("final_content") or ""),
        "hit_iteration_limit": bool(trajectory.get("hit_iteration_limit")),
        "driver_errors": trajectory.get("errors") or [],
    }
