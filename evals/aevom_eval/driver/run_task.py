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

"""Drive one Aevom task from inside the Harbor environment.

Invoked by ``AevomAgent`` via ``environment.exec`` in the ``main`` container,
which sits on the same Compose network as the api. Writes the trajectory to
``--output`` for the host to collect, and prints a one-line summary to stdout so
a failure is legible in Harbor's logs without opening the JSON.

Usage:
    python -m aevom_eval.driver.run_task \
        --instruction-file /harness/instruction.md \
        --output /harness/trajectory.json \
        [--model nemotron-3-ultra] [--api-base http://api:8000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .aevom_client import AevomClient
from .molstar_bridge import MolstarBridge
from .trajectory import Trajectory

logger = logging.getLogger("aevom_eval.driver")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-file")
    parser.add_argument("--instruction")
    parser.add_argument("--output", default="/harness/trajectory.json")
    parser.add_argument(
        "--api-base",
        default="http://api:8000",
        help="Aevom api, resolved on the Compose network.",
    )
    parser.add_argument("--molstar-base", default="http://molstar-headless:8020")
    parser.add_argument(
        "--model",
        default=None,
        help="Passed through as ChatRequest.model_id; Harbor's --model arrives here.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Omitted by default so the model's own default applies, as in production.",
    )
    parser.add_argument("--health-timeout", type=float, default=300.0)
    parser.add_argument("--turn-timeout", type=float, default=1800.0)
    return parser.parse_args(argv)


def load_instruction(args: argparse.Namespace) -> str:
    if args.instruction:
        return args.instruction
    if args.instruction_file:
        return Path(args.instruction_file).read_text()
    raise SystemExit("one of --instruction or --instruction-file is required")


def run(args: argparse.Namespace) -> int:
    instruction = load_instruction(args)
    client = AevomClient(args.api_base)

    client.wait_until_healthy(args.health_timeout)
    session_id = client.create_session()
    bridge = MolstarBridge(args.api_base, args.molstar_base, client.owner_token)
    trajectory = Trajectory(
        instruction=instruction, session_id=session_id, model_id=args.model
    )

    try:
        trajectory.info = client.info()
    except Exception as e:
        # Informational only; a run without it is still scoreable.
        trajectory.errors.append(f"/api/info failed: {e}")

    started = time.monotonic()
    try:
        for event in client.chat(
            instruction,
            session_id,
            model_id=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.turn_timeout,
        ):
            trajectory.record(event)

            tool_event = event.get("tool_event")
            if tool_event:
                # Synchronous on purpose. The backend is blocked in
                # drain_pending_ops waiting for this POST, and the SSE stream is
                # the only thing feeding us, so there is nothing to overlap with.
                try:
                    bridge.note_tool_event(tool_event, session_id)
                except Exception as e:
                    logger.exception("bridge failed on %s", tool_event.get("type"))
                    trajectory.errors.append(f"bridge error: {e}")
    except Exception as e:
        # A stream that dies partway still produced a trajectory worth scoring;
        # the error is recorded and the verifier decides what it means.
        logger.exception("chat stream failed")
        trajectory.errors.append(f"chat stream failed: {e}")
    finally:
        trajectory.duration_sec = round(time.monotonic() - started, 3)
        trajectory.bridge_events = bridge.events

    # End-of-turn API state, gathered after the stream closes so it reflects
    # everything the turn produced.
    for label, getter in (
        ("artifacts", lambda: client.artifacts(session_id)),
        ("selections", lambda: client.selections(session_id)),
        ("conversation", lambda: client.conversation(session_id)),
    ):
        try:
            setattr(trajectory, label, getter())
        except Exception as e:
            trajectory.errors.append(f"/{label} fetch failed: {e}")

    trajectory.write(args.output)

    summary = {
        "session_id": session_id,
        "duration_sec": trajectory.duration_sec,
        "tools": trajectory.tool_names,
        "failed_tools": [call.get("name") for call in trajectory.failed_tool_calls],
        "structure_queries": [
            event
            for event in trajectory.bridge_events
            if event.get("kind") == "structure_query"
        ],
        "artifacts": len(trajectory.artifacts),
        "answer_chars": len(trajectory.final_content),
        "errors": trajectory.errors,
    }
    print(json.dumps(summary), flush=True)

    # A turn that produced no answer and no tool calls means the harness failed
    # to drive the system, which is different from the agent doing badly.
    if not trajectory.final_content and not trajectory.tool_calls:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
