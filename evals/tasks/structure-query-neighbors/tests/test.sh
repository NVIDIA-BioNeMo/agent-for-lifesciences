#!/bin/sh
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

# Harness liveness check -- NOT the real grader.
#
# Harbor requires every task to carry a test script, and this is the one that
# runs if a job forgets `--verifier aevom_eval.verifier:AevomVerifier`. Task
# expectations live in [metadata.expect] in task.toml, which is not present in
# the container, so this script cannot grade correctness. What it can do is tell
# you whether the harness drove Aevom at all.
#
# It deliberately reports under the key `harness_ok` rather than `reward`, so a
# job that skipped the real verifier cannot satisfy evals/scripts/gate.py by
# accident.
set -u

TRAJECTORY=/logs/agent/trajectory.json
REWARD=/logs/verifier/reward.json
mkdir -p /logs/verifier

echo "WARNING: running the harness liveness check, not the Aevom grader."
echo "         Pass --verifier aevom_eval.verifier:AevomVerifier to grade this task."

python3 - "$TRAJECTORY" "$REWARD" <<'PY'
import json
import sys

trajectory_path, reward_path = sys.argv[1], sys.argv[2]

try:
    with open(trajectory_path) as handle:
        document = json.load(handle)
except Exception as e:
    print(f"FAIL: no usable trajectory at {trajectory_path}: {e}")
    with open(reward_path, "w") as handle:
        json.dump({"harness_ok": 0.0}, handle)
    raise SystemExit(1)

# The agent writes this file as ATIF, since Harbor's viewer reads the path as
# one. The driver's own record rides under `extra`.
trajectory = (document.get("extra") or {}).get("aevom") or document

answered = bool(trajectory.get("final_content"))
called_tools = bool(trajectory.get("tool_calls"))
ok = answered or called_tools

print(f"session_id: {trajectory.get('session_id')}")
print(f"tool calls: {len(trajectory.get('tool_calls') or [])}")
print(f"answer chars: {len(trajectory.get('final_content') or '')}")
print(f"harness errors: {trajectory.get('errors') or 'none'}")

with open(reward_path, "w") as handle:
    json.dump({"harness_ok": 1.0 if ok else 0.0}, handle)

raise SystemExit(0 if ok else 1)
PY
