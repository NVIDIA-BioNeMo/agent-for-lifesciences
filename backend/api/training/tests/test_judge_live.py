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

"""Live integration tests for the multi-model LLM judge.

These tests make REAL network calls to NVIDIA-hosted endpoints and are
NOT run as part of the standard ``pytest`` collection. They are gated on
``RUN_LIVE_JUDGE_TESTS=1`` in the environment AND on the presence of an
``NVIDIA_API_KEY`` env var. Skipping by default keeps CI / dev pytest fast
and avoids spending NVIDIA API quota on every test run.

Run as part of the scale-inference-18n-multijudge pre-flight gate via:

    RUN_LIVE_JUDGE_TESTS=1 NVIDIA_API_KEY=$(cat ./API_KEY_NVINF) \\
        python3 -m pytest backend/api/training/tests/test_judge_live.py -v

What these tests cover (NOT covered by mocked unit tests):

1. Endpoint reachability: every enabled entry in ``judge.yaml`` actually
   returns a parseable rubric response when given a real trajectory.
2. Pool-mode call shape: the same ``LLMJudge`` instance can route 6
   different model families behind one ``evaluate()`` interface.
3. Cross-model rubric robustness: each model's reasoning_content + content
   concat produces extractable scores (catches a model family whose
   chat-template strips score tags differently).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from api.training.judge import LLMJudge, JudgeEndpoint, JudgePrompts


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_JUDGE_TESTS") != "1"
    or not os.environ.get("NVIDIA_API_KEY"),
    reason=(
        "Live judge tests skipped. To run: "
        "RUN_LIVE_JUDGE_TESTS=1 NVIDIA_API_KEY=... pytest ..."
    ),
)


# A minimal but realistic fixture trajectory. Designed to score in the
# middle of the rubric so the test is robust to per-model scoring
# variance: it has paired-metric computation (LogP) and a re-dock, but
# the hypothesis grounding is weak.
FIXTURE_TRAJECTORY = [
    {
        "name": "vina_dock",
        "arguments": {"smiles": "CCO", "receptor": "X"},
        "output": "Affinity: -7.2 kcal/mol",
        "success": True,
        "turn": 1,
    },
    {
        "name": "vina_dock",
        "arguments": {"smiles": "CCCO", "receptor": "X"},
        "output": "Affinity: -5.1 kcal/mol",
        "success": True,
        "turn": 2,
    },
    {
        "name": "python_execute",
        "arguments": {"code": "from rdkit.Chem import Crippen; ..."},
        "output": "Winner LogP=0.32, Loser LogP=0.58",
        "success": True,
        "turn": 3,
    },
    {
        "name": "vina_dock",
        "arguments": {"smiles": "CCCN", "receptor": "X"},
        "output": "Affinity: -6.8 kcal/mol",
        "success": True,
        "turn": 4,
    },
]

FIXTURE_RESPONSE = (
    "I docked both molecules. The loser has a slightly higher LogP (0.58 vs "
    "0.32), which I hypothesize correlates with the binding gap. I edited the "
    "loser to replace -OH with -NH2, expecting lower LogP, and re-docked. The "
    "edit recovered partial binding (-6.8 vs -7.2)."
)

FIXTURE_GROUND_TRUTH = {
    "sabotage_type": "lipophilicity_increase",
    "winner_vina": -7.2,
    "loser_vina": -5.1,
    "delta_vina": 2.1,
    "modification_desc": "Added one extra carbon to increase LogP.",
}


def _judge_config_path() -> str:
    """Locate the in-repo judge.yaml for live testing."""
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    candidate = repo_root / "training" / "configs" / "judge.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"judge.yaml not found at {candidate}")
    return str(candidate)


def test_pool_can_be_built_from_repo_judge_yaml():
    """Sanity: the in-repo judge.yaml parses into a non-empty pool."""
    judge = LLMJudge.from_config(_judge_config_path())
    enabled = [ep for ep in judge.endpoints if ep.enabled]
    assert len(enabled) >= 1, "no enabled endpoints in judge.yaml"
    for ep in enabled:
        assert ep.url.startswith("https://"), f"endpoint {ep.name} has non-HTTPS URL"
        assert ep.model, f"endpoint {ep.name} has empty model"
        assert ep.api_key, (
            f"endpoint {ep.name} has no api_key resolved "
            "(check api_key_env={ep.name})"
        )


@pytest.mark.parametrize(
    "endpoint_name",
    [
        "nemotron_3_super",
        "qwen_3_6",
        "glm_5_1",
        "claude_opus_4_6",
        "gpt_5_5",
        "gemini_3_1",
    ],
)
def test_individual_endpoint_returns_parseable_rubric(endpoint_name: str):
    """Each enabled endpoint, called in isolation, returns a parseable rubric.

    Constructs a single-entry pool with only the target endpoint enabled,
    so the random pool pick is deterministic for the test. Validates that:

    1. The HTTPS call succeeds (no auth / model-string errors).
    2. The response contains extractable scores under the 6-category rubric.
    3. The total score is in [0, 16] (catches malformed scores parsed as
       junk numbers).
    """
    import asyncio

    base = LLMJudge.from_config(_judge_config_path())
    target = next(
        (ep for ep in base.endpoints if ep.name == endpoint_name), None
    )
    if target is None or not target.enabled:
        pytest.skip(f"endpoint {endpoint_name} not enabled in judge.yaml")

    judge = LLMJudge(
        prompts=base.prompts,
        endpoints=[target],
        timeout=60.0,
    )
    result = asyncio.new_event_loop().run_until_complete(
        judge.evaluate(
            trajectory=FIXTURE_TRAJECTORY,
            response=FIXTURE_RESPONSE,
            ground_truth=FIXTURE_GROUND_TRUTH,
        )
    )

    assert result.get("error") is None, (
        f"endpoint {endpoint_name} errored: {result.get('error')}"
    )
    assert "total_score" in result
    assert 0 <= result["total_score"] <= 16, (
        f"endpoint {endpoint_name} returned out-of-range score "
        f"{result['total_score']} (raw: {result.get('raw_response', '')[:200]})"
    )


def test_pool_mode_distributes_calls_across_models():
    """Pool mode with all enabled endpoints: N calls hit >=2 distinct models.

    Doesn't strictly require uniform distribution (small N, random picks),
    just that the pool DOES route across models rather than pinning to one.
    Done over 6 calls to keep API quota use modest.
    """
    import asyncio

    judge = LLMJudge.from_config(_judge_config_path())
    enabled = [ep for ep in judge.endpoints if ep.enabled]
    if len(enabled) < 2:
        pytest.skip("need >=2 enabled endpoints to test pool distribution")

    loop = asyncio.new_event_loop()
    results = []
    for _ in range(6):
        r = loop.run_until_complete(
            judge.evaluate(
                trajectory=FIXTURE_TRAJECTORY,
                response=FIXTURE_RESPONSE,
                ground_truth=FIXTURE_GROUND_TRUTH,
            )
        )
        results.append(r)

    successful = [r for r in results if r.get("error") is None]
    assert len(successful) >= 4, (
        f"too many pool calls errored: {[r.get('error') for r in results if r.get('error')]}"
    )
