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

"""Unit tests for the LLM judge."""

import pytest
from api.training.judge import (
    JudgeEndpoint,
    LLMJudge,
    JudgePrompts,
    JudgeResult,
    MockJudge,
)
from api.training.judge_prompts import (
    JUDGE_PROMPTS,
    get_judge_prompts,
)


class TestJudgePrompts:
    """Tests for judge prompts configuration."""
    
    def test_default_prompts_exist(self):
        """Test that default prompts are available."""
        prompts = JudgePrompts()
        
        assert prompts.system_prompt is not None
        assert len(prompts.system_prompt) > 100
        assert prompts.prompt_template is not None
        assert len(prompts.prompt_template) > 100
    
    def test_score_patterns_defined(self):
        """Test that the 6-category advanced rubric score patterns are defined."""
        prompts = JudgePrompts()

        assert "logical_consistency" in prompts.score_patterns
        assert "differentiating_metric" in prompts.score_patterns
        assert "hypothesis_grounded" in prompts.score_patterns
        assert "scientific_accuracy" in prompts.score_patterns
        assert "edit_tests_hypothesis" in prompts.score_patterns
        assert "diagnostic_quality" in prompts.score_patterns
    
    def test_verdict_patterns(self):
        """Test verdict patterns."""
        prompts = JudgePrompts()
        
        assert "SOUND" in prompts.sound_verdict_pattern
        assert "FLAWED" in prompts.flawed_verdict_pattern
    
    def test_get_judge_prompts_default(self):
        """Test getting default judge prompts."""
        prompts = get_judge_prompts("default")
        
        assert prompts is not None
    
    def test_get_judge_prompts_strict(self):
        """Test getting strict scientific prompts."""
        prompts = get_judge_prompts("strict_scientific")
        
        assert "system_prompt" in prompts
        assert "SCIENTIFIC_METHOD" in prompts["system_prompt"]
    
    def test_get_judge_prompts_binary(self):
        """Test getting binary prompts."""
        prompts = get_judge_prompts("binary")
        
        assert "ACCEPT" in prompts["sound_verdict_pattern"]
        assert "REJECT" in prompts["flawed_verdict_pattern"]
    
    def test_get_judge_prompts_unknown(self):
        """Test error on unknown prompt set."""
        with pytest.raises(ValueError):
            get_judge_prompts("nonexistent_prompt_set")
    
    def test_all_prompt_sets_available(self):
        """Test all prompt sets in registry."""
        expected_sets = ["default", "strict_scientific", "gentle_mentor", "binary", "chemistry_detailed"]
        
        for name in expected_sets:
            prompts = get_judge_prompts(name)
            assert prompts is not None


class TestJudgeResponseParsing:
    """Tests for judge response parsing."""
    
    @pytest.fixture
    def judge(self):
        """Create a judge with default config."""
        prompts = JudgePrompts()
        return LLMJudge(
            prompts=prompts,
            model_url="http://localhost:8000/v1",
        )
    
    def test_parse_sound_verdict(self, judge):
        """Test parsing a SOUND verdict under the 6-category advanced rubric."""
        response = """
LOGICAL_CONSISTENCY: 3
DIFFERENTIATING_METRIC: 3
HYPOTHESIS_GROUNDED: 2
SCIENTIFIC_ACCURACY: 2
EDIT_TESTS_HYPOTHESIS: 2
DIAGNOSTIC_QUALITY: 1

REASONING: The reasoning was well-structured and metric-grounded.

VERDICT: [[SOUND]]
"""
        result = judge._parse_response(response)

        assert result.is_sound is True
        assert result.logical_consistency == 3
        assert result.differentiating_metric == 3
        assert result.hypothesis_grounded == 2
        assert result.scientific_accuracy == 2
        assert result.edit_tests_hypothesis == 2
        assert result.diagnostic_quality == 1
        assert result.total_score == 13

    def test_parse_flawed_verdict(self, judge):
        """Test parsing a FLAWED verdict."""
        response = """
LOGICAL_CONSISTENCY: 1
DIFFERENTIATING_METRIC: 0
HYPOTHESIS_GROUNDED: 0
SCIENTIFIC_ACCURACY: 1
EDIT_TESTS_HYPOTHESIS: 0
DIAGNOSTIC_QUALITY: 0

REASONING: The analysis lacked metric support and made no edit.

VERDICT: [[FLAWED]]
"""
        result = judge._parse_response(response)

        assert result.is_sound is False
        assert result.total_score == 2

    def test_parse_extracts_reasoning(self, judge):
        """Test that reasoning is extracted."""
        response = """
LOGICAL_CONSISTENCY: 2
DIFFERENTIATING_METRIC: 2
HYPOTHESIS_GROUNDED: 2
SCIENTIFIC_ACCURACY: 2
EDIT_TESTS_HYPOTHESIS: 2
DIAGNOSTIC_QUALITY: 1

REASONING: The model showed good understanding but missed some key points.

VERDICT: [[SOUND]]
"""
        result = judge._parse_response(response)

        assert "good understanding" in result.reasoning

    def test_parse_with_missing_scores(self, judge):
        """Test parsing with missing scores defaults to 0."""
        response = """
LOGICAL_CONSISTENCY: 2

VERDICT: [[FLAWED]]
"""
        result = judge._parse_response(response)

        assert result.logical_consistency == 2
        assert result.differentiating_metric == 0
        assert result.hypothesis_grounded == 0
        assert result.scientific_accuracy == 0
        assert result.edit_tests_hypothesis == 0
        assert result.diagnostic_quality == 0

    def test_reward_calculation(self, judge):
        """Test reward is calculated from score (max 16, scale 0.3)."""
        response = """
LOGICAL_CONSISTENCY: 3
DIFFERENTIATING_METRIC: 3
HYPOTHESIS_GROUNDED: 3
SCIENTIFIC_ACCURACY: 3
EDIT_TESTS_HYPOTHESIS: 3
DIAGNOSTIC_QUALITY: 1

VERDICT: [[SOUND]]
"""
        result = judge._parse_response(response)

        # Total is 16/16, reward_scale is 0.3 by default
        assert result.reward == pytest.approx(0.3, rel=0.01)

    def test_parse_response_last_match_wins_over_cot_leak(self, judge):
        """Locks in the v0528a verdict-leak + tentative-score-leak fix.

        Reasoning models (Nemotron Super/Ultra, Qwen 3.6, GLM 5.1, etc.) emit
        chain-of-thought in ``reasoning_content`` and the formal answer in
        ``content``. While deliberating they can write tentative scores
        ("LOGICAL_CONSISTENCY: 3 -- wait, 2") and opposing-verdict markers
        ("could be SOUND if X, but FLAWED because Y"). Pre-2026-05-28 the
        parser used ``re.search`` (first-match) on a concatenated
        ``reasoning_content + content`` text, so the FIRST tentative score /
        stray verdict marker won over the FINAL committed one in content.
        That silently flipped the reward sign on borderline cases.

        The fix is last-match for both score patterns and verdict patterns.
        The model's final commitment is always at the END of its output --
        either in content (when it emits one) or at the tail of reasoning
        (truncation case). Last-match locks onto that commitment.

        This test mirrors the exact shape ``_call_judge`` produces:
        ``response_text = reasoning_content + "\\n" + content``. If anyone
        reverts the parser to first-match, this test fails.
        """
        # ``reasoning_content`` is verbose CoT with tentative alternatives:
        # earlier tentative scores (3 instead of 2), and a stray [[SOUND]]
        # marker the model wrote while considering the opposite verdict.
        reasoning_content = """\
The user wants me to evaluate the assistant's reasoning.
Initial impression: this could be [[SOUND]] if the LogP-causation argument
holds. Tentative scores:
LOGICAL_CONSISTENCY: 3 -- wait, the causal link is actually weak.
DIFFERENTIATING_METRIC: 3 -- LogP is computed for both pairs.
HYPOTHESIS_GROUNDED: 2 -- partially.
On reflection the mechanism is misattributed (LogP is correlative, not the
direct cause of H-bond loss). So this is [[FLAWED]] after all.
"""

        # ``content`` is the formal answer (final scores + verdict).
        content = """\
LOGICAL_CONSISTENCY: 2
DIFFERENTIATING_METRIC: 2
HYPOTHESIS_GROUNDED: 2
SCIENTIFIC_ACCURACY: 1
EDIT_TESTS_HYPOTHESIS: 2
DIAGNOSTIC_QUALITY: 0

REASONING: Mechanism misattributed; LogP correlative not causal.

VERDICT: [[FLAWED]]"""

        # Mirror _call_judge's concat: reasoning + content.
        response_text = reasoning_content + "\n" + content
        result = judge._parse_response(response_text)

        # Verdict must be FLAWED (last marker = content's [[FLAWED]]),
        # not the in-CoT [[SOUND]] near the start of reasoning.
        assert result.is_sound is False, (
            "verdict must be FLAWED (last marker = content's commitment); "
            "a SOUND result means the parser leaked an early CoT marker -- "
            "regression of the v0528a verdict-leak fix"
        )
        # Scores must come from content (final values, last occurrence of
        # each label), not the tentative 3-3 the model wrote earlier in CoT.
        assert result.logical_consistency == 2, "tentative 3 in CoT must lose to content's 2"
        assert result.differentiating_metric == 2, "tentative 3 in CoT must lose to content's 2"
        assert result.hypothesis_grounded == 2
        assert result.scientific_accuracy == 1
        assert result.edit_tests_hypothesis == 2
        assert result.diagnostic_quality == 0
        assert result.total_score == 9

    def test_parse_response_truncation_falls_back_to_reasoning(self, judge):
        """When content is empty (finish_reason=length), parser uses reasoning.

        This is the truncation case: the model ran out of max_tokens budget
        before emitting any content. _call_judge's ``content or
        reasoning_content`` fallback kicks in, and _parse_response sees only
        the reasoning text. We must still extract whatever scores / verdict
        markers the reasoning emitted before being cut off.
        """
        reasoning_only = """\
Let me work through the rubric.
LOGICAL_CONSISTENCY: 2
DIFFERENTIATING_METRIC: 3
HYPOTHESIS_GROUNDED: 2
SCIENTIFIC_ACCURACY: 2
EDIT_TESTS_HYPOTHESIS: 3
DIAGNOSTIC_QUALITY: 1
The verdict is [[SOUND]] given the total of 13/16.
"""
        result = judge._parse_response(reasoning_only)

        assert result.is_sound is True
        assert result.total_score == 13


class TestMockJudge:
    """Tests for MockJudge."""

    def test_mock_judge_returns_result(self):
        """Test mock judge returns a result under the /16 advanced rubric."""
        import asyncio
        judge = MockJudge(default_score=11, is_sound=True)

        result = asyncio.run(
            judge.evaluate(trajectory=[], response={}, ground_truth={})
        )

        assert result["is_sound"] is True
        assert result["total_score"] == 11
        assert result["max_score"] == 16
        assert result["reward"] > 0
        assert "differentiating_metric" in result["scores"]
        assert "hypothesis_grounded" in result["scores"]
        assert "edit_tests_hypothesis" in result["scores"]

    def test_mock_judge_flawed(self):
        """Test mock judge can return flawed."""
        import asyncio
        judge = MockJudge(default_score=3, is_sound=False)

        result = asyncio.run(
            judge.evaluate(trajectory=[], response={}, ground_truth={})
        )

        assert result["is_sound"] is False
        assert result["total_score"] == 3


class TestJudgeToolCallsFormatting:
    """Tests for tool calls formatting."""
    
    @pytest.fixture
    def judge(self):
        """Create a judge."""
        prompts = JudgePrompts()
        return LLMJudge(
            prompts=prompts,
            model_url="http://localhost:8000/v1",
        )
    
    def test_format_empty_tool_calls(self, judge):
        """Test formatting empty tool calls."""
        result = judge._format_tool_calls([])
        
        assert "No tool calls" in result
    
    def test_format_tool_calls_with_success(self, judge):
        """Test formatting successful tool calls."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5 kcal/mol", "success": True},
        ]
        
        result = judge._format_tool_calls(tool_calls)
        
        assert "vina_dock" in result
        assert "✓" in result
    
    def test_format_tool_calls_with_failure(self, judge):
        """Test formatting failed tool calls."""
        tool_calls = [
            {"name": "python_exec", "output": "Error: syntax error", "success": False},
        ]
        
        result = judge._format_tool_calls(tool_calls)
        
        assert "python_exec" in result
        assert "✗" in result
    
    def test_format_tool_calls_truncates_long_output(self, judge):
        """Test that long outputs are truncated."""
        long_output = "x" * 200
        tool_calls = [
            {"name": "test_tool", "output": long_output, "success": True},
        ]
        
        result = judge._format_tool_calls(tool_calls)
        
        assert "..." in result
        assert len(result) < len(long_output)


class TestJudgePromptBuilding:
    """Tests for judge prompt building."""
    
    @pytest.fixture
    def judge(self):
        """Create a judge."""
        prompts = JudgePrompts()
        return LLMJudge(
            prompts=prompts,
            model_url="http://localhost:8000/v1",
        )
    
    def test_build_prompt_includes_ground_truth(self, judge):
        """Test that prompt includes ground truth."""
        ground_truth = {
            "sabotage_type": "steric_clash",
            "winner_vina": -9.5,
            "loser_vina": -6.1,
            "delta_vina": 3.4,
            "modification_desc": "Added tert-butyl group",
        }
        
        prompt = judge._build_prompt([], {}, ground_truth)
        
        assert "steric_clash" in prompt
        assert "-9.5" in prompt
        assert "-6.1" in prompt
    
    def test_build_prompt_includes_tool_calls(self, judge):
        """Test that prompt includes tool call summary."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True},
        ]
        ground_truth = {"sabotage_type": "test"}
        
        prompt = judge._build_prompt(tool_calls, {}, ground_truth)
        
        assert "vina_dock" in prompt

    def test_build_prompt_includes_reasoning_traces(self, judge):
        """Test that reasoning traces are interleaved into the prompt."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True, "turn": 1},
            {"name": "python_execute", "output": "LogP=3.2", "success": True, "turn": 2},
        ]
        reasoning_traces = [
            {"turn": 1, "reasoning": "I should dock the ligand first to compare affinities."},
            {"turn": 2, "reasoning": "Now let me compute LogP to check lipophilicity."},
        ]
        ground_truth = {"sabotage_type": "steric_clash"}

        prompt = judge._build_prompt(
            tool_calls, "Final analysis complete.", ground_truth,
            reasoning_traces=reasoning_traces,
        )

        assert "dock the ligand first" in prompt
        assert "compute LogP" in prompt
        assert "vina_dock" in prompt

    def test_format_reasoning_trace_without_traces(self, judge):
        """Test formatting works when no reasoning traces are provided."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -7.0", "success": True, "turn": 1},
        ]
        result = judge._format_reasoning_trace(tool_calls, "Done.")
        assert "vina_dock" in result
        assert "Reasoning" not in result


class TestJudgeEndpointPool:
    """Tests for the multi-model JudgeEndpoint pool.

    Covers the new pool-mode path added by the scale-inference-18n-multijudge
    plan: random per-call endpoint selection, disabled filtering, retry on
    different endpoint, backward-compat fallback, per-call (model, api_key)
    headers, and YAML schema parsing via ``from_config``.
    """

    def _make_endpoints(self):
        return [
            JudgeEndpoint(
                name="alpha",
                url="https://alpha.test/v1",
                model="alpha-model",
                api_key="key-alpha",
                enabled=True,
            ),
            JudgeEndpoint(
                name="beta",
                url="https://beta.test/v1",
                model="beta-model",
                api_key="key-beta",
                enabled=True,
            ),
            JudgeEndpoint(
                name="gamma",
                url="https://gamma.test/v1",
                model="gamma-model",
                api_key="key-gamma",
                enabled=False,
            ),
        ]

    def test_pool_pick_round_robin_distribution(self):
        """Over many calls, picks should be approximately uniform over enabled endpoints."""
        judge = LLMJudge(prompts=JudgePrompts(), endpoints=self._make_endpoints())
        counts = {"alpha": 0, "beta": 0, "gamma": 0}
        for _ in range(2000):
            counts[judge._pick_endpoint().name] += 1
        assert counts["gamma"] == 0, "disabled endpoint must never be picked"
        # Allow +/- 20% deviation around 1000/1000 (uniform-random over 2 enabled)
        assert 800 <= counts["alpha"] <= 1200
        assert 800 <= counts["beta"] <= 1200

    def test_pool_disabled_filtered_at_construction(self):
        """Pool URL list and pool model_url should exclude disabled entries."""
        judge = LLMJudge(prompts=JudgePrompts(), endpoints=self._make_endpoints())
        assert "https://gamma.test/v1" not in judge.model_urls
        assert "gamma.test" not in judge.model_url

    def test_pool_all_disabled_raises(self):
        """Constructing with all-disabled endpoints must fail loudly."""
        eps = [JudgeEndpoint(name="x", url="u", model="m", enabled=False)]
        with pytest.raises(ValueError, match="at least one enabled endpoint"):
            LLMJudge(prompts=JudgePrompts(), endpoints=eps)

    def test_pool_backward_compat_legacy_single_endpoint(self):
        """No endpoints argument -> single-endpoint legacy mode still works."""
        judge = LLMJudge(
            prompts=JudgePrompts(),
            model_url="http://localhost:8000/v1",
            model_name="judge-legacy",
            api_key="legacy-key",
        )
        assert judge.endpoints == []
        assert judge.model_urls == ["http://localhost:8000/v1"]
        assert judge.model_name == "judge-legacy"
        assert judge.api_key == "legacy-key"

    def test_pool_from_config_parses_judge_models_list(self, tmp_path):
        """`from_config` builds JudgeEndpoint list from the new YAML schema."""
        import os
        import yaml

        cfg = {
            "model_url": "http://localhost:8000/v1",
            "model_name": "fallback-model",
            "judge_models": [
                {
                    "name": "nemotron_3_super",
                    "enabled": True,
                    "endpoint": "https://inference-api.nvidia.com/v1",
                    "model": "nvidia/nvidia/nemotron-3-super-v3",
                    "api_key_env": "TEST_NV_KEY",
                },
                {
                    "name": "claude_opus_4_6",
                    "enabled": False,
                    "endpoint": "https://inference-api.nvidia.com/v1",
                    "model": "aws/anthropic/bedrock-claude-opus-4-6",
                    "api_key_env": "TEST_NV_KEY",
                },
            ],
        }
        cfg_path = tmp_path / "judge.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        os.environ["TEST_NV_KEY"] = "sk-from-env"
        try:
            judge = LLMJudge.from_config(str(cfg_path))
        finally:
            os.environ.pop("TEST_NV_KEY", None)

        assert len(judge.endpoints) == 2
        assert judge.endpoints[0].name == "nemotron_3_super"
        assert judge.endpoints[0].enabled is True
        assert judge.endpoints[0].api_key == "sk-from-env"
        assert judge.endpoints[1].enabled is False

    def test_pool_from_config_model_url_override_forces_single(self, tmp_path):
        """Passing model_url override forces single-endpoint mode (local-vLLM path)."""
        import yaml

        cfg = {
            "model_name": "fallback-model",
            "judge_models": [
                {
                    "name": "x",
                    "enabled": True,
                    "endpoint": "https://inference-api.nvidia.com/v1",
                    "model": "some-model",
                },
            ],
        }
        cfg_path = tmp_path / "judge.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        judge = LLMJudge.from_config(
            str(cfg_path),
            model_url="http://10.0.0.5:9090/v1",
        )
        assert judge.endpoints == []
        assert judge.model_urls == ["http://10.0.0.5:9090/v1"]
        assert judge.model_name == "fallback-model"

    def test_pool_from_config_production_yaml_engages_pool(self):
        """Production training/configs/judge.yaml + model_url=None -> pool with all 6 enabled.

        Locks in the v0528a fix: when the wrapper passes model_url=None
        (the new default for JUDGE_NODES=0 runs), from_config must engage
        the 6-model pool, not fall back to single-endpoint mode. See
        internal_docs/docs/training_scaling_curve.md v0528a post-mortem.
        """
        from pathlib import Path

        # Walk up from the test file to find the repo root, then the prod yaml.
        # backend/api/training/tests/test_judge.py -> repo root is 4 levels up.
        repo_root = Path(__file__).resolve().parents[4]
        prod_yaml = repo_root / "training" / "configs" / "judge.yaml"
        if not prod_yaml.exists():
            pytest.skip(f"production judge yaml not found at {prod_yaml}")

        judge = LLMJudge.from_config(str(prod_yaml), model_url=None)

        assert len(judge.endpoints) == 7, (
            f"expected 7-model pool, got {len(judge.endpoints)} endpoints"
        )
        enabled = [ep for ep in judge.endpoints if ep.enabled]
        assert len(enabled) == 7, "all 7 production endpoints should ship enabled"
        # All entries currently target the NVIDIA Inference API (per judge.yaml).
        assert all(
            ep.url == "https://inference-api.nvidia.com/v1" for ep in enabled
        ), "production yaml should route every entry to the NVIDIA Inference API"
        # Names must be a stable set we can grep for in postmortems.
        expected_names = {
            "nemotron_3_super",
            "nemotron_3_ultra_preview",
            "qwen_3_6",
            "glm_5_1",
            "claude_opus_4_6",
            "gpt_5_5",
            "gemini_3_1",
        }
        assert {ep.name for ep in enabled} == expected_names

    def test_pool_from_config_production_yaml_with_url_override_stays_single(self):
        """Production yaml + explicit model_url -> single-endpoint (preserves JUDGE_NODES>0)."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        prod_yaml = repo_root / "training" / "configs" / "judge.yaml"
        if not prod_yaml.exists():
            pytest.skip(f"production judge yaml not found at {prod_yaml}")

        judge = LLMJudge.from_config(
            str(prod_yaml),
            model_url="http://10.52.50.99:9090/v1",
        )
        # The override must short-circuit the pool branch -- the JUDGE_NODES>0
        # local-vLLM contract relies on this.
        assert judge.endpoints == []
        assert judge.model_urls == ["http://10.52.50.99:9090/v1"]

    def test_pool_from_config_skips_incomplete_entries(self, tmp_path, caplog):
        """Entries missing name/endpoint/model are skipped with a warning."""
        import yaml

        cfg = {
            "judge_models": [
                {"name": "good", "endpoint": "https://x", "model": "m"},
                {"name": "no_url"},
                {"endpoint": "https://y", "model": "m2"},
                "not-a-dict",
            ],
        }
        cfg_path = tmp_path / "judge.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        with caplog.at_level("WARNING"):
            judge = LLMJudge.from_config(str(cfg_path))

        assert len(judge.endpoints) == 1
        assert judge.endpoints[0].name == "good"
        assert any("Skipping" in r.message for r in caplog.records)

    def test_pool_call_uses_per_endpoint_model_and_key(self):
        """_call_judge sends the picked endpoint's model name + api_key in headers."""
        import asyncio
        from unittest.mock import patch

        endpoints = [
            JudgeEndpoint(
                name="only",
                url="https://only.test/v1",
                model="only-model",
                api_key="key-only",
                enabled=True,
            ),
        ]
        judge = LLMJudge(prompts=JudgePrompts(), endpoints=endpoints, timeout=5.0)

        captured = {}

        class FakeResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "LOGICAL_CONSISTENCY: 3\n"
                                    "DIFFERENTIATING_METRIC: 3\n"
                                    "HYPOTHESIS_GROUNDED: 3\n"
                                    "SCIENTIFIC_ACCURACY: 3\n"
                                    "EDIT_TESTS_HYPOTHESIS: 3\n"
                                    "DIAGNOSTIC_QUALITY: 1\n"
                                    "VERDICT: [[SOUND]]"
                                ),
                                "reasoning_content": "",
                            }
                        }
                    ]
                }

            def raise_for_status(self):
                return None

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers, json):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return FakeResponse()

        with patch("api.training.judge.httpx.AsyncClient", FakeClient):
            result = asyncio.new_event_loop().run_until_complete(
                judge._call_judge("test prompt")
            )

        assert result.total_score == 16
        assert captured["url"] == "https://only.test/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer key-only"
        assert captured["json"]["model"] == "only-model"

    def test_pool_retry_picks_fresh_endpoint(self, monkeypatch):
        """When the first attempt returns 502, retry should pick from pool again (likely a different one)."""
        import asyncio
        from unittest.mock import patch

        endpoints = [
            JudgeEndpoint(
                name="bad",
                url="https://bad.test/v1",
                model="m-bad",
                api_key="k-bad",
                enabled=True,
            ),
            JudgeEndpoint(
                name="good",
                url="https://good.test/v1",
                model="m-good",
                api_key="k-good",
                enabled=True,
            ),
        ]
        judge = LLMJudge(prompts=JudgePrompts(), endpoints=endpoints, timeout=1.0)

        seq = iter(["bad", "good"])

        def fake_pick():
            name = next(seq)
            return next(ep for ep in endpoints if ep.name == name)

        monkeypatch.setattr(judge, "_pick_endpoint", fake_pick)

        import asyncio as _asyncio

        async def fast_sleep(*a, **kw):
            return None

        monkeypatch.setattr(_asyncio, "sleep", fast_sleep)

        call_log = []

        class FakeResponse:
            def __init__(self, code, body=None):
                self.status_code = code
                self.headers = {"retry-after": "0"}
                self._body = body

            def json(self):
                return self._body

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers, json):
                call_log.append((url, json["model"]))
                if "bad" in url:
                    return FakeResponse(502)
                return FakeResponse(
                    200,
                    body={
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "LOGICAL_CONSISTENCY: 2\n"
                                        "DIFFERENTIATING_METRIC: 2\n"
                                        "HYPOTHESIS_GROUNDED: 2\n"
                                        "SCIENTIFIC_ACCURACY: 2\n"
                                        "EDIT_TESTS_HYPOTHESIS: 2\n"
                                        "DIAGNOSTIC_QUALITY: 1\n"
                                        "VERDICT: [[SOUND]]"
                                    ),
                                    "reasoning_content": "",
                                }
                            }
                        ]
                    },
                )

        with patch("api.training.judge.httpx.AsyncClient", FakeClient):
            result = asyncio.new_event_loop().run_until_complete(
                judge._call_judge("test prompt")
            )

        assert result.total_score == 11
        assert len(call_log) == 2
        assert "bad" in call_log[0][0]
        assert call_log[0][1] == "m-bad"
        assert "good" in call_log[1][0]
        assert call_log[1][1] == "m-good"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
