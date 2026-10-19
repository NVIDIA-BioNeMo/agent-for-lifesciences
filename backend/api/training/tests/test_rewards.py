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

"""Unit tests for the tiered reward system."""

import pytest
from api.training.rewards import (
    TieredRewardComputer,
    ToolUsageDetector,
    ToolUsageStats,
    TOOL_REWARD_SCHEDULE,
    build_reward_annotations,
    compute_tool_schedule_reward,
    detect_sabotage_type_in_text,
    detect_differentiation,
    DEFAULT_REWARD_CONFIG,
)


class TestToolUsageDetector:
    """Tests for ToolUsageDetector."""
    
    def test_detect_dock_attempts(self):
        """Test detection of docking tool calls."""
        detector = ToolUsageDetector()
        
        tool_calls = [
            {"name": "vina_dock", "arguments": {"ligand": "sdf_001"}, "output": "Affinity: -8.5 kcal/mol", "success": True},
            {"name": "vina_dock", "arguments": {"ligand": "sdf_002"}, "output": "Affinity: -6.2 kcal/mol", "success": True},
        ]
        
        stats = detector.analyze_trajectory(tool_calls)
        
        assert stats.dock_attempts == 2
        assert stats.dock_successes == 2
        assert len(stats.dock_results) == 2
    
    def test_detect_python_exec_success(self):
        """Test detection of successful Python execution."""
        detector = ToolUsageDetector()
        
        tool_calls = [
            {"name": "python_exec", "arguments": {"code": "print('hello')"}, "output": "hello", "success": True},
            {"name": "python_exec", "arguments": {"code": "x = 1/0"}, "output": "Error: division by zero", "success": False},
        ]
        
        stats = detector.analyze_trajectory(tool_calls)
        
        assert stats.python_exec_calls == 2
        assert stats.python_exec_successes == 1
        assert len(stats.python_exec_errors) == 1
    
    def test_detect_rdkit_usage(self):
        """Test detection of RDKit usage in Python code."""
        detector = ToolUsageDetector()
        
        tool_calls = [
            {
                "name": "python_exec",
                "arguments": {"code": "from rdkit import Chem\nmol = Chem.MolFromSmiles('CCO')"},
                "output": "",
                "success": True,
            },
        ]
        
        stats = detector.analyze_trajectory(tool_calls)
        
        assert stats.rdkit_usage is True
        assert "MolFromSmiles" in stats.rdkit_functions_used
    
    def test_detect_rdkit_descriptors(self):
        """Test detection of RDKit descriptor imports."""
        detector = ToolUsageDetector()
        
        tool_calls = [
            {
                "name": "python_exec",
                "arguments": {"code": "from rdkit.Chem import Descriptors\nlogp = Descriptors.MolLogP(mol)"},
                "output": "LogP: 1.23",
                "success": True,
            },
        ]
        
        stats = detector.analyze_trajectory(tool_calls)
        
        assert stats.rdkit_usage is True
        assert "MolLogP" in stats.rdkit_functions_used
    
    def test_detect_metrics_computed(self):
        """Test detection of computed metrics."""
        detector = ToolUsageDetector()
        
        tool_calls = [
            {
                "name": "python_exec",
                "arguments": {"code": "logp = calculate_logp(mol)\ntpsa = calculate_tpsa(mol)"},
                "output": "LogP: 2.5, TPSA: 40.0",
                "success": True,
            },
        ]
        
        stats = detector.analyze_trajectory(tool_calls)
        
        assert "logp" in stats.metrics_computed or "tpsa" in stats.metrics_computed
    
    def test_no_rdkit_in_plain_python(self):
        """Test that plain Python doesn't trigger RDKit detection."""
        detector = ToolUsageDetector()
        
        tool_calls = [
            {
                "name": "python_exec",
                "arguments": {"code": "x = 1 + 2\nprint(x)"},
                "output": "3",
                "success": True,
            },
        ]
        
        stats = detector.analyze_trajectory(tool_calls)
        
        assert stats.rdkit_usage is False


class TestSabotageDetection:
    """Tests for sabotage type detection."""
    
    def test_detect_steric_clash(self):
        """Test detection of steric clash mentions."""
        text = "The binding is poor due to a steric clash with the bulky tert-butyl group."
        assert detect_sabotage_type_in_text(text) == "steric_clash"
    
    def test_detect_electrostatic_repulsion(self):
        """Test detection of electrostatic repulsion mentions."""
        text = "The electrostatic repulsion disrupts the binding interaction."
        assert detect_sabotage_type_in_text(text) == "electrostatic_repulsion"
    
    def test_detect_loss_of_hbond(self):
        """Test detection of hydrogen bond loss mentions."""
        text = "The hydrogen bond donor was removed, breaking a key interaction."
        assert detect_sabotage_type_in_text(text) == "loss_of_hbond"
    
    def test_detect_poor_solubility(self):
        """Test detection of poor solubility."""
        text = "The molecule has poor solubility due to excessive lipophilicity."
        assert detect_sabotage_type_in_text(text) == "poor_solubility"
    
    def test_no_sabotage_detected(self):
        """Test that irrelevant text returns None."""
        text = "The weather is nice today."
        assert detect_sabotage_type_in_text(text) is None


class TestDifferentiationDetection:
    """Tests for differentiation detection."""
    
    def test_detect_differentiation_with_two_docks(self):
        """Test differentiation detection with two successful docks."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -9.5 kcal/mol", "success": True},
            {"name": "vina_dock", "output": "Affinity: -6.1 kcal/mol", "success": True},
        ]
        response_text = "The winner binds better due to steric clash in the loser."
        ground_truth = {"sabotage_type": "steric_clash"}
        
        result = detect_differentiation(tool_calls, response_text, ground_truth)
        
        assert result["differentiated"] is True
        assert result["sabotage_detected"] == "steric_clash"
        assert result["sabotage_correct"] is True
    
    def test_detect_wrong_sabotage_type(self):
        """Test detection of incorrect sabotage type."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -9.5 kcal/mol", "success": True},
            {"name": "vina_dock", "output": "Affinity: -6.1 kcal/mol", "success": True},
        ]
        response_text = "I believe the issue is a charge flip."
        ground_truth = {"sabotage_type": "steric_clash"}
        
        result = detect_differentiation(tool_calls, response_text, ground_truth)
        
        assert result["sabotage_detected"] == "electrostatic_repulsion"
        assert result["sabotage_correct"] is False
    
    def test_no_differentiation_without_docking(self):
        """Test that differentiation requires docking."""
        tool_calls = [
            {"name": "python_exec", "output": "some output", "success": True},
        ]
        response_text = "The molecules are different."
        ground_truth = {"sabotage_type": "steric_clash"}
        
        result = detect_differentiation(tool_calls, response_text, ground_truth)
        
        assert result["differentiated"] is False


class TestTieredRewardComputer:
    """Tests for TieredRewardComputer."""
    
    @pytest.fixture
    def computer(self):
        """Create a reward computer with default config."""
        return TieredRewardComputer(DEFAULT_REWARD_CONFIG)
    
    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def test_tier1_dock_reward(self, computer):
        """Test tier 1 reward for docking."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True},
        ]
        ground_truth = {"sabotage_type": "steric_clash"}

        result = self._run(computer.compute(
            tool_calls=tool_calls,
            ground_truth=ground_truth,
            response={},
        ))

        assert result["tier1_reward"] > 0
        assert result["tier1_details"]["dock_successes"] == 1

    def test_tier1_rdkit_reward(self, computer):
        """Test tier 1 reward for RDKit usage."""
        tool_calls = [
            {
                "name": "python_exec",
                "arguments": {"code": "from rdkit import Chem"},
                "output": "",
                "success": True,
            },
        ]
        ground_truth = {"sabotage_type": "steric_clash"}

        result = self._run(computer.compute(
            tool_calls=tool_calls,
            ground_truth=ground_truth,
            response={},
        ))

        assert "rdkit_reward" in result["tier1_details"]
        assert result["tier1_details"]["rdkit_reward"] > 0

    def test_tier1_max_cap(self, computer):
        """Test that tier 1 rewards are capped."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True}
            for _ in range(20)
        ]
        ground_truth = {"sabotage_type": "steric_clash"}

        result = self._run(computer.compute(
            tool_calls=tool_calls,
            ground_truth=ground_truth,
            response={},
        ))

        max_tier1 = computer.config["tier1"]["max_tier1_reward"]
        assert result["tier1_reward"] <= max_tier1

    def test_tier3_hypothesis_reward(self, computer):
        """Test tier 3 reward for hypothesis detection."""
        tool_calls = []
        ground_truth = {"sabotage_type": "steric_clash"}

        result = self._run(computer.compute(
            tool_calls=tool_calls,
            ground_truth=ground_truth,
            response={},
            hypothesis_detected=True,
        ))

        assert result["tier3_reward"] > 0
        assert result["tier3_details"]["hypothesis_detected"] is True

    def test_total_reward_sum(self, computer):
        """Test that total reward is sum of tiers."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True},
            {"name": "vina_dock", "output": "Affinity: -6.1", "success": True},
        ]
        ground_truth = {"sabotage_type": "steric_clash"}

        result = self._run(computer.compute(
            tool_calls=tool_calls,
            ground_truth=ground_truth,
            response={},
        ))

        expected_total = (
            result["tier1_reward"] +
            result["tier2_reward"] +
            result["tier3_reward"]
        )
        assert abs(result["total_reward"] - expected_total) < 0.001


class TestToolScheduleReward:
    """Tests for the v0601a per-tool, per-call Tier-1 schedule."""

    def test_single_successful_dock(self):
        tier1, breakdown = compute_tool_schedule_reward(
            [{"tool_name": "vina_dock", "success": True}]
        )
        # attempt 0.02 + success 0.02 = 0.04
        assert tier1 == pytest.approx(0.04)
        assert breakdown["vina_dock"]["num_calls"] == 1
        assert breakdown["vina_dock"]["num_successful"] == 1
        assert breakdown["vina_dock"]["capped"] == pytest.approx(0.04)

    def test_failed_call_earns_attempt_only(self):
        tier1, breakdown = compute_tool_schedule_reward(
            [{"tool_name": "python_execute", "success": False}]
        )
        # attempt 0.06, no success
        assert tier1 == pytest.approx(0.06)
        assert breakdown["python_execute"]["num_successful"] == 0

    def test_dock_cap_binds(self):
        # 3 successful docks -> raw 0.12, capped at 0.08
        calls = [{"tool_name": "vina_dock", "success": True} for _ in range(3)]
        tier1, breakdown = compute_tool_schedule_reward(calls)
        assert breakdown["vina_dock"]["raw"] == pytest.approx(0.12)
        assert breakdown["vina_dock"]["capped"] == pytest.approx(0.08)
        assert tier1 == pytest.approx(0.08)

    def test_pyexec_cap_binds(self):
        # 5 successful pyexec -> raw 0.60, capped at 0.48
        calls = [{"tool_name": "python_execute", "success": True} for _ in range(5)]
        tier1, _ = compute_tool_schedule_reward(calls)
        assert tier1 == pytest.approx(0.48)

    def test_visualize_pdb_attempt_only(self):
        calls = [{"tool_name": "visualize_pdb", "success": True} for _ in range(3)]
        tier1, breakdown = compute_tool_schedule_reward(calls)
        # attempt 0.02 each, no success reward, capped at 0.04
        assert breakdown["visualize_pdb"]["num_successful"] == 3
        assert tier1 == pytest.approx(0.04)

    def test_unscheduled_tool_earns_zero(self):
        tier1, breakdown = compute_tool_schedule_reward(
            [{"tool_name": "search_pubchem", "success": True}]
        )
        assert tier1 == 0.0
        assert breakdown == {}

    def test_mixed_tools_sum(self):
        # 1 ok dock (0.04) + 2 ok pyexec (0.24) = 0.28
        calls = [
            {"tool_name": "vina_dock", "success": True},
            {"tool_name": "python_execute", "success": True},
            {"tool_name": "python_execute", "success": True},
        ]
        tier1, _ = compute_tool_schedule_reward(calls)
        assert tier1 == pytest.approx(0.28)

    def test_no_calls_zero(self):
        tier1, breakdown = compute_tool_schedule_reward([])
        assert tier1 == 0.0
        assert breakdown == {}

    def test_effective_max_equals_sum_of_caps(self):
        assert sum(s["cap"] for s in TOOL_REWARD_SCHEDULE.values()) == pytest.approx(1.0)


class TestPerCallAnnotations:
    """Tests for the generalized per-call annotation path."""

    def test_attempt_and_success_coexist_on_same_span(self):
        per_tool_call_rewards = [
            {
                "tc_index": 0,
                "tool_name": "vina_dock",
                "span_id": "abc123",
                "label": "tool_call_attempt",
                "value": 0.02,
            },
            {
                "tc_index": 0,
                "tool_name": "vina_dock",
                "span_id": "abc123",
                "label": "tool_call_success",
                "value": 0.02,
            },
        ]
        annotations = build_reward_annotations(
            tool_calls=[{"tool_name": "vina_dock", "success": True}],
            tool_span_ids=[("abc123", "vina_dock", 0)],
            root_span_id="root",
            awarded_rewards=[],
            per_tool_call_rewards=per_tool_call_rewards,
        )
        per_call = [a for a in annotations if a["span_id"] == "abc123"]
        assert len(per_call) == 2
        labels = {a["label"] for a in per_call}
        assert labels == {"tool_call_attempt", "tool_call_success"}
        identifiers = {a["identifier"] for a in per_call}
        assert identifiers == {"tool_call_attempt:0", "tool_call_success:0"}
        assert sum(a["score"] for a in per_call) == pytest.approx(0.04)

    def test_fallback_to_root_when_no_span(self):
        annotations = build_reward_annotations(
            tool_calls=[{"tool_name": "python_execute", "success": True}],
            tool_span_ids=[],
            root_span_id="root",
            awarded_rewards=[],
            per_tool_call_rewards=[
                {
                    "tc_index": 0,
                    "tool_name": "python_execute",
                    "span_id": "",
                    "label": "tool_call_attempt",
                    "value": 0.06,
                }
            ],
        )
        rec = [a for a in annotations if a["label"] == "tool_call_attempt"][0]
        assert rec["span_id"] == "root"
        assert rec["metadata"]["note"] == "no matching tool span found"
        assert rec["metadata"]["tool_name"] == "python_execute"


class TestSingleRootAnnotation:
    """The production verify() path emits ONLY the root summed reward.

    Per-tool-call / Tier-2-3 catalog / judge_reasoning annotations were removed
    to kill the Phoenix span_annotations 404 flood; the reward detail survives
    as span attributes + episodes.jsonl. These tests pin the verify() call
    shape so the single-annotation contract cannot silently regress.
    """

    def test_only_total_effective_reward_emitted(self):
        annotations = build_reward_annotations(
            tool_calls=[
                {"tool_name": "vina_dock", "success": True},
                {"tool_name": "python_execute", "success": True},
            ],
            tool_span_ids=[("abc123", "vina_dock", 0), ("def456", "python_execute", 1)],
            root_span_id="root",
            awarded_rewards=[],
            episode_id="ep-1",
            total_reward=0.42,
            reward_breakdown={"tier1_tool_usage": 0.42},
            judge_result=None,
        )
        assert len(annotations) == 1
        ann = annotations[0]
        assert ann["label"] == "total_effective_reward"
        assert ann["span_id"] == "root"
        assert ann["score"] == pytest.approx(0.42)
        assert ann["metadata"]["breakdown"] == {"tier1_tool_usage": 0.42}

    def test_no_annotations_without_total_reward(self):
        # Defensive: with no extra sources and no total_reward, nothing is emitted.
        annotations = build_reward_annotations(
            tool_calls=[{"tool_name": "vina_dock", "success": True}],
            tool_span_ids=[("abc123", "vina_dock", 0)],
            root_span_id="root",
            awarded_rewards=[],
            judge_result=None,
        )
        assert annotations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
