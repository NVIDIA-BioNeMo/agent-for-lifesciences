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

"""Unit tests for the conversation controller (advanced mode)."""

import pytest
from api.training.conversation_controller import (
    ConversationController,
    AdvancedModeConfig,
    AdvancedModeState,
    HypothesisDetectionResult,
    ModificationDetectionResult,
    detect_hypothesis_in_text,
    detect_modification_in_code,
)


class TestHypothesisDetection:
    """Tests for hypothesis detection."""
    
    @pytest.fixture
    def controller(self):
        """Create a conversation controller with default config."""
        return ConversationController(AdvancedModeConfig())
    
    def test_detect_hypothesis_basic(self, controller):
        """Test basic hypothesis detection."""
        text = "I have a hypothesis about why the binding is poor."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is True
        assert result.confidence > 0
    
    def test_detect_hypothesis_believe(self, controller):
        """Test hypothesis detection with 'I believe'."""
        text = "I believe the reason is a steric clash."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is True
    
    def test_detect_hypothesis_diagnosis(self, controller):
        """Test hypothesis detection with 'My diagnosis is'."""
        text = "My diagnosis is that the tert-butyl group causes steric hindrance."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is True
    
    def test_detect_hypothesis_conclude(self, controller):
        """Test hypothesis detection with 'I conclude that'."""
        text = "Based on the docking scores, I conclude that the charge flip is responsible."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is True
    
    def test_detect_hypothesis_in_conclusion(self, controller):
        """Test hypothesis detection with 'In conclusion'."""
        text = "In conclusion, the main difference is the loss of a hydrogen bond."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is True
    
    def test_no_hypothesis_in_question(self, controller):
        """Test that questions don't trigger hypothesis detection."""
        text = "Let me dock the molecules to see what the binding scores are."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is False
    
    def test_no_hypothesis_in_tool_description(self, controller):
        """Test that tool descriptions don't trigger hypothesis detection."""
        text = "I will use vina_dock to compute the binding affinity."
        result = controller.detect_hypothesis(text)
        
        assert result.detected is False
    
    def test_convenience_function(self):
        """Test the convenience function for hypothesis detection."""
        result = detect_hypothesis_in_text("My hypothesis is that...")
        
        assert result.detected is True


class TestModificationDetection:
    """Tests for molecule modification detection."""
    
    @pytest.fixture
    def controller(self):
        """Create a conversation controller with default config."""
        return ConversationController(AdvancedModeConfig())
    
    def test_detect_rwmol(self, controller):
        """Test detection of RWMol usage."""
        code = """
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
rwmol = Chem.RWMol(mol)
rwmol.RemoveAtom(0)
"""
        result = controller.detect_modification(code)
        
        assert result.detected is True
        assert len(result.patterns_matched) > 0
    
    def test_detect_remove_atom(self, controller):
        """Test detection of RemoveAtom."""
        code = "mol.RemoveAtom(5)"
        result = controller.detect_modification(code)
        
        assert result.detected is True
    
    def test_detect_add_atom(self, controller):
        """Test detection of AddAtom."""
        code = "rwmol.AddAtom(Chem.Atom(6))"
        result = controller.detect_modification(code)
        
        assert result.detected is True
    
    def test_detect_replace_substructs(self, controller):
        """Test detection of ReplaceSubstructs."""
        code = "new_mol = Chem.ReplaceSubstructs(mol, patt, repl)"
        result = controller.detect_modification(code)
        
        assert result.detected is True
    
    def test_detect_delete_substructs(self, controller):
        """Test detection of DeleteSubstructs."""
        code = "new_mol = Chem.DeleteSubstructs(mol, patt)"
        result = controller.detect_modification(code)
        
        assert result.detected is True
    
    def test_no_modification_in_analysis(self, controller):
        """Test that analysis code doesn't trigger modification detection."""
        code = """
from rdkit import Chem
from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles('CCO')
logp = Descriptors.MolLogP(mol)
print(logp)
"""
        result = controller.detect_modification(code)
        
        assert result.detected is False
    
    def test_convenience_function(self):
        """Test the convenience function for modification detection."""
        result = detect_modification_in_code("mol.RemoveAtom(0)")
        
        assert result.detected is True


class TestAdvancedModeState:
    """Tests for AdvancedModeState management."""
    
    @pytest.fixture
    def controller(self):
        """Create a conversation controller."""
        return ConversationController(AdvancedModeConfig())
    
    def test_create_state(self, controller):
        """Test state creation from ground truth."""
        ground_truth = {
            "winner_vina": -9.5,
            "loser_vina": -6.1,
        }
        
        state = controller.create_state(ground_truth)
        
        assert state.winner_affinity == -9.5
        assert state.loser_affinity == -6.1
        assert state.target_affinity == -9.5 * 0.80  # 80% of winner
        assert state.hypothesis_detected is False
        assert state.modification_detected is False
    
    def test_injection_prompt_formatting(self, controller):
        """Test injection prompt formatting."""
        ground_truth = {
            "winner_vina": -10.0,
            "loser_vina": -5.0,
            "sabotage_type": "steric_clash",
        }
        
        prompt = controller.get_injection_prompt(ground_truth)
        
        assert "80%" in prompt
        assert "-8.00" in prompt  # 80% of -10.0
        assert "prove it by editing" in prompt.lower()


class TestRedockingEvaluation:
    """Tests for redocking success evaluation."""
    
    @pytest.fixture
    def controller(self):
        """Create a conversation controller."""
        return ConversationController(AdvancedModeConfig())
    
    def test_redock_success_target_reached(self, controller):
        """Test redocking success when target is reached."""
        state = AdvancedModeState(
            winner_affinity=-10.0,
            loser_affinity=-5.0,
            target_affinity=-8.0,  # 80% of winner
        )
        
        # Modified molecule reaches target
        result = controller.check_redock_success(-8.5, state)
        
        assert result["success"] is True
        assert result["recovery_ratio"] > 0
    
    def test_redock_failure_target_not_reached(self, controller):
        """Test redocking failure when target is not reached."""
        state = AdvancedModeState(
            winner_affinity=-10.0,
            loser_affinity=-5.0,
            target_affinity=-8.0,
        )
        
        # Modified molecule doesn't reach target
        result = controller.check_redock_success(-6.0, state)
        
        assert result["success"] is False
    
    def test_redock_exceeds_winner(self, controller):
        """Test redocking when modified molecule exceeds winner."""
        state = AdvancedModeState(
            winner_affinity=-10.0,
            loser_affinity=-5.0,
            target_affinity=-8.0,
        )
        
        # Modified molecule is better than winner
        result = controller.check_redock_success(-11.0, state)
        
        assert result["success"] is True
        assert result["recovery_ratio"] >= 1.0


class TestAdvancedModeRewards:
    """Tests for advanced mode reward computation."""
    
    @pytest.fixture
    def controller(self):
        """Create a conversation controller."""
        return ConversationController(AdvancedModeConfig())
    
    def test_no_rewards_initial_state(self, controller):
        """Test no rewards for initial state."""
        state = AdvancedModeState()
        
        rewards = controller.compute_advanced_rewards(state)
        
        assert rewards["total"] == 0
    
    def test_hypothesis_reward(self, controller):
        """Test reward for hypothesis detection."""
        state = AdvancedModeState(
            hypothesis_detected=True,
        )
        
        rewards = controller.compute_advanced_rewards(state)
        
        assert rewards["hypothesis_reward"] > 0
    
    def test_modification_reward(self, controller):
        """Test reward for molecule modification."""
        state = AdvancedModeState(
            modification_detected=True,
        )
        
        rewards = controller.compute_advanced_rewards(state)
        
        assert rewards["modification_reward"] > 0
    
    def test_full_success_rewards(self, controller):
        """Test full rewards for complete success."""
        state = AdvancedModeState(
            hypothesis_detected=True,
            modification_detected=True,
            redock_success=True,
            redock_affinity=-9.0,
            target_affinity=-8.0,
        )
        
        rewards = controller.compute_advanced_rewards(state)
        
        assert rewards["hypothesis_reward"] > 0
        assert rewards["modification_reward"] > 0
        assert rewards["redock_reward"] > 0
        assert rewards["affinity_reward"] > 0
        assert rewards["total"] > 2.0  # Should be substantial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
