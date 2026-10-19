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

"""Conversation controller for advanced mode training.

This module handles the hypothesis -> fix flow in advanced mode training:
1. Detect when the model forms a hypothesis
2. Inject the "prove it by editing the molecule" prompt
3. Track molecule modification and redocking attempts
4. Compute advanced mode rewards

The hypothesis detection patterns and injection prompts are configurable.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_HYPOTHESIS_PATTERNS = [
    r"I\s+have\s+a\s+hypothesis",
    r"I\s+believe\s+(?:the\s+)?(?:reason|cause)\s+is",
    r"My\s+(?:hypothesis|diagnosis)\s+is",
    r"The\s+(?:likely|probable)\s+cause\s+is",
    r"Based\s+on\s+(?:the|my)\s+analysis[,.]?\s+(?:I\s+(?:think|believe)|the\s+(?:reason|cause))",
    r"I\s+conclude\s+that",
    r"(?:In\s+)?conclusion[,:]",
    r"The\s+(?:main|primary|key)\s+(?:difference|issue|problem)\s+(?:is|appears\s+to\s+be)",
]

DEFAULT_INJECTION_PROMPT = """Good hypothesis! Now PROVE it by editing the molecule and re-docking.

Modify the loser molecule to address the defect you identified. Make a real chemical change motivated by your hypothesis -- do NOT simply copy or regenerate the winner molecule (a Tanimoto-similarity check vs the winner will detect that and apply a penalty).

**Target**: Achieve at least {target_percentage}% of the winner's binding affinity ({target_affinity:.2f} kcal/mol).

**Steps**:
1. Use `python_execute` with RDKit to modify the loser molecule's structure
2. Generate new 3D coordinates for the modified molecule
3. Store the new molecule with `aevom.store_sdf(...)` to obtain a new sdf reference
4. Dock the modified molecule using `vina_dock` with the new reference
5. Compare the new binding affinity to the winner and explain whether the result confirms or refutes your hypothesis
"""

# Patterns for detecting molecule modification attempts
MODIFICATION_PATTERNS = [
    r"(?:Chem\.)?RWMol",
    r"EditableMol",
    r"ReplaceAtom",
    r"RemoveAtom",
    r"AddAtom",
    r"ReplaceSidechains",
    r"ReplaceCore",
    r"DeleteSubstructs",
    r"AllChem\.(?:Replace|Remove|Add)",
    r"Chem\.(?:ReplaceSubstructs|DeleteSubstructs)",
    r"mol\.(?:RemoveAtom|AddAtom|ReplaceAtom)",
]


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AdvancedModeConfig:
    """Configuration for advanced mode training."""

    # Hypothesis detection
    hypothesis_patterns: list[str] = field(default_factory=lambda: DEFAULT_HYPOTHESIS_PATTERNS)
    hypothesis_confidence_threshold: float = 0.7

    # Injection settings
    injection_prompt: str = DEFAULT_INJECTION_PROMPT
    target_recovery_fraction: float = 0.80  # 80% of winner affinity
    max_fix_attempts: int = 3

    # Edit-similarity guards (advanced mode)
    edit_max_winner_sim: float = 0.99  # above this = trivial copy of winner
    edit_min_loser_sim: float = 0.40   # below this = drifted too far from loser

    # Modification detection
    modification_patterns: list[str] = field(default_factory=lambda: MODIFICATION_PATTERNS)

    # Rewards
    hypothesis_reward: float = 0.1
    modification_reward: float = 0.5
    redock_success_reward: float = 0.5
    affinity_recovered_reward: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "AdvancedModeConfig":
        """Load configuration from YAML file."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Advanced mode config not found: {path}, using defaults")
            return cls()

        with open(path_obj) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            hypothesis_patterns=data.get("hypothesis_patterns", DEFAULT_HYPOTHESIS_PATTERNS),
            hypothesis_confidence_threshold=data.get("hypothesis_confidence_threshold", 0.7),
            injection_prompt=data.get("injection_prompt", DEFAULT_INJECTION_PROMPT),
            target_recovery_fraction=data.get("target_recovery_fraction", 0.80),
            max_fix_attempts=data.get("max_fix_attempts", 3),
            edit_max_winner_sim=data.get("edit_max_winner_sim", 0.99),
            edit_min_loser_sim=data.get("edit_min_loser_sim", 0.40),
            modification_patterns=data.get("modification_patterns", MODIFICATION_PATTERNS),
            hypothesis_reward=data.get("hypothesis_reward", 0.1),
            modification_reward=data.get("modification_reward", 0.5),
            redock_success_reward=data.get("redock_success_reward", 0.5),
            affinity_recovered_reward=data.get("affinity_recovered_reward", 1.0),
        )


@dataclass
class HypothesisDetectionResult:
    """Result of hypothesis detection."""
    
    detected: bool = False
    confidence: float = 0.0
    pattern_matched: Optional[str] = None
    hypothesis_text: str = ""
    position: int = -1  # Position in response where hypothesis was found


@dataclass
class ModificationDetectionResult:
    """Result of molecule modification detection."""
    
    detected: bool = False
    patterns_matched: list[str] = field(default_factory=list)
    modification_code: str = ""


@dataclass
class AdvancedModeState:
    """State tracking for advanced mode in an episode."""
    
    # Hypothesis tracking
    hypothesis_detected: bool = False
    hypothesis_text: str = ""
    hypothesis_turn: int = -1
    injection_done: bool = False
    injection_turn: int = -1
    
    # Modification tracking
    modification_attempts: int = 0
    modification_detected: bool = False
    modified_molecule_ref: Optional[str] = None
    modification_code: str = ""
    
    # Redocking tracking
    redock_attempts: int = 0
    redock_success: bool = False
    redock_affinity: Optional[float] = None
    
    # Ground truth for comparison
    winner_affinity: float = 0.0
    loser_affinity: float = 0.0
    target_affinity: float = 0.0


# =============================================================================
# Conversation Controller
# =============================================================================

class ConversationController:
    """Controls conversation flow for advanced mode training.
    
    This class manages the hypothesis -> fix -> verify flow:
    1. Monitors agent responses for hypothesis markers
    2. Injects the "prove it" prompt when hypothesis detected
    3. Tracks modification attempts
    4. Evaluates redocking results
    
    Example:
        controller = ConversationController.from_config("training/configs/advanced_mode.yaml")
        
        # Check for hypothesis in response
        detection = controller.detect_hypothesis(response_text)
        if detection.detected and not state.injection_done:
            injection = controller.get_injection_prompt(ground_truth)
            # Add injection to conversation
            state.injection_done = True
        
        # Check for modification in tool calls
        for call in tool_calls:
            if call["name"] == "python_exec":
                mod = controller.detect_modification(call["arguments"]["code"])
                if mod.detected:
                    state.modification_detected = True
    """
    
    def __init__(self, config: AdvancedModeConfig):
        """Initialize the controller.
        
        Args:
            config: Advanced mode configuration.
        """
        self.config = config
        self._compiled_hypothesis_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in config.hypothesis_patterns
        ]
        self._compiled_modification_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in config.modification_patterns
        ]
    
    @classmethod
    def from_config(cls, config_path: str) -> "ConversationController":
        """Create from YAML configuration file.
        
        Args:
            config_path: Path to YAML config file.
            
        Returns:
            Configured ConversationController instance.
        """
        config = AdvancedModeConfig.from_yaml(config_path)
        return cls(config)
    
    def create_state(self, ground_truth: dict) -> AdvancedModeState:
        """Create initial state for an episode.
        
        Args:
            ground_truth: Ground truth with winner/loser affinities.
            
        Returns:
            Initialized AdvancedModeState.
        """
        winner_affinity = ground_truth.get("winner_vina", 0.0)
        loser_affinity = ground_truth.get("loser_vina", 0.0)
        
        # Calculate target (fraction of winner's binding)
        # Vina scores are negative, so target should be more negative
        target_affinity = winner_affinity * self.config.target_recovery_fraction
        
        return AdvancedModeState(
            winner_affinity=winner_affinity,
            loser_affinity=loser_affinity,
            target_affinity=target_affinity,
        )
    
    def detect_hypothesis(self, text: str) -> HypothesisDetectionResult:
        """Detect if text contains a hypothesis statement.
        
        Args:
            text: Response text to analyze.
            
        Returns:
            HypothesisDetectionResult with detection details.
        """
        result = HypothesisDetectionResult()
        
        for i, pattern in enumerate(self._compiled_hypothesis_patterns):
            match = pattern.search(text)
            if match:
                result.detected = True
                result.pattern_matched = self.config.hypothesis_patterns[i]
                result.position = match.start()
                
                # Extract hypothesis context (surrounding text)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 200)
                result.hypothesis_text = text[start:end].strip()
                
                # Confidence based on pattern strength and context
                # Patterns earlier in the list are considered stronger
                result.confidence = 1.0 - (i * 0.1)
                
                break
        
        return result
    
    def detect_modification(self, code: str) -> ModificationDetectionResult:
        """Detect if code contains molecule modification operations.
        
        Args:
            code: Python code to analyze.
            
        Returns:
            ModificationDetectionResult with detection details.
        """
        result = ModificationDetectionResult()
        result.modification_code = code
        
        for i, pattern in enumerate(self._compiled_modification_patterns):
            if pattern.search(code):
                result.detected = True
                result.patterns_matched.append(
                    self.config.modification_patterns[i]
                )
        
        return result
    
    def get_injection_prompt(
        self,
        ground_truth: dict,
        state: Optional[AdvancedModeState] = None,
    ) -> str:
        """Get the prompt to inject after hypothesis detection.
        
        Args:
            ground_truth: Ground truth information.
            state: Optional current state.
            
        Returns:
            Formatted injection prompt.
        """
        winner_affinity = ground_truth.get("winner_vina", 0.0)
        target_affinity = winner_affinity * self.config.target_recovery_fraction
        target_percentage = int(self.config.target_recovery_fraction * 100)
        
        return self.config.injection_prompt.format(
            target_percentage=target_percentage,
            target_affinity=target_affinity,
            winner_affinity=winner_affinity,
            loser_affinity=ground_truth.get("loser_vina", 0.0),
            sabotage_type=ground_truth.get("sabotage_type", "unknown"),
        )
    
    def check_redock_success(
        self,
        redock_affinity: float,
        state: AdvancedModeState,
    ) -> dict:
        """Check if redocking achieved target affinity.
        
        Args:
            redock_affinity: Affinity from redocking.
            state: Current episode state.
            
        Returns:
            Dictionary with success details.
        """
        # Vina scores are negative, lower is better
        # Success means getting close to (or better than) target
        target_reached = redock_affinity <= state.target_affinity
        
        # Calculate recovery ratio
        # How close did we get to winner's affinity from loser's?
        loser_winner_diff = state.loser_affinity - state.winner_affinity
        if abs(loser_winner_diff) > 0.01:
            improvement = state.loser_affinity - redock_affinity
            recovery_ratio = improvement / loser_winner_diff
        else:
            recovery_ratio = 1.0 if target_reached else 0.0
        
        return {
            "success": target_reached,
            "redock_affinity": redock_affinity,
            "target_affinity": state.target_affinity,
            "winner_affinity": state.winner_affinity,
            "loser_affinity": state.loser_affinity,
            "recovery_ratio": min(1.0, max(0.0, recovery_ratio)),
            "improvement": state.loser_affinity - redock_affinity,
        }
    
    def process_turn(
        self,
        response_text: str,
        tool_calls: list[dict],
        state: AdvancedModeState,
        turn_number: int,
    ) -> dict:
        """Process a turn in the conversation.
        
        Args:
            response_text: Model's response text.
            tool_calls: Tool calls made in this turn.
            state: Current episode state.
            turn_number: Current turn number.
            
        Returns:
            Dictionary with processing results and any actions to take.
        """
        result = {
            "should_inject": False,
            "injection_prompt": None,
            "hypothesis_detected": False,
            "modification_detected": False,
            "redock_detected": False,
            "redock_result": None,
        }
        
        # Check for hypothesis if not already detected
        if not state.hypothesis_detected:
            detection = self.detect_hypothesis(response_text)
            if detection.detected and detection.confidence >= self.config.hypothesis_confidence_threshold:
                state.hypothesis_detected = True
                state.hypothesis_text = detection.hypothesis_text
                state.hypothesis_turn = turn_number
                result["hypothesis_detected"] = True
                
                # Prepare injection if not already done
                if not state.injection_done:
                    result["should_inject"] = True
                    # Note: injection_prompt will need ground_truth to format
        
        # Check tool calls for modification and redocking
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            output = call.get("output", "")
            success = call.get("success", True)
            
            # Check for modification in python_exec
            if name in ("python_exec", "python_execute"):
                code = args.get("code", "")
                mod_detection = self.detect_modification(code)
                if mod_detection.detected:
                    state.modification_attempts += 1
                    state.modification_detected = True
                    state.modification_code = code
                    result["modification_detected"] = True
            
            # Check for redocking
            elif name in ("vina_dock", "dock_ligand"):
                if state.modification_detected:
                    state.redock_attempts += 1
                    result["redock_detected"] = True
                    
                    # Try to extract affinity from output
                    if success:
                        affinity = self._extract_affinity(output)
                        if affinity is not None:
                            state.redock_affinity = affinity
                            state.redock_success = True
                            result["redock_result"] = self.check_redock_success(
                                affinity, state
                            )
        
        return result
    
    def _extract_affinity(self, output: str) -> Optional[float]:
        """Extract affinity value from docking output.
        
        Args:
            output: Docking tool output.
            
        Returns:
            Affinity value or None if not found.
        """
        patterns = [
            r"affinity[:\s]+(-?\d+\.?\d*)",
            r"score[:\s]+(-?\d+\.?\d*)",
            r"(-?\d+\.?\d*)\s*kcal",
            r"binding\s+energy[:\s]+(-?\d+\.?\d*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def compute_advanced_rewards(
        self,
        state: AdvancedModeState,
    ) -> dict:
        """Compute advanced mode rewards from state.
        
        Args:
            state: Final episode state.
            
        Returns:
            Dictionary with reward breakdown.
        """
        rewards = {
            "hypothesis_reward": 0.0,
            "modification_reward": 0.0,
            "redock_reward": 0.0,
            "affinity_reward": 0.0,
            "total": 0.0,
        }
        
        if state.hypothesis_detected:
            rewards["hypothesis_reward"] = self.config.hypothesis_reward
        
        if state.modification_detected:
            rewards["modification_reward"] = self.config.modification_reward
        
        if state.redock_success:
            rewards["redock_reward"] = self.config.redock_success_reward
            
            # Check if affinity target was reached
            if state.redock_affinity is not None:
                if state.redock_affinity <= state.target_affinity:
                    rewards["affinity_reward"] = self.config.affinity_recovered_reward
        
        rewards["total"] = sum([
            rewards["hypothesis_reward"],
            rewards["modification_reward"],
            rewards["redock_reward"],
            rewards["affinity_reward"],
        ])
        
        return rewards


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_hypothesis_in_text(
    text: str,
    patterns: Optional[list[str]] = None,
) -> HypothesisDetectionResult:
    """Convenience function to detect hypothesis.
    
    Args:
        text: Text to analyze.
        patterns: Optional custom patterns.
        
    Returns:
        Detection result.
    """
    config = AdvancedModeConfig()
    if patterns:
        config.hypothesis_patterns = patterns
    
    controller = ConversationController(config)
    return controller.detect_hypothesis(text)


def detect_modification_in_code(
    code: str,
    patterns: Optional[list[str]] = None,
) -> ModificationDetectionResult:
    """Convenience function to detect modification.
    
    Args:
        code: Python code to analyze.
        patterns: Optional custom patterns.
        
    Returns:
        Detection result.
    """
    config = AdvancedModeConfig()
    if patterns:
        config.modification_patterns = patterns
    
    controller = ConversationController(config)
    return controller.detect_modification(code)
