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

"""Training infrastructure for Aevom.

This module provides infrastructure for both DPO (Direct Preference Optimization)
training data collection and GRPO (Group Relative Policy Optimization) RL training.

DPO Training:
    - PromptMode, get_training_prompt: Interactive prompt comparison and voting

GRPO Training (NeMo-Gym Integration):
    - CDDDatasetAdapter: Convert aevom-data-gen to NeMo-Gym format
    - TrainingDataPreloader: Load molecules into storage at episode start
    - TieredRewardComputer: Multi-tier reward computation
    - LLMJudge: Reasoning quality evaluation
    - ConversationController: Advanced mode hypothesis->fix flow

The NeMo-Gym resource server lives at training/nemo_gym_wrapper/app.py.
"""

# DPO training components
from api.training.prompts import (
    PromptMode,
    get_training_prompt,
    get_prompt_modes,
    PROMPT_DESCRIPTIONS,
)

# GRPO training components
from api.training.dataset_adapter import (
    CDDDatasetAdapter,
    MoleculePair,
    TrainingTask,
    convert_dataset,
)

from api.training.data_preloader import (
    TrainingDataPreloader,
    PreloadResult,
    MockStorageService,
    load_training_task,
    iter_training_tasks,
    preload_task_data,
)

from api.training.rewards import (
    TieredRewardComputer,
    ToolUsageStats,
    ToolUsageDetector,
    detect_sabotage_type_in_text,
    detect_differentiation,
    compute_reward,
    DEFAULT_REWARD_CONFIG,
)

from api.training.tool_detector import (
    TrajectoryAnalyzer,
    TrajectoryAnalysis,
    PythonCodeAnalysis,
    DockingAnalysis,
    analyze_tool_calls,
    extract_docking_results,
    extract_computed_metrics,
)

from api.training.judge import (
    LLMJudge,
    JudgePrompts,
    JudgeResult,
    MockJudge,
    evaluate_reasoning,
)

from api.training.conversation_controller import (
    ConversationController,
    AdvancedModeConfig,
    AdvancedModeState,
    HypothesisDetectionResult,
    ModificationDetectionResult,
    detect_hypothesis_in_text,
    detect_modification_in_code,
)

__all__ = [
    # DPO
    "PromptMode",
    "get_training_prompt",
    "get_prompt_modes",
    "PROMPT_DESCRIPTIONS",
    # Dataset
    "CDDDatasetAdapter",
    "MoleculePair",
    "TrainingTask",
    "convert_dataset",
    # Preloader
    "TrainingDataPreloader",
    "PreloadResult",
    "MockStorageService",
    "load_training_task",
    "iter_training_tasks",
    "preload_task_data",
    # Rewards
    "TieredRewardComputer",
    "ToolUsageStats",
    "ToolUsageDetector",
    "detect_sabotage_type_in_text",
    "detect_differentiation",
    "compute_reward",
    "DEFAULT_REWARD_CONFIG",
    # Tool Detection
    "TrajectoryAnalyzer",
    "TrajectoryAnalysis",
    "PythonCodeAnalysis",
    "DockingAnalysis",
    "analyze_tool_calls",
    "extract_docking_results",
    "extract_computed_metrics",
    # Judge
    "LLMJudge",
    "JudgePrompts",
    "JudgeResult",
    "MockJudge",
    "evaluate_reasoning",
    # Advanced Mode
    "ConversationController",
    "AdvancedModeConfig",
    "AdvancedModeState",
    "HypothesisDetectionResult",
    "ModificationDetectionResult",
    "detect_hypothesis_in_text",
    "detect_modification_in_code",
]
