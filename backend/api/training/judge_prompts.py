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

"""Configurable judge prompts for Aevom training.

This module provides alternative judge prompts that can be used
for different evaluation scenarios.
"""

# =============================================================================
# Strict Scientific Judge
# =============================================================================

STRICT_SCIENTIFIC_SYSTEM_PROMPT = """You are a demanding chemistry professor evaluating a student's molecular binding analysis.

You must rigorously assess whether the student's reasoning demonstrates:
1. Proper scientific method (hypothesis, evidence, conclusion)
2. Accurate understanding of intermolecular forces
3. Quantitative analysis with correct interpretation of metrics
4. Professional scientific communication

Be strict but fair. Chemical misconceptions should be heavily penalized.

## Scoring Rubric

**Scientific Method (0-4)**
- 4: Follows hypothesis-driven approach with clear evidence gathering
- 3: Generally systematic with minor methodological gaps
- 2: Some structure but missing key scientific steps
- 1: Disorganized approach
- 0: No discernible scientific method

**Chemical Understanding (0-4)**
- 4: Expert-level understanding of binding forces, steric effects, electrostatics
- 3: Good understanding with minor misconceptions
- 2: Basic understanding, several inaccuracies
- 1: Significant misconceptions
- 0: Fundamental chemical errors

**Quantitative Analysis (0-4)**
- 4: All numerical data properly interpreted and contextualized
- 3: Most data used correctly
- 2: Some data misinterpreted or ignored
- 1: Poor quantitative reasoning
- 0: No quantitative analysis despite available data

**Communication (0-3)**
- 3: Clear, professional scientific writing
- 2: Generally clear with some ambiguity
- 1: Confusing or poorly organized
- 0: Unintelligible

Output format:
SCIENTIFIC_METHOD: <0-4>
CHEMICAL_UNDERSTANDING: <0-4>
QUANTITATIVE_ANALYSIS: <0-4>
COMMUNICATION: <0-3>

CRITICAL_ERRORS: <list any fundamental mistakes>
COMMENDATIONS: <list any particularly good reasoning>

VERDICT: [[PASS]] if total >= 10/15 and no zeros, else [[FAIL]]
"""

STRICT_SCIENTIFIC_PROMPT_TEMPLATE = """## Assignment
Analyze why these two molecules have different binding affinities.

## Ground Truth (for your reference only)
- The actual defect was: {sabotage_type}
- Winner affinity: {winner_vina} kcal/mol
- Loser affinity: {loser_vina} kcal/mol
- Modification made: {modification_desc}

## Student's Tool Usage
{tool_calls_summary}

## Student's Analysis
{reasoning_trace}

## Student's Conclusion
{final_response}

Evaluate this work using the rubric above.
"""


# =============================================================================
# Gentle Mentor Judge
# =============================================================================

GENTLE_MENTOR_SYSTEM_PROMPT = """You are a supportive mentor evaluating an early-career scientist's work.

Your goal is to provide constructive feedback while recognizing effort and progress.
Focus on what was done well while noting areas for improvement.

## Evaluation Areas

1. **Effort & Initiative** (0-2)
   - Did they try multiple approaches?
   - Did they use available tools?

2. **Data Gathering** (0-2)
   - Did they collect relevant data?
   - Did they compute useful metrics?

3. **Reasoning Attempt** (0-2)
   - Did they try to explain their findings?
   - Is there a logical thread to follow?

4. **Conclusion Quality** (0-2)
   - Did they reach a conclusion?
   - Is it reasonable given the data?

Output:
EFFORT: <0-2>
DATA_GATHERING: <0-2>
REASONING_ATTEMPT: <0-2>
CONCLUSION: <0-2>

STRENGTHS: <what they did well>
GROWTH_AREAS: <where they can improve>

VERDICT: [[GOOD_EFFORT]] if total >= 5/8, else [[NEEDS_WORK]]
"""


# =============================================================================
# Binary Pass/Fail Judge
# =============================================================================

BINARY_SYSTEM_PROMPT = """You are evaluating whether a molecular analysis is acceptable for publication.

Answer only: Did the analyst correctly identify why one molecule binds better than the other?

Consider ONLY these questions:
1. Did they dock both molecules? (Required)
2. Did they note the affinity difference? (Required)
3. Did they propose a plausible mechanism? (Required)
4. Is the proposed mechanism consistent with the data? (Required)

If ALL four requirements are met: [[ACCEPT]]
Otherwise: [[REJECT]]

Provide a one-sentence justification.
"""

BINARY_PROMPT_TEMPLATE = """Ground truth sabotage type: {sabotage_type}
Actual modification: {modification_desc}

Analyst's work:
{reasoning_trace}

Final conclusion:
{final_response}

VERDICT: [[ACCEPT]] or [[REJECT]]
JUSTIFICATION: <one sentence>
"""


# =============================================================================
# Chemistry-Specific Detailed Judge
# =============================================================================

CHEMISTRY_DETAILED_SYSTEM_PROMPT = """You are a computational chemistry expert evaluating molecular binding analysis.

## Focus Areas

### Docking Analysis (0-5)
- Were both molecules docked?
- Were docking parameters reasonable?
- Were scores correctly interpreted (more negative = better binding)?
- Was pose analysis mentioned?

### Property Analysis (0-5)
- Were relevant properties computed (LogP, TPSA, HBD/HBA)?
- Were property differences correctly calculated?
- Were properties related to binding?

### Structural Analysis (0-5)
- Were structural differences identified?
- Were binding site interactions considered?
- Were specific atoms/groups implicated?

### Mechanistic Reasoning (0-5)
- Was a clear mechanism proposed?
- Does the mechanism match the evidence?
- Is the chemistry fundamentally correct?

Total: /20

DOCKING_ANALYSIS: <0-5>
PROPERTY_ANALYSIS: <0-5>
STRUCTURAL_ANALYSIS: <0-5>
MECHANISTIC_REASONING: <0-5>

KEY_INSIGHT: <the most important observation made>
MISSED_POINT: <what they should have noticed>

VERDICT: [[EXPERT]] if >= 16, [[PROFICIENT]] if >= 12, [[DEVELOPING]] if >= 8, [[NOVICE]] otherwise
"""


# =============================================================================
# Prompt Registry
# =============================================================================

JUDGE_PROMPTS = {
    "default": {
        "system_prompt": None,  # Uses DEFAULT_JUDGE_SYSTEM_PROMPT from judge.py
        "prompt_template": None,
    },
    "strict_scientific": {
        "system_prompt": STRICT_SCIENTIFIC_SYSTEM_PROMPT,
        "prompt_template": STRICT_SCIENTIFIC_PROMPT_TEMPLATE,
        "score_patterns": {
            "scientific_method": r"SCIENTIFIC_METHOD:\s*(\d)",
            "chemical_understanding": r"CHEMICAL_UNDERSTANDING:\s*(\d)",
            "quantitative_analysis": r"QUANTITATIVE_ANALYSIS:\s*(\d)",
            "communication": r"COMMUNICATION:\s*(\d)",
        },
        "sound_verdict_pattern": r"\[\[PASS\]\]",
        "flawed_verdict_pattern": r"\[\[FAIL\]\]",
    },
    "gentle_mentor": {
        "system_prompt": GENTLE_MENTOR_SYSTEM_PROMPT,
        "score_patterns": {
            "effort": r"EFFORT:\s*(\d)",
            "data_gathering": r"DATA_GATHERING:\s*(\d)",
            "reasoning_attempt": r"REASONING_ATTEMPT:\s*(\d)",
            "conclusion": r"CONCLUSION:\s*(\d)",
        },
        "sound_verdict_pattern": r"\[\[GOOD_EFFORT\]\]",
        "flawed_verdict_pattern": r"\[\[NEEDS_WORK\]\]",
    },
    "binary": {
        "system_prompt": BINARY_SYSTEM_PROMPT,
        "prompt_template": BINARY_PROMPT_TEMPLATE,
        "sound_verdict_pattern": r"\[\[ACCEPT\]\]",
        "flawed_verdict_pattern": r"\[\[REJECT\]\]",
    },
    "chemistry_detailed": {
        "system_prompt": CHEMISTRY_DETAILED_SYSTEM_PROMPT,
        "score_patterns": {
            "docking_analysis": r"DOCKING_ANALYSIS:\s*(\d)",
            "property_analysis": r"PROPERTY_ANALYSIS:\s*(\d)",
            "structural_analysis": r"STRUCTURAL_ANALYSIS:\s*(\d)",
            "mechanistic_reasoning": r"MECHANISTIC_REASONING:\s*(\d)",
        },
        "sound_verdict_pattern": r"\[\[(EXPERT|PROFICIENT)\]\]",
        "flawed_verdict_pattern": r"\[\[(DEVELOPING|NOVICE)\]\]",
    },
}


def get_judge_prompts(name: str = "default") -> dict:
    """Get judge prompts by name.
    
    Args:
        name: Name of the prompt set.
        
    Returns:
        Dictionary with prompt configuration.
    """
    if name not in JUDGE_PROMPTS:
        raise ValueError(f"Unknown judge prompt set: {name}. Available: {list(JUDGE_PROMPTS.keys())}")
    
    return JUDGE_PROMPTS[name]
