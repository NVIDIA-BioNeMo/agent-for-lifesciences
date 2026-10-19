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

"""Tiered reward computation for Aevom training.

This module implements the multi-tier reward structure:
- Tier 1: Tool usage rewards (docking, python_exec, RDKit, metrics)
- Tier 2: Differentiation rewards (correctly identifying binding difference)
- Tier 3: Advanced mode rewards (hypothesis, molecule modification, redocking)

All reward weights are configurable via YAML configuration.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Default Reward Configuration
# =============================================================================

DEFAULT_REWARD_CONFIG = {
    # Tier 1: Tool Usage (per-step, accumulated)
    "tier1": {
        "dock_attempt": 0.05,  # Called vina_dock
        "python_exec_success": 0.05,  # python_exec without error
        "rdkit_usage": 0.03,  # Used RDKit in python_exec
        "metric_computed": 0.05,  # Computed LogP, TPSA, etc.
        "differentiating_metric_computed": 0.10,  # Paired metric that differs
        "metric_noise_threshold": 0.05,
        "max_tier1_reward": 0.60,  # Cap on tier 1 rewards
    },
    # Tier 2: Differentiation (episode end)
    "tier2": {
        "correct_differentiation": 0.30,  # Identified score difference
        "correct_sabotage_type": 0.20,  # Named correct defect type
        "partial_diagnosis": 0.10,  # Partially correct reasoning
    },
    # Tier 3: Advanced Mode (hypothesis -> edit -> redock)
    "tier3": {
        "hypothesis_with_metric": 0.20,
        "non_trivial_edit": 0.50,
        "edit_redock_success": 0.50,
        "affinity_recovered": 1.00,
        "trivial_edit_penalty": -0.50,
        "edit_max_winner_sim": 0.99,
        "edit_min_loser_sim": 0.40,
        "target_recovery_fraction": 0.80,
        # Legacy keys retained for backwards compatibility with existing tests
        # and the non-wired TieredRewardComputer code path.
        "molecule_modified": 0.50,
        "redock_success": 0.50,
    },
    # Judge weight (judge reward is computed separately)
    "judge_weight": 0.30,
}


# =============================================================================
# Reward Catalog -- single source of truth for reward names and values
# =============================================================================


@dataclass(frozen=True)
class RewardEntry:
    """One entry in the reward catalog.

    Attributes:
        value: Reward magnitude awarded when this condition is met.
        tier: Which tier this reward belongs to (1=tool usage, 2=differentiation, 3=fix).
        description: Human-readable explanation of the reward trigger.
        tool_match: Tool names that trigger this reward and should receive the
            annotation.  ``None`` means the reward is episode-level and should
            be annotated on the root ``chat_request`` span.
    """

    value: float
    tier: int
    description: str = ""
    tool_match: Optional[frozenset[str]] = None


REWARD_CATALOG: dict[str, RewardEntry] = {
    # ------------------------------------------------------------------
    # Tier 1: Tool usage  (per-step, accumulated, capped at TIER1_MAX_REWARD)
    # ------------------------------------------------------------------
    # Hard floor reward for using ANY tool successfully. Awarded once PER
    # successful tool call and annotated on that call's own tool.{name} span
    # (see build_reward_annotations' per_tool_call_rewards path). tool_match is
    # left None because the reward is not tied to a specific tool name; the
    # per-call annotation targeting is handled explicitly by the caller.
    "any_tool_usage": RewardEntry(
        value=0.02,
        tier=1,
        description="Successfully called any tool (per-call floor reward)",
        tool_match=None,
    ),
    "successful_dock": RewardEntry(
        value=0.05,
        tier=1,
        description="Called vina_dock and received a valid affinity score",
        tool_match=frozenset({"vina_dock"}),
    ),
    "python_exec_success": RewardEntry(
        value=0.05,
        tier=1,
        description="Ran python_execute without error",
        tool_match=frozenset({"python_execute"}),
    ),
    "rdkit_usage": RewardEntry(
        value=0.03,
        tier=1,
        description="Used RDKit libraries in Python code",
        tool_match=frozenset({"python_execute"}),
    ),
    "metric_computed": RewardEntry(
        value=0.05,
        tier=1,
        description="Computed a molecular property (LogP, TPSA, QED, etc.)",
        tool_match=frozenset({"python_execute"}),
    ),
    "differentiating_metric_computed": RewardEntry(
        value=0.10,
        tier=1,
        description=(
            "Computed at least one paired metric for winner & loser whose "
            "absolute difference exceeds the noise threshold"
        ),
        tool_match=frozenset({"python_execute"}),
    ),
    # ------------------------------------------------------------------
    # Tier 2: Differentiation  (episode-level, annotated on root span)
    # ------------------------------------------------------------------
    "correct_sabotage_id": RewardEntry(
        value=0.5,
        tier=2,
        description="Correctly identified the sabotage type",
        tool_match=None,
    ),
    "partial_sabotage_id": RewardEntry(
        value=0.3,
        tier=2,
        description="Detected some defect type, even if not the exact label",
        tool_match=None,
    ),
    # ------------------------------------------------------------------
    # Tier 3: Molecule fix  (advanced mode)
    # ------------------------------------------------------------------
    "hypothesis_with_metric": RewardEntry(
        value=0.20,
        tier=3,
        description=(
            "Stated a hypothesis that explicitly references a metric value "
            "previously computed in paired form (winner vs loser)"
        ),
        tool_match=None,
    ),
    "non_trivial_edit": RewardEntry(
        value=0.50,
        tier=3,
        description=(
            "Created a new molecule that is similar to the loser but is "
            "NOT a copy of the winner (Tanimoto guards: edit~loser >= "
            "edit_min_loser_sim AND edit~winner <= edit_max_winner_sim)"
        ),
        tool_match=None,
    ),
    "edit_redock_success": RewardEntry(
        value=0.50,
        tier=3,
        description="Successfully docked the edited molecule with vina_dock",
        tool_match=frozenset({"vina_dock"}),
    ),
    "affinity_recovered": RewardEntry(
        value=1.00,
        tier=3,
        description=(
            "Edit's docking affinity reached the recovery threshold "
            "(<= target_recovery_fraction * winner_vina)"
        ),
        tool_match=None,
    ),
    "trivial_edit_penalty": RewardEntry(
        value=-0.50,
        tier=3,
        description=(
            "Edit is a trivial copy of the winner "
            "(Tanimoto(edit, winner) > edit_max_winner_sim). "
            "Total reward is floored at 0 so this cannot push reward negative."
        ),
        tool_match=None,
    ),
}

TIER1_MAX_REWARD = 0.60


# =============================================================================
# Per-tool Tier-1 reward schedule (v0601a)  --  single source of truth
# =============================================================================
#
# Tier-1 is computed as a per-tool, per-call schedule rather than episode-level
# binary flags.  For each scheduled tool the agent earns ``attempt`` for every
# call (success or fail) plus ``success`` for each successful call, with the
# accumulated per-tool total clamped by that tool's ``cap``.  There is NO global
# Tier-1 cap -- the per-tool caps bound it (effective Tier-1 max = sum of caps =
# 1.00).  The training agent is advertised exactly these 5 tools
# (get_training_tool_schemas()); any other/unknown tool name earns 0.
#
# This restores within-group reward variance (the count-sensitive per-call
# accumulation differs across trajectories that touch the same tools), which the
# previous episode-level binary flags collapsed to zero.
TOOL_REWARD_SCHEDULE: dict[str, dict[str, float]] = {
    "vina_dock": {"attempt": 0.02, "success": 0.02, "cap": 0.08},
    "molecule_info": {"attempt": 0.02, "success": 0.02, "cap": 0.08},
    "structure_query": {"attempt": 0.04, "success": 0.04, "cap": 0.32},
    "python_execute": {"attempt": 0.06, "success": 0.06, "cap": 0.48},
    "visualize_pdb": {"attempt": 0.02, "success": 0.00, "cap": 0.04},
}


def compute_tool_schedule_reward(
    tool_calls: list[dict],
    schedule: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute the per-tool, per-call Tier-1 reward.

    For each scheduled tool, accumulate ``attempt`` for every call plus
    ``success`` for each successful call, then clamp the per-tool total by that
    tool's ``cap``.  Unscheduled tools contribute nothing.  There is no global
    cap; the returned ``tier1`` is the sum of the per-tool capped contributions.

    Args:
        tool_calls: Episode tool-call records (each with ``tool_name``/``name``
            and ``success``).
        schedule: Override the default ``TOOL_REWARD_SCHEDULE`` (mainly for
            tests).

    Returns:
        Tuple of ``(tier1_reward, per_tool_breakdown)`` where
        ``per_tool_breakdown`` maps each tool name seen to a dict with
        ``num_calls``, ``num_successful``, ``raw`` (pre-cap) and ``capped``.
    """
    sched = schedule or TOOL_REWARD_SCHEDULE
    breakdown: dict[str, dict[str, float]] = {}

    for tc in tool_calls:
        name = tc.get("tool_name", tc.get("name", ""))
        spec = sched.get(name)
        if spec is None:
            continue  # unscheduled tools earn nothing
        success = bool(tc.get("success", True))

        entry = breakdown.setdefault(
            name, {"num_calls": 0, "num_successful": 0, "raw": 0.0, "capped": 0.0}
        )
        entry["num_calls"] += 1
        call_reward = spec["attempt"]
        if success:
            entry["num_successful"] += 1
            call_reward += spec["success"]
        entry["raw"] += call_reward

    tier1 = 0.0
    for name, entry in breakdown.items():
        capped = min(entry["raw"], sched[name]["cap"])
        entry["capped"] = capped
        tier1 += capped

    return tier1, breakdown


def compute_tier1_reward(
    tool_calls: list[dict],
    catalog: Optional[dict[str, "RewardEntry"]] = None,
    max_reward: float = TIER1_MAX_REWARD,
) -> tuple[float, list[str]]:
    """Compute per-step accumulated tier-1 reward from tool calls.

    Walks each tool call and awards:
    - ``successful_dock`` per successful vina_dock call
    - ``python_exec_success`` per error-free python_execute call
    - ``rdkit_usage`` once per episode if RDKit imports are detected
    - ``metric_computed`` once per distinct metric keyword found

    Returns:
        Tuple of (capped_reward, list_of_awarded_reward_names).
    """
    cat = catalog or REWARD_CATALOG
    reward = 0.0
    awarded: list[str] = []
    rdkit_awarded = False
    metrics_awarded: set[str] = set()

    for tc in tool_calls:
        name = tc.get("tool_name", tc.get("name", ""))
        success = tc.get("success", True)
        output = str(tc.get("output", ""))

        if name in ("vina_dock", "dock_ligand"):
            if success and "error" not in output.lower()[:100]:
                reward += cat.get("successful_dock", REWARD_CATALOG["successful_dock"]).value
                awarded.append("successful_dock")

        elif name in ("python_execute", "python_exec", "execute_python"):
            if success and "error" not in output.lower()[:100]:
                reward += cat.get("python_exec_success", REWARD_CATALOG["python_exec_success"]).value
                awarded.append("python_exec_success")

            code = tc.get("arguments", {}).get("code", "")
            if not rdkit_awarded and code:
                for pattern in RDKIT_PATTERNS:
                    if re.search(pattern, code, re.IGNORECASE):
                        reward += cat.get("rdkit_usage", REWARD_CATALOG["rdkit_usage"]).value
                        awarded.append("rdkit_usage")
                        rdkit_awarded = True
                        break

            combined = f"{code} {output}".lower()
            for kw in METRIC_KEYWORDS:
                if kw not in metrics_awarded and kw in combined:
                    reward += cat.get("metric_computed", REWARD_CATALOG["metric_computed"]).value
                    awarded.append("metric_computed")
                    metrics_awarded.add(kw)

    capped = min(reward, max_reward)
    return capped, awarded


# =============================================================================
# Tool Usage Detection
# =============================================================================


@dataclass
class ToolUsageStats:
    """Statistics about tool usage in an episode."""

    # Docking
    dock_attempts: int = 0
    dock_successes: int = 0
    dock_results: list[dict] = field(default_factory=list)

    # Python execution
    python_exec_calls: int = 0
    python_exec_successes: int = 0
    python_exec_errors: list[str] = field(default_factory=list)

    # RDKit usage
    rdkit_usage: bool = False
    rdkit_functions_used: set[str] = field(default_factory=set)

    # Metrics
    metrics_computed: set[str] = field(default_factory=set)
    metric_values: dict[str, float] = field(default_factory=dict)

    # Other tools
    load_molecule_calls: int = 0
    other_tool_calls: int = 0


# RDKit import patterns to detect
RDKIT_PATTERNS = [
    r"from\s+rdkit",
    r"import\s+rdkit",
    r"from\s+Chem\s+import",
    r"from\s+rdkit\.Chem",
    r"Chem\.",
    r"AllChem\.",
    r"Descriptors\.",
]

# Metric keywords to detect
METRIC_KEYWORDS = [
    "logp",
    "log_p",
    "logp",
    "tpsa",
    "molecular_weight",
    "mw",
    "hbd",
    "h_bond_donor",
    "hbond_donor",
    "hba",
    "h_bond_acceptor",
    "hbond_acceptor",
    "num_rotatable_bonds",
    "rotatable_bonds",
    "qed",
    "sasa",
    "solvent_accessible",
    "fsp3",
    "num_rings",
    "ring_count",
]

# Sabotage type detection patterns -- keys MUST match dataset sabotage_type values
# (steric_clash, electrostatic_repulsion, loss_of_hbond, poor_solubility, poor_permeability).
SABOTAGE_TYPE_PATTERNS: dict[str, list[str]] = {
    "steric_clash": [
        r"steric\s*clash",
        r"steric\s*hindrance",
        r"steric\s*strain",
        r"steric\s*repulsion",
        r"bulky\s+group",
        r"bulkier",
        r"too\s+large",
        r"overcrowd",
        r"van\s+der\s+waals.*clash",
        r"size.*pocket",
    ],
    "electrostatic_repulsion": [
        r"electrostatic\s*repulsion",
        r"charge\s*(flip|change|reversal|inversion|mismatch)",
        r"salt\s+bridge.*lost",
        r"lost.*salt\s+bridge",
        r"ionic\s*(repulsion|interaction)",
        r"formal\s+charge.*(change|differ|flip|revers)",
        r"(change|differ|flip|revers).*formal\s+charge",
        r"opposite\s+charge",
        r"same[- ]charge\s*repulsion",
        r"charge\s*complementarity",
        r"charge\s*repulsion",
    ],
    "loss_of_hbond": [
        r"hydrogen\s*bond.*loss",
        r"loss.*hydrogen\s*bond",
        r"h[- ]?bond.*lost",
        r"lost.*h[- ]?bond",
        r"hbond.*remov",
        r"remov.*hbond",
        r"hydrogen\s*bond\s*donor",
        r"hydrogen\s*bond\s*acceptor",
        r"h[- ]?bond\s*donor",
        r"h[- ]?bond\s*acceptor",
        r"fewer.*h[- ]?bond",
        r"reduced.*h[- ]?bond",
        r"hydroxyl.*replaced",
        r"replaced.*hydroxyl",
        r"extra.*h[- ]?bond\s*(donor|acceptor)",
        r"additional.*h[- ]?bond",
        r"lacks?.*polar\s*contact",
        r"lacks?.*h[- ]?bond",
        r"missing.*hydrogen",
        r"(hbd|hba).*(differ|fewer|less|more|extra|additional)",
        r"(differ|fewer|less|more|extra|additional).*(hbd|hba)",
        r"polar\s*contact.*(key|critical|driv|explain|account|responsible)",
    ],
    "poor_solubility": [
        r"poor\s*solubility",
        r"insolub",
        r"hydrophobic.*excess",
        r"too\s*(hydrophobic|lipophilic)",
        r"logp.*(high|increas|exceed|above\s+[56])",
        r"(high|excessive)\s*logp",
        r"lipophili",
        r"lipinski.*violat",
    ],
    "poor_permeability": [
        r"poor\s*permeability",
        r"impermeab",
        r"(high|excessive)\s*tpsa",
        r"tpsa.*(high|increas|exceed|above\s+1[45]0)",
        r"polar\s+surface\s+area.*(high|excessive|too)",
        r"too\s*(polar|hydrophilic)",
        r"membrane.*permeab",
    ],
}

# Legacy aliases for backwards compatibility with older reward configs
_SABOTAGE_TYPE_ALIASES: dict[str, str] = {
    "charge_flip": "electrostatic_repulsion",
    "h_bond_loss": "loss_of_hbond",
    "hbond_loss": "loss_of_hbond",
    "hydrophobic_mismatch": "poor_solubility",
}


class ToolUsageDetector:
    """Detects and categorizes tool usage from trajectory."""

    def analyze_trajectory(self, tool_calls: list[dict]) -> ToolUsageStats:
        """Analyze a list of tool calls to extract usage statistics.

        Args:
            tool_calls: List of tool call records with name, arguments, output.

        Returns:
            ToolUsageStats with detected usage patterns.
        """
        stats = ToolUsageStats()

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            output = call.get("output", "")
            success = call.get("success", True)

            # Docking calls
            if name in ("vina_dock", "dock_ligand", "docking"):
                stats.dock_attempts += 1
                if success and "error" not in output.lower():
                    stats.dock_successes += 1
                    # Try to extract affinity
                    stats.dock_results.append(
                        {
                            "arguments": args,
                            "output": output,
                            "success": success,
                        }
                    )

            # Python execution
            elif name in ("python_exec", "python_execute", "execute_python"):
                stats.python_exec_calls += 1

                if success and "error" not in output.lower()[:100]:
                    stats.python_exec_successes += 1
                else:
                    stats.python_exec_errors.append(output[:200])

                # Check for RDKit usage in code
                code = args.get("code", "")
                if self._detect_rdkit_usage(code):
                    stats.rdkit_usage = True
                    stats.rdkit_functions_used.update(
                        self._extract_rdkit_functions(code)
                    )

                # Check for metric computation
                metrics = self._detect_metrics(code, output)
                stats.metrics_computed.update(metrics)

            # Metric computation
            elif name in ("compute_metrics", "calculate_properties"):
                metrics_requested = args.get("metrics", [])
                stats.metrics_computed.update(metrics_requested)

            # Load molecule
            elif name in ("load_molecule", "load_sdf"):
                stats.load_molecule_calls += 1

            else:
                stats.other_tool_calls += 1

        return stats

    def _detect_rdkit_usage(self, code: str) -> bool:
        """Check if code uses RDKit."""
        for pattern in RDKIT_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def _extract_rdkit_functions(self, code: str) -> set[str]:
        """Extract RDKit function names from code."""
        functions = set()

        # Common RDKit function patterns
        patterns = [
            r"Chem\.(\w+)",
            r"AllChem\.(\w+)",
            r"Descriptors\.(\w+)",
            r"Draw\.(\w+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code)
            functions.update(matches)

        return functions

    def _detect_metrics(self, code: str, output: str) -> set[str]:
        """Detect computed metrics from code and output."""
        metrics = set()
        combined = f"{code} {output}".lower()

        for keyword in METRIC_KEYWORDS:
            if keyword.lower() in combined:
                metrics.add(keyword)

        return metrics


def detect_sabotage_type_in_text(text: str) -> Optional[str]:
    """Detect if text mentions a specific sabotage type.

    Returns dataset-aligned sabotage type names (e.g. ``loss_of_hbond``,
    ``electrostatic_repulsion``).

    Args:
        text: Text to search (e.g., model response).

    Returns:
        Sabotage type name if found, None otherwise.
    """
    text_lower = text.lower()

    for stype, patterns in SABOTAGE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return stype

    return None


def check_sabotage_match(
    response_text: str,
    ground_truth_type: str,
) -> tuple[bool, bool]:
    """Check if model response correctly identifies the sabotage type.

    Handles alias resolution so ground truth types from older datasets
    (e.g. ``hbond_loss``) match the canonical patterns.

    Args:
        response_text: The model's final response text.
        ground_truth_type: The sabotage type from ground truth metadata.

    Returns:
        Tuple of (exact_match, partial_match).
        exact_match: True if the detected type matches the ground truth.
        partial_match: True if ANY sabotage type was detected (even if wrong).
    """
    canonical_gt = _SABOTAGE_TYPE_ALIASES.get(
        ground_truth_type.lower(), ground_truth_type.lower()
    )
    detected = detect_sabotage_type_in_text(response_text)
    if detected:
        canonical_detected = _SABOTAGE_TYPE_ALIASES.get(detected, detected)
        return (canonical_detected == canonical_gt, True)
    return (False, False)


def detect_differentiation(
    tool_calls: list[dict],
    response_text: str,
    ground_truth: dict,
) -> dict:
    """Detect if the model correctly differentiated the molecules.

    Args:
        tool_calls: List of tool call records.
        response_text: Final model response text.
        ground_truth: Ground truth with winner/loser Vina scores.

    Returns:
        Dict with differentiation results.
    """
    result = {
        "differentiated": False,
        "correct_direction": False,
        "mentioned_scores": False,
        "sabotage_detected": None,
        "sabotage_correct": False,
    }

    # Check if docking was performed for both molecules
    dock_results = []
    for call in tool_calls:
        if call.get("name") in ("vina_dock", "dock_ligand"):
            if call.get("success"):
                dock_results.append(call)

    if len(dock_results) >= 2:
        result["differentiated"] = True

        # Check if output mentions score difference
        combined_output = " ".join(call.get("output", "") for call in dock_results)
        if any(
            x in combined_output.lower()
            for x in ["kcal", "affinity", "score", "binding"]
        ):
            result["mentioned_scores"] = True

    # Check sabotage type detection
    detected_type = detect_sabotage_type_in_text(response_text)
    result["sabotage_detected"] = detected_type

    if detected_type:
        true_type = ground_truth.get("sabotage_type", "")
        result["sabotage_correct"] = detected_type == true_type

    return result


# =============================================================================
# Paired-Metric Extraction (advanced mode tier-1 signal)
# =============================================================================


# Canonical metric names with the tokens we accept as the "metric label" in
# python_execute output.  All matching is case-insensitive.
_PAIRED_METRIC_ALIASES: dict[str, list[str]] = {
    "logp": ["logp", "log_p", "clogp", "alogp", "mol_logp"],
    "tpsa": ["tpsa", "polar_surface_area", "psa"],
    "mw": ["mw", "mol_wt", "molecular_weight", "molwt"],
    "hbd": ["hbd", "h_bond_donor", "hbond_donor", "num_hdonors", "num_h_donors"],
    "hba": [
        "hba",
        "h_bond_acceptor",
        "hbond_acceptor",
        "num_hacceptors",
        "num_h_acceptors",
    ],
    "qed": ["qed"],
    "rotatable_bonds": ["rotatable_bonds", "num_rotatable_bonds", "rotbonds"],
    "num_rings": ["num_rings", "ring_count", "ringcount"],
    "fsp3": ["fsp3", "frac_sp3", "fraction_sp3"],
    "heavy_atoms": ["heavy_atoms", "num_heavy_atoms", "heavyatomcount"],
    "formal_charge": ["formal_charge", "net_charge", "total_charge", "charge"],
    "vina_score": ["vina_score", "vina", "binding_affinity", "affinity"],
}

# Per-metric noise thresholds.  A paired difference must exceed this to count
# as "differentiating".  Falls back to ``noise_threshold`` if not listed.
_METRIC_NOISE_THRESHOLDS: dict[str, float] = {
    "logp": 0.3,
    "tpsa": 5.0,
    "mw": 5.0,
    "hbd": 0.5,
    "hba": 0.5,
    "qed": 0.05,
    "rotatable_bonds": 0.5,
    "num_rings": 0.5,
    "fsp3": 0.05,
    "heavy_atoms": 0.5,
    "formal_charge": 0.5,
    "vina_score": 0.5,
}


def _canonicalize_metric_token(token: str) -> Optional[str]:
    """Map an alias like 'cLogP' or 'molecular_weight' to a canonical name."""
    t = token.strip().lower().replace(" ", "_").replace("-", "_")
    for canonical, aliases in _PAIRED_METRIC_ALIASES.items():
        if t == canonical or t in aliases:
            return canonical
    return None


# Regexes that capture a paired (winner, loser) numeric metric from a single
# line of python_execute output.  Each regex MUST yield three groups:
# (metric_label, winner_value, loser_value) or (winner_value, loser_value,
# metric_label) -- handled per-pattern below.
_PAIRED_METRIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "LogP winner: 3.4, loser: 4.8"   |  "LogP winner=3.4 loser=4.8"
    (
        re.compile(
            r"([A-Za-z][A-Za-z0-9_ \-]{0,30})\s*[:\-]?\s*"
            r"winner\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            r"[,;\s]+loser\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "metric_winner_loser",
    ),
    # "winner_logp = 3.4 / loser_logp = 4.8"
    (
        re.compile(
            r"winner[_ ]([A-Za-z][A-Za-z0-9_]{0,30})\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            r".{0,80}?"
            r"loser[_ ]\1\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE | re.DOTALL,
        ),
        "metric_winner_loser",
    ),
    # "metric=logp winner=3.4 loser=4.8"
    (
        re.compile(
            r"metric\s*[:=]\s*([A-Za-z][A-Za-z0-9_]{0,30})"
            r".{0,40}?winner\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            r".{0,40}?loser\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE | re.DOTALL,
        ),
        "metric_winner_loser",
    ),
    # "LogP: winner 3.4 vs loser 4.8"
    (
        re.compile(
            r"([A-Za-z][A-Za-z0-9_ \-]{0,30})\s*[:\-]\s*"
            r"winner\s+(-?\d+(?:\.\d+)?)"
            r"\s+(?:vs|/|\|)\s+loser\s+(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "metric_winner_loser",
    ),
]


def extract_paired_metrics(
    tool_calls: list[dict],
    noise_threshold: float = 0.05,
) -> list[dict]:
    """Parse python_execute outputs for paired (winner, loser) numeric metrics.

    Walks every successful ``python_execute`` call and tries several regex
    formats that researchers commonly use to print a comparison:

    - ``LogP winner: 3.4, loser: 4.8``
    - ``winner_logp = 3.4 ... loser_logp = 4.8``
    - ``metric=logp winner=3.4 loser=4.8``
    - ``LogP: winner 3.4 vs loser 4.8``

    Each match is canonicalized against ``_PAIRED_METRIC_ALIASES`` and
    classified as ``differentiates=True`` when the absolute difference exceeds
    the per-metric noise threshold (or ``noise_threshold`` if unlisted).

    Args:
        tool_calls: Episode tool-call records.
        noise_threshold: Fallback noise threshold for unknown metrics.

    Returns:
        List of dicts ``{metric, winner_value, loser_value, abs_diff,
        differentiates, source_turn}`` -- de-duplicated by canonical metric
        name (first occurrence wins).
    """
    found: dict[str, dict] = {}

    for call in tool_calls:
        name = call.get("tool_name", call.get("name", ""))
        if name not in ("python_execute", "python_exec", "execute_python"):
            continue
        if not call.get("success", True):
            continue

        output = call.get("output", "") or ""
        if not output:
            continue

        turn = call.get("turn", 0)

        for pattern, _kind in _PAIRED_METRIC_PATTERNS:
            for m in pattern.finditer(output):
                raw_label, w_str, l_str = m.group(1), m.group(2), m.group(3)
                canonical = _canonicalize_metric_token(raw_label)
                if canonical is None:
                    continue
                if canonical in found:
                    continue

                try:
                    w = float(w_str)
                    l = float(l_str)
                except ValueError:
                    continue

                threshold = _METRIC_NOISE_THRESHOLDS.get(canonical, noise_threshold)
                abs_diff = abs(w - l)

                found[canonical] = {
                    "metric": canonical,
                    "winner_value": w,
                    "loser_value": l,
                    "abs_diff": abs_diff,
                    "differentiates": abs_diff > threshold,
                    "source_turn": turn,
                }

    return list(found.values())


def hypothesis_cites_metric(
    response_text: str,
    paired_metrics: list[dict],
    hypothesis_patterns: Optional[list[str]] = None,
) -> Optional[str]:
    """Check whether the response's hypothesis sentence references one of the
    paired metrics that were actually computed.

    A "hypothesis" is the first sentence/clause that matches one of the
    standard hypothesis cue patterns (``"my hypothesis is"``, ``"i believe
    the reason is"``, ...).  We then look in the surrounding ~300 chars for
    any alias of one of the supplied paired metrics.

    Args:
        response_text: The model's final response text.
        paired_metrics: Output of ``extract_paired_metrics``.
        hypothesis_patterns: Override the default hypothesis cue patterns.

    Returns:
        The canonical metric name that the hypothesis cites, or ``None`` if
        no hypothesis cue is found or the hypothesis does not cite any
        computed metric.
    """
    if not response_text or not paired_metrics:
        return None

    differentiating = {m["metric"] for m in paired_metrics if m.get("differentiates")}
    if not differentiating:
        return None

    cues = hypothesis_patterns or [
        r"my\s+hypothesis\s+is",
        r"i\s+(?:believe|think|hypothesize)\s+(?:that\s+|the\s+(?:reason|cause)\s+is)?",
        r"i\s+conclude\s+that",
        r"the\s+(?:likely|probable)\s+cause\s+is",
        r"my\s+diagnosis\s+is",
        r"the\s+(?:main|primary|key)\s+(?:difference|issue|problem)\s+is",
    ]

    text_lower = response_text.lower()
    for cue in cues:
        m = re.search(cue, text_lower)
        if not m:
            continue
        # Look at the 300-char window around the cue for a metric mention
        start = m.start()
        window = text_lower[start : start + 300]
        for metric in differentiating:
            for alias in _PAIRED_METRIC_ALIASES[metric]:
                # Word-boundary match so "logp" doesn't match inside "phlogp..."
                if re.search(rf"\b{re.escape(alias)}\b", window):
                    return metric
        # Found a hypothesis cue but no metric mention -- stop scanning further
        # cues; we don't want to award credit for a later "in conclusion" that
        # happens to mention a metric without being the actual hypothesis.
        return None

    return None


# =============================================================================
# Molecule-Edit Detection (advanced mode tier-3 signal)
# =============================================================================


def _morgan_tanimoto(smiles_a: str, smiles_b: str) -> Optional[float]:
    """Compute Morgan-fingerprint Tanimoto similarity between two SMILES.

    Returns ``None`` if either SMILES fails to parse or RDKit is not
    available (e.g. running in a stripped-down environment).
    """
    if not smiles_a or not smiles_b:
        return None
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        logger.warning("RDKit not available; skipping Tanimoto computation")
        return None

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _smiles_from_sdf_bytes(sdf_bytes: bytes) -> Optional[str]:
    """Parse an SDF byte string and return canonical SMILES of the first mol."""
    try:
        from rdkit import Chem
    except ImportError:
        return None
    try:
        text = sdf_bytes.decode("utf-8", errors="replace")
        suppl = Chem.SDMolSupplier()
        suppl.SetData(text, sanitize=True, removeHs=False)
        for mol in suppl:
            if mol is not None:
                return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        pass
    return None


def detect_molecule_edits(
    tool_calls: list[dict],
    storage_artifacts: list[dict],
    ground_truth: dict,
    *,
    preload_refs: Optional[dict[str, str]] = None,
    artifact_loader: Optional[Any] = None,
    edit_max_winner_sim: float = 0.99,
    edit_min_loser_sim: float = 0.40,
) -> dict:
    """Detect non-trivial molecule edits and any subsequent redocking.

    Walks ``storage_artifacts`` to find sdf/smiles refs that were created
    during the episode (i.e. are NOT in ``preload_refs`` -- those are the
    original winner/loser/receptor that the resource server preloaded).

    The first such "edit" artifact is loaded via ``artifact_loader`` (an
    awaitable ``async def(reference) -> bytes``) and Tanimoto similarity is
    computed against both the winner and loser SMILES from
    ``ground_truth["winner_metrics"]["smiles"]`` and
    ``ground_truth["loser_metrics"]["smiles"]``.

    An edit is "non-trivial" when:

    - ``Tanimoto(edit, winner) <= edit_max_winner_sim`` (not a copy of winner)
    - ``Tanimoto(edit, loser)  >= edit_min_loser_sim``  (still resembles loser)

    "Trivial" means ``Tanimoto(edit, winner) > edit_max_winner_sim`` -- the
    agent just regenerated the winner.

    Subsequent ``vina_dock`` calls whose ``ligand_sdf_ref`` matches the edit
    ref are recorded as ``redock_attempted``; if any succeeded and produced a
    parseable affinity, ``redock_affinity`` is the most recent such value.

    Args:
        tool_calls: Episode tool-call records.
        storage_artifacts: ``await session.storage.list_artifacts()`` output.
        ground_truth: Episode ground truth dict (with winner/loser SMILES).
        preload_refs: Mapping of preloaded ref name -> source path. Used to
            skip the original winner/loser/receptor refs when scanning for
            new artifacts.
        artifact_loader: ``async def(reference) -> bytes``. Used to load the
            edit's SDF/SMILES content. If ``None``, only ref-name based
            classification is performed (no Tanimoto, no triviality check).
        edit_max_winner_sim: Tanimoto upper bound vs winner for a non-trivial
            edit (above this is considered a copy of the winner).
        edit_min_loser_sim: Tanimoto lower bound vs loser for a non-trivial
            edit (below this means the edit drifted too far from the loser).

    Returns:
        Dict with keys: ``edit_ref``, ``edit_smiles``, ``tanimoto_winner``,
        ``tanimoto_loser``, ``is_trivial``, ``is_non_trivial``,
        ``redock_attempted``, ``redock_affinity``, ``redock_success``.
    """
    result: dict[str, Any] = {
        "edit_ref": None,
        "edit_smiles": None,
        "tanimoto_winner": None,
        "tanimoto_loser": None,
        "is_trivial": False,
        "is_non_trivial": False,
        "redock_attempted": False,
        "redock_affinity": None,
        "redock_success": False,
    }

    preloaded_refs: set[str] = set((preload_refs or {}).keys())

    # ------------------------------------------------------------------
    # Find the first new sdf_*/smiles_* artifact created during the episode
    # ------------------------------------------------------------------
    edit_ref: Optional[str] = None
    edit_data_type: Optional[str] = None
    for artifact in sorted(
        storage_artifacts, key=lambda a: a.get("created_at") or ""
    ):
        ref = artifact.get("reference", "")
        dtype = (artifact.get("data_type") or "").lower()
        if ref in preloaded_refs:
            continue
        if dtype in ("sdf", "smiles", "mol"):
            edit_ref = ref
            edit_data_type = dtype
            break

    if edit_ref is None:
        return result

    result["edit_ref"] = edit_ref

    # ------------------------------------------------------------------
    # Compute Tanimoto vs winner & loser
    # ------------------------------------------------------------------
    winner_smiles = (ground_truth.get("winner_metrics") or {}).get("smiles") or ""
    loser_smiles = (ground_truth.get("loser_metrics") or {}).get("smiles") or ""

    edit_smiles: Optional[str] = None
    if artifact_loader is not None:
        try:
            content = artifact_loader(edit_ref)
            # Support both sync bytes-returning loaders and async coroutines
            if hasattr(content, "__await__"):
                # The caller is responsible for awaiting; we just received the
                # coroutine.  Skip Tanimoto in that case -- callers should pass
                # an already-awaited byte string via a sync wrapper instead.
                logger.debug(
                    "detect_molecule_edits got an awaitable loader -- caller "
                    "must use detect_molecule_edits_async; skipping Tanimoto"
                )
                content = None
        except Exception as e:
            logger.warning(f"Failed to load edit artifact {edit_ref}: {e}")
            content = None

        if content is not None:
            if edit_data_type == "smiles":
                try:
                    edit_smiles = (
                        content.decode("utf-8", errors="replace").strip().split()[0]
                    )
                except Exception:
                    edit_smiles = None
            else:
                edit_smiles = _smiles_from_sdf_bytes(content)

    if edit_smiles:
        result["edit_smiles"] = edit_smiles
        if winner_smiles:
            result["tanimoto_winner"] = _morgan_tanimoto(edit_smiles, winner_smiles)
        if loser_smiles:
            result["tanimoto_loser"] = _morgan_tanimoto(edit_smiles, loser_smiles)

        tw = result["tanimoto_winner"]
        tl = result["tanimoto_loser"]

        if tw is not None and tw > edit_max_winner_sim:
            result["is_trivial"] = True
        elif tw is not None and tl is not None:
            result["is_non_trivial"] = (
                tw <= edit_max_winner_sim and tl >= edit_min_loser_sim
            )
        elif tw is not None and loser_smiles == "":
            # No loser SMILES to compare -- accept as non-trivial if not a winner copy
            result["is_non_trivial"] = tw <= edit_max_winner_sim
    else:
        # No way to compute Tanimoto -- fall back to: any new sdf/smiles
        # artifact counts as a non-trivial edit attempt.  We cannot detect
        # the trivial-winner case in this fallback path.
        result["is_non_trivial"] = True

    # ------------------------------------------------------------------
    # Look for subsequent vina_dock calls that referenced the edit
    # ------------------------------------------------------------------
    for call in tool_calls:
        name = call.get("tool_name", call.get("name", ""))
        if name not in ("vina_dock", "dock_ligand"):
            continue
        args = call.get("arguments", {}) or {}
        ligand_ref = args.get("ligand_sdf_ref") or args.get("ligand_ref") or ""
        if ligand_ref != edit_ref:
            continue

        result["redock_attempted"] = True
        if call.get("success", True):
            output = call.get("output", "") or ""
            affinity = _extract_affinity_from_output(output)
            if affinity is not None:
                result["redock_affinity"] = affinity
                result["redock_success"] = True

    return result


async def detect_molecule_edits_async(
    tool_calls: list[dict],
    storage_artifacts: list[dict],
    ground_truth: dict,
    *,
    preload_refs: Optional[dict[str, str]] = None,
    storage: Optional[Any] = None,
    edit_max_winner_sim: float = 0.99,
    edit_min_loser_sim: float = 0.40,
) -> dict:
    """Async variant of ``detect_molecule_edits`` that awaits ``storage.retrieve``.

    Use this from the resource server's ``verify()`` endpoint; it handles
    fetching the edit artifact bytes via ``await storage.retrieve(ref)``.
    """
    pre_loaded_bytes: dict[str, bytes] = {}
    if storage is not None:
        # Find candidate edit refs first, then pre-fetch their bytes so the
        # sync path doesn't need an awaitable loader.
        preloaded_refs: set[str] = set((preload_refs or {}).keys())
        for artifact in sorted(
            storage_artifacts, key=lambda a: a.get("created_at") or ""
        ):
            ref = artifact.get("reference", "")
            dtype = (artifact.get("data_type") or "").lower()
            if ref in preloaded_refs:
                continue
            if dtype in ("sdf", "smiles", "mol"):
                try:
                    pre_loaded_bytes[ref] = await storage.retrieve(ref)
                except Exception as e:
                    logger.warning(f"Failed to retrieve edit artifact {ref}: {e}")
                break  # only need the first edit

    def _loader(ref: str) -> Optional[bytes]:
        return pre_loaded_bytes.get(ref)

    return detect_molecule_edits(
        tool_calls=tool_calls,
        storage_artifacts=storage_artifacts,
        ground_truth=ground_truth,
        preload_refs=preload_refs,
        artifact_loader=_loader,
        edit_max_winner_sim=edit_max_winner_sim,
        edit_min_loser_sim=edit_min_loser_sim,
    )


def _extract_affinity_from_output(output: str) -> Optional[float]:
    """Extract a Vina binding affinity (kcal/mol) from a docking tool output."""
    if not output:
        return None
    patterns = [
        r"affinity[:\s]+(-?\d+\.?\d*)\s*kcal",
        r"binding\s+affinity[:\s]+(-?\d+\.?\d*)",
        r"binding\s+energy[:\s]+(-?\d+\.?\d*)",
        r"vina\s+score[:\s]+(-?\d+\.?\d*)",
        r"score[:\s]+(-?\d+\.?\d*)\s*kcal",
        r"(-?\d+\.\d+)\s*kcal/mol",
    ]
    for pattern in patterns:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


# =============================================================================
# Tiered Reward Computer
# =============================================================================


class TieredRewardComputer:
    """Computes tiered rewards for training episodes.

    This class implements the complete reward computation pipeline:
    - Tier 1: Tool usage rewards (accumulated during episode)
    - Tier 2: Differentiation rewards (at episode end)
    - Tier 3: Advanced mode rewards (if enabled)

    Example:
        computer = TieredRewardComputer.from_config("training/configs/rewards.yaml")

        result = await computer.compute(
            tool_calls=trajectory,
            ground_truth=ground_truth,
            response=model_response,
        )

        print(f"Total reward: {result['total_reward']}")
    """

    def __init__(self, config: dict):
        """Initialize with configuration.

        Args:
            config: Reward configuration dictionary.
        """
        self.config = {**DEFAULT_REWARD_CONFIG, **config}
        self.tool_detector = ToolUsageDetector()

    @classmethod
    def from_config(cls, config_path: str) -> "TieredRewardComputer":
        """Create from YAML configuration file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Configured TieredRewardComputer instance.
        """
        config = {}
        path = Path(config_path)

        if path.exists():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Rewards config not found at {config_path}, using defaults")

        return cls(config)

    async def compute(
        self,
        tool_calls: list[dict],
        ground_truth: dict,
        response: Any,
        hypothesis_detected: bool = False,
        molecule_modified: bool = False,
        redock_result: Optional[dict] = None,
    ) -> dict:
        """Compute all tiers of rewards.

        Args:
            tool_calls: List of tool call records from the episode.
            ground_truth: Ground truth information for the task.
            response: The model's response object.
            hypothesis_detected: Whether a hypothesis was detected.
            molecule_modified: Whether the molecule was modified.
            redock_result: Results from redocking modified molecule.

        Returns:
            Dictionary with reward breakdown and details.
        """
        result = {
            "tier1_reward": 0.0,
            "tier1_details": {},
            "tier2_reward": 0.0,
            "tier2_details": {},
            "tier3_reward": 0.0,
            "tier3_details": {},
            "total_reward": 0.0,
        }

        # Analyze tool usage
        usage_stats = self.tool_detector.analyze_trajectory(tool_calls)

        # =================================================================
        # Tier 1: Tool Usage Rewards
        # =================================================================
        tier1_config = self.config.get("tier1", {})
        tier1_reward = 0.0
        tier1_details = {}

        # Docking attempts
        dock_reward = usage_stats.dock_successes * tier1_config.get(
            "dock_attempt", 0.05
        )
        tier1_details["dock_reward"] = dock_reward
        tier1_details["dock_successes"] = usage_stats.dock_successes
        tier1_reward += dock_reward

        # Python exec success
        exec_reward = usage_stats.python_exec_successes * tier1_config.get(
            "python_exec_success", 0.05
        )
        tier1_details["exec_reward"] = exec_reward
        tier1_details["exec_successes"] = usage_stats.python_exec_successes
        tier1_reward += exec_reward

        # RDKit usage
        if usage_stats.rdkit_usage:
            rdkit_reward = tier1_config.get("rdkit_usage", 0.03)
            tier1_details["rdkit_reward"] = rdkit_reward
            tier1_details["rdkit_functions"] = list(usage_stats.rdkit_functions_used)
            tier1_reward += rdkit_reward

        # Metrics computed
        num_metrics = len(usage_stats.metrics_computed)
        metric_reward = num_metrics * tier1_config.get("metric_computed", 0.05)
        tier1_details["metric_reward"] = metric_reward
        tier1_details["metrics_computed"] = list(usage_stats.metrics_computed)
        tier1_reward += metric_reward

        # Cap tier 1 reward
        max_tier1 = tier1_config.get("max_tier1_reward", 0.50)
        tier1_reward = min(tier1_reward, max_tier1)

        result["tier1_reward"] = tier1_reward
        result["tier1_details"] = tier1_details

        # =================================================================
        # Tier 2: Differentiation Rewards
        # =================================================================
        tier2_config = self.config.get("tier2", {})
        tier2_reward = 0.0
        tier2_details = {}

        # Extract response text for analysis
        response_text = ""
        if hasattr(response, "output"):
            for item in response.output:
                if isinstance(item, dict) and item.get("type") == "message":
                    content = item.get("content", [])
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            response_text += c.get("text", "")
        elif isinstance(response, dict):
            response_text = str(response.get("output", ""))

        # Detect differentiation
        diff_result = detect_differentiation(tool_calls, response_text, ground_truth)
        tier2_details.update(diff_result)

        if diff_result["differentiated"]:
            tier2_reward += tier2_config.get("correct_differentiation", 0.30)

        if diff_result["sabotage_correct"]:
            tier2_reward += tier2_config.get("correct_sabotage_type", 0.20)
        elif diff_result["sabotage_detected"]:
            # Partial credit for detecting some type
            tier2_reward += tier2_config.get("partial_diagnosis", 0.10)

        result["tier2_reward"] = tier2_reward
        result["tier2_details"] = tier2_details

        # =================================================================
        # Tier 3: Advanced Mode Rewards
        # =================================================================
        tier3_config = self.config.get("tier3", {})
        tier3_reward = 0.0
        tier3_details = {
            "hypothesis_detected": hypothesis_detected,
            "molecule_modified": molecule_modified,
        }

        if hypothesis_detected:
            # Small reward for forming hypothesis
            tier3_details["hypothesis_reward"] = 0.1
            tier3_reward += 0.1

        if molecule_modified:
            tier3_reward += tier3_config.get("molecule_modified", 0.50)
            tier3_details["modification_reward"] = tier3_config.get(
                "molecule_modified", 0.50
            )

        if redock_result:
            tier3_details["redock_result"] = redock_result

            if redock_result.get("success"):
                tier3_reward += tier3_config.get("redock_success", 0.50)
                tier3_details["redock_reward"] = tier3_config.get(
                    "redock_success", 0.50
                )

                # Check affinity recovery
                modified_affinity = redock_result.get("modified_affinity")
                winner_affinity = ground_truth.get("winner_vina", 0)
                target_fraction = tier3_config.get("target_recovery_fraction", 0.80)

                if modified_affinity and winner_affinity:
                    # Vina scores are negative, lower is better
                    # Recovery means getting close to winner's (more negative) score
                    if modified_affinity <= winner_affinity * target_fraction:
                        tier3_reward += tier3_config.get("affinity_recovered", 1.00)
                        tier3_details["affinity_recovered"] = True
                        tier3_details["recovery_ratio"] = (
                            modified_affinity / winner_affinity
                        )

        result["tier3_reward"] = tier3_reward
        result["tier3_details"] = tier3_details

        # =================================================================
        # Total Reward
        # =================================================================
        result["total_reward"] = (
            result["tier1_reward"] + result["tier2_reward"] + result["tier3_reward"]
        )

        return result


# =============================================================================
# Reward Annotation Builder
# =============================================================================


def build_reward_annotations(
    tool_calls: list[dict],
    tool_span_ids: list[tuple[str, str, int]],
    root_span_id: str,
    awarded_rewards: list[str],
    episode_id: str = "",
    total_reward: Optional[float] = None,
    reward_breakdown: Optional[dict[str, float]] = None,
    catalog: Optional[dict[str, RewardEntry]] = None,
    judge_result: Optional[dict] = None,
    per_tool_call_rewards: Optional[list[dict]] = None,
) -> list[dict]:
    """Map awarded rewards to per-span Phoenix annotation dicts.

    For each reward in *awarded_rewards*, looks up the ``REWARD_CATALOG`` entry.
    If the entry has ``tool_match``, annotates only the FIRST matching tool
    span (not every one) so annotation sums are meaningful.  If ``tool_match``
    is ``None`` the annotation goes on the root ``chat_request`` span.

    When *total_reward* is provided, a ``total_effective_reward`` annotation
    is added to the root span with the full capped/summed reward and a
    per-tier breakdown in metadata.

    When *judge_result* is provided (dict from ``LLMJudge.evaluate()``),
    a ``judge_reasoning`` annotation is added to the ``judge_evaluation``
    span (or root span if the judge span ID is missing).

    Args:
        tool_calls: Episode tool-call records (same list passed to verify).
        tool_span_ids: ``(span_id_hex, tool_name, tc_index)`` tuples produced
            by ``EpisodeSpan.build_trace()``.
        root_span_id: Hex span-id of the root ``chat_request`` span.
        awarded_rewards: Names of rewards that were actually awarded this
            episode (subset of ``REWARD_CATALOG`` keys).
        episode_id: Optional episode identifier for annotation metadata.
        total_reward: The final effective reward for the episode (after
            caps and summing).  Added as a ``total_effective_reward``
            annotation on the root span.
        reward_breakdown: Per-tier reward values (e.g.
            ``{"tier1_tool_usage": 0.6, ...}``).  Stored in annotation
            metadata.
        catalog: Override the default ``REWARD_CATALOG`` (mainly for tests).
        judge_result: Result dict from ``LLMJudge.evaluate()``, containing
            ``reward``, ``span_id``, ``scores``, ``reasoning``, etc.
        per_tool_call_rewards: Optional list of per-call Tier-1 reward records.
            Each record is a dict with keys ``span_id`` (hex id of that call's
            tool span, or empty to fall back to root), ``label`` (e.g.
            ``"tool_call_attempt"`` / ``"tool_call_success"``), ``value`` (the
            per-tool reward magnitude), ``tool_name``, ``tc_index`` (the
            tool-call index, used to build a unique ``label:tc_index``
            identifier), and optional ``description``.

    Returns:
        List of annotation dicts ready for ``EpisodeSpan.annotate_rewards()``.
        Each dict has keys: ``span_id``, ``label``, ``score``, ``metadata``.
    """
    cat = catalog or REWARD_CATALOG
    annotations: list[dict] = []

    # -- Total effective reward annotation on root span --
    if total_reward is not None:
        total_meta: dict[str, Any] = {
            "reward_name": "total_effective_reward",
            "description": "Total episode reward after caps and summing",
        }
        if episode_id:
            total_meta["episode_id"] = episode_id
        if reward_breakdown:
            total_meta["breakdown"] = reward_breakdown
        annotations.append(
            {
                "span_id": root_span_id,
                "label": "total_effective_reward",
                "score": float(total_reward),
                "metadata": total_meta,
            }
        )

    # -- Per-reward annotations --
    for reward_name in awarded_rewards:
        entry = cat.get(reward_name)
        if entry is None:
            logger.warning(
                f"Reward name '{reward_name}' not found in catalog, skipping"
            )
            continue

        metadata: dict[str, Any] = {
            "tier": entry.tier,
            "reward_name": reward_name,
            "description": entry.description,
        }
        if episode_id:
            metadata["episode_id"] = episode_id

        if entry.tool_match is not None:
            # Annotate only the FIRST matching tool span so annotation
            # sums equal the actual tier reward (not N * value).
            first_match_id = None
            for span_id_hex, tool_name, _tc_idx in tool_span_ids:
                if tool_name in entry.tool_match:
                    first_match_id = span_id_hex
                    metadata["tool_name"] = tool_name
                    break
            annotations.append(
                {
                    "span_id": first_match_id or root_span_id,
                    "label": reward_name,
                    "score": entry.value,
                    "metadata": {
                        **metadata,
                        **(
                            {"note": "no matching tool span found"}
                            if not first_match_id
                            else {}
                        ),
                    },
                }
            )
        else:
            # Episode-level reward -> annotate root span
            annotations.append(
                {
                    "span_id": root_span_id,
                    "label": reward_name,
                    "score": entry.value,
                    "metadata": metadata,
                }
            )

    # -- Per-tool-call reward annotations (Tier-1 schedule) --
    # One annotation per record, placed on that call's OWN tool.{name} span (not
    # the root) so the reward is visible exactly where the work happened. Each
    # record carries its own ``label`` (e.g. "tool_call_attempt" /
    # "tool_call_success") and per-tool ``value`` -- no catalog lookup. Each
    # annotation carries a unique ``identifier`` (``label:tc_index``) so attempt
    # and success annotations coexist on the same span without overwriting each
    # other under the shared label+span.
    if per_tool_call_rewards:
        for rec in per_tool_call_rewards:
            span_id = rec.get("span_id") or root_span_id
            label = rec.get("label", "tool_call_attempt")
            value = rec.get("value", 0.0)
            call_meta: dict[str, Any] = {
                "tier": 1,
                "reward_name": label,
                "description": rec.get("description", "Tier-1 per-call tool reward"),
                "tool_name": rec.get("tool_name", ""),
            }
            if episode_id:
                call_meta["episode_id"] = episode_id
            if not rec.get("span_id"):
                call_meta["note"] = "no matching tool span found"
            annotations.append(
                {
                    "span_id": span_id,
                    "label": label,
                    # Unique per (label, call) so root-fallback duplicates and
                    # attempt/success pairs on the same span don't clobber.
                    "identifier": f"{label}:{rec.get('tc_index', len(annotations))}",
                    "score": float(value),
                    "metadata": call_meta,
                }
            )

    # -- Judge reasoning annotation --
    if judge_result and judge_result.get("reward", 0.0) != 0.0:
        judge_meta: dict[str, Any] = {
            "reward_name": "judge_reasoning",
            "description": "LLM-as-judge reasoning quality evaluation",
            "verdict": "SOUND" if judge_result.get("is_sound") else "FLAWED",
            "total_score": judge_result.get("total_score", 0),
            "max_score": judge_result.get("max_score", 10),
        }
        if episode_id:
            judge_meta["episode_id"] = episode_id
        scores = judge_result.get("scores")
        if scores:
            judge_meta["scores"] = scores
        reasoning = judge_result.get("reasoning")
        if reasoning:
            judge_meta["reasoning"] = reasoning

        annotations.append(
            {
                "span_id": judge_result.get("span_id") or root_span_id,
                "label": "judge_reasoning",
                "score": float(judge_result["reward"]),
                "metadata": judge_meta,
            }
        )

    return annotations


def compute_reward(
    tool_calls: list[dict],
    ground_truth: dict,
    response: Any,
    config: Optional[dict] = None,
) -> dict:
    """Convenience function to compute rewards.

    Args:
        tool_calls: List of tool call records.
        ground_truth: Ground truth information.
        response: Model response.
        config: Optional reward configuration.

    Returns:
        Reward computation result.
    """
    import asyncio

    computer = TieredRewardComputer(config or {})
    return asyncio.run(
        computer.compute(
            tool_calls=tool_calls,
            ground_truth=ground_truth,
            response=response,
        )
    )
