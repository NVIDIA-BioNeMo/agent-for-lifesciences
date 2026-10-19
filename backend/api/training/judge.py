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

"""LLM-as-Judge for reasoning quality evaluation.

This module implements an LLM judge that evaluates the reasoning quality
of agent trajectories for computing soft rewards during training.

The judge evaluates:
1. Logical consistency - Does reasoning follow from observations?
2. Data-driven - Are conclusions supported by computed data?
3. Scientific accuracy - Are chemistry concepts used correctly?

Judge prompts are configurable via YAML configuration.
"""

import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)


# =============================================================================
# Judge Endpoint (multi-model pool entry)
# =============================================================================

@dataclass
class JudgeEndpoint:
    """One backend in the judge pool.

    Pool semantics (see ``LLMJudge._pick_endpoint``):
    - Each ``_call_judge`` attempt picks one endpoint uniformly at random
      from the enabled subset. Lock-free, asyncio-safe; over many concurrent
      calls this approximates round-robin.
    - A retry-after-failure picks a fresh endpoint, so a transiently-bad
      model (e.g. one returning 502) gets routed around within a single call.
    - Endpoints with ``enabled=False`` are skipped entirely (kept in the
      config for reproducibility / future re-enablement).
    """

    name: str
    url: str
    model: str
    api_key: Optional[str] = None
    enabled: bool = True


# =============================================================================
# Default Judge Configuration
# =============================================================================

DEFAULT_JUDGE_SYSTEM_PROMPT = """You are evaluating a computational chemist's reasoning about molecular binding.

The assistant was asked to (1) diagnose why one of two similar molecules binds better to a protein target, (2) state a hypothesis grounded in a *quantitative* metric that distinguishes the two molecules, and (3) test the hypothesis by editing the loser molecule and re-docking it.

Your job is to score the assistant on six categories of advanced-mode behavior. Be strict about the new metric-grounding and edit-tests-hypothesis categories -- those are the central skills we are training.

## Evaluation Criteria

1. **Logical Consistency** (0-3 points)
   - 3: Reasoning flows logically, each step builds on previous observations
   - 2: Mostly logical with minor gaps
   - 1: Some logical jumps or inconsistencies
   - 0: Reasoning is disjointed or contradictory

2. **Differentiating Metric Identified** (0-3 points)
   - 3: A clearly-named quantitative metric (e.g. LogP, TPSA, MW, HBD, HBA, formal charge, Vina score) was computed for BOTH molecules and the values genuinely differ in a way that could explain the binding gap
   - 2: A paired metric was computed but the difference is small / borderline, OR only one molecule was measured and the other was inferred
   - 1: A metric was mentioned but no actual paired computation was performed
   - 0: No quantitative metric was used to distinguish the molecules

3. **Hypothesis Grounded in Metric** (0-3 points)
   - 3: A hypothesis is stated explicitly (e.g. "My hypothesis is...") AND it cites the metric value from category 2 as the cause of the binding loss
   - 2: A hypothesis is stated and refers to the metric category but not the specific computed value
   - 1: A hypothesis is stated but is not grounded in a computed metric (qualitative only)
   - 0: No clear hypothesis was stated

4. **Scientific Accuracy** (0-3 points)
   - 3: Correct use of chemical concepts (H-bonds, steric clash, electrostatics, lipophilicity, polar surface area)
   - 2: Mostly correct with minor inaccuracies
   - 1: Contains significant chemical misconceptions
   - 0: Fundamentally incorrect chemistry

5. **Edit Tests Hypothesis** (0-3 points)
   - 3: A new molecule was constructed via RDKit (NOT a copy of the winner) AND the chemical change directly addresses the hypothesized defect AND the new molecule was re-docked
   - 2: A new molecule was constructed and re-docked but the change only weakly addresses the hypothesis, OR the edit addresses the hypothesis but was not re-docked
   - 1: A modification was attempted but is unrelated to the hypothesis, or is a trivial copy of the winner
   - 0: No molecule edit was attempted

6. **Diagnostic Quality** (0-1 point)
   - 1: Correctly identified the type of molecular defect (matches the ground-truth sabotage type)
   - 0: Did not correctly identify the defect

## Output Format

Provide your evaluation in this exact format (do NOT add extra punctuation around the scores):

LOGICAL_CONSISTENCY: <score 0-3>
DIFFERENTIATING_METRIC: <score 0-3>
HYPOTHESIS_GROUNDED: <score 0-3>
SCIENTIFIC_ACCURACY: <score 0-3>
EDIT_TESTS_HYPOTHESIS: <score 0-3>
DIAGNOSTIC_QUALITY: <score 0-1>

REASONING: <brief explanation of your scores; cite the specific tool call or response sentence supporting each score>

VERDICT: [[SOUND]] or [[FLAWED]]

A verdict of SOUND requires total score >= 11/16 AND no zero in any of the five 0-3 categories. (Diagnostic Quality may be 0.)
"""

DEFAULT_JUDGE_PROMPT_TEMPLATE = """<|Task|>
The assistant was asked to diagnose why one of two similar molecules binds better to a protein target, ground its hypothesis in a quantitative metric, edit the loser molecule, and re-dock the edit.

<|Ground Truth|>
- Sabotage type: {sabotage_type}
- Winner Vina score: {winner_vina} kcal/mol
- Loser Vina score: {loser_vina} kcal/mol
- Score difference: {delta_vina} kcal/mol
- Ground-truth modification (NOT to be revealed in your reasoning): {modification_desc}

<|Tool Calls Made|>
{tool_calls_summary}

<|Assistant's Final Response|>
{final_response}

<|Full Reasoning Trace|>
{reasoning_trace}

Please evaluate the assistant's reasoning quality using the six-category rubric above. Remember:
- "Differentiating Metric" requires that a metric was actually COMPUTED (look for python_execute output with both winner and loser values), not just mentioned.
- "Edit Tests Hypothesis" requires both a new molecule AND a re-dock; a trivial copy of the winner does NOT count.
"""


# =============================================================================
# Judge Prompts Container
# =============================================================================

@dataclass
class JudgePrompts:
    """Container for configurable judge prompts."""
    
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT
    prompt_template: str = DEFAULT_JUDGE_PROMPT_TEMPLATE
    
    # Additional configurable patterns
    sound_verdict_pattern: str = r"\[\[SOUND\]\]"
    flawed_verdict_pattern: str = r"\[\[FLAWED\]\]"
    score_patterns: dict[str, str] = None

    # Prometheus-2 absolute-grading output ends in "[RESULT] N" (N in 1-5).
    # Used by the ``prometheus`` score_format (and the ``auto`` fallback) to
    # parse models that ignore the 6-category Aevom rubric. See
    # https://github.com/prometheus-eval/prometheus-eval (ABS output format).
    prometheus_score_pattern: str = r"\[RESULT\]\s*([1-5])"
    
    def __post_init__(self):
        if self.score_patterns is None:
            self.score_patterns = {
                "logical_consistency": r"LOGICAL_CONSISTENCY:\s*(\d)",
                "differentiating_metric": r"DIFFERENTIATING_METRIC:\s*(\d)",
                "hypothesis_grounded": r"HYPOTHESIS_GROUNDED:\s*(\d)",
                "scientific_accuracy": r"SCIENTIFIC_ACCURACY:\s*(\d)",
                "edit_tests_hypothesis": r"EDIT_TESTS_HYPOTHESIS:\s*(\d)",
                "diagnostic_quality": r"DIAGNOSTIC_QUALITY:\s*(\d)",
            }
    
    @classmethod
    def from_config(cls, config: dict) -> "JudgePrompts":
        """Create from configuration dictionary."""
        return cls(
            system_prompt=config.get("system_prompt", DEFAULT_JUDGE_SYSTEM_PROMPT),
            prompt_template=config.get("prompt_template", DEFAULT_JUDGE_PROMPT_TEMPLATE),
            sound_verdict_pattern=config.get("sound_verdict_pattern", r"\[\[SOUND\]\]"),
            flawed_verdict_pattern=config.get("flawed_verdict_pattern", r"\[\[FLAWED\]\]"),
            score_patterns=config.get("score_patterns"),
            prometheus_score_pattern=config.get(
                "prometheus_score_pattern", r"\[RESULT\]\s*([1-5])"
            ),
        )


# =============================================================================
# Judge Result
# =============================================================================

@dataclass
class JudgeResult:
    """Result from judge evaluation (advanced-mode rubric, total /16)."""

    # Overall
    is_sound: bool = False
    total_score: int = 0
    max_score: int = 16

    # Component scores (advanced-mode rubric: 5 x 0-3 + 1 x 0-1 = /16)
    logical_consistency: int = 0
    differentiating_metric: int = 0
    hypothesis_grounded: int = 0
    scientific_accuracy: int = 0
    edit_tests_hypothesis: int = 0
    diagnostic_quality: int = 0

    # Computed reward
    reward: float = 0.0

    # Metadata
    reasoning: str = ""
    raw_response: str = ""
    error: Optional[str] = None


# =============================================================================
# LLM Judge
# =============================================================================

class LLMJudge:
    """LLM-as-Judge for evaluating reasoning quality.
    
    This class uses an LLM to evaluate the quality of agent reasoning
    during molecular binding analysis tasks.
    
    Example:
        judge = LLMJudge.from_config("training/configs/judge.yaml")
        
        result = await judge.evaluate(
            trajectory=tool_calls,
            response=model_response,
            ground_truth=ground_truth,
        )
        
        print(f"Sound reasoning: {result.is_sound}")
        print(f"Reward: {result.reward}")
    """
    
    def __init__(
        self,
        prompts: JudgePrompts,
        model_url: Optional[str] = None,
        model_name: str = "judge",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        reward_scale: float = 0.3,
        endpoints: Optional[list[JudgeEndpoint]] = None,
        score_format: str = "auto",
    ):
        """Initialize the judge.

        Two modes:

        (a) Single-endpoint / local-vLLM mode (legacy): pass ``model_url``
            (single URL or comma-separated list) + ``model_name`` + ``api_key``.
            Used by the local-vLLM judge path (``JUDGE_NODES>0``) and any
            caller that hasn't migrated to the pool config.

        (b) Multi-model pool mode: pass ``endpoints`` (a list of
            ``JudgeEndpoint``). Each per-call attempt picks a random enabled
            endpoint and sends the call with that endpoint's
            (url, model, api_key) triple. ``model_url`` / ``model_name`` /
            ``api_key`` are ignored in this mode.

        Args:
            prompts: JudgePrompts configuration.
            model_url: URL of the judge LLM API. May be a single URL or a
                comma-separated list of URLs for client-side load balancing
                across multiple judge backends (see Phase B of
                ``internal_docs/docs/training_scaling_curve.md`` ``v0517e-judge5`` probe).
                Ignored when ``endpoints`` is provided.
            model_name: Name of the model to use.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            reward_scale: Scale factor for converting score to reward.
            endpoints: Optional multi-model pool. When non-empty, supersedes
                the (model_url, model_name, api_key) triple.
            score_format: How to parse judge responses into scores. One of:
                ``"aevom"`` (the 6-category rubric -> /16), ``"prometheus"``
                (absolute-grading ``[RESULT] N`` -> /5), or ``"auto"`` (try
                aevom first, fall back to prometheus when the aevom parse
                yields 0 and a ``[RESULT]`` token is present). The env var
                ``AEVOM_JUDGE_FORMAT`` overrides this argument when set, so the
                local Prometheus-2 sbatch can force prometheus parsing without
                touching the shared judge.yaml. Defaults to ``"auto"``.
        """
        self.prompts = prompts
        self.timeout = timeout
        self.reward_scale = reward_scale

        # Env var wins so the local-judge sbatch can force a format without
        # editing the shared judge.yaml; "auto" still parses Prometheus via
        # the [RESULT] fallback even if the env var is absent.
        env_fmt = os.environ.get("AEVOM_JUDGE_FORMAT", "").strip().lower()
        self.score_format = env_fmt or (score_format or "auto").lower()

        if endpoints:
            self.endpoints = list(endpoints)
            enabled = [ep for ep in self.endpoints if ep.enabled]
            if not enabled:
                raise ValueError(
                    "LLMJudge requires at least one enabled endpoint, "
                    f"got {len(self.endpoints)} endpoints all disabled."
                )
            self.model_url = ",".join(ep.url for ep in enabled)
            self.model_urls = [ep.url for ep in enabled]
            self.model_name = enabled[0].model
            self.api_key = enabled[0].api_key
        else:
            self.endpoints = []
            self.model_url = model_url or ""
            self.model_urls = [u.strip() for u in str(model_url or "").split(",") if u.strip()]
            if not self.model_urls and model_url is not None:
                self.model_urls = [model_url]
            self.model_name = model_name
            self.api_key = api_key

        # Startup health log -- greppable in services-*.log to confirm the
        # configured mode matches what's actually running. The v0528a post-mortem
        # in internal_docs/docs/training_scaling_curve.md spent significant time tracing
        # the dead-pool failure mode that this single line would have caught.
        if self.endpoints:
            enabled_names = [ep.name for ep in self.endpoints if ep.enabled]
            logger.info(
                "[AEVOM] LLMJudge: pool mode engaged with %d/%d endpoints enabled: %s",
                len(enabled_names),
                len(self.endpoints),
                ", ".join(enabled_names),
            )
        else:
            logger.info(
                "[AEVOM] LLMJudge: single-endpoint mode -> %s (model=%s)",
                self.model_url,
                self.model_name,
            )

    def _pick_endpoint(self) -> JudgeEndpoint:
        """Pick one enabled endpoint per call.

        Used in multi-model pool mode. Uniform-random across enabled
        endpoints; lock-free (safe under asyncio concurrency).
        """
        enabled = [ep for ep in self.endpoints if ep.enabled]
        if not enabled:
            raise RuntimeError(
                "LLMJudge has no enabled endpoints (all entries disabled)."
            )
        if len(enabled) == 1:
            return enabled[0]
        return random.choice(enabled)

    def _pick_url(self) -> str:
        """Pick one judge backend URL per call (single-endpoint mode).

        Uniform-random across configured backends. With many concurrent calls
        this approximates round-robin without needing a lock, which would be
        a real concern given ``evaluate`` is called from many asyncio tasks.
        """
        if len(self.model_urls) == 1:
            return self.model_urls[0]
        return random.choice(self.model_urls)
    
    @classmethod
    def from_config(
        cls,
        config_path: str,
        model_url: Optional[str] = None,
    ) -> "LLMJudge":
        """Create from YAML configuration file.

        Resolution order:

        1. If ``model_url`` is passed *and* non-empty (override path -- used
           by the local-vLLM judge path that sets ``AEVOM_JUDGE_MODEL_URL`` to
           the local backend before LLMJudge is constructed), build a
           single-endpoint judge from the config's ``model_name`` / ``api_key``
           with that URL. This preserves backward compatibility with
           ``JUDGE_NODES>0`` deployments.
        2. Else, if the config has a non-empty ``judge_models`` list, build a
           multi-model pool from it. Each entry maps to a ``JudgeEndpoint``.
           Disabled entries are kept (for reproducibility) but skipped at
           pick time.
        3. Else, fall back to single-endpoint legacy behavior using the
           config's ``model_url`` / ``model_name`` / ``api_key`` fields.

        Args:
            config_path: Path to YAML config file.
            model_url: Override URL for judge model (forces single-endpoint
                mode when non-empty).

        Returns:
            Configured LLMJudge instance.
        """
        config = {}
        path = Path(config_path)

        if path.exists():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Judge config not found at {config_path}, using defaults")

        prompts = JudgePrompts.from_config(config.get("prompts", {}))
        timeout = config.get("timeout", 60.0)
        reward_scale = config.get("reward_scale", 0.3)
        score_format = config.get("score_format", "auto")

        fallback_api_key = (
            config.get("api_key")
            or os.environ.get("AEVOM_JUDGE_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
        )

        if model_url:
            return cls(
                prompts=prompts,
                model_url=model_url,
                model_name=config.get("model_name", "judge"),
                api_key=fallback_api_key,
                timeout=timeout,
                reward_scale=reward_scale,
                score_format=score_format,
            )

        judge_models = config.get("judge_models") or []
        if judge_models:
            endpoints = cls._parse_endpoints(judge_models, fallback_api_key)
            if any(ep.enabled for ep in endpoints):
                logger.info(
                    "LLMJudge: %d endpoint(s) configured, %d enabled (%s)",
                    len(endpoints),
                    sum(1 for ep in endpoints if ep.enabled),
                    ", ".join(ep.name for ep in endpoints if ep.enabled),
                )
                return cls(
                    prompts=prompts,
                    endpoints=endpoints,
                    timeout=timeout,
                    reward_scale=reward_scale,
                    score_format=score_format,
                )
            logger.warning(
                "LLMJudge: judge_models present but ALL disabled; "
                "falling back to single-endpoint model_url / model_name."
            )

        return cls(
            prompts=prompts,
            model_url=config.get("model_url", "http://localhost:8000/v1"),
            model_name=config.get("model_name", "judge"),
            api_key=fallback_api_key,
            timeout=timeout,
            reward_scale=reward_scale,
            score_format=score_format,
        )

    @staticmethod
    def _parse_endpoints(
        entries: list[dict],
        fallback_api_key: Optional[str],
    ) -> list[JudgeEndpoint]:
        """Parse a list of ``judge_models`` YAML entries into ``JudgeEndpoint``.

        Each entry must define ``name``, ``endpoint`` (URL), and ``model``.
        ``enabled`` defaults to True. ``api_key_env`` names an environment
        variable to read; if absent or unset, ``fallback_api_key`` is used.
        """
        endpoints: list[JudgeEndpoint] = []
        for raw in entries:
            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict judge_models entry: %r", raw)
                continue
            name = raw.get("name")
            url = raw.get("endpoint") or raw.get("url")
            model = raw.get("model")
            if not (name and url and model):
                logger.warning(
                    "Skipping incomplete judge_models entry (need name/endpoint/model): %r",
                    raw,
                )
                continue
            api_key_env = raw.get("api_key_env")
            api_key = (
                os.environ.get(api_key_env) if api_key_env else None
            ) or raw.get("api_key") or fallback_api_key
            endpoints.append(
                JudgeEndpoint(
                    name=str(name),
                    url=str(url),
                    model=str(model),
                    api_key=api_key,
                    enabled=bool(raw.get("enabled", True)),
                )
            )
        return endpoints
    
    def _format_tool_calls(self, trajectory: list[dict]) -> str:
        """Format tool calls for judge prompt."""
        lines = []
        for i, call in enumerate(trajectory, 1):
            name = call.get("name", "unknown")
            success = "✓" if call.get("success", True) else "✗"
            output_preview = str(call.get("output", ""))[:100]
            if len(str(call.get("output", ""))) > 100:
                output_preview += "..."
            lines.append(f"{i}. [{success}] {name}: {output_preview}")
        
        return "\n".join(lines) if lines else "(No tool calls)"
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text from response object."""
        if isinstance(response, str):
            return response
        
        if hasattr(response, "output"):
            text_parts = []
            for item in response.output:
                if isinstance(item, dict):
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                text_parts.append(c.get("text", ""))
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)
        
        if isinstance(response, dict):
            return str(response.get("output", response.get("text", "")))
        
        return str(response)
    
    def _format_reasoning_trace(
        self,
        trajectory: list[dict],
        response: Any,
        reasoning_traces: Optional[list[dict]] = None,
    ) -> str:
        """Format full reasoning trace for judge.

        Args:
            trajectory: Tool call records.
            response: The model's final response.
            reasoning_traces: Optional list of ``{"turn": N, "reasoning": "..."}``
                dicts containing the model's thinking between tool calls.
        """
        trace_parts = []
        reasoning_by_turn: dict[int, str] = {}
        if reasoning_traces:
            for rt in reasoning_traces:
                reasoning_by_turn[rt.get("turn", 0)] = rt.get("reasoning", "")

        for i, call in enumerate(trajectory, 1):
            name = call.get("name", "unknown")
            args = call.get("arguments", {})
            output = call.get("output", "")
            turn = call.get("turn", i)

            thinking = reasoning_by_turn.get(turn, "")
            if thinking:
                trace_parts.append(f"--- Turn {turn}: Reasoning ---")
                trace_parts.append(thinking[:2000])
                trace_parts.append("")

            trace_parts.append(f"--- Turn {turn}: {name} ---")

            if name in ("python_exec", "python_execute"):
                code = args.get("code", "")
                trace_parts.append(f"Code:\n```python\n{code}\n```")

            if args:
                arg_str = ", ".join(f"{k}={v}" for k, v in args.items() if k != "code")
                if arg_str:
                    trace_parts.append(f"Arguments: {arg_str}")

            trace_parts.append(f"Output: {output[:500]}")
            trace_parts.append("")

        final_text = self._extract_response_text(response)
        trace_parts.append("--- Final Response ---")
        trace_parts.append(final_text)

        return "\n".join(trace_parts)
    
    def _build_prompt(
        self,
        trajectory: list[dict],
        response: Any,
        ground_truth: dict,
        reasoning_traces: Optional[list[dict]] = None,
    ) -> str:
        """Build the judge prompt."""
        return self.prompts.prompt_template.format(
            sabotage_type=ground_truth.get("sabotage_type", "unknown"),
            winner_vina=ground_truth.get("winner_vina", "N/A"),
            loser_vina=ground_truth.get("loser_vina", "N/A"),
            delta_vina=ground_truth.get("delta_vina", "N/A"),
            modification_desc=ground_truth.get("modification_desc", "N/A"),
            tool_calls_summary=self._format_tool_calls(trajectory),
            final_response=self._extract_response_text(response),
            reasoning_trace=self._format_reasoning_trace(
                trajectory, response, reasoning_traces=reasoning_traces,
            ),
        )
    
    def _parse_response(self, response_text: str) -> JudgeResult:
        """Dispatch to the configured score-format parser.

        - ``prometheus``: parse the absolute-grading ``[RESULT] N`` token.
        - ``aevom``: parse the 6-category rubric (legacy default).
        - ``auto``: try aevom first; if it yields total_score 0 AND a
          ``[RESULT]`` token is present, fall back to prometheus. This makes
          local Prometheus-2 judges work even if ``AEVOM_JUDGE_FORMAT`` did
          not propagate, while never misfiring on aevom-format pool models
          (which lack ``[RESULT]``).
        """
        fmt = self.score_format
        if fmt == "prometheus":
            return self._parse_prometheus(response_text)
        if fmt == "auto":
            result = self._parse_aevom(response_text)
            if result.total_score == 0 and re.search(
                self.prompts.prometheus_score_pattern, response_text
            ):
                return self._parse_prometheus(response_text)
            return result
        # Default / explicit "aevom".
        return self._parse_aevom(response_text)

    def _parse_prometheus(self, response_text: str) -> JudgeResult:
        """Parse a Prometheus-2 absolute-grading response.

        Prometheus-2 emits ``Feedback: ... [RESULT] N`` where N is an integer
        1-5 on a Likert scale (it ignores our 6-category template). We take the
        LAST ``[RESULT] N`` (reasoning models may rehearse earlier values),
        scale linearly as ``(N/5) * reward_scale`` (so N=5 -> full reward_scale,
        N=1 -> 0.2*reward_scale), and treat N>=4 as SOUND. When no ``[RESULT]``
        token is found the score is 0 and the reward is 0.
        """
        result = JudgeResult(raw_response=response_text, max_score=5)

        matches = list(re.finditer(self.prompts.prometheus_score_pattern, response_text))
        score = 0
        if matches:
            try:
                score = int(matches[-1].group(1))
            except (ValueError, AttributeError):
                score = 0
            # Feedback is everything before the final [RESULT] marker.
            result.reasoning = response_text[: matches[-1].start()].strip()
        else:
            result.reasoning = response_text.strip()

        result.total_score = score
        # Likert SOUND threshold: 4-5 of 5. Mirrors the aevom "high score"
        # intent; only used for the judge.verdict telemetry attribute.
        result.is_sound = score >= 4
        # Linear scale into our reward range, identical in form to the aevom
        # (total_score / max_score) * reward_scale mapping.
        result.reward = (score / 5.0) * self.reward_scale if score else 0.0
        return result

    def _parse_aevom(self, response_text: str) -> JudgeResult:
        """Parse the 6-category Aevom rubric response.

        Uses LAST-MATCH for both score patterns and verdict markers. Reasoning
        models emit chain-of-thought before the formal answer, and the CoT can
        contain tentative scores ("LOGICAL_CONSISTENCY: 3 -- wait, 2") and
        stray verdict markers ("could be SOUND if X, but FLAWED because Y").
        Picking the LAST occurrence ignores those rehearsals and locks onto
        the model's final commitment. _call_judge concatenates
        ``reasoning_content + content`` so the formal answer is at the end of
        the text; the truncation case (content empty, reasoning has the
        answer) is also handled because the last marker emitted before
        truncation is still the latest commitment.

        Pre-2026-05-28 the parser used re.search (first-match), which on
        long-reasoning model outputs silently flipped the verdict and locked
        in tentative scores. See v0528a postmortem in
        ``internal_docs/docs/training_scaling_curve.md``.
        """
        result = JudgeResult(raw_response=response_text)

        # Extract scores: last occurrence of each label wins.
        for name, pattern in self.prompts.score_patterns.items():
            matches = list(re.finditer(pattern, response_text))
            if matches:
                try:
                    score = int(matches[-1].group(1))
                    setattr(result, name, score)
                except (ValueError, AttributeError):
                    pass

        # Calculate total score
        result.total_score = (
            result.logical_consistency +
            result.differentiating_metric +
            result.hypothesis_grounded +
            result.scientific_accuracy +
            result.edit_tests_hypothesis +
            result.diagnostic_quality
        )

        # Check verdict: pick whichever marker appears LATER in the text.
        # If only one type appears, it wins.
        sound_matches = list(re.finditer(self.prompts.sound_verdict_pattern, response_text))
        flawed_matches = list(re.finditer(self.prompts.flawed_verdict_pattern, response_text))
        if sound_matches and flawed_matches:
            result.is_sound = sound_matches[-1].start() > flawed_matches[-1].start()
        elif sound_matches:
            result.is_sound = True
        elif flawed_matches:
            result.is_sound = False
        else:
            # Infer from score: SOUND requires >=11/16 and no zero in the
            # five 0-3 categories (Diagnostic Quality may be 0).
            result.is_sound = (
                result.total_score >= 11 and
                min(
                    result.logical_consistency,
                    result.differentiating_metric,
                    result.hypothesis_grounded,
                    result.scientific_accuracy,
                    result.edit_tests_hypothesis,
                ) > 0
            )
        
        # Extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=VERDICT:|$)",
            response_text,
            re.DOTALL,
        )
        if reasoning_match:
            result.reasoning = reasoning_match.group(1).strip()
        
        # Compute reward
        result.reward = (result.total_score / result.max_score) * self.reward_scale
        
        return result
    
    async def evaluate(
        self,
        trajectory: list[dict],
        response: Any,
        ground_truth: dict,
        reasoning_traces: Optional[list[dict]] = None,
        parent_span: Optional[Span] = None,
    ) -> dict:
        """Evaluate a trajectory using the LLM judge.
        
        Args:
            trajectory: List of tool call records.
            response: The model's final response.
            ground_truth: Ground truth information.
            reasoning_traces: Optional per-turn reasoning from ``<think>`` blocks.
            parent_span: Optional OTel span to nest judge spans under.
                Falls back to the current chat span ContextVar if not provided.
            
        Returns:
            Dictionary with evaluation results.
        """
        from api.telemetry import get_tracer, get_current_chat_span, set_io, set_span_kind

        tracer = get_tracer()
        parent = parent_span or get_current_chat_span()
        ctx = trace.set_span_in_context(parent) if parent else None

        with tracer.start_as_current_span("judge_evaluation", context=ctx) as judge_span:
            judge_span_id = format(judge_span.get_span_context().span_id, "016x")
            # OpenInference: render as an EVALUATOR span. Additive -- all
            # judge.* attributes below are preserved for profile_run.py.
            set_span_kind(judge_span, "EVALUATOR")
            if self.endpoints:
                enabled_names = [ep.name for ep in self.endpoints if ep.enabled]
                judge_span.set_attribute("judge.pool_size", len(enabled_names))
                judge_span.set_attribute("judge.pool_endpoints", ",".join(enabled_names))
                judge_span.set_attribute("judge.mode", "pool")
            else:
                judge_span.set_attribute("judge.mode", "single")
                judge_span.set_attribute("judge.model", self.model_name)
            judge_span.set_attribute("judge.model_url", self.model_url)
            judge_span.set_attribute("judge.reward_scale", self.reward_scale)
            judge_span.set_attribute("judge.score_format", self.score_format)
            judge_span.set_attribute(
                "judge.ground_truth.sabotage_type",
                ground_truth.get("sabotage_type", "unknown"),
            )

            prompt = self._build_prompt(
                trajectory, response, ground_truth,
                reasoning_traces=reasoning_traces,
            )
            set_io(judge_span, input_value=prompt)

            try:
                result = await self._call_judge(prompt)
            except Exception as e:
                logger.error(f"Judge evaluation failed: {e}")
                judge_span.set_status(Status(StatusCode.ERROR, str(e)))
                judge_span.record_exception(e)
                return {
                    "is_sound": False,
                    "reward": 0.0,
                    "error": str(e),
                    "span_id": judge_span_id,
                }

            judge_span.set_attribute("judge.verdict", "SOUND" if result.is_sound else "FLAWED")
            judge_span.set_attribute("judge.total_score", result.total_score)
            judge_span.set_attribute("judge.max_score", result.max_score)
            judge_span.set_attribute("judge.reward", result.reward)
            judge_span.set_attribute("judge.reasoning", result.reasoning)
            judge_span.set_attribute("judge.raw_response", result.raw_response)
            judge_span.set_attribute("judge.score.logical_consistency", result.logical_consistency)
            judge_span.set_attribute("judge.score.differentiating_metric", result.differentiating_metric)
            judge_span.set_attribute("judge.score.hypothesis_grounded", result.hypothesis_grounded)
            judge_span.set_attribute("judge.score.scientific_accuracy", result.scientific_accuracy)
            judge_span.set_attribute("judge.score.edit_tests_hypothesis", result.edit_tests_hypothesis)
            judge_span.set_attribute("judge.score.diagnostic_quality", result.diagnostic_quality)
            set_io(
                judge_span,
                output_value=(
                    f"{'SOUND' if result.is_sound else 'FLAWED'} "
                    f"(reward={result.reward})\n\n{result.reasoning}"
                ),
            )
            judge_span.set_status(Status(StatusCode.OK))

            return {
                "is_sound": result.is_sound,
                "reward": result.reward,
                "total_score": result.total_score,
                "max_score": result.max_score,
                "scores": {
                    "logical_consistency": result.logical_consistency,
                    "differentiating_metric": result.differentiating_metric,
                    "hypothesis_grounded": result.hypothesis_grounded,
                    "scientific_accuracy": result.scientific_accuracy,
                    "edit_tests_hypothesis": result.edit_tests_hypothesis,
                    "diagnostic_quality": result.diagnostic_quality,
                },
                "reasoning": result.reasoning,
                "raw_response": result.raw_response,
                "span_id": judge_span_id,
            }
    
    async def _call_judge(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> JudgeResult:
        """Call the judge LLM with retry on transient failures.

        Retries with exponential backoff on 429 (rate limit) and
        502/503/504 (gateway/unavailable) responses.

        Creates an ``llm.complete`` child span (inherits from the
        ``judge_evaluation`` parent set by ``evaluate()``).
        """
        import asyncio
        from api.telemetry import (
            get_tracer,
            set_input_messages,
            set_output_message,
            set_span_kind,
        )

        tracer = get_tracer()

        with tracer.start_as_current_span("llm.complete") as llm_span:
            llm_span.set_attribute("llm.temperature", 0.0)
            llm_span.set_attribute("llm.max_tokens", 4096)
            llm_span.set_attribute("llm.role", "judge")
            llm_span.set_attribute("llm.input.system_prompt", self.prompts.system_prompt)
            llm_span.set_attribute("llm.input.user_prompt", prompt)
            # OpenInference: render the judge LLM call natively. Additive --
            # the llm.input.* / llm.raw.* attrs are preserved.
            set_span_kind(llm_span, "LLM")
            set_input_messages(
                llm_span,
                [
                    {"role": "system", "content": self.prompts.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            in_pool_mode = bool(self.endpoints)
            if not in_pool_mode:
                llm_span.set_attribute("llm.model", self.model_name)

            retryable_codes = {429, 502, 503, 504}
            last_exc: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                if in_pool_mode:
                    endpoint = self._pick_endpoint()
                    chosen_url = endpoint.url
                    chosen_model = endpoint.model
                    chosen_api_key = endpoint.api_key
                    chosen_name = endpoint.name
                else:
                    chosen_url = self._pick_url()
                    chosen_model = self.model_name
                    chosen_api_key = self.api_key
                    chosen_name = self.model_name

                headers = {"Content-Type": "application/json"}
                if chosen_api_key:
                    headers["Authorization"] = f"Bearer {chosen_api_key}"

                payload = {
                    "model": chosen_model,
                    "messages": [
                        {"role": "system", "content": self.prompts.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    # 1024 was insufficient: even nemotron-3-super-v3 occasionally
                    # hit finish_reason="length" before emitting the formal answer
                    # in `content`, and nemotron-3-ultra-preview (added 2026-05-28)
                    # routinely emits ~3000 reasoning tokens on judge prompts. At
                    # 4096 the smoke test showed all pool members finish cleanly
                    # with finish_reason="stop". See v0528a postmortem in
                    # internal_docs/docs/training_scaling_curve.md.
                    "max_tokens": 4096,
                }

                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            f"{chosen_url}/chat/completions",
                            headers=headers,
                            json=payload,
                        )

                        if response.status_code in retryable_codes and attempt < max_retries:
                            retry_after = float(
                                response.headers.get("retry-after", base_delay * (2 ** attempt))
                            )
                            logger.warning(
                                f"Judge got HTTP {response.status_code} from {chosen_name}, "
                                f"retrying in {retry_after:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            llm_span.add_event(
                                "retry",
                                {
                                    "attempt": attempt + 1,
                                    "status_code": response.status_code,
                                    "endpoint_name": chosen_name,
                                },
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        response.raise_for_status()

                        data = response.json()
                        msg = data["choices"][0]["message"]
                        content = msg.get("content") or ""
                        reasoning_content = msg.get("reasoning_content") or ""

                        # Reasoning models (Nemotron Super / Ultra, Qwen 3.6,
                        # GLM 5.1) return the full chain-of-thought in
                        # reasoning_content and a short formal answer in
                        # content. We concatenate reasoning + content so the
                        # text follows the model's natural temporal flow
                        # (deliberation, then commitment). The parser then
                        # uses LAST-MATCH for both score patterns and verdict
                        # markers (see _parse_response): the final occurrence
                        # of any pattern is the model's commitment, which
                        # wins over earlier tentative scores ("3 -- wait, 2")
                        # or stray verdict markers in CoT ("could be SOUND if
                        # X, but FLAWED because Y"). The pre-2026-05-28
                        # layout was reasoning + content with FIRST-match,
                        # which leaked CoT into the parsed answer.
                        if reasoning_content and content:
                            response_text = reasoning_content + "\n" + content
                        else:
                            response_text = content or reasoning_content

                        llm_span.set_attribute("llm.raw.content", content)
                        llm_span.set_attribute("llm.raw.reasoning", reasoning_content or content)
                        llm_span.set_attribute("llm.has_reasoning", bool(reasoning_content))
                        llm_span.set_attribute("llm.endpoint_url", chosen_url)
                        llm_span.set_attribute("llm.model", chosen_model)
                        llm_span.set_attribute("llm.model_name", chosen_model)
                        llm_span.set_attribute("judge.endpoint.name", chosen_name)
                        set_output_message(
                            llm_span,
                            reasoning=reasoning_content or None,
                            content=content or None,
                        )
                        if attempt > 0:
                            llm_span.set_attribute("llm.retries", attempt)
                        llm_span.set_status(Status(StatusCode.OK))

                        return self._parse_response(response_text)

                except httpx.TimeoutException as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Judge request to {chosen_name} timed out, retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        llm_span.add_event(
                            "timeout_retry",
                            {
                                "attempt": attempt + 1,
                                "error": str(e),
                                "endpoint_name": chosen_name,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue
                    llm_span.set_status(Status(StatusCode.ERROR, str(e)))
                    llm_span.record_exception(e)
                    raise

            error = last_exc or RuntimeError("Judge call failed after retries")
            llm_span.set_status(Status(StatusCode.ERROR, str(error)))
            llm_span.record_exception(error)
            raise error


# =============================================================================
# Convenience Functions
# =============================================================================

async def evaluate_reasoning(
    trajectory: list[dict],
    response: Any,
    ground_truth: dict,
    model_url: str = "http://localhost:8000/v1",
    config_path: Optional[str] = None,
    reasoning_traces: Optional[list[dict]] = None,
    parent_span: Optional[Span] = None,
) -> dict:
    """Convenience function to evaluate reasoning quality.
    
    Args:
        trajectory: List of tool call records.
        response: Model response.
        ground_truth: Ground truth information.
        model_url: URL for judge LLM.
        config_path: Optional path to judge config.
        reasoning_traces: Optional per-turn reasoning from ``<think>`` blocks.
        parent_span: Optional OTel span to nest judge spans under.
        
    Returns:
        Evaluation result dictionary.
    """
    if config_path:
        judge = LLMJudge.from_config(config_path, model_url=model_url)
    else:
        prompts = JudgePrompts()
        judge = LLMJudge(prompts=prompts, model_url=model_url)
    
    return await judge.evaluate(
        trajectory, response, ground_truth,
        reasoning_traces=reasoning_traces,
        parent_span=parent_span,
    )


class MockJudge:
    """Mock judge for testing without LLM (advanced-mode rubric, /16)."""

    def __init__(self, default_score: int = 11, is_sound: bool = True):
        self.default_score = default_score
        self.is_sound = is_sound

    async def evaluate(
        self,
        trajectory: list[dict],
        response: Any,
        ground_truth: dict,
        reasoning_traces: Optional[list[dict]] = None,
        parent_span: Optional[Span] = None,
    ) -> dict:
        """Return mock evaluation under the advanced-mode 6-category rubric."""
        return {
            "is_sound": self.is_sound,
            "reward": self.default_score / 16 * 0.3,
            "total_score": self.default_score,
            "max_score": 16,
            "scores": {
                "logical_consistency": 2,
                "differentiating_metric": 2,
                "hypothesis_grounded": 2,
                "scientific_accuracy": 2,
                "edit_tests_hypothesis": 2,
                "diagnostic_quality": 1,
            },
            "reasoning": "Mock evaluation",
            "raw_response": "",
        }
