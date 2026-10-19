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

"""Token budgeting for LLM prompts.

``ModelConfig.max_tokens`` has always been the *generation* limit; nothing in
the system knew how large a prompt was allowed to be. The result was that
context overflow surfaced as a provider 400 with no diagnostics beyond a
message count, which is why ``LLMClient`` logs one defensively.

This module gives that pressure a number. It estimates prompt size before the
call, compares it against ``context_window - max_tokens - margin``, and warns
on breach. It deliberately does not truncate: Stage 1 is about learning
whether real sessions actually reach the ceiling, and silent degradation would
destroy the evidence.

The estimator is character-based on purpose. A real tokenizer would mean
loading per-model vocabularies into the request path for a number that only
feeds a warning; ``context.tokens.estimate_error`` telemetry compares each
estimate against the provider's own ``prompt_tokens`` so the constant can be
tuned against evidence instead of assumption.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Characters per token. Declared here rather than alongside the tool response
# cap so that this package stays free of any api.* import and can be depended
# on from anywhere in the request path.
#
# Calibrated, not guessed. The obvious value of 4.0 measured at an
# ``estimate_ratio`` of ~0.885 against Nemotron 3 Super on real agent
# trajectories -- an 11% under-count, which is the dangerous direction for a
# budget, since it fails to warn exactly when the prompt is largest. 3.5 puts
# the ratio just above 1.0, so the budget errs toward warning early.
# ``context.tokens.estimate_ratio`` is the telemetry that keeps this honest;
# re-tune if it drifts below 1.0 on the models we actually run.
CHARS_PER_TOKEN = 3.5

# Headroom between the estimated prompt and the hard window. Absorbs chat
# template scaffolding, the provider's own tool-schema serialization, and
# estimator error. Cheap insurance: the cost of being wrong in this direction
# is a spurious warning, the cost of being wrong in the other is a 400.
CONTEXT_SAFETY_MARGIN_TOKENS = 2048

# Per-message wire overhead (role, delimiters, tool_call_id and friends).
# Roughly what the OpenAI cookbook uses for chat message framing.
_TOKENS_PER_MESSAGE = 4


@dataclass(frozen=True)
class BudgetReport:
    """The result of measuring one prompt against one model's window.

    Attributes:
        estimated_tokens: Estimated size of the prompt, messages plus tools.
        available_tokens: What the prompt was allowed to be.
        context_window: The model's declared total capacity.
        reserved_output_tokens: Capacity held back for generation.
        message_count: How many messages were measured, useful when
            correlating a breach with a long tool-calling turn.
    """

    estimated_tokens: int
    available_tokens: int
    context_window: int
    reserved_output_tokens: int
    message_count: int

    @property
    def over_budget(self) -> bool:
        return self.estimated_tokens > self.available_tokens

    @property
    def headroom_tokens(self) -> int:
        return self.available_tokens - self.estimated_tokens

    @property
    def utilization(self) -> float:
        """Fraction of the available budget consumed, 0.0 when unbounded."""
        if self.available_tokens <= 0:
            return 0.0
        return self.estimated_tokens / self.available_tokens


def estimate_tokens_from_chars(char_count: int) -> int:
    """Estimate tokens from a character count already measured."""
    if char_count <= 0:
        return 0
    return math.ceil(char_count / CHARS_PER_TOKEN)


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of a string."""
    return estimate_tokens_from_chars(len(text) if text else 0)


def _content_length(content: Any) -> int:
    """Character length of a message's content, whatever shape it takes.

    Content is usually a string, but multimodal messages carry a list of
    parts and tool calls carry structured arguments. Serializing non-strings
    over-counts slightly, which is the safe direction here.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, default=str))
    except (TypeError, ValueError):
        return len(str(content))


def estimate_messages_tokens(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
) -> int:
    """Estimate the prompt cost of a message list and its tool schemas.

    Tool schemas are counted because they are resent on every single call and
    are a large fixed floor in this system -- roughly forty tools, each with a
    full JSON Schema. Omitting them would make every estimate optimistic by a
    constant that grows every time a tool is added.

    Args:
        messages: OpenAI-format chat messages.
        tools: OpenAI-format tool schemas sent alongside them.

    Returns:
        Estimated prompt tokens.
    """
    total = 0
    for message in messages:
        total += _TOKENS_PER_MESSAGE
        total += estimate_tokens_from_chars(_content_length(message.get("content")))
        if message.get("reasoning_content"):
            total += estimate_tokens_from_chars(
                _content_length(message["reasoning_content"])
            )
        if message.get("tool_calls"):
            total += estimate_tokens_from_chars(
                _content_length(message["tool_calls"])
            )

    if tools:
        total += estimate_tokens_from_chars(_content_length(tools))

    return total


def check_context_budget(
    config: Any,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    max_output_tokens: Optional[int] = None,
    margin_tokens: int = CONTEXT_SAFETY_MARGIN_TOKENS,
) -> BudgetReport:
    """Measure a prompt against a model's context window and warn on breach.

    Args:
        config: The :class:`ModelConfig` the call will use.
        messages: The messages about to be sent.
        tools: The tool schemas about to be sent.
        max_output_tokens: Generation limit for this call, defaulting to the
            model's configured ``max_tokens``.
        margin_tokens: Headroom held back on top of the generation reserve.

    Returns:
        A :class:`BudgetReport`. Callers decide what to do with it; at Stage 1
        that means recording it on a span and otherwise proceeding unchanged.
    """
    context_window = getattr(config, "context_window", 0) or 0
    reserved = max_output_tokens or getattr(config, "max_tokens", 0) or 0
    available = max(0, context_window - reserved - margin_tokens)

    report = BudgetReport(
        estimated_tokens=estimate_messages_tokens(messages, tools),
        available_tokens=available,
        context_window=context_window,
        reserved_output_tokens=reserved,
        message_count=len(messages),
    )

    if report.over_budget:
        logger.warning(
            "[AEVOM-CONTEXT-BUDGET] %s prompt is over budget: ~%d tokens "
            "estimated against %d available (window %d, reserved for output "
            "%d, margin %d) across %d messages. The provider will likely "
            "reject this call with a 400.",
            getattr(config, "id", "unknown"),
            report.estimated_tokens,
            report.available_tokens,
            report.context_window,
            report.reserved_output_tokens,
            margin_tokens,
            report.message_count,
        )
    elif report.utilization >= 0.8:
        logger.info(
            "[AEVOM-CONTEXT-BUDGET] %s prompt at %.0f%% of budget "
            "(~%d of %d tokens, %d messages).",
            getattr(config, "id", "unknown"),
            report.utilization * 100,
            report.estimated_tokens,
            report.available_tokens,
            report.message_count,
        )

    return report


def record_budget_on_span(span: Any, report: BudgetReport) -> None:
    """Attach a budget report to a span.

    Kept separate from :func:`check_context_budget` so the measurement stays
    usable outside a traced path (tests, training rollouts with tool spans
    suppressed).
    """
    if span is None:
        return
    try:
        span.set_attribute("context.tokens.used", report.estimated_tokens)
        span.set_attribute("context.tokens.budget", report.available_tokens)
        span.set_attribute("context.tokens.window", report.context_window)
        span.set_attribute(
            "context.tokens.reserved_output", report.reserved_output_tokens
        )
        span.set_attribute("context.tokens.headroom", report.headroom_tokens)
        span.set_attribute("context.tokens.utilization", round(report.utilization, 4))
        span.set_attribute("context.over_budget", report.over_budget)
        span.set_attribute("context.messages.count", report.message_count)
    except Exception:
        logger.debug("Failed to record context budget on span", exc_info=True)


def record_estimate_error(span: Any, estimated: int, actual: int) -> None:
    """Record how far the estimate was from the provider's own count.

    This is the calibration loop. ``actual`` comes from ``usage.prompt_tokens``,
    which the provider computes with the real tokenizer; the ratio tells us
    whether CHARS_PER_TOKEN is right for the models we actually run.

    Args:
        span: The LLM span, or None.
        estimated: What :func:`estimate_messages_tokens` predicted.
        actual: The provider's ``prompt_tokens``, or 0 when not reported.
    """
    if span is None or not actual:
        return
    try:
        span.set_attribute("context.tokens.actual", actual)
        span.set_attribute("context.tokens.estimate_error", estimated - actual)
        span.set_attribute(
            "context.tokens.estimate_ratio", round(estimated / actual, 4)
        )
    except Exception:
        logger.debug("Failed to record estimate error on span", exc_info=True)
