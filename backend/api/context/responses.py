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

"""The degradation ladder applied to Responses API input items.

Production assembles LangChain messages; training's NeMo-Gym rollout loop
carries an OpenAI Responses ``input`` array instead. The two shapes differ but
the policy must not: whatever production hides from the model at a given level
of context pressure, training has to hide too, or the trained observation
distribution stops matching the deployed one.

So this module reuses the same transforms and the same oldest-first order as
:mod:`api.context.assembler`, applied to the other shape.

Two things make this side simpler. Training episodes are single-user-turn --
advanced mode front-loads all four steps into the initial prompt -- so there
is no earlier turn to summarize and the ladder stops at level 2. And because
it only ever rewrites a ``function_call_output`` in place, never removing an
item, a call can never lose its result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from api.context.assembler import (
    LEVEL_COLLAPSED,
    LEVEL_PREVIEW,
    LEVEL_VERBATIM,
    _one_line,
    _preview,
)
from api.context.budget import _TOKENS_PER_MESSAGE, estimate_tokens_from_chars

logger = logging.getLogger(__name__)


@dataclass
class DegradeReport:
    """What degrading the input array cost."""

    level: int = LEVEL_VERBATIM
    degraded_messages: int = 0
    estimated_tokens: int = 0
    available_tokens: int = 0
    still_over_budget: bool = False


def _item_chars(item: Any) -> int:
    """Rough serialized size of one input item."""
    total = 0
    for attr in ("content", "output", "arguments", "text", "summary"):
        value = getattr(item, attr, None)
        if value is None and isinstance(item, dict):
            value = item.get(attr)
        if value is not None:
            total += len(value) if isinstance(value, str) else len(str(value))
    return total


def _item_tokens(item: Any) -> int:
    return _TOKENS_PER_MESSAGE + estimate_tokens_from_chars(_item_chars(item))


def _output_text(item: Any) -> Optional[str]:
    """The result text of a function_call_output, or None if not one."""
    if isinstance(item, dict):
        if item.get("type") == "function_call_output":
            output = item.get("output")
            return output if isinstance(output, str) else None
        return None

    if getattr(item, "type", None) != "function_call_output":
        return None
    output = getattr(item, "output", None)
    return output if isinstance(output, str) else None


def _set_output_text(item: Any, text: str) -> None:
    if isinstance(item, dict):
        item["output"] = text
    else:
        item.output = text


def degrade_responses_input(
    input_items: list[Any],
    available_tokens: Optional[int],
) -> DegradeReport:
    """Shrink tool outputs in ``input_items`` in place until they fit.

    Args:
        input_items: The Responses API ``input`` array, mutated in place.
        available_tokens: Prompt budget, or None to leave the array untouched.

    Returns:
        What was degraded, for telemetry.
    """
    total = sum(_item_tokens(item) for item in input_items)
    report = DegradeReport(
        estimated_tokens=total, available_tokens=available_tokens or 0
    )
    if available_tokens is None or total <= available_tokens:
        return report

    outputs = [
        index for index, item in enumerate(input_items) if _output_text(item) is not None
    ]

    for level, transform in ((LEVEL_PREVIEW, _preview), (LEVEL_COLLAPSED, _one_line)):
        for index in outputs:
            item = input_items[index]
            before = _item_tokens(item)
            _set_output_text(item, transform(_output_text(item) or ""))
            total -= before - _item_tokens(item)

            report.level = level
            report.degraded_messages += 1
            report.estimated_tokens = total
            if total <= available_tokens:
                return report

    report.still_over_budget = True
    logger.warning(
        "[aevom-context] Rollout prompt is still ~%d tokens against %d available "
        "after collapsing every tool output. The prompt itself, not the "
        "trajectory, is over budget.",
        total,
        available_tokens,
    )
    return report
