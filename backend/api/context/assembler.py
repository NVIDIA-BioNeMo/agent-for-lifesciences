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

"""Assemble a prompt from the conversation under a declared token budget.

Context assembly used to be implicit: every message the frontend still held
was sent, and the only thing standing between a long session and a provider
400 was luck. This module makes it an explicit, measured step.

Fill order is a floor first, then the newest content:

1. The floor -- system prompt, tool schemas, state envelope, session log.
   Never degraded. The session log lives here for a reason: it holds every
   number the session measured, so degrading the trajectory below cannot lose
   a measurement, only the prose around it.
2. The current user message and the turn in progress.
3. Earlier turns, newest to oldest.

When that does not fit, content is *degraded* rather than dropped:

- Level 0: the message verbatim.
- Level 1: tool results replaced by a preview plus their artifact reference.
- Level 2: a call and its results collapsed to one line each.
- Level 3: a whole turn reduced to its findings and its final assistant text.

Degradation always operates on a whole assistant message together with the
tool results answering it. Orphaning a tool result from its call produces a
provider 400 that is much harder to diagnose than the overflow it replaced --
the error names neither the message nor the tool.

Why there is no LLM-written compaction below level 3
----------------------------------------------------

The plan made anchored compaction conditional on this ladder proving
insufficient. Measured against ``nemotron-3-super-rc-nim`` (131k window, 32k
reserved for generation) with a worst case of every turn running all 20
iterations and every tool result saturating its 2000-token cap:

    turns    level   estimated   available   over budget
        1        0      52,328      75,485   no
        2        1      74,816      75,485   no
       10        3      67,216      75,485   no
      100        3      74,108      75,485   no
      150        3      74,158      75,485   no
      200        3      78,208      75,485   yes

Real sessions measured in Phoenix sit at 22% of budget. The ladder therefore
holds well past any plausible conversation, and the failure point is 150
maximal turns rather than the handful the unbounded design managed.

Compaction would also buy less than it looks. At level 3 a turn is already
reduced to its findings plus its closing sentence, and findings are exactly
what anchored compaction preserves verbatim; the only thing left to compress
is one sentence of narrative per turn. ``context.still_over_budget`` is the
signal that this conclusion has expired.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from api.context.budget import (
    CONTEXT_SAFETY_MARGIN_TOKENS,
    _TOKENS_PER_MESSAGE,
    estimate_tokens_from_chars,
)

logger = logging.getLogger(__name__)

# How much of a tool result survives at level 1. Enough to keep the shape of
# the answer -- a header, the first few fields -- without the payload.
PREVIEW_CHARS = 400

# Same reference shape the ledger and python_execute scan for.
_REF_PATTERN = re.compile(r"\b([a-z]+(?:_[a-z]+)*_[a-f0-9]{8})\b")

LEVEL_VERBATIM = 0
LEVEL_PREVIEW = 1
LEVEL_COLLAPSED = 2
LEVEL_SUMMARY = 3


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    return len(str(content))


def _message_tokens(message: BaseMessage) -> int:
    """Estimate one message's cost, reasoning and tool calls included."""
    total = _TOKENS_PER_MESSAGE
    total += estimate_tokens_from_chars(_content_chars(message.content))

    reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning")
    if reasoning:
        total += estimate_tokens_from_chars(_content_chars(reasoning))

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        total += estimate_tokens_from_chars(_content_chars(tool_calls))

    return total


def _messages_tokens(messages: list[BaseMessage]) -> int:
    return sum(_message_tokens(message) for message in messages)


def _refs_in(text: str) -> list[str]:
    seen: list[str] = []
    for ref in _REF_PATTERN.findall(text or ""):
        if ref not in seen:
            seen.append(ref)
    return seen


def _preview(text: str) -> str:
    """Head of a tool result plus a pointer to the whole thing.

    The reference matters more than the text. A model that can see
    ``pdb_c3c3c3c3`` can pass it to another tool or read it with
    ``python_execute``; a model shown four hundred characters of a PDB file
    can do neither.
    """
    text = text or ""
    refs = _refs_in(text)
    if len(text) <= PREVIEW_CHARS:
        return text

    parts = [text[:PREVIEW_CHARS].rstrip(), "...[truncated to fit the context budget]"]
    if refs:
        parts.append(
            f"Full output is stored as {', '.join(refs)}; read it with python_execute."
        )
    return "\n".join(parts)


def _one_line(text: str) -> str:
    """Collapse a tool result to a single line.

    Safe to be this aggressive because the [SESSION LOG] block in the floor
    holds the scalars this call produced, verbatim and undegraded. What is
    dropped here is prose, not measurement.
    """
    text = (text or "").strip()
    first_line = text.split("\n", 1)[0].strip() if text else ""
    refs = _refs_in(text)

    line = first_line[:160] if first_line else "(result elided)"
    if refs:
        line += f" [{', '.join(refs)}]"
    return line


@dataclass
class Exchange:
    """One assistant message and the tool results answering it.

    The atomic unit of degradation. These never come apart: a tool result
    without its call, or a call without its result, is a provider 400.
    """

    assistant: AIMessage
    results: list[ToolMessage] = field(default_factory=list)
    _rendered: dict = field(default_factory=dict, repr=False, compare=False)

    def render(self, level: int) -> list[BaseMessage]:
        """Render at a level, memoized.

        The fit loop asks for the same level of the same exchange repeatedly,
        and tool results are large enough that re-transforming them dominates
        assembly. Caching keeps a long session's assembly linear.
        """
        if not self.results:
            level = LEVEL_VERBATIM

        cached = self._rendered.get(level)
        if cached is not None:
            return cached

        if level <= LEVEL_VERBATIM:
            rendered: list[BaseMessage] = [self.assistant, *self.results]
        else:
            transform = _preview if level == LEVEL_PREVIEW else _one_line
            rendered = [
                self.assistant,
                *[
                    ToolMessage(
                        content=transform(str(result.content)),
                        tool_call_id=result.tool_call_id,
                        name=result.name,
                    )
                    for result in self.results
                ],
            ]

        self._rendered[level] = rendered
        return rendered

    def tokens(self, level: int) -> int:
        return _messages_tokens(self.render(level))

    def finding_lines(self) -> list[str]:
        """One line per tool call, for when the whole turn is summarized."""
        lines = []
        for call, result in zip(self.assistant.tool_calls or [], self.results):
            lines.append(f"- {call.get('name', 'tool')}: {_one_line(str(result.content))}")
        return lines


@dataclass
class Turn:
    """A user message and everything the agent did in response."""

    user: Optional[HumanMessage]
    exchanges: list[Exchange] = field(default_factory=list)
    trailing: list[BaseMessage] = field(default_factory=list)
    _summary: Optional[list] = field(default=None, repr=False, compare=False)

    def render(self, levels: dict[int, int], turn_level: int) -> list[BaseMessage]:
        if turn_level >= LEVEL_SUMMARY:
            if self._summary is None:
                self._summary = self._render_summary()
            return self._summary

        rendered: list[BaseMessage] = []
        if self.user is not None:
            rendered.append(self.user)
        for index, exchange in enumerate(self.exchanges):
            rendered.extend(exchange.render(levels.get(index, LEVEL_VERBATIM)))
        rendered.extend(self.trailing)
        return rendered

    def fixed_tokens(self) -> int:
        """Cost of the parts of a turn that no level changes."""
        fixed = [m for m in ([self.user] if self.user else []) + self.trailing]
        return _messages_tokens(fixed)

    def summary_tokens(self) -> int:
        if self._summary is None:
            self._summary = self._render_summary()
        return _messages_tokens(self._summary)

    def _render_summary(self) -> list[BaseMessage]:
        """Reduce the turn to what it established plus how it answered.

        No tool calls survive, so there is nothing left to orphan. The
        assistant's closing text is kept because it is what the user actually
        saw, and the model's later references point at it.
        """
        rendered: list[BaseMessage] = []
        if self.user is not None:
            rendered.append(self.user)

        lines: list[str] = []
        for exchange in self.exchanges:
            lines.extend(exchange.finding_lines())

        closing = " ".join(
            str(message.content) for message in self.trailing if message.content
        ).strip()

        parts = []
        if lines:
            parts.append("[EARLIER TURN, CONDENSED]\n" + "\n".join(lines))
        if closing:
            parts.append(closing)

        if parts:
            rendered.append(AIMessage(content="\n\n".join(parts)))
        return rendered


def split_into_turns(messages: list[BaseMessage]) -> list[Turn]:
    """Group a flat message list into turns and tool-call exchanges.

    Anything that arrives orphaned -- a tool result with no preceding call --
    is discarded here rather than passed on, since it would be rejected at the
    provider with a far less informative error.
    """
    turns: list[Turn] = []
    current: Optional[Turn] = None

    for message in messages:
        if isinstance(message, HumanMessage):
            current = Turn(user=message)
            turns.append(current)
            continue

        if current is None:
            current = Turn(user=None)
            turns.append(current)

        if isinstance(message, AIMessage):
            if message.tool_calls:
                current.exchanges.append(Exchange(assistant=message))
            else:
                current.trailing.append(message)
            continue

        if isinstance(message, ToolMessage):
            if current.exchanges:
                current.exchanges[-1].results.append(message)
            else:
                logger.debug("Discarding an orphaned tool result during assembly")
            continue

        current.trailing.append(message)

    return turns


@dataclass
class AssemblyReport:
    """What the assembler had to do to make the prompt fit."""

    level: int = LEVEL_VERBATIM
    degraded_messages: int = 0
    estimated_tokens: int = 0
    available_tokens: int = 0
    still_over_budget: bool = False


class ContextAssembler:
    """Render the conversation into a prompt that fits the model's window.

    Example:
        assembler = ContextAssembler(system_prompt, config=client.config)
        messages = assembler.assemble(state)
    """

    def __init__(
        self,
        base_system_prompt: str,
        config: Any = None,
        tool_schemas: Optional[list[dict[str, Any]]] = None,
        margin_tokens: int = CONTEXT_SAFETY_MARGIN_TOKENS,
    ):
        self._base_system_prompt = base_system_prompt
        self._config = config
        self._margin_tokens = margin_tokens
        self._tool_tokens = (
            estimate_tokens_from_chars(_content_chars(tool_schemas))
            if tool_schemas
            else 0
        )
        self.last_report = AssemblyReport()

    def _available_tokens(self) -> Optional[int]:
        """What the prompt is allowed to cost, generation and tools reserved.

        ``None`` means the model declared no window, so there is no budget to
        enforce and assembly degenerates to the previous behaviour: build the
        prompt and send it. That is distinct from a budget of zero, which
        means the floor alone has already consumed the window -- a real
        problem worth degrading and reporting.
        """
        window = getattr(self._config, "context_window", 0) or 0
        if not window:
            return None
        reserved = getattr(self._config, "max_tokens", 0) or 0
        return max(0, window - reserved - self._margin_tokens - self._tool_tokens)

    def _build_system_message(self, state: dict) -> SystemMessage:
        from api.state.format import inject_state_into_prompt

        return SystemMessage(
            content=inject_state_into_prompt(
                self._base_system_prompt,
                state.get("viewer_state") or {},
                state.get("storage_state") or {},
                state.get("selections") or [],
                findings=state.get("findings") or [],
                tool_budgets=state.get("tool_budgets") or [],
                candidates=state.get("candidates") or [],
                exclusions=state.get("exclusions") or [],
                nim_notices=state.get("nim_notices") or [],
            )
        )

    def assemble(self, state: dict) -> list[BaseMessage]:
        """Return the messages to send, degrading older content if needed."""
        system = self._build_system_message(state)
        turns = split_into_turns(
            [m for m in state.get("messages") or [] if not isinstance(m, SystemMessage)]
        )

        budget = self._available_tokens()
        available = None if budget is None else max(0, budget - _message_tokens(system))
        rendered, report = self._fit(turns, available)
        self.last_report = report

        return [system, *rendered]

    def _fit(
        self, turns: list[Turn], available: Optional[int]
    ) -> tuple[list[BaseMessage], AssemblyReport]:
        """Degrade oldest-first until the trajectory fits, or cannot shrink.

        Each step adjusts a running total by the difference the step makes,
        rather than re-measuring the whole prompt. Re-measuring is quadratic
        in turns and, with tool results running to tens of kilobytes, that is
        the difference between assembly costing microseconds and seconds.
        """
        levels: list[dict[int, int]] = [{} for _ in turns]
        turn_levels: list[int] = [LEVEL_VERBATIM for _ in turns]

        def render() -> list[BaseMessage]:
            out: list[BaseMessage] = []
            for index, turn in enumerate(turns):
                out.extend(turn.render(levels[index], turn_levels[index]))
            return out

        total = sum(
            turn.fixed_tokens() + sum(e.tokens(LEVEL_VERBATIM) for e in turn.exchanges)
            for turn in turns
        )
        report = AssemblyReport(
            estimated_tokens=total,
            available_tokens=available or 0,
        )
        if available is None or total <= available:
            return render(), report

        # The turn in progress carries the user's actual request and the work
        # answering it. Everything before it degrades first.
        degradable = list(range(max(0, len(turns) - 1)))

        def apply(turn_index: int, exchange_index: int, level: int) -> bool:
            nonlocal total
            exchange = turns[turn_index].exchanges[exchange_index]
            current = levels[turn_index].get(exchange_index, LEVEL_VERBATIM)
            total -= exchange.tokens(current) - exchange.tokens(level)
            levels[turn_index][exchange_index] = level
            report.level = max(report.level, level)
            report.degraded_messages += 1
            report.estimated_tokens = total
            return total <= available

        for level in (LEVEL_PREVIEW, LEVEL_COLLAPSED):
            for turn_index in degradable:
                for exchange_index in range(len(turns[turn_index].exchanges)):
                    if apply(turn_index, exchange_index, level):
                        return render(), report

        for turn_index in degradable:
            turn = turns[turn_index]
            before = turn.fixed_tokens() + sum(
                e.tokens(levels[turn_index].get(i, LEVEL_VERBATIM))
                for i, e in enumerate(turn.exchanges)
            )
            turn_levels[turn_index] = LEVEL_SUMMARY
            total -= before - turn.summary_tokens()
            report.level = LEVEL_SUMMARY
            report.degraded_messages += 1
            report.estimated_tokens = total
            if total <= available:
                return render(), report

        # The current turn alone exceeds the window. Degrade all but its most
        # recent exchange, which is what the model is about to respond to.
        if turns:
            last = len(turns) - 1
            for exchange_index in range(max(0, len(turns[last].exchanges) - 1)):
                if apply(last, exchange_index, LEVEL_COLLAPSED):
                    return render(), report

        report.still_over_budget = total > available
        if report.still_over_budget:
            logger.warning(
                "[AEVOM-CONTEXT] Prompt is still ~%d tokens against %d available "
                "after full degradation. A single message is larger than the "
                "window allows; the provider will likely reject this call.",
                total,
                available,
            )
        return render(), report


def record_assembly_on_span(span: Any, report: AssemblyReport) -> None:
    """Attach assembly outcomes to a span.

    Silent degradation is worse than a loud failure -- a model that quietly
    stopped being shown half its trajectory looks like a model that got worse
    -- so this telemetry is part of the feature, not an extra.
    """
    if span is None:
        return
    try:
        span.set_attribute("context.degradation_level", report.level)
        span.set_attribute("context.messages.degraded", report.degraded_messages)
        span.set_attribute("context.assembled.tokens", report.estimated_tokens)
        span.set_attribute("context.assembled.available", report.available_tokens)
        span.set_attribute("context.still_over_budget", report.still_over_budget)
    except Exception:
        logger.debug("Failed to record assembly on span", exc_info=True)


def build_context_assembler(
    base_system_prompt: str,
    config: Any = None,
    tool_schemas: Optional[list[dict[str, Any]]] = None,
):
    """Return a callable with the signature the agent graph already expects.

    Keeps ``agent_node`` unchanged: it still calls one function with the state
    and gets back a message list.
    """
    assembler = ContextAssembler(base_system_prompt, config, tool_schemas)

    def _assemble(state: dict) -> list[BaseMessage]:
        messages = assembler.assemble(state)
        report = assembler.last_report
        # Exhaustion is reported even at level 0, which happens when a single
        # message is itself too large: nothing degraded, and the call is about
        # to fail. That is precisely the case worth seeing in Phoenix.
        if report.level > LEVEL_VERBATIM or report.still_over_budget:
            from api.telemetry import get_current_chat_span

            record_assembly_on_span(get_current_chat_span(), report)
        return messages

    _assemble.assembler = assembler
    return _assemble
