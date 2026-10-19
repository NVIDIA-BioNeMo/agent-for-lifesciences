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

"""Per-session call budgets for expensive tools.

Some tools cost real money or minutes of GPU time per call -- docking, co-folding,
ADMET ensembles. This module caps how many times a session may call them and, just
as importantly, tells the agent how many calls are left while it still has some.
An agent that knows it has two docking calls remaining spends them on its two best
candidates; an agent that discovers exhaustion at call nine has already wasted
eight.

The mechanism is generic and off by default. Which tools have budgets, and how
large, is a deployment decision: set ``AEVOM_TOOL_BUDGETS`` (see
:func:`configured_budgets`), or call :func:`set_budgets` from a harness that owns
the limits for a particular run.

Counters live in Redis under ``budget:{session_id}``, so they are shared across API
pods and expire with the session. When Redis is unavailable the budget opens rather
than closes: refusing real work because a counter could not be read would be a
worse failure than allowing an extra call.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Matches the selection registry: long enough to outlive a working session,
# refreshed on every write.
BUDGET_TTL = 3600

# Read once at import so a deployment cannot change the limits underneath a
# running session. Format: "tool=n,tool=n", e.g. "boltz2=8,admet_predict=15".
BUDGET_ENV_VAR = "AEVOM_TOOL_BUDGETS"


def _redis_key(session_id: str) -> str:
    return f"budget:{session_id}"


def parse_budgets(spec: str) -> dict[str, int]:
    """Parse a ``tool=n,tool=n`` budget specification.

    Unparseable entries are logged and skipped rather than raising: a typo in a
    deployment variable should not stop the service from starting, and a budget
    that fails to apply is visible in the logs and in the absence of the prompt
    block.

    Args:
        spec: The specification string. Empty or whitespace yields no budgets.

    Returns:
        Tool name to call limit, for every entry that parsed.
    """
    budgets: dict[str, int] = {}
    for entry in (spec or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        tool, separator, raw_limit = entry.partition("=")
        tool = tool.strip()
        if not separator or not tool:
            logger.warning("Ignoring budget entry %r: expected tool=limit", entry)
            continue
        try:
            limit = int(raw_limit.strip())
        except ValueError:
            logger.warning(
                "Ignoring budget for %s: %r is not a number", tool, raw_limit
            )
            continue
        if limit < 0:
            logger.warning("Ignoring budget for %s: %d is negative", tool, limit)
            continue
        budgets[tool] = limit
    return budgets


def configured_budgets() -> dict[str, int]:
    """Budgets from the environment, or an empty mapping when unset."""
    return parse_budgets(os.environ.get(BUDGET_ENV_VAR, ""))


def describe_budgets(budgets: dict[str, int]) -> str:
    """One line stating a budget configuration, including the empty one.

    Silence cannot say "no budgets": it is equally consistent with logging being
    misconfigured. A benchmark harness that needs to prove a run was unbudgeted
    reads this line back out of the container's log, so it has to be emitted in
    both cases and it has to be greppable.
    """
    if not budgets:
        return "Tool call budgets: none configured -- every tool may be called without limit"
    return "Tool call budgets active: " + ", ".join(
        f"{tool}={limit}" for tool, limit in sorted(budgets.items())
    )


_budgets: dict[str, int] = configured_budgets()


def log_budget_state() -> None:
    """Emit the active budget configuration, once the app can log.

    Deliberately not called at import. This module is imported long before
    api.main configures logging, so a record emitted then goes nowhere -- which
    is exactly what happened the first time, and it turned "no budgets" back
    into an absence rather than a statement. Called from the lifespan startup
    instead, where the handlers exist and where the state being described is
    the one the process will actually serve requests with.
    """
    logger.info("%s (from %s)", describe_budgets(_budgets), BUDGET_ENV_VAR)


def set_budgets(budgets: Optional[dict[str, int]]) -> None:
    """Replace the active budgets. Pass None or {} to disable enforcement.

    For harnesses that own the limits for a run rather than expressing them in the
    deployment environment.
    """
    global _budgets
    _budgets = dict(budgets or {})
    logger.info(describe_budgets(_budgets))


def active_budgets() -> dict[str, int]:
    """The budgets currently being enforced."""
    return dict(_budgets)


def limit_for(tool_name: str) -> Optional[int]:
    """The call limit for a tool, or None when it is unbudgeted."""
    return _budgets.get(tool_name)


@dataclass(frozen=True)
class BudgetState:
    """How much of one tool's budget a session has spent."""

    tool_name: str
    limit: int
    used: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


@dataclass(frozen=True)
class Verdict:
    """The outcome of asking to spend one call against a budget."""

    allowed: bool
    state: Optional[BudgetState] = None

    @property
    def unbudgeted(self) -> bool:
        return self.state is None


async def _counts(redis_client, session_id: str) -> dict[str, int]:
    """Read the per-tool call counts for a session."""
    raw = await redis_client.get(_redis_key(session_id))
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Discarding unreadable budget counter for session %s", session_id
        )
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(tool): int(count)
        for tool, count in data.items()
        if isinstance(count, (int, float))
    }


async def spend(session_id: Optional[str], tool_name: str) -> Verdict:
    """Charge one call against ``tool_name``'s budget for this session.

    Returns a verdict whose ``allowed`` is False only when the tool is budgeted
    and the budget is already spent. The count is incremented only when the call
    is allowed, so ``used`` never exceeds ``limit`` and a refusal does not deepen
    the hole.

    Args:
        session_id: The session to charge. None means there is nothing to track
            against, so the call proceeds unbudgeted.
        tool_name: The tool about to run.
    """
    limit = limit_for(tool_name)
    if limit is None or not session_id:
        return Verdict(allowed=True)

    try:
        from api.redis import get_redis

        redis_client = await get_redis()
        counts = await _counts(redis_client, session_id)
        used = counts.get(tool_name, 0)

        if used >= limit:
            return Verdict(
                allowed=False,
                state=BudgetState(tool_name=tool_name, limit=limit, used=used),
            )

        counts[tool_name] = used + 1
        await redis_client.set(
            _redis_key(session_id), json.dumps(counts), ex=BUDGET_TTL
        )
        return Verdict(
            allowed=True,
            state=BudgetState(tool_name=tool_name, limit=limit, used=used + 1),
        )
    except Exception:
        # Enforcement is a cost control, not a safety control. A Redis outage
        # must not turn every budgeted tool into an error.
        logger.warning(
            "Could not apply the %s budget for session %s; allowing the call",
            tool_name,
            session_id,
            exc_info=True,
        )
        return Verdict(allowed=True)


async def refund(session_id: Optional[str], tool_name: str) -> None:
    """Give back one call, for when a charged call never really ran.

    A tool that fails before doing its expensive work has not spent the resource
    the budget exists to protect, and charging for it would make a transient
    error permanently expensive.
    """
    if limit_for(tool_name) is None or not session_id:
        return

    try:
        from api.redis import get_redis

        redis_client = await get_redis()
        counts = await _counts(redis_client, session_id)
        used = counts.get(tool_name, 0)
        if used <= 0:
            return
        counts[tool_name] = used - 1
        await redis_client.set(
            _redis_key(session_id), json.dumps(counts), ex=BUDGET_TTL
        )
    except Exception:
        logger.warning(
            "Could not refund the %s budget for session %s",
            tool_name,
            session_id,
            exc_info=True,
        )


async def current_state(
    session_id: Optional[str], tool_name: str
) -> Optional["Verdict"]:
    """This tool's budget as it stands right now.

    For reporting after a :func:`refund`. ``spend`` returns the count it just
    incremented, so reusing its verdict after a refund tells the model one call
    fewer than it really has.
    """
    limit = limit_for(tool_name)
    if limit is None or not session_id:
        return None

    try:
        from api.redis import get_redis

        redis_client = await get_redis()
        counts = await _counts(redis_client, session_id)
    except Exception:
        logger.warning(
            "Could not read the %s budget for session %s", tool_name, session_id
        )
        return None
    return Verdict(
        allowed=True,
        state=BudgetState(
            tool_name=tool_name, limit=limit, used=counts.get(tool_name, 0)
        ),
    )


async def get_budget_states(session_id: Optional[str]) -> list[BudgetState]:
    """Every active budget and what the session has spent, for the prompt block.

    Ordered most-depleted first, so the constraint the agent is closest to
    hitting reads first.
    """
    if not _budgets or not session_id:
        return []

    counts: dict[str, int] = {}
    try:
        from api.redis import get_redis

        redis_client = await get_redis()
        counts = await _counts(redis_client, session_id)
    except Exception:
        logger.warning(
            "Could not read budget counters for session %s", session_id, exc_info=True
        )

    states = [
        BudgetState(tool_name=tool, limit=limit, used=counts.get(tool, 0))
        for tool, limit in _budgets.items()
    ]
    states.sort(key=lambda state: (state.remaining, state.tool_name))
    return states


async def reset(session_id: str) -> None:
    """Clear a session's counters. For tests and for restarting a task."""
    try:
        from api.redis import get_redis

        redis_client = await get_redis()
        await redis_client.delete(_redis_key(session_id))
    except Exception:
        logger.warning(
            "Could not reset budgets for session %s", session_id, exc_info=True
        )


def exhausted_message(state: BudgetState) -> str:
    """Explain a refusal in terms the agent can act on."""
    return (
        f"The call budget for {state.tool_name} is spent: {state.used} of "
        f"{state.limit} calls used, and this session gets no more. Work with the "
        f"results you already have -- earlier {state.tool_name} output is in the "
        f"session log and in storage -- or use a cheaper tool. Calling "
        f"{state.tool_name} again will keep returning this message."
    )


def remaining_note(state: BudgetState) -> str:
    """The line appended to a budgeted tool's response."""
    if state.remaining == 0:
        return (
            f"[Budget: {state.tool_name} {state.used}/{state.limit} used. That was "
            "the last one.]"
        )
    call_word = "call" if state.remaining == 1 else "calls"
    return (
        f"[Budget: {state.tool_name} {state.used}/{state.limit} used, "
        f"{state.remaining} {call_word} left.]"
    )
