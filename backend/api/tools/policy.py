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

"""Registry-level policy hooks for tool execution.

Part of the P0 architecture-contracts work (see
``internal_docs/docs/0001-architecture-boundaries.md``). Before this module, the only
"gate before execute" pattern in the codebase was the SSRF/host-allowlist
check inlined into ``HttpRequestTool`` (``api.tools.http_common``). This
module generalizes that shape into a hook every tool call passes through,
modeled directly on the existing ``ToolFilter`` pattern in
``api.tools.registry`` (env-var driven, an ``active`` short-circuit so an
unconfigured deployment pays no cost).

Default behavior is a no-op allow-all -- registering a ``Policy`` is opt-in
and changes nothing for a deployment that doesn't configure one.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    """The outcome of evaluating one policy against one tool call."""

    allowed: bool = True
    reason: Optional[str] = None
    requires_approval: bool = False

    @classmethod
    def allow(cls) -> "PolicyResult":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str, *, requires_approval: bool = False) -> "PolicyResult":
        return cls(allowed=False, reason=reason, requires_approval=requires_approval)


class Policy(Protocol):
    """A single named check run before a tool executes.

    Implementations should be fast and side-effect-free (no network calls,
    no writes) unless the check itself requires one (e.g. an SSRF DNS
    resolution). Raising is treated as a deny, not a crash -- see
    ``PolicyEngine.evaluate``.
    """

    name: str

    async def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str],
        request_id: Optional[str],
    ) -> PolicyResult:
        ...


class ApprovalRequiredPolicy:
    """Deny named tools pending human approval, via an env-configured list.

    There is no approval UI yet (future phase, per the audit and ADR 0001);
    this proves the ``requires_approval`` field end-to-end for the two P0
    reference tools without building one. A tool named in
    ``AEVOM_REQUIRE_APPROVAL_FOR`` is always denied with
    ``requires_approval=True`` until that changes.
    """

    name = "approval_required"

    def __init__(self, tool_names: Optional[frozenset[str]] = None):
        self.tool_names = (
            _names_from_env("AEVOM_REQUIRE_APPROVAL_FOR")
            if tool_names is None
            else frozenset(tool_names)
        )

    @property
    def active(self) -> bool:
        return bool(self.tool_names)

    async def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str],
        request_id: Optional[str],
    ) -> PolicyResult:
        if tool_name not in self.tool_names:
            return PolicyResult.allow()
        return PolicyResult.deny(
            f"'{tool_name}' requires human approval before it can run, and "
            "Aevom does not yet have an approval flow to grant it. Ask the "
            "user to run this outside Aevom, or remove it from "
            "AEVOM_REQUIRE_APPROVAL_FOR.",
            requires_approval=True,
        )


class HttpAllowlistPolicy:
    """Register the existing http_request SSRF/allowlist gate as a Policy.

    ``HttpRequestTool`` still performs its own inline check
    (``api.tools.http_common.validate_target``) inside ``execute()`` --
    intentionally redundant during this phase, per ADR 0001, so this hook can
    be proven without weakening existing coverage if the two ever disagree.
    """

    name = "http_allowlist"

    @property
    def active(self) -> bool:
        return True

    async def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str],
        request_id: Optional[str],
    ) -> PolicyResult:
        if tool_name != "http_request":
            return PolicyResult.allow()

        from .http_common import HttpToolError, validate_target

        url = arguments.get("url")
        method = arguments.get("method", "POST")
        if not url:
            # Missing url is a tool-argument error, not a policy matter;
            # let execute() produce its usual message.
            return PolicyResult.allow()
        try:
            validate_target(url, method)
        except HttpToolError as exc:
            return PolicyResult.deny(str(exc))
        return PolicyResult.allow()


def _names_from_env(variable: str) -> frozenset[str]:
    return frozenset(
        name.strip() for name in os.getenv(variable, "").split(",") if name.strip()
    )


class PolicyEngine:
    """Runs the configured policies before a tool executes.

    Default construction (no policies passed, nothing active) is a no-op:
    ``evaluate()`` returns ``PolicyResult.allow()`` immediately, matching
    behavior before this module existed.
    """

    def __init__(self, policies: Optional[list[Policy]] = None):
        self.policies = (
            [ApprovalRequiredPolicy(), HttpAllowlistPolicy()]
            if policies is None
            else policies
        )

    @property
    def active(self) -> bool:
        return any(getattr(p, "active", True) for p in self.policies)

    async def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str],
        request_id: Optional[str],
    ) -> PolicyResult:
        if not self.active:
            return PolicyResult.allow()
        for policy in self.policies:
            if not getattr(policy, "active", True):
                continue
            try:
                result = await policy.evaluate(
                    tool_name, arguments, session_id, request_id
                )
            except Exception:  # noqa: BLE001 - a broken policy must not break execution
                logger.warning(
                    "Policy %s raised evaluating %s; treating as deny",
                    getattr(policy, "name", policy),
                    tool_name,
                    exc_info=True,
                )
                return PolicyResult.deny(
                    f"Policy check '{getattr(policy, 'name', 'unknown')}' failed "
                    "unexpectedly, so the call was refused rather than allowed "
                    "unchecked."
                )
            if not result.allowed:
                return result
        return PolicyResult.allow()
