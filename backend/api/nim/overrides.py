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

"""Where a built-in NIM tool's requests go, for one session.

Built-in tools are registry singletons shared by every session, so a user
pointing ``boltz2`` at their own GPU cannot be recorded on the tool. It is
recorded here instead, keyed by session, and ``NvidiaApiTool._resolve_endpoint``
reads it on each call.

This is process-local, rebuilt from Redis whenever the session store loads a
session's NIMs. That is the same shape as the MCP connection cache: Redis
holds the registration, the process holds what it derived from it. A pod that
has never seen the session simply has no override until it loads one, and the
call falls through to the hosted endpoint -- wrong, but only until the first
list or probe. The session store refreshes it on every read for that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndpointOverride:
    """A self-hosted endpoint standing in for a hosted NVIDIA one.

    Attributes:
        url: Full inference URL on the self-hosted NIM.
        api_key: Bearer token, or None when the endpoint needs no auth --
            the usual case, and the reason this is not simply a URL string.
        server_id: The NIM registration this came from, for diagnostics.
    """

    url: str
    api_key: Optional[str]
    server_id: str


# session_id -> {builtin tool name -> override}
_overrides: dict[str, dict[str, EndpointOverride]] = {}


def set_overrides(session_id: str, overrides: dict[str, EndpointOverride]) -> None:
    """Replace every override for a session.

    Replacing wholesale rather than merging is deliberate: this is called with
    what the session's confirmed NIMs currently say, so a NIM that was removed
    or un-confirmed must stop re-pointing anything.
    """
    if overrides:
        _overrides[session_id] = dict(overrides)
        logger.info(
            "Session %s re-points %s to self-hosted NIMs",
            session_id, ", ".join(sorted(overrides)),
        )
    else:
        _overrides.pop(session_id, None)


def get_endpoint_override(
    session_id: str, tool_name: str
) -> Optional[EndpointOverride]:
    """The override for this tool in this session, if any."""
    return _overrides.get(session_id, {}).get(tool_name)


def clear_session(session_id: str) -> None:
    """Drop every override for a session."""
    _overrides.pop(session_id, None)


def overridden_tools(session_id: str) -> list[str]:
    """Names of built-in tools currently re-pointed for this session."""
    return sorted(_overrides.get(session_id, {}))
