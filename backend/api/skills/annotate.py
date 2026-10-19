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

"""Tell the model about a skill's tools, in the skill's own prose.

Registering a tool is not enough. A ``SKILL.md`` was written for a human (or for
an agent armed with a shell and ``requests``), so its prose says things like
"POST the sequence to /predict and read ``structures[0].structure`` from the
response". A model that reads those instructions will follow them literally --
pasting raw sequences into a request body -- even while a typed tool sits in its
schema list.

So the announcement is repeated at every disclosure level the model can read:

1. the always-in-context catalog line (:func:`catalog_tool_hint`),
2. a banner prepended to the body returned by ``load_skill``
   (:func:`skill_body_banner`), which also maps the response fields the prose
   names onto what the tool returns instead,
3. a one-line reminder on each bundled resource (:func:`resource_banner`),
   because a resource file may repeat the raw-request instructions.

The tools being announced come from one of two places -- a synthesized
``SkillContract`` or a ``SkillBinding`` naming builtin tools -- and this module
does not care which. Both arrive as a ``ToolAnnouncement``; see
:mod:`api.skills.binding` for why the two paths exist.
"""

from __future__ import annotations

from typing import Optional

from api.tools.http_common import http_request_tool_enabled

from .binding import HAND_BUILT_HTTP, ToolAnnouncement


def _replaces(announcement: ToolAnnouncement) -> str:
    """What the banner tells the model to stop doing, naming only tools it has."""
    if announcement.replaces == HAND_BUILT_HTTP and http_request_tool_enabled():
        return f"{HAND_BUILT_HTTP} with `http_request`"
    return announcement.replaces


def catalog_tool_hint(announcement: Optional[ToolAnnouncement]) -> str:
    """One-line tool announcement for the Level 1 catalog entry.

    Returns an empty string when the skill has no tools, so the
    always-in-context block stays as cheap as it is today.
    """
    if announcement is None or announcement.is_empty:
        return ""
    return f" [has tools: {', '.join(announcement.short_names)}]"


def skill_body_banner(announcement: Optional[ToolAnnouncement]) -> str:
    """Banner prepended to the Level 2 ``load_skill`` body.

    Placed *before* the prose because the model acts on the first applicable
    instruction it reads.
    """
    if announcement is None or announcement.is_empty:
        return ""

    lines = [
        "[READ FIRST — this skill has dedicated tools]",
        "",
        "Aevom provides a tool for this skill's work. Call these tools instead of "
        f"{_replaces(announcement)}:",
        "",
    ]
    lines += list(announcement.body_lines)
    lines += ["", announcement.trailer, "", "---", ""]
    return "\n".join(lines)


def resource_banner(announcement: Optional[ToolAnnouncement]) -> str:
    """One-line Level 3 reminder, prepended to a bundled resource's text."""
    if announcement is None or announcement.is_empty:
        return ""
    names = ", ".join(f"`{name}`" for name in announcement.qualified_names)
    return (
        f"[Reminder: use {names} for this skill's work. Any raw request, response "
        "handling, local deployment or file output described below is already "
        "handled by those tools.]\n\n"
    )
