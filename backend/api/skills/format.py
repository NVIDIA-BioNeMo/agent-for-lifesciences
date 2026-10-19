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

"""Formatting helpers for surfacing skills to the LLM (Level 1 disclosure)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from api.tools.http_common import http_request_tool_enabled

from .annotate import catalog_tool_hint

if TYPE_CHECKING:
    from .session_store import SkillMeta

# Keep catalog descriptions compact so the always-in-context block stays cheap.
MAX_CATALOG_DESCRIPTION_LEN = 300


def format_skill_catalog(catalog: Iterable["SkillMeta"]) -> str:
    """Render the always-in-context skill catalog (Level 1 disclosure).

    Returns only the skill ``name`` + ``description`` per skill, plus a hint
    on how to load the full instructions. Returns an empty string when no
    skills are available so nothing is added to the prompt.
    """
    entries = list(catalog)
    if not entries:
        return ""

    lines = [f"[SKILLS AVAILABLE: {len(entries)}]"]
    any_tools = False
    for meta in entries:
        description = (meta.description or "").strip().replace("\n", " ")
        if len(description) > MAX_CATALOG_DESCRIPTION_LEN:
            description = description[:MAX_CATALOG_DESCRIPTION_LEN].rstrip() + "…"
        hint = catalog_tool_hint(meta.announcement)
        any_tools = any_tools or bool(hint)
        lines.append(f"- {meta.id}: {meta.name} — {description}{hint}")

    lines.append(
        "When a task matches a skill's description, call "
        'load_skill(skill_id="<id>") to read its full instructions before '
        "proceeding. Skills tell you how to use the available tools; follow "
        "their instructions and load any referenced files with "
        "load_skill_resource(skill_id, path) as needed."
    )
    if any_tools:
        # The tool names are listed inline above so the model can connect a tool
        # in its schema list back to the skill that owns it. Naming a pattern
        # instead would be wrong for bound skills, whose tools are builtins with
        # plain names rather than synthesized `skill_<id>_<name>` ones.
        prefer = (
            "Prefer those over http_request — they take and return data references."
            if http_request_tool_enabled()
            else "They take and return data references."
        )
        lines.append(
            "Skills marked 'has tools' name the tools to use for that skill's "
            f"work. {prefer} load_skill first to see how to use them."
        )
    return "\n".join(lines)
