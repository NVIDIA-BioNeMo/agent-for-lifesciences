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

"""Skill disclosure tools (Level 2 and Level 3 of progressive disclosure).

- ``load_skill`` returns the full ``SKILL.md`` body once the model decides a
  skill is relevant from its catalog description (Level 2).
- ``load_skill_resource`` returns a bundled reference file's text, loaded only
  when the skill body instructs the model to do so (Level 3).

Both are LOCAL builtin tools registered in ``api.tools.registry`` so they flow
automatically into the LangGraph agent and the training tool registry. They
declare ``INPUT_RETRIEVES`` so the registry injects the per-request storage
service (needed to read session-installed skill artifacts from Postgres).
"""

from __future__ import annotations

from typing import Any

from api.tools.base import BaseTool, DataIOCategory, ToolResult, ToolType

from .annotate import resource_banner, skill_body_banner
from .session_store import get_skill_session_store


def _format_skill_body(skill) -> str:
    """Assemble the Level 2 body: tool banner, prose, then operational footers.

    The banner goes first so the model reads "call this tool" before it reads the
    prose telling it to hand-build an HTTP request.
    """
    parts = [skill_body_banner(skill.announcement)]
    parts.append(skill.body.strip())
    if skill.allowed_tools:
        parts.append(
            "\n\n---\nThis skill is intended to be used with these tools: "
            + ", ".join(skill.allowed_tools)
            + "."
        )
    if skill.resource_paths:
        listing = "\n".join(f"- {path}" for path in skill.resource_paths)
        parts.append(
            "\n\n---\nBundled reference files (load with "
            f'load_skill_resource(skill_id="{skill.id}", path="<path>")):\n{listing}'
        )
    return "".join(parts)


class LoadSkillTool(BaseTool):
    """Return the full SKILL.md body for a skill by id (Level 2 disclosure)."""

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        # Needs the per-request storage service to read session skill bodies.
        return DataIOCategory.INPUT_RETRIEVES

    @property
    def summary_template(self) -> str:
        return "Loading skill {skill_id}..."

    @property
    def max_response_tokens(self) -> int:
        """Skill bodies are author-controlled prose of unbounded length."""
        return 1200

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Load the full instructions (SKILL.md body) for an available "
                    "skill by its id. Call this when a user's task matches a skill "
                    "listed in [SKILLS AVAILABLE], then follow the returned "
                    "instructions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "The id of the skill to load, from the skill catalog.",
                        }
                    },
                    "required": ["skill_id"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        skill_id = (arguments.get("skill_id") or "").strip()
        if not skill_id:
            return ToolResult.error("skill_id is required.")

        store = get_skill_session_store()
        skill = await store.get_skill(self._session_id, self._storage, skill_id)
        if skill is None:
            return ToolResult.error(
                f"Skill '{skill_id}' not found. Check [SKILLS AVAILABLE] for valid ids."
            )
        return ToolResult.text(_format_skill_body(skill))


class LoadSkillResourceTool(BaseTool):
    """Return a bundled resource file's text for a skill (Level 3 disclosure)."""

    @property
    def name(self) -> str:
        return "load_skill_resource"

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.INPUT_RETRIEVES

    @property
    def summary_template(self) -> str:
        return "Loading skill resource {path}..."

    @property
    def max_response_tokens(self) -> int:
        """Bundled resources are arbitrary files shipped with a skill."""
        return 1200

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Load a bundled reference file for a skill (e.g. reference.md). "
                    "Only call this when a loaded skill's instructions tell you to "
                    "read a specific file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "The id of the skill that owns the resource.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path of the resource file within the skill.",
                        },
                    },
                    "required": ["skill_id", "path"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        skill_id = (arguments.get("skill_id") or "").strip()
        path = (arguments.get("path") or "").strip()
        if not skill_id or not path:
            return ToolResult.error("Both skill_id and path are required.")

        store = get_skill_session_store()
        content = await store.get_resource(
            self._session_id, self._storage, skill_id, path
        )
        if content is None:
            return ToolResult.error(
                f"Resource '{path}' not found for skill '{skill_id}'."
            )

        # A resource file often repeats the raw-request instructions, so restate
        # which tool supersedes them.
        skill = await store.get_skill(self._session_id, self._storage, skill_id)
        banner = resource_banner(skill.announcement) if skill else ""
        return ToolResult.text(banner + content)
