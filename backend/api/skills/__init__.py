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

"""Skills: progressive-disclosure extensions for the Aevom agent.

A skill is a ``SKILL.md`` (YAML frontmatter + markdown body) plus optional
bundled read-only reference files. Skills teach the model how to orchestrate
existing Aevom tools. They are surfaced via three-level progressive
disclosure:

- Level 1: name + description in the system prompt (``format_skill_catalog``).
- Level 2: full ``SKILL.md`` body via the ``load_skill`` tool.
- Level 3: bundled files via the ``load_skill_resource`` tool.

Curated builtin skills ship in ``backend/skills/`` and load at startup into a
global registry, alongside the NIM skills allowlisted in ``binding.py`` whose
prose comes from the pinned bionemo-agent-toolkit submodule. User-installed
skills live in Redis (catalog) + Postgres (bodies/resources), mirroring the MCP
session-store pattern.
"""

from .annotate import catalog_tool_hint, resource_banner, skill_body_banner
from .binding import (
    NIM_SKILL_BINDINGS,
    BoundTool,
    SkillBinding,
    ToolAnnouncement,
    bound_skill_ids,
    get_binding,
    nim_skills_root,
    resolve_announcement,
)
from .contract import (
    SkillContract,
    SkillContractError,
    SkillEndpoint,
    SkillInputParam,
    SkillOutputField,
    SkillScalarParam,
    parse_contract,
)
from .format import format_skill_catalog
from .parser import (
    ParsedResource,
    ParsedSkill,
    SkillParseError,
    parse_skill_bundle,
    parse_skill_markdown,
)
from .endpoint_tool import SkillEndpointTool, synthesize_skill_tools
from .extract import extract_contract
from .registry import SkillDefinition, SkillRegistry, get_skill_registry
from .routes import router as skills_router
from .session_store import (
    LoadedSkill,
    SkillMeta,
    SkillSessionStore,
    get_skill_session_store,
)
from .tools import LoadSkillResourceTool, LoadSkillTool

__all__ = [
    # Annotation (tool announcement at each disclosure level)
    "catalog_tool_hint",
    "resource_banner",
    "skill_body_banner",
    # Binding (skills whose work is served by a builtin tool)
    "NIM_SKILL_BINDINGS",
    "BoundTool",
    "SkillBinding",
    "ToolAnnouncement",
    "bound_skill_ids",
    "get_binding",
    "nim_skills_root",
    "resolve_announcement",
    # Contract
    "SkillContract",
    "SkillContractError",
    "SkillEndpoint",
    "SkillInputParam",
    "SkillOutputField",
    "SkillScalarParam",
    "parse_contract",
    # Parser
    "ParsedResource",
    "ParsedSkill",
    "SkillParseError",
    "parse_skill_bundle",
    "parse_skill_markdown",
    # Synthesized endpoint tools
    "SkillEndpointTool",
    "synthesize_skill_tools",
    # Contract extraction
    "extract_contract",
    # Registry
    "SkillDefinition",
    "SkillRegistry",
    "get_skill_registry",
    # Session store
    "LoadedSkill",
    "SkillMeta",
    "SkillSessionStore",
    "get_skill_session_store",
    # Tools
    "LoadSkillTool",
    "LoadSkillResourceTool",
    # Format
    "format_skill_catalog",
    # Routes
    "skills_router",
]
