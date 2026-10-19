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

"""Per-session skill storage: Redis catalog + Postgres artifact bodies.

Architecture (mirrors ``api.mcp.session_store``):
- **Redis** (``skill:{session_id}``, TTL) holds the installed-skill *catalog*
  (metadata + artifact references). This is the source of truth for what a
  session has installed and keeps API pods stateless.
- **Postgres** (``data_artifacts`` via ``DataStorageService``) holds the skill
  ``SKILL.md`` body and each bundled resource as session-scoped artifacts.
- **Curated builtin skills** come from the in-memory
  :class:`~api.skills.registry.SkillRegistry` and are merged into every view.

Skill bodies stored in ``data_artifacts`` are ephemeral (the table is
recreated on API restart, see ``api.db.config.init_db``); the Redis catalog
shares the same session lifetime, so the two stay consistent for a session.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .binding import (
    SkillBinding,
    ToolAnnouncement,
    get_binding,
    resolve_announcement,
)
from .contract import SkillContract, SkillContractError, parse_contract
from .parser import ParsedSkill
from .registry import BUILTIN_SOURCE, get_skill_registry

if TYPE_CHECKING:
    from api.storage.scoped import SessionScopedStorage

    from .endpoint_tool import SkillEndpointTool

logger = logging.getLogger(__name__)

# Default session TTL: 4 hours (matches MCP sessions).
DEFAULT_SESSION_TTL = 4 * 60 * 60

SESSION_SOURCE = "session"
SKILL_ARTIFACT_TYPE = "skill_file"


@dataclass
class SkillMeta:
    """Serializable catalog entry for an installed skill (stored in Redis)."""

    id: str
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    body_ref: str = ""
    resources: list[dict[str, Any]] = field(default_factory=list)  # {path, ref, bytes}
    source: str = SESSION_SOURCE
    installed_at: float = field(default_factory=time.time)
    # Serialized SkillContract. Stored as a plain dict because the catalog is
    # JSON in Redis; use `parsed_contract` to get the validated model back.
    contract: Optional[dict[str, Any]] = None
    # Key into the static binding table, set only for skills the registry loaded
    # against a binding. Stored as an id rather than inferred from `id` so that a
    # session-installed skill cannot claim a NIM's builtin tools by reusing its
    # name.
    binding_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillMeta":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            allowed_tools=list(data.get("allowed_tools") or []),
            body_ref=data.get("body_ref", ""),
            resources=list(data.get("resources") or []),
            source=data.get("source", SESSION_SOURCE),
            installed_at=data.get("installed_at", time.time()),
            contract=data.get("contract"),
            binding_id=data.get("binding_id"),
        )

    @property
    def resource_paths(self) -> list[str]:
        return [r.get("path", "") for r in self.resources if r.get("path")]

    @property
    def binding(self) -> Optional[SkillBinding]:
        """Builtin tools bound to this skill, if any."""
        return get_binding(self.binding_id) if self.binding_id else None

    @property
    def announcement(self) -> Optional[ToolAnnouncement]:
        """Tools to announce to the model alongside this catalog entry."""
        return resolve_announcement(self.id, self.binding, self.parsed_contract)

    @property
    def parsed_contract(self) -> Optional[SkillContract]:
        """Validated contract, or None when absent or no longer valid.

        A catalog entry can outlive a code change that tightens validation, so a
        contract that no longer parses is treated as absent rather than raising
        and breaking every prompt that renders the catalog.
        """
        if not self.contract:
            return None
        try:
            return parse_contract(self.contract)
        except SkillContractError as exc:
            logger.warning("Discarding invalid contract for skill %s: %s", self.id, exc)
            return None

    def to_public_dict(self) -> dict[str, Any]:
        """API-facing view (no internal artifact refs)."""
        contract = self.parsed_contract
        announcement = self.announcement
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "allowed_tools": self.allowed_tools,
            "resources": self.resource_paths,
            "source": self.source,
            "removable": self.source != BUILTIN_SOURCE,
            "installed_at": self.installed_at,
            "contract": contract.to_dict() if contract else None,
            # Names the model can actually call, whether they were synthesized
            # from a contract or bound to builtin tools.
            "tool_names": list(announcement.qualified_names) if announcement else [],
        }


@dataclass
class LoadedSkill:
    """Fully resolved skill content returned to the load_skill tool."""

    id: str
    name: str
    body: str
    allowed_tools: list[str]
    resource_paths: list[str]
    source: str
    # Together these drive the "a tool supersedes the prose below" banner:
    # `contract` for a synthesized tool, `binding` for a builtin one.
    contract: Optional[SkillContract] = None
    binding: Optional[SkillBinding] = None

    @property
    def announcement(self) -> Optional[ToolAnnouncement]:
        """Tools to announce to the model when it reads this skill."""
        return resolve_announcement(self.id, self.binding, self.contract)


def _redis_catalog_key(session_id: str) -> str:
    """Redis key for a session's installed-skill catalog."""
    return f"skill:{session_id}"


def _candidate_skill_ids(skill_id: str) -> list[str]:
    """Robust id candidates for lookup.

    Models frequently drop the ``skill_`` prefix (calling ``load_skill`` with
    the bare hex), so accept both the prefixed and unprefixed forms.
    """
    sid = (skill_id or "").strip()
    candidates = [sid]
    if sid.startswith("skill_"):
        candidates.append(sid[len("skill_") :])
    else:
        candidates.append(f"skill_{sid}")
    seen: set[str] = set()
    result: list[str] = []
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            result.append(cand)
    return result


class SkillSessionStore:
    """Manages installed skills per browser session.

    Redis stores the catalog (source of truth); Postgres stores bodies and
    resource files as session-scoped artifacts. Curated builtin skills are
    merged in from the global registry.
    """

    def __init__(self, session_ttl: float = DEFAULT_SESSION_TTL):
        self._session_ttl = session_ttl

    async def _get_redis(self):
        from api.redis import get_redis

        return await get_redis()

    async def _load_catalog(self, session_id: str) -> dict[str, SkillMeta]:
        redis = await self._get_redis()
        raw = await redis.get(_redis_catalog_key(session_id))
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return {sid: SkillMeta.from_dict(meta) for sid, meta in data.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load skill catalog for %s: %s", session_id, exc)
            return {}

    async def _save_catalog(
        self, session_id: str, catalog: dict[str, SkillMeta]
    ) -> None:
        redis = await self._get_redis()
        data = {sid: meta.to_dict() for sid, meta in catalog.items()}
        await redis.set(
            _redis_catalog_key(session_id),
            json.dumps(data),
            ex=int(self._session_ttl),
        )

    async def install_skill(
        self,
        session_id: str,
        storage: "SessionScopedStorage",
        parsed: ParsedSkill,
        contract: Optional[SkillContract] = None,
    ) -> SkillMeta:
        """Store a parsed skill's body + resources and register it in Redis.

        Args:
            contract: Overrides ``parsed.contract``. The install route passes an
                LLM-extracted (unconfirmed) contract here when the skill did not
                declare one.
        """
        skill_id = f"skill_{uuid.uuid4().hex[:8]}"

        body_ref = await storage.store(
            SKILL_ARTIFACT_TYPE,
            parsed.body.encode("utf-8"),
            metadata={
                "skill_id": skill_id,
                "skill_name": parsed.name,
                "role": "body",
            },
            description=f"skill body: {parsed.name}"[:256],
        )

        resources: list[dict[str, Any]] = []
        for resource in parsed.resources:
            ref = await storage.store(
                SKILL_ARTIFACT_TYPE,
                resource.content,
                metadata={
                    "skill_id": skill_id,
                    "skill_name": parsed.name,
                    "role": "resource",
                    "path": resource.path,
                },
                description=f"skill resource: {parsed.name}/{resource.path}"[:256],
            )
            resources.append(
                {"path": resource.path, "ref": ref, "bytes": len(resource.content)}
            )

        meta = SkillMeta(
            id=skill_id,
            name=parsed.name,
            description=parsed.description,
            allowed_tools=parsed.allowed_tools,
            body_ref=body_ref,
            resources=resources,
            source=SESSION_SOURCE,
            contract=(contract or parsed.contract).to_dict()
            if (contract or parsed.contract)
            else None,
        )

        catalog = await self._load_catalog(session_id)
        catalog[skill_id] = meta
        await self._save_catalog(session_id, catalog)
        await storage.commit()

        logger.info(
            "Installed skill '%s' (%s) with %d resources for session %s",
            parsed.name,
            skill_id,
            len(resources),
            session_id,
        )
        return meta

    async def remove_skill(
        self,
        session_id: str,
        storage: "SessionScopedStorage",
        skill_id: str,
    ) -> bool:
        """Remove a session-installed skill (builtin skills are not removable)."""
        catalog = await self._load_catalog(session_id)
        meta = catalog.get(skill_id)
        if meta is None:
            return False

        refs = [meta.body_ref] + [r.get("ref", "") for r in meta.resources]
        for ref in refs:
            if not ref:
                continue
            try:
                await storage.delete(ref)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Failed to delete skill artifact %s: %s", ref, exc)

        del catalog[skill_id]
        await self._save_catalog(session_id, catalog)
        await storage.commit()
        logger.info("Removed skill %s from session %s", skill_id, session_id)
        return True

    async def get_tools(self, session_id: Optional[str]) -> list["SkillEndpointTool"]:
        """Synthesize the endpoint tools for every skill visible to a session.

        Mirrors ``MCPSessionStore.get_tools``: the registry calls this to merge
        skill tools into the schema list and to resolve a tool at execution time.
        Only confirmed contracts yield tools.
        """
        from .endpoint_tool import synthesize_skill_tools

        tools: list["SkillEndpointTool"] = []
        for meta in await self.get_catalog(session_id):
            tools.extend(
                synthesize_skill_tools(meta.id, meta.name, meta.parsed_contract)
            )
        return tools

    async def get_tool_schemas(self, session_id: Optional[str]) -> list[dict[str, Any]]:
        """OpenAI-compatible schemas for all synthesized skill tools."""
        return [tool.get_schema() for tool in await self.get_tools(session_id)]

    async def get_tool_by_name(
        self,
        session_id: Optional[str],
        tool_name: str,
    ) -> Optional["SkillEndpointTool"]:
        """Resolve a synthesized skill tool by its full name."""
        for tool in await self.get_tools(session_id):
            if tool.name == tool_name:
                return tool
        return None

    async def set_contract(
        self,
        session_id: str,
        skill_id: str,
        contract: SkillContract,
    ) -> Optional[SkillMeta]:
        """Persist a reviewed contract for a session-installed skill.

        Returns the updated meta, or None when the skill is unknown to this
        session. Builtin skills declare their contract in frontmatter and are
        not editable at runtime.
        """
        catalog = await self._load_catalog(session_id)
        meta = None
        for cand in _candidate_skill_ids(skill_id):
            meta = catalog.get(cand)
            if meta is not None:
                skill_id = cand
                break
        if meta is None:
            return None

        meta.contract = contract.to_dict()
        catalog[skill_id] = meta
        await self._save_catalog(session_id, catalog)
        logger.info(
            "Set contract for skill %s (session %s): %d endpoint(s), confirmed=%s",
            skill_id,
            session_id,
            len(contract.endpoints),
            contract.confirmed,
        )
        return meta

    async def get_catalog(self, session_id: Optional[str]) -> list[SkillMeta]:
        """Return the merged catalog (curated builtin + session-installed)."""
        merged: list[SkillMeta] = []
        for definition in get_skill_registry().list():
            merged.append(
                SkillMeta(
                    id=definition.id,
                    name=definition.name,
                    description=definition.description,
                    allowed_tools=definition.allowed_tools,
                    body_ref="",
                    resources=[
                        {"path": path, "ref": "", "bytes": len(content.encode("utf-8"))}
                        for path, content in definition.resources.items()
                    ],
                    source=BUILTIN_SOURCE,
                    installed_at=0.0,
                    contract=definition.contract.to_dict()
                    if definition.contract
                    else None,
                    binding_id=definition.binding.skill_id
                    if definition.binding
                    else None,
                )
            )

        if session_id:
            catalog = await self._load_catalog(session_id)
            merged.extend(catalog.values())
        return merged

    async def get_skill(
        self,
        session_id: Optional[str],
        storage: Optional["SessionScopedStorage"],
        skill_id: str,
    ) -> Optional[LoadedSkill]:
        """Resolve a skill's full body (curated in-memory or session Postgres)."""
        candidates = _candidate_skill_ids(skill_id)
        definition = None
        for cand in candidates:
            definition = get_skill_registry().get(cand)
            if definition is not None:
                break
        if definition is not None:
            return LoadedSkill(
                id=definition.id,
                name=definition.name,
                body=definition.body,
                allowed_tools=definition.allowed_tools,
                resource_paths=list(definition.resources.keys()),
                source=BUILTIN_SOURCE,
                contract=definition.contract,
                binding=definition.binding,
            )

        if not session_id:
            return None
        catalog = await self._load_catalog(session_id)
        meta = None
        for cand in candidates:
            meta = catalog.get(cand)
            if meta is not None:
                break
        if meta is None:
            return None
        if storage is None:
            logger.warning("No storage available to load skill body %s", skill_id)
            return None
        body = (await storage.retrieve(meta.body_ref)).decode("utf-8", errors="replace")
        return LoadedSkill(
            id=meta.id,
            name=meta.name,
            body=body,
            allowed_tools=meta.allowed_tools,
            resource_paths=meta.resource_paths,
            source=meta.source,
            contract=meta.parsed_contract,
            binding=meta.binding,
        )

    async def get_resource(
        self,
        session_id: Optional[str],
        storage: Optional["SessionScopedStorage"],
        skill_id: str,
        path: str,
    ) -> Optional[str]:
        """Resolve a bundled resource file's text content."""
        candidates = _candidate_skill_ids(skill_id)
        definition = None
        for cand in candidates:
            definition = get_skill_registry().get(cand)
            if definition is not None:
                break
        if definition is not None:
            return definition.resources.get(path)

        if not session_id:
            return None
        catalog = await self._load_catalog(session_id)
        meta = None
        for cand in candidates:
            meta = catalog.get(cand)
            if meta is not None:
                break
        if meta is None:
            return None
        ref = next((r.get("ref") for r in meta.resources if r.get("path") == path), None)
        if not ref:
            return None
        if storage is None:
            logger.warning("No storage available to load skill resource %s", skill_id)
            return None
        return (await storage.retrieve(ref)).decode("utf-8", errors="replace")


_skill_session_store: Optional[SkillSessionStore] = None


def get_skill_session_store() -> SkillSessionStore:
    """Get the skill session store singleton."""
    global _skill_session_store
    if _skill_session_store is None:
        _skill_session_store = SkillSessionStore()
        logger.info("Created skill session store")
    return _skill_session_store
