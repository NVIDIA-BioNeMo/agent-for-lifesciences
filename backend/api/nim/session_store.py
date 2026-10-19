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

"""Per-session NIM registrations, in Redis.

Modelled directly on ``api.mcp.session_store``, with one simplification: a
NIM has no live connection to cache. Registration is a probe whose result is
small and serialisable, so Redis holds everything and the process holds only
the tool objects it derived -- which are cheap to rebuild and are rebuilt on
any pod that has not seen the session.

The confirm gate is the same as MCP's, and for the same reason. Registering a
NIM introduces tools the agent did not have, pointed at a host Aevom has
never spoken to. Those tools stay withheld until a human says otherwise.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from api.tools.base import BaseTool

from .adapters import get_adapter
from .adapters.base import NIMEndpoint
from .client import NIMClient, NIMInfo
from .overrides import EndpointOverride, clear_session, set_overrides

logger = logging.getLogger(__name__)

# Same 4 hours as MCP: long enough for a working session, short enough that a
# forgotten endpoint does not linger.
DEFAULT_SESSION_TTL = 4 * 60 * 60


def _redis_key(session_id: str) -> str:
    return f"nim:{session_id}"


@dataclass
class NIMServerConfig:
    """A registered NIM, as stored in Redis.

    Attributes:
        id: Aevom's id for this registration.
        name: The user's name for it.
        url: Base URL.
        api_key: Bearer token, or None.
        info: What the probe found, kept so the session can be rebuilt
            without re-probing a NIM that may be mid-restart.
        confirmed: Whether a human authorised this NIM's tools. False until
            ``confirm_server``; tools are withheld meanwhile.
        registered_at: When it was added.
    """

    id: str
    name: str
    url: str
    api_key: Optional[str] = None
    info: Optional[NIMInfo] = None
    confirmed: bool = False
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Redis storage."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "api_key": self.api_key,
            "info": self.info.to_dict() if self.info else None,
            "confirmed": self.confirmed,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NIMServerConfig":
        """Deserialize from Redis."""
        info = data.get("info")
        return cls(
            id=data["id"],
            name=data["name"],
            url=data["url"],
            api_key=data.get("api_key"),
            info=NIMInfo.from_dict(info) if info else None,
            confirmed=data.get("confirmed", False),
            registered_at=data.get("registered_at", time.time()),
        )

    @property
    def endpoint(self) -> Optional[NIMEndpoint]:
        """This registration as an adapter sees it."""
        if self.info is None:
            return None
        return NIMEndpoint(
            server_id=self.id,
            name=self.name,
            url=self.url,
            api_key=self.api_key,
            info=self.info,
        )

    def to_response(self) -> dict[str, Any]:
        """JSON for the API, describing what this NIM offers.

        ``tools`` is what it *would* contribute; while ``confirmed`` is false
        none of it is reachable, and a client must render that difference.
        """
        adapter = get_adapter(self.info.model_id) if self.info else None
        endpoint = self.endpoint
        tool_names: list[str] = []
        repointed: list[str] = []
        if adapter and endpoint:
            tool_names = [tool.name for tool in adapter.build_tools(endpoint)]
            repointed = sorted(adapter.builtin_endpoints(endpoint))

        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "confirmed": self.confirmed,
            "registered_at": self.registered_at,
            "model_id": self.info.model_id if self.info else None,
            "model_version": self.info.model_version if self.info else None,
            "description": self.info.description if self.info else None,
            "ready": self.info.ready if self.info else False,
            "tasks": self.info.tasks if self.info else [],
            "adapter": adapter.key if adapter else None,
            "adapter_label": adapter.label if adapter else None,
            "adapter_summary": adapter.summary if adapter else "",
            "tools": tool_names,
            "tools_count": len(tool_names),
            "repointed_tools": repointed,
        }


class NIMSessionStore:
    """Manages a session's registered NIMs and the tools they contribute."""

    def __init__(self, session_ttl: float = DEFAULT_SESSION_TTL):
        """Initialize the store.

        Args:
            session_ttl: TTL for a session's registrations, in seconds.
        """
        self._session_ttl = session_ttl

    async def _get_redis(self):
        from api.redis import get_redis

        return await get_redis()

    async def _load(self, session_id: str) -> dict[str, NIMServerConfig]:
        """Read a session's registrations from Redis."""
        redis = await self._get_redis()
        raw = await redis.get(_redis_key(session_id))
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return {
                server_id: NIMServerConfig.from_dict(config)
                for server_id, config in data.items()
            }
        except Exception as exc:  # noqa: BLE001 - a bad blob must not kill the session
            logger.warning("Failed to load NIM configs for %s: %s", session_id, exc)
            return {}

    async def _save(
        self, session_id: str, configs: dict[str, NIMServerConfig]
    ) -> None:
        """Write a session's registrations, and refresh its endpoint overrides."""
        redis = await self._get_redis()
        await redis.set(
            _redis_key(session_id),
            json.dumps({sid: cfg.to_dict() for sid, cfg in configs.items()}),
            ex=int(self._session_ttl),
        )
        self._refresh_overrides(session_id, configs)

    def _refresh_overrides(
        self, session_id: str, configs: dict[str, NIMServerConfig]
    ) -> None:
        """Rebuild which built-in tools this session re-points, and where.

        Only confirmed NIMs count. An unconfirmed registration must not
        quietly redirect a tool the agent already had -- that would be a
        larger change than adding a withheld tool, made with less consent.
        """
        overrides: dict[str, EndpointOverride] = {}
        for config in configs.values():
            if not config.confirmed or config.info is None:
                continue
            adapter = get_adapter(config.info.model_id)
            endpoint = config.endpoint
            if adapter is None or endpoint is None:
                continue
            for tool_name, url in adapter.builtin_endpoints(endpoint).items():
                overrides[tool_name] = EndpointOverride(
                    url=url, api_key=config.api_key, server_id=config.id
                )
        set_overrides(session_id, overrides)

    async def add_server(
        self,
        session_id: str,
        name: str,
        url: str,
        api_key: Optional[str] = None,
    ) -> NIMServerConfig:
        """Probe a NIM and register it for this session.

        The probe is what validates the URL, so a NIM that cannot be reached
        is never registered. Raises the client's ``NIMConnectionError`` or
        ``NIMAuthError`` in that case.
        """
        logger.info("Registering NIM '%s' (%s) for session %s", name, url, session_id)
        info = await NIMClient(url=url, api_key=api_key).probe()

        config = NIMServerConfig(
            id=uuid.uuid4().hex[:8],
            name=name,
            url=url.rstrip("/"),
            api_key=api_key,
            info=info,
        )

        configs = await self._load(session_id)
        configs[config.id] = config
        await self._save(session_id, configs)

        adapter = get_adapter(info.model_id)
        logger.info(
            "Registered NIM '%s' as %s (model %s, adapter %s)",
            name, config.id, info.model_id, adapter.key if adapter else "none",
        )
        return config

    async def remove_server(self, session_id: str, server_id: str) -> bool:
        """Remove a registration. Returns False if it was not there."""
        configs = await self._load(session_id)
        if server_id not in configs:
            return False
        del configs[server_id]
        await self._save(session_id, configs)
        logger.info("Removed NIM %s from session %s", server_id, session_id)
        return True

    async def confirm_server(self, session_id: str, server_id: str) -> bool:
        """Authorise a NIM's tools after a human has reviewed them.

        Idempotent; returns False only when no such registration exists.
        """
        configs = await self._load(session_id)
        config = configs.get(server_id)
        if config is None:
            return False
        config.confirmed = True
        await self._save(session_id, configs)
        logger.info("Confirmed NIM %s (%s) for session %s", config.name, server_id, session_id)
        return True

    async def get_servers(self, session_id: str) -> list[NIMServerConfig]:
        """Every NIM registered for this session."""
        configs = await self._load(session_id)
        # Reading is also when a pod that has never seen this session learns
        # about its endpoint overrides, so refresh them here too.
        self._refresh_overrides(session_id, configs)
        return list(configs.values())

    async def get_tools(self, session_id: str) -> list[BaseTool]:
        """The NIM tools this session may actually use.

        Withholds everything from unconfirmed registrations, and is the single
        chokepoint for both what the model is shown and what the registry will
        dispatch to -- same as MCP.
        """
        tools: list[BaseTool] = []
        for config in await self.get_servers(session_id):
            if not config.confirmed:
                logger.info(
                    "Withholding tools from unconfirmed NIM %s (%s)",
                    config.name, config.id,
                )
                continue
            if config.info is None:
                continue
            adapter = get_adapter(config.info.model_id)
            endpoint = config.endpoint
            if adapter is None or endpoint is None:
                continue
            try:
                tools.extend(adapter.build_tools(endpoint))
            except Exception:  # noqa: BLE001 - one bad adapter must not blank the rest
                logger.exception(
                    "Adapter %s failed to build tools for NIM %s",
                    adapter.key, config.id,
                )
        return tools

    async def get_tool_schemas(self, session_id: str) -> list[dict[str, Any]]:
        """OpenAI-compatible schemas for this session's NIM tools."""
        return [tool.get_schema() for tool in await self.get_tools(session_id)]

    async def get_tool_by_name(
        self, session_id: Optional[str], tool_name: str
    ) -> Optional[BaseTool]:
        """Resolve a NIM tool by name, or None."""
        if not session_id:
            return None
        for tool in await self.get_tools(session_id):
            if tool.name == tool_name:
                return tool
        return None

    async def clear(self, session_id: str) -> None:
        """Forget a session's registrations entirely."""
        redis = await self._get_redis()
        await redis.delete(_redis_key(session_id))
        clear_session(session_id)


_nim_session_store: Optional[NIMSessionStore] = None


def get_nim_session_store() -> NIMSessionStore:
    """Get the NIM session store singleton."""
    global _nim_session_store
    if _nim_session_store is None:
        _nim_session_store = NIMSessionStore()
        logger.info("Created NIM session store")
    return _nim_session_store
