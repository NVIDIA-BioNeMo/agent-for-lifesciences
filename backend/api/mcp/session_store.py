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

"""Per-session MCP server storage with Redis config + local connection cache.

This module manages MCP server connections on a per-session basis,
allowing different browser sessions to have their own set of MCP servers.

Architecture:
- **Redis** stores MCP server configurations (name, URL, api_key) per session.
  This is the source of truth for what servers a session has.
- **Process-local cache** holds live MCPClient WebSocket connections. On cache
  miss, the config is read from Redis and the connection is re-established.
- This allows horizontal scaling: any pod can serve any session by lazily
  reconnecting from the Redis-stored config.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from api.tools.base import BaseTool

from .client import MCPClient, MCPToolSchema
from .wrapper import MCPToolWrapper, create_mcp_tools
from .adapter import AdaptedMCPToolWrapper, adapt_mcp_tools_async

logger = logging.getLogger(__name__)

# Default session TTL: 4 hours
DEFAULT_SESSION_TTL = 4 * 60 * 60


@dataclass
class MCPServerConfig:
    """Serializable configuration for an MCP server (stored in Redis).
    
    This contains everything needed to reconnect to the server.
    """
    id: str
    name: str
    url: str
    api_key: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    confirmed: bool = False
    """Whether a human has reviewed this server's tools and authorised them.

    False until :meth:`MCPSessionStore.confirm_server` says otherwise, and
    ``get_tools`` withholds the server's tools while it is False.

    This mirrors how a skill is already treated: a skill declaring one HTTP
    endpoint yields no tool until its contract is confirmed
    (``skills/contract.py``), and every call it makes is then checked against
    an allowlist. An MCP server is strictly more powerful -- it contributes a
    whole toolset, of tools Aevom has never seen -- so binding it on a single
    unreviewed POST was the weaker of the two paths by a wide margin.

    Defaults False on deserialisation too, so servers already in Redis when
    this shipped are treated as unreviewed rather than grandfathered in.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Redis storage."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "api_key": self.api_key,
            "connected_at": self.connected_at,
            "confirmed": self.confirmed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        """Deserialize from Redis."""
        return cls(
            id=data["id"],
            name=data["name"],
            url=data["url"],
            api_key=data.get("api_key"),
            connected_at=data.get("connected_at", time.time()),
            confirmed=data.get("confirmed", False),
        )


@dataclass
class MCPServerConnection:
    """Represents a connected MCP server (lives in process-local cache).
    
    Attributes:
        config: Server configuration (serializable).
        client: Connected MCPClient instance (not serializable).
        tools: List of wrapped MCP tools.
        schemas: Raw tool schemas from the server.
        last_used: Unix timestamp of last tool execution.
        status: Current connection status.
    """
    config: MCPServerConfig
    client: MCPClient
    tools: list[Union[MCPToolWrapper, AdaptedMCPToolWrapper]] = field(default_factory=list)
    schemas: list[MCPToolSchema] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)
    status: str = "connected"
    
    @property
    def id(self) -> str:
        return self.config.id
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def url(self) -> str:
        return self.config.url
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        tool_dicts = []
        for tool in self.tools:
            schema = tool.get_schema()
            func = schema.get("function", {})
            tool_info: dict[str, Any] = {
                "name": tool.original_name,
                "description": func.get("description", ""),
            }
            # Include adaptation info for adapted tools
            if isinstance(tool, AdaptedMCPToolWrapper):
                tool_info["adapted"] = True
                tool_info["adaptation_info"] = tool.adaptation_info
            tool_dicts.append(tool_info)
        
        return {
            "id": self.config.id,
            "name": self.config.name,
            "url": self.config.url,
            "status": self.status,
            "tools": tool_dicts,
            "tools_count": len(self.tools),
            "connected_at": self.config.connected_at,
            # The tools above are what the server *offers*. Until confirmed is
            # true none of them are reachable by the agent, so a client
            # rendering this list must say so rather than implying they are live.
            "confirmed": self.config.confirmed,
        }


def _redis_config_key(session_id: str) -> str:
    """Redis key for MCP session config."""
    return f"mcp:{session_id}"


class MCPSessionStore:
    """Manages MCP server connections per browser session.
    
    Two-layer architecture:
    1. Redis stores server *configurations* (source of truth)
    2. Process-local cache holds live WebSocket connections
    
    On cache miss, configs are read from Redis and connections re-established.
    """
    
    def __init__(self, session_ttl: float = DEFAULT_SESSION_TTL):
        """Initialize the session store.
        
        Args:
            session_ttl: TTL for sessions in seconds.
        """
        self._session_ttl = session_ttl
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Process-local connection cache: session_id -> {server_id -> MCPServerConnection}
        self._local_cache: dict[str, dict[str, MCPServerConnection]] = {}
    
    async def _get_redis(self):
        """Get Redis client."""
        from api.redis import get_redis
        return await get_redis()
    
    async def _load_configs(self, session_id: str) -> dict[str, MCPServerConfig]:
        """Load server configs from Redis.
        
        Args:
            session_id: Browser session ID.
            
        Returns:
            Dict of server_id -> MCPServerConfig.
        """
        redis = await self._get_redis()
        raw = await redis.get(_redis_config_key(session_id))
        if not raw:
            return {}
        
        try:
            data = json.loads(raw)
            return {
                sid: MCPServerConfig.from_dict(cfg)
                for sid, cfg in data.items()
            }
        except Exception as e:
            logger.warning(f"Failed to load MCP configs for {session_id}: {e}")
            return {}
    
    async def _save_configs(self, session_id: str, configs: dict[str, MCPServerConfig]) -> None:
        """Save server configs to Redis.
        
        Args:
            session_id: Browser session ID.
            configs: Dict of server_id -> MCPServerConfig.
        """
        redis = await self._get_redis()
        data = {sid: cfg.to_dict() for sid, cfg in configs.items()}
        await redis.set(
            _redis_config_key(session_id),
            json.dumps(data),
            ex=int(self._session_ttl),
        )
    
    async def _get_or_connect(
        self, session_id: str, config: MCPServerConfig
    ) -> MCPServerConnection:
        """Get a live connection from local cache, or establish a new one.
        
        Args:
            session_id: Browser session ID.
            config: Server configuration.
            
        Returns:
            Active MCPServerConnection.
        """
        # Check local cache
        if session_id in self._local_cache:
            conn = self._local_cache[session_id].get(config.id)
            if conn and conn.status == "connected":
                conn.last_used = time.time()
                return conn
        
        # Cache miss -- reconnect from config
        logger.info(f"Reconnecting to MCP server '{config.name}' ({config.url}) for session {session_id}")
        
        client = MCPClient(url=config.url, api_key=config.api_key)
        await client.connect()
        
        schemas = await client.list_tools()
        raw_tools = create_mcp_tools(
            mcp_client=client,
            schemas=schemas,
            server_id=config.id,
            server_name=config.name,
        )
        
        # Detect and adapt heavy data types in MCP tool schemas
        tools = await adapt_mcp_tools_async(raw_tools)
        
        connection = MCPServerConnection(
            config=config,
            client=client,
            tools=tools,
            schemas=schemas,
        )
        
        # Store in local cache
        if session_id not in self._local_cache:
            self._local_cache[session_id] = {}
        self._local_cache[session_id][config.id] = connection
        
        return connection
    
    async def add_server(
        self,
        session_id: str,
        name: str,
        url: str,
        api_key: Optional[str] = None,
    ) -> MCPServerConnection:
        """Add and connect to an MCP server for a session.
        
        Args:
            session_id: The browser session ID.
            name: Human-readable name for the server.
            url: MCP server URL.
            api_key: Optional API key for authentication.
            
        Returns:
            MCPServerConnection with discovered tools.
        """
        server_id = uuid.uuid4().hex[:8]
        
        logger.info(f"Adding MCP server '{name}' ({url}) for session {session_id}")
        
        # Create config
        config = MCPServerConfig(
            id=server_id,
            name=name,
            url=url,
            api_key=api_key,
        )
        
        # Connect and discover tools
        connection = await self._get_or_connect(session_id, config)
        
        # Save config to Redis
        configs = await self._load_configs(session_id)
        configs[server_id] = config
        await self._save_configs(session_id, configs)
        
        logger.info(
            f"Added MCP server '{name}' with {len(connection.tools)} tools "
            f"(session: {session_id}, server_id: {server_id})"
        )
        
        return connection
    
    async def remove_server(self, session_id: str, server_id: str) -> bool:
        """Remove an MCP server from a session.
        
        Args:
            session_id: The browser session ID.
            server_id: ID of the server to remove.
            
        Returns:
            True if server was removed, False if not found.
        """
        # Remove from Redis
        configs = await self._load_configs(session_id)
        if server_id not in configs:
            return False
        
        del configs[server_id]
        await self._save_configs(session_id, configs)
        
        # Disconnect from local cache
        if session_id in self._local_cache:
            conn = self._local_cache[session_id].pop(server_id, None)
            if conn:
                try:
                    await conn.client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting MCP server: {e}")
        
        logger.info(f"Removed MCP server {server_id} from session {session_id}")
        return True
    
    async def get_servers(self, session_id: str) -> list[MCPServerConnection]:
        """Get all MCP servers for a session.
        
        Args:
            session_id: The browser session ID.
            
        Returns:
            List of MCPServerConnection objects.
        """
        configs = await self._load_configs(session_id)
        if not configs:
            return []
        
        connections = []
        for config in configs.values():
            try:
                conn = await self._get_or_connect(session_id, config)
                connections.append(conn)
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server '{config.name}': {e}")
        
        return connections
    
    async def get_tools(self, session_id: str) -> list[BaseTool]:
        """Get the MCP tools a session may actually use.

        Withholds tools from servers that have not been confirmed. This is the
        single chokepoint for both halves of "may use": ``get_tool_schemas``
        (what the model is shown) and ``get_tool_by_name`` (what the registry
        will dispatch to) both read from here, so an unconfirmed server is
        neither offered nor callable, rather than merely hidden.

        Args:
            session_id: The browser session ID.

        Returns:
            List of BaseTool instances (MCPToolWrapper or AdaptedMCPToolWrapper).
        """
        servers = await self.get_servers(session_id)
        tools: list[BaseTool] = []
        for server in servers:
            if server.status != "connected":
                continue
            if not server.config.confirmed:
                logger.info(
                    "Withholding %d tool(s) from unconfirmed MCP server %s (%s)",
                    len(server.tools),
                    server.name,
                    server.id,
                )
                continue
            tools.extend(server.tools)
        return tools

    async def confirm_server(self, session_id: str, server_id: str) -> bool:
        """Authorise a server's tools after a human has reviewed them.

        Returns True if the server was found and is now confirmed, False if no
        such server exists for the session. Confirming an already-confirmed
        server is a no-op and still returns True, so the call is idempotent.
        """
        configs = await self._load_configs(session_id)
        config = configs.get(server_id)
        if config is None:
            return False

        config.confirmed = True
        await self._save_configs(session_id, configs)

        # The live connection carries its own copy of the config, so update it
        # too or the running process keeps withholding tools until the local
        # cache is next rebuilt from Redis.
        cached = self._local_cache.get(session_id, {}).get(server_id)
        if cached is not None:
            cached.config.confirmed = True

        logger.info(
            "Confirmed MCP server %s (%s) for session %s",
            config.name,
            server_id,
            session_id,
        )
        return True
    
    async def get_tool_schemas(self, session_id: str) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all session MCP tools.
        
        Args:
            session_id: The browser session ID.
            
        Returns:
            List of OpenAI function schemas.
        """
        tools = await self.get_tools(session_id)
        return [tool.get_schema() for tool in tools]
    
    async def get_tool_by_name(
        self,
        session_id: str,
        tool_name: str,
    ) -> Optional[BaseTool]:
        """Get a specific MCP tool by name.
        
        Args:
            session_id: The browser session ID.
            tool_name: Full tool name (with mcp_ prefix).
            
        Returns:
            BaseTool if found (MCPToolWrapper or AdaptedMCPToolWrapper),
            None otherwise.
        """
        tools = await self.get_tools(session_id)
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def cleanup_local_cache(self) -> int:
        """Clean up stale entries from the local connection cache.
        
        Disconnects servers that haven't been used recently.
        
        Returns:
            Number of connections cleaned up.
        """
        now = time.time()
        cleaned = 0
        
        stale_sessions = []
        for session_id, servers in self._local_cache.items():
            stale_servers = []
            for server_id, conn in servers.items():
                if now - conn.last_used > self._session_ttl:
                    stale_servers.append(server_id)
            
            for server_id in stale_servers:
                conn = servers.pop(server_id)
                try:
                    await conn.client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting stale MCP server: {e}")
                cleaned += 1
            
            if not servers:
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            del self._local_cache[session_id]
        
        return cleaned
    
    async def start_cleanup_task(self, interval: float = 300.0) -> None:
        """Start background task for cleaning up stale local cache entries.
        
        Args:
            interval: Cleanup interval in seconds (default: 5 minutes).
        """
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    count = await self.cleanup_local_cache()
                    if count > 0:
                        logger.info(f"Cleaned up {count} stale MCP connections from local cache")
                except Exception as e:
                    logger.error(f"Error in MCP cache cleanup: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started MCP local cache cleanup task")
    
    def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("Stopped MCP local cache cleanup task")
    
    @property
    def session_count(self) -> int:
        """Get the number of locally cached sessions."""
        return len(self._local_cache)


# Module-level singleton (stateless facade + local cache, safe as singleton)
_mcp_session_store: Optional[MCPSessionStore] = None


def get_mcp_session_store() -> MCPSessionStore:
    """Get the MCP session store singleton.
    
    Returns:
        The MCPSessionStore instance.
    """
    global _mcp_session_store
    if _mcp_session_store is None:
        _mcp_session_store = MCPSessionStore()
        logger.info("Created MCP session store")
    return _mcp_session_store
