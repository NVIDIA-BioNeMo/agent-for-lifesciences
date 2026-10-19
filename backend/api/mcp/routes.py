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

"""API routes for MCP server management.

Provides endpoints for adding, listing, and removing MCP servers
on a per-session basis.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.deps import optional_owned_session, require_owned_session

from .adapter import AdaptedMCPToolWrapper
from .client import MCPConnectionError, MCPAuthError, MCPResponseTooLargeError
from .session_store import get_mcp_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class AddMCPServerRequest(BaseModel):
    """Request body for adding an MCP server."""
    name: str
    url: str  # Using str instead of HttpUrl for flexibility with localhost
    api_key: Optional[str] = None


class MCPServerResponse(BaseModel):
    """Response for a single MCP server."""
    id: str
    name: str
    url: str
    status: str
    tools_count: int
    tools: list[dict]
    confirmed: bool = False
    """Whether the listed tools are actually reachable by the agent.

    A newly added server is connected and enumerated but *not* authorised, so
    ``tools`` describes what it offers rather than what the model can call.
    Clients must render the difference; ``POST /servers/{id}/confirm`` closes it.
    """


class MCPServersListResponse(BaseModel):
    """Response for listing MCP servers."""
    servers: list[dict]
    count: int


class ToolInfo(BaseModel):
    """Information about a single tool."""
    name: str
    description: str
    source: str  # "builtin" or server name


class ToolsListResponse(BaseModel):
    """Response for listing all tools."""
    tools: list[ToolInfo]
    builtin_count: int
    mcp_count: int
    total: int


@router.post("/servers", response_model=MCPServerResponse)
async def add_mcp_server(
    request: AddMCPServerRequest,
    session_id: str = Depends(require_owned_session),
):
    """Add and connect to an MCP server."""
    store = get_mcp_session_store()
    
    try:
        logger.info(f"Adding MCP server: {request.name} ({request.url})")
        
        connection = await store.add_server(
            session_id=session_id,
            name=request.name,
            url=request.url,
            api_key=request.api_key,
        )
        
        tools_list: list[dict[str, Any]] = []
        for tool in connection.tools:
            schema = tool.get_schema()
            func = schema.get("function", {})
            tool_info: dict[str, Any] = {
                "name": tool.original_name,
                "description": func.get("description", ""),
                "full_name": tool.name,
            }
            if isinstance(tool, AdaptedMCPToolWrapper):
                tool_info["adapted"] = True
                tool_info["adaptation_info"] = tool.adaptation_info
            tools_list.append(tool_info)
        
        return MCPServerResponse(
            id=connection.id,
            name=connection.name,
            url=connection.url,
            status=connection.status,
            tools_count=len(connection.tools),
            tools=tools_list,
            confirmed=connection.config.confirmed,
        )
        
    except MCPAuthError as e:
        logger.warning(f"MCP auth error: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}",
        )
    except MCPConnectionError as e:
        logger.warning(f"MCP connection error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to MCP server: {str(e)}",
        )
    except MCPResponseTooLargeError as e:
        logger.warning(f"MCP response too large: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error adding MCP server: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}",
        )


@router.get("/servers", response_model=MCPServersListResponse)
async def list_mcp_servers(
    session_id: Optional[str] = Depends(optional_owned_session),
):
    """List all MCP servers for a session."""
    if not session_id:
        return MCPServersListResponse(servers=[], count=0)
    
    store = get_mcp_session_store()
    servers = await store.get_servers(session_id)
    
    return MCPServersListResponse(
        servers=[server.to_dict() for server in servers],
        count=len(servers),
    )


@router.delete("/servers/{server_id}")
async def remove_mcp_server(
    server_id: str,
    session_id: str = Depends(require_owned_session),
):
    """Remove an MCP server from a session."""
    store = get_mcp_session_store()
    
    removed = await store.remove_server(session_id, server_id)
    
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server {server_id} not found",
        )
    
    return {"status": "removed", "server_id": server_id}


@router.post("/servers/{server_id}/confirm")
async def confirm_mcp_server(
    server_id: str,
    session_id: str = Depends(require_owned_session),
):
    """Authorise a connected server's tools after a human has reviewed them.

    Adding a server connects and enumerates it, but its tools stay withheld
    from the agent until this is called -- the same shape as a skill contract,
    which yields no tool until confirmed. Idempotent: confirming an already
    confirmed server succeeds.
    """
    store = get_mcp_session_store()
    confirmed = await store.confirm_server(session_id, server_id)

    if not confirmed:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server {server_id} not found",
        )

    return {"status": "confirmed", "server_id": server_id}


@router.get("/tools", response_model=ToolsListResponse)
async def list_all_tools(
    session_id: Optional[str] = Depends(optional_owned_session),
):
    """List all available tools (builtin + MCP)."""
    from api.tools.registry import get_registry
    
    registry = get_registry()
    store = get_mcp_session_store()
    
    tools_info = []
    
    # Add builtin tools
    for tool_name in registry.tool_names:
        tool = registry.get(tool_name)
        if tool:
            schema = tool.get_schema()
            tools_info.append(ToolInfo(
                name=tool_name,
                description=schema.get("function", {}).get("description", ""),
                source="builtin",
            ))
    
    builtin_count = len(tools_info)
    
    # Add MCP tools if session provided
    mcp_count = 0
    if session_id:
        servers = await store.get_servers(session_id)
        for server in servers:
            for tool in server.tools:
                schema = tool.get_schema()
                func = schema.get("function", {})
                tools_info.append(ToolInfo(
                    name=tool.original_name,
                    description=func.get("description", ""),
                    source=server.name,
                ))
                mcp_count += 1
    
    return ToolsListResponse(
        tools=tools_info,
        builtin_count=builtin_count,
        mcp_count=mcp_count,
        total=len(tools_info),
    )
