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

"""Training data collection API routes.

This module provides endpoints for managing training queries, tracking pair
collection status, and exporting DPO training datasets.
"""

import json
import logging
import os
import secrets
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from api.db.config import get_async_session
from api.db.models import TrainingQuery
from api.training.prompts import PromptMode, get_prompt_modes, get_training_prompt
from api.training.events import broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

# Phoenix configuration
AEVOM_PROJECT_ID = "UHJvamVjdDoy"  # Base64 encoded project ID for "aevom"


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateQueryRequest(BaseModel):
    """Request to create a new training query."""
    query_text: str
    expected_tool: Optional[str] = None
    category: Optional[str] = None
    oracle_prompt: Optional[str] = None


class UpdateOraclePromptRequest(BaseModel):
    """Request to update a query's oracle prompt."""
    oracle_prompt: str


class QueryResponse(BaseModel):
    """Response for a training query with collection status."""
    id: str
    query_text: str
    expected_tool: Optional[str]
    category: Optional[str]
    created_at: str
    is_active: bool
    # Collection status (computed from Phoenix)
    status: str  # none, chosen_only, rejected_only, complete, multiple
    chosen_count: int
    rejected_count: int
    total_responses: int


class StatusSummary(BaseModel):
    """Summary of pair collection progress."""
    total_queries: int
    complete_pairs: int
    in_progress: int
    not_started: int


class QueriesResponse(BaseModel):
    """Response containing all queries with status."""
    queries: list[QueryResponse]
    summary: StatusSummary


# ============================================================================
# Phoenix Query Helpers
# ============================================================================

def _get_phoenix_url() -> str:
    """Get Phoenix GraphQL endpoint URL."""
    # Check for explicit URL
    env_url = os.environ.get("PHOENIX_CLIENT_URL", "http://phoenix:6006")
    return f"{env_url.rstrip('/')}/graphql"


def _query_phoenix(query: str) -> dict[str, Any]:
    """Execute a GraphQL query against Phoenix.
    
    Args:
        query: GraphQL query string.
        
    Returns:
        JSON response from Phoenix.
    """
    url = _get_phoenix_url()
    payload = json.dumps({"query": query}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
            if "errors" in result:
                logger.warning(f"Phoenix GraphQL errors: {result['errors']}")
            return result
    except urllib.error.URLError as e:
        logger.error(f"Failed to query Phoenix at {url}: {e}")
        return {"data": None, "error": str(e)}


def _get_training_spans() -> list[dict]:
    """Get all spans with training metadata from Phoenix.
    
    Returns:
        List of span data with training attributes.
    """
    # Use a high limit to ensure we get all training spans
    # Both this and _get_feedback_annotations must return the same spans
    query = f'''{{
        node(id: "{AEVOM_PROJECT_ID}") {{
            ... on Project {{
                spans(
                    first: 10000,
                    sort: {{col: startTime, dir: desc}}
                ) {{
                    edges {{
                        node {{
                            spanId
                            name
                            attributes
                        }}
                    }}
                }}
            }}
        }}
    }}'''
    
    result = _query_phoenix(query)
    
    if not result.get("data") or not result["data"].get("node"):
        return []
    
    spans = []
    for edge in result["data"]["node"]["spans"]["edges"]:
        span = edge["node"]
        if span.get("attributes"):
            try:
                attrs = json.loads(span["attributes"]) if isinstance(span["attributes"], str) else span["attributes"]
                training = attrs.get("training", {})
                if training.get("query_id"):
                    spans.append({
                        "span_id": span["spanId"],
                        "name": span["name"],
                        "query_id": training.get("query_id"),
                        "mode": training.get("mode"),
                    })
            except json.JSONDecodeError:
                pass
    
    return spans


@dataclass
class FeedbackInfo:
    """Feedback annotation with version metadata."""
    label: str  # thumbs_up or thumbs_down
    version: Optional[str] = None


def _get_feedback_annotations() -> dict[str, FeedbackInfo]:
    """Get feedback annotations (thumbs up/down) from Phoenix with version info.
    
    Returns:
        Dict mapping span_id to FeedbackInfo (label and version).
    """
    # Use the same high limit as _get_training_spans to ensure consistency
    query = f'''{{
        node(id: "{AEVOM_PROJECT_ID}") {{
            ... on Project {{
                spans(
                    first: 10000,
                    sort: {{col: startTime, dir: desc}}
                ) {{
                    edges {{
                        node {{
                            spanId
                            spanAnnotations {{
                                name
                                label
                                metadata
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}'''
    
    result = _query_phoenix(query)
    
    if not result.get("data") or not result["data"].get("node"):
        return {}
    
    feedback = {}
    for edge in result["data"]["node"]["spans"]["edges"]:
        span = edge["node"]
        span_id = span.get("spanId")
        annotations = span.get("spanAnnotations", [])
        
        for ann in annotations:
            if ann.get("name") == "user_feedback":
                label = ann.get("label")
                if label in ("thumbs_up", "thumbs_down"):
                    # Extract version from annotation metadata
                    # Try version_number first, fall back to training_version for old data
                    version = None
                    metadata = ann.get("metadata")
                    if metadata:
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except json.JSONDecodeError:
                                metadata = {}
                        version = metadata.get("version_number") or metadata.get("training_version")
                    feedback[span_id] = FeedbackInfo(label=label, version=version)
                    break
    
    return feedback


def _parse_version(version_str: str) -> tuple:
    """Parse a semantic version string into a comparable tuple.
    
    Handles versions like "0.1.0", "v0.1.0", "1.2.3-dev", etc.
    """
    if not version_str:
        return (0, 0, 0)
    
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")
    
    # Remove any suffix after hyphen (e.g., "0.0.0-dev" -> "0.0.0")
    version_str = version_str.split("-")[0]
    
    # Split by dots and convert to integers
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    
    # Pad to at least 3 components
    while len(parts) < 3:
        parts.append(0)
    
    return tuple(parts)


def _version_gte(version: str, min_version: str) -> bool:
    """Check if version is greater than or equal to min_version."""
    return _parse_version(version) >= _parse_version(min_version)


def _get_unique_versions(feedback: dict[str, FeedbackInfo]) -> list[str]:
    """Extract unique versions from feedback annotations, sorted descending."""
    versions = set()
    for info in feedback.values():
        if info.version:
            versions.add(info.version)
    
    # Sort by parsed version, descending (newest first)
    return sorted(versions, key=_parse_version, reverse=True)


def _compute_query_status(
    query_id: str,
    spans: list[dict],
    feedback: dict[str, FeedbackInfo],
    min_version: Optional[str] = None,
) -> tuple[str, int, int, int]:
    """Compute collection status for a query.
    
    Args:
        query_id: The training query ID.
        spans: List of spans with training metadata.
        feedback: Dict mapping span_id to FeedbackInfo.
        min_version: If provided, only count feedback with version >= min_version.
        
    Returns:
        Tuple of (status, chosen_count, rejected_count, total_responses).
    """
    # Filter spans for this query
    query_spans = [s for s in spans if s.get("query_id") == query_id]
    
    # Count feedback types
    chosen_count = 0
    rejected_count = 0
    total_responses = len(query_spans)
    
    for span in query_spans:
        span_id = span.get("span_id")
        info = feedback.get(span_id)
        if not info:
            continue
        
        # Apply version filter if specified
        if min_version:
            # Skip feedback without version or with version below minimum
            if not info.version or not _version_gte(info.version, min_version):
                continue
        
        if info.label == "thumbs_up":
            chosen_count += 1
        elif info.label == "thumbs_down":
            rejected_count += 1
    
    # Determine status
    if chosen_count == 0 and rejected_count == 0:
        status = "none"
    elif chosen_count > 0 and rejected_count > 0:
        if chosen_count > 1 or rejected_count > 1:
            status = "multiple"
        else:
            status = "complete"
    elif chosen_count > 0:
        status = "chosen_only"
    else:
        status = "rejected_only"
    
    return status, chosen_count, rejected_count, total_responses


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/events")
async def query_events(request: Request):
    """SSE endpoint for real-time query updates.
    
    Clients subscribe to this endpoint to receive notifications when
    queries are created, updated, or deleted.
    
    Event types:
        - query_created: New query added
        - query_deleted: Query deactivated
        - query_updated: Query modified (future use)
    """
    async def event_generator():
        async for event in broadcaster.subscribe():
            # Check if client disconnected
            if await request.is_disconnected():
                break
            yield event
    
    return EventSourceResponse(event_generator())


@router.get("/prompts")
async def list_prompts():
    """List available prompt modes for training."""
    return {"prompts": get_prompt_modes()}


@router.get("/versions")
async def list_versions():
    """Get available training versions from feedback annotations.
    
    Returns the current version and all versions found in Phoenix feedback
    annotations, sorted descending (newest first).
    
    Returns:
        current_version: The currently running version from VERSION_NUMBER.
        versions: List of unique versions found in feedback annotations.
    """
    # Get current version from environment
    current_version = os.environ.get("VERSION_NUMBER", "0.0.0-dev")
    
    # Get feedback annotations and extract unique versions
    feedback = _get_feedback_annotations()
    versions = _get_unique_versions(feedback)
    
    # Ensure current version is in the list (even if no feedback exists for it yet)
    if current_version not in versions:
        versions = [current_version] + versions
    
    return {
        "current_version": current_version,
        "versions": versions,
    }


@router.get("/queries", response_model=QueriesResponse)
async def list_queries(
    session: AsyncSession = Depends(get_async_session),
    include_inactive: bool = False,
    min_version: Optional[str] = None,
):
    """List all training queries with their collection status.
    
    Args:
        include_inactive: Whether to include inactive queries.
        min_version: If provided, only count feedback with version >= this value.
        
    Returns:
        Queries with collection status and summary.
    """
    # Fetch queries from database
    stmt = select(TrainingQuery)
    if not include_inactive:
        stmt = stmt.where(TrainingQuery.is_active == True)
    stmt = stmt.order_by(TrainingQuery.created_at.desc())
    
    result = await session.execute(stmt)
    db_queries = result.scalars().all()
    
    # Get spans and feedback from Phoenix
    spans = _get_training_spans()
    feedback = _get_feedback_annotations()
    
    # Build response with status
    queries = []
    complete_pairs = 0
    in_progress = 0
    not_started = 0
    
    for q in db_queries:
        status, chosen, rejected, total = _compute_query_status(
            q.id, spans, feedback, min_version=min_version
        )
        
        queries.append(QueryResponse(
            id=q.id,
            query_text=q.query_text,
            expected_tool=q.expected_tool,
            category=q.category,
            created_at=q.created_at.isoformat() if q.created_at else "",
            is_active=q.is_active,
            status=status,
            chosen_count=chosen,
            rejected_count=rejected,
            total_responses=total,
        ))
        
        # Update summary counts
        if status in ("complete", "multiple"):
            complete_pairs += 1
        elif status == "none":
            not_started += 1
        else:
            in_progress += 1
    
    return QueriesResponse(
        queries=queries,
        summary=StatusSummary(
            total_queries=len(queries),
            complete_pairs=complete_pairs,
            in_progress=in_progress,
            not_started=not_started,
        ),
    )


@router.post("/queries")
async def create_query(
    request: CreateQueryRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Create a new training query.
    
    Args:
        request: Query creation request.
        
    Returns:
        The created query.
    """
    # Generate unique ID
    query_id = f"q_{secrets.token_hex(4)}"
    
    query = TrainingQuery(
        id=query_id,
        query_text=request.query_text,
        expected_tool=request.expected_tool,
        category=request.category,
        oracle_prompt=request.oracle_prompt,
        oracle_prompt_updated_at=datetime.now(timezone.utc) if request.oracle_prompt else None,
        is_active=True,
    )
    
    session.add(query)
    await session.commit()
    await session.refresh(query)
    
    logger.info(f"Created training query: {query_id}")
    
    # Build response with initial status (new queries have no responses yet)
    query_response = QueryResponse(
        id=query.id,
        query_text=query.query_text,
        expected_tool=query.expected_tool,
        category=query.category,
        created_at=query.created_at.isoformat() if query.created_at else "",
        is_active=query.is_active,
        status="none",
        chosen_count=0,
        rejected_count=0,
        total_responses=0,
    )
    
    # Broadcast event to all subscribers
    await broadcaster.publish("query_created", {"query": query_response.model_dump()})
    
    return query.to_dict()


@router.delete("/queries/{query_id}")
async def deactivate_query(
    query_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Deactivate a training query (soft delete).
    
    Args:
        query_id: The query ID to deactivate.
        
    Returns:
        Success status.
    """
    stmt = select(TrainingQuery).where(TrainingQuery.id == query_id)
    result = await session.execute(stmt)
    query = result.scalar_one_or_none()
    
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    query.is_active = False
    await session.commit()
    
    logger.info(f"Deactivated training query: {query_id}")
    
    # Broadcast deletion event to all subscribers
    await broadcaster.publish("query_deleted", {"id": query_id})
    
    return {"status": "deactivated", "id": query_id}


@router.get("/status")
async def get_collection_status(
    session: AsyncSession = Depends(get_async_session),
    min_version: Optional[str] = None,
):
    """Get pair collection status summary.
    
    Args:
        min_version: If provided, only count feedback with version >= this value.
    
    Returns:
        Summary of collection progress.
    """
    # This is essentially the same as list_queries but returns just the summary
    response = await list_queries(
        session=session, include_inactive=False, min_version=min_version
    )
    return {"summary": response.summary}


@router.get("/prompt-preview")
async def preview_prompt(
    mode: str,
    expected_tool: Optional[str] = None,
):
    """Preview a prompt for a given mode.
    
    Args:
        mode: The prompt mode (production, oracle, weak).
        expected_tool: Required for oracle mode.
        
    Returns:
        The rendered prompt.
    """
    try:
        prompt_mode = PromptMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    
    try:
        prompt = get_training_prompt(prompt_mode, expected_tool)
        return {
            "mode": mode,
            "prompt": prompt,
            "length": len(prompt),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/queries/{query_id}/oracle-prompt")
async def get_oracle_prompt(
    query_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Get the oracle prompt for a specific query.
    
    Args:
        query_id: The query ID.
        
    Returns:
        The oracle prompt text and metadata.
    """
    stmt = select(TrainingQuery).where(TrainingQuery.id == query_id)
    result = await session.execute(stmt)
    query = result.scalar_one_or_none()
    
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    return {
        "query_id": query_id,
        "oracle_prompt": query.oracle_prompt,
        "oracle_prompt_updated_at": query.oracle_prompt_updated_at.isoformat() if query.oracle_prompt_updated_at else None,
    }


@router.put("/queries/{query_id}/oracle-prompt")
async def update_oracle_prompt(
    query_id: str,
    request: UpdateOraclePromptRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Update the oracle prompt for a specific query.
    
    Args:
        query_id: The query ID.
        request: The new oracle prompt text.
        
    Returns:
        The updated oracle prompt and metadata.
    """
    stmt = select(TrainingQuery).where(TrainingQuery.id == query_id)
    result = await session.execute(stmt)
    query = result.scalar_one_or_none()
    
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    query.oracle_prompt = request.oracle_prompt
    query.oracle_prompt_updated_at = datetime.now(timezone.utc)
    
    await session.commit()
    await session.refresh(query)
    
    logger.info(f"Updated oracle prompt for query: {query_id}")
    
    return {
        "query_id": query_id,
        "oracle_prompt": query.oracle_prompt,
        "oracle_prompt_updated_at": query.oracle_prompt_updated_at.isoformat() if query.oracle_prompt_updated_at else None,
    }
