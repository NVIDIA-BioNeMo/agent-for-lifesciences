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

"""FastAPI server that proxies chat requests to LLM with tool calling."""

import asyncio
import logging
import os
import json
import uuid
from contextlib import asynccontextmanager

STORAGE_ACCESS_API_ENABLED = os.getenv(
    "AEVOM_FEAT_STORAGE_ACCESS_API", "true"
).lower() in ("true", "1")
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.deps import (
    assert_owned_session,
    get_scoped_storage,
    get_session_ownership,
    require_owned_path_session,
    require_owned_session,
)
from api.auth.ownership import SessionOwnership
from api.auth.session import open_owned_session
from api.db.config import init_db, close_db, get_async_session
from api.redis import get_redis, close_redis, redis_health_check
from api.jobs.manager import JobManager
from api.artifacts import artifacts_router
from api.feedback import FEEDBACK_ENABLED, feedback_router
from api.prompts import get_system_prompt
from api.sandbox import sandbox_router
from api.selections import get_available_selections
from api.selections.models import SelectionData
from api.selections.roles import registry_payload as design_roles_payload
from api.selections.service import store_selection_data
from api.pending_ops import acknowledge_pending_op
from api.mcp import mcp_router, get_mcp_session_store
from api.nim import nim_router
from api.skills import (
    skills_router,
    get_skill_registry,
    get_skill_session_store,
    format_skill_catalog,
)
from api.secret_store import secrets_router
from api.spa import mount_spa
from api.storage import (
    ArtifactNotFoundError,
    DataType,
    InvalidReferenceError,
    SessionScopedStorage,
    artifact_owner,
    content_type_for,
    scoped_storage,
)
from api.storage.datatypes import is_text_type
from api.storage.images import normalize_png_reference
from api.tools.http_common import http_request_tool_enabled
from api.tools.registry import get_registry
from api.llm import get_client, get_active_model
from api.telemetry import init_telemetry
from api.training import PromptMode, get_training_prompt
from api.training.routes import router as training_router
from api.training.events import broadcaster

# Maximum tool call iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 20


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Initialize database, Redis, tool registry, telemetry, and LLM client
    logger.info("Starting Aevom API server...")

    # Every key mount is optional, so "at least one" is enforced here, first,
    # so that a pod with no key crash-loops with this as its first error.
    from api.llm.anthropic_models import refresh_anthropic_models
    from api.llm.config import require_llm_key
    from api.llm.openai_models import refresh_openai_models

    require_llm_key()

    # What this deployment will and will not meter. Stated on every boot so a
    # run can be attested from the container afterwards rather than inferred.
    from api.tools.budget import log_budget_state

    log_budget_state()

    await init_db()

    # Initialize Redis connection pool
    redis = get_redis()
    logger.info("Redis connection pool initialized")

    # Initialize OpenTelemetry with Phoenix
    init_telemetry()

    # Initialize tool registry on startup to catch errors early
    registry = get_registry()
    logger.info(f"Available tools: {registry.tool_names}")

    # Load curated first-party skills into the global registry
    get_skill_registry().load_builtin_skills()

    await asyncio.to_thread(refresh_anthropic_models)
    await asyncio.to_thread(refresh_openai_models)

    # Log active model
    model_config = get_active_model()
    logger.info(
        f"Active LLM model: {model_config.name} (reasoning: {model_config.supports_reasoning})"
    )

    # Start MCP session cleanup task
    mcp_store = get_mcp_session_store()
    await mcp_store.start_cleanup_task()

    # python_execute under OpenShell: fail at boot on a bad policy, then keep
    # warm spares ready so new sessions rarely wait for a sandbox.
    from api.sandbox.openshell_manager import get_openshell_manager, openshell_enabled

    if openshell_enabled():
        await get_openshell_manager().start()
        logger.info("python_execute runs in OpenShell sandboxes")

    yield

    if openshell_enabled():
        await get_openshell_manager().stop()

    # Stop MCP cleanup task
    mcp_store.stop_cleanup_task()

    # Shutdown: Close database and Redis connections
    logger.info("Shutting down Aevom API server...")
    await close_redis()
    await close_db()


# FastAPI's native OpenTelemetry is off: it uses the global provider Phoenix
# registers, so it would wrap every chat_request trace in an HTTP server span
# and turn every polling request into a trace of its own.
app = FastAPI(
    title="Aevom Chat API",
    lifespan=lifespan,
    telemetry={"tracing": False, "metrics": False, "logs": False},
)

# Mount sandbox routes for pyexec data access
app.include_router(sandbox_router)

# Mount training routes for DPO data collection
app.include_router(training_router)

# Mount MCP server management routes
app.include_router(mcp_router)

# Mount self-hosted NIM registration routes
app.include_router(nim_router)

# Mount skill management routes
app.include_router(skills_router)

# Mount secret (API key) management routes
app.include_router(secrets_router)

# Mount session artifact seeding + submission read-back. Registered before the
# SPA catch-all, like every other API router.
app.include_router(artifacts_router)

# Mount the free-text feedback email relay. Its POST /api/feedback/email is
# unrelated to the POST /api/feedback below, which annotates a Phoenix span.
app.include_router(feedback_router)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # The current contract: the new user message only. History comes from the
    # backend's record, so the browser can no longer silently truncate the
    # trajectory by keeping only role and content.
    message: str | None = None
    # Legacy contract, kept for scripts and eval harnesses that post a whole
    # transcript. Ignored when `message` is set.
    messages: list[Message] = []
    max_tokens: int = 8192
    temperature: float | None = None  # Use model default if not specified
    top_p: float = 1.0
    viewer_state: dict | None = None  # Current state of the Molstar viewer
    # Required; from POST /api/session/create, and owned by the caller's token.
    session_id: str | None = None
    model_id: str | None = None  # Override active model for this request
    # Training mode parameters
    prompt_mode: str | None = None  # "production", "oracle", or "weak"
    training_query_id: str | None = None  # Links responses to a training query
    expected_tool: str | None = None  # Required for oracle mode
    oracle_prompt: str | None = None  # Per-query oracle prompt (fetched from DB)


# Note: Selection tracking is now handled by the selection registry
# (api.selections.registry) which persists selections across conversation turns


class FeedbackRequest(BaseModel):
    """User feedback on a chat response (thumbs up/down).

    The span_id links this feedback to the Phoenix trace for annotation.
    """

    span_id: str  # The span_id from the chat_request span
    label: str  # "thumbs_up" or "thumbs_down"
    score: int  # 1 for thumbs_up, 0 for thumbs_down
    metadata: dict | None = None  # Optional: user_id, session_id, explanation, etc.


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
    ownership: SessionOwnership = Depends(get_session_ownership),
):
    """Stream chat responses using the LangGraph agent."""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    await assert_owned_session(http_request, request.session_id, ownership)
    return EventSourceResponse(
        stream_chat_with_langgraph(request), media_type="text/event-stream"
    )


@app.get("/api/conversation/{session_id}")
async def get_conversation(session_id: str = Depends(require_owned_path_session)):
    """Return the session's transcript and findings.

    The frontend renders from this rather than from its own memory, so a
    reload -- or a second browser tab -- sees the same conversation the model
    does. Findings are returned alongside because they survive a model switch
    that starts a fresh transcript.
    """
    from api.conversations import ConversationStore
    from api.db.config import get_session_factory

    async with get_session_factory()() as db_session:
        store = ConversationStore(db_session)
        return {
            "session_id": session_id,
            "messages": await store.get_transcript(session_id),
            "findings": await store.get_findings(session_id),
        }


async def _open_turn(
    request: ChatRequest,
    session_id: str,
    db_session_factory: Any,
    model_id: str,
) -> tuple[list[dict], Any]:
    """Start a turn and return the history to run it against.

    Under the current contract the request carries only the new user message
    and history is read back from the record, which is what stops the browser
    from deciding how much of the trajectory the model gets to see.

    Legacy callers post a full transcript instead. Those are honoured as sent
    -- an eval harness constructing an exact prompt must get that prompt -- but
    the turn is still recorded, so the record stays complete either way.

    Returns:
        The conversation history to seed the graph with, and the recorder that
        will capture this turn.
    """
    from api.conversations import TurnRecorder

    try:
        recorder = await TurnRecorder.open(db_session_factory, session_id, model_id)
    except Exception:
        logger.warning("Could not open a turn; proceeding unrecorded", exc_info=True)
        recorder = None

    if request.message is None:
        history = [{"role": m.role, "content": m.content} for m in request.messages]
        if recorder is not None and history and history[-1]["role"] == "user":
            await recorder.record_user(history[-1]["content"])
        return history, recorder

    history: list[dict] = []
    if recorder is not None:
        from api.conversations import ConversationStore

        try:
            async with db_session_factory() as db_session:
                store = ConversationStore(db_session)
                history = await store.get_transcript(session_id)
        except Exception:
            logger.warning("Could not load conversation history", exc_info=True)
        await recorder.record_user(request.message)

    history.append({"role": "user", "content": request.message})
    return history, recorder


async def stream_chat_with_langgraph(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream chat using the LangGraph agent architecture.

    Replaces the bespoke env/policy/runner loop with a LangGraph StateGraph.
    Yields SSE-formatted JSON events identical to the v1 endpoint.
    """
    from api.db.config import get_session_factory
    from api.graph.llm import ChatAevomLLM
    from api.graph.tools import build_langchain_tools
    from api.graph.agent import build_agent_graph
    from api.graph.streaming import stream_graph

    session_id = request.session_id
    request_id = str(uuid.uuid4())
    _claim_sandbox_in_background(session_id)

    try:
        # Build system prompt
        if request.prompt_mode:
            try:
                prompt_mode = PromptMode(request.prompt_mode)
                base_prompt = get_training_prompt(
                    prompt_mode, request.expected_tool, request.oracle_prompt,
                )
            except ValueError:
                base_prompt = get_system_prompt()
        else:
            base_prompt = get_system_prompt()

        # Use per-request client when model_id is specified, otherwise singleton
        if request.model_id:
            from api.llm.config import get_model_config
            from api.llm.client import create_client
            model_config = get_model_config(request.model_id)
            client = create_client(model_config)
        else:
            client = get_client()
        registry = get_registry()
        tool_schemas = await registry.get_all_schemas(session_id=session_id)

        # prompts.md describes the NIM tools; when they were withheld for want
        # of an NVIDIA key, say so rather than let the model plan around them.
        unavailable = getattr(registry, "unavailable", None)
        if unavailable:
            base_prompt = (
                f"{base_prompt}\n\n[UNAVAILABLE TOOLS]\n"
                f"These tools are not configured in this deployment (no NVIDIA "
                f"API key): {', '.join(unavailable)}, and the "
                f"run_structure_prediction workflow. Do not call them or plan "
                f"around them; say so if the user's request needs one."
            )

        # Level 1 progressive disclosure: inject the compact skill catalog
        # (name + description only) into the system prompt so the model knows
        # which skills exist and can load them on demand via load_skill.
        try:
            skill_catalog = await get_skill_session_store().get_catalog(session_id)
            catalog_block = format_skill_catalog(skill_catalog)
            if catalog_block:
                base_prompt = f"{base_prompt}\n\n{catalog_block}"
        except Exception as e:
            logger.warning(f"Failed to inject skill catalog into prompt: {e}")

        try:
            from api.secret_store import get_secret_session_store
            from api.secret_store.prompt import stored_keys_block

            stored_secrets = await get_secret_session_store().list_secrets(session_id)
            keys_block = stored_keys_block([s.name for s in stored_secrets])
            if keys_block:
                base_prompt = f"{base_prompt}\n\n{keys_block}"
        except Exception as e:
            logger.warning(f"Failed to inject stored key names into prompt: {e}")

        rendered_prompt = client.render_system_prompt(base_prompt, tool_schemas)

        # Build LangGraph components
        llm = ChatAevomLLM(client=client)
        tools = await build_langchain_tools(session_id=session_id)
        graph = build_agent_graph(llm, tools, rendered_prompt)

        # Create storage (same pattern as v1)
        db_session_factory = get_session_factory()
        db_session = db_session_factory()
        storage = scoped_storage(db_session, session_id)
        # Durable job semantics (ADR 0001, P0): tools that opt into job
        # tracking (tool.is_async == True, e.g. vina_dock) get their call
        # recorded as a Job row on the same session as storage. Every other
        # tool's execution path is unaffected.
        job_manager = JobManager(db_session)

        conversation_history, recorder = await _open_turn(
            request, session_id, db_session_factory, client.config.id
        )

        # Stream the graph
        async for event in stream_graph(
            graph,
            system_prompt=rendered_prompt,
            conversation_history=conversation_history,
            viewer_state=request.viewer_state,
            session_id=session_id,
            storage=storage,
            request_id=request_id,
            db_session_factory=db_session_factory,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            model_name=client.config.name,
            recorder=recorder,
            job_manager=job_manager,
        ):
            yield json.dumps(event)

    except Exception as e:
        logger.exception(f"Error in LangGraph chat: {e}")
        yield json.dumps({"content": f"Error: {str(e)}"})
    finally:
        if "db_session" in locals():
            await db_session.commit()
            await db_session.close()


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    from api.conversations.recorder import swallowed_writes

    redis_ok = await redis_health_check()
    status = "healthy" if redis_ok else "degraded"
    # Ledger writes fail silently on purpose -- losing a findings row must not
    # take down a turn that already did real work -- but silence made a live
    # data-loss bug invisible for a whole test run. Reported here so "the
    # ledger is quietly dropping writes" is observable. Non-empty does not
    # make the service unhealthy; it means look at the logs.
    return {
        "status": status,
        "redis": redis_ok,
        "swallowed_ledger_writes": swallowed_writes(),
    }


@app.get("/api/info")
async def info():
    """Return application info including active model and feature flags."""
    from api.llm.config import get_active_model, pretty_name
    model = get_active_model()
    return {
        "model_id": model.id,
        "model_name": model.name,
        "pretty_name": pretty_name(model),
        "provider": model.provider,
        "supports_reasoning": model.supports_reasoning,
        "features": {
            "storage_access_api": STORAGE_ACCESS_API_ENABLED,
            "feedback_email": FEEDBACK_ENABLED,
            "http_request_tool": http_request_tool_enabled(),
        },
    }


@app.get("/api/models")
async def list_models():
    """Return available models filtered by API key availability.

    Anthropic and OpenAI models are discovered from the keys themselves
    (refreshed on a TTL), so populating API_KEY_ANTHROPIC or API_KEY_OPENAI is
    all it takes for them to appear.
    """
    from api.llm.anthropic_models import refresh_anthropic_models
    from api.llm.config import available_models, get_active_model, pretty_name
    from api.llm.openai_models import refresh_openai_models

    await asyncio.to_thread(refresh_anthropic_models)
    await asyncio.to_thread(refresh_openai_models)
    available = [
        {
            "id": config.id,
            "pretty_name": pretty_name(config),
            "provider": config.provider,
            "supports_reasoning": config.supports_reasoning,
        }
        for config in available_models()
    ]
    return {"models": available, "default": get_active_model().id}


def _require_storage_api():
    if not STORAGE_ACCESS_API_ENABLED:
        raise HTTPException(status_code=404, detail="Storage access API is disabled")


@app.post("/api/session/create")
async def create_session(
    request: Request,
    response: Response,
    ownership: SessionOwnership = Depends(get_session_ownership),
):
    """Create a new session owned by the caller. See ``open_owned_session``."""
    created = await open_owned_session(request, response, ownership)
    _claim_sandbox_in_background(created["session_id"])
    return created


_background_claims: set[asyncio.Task] = set()


def _claim_sandbox_in_background(session_id: str) -> None:
    """Bind an OpenShell sandbox to the session before python_execute needs it.

    Fire-and-forget: python_execute waits for the sandbox itself, so a failed
    or slow claim here costs latency, never correctness.
    """
    from api.sandbox.openshell_manager import get_openshell_manager, openshell_enabled

    if not openshell_enabled():
        return

    async def _claim() -> None:
        try:
            await get_openshell_manager().claim(session_id)
        except Exception as exc:
            logger.warning("Could not pre-claim a sandbox for session %s: %s", session_id, exc)

    task = asyncio.create_task(_claim())
    _background_claims.add(task)
    task.add_done_callback(_background_claims.discard)


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback (thumbs up/down) for a chat response.

    This endpoint uses the Phoenix client to add a span annotation to the
    trace identified by span_id. The annotation includes the feedback label,
    score, and any additional metadata.

    Args:
        request: Feedback request with span_id, label, score, and optional metadata.

    Returns:
        Success status and the span_id that was annotated.
    """
    try:
        from phoenix.client import Client

        # Get Phoenix base URL from environment (same host as telemetry, but HTTP port)
        phoenix_base_url = os.environ.get("PHOENIX_CLIENT_URL", "http://phoenix:6006")

        logger.info(f"Submitting feedback for span {request.span_id}: {request.label}")

        # Create Phoenix client and add the span annotation
        client = Client(base_url=phoenix_base_url)

        # Build metadata with defaults
        annotation_metadata = request.metadata or {}

        # Add the span annotation using the spans resource
        # annotator_kind is a string literal: "HUMAN", "LLM", or "CODE"
        client.spans.add_span_annotation(
            span_id=request.span_id,
            annotation_name="user_feedback",
            annotator_kind="HUMAN",
            label=request.label,
            score=float(request.score),
            metadata=annotation_metadata,
            sync=True,  # Wait for confirmation
        )

        logger.info(
            f"Successfully logged feedback annotation for span {request.span_id}"
        )

        # Broadcast feedback event for training mode status updates
        training_query_id = annotation_metadata.get("training_query_id")
        if training_query_id:
            await broadcaster.publish(
                "feedback_submitted",
                {
                    "query_id": training_query_id,
                    "label": request.label,
                    "span_id": request.span_id,
                },
            )

        return {
            "status": "success",
            "span_id": request.span_id,
            "label": request.label,
        }

    except ImportError as e:
        logger.error(f"Phoenix client not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Phoenix client not available. Install arize-phoenix to enable feedback.",
        )
    except Exception as e:
        logger.exception(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit feedback: {str(e)}"
        )


@app.get("/api/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    session_id: str = Depends(require_owned_session),
    session: AsyncSession = Depends(get_async_session),
):
    """Get status and result of an async job the caller's session started.

    Args:
        job_id: UUID of the job.
        session_id: The owned session; another session's job is a 404.

    Returns:
        Job status and result if complete.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    manager = JobManager(session)
    status = await manager.get_job_status(job_uuid, session_id=session_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return status


@app.post("/api/selection")
async def store_selection(
    data: SelectionData,
    session_id: str = Depends(require_owned_session),
    session: AsyncSession = Depends(get_async_session),
    redis_client=Depends(get_redis),
):
    """Store a selection posted by the frontend and register it for the session.

    Two producers post here. After a ``structure_query`` the viewer posts what
    the agent's MolQL matched, with the ``query_id`` that unblocks the waiting
    tool call. When the user picks residues by hand it posts them with
    ``provenance="user"`` and no ``query_id``. Either way the result is the same
    kind of artifact; see :func:`api.selections.service.store_selection_data`.

    Returns:
        The reference, counts, provenance and compact residue listing.
    """
    storage = scoped_storage(session, session_id)

    async def acknowledge(query_id: str) -> bool:
        return await acknowledge_pending_op(redis_client, query_id)

    result = await store_selection_data(
        data,
        storage=storage,
        redis_client=redis_client,
        session_id=session_id,
        acknowledge=acknowledge,
        commit=session.commit,
    )
    return result


@app.get("/api/selections")
async def get_selections(
    session_id: str = Depends(require_owned_session),
    redis_client=Depends(get_redis),
):
    """Get all available selections from the registry.

    This endpoint allows clients to query what selections are currently
    available for analysis.

    Returns:
        List of available selections with their metadata.
    """
    selections = await get_available_selections(redis_client, session_id)
    return {
        "selections": [s.to_dict() for s in selections],
        "count": len(selections),
    }


@app.get("/api/design-roles")
async def get_design_roles():
    """The roles a selection can play in a design, and the engines they run on.

    One definition, read by the viewer (which buttons to offer for a pick and
    how to colour them), the language model and the backend compiler.
    """
    return design_roles_payload()


@app.get("/api/storage/artifacts")
async def list_storage_artifacts(
    storage: SessionScopedStorage = Depends(get_scoped_storage),
):
    """List stored data artifacts for the frontend storage strip.

    Deliberately not behind ``_require_storage_api()``: the storage strip is
    core UI rather than an access API, and this returns only the metadata the
    strip renders (reference, type, description) for the caller's own session,
    never artifact content. The flag gates the two endpoints below, which do
    return content. Gating this one blanked the strip wherever the flag was
    off -- the k8s deployment -- because the frontend hydrates from here.
    """
    artifacts = await storage.list_artifacts(limit=50)
    return {"artifacts": artifacts}


@app.get("/api/storage/artifact/{reference}")
async def get_artifact_content(
    reference: str,
    storage: SessionScopedStorage = Depends(get_scoped_storage),
):
    """Get a single artifact's content by reference ID.

    Another session's artifact is a 404, the same as a missing one.
    """
    _require_storage_api()
    try:
        content, metadata = await storage.retrieve_with_metadata(reference)
        data_type = storage.get_data_type(reference)
    except (ArtifactNotFoundError, InvalidReferenceError):
        raise HTTPException(status_code=404, detail=f"Artifact not found: {reference}")

    if is_text_type(data_type):
        content_str = content.decode("utf-8")
    else:
        import base64
        content_str = base64.b64encode(content).decode("utf-8")

    return {
        "reference": reference,
        "data_type": data_type,
        "content": content_str,
        "metadata": metadata,
    }


@app.get("/api/storage/export")
async def export_artifacts(
    data_type: str | None = None,
    format: str = "json",
    storage: SessionScopedStorage = Depends(get_scoped_storage),
):
    """Export multiple artifacts with their content for external analysis.

    Uses a single batch query instead of N+1 retrieval.
    """
    _require_storage_api()
    results = await storage.export_artifacts(data_type=data_type, limit=100)

    if format == "zip":
        import io
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in results:
                ext = item["data_type"]
                filename = f"{item['reference']}.{ext}"
                zf.writestr(filename, item["content"])

        zip_buffer.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=aevom_export.zip"}
        )

    return {"artifacts": results, "count": len(results)}


@app.get("/api/images/{reference}")
async def get_image(
    reference: str,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    ownership: SessionOwnership = Depends(get_session_ownership),
):
    """Serve a molecule depiction stored as a `png` data artifact.

    Takes no session_id, because model-authored markdown embeds bare
    ``/api/images/png_<hex>`` URLs. The owning session is read from the
    artifact instead, and the caller's owner token (the cookie an ``<img>``
    sends) must own it. Restricted to `png_` references so it cannot be used as
    a generic artifact reader.

    Args:
        reference: A `png_<hex>` artifact reference, optionally with a trailing
            `.png` (model-authored markdown tends to add the extension).

    Returns:
        The raw PNG bytes.
    """
    artifact_reference = normalize_png_reference(reference)
    if artifact_reference is None:
        raise HTTPException(status_code=404, detail="Image not found")

    owner = await artifact_owner(session, artifact_reference)
    if owner is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        await assert_owned_session(request, owner, ownership)
    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Image not found")
        raise
    try:
        content = await scoped_storage(session, owner).retrieve(artifact_reference)
    except (ArtifactNotFoundError, InvalidReferenceError):
        raise HTTPException(status_code=404, detail="Image not found")

    return Response(content=content, media_type=content_type_for(DataType.PNG))


# Serve static files (React build)
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    mount_spa(app, static_path)
