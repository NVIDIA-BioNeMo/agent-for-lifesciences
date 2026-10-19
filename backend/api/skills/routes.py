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

"""API routes for skill management (install, list, remove).

Mirrors ``api.mcp.routes``. Skills are installed per browser session; curated
builtin skills are always present and are not removable.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.auth.deps import optional_owned_session, require_owned_session
from api.db.config import get_session_factory
from api.storage import scoped_storage
from api.tools.http_common import load_allowed_hosts

from .contract import SkillContract, SkillContractError, parse_contract
from .extract import (
    ExtractionProgress,
    build_extraction_prose,
    extract_contract_events,
)
from .parser import (
    MAX_FILE_BYTES,
    MAX_SKILL_FILES,
    SkillParseError,
    parse_skill_bundle,
)
from .session_store import get_skill_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills", tags=["skills"])

# Each captured part is spliced into an api.github.com URL that carries
# GITHUB_TOKEN, so each is held to GitHub's own character set: a looser class
# let "..", "?" or "#" steer the authenticated request to another API path.
_GITHUB_TREE_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9-]{1,39})/(?P<repo>[A-Za-z0-9._-]{1,100})"
    r"(?:/tree/(?P<branch>[A-Za-z0-9._-]{1,255})(?:/(?P<subpath>[^?#]*))?)?/?$"
)
_DOT_SEGMENTS = {".", ".."}


class SkillFilePayload(BaseModel):
    """A single uploaded skill file (content base64-encoded)."""

    path: str
    content_base64: str


class InstallSkillRequest(BaseModel):
    """Request body for installing a skill.

    Provide either an uploaded ``files`` bundle (from a directory drop / zip)
    or a ``github_url`` pointing at a skill directory to fetch server-side.
    """

    files: Optional[list[SkillFilePayload]] = None
    github_url: Optional[str] = None


class SkillResponse(BaseModel):
    """Response describing a single installed skill."""

    id: str
    name: str
    description: str
    allowed_tools: list[str]
    resources: list[str]
    source: str
    removable: bool
    # Present when the skill declared a contract, or when one was proposed for
    # review. A proposal is unconfirmed and yields no tools until confirmed.
    contract: Optional[dict] = None
    tool_names: list[str] = []
    # Hosts the user is being asked to authorize, and whether the deployment's
    # allowlist actually permits them. Drives the install review step.
    contract_hosts: list[str] = []
    blocked_hosts: list[str] = []
    # Why no contract was proposed, when the skill documents an API but reading
    # it failed. Install still succeeds, so this is the only signal the user gets
    # that the skill has no synthesized tool and the model will fall back to
    # hand-building requests with http_request.
    contract_error: Optional[str] = None


class SkillsListResponse(BaseModel):
    """Response for listing skills."""

    skills: list[dict]
    count: int


class ConfirmContractRequest(BaseModel):
    """Request body for confirming (or clearing) a skill's contract.

    The body is the reviewed contract, possibly edited in the UI. Sending
    ``contract: null`` clears it, which removes the skill's synthesized tools.
    """

    contract: Optional[dict] = None


def _blocked_hosts(contract: Optional[SkillContract]) -> list[str]:
    """Contract hosts the deployment's HTTP allowlist would refuse.

    Surfaced at install time so the user learns the endpoint is unreachable now,
    rather than from a failed tool call later.
    """
    if contract is None:
        return []
    allowed = load_allowed_hosts()
    return [host for host in contract.hosts if host not in allowed]


def _skill_response(meta, contract_error: Optional[str] = None) -> SkillResponse:
    public = meta.to_public_dict()
    contract = meta.parsed_contract
    return SkillResponse(
        **public,
        contract_hosts=contract.hosts if contract else [],
        blocked_hosts=_blocked_hosts(contract),
        contract_error=contract_error,
    )


async def _fetch_github_skill(github_url: str) -> list[tuple[str, bytes]]:
    """Fetch a skill directory from a public GitHub URL.

    Supports ``https://github.com/{owner}/{repo}/tree/{branch}/{subpath}`` and
    plain repo URLs (default branch, repo root). Returns repo-relative paths;
    the parser locates SKILL.md and derives the skill root.
    """
    match = _GITHUB_TREE_RE.match(github_url.strip())
    if not match:
        raise SkillParseError(
            "Unsupported GitHub URL. Use a repo or /tree/<branch>/<path> URL."
        )
    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch")
    subpath = (match.group("subpath") or "").strip("/")
    if repo in _DOT_SEGMENTS or branch in _DOT_SEGMENTS:
        raise SkillParseError(
            "Unsupported GitHub URL. Use a repo or /tree/<branch>/<path> URL."
        )

    import os

    headers = {"User-Agent": "aevom-skills", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    owner_q, repo_q = quote(owner, safe=""), quote(repo, safe="")
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        if not branch:
            repo_resp = await client.get(
                f"https://api.github.com/repos/{owner_q}/{repo_q}"
            )
            repo_resp.raise_for_status()
            branch = repo_resp.json().get("default_branch", "main")
        branch_q = quote(branch, safe="")

        tree_resp = await client.get(
            f"https://api.github.com/repos/{owner_q}/{repo_q}/git/trees/{branch_q}",
            params={"recursive": "1"},
        )
        tree_resp.raise_for_status()
        tree = tree_resp.json().get("tree", [])

        prefix = f"{subpath}/" if subpath else ""
        blob_paths = [
            entry["path"]
            for entry in tree
            if entry.get("type") == "blob"
            and (not prefix or entry["path"].startswith(prefix))
        ]
        if not blob_paths:
            raise SkillParseError(
                f"No files found under '{subpath or repo}' on branch '{branch}'."
            )
        if len(blob_paths) > MAX_SKILL_FILES:
            blob_paths = blob_paths[:MAX_SKILL_FILES]

        files: list[tuple[str, bytes]] = []
        for path in blob_paths:
            raw_resp = await client.get(
                f"https://raw.githubusercontent.com/{owner_q}/{repo_q}/{branch_q}/"
                f"{quote(path, safe='/')}"
            )
            if raw_resp.status_code != 200:
                continue
            content = raw_resp.content
            if len(content) > MAX_FILE_BYTES:
                continue
            files.append((path, content))
    return files


def _decode_files(payloads: list[SkillFilePayload]) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for payload in payloads:
        try:
            content = base64.b64decode(payload.content_base64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise SkillParseError(
                f"Invalid base64 content for '{payload.path}'."
            ) from exc
        files.append((payload.path, content))
    return files


def _extraction_sources(resource_count: int) -> str:
    """Name the documentation the extractor is about to read."""
    if resource_count <= 0:
        return "SKILL.md"
    plural = "s" if resource_count != 1 else ""
    return f"SKILL.md and {resource_count} reference file{plural}"


async def _install_events(
    request: InstallSkillRequest, session_id: str
) -> AsyncIterator[dict[str, Any]]:
    """Run an install, yielding a progress event before each slow step.

    Deriving a contract sends the skill's documentation through an LLM and can
    take the better part of a minute, so a client that only sees the final
    response has to invent something to show meanwhile.

    The last event is either ``{"phase": "installed", "skill": SkillResponse}``
    or ``{"phase": "error", "status": int, "detail": str}``. Both install routes
    consume this, so the plain and streaming versions cannot drift apart.
    """
    try:
        if request.files:
            count = len(request.files)
            plural = "s" if count != 1 else ""
            yield {
                "phase": "reading_files",
                "message": f"Reading {count} file{plural}...",
            }
            files = _decode_files(request.files)
        elif request.github_url:
            yield {
                "phase": "fetching",
                "message": "Fetching the skill from GitHub...",
            }
            files = await _fetch_github_skill(request.github_url)
        else:
            raise SkillParseError("Provide either 'files' or 'github_url'.")

        parsed = parse_skill_bundle(files)
    except SkillParseError as exc:
        yield {"phase": "error", "status": 400, "detail": str(exc)}
        return
    except httpx.HTTPError as exc:
        yield {
            "phase": "error",
            "status": 502,
            "detail": f"Failed to fetch skill: {exc}",
        }
        return

    # A declared contract is authored deliberately and already confirmed. Only
    # skills without one get an LLM-proposed contract, which stays unconfirmed
    # until the user reviews it (confirming also authorizes its host).
    contract = parsed.contract
    contract_error: Optional[str] = None
    if contract is None:
        # Bundled reference files usually hold the request/response schema the
        # body only gestures at, so the extractor reads them too.
        prose = build_extraction_prose(parsed.body, parsed.resources)
        yield {
            "phase": "extracting",
            "message": (
                f"Reading {_extraction_sources(len(parsed.resources or []))} to "
                "work out the API contract..."
            ),
        }
        async for event in extract_contract_events(
            parsed.name, parsed.description, prose
        ):
            if isinstance(event, ExtractionProgress):
                yield {"phase": "extracting", "message": event.message}
                continue
            contract = event.contract
            contract_error = event.error

    yield {"phase": "storing", "message": "Saving the skill..."}
    session_factory = get_session_factory()
    try:
        async with session_factory() as db_session:
            storage = scoped_storage(db_session, session_id)
            store = get_skill_session_store()
            meta = await store.install_skill(session_id, storage, parsed, contract)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to install skill")
        yield {"phase": "error", "status": 500, "detail": f"Internal error: {exc}"}
        return

    yield {
        "phase": "installed",
        "skill": _skill_response(meta, contract_error=contract_error),
    }


@router.post("/", response_model=SkillResponse)
async def install_skill(
    request: InstallSkillRequest,
    session_id: str = Depends(require_owned_session),
):
    """Install a skill for a session from an uploaded bundle or GitHub URL."""

    terminal: Optional[dict[str, Any]] = None
    async for event in _install_events(request, session_id):
        if event["phase"] in ("installed", "error"):
            terminal = event

    if terminal is None:
        raise HTTPException(status_code=500, detail="Install produced no result.")
    if terminal["phase"] == "error":
        raise HTTPException(
            status_code=terminal.get("status", 500), detail=terminal["detail"]
        )
    return terminal["skill"]


@router.post("/stream")
async def install_skill_stream(
    request: InstallSkillRequest,
    session_id: str = Depends(require_owned_session),
):
    """Install a skill, emitting an SSE event as each phase begins.

    Same work and same result as ``POST /``, which remains for callers that do
    not want a stream.
    """

    async def events() -> AsyncIterator[str]:
        async for event in _install_events(request, session_id):
            payload = dict(event)
            skill = payload.get("skill")
            if skill is not None:
                payload["skill"] = skill.model_dump()
            yield json.dumps(payload)

    return EventSourceResponse(events(), media_type="text/event-stream")


@router.get("/", response_model=SkillsListResponse)
async def list_skills(
    session_id: Optional[str] = Depends(optional_owned_session),
):
    """List all skills for a session (curated builtin + session-installed)."""
    store = get_skill_session_store()
    catalog = await store.get_catalog(session_id)
    skills = [meta.to_public_dict() for meta in catalog]
    return SkillsListResponse(skills=skills, count=len(skills))


@router.put("/{skill_id}/contract", response_model=SkillResponse)
async def confirm_skill_contract(
    skill_id: str,
    request: ConfirmContractRequest,
    session_id: str = Depends(require_owned_session),
):
    """Save a reviewed contract for a skill, enabling its synthesized tools.

    Confirming is the consent gate: it authorizes the skill's tools to send
    requests (and the configured API key) to the contract's hosts, so it is an
    explicit user action rather than a side effect of installing.
    """

    if request.contract is None:
        # Clearing is allowed: it withdraws the tools and returns the skill to
        # prose-plus-http_request behavior.
        contract = SkillContract(source="none", confirmed=False)
    else:
        try:
            contract = parse_contract(request.contract)
        except SkillContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not contract.endpoints:
            raise HTTPException(
                status_code=400, detail="Contract must define at least one endpoint."
            )
        contract = contract.model_copy(update={"confirmed": True})

    store = get_skill_session_store()
    meta = await store.set_contract(session_id, skill_id, contract)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Skill {skill_id} not found for this session. Builtin skills declare "
                "their contract in SKILL.md and cannot be edited at runtime."
            ),
        )
    return _skill_response(meta)


@router.delete("/{skill_id}")
async def remove_skill(
    skill_id: str,
    session_id: str = Depends(require_owned_session),
):
    """Remove a session-installed skill (curated builtin skills cannot be removed)."""
    session_factory = get_session_factory()
    async with session_factory() as db_session:
        storage = scoped_storage(db_session, session_id)
        store = get_skill_session_store()
        removed = await store.remove_skill(session_id, storage, skill_id)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Skill {skill_id} not found (or is a non-removable builtin skill)",
        )
    return {"status": "removed", "skill_id": skill_id}
