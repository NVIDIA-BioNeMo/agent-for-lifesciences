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

"""Tests for the /api/skills route handlers.

The handlers are exercised directly (as async functions) with the database
session factory and storage service monkeypatched to an in-memory backend, so
no PostgreSQL/Redis services are required.
"""

import base64

import pytest
from fastapi import HTTPException

from api.skills import routes
from api.skills.contract import parse_contract
from api.skills.extract import ExtractionOutcome, ExtractionProgress
from api.skills.registry import get_skill_registry
from api.skills.session_store import SkillSessionStore
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

VALID_SKILL_MD = b"""---
name: Route Skill
description: Installed via the route.
allowed_tools: [search_pubchem]
---
# Body
Steps.
"""


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


class _DummyCM:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def wired(monkeypatch):
    """Wire routes to an in-memory storage + fake redis, isolated per test."""
    redis = FakeRedis()
    repo = InMemoryDataArtifactRepository()

    async def _get_redis(self):
        return redis

    monkeypatch.setattr(SkillSessionStore, "_get_redis", _get_redis)
    monkeypatch.setattr(routes, "get_session_factory", lambda: (lambda: _DummyCM()))
    storage = DataStorageService(repository=repo)
    monkeypatch.setattr(
        routes,
        "scoped_storage",
        lambda _db, session_id: SessionScopedStorage(storage, session_id),
    )
    # Fresh session-store singleton per test.
    monkeypatch.setattr("api.skills.session_store._skill_session_store", None)
    get_skill_registry()._skills.clear()
    yield
    get_skill_registry()._skills.clear()


def _b64_files():
    return [
        routes.SkillFilePayload(
            path="skill/SKILL.md",
            content_base64=base64.b64encode(VALID_SKILL_MD).decode(),
        ),
        routes.SkillFilePayload(
            path="skill/reference.md",
            content_base64=base64.b64encode(b"reference").decode(),
        ),
    ]


class TestInstallRoute:
    @pytest.mark.asyncio
    async def test_install_success(self, wired):
        req = routes.InstallSkillRequest(files=_b64_files())
        resp = await routes.install_skill(req, session_id="sess_1")
        assert resp.name == "Route Skill"
        assert resp.source == "session"
        assert resp.removable is True
        assert resp.resources == ["reference.md"]


    @pytest.mark.asyncio
    async def test_install_requires_payload(self, wired):
        req = routes.InstallSkillRequest()
        with pytest.raises(HTTPException) as exc:
            await routes.install_skill(req, session_id="sess_1")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_install_rejects_bad_bundle(self, wired):
        req = routes.InstallSkillRequest(
            files=[
                routes.SkillFilePayload(
                    path="skill/reference.md",
                    content_base64=base64.b64encode(b"no skill md").decode(),
                )
            ]
        )
        with pytest.raises(HTTPException) as exc:
            await routes.install_skill(req, session_id="sess_1")
        assert exc.value.status_code == 400


class TestListAndDeleteRoutes:
    @pytest.mark.asyncio
    async def test_list_includes_installed(self, wired):
        await routes.install_skill(
            routes.InstallSkillRequest(files=_b64_files()), session_id="sess_1"
        )
        listing = await routes.list_skills(session_id="sess_1")
        assert listing.count == 1
        assert listing.skills[0]["name"] == "Route Skill"

    @pytest.mark.asyncio
    async def test_delete_installed(self, wired):
        resp = await routes.install_skill(
            routes.InstallSkillRequest(files=_b64_files()), session_id="sess_1"
        )
        result = await routes.remove_skill(resp.id, session_id="sess_1")
        assert result["status"] == "removed"

        listing = await routes.list_skills(session_id="sess_1")
        assert listing.count == 0

    @pytest.mark.asyncio
    async def test_delete_unknown_404(self, wired):
        with pytest.raises(HTTPException) as exc:
            await routes.remove_skill("skill_missing", session_id="sess_1")
        assert exc.value.status_code == 404


DECLARED_SKILL_MD = b"""---
name: Declared Skill
description: Ships its own contract.
endpoints:
  - tool_suffix: predict
    summary: Predict a structure.
    url: https://api.test/predict
    inputs:
      - name: sequence
        data_type: sequence
    outputs:
      - name: structure
        json_path: $.structure
        data_type: mmcif
---
# Body
POST the sequence to https://api.test/predict.
"""

PROSE_SKILL_MD = b"""---
name: Prose Skill
description: Documents an API in prose only.
---
# Body
POST a protein sequence to https://api.test/predict and read $.structure.
"""

PROPOSAL = {
    "endpoints": [
        {
            "tool_suffix": "predict",
            "summary": "Predict a structure.",
            "url": "https://api.test/predict",
            "inputs": [{"name": "sequence", "data_type": "sequence"}],
            "outputs": [
                {"name": "structure", "json_path": "$.structure", "data_type": "mmcif"}
            ],
        }
    ]
}


def _files(skill_md: bytes):
    return [
        routes.SkillFilePayload(
            path="skill/SKILL.md",
            content_base64=base64.b64encode(skill_md).decode(),
        )
    ]


@pytest.fixture
def allowlisted(monkeypatch):
    monkeypatch.setenv("HTTP_TOOL_ALLOWED_HOSTS", "api.test")


@pytest.fixture
def stub_extraction(monkeypatch):
    """Stand in for the extraction model call at install time.

    Returns the list of prose strings extraction was called with, so a test can
    assert that bundled reference files reached the extractor.
    """

    def _install(contract, error=None, progress=()):
        seen: list[str] = []

        async def _extract(name, description, prose, **kwargs):
            seen.append(prose)
            for message in progress:
                yield ExtractionProgress(message=message)
            yield ExtractionOutcome(contract=contract, error=error)

        monkeypatch.setattr(routes, "extract_contract_events", _extract)
        return seen

    return _install


class TestDeclaredContract:
    @pytest.mark.asyncio
    async def test_frontmatter_contract_is_live_immediately(self, wired, allowlisted):
        # The skill author wrote it deliberately, so there is nothing to review.
        resp = await routes.install_skill(
            routes.InstallSkillRequest(files=_files(DECLARED_SKILL_MD)),
            session_id="sess_1",
        )
        assert resp.contract["source"] == "declared"
        assert resp.contract["confirmed"] is True
        assert resp.tool_names == [f"{resp.id}_predict"]
        assert resp.contract_hosts == ["api.test"]
        assert resp.blocked_hosts == []

    @pytest.mark.asyncio
    async def test_extraction_is_skipped_when_declared(self, wired, monkeypatch):
        called = []

        async def _extract(*args, **kwargs):
            called.append(args)
            yield ExtractionOutcome()

        monkeypatch.setattr(routes, "extract_contract_events", _extract)
        await routes.install_skill(
            routes.InstallSkillRequest(files=_files(DECLARED_SKILL_MD)),
            session_id="sess_1",
        )
        assert called == []

    @pytest.mark.asyncio
    async def test_invalid_frontmatter_contract_is_a_400(self, wired):
        bad = DECLARED_SKILL_MD.replace(b"https://api.test/predict", b"not-a-url")
        with pytest.raises(HTTPException) as exc:
            await routes.install_skill(
                routes.InstallSkillRequest(files=_files(bad)), session_id="sess_1"
            )
        assert exc.value.status_code == 400


class TestProposedContract:
    @pytest.mark.asyncio
    async def test_proposal_is_returned_unconfirmed_with_no_tools(
        self, wired, allowlisted, stub_extraction
    ):
        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        resp = await routes.install_skill(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD)),
            session_id="sess_1",
        )
        assert resp.contract["source"] == "llm"
        assert resp.contract["confirmed"] is False
        # Nothing is callable until the user approves the host.
        assert resp.tool_names == []
        assert resp.contract_hosts == ["api.test"]

    @pytest.mark.asyncio
    async def test_a_host_outside_the_allowlist_is_flagged(
        self, wired, monkeypatch, stub_extraction
    ):
        monkeypatch.setenv("HTTP_TOOL_ALLOWED_HOSTS", "health.api.nvidia.com")
        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        resp = await routes.install_skill(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD)),
            session_id="sess_1",
        )
        assert resp.blocked_hosts == ["api.test"]

    @pytest.mark.asyncio
    async def test_failed_extraction_still_installs_the_skill(
        self, wired, stub_extraction
    ):
        stub_extraction(None)
        resp = await routes.install_skill(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD)),
            session_id="sess_1",
        )
        assert resp.name == "Prose Skill"
        assert resp.contract is None
        assert resp.tool_names == []

    @pytest.mark.asyncio
    async def test_failed_extraction_reports_why(self, wired, stub_extraction):
        # Without this the install looks identical to a successful one, and the
        # user has no way to know the skill gained no tool.
        stub_extraction(None, error="the model returned an empty response")
        resp = await routes.install_skill(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD)),
            session_id="sess_1",
        )
        assert resp.contract is None
        assert resp.contract_error == "the model returned an empty response"

    @pytest.mark.asyncio
    async def test_reference_files_reach_the_extractor(self, wired, stub_extraction):
        # The request/response schema usually lives in a reference file rather
        # than in the body's prose.
        seen = stub_extraction(None)
        files = _files(PROSE_SKILL_MD) + [
            routes.SkillFilePayload(
                path="skill/references/api.md",
                content_base64=base64.b64encode(b"| protein | string |").decode(),
            )
        ]
        await routes.install_skill(
            routes.InstallSkillRequest(files=files), session_id="sess_1"
        )
        assert len(seen) == 1
        assert "references/api.md" in seen[0]
        assert "| protein | string |" in seen[0]


class TestInstallProgressStream:
    """Phases are reported as they start, so the client narrates real work.

    The client used to show messages on a 3s timer, which drifted from whatever
    the server was actually doing and had no way to know when extraction began.
    """

    async def _phases(self, request, session_id="sess_1"):
        """Collect the progress events, and the terminal event separately."""
        progress: list[dict] = []
        terminal: dict | None = None
        async for event in routes._install_events(request, session_id):
            if event["phase"] in ("installed", "error"):
                terminal = event
            else:
                progress.append(event)
        return progress, terminal

    @pytest.mark.asyncio
    async def test_extraction_is_announced_before_it_runs(
        self, wired, allowlisted, stub_extraction
    ):
        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        progress, terminal = await self._phases(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD))
        )
        phases = [event["phase"] for event in progress]
        # The long step must be announced before storing, not after.
        assert phases == ["reading_files", "extracting", "storing"]
        assert terminal["phase"] == "installed"

    @pytest.mark.asyncio
    async def test_every_progress_event_carries_something_to_display(
        self, wired, allowlisted, stub_extraction
    ):
        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        progress, _ = await self._phases(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD))
        )
        assert all(event.get("message") for event in progress)

    @pytest.mark.asyncio
    async def test_the_extraction_message_names_the_files_being_read(
        self, wired, allowlisted, stub_extraction
    ):
        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        files = _files(PROSE_SKILL_MD) + [
            routes.SkillFilePayload(
                path="skill/references/api.md",
                content_base64=base64.b64encode(b"# API").decode(),
            )
        ]
        progress, _ = await self._phases(routes.InstallSkillRequest(files=files))
        extracting = next(e for e in progress if e["phase"] == "extracting")
        assert "SKILL.md and 1 reference file" in extracting["message"]

    @pytest.mark.asyncio
    async def test_extraction_progress_is_forwarded(
        self, wired, allowlisted, stub_extraction
    ):
        """The model call is the minute-long part, so its own updates must pass
        through. Announcing extraction and then going quiet is what made the
        install look hung."""
        stub_extraction(
            parse_contract(PROPOSAL, source="llm"),
            progress=["Working through the documentation...", "Writing it out..."],
        )
        progress, terminal = await self._phases(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD))
        )
        messages = [event["message"] for event in progress]

        assert "Working through the documentation..." in messages
        assert "Writing it out..." in messages
        # Still bracketed by the phases either side of the model call.
        assert messages[0].startswith("Reading 1 file")
        assert messages[-1] == "Saving the skill..."
        assert terminal["phase"] == "installed"

    @pytest.mark.asyncio
    async def test_a_declared_contract_skips_the_extraction_phase(
        self, wired, allowlisted
    ):
        progress, terminal = await self._phases(
            routes.InstallSkillRequest(files=_files(DECLARED_SKILL_MD))
        )
        assert "extracting" not in [event["phase"] for event in progress]
        assert terminal["phase"] == "installed"

    @pytest.mark.asyncio
    async def test_a_bad_bundle_ends_the_stream_with_an_error(self, wired):
        request = routes.InstallSkillRequest(
            files=[
                routes.SkillFilePayload(
                    path="skill/reference.md",
                    content_base64=base64.b64encode(b"no skill md").decode(),
                )
            ]
        )
        _, terminal = await self._phases(request)
        assert terminal["phase"] == "error"
        assert terminal["status"] == 400
        assert terminal["detail"]


    @pytest.mark.asyncio
    async def test_the_stream_serializes_the_installed_skill(
        self, wired, allowlisted, stub_extraction
    ):
        """The terminal event carries a model, which must survive JSON encoding."""
        import json

        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        response = await routes.install_skill_stream(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD)),
            session_id="sess_1",
        )
        payloads = [json.loads(chunk) async for chunk in response.body_iterator]
        installed = next(p for p in payloads if p["phase"] == "installed")
        assert installed["skill"]["name"] == "Prose Skill"
        assert installed["skill"]["contract"]["confirmed"] is False


class TestConfirmContractRoute:
    async def _install_proposal(self, stub_extraction):
        stub_extraction(parse_contract(PROPOSAL, source="llm"))
        return await routes.install_skill(
            routes.InstallSkillRequest(files=_files(PROSE_SKILL_MD)),
            session_id="sess_1",
        )

    @pytest.mark.asyncio
    async def test_confirming_creates_the_tools(
        self, wired, allowlisted, stub_extraction
    ):
        installed = await self._install_proposal(stub_extraction)
        resp = await routes.confirm_skill_contract(
            installed.id,
            routes.ConfirmContractRequest(contract=PROPOSAL),
            session_id="sess_1",
        )
        assert resp.contract["confirmed"] is True
        assert resp.tool_names == [f"{installed.id}_predict"]

    @pytest.mark.asyncio
    async def test_user_edits_are_persisted(self, wired, allowlisted, stub_extraction):
        installed = await self._install_proposal(stub_extraction)
        edited = {
            "endpoints": [
                dict(PROPOSAL["endpoints"][0], tool_suffix="fold", summary="Fold it.")
            ]
        }
        resp = await routes.confirm_skill_contract(
            installed.id,
            routes.ConfirmContractRequest(contract=edited),
            session_id="sess_1",
        )
        assert resp.tool_names == [f"{installed.id}_fold"]

    @pytest.mark.asyncio
    async def test_clearing_withdraws_the_tools(
        self, wired, allowlisted, stub_extraction
    ):
        installed = await self._install_proposal(stub_extraction)
        await routes.confirm_skill_contract(
            installed.id,
            routes.ConfirmContractRequest(contract=PROPOSAL),
            session_id="sess_1",
        )
        resp = await routes.confirm_skill_contract(
            installed.id,
            routes.ConfirmContractRequest(contract=None),
            session_id="sess_1",
        )
        assert resp.tool_names == []
        assert resp.contract_hosts == []

    @pytest.mark.asyncio
    async def test_invalid_edit_is_a_400(self, wired, allowlisted, stub_extraction):
        installed = await self._install_proposal(stub_extraction)
        broken = {"endpoints": [dict(PROPOSAL["endpoints"][0], url="not-a-url")]}
        with pytest.raises(HTTPException) as exc:
            await routes.confirm_skill_contract(
                installed.id,
                routes.ConfirmContractRequest(contract=broken),
                session_id="sess_1",
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_endpoint_list_is_a_400(
        self, wired, allowlisted, stub_extraction
    ):
        installed = await self._install_proposal(stub_extraction)
        with pytest.raises(HTTPException) as exc:
            await routes.confirm_skill_contract(
                installed.id,
                routes.ConfirmContractRequest(contract={"endpoints": []}),
                session_id="sess_1",
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_skill_is_a_404(self, wired):
        with pytest.raises(HTTPException) as exc:
            await routes.confirm_skill_contract(
                "skill_missing",
                routes.ConfirmContractRequest(contract=PROPOSAL),
                session_id="sess_1",
            )
        assert exc.value.status_code == 404

