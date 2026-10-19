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

"""Session ownership: a session id names a session, the owner token unlocks it.

Before this, holding a session UUID (or an artifact reference) was the whole of
access control. These tests pin the replacement: ``/api/session/create`` binds
each session to an owner token, and every session-scoped route refuses a caller
that does not hold it -- 401 with no token, 404 with someone else's, so a
foreign session cannot be told apart from a missing one.

``api.main`` cannot be imported under pytest, so its routes are checked by
reading their source, as test_storage_access_flag.py does.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import pytest
from fastapi import Depends, FastAPI, Request, Response
from fastapi.testclient import TestClient

from api.artifacts.routes import router as artifacts_router
from api.auth import deps
from api.auth.ownership import KEY_PREFIX, SessionAlreadyBoundError, SessionOwnership
from api.auth.principal import (
    OWNER_COOKIE,
    hash_token,
    is_well_formed,
    mint_owner_token,
    owner_token_from,
)
from api.auth.session import open_owned_session
from api.db.config import get_async_session
from api.mcp.routes import router as mcp_router
from api.nim.routes import router as nim_router
from api.secret_store.routes import router as secrets_router
from api.skills.routes import router as skills_router


class FakeRedis:
    """The four commands SessionOwnership uses, with TTLs recorded not enforced."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.ttls[key] = ex
        return True

    async def get(self, key):
        return self.store.get(key)

    async def expire(self, key, seconds):
        self.ttls[key] = seconds
        return key in self.store


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def ownership(redis):
    return SessionOwnership(redis)


def _request(headers: Optional[dict] = None, cookies: Optional[dict] = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookies:
        cookie = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw.append((b"cookie", cookie.encode()))
    return Request({"type": "http", "headers": raw})


class TestOwnerToken:
    def test_a_minted_token_is_well_formed_and_unique(self):
        tokens = {mint_owner_token() for _ in range(50)}
        assert len(tokens) == 50
        assert all(is_well_formed(t) for t in tokens)

    def test_a_short_or_odd_token_is_not_an_identity(self):
        # A client-chosen value must not become a (guessable) owner.
        for token in ["", "a", "x" * 42, "!" * 43, "a b" * 20]:
            assert not is_well_formed(token)

    def test_the_bearer_header_is_read(self):
        token = mint_owner_token()
        assert owner_token_from(_request({"Authorization": f"Bearer {token}"})) == token

    def test_the_cookie_is_read(self):
        token = mint_owner_token()
        assert owner_token_from(_request(cookies={OWNER_COOKIE: token})) == token

    def test_bearer_wins_over_the_cookie(self):
        bearer, cookie = mint_owner_token(), mint_owner_token()
        request = _request({"Authorization": f"Bearer {bearer}"}, {OWNER_COOKIE: cookie})
        assert owner_token_from(request) == bearer

    def test_a_malformed_token_reads_as_none(self):
        assert owner_token_from(_request({"Authorization": "Bearer short"})) is None
        assert owner_token_from(_request(cookies={OWNER_COOKIE: "short"})) is None
        assert owner_token_from(_request()) is None


@pytest.mark.asyncio
class TestOwnershipStore:
    async def test_a_bound_session_is_owned_by_its_token(self, ownership):
        token = mint_owner_token()
        await ownership.bind("s1", token)
        assert await ownership.owns("s1", token)

    async def test_another_token_does_not_own_it(self, ownership):
        await ownership.bind("s1", mint_owner_token())
        assert not await ownership.owns("s1", mint_owner_token())

    async def test_an_unbound_session_is_owned_by_nobody(self, ownership):
        assert not await ownership.owns("never-created", mint_owner_token())

    async def test_missing_inputs_own_nothing(self, ownership):
        token = mint_owner_token()
        await ownership.bind("s1", token)
        assert not await ownership.owns("s1", None)
        assert not await ownership.owns(None, token)

    async def test_a_session_cannot_be_claimed_twice(self, ownership):
        await ownership.bind("s1", mint_owner_token())
        with pytest.raises(SessionAlreadyBoundError):
            await ownership.bind("s1", mint_owner_token())

    async def test_only_a_hash_of_the_token_is_stored(self, ownership, redis):
        token = mint_owner_token()
        await ownership.bind("s1", token)
        stored = redis.store[f"{KEY_PREFIX}s1"]
        assert stored == hash_token(token)
        assert token not in stored

    async def test_the_binding_expires_and_use_refreshes_it(
        self, ownership, redis, monkeypatch
    ):
        monkeypatch.setenv("AEVOM_SESSION_OWNER_TTL_S", "60")
        token = mint_owner_token()
        await ownership.bind("s1", token)
        key = f"{KEY_PREFIX}s1"
        assert redis.ttls[key] == 60
        redis.ttls[key] = 1
        await ownership.owns("s1", token)
        assert redis.ttls[key] == 60

    async def test_an_expired_binding_fails_closed(self, ownership, redis):
        token = mint_owner_token()
        await ownership.bind("s1", token)
        del redis.store[f"{KEY_PREFIX}s1"]
        assert not await ownership.owns("s1", token)


def _app(ownership: SessionOwnership) -> FastAPI:
    app = FastAPI()

    @app.post("/api/session/create")
    async def create(request: Request, response: Response):
        return await open_owned_session(request, response, ownership)

    @app.get("/query")
    async def query(session_id: str = Depends(deps.require_owned_session)):
        return {"session_id": session_id}

    @app.get("/path/{session_id}")
    async def path(session_id: str = Depends(deps.require_owned_path_session)):
        return {"session_id": session_id}

    @app.get("/list")
    async def listing(session_id: Optional[str] = Depends(deps.optional_owned_session)):
        return {"session_id": session_id}

    for router in (mcp_router, nim_router, secrets_router, skills_router, artifacts_router):
        app.include_router(router)

    async def no_db():
        yield None

    app.dependency_overrides[deps.get_session_ownership] = lambda: ownership
    app.dependency_overrides[get_async_session] = no_db
    return app


@pytest.fixture
def client(ownership):
    # https, because the owner cookie is Secure by default and a client (like a
    # browser) will not send it back over plain http.
    return TestClient(_app(ownership), base_url="https://testserver")


def _create(client: TestClient, token: Optional[str] = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = client.post("/api/session/create", headers=headers)
    assert response.status_code == 200
    return response.json()


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestSessionCreate:
    def test_returns_a_session_and_a_token(self, client):
        body = _create(client)
        assert body["session_id"]
        assert is_well_formed(body["owner_token"])

    def test_sets_an_http_only_lax_secure_cookie(self, client, monkeypatch):
        monkeypatch.delenv("AEVOM_COOKIE_SECURE", raising=False)
        response = client.post("/api/session/create")
        cookie = response.headers["set-cookie"]
        assert cookie.startswith(f"{OWNER_COOKIE}={response.json()['owner_token']}")
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
        assert "Secure" in cookie
        assert "Path=/" in cookie

    def test_secure_can_be_turned_off_for_plain_http(self, ownership, monkeypatch):
        monkeypatch.setenv("AEVOM_COOKIE_SECURE", "false")
        plain = TestClient(_app(ownership), base_url="http://testserver")
        response = plain.post("/api/session/create")
        assert "Secure" not in response.headers["set-cookie"]
        sid = response.json()["session_id"]
        assert plain.get(f"/query?session_id={sid}").status_code == 200

    def test_a_secure_cookie_is_not_sent_over_plain_http(self, ownership, monkeypatch):
        # Why compose turns Secure off: otherwise a local browser gets 401s.
        monkeypatch.delenv("AEVOM_COOKIE_SECURE", raising=False)
        plain = TestClient(_app(ownership), base_url="http://testserver")
        sid = plain.post("/api/session/create").json()["session_id"]
        assert plain.get(f"/query?session_id={sid}").status_code == 401

    def test_a_caller_with_a_token_keeps_it(self, client):
        first = _create(client)
        second = _create(client, first["owner_token"])
        assert second["owner_token"] == first["owner_token"]
        assert second["session_id"] != first["session_id"]

    def test_a_browser_cookie_is_reused(self, client):
        # TestClient keeps cookies between requests, as a browser does.
        first = client.post("/api/session/create").json()
        second = client.post("/api/session/create").json()
        assert second["owner_token"] == first["owner_token"]

    def test_a_malformed_token_gets_a_fresh_one(self, client):
        body = _create(client, "not-a-real-token")
        assert body["owner_token"] != "not-a-real-token"
        assert is_well_formed(body["owner_token"])


class TestOwnedSessionDependencies:
    def test_the_owner_gets_through(self, client):
        body = _create(client)
        token, sid = body["owner_token"], body["session_id"]
        assert client.get(f"/query?session_id={sid}", headers=_bearer(token)).status_code == 200
        assert client.get(f"/path/{sid}", headers=_bearer(token)).status_code == 200

    def test_the_cookie_alone_is_enough(self, client):
        sid = client.post("/api/session/create").json()["session_id"]
        assert client.get(f"/query?session_id={sid}").status_code == 200

    def test_no_token_is_401(self, ownership):
        sid = _create(TestClient(_app(ownership)))["session_id"]
        fresh = TestClient(_app(ownership))
        assert fresh.get(f"/query?session_id={sid}").status_code == 401
        assert fresh.get(f"/path/{sid}").status_code == 401

    def test_another_owner_is_404(self, client):
        victim = _create(client)["session_id"]
        attacker = mint_owner_token()
        response = client.get(f"/query?session_id={victim}", headers=_bearer(attacker))
        assert response.status_code == 404
        assert client.get(f"/path/{victim}", headers=_bearer(attacker)).status_code == 404

    def test_a_session_nobody_created_is_404(self, client):
        token = _create(client)["owner_token"]
        response = client.get("/query?session_id=made-up", headers=_bearer(token))
        assert response.status_code == 404

    def test_a_missing_session_is_400(self, client):
        token = _create(client)["owner_token"]
        assert client.get("/query", headers=_bearer(token)).status_code == 400

    def test_a_listing_without_a_session_is_allowed(self, client):
        response = TestClient(client.app).get("/list")
        assert response.status_code == 200
        assert response.json() == {"session_id": None}

    def test_a_listing_with_someone_elses_session_is_404(self, client):
        victim = _create(client)["session_id"]
        response = client.get(
            f"/list?session_id={victim}", headers=_bearer(mint_owner_token())
        )
        assert response.status_code == 404


# Every session-scoped route on the real routers. Each must refuse before its
# handler runs, which is also why no store or database is wired up here.
SCOPED_ROUTES = [
    ("POST", "/api/mcp/servers?session_id={sid}", {"name": "x", "url": "http://x"}),
    ("DELETE", "/api/mcp/servers/srv?session_id={sid}", None),
    ("POST", "/api/mcp/servers/srv/confirm?session_id={sid}", None),
    ("GET", "/api/mcp/servers?session_id={sid}", None),
    ("GET", "/api/mcp/tools?session_id={sid}", None),
    ("POST", "/api/nim/servers?session_id={sid}", {"name": "x", "url": "http://x"}),
    ("GET", "/api/nim/servers?session_id={sid}", None),
    ("DELETE", "/api/nim/servers/n?session_id={sid}", None),
    ("POST", "/api/nim/servers/n/confirm?session_id={sid}", None),
    ("POST", "/api/secrets/?session_id={sid}", {"name": "K", "value": "v"}),
    ("GET", "/api/secrets/?session_id={sid}", None),
    ("DELETE", "/api/secrets/sk_1?session_id={sid}", None),
    ("POST", "/api/skills/?session_id={sid}", {"files": []}),
    ("POST", "/api/skills/stream?session_id={sid}", {"files": []}),
    ("GET", "/api/skills/?session_id={sid}", None),
    ("PUT", "/api/skills/sk/contract?session_id={sid}", {"contract": {}}),
    ("DELETE", "/api/skills/sk?session_id={sid}", None),
    ("POST", "/api/session/{sid}/artifacts", {"artifacts": []}),
    ("GET", "/api/session/{sid}/submissions", None),
]


@pytest.mark.parametrize("method,url,body", SCOPED_ROUTES)
class TestRoutersRequireOwnership:
    def test_no_token_is_refused(self, ownership, method, url, body):
        sid = _create(TestClient(_app(ownership)))["session_id"]
        fresh = TestClient(_app(ownership))
        response = fresh.request(method, url.format(sid=sid), json=body)
        assert response.status_code == 401

    def test_another_owner_is_refused(self, client, method, url, body):
        sid = _create(client)["session_id"]
        response = client.request(
            method, url.format(sid=sid), json=body, headers=_bearer(mint_owner_token())
        )
        assert response.status_code == 404


# The mutations among them. The listings return nothing without a session.
REQUIRED_SESSION_ROUTES = [
    ("POST", "/api/mcp/servers", {"name": "x", "url": "http://x"}),
    ("DELETE", "/api/mcp/servers/srv", None),
    ("POST", "/api/mcp/servers/srv/confirm", None),
    ("POST", "/api/nim/servers", {"name": "x", "url": "http://x"}),
    ("DELETE", "/api/nim/servers/n", None),
    ("POST", "/api/nim/servers/n/confirm", None),
    ("POST", "/api/secrets/", {"name": "K", "value": "v"}),
    ("DELETE", "/api/secrets/sk_1", None),
    ("POST", "/api/skills/", {"files": []}),
    ("POST", "/api/skills/stream", {"files": []}),
    ("PUT", "/api/skills/sk/contract", {"contract": {}}),
    ("DELETE", "/api/skills/sk", None),
]


@pytest.mark.parametrize("method,url,body", REQUIRED_SESSION_ROUTES)
def test_a_mutation_without_a_session_is_400(client, method, url, body):
    token = _create(client)["owner_token"]
    response = client.request(method, url, json=body, headers=_bearer(token))
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# main.py, by source
# ---------------------------------------------------------------------------

OWNED_DEPS = {
    "require_owned_session",
    "require_owned_path_session",
    "optional_owned_session",
    "get_scoped_storage",
}


def _main_functions() -> dict[str, ast.AsyncFunctionDef]:
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }


def _route_path(function: ast.AST) -> Optional[str]:
    for decorator in getattr(function, "decorator_list", []):
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
        ):
            return decorator.args[0].value
    return None


def _depends_on(function: ast.AST) -> set[str]:
    names = set()
    for default in function.args.defaults + function.args.kw_defaults:
        if (
            isinstance(default, ast.Call)
            and isinstance(default.func, ast.Name)
            and default.func.id == "Depends"
            and default.args
            and isinstance(default.args[0], ast.Name)
        ):
            names.add(default.args[0].id)
    return names


def _calls(function: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(function)
    )


MAIN = _main_functions()
MAIN_ROUTES = {name: fn for name, fn in MAIN.items() if _route_path(fn)}


class TestMainRoutes:
    @pytest.mark.parametrize(
        "name",
        sorted(
            name
            for name, fn in MAIN_ROUTES.items()
            if any(arg.arg == "session_id" for arg in fn.args.args)
        ),
    )
    def test_any_route_taking_a_session_id_takes_an_owned_one(self, name):
        assert _depends_on(MAIN_ROUTES[name]) & OWNED_DEPS, name

    @pytest.mark.parametrize(
        "name",
        [
            "list_storage_artifacts",
            "get_artifact_content",
            "export_artifacts",
            "store_selection",
            "get_selections",
            "get_job_status",
            "get_conversation",
        ],
    )
    def test_each_scoped_route_is_gated(self, name):
        assert _depends_on(MAIN_ROUTES[name]) & OWNED_DEPS, name

    def test_chat_checks_the_body_session(self):
        assert _calls(MAIN_ROUTES["chat"], "assert_owned_session")

    def test_chat_no_longer_falls_back_to_a_shared_session(self):
        source = ast.unparse(MAIN["stream_chat_with_langgraph"])
        assert "'default'" not in source

    def test_images_check_the_artifacts_owner(self):
        image = MAIN_ROUTES["get_image"]
        assert _calls(image, "artifact_owner")
        assert _calls(image, "assert_owned_session")

    def test_jobs_are_filtered_by_session(self):
        assert "session_id=session_id" in ast.unparse(MAIN_ROUTES["get_job_status"])

    def test_session_create_binds_an_owner(self):
        assert _calls(MAIN_ROUTES["create_session"], "open_owned_session")
