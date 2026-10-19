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

"""Tests for the /api/secrets route handlers.

Handlers are exercised directly with a fake Redis backing the session store, so
no Redis service is required. The key invariant checked here is that a secret's
value is never returned by any endpoint.
"""

import pytest
from fastapi import HTTPException

from api.secret_store import routes
from api.secret_store.session_store import SecretSessionStore


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture
def wired(monkeypatch):
    redis = FakeRedis()

    async def _get_redis(self):
        return redis

    monkeypatch.setattr(SecretSessionStore, "_get_redis", _get_redis)
    # Fresh singleton per test.
    monkeypatch.setattr("api.secret_store.session_store._secret_session_store", None)
    yield redis


class TestAddRoute:
    @pytest.mark.asyncio
    async def test_add_success_returns_public_only(self, wired):
        req = routes.AddSecretRequest(name="NVIDIA_API_KEY", value="super-secret")
        resp = await routes.add_secret(req, session_id="sess_1")
        assert resp.name == "NVIDIA_API_KEY"
        assert resp.id.startswith("sk_")
        # Serialized response must not contain the value anywhere.
        assert "super-secret" not in resp.model_dump_json()


    @pytest.mark.asyncio
    async def test_add_requires_name_and_value(self, wired):
        with pytest.raises(HTTPException):
            await routes.add_secret(
                routes.AddSecretRequest(name="  ", value="b"), session_id="s"
            )
        with pytest.raises(HTTPException):
            await routes.add_secret(
                routes.AddSecretRequest(name="a", value="  "), session_id="s"
            )


class TestListAndDeleteRoutes:
    @pytest.mark.asyncio
    async def test_list_masks_values(self, wired):
        await routes.add_secret(
            routes.AddSecretRequest(name="KEY", value="do-not-leak"),
            session_id="sess_1",
        )
        listing = await routes.list_secrets(session_id="sess_1")
        assert listing.count == 1
        assert listing.secrets[0].name == "KEY"
        assert "do-not-leak" not in listing.model_dump_json()

    @pytest.mark.asyncio
    async def test_list_empty_without_session(self, wired):
        listing = await routes.list_secrets(session_id=None)
        assert listing.count == 0

    @pytest.mark.asyncio
    async def test_delete(self, wired):
        resp = await routes.add_secret(
            routes.AddSecretRequest(name="KEY", value="v"), session_id="sess_1"
        )
        result = await routes.remove_secret(resp.id, session_id="sess_1")
        assert result["status"] == "removed"
        listing = await routes.list_secrets(session_id="sess_1")
        assert listing.count == 0

    @pytest.mark.asyncio
    async def test_delete_unknown_404(self, wired):
        with pytest.raises(HTTPException) as exc:
            await routes.remove_secret("sk_missing", session_id="sess_1")
        assert exc.value.status_code == 404
