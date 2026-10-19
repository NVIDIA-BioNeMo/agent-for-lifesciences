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

"""Tests for SecretSessionStore (Redis-backed session secrets)."""

import json

import pytest

from api.secret_store.session_store import SecretSessionStore


class FakeRedis:
    """Minimal async Redis stand-in supporting get/set (decode_responses)."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.last_ex: int | None = None

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        self.last_ex = ex


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()

    async def _get_redis(self):
        return redis

    monkeypatch.setattr(SecretSessionStore, "_get_redis", _get_redis)
    return redis


class TestAddAndList:
    @pytest.mark.asyncio
    async def test_add_persists_and_sets_ttl(self, fake_redis):
        store = SecretSessionStore()
        meta = await store.add_secret("sess_1", "NVIDIA_API_KEY", "secret-value")

        assert meta.id.startswith("sk_")
        assert meta.name == "NVIDIA_API_KEY"
        # TTL applied on write.
        assert fake_redis.last_ex == int(store._session_ttl)
        # Value is persisted in Redis (source of truth).
        raw = json.loads(fake_redis.store["secret:sess_1"])
        assert raw[meta.id]["value"] == "secret-value"

    @pytest.mark.asyncio
    async def test_public_dict_omits_value(self, fake_redis):
        store = SecretSessionStore()
        meta = await store.add_secret("sess_1", "KEY", "v")
        public = meta.to_public_dict()
        assert "value" not in public
        assert set(public) == {"id", "name", "created_at"}

    @pytest.mark.asyncio
    async def test_list_returns_all(self, fake_redis):
        store = SecretSessionStore()
        await store.add_secret("sess_1", "A", "1")
        await store.add_secret("sess_1", "B", "2")
        names = {s.name for s in await store.list_secrets("sess_1")}
        assert names == {"A", "B"}

    @pytest.mark.asyncio
    async def test_add_same_name_replaces(self, fake_redis):
        store = SecretSessionStore()
        await store.add_secret("sess_1", "KEY", "old")
        await store.add_secret("sess_1", "KEY", "new")
        secrets = await store.list_secrets("sess_1")
        assert len(secrets) == 1
        assert await store.get_value("sess_1", "KEY") == "new"


class TestGetValueAndRemove:
    @pytest.mark.asyncio
    async def test_get_value_by_name(self, fake_redis):
        store = SecretSessionStore()
        await store.add_secret("sess_1", "KEY", "the-value")
        assert await store.get_value("sess_1", "KEY") == "the-value"

    @pytest.mark.asyncio
    async def test_get_value_missing(self, fake_redis):
        store = SecretSessionStore()
        assert await store.get_value("sess_1", "NOPE") is None
        assert await store.get_value("", "KEY") is None

    @pytest.mark.asyncio
    async def test_isolation_between_sessions(self, fake_redis):
        store = SecretSessionStore()
        await store.add_secret("sess_1", "KEY", "one")
        assert await store.get_value("sess_2", "KEY") is None

    @pytest.mark.asyncio
    async def test_remove(self, fake_redis):
        store = SecretSessionStore()
        meta = await store.add_secret("sess_1", "KEY", "v")
        assert await store.remove_secret("sess_1", meta.id) is True
        assert await store.list_secrets("sess_1") == []
        assert await store.remove_secret("sess_1", meta.id) is False
