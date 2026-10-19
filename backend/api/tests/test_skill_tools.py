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

"""Tests for the load_skill / load_skill_resource disclosure tools."""

import pytest

from api.skills.registry import SkillDefinition, get_skill_registry
from api.skills.session_store import SkillSessionStore
from api.skills.tools import LoadSkillResourceTool, LoadSkillTool
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService
from api.tools.base import ContentType


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()

    async def _get_redis(self):
        return redis

    monkeypatch.setattr(SkillSessionStore, "_get_redis", _get_redis)
    return redis


@pytest.fixture(autouse=True)
def clear_registry():
    get_skill_registry()._skills.clear()
    yield
    get_skill_registry()._skills.clear()


class TestLoadSkillTool:
    @pytest.mark.asyncio
    async def test_loads_builtin_skill(self):
        get_skill_registry()._skills["mine"] = SkillDefinition(
            id="mine",
            name="Mine",
            description="d",
            body="curated instructions",
            allowed_tools=["search_pubchem"],
            resources={"reference.md": "ref"},
        )
        tool = LoadSkillTool()
        tool._session_id = None
        tool._storage = None

        result = await tool.execute({"skill_id": "mine"})
        assert result.content_type == ContentType.TEXT
        assert "curated instructions" in result.data
        assert "search_pubchem" in result.data  # allowed_tools footer
        assert "reference.md" in result.data  # resource listing footer

    @pytest.mark.asyncio
    async def test_missing_skill_id_errors(self):
        tool = LoadSkillTool()
        result = await tool.execute({})
        assert result.content_type == ContentType.ERROR

    @pytest.mark.asyncio
    async def test_unknown_skill_errors(self):
        tool = LoadSkillTool()
        tool._session_id = None
        tool._storage = None
        result = await tool.execute({"skill_id": "nope"})
        assert result.content_type == ContentType.ERROR

    @pytest.mark.asyncio
    async def test_loads_session_skill(self, fake_redis):
        from api.skills.parser import ParsedResource, ParsedSkill

        repo = InMemoryDataArtifactRepository()
        storage = SessionScopedStorage(DataStorageService(repository=repo), "sess_x")
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_x",
            storage,
            ParsedSkill(
                name="S",
                description="d",
                body="session body text",
                allowed_tools=[],
                resources=[ParsedResource(path="reference.md", content=b"r")],
            ),
        )

        tool = LoadSkillTool()
        tool._session_id = "sess_x"
        tool._storage = storage
        result = await tool.execute({"skill_id": meta.id})
        assert "session body text" in result.data


class TestLoadSkillResourceTool:
    @pytest.mark.asyncio
    async def test_loads_builtin_resource(self):
        get_skill_registry()._skills["mine"] = SkillDefinition(
            id="mine",
            name="Mine",
            description="d",
            body="b",
            allowed_tools=[],
            resources={"reference.md": "the reference content"},
        )
        tool = LoadSkillResourceTool()
        tool._session_id = None
        tool._storage = None
        result = await tool.execute({"skill_id": "mine", "path": "reference.md"})
        assert result.content_type == ContentType.TEXT
        assert result.data == "the reference content"

    @pytest.mark.asyncio
    async def test_missing_args_error(self):
        tool = LoadSkillResourceTool()
        result = await tool.execute({"skill_id": "mine"})
        assert result.content_type == ContentType.ERROR

    @pytest.mark.asyncio
    async def test_unknown_resource_error(self):
        get_skill_registry()._skills["mine"] = SkillDefinition(
            id="mine", name="M", description="d", body="b", allowed_tools=[], resources={}
        )
        tool = LoadSkillResourceTool()
        tool._session_id = None
        tool._storage = None
        result = await tool.execute({"skill_id": "mine", "path": "nope.md"})
        assert result.content_type == ContentType.ERROR
