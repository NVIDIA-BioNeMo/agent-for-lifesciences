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

"""Tests for SkillSessionStore (Redis catalog + Postgres artifact bodies)."""

import pytest

from api.skills.contract import SkillContract, parse_contract
from api.skills.parser import ParsedResource, ParsedSkill
from api.skills.registry import SkillDefinition, get_skill_registry
from api.skills.session_store import SkillSessionStore
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService


class FakeRedis:
    """Minimal async Redis stand-in supporting get/set (decode_responses)."""

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
def clear_builtin_registry():
    get_skill_registry()._skills.clear()
    yield
    get_skill_registry()._skills.clear()


@pytest.fixture
def storage():
    repo = InMemoryDataArtifactRepository()
    base = DataStorageService(repository=repo)
    return SessionScopedStorage(base, session_id="sess_1")


def _parsed_skill():
    return ParsedSkill(
        name="Demo",
        description="A demo skill.",
        body="# Demo\nInstructions here.",
        allowed_tools=["search_pubchem"],
        resources=[ParsedResource(path="reference.md", content=b"reference text")],
    )


class TestInstallAndLoad:
    @pytest.mark.asyncio
    async def test_install_persists_catalog_and_body(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill("sess_1", storage, _parsed_skill())

        assert meta.id.startswith("skill_")
        assert meta.name == "Demo"
        assert meta.body_ref
        assert len(meta.resources) == 1

        # Body is retrievable from storage.
        body = (await storage.retrieve(meta.body_ref)).decode()
        assert "Instructions here." in body

        # Catalog persisted to redis.
        assert fake_redis.store  # non-empty

    @pytest.mark.asyncio
    async def test_get_skill_returns_body(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill("sess_1", storage, _parsed_skill())

        loaded = await store.get_skill("sess_1", storage, meta.id)
        assert loaded is not None
        assert "Instructions here." in loaded.body
        assert loaded.allowed_tools == ["search_pubchem"]
        assert loaded.resource_paths == ["reference.md"]

    @pytest.mark.asyncio
    async def test_get_resource_returns_content(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill("sess_1", storage, _parsed_skill())

        content = await store.get_resource("sess_1", storage, meta.id, "reference.md")
        assert content == "reference text"

    @pytest.mark.asyncio
    async def test_get_unknown_skill_returns_none(self, fake_redis, storage):
        store = SkillSessionStore()
        assert await store.get_skill("sess_1", storage, "skill_missing") is None


class TestCatalogMerge:
    @pytest.mark.asyncio
    async def test_catalog_merges_builtin_and_session(self, fake_redis, storage):
        get_skill_registry()._skills["builtin_one"] = SkillDefinition(
            id="builtin_one",
            name="Builtin One",
            description="curated",
            body="body",
            allowed_tools=[],
            resources={},
        )
        store = SkillSessionStore()
        await store.install_skill("sess_1", storage, _parsed_skill())

        catalog = await store.get_catalog("sess_1")
        by_id = {m.id: m for m in catalog}
        assert "builtin_one" in by_id
        assert by_id["builtin_one"].source == "builtin"
        session_entries = [m for m in catalog if m.source == "session"]
        assert len(session_entries) == 1
        assert session_entries[0].to_public_dict()["removable"] is True

    @pytest.mark.asyncio
    async def test_builtin_get_skill_without_storage(self, fake_redis):
        get_skill_registry()._skills["b"] = SkillDefinition(
            id="b",
            name="B",
            description="d",
            body="curated body",
            allowed_tools=[],
            resources={"reference.md": "curated ref"},
        )
        store = SkillSessionStore()
        loaded = await store.get_skill(None, None, "b")
        assert loaded is not None
        assert loaded.body == "curated body"
        res = await store.get_resource(None, None, "b", "reference.md")
        assert res == "curated ref"


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_deletes_catalog_and_artifacts(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill("sess_1", storage, _parsed_skill())

        removed = await store.remove_skill("sess_1", storage, meta.id)
        assert removed is True
        assert await store.get_skill("sess_1", storage, meta.id) is None
        assert await storage.exists(meta.body_ref) is False

    @pytest.mark.asyncio
    async def test_remove_unknown_returns_false(self, fake_redis, storage):
        store = SkillSessionStore()
        assert await store.remove_skill("sess_1", storage, "skill_missing") is False


ENDPOINT_DATA = {
    "tool_suffix": "predict",
    "summary": "Predict a structure.",
    "url": "https://api.test/predict",
    "inputs": [{"name": "sequence", "data_type": "sequence"}],
    "outputs": [
        {"name": "structure", "json_path": "$.structure", "data_type": "mmcif"}
    ],
}


def _contract(*, confirmed: bool) -> SkillContract:
    return parse_contract([ENDPOINT_DATA], source="llm").model_copy(
        update={"confirmed": confirmed}
    )


class TestContractPersistence:
    @pytest.mark.asyncio
    async def test_install_stores_the_contract_as_json(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=False)
        )

        assert meta.contract["endpoints"][0]["tool_suffix"] == "predict"
        assert meta.parsed_contract.source == "llm"
        assert meta.parsed_contract.confirmed is False

    @pytest.mark.asyncio
    async def test_contract_survives_a_redis_round_trip(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )

        reloaded = next(
            m for m in await SkillSessionStore().get_catalog("sess_1") if m.id == meta.id
        )
        assert reloaded.parsed_contract == meta.parsed_contract

    @pytest.mark.asyncio
    async def test_an_explicit_contract_overrides_the_parsed_one(
        self, fake_redis, storage
    ):
        # The install route passes an LLM proposal here for skills whose
        # frontmatter declared nothing.
        parsed = _parsed_skill()
        parsed.contract = None
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, parsed, _contract(confirmed=False)
        )
        assert meta.parsed_contract is not None

    @pytest.mark.asyncio
    async def test_a_contract_that_no_longer_validates_is_treated_as_absent(
        self, fake_redis, storage
    ):
        # A catalog entry can outlive a code change that tightens validation;
        # that must not break every prompt that renders the catalog.
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )
        meta.contract["endpoints"][0]["url"] = "not-a-url"
        assert meta.parsed_contract is None
        assert meta.to_public_dict()["contract"] is None

    @pytest.mark.asyncio
    async def test_public_dict_lists_tool_names_only_once_confirmed(
        self, fake_redis, storage
    ):
        store = SkillSessionStore()
        unconfirmed = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=False)
        )
        assert unconfirmed.to_public_dict()["tool_names"] == []

        confirmed = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )
        assert confirmed.to_public_dict()["tool_names"] == [
            f"{confirmed.id}_predict"
        ]


class TestSynthesizedTools:
    @pytest.mark.asyncio
    async def test_confirmed_contract_yields_a_tool(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )

        tools = await store.get_tools("sess_1")
        assert [t.name for t in tools] == [f"{meta.id}_predict"]

    @pytest.mark.asyncio
    async def test_unconfirmed_contract_yields_nothing(self, fake_redis, storage):
        store = SkillSessionStore()
        await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=False)
        )
        assert await store.get_tools("sess_1") == []

    @pytest.mark.asyncio
    async def test_schemas_match_the_tools(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )

        schemas = await store.get_tool_schemas("sess_1")
        assert [s["function"]["name"] for s in schemas] == [f"{meta.id}_predict"]

    @pytest.mark.asyncio
    async def test_lookup_by_name(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )

        found = await store.get_tool_by_name("sess_1", f"{meta.id}_predict")
        assert found is not None
        assert await store.get_tool_by_name("sess_1", "nope") is None

    @pytest.mark.asyncio
    async def test_confirming_a_contract_activates_the_tool(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=False)
        )
        assert await store.get_tools("sess_1") == []

        await store.set_contract("sess_1", meta.id, _contract(confirmed=True))
        assert [t.name for t in await store.get_tools("sess_1")] == [
            f"{meta.id}_predict"
        ]

    @pytest.mark.asyncio
    async def test_clearing_a_contract_removes_the_tool(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )

        await store.set_contract("sess_1", meta.id, SkillContract(source="none"))
        assert await store.get_tools("sess_1") == []

    @pytest.mark.asyncio
    async def test_set_contract_on_an_unknown_skill_returns_none(
        self, fake_redis, storage
    ):
        store = SkillSessionStore()
        result = await store.set_contract(
            "sess_1", "skill_missing", _contract(confirmed=True)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_removing_a_skill_removes_its_tools(self, fake_redis, storage):
        store = SkillSessionStore()
        meta = await store.install_skill(
            "sess_1", storage, _parsed_skill(), _contract(confirmed=True)
        )

        await store.remove_skill("sess_1", storage, meta.id)
        assert await store.get_tools("sess_1") == []
