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

"""Search is the alternative to re-running work, so it has to actually find things."""

from typing import Any, Optional

import pytest

from api.tools.session_history import SearchSessionHistoryTool


class FakeRecorder:
    """Stands in for the request's TurnRecorder."""

    def __init__(self, findings: list[dict[str, Any]], fail: bool = False):
        self._findings = findings
        self._fail = fail
        self.calls: list[dict] = []

    async def search_findings(
        self,
        query: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "tool_name": tool_name, "limit": limit})
        if self._fail:
            raise RuntimeError("database is down")

        rows = self._findings
        if tool_name:
            rows = [r for r in rows if r["tool_name"] == tool_name]
        if query:
            needle = query.lower()
            rows = [
                r
                for r in rows
                if needle in r["summary"].lower() or needle in r["tool_name"].lower()
            ]
        return rows[:limit]


class FakeStorage:
    def __init__(self, artifacts: list[dict[str, Any]], fail: bool = False):
        self._artifacts = artifacts
        self._fail = fail

    async def search_artifacts(
        self,
        query: Optional[str] = None,
        data_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if self._fail:
            raise RuntimeError("database is down")

        rows = self._artifacts
        if data_type:
            rows = [r for r in rows if r["data_type"] == data_type]
        if query:
            needle = query.lower()
            rows = [r for r in rows if needle in (r["description"] or "").lower()]
        return rows[:limit]


_FINDINGS = [
    {
        "turn_seq": 1,
        "iteration": 2,
        "tool_name": "vina_dock",
        "summary": "vina_dock(ligand=sdf_a1) -> -9.50 kcal/mol",
        "success": True,
    },
    {
        "turn_seq": 1,
        "iteration": 4,
        "tool_name": "vina_dock",
        "summary": "vina_dock(ligand=sdf_d4) -> -6.10 kcal/mol",
        "success": True,
    },
    {
        "turn_seq": 2,
        "iteration": 1,
        "tool_name": "openfold3",
        "summary": "openfold3(seq) -> pLDDT 88.2",
        "success": True,
    },
    {
        "turn_seq": 2,
        "iteration": 3,
        "tool_name": "openfold3",
        "summary": "openfold3(seq) -> failed",
        "success": False,
    },
]

_ARTIFACTS = [
    {"reference": "pdb_b2", "data_type": "pdb", "description": "CDK2 receptor"},
    {"reference": "sdf_a1", "data_type": "sdf", "description": "aspirin analog"},
]


@pytest.fixture
def tool(monkeypatch):
    instance = SearchSessionHistoryTool()
    recorder = FakeRecorder(_FINDINGS)
    storage = FakeStorage(_ARTIFACTS)

    monkeypatch.setattr(
        "api.conversations.get_recorder", lambda: recorder, raising=False
    )
    monkeypatch.setattr(
        type(instance), "_storage", property(lambda self: storage), raising=False
    )
    instance._fake_recorder = recorder
    return instance


@pytest.mark.asyncio
async def test_a_number_from_earlier_in_the_session_is_recoverable(tool):
    result = await tool.execute({"query": "kcal"})

    assert "-9.50 kcal/mol" in result.data
    assert "-6.10 kcal/mol" in result.data


@pytest.mark.asyncio
async def test_results_carry_their_turn_and_iteration(tool):
    result = await tool.execute({"tool": "vina_dock"})

    assert "T1.2" in result.data
    assert "T1.4" in result.data


@pytest.mark.asyncio
async def test_filtering_by_tool_excludes_other_tools(tool):
    result = await tool.execute({"tool": "openfold3"})

    assert "pLDDT 88.2" in result.data
    assert "kcal/mol" not in result.data


@pytest.mark.asyncio
async def test_a_tool_filter_alone_does_not_dump_the_artifact_store(tool):
    """Burying the answer in unrelated artifacts is the failure being avoided."""
    result = await tool.execute({"tool": "openfold3"})

    assert "ARTIFACTS" not in result.data
    assert result.metadata["artifact_count"] == 0


@pytest.mark.asyncio
async def test_a_tool_filter_with_a_query_still_searches_artifacts(tool):
    result = await tool.execute({"tool": "openfold3", "query": "CDK2"})

    assert "pdb_b2" in result.data


@pytest.mark.asyncio
async def test_searching_is_not_itself_recorded_as_a_finding():
    """Otherwise each search becomes findable and the ledger echoes itself."""
    from api.tools.registry import _NOT_FINDINGS

    assert "search_session_history" in _NOT_FINDINGS


@pytest.mark.asyncio
async def test_a_failed_call_is_labelled_rather_than_hidden(tool):
    """A failure the model cannot see is a failure it will repeat."""
    result = await tool.execute({"tool": "openfold3"})

    assert "[failed]" in result.data


@pytest.mark.asyncio
async def test_artifacts_are_searched_alongside_results(tool):
    result = await tool.execute({"query": "CDK2"})

    assert "pdb_b2" in result.data
    assert "CDK2 receptor" in result.data


@pytest.mark.asyncio
async def test_an_empty_search_says_the_work_has_not_been_done(tool):
    result = await tool.execute({"query": "zzzznothing"})

    assert "No session history matches" in result.data
    assert result.metadata["match_count"] == 0


@pytest.mark.asyncio
async def test_limit_is_clamped_so_a_search_cannot_flood_the_context(tool):
    await tool.execute({"query": "kcal", "limit": 10_000})

    assert tool._fake_recorder.calls[-1]["limit"] == 50


@pytest.mark.asyncio
async def test_a_nonsense_limit_falls_back_to_the_default(tool):
    await tool.execute({"query": "kcal", "limit": "many"})

    assert tool._fake_recorder.calls[-1]["limit"] == 15


@pytest.mark.asyncio
async def test_artifacts_still_search_when_there_is_no_conversation_record(monkeypatch):
    """Training rollouts have artifacts but no recorder; the tool still works."""
    instance = SearchSessionHistoryTool()
    monkeypatch.setattr("api.conversations.get_recorder", lambda: None, raising=False)
    monkeypatch.setattr(
        type(instance),
        "_storage",
        property(lambda self: FakeStorage(_ARTIFACTS)),
        raising=False,
    )

    result = await instance.execute({"query": "aspirin"})

    assert "sdf_a1" in result.data


class TestTrainingParity:
    """A rollout has no database, so its ledger is searched in process."""

    @pytest.fixture(autouse=True)
    def no_recorder(self, monkeypatch):
        monkeypatch.setattr(
            "api.conversations.get_recorder", lambda: None, raising=False
        )

    @pytest.fixture
    def tool(self, monkeypatch):
        from api.conversations import reset_findings_source, set_findings_source

        instance = SearchSessionHistoryTool()
        monkeypatch.setattr(
            type(instance),
            "_storage",
            property(lambda self: FakeStorage([])),
            raising=False,
        )
        token = set_findings_source(lambda: _FINDINGS)
        yield instance
        reset_findings_source(token)

    @pytest.mark.asyncio
    async def test_an_in_process_ledger_is_searchable(self, tool):
        result = await tool.execute({"query": "kcal"})

        assert "-9.50 kcal/mol" in result.data
        assert "-6.10 kcal/mol" in result.data

    @pytest.mark.asyncio
    async def test_the_tool_filter_works_the_same_way(self, tool):
        result = await tool.execute({"tool": "openfold3"})

        assert "pLDDT 88.2" in result.data
        assert "kcal/mol" not in result.data

    @pytest.mark.asyncio
    async def test_results_come_back_newest_first(self, tool):
        from api.conversations import search_findings_in_memory

        rows = search_findings_in_memory(_FINDINGS, limit=2)

        assert [r["turn_seq"] for r in rows] == [2, 2]

    @pytest.mark.asyncio
    async def test_without_a_ledger_the_tool_reports_nothing_rather_than_failing(
        self, monkeypatch
    ):
        instance = SearchSessionHistoryTool()
        monkeypatch.setattr(
            type(instance),
            "_storage",
            property(lambda self: FakeStorage([])),
            raising=False,
        )

        result = await instance.execute({"query": "kcal"})

        assert result.metadata["match_count"] == 0


@pytest.mark.asyncio
async def test_a_database_failure_degrades_instead_of_crashing_the_turn(monkeypatch):
    instance = SearchSessionHistoryTool()
    monkeypatch.setattr(
        "api.conversations.get_recorder",
        lambda: FakeRecorder(_FINDINGS, fail=True),
        raising=False,
    )
    monkeypatch.setattr(
        type(instance),
        "_storage",
        property(lambda self: FakeStorage(_ARTIFACTS, fail=True)),
        raising=False,
    )

    result = await instance.execute({"query": "anything"})

    assert result.metadata["match_count"] == 0


class TestAgainstRealStorage:
    """The in-memory repository is what training rollouts actually use."""

    @staticmethod
    async def _storage():
        from api.storage.memory import InMemoryDataArtifactRepository
        from api.storage.service import DataStorageService

        service = DataStorageService(repository=InMemoryDataArtifactRepository())
        await service.store("pdb", b"ATOM", description="CDK2 kinase receptor")
        await service.store("sdf", b"MOL", description="aspirin analog series A")
        await service.store("sdf", b"MOL", description="ibuprofen reference")
        return service

    @pytest.mark.asyncio
    async def test_description_search_finds_the_right_artifact(self):
        storage = await self._storage()
        found = await storage.search_artifacts(query="aspirin")

        assert len(found) == 1
        assert found[0]["description"] == "aspirin analog series A"

    @pytest.mark.asyncio
    async def test_data_type_narrows_without_a_query(self):
        storage = await self._storage()
        found = await storage.search_artifacts(data_type="sdf")

        assert {f["data_type"] for f in found} == {"sdf"}
        assert len(found) == 2

    @pytest.mark.asyncio
    async def test_search_is_case_insensitive(self):
        storage = await self._storage()

        assert await storage.search_artifacts(query="CDK2")
        assert await storage.search_artifacts(query="cdk2")

    @pytest.mark.asyncio
    async def test_search_returns_metadata_not_payloads(self):
        storage = await self._storage()
        found = await storage.search_artifacts(query="aspirin")

        assert set(found[0]) == {
            "reference",
            "data_type",
            "description",
            "metadata",
            "created_at",
        }

    @pytest.mark.asyncio
    async def test_limit_is_honoured(self):
        storage = await self._storage()

        assert len(await storage.search_artifacts(limit=2)) == 2


def test_the_schema_tells_the_model_not_to_re_run_work():
    schema = SearchSessionHistoryTool().get_schema()
    description = schema["function"]["description"]

    assert "Do NOT re-run" in description
    assert set(schema["function"]["parameters"]["properties"]) == {
        "query",
        "tool",
        "data_type",
        "limit",
    }
