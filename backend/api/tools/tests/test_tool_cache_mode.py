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

"""Tests for AEVOM_TOOL_CACHE_MODE and replay of storage-backed tool responses.

The interesting case is not "does the cache return the recorded string" -- it is
whether a replayed response still refers to data that exists. Aevom's tools pass
artifacts by reference, so a cache that replays references without replaying the
underlying rows produces a run where every downstream tool fails on a lookup.
"""

import base64
import inspect
import json

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import BaseTool, ContentType, DataIOCategory, ToolResult, ToolType
from api.tools.mock_registry import CachedResponse, CachingRegistry, MockRegistry
from api.tools.registry import ToolRegistry, get_registry


class StoringTool(BaseTool):
    """Stands in for the OUTPUT_STORES family (fetch_and_clean_pdb et al.).

    Records how many times it really ran so tests can prove the cache took over.
    """

    def __init__(self):
        self.calls = 0

    @property
    def name(self) -> str:
        return "store_thing"

    @property
    def description(self) -> str:
        return "Stores a payload and returns a reference"

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"payload": {"type": "string"}}}

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.OUTPUT_STORES

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments):
        self.calls += 1
        payload = arguments["payload"].encode()
        # Storage arrives via the registry's ContextVar, same as real tools.
        primary = await self._storage.store(
            "pdb", payload, {"origin": "test"}, description="primary"
        )
        secondary = await self._storage.store("array", b"\x00\x01binary", {})
        return ToolResult.data_reference(
            reference=primary,
            data_type="pdb",
            message=f"Stored as '{primary}'",
            # A second reference reachable only through metadata, mirroring how
            # vina_dock reports its docked complex.
            metadata={"extra_ref": secondary, "nested": {"refs": [secondary]}},
        )


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def base_registry():
    registry = ToolRegistry()
    registry.register(StoringTool())
    return registry


async def _record(cache_dir, base_registry, storage, payload="HELLO"):
    registry = CachingRegistry(
        base_registry=base_registry, cache_dir=cache_dir, mode="record"
    )
    return await registry.execute("store_thing", {"payload": payload}, storage)


@pytest.mark.asyncio
async def test_record_captures_artifact_payloads(tmp_path, base_registry, storage):
    await _record(tmp_path, base_registry, storage)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    # These files are committed under evals/fixtures/tool_cache/, and a colon in
    # a tracked path makes `git checkout` fail on Windows.
    assert ":" not in files[0].name
    entry = json.loads(files[0].read_text())

    artifacts = {a["reference"]: a for a in entry["artifacts"]}
    assert len(artifacts) == 2, "both the primary and metadata-only ref must be captured"

    primary = next(a for a in artifacts.values() if a["data_type"] == "pdb")
    assert base64.b64decode(primary["content_base64"]) == b"HELLO"
    assert primary["description"] == "primary"

    binary = next(a for a in artifacts.values() if a["data_type"] == "array")
    assert base64.b64decode(binary["content_base64"]) == b"\x00\x01binary"


@pytest.mark.asyncio
async def test_replay_rehydrates_into_a_fresh_store(tmp_path, base_registry, storage):
    recorded = await _record(tmp_path, base_registry, storage)
    tool = base_registry.get("store_thing")
    assert tool.calls == 1

    # A brand new store, as a fresh eval trial would have.
    fresh = DataStorageService(repository=InMemoryDataArtifactRepository())
    replay = CachingRegistry(cache_dir=tmp_path, mode="replay")
    replayed = await replay.execute("store_thing", {"payload": "HELLO"}, fresh)

    assert tool.calls == 1, "replay must not reach the real tool"
    assert replayed.content_type == ContentType.DATA_REFERENCE

    # The reference is new, and it resolves.
    assert replayed.data != recorded.data
    assert await fresh.retrieve(replayed.data) == b"HELLO"


@pytest.mark.asyncio
async def test_replay_rewrites_references_everywhere(tmp_path, base_registry, storage):
    await _record(tmp_path, base_registry, storage)

    fresh = DataStorageService(repository=InMemoryDataArtifactRepository())
    replay = CachingRegistry(cache_dir=tmp_path, mode="replay")
    replayed = await replay.execute("store_thing", {"payload": "HELLO"}, fresh)

    extra = replayed.metadata["extra_ref"]
    assert await fresh.retrieve(extra) == b"\x00\x01binary"
    # The same rewrite must reach nested containers, not just top-level keys.
    assert replayed.metadata["nested"]["refs"] == [extra]
    # And the message text, which the model reads, must name the live reference.
    assert replayed.metadata["message"] == f"Stored as '{replayed.data}'"


@pytest.mark.asyncio
async def test_replay_miss_is_an_explicit_error(tmp_path, storage):
    replay = CachingRegistry(cache_dir=tmp_path, mode="replay")
    result = await replay.execute("store_thing", {"payload": "unseen"}, storage)

    assert result.content_type == ContentType.ERROR
    assert "CACHE MISS" in result.data


@pytest.mark.asyncio
async def test_mixed_mode_falls_through_then_serves_cache(
    tmp_path, base_registry, storage
):
    mixed = CachingRegistry(
        base_registry=base_registry, cache_dir=tmp_path, mode="mixed"
    )
    tool = base_registry.get("store_thing")

    await mixed.execute("store_thing", {"payload": "A"}, storage)
    assert tool.calls == 1

    fresh = DataStorageService(repository=InMemoryDataArtifactRepository())
    second = await mixed.execute("store_thing", {"payload": "A"}, fresh)
    assert tool.calls == 1, "second identical call should come from cache"
    assert await fresh.retrieve(second.data) == b"A"


class TestCacheKey:
    def test_cosmetic_arguments_do_not_change_the_key(self):
        """The model rewrites `description` every run; the response is identical.

        Observed: a recorded fetch_and_clean_pdb of 1CRN missed on replay only
        because the model called it "1CRN crambin structure" the second time.
        """
        base = {"pdb_id": "1CRN", "keep_ligands": False}
        assert CachedResponse.hash_args(
            {**base, "description": "1CRN structure for analysis"}
        ) == CachedResponse.hash_args({**base, "description": "1CRN crambin"})
        assert CachedResponse.hash_args({**base, "label": "a"}) == (
            CachedResponse.hash_args(base)
        )

    def test_significant_arguments_still_change_the_key(self):
        """keep_ligands changes which atoms come back, so it must not collapse."""
        assert CachedResponse.hash_args(
            {"pdb_id": "1CRN", "keep_ligands": True}
        ) != CachedResponse.hash_args({"pdb_id": "1CRN", "keep_ligands": False})
        assert CachedResponse.hash_args({"pdb_id": "1CRN"}) != (
            CachedResponse.hash_args({"pdb_id": "1HSG"})
        )

    def test_key_is_order_independent(self):
        assert CachedResponse.hash_args({"a": 1, "b": 2}) == (
            CachedResponse.hash_args({"b": 2, "a": 1})
        )

    @pytest.mark.asyncio
    async def test_a_reworded_description_replays(self, tmp_path, storage):
        """End to end, since the hash and the lookup could disagree."""

        class FetchTool(StoringTool):
            @property
            def name(self) -> str:
                return "fetch_thing"

            async def execute(self, arguments):
                self.calls += 1
                return ToolResult.text("ATOM  1")

        base = ToolRegistry()
        base.register(FetchTool())
        registry = CachingRegistry(
            base_registry=base, cache_dir=tmp_path, mode="record"
        )
        await registry.execute(
            "fetch_thing", {"pdb_id": "1CRN", "description": "as recorded"}, storage
        )

        replay = CachingRegistry(cache_dir=tmp_path, mode="replay")
        result = await replay.execute(
            "fetch_thing", {"pdb_id": "1CRN", "description": "worded differently"}, storage
        )
        assert result.data == "ATOM  1"


class TestNeverCached:
    """Session-coupled tools must reach the real tool in every mode.

    The motivating case is structure_query: it blocks on the client POSTing a
    selection back, so a cached reply would make the eval that exists to prove
    the browser bridge works pass with the bridge unplugged.
    """

    @pytest.fixture
    def viewer_registry(self, base_registry):
        class ViewerTool(StoringTool):
            @property
            def name(self) -> str:
                return "structure_query"

            async def execute(self, arguments):
                self.calls += 1
                return ToolResult.text(f"selected {self.calls}")

        base_registry.register(ViewerTool())
        return base_registry

    @pytest.mark.asyncio
    async def test_record_does_not_write_a_cache_entry(
        self, tmp_path, viewer_registry, storage
    ):
        registry = CachingRegistry(
            base_registry=viewer_registry, cache_dir=tmp_path, mode="record"
        )
        await registry.execute("structure_query", {"molql": "(sel.atom.all)"}, storage)

        assert list(tmp_path.glob("structure_query*.json")) == []

    @pytest.mark.asyncio
    async def test_replay_still_runs_the_real_tool(
        self, tmp_path, viewer_registry, storage
    ):
        registry = CachingRegistry(
            base_registry=viewer_registry, cache_dir=tmp_path, mode="replay"
        )
        tool = viewer_registry.get("structure_query")

        first = await registry.execute("structure_query", {"q": 1}, storage)
        second = await registry.execute("structure_query", {"q": 1}, storage)

        assert tool.calls == 2, "identical calls must both reach the live tool"
        assert (first.data, second.data) == ("selected 1", "selected 2")

    @pytest.mark.asyncio
    async def test_other_tools_are_still_cached(
        self, tmp_path, viewer_registry, storage
    ):
        """The exemption is per tool, not a blanket disable."""
        registry = CachingRegistry(
            base_registry=viewer_registry, cache_dir=tmp_path, mode="record"
        )
        await registry.execute("store_thing", {"payload": "A"}, storage)

        assert len(list(tmp_path.glob("store_thing*.json"))) == 1

    @pytest.mark.asyncio
    async def test_replay_without_a_base_registry_says_why(self, tmp_path, storage):
        """Replay normally needs no base registry; these tools do."""
        registry = CachingRegistry(cache_dir=tmp_path, mode="replay")
        result = await registry.execute("structure_query", {}, storage)

        assert result.content_type == ContentType.ERROR
        assert "never cached" in result.data

    @pytest.mark.asyncio
    async def test_the_list_is_overridable(self, tmp_path, viewer_registry, storage):
        registry = CachingRegistry(
            base_registry=viewer_registry,
            cache_dir=tmp_path,
            mode="record",
            never_cached=[],
        )
        await registry.execute("structure_query", {"q": 1}, storage)

        assert len(list(tmp_path.glob("structure_query*.json"))) == 1

    def test_the_default_list_names_real_tools(self):
        """A rename would otherwise silently re-enable caching for that tool."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.delenv("AEVOM_TOOL_CACHE_MODE", raising=False)
        import api.tools.registry as registry_module

        monkeypatch.setattr(registry_module, "_registry", None)
        try:
            live = set(get_registry().tool_names)
        finally:
            monkeypatch.undo()

        unknown = sorted(CachingRegistry.NEVER_CACHED - live)
        assert not unknown, (
            f"NEVER_CACHED names tools that no longer exist: {unknown}; caching "
            "is silently back on for whatever they were renamed to"
        )


@pytest.mark.asyncio
async def test_input_references_are_not_captured(tmp_path, base_registry, storage):
    """Only artifacts this call created belong in its cache entry.

    An input ref already exists in the replaying run's store, so re-storing it
    would duplicate rows and make the artifact listing disagree with a live run.
    """
    existing = await storage.store("pdb", b"pre-existing", {})

    class EchoRefTool(StoringTool):
        @property
        def name(self) -> str:
            return "echo_ref"

        async def execute(self, arguments):
            self.calls += 1
            return ToolResult.text(f"used {arguments['pdb_ref']}")

    base_registry.register(EchoRefTool())
    registry = CachingRegistry(
        base_registry=base_registry, cache_dir=tmp_path, mode="record"
    )
    await registry.execute("echo_ref", {"pdb_ref": existing}, storage)

    entry = json.loads(next(tmp_path.glob("echo_ref*.json")).read_text())
    captured = {a["reference"] for a in entry["artifacts"]}
    assert existing not in captured, (
        "the input reference was re-captured; replaying it would duplicate an "
        "artifact the run already has"
    )
    assert captured == set()


@pytest.mark.asyncio
async def test_reference_shaped_text_is_not_mistaken_for_an_artifact(
    tmp_path, base_registry, storage
):
    """The embedded-reference pattern is permissive; storage lookup is the filter."""

    class ChattyTool(StoringTool):
        @property
        def name(self) -> str:
            return "chatty"

        async def execute(self, arguments):
            self.calls += 1
            return ToolResult.text("commit abcdef1234 and token_deadbeef are not refs")

    base_registry.register(ChattyTool())
    registry = CachingRegistry(
        base_registry=base_registry, cache_dir=tmp_path, mode="record"
    )
    await registry.execute("chatty", {}, storage)

    entry = json.loads(next(tmp_path.glob("chatty*.json")).read_text())
    assert entry["artifacts"] == []

    # And replay leaves the prose untouched.
    replay = CachingRegistry(cache_dir=tmp_path, mode="replay")
    replayed = await replay.execute("chatty", {}, storage)
    assert replayed.data == "commit abcdef1234 and token_deadbeef are not refs"


class TestGetRegistryCacheMode:
    """The env var wiring, including the failure mode for a typo."""

    @pytest.fixture(autouse=True)
    def reset_global_registry(self, monkeypatch):
        import api.tools.registry as registry_module

        monkeypatch.setattr(registry_module, "_registry", None)
        yield
        monkeypatch.setattr(registry_module, "_registry", None)

    def test_default_is_a_plain_registry(self, monkeypatch):
        monkeypatch.delenv("AEVOM_TOOL_CACHE_MODE", raising=False)
        registry = get_registry()
        assert type(registry) is ToolRegistry

    def test_off_is_a_plain_registry(self, monkeypatch):
        monkeypatch.setenv("AEVOM_TOOL_CACHE_MODE", "off")
        assert type(get_registry()) is ToolRegistry

    def test_replay_wraps_in_caching_registry(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEVOM_TOOL_CACHE_MODE", "replay")
        monkeypatch.setenv("AEVOM_TOOL_CACHE_DIR", str(tmp_path))
        registry = get_registry()
        assert isinstance(registry, CachingRegistry)

    def test_wrapper_still_advertises_the_real_tools(self, monkeypatch, tmp_path):
        """Schemas must come from the base registry, or the model sees no tools."""
        monkeypatch.delenv("AEVOM_TOOL_CACHE_MODE", raising=False)
        import api.tools.registry as registry_module

        live_tools = set(get_registry().tool_names)
        monkeypatch.setattr(registry_module, "_registry", None)

        monkeypatch.setenv("AEVOM_TOOL_CACHE_MODE", "replay")
        monkeypatch.setenv("AEVOM_TOOL_CACHE_DIR", str(tmp_path))
        cached = get_registry()
        assert set(cached._base_registry.tool_names) == live_tools

    def test_unknown_mode_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("AEVOM_TOOL_CACHE_MODE", "relpay")
        with pytest.raises(ValueError, match="AEVOM_TOOL_CACHE_MODE"):
            get_registry()


@pytest.mark.parametrize("subclass", [CachingRegistry, MockRegistry])
def test_execute_signature_tracks_the_base_registry(subclass):
    """A drifted override is a TypeError at the first tool call, nowhere sooner.

    ToolRegistry.execute() gained `request_id` while these overrides did not,
    and because the graph passes it by keyword every tool call failed the moment
    a cache mode was switched on -- with the type checker perfectly happy.
    """
    base = inspect.signature(ToolRegistry.execute).parameters
    override = inspect.signature(subclass.execute).parameters

    missing = [name for name in base if name not in override]
    assert not missing, (
        f"{subclass.__name__}.execute() is missing {missing}; callers pass these "
        "by keyword, so every tool call would raise TypeError"
    )
    for name, parameter in base.items():
        assert override[name].default == parameter.default, (
            f"{subclass.__name__}.execute() changes the default of {name!r}"
        )
