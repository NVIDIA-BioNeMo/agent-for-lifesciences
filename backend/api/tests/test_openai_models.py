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

"""Tests for discovering GPT models from an OpenAI key.

``GET /v1/models`` carries only ids and creation times, so these tests pin the
rules that turn that listing into dropdown entries: which ids belong to a
family, newest model per family, Rosalind behind FEAT_ROSALIND, fixed limits,
and a refresh that never empties the dropdown on a transient failure.
"""

import time

import httpx
import pytest
from api.llm import openai_models
from api.llm.config import PROVIDER_OPENAI, all_models, available_models
from api.llm.openai_models import (
    MAX_OUTPUT_TOKENS_CAP,
    discover,
    family_of,
    latest_per_family,
    pretty_name_for,
    refresh_openai_models,
    reset_discovery_state,
    rosalind_enabled,
    to_model_config,
)


def model(model_id, created, **extra):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "system", **extra}


LISTING = [
    model("gpt-6-astra", 1_780_000_000),
    model("gpt-6-astra-2026-05-01", 1_779_000_000),
    model("gpt-6.1-sol", 1_790_000_000),
    model("gpt-6-sol", 1_770_000_000),
    model("gpt-6.1-sol-pro", 1_791_000_000),
    model("gpt-6.1-sol-chat-latest", 1_792_000_000),
    model("gpt-6-luna", 1_775_000_000),
    model("gpt-5.5-rosalind", 1_760_000_000),
    model("gpt-rosalind-research", 1_785_000_000),
    model("gpt-rosalind-preview", 1_795_000_000),
    model("gpt-rosalind-research-2026-06-01", 1_784_000_000),
    model("text-embedding-3-large", 1_700_000_000),
    model("o3", 1_740_000_000),
]


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.delenv("FEAT_ROSALIND", raising=False)
    reset_discovery_state()
    yield
    reset_discovery_state()


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "API_KEY_OPENAI"
    path.write_text("test-key-not-real\n")
    return path


@pytest.fixture
def listing(monkeypatch):
    """Stub the HTTP call; returns a dict to change the listing or make it fail."""
    state = {"models": LISTING, "error": None, "calls": 0, "keys": []}

    def fake_list_models(api_key, timeout=10.0):
        state["calls"] += 1
        state["keys"].append(api_key)
        if state["error"]:
            raise state["error"]
        return state["models"]

    monkeypatch.setattr(openai_models, "list_models", fake_list_models)
    return state


def openai_ids():
    return [m.id for m in all_models().values() if m.provider == PROVIDER_OPENAI]


class TestFamilyOf:
    @pytest.mark.parametrize(
        "model_id, family",
        [
            ("gpt-6-astra", "astra"),
            ("gpt-6.1-sol", "sol"),
            ("gpt-6-luna", "luna"),
            ("gpt-rosalind-research", "rosalind"),
            ("gpt-5.5-rosalind", "rosalind"),
        ],
    )
    def test_offered(self, model_id, family):
        assert family_of(model_id) == family

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-6-astra-2026-05-01",
            "gpt-6.1-sol-pro",
            "gpt-6.1-sol-chat-latest",
            "gpt-rosalind-preview",
            "gpt-rosalind-research-2026-06-01",
            "gpt-rosalind-research-260601",
            "o3",
            "",
        ],
    )
    def test_not_offered(self, model_id):
        assert family_of(model_id) is None


class TestLatestPerFamily:
    def test_core_families_only_by_default(self):
        names = {m["id"] for m in latest_per_family(LISTING)}
        assert names == {"gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"}

    def test_rosalind_when_included(self):
        names = {m["id"] for m in latest_per_family(LISTING, include_rosalind=True)}
        assert names == {"gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "gpt-rosalind-research"}

    def test_version_breaks_a_created_tie(self):
        tied = [model("gpt-6-sol", 1), model("gpt-6.1-sol", 1)]
        assert [m["id"] for m in latest_per_family(tied)] == ["gpt-6.1-sol"]

    def test_shut_down_models_are_skipped(self):
        listing = [
            model("gpt-6.1-sol", 2, shutdown_date=int(time.time()) - 60),
            model("gpt-6-sol", 1, shutdown_date="2999-01-01"),
            model("gpt-6-luna", 1, shutdown_date="2000-01-01"),
            model("gpt-6-astra", 1, shutdown_date="not-a-date"),
        ]
        assert {m["id"] for m in latest_per_family(listing)} == {"gpt-6-sol", "gpt-6-astra"}


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), ("false", False), ("1", False), ("true", True), ("TRUE", True), (" True ", True)],
)
def test_rosalind_flag(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("FEAT_ROSALIND", value)
    assert rosalind_enabled() is expected


@pytest.mark.parametrize(
    "model_id, pretty",
    [
        ("gpt-6.1-sol", "GPT-6.1 Sol"),
        ("gpt-6-astra", "GPT-6 Astra"),
        ("gpt-rosalind-research", "GPT Rosalind Research"),
        ("gpt-5.5-rosalind", "GPT-5.5 Rosalind"),
        ("o3", "o3"),
    ],
)
def test_pretty_name(model_id, pretty):
    assert pretty_name_for(model_id) == pretty


class TestToModelConfig:
    def test_fields(self):
        config = to_model_config(model("gpt-6.1-sol", 1), "/k")
        assert config.id == "openai/gpt-6.1-sol"
        assert config.name == "gpt-6.1-sol"
        assert config.provider == PROVIDER_OPENAI
        assert config.api_key_file == "/k"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.pretty_name == "GPT-6.1 Sol"
        assert config.supports_reasoning and config.retains_reasoning
        assert config.supports_temperature is False
        assert config.max_tokens == MAX_OUTPUT_TOKENS_CAP
        assert config.context_window == openai_models.CONTEXT_CAP

    def test_effort_override(self, monkeypatch):
        assert to_model_config(model("gpt-6-luna", 1)).effort is None
        monkeypatch.setattr(openai_models, "EFFORT", "high")
        assert to_model_config(model("gpt-6-luna", 1)).effort == "high"


class TestDiscover:
    def test_orders_by_family_preference(self, key_file, listing):
        assert [c.name for c in discover(str(key_file))] == ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"]

    def test_rosalind_behind_the_flag(self, key_file, listing, monkeypatch):
        monkeypatch.setenv("FEAT_ROSALIND", "TRUE")
        names = [c.name for c in discover(str(key_file))]
        assert names == ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "gpt-rosalind-research"]

    def test_absent_or_empty_key_skips_the_request(self, tmp_path, listing):
        empty = tmp_path / "empty"
        empty.write_text("  \n")
        assert discover(str(empty)) == []
        assert discover(str(tmp_path / "missing")) == []
        assert discover(str(tmp_path)) == []  # k8s: a missing optional secret leaves a directory
        assert listing["calls"] == 0

    @pytest.mark.parametrize("error", [httpx.ConnectError("down"), ValueError("bad json"), OSError("eio")])
    def test_failure_returns_nothing(self, key_file, listing, error):
        listing["error"] = error
        assert discover(str(key_file)) == []


class TestRefresh:
    def test_registers_discovered_models_as_available(self, key_file, listing):
        refresh_openai_models(key_file=str(key_file))
        assert openai_ids() == ["openai/gpt-6-astra", "openai/gpt-6.1-sol", "openai/gpt-6-luna"]
        assert {m.id for m in available_models()} >= set(openai_ids())

    def test_within_ttl_does_not_refetch(self, key_file, listing):
        refresh_openai_models(key_file=str(key_file))
        refresh_openai_models(key_file=str(key_file))
        assert listing["calls"] == 1
        refresh_openai_models(force=True, key_file=str(key_file))
        assert listing["calls"] == 2

    def test_failure_keeps_the_previous_list(self, key_file, listing):
        refresh_openai_models(key_file=str(key_file))
        before = openai_ids()
        listing["error"] = httpx.ConnectError("down")
        refresh_openai_models(force=True, key_file=str(key_file))
        assert openai_ids() == before

    def test_failure_is_retried_before_the_ttl(self, key_file, listing, monkeypatch):
        listing["error"] = httpx.ConnectError("down")
        refresh_openai_models(key_file=str(key_file))
        assert openai_ids() == []

        listing["error"] = None
        monkeypatch.setattr(openai_models, "FAILURE_RETRY_S", 0.0)
        refresh_openai_models(key_file=str(key_file))
        assert len(openai_ids()) == 3

    def test_removing_the_key_removes_the_models(self, key_file, listing):
        refresh_openai_models(key_file=str(key_file))
        key_file.write_text("")
        refresh_openai_models(key_file=str(key_file))
        assert openai_ids() == []

    def test_changed_key_refetches_and_failure_drops_old_models(self, key_file, listing):
        refresh_openai_models(key_file=str(key_file))
        key_file.write_text("another-key")
        listing["error"] = httpx.HTTPStatusError(
            "401", request=httpx.Request("GET", "https://x"), response=httpx.Response(401)
        )
        refresh_openai_models(key_file=str(key_file))
        assert listing["keys"] == ["test-key-not-real", "another-key"]
        assert openai_ids() == []


def test_list_models_sends_bearer_key(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"object": "list", "data": [{"id": "a"}, {"id": "b"}]})

    real_client = httpx.Client
    monkeypatch.setattr(
        openai_models.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert [m["id"] for m in openai_models.list_models("test-key-not-real")] == ["a", "b"]
    assert seen[0].headers["authorization"] == "Bearer test-key-not-real"
    assert str(seen[0].url) == "https://api.openai.com/v1/models"
