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

"""Tests for discovering Claude models from an Anthropic key.

Nothing about individual Claude models is hardcoded, so these tests pin the
rules that turn a ``GET /v1/models`` listing into dropdown entries: newest
model per family, capability-driven thinking mode, capped limits, and a
refresh that never empties the dropdown on a transient failure.
"""

import httpx
import pytest
from api.llm import anthropic_models
from api.llm.anthropic_models import (
    MAX_OUTPUT_TOKENS_CAP,
    discover,
    latest_per_line,
    refresh_anthropic_models,
    reset_discovery_state,
    to_model_config,
)
from api.llm.config import PROVIDER_ANTHROPIC, all_models, available_models

ADAPTIVE = {"thinking": {"supported": True, "types": {"adaptive": {"supported": True}, "enabled": {"supported": False}}}}
BUDGETED = {"thinking": {"supported": True, "types": {"adaptive": {"supported": False}, "enabled": {"supported": True}}}}


def model(model_id, line, created_at, capabilities=ADAPTIVE, lifecycle="active", **extra):
    return {
        "id": model_id,
        "display_name": model_id.replace("-", " ").title(),
        "line": line,
        "created_at": created_at,
        "lifecycle": lifecycle,
        "capabilities": capabilities,
        "max_input_tokens": 1_000_000,
        "max_tokens": 128_000,
        **extra,
    }


LISTING = [
    model("claude-haiku-5-5", "haiku", "2026-08-01T00:00:00Z"),
    model("claude-sonnet-5-5", "sonnet", "2026-07-01T00:00:00Z"),
    model("claude-sonnet-5-0", "sonnet", "2026-01-01T00:00:00Z"),
    model("claude-opus-5-5", "opus", "2026-06-01T00:00:00Z"),
    model("claude-opus-6-0-preview", "opus", "2026-09-01T00:00:00Z", lifecycle="deprecated"),
    model("claude-sonnet-4-5", "sonnet", "2025-09-29T00:00:00Z", capabilities=BUDGETED),
]


@pytest.fixture(autouse=True)
def clean_registry():
    reset_discovery_state()
    yield
    reset_discovery_state()


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "API_KEY_ANTHROPIC"
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

    monkeypatch.setattr(anthropic_models, "list_models", fake_list_models)
    return state


def anthropic_ids():
    return [m.id for m in all_models().values() if m.provider == PROVIDER_ANTHROPIC]


class TestLatestPerLine:
    def test_keeps_newest_active_model_of_each_family(self):
        names = {m["id"] for m in latest_per_line(LISTING)}
        assert names == {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}

    def test_model_without_line_is_its_own_family(self):
        listing = [{"id": "claude-x", "created_at": "2026-01-01T00:00:00Z"}]
        assert [m["id"] for m in latest_per_line(listing)] == ["claude-x"]


class TestToModelConfig:
    def test_adaptive_model(self):
        config = to_model_config(LISTING[1], "/k")
        assert config.id == "anthropic/claude-sonnet-5-5"
        assert config.name == "claude-sonnet-5-5"
        assert config.provider == PROVIDER_ANTHROPIC
        assert config.api_key_file == "/k"
        assert config.pretty_name == "Claude Sonnet 5 5"
        assert config.thinking_mode == "adaptive"
        assert config.supports_reasoning and config.retains_reasoning
        assert config.supports_temperature is False

    def test_budgeted_model(self):
        config = to_model_config(LISTING[-1], "/k")
        assert config.thinking_mode == "enabled"
        assert 1024 <= config.thinking_budget_tokens < config.max_tokens

    def test_model_without_thinking(self):
        config = to_model_config(model("claude-old", "old", "2024-01-01T00:00:00Z", capabilities={}), "/k")
        assert config.thinking_mode is None
        assert config.supports_reasoning is False

    def test_limits_are_capped(self):
        config = to_model_config(LISTING[1], "/k")
        assert config.max_tokens == MAX_OUTPUT_TOKENS_CAP
        assert config.context_window == anthropic_models.CONTEXT_CAP

    def test_effort_only_when_the_model_supports_it(self, monkeypatch):
        monkeypatch.setattr(anthropic_models, "EFFORT", "high")
        supported = model("a", "a", "", capabilities={**ADAPTIVE, "effort": {"high": {"supported": True}}})
        unsupported = model("b", "b", "", capabilities={**ADAPTIVE, "effort": {"high": {"supported": False}}})
        assert to_model_config(supported).effort == "high"
        assert to_model_config(unsupported).effort is None


class TestDiscover:
    def test_orders_by_family_preference(self, key_file, listing):
        names = [c.name for c in discover(str(key_file))]
        assert names == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]

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
        refresh_anthropic_models(key_file=str(key_file))
        assert anthropic_ids() == [
            "anthropic/claude-opus-5-5",
            "anthropic/claude-sonnet-5-5",
            "anthropic/claude-haiku-5-5",
        ]
        assert {m.id for m in available_models()} >= set(anthropic_ids())

    def test_within_ttl_does_not_refetch(self, key_file, listing):
        refresh_anthropic_models(key_file=str(key_file))
        refresh_anthropic_models(key_file=str(key_file))
        assert listing["calls"] == 1
        refresh_anthropic_models(force=True, key_file=str(key_file))
        assert listing["calls"] == 2

    def test_failure_keeps_the_previous_list(self, key_file, listing):
        refresh_anthropic_models(key_file=str(key_file))
        before = anthropic_ids()
        listing["error"] = httpx.ConnectError("down")
        refresh_anthropic_models(force=True, key_file=str(key_file))
        assert anthropic_ids() == before

    def test_failure_is_retried_before_the_ttl(self, key_file, listing, monkeypatch):
        listing["error"] = httpx.ConnectError("down")
        refresh_anthropic_models(key_file=str(key_file))
        assert anthropic_ids() == []

        listing["error"] = None
        monkeypatch.setattr(anthropic_models, "FAILURE_RETRY_S", 0.0)
        refresh_anthropic_models(key_file=str(key_file))
        assert len(anthropic_ids()) == 3

    def test_removing_the_key_removes_the_models(self, key_file, listing):
        refresh_anthropic_models(key_file=str(key_file))
        key_file.write_text("")
        refresh_anthropic_models(key_file=str(key_file))
        assert anthropic_ids() == []

    def test_changed_key_refetches_and_failure_drops_old_models(self, key_file, listing):
        refresh_anthropic_models(key_file=str(key_file))
        key_file.write_text("another-key")
        listing["error"] = httpx.HTTPStatusError(
            "401", request=httpx.Request("GET", "https://x"), response=httpx.Response(401)
        )
        refresh_anthropic_models(key_file=str(key_file))
        assert listing["keys"] == ["test-key-not-real", "another-key"]
        assert anthropic_ids() == []


def test_list_models_follows_pagination(monkeypatch):
    pages = {
        None: {"data": [{"id": "a"}], "has_more": True, "last_id": "a"},
        "a": {"data": [{"id": "b"}], "has_more": False, "last_id": "b"},
    }
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        return httpx.Response(200, json=pages[request.url.params.get("after_id")])

    real_client = httpx.Client
    monkeypatch.setattr(
        anthropic_models.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert [m["id"] for m in anthropic_models.list_models("test-key-not-real")] == ["a", "b"]
    assert seen_headers[0]["x-api-key"] == "test-key-not-real"
    assert seen_headers[0]["anthropic-version"] == anthropic_models.ANTHROPIC_VERSION
