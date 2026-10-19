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

"""Registering a self-hosted NIM: discovery, adapters, and the confirm gate.

The live responses the probe tests assert against were captured from a real
RFD4-Proteina NIM on localhost:8000, including the detail that its readiness
endpoint answers a bare ``true`` rather than the object its own OpenAPI
document describes.

The confirm gate is the same contract MCP has, for the same reason: a
registered NIM contributes tools pointed at a host Aevom has never spoken to,
and may re-point tools the agent already had. Neither happens until a human
says so.
"""

import httpx
import pytest

from api.nim.adapters import all_adapters, get_adapter
from api.nim.adapters.base import NIMEndpoint
from api.nim.client import NIMClient, NIMConnectionError, NIMInfo
from api.nim.overrides import clear_session, get_endpoint_override
from api.nim.session_store import NIMServerConfig, NIMSessionStore

# Exactly what localhost:8000 returned, trimmed to the fields that are read.
LIVE_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "BakerLab/rfd4-proteina",
            "object": "model",
            "owned_by": "ipd",
            "version": "1.0.0",
            "description": "RFD4-Proteina generative protein design model",
        }
    ],
}

LIVE_TASKS = {
    "tasks": [
        {"name": "unconditional", "summary": "..."},
        {"name": "motif_scaffolding", "summary": "..."},
        {"name": "inverse_folding", "summary": "..."},
    ],
    "metric_tags": {
        "design": {"available": True, "reason": None},
        "hotspot_conditioning": {"available": True, "reason": None},
        # This deployment has no oracle, so refolding cannot be computed.
        "unconditional_refolding": {"available": False, "reason": "no oracle"},
        "developability": {"available": False, "reason": "no PyRosetta"},
    },
}

LIVE_OPENAPI = {
    "paths": {
        "/v1/health/ready": {"get": {}},
        "/v1/metadata": {"get": {}},
        "/biology/ipd/rfd4-proteina/generative": {"post": {}},
    }
}

LIVE_ROUTES = {
    "/v1/models": LIVE_MODELS,
    "/v1/health/ready": True,
    "/v1/metadata": {"version": "1.0.0"},
    "/v1/tasks": LIVE_TASKS,
    "/openapi.json": LIVE_OPENAPI,
}


class FakeNIM:
    """Serves a fixed routing table, and records the headers it was sent.

    A route maps to a JSON body, a bare status code, or a ready-made
    ``httpx.Response`` for anything more specific.
    """

    def __init__(self, routes):
        self.routes = dict(routes)
        self.seen_headers: list = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen_headers.append(request.headers)
        for path, payload in self.routes.items():
            if request.url.path.endswith(path):
                if isinstance(payload, httpx.Response):
                    return payload
                # bool is a subclass of int, and `True` is a legitimate body
                # here -- the readiness route returns exactly that -- so the
                # status-code shorthand must exclude it.
                if isinstance(payload, int) and not isinstance(payload, bool):
                    return httpx.Response(payload)
                return httpx.Response(200, json=payload)
        return httpx.Response(404)


@pytest.fixture
def fake_http(monkeypatch):
    """Point the probe at a routing table instead of the network."""
    real_client = httpx.AsyncClient

    def _install(routes):
        fake = FakeNIM(routes)

        def factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return real_client(*args, transport=httpx.MockTransport(fake), **kwargs)

        monkeypatch.setattr("api.nim.client.httpx.AsyncClient", factory)
        return fake

    return _install


@pytest.mark.asyncio
class TestDiscoveryProbe:
    async def test_reads_the_model_id_the_adapters_key_on(self, fake_http):
        fake_http(LIVE_ROUTES)
        info = await NIMClient("http://localhost:8000").probe()
        assert info.model_id == "BakerLab/rfd4-proteina"

    async def test_bare_true_readiness_body_is_understood(self, fake_http):
        # The NIM answers `true`, not the object its OpenAPI document shows.
        fake_http(LIVE_ROUTES)
        assert (await NIMClient("http://x").probe()).ready is True

    async def test_tasks_are_read_from_the_nim_not_hardcoded(self, fake_http):
        # The published docs claim a task named "generation"; the live enum
        # has no such entry. Reading the endpoint is the only safe source.
        fake_http(LIVE_ROUTES)
        info = await NIMClient("http://x").probe()
        assert "unconditional" in info.tasks
        assert "generation" not in info.tasks

    async def test_unavailable_metric_tags_are_recorded_as_unavailable(
        self, fake_http
    ):
        fake_http(LIVE_ROUTES)
        info = await NIMClient("http://x").probe()
        assert info.metric_tags["developability"] is False
        assert "developability" not in info.available_metric_tags
        assert "hotspot_conditioning" in info.available_metric_tags

    async def test_infrastructure_routes_are_not_mistaken_for_inference(
        self, fake_http
    ):
        fake_http(LIVE_ROUTES)
        info = await NIMClient("http://x").probe()
        assert info.inference_paths == ["/biology/ipd/rfd4-proteina/generative"]

    async def test_no_authorization_header_without_a_key(self, fake_http):
        # A self-hosted NIM has no bearer token, and some reject a bogus one.
        client = fake_http(LIVE_ROUTES)
        await NIMClient("http://x").probe()
        assert all("Authorization" not in h for h in client.seen_headers)

    async def test_key_is_sent_when_one_was_registered(self, fake_http):
        client = fake_http(LIVE_ROUTES)
        await NIMClient("http://x", api_key="secret").probe()
        assert all(
            h.get("Authorization") == "Bearer secret" for h in client.seen_headers
        )

    async def test_endpoint_without_a_model_list_is_rejected(self, fake_http):
        fake_http({"/v1/health/ready": True})
        with pytest.raises(NIMConnectionError):
            await NIMClient("http://not-a-nim").probe()

    async def test_optional_endpoints_may_all_be_missing(self, fake_http):
        # Only /v1/models is required; a leaner NIM still registers.
        fake_http({"/v1/models": LIVE_MODELS})
        info = await NIMClient("http://x").probe()
        assert info.model_id == "BakerLab/rfd4-proteina"
        assert info.tasks == []

    async def test_redirect_is_not_followed_and_names_the_target(self, fake_http):
        fake = fake_http({
            "/v1/models": httpx.Response(
                302, headers={"location": "http://169.254.169.254/v1/models"}
            ),
        })
        with pytest.raises(NIMConnectionError, match="169.254.169.254") as exc_info:
            await NIMClient("http://x").probe()
        assert "does not follow redirects" in str(exc_info.value)
        assert len(fake.seen_headers) == 1

    async def test_oversized_response_is_rejected_with_the_limit(
        self, fake_http, monkeypatch
    ):
        monkeypatch.setattr("api.bounded_http.MAX_RESPONSE_BYTES", 64)
        fake_http({"/v1/models": {"data": [{"id": "x" * 200}]}})
        with pytest.raises(NIMConnectionError, match="64 bytes limit"):
            await NIMClient("http://x").probe()


class TestAdapterMatching:
    def test_live_model_id_resolves_to_the_rfd4_adapter(self):
        assert get_adapter("BakerLab/rfd4-proteina").key == "rfd4_proteina"

    def test_match_is_case_insensitive(self):
        assert get_adapter("bakerlab/RFD4-Proteina").key == "rfd4_proteina"

    def test_retagged_local_model_still_matches(self):
        # Re-tagging on push is routine, and the owner prefix is what changes.
        assert get_adapter("mylab/rfd4-proteina:1.0.0").key == "rfd4_proteina"

    def test_unknown_model_has_no_adapter(self):
        # None is a supported state, not an error: the NIM still registers.
        assert get_adapter("acme/some-other-model") is None

    def test_adapter_keys_are_unique(self):
        keys = [adapter.key for adapter in all_adapters()]
        assert len(keys) == len(set(keys))


class TestAdapterContributions:
    @pytest.fixture
    def endpoint(self):
        return NIMEndpoint(
            server_id="srv1",
            name="Local RFD4",
            url="http://localhost:8000",
            api_key=None,
            info=NIMInfo(
                model_id="BakerLab/rfd4-proteina",
                metric_tags={
                    "design": True,
                    "hotspot_conditioning": True,
                    "interchain_interactions": False,
                },
                inference_paths=["/biology/ipd/rfd4-proteina/generative"],
            ),
        )

    def test_rfd4_contributes_the_binder_and_motif_tools(self, endpoint):
        tools = get_adapter("BakerLab/rfd4-proteina").build_tools(endpoint)
        assert [tool.name for tool in tools] == ["design_protein_binder", "design_motif_scaffold"]

    def test_a_nim_that_lists_its_tasks_is_believed(self, endpoint):
        endpoint.info.tasks = ["unconditional"]
        tools = get_adapter("BakerLab/rfd4-proteina").build_tools(endpoint)
        assert [tool.name for tool in tools] == ["design_protein_binder"]
        endpoint.info.tasks = ["unconditional", "motif_scaffolding"]
        tools = get_adapter("BakerLab/rfd4-proteina").build_tools(endpoint)
        assert len(tools) == 2

    def test_the_tool_is_bound_to_the_registered_endpoint(self, endpoint):
        tool = get_adapter("BakerLab/rfd4-proteina").build_tools(endpoint)[0]
        assert tool.api_endpoint == (
            "http://localhost:8000/biology/ipd/rfd4-proteina/generative"
        )

    def test_the_tool_only_asks_for_metrics_this_deployment_has(self, endpoint):
        # Requesting an unavailable tag is a request error, not a skipped
        # metric, so the filter has to happen before the call.
        tool = get_adapter("BakerLab/rfd4-proteina").build_tools(endpoint)[0]
        assert tool._metric_tags == ["design", "hotspot_conditioning"]

    def test_discovered_route_wins_over_the_hardcoded_one(self, endpoint):
        moved = NIMEndpoint(
            server_id="srv2",
            name="Moved",
            url="http://localhost:9000",
            api_key=None,
            info=NIMInfo(
                model_id="BakerLab/rfd4-proteina",
                inference_paths=["/v2/biology/ipd/rfd4-proteina/generative"],
            ),
        )
        tool = get_adapter("BakerLab/rfd4-proteina").build_tools(moved)[0]
        assert tool.api_endpoint.endswith("/v2/biology/ipd/rfd4-proteina/generative")

    def test_boltz2_repoints_the_builtin_rather_than_adding_a_tool(self):
        adapter = get_adapter("mit/boltz2")
        endpoint = NIMEndpoint(
            server_id="srv3",
            name="Local Boltz",
            url="http://gpu-box:8000",
            api_key=None,
            info=NIMInfo(model_id="mit/boltz2"),
        )
        assert adapter.build_tools(endpoint) == []
        assert adapter.builtin_endpoints(endpoint) == {
            "boltz2": "http://gpu-box:8000/biology/mit/boltz2/predict"
        }


@pytest.fixture
def store(monkeypatch):
    """A session store whose Redis round-trip is a plain dict."""
    instance = NIMSessionStore()
    saved: dict[str, dict[str, NIMServerConfig]] = {}

    async def _load(session_id):
        return dict(saved.get(session_id, {}))

    async def _save(session_id, configs):
        saved[session_id] = dict(configs)
        instance._refresh_overrides(session_id, configs)

    monkeypatch.setattr(instance, "_load", _load)
    monkeypatch.setattr(instance, "_save", _save)
    instance._saved = saved
    return instance


def register(store, session_id="s1", confirmed=False, model="BakerLab/rfd4-proteina"):
    config = NIMServerConfig(
        id="srv1",
        name="Local RFD4",
        url="http://localhost:8000",
        info=NIMInfo(
            model_id=model,
            metric_tags={"design": True},
            inference_paths=["/biology/ipd/rfd4-proteina/generative"],
        ),
        confirmed=confirmed,
    )
    store._saved.setdefault(session_id, {})[config.id] = config
    return config


@pytest.mark.asyncio
class TestConfirmGate:
    async def test_unconfirmed_nim_contributes_no_tools(self, store):
        register(store, confirmed=False)
        assert await store.get_tools("s1") == []

    async def test_unconfirmed_tool_is_not_dispatchable_by_name(self, store):
        # Hiding it from the schema list alone would leave it callable, which
        # is the gap this gate exists to close.
        register(store, confirmed=False)
        assert await store.get_tool_by_name("s1", "design_protein_binder") is None

    async def test_confirming_releases_the_tools(self, store):
        register(store, confirmed=False)
        assert await store.confirm_server("s1", "srv1") is True
        tools = await store.get_tools("s1")
        assert [tool.name for tool in tools] == ["design_protein_binder", "design_motif_scaffold"]

    async def test_confirming_is_idempotent(self, store):
        register(store, confirmed=True)
        assert await store.confirm_server("s1", "srv1") is True

    async def test_confirming_an_unknown_server_reports_failure(self, store):
        assert await store.confirm_server("s1", "nope") is False

    async def test_listing_shows_what_would_be_added_before_confirming(self, store):
        config = register(store, confirmed=False)
        response = config.to_response()
        assert response["tools"] == ["design_protein_binder", "design_motif_scaffold"]
        assert response["confirmed"] is False

    async def test_unknown_model_registers_with_no_adapter(self, store):
        config = register(store, confirmed=True, model="acme/mystery")
        response = config.to_response()
        assert response["adapter"] is None
        assert response["tools"] == []
        # Still listed, so the user learns the endpoint is reachable.
        assert response["model_id"] == "acme/mystery"
        assert await store.get_tools("s1") == []


@pytest.mark.asyncio
class TestUnconfirmedNimNotice:
    """What the model is told about a NIM it cannot use yet."""

    @pytest.fixture(autouse=True)
    def _use_store(self, store, monkeypatch):
        monkeypatch.setattr("api.nim.get_nim_session_store", lambda: store)

    async def test_an_unconfirmed_nim_is_reported_with_the_tools_it_would_add(self, store):
        from api.graph.state import _load_nim_notices

        register(store, confirmed=False)
        assert await _load_nim_notices("s1") == [
            {"name": "Local RFD4", "tools": ["design_protein_binder", "design_motif_scaffold"]}
        ]

    async def test_a_confirmed_nim_is_not_reported(self, store):
        from api.graph.state import _load_nim_notices

        register(store, confirmed=True)
        assert await _load_nim_notices("s1") == []

    async def test_confirming_clears_the_notice(self, store):
        from api.graph.state import _load_nim_notices

        register(store, confirmed=False)
        await store.confirm_server("s1", "srv1")
        assert await _load_nim_notices("s1") == []

    async def test_a_nim_with_no_adapter_is_not_reported(self, store):
        # Confirming it would add nothing, so asking for it would be a dead end.
        from api.graph.state import _load_nim_notices

        register(store, confirmed=False, model="acme/mystery")
        assert await _load_nim_notices("s1") == []

    async def test_another_sessions_nim_is_not_reported(self, store):
        from api.graph.state import _load_nim_notices

        register(store, session_id="other", confirmed=False)
        assert await _load_nim_notices("s1") == []

    async def test_an_unreadable_store_is_no_notice_not_a_failed_turn(self, monkeypatch):
        from api.graph.state import _load_nim_notices

        class Broken:
            async def get_servers(self, session_id):
                raise RuntimeError("redis down")

        monkeypatch.setattr("api.nim.get_nim_session_store", lambda: Broken())
        assert await _load_nim_notices("s1") == []


@pytest.mark.asyncio
class TestEndpointOverrides:
    @pytest.fixture(autouse=True)
    def _clean(self):
        clear_session("s1")
        yield
        clear_session("s1")

    def _boltz(self, store, confirmed):
        config = NIMServerConfig(
            id="srv-boltz",
            name="Local Boltz",
            url="http://gpu-box:8000",
            info=NIMInfo(model_id="mit/boltz2"),
            confirmed=confirmed,
        )
        store._saved.setdefault("s1", {})["srv-boltz"] = config
        return config

    async def test_confirmed_nim_repoints_the_builtin_tool(self, store):
        self._boltz(store, confirmed=True)
        await store.get_servers("s1")
        override = get_endpoint_override("s1", "boltz2")
        assert override.url == "http://gpu-box:8000/biology/mit/boltz2/predict"

    async def test_unconfirmed_nim_repoints_nothing(self, store):
        # Silently redirecting a tool the agent already had is a bigger change
        # than adding a withheld one, so it needs at least as much consent.
        self._boltz(store, confirmed=False)
        await store.get_servers("s1")
        assert get_endpoint_override("s1", "boltz2") is None

    async def test_removing_the_nim_restores_the_hosted_endpoint(self, store):
        self._boltz(store, confirmed=True)
        await store.get_servers("s1")
        await store.remove_server("s1", "srv-boltz")
        assert get_endpoint_override("s1", "boltz2") is None

    async def test_override_does_not_leak_between_sessions(self, store):
        self._boltz(store, confirmed=True)
        await store.get_servers("s1")
        assert get_endpoint_override("s2", "boltz2") is None
