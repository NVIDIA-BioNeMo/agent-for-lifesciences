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

"""Unit tests for ``api.tools.base``.

Focused on the post-2026-05-27 ``tool_service.http`` OTel span that
ContainerTool._call_service emits per HTTP attempt, with per-service-host
attribution attributes so the agent-loop profiling pipeline can answer
"which container service node is taking the load?"
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from api.tools.base import ContainerTool, ToolResult


@pytest.fixture
def otel_exporter():
    """Install an in-memory OTel SDK and return the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]

    # api.telemetry caches a tracer; reset so it grabs the new provider.
    import api.telemetry as _tel

    _tel._tracer = otel_trace.get_tracer("aevom", tracer_provider=provider)
    _tel._tracer_provider = provider
    _tel._initialized = True

    yield exporter
    exporter.clear()


class _FakeTool(ContainerTool):
    """Minimal ContainerTool subclass for testing _call_service.

    `service_url` is a comma-separated list to exercise the multi-host
    random-pick path that the per-host attribution attributes rely on.
    """

    def __init__(self, url: str = "http://svc-node-0:8001,http://svc-node-1:8001"):
        super().__init__()
        self._url = url

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def service_url(self) -> str:
        return self._url

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": "fake"},
        }

    async def execute(self, arguments: dict) -> ToolResult:  # pragma: no cover
        return ToolResult.text("ok")


class TestToolServiceHttpSpan:
    def test_span_emitted_with_per_host_attrs(self, otel_exporter, monkeypatch):
        """Successful call: tool_service.http span has tool.service_host /
        tool.service_port / tool.service_base_url set, status OK."""
        tool = _FakeTool(url="http://pyexec-svc-0:8001")

        # Mock the httpx AsyncClient to avoid real network I/O.
        class _MockResponse:
            status_code = 200
            headers = {"content-length": "12"}

            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        class _MockClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                # Capture the outbound headers so we can assert on the
                # x-aevom-client-send-ns wire-time header below.
                _MockClient.last_post_headers = headers
                return _MockResponse()

            async def get(self, url, params=None, headers=None):
                return _MockResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _MockClient)

        result = asyncio.new_event_loop().run_until_complete(
            tool._call_service("/run", json_data={"x": 1})
        )
        assert result == {"ok": True}

        otel_exporter.force_flush()
        spans = otel_exporter.get_finished_spans()
        svc = next(s for s in spans if s.name == "tool_service.http")
        attrs = dict(svc.attributes)
        assert attrs["tool.name"] == "fake_tool"
        assert attrs["tool.service_base_url"] == "http://pyexec-svc-0:8001"
        assert attrs["tool.service_host"] == "pyexec-svc-0"
        assert attrs["tool.service_port"] == 8001
        assert attrs["tool.service_scheme"] == "http"
        assert attrs["tool.endpoint"] == "/run"
        assert attrs["tool.attempt"] == 1
        assert attrs["tool.http.method"] == "POST"
        assert attrs["tool.http.status_code"] == 200
        # Client latency attribute is recorded on success spans so the
        # post-2026-05-28 backpressure analysis scripts can read it
        # directly.
        assert "tool.http.client_latency_s" in attrs
        assert attrs["tool.http.client_latency_s"] >= 0.0
        # And the wire-time header is injected so the server can compute
        # kernel-queue wait for [AEVOM-PYEXEC-BACKPRESSURE] warnings.
        sent_hdrs = getattr(_MockClient, "last_post_headers", None) or {}
        assert "x-aevom-client-send-ns" in sent_hdrs
        assert int(sent_hdrs["x-aevom-client-send-ns"]) > 0

    def test_default_port_inferred(self, otel_exporter, monkeypatch):
        """If the URL omits the port, fall back to scheme defaults."""
        tool = _FakeTool(url="https://example-svc/")  # implicit 443
        import httpx

        class _MR:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class _MC:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return _MR()

            async def get(self, url, params=None, headers=None):
                return _MR()

        monkeypatch.setattr(httpx, "AsyncClient", _MC)
        asyncio.new_event_loop().run_until_complete(
            tool._call_service("/x", json_data={})
        )
        otel_exporter.force_flush()
        svc = next(
            s
            for s in otel_exporter.get_finished_spans()
            if s.name == "tool_service.http"
        )
        attrs = dict(svc.attributes)
        assert attrs["tool.service_scheme"] == "https"
        assert attrs["tool.service_port"] == 443

    def test_pool_random_pick_records_host(self, otel_exporter, monkeypatch):
        """With multi-URL pool, the chosen host is reflected on the span."""
        tool = _FakeTool(url="http://node-a:8001,http://node-b:8001")
        import random
        import httpx

        # Deterministic pick.
        monkeypatch.setattr(random, "choice", lambda xs: xs[1])

        class _MR:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class _MC:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return _MR()

            async def get(self, url, params=None, headers=None):
                return _MR()

        monkeypatch.setattr(httpx, "AsyncClient", _MC)
        asyncio.new_event_loop().run_until_complete(
            tool._call_service("/x", json_data={})
        )
        otel_exporter.force_flush()
        svc = next(
            s
            for s in otel_exporter.get_finished_spans()
            if s.name == "tool_service.http"
        )
        attrs = dict(svc.attributes)
        assert attrs["tool.service_host"] == "node-b"

    def test_retry_sets_attempt_attr(self, otel_exporter, monkeypatch):
        """One transient failure + retry success -> two spans, attempts 1 & 2."""
        tool = _FakeTool(url="http://svc:8001")
        import httpx

        call_count = {"n": 0}

        class _OkResp:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class _MC:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise httpx.ReadTimeout("timeout")
                return _OkResp()

            async def get(self, url, params=None, headers=None):
                return _OkResp()

        monkeypatch.setattr(httpx, "AsyncClient", _MC)
        # Make asyncio.sleep return immediately
        import asyncio as _asyncio

        async def _no_sleep(s):
            return None

        monkeypatch.setattr(_asyncio, "sleep", _no_sleep)

        asyncio.new_event_loop().run_until_complete(
            tool._call_service("/x", json_data={}, retries=1)
        )
        otel_exporter.force_flush()
        svcs = [
            s
            for s in otel_exporter.get_finished_spans()
            if s.name == "tool_service.http"
        ]
        assert len(svcs) == 2
        attempts = sorted(dict(s.attributes)["tool.attempt"] for s in svcs)
        assert attempts == [1, 2]
        # First span has an error_type, second has the status_code.
        first = next(
            s
            for s in svcs
            if dict(s.attributes)["tool.attempt"] == 1
        )
        assert dict(first.attributes).get("tool.http.error_type") == "ReadTimeout"
        # httpx timeouts usually carry no message, so the status names the
        # exception and the budget rather than relying on str(e).
        assert first.status.description.startswith("ReadTimeout after ")
        assert "(timeout 30s)" in first.status.description
        # Both attempts now record client_latency_s on the span -- the
        # error path used to drop it pre-2026-05-28 (v0527c post-mortem).
        assert "tool.http.client_latency_s" in dict(first.attributes)
        second = next(
            s
            for s in svcs
            if dict(s.attributes)["tool.attempt"] == 2
        )
        assert "tool.http.client_latency_s" in dict(second.attributes)


class TestBackpressureWarning:
    """The post-2026-05-28 [AEVOM-TOOL-BACKPRESSURE] alarm.

    Designed (per v0527c post-mortem) to make sandbox saturation
    impossible to miss in the driver/service logs without dropping back
    to Phoenix queries. We deliberately do NOT throttle / cap; the
    warning is the signal to scale workers.
    """

    def test_slow_attempt_emits_loud_warning(self, otel_exporter, monkeypatch, caplog):
        """A success that took longer than the warn threshold logs the
        loud, structured ``[AEVOM-TOOL-BACKPRESSURE]`` line."""
        import time

        tool = _FakeTool(url="http://svc:8001")
        # Force the warn threshold to 0s so any non-zero latency trips
        # it without us having to actually sleep in the test.
        monkeypatch.setenv("AEVOM_TOOL_BACKPRESSURE_WARN_S", "0.0")

        class _MR:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class _MC:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                # Burn a tiny but nonzero amount of perf_counter time so
                # the latency probe sees > 0.0 in the handler. Using
                # busy-wait keeps the test fast and deterministic.
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 0.001:
                    pass
                return _MR()

            async def get(self, url, params=None, headers=None):
                return _MR()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _MC)

        import logging

        with caplog.at_level(logging.WARNING, logger="api.tools.base"):
            asyncio.new_event_loop().run_until_complete(
                tool._call_service("/x", json_data={})
            )

        backpressure_msgs = [
            r.getMessage()
            for r in caplog.records
            if "[AEVOM-TOOL-BACKPRESSURE]" in r.getMessage()
        ]
        assert backpressure_msgs, (
            "expected [AEVOM-TOOL-BACKPRESSURE] warning in logs; "
            f"got {[r.getMessage() for r in caplog.records]}"
        )
        # Warning must include the tool name + endpoint so it's grep-able
        # alongside the per-host attribution from OTel.
        msg = backpressure_msgs[0]
        assert "fake_tool" in msg
        assert "/x" in msg

    def test_fast_attempt_emits_no_warning(self, otel_exporter, monkeypatch, caplog):
        """A fast success below the warn threshold does NOT log the
        backpressure line (we don't want to spam the steady-state log)."""
        tool = _FakeTool(url="http://svc:8001")
        # Generous threshold so the fast path won't trip it.
        monkeypatch.setenv("AEVOM_TOOL_BACKPRESSURE_WARN_S", "60.0")

        class _MR:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class _MC:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return _MR()

            async def get(self, url, params=None, headers=None):
                return _MR()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _MC)

        import logging

        with caplog.at_level(logging.WARNING, logger="api.tools.base"):
            asyncio.new_event_loop().run_until_complete(
                tool._call_service("/x", json_data={})
            )

        backpressure_msgs = [
            r.getMessage()
            for r in caplog.records
            if "[AEVOM-TOOL-BACKPRESSURE]" in r.getMessage()
        ]
        assert not backpressure_msgs, (
            f"expected no backpressure warnings on fast path; "
            f"got {backpressure_msgs}"
        )

    def test_timeout_emits_backpressure_warning(self, otel_exporter, monkeypatch, caplog):
        """A httpx timeout always emits the backpressure warning; the
        most reliable saturation signal is when an attempt's budget is
        exhausted."""
        import httpx

        tool = _FakeTool(url="http://svc:8001")

        class _MC:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                raise httpx.WriteTimeout("simulated saturation")

            async def get(self, url, params=None, headers=None):
                raise httpx.WriteTimeout("simulated saturation")

        monkeypatch.setattr(httpx, "AsyncClient", _MC)

        import logging

        with caplog.at_level(logging.WARNING, logger="api.tools.base"):
            with pytest.raises(httpx.WriteTimeout):
                asyncio.new_event_loop().run_until_complete(
                    tool._call_service("/x", json_data={})
                )

        backpressure_msgs = [
            r.getMessage()
            for r in caplog.records
            if "[AEVOM-TOOL-BACKPRESSURE]" in r.getMessage()
        ]
        assert backpressure_msgs, (
            "expected at least one [AEVOM-TOOL-BACKPRESSURE] warning on "
            "WriteTimeout; got "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        assert any("WriteTimeout" in m for m in backpressure_msgs)


class TestNvidiaApiKeyFile:
    """``NvidiaApiTool._get_api_key`` must refuse an unusable key file.

    An empty file used to be read as the empty string and sent as ``Bearer ``.
    health.api.nvidia.com answers a missing credential with HTTP 500 and an axum
    "Missing request extension: Authorization<Bearer>" body rather than a 401, so
    the tool reported what looked like an upstream outage. That is how an unset
    ``$NVIDIA_API_KEY`` in the eval jobs cost three tasks per pipeline while the
    LLM -- which reads the other key, and whose reader already raised on empty --
    kept working and made the run look like a scoring regression.
    """

    def _tool(self):
        from api.tools.msa_search import MSASearchTool

        return MSASearchTool()

    def test_an_empty_key_file_is_rejected(self, tmp_path, monkeypatch):
        key_file = tmp_path / "nvidia_api_key"
        key_file.write_text("")
        monkeypatch.setenv("NVIDIA_API_KEY_FILE", str(key_file))

        with pytest.raises(RuntimeError, match="API key file is empty"):
            self._tool()._get_api_key()

    def test_a_whitespace_only_key_file_is_rejected(self, tmp_path, monkeypatch):
        """``write_key_file`` writes no newline, but a hand-made file often has one."""
        key_file = tmp_path / "nvidia_api_key"
        key_file.write_text("\n  \n")
        monkeypatch.setenv("NVIDIA_API_KEY_FILE", str(key_file))

        with pytest.raises(RuntimeError, match="API key file is empty"):
            self._tool()._get_api_key()

    def test_a_missing_key_file_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY_FILE", str(tmp_path / "absent"))

        with pytest.raises(RuntimeError, match="API key file not found"):
            self._tool()._get_api_key()

    def test_a_populated_key_file_is_returned_stripped(self, tmp_path, monkeypatch):
        key_file = tmp_path / "nvidia_api_key"
        key_file.write_text("nvapi-abc123\n")
        monkeypatch.setenv("NVIDIA_API_KEY_FILE", str(key_file))

        assert self._tool()._get_api_key() == "nvapi-abc123"
