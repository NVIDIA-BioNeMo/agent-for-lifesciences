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

"""Tests for the OpenShell sandbox pool behind python_execute.

The gateway and Redis are faked; what is covered is the pool's bookkeeping:
spares are claimed before anything is started, a session keeps its sandbox,
the pool never grows past its cap, failures reach the caller instead of being
retried silently, and the reaper cleans up both directions of drift between
Redis and the gateway.
"""

import asyncio
import datetime
import fnmatch
import http.server
import ssl
import threading
from types import SimpleNamespace

import pytest

from api.sandbox.openshell_manager import (
    CAPACITY_KEY,
    RESERVE_SCRIPT,
    SPARES_KEY,
    OpenShellSandboxManager,
    OpenShellSettings,
    SandboxUnavailableError,
    sandbox_key,
    session_key,
)


def _mtls_files(directory):
    """Write a CA, a localhost server pair and a client pair; return their paths."""
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    def leaf(name, usage, san):
        key = ec.generate_private_key(ec.SECP256R1())
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
            )
        )
        if san:
            builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(san)]), critical=False)
        cert_path, key_path = directory / f"{name}.crt", directory / f"{name}.key"
        cert_path.write_bytes(builder.sign(ca_key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        return str(cert_path), str(key_path)

    ca_path = directory / "ca.crt"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    server = leaf("server", ExtendedKeyUsageOID.SERVER_AUTH, "localhost")
    client = leaf("client", ExtendedKeyUsageOID.CLIENT_AUTH, None)
    return str(ca_path), server, client


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class FakeRedis:
    """The subset of redis.asyncio (decode_responses=True) the manager uses.

    ``get`` and ``set`` yield to the event loop, as a network round trip would,
    so concurrent callers interleave between commands the way they do against a
    real server.
    """

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.ttl: dict[str, float] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def eval(self, script, numkeys, *keys_and_args):
        # Lua scripts run atomically on the server: no yield in here.
        assert script == RESERVE_SCRIPT
        capacity, record_key = keys_and_args[:numkeys]
        now, max_active, expiry, name, record, ttl = keys_and_args[numkeys:]
        leases = self.zsets.setdefault(capacity, {})
        for member in [m for m, score in leases.items() if score <= float(now)]:
            del leases[member]
        if len(leases) >= int(max_active):
            return 0
        leases[name] = float(expiry)
        self.kv[record_key] = record
        self.ttl[record_key] = int(ttl)
        return 1

    async def zadd(self, key, mapping, xx=False):
        zset = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            if xx and member not in zset:
                continue
            zset[member] = float(score)

    async def zrem(self, key, *members):
        for member in members:
            self.zsets.get(key, {}).pop(member, None)

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

    async def zrange(self, key, start, end):
        return sorted(self.zsets.get(key, {}), key=self.zsets[key].get) if key in self.zsets else []

    async def zremrangebyscore(self, key, low, high):
        zset = self.zsets.get(key, {})
        for member in [m for m, score in zset.items() if float(low) <= score <= float(high)]:
            del zset[member]

    async def get(self, key):
        await asyncio.sleep(0)
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=False):
        await asyncio.sleep(0)
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        if ex:
            self.ttl[key] = ex
        else:
            self.ttl.pop(key, None)
        return True

    async def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)
            self.lists.pop(key, None)
            self.ttl.pop(key, None)

    async def expire(self, key, seconds):
        if key in self.kv:
            self.ttl[key] = seconds

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def rpop(self, key):
        items = self.lists.get(key) or []
        return items.pop() if items else None

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def lrem(self, key, count, value):
        self.lists[key] = [v for v in self.lists.get(key, []) if v != value]

    async def scan_iter(self, match):
        for key in list(self.kv):
            if fnmatch.fnmatch(key, match):
                yield key


class FakeGateway:
    """Synchronous stand-in for openshell.SandboxClient."""

    def __init__(self, fail=False):
        self.sandboxes: set[str] = set()
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.phases: dict[str, int] = {}
        self.fail = fail

    def create(self, *, workspace, name, spec, labels, service_exposures):
        if self.fail:
            raise RuntimeError("image not found")
        self.sandboxes.add(name)
        self.created.append(name)
        return SimpleNamespace(service_urls={"": f"http://default--{name}.openshell.localhost:8090"})

    def wait_ready(self, name, *, workspace, timeout_seconds):
        return SimpleNamespace(name=name)

    def delete(self, name, *, workspace, allow_missing):
        self.sandboxes.discard(name)
        self.deleted.append(name)

    def list_all(self, *, workspace, label_selector):
        return [SimpleNamespace(name=n, phase=self.phases.get(n, 2)) for n in sorted(self.sandboxes)]


class Manager(OpenShellSandboxManager):
    def _create_spec(self):
        return "spec"

    def _create_and_wait(self, name):
        created = self._sdk().create(
            workspace=self.settings.workspace, name=name, spec=self._create_spec(), labels={}, service_exposures=[]
        )
        return created.service_urls[""]


async def healthy(base_url, headers, httpx_kwargs):
    return True


async def never_healthy(base_url, headers, httpx_kwargs):
    return False


def make(redis=None, gateway=None, probe=healthy, **settings):
    settings.setdefault("spare_pool", 1)
    settings.setdefault("max_active", 3)
    settings.setdefault("ready_timeout_s", 2.0)
    manager = Manager(
        OpenShellSettings(gateway="gw:8090", **settings),
        redis=redis or FakeRedis(),
        client=gateway or FakeGateway(),
        health_probe=probe,
        policy=object(),
    )
    return manager


async def settle(manager):
    while manager._tasks:
        await asyncio.gather(*list(manager._tasks), return_exceptions=True)


class TestSparePool:
    @pytest.mark.asyncio
    async def test_refill_starts_sandboxes_until_the_target_is_met(self):
        manager = make(spare_pool=2)
        assert await manager.refill() == 2
        await settle(manager)
        assert await manager._redis.llen(SPARES_KEY) == 2
        assert await manager.refill() == 0

    @pytest.mark.asyncio
    async def test_a_session_claims_a_ready_spare_without_starting_one(self):
        manager = make(spare_pool=1)
        await manager.refill()
        await settle(manager)
        spare = manager._redis.lists[SPARES_KEY][0]

        endpoint = await manager.get_endpoint("s1")

        assert endpoint.name == spare
        assert endpoint.cold_start is False
        assert endpoint.base_url == "http://gw:8090"
        assert endpoint.headers == {"Host": f"default--{spare}.openshell.localhost:8090"}

    @pytest.mark.asyncio
    async def test_claiming_a_spare_replaces_it(self):
        manager = make(spare_pool=1)
        await manager.refill()
        await settle(manager)
        await manager.get_endpoint("s1")
        await settle(manager)
        assert await manager._redis.llen(SPARES_KEY) == 1
        assert len(manager._client.created) == 2

    @pytest.mark.asyncio
    async def test_the_spare_pool_respects_the_cap(self):
        manager = make(spare_pool=5, max_active=2)
        assert await manager.refill() == 2


class TestClaim:
    @pytest.mark.asyncio
    async def test_without_a_spare_the_session_gets_a_cold_sandbox(self):
        manager = make(spare_pool=0)
        endpoint = await manager.get_endpoint("s1")
        assert endpoint.cold_start is True
        assert manager._client.created == [endpoint.name]

    @pytest.mark.asyncio
    async def test_a_session_keeps_its_sandbox(self):
        manager = make(spare_pool=0)
        first = await manager.get_endpoint("s1")
        second = await manager.get_endpoint("s1")
        assert first.name == second.name
        assert len(manager._client.created) == 1

    @pytest.mark.asyncio
    async def test_sessions_do_not_share_sandboxes(self):
        manager = make(spare_pool=0)
        a = await manager.get_endpoint("a")
        b = await manager.get_endpoint("b")
        assert a.name != b.name

    @pytest.mark.asyncio
    async def test_claim_is_idempotent(self):
        manager = make(spare_pool=0)
        name = await manager.claim("s1")
        assert await manager.claim("s1") == name

    @pytest.mark.asyncio
    async def test_use_refreshes_the_idle_ttl(self):
        manager = make(spare_pool=0, idle_ttl_s=900)
        endpoint = await manager.get_endpoint("s1")
        redis = manager._redis
        redis.ttl[session_key("s1")] = 1
        await manager.get_endpoint("s1")
        assert redis.ttl[session_key("s1")] == 900
        assert redis.ttl[sandbox_key(endpoint.name)] == 900

    @pytest.mark.asyncio
    async def test_a_full_pool_fails_instead_of_overcommitting(self):
        manager = make(spare_pool=0, max_active=1)
        await manager.get_endpoint("a")
        with pytest.raises(SandboxUnavailableError, match="in use"):
            await manager.get_endpoint("b")
        assert len(manager._client.created) == 1


class TestCapacity:
    @pytest.mark.asyncio
    async def test_concurrent_sessions_cannot_exceed_the_cap(self):
        manager = make(spare_pool=0, max_active=2)
        names = await asyncio.gather(*(manager.claim(f"s{i}") for i in range(6)))
        await settle(manager)
        assert len([n for n in names if n]) == 2
        assert len(manager._client.created) == 2

    @pytest.mark.asyncio
    async def test_claims_racing_a_refill_cannot_exceed_the_cap(self):
        manager = make(spare_pool=2, max_active=3)
        await asyncio.gather(manager.refill(), *(manager.claim(f"s{i}") for i in range(4)))
        await settle(manager)
        assert len(manager._client.created) == 3
        assert await manager.active_count() == 3

    @pytest.mark.asyncio
    async def test_leases_follow_the_sandbox_lifecycle(self):
        manager = make(spare_pool=1, idle_ttl_s=900)
        await manager.refill()
        await settle(manager)
        spare = manager._redis.lists[SPARES_KEY][0]
        assert manager._redis.zsets[CAPACITY_KEY][spare] == float("inf")

        await manager.get_endpoint("s1")
        assert manager._redis.zsets[CAPACITY_KEY][spare] < float("inf")

    @pytest.mark.asyncio
    async def test_a_failed_start_frees_its_slot(self):
        gateway = FakeGateway(fail=True)
        manager = make(spare_pool=0, max_active=1, gateway=gateway)
        with pytest.raises(SandboxUnavailableError, match="image not found"):
            await manager.get_endpoint("a")
        assert await manager.active_count() == 0
        gateway.fail = False
        assert (await manager.get_endpoint("b")).name in gateway.sandboxes

    @pytest.mark.asyncio
    async def test_the_reaper_frees_the_slots_it_reaps(self):
        manager = make(spare_pool=0, max_active=1)
        endpoint = await manager.get_endpoint("a")
        await manager._redis.delete(session_key("a"), sandbox_key(endpoint.name))

        assert await manager.reap() == [endpoint.name]
        assert await manager.active_count() == 0
        assert (await manager.get_endpoint("b")).name != endpoint.name

    @pytest.mark.asyncio
    async def test_the_reaper_drops_a_lease_whose_record_is_gone(self):
        manager = make(spare_pool=0, max_active=1)
        endpoint = await manager.get_endpoint("a")
        manager._client.sandboxes.clear()
        await manager._redis.delete(sandbox_key(endpoint.name))

        await manager.reap()
        assert await manager.active_count() == 0

    @pytest.mark.asyncio
    async def test_an_expired_lease_frees_its_slot(self):
        manager = make(spare_pool=0, max_active=1)
        endpoint = await manager.get_endpoint("a")
        # The record and the lease share a TTL; simulate both lapsing.
        await manager._redis.delete(sandbox_key(endpoint.name))
        manager._redis.zsets[CAPACITY_KEY][endpoint.name] = 0.0

        assert (await manager.get_endpoint("b")).name != endpoint.name


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_failed_start_reaches_the_caller(self):
        manager = make(spare_pool=0, gateway=FakeGateway(fail=True))
        with pytest.raises(SandboxUnavailableError, match="image not found"):
            await manager.get_endpoint("s1")

    @pytest.mark.asyncio
    async def test_after_a_failure_the_next_call_starts_over(self):
        gateway = FakeGateway(fail=True)
        manager = make(spare_pool=0, gateway=gateway)
        with pytest.raises(SandboxUnavailableError):
            await manager.get_endpoint("s1")
        gateway.fail = False
        endpoint = await manager.get_endpoint("s1")
        assert endpoint.name in gateway.sandboxes

    @pytest.mark.asyncio
    async def test_a_sandbox_that_never_serves_is_deleted(self):
        manager = make(spare_pool=0, probe=never_healthy, ready_timeout_s=0.5)
        with pytest.raises(SandboxUnavailableError):
            await manager.get_endpoint("s1", timeout=5)
        await settle(manager)
        assert manager._client.sandboxes == set()


class TestReaper:
    @pytest.mark.asyncio
    async def test_an_idle_session_sandbox_is_deleted(self):
        manager = make(spare_pool=0)
        endpoint = await manager.get_endpoint("s1")
        # Both keys share the idle TTL; simulate it lapsing.
        await manager._redis.delete(session_key("s1"), sandbox_key(endpoint.name))

        assert await manager.reap() == [endpoint.name]
        assert manager._client.sandboxes == set()

    @pytest.mark.asyncio
    async def test_an_active_sandbox_is_kept(self):
        manager = make(spare_pool=1)
        await manager.refill()
        await settle(manager)
        await manager.get_endpoint("s1")
        await settle(manager)
        assert await manager.reap() == []
        assert len(manager._client.sandboxes) == 2

    @pytest.mark.asyncio
    async def test_a_dead_spare_is_replaced(self):
        manager = make(spare_pool=1)
        await manager.refill()
        await settle(manager)
        spare = manager._redis.lists[SPARES_KEY][0]
        manager._client.phases[spare] = 3  # SANDBOX_PHASE_ERROR

        assert await manager.reap() == [spare]
        assert await manager._get_record(spare) is None
        assert await manager._redis.llen(SPARES_KEY) == 0
        assert await manager.refill() == 1

    @pytest.mark.asyncio
    async def test_records_for_vanished_sandboxes_are_dropped(self):
        manager = make(spare_pool=1)
        await manager.refill()
        await settle(manager)
        spare = manager._redis.lists[SPARES_KEY][0]
        manager._client.sandboxes.clear()

        await manager.reap()

        assert await manager._redis.get(sandbox_key(spare)) is None
        assert await manager._redis.llen(SPARES_KEY) == 0

    @pytest.mark.asyncio
    async def test_a_session_whose_sandbox_vanished_gets_a_new_one(self):
        manager = make(spare_pool=0)
        first = await manager.get_endpoint("s1")
        manager._client.sandboxes.clear()
        await manager.reap()
        second = await manager.get_endpoint("s1")
        assert second.name != first.name


class TestSettings:
    def test_tls_switches_the_service_scheme(self):
        s = OpenShellSettings(gateway="gw:443", tls_ca="/ca.pem", tls_cert="/c.pem", tls_key="/k.pem")
        assert s.service_base_url == "https://gw:443"

    def test_no_tls_means_no_httpx_options(self):
        assert OpenShellSettings(gateway="gw:80").httpx_kwargs() == {}

    def test_httpx_presents_the_client_certificate(self, tmp_path):
        # The gateway refuses handshakes without a client certificate. httpx
        # 0.28 silently drops cert= when verify is a path, so exercise a real
        # mTLS server rather than inspect the kwargs.
        httpx = pytest.importorskip("httpx")
        ca, server, client = _mtls_files(tmp_path)

        server_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=ca)
        server_context.load_cert_chain(*server)
        server_context.verify_mode = ssl.CERT_REQUIRED
        httpd = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
        httpd.socket = server_context.wrap_socket(httpd.socket, server_side=True)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            s = OpenShellSettings(gateway="localhost", tls_ca=ca, tls_cert=client[0], tls_key=client[1])
            with httpx.Client(timeout=5, **s.httpx_kwargs()) as client_session:
                response = client_session.get(f"https://localhost:{httpd.server_port}/health")
            assert response.status_code == 200
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENSHELL_GATEWAY", "openshell:8090")
        monkeypatch.setenv("OPENSHELL_SPARE_POOL", "4")
        monkeypatch.setenv("OPENSHELL_MAX_ACTIVE", "40")
        s = OpenShellSettings.from_env()
        assert (s.gateway, s.spare_pool, s.max_active) == ("openshell:8090", 4, 40)
        assert s.service_base_url == "http://openshell:8090"

    def test_uvicorn_keeps_idle_connections_longer_than_the_gateway(self):
        # The gateway pools idle connections for under 20s. If uvicorn closes
        # one first, the gateway's next request on it hangs.
        from openshell._proto import openshell_pb2

        manager = OpenShellSandboxManager(
            OpenShellSettings(gateway="gw:8090"),
            redis=FakeRedis(),
            client=FakeGateway(),
            health_probe=healthy,
            policy=type(openshell_pb2.SandboxSpec().policy)(),
        )
        command = list(manager._create_spec().command)
        keep_alive = int(command[command.index("--timeout-keep-alive") + 1])
        assert keep_alive >= 60

    def test_url_routing_uses_the_reported_service_url(self):
        manager = make(service_routing="url")
        record = {"service_url": "https://default--x.sandboxes.example/", "host_header": "ignored"}
        endpoint = manager._endpoint("x", record, cold=False, wait_ms=0)
        assert endpoint.base_url == "https://default--x.sandboxes.example"
        assert endpoint.headers == {}
