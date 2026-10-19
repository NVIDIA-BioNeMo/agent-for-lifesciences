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

"""OpenShell sandbox pool for python_execute (``PYEXEC_MODE=openshell``).

Each chat session gets its own OpenShell sandbox running the pyexec service,
reached through the gateway's service routing. Sandboxes are expensive to
start (seconds, mostly Python imports), so a small pool of warm spares is kept
ready and a session claims one as soon as the browser asks for a session id;
by the time the first python_execute call arrives the sandbox is normally
already serving.

All state lives in Redis so several backend replicas share one pool:

``openshell:session:{session_id}``
    Name of the sandbox owned by the session. Expires after the idle TTL and
    is refreshed on every python_execute call.
``openshell:sandbox:{name}``
    JSON record ``{state, session_id, host_header, cold}``. ``state`` is
    ``starting``, ``ready`` or ``failed``. Records of claimed sandboxes share
    the session TTL; starting records expire so a replica that dies mid-create
    cannot leak a sandbox.
``openshell:spares``
    List of ready, unclaimed sandbox names.
``openshell:capacity``
    Sorted set of every sandbox counted against ``max_active``, scored by when
    its lease lapses (``+inf`` for a ready spare). Leases track the record TTLs.
    A sandbox is admitted only by ``RESERVE_SCRIPT``, which checks the count,
    takes the lease and writes the ``starting`` record in one atomic step, so
    concurrent sessions, refills and replicas cannot all pass the cap at once.

The reaper deletes any labelled sandbox in the gateway whose record has
expired, which covers idle sessions and orphans alike.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import ssl
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

KEY_PREFIX = "openshell"
SPARES_KEY = f"{KEY_PREFIX}:spares"
REFILL_LOCK_KEY = f"{KEY_PREFIX}:refill-lock"
CAPACITY_KEY = f"{KEY_PREFIX}:capacity"
# KEYS: capacity set, sandbox record. ARGV: now, max_active, lease expiry,
# sandbox name, record JSON, record TTL in seconds. Returns 1 if admitted.
RESERVE_SCRIPT = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then
  return 0
end
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
redis.call('SET', KEYS[2], ARGV[5], 'EX', ARGV[6])
return 1
"""
POOL_LABEL_KEY = "aevom.pool"
POOL_LABEL_VALUE = "pyexec"
SANDBOX_PORT = 8000
# Must outlast the gateway's idle connection pool (measured under 20 s). If
# uvicorn drops an idle connection first, the gateway reuses the dead connection
# and the request hangs until the caller times out. pyexec's
# PYEXEC_CONNECTION_CLOSE normally prevents reuse; this covers a gateway that
# ignores it.
SANDBOX_KEEP_ALIVE_S = 75
# SandboxPhase ERROR, STOPPED and COMPLETED (openshell.proto). Kept as ints so the
# backend imports without the generated protos.
TERMINAL_PHASES = frozenset({3, 7, 9})


def session_key(session_id: str) -> str:
    return f"{KEY_PREFIX}:session:{session_id}"


def sandbox_key(name: str) -> str:
    return f"{KEY_PREFIX}:sandbox:{name}"


def claim_lock_key(session_id: str) -> str:
    return f"{KEY_PREFIX}:claim-lock:{session_id}"


class SandboxUnavailableError(RuntimeError):
    """No sandbox could be provided for the session."""


def pyexec_mode() -> str:
    return os.environ.get("PYEXEC_MODE", "direct").strip().lower()


def openshell_enabled() -> bool:
    return pyexec_mode() == "openshell"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


@dataclass
class OpenShellSettings:
    gateway: str = "openshell-gateway:8090"
    # "host-header": connect to service_base_url and route with the Host header
    # (plaintext local gateway). "url": connect to the service URL the gateway
    # reports, for deployments where its wildcard domain resolves in-cluster.
    service_routing: str = "host-header"
    service_base_url: str = ""
    workspace: str = "default"
    image: str = "aevom-pyexec-sandbox:latest"
    cpu_limit: str = "1"
    memory_limit: str = "2Gi"
    pyexec_workers: int = 2
    spare_pool: int = 2
    max_active: int = 20
    idle_ttl_s: int = 1800
    ready_timeout_s: float = 60.0
    reap_interval_s: float = 15.0
    policy_path: Optional[str] = None
    tls_ca: Optional[str] = None
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.service_base_url:
            scheme = "https" if self.tls_ca or self.tls_cert else "http"
            self.service_base_url = f"{scheme}://{self.gateway}"

    @classmethod
    def from_env(cls) -> "OpenShellSettings":
        env = os.environ.get
        return cls(
            gateway=env("OPENSHELL_GATEWAY", cls.gateway),
            service_routing=env("OPENSHELL_SERVICE_ROUTING", cls.service_routing),
            service_base_url=env("OPENSHELL_SERVICE_BASE_URL", ""),
            workspace=env("OPENSHELL_WORKSPACE", cls.workspace),
            image=env("OPENSHELL_SANDBOX_IMAGE", cls.image),
            cpu_limit=env("OPENSHELL_SANDBOX_CPU", cls.cpu_limit),
            memory_limit=env("OPENSHELL_SANDBOX_MEMORY", cls.memory_limit),
            pyexec_workers=_env_int("OPENSHELL_PYEXEC_WORKERS", cls.pyexec_workers),
            spare_pool=_env_int("OPENSHELL_SPARE_POOL", cls.spare_pool),
            max_active=_env_int("OPENSHELL_MAX_ACTIVE", cls.max_active),
            idle_ttl_s=_env_int("OPENSHELL_IDLE_TTL_S", cls.idle_ttl_s),
            ready_timeout_s=_env_float("OPENSHELL_READY_TIMEOUT_S", cls.ready_timeout_s),
            reap_interval_s=_env_float("OPENSHELL_REAP_INTERVAL_S", cls.reap_interval_s),
            policy_path=env("OPENSHELL_POLICY_PATH") or None,
            tls_ca=env("OPENSHELL_TLS_CA") or None,
            tls_cert=env("OPENSHELL_TLS_CERT") or None,
            tls_key=env("OPENSHELL_TLS_KEY") or None,
        )

    def httpx_kwargs(self) -> dict[str, Any]:
        """TLS options for HTTP calls to sandbox services through the gateway."""
        # A context, not verify=<path> plus cert=: httpx 0.28 returns early for
        # a string verify and never loads cert, so the gateway sees no client
        # certificate and refuses the handshake.
        if not (self.tls_ca or self.tls_cert):
            return {}
        context = ssl.create_default_context(cafile=self.tls_ca)
        if self.tls_cert and self.tls_key:
            context.load_cert_chain(self.tls_cert, self.tls_key)
        return {"verify": context}


@dataclass(frozen=True)
class SandboxEndpoint:
    """Where to send pyexec requests for one session."""

    name: str
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    httpx_kwargs: dict[str, Any] = field(default_factory=dict)
    cold_start: bool = False
    wait_ms: float = 0.0


def _get_tracer():
    # api.telemetry.get_tracer() initialises Phoenix on first use; only borrow
    # a tracer the application has already set up.
    telemetry = sys.modules.get("api.telemetry")
    return getattr(telemetry, "_tracer", None) if telemetry else None


def _span(name: str):
    tracer = _get_tracer()
    return tracer.start_as_current_span(name) if tracer is not None else nullcontext()


def _set(span, key: str, value: Any) -> None:
    if span is not None:
        try:
            span.set_attribute(key, value)
        except Exception:
            pass


HealthProbe = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[bool]]


async def _http_health_probe(base_url: str, headers: dict[str, str], httpx_kwargs: dict[str, Any]) -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0, **httpx_kwargs) as client:
            response = await client.get(f"{base_url}/health", headers=headers)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


class OpenShellSandboxManager:
    """Pool of per-session OpenShell sandboxes, coordinated through Redis."""

    def __init__(
        self,
        settings: Optional[OpenShellSettings] = None,
        *,
        redis: Any = None,
        client: Any = None,
        health_probe: Optional[HealthProbe] = None,
        policy: Any = None,
    ) -> None:
        self.settings = settings or OpenShellSettings.from_env()
        self._redis = redis
        self._client = client
        self._health_probe = health_probe or _http_health_probe
        self._policy = policy
        self._tasks: set[asyncio.Task] = set()
        self._loop_task: Optional[asyncio.Task] = None

    # -- dependencies -----------------------------------------------------

    async def _r(self):
        if self._redis is None:
            from api.redis import get_redis

            self._redis = get_redis()
        return self._redis

    def _sdk(self):
        if self._client is None:
            from pathlib import Path

            from openshell import SandboxClient, TlsConfig

            s = self.settings
            tls = None
            if s.tls_ca or s.tls_cert:
                tls = TlsConfig(
                    ca_path=Path(s.tls_ca) if s.tls_ca else None,
                    cert_path=Path(s.tls_cert) if s.tls_cert else None,
                    key_path=Path(s.tls_key) if s.tls_key else None,
                )
            self._client = SandboxClient(s.gateway, timeout=30.0, tls=tls)
        return self._client

    def _policy_proto(self):
        if self._policy is None:
            from api.sandbox.openshell_policy import build_policy_proto, load_policy_document

            self._policy = build_policy_proto(load_policy_document(self.settings.policy_path))
        return self._policy

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Validate the policy, then keep the spare pool filled and reap idle sandboxes."""
        self._policy_proto()
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._maintenance_loop())

    async def stop(self) -> None:
        tasks = [t for t in (self._loop_task, *self._tasks) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._loop_task = None

    async def _maintenance_loop(self) -> None:
        while True:
            try:
                await self.reap()
                await self.refill()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("openshell pool maintenance failed: %s", exc)
            await asyncio.sleep(self.settings.reap_interval_s)

    # -- records ----------------------------------------------------------

    async def _get_record(self, name: str) -> Optional[dict[str, Any]]:
        raw = await (await self._r()).get(sandbox_key(name))
        return json.loads(raw) if raw else None

    async def _put_record(self, name: str, record: dict[str, Any], ttl: Optional[float]) -> None:
        redis = await self._r()
        if ttl:
            await redis.set(sandbox_key(name), json.dumps(record), ex=int(ttl))
        else:
            await redis.set(sandbox_key(name), json.dumps(record))

    async def _records(self) -> dict[str, dict[str, Any]]:
        redis = await self._r()
        prefix = sandbox_key("")
        records: dict[str, dict[str, Any]] = {}
        async for key in redis.scan_iter(match=f"{prefix}*"):
            raw = await redis.get(key)
            if raw:
                records[key[len(prefix):]] = json.loads(raw)
        return records

    def _starting_ttl(self) -> float:
        return self.settings.ready_timeout_s * 3

    # -- capacity ---------------------------------------------------------

    @staticmethod
    def _lease_expiry(ttl: Optional[float]) -> str:
        return "+inf" if ttl is None else f"{time.time() + ttl:.3f}"

    async def _reserve(self, name: str, record: dict[str, Any]) -> bool:
        """Take a capacity lease and write the ``starting`` record, if the pool has room."""
        ttl = max(1, math.ceil(self._starting_ttl()))
        admitted = await (await self._r()).eval(
            RESERVE_SCRIPT,
            2,
            CAPACITY_KEY,
            sandbox_key(name),
            f"{time.time():.3f}",
            self.settings.max_active,
            self._lease_expiry(ttl),
            name,
            json.dumps(record),
            ttl,
        )
        return bool(admitted)

    async def _extend(self, name: str, ttl: Optional[float]) -> None:
        """Move an existing lease to ``ttl`` from now, or make it open-ended."""
        await (await self._r()).zadd(CAPACITY_KEY, {name: self._lease_expiry(ttl)}, xx=True)

    async def _release(self, name: str) -> None:
        await (await self._r()).zrem(CAPACITY_KEY, name)

    # -- claim and lookup -------------------------------------------------

    async def claim(self, session_id: str) -> Optional[str]:
        """Bind a sandbox to the session, starting one if no spare is ready.

        Idempotent and cheap when the session already has a sandbox; safe to
        fire-and-forget from session creation. Returns the sandbox name, or
        ``None`` when the pool is at capacity.
        """
        redis = await self._r()
        existing = await redis.get(session_key(session_id))
        if existing:
            return existing
        if not await redis.set(claim_lock_key(session_id), "1", nx=True, ex=30):
            return None
        try:
            existing = await redis.get(session_key(session_id))
            if existing:
                return existing
            name = await self._take_spare(session_id)
            if name is None:
                name = await self._start_for_session(session_id)
            if name is not None:
                self._spawn(self.refill())
            return name
        finally:
            await redis.delete(claim_lock_key(session_id))

    async def _take_spare(self, session_id: str) -> Optional[str]:
        redis = await self._r()
        ttl = self.settings.idle_ttl_s
        while True:
            name = await redis.rpop(SPARES_KEY)
            if name is None:
                return None
            record = await self._get_record(name)
            if not record or record.get("state") != "ready" or record.get("session_id"):
                continue
            record["session_id"] = session_id
            record["cold"] = False
            await self._put_record(name, record, ttl)
            await self._extend(name, ttl)
            await redis.set(session_key(session_id), name, ex=ttl)
            logger.info("openshell: session %s claimed spare sandbox %s", session_id, name)
            return name

    async def _start_for_session(self, session_id: str) -> Optional[str]:
        name = self._new_name()
        if not await self._reserve(name, {"state": "starting", "session_id": session_id, "cold": True}):
            logger.warning(
                "openshell: pool at capacity (%d sandboxes); session %s has no sandbox",
                self.settings.max_active,
                session_id,
            )
            return None
        await (await self._r()).set(session_key(session_id), name, ex=self.settings.idle_ttl_s)
        logger.info("openshell: no spare ready; starting sandbox %s for session %s", name, session_id)
        self._spawn(self._provision(name))
        return name

    async def get_endpoint(self, session_id: str, timeout: Optional[float] = None) -> SandboxEndpoint:
        """Return the session's sandbox endpoint, waiting for it to become ready.

        Raises ``SandboxUnavailableError`` on capacity exhaustion, a failed
        start, or timeout. Callers must not fall back to an ungoverned path.
        """
        deadline_s = self.settings.ready_timeout_s if timeout is None else timeout
        started = time.monotonic()
        redis = await self._r()
        with _span("openshell.sandbox.claim") as span:
            _set(span, "session.id", session_id)
            while True:
                name = await redis.get(session_key(session_id)) or await self.claim(session_id)
                if name is None:
                    # Either another request for this session holds the claim
                    # lock, or the pool is full.
                    if await self.active_count() >= self.settings.max_active:
                        raise SandboxUnavailableError("all Python sandboxes are in use; try again shortly")
                    record = None
                else:
                    record = await self._get_record(name)
                    if record and record.get("state") == "ready":
                        break
                    if record is None or record.get("state") == "failed":
                        await redis.delete(session_key(session_id))
                        reason = (record or {}).get("error", "sandbox record expired")
                        raise SandboxUnavailableError(f"Python sandbox failed to start: {reason}")
                if time.monotonic() - started > deadline_s:
                    raise SandboxUnavailableError(f"Python sandbox was not ready after {deadline_s:.0f}s")
                _set(span, "sandbox.waited", True)
                await asyncio.sleep(0.25)

            ttl = self.settings.idle_ttl_s
            await redis.expire(session_key(session_id), ttl)
            await redis.expire(sandbox_key(name), ttl)
            await self._extend(name, ttl)
            wait_ms = (time.monotonic() - started) * 1000.0
            cold = bool(record.get("cold"))
            _set(span, "sandbox.name", name)
            _set(span, "sandbox.wait_ms", round(wait_ms, 1))
            _set(span, "sandbox.cold_start", cold)
            return self._endpoint(name, record, cold=cold, wait_ms=wait_ms)

    def _endpoint(self, name: str, record: dict[str, Any], *, cold: bool, wait_ms: float) -> SandboxEndpoint:
        s = self.settings
        if s.service_routing == "url" and record.get("service_url"):
            base_url, headers = record["service_url"].rstrip("/"), {}
        else:
            base_url, headers = s.service_base_url.rstrip("/"), {"Host": record["host_header"]}
        return SandboxEndpoint(
            name=name,
            base_url=base_url,
            headers=headers,
            httpx_kwargs=s.httpx_kwargs(),
            cold_start=cold,
            wait_ms=wait_ms,
        )

    # -- provisioning -----------------------------------------------------

    def _new_name(self) -> str:
        return f"pyexec-{uuid.uuid4().hex[:12]}"

    def _create_spec(self):
        from openshell._proto import openshell_pb2

        s = self.settings
        spec = openshell_pb2.SandboxSpec()
        spec.policy.CopyFrom(self._policy_proto())
        spec.template.image = s.image
        spec.template.resources.update({"limits": {"cpu": s.cpu_limit, "memory": s.memory_limit}})
        spec.command.extend(
            [
                "/usr/local/bin/uvicorn",
                "app.main:app",
                "--app-dir",
                "/app",
                "--host",
                "127.0.0.1",
                "--port",
                str(SANDBOX_PORT),
                "--workers",
                str(s.pyexec_workers),
                "--timeout-keep-alive",
                str(SANDBOX_KEEP_ALIVE_S),
            ]
        )
        return spec

    def _create_and_wait(self, name: str) -> str:
        """Create the sandbox, wait for the Ready phase, and return its service URL."""
        from openshell import ServiceExposure

        client = self._sdk()
        created = client.create(
            workspace=self.settings.workspace,
            name=name,
            spec=self._create_spec(),
            labels={POOL_LABEL_KEY: POOL_LABEL_VALUE},
            service_exposures=[ServiceExposure(target_port=SANDBOX_PORT)],
        )
        # Only the create response carries service URLs; "" is the unnamed service.
        service_url = dict(created.service_urls).get("", "")
        client.wait_ready(name, workspace=self.settings.workspace, timeout_seconds=self.settings.ready_timeout_s)
        return service_url

    async def _provision(self, name: str) -> None:
        """Create a sandbox and mark it ready once pyexec answers /health."""
        started = time.monotonic()
        with _span("openshell.sandbox.wait_ready") as span:
            _set(span, "sandbox.name", name)
            try:
                service_url = await asyncio.to_thread(self._create_and_wait, name)
                if not service_url:
                    raise SandboxUnavailableError("gateway returned no service URL")
                record = await self._get_record(name) or {"state": "starting", "session_id": None}
                record["service_url"] = service_url
                record["host_header"] = urlparse(service_url).netloc
                endpoint = self._endpoint(name, record, cold=False, wait_ms=0.0)
                while not await self._health_probe(endpoint.base_url, endpoint.headers, endpoint.httpx_kwargs):
                    if time.monotonic() - started > self.settings.ready_timeout_s:
                        raise SandboxUnavailableError("pyexec did not answer /health in time")
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("openshell: sandbox %s failed to start: %s", name, exc)
                _set(span, "error", str(exc)[:200])
                record = await self._get_record(name) or {}
                record.update(state="failed", error=str(exc)[:200])
                await self._put_record(name, record, self._starting_ttl())
                await self._release(name)
                await self._delete_sandbox(name)
                return

            elapsed_ms = (time.monotonic() - started) * 1000.0
            _set(span, "sandbox.ready_ms", round(elapsed_ms, 1))
            # A session may have claimed this sandbox while it was starting.
            current = await self._get_record(name) or {}
            record["session_id"] = current.get("session_id", record.get("session_id"))
            record["state"] = "ready"
            if record.get("session_id"):
                await self._put_record(name, record, self.settings.idle_ttl_s)
                await self._extend(name, self.settings.idle_ttl_s)
            else:
                await self._put_record(name, record, None)
                await self._extend(name, None)
                await (await self._r()).lpush(SPARES_KEY, name)
            logger.info("openshell: sandbox %s ready in %.0f ms", name, elapsed_ms)

    async def _delete_sandbox(self, name: str) -> None:
        try:
            await asyncio.to_thread(
                self._sdk().delete, name, workspace=self.settings.workspace, allow_missing=True
            )
        except Exception as exc:
            logger.warning("openshell: deleting sandbox %s failed: %s", name, exc)

    # -- pool maintenance -------------------------------------------------

    async def active_count(self) -> int:
        """Sandboxes holding a capacity lease: starting, ready spares and claimed ones."""
        redis = await self._r()
        await redis.zremrangebyscore(CAPACITY_KEY, "-inf", f"{time.time():.3f}")
        return await redis.zcard(CAPACITY_KEY)

    async def refill(self) -> int:
        """Start sandboxes until the spare pool target is met. Returns how many were started.

        The lock only keeps replicas from topping up the same shortfall twice;
        the cap is enforced by ``_reserve``.
        """
        redis = await self._r()
        if not await redis.set(REFILL_LOCK_KEY, "1", nx=True, ex=30):
            return 0
        try:
            records = await self._records()
            unclaimed = sum(1 for r in records.values() if r.get("state") != "failed" and not r.get("session_id"))
            started = 0
            for _ in range(max(0, self.settings.spare_pool - unclaimed)):
                name = self._new_name()
                if not await self._reserve(name, {"state": "starting", "session_id": None}):
                    break
                self._spawn(self._provision(name))
                started += 1
            return started
        finally:
            await redis.delete(REFILL_LOCK_KEY)

    async def reap(self) -> list[str]:
        """Delete sandboxes whose record expired or that died; drop records whose sandbox is gone."""
        redis = await self._r()
        refs = await asyncio.to_thread(
            self._sdk().list_all,
            workspace=self.settings.workspace,
            label_selector=f"{POOL_LABEL_KEY}={POOL_LABEL_VALUE}",
        )
        in_gateway = {ref.name for ref in refs}
        dead = {ref.name for ref in refs if getattr(ref, "phase", None) in TERMINAL_PHASES}
        # Leases before records: a lease and its record are written together,
        # so a lease read here without a record means the record is gone.
        leased = await redis.zrange(CAPACITY_KEY, 0, -1)
        records = await self._records()
        reaped = []
        for name in sorted(in_gateway):
            record = records.get(name)
            if record is not None and (name not in dead or record.get("state") == "starting"):
                continue
            logger.info("openshell: reaping idle, orphaned or dead sandbox %s", name)
            await self._delete_sandbox(name)
            await self._release(name)
            in_gateway.discard(name)
            reaped.append(name)
        for name, record in records.items():
            if name in in_gateway or record.get("state") == "starting":
                continue
            await redis.delete(sandbox_key(name))
            await redis.lrem(SPARES_KEY, 0, name)
            await self._release(name)
            session_id = record.get("session_id")
            if session_id and await redis.get(session_key(session_id)) == name:
                await redis.delete(session_key(session_id))
        for name in leased:
            if name not in records:
                await self._release(name)
        return reaped


_manager: Optional[OpenShellSandboxManager] = None


def get_openshell_manager() -> OpenShellSandboxManager:
    global _manager
    if _manager is None:
        _manager = OpenShellSandboxManager()
    return _manager


def set_openshell_manager(manager: Optional[OpenShellSandboxManager]) -> None:
    global _manager
    _manager = manager
