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

"""Escape suite for python_execute under OpenShell (`make test-openshell`).

Runs raw Python inside real sandboxes through the gateway's exec API. That
bypasses RestrictedPython on purpose: what is tested is the layer underneath
it -- Landlock, seccomp, the network namespace and the egress policy -- for the
day an exploit gets past the interpreter.

Run inside the backend container (scripts/test-openshell.sh does this), which
has the SDK, reaches the gateway, and carries the policy environment. Targets
are passed in ESCAPE_TARGETS as JSON {"redis": "ip", "postgres": "ip", ...}.
Sandboxes are labelled aevom.pool=escape-suite so the pool reaper ignores them,
and are deleted on exit.
"""

import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

SANDBOX_DIR = Path("/app/api/sandbox")
LABELS = {"aevom.pool": "escape-suite"}

# Probes executed inside the sandbox. Each prints one JSON object.
PROBE = r'''
import json, os, socket, subprocess, sys, urllib.error, urllib.request

targets = json.loads(sys.argv[1])
backend = targets["backend"]
out = {}

def run(name, fn):
    try:
        out[name] = {"ok": True, "detail": str(fn())[:160]}
    except Exception as exc:
        out[name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"[:200]}

def tcp(host, port):
    socket.create_connection((host, port), timeout=3).close()
    return "connected"

def http(method, url, body=None):
    req = urllib.request.Request(url, data=body, method=method, headers={"content-type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return f"status={resp.status} policy={resp.headers.get('X-OpenShell-Policy')}"
    except urllib.error.HTTPError as exc:
        return f"status={exc.code} policy={exc.headers.get('X-OpenShell-Policy')} body={exc.read()[:80]!r}"

def raw(payload):
    host, port = backend.split(":")
    s = socket.create_connection((host, int(port)), timeout=5)
    s.sendall(payload)
    s.settimeout(5)
    data = b""
    try:
        while len(data) < 4096:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    s.close()
    return data.decode(errors="replace").replace("\r\n", " / ")

def readable_credentials():
    found = []
    for root in ("/etc", "/.openshell", "/run", "/var/run", "/tmp", "/sandbox", "/root", "/home"):
        for dirpath, _, files in os.walk(root, onerror=lambda e: None):
            for name in files:
                path = os.path.join(dirpath, name)
                if not os.path.isfile(path):
                    continue
                try:
                    with open(path, "rb") as fh:
                        head = fh.read(65536)
                except OSError:
                    continue
                if b"PRIVATE KEY-----" in head or (name == "token" and "serviceaccount" in dirpath):
                    found.append(path)
    return found

for svc in ("redis", "postgres", "rdkit"):
    if targets.get(svc):
        port = {"redis": 6379, "postgres": 5432, "rdkit": 8000}[svc]
        run(f"tcp_{svc}", lambda h=targets[svc], p=port: tcp(h, p))
# The gateway trusts any holder of its client certificate, so sandboxed code
# must neither reach it nor find a key it could present.
if targets.get("gateway"):
    gateway_host, gateway_port = targets["gateway"].rsplit(":", 1)
    run("tcp_gateway", lambda: tcp(gateway_host, int(gateway_port)))
run("read_credentials", readable_credentials)
# A socket to the backend opens -- it is a declared endpoint -- but lands in the
# policy proxy, so the bytes sent over it are still held to the route rules.
run("raw_backend_wrong_authority", lambda: raw(b"GET /api/health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"))
run("raw_backend_disallowed_route", lambda: raw(
    f"GET /api/health HTTP/1.1\r\nHost: {backend}\r\nConnection: close\r\n\r\n".encode()))
run("raw_backend_non_http", lambda: raw(b"\x16\x03\x01\x00\x05hello-not-http\r\n\r\n"))
run("raw_backend_smuggle", lambda: raw(
    f"POST /api/sandbox/store HTTP/1.1\r\nHost: {backend}\r\nContent-Length: 4\r\n"
    f"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\nGET /api/health HTTP/1.1\r\nHost: {backend}\r\n\r\n".encode()))
run("http_metadata", lambda: http("GET", "http://169.254.169.254/latest/meta-data/"))
run("https_example", lambda: http("GET", "https://example.com/"))
run("backend_health_route", lambda: http("GET", f"http://{backend}/api/health"))
run("backend_delete_route", lambda: http("DELETE", f"http://{backend}/api/sandbox/data/pdb_00000000"))
run("backend_store_route", lambda: http("POST", f"http://{backend}/api/sandbox/store", b"{}"))
run("backend_selection_route", lambda: http("POST", f"http://{backend}/api/sandbox/selection", b"{}"))
run("pubchem", lambda: http("GET", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON"))
run("rcsb_search", lambda: http("POST", "https://search.rcsb.org/rcsbsearch/v2/query",
    json.dumps({"query": {"type": "terminal", "service": "full_text", "parameters": {"value": "caffeine"}},
                "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": 1}}}).encode()))
run("write_app", lambda: open("/app/escape", "w").write("x"))
run("write_site_packages", lambda: open("/usr/local/lib/python3.12/site-packages/escape.py", "w").write("x"))
run("write_etc", lambda: open("/etc/escape", "w").write("x"))
run("read_shadow", lambda: open("/etc/shadow").read())
run("read_supervisor_env", lambda: open("/proc/1/environ", "rb").read())
# World-readable and world-writable respectively, and in neither policy list:
# only Landlock stands between these and the sandbox user.
run("landlock_read_unlisted", lambda: open("/var/lib/dpkg/status").read())
run("landlock_write_unlisted", lambda: open("/dev/shm/escape", "w").write("x"))
run("setuid_root", lambda: os.setuid(0))
run("curl_binary", lambda: subprocess.run(["curl", "-sS", "-m", "5", "https://example.com"], capture_output=True, check=True))
run("python_subprocess_egress", lambda: subprocess.run(
    [sys.executable, "-c", "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)"],
    capture_output=True, check=True))
run("write_tmp", lambda: open("/tmp/escape-ok", "w").write("ok"))
run("write_workdir", lambda: open("/sandbox/escape-ok", "w").write("ok"))
print(json.dumps(out))
'''

MARK = r'''
import os, sys
open("/sandbox/marker-" + sys.argv[1], "w").write("x")
open("/tmp/marker-" + sys.argv[1], "w").write("x")
print("marked")
'''

LOOK = r'''
import glob, json
print(json.dumps(sorted(glob.glob("/sandbox/marker-*") + glob.glob("/tmp/marker-*"))))
'''


def blocked(result):
    return not result["ok"] or "policy=" in result["detail"] and "policy=None" not in result["detail"]


def allowed_by_policy(result):
    # Reached the destination: a response that did not come from the proxy.
    return result["ok"] and "policy=None" in result["detail"]


def never_reaches_backend(result):
    return "server: uvicorn" not in result["detail"]


EXPECT = {
    "tcp_redis": blocked,
    "tcp_postgres": blocked,
    "tcp_rdkit": blocked,
    "tcp_gateway": blocked,
    "read_credentials": lambda r: r["ok"] and r["detail"] == "[]",
    "raw_backend_wrong_authority": lambda r: "request_authority_mismatch" in r["detail"],
    "raw_backend_disallowed_route": lambda r: "403 Forbidden" in r["detail"] and "X-OpenShell-Policy" in r["detail"],
    "raw_backend_non_http": never_reaches_backend,
    "raw_backend_smuggle": never_reaches_backend,
    "http_metadata": blocked,
    "https_example": blocked,
    "backend_health_route": blocked,
    "backend_delete_route": blocked,
    "write_app": blocked,
    "write_site_packages": blocked,
    "write_etc": blocked,
    "read_shadow": blocked,
    "read_supervisor_env": blocked,
    "landlock_read_unlisted": blocked,
    "landlock_write_unlisted": blocked,
    "setuid_root": blocked,
    "curl_binary": blocked,
    "python_subprocess_egress": blocked,
    "backend_store_route": lambda r: "policy=None" in r["detail"],
    "backend_selection_route": lambda r: "policy=None" in r["detail"],
    "pubchem": allowed_by_policy,
    "rcsb_search": allowed_by_policy,
    "write_tmp": lambda r: r["ok"],
    "write_workdir": lambda r: r["ok"],
}
# External services can be down for reasons that have nothing to do with the
# policy; a failure there is reported, not counted.
EXTERNAL = {"pubchem", "rcsb_search"}


def load(name):
    spec = importlib.util.spec_from_file_location(f"escape_{name}", SANDBOX_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from openshell import ServiceExposure  # noqa: F401  (fail fast if the SDK is missing)

    policy_mod = load("openshell_policy")
    manager_mod = load("openshell_manager")
    targets = json.loads(os.environ["ESCAPE_TARGETS"])
    settings = manager_mod.OpenShellSettings.from_env()
    policy = policy_mod.build_policy_proto(policy_mod.load_policy_document(settings.policy_path))
    manager = manager_mod.OpenShellSandboxManager(settings, redis=object(), policy=policy)
    client, workspace = manager._sdk(), settings.workspace

    names = [f"escape-{uuid.uuid4().hex[:8]}-{side}" for side in ("a", "b")]
    started = time.monotonic()
    try:
        for name in names:
            client.create(workspace=workspace, name=name, spec=manager._create_spec(), labels=LABELS)
        for name in names:
            client.wait_ready(name, workspace=workspace, timeout_seconds=120)
        print(f"two sandboxes ready in {time.monotonic() - started:.1f}s")

        def py(name, code, *args):
            res = client.exec(name, ["/usr/local/bin/python3", "-c", code, *args], workspace=workspace,
                              timeout_seconds=120)
            if res.exit_code != 0:
                raise RuntimeError(f"probe exited {res.exit_code}: {res.stderr[-400:]}")
            return res.stdout.strip().splitlines()[-1]

        results = json.loads(py(names[0], PROBE, json.dumps(targets)))
        py(names[0], MARK, "a")
        seen_from_b = json.loads(py(names[1], LOOK))
        results["cross_sandbox_files"] = {"ok": not seen_from_b, "detail": f"b sees {seen_from_b}"}
        EXPECT["cross_sandbox_files"] = lambda r: r["ok"]
    finally:
        for name in names:
            try:
                client.delete(name, workspace=workspace, allow_missing=True)
            except Exception as exc:
                print(f"warning: could not delete {name}: {exc}")

    failures, warnings = [], []
    width = max(map(len, EXPECT))
    for name, check in EXPECT.items():
        result = results.get(name)
        if result is None:
            continue
        passed = check(result)
        mark = "PASS" if passed else ("WARN" if name in EXTERNAL else "FAIL")
        print(f"{mark}  {name:<{width}}  {result['detail'][:110]}")
        if not passed:
            (warnings if name in EXTERNAL else failures).append(name)
    print()
    if warnings:
        print(f"{len(warnings)} external check(s) could not reach their service: {', '.join(warnings)}")
    print(f"{len(failures)} failure(s)" if failures else "all policy checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
