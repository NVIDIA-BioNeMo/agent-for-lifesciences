# OpenShell Governance for `python_execute`

`python_execute` runs model-generated Python. With `PYEXEC_MODE=openshell` that
code runs inside an [OpenShell](https://github.com/NVIDIA/OpenShell) sandbox, one
per chat session, which confines what the code can reach on the network, write
on disk and do as a process. With `PYEXEC_MODE=direct` the backend calls the
shared `pyexec` service exactly as before.

Deployments use OpenShell by default: `make up`, `make dev` and their GPU and
quick variants, and the Helm chart (`pyexec.mode: openshell`), production
included. The backend's own default is still `direct`, so anything that does not
set the variable keeps the shared service. That includes training, the eval
stack (`docker-compose.eval.yml`) and `make test`.

RestrictedPython in `services/pyexec` stays in place in both modes. It is the
first gate and blocks `socket`, `urllib.request`, `open(` and the rest before
the OS sees them. OpenShell is the layer that still holds when that gate has a
hole.

## Architecture

```
backend ──claim/wait──▶ OpenShell gateway (gRPC) ──creates──▶ sandbox pod/container
   │                                                          ├─ supervisor (proxy, OPA, Landlock, seccomp)
   └──POST /execute (Host: default--<name>.openshell.localhost)──▶ └─ workload: pyexec uvicorn on 127.0.0.1:8000
                                                                       │
                       aevom client callbacks ◀── L7 proxy ◀────────────┘
```

- **Sandbox image.** `services/pyexec/Dockerfile.openshell` builds
  `aevom-pyexec-sandbox`, which is the pyexec image with the `sandbox` user the
  supervisor drops to. The workload is the same pyexec FastAPI app, bound to
  loopback and reached through an OpenShell service exposure.
- **Manager.** `backend/api/sandbox/openshell_manager.py` keeps state in Redis,
  so every backend replica sees the same pool:
  - `openshell:session:{sid}` maps a session to its sandbox;
  - `openshell:sandbox:{name}` holds the record (`starting`, `ready`, `failed`);
  - `openshell:spares` is the warm pool.
- **Pre-claiming.** Creating a session, or the first chat turn, claims a sandbox
  in the background. A spare is taken if one is ready. Otherwise a sandbox is
  started for the session, while respecting `OPENSHELL_MAX_ACTIVE`. The pool is
  refilled afterwards. By the time the model calls `python_execute`, the sandbox
  is normally ready.
- **Capacity.** `openshell:capacity` is a sorted set holding one lease per
  sandbox that counts against `OPENSHELL_MAX_ACTIVE`, scored by when the lease
  lapses. Leases follow the record TTLs, and a ready spare's lease never lapses.
  Session claims and pool refills both admit a sandbox through a single Lua
  script, which checks the count, takes the lease and writes the `starting`
  record in one step. Concurrent sessions, refills and backend replicas
  therefore cannot overshoot the cap between checking it and starting a sandbox.
  A failed start or a reaped sandbox gives its lease back.
- **Reaping.** Session and sandbox keys expire after `OPENSHELL_IDLE_TTL_S`, and
  every tool call refreshes them. A loop running every
  `OPENSHELL_REAP_INTERVAL_S` deletes any gateway sandbox whose record has
  expired or never existed, and drops any record whose sandbox has gone.
- **Failing closed.** If no sandbox can be had (the gateway is down, the cap is
  reached, or startup times out), the tool returns `Python sandbox unavailable:
  …` and does **not** fall back to the shared pyexec service.

## Policy

`deploy/openshell/pyexec-policy.yaml` is sent with every `CreateSandbox`. The
backend substitutes `${OPENSHELL_BACKEND_HOST}` and `${OPENSHELL_BACKEND_PORT}`
and validates the policy at startup, so a broken policy stops the backend from
booting rather than failing the first call.

| Layer | Allowed | Denied |
| --- | --- | --- |
| Network | backend `/api/sandbox/{data/**,store,selection,visualize,probe-grid,constrained-pose}`; PubChem PUG-REST; RCSB chemical search | everything else, including postgres, redis, rdkit, vina, phoenix, the cloud metadata address, and other backend routes |
| Filesystem | read: `/usr /lib /lib64 /bin /etc /opt /app /proc`; write: `/tmp /sandbox` | writes to code and libraries; `/etc/shadow` |
| Process | runs as `sandbox`, under seccomp, and locally a cap of 512 PIDs | setuid to root; egress from `curl` or any binary other than python |

The 512-PID cap (the fork-bomb guard) is `sandbox_pids_limit` in the local
Docker gateway's `deploy/openshell/gateway.toml`. Nothing in this repository sets
one for Kubernetes, so in production the limit is whatever the OpenShell chart
or the node's kubelet (`podPidsLimit`) enforces.

Network rules are bound to the `python3*` binary, and are enforced at L7 by the
supervisor's proxy. A raw socket to the backend lands in the proxy as well. A
wrong `Host` gets `request_authority_mismatch`, a route not listed gets
`policy_denied`, and non-HTTP bytes get the connection closed.

To add an endpoint, add a `network_policies` entry with the narrowest method
and path rules that work, then rerun `make test-openshell`. Policy applies at
create time, so recycle the pool (`make openshell-clean`, then restart the
backend) for the change to take effect.

When sandboxed code hits a denial, `python_execute` appends a short note to the
error saying the sandbox blocked network or filesystem access, so the model
stops retrying a path that cannot work.

## Local use (docker compose)

```bash
make up                 # or make dev; both build aevom-pyexec-sandbox as needed
make test-openshell     # escape suite against the running stack
make openshell-clean    # remove sandboxes, the gateway and its keys
make dev PYEXEC_MODE=direct   # the shared pyexec service instead
```

Every start target adds `docker-compose.openshell.yml` unless
`PYEXEC_MODE=direct` is given. The overlay adds the gateway (plain HTTP on
loopback port 8090) and a one-shot `openshell-keys` service that mints the
supervisor JWT keys, and it switches the backend to `PYEXEC_MODE=openshell`.
The gateway reaches the host Docker daemon through its socket, which is
root-equivalent on the host. The local gateway also has TLS and user auth turned
off, so never reuse `deploy/openshell/gateway.toml` for a shared gateway.

The sandbox image is built through the production compose file even for
`make dev`, because only that file tags `aevom-pyexec:latest`. Sandboxes run the
code baked into the image rather than a live-reload mount, so any change under
`services/pyexec/app/` rebuilds the image on the next `make dev`.
`make openshell-images` forces a rebuild. Sandboxes already running keep the old
image: run `make openshell-clean` and restart the backend to replace them.

Tuning, all set as backend environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENSHELL_SPARE_POOL` | 2 | warm sandboxes kept ready |
| `OPENSHELL_MAX_ACTIVE` | 20 (compose: 8) | cap on sandboxes, spares included |
| `OPENSHELL_IDLE_TTL_S` | 1800 | idle time before a session's sandbox is reaped |
| `OPENSHELL_PYEXEC_WORKERS` | 2 | uvicorn workers per sandbox |
| `OPENSHELL_SANDBOX_CPU` / `_MEMORY` | `1` / `2Gi` | per-sandbox limits |
| `OPENSHELL_READY_TIMEOUT_S` | 60 | how long a tool call waits for a starting sandbox |

## Production (Helm)

`pyexec.mode: openshell` is the chart's default, and production inherits it. The
chart then drops the pyexec Deployment and HPA, and points the api at the
gateway with mTLS. It also renders a NetworkPolicy that admits traffic to
non-api pods only from within the namespace. On a cluster without the
prerequisites below, set `pyexec.mode: direct`; otherwise `python_execute` fails
closed. Switching an existing release to `openshell` deletes its pyexec
Deployment, which holds no data.

Sandbox nodes need Linux 6.2 or newer. OpenShell 0.1.2's sandbox requires
Landlock ABI v3, which added truncation control, and exits at startup on older
kernels with `sandbox self-protection requires Landlock ABI v3 or newer`.
Setting `landlock.compatibility: best_effort` in the policy does not get past
this check. The gateway itself runs on older nodes, so the failure shows up
only when the first sandbox starts, as a workload pod in `Error`.

NVIDIA's production cluster runs Ubuntu 22.04 with the HWE 6.8 kernel (Landlock
ABI v4); its GA 5.15 kernel has ABI v1 and cannot run sandboxes. The escape
suite passes against that cluster's gateway.

The gateway authenticates the api by its client certificate. Chart 0.1.2
authorizes API callers only through OIDC bearer tokens or
`server.auth.allowUnauthenticatedUsers`; with neither, it answers
`UNAUTHENTICATED: missing authorization header`. The install script sets
`allowUnauthenticatedUsers`, so mutual TLS is the whole boundary: the handshake
refuses callers without a certificate from OpenShell's CA, and anyone holding
`openshell-client-tls` can drive every sandbox. On Kubernetes that Secret is
mounted only in the api and in each sandbox's supervisor pod, never in the
workload pod where sandboxed code runs. The escape suite's `tcp_gateway` and
`read_credentials` probes check that sandboxed code can neither reach the
gateway nor find a key.

The Kubernetes driver runs the workload as UID and GID 10001, so the sandbox
image's `sandbox` user has that identity. With any other owner, the
`workspace-init` container cannot read `/sandbox` and the sandbox never starts.

Prerequisites the chart does not install. `deploy/openshell/install-cluster.sh`
installs all four (dry run by default, `--apply` to install), and warns about
nodes older than Linux 6.2:

1. The OpenShell Helm chart at 0.1.2, matching the pinned `openshell` SDK, plus
   the Agent Sandbox controller it depends on, installed in
   `openshell.sandboxNamespace` (default `openshell`). Leave
   `pkiInitJob.enabled=true` (the default), which creates the client certificate
   described below. Set `server.telemetryEnabled=false` unless you want
   OpenShell's anonymous usage telemetry, and `server.enableUserNamespaces=true`
   on nodes that support it.
2. A CNI that enforces NetworkPolicy. Without one, the NetworkPolicy is
   accepted but does nothing.
3. Read access for the CI deploy identity to that certificate. Create a Role
   and RoleBinding in `openshell` that grant `get` on `secrets`, limited to
   `resourceNames: [openshell-client-tls]`.
4. A GitLab registry pull secret in the `openshell` namespace, listed in the
   OpenShell chart's `server.sandboxImagePullSecrets`. Sandbox pods run there,
   not in the Aevom namespace, so without it they cannot pull `pyexec-sandbox`.

CI builds and pushes `pyexec-sandbox` (job `build-pyexec-sandbox`, from
`services/pyexec/Dockerfile.openshell`) next to the other images.
`openshell.sandboxImage` overrides the default `<registry>/pyexec-sandbox:<tag>`.

### The client certificate

The gateway requires mutual TLS (`server.tls.enableMtls: true` in the OpenShell
chart). On install, the chart's `pkiInitJob` hook generates a self-signed CA and
writes the client certificate to the Secret `openshell-client-tls` in its own
namespace, with keys `ca.crt`, `tls.crt` and `tls.key`. The hook skips if the
Secret already exists, so the certificate changes only when OpenShell is
reinstalled, or every year if the chart runs with `certManager.enabled=true`.

A pod cannot mount a Secret from another namespace. The `deploy-production` job
therefore copies the Secret into the Aevom namespace, under the name in
`openshell.clientTlsSecret`, on every deploy. It also passes a checksum of the
Secret as `openshell.clientTlsChecksum`, which becomes an api pod annotation.
The api builds its gateway client once at startup, so a new certificate takes
effect only when the pods restart, and the changed annotation restarts them.
After a certificate change between deploys, rerun the deploy job.

This is the same certificate the sandbox supervisors use, not a separate
identity for the api. The supervisor's copy is staged under
`/.openshell/supervisor/`, which the sandbox's Landlock policy does not grant to
the workload, and the network policy does not let sandboxed code reach the
gateway in any case.

The direct-mode pyexec pod is hardened as well: it runs non-root with a
read-only root filesystem, drops all capabilities, has no service-account token,
and gets a `/tmp` emptyDir. Setting `pyexec.hostUsers: false` adds user
namespaces where the cluster supports them.

## Observability

- **Backend traces (Phoenix).** The span `openshell.sandbox.claim` covers
  claiming a sandbox, and `openshell.sandbox.wait_ready` covers its startup. The
  `tool.python_execute` span carries `sandbox.name`, `sandbox.wait_ms` and
  `sandbox.cold_start`. A nonzero `wait_ms` means pre-claiming did not hide the
  startup.
- **Denials (OCSF).** Each sandbox's supervisor writes an OCSF line for every
  network decision, naming the binary, the destination and the rule that
  matched. For example: `NET:OPEN [MED] DENIED /usr/local/bin/python3.12 ->
  host:443 [policy:-]`. Locally, these are in the sandbox containers' logs:

  ```bash
  for c in $(docker ps -q --filter label=openshell.ai/managed-by=openshell); do
    docker logs "$c" 2>&1 | grep OCSF | grep DENIED
  done
  ```

  The gateway logs its own `ocsf` lines. `DENIED 127.0.0.1:8000
  [policy:sandbox_service_relay] [reason:service endpoint unreachable]` is the
  backend's health poll arriving before uvicorn is up, which is expected while a
  sandbox starts. In production, enable `server.ocsfLog.enabled` on the
  OpenShell chart and ship the JSONL file to your log store.
- **Gateway traces.** The OpenShell chart can export the gateway's own spans
  with `server.otlp.endpoint`. It speaks OTLP over gRPC only, so point it at
  Phoenix on 4317. The Aevom NetworkPolicy then needs an ingress exception for
  the `openshell` namespace on that port.

## Measurements

Measured locally on the Docker driver:

| | |
| --- | --- |
| spare ready from create | ~9–11 s |
| memory per sandbox | ~550 MiB (2 workers) |
| first `python_execute` in a session | ~1.2 s (module warm-up) |
| later calls | ~20 ms overhead |

## Escape suite

`scripts/openshell_escape_suite.py`, run with `make test-openshell`, creates two
throwaway sandboxes with the production policy. It probes every denial in the
table above, checks that the allowed callbacks and external APIs still work,
and checks that the two sandboxes cannot see each other's files. External APIs
are reported as `WARN` rather than `FAIL`, so an outage at PubChem does not
fail the suite. Run it after any policy change.

## Known issues

- **Leftover staging containers.** Deleting a sandbox while it is still
  provisioning can leave a `supervisor-stage` container behind, or make Docker
  report a 409 "volume in use". The reaper retries, and `make openshell-clean`
  removes everything labelled `openshell.ai/managed-by=openshell`.
- **A stack started before `make test` had its own project name.**
  `docker-compose.test.yml` now runs as project `aevom-test`, so `make test`
  leaves the app stack's network alone. A stack that lost the backend's pinned
  `192.168.249.250` to an older `make test` needs `make down` and then `make up`
  (or `make dev`). If compose complains about endpoints on the old network,
  remove the empty `aevom_default` network first.
- **A restart during provisioning delays the pool.** Sandboxes still starting
  when the backend stops keep their `starting` records until those expire,
  after three times `OPENSHELL_READY_TIMEOUT_S`. The reaper then deletes the
  sandboxes and refill replaces them.
- **Stale relay connections (fixed).** The gateway pools idle connections into a
  sandbox for under 20 s, but when uvicorn closes one, the close is not passed
  back through the relay. Before the fix, a call 5–20 s after the previous one
  went down a dead connection and hung until the backend's timeout
  (`timeout_seconds + 10`). The sandbox image sets `PYEXEC_CONNECTION_CLOSE=1`,
  so every response carries `Connection: close` and the gateway opens a new
  relay per request. As a backstop, uvicorn's `--timeout-keep-alive` is set to
  75 s, longer than the gateway's pool. In the gateway log, one
  `relay opened successfully` line per `/execute` means the fix is in effect.
- **`docker exec` output.** Importing `api.sandbox` inside the backend container
  detaches the process from `docker exec`'s stdio. The code still runs but its
  output is lost. This predates OpenShell. Write results to a file, as
  `scripts/test-openshell.sh` does.
