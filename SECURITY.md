## What is worth knowing before you deploy this

Aevom runs LLM-generated Python. Some properties of the current design are
deliberate but are not the defaults a hardened deployment would want, and they
should be understood rather than discovered:

- **The Python sandbox executes model-generated code.** Deployments give each
  chat session its own OpenShell sandbox by default: the Helm chart
  (`pyexec.mode: openshell`) and `make up` / `make dev` locally; see
  [docs/openshell_sandbox.md](docs/openshell_sandbox.md#production-helm). The sandbox enforces a deny-by-default network policy, read-only
  code and libraries, a non-root user, and seccomp. In `direct` mode, which is
  the backend's own default and what training, the eval stack and
  `make test` use, `services/pyexec` is the isolation boundary instead. It is a
  single shared container, not a VM, and it can reach the API and the other tool
  services over the cluster network. Locally, the OpenShell gateway drives the
  host Docker daemon through its socket, which is root-equivalent on the host.
  See [docs/openshell_sandbox.md](docs/openshell_sandbox.md).
- **Sessions are owned, but owners are anonymous.** `POST /api/session/create`
  mints a random owner token, sets it as an HttpOnly `SameSite=Lax` cookie and
  returns it for scripts to send as `Authorization: Bearer`. Every session-scoped
  route (chat, conversation, storage, images, selections, jobs, artifacts, MCP,
  NIM, skills, secrets) requires the token the session was bound to. A request
  with no token gets 401, and another owner's session or artifact is a 404,
  indistinguishable from one that does not exist. Holding a session id or an
  artifact reference is no longer access. The token says nothing about who the
  owner is, though, and anyone can mint one. Bindings live in Redis for
  `AEVOM_SESSION_OWNER_TTL_S` (24h) after last use. Flushing Redis orphans every
  session.
- **The generic `http_request` tool is off by default.** With
  `config.AEVOM_FEAT_HTTP_REQUEST_TOOL` on, the model can send any method, URL,
  headers and body to any host in `HTTP_TOOL_ALLOWED_HOSTS`, with a stored API
  key attached by name. The URL is the model's choice, so a prompt injection in
  anything it reads can steer it. Each request still has to pass the host
  allowlist and an SSRF check that refuses private, loopback, link-local and
  unresolvable addresses, and redirects are not followed. Skill endpoint tools,
  which call only the endpoints a user confirmed in a skill contract, work with
  the flag off.
- **MCP servers and self-hosted NIMs are reached at whatever URL is
  registered.** Any caller with a session owner token, which means anyone who
  can reach the API (see below), can register an endpoint for their own
  session. The API pod then connects to that address from inside the cluster.
  The address is deliberately not checked against an allowlist or for private,
  loopback or link-local ranges, because pointing Aevom at your own MCP server
  or NIM, often on an internal network, is the intended use. Register only
  endpoints you operate or trust. Redirects are not followed, and response
  bodies over 1 GiB are rejected. Because of this, the API pod's egress is the
  control that matters: restrict it with a NetworkPolicy or egress proxy to the
  hosts it needs (the LLM provider, the NVIDIA API, the in-cluster services it
  depends on, and the MCP and NIM hosts you approve), and block cloud metadata
  endpoints such as `169.254.169.254` and the Kubernetes API server.
- **The chart ships only one NetworkPolicy, and pod security contexts only for
  pyexec.** The NetworkPolicy is rendered in OpenShell mode, the default, and
  admits traffic to non-api pods from within the namespace only. The
  direct-mode pyexec pod runs non-root with a read-only root filesystem. Every
  other pod can talk to anything the cluster allows.
- **There is no user authentication in front of the API.** Session ownership
  keeps one caller out of another's data, but anyone who can reach the ingress
  can create a session, run tools and spend your API key. Put your own
  authentication in front of it, or keep it off the public internet.
- **Secrets are referenced, never created by the chart.** API keys and the
  Postgres password are Kubernetes Secrets you create out of band, so key
  material does not pass through a values file or a Helm release revision. Keep
  it that way.

