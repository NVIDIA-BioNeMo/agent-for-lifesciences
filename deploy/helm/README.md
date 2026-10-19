# Deploying Aevom on Kubernetes

Aevom ships as a Helm chart at [aevom/](aevom). One `helm install` brings up the
API and frontend, the three tool services (RDKit, the Python sandbox, AutoDock
Vina), Postgres, Redis and Phoenix for telemetry.

Two sections:

- [Install](#install) — a fresh cluster, and every value you can set.
- [Building your own images](#building-your-own-images) — required today, since
  there are no published Aevom images yet.

## Install

### What the cluster needs

| Requirement | Why | If missing |
| --- | --- | --- |
| A default StorageClass | Postgres, Redis and Phoenix each claim a ReadWriteOnce volume | PVCs stay `Pending` and nothing starts. Set `persistence.storageClass` if your cluster has no default. |
| metrics-server | The four HPAs read CPU from it | HPAs report `<unknown>` and never scale. Everything runs at one replica — degraded, not broken. |
| An ingress controller | Only if you set `ingress.enabled=true` | An Ingress with no controller is a silent no-op. Use `kubectl port-forward` instead. |
| cert-manager | Only for `ingress.tls` with a `clusterIssuer` | Certificates never issue and the host serves the controller's self-signed default. |

No ReadWriteMany volume is needed. Molecule depictions are rows in Postgres, so
every volume is ReadWriteOnce and any ordinary storage provisioner will do.

### Secrets

The chart references Secrets but never creates them, so no key material passes
through a values file or a Helm release revision. Create them first:

```bash
kubectl create namespace aevom

# Postgres password. Any value; the api and postgres both read it from here.
kubectl -n aevom create secret generic aevom-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)"

# LLM keys: create at least one of these three. The key inside each secret must
# be named `key`.

# Your build.nvidia.com API key, mounted at /run/secrets/nvidia_api_key.
kubectl -n aevom create secret generic nvidia-api-key --from-file=key=./API_KEY_NV

# Your Anthropic API key, mounted at /run/secrets/anthropic_api_key.
kubectl -n aevom create secret generic anthropic-api-key --from-file=key=./API_KEY_ANTHROPIC

# Your OpenAI API key, mounted at /run/secrets/openai_api_key.
kubectl -n aevom create secret generic openai-api-key --from-file=key=./API_KEY_OPENAI
```

All three key Secrets are optional, but the api refuses to start without at least
one of them (see [Troubleshooting](#troubleshooting)). Creating them is always a
manual step: the chart never creates a key Secret, so you register a new key by
creating its Secret yourself.

Get a build.nvidia.com key at <https://build.nvidia.com>. The default model,
`nemotron-3-ultra`, runs there, and so do the NIM-backed tools (OpenFold3,
Boltz-2 and the rest). Without `nvidia-api-key` those tools are withheld from
the agent and everything else keeps working. `nemotron-3-ultra-nvinf` is the
same weights behind an NVIDIA-internal gateway and is not reachable from outside
NVIDIA; see `backend/api/llm/config.py` for the other models and which endpoint
each uses.

Get an Anthropic key at <https://console.anthropic.com>. With `anthropic-api-key`
present, the api lists the models available to that key and offers the newest
model of each Claude family (Opus, Sonnet, Haiku, ...) in the model dropdown.
The list is refreshed every ten minutes, so new models appear without a deploy.
With only the Anthropic key, the api starts on the best Claude model even though
`config.AEVOM_MODEL` names an NVIDIA one.

Get an OpenAI key at <https://platform.openai.com/api-keys>. With
`openai-api-key` present, the api offers the newest GPT Astra, Sol and Luna
models available to that key, refreshed on the same ten-minute cycle. The GPT
Rosalind life-sciences models are listed only when `config.FEAT_ROSALIND` is
`"true"`. With only the OpenAI key, the api starts on the best GPT model. OpenAI
models run on the Responses API with `store=false`, so OpenAI keeps no
conversation state; the encrypted reasoning travels with each request instead.

A Secret created or changed after the api is running is not picked up by the
running pods, because the chart mounts each key with `subPath`. Restart them:

```bash
kubectl -n aevom rollout restart deployment/aevom-api
```

### Install

```bash
helm install aevom deploy/helm/aevom \
  --namespace aevom --create-namespace \
  --set image.registry=your-registry.example.com/aevom

kubectl -n aevom get pods -w        # rdkit and vina take a couple of minutes
kubectl -n aevom port-forward svc/aevom-api 8000:8000
```

Then open <http://localhost:8000>.

`image.registry` has no default and the chart fails to render without it —
see [Building your own images](#building-your-own-images).

For anything beyond a smoke test, put your settings in a file rather than piling
up `--set` flags:

```yaml
# my-values.yaml
image:
  registry: your-registry.example.com/aevom
  tag: v0.1.0

persistence:
  storageClass: gp3

ingress:
  enabled: true
  host: aevom.example.com
  tls:
    enabled: true
    clusterIssuer: letsencrypt-prod

certManager:
  enabled: true
  email: you@example.com
```

```bash
helm upgrade --install aevom deploy/helm/aevom \
  --namespace aevom --create-namespace -f my-values.yaml --wait
```

### Values

Everything the chart exposes. Defaults are in
[aevom/values.yaml](aevom/values.yaml), which carries the reasoning behind each
one; this table is the index.

**Images**

| Value | Default | Notes |
| --- | --- | --- |
| `image.registry` | *(none — required)* | Registry path holding `api`, `rdkit`, `pyexec` and `vina`. Rendering fails if unset. |
| `image.tag` | `latest` | Applied to all four Aevom images. |
| `image.pullPolicy` | `IfNotPresent` | |
| `imagePullSecrets` | `[]` | List of `{name: ...}`. A public registry needs none. |

**Storage**

| Value | Default | Notes |
| --- | --- | --- |
| `persistence.storageClass` | `""` | Empty means the cluster default. Immutable once a claim exists — changing it means deleting the claim, and its data. |
| `persistence.postgres.size` | `10Gi` | Conversation history and stored artifacts. |
| `persistence.redis.size` | `1Gi` | Session and viewer state. |
| `persistence.phoenix.size` | `5Gi` | Telemetry traces. |

**Secrets** (referenced, never created)

| Value | Default | Notes |
| --- | --- | --- |
| `secrets.app.secretName` | `aevom-secrets` | Must contain `POSTGRES_PASSWORD`. |
| `secrets.nvidiaApiKey.secretName` | `nvidia-api-key` | Mounted at `/run/secrets/nvidia_api_key`. Optional, but at least one of `nvidia-api-key`, `anthropic-api-key` and `openai-api-key` must exist. Without it the NIM-backed tools are disabled. |
| `secrets.nvidiaApiKey.key` | `key` | Key within that Secret. |
| `secrets.anthropic.secretName` | `anthropic-api-key` | Mounted at `/run/secrets/anthropic_api_key`. Optional. When present, Anthropic models appear in the model dropdown. |
| `secrets.anthropic.key` | `key` | Key within that Secret. |
| `secrets.openai.secretName` | `openai-api-key` | Mounted at `/run/secrets/openai_api_key`. Optional. When present, OpenAI models appear in the model dropdown. |
| `secrets.openai.key` | `key` | Key within that Secret. |
| `secrets.nvinf.enabled` | `false` | Turn on only for an `inference-api.nvidia.com` model. Enabling it without the Secret leaves the api in `ContainerCreating`. |
| `secrets.nvinf.secretName` | `nvidia-api-key-nvinf` | Mounted at `/run/secrets/nvidia_api_key_nvinf`. |
| `secrets.smtp.enabled` | `false` | Injects `SMTP_PASSWORD` for the feedback relay. Most in-cluster relays accept unauthenticated mail, so leave it off unless yours does not. |
| `secrets.smtp.secretName` / `.key` | `aevom-smtp` / `SMTP_PASSWORD` | Only read when `secrets.smtp.enabled` is true. |

**Application config** → ConfigMap `aevom-config`, injected into the api with `envFrom`

| Value | Default | Notes |
| --- | --- | --- |
| `config.AEVOM_MODEL` | `nemotron-3-ultra` | Must name an entry in `backend/api/llm/config.py`, or a discovered model as `anthropic/<model id>` or `openai/<model id>`. If that model's key is not mounted, the api falls back to a model whose key is, and logs a warning. |
| `config.LOG_LEVEL` | `INFO` | |
| `config.AEVOM_FEAT_STORAGE_ACCESS_API` | `"false"` | Enables `GET /api/storage/artifact/{ref}` and `/api/storage/export`, which serve raw artifact content to the session's owner. |
| `config.AEVOM_FEAT_HTTP_REQUEST_TOOL` | `"false"` | Offers the model the generic `http_request` tool, which can call any host in `HTTP_TOOL_ALLOWED_HOSTS` with a stored key. Skill endpoint tools work without it. |
| `config.FEAT_ROSALIND` | `"false"` | `"true"` also lists OpenAI's GPT Rosalind models in the model dropdown. Needs `openai-api-key`. |
| `config.AEVOM_FEAT_FEEDBACK_EMAIL` | `"false"` | Shows the Feedback button at the foot of the tools panel and enables `POST /api/feedback/email`. Needs the four values below to do anything; on its own it produces a button that 503s. |
| `config.FEEDBACK_EMAIL_ADDRESS` | `""` | Where feedback lands — normally a distribution list. |
| `config.FEEDBACK_FROM_ADDRESS` | `""` | Envelope sender. An authenticated relay requires this to be the mailbox of the `SMTP_USERNAME` account and rejects anything else. Empty falls back to the recipient, which only works on an unauthenticated relay. |
| `config.SMTP_HOST` | `""` | Relay hostname. Together with `FEEDBACK_EMAIL_ADDRESS`, the two values with no usable default. |
| `config.SMTP_PORT` / `SMTP_STARTTLS` | `"587"` / `"true"` | Submission port with STARTTLS. |
| `config.SMTP_USERNAME` | `""` | Empty means unauthenticated. The password comes from `secrets.smtp`, never from here. |
| `config.RDKIT_SERVICE_URL` etc. | Service DNS names | Only change these if you run a tool service outside the release. |
| `apiEnv` | see values.yaml | Literal env on the api Deployment: `DATABASE_URL`, the key file path, the Phoenix trio. |
| `pyexecEnv.BACKEND_URL` | `http://aevom-api:8000` | The sandbox's callback. Must match `config.SANDBOX_BACKEND_URL`. |

Adding a key under `config` reaches the api with no template change.

Two settings are left at their code defaults, so they appear in no values file.
`AEVOM_COOKIE_SECURE` (default `"true"`) marks the session owner cookie `Secure`.
Set it to `"false"` only if the release is served over plain HTTP, or browsers
will drop the cookie. `AEVOM_SESSION_OWNER_TTL_S` (default `86400`) is how long
an idle session stays bound to its owner. After that it is unreachable.

**Prompt**

| Value | Default | Notes |
| --- | --- | --- |
| `prompts.content` | `""` | Empty uses the `prompts.md` baked into the image. Set it to override without a rebuild: `--set-file prompts.content=prompts.md`. |

**Workloads** — `api`, `rdkit`, `pyexec`, `vina`, `postgres`, `redis`, `phoenix`

| Value | Default | Notes |
| --- | --- | --- |
| `<workload>.enabled` | `true` | Turning off `phoenix` also needs `apiEnv.PHOENIX_COLLECTOR_ENDPOINT` cleared, or the api retries spans against nothing. |
| `<workload>.resources` | see values.yaml | Requests and limits. `vina.resources.limits.memory` is 16Gi: docking is the memory-hungry one. |
| `<workload>.autoscaling.enabled` | `true` for api, rdkit, pyexec, vina | Off means a single replica. `replicas` is deliberately not templated when this is on, or every upgrade would fight the HPA back to its value. |
| `<workload>.autoscaling.minReplicas` / `maxReplicas` / `targetCPUUtilizationPercentage` / `behavior` | see values.yaml | |
| `api.service.port` | `8000` | |
| `api.podDisruptionBudget.enabled` | `true` | |
| `vina.env` | see values.yaml | `VINA_GPU`, `CONFORMER_COUNT`, `ROT_BOND_THRESHOLD`, `UNIDOCK_SEARCH_MODE`. |
| `postgres.image` / `redis.image` / `phoenix.image` | pinned upstream tags | Third-party images, unaffected by `image.registry`. |

**Python sandbox**

| Value | Default | Notes |
| --- | --- | --- |
| `pyexec.mode` | `openshell` | Runs `python_execute` in a per-session OpenShell sandbox and drops the pyexec Deployment. Needs an OpenShell gateway, a NetworkPolicy-enforcing CNI, the client TLS Secret and the `pyexec-sandbox` image: see [docs/openshell_sandbox.md](../../docs/openshell_sandbox.md#production-helm). `direct` runs the shared pyexec Deployment instead, for clusters without those prerequisites. |
| `pyexec.hostUsers` | `true` | `false` runs the direct-mode pyexec pod in a user namespace, on clusters that support it. |
| `openshell.gateway` | `openshell.openshell.svc.cluster.local:8080` | gRPC address of the OpenShell gateway. |
| `openshell.clientTlsSecret` | `openshell-client-tls` | Secret in the Aevom namespace with `ca.crt`, `tls.crt`, `tls.key`, copied from the OpenShell chart's Secret of the same name. The production deploy job copies it on every deploy in `openshell` mode. |
| `openshell.clientTlsChecksum` | `""` | sha256 of that Secret's data, rendered as an api pod annotation so a new certificate restarts the api. Set by the deploy job. |
| `openshell.sandboxImage` | `""` | Empty means `<image.registry>/pyexec-sandbox:<image.tag>`. |
| `openshell.sparePool` / `maxActive` / `idleTtlSeconds` | `2` / `20` / `1800` | Warm sandboxes, the cap including spares, and the idle time before a session's sandbox is reaped. |
| `openshell.readyTimeoutSeconds` | `180` | How long a sandbox may take to start. The first sandbox on a node after a deploy pulls the new sandbox image, about 600MB. |
| `openshell.networkPolicy.enabled` | `true` | Admits traffic to non-api pods only from within the namespace. |

**Ingress and TLS**

| Value | Default | Notes |
| --- | --- | --- |
| `ingress.enabled` | `false` | |
| `ingress.className` | `nginx` | |
| `ingress.host` | `aevom.local` | |
| `ingress.path` / `pathType` | `/` / `Prefix` | |
| `ingress.annotations` | NGINX SSE + sticky-session set | Do not drop these wholesale: chat is server-sent events, so buffering must be off and the read timeout must outlast an agent turn. The defaults do that. |
| `ingress.tls.enabled` | `false` | |
| `ingress.tls.secretName` | `aevom-tls` | |
| `ingress.tls.clusterIssuer` | `""` | cert-manager Issuer to mint the certificate. |
| `certManager.enabled` | `false` | Creates Let's Encrypt `ClusterIssuer`s. Cluster-scoped, so a namespaced release affects the whole cluster — off unless you want that. |
| `certManager.email` | `""` | Required when enabled. |
| `certManager.ingressClass` | `nginx` | Solver class; must match `ingress.className`. |

### Verifying and operating

```bash
kubectl -n aevom get pods              # seven workloads
kubectl -n aevom get pvc               # three claims, all Bound
kubectl -n aevom logs deployment/aevom-api --tail=50
kubectl -n aevom exec deployment/aevom-api -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/api/health').status)"

kubectl -n aevom port-forward deployment/phoenix 6006:6006   # telemetry UI
```

Upgrades and rollbacks are ordinary Helm:

```bash
helm upgrade aevom deploy/helm/aevom -f my-values.yaml --wait --timeout 10m
helm -n aevom history aevom
helm -n aevom rollback aevom <revision>
```

`helm uninstall` deletes the PersistentVolumeClaims along with everything else,
which is the database. Back up first, or exclude them by hand.

### Troubleshooting

**API pod stuck in `ContainerCreating`.** A referenced Secret does not exist.
`kubectl -n aevom describe pod <name>` names it. If it is
`nvidia-api-key-nvinf`, either create it or set `secrets.nvinf.enabled=false`.
`nvidia-api-key`, `anthropic-api-key` and `openai-api-key` are optional and
never cause this.

**API pod in `CrashLoopBackOff` with `No LLM API key found` in its log.**
None of `nvidia-api-key`, `anthropic-api-key` and `openai-api-key` exists in the
namespace, or all are empty. Create at least one (see [Secrets](#secrets)), then
`kubectl -n aevom rollout restart deployment/aevom-api`. A Secret whose data key
is not named `key` behaves the same as a missing one.

**Anthropic models missing from the dropdown.** Check that `anthropic-api-key`
exists and that the pods were restarted after it was created. The api log shows
`Anthropic model discovery failed` when the key is rejected or
`api.anthropic.com` is unreachable from the cluster. Egress to it has to be
allowed.

**OpenAI models missing from the dropdown.** The same checks for
`openai-api-key`; the log line is `OpenAI model discovery failed` and the host is
`api.openai.com`. GPT Rosalind is hidden unless `config.FEAT_ROSALIND` is
`"true"`, and the ConfigMap is read at pod start, so restart after changing it.

**PVCs `Pending`.** No default StorageClass. `kubectl get storageclass`, then set
`persistence.storageClass`.

**`ErrImagePull` / `ImagePullBackOff`.** `image.registry` points somewhere the
cluster cannot reach, or a private registry needs `imagePullSecrets`. Confirm
with `kubectl -n aevom describe pod <name>`.

**HPAs show `<unknown>` targets.** No metrics-server. Install it, or set
`autoscaling.enabled=false` per workload to stop the noise.

**Phoenix crashes on startup complaining about `PHOENIX_PORT`.** Kubernetes
injects `PHOENIX_PORT` for any Service named `phoenix`, in `host:port` form,
which the Phoenix client misreads as its listen port. The chart sets an explicit
`PHOENIX_PORT=6006` on both the api and phoenix containers to shadow it. If you
see this, something removed that env var.

**Chat streams stall or cut off mid-answer through the ingress.** The SSE
annotations were changed or dropped. `proxy-buffering` must be `off` and
`proxy-read-timeout` has to outlast the longest agent turn.

**The API answers but the model 401s several minutes in.** The mounted key does
not match `config.AEVOM_MODEL`'s endpoint. Check which endpoint the model uses
in `backend/api/llm/config.py`.

## Building your own images

There are no published Aevom images yet, so the four images have to come from a
registry you control. This is the only extra step; the chart itself assumes
nothing about where they live.

```bash
git clone <repo> && cd aevom
make build                                    # builds all four locally

REGISTRY=your-registry.example.com/aevom
TAG=v0.1.0
for image in api rdkit pyexec vina; do
    docker tag aevom-$image:latest $REGISTRY/$image:$TAG
    docker push $REGISTRY/$image:$TAG
done

helm install aevom deploy/helm/aevom --namespace aevom --create-namespace \
  --set image.registry=$REGISTRY --set image.tag=$TAG
```

The names matter: the chart appends `/api`, `/rdkit`, `/pyexec` and `/vina` to
`image.registry`, and `make build` produces exactly those with an `aevom-`
prefix. `molstar-headless`, which `make build` also produces, is eval harness
infrastructure and is not deployed — production has a real browser.

`pyexec.mode` defaults to `openshell`, so also push the sandbox image, which
`make openshell-images` builds:

```bash
docker tag aevom-pyexec-sandbox:latest $REGISTRY/pyexec-sandbox:$TAG
docker push $REGISTRY/pyexec-sandbox:$TAG
```

The cluster also needs the OpenShell prerequisites in
[docs/openshell_sandbox.md](../../docs/openshell_sandbox.md#production-helm).
On a cluster without them, add `--set pyexec.mode=direct` to run the shared
pyexec Deployment instead; otherwise `python_execute` fails closed.

`make build` needs Docker and takes a while on a cold cache (the RDKit image
builds RDKit from source; `pyexec` builds on top of it). See the root
[README.md](../../README.md) for the local development workflow, which needs no
Kubernetes at all.

When public images are published, `image.registry` gains a default and this
section becomes a note for people who want to build from a fork.
