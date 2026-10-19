# Aevom evals

End-to-end evals that drive the real Aevom stack — the same images and the same
config values production runs — and grade what came back. They catch the class
of regression the unit suite structurally cannot: a tool the model can no longer
chain, a viewer contract that broke, a prompt edit that lost the answer.

## Quick start

```bash
make eval-smoke              # cached tools, one attempt per task, a few minutes
make eval-smoke TASK=structure-query-neighbors
make eval                    # live tools, three attempts, the nightly tier
```

Needs an API key at the repo root — the model is live in every mode, since only
tool responses are ever replayed. Which key depends on the endpoint your
`AEVOM_MODEL` uses. The eval stack pins `nemotron-3-ultra-nvinf`, matching
production, so `API_KEY_NVINF` is the one a default run needs:

| File | Endpoint | CI variable |
| --- | --- | --- |
| `API_KEY_NVINF` | `inference-api.nvidia.com` (`nemotron-3-ultra-nvinf`) | `NVIDIA_API_KEY_NVINF` |
| `API_KEY_NV` | `integrate.api.nvidia.com` (`nemotron-3-ultra`) | `NVIDIA_API_KEY` |

Both files must exist for the stack's bind mounts, so `run_suite.sh` creates an
empty one for whichever key you did not supply and says so. That is fine until a
model on that endpoint is selected, at which point it surfaces as a 401 minutes
into the run.

`API_KEY_NV` has a second, unrelated job: the stack mounts it at
`/run/secrets/nvidia_api_key`, which is where `NvidiaApiTool` reads the key for
`health.api.nvidia.com`. So the three NIM tasks need it even though the pinned
model does not. `run_suite.sh` names them if it is empty.

Harbor itself is baked into the devcontainer image, in a venv of its own at
`/opt/harbor-venv` symlinked onto `PATH`. It is isolated rather than installed
alongside everything else because it pulls litellm and supabase, whose transitive
pins on `openai`, `httpx` and `tiktoken` would be resolved against the backend
dependencies `make test-fast` runs on. Nothing needs them to share an
interpreter: harbor imports `aevom_eval` off `PYTHONPATH`, and the driver it
uploads into the container is stdlib-only.

If `harbor` is missing, the make targets stop before building anything and say
what to do: rebuild a stale devcontainer, or — on the host — create a venv and
re-run. `scripts/check-eval-prerequisites.sh` is the guard.

Output lands in `jobs/<timestamp>/`: a trajectory, a per-task verdict from the
verifier, and `gate.json`. Both are gitignored.

## How it fits together

[Harbor](https://github.com/laude-institute/harbor) inverts Aevom's vocabulary:
its "agent" is the system under evaluation. So **Aevom is the agent**, the Aevom
Compose stack is the **environment**, and the code here is the client that drives
it plus the grader that reads the result.

```
harbor run
  └── per task: docker compose up   (docker-compose.eval.yml, production config)
       ├── AevomAgent            uploads the driver into `main`, execs it
       │    └── driver/run_task  POST /api/chat, consume SSE, record everything
       │         └── MolstarBridge   stands in for the browser (see below)
       └── AevomVerifier         scores trajectory.json against [metadata.expect]
gate.py  turns the job's rewards into a CI pass or fail
```

The driver runs *inside* the environment's `main` container rather than on the
host, because the eval stack publishes no host ports — fixed ports would collide
the moment `--n-concurrent` exceeds one. `AevomAgent` uploads `driver/`, execs it,
and brings `trajectory.json` back.

`driver/` is stdlib-only for the same reason: it has to run on the task's plain
`python:3.12-slim` with nothing installed.

### The browser stand-in

`structure_query` emits an SSE `tool_event` and then blocks until a client POSTs a
selection back to `/api/selection`. In production that client is Molstar in the
browser. Headless, nothing answers and the graph stalls until its timeout, so the
eval would measure a different system than the one users run.

`MolstarBridge` plays the browser: it mirrors `frontend/src/App.tsx`, executing the
MolQL against the `molstar-headless` service and posting back the same
`SelectionData` shape the frontend does. The backend is untouched — the harness is
a headless frontend client against the unmodified production path.

## Layout

| Path | What it is |
| --- | --- |
| `aevom_eval/agent.py` | `AevomAgent(BaseAgent)` — uploads and runs the driver |
| `aevom_eval/verifier.py` | `AevomVerifier(BaseVerifier)` — the Harbor adapter |
| `aevom_eval/scoring.py` | The grading rules, importable without Harbor |
| `aevom_eval/driver/` | Runs in `main`: SSE client, bridge, trajectory recorder |
| `tasks/<name>/` | `task.toml`, `instruction.md`, `environment/`, `tests/` |
| `fixtures/tool_cache/` | Recorded tool responses; see its README |
| `scripts/run_suite.sh` | One definition of a tier, shared by `make` and CI |
| `scripts/gate.py` | Reads `jobs/<id>/result.json`, exits non-zero on regression |
| `scripts/record_fixtures.py` | Refreshes the tool cache against live tools |

## Writing a task

Four files, and the expectations live next to the instruction:

```
tasks/my-task/
├── task.toml                        # [metadata.expect] is the grading contract
├── instruction.md                   # what Aevom is asked, verbatim
├── environment/docker-compose.yaml  # one line: include docker-compose.eval.yml
└── tests/test.sh                    # a copy of an existing one, see below
```

`[metadata.expect]` keys are graded in four dimensions, and **only declared
dimensions are scored** — a task that says nothing about artifacts gets no
`artifacts` key rather than a free 1.0 that would inflate the mean:

- `tool_selection` — `required_tools`, `forbidden_tools`,
  `required_tool_sequence`, `required_tool_arguments`
- `artifacts` — `required_artifact_types`, `min_artifacts`
- `answer` — `answer_contains`, `answer_contains_any`, `answer_numbers`,
  `answer_excludes`, `answer_min_chars`, `structure_query_must_succeed`
- `no_errors` — `max_failed_tool_calls`, `allow_iteration_limit`,
  `max_driver_errors`

`reward` is the mean over the declared dimensions, and it is what the gate reads.

### Grading the arguments, not just the tool name

For some tools, "was it called" says almost nothing.
`rfdiffusion_design_backbone` selects de novo generation, motif scaffolding and
binder design through one `contigs` string, so a run that ignored the target
structure entirely looks identical to one that built against it. The trajectory
has recorded `args` all along; `required_tool_arguments` reads them:

```toml
[[metadata.expect.required_tool_arguments]]
tool = "rfdiffusion_design_backbone"
arg = "contigs"
matches = "[A-Za-z]\\d+-\\d+/0"
example = "A25-109/0 100"
```

Predicates are `present` (a bool; an empty string or list counts as absent),
`equals`, `matches` (a regex against the value's string form) and `min_items`.
All of the ones you declare must hold, and the check passes if **any** call to
that tool satisfies them — so a model that gets an argument wrong and then
corrects itself passes here, and pays for the retry under `max_failed_tool_calls`
instead.

`example` is not a predicate. It is a value that *should* pass, and a unit test
asserts it does. Write it: a typo in `matches` otherwise rejects every
trajectory, and the task then fails every run looking exactly like a model
regression. Omit it only when asserting `present = false`.

Two things worth knowing before you tune thresholds:

**Calibrate budgets from a baseline, not from principle.** `max_failed_tool_calls
= 0` looks rigorous and measures the weather: PubChem returns 503
`PUGREST.ServerBusy` often enough that recovering from it is correct behaviour, and
one baseline burned five calls doing so. Run the task, read the trajectory, then
set the budget with headroom.

**Do not enumerate phrasings of a semantic property.** The failure-recovery task
originally listed "could not", "not found" and friends, and was wrong on three
consecutive baselines — each new model wording slipped past a list tuned on the
previous one. Match the structural property instead (any honest report of a miss
contains a negation), or leave it to the LLM judge on the live tier.

`tests/test.sh` is a copy in every task by necessity — Harbor uploads each task's
own `tests` directory. It is only a harness liveness check: it reports under
`harness_ok`, not `reward`, so a job that forgot `--verifier` cannot satisfy the
gate by accident. Copy an existing one verbatim; a unit test pins them together.

### The NIM tasks, and why the upstream evals are not reused

`msa-alignment-coverage`, `binder-backbone-design` and `de-novo-sequence-design`
grade the three NIM skills bound in `backend/api/skills/binding.py`. Each ships
an eval of its own in `third_party/bionemo-agent-toolkit/nim-skills/*/evals/`,
and none of them is usable here — not because they are too easy, but because they
grade the opposite behaviour. Their assertions inspect a Python script the model
was expected to write:

> `check`: "Script contains 'Authorization' and 'Bearer' and 'NGC_API_KEY'"

In Aevom that is the failure mode, not the goal. The model must call
`rfdiffusion_design_backbone`; a run that hand-built that request would mean the
Level 2 banner stopped superseding the skill's prose, which is exactly what these
tasks assert the negative of via `forbidden_tools = ["http_request"]`. The
upstream evals also expect files on disk and an `NGC_API_KEY` in the model's
environment, neither of which exists here — outputs are storage references and
the key is the backend's.

What did carry over is the science. The upstream `deferred_evals` are where the
harder variants live (chain-restricted redesign, motif scaffolding, excluding
cysteines), and those informed the three tasks. Every threshold in them was
baselined against the live endpoints on 2026-08-08; the numbers and the runs they
came from are recorded in comments next to each expectation.

One thing to know if a NIM task starts failing: `openfold3` is deliberately
absent from all three. It is the natural consumer of an MSA and the natural
validator of a design, but the hosted endpoint was returning 500 on every
inference during calibration. Wiring it in would have graded NVIDIA's uptime.

## The two tiers

|  | `eval-smoke` (merge requests) | `eval-live` (nightly) |
| --- | --- | --- |
| Tools | `mixed` — cached, live on a miss | `off` — everything live |
| Images | built from the branch | the registry tags that deploy |
| Attempts | 1 | 3 |
| Gate | mean 0.8, per-task 0.5 | mean 0.7, per-task 0.34 |

Smoke does not run on `main`: the same code was just graded on the merge request,
and re-grading would only insert LLM variance between merge and deploy.

`eval-live` runs only when `$CI_PIPELINE_SOURCE == "schedule"`, and **the schedule
itself is not in this repository** — GitLab keeps schedules in project settings.
Until someone creates one under **Build → Pipeline schedules**, the nightly tier
never runs anywhere. Point it at `main`, since `eval-live` pins the
`$CI_REGISTRY_IMAGE/*:latest` tags and only `main` pushes those.

A scheduled pipeline on `main` still sets `CI_COMMIT_BRANCH=main`, which is the
entire rule for the build jobs and for `deploy-production`. Both carry a
`schedule → when: never` guard (`.publish-rules` in `.gitlab-ci.yml`) so the
nightly grades the deployed images instead of rebuilding and redeploying them
first; `test_the_nightly_schedule_runs_nothing_but_the_evals` keeps that true as
jobs are added. The unit test jobs do still run, which is deliberate — there is
no sense spending three hours of evals on a tree whose tests are broken.

The cache is `mixed` rather than strict `replay` because a miss is unavoidable —
`vina_dock` takes a docking box as floats the model recomputes every run, so its
key essentially never repeats. See `fixtures/tool_cache/README.md`.

Concurrency is 1. Each trial starts a whole Aevom stack (postgres, redis, rdkit,
pyexec, vina, phoenix, molstar-headless) and two at once will exhaust a runner.

## Production parity, and the one gap

`docker-compose.eval.yml` pins images and carries the production ConfigMap values
verbatim — no hot-reload mounts, no `--reload`. A unit test
(`backend/api/tests/test_eval_compose_parity.py`) asserts its environment matches
the Helm chart's production values key for key, reading
`deploy/helm/aevom/values.yaml` and `values-production.yaml` directly, so a
production config change cannot silently desync the eval stack. It also asserts
every enabled chart workload has a counterpart Compose service, so a new
service cannot be deployed and left ungraded.

`AEVOM_MODEL` is pinned to `nemotron-3-ultra-nvinf` here, the same internal
endpoint production uses. The gate thresholds in `.gitlab-ci.yml` are calibrated
against that model; grading a different one means re-baselining them.

The deliberate residual gap: production sets
`AEVOM_FEAT_STORAGE_ACCESS_API=false`, which disables
`/api/storage/artifact/{ref}`. Artifacts are therefore graded on metadata — type,
count, description — and not on content. Grading content would mean diverging from
production config, which costs more than it buys.

The orchestrator differs too: Compose here, Kubernetes in production. Real-cluster
parity would need a custom Harbor environment backend and is deferred.

## Troubleshooting

**"stack did not become healthy"** is usually the bind mount, not the stack. The
stack mounts secrets and `prompts.md` from `$HOST_PROJECT_PATH`, which the Docker
daemon resolves in *its own* filesystem — under docker-outside-of-docker that is
the host, where the devcontainer's `/workspaces/aevom` does not exist.
`run_suite.sh` preflights this and explains itself.

**Rewards contain `harness_ok` instead of `reward`.** The task's stub `test.sh`
graded the run: the job did not pass `--verifier aevom_eval.verifier:AevomVerifier`.
The gate fails on this rather than reading it as a score.

**The smoke tier got slow but still passes.** Cache keys stopped matching, so
everything falls through to live tools. Re-record the fixtures.

**A task fails and re-recording would fix it.** Don't. The failure is the signal;
a recording of a bad run pins that bad run in place.
