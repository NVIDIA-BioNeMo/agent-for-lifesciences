# Tool cache fixtures

Recorded tool responses, replayed by the merge-request eval tier so it does not
spend a CI job's budget waiting on PubChem, the RCSB, or a Vina docking run.

One JSON file per cache key. A key is the tool name plus a hash of its
*significant* arguments — `description` and `label` are excluded, because the
model rewrites that prose every run and it has no bearing on the response. Each
file holds the `ToolResult` and, for tools that store data, the artifact payloads
base64-encoded; replay re-stores those into the current session and rewrites the
references, which is what keeps a chained call like `fetch_and_clean_pdb` →
`vina_dock` working off cache.

## Two things these fixtures deliberately do not cover

**Session-coupled tools are never cached.** `structure_query`, `visualize_pdb`,
`get_viewer_state`, `python_execute` and `search_session_history` always run live,
in every mode — see `CachingRegistry.NEVER_CACHED`. They make no network call, so
caching buys no determinism, and replaying `structure_query` would let the eval
that exists to prove the browser bridge works pass with the bridge unplugged.

**The model is never cached.** Only tool responses are. Every run is a fresh
sample, which is why the suite runs several attempts on the live tier.

## The tier uses `mixed`, not `replay`

`replay` errors on a miss, and a miss is unavoidable: `vina_dock` takes a docking
box as floats the model recomputes each run, so its key essentially never
repeats. Measured on the seed suite, strict replay scored 0.98 mean with the
docking task collapsing into 26 failed calls as the model tried to work around
the cache-miss errors — it went off and predicted the structure with OpenFold.
Under `mixed` the same suite scores 1.0 in about 45 seconds, hitting cache for the
PubChem and RCSB lookups and falling through to the live tool otherwise.

## Refreshing

```bash
make eval-record                            # all tasks, live tools
make eval-record TASK=docking-affinity      # one task
evals/scripts/record_fixtures.py --fresh    # discard and re-record from scratch
```

Then verify, which writes to a scratch copy so these files stay pristine:

```bash
evals/scripts/record_fixtures.py --mode mixed --results-dir evals/results/mixed
```

Recording needs `API_KEY_NV` and calls the real model, so it runs locally rather
than in CI. Read the trajectories under `evals/results/record/` before
committing: a recording of a run that went badly pins that bad run in place.

## When to re-record

- A tool's arguments or its result shape changed, so old keys no longer match and
  `mixed` silently stops hitting cache. Watch the wall-clock time of the smoke
  tier for this — it gets slower, it does not fail.
- A task's instruction changed enough to take a different path.
- The recorded upstream data is stale in a way the task depends on.

Do not re-record to make a failing eval pass. The failure is the signal.
