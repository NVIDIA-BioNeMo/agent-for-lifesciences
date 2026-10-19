# Aevom Instrumentation Guide

This document describes how to use Arize Phoenix for observing and debugging Aevom's agent behavior, including reasoning traces, tool calls, and LLM interactions. It also covers the **autonomous AI development workflow** that enables programmatic trace analysis.

## Overview

Aevom uses [OpenTelemetry](https://opentelemetry.io/) for instrumentation and [Arize Phoenix](https://docs.arize.com/phoenix) as the trace collector and visualization UI. This setup allows you to:

- **View reasoning traces**: See the full chain of thought from user query to final response
- **Debug tool calls**: Inspect arguments, execution time, and results for each tool
- **Analyze LLM interactions**: Monitor model calls, token usage, and tool call decisions
- **Identify bottlenecks**: Find slow tools or excessive iteration loops
- **Autonomous debugging**: Query traces programmatically via GraphQL for AI-assisted development

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Docker Compose Stack                                │
│                                                                             │
│  ┌─────────────┐         ┌─────────────────────────────────┐                │
│  │   Backend   │         │          Phoenix                │                │
│  │   (FastAPI) │ ──────► │  - OTLP Receiver (gRPC:4317)    │                │
│  │             │  traces │  - Web UI (HTTP:6006)           │                │
│  └─────────────┘         │  - GraphQL API                  │                │
│                          │  - Trace Storage                │                │
│                          └─────────────────────────────────┘                │
│                                      ▲                                      │
│                                      │                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Debugging Tools                                  │    │
│  │  scripts/phoenix_query.py                                           |    |
│  │  scripts/test_chat.sh                                               │    |
│  └─────────────────────────────────────────────────────────────────────┘    |
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the Stack

```bash
# Production mode
make up

# Development mode (with hot-reload)
make dev
```

Services will be available at:

| Service        | URL                   | Description                |
| -------------- | --------------------- | -------------------------- |
| Aevom Frontend | http://localhost:5180 | Chat interface             |
| Backend API    | http://localhost:8080 | FastAPI backend (dev mode) |
| Phoenix UI     | http://localhost:6080 | Trace visualization        |

### 2. Generate Traces

Use Aevom normally through the chat interface. Every chat interaction generates traces that are sent to Phoenix.

### 3. View Traces in Phoenix

1. Open http://localhost:6080 in your browser
2. Navigate to the **Traces** tab
3. Select the **aevom** project
4. You'll see traces for each chat request

---

## Autonomous AI Development Workflow

Aevom includes CLI tools for programmatic trace analysis, enabling AI assistants to autonomously test features and debug issues without manual intervention.

### CLI Tools

#### `scripts/phoenix_query.py`

Query Phoenix traces from the command line:

```bash
# Get 10 most recent spans
python3 scripts/phoenix_query.py recent 10

# Filter by span name pattern
python3 scripts/phoenix_query.py recent 20 --filter "tool."

# Show full LLM reasoning content
python3 scripts/phoenix_query.py recent 5 --show-reasoning

# Filter to reasoning iterations with full content
python3 scripts/phoenix_query.py recent 10 --filter "reasoning_iteration" --show-reasoning

# Show project statistics
python3 scripts/phoenix_query.py stats
```

Example output:

```
============================================================
Found 5 spans
============================================================

✓ chat_request (6870ms)
   Time: 2026-01-12T21:23:48.634137+00:00
   User: What is the molecular formula of caffeine?
   Iterations: 1
----------------------------------------
✓ reasoning_iteration.1 (5230ms)
   Time: 2026-01-12T21:23:48.635397+00:00
   Tool calls: 0
   Exit reason: no_tool_calls
----------------------------------------
✓ tool.python_execute (21ms)
   Time: 2026-01-12T20:44:26.479476+00:00
   Args: {"code": "import numpy as np\natoms = selection['atoms']...
   Result: text (success=True)
   Output: Alpha carbon of residue 14 and residue 50 are 8.88 Angstroms apart
```

#### `scripts/test_chat.sh`

Combined workflow: send a chat request and immediately view the trace:

```bash
# Run test and show trace
./scripts/test_chat.sh "Generate 3D coordinates for benzene"

# With full reasoning content
./scripts/test_chat.sh "What tools are available?" --show-reasoning

# Custom span limit
./scripts/test_chat.sh "Test message" --limit 20
```

### DevContainer Network Setup

When running in a devcontainer, connect to the Aevom Docker network for direct service access:

```bash
# Connect to network (run once per session)
docker network connect aevom_default $(hostname)

# Service endpoints after connection
# Backend: http://aevom-backend-1:8000
# Phoenix: http://aevom-phoenix-1:6006/graphql
```

### Phoenix GraphQL API

Phoenix exposes a GraphQL API for programmatic trace queries. The Aevom project ID is `UHJvamVjdDoy`.

**Query recent spans:**

```bash
curl -s "http://aevom-phoenix-1:6006/graphql" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ node(id: \"UHJvamVjdDoy\") { ... on Project { spans(first: 10, sort: {col: startTime, dir: desc}, timeRange: {start: \"2026-01-01T00:00:00Z\", end: \"2027-01-01T00:00:00Z\"}) { edges { node { name statusCode latencyMs attributes } } } } } }"}' \
  | python3 -m json.tool
```

**Query project stats:**

```bash
curl -s "http://aevom-phoenix-1:6006/graphql" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ projects { edges { node { name traceCount } } } }"}'
```

### AI Development Workflow

When implementing or debugging features, follow this workflow:

1. **Ensure environment is ready**

   ```bash
   make dev  # Start services if not running
   docker network connect aevom_default $(hostname) 2>/dev/null || true
   ```

2. **Send test request**

   ```bash
   curl -s -X POST "http://aevom-backend-1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Your test message"}], "max_tokens": 1024}'
   ```

3. **Query traces**

   ```bash
   python3 scripts/phoenix_query.py recent 15
   ```

4. **Analyze results**

   - Check `statusCode` for errors
   - Read `llm.raw.reasoning` in child `llm.complete` spans to understand LLM decisions
   - Verify tool arguments and results
   - Identify unexpected behavior

5. **Iterate**
   ```bash
   docker restart aevom-backend-1  # After code changes
   # Re-run test and compare traces
   ```

---

## Trace Hierarchy

Each chat request creates a hierarchical trace structure:

```
chat_request (root span)
├── reasoning_iteration.1
│   ├── llm.complete (tool detection call)
│   └── tool.generate_3d_coordinates
├── reasoning_iteration.2
│   ├── llm.complete (tool detection call)
│   └── tool.visualize_3d
└── llm.stream (final response streaming)
```

### Span Types

| Span Name               | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| `chat_request`          | Root span for the entire user interaction                        |
| `reasoning_iteration.N` | One iteration of the tool-calling loop                           |
| `llm.complete`          | Non-streaming LLM call (used for tool detection)                 |
| `llm.stream`            | Streaming LLM call (used for final response)                     |
| `tool.{name}`           | Individual tool execution (e.g., `tool.generate_3d_coordinates`) |

### Span Attributes

**Chat Request Span:**

| Attribute                   | Description                        |
| --------------------------- | ---------------------------------- |
| `chat.user_message_preview` | First 200 chars of user message    |
| `chat.max_tokens`           | Max tokens setting                 |
| `chat.temperature`          | Temperature setting (if specified) |
| `chat.tools_available`      | Number of tools available          |
| `chat.total_iterations`     | Total reasoning iterations         |
| `chat.total_tool_calls`     | Total tool calls made              |
| `chat.hit_iteration_limit`  | Whether max iterations was reached |

**LLM Spans:**

| Attribute               | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `llm.model`             | Model name (e.g., `nvidia/nemotron-3-nano-30b-a3b`)   |
| `llm.temperature`       | Temperature used                                      |
| `llm.max_tokens`        | Max tokens setting                                    |
| `llm.tools_count`       | Number of tools provided                              |
| `llm.tool_calls_count`  | Number of tools requested by model                    |
| `llm.tool_calls`        | Comma-separated list of tool names called             |
| `llm.has_reasoning`     | Whether reasoning content was returned                |
| `llm.raw.reasoning`     | **Full text of the LLM's reasoning/thinking process** |
| `llm.raw.content`       | **Full response content (no truncation)**             |
| `llm.raw.tool_calls`    | **Full JSON of tool calls requested**                 |

**Reasoning Iteration Spans:**

| Attribute                     | Description                                           |
| ----------------------------- | ----------------------------------------------------- |
| `iteration.number`            | Iteration number (1-based)                            |
| `iteration.messages_count`    | Number of messages in context                         |
| `iteration.tool_calls_count`  | Tools called in this iteration                        |
| `iteration.reasoning_length`  | Length of reasoning content                           |
| `iteration.exit_reason`       | Why the loop exited (`no_tool_calls` or limit)        |

> **Note:** Reasoning content is stored in the child `llm.complete` span as `llm.raw.reasoning` to avoid duplication.

**Tool Spans:**

| Attribute                  | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `tool.name`                | Tool name                                         |
| `tool.type`                | Tool type (local, container, nvidia_api)          |
| `tool.data_io_category`    | Data I/O pattern (text_only, output_stores, etc.) |
| `tool.arg.{key}`           | Tool arguments (truncated if large, for UI)       |
| `tool.raw.arguments`       | **Full JSON arguments (no truncation)**           |
| `tool.raw.output`          | **Full tool output (no truncation)**              |
| `tool.result.content_type` | Result type (text, sdf_3d, error, etc.)           |
| `tool.result.success`      | Whether execution succeeded                       |
| `tool.error`               | Error message if tool failed                      |

---

## Training Span Tree

Training rollouts use a different (live) span tree because the simple_agent
runs in a separate process from the resource server. The 2026-05-27
refactor replaced the retroactive `reasoning_iteration.* / llm.complete`
reconstruction with real-time per-turn spans, and the 2026-05-30 pass
restored full LLM output capture on `policy_llm.responses` (the
production-equivalent of `llm.complete`) and dropped the noise spans
`agent.prepare` / `agent.parse`:

```
chat_request                       [seed_session -> verify, root]
├── agent.turn.{1..N}              [per loop iteration]
│   ├── policy_llm.responses       [vLLM HTTP wall + LLM raw output]
│   └── tool.{name}                [live, from ToolRegistry]
│       └── tool_service.http      [per-host HTTP wall]
└── judge_evaluation               [optional]
    └── llm.complete               [judge call]
```

**Cross-process parenting:** the `EpisodeSpan` (created at `seed_session`)
serialises a W3C `traceparent` header into the seed-session response.
The simple_agent extracts and attaches it on first turn, and re-injects
it on every outgoing tool POST so the resource server's
`/{tool_name}` handler can attach the same context before calling
`ToolRegistry.execute` (otherwise the live `tool.{name}` spans become
orphans, on a different trace from `chat_request`).

### Training Span Attributes

**`policy_llm.responses` (per-turn LLM call):**

| Attribute               | Description                                                            |
| ----------------------- | ---------------------------------------------------------------------- |
| `turn.number`           | Turn number (1-based, matches the parent `agent.turn.{N}`)             |
| `llm.server`            | Configured NeMo-Gym model server name                                  |
| `llm.wall_ms`           | HTTP wall time for the `/v1/responses` POST                            |
| `llm.input_tokens`      | Input/prompt tokens (when vLLM reports usage)                          |
| `llm.output_tokens`     | Output/completion tokens (when vLLM reports usage)                     |
| `llm.has_reasoning`     | Whether reasoning content was returned                                 |
| `llm.tool_calls_count`  | Number of function calls the model requested this turn                 |
| `llm.functions`         | Comma-separated function names called this turn (e.g. `"vina_dock,molecule_info"`) |
| `llm.raw.reasoning`     | **Full text of the model's reasoning/chain-of-thought (no truncation)** |
| `llm.raw.content`       | **Full assistant message content (no truncation)**                     |
| `llm.raw.tool_calls`    | **Full JSON of function calls (no truncation); matches production's `llm.raw.tool_calls`** |
| `llm.parse.error`       | Pydantic validation error if `NeMoGymResponse.model_validate` fails    |

**`agent.turn.{N}` (per loop iteration):**

| Attribute              | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| `turn.number`          | Turn number (1-based)                                             |
| `turn.fn_calls_count`  | Number of function calls dispatched this turn                     |
| `turn.exit_reason`     | Why the turn ended (`tool_calls`, `final_message`, `max_output_tokens`) |

**`chat_request` (training-only attributes):**

| Attribute                       | Description                                          |
| ------------------------------- | ---------------------------------------------------- |
| `training.episode_id`           | NeMo-Gym session ID for this episode                 |
| `training.is_eval`              | Whether this is an eval (vs. train) episode          |
| `training.wandb_run_id`         | Linked Weights & Biases run                          |
| `training.slurm_job_id`         | SLURM job ID (when running on the cluster)           |
| `training.reward`               | Total episode reward                                 |
| `training.reward.{tier}`        | Per-tier reward breakdown (`tier1_tool_usage`, ...)  |
| `training.reward.tier1.{tool}`  | Per-tool capped Tier-1 contribution (e.g. `...tier1.vina_dock`) |
| `training.reward_gated_no_tools`| `True` when the episode made no tool calls (reward forced to 0) |
| `training.sabotage_type`        | Ground-truth sabotage type for this episode          |
| `training.num_turns`            | Number of `agent.turn.{N}` iterations                |
| `training.num_tool_calls`       | Total tool calls dispatched                          |
| `training.num_tool_errors`      | Tool calls that returned an error                    |
| `chat.final_response`           | Final assistant message (falls back to last reasoning) |
| `chat.episode_outcome`          | `complete` / `no_response` / `no_tools` / `empty`    |

### OpenInference Rendering Attributes

Phoenix 13.x renders LLM messages, reasoning, tool calls, and trace-level
I/O **only** from [OpenInference semantic conventions](https://arize.com/docs/phoenix/references/openinference),
not from the custom `llm.raw.*` / `chat.*` / `judge.*` / `tool.*` attributes
above. Each span-creation site therefore emits these conventions **in
addition to** (never instead of) the custom + profiling attributes — the
custom families are kept verbatim so `scripts/profile_run.py` is unaffected.
The shared helpers live in `backend/api/telemetry.py` (`set_span_kind`,
`set_io`, `set_input_messages`, `set_output_message`, `set_llm_tools`) and are
all try/except-guarded so telemetry can never break the training loop.

| Span                    | `openinference.span.kind` | OpenInference attributes added                                                                 |
| ----------------------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| `chat_request`          | `AGENT`                   | `input.value` (task prompt), `output.value` (final response), `llm.tools.*` (advertised tools) |
| `policy_llm.responses`  | `LLM`                     | `llm.model_name`, `llm.input_messages.*` (full chat history), `llm.output_messages.0.*` (reasoning + final content concatenated into `message.content`, plus `message.tool_calls.*`), `llm.tools.*`, `llm.invocation_parameters` |
| `judge_evaluation`      | `EVALUATOR`               | `input.value` (judge prompt), `output.value` (verdict + reward + reasoning)                    |
| `llm.complete` (judge)  | `LLM`                     | `llm.model_name`, `llm.input_messages.*` (system + user prompt), `llm.output_messages.0.*` (reasoning + content) |
| `tool.{name}`           | `TOOL`                    | `tool.parameters` + `input.value` (arguments JSON), `output.value` (tool result text)          |

Reasoning is surfaced via the reliably-rendered `message.content` string
(reasoning concatenated with any final assistant text) rather than a
non-standard reasoning content block, so per-turn chain-of-thought is visible
even on tool-calling turns that emit no final assistant message. The original
`llm.raw.reasoning` / `judge.reasoning` attributes are retained as the
untruncated source of truth.

### Reward Annotation Targeting

Each episode emits exactly **one** Phoenix span annotation: `total_effective_reward`
on the root `chat_request` span, carrying the final summed/capped/no-tools-gated
reward as its score and the per-tier `reward_breakdown` in its metadata.

This is deliberately minimal. Earlier revisions also emitted per-tool-call
(`tool_call_attempt` / `tool_call_success`), Tier-2/Tier-3 catalog, and
`judge_reasoning` annotations -- 10-40 annotations/episode. At pps=32 that
flooded Phoenix's `span_annotations` REST API with `404 Not Found`s whenever
ingestion fell behind, and the synchronous retry backoff
(`_submit_one_annotation`, up to ~37s/annotation) saturated the shared verify
threadpool, back-pressuring rollout completion. Collapsing to a single
root-span annotation removes both failure modes.

**No reward detail is lost** -- the annotations were only a Phoenix
visualization overlay. The same data is persisted as span attributes and in
`episodes.jsonl`:

- Per-tool Tier-1 contributions: `training.reward.tier1.{tool}` attributes (+ `tier1_details` in the verify response).
- Tier-2 / Tier-3: `verify.reward_tiers` span attributes (+ `tier2_details` / `tier3_details`).
- Judge scores + reasoning: `judge_evaluation` span attributes (`judge.total_score`, `judge.reasoning`, the six `judge.score.*`, etc.) (+ `judge_reasoning_reward`).
- Total + per-tier breakdown: `training.reward.*` attributes + the `total_effective_reward` annotation.

The training signal itself flows through the verify response (`reward` /
`reward_breakdown`) and is entirely independent of annotations.

---

## Common Debugging Scenarios

### Finding Slow Tool Calls

```bash
# Using CLI
python3 scripts/phoenix_query.py recent 20 --filter "tool."
# Look at latencyMs for each tool span
```

Or via Phoenix UI:

1. Open Phoenix UI → Traces
2. Sort by duration (descending)
3. Expand traces to find `tool.*` spans with long durations

### Debugging Tool Failures

```bash
# Using CLI - look for ERROR status
python3 scripts/phoenix_query.py recent 20 --filter "tool."
# Check tool.error attribute in output
```

Or via Phoenix UI:

1. Filter by status: ERROR
2. Look for `tool.*` spans with error status
3. Check `tool.error` attribute for error message

### Analyzing LLM Decisions

```bash
# See what tools the LLM chose to call
python3 scripts/phoenix_query.py recent 10 --filter "llm.complete"

# See full reasoning for why
python3 scripts/phoenix_query.py recent 5 --filter "reasoning_iteration" --show-reasoning
```

### Viewing LLM Reasoning Content

The full reasoning/thinking process is captured in spans. Use `--show-reasoning` to view:

```bash
python3 scripts/phoenix_query.py recent 5 --show-reasoning
```

This shows:

- Why the LLM chose specific tools
- Tool argument generation logic
- Chain-of-thought before tool calls
- Reasoning errors or hallucinations

### Identifying Infinite Loops

```bash
# Look for traces that hit iteration limit
python3 scripts/phoenix_query.py recent 10 --filter "chat_request"
# Check for "⚠ Hit iteration limit!" in output
```

---

## Environment Variables

Phoenix configuration is controlled via environment variables:

| Variable                     | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://phoenix:4317` | gRPC endpoint for trace collection |
| `PHOENIX_PROJECT_NAME`       | `aevom`               | Project name in Phoenix UI         |

These are configured in `docker-compose.yml` and `docker-compose.dev.yml` for the backend service.

---

## Local Development (Without Docker)

If you're running Aevom outside Docker for development:

### 1. Start Phoenix Locally

```bash
# Install Phoenix
pip install arize-phoenix

# Run Phoenix server
python -m phoenix.server.main serve
```

Phoenix will start on http://localhost:6006 with gRPC on port 4317.

### 2. Configure Environment

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
export PHOENIX_PROJECT_NAME=aevom-dev
```

### 3. Run Aevom

```bash
cd backend/api
uvicorn api.main:app --reload
```

---

## Disabling Telemetry

To disable telemetry (e.g., in production without Phoenix):

1. Don't set `PHOENIX_COLLECTOR_ENDPOINT`, or
2. Set it to an unreachable endpoint

The telemetry module gracefully handles connection failures and won't crash the application.

---

## Troubleshooting

### No Traces Appearing

1. Check Phoenix is running: `docker compose ps phoenix`
2. Check Phoenix logs: `docker compose logs phoenix`
3. Verify backend can reach Phoenix: Check for connection errors in backend logs
4. Ensure `PHOENIX_COLLECTOR_ENDPOINT` is correct

### Incomplete Traces

1. Check for errors in backend logs
2. Look for spans with ERROR status in Phoenix
3. Verify all services are healthy: `docker compose ps`

### Phoenix UI Not Loading

1. Ensure port 6080 is not in use: `lsof -i :6080`
2. Check Phoenix container logs: `docker compose logs phoenix`
3. Try restarting: `docker compose restart phoenix`

### CLI Tools Not Connecting

1. Ensure you're connected to the Docker network:
   ```bash
   docker network connect aevom_default $(hostname)
   ```
2. Verify services are running: `docker ps | grep aevom`
3. Test connectivity: `curl -s http://aevom-backend-1:8000/api/health`

---

## Implementation Details

The instrumentation is implemented in [`backend/api/telemetry.py`](backend/api/telemetry.py):

- **`ChatSpanContext`**: Creates root span for chat requests (used in `stream_graph()`)
- **`IterationSpanContext`**: Creates child spans for each reasoning iteration (managed via `_active_iter_spans` dict in `agent.py` — see ContextVar note below)
- **`trace_llm_call`**: Decorator for instrumenting LLM API calls (called by `ChatAevomLLM`)
- **`trace_tool_execution`**: Creates spans for tool execution (in `registry.py`)

### `chat_request` Is Always the Trace Root

`ChatSpanContext` (and the training `EpisodeSpan`) start `chat_request` with an
empty OpenTelemetry context, so it never adopts whatever span is active at the
time. Children are parented explicitly through the `_current_chat_span` /
`_current_iteration_span` ContextVars, not through the ambient OTel context.

FastAPI 0.142.0 added native OpenTelemetry tracing, on by default whenever a
global TracerProvider exists — which `phoenix.otel.register()` sets. Left on, it
wrapped each turn in a `POST /api/chat` server span and gave every polling
endpoint (`/api/skills/`, `/api/secrets/`, `/api/mcp/tools`, ...) a trace of its
own. [`backend/api/main.py`](backend/api/main.py) therefore constructs the app
with `telemetry={"tracing": False, "metrics": False, "logs": False}`, which is
why `backend/requirements.txt` requires `fastapi>=0.142.0`.

Tool execution tracing happens in [`backend/api/tools/registry.py`](backend/api/tools/registry.py), which captures:

- Tool arguments
- Execution timing
- Result content type and success status
- Error messages

### LangGraph ContextVar Workaround

LangGraph isolates ContextVars per node — each node runs with a copy of the parent's context, so mutations to `_current_iteration_span` in the `agent` node are not visible in the `tools` node. To work around this, iteration spans are shared via a **module-level dict** (`_active_iter_spans`) in `agent.py`, keyed by `request_id`:

- **`_rotate_iteration_span(request_id)`**: Called at the start of each `agent` node invocation. Closes the previous iteration span and opens a new one.
- **`restore_iteration_span(request_id)`**: Called at the start of other nodes (e.g. `tool_node`) to restore the current iteration span from the shared dict.
- **`reset_iteration_tracking(request_id)`**: Called in the `finally` block of `stream_graph()` to clean up the dict entry after the request completes.

---

## Further Reading

- [Arize Phoenix Documentation](https://docs.arize.com/phoenix)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
- [arize-phoenix-otel Package](https://arize.com/docs/phoenix/sdk-api-reference/python/arize-phoenix-otel)
