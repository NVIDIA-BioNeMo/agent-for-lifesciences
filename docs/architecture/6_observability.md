# 6. Observability

OpenTelemetry instrumentation, Arize Phoenix trace collection, span hierarchy, metrics, and feedback annotations.

## Telemetry Architecture

```mermaid
flowchart TB
    subgraph backend ["Aevom Backend"]
        subgraph instrumentation ["Instrumentation Points"]
            InitTelemetry["init_telemetry()\nphoenix.otel.register()\nTracerProvider + gRPC exporter\nMeterProvider + OTLP exporter"]

            ChatSpan["ChatSpanContext\nRoot span: chat_request\nAttrs: user_message, model"]
            IterSpan["IterationSpanContext\nChild span: reasoning_iteration.N\nAttrs: iteration.number"]
            LLMTrace["@trace_llm_call\nChild span: llm.complete / llm.stream\nAttrs: model, temperature,\nmax_tokens, tools_count,\ntool_calls, reasoning"]
            ToolTrace["ToolRegistry.execute()\nChild span: tool.{name}\nAttrs: tool.name, tool.type,\ntool.arg.*, tool.raw.output,\ntool.result.content_type,\ntool.error"]
        end

        subgraph contextvar_workaround ["LangGraph ContextVar Workaround"]
            ActiveIter["_active_iter_spans: dict\nKeyed by request_id\nShared across graph nodes"]
            RotateSpan["_rotate_iteration_span()\nCalled in agent_node\nCloses previous, opens new"]
            RestoreSpan["restore_iteration_span()\nCalled in tool_node\nRestores from shared dict"]
            ResetSpan["reset_iteration_tracking()\nCalled in stream_graph finally\nCleans up dict entry"]
        end

        TokenAcc["TokenAccumulator\nThread-safe token counter\nShared via mutable ContextVar\nRecords input/output tokens"]
        TokenCounter["OTel Counter: llm_agent_tokens_total\nUnit: tokens"]

        FeedbackEndpoint["POST /api/feedback\nPhoenix client.spans.\nadd_span_annotation()\nannotation_name: user_feedback\nannotator_kind: HUMAN"]
        SpanIdEmit["get_current_span_id()\nEmits span_id via SSE\nfor feedback correlation"]
    end

    subgraph phoenix ["Arize Phoenix"]
        OTLP["OTLP gRPC Receiver\n:4317"]
        TraceStore["Trace Storage\n(phoenix_data volume)"]
        WebUI["Web UI :6006\nTrace visualization\nSpan inspector\nDuration analysis"]
        GraphQL["GraphQL API\nProgrammatic queries\nProject stats"]
        Annotations["Span Annotations\nuser_feedback: thumbs_up/down\nscore, metadata"]
    end

    subgraph cli_tools ["CLI Debugging Tools"]
        PhoenixQuery["scripts/phoenix_query.py\nstats, errors, recent\n--filter, --show-reasoning\n--no-limit, --prod/--local"]
        TestChat["scripts/test_chat.sh\nSend test + view trace"]
    end

    subgraph frontend ["Frontend"]
        FeedbackUI["Thumbs up/down buttons\nSends span_id + label"]
    end

    InitTelemetry -->|"gRPC export"| OTLP
    ChatSpan --> IterSpan
    IterSpan --> LLMTrace
    IterSpan --> ToolTrace
    TokenAcc --> TokenCounter
    TokenCounter -->|"OTLP metrics"| OTLP

    OTLP --> TraceStore
    TraceStore --> WebUI
    TraceStore --> GraphQL

    FeedbackUI -->|"POST /api/feedback"| FeedbackEndpoint
    FeedbackEndpoint -->|"Phoenix client"| Annotations
    SpanIdEmit -->|"SSE span_id event"| FeedbackUI

    PhoenixQuery -->|"GraphQL"| GraphQL
```

## Span Hierarchy

```mermaid
flowchart TB
    subgraph trace ["Single Chat Request Trace"]
        ChatRequest["chat_request (root)\nAttrs: chat.user_message,\nchat.total_iterations,\nchat.total_tool_calls,\nchat.hit_iteration_limit"]

        Iter1["reasoning_iteration.1\nAttrs: iteration.number=1,\niteration.tool_calls_count,\niteration.exit_reason"]
        LLM1["llm.complete\nAttrs: llm.model, llm.tools_count,\nllm.tool_calls, llm.has_reasoning,\nllm.raw.reasoning, llm.raw.content,\nllm.raw.tool_calls"]
        Tool1a["tool.fetch_protein_sequence\nAttrs: tool.name, tool.type=external_api,\ntool.arg.gene_name, tool.raw.output,\ntool.result.content_type=text"]

        Iter2["reasoning_iteration.2\nAttrs: iteration.number=2"]
        LLM2["llm.complete"]
        Tool2a["tool.openfold3\nAttrs: tool.type=nvidia_api,\ntool.arg.sequence_ref,\ntool.raw.output"]

        Iter3["reasoning_iteration.3\nAttrs: iteration.number=3"]
        LLM3["llm.complete"]
        Tool3a["tool.visualize_pdb\nAttrs: tool.type=local,\ntool.arg.pdb_ref"]

        FinalLLM["llm.stream (final response)\nAttrs: llm.model,\nllm.raw.content"]
    end

    ChatRequest --> Iter1
    ChatRequest --> Iter2
    ChatRequest --> Iter3
    ChatRequest --> FinalLLM

    Iter1 --> LLM1
    Iter1 --> Tool1a
    Iter2 --> LLM2
    Iter2 --> Tool2a
    Iter3 --> LLM3
    Iter3 --> Tool3a
```

## Span Attribute Reference

### Chat Request Span

| Attribute | Type | Description |
|-----------|------|-------------|
| `chat.user_message` | string | Full user message (no truncation) |
| `chat.total_iterations` | int | Total reasoning iterations |
| `chat.total_tool_calls` | int | Total tool calls made |
| `chat.hit_iteration_limit` | bool | Whether max iterations (20) was reached |
| `llm.model` | string | Model name |

### LLM Spans

| Attribute | Type | Description |
|-----------|------|-------------|
| `llm.model` | string | Model name (e.g., `nvidia/nemotron-3-nano-30b-a3b`) |
| `llm.temperature` | float | Temperature used |
| `llm.max_tokens` | int | Max tokens setting |
| `llm.tools_count` | int | Number of tools provided |
| `llm.tool_calls_count` | int | Number of tools requested |
| `llm.tool_calls` | string | Comma-separated tool names |
| `llm.has_reasoning` | bool | Whether reasoning content returned |
| `llm.raw.reasoning` | string | Full reasoning/thinking content |
| `llm.raw.content` | string | Full response content |
| `llm.raw.tool_calls` | string | Full JSON of tool calls |

### Reasoning Iteration Spans

| Attribute | Type | Description |
|-----------|------|-------------|
| `iteration.number` | int | 1-based iteration number |
| `iteration.messages_count` | int | Messages in context |
| `iteration.tool_calls_count` | int | Tools called this iteration |
| `iteration.reasoning_length` | int | Length of reasoning content |
| `iteration.exit_reason` | string | `no_tool_calls` or `limit` |

### Tool Spans

| Attribute | Type | Description |
|-----------|------|-------------|
| `tool.name` | string | Tool name |
| `tool.type` | string | `local`, `container`, `nvidia_api`, `external_api` |
| `tool.data_io_category` | string | `text_only`, `output_stores`, `input_retrieves`, `both` |
| `tool.arg.{key}` | string | Tool arguments (truncated at 500 chars for UI) |
| `tool.raw.arguments` | string | Full JSON arguments (no truncation) |
| `tool.raw.output` | string | Full tool output (no truncation) |
| `tool.result.content_type` | string | Result type (text, sdf_3d, error, etc.) |
| `tool.result.success` | bool | Whether execution succeeded |
| `tool.error` | string | Error message if failed |
| `tool.source` | string | `builtin` or `mcp` |
| `tool.adapted` | bool | Whether MCP tool was adapted |

## Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `llm_agent_tokens_total` | Counter | tokens | Cumulative LLM tokens consumed (input + output) |

## Source References

- `backend/api/telemetry.py` -- `init_telemetry()`, `ChatSpanContext`, `IterationSpanContext`, `trace_llm_call()`, `trace_tool_execution()`, `TokenAccumulator`, `get_current_span_id()`, `flush_spans()`
- `backend/api/graph/agent.py` -- `_active_iter_spans` dict, `_rotate_iteration_span()`, `restore_iteration_span()`, `reset_iteration_tracking()`
- `backend/api/graph/streaming.py` -- `ChatSpanContext` usage, span_id emission via SSE
- `backend/api/tools/registry.py` -- `ToolRegistry.execute()` span creation, `should_skip_tool_spans()` for training
- `backend/api/main.py` -- `POST /api/feedback` -> Phoenix `client.spans.add_span_annotation()`
- `scripts/phoenix_query.py` -- CLI for querying Phoenix traces via GraphQL
