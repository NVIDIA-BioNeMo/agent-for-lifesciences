# Multi-Turn Tool Chaining

## Architecture

Aevom uses a **LangGraph StateGraph** to orchestrate multi-turn tool chaining. The graph is a typed state machine with conditional routing:

| Component               | Class / Function              | Responsibility                                                                     |
| ----------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
| **State**               | `AevomState` (TypedDict)      | Typed state with `add_messages` reducer, viewer/storage/selection state             |
| **Agent Node**          | `ChatAevomLLM` + `state_modifier` | LLM inference with prompt injection of viewer/storage/selection state           |
| **Tool Node**           | `ToolNode` + `StructuredTool` | Executes tools via `ToolRegistry.execute()`                                        |
| **Await Pending Ops**   | `_await_pending_ops_node`     | Drains frontend operations (e.g. selection POSTs) after tool execution             |
| **Refresh State**       | `refresh_state_from_backends` | Pulls viewer/storage/selections from Redis/DB after tools run                      |
| **Routing**             | `_should_continue`            | Routes to `tools` if tool_calls present, otherwise `END`                           |

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH AGENT LOOP                             │
│                                                                      │
│  START                                                               │
│    ↓                                                                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. agent — state_modifier injects viewer/storage/selection     │  │
│  │    state into system prompt, ChatAevomLLM calls LLM with       │  │
│  │    bound tools                                                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                          ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 2. _should_continue — tool_calls in last message?              │  │
│  │    ├─ NO  → END (stream final content to user)                 │  │
│  │    └─ YES ↓                                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                          ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 3. tools — ToolNode executes each StructuredTool wrapper       │  │
│  │    → delegates to ToolRegistry.execute()                       │  │
│  │    → visualization events queued for streaming layer           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                          ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 4. await_pending_ops — drain frontend operations (timeout 10s) │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                          ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 5. refresh_state — pull viewer/storage/selections from         │  │
│  │    Redis/PostgreSQL                                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                          ↓                                           │
│              LOOP BACK TO STEP 1 (max 15 iterations)                 │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File                                        | Purpose                                                                          |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| `backend/api/graph/agent.py`                | `build_agent_graph()` — graph definition, routing, iteration span management     |
| `backend/api/graph/state.py`                | `AevomState` TypedDict, `refresh_state_from_backends()`                          |
| `backend/api/graph/state_modifier.py`       | `build_state_modifier()` — prompt injection callback                             |
| `backend/api/graph/llm.py`                  | `ChatAevomLLM` — LangChain `BaseChatModel` adapter wrapping `LLMClient`         |
| `backend/api/graph/tools.py`                | `build_langchain_tools()` — `StructuredTool` wrappers, viz event queue, context  |
| `backend/api/graph/streaming.py`            | `stream_graph()` — maps `astream_events()` to SSE events                        |
| `backend/api/graph/workflows/`              | Four workflow subgraphs (small molecule, structure prediction, docking, protein-ligand) |
| `backend/api/main.py`                       | `stream_chat_with_langgraph()` — entry point that wires everything together      |

## Step-by-Step Breakdown

### 1. Entry Point (`main.py`)

The `/api/chat` endpoint creates the LangGraph components and delegates to `stream_graph()`:

```python
async def stream_chat_with_langgraph(request: ChatRequest):
    client = get_client()
    rendered_prompt = client.render_system_prompt(base_prompt, tool_schemas)

    llm = ChatAevomLLM(client=client)
    tools = await build_langchain_tools(session_id=session_id)
    graph = build_agent_graph(llm, tools, rendered_prompt)

    storage = SessionScopedStorage(DataStorageService(...), session_id)

    async for event in stream_graph(
        graph,
        system_prompt=rendered_prompt,
        conversation_history=conversation_history,
        viewer_state=request.viewer_state,
        session_id=session_id,
        storage=storage,
        request_id=request_id,
        db_session_factory=...,
    ):
        yield json.dumps(event)
```

### 2. State Initialization (`stream_graph`)

`stream_graph()` converts the conversation history to LangChain messages, then seeds every state field through `refresh_state_from_backends()` — the same function the `refresh_state` node runs:

```python
initial_state = {
    "messages": lc_messages,
    # Empty-shaped defaults, then filled from the backends. The refresh omits
    # findings and candidates when they cannot be read (None means "unknown"),
    # so the keys have to exist first.
    "viewer_state": {"is_empty": True, "structures": []},
    "storage_state": {"is_empty": True, "artifacts": []},
    "selections": [],
    "findings": [],
    "tool_budgets": [],
    "candidates": [],
    "exclusions": [],
    "request_id": request_id,
    "session_id": session_id,
    ...
}
initial_state.update(await refresh_state_from_backends(initial_state, graph_config))
```

The graph runs `START -> agent`, so the first LLM call of the turn — the one that plans it — reads this seed and nothing else. Seeding it any other way is how it once came to omit `[CANDIDATES]`, `[EXCLUSIONS]` and `[TOOL BUDGET]` from the planning call while every later call in the same turn had them.

Services travel in `graph_config["configurable"]` (`storage`, `db_session_factory`), never in state. The same config object is passed to `astream_events()`, so the seed and the run cannot disagree about which storage service they mean.

### 3. Agent Node (`agent` in graph)

The agent node calls the LLM with a `state_modifier` callback that:

1. Reads the state envelope fields from `AevomState` — `viewer_state`, `storage_state`, `selections`, `findings`, `candidates`, `exclusions`, `tool_budgets`
2. Injects the corresponding `[VIEWER STATE]`, `[STORAGE STATE]`, `[SELECTION STATE]`, `[SESSION LOG]`, `[CANDIDATES]`, `[EXCLUSIONS]` and `[TOOL BUDGET]` blocks into the system prompt
3. Returns the modified message list, degrading older turns if the trajectory exceeds the model's window

`ChatAevomLLM` then calls the underlying `LLMClient` with bound tools. The response may contain `tool_calls`:

```python
[{
    "id": "call_123",
    "function": {
        "name": "generate_3d_coordinates",
        "arguments": "{\"smiles\": \"c1ccccc1\"}"
    }
}]
```

### 4. Routing (`_should_continue`)

```python
def _should_continue(state: AevomState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"
```

### 5. Tool Node (`tools` in graph)

`ToolNode` executes each tool call via `StructuredTool` wrappers. Each wrapper:

1. Reads ContextVars for storage, session_id, request_id
2. Calls `ToolRegistry.execute(tool_name, arguments, ...)`
3. Pushes visualization events (SDF, PDB, structure queries) to the `_viz_event_queue`
4. Returns the tool result as a `ToolMessage`

### 6. Await Pending Ops (`await_pending_ops` in graph)

After tools execute, some operations need frontend acknowledgment (e.g. structure query selections). `_await_pending_ops_node` calls `drain_pending_ops(timeout=10.0)` to wait for these.

### 7. Refresh State (`refresh_state` in graph)

`refresh_state_from_backends()` pulls the latest state from Redis/PostgreSQL:

- **Viewer state**: structures loaded in Molstar, chain/residue info
- **Storage state**: data artifacts (SDFs, PDBs, sequences) with references
- **Selections**: active molecular selections from the frontend
- **Findings**, **candidates**, **exclusions**, **tool budgets**: the ledgers and limits behind the `[SESSION LOG]`, `[CANDIDATES]`, `[EXCLUSIONS]` and `[TOOL BUDGET]` blocks

This ensures the agent node sees up-to-date state in its next iteration. It is also the function that seeds the turn's first call (step 2) and every workflow subgraph, so there is exactly one definition of what a state envelope contains — a field added here reaches all three at once.

### 8. Event Emission (`stream_graph`)

`stream_graph()` maps LangGraph `astream_events(version="v2")` to SSE payloads:

| LangGraph Event                          | SSE Output                       | Description                            |
| ---------------------------------------- | -------------------------------- | -------------------------------------- |
| `on_chat_model_stream` (agent node)      | `reasoning`, `content`           | Streaming reasoning tokens and content |
| `on_chat_model_end` (agent node)         | `tool_call_start`                | Tool call detected in LLM response     |
| `on_tool_end` (tools node)               | `tool_event`, `tool_call_end`    | Visualization events + tool completion |
| `on_chain_end` (refresh_state node)      | `storage_update`                 | Updated storage artifacts              |

Visualization events are drained from `_viz_event_queue` on each `on_tool_end` before emitting `tool_call_end`:

```python
# Visualization events (sent to Molstar)
yield {"tool_event": {"type": "visualize_pdb", "pdb": "...", "label": "PGK1"}}

# Tool execution complete
yield {"tool_call_end": {"id": call_id, "name": "openfold3", "success": True}, "iteration": 2}

# Storage updates (sent to frontend artifact strip)
yield {"storage_update": {"artifacts": [...]}}
```

## Workflow Subgraphs

Four domain-specific subgraphs are exposed as callable tools via `build_workflow_tools()`:

| Workflow               | Trigger Examples                          | Tool Subset                                          |
| ---------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Small Molecule         | "Show me aspirin", "Properties of caffeine" | PubChem search, molecule info, 3D coordinates, viz |
| Structure Prediction   | "Predict structure", "fold this sequence" | Fetch sequence, OpenFold3, visualize PDB             |
| Docking                | "Dock", "binding affinity", "Vina"        | Receptor prep, ligand prep, Vina dock, interpret     |
| Protein-Ligand         | "How does X interact with Y?"             | Ligand identification, 3D structure, interactions    |

Each subgraph has its own agent loop (`agent → should_continue → tools → refresh → agent`) with a specialized system prompt and filtered tool set. They run within the parent graph's tool execution.

The child starts from a fresh `[HumanMessage]` — the parent's trajectory never crosses the boundary, which is the point of a subgraph — but its *state* is the parent's, seeded by `_run_subgraph` through `refresh_state_from_backends()`. Storage, viewer, selections and the ledgers are all read under the parent's session, so a workflow is never told the session is empty while the parent is looking at a docked complex.

## Concrete Example

```
Iteration 1:
  agent — state_modifier injects state, LLM called with tools
  → AIMessage: tool_calls=[generate_3d_coordinates(smiles="c1ccccc1")]
  _should_continue → "tools"
  tools — ToolNode executes generate_3d_coordinates → stores SDF as "sdf_xyz"
  await_pending_ops — no pending ops
  refresh_state — updates storage_state with new artifact

Iteration 2:
  agent — state_modifier injects updated state (now includes "sdf_xyz")
  → AIMessage: tool_calls=[visualize_3d(sdf_ref="sdf_xyz")]
  _should_continue → "tools"
  tools — ToolNode executes visualize_3d → sends viz event to Molstar
  await_pending_ops — no pending ops
  refresh_state — updates viewer_state with loaded structure

Iteration 3:
  agent — state_modifier injects updated state (viewer now shows benzene)
  → AIMessage: content="Here is your benzene molecule..."
  _should_continue → "__end__"
  → Stream final content to user
```

## Design Benefits

**LangGraph handles the loop**: Routing, state management, and the iteration loop are handled by LangGraph's `StateGraph` — less custom orchestration code to maintain.

**Typed state with reducers**: `AevomState` is a TypedDict with `add_messages` reducer, preventing state bugs. The `refresh_state_from_backends()` pattern ensures the agent always sees current state.

**Workflow subgraphs as tools**: Domain-specific multi-step sequences (structure prediction, docking, etc.) are encapsulated as subgraphs with their own agent loops and system prompts, callable as regular tools.

**Same SSE contract**: `stream_graph()` maps LangGraph events to the same SSE event format — the frontend is unchanged.

**Dual-mode execution**: Deployment uses LangGraph; training uses the NeMo-Gym resource server which wraps `ToolRegistry` directly. The shared `ToolRegistry` ensures consistent tool behavior across both modes.
