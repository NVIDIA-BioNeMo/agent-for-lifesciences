# View State Injection Pattern

## The Challenge: Agentic AI with Visual Context

### The Problem Space

Aevom is an agentic AI system that combines large language model reasoning with computational chemistry tools and interactive 3D molecular visualization. This architecture presents a unique challenge: **the AI agent operates on the backend, but the visualization happens on the frontend**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THE VISIBILITY GAP                           │
│                                                                     │
│   ┌──────────────┐         ┌──────────────┐         ┌────────────┐  │
│   │   Frontend   │         │   Backend    │         │    LLM     │  │
│   │   (Browser)  │ ──SSE──▶│   (FastAPI)  │ ──API──▶│   Agent    │  │
│   │              │         │              │         │            │  │
│   │  ┌────────┐  │         │  ┌────────┐  │         │  "What is  │  │
│   │  │Molstar │  │         │  │ Tools  │  │         │  currently │  │
│   │  │Viewer  │  │         │  │Registry│  │         │  displayed?│  │
│   │  └────────┘  │         │  └────────┘  │         │     ???    │  │
│   │      ▲       │         │              │         │            │  │
│   │      │       │         │              │         └────────────┘  │
│   │   USER SEES  │         │  AGENT CAN'T │                         │
│   │   THE 3D     │         │  SEE THE     │                         │
│   │   MOLECULE   │         │  VIEWER      │                         │
│   └──────────────┘         └──────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

When a user asks "highlight the binding site," the agent needs to know:

1. Is there a structure currently loaded?
2. What is it? (protein name, chains, ligands)
3. Should I load something first, or can I proceed directly?

Without this context, the agent either:

- **Wastes time**: Calls a tool to check viewer state, burning an LLM turn
- **Duplicates work**: Re-fetches and re-predicts structures already displayed
- **Fails confusingly**: Tries to highlight something in an empty viewer

### The Multi-Turn Tool Chaining Context

Aevom uses a multi-turn tool chaining loop where the LLM can autonomously call multiple tools across reasoning iterations:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MULTI-TURN LOOP                             │
│                                                                 │
│  1. Send messages[] to LLM with available tools                 │
│                          ↓                                      │
│  2. LLM responds with content and/or tool_calls[]               │
│                          ↓                                      │
│  3. IF no tool_calls → EXIT LOOP, stream final response         │
│                          ↓                                      │
│  4. Execute each tool, append results to messages[]             │
│                          ↓                                      │
│              LOOP BACK TO STEP 1 (max 10 iterations)            │
└─────────────────────────────────────────────────────────────────┘
```

Each iteration costs time and tokens. If the agent must call `get_viewer_state` before every visualization-related action, that's an extra round trip through the entire loop—typically adding 1-3 seconds of latency.

---

## System Architecture Overview

### Data Flow

```
┌─────────────┐     POST /api/chat      ┌─────────────────────────────┐
│   Browser   │ ──────────────────────▶ │         FastAPI             │
│   (React)   │   {messages, viewer_    │                             │
│             │    state}               │  ┌─────────────────────┐    │
│  ┌───────┐  │                         │  │   System Prompt     │    │
│  │Molstar│  │ ◀─────────SSE─────────  │  │   + [VIEWER STATE]  │    │
│  │Viewer │  │   {content, tool_event} │  └─────────────────────┘    │
│  └───────┘  │                         │            │                │
│      │      │                         │            ▼                │
│      │      │                         │  ┌─────────────────────┐    │
│      ▼      │                         │  │    LLM Client       │───▶│ NVIDIA API
│  getState() │                         │  └─────────────────────┘    │
│      │      │                         │            │                │
│      ▼      │                         │            ▼                │
│  ViewerState│                         │  ┌─────────────────────┐    │
│  {structures,                         │  │   Tool Registry     │    │
│   chains,   │                         │  │   (execute tools)   │    │
│   ...}      │                         │  └─────────────────────┘    │
└─────────────┘                         └─────────────────────────────┘
```

### Key Components

1. **Frontend (React + Molstar)**: Renders 3D molecular structures, tracks viewer state
2. **Backend (FastAPI)**: Orchestrates LLM calls, executes tools, streams responses
3. **LLM Agent**: Reasons about user requests, decides which tools to call
4. **Tool Registry**: Collection of tools (structure prediction, visualization, queries)

### The SSE Communication Pattern

The backend communicates with the frontend via Server-Sent Events (SSE):

- **Backend → Frontend**: Tool events (load structure, highlight atoms, etc.)
- **Frontend → Backend**: Viewer state sent with each new chat request

This is **unidirectional per-request**. The frontend can't "push" state changes to the backend mid-conversation—it can only send state when initiating a new request.

---

## The View State Injection Pattern

### Solution Overview

Instead of requiring the agent to call a tool to discover viewer state, we **inject a condensed representation of the viewer state directly into the system prompt** at the start of every LLM call.

```
┌────────────────────────────────────────────────────────────────────┐
│                    VIEW STATE INJECTION FLOW                       │
│                                                                    │
│  1. Frontend sends viewer_state with chat request                  │
│                          ↓                                         │
│  2. Backend parses and stores viewer state                         │
│                          ↓                                         │
│  3. Generate condensed summary: "[VIEWER STATE: PGK1 (protein)...]"│
│                          ↓                                         │
│  4. Append to system prompt before sending to LLM                  │
│                          ↓                                         │
│  5. LLM has immediate context—no tool call needed                  │
└────────────────────────────────────────────────────────────────────┘
```

### Implementation

#### 1. Define Viewer State Data Structures

```python
# viewer_state.py

@dataclass
class StructureInfo:
    """Information about a single loaded structure."""
    label: str = ""
    structure_type: str = ""  # "protein", "small_molecule", "complex"
    chain_ids: list[str] = field(default_factory=list)
    residue_count: int = 0
    atom_count: int = 0
    has_polymer: bool = False
    has_ligand: bool = False
    has_water: bool = False

@dataclass
class ViewerState:
    """Complete state of the molecular viewer."""
    is_empty: bool = True
    structures: list[StructureInfo] = field(default_factory=list)
    has_selection: bool = False
    selection_info: Optional[str] = None
```

#### 2. Create Condensed Summary Method

```python
# viewer_state.py

def to_condensed(self) -> str:
    """Generate a condensed one-line summary for system prompt injection."""
    if self.is_empty:
        return "[VIEWER STATE: Empty - no structures loaded]"

    parts = []
    for struct in self.structures:
        struct_parts = []
        if struct.label:
            struct_parts.append(struct.label)
        if struct.structure_type:
            struct_parts.append(f"({struct.structure_type})")
        if struct.chain_ids:
            struct_parts.append(f"chains: {','.join(struct.chain_ids)}")
        if struct.residue_count > 0:
            struct_parts.append(f"{struct.residue_count} residues")
        if struct.has_ligand:
            struct_parts.append("has ligand")
        parts.append(" ".join(struct_parts))

    return f"[VIEWER STATE: {'; '.join(parts)}]"
```

**Example outputs:**

- `[VIEWER STATE: Empty - no structures loaded]`
- `[VIEWER STATE: PGK1 (protein) chains: A 417 residues]`
- `[VIEWER STATE: CDK2-Aspirin (complex) chains: A,B 298 residues has ligand]`

#### 3. Inject Into System Prompt

```python
# LLMPolicy._inject_state_into_prompt() (api/policy/llm_policy.py)

def _inject_state_into_prompt(self, base_prompt, observation):
    # Generate condensed summary and inject into system prompt
    viewer_state_summary = observation.viewer_state.get("condensed", "")
    storage_state_summary = observation.storage_state.get("condensed", "")
    effective_system_prompt = f"{base_prompt}\n\n{viewer_state_summary}\n{storage_state_summary}"

    # ... continue with LLM call
```

#### 4. Update System Prompt Documentation

Tell the agent where to find and how to use the injected state:

```
### VIEWER AWARENESS

The current viewer state is automatically provided at the end of this prompt.
Look for `[VIEWER STATE: ...]` to see what structures are currently loaded.

Use this information to:
- Know what structures are already loaded
- Avoid redundant work (don't re-fetch/re-predict if already displayed)
- Understand context for follow-up queries

**Workflow for Follow-Up Questions:**
1. CHECK the [VIEWER STATE] at the end of this prompt
2. IF structure is loaded: Use structure_query directly
3. IF viewer is empty: Run full workflow (fetch → predict → visualize)
```

### Frontend Integration

The frontend must send viewer state with each request:

```typescript
// Chat.tsx - sending request
const response = await fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [...],
    viewer_state: getViewerState(),  // Current viewer state
  }),
});

// MolstarViewer.tsx - collecting state
function getViewerState(): ViewerStateInfo {
  const structures = getAllStructures(plugin);
  return {
    isEmpty: structures.length === 0,
    structures: structures.map(s => ({
      label: s.label,
      structureType: detectType(s),
      chainIds: getChainIds(s),
      residueCount: countResidues(s),
      hasLigand: hasNonPolymer(s),
      // ...
    })),
  };
}
```

---

## Benefits

| Before (Tool-Based)                 | After (Injection Pattern)           |
| ----------------------------------- | ----------------------------------- |
| Agent calls `get_viewer_state` tool | Agent reads from system prompt      |
| Extra LLM round trip (1-3 seconds)  | Zero additional latency             |
| Agent might forget to check         | Context always available            |
| Token cost for tool call + response | Minimal token overhead (~20 tokens) |
| Agent narrates "Let me check..."    | Agent proceeds directly             |

### Latency Comparison

```
BEFORE:
User: "Highlight the binding site"
  → LLM decides to call get_viewer_state (500ms)
  → Tool executes (10ms)
  → LLM receives result, decides next action (500ms)
  → LLM calls structure_query (500ms)
  → Tool executes, returns
Total: ~1.5 seconds + actual work

AFTER:
User: "Highlight the binding site"
  → LLM sees [VIEWER STATE: CDK2...] in prompt
  → LLM calls structure_query directly (500ms)
  → Tool executes, returns
Total: ~500ms + actual work
```

---

## Design Considerations

### When to Use This Pattern

✅ **Good fit:**

- Visual state that changes infrequently (per-request, not real-time)
- State that's small enough to summarize concisely
- Context the agent needs for almost every decision
- Reducing latency is important

❌ **Poor fit:**

- Real-time state changes (would require WebSocket)
- Large state that would bloat the prompt
- State only needed occasionally (keep as a tool)
- State that requires complex queries to access

### Keeping the Tool Available

We keep `get_viewer_state` as a tool for cases where the agent needs more detail than the condensed summary provides:

- Full residue lists
- Exact atom counts
- Detailed chain composition
- Selection state details

The condensed summary handles 90% of cases; the tool handles edge cases.

### Token Budget

The condensed summary adds ~20-50 tokens to the system prompt. This is negligible compared to:

- A tool call round trip (~200-500 tokens for call + response)
- The base system prompt (~2000-4000 tokens)

---

## Generalizing the Pattern

This pattern applies to any agentic AI system where:

1. **The agent needs context from a separate UI component**
2. **That context changes between requests but not within a request**
3. **The context can be summarized concisely**

### Examples in Other Domains

| Domain              | UI State                        | Condensed Injection                               |
| ------------------- | ------------------------------- | ------------------------------------------------- |
| Code Editor Agent   | Open files, cursor position     | `[EDITOR: main.py L45, utils.py L12]`             |
| Data Analysis Agent | Loaded datasets, current chart  | `[DATA: sales.csv (1000 rows), bar chart active]` |
| Design Agent        | Canvas objects, selected layers | `[CANVAS: 3 shapes, Layer "Header" selected]`     |
| Document Agent      | Open document, cursor section   | `[DOC: "Q3 Report" Section 2.3]`                  |

### Implementation Template

```python
# 1. Define state structure
@dataclass
class UIState:
    # ... relevant fields

    def to_condensed(self) -> str:
        # Return single-line summary
        return f"[STATE: {summary}]"

# 2. Receive state with request
@app.post("/api/chat")
async def chat(request: ChatRequest):
    ui_state = parse_ui_state(request.ui_state)

# 3. Inject into system prompt
effective_prompt = f"{base_prompt}\n\n{ui_state.to_condensed()}"

# 4. Document in system prompt
"""
The current UI state is provided at the end of this prompt.
Look for [STATE: ...] to understand the current context.
"""
```

---

## Conclusion

The View State Injection Pattern solves a fundamental challenge in agentic AI systems with visual interfaces: giving the agent context about what the user sees without incurring the latency cost of a tool call. By sending UI state with each request and injecting a condensed summary into the system prompt, we achieve:

- **Lower latency**: No extra LLM round trips
- **Better reliability**: Agent always has context
- **Simpler reasoning**: Agent doesn't need to decide whether to check state
- **Minimal overhead**: ~20 tokens vs ~200+ for a tool call

This pattern is particularly valuable in multi-turn tool chaining architectures where each LLM call is expensive and latency compounds across iterations.
