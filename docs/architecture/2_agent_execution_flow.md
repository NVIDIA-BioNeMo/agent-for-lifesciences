# 2. Agent Execution Flow

The LangGraph StateGraph loop in full detail, from the `/api/chat` entrypoint through tool dispatch and state refresh.

## Agent Graph

```mermaid
flowchart TB
    subgraph entrypoint ["Entrypoint"]
        ChatEndpoint["POST /api/chat\nmain.py: stream_chat_with_langgraph()"]
        BuildGraph["Build graph components:\nChatAevomLLM, build_langchain_tools(),\nbuild_agent_graph(), SessionScopedStorage"]
    end

    subgraph langgraph_loop ["LangGraph StateGraph (build_agent_graph)"]
        AgentNode["agent node\n- _rotate_iteration_span(request_id)\n- state_modifier(state) injects:\n  VIEWER STATE, STORAGE STATE, SELECTIONS\n- llm_with_tools.ainvoke(modified_messages)"]
        ShouldContinue{"_should_continue\ntool_calls in\nlast message?"}
        ToolNode["tools node\n- restore_iteration_span(request_id)\n- ToolNode executes StructuredTool wrappers\n- viz events queued to _viz_event_queue"]
        AwaitPending["await_pending_ops node\n- drain_pending_ops(timeout=10s)\n- patch structure_query ToolMessages\n  with selection data from Redis"]
        RefreshState["refresh_state node\n- refresh_state_from_backends()\n- pull viewer_state from Redis\n- pull storage_state from PostgreSQL\n- pull selections from Redis"]
        ForceRespond["force_respond node\n- stub pending tool calls\n- inject nudge message\n- call LLM without tools"]
    end

    subgraph sse_bridge ["SSE Streaming Bridge"]
        StreamGraph["stream_graph()\n- ChatSpanContext (root span)\n- astream_events(version=v2)\n- maps LangGraph events to SSE"]
    end

    subgraph sse_events ["SSE Event Types"]
        Reasoning["reasoning: chain-of-thought tokens"]
        ToolStart["tool_call_start: {id, name, args, summary}"]
        ToolEnd["tool_call_end: {id, name, success, output_preview}"]
        ToolEvent["tool_event: {type, pdb/sdf/query data}"]
        StorageUpdate["storage_update: {artifacts}"]
        Content["content: final response tokens"]
        SpanId["span_id: root trace ID for feedback"]
    end

    ChatEndpoint --> BuildGraph
    BuildGraph --> StreamGraph
    StreamGraph --> AgentNode

    AgentNode --> ShouldContinue
    ShouldContinue -->|"tool_calls present\n(iteration < 20)"| ToolNode
    ShouldContinue -->|"no tool_calls"| Content
    ShouldContinue -->|"iteration >= 20"| ForceRespond
    ToolNode --> AwaitPending
    AwaitPending --> RefreshState
    RefreshState -->|"loop back"| AgentNode
    ForceRespond --> Content

    StreamGraph --> Reasoning
    StreamGraph --> ToolStart
    StreamGraph --> ToolEnd
    StreamGraph --> ToolEvent
    StreamGraph --> StorageUpdate
    StreamGraph --> SpanId
```

## Tool Dispatch

```mermaid
flowchart TB
    subgraph tool_dispatch ["ToolRegistry.execute()"]
        Lookup["Lookup builtin tool by name"]
        MCPFallback["MCP fallback:\nmcp_store.get_tool_by_name()"]
        Inject["Inject ContextVars:\nstorage, session_id, request_id"]
        Execute["tool.execute(arguments)"]
    end

    subgraph builtin_tools ["Builtin Tools (16 registered)"]
        subgraph bionemo_tools ["BioNeMo / NVIDIA API"]
            openfold3["openfold3\n(OpenFold3Tool)"]
            msa_search["msa_search\n(MSASearchTool)"]
        end

        subgraph container_tools ["Container Microservices"]
            molecule_info["molecule_info\n(RDKitMoleculeInfoTool)\n-> RDKit :8000"]
            python_execute["python_execute\n(PythonExecuteTool)\n-> PyExec :8000"]
            vina_dock["vina_dock\n(VinaDockTool)\n-> Vina :8000"]
        end

        subgraph external_tools ["External REST APIs"]
            fetch_protein_sequence["fetch_protein_sequence\n(UniProt)"]
            search_rcsb["search_rcsb\n(RCSB PDB)"]
            fetch_and_clean_pdb["fetch_and_clean_pdb\n(RCSB PDB)"]
            search_pubchem["search_pubchem\n(PubChem)"]
            get_pubchem_properties["get_pubchem_properties\n(PubChem)"]
            download_pubchem_sdf["download_pubchem_sdf\n(PubChem)"]
            lookup_ccd_code["lookup_ccd_code\n(PubChem/CCD)"]
        end

        subgraph viewer_tools ["Viewer / State"]
            visualize_pdb["visualize_pdb\n(VisualizePdbTool)"]
            visualize_sdf["visualize_sdf\n(VisualizeSdfTool)"]
            structure_query["structure_query\n(MolstarQueryTool)"]
            get_viewer_state["get_viewer_state\n(GetViewerStateTool)"]
        end
    end

    subgraph workflow_subgraphs ["Workflow Subgraph Tools (4)"]
        sm["run_small_molecule_analysis\n(PubChem, molecule_info, 3D coords, viz)"]
        sp["run_structure_prediction\n(fetch sequence, OpenFold3, visualize PDB)"]
        dk["run_docking_workflow\n(receptor prep, ligand prep, Vina, interpret)"]
        pl["run_protein_ligand_analysis\n(ligand ID, 3D structure, interactions)"]
    end

    subgraph mcp_tools ["MCP Dynamic Tools"]
        MCPServers["Per-session MCP servers\n(added via /api/mcp/servers)"]
    end

    Lookup -->|"found"| Inject
    Lookup -->|"not found"| MCPFallback
    MCPFallback -->|"found"| Inject
    Inject --> Execute

    Execute --> builtin_tools
    Execute --> workflow_subgraphs
    Execute --> mcp_tools
```

## State Modifier Detail

```mermaid
flowchart LR
    subgraph inputs ["AevomState Fields"]
        ViewerState["viewer_state\n(from Redis)"]
        StorageState["storage_state\n(from PostgreSQL)"]
        Selections["selections\n(from Redis registry)"]
        BasePrompt["base system prompt\n(prompts.md + Jinja2 template)"]
    end

    StateModifier["build_state_modifier()\ninject_state_into_prompt()"]

    subgraph output ["Augmented System Prompt"]
        Prompt["base prompt text\n+\nVIEWER STATE: structures, chains, residues\n+\nSTORAGE STATE: refs, types, descriptions\n+\nSELECTION STATE: refs, atom/residue counts"]
    end

    ViewerState --> StateModifier
    StorageState --> StateModifier
    Selections --> StateModifier
    BasePrompt --> StateModifier
    StateModifier --> output
```

## Source References

- `backend/api/main.py` -- `/api/chat` endpoint, `stream_chat_with_langgraph()`
- `backend/api/graph/agent.py` -- `build_agent_graph()`, 5 nodes (agent, tools, await_pending_ops, refresh_state, force_respond), `_should_continue` routing, iteration span management
- `backend/api/graph/state_modifier.py` -- `build_state_modifier()`, `inject_state_into_prompt()`, `format_viewer_state/storage_state/selections`
- `backend/api/graph/tools.py` -- `build_langchain_tools()`, StructuredTool wrappers, `_viz_event_queue`, pending ops
- `backend/api/graph/streaming.py` -- `stream_graph()`, LangGraph event-to-SSE mapping
- `backend/api/tools/registry.py` -- `ToolRegistry`, `_register_default_tools()` (16 builtin tools), MCP fallback
- `backend/api/graph/workflows/__init__.py` -- `build_workflow_tools()`, 4 subgraph tools
- `backend/api/graph/llm.py` -- `ChatAevomLLM` LangChain adapter
- `backend/api/llm/client.py` -- `LLMClient`, NVIDIA API calls
- `backend/api/llm/config.py` -- `ModelConfig`, model selection
