# 3. Frontend Components

React component tree, Molstar integration, and all data flows between browser and backend.

## Component Tree

```mermaid
flowchart TB
    subgraph app_root ["App.tsx (root)"]
        RouteCheck{"window.location.pathname\nstarts with /training?"}
    end

    RouteCheck -->|"yes"| TrainingMode
    RouteCheck -->|"no"| MainApp

    subgraph main_app ["MainApp (react-resizable-panels layout)"]
        subgraph left_panel ["Left Panel"]
            StorageStrip["StorageStrip\nHorizontal artifact bar\nicons by data_type\nclick to inject ref"]
            Chat["Chat\nfetch + ReadableStream SSE\nreact-markdown + remark-gfm\nrehype-highlight\n@ artifact autocomplete\nfeedback thumbs up/down"]
            Timeline["ToolCallTimeline\n(embedded in Chat messages)\nreasoning + tool_call items\nstreaming status indicators"]
        end

        subgraph right_panel ["Right Panel"]
            MolstarViewer["MolstarViewer\nMolstar createPluginUI\nWebGL 3D rendering\ndark skin SCSS"]
            StatusOverlay["Selection status overlay\natom/residue counts"]
        end

        subgraph overlays ["Slide-out Panels / Modals"]
            ToolManager["ToolManagerPanel\nbuiltin tools list\nMCP servers list\nsession ID copy"]
            AddMCPModal["AddMCPServerModal\nname + URL input\nconnect + discover tools"]
        end

        Header["Header bar\nmodel selector dropdown\nhamburger menu toggle"]
    end

    subgraph training_app ["TrainingMode"]
        TrainingMode["TrainingMode.tsx\nQuery list sidebar\nChat + Molstar + StorageStrip reuse\nEventSource /api/training/events\nPrompt mode selector\nFeedback integration"]
    end

    subgraph molstar_lib ["lib/molstar/"]
        Queries["queries.ts\nMolQL builders: chain, ligand,\nresidue, atom, backbone,\nsidechain, alphaCarbon,\nwithinDistance, polymer"]
        SelectionOps["selections.ts\nqueryStructure, highlightSelection,\nclearHighlights, selectAtoms,\nfocusOnSelection, measureDistance,\nmeasureAngle, measureDihedral,\ngetCenterOfMass, getBoundingBox,\ngetViewerState"]
        Types["types.ts\nQueryResult, ResidueInfo,\nAtomInfo, ViewerStateInfo,\nHighlightOptions, ChainSelector,\nResidueSelector, AtomSelector"]
    end

    MolstarViewer --> Queries
    MolstarViewer --> SelectionOps
    MolstarViewer --> Types
```

## Data Flows

```mermaid
sequenceDiagram
    participant User
    participant Chat as Chat.tsx
    participant App as App.tsx
    participant Molstar as MolstarViewer
    participant Strip as StorageStrip
    participant API as Aevom API

    Note over User,API: Chat Request Flow (fetch + ReadableStream)
    User->>Chat: type message + send
    Chat->>App: getState() from MolstarViewer ref
    Chat->>API: POST /api/chat {messages, viewer_state, session_id, model_id}
    API-->>Chat: SSE stream (data: JSON lines)

    loop Stream Processing
        Chat->>Chat: Parse SSE line
        alt reasoning event
            Chat->>Chat: Update timeline (reasoning item)
        else tool_call_start event
            Chat->>Chat: Add timeline item (pending)
        else tool_event (visualize_pdb/sdf)
            Chat->>App: onToolEvent callback
            App->>Molstar: loadPdb() or loadSdf()
        else tool_event (structure_query)
            Chat->>App: onToolEvent callback
            App->>Molstar: executeQuery() -> highlight/focus/select
            Molstar->>API: POST /api/selection {query_id, atoms, residues}
        else tool_call_end event
            Chat->>Chat: Update timeline (complete/error)
        else storage_update event
            Chat->>App: onStorageUpdate callback
            App->>Strip: Update artifacts prop
        else content event
            Chat->>Chat: Append to assistant message
        else span_id event
            Chat->>Chat: Store for feedback
        end
    end

    Note over User,API: Artifact Strip Flow
    App->>API: GET /api/storage/artifacts?session_id=...
    API-->>Strip: {artifacts: [...]}
    User->>Strip: Click artifact icon
    Strip->>Chat: onInjectRef(reference) -> insert @ref into input

    Note over User,API: MCP Tool Management
    User->>ToolManager: Add MCP server
    ToolManager->>API: POST /api/mcp/servers {name, url}
    API-->>ToolManager: {server with discovered tools}
    ToolManager->>API: GET /api/mcp/tools?session_id=...
    API-->>ToolManager: {builtin + MCP tools}

    Note over User,API: Training Mode (EventSource)
    User->>TrainingMode: Navigate to /training
    TrainingMode->>API: new EventSource(/api/training/events)
    API-->>TrainingMode: SSE: query_created, query_deleted, feedback_submitted
```

## Molstar Imperative API (MolstarViewerRef)

```mermaid
flowchart LR
    subgraph ref_api ["MolstarViewerRef (useImperativeHandle)"]
        subgraph loading ["Loading"]
            loadSdf["loadSdf(sdfData, label)"]
            loadPdb["loadPdb(pdbData, label)"]
            clearAll["clear()"]
        end

        subgraph querying ["Querying"]
            query["query(expression)"]
            getResidues["getResidues(expression)"]
            getAtoms["getAtoms(expression)"]
        end

        subgraph presets ["Preset Queries"]
            selectLigand["selectLigand()"]
            selectNearLigand["selectResiduesNearLigand(radius)"]
            selectChain["selectChain(chainId)"]
        end

        subgraph viz ["Visualization"]
            highlight["highlight(expression, options)"]
            clearHighlights["clearHighlights()"]
            focus["focus(expression)"]
            select["select(expression)"]
        end

        subgraph measurement ["Measurements"]
            measureDist["measureDistance(a, b)"]
            measureAng["measureAngle(a, b, c)"]
            measureDih["measureDihedral(a, b, c, d)"]
            getCOM["getCenterOfMass(expression)"]
            getBBox["getBoundingBox(expression)"]
        end

        subgraph state ["State"]
            getState["getState() -> ViewerStateInfo"]
        end
    end

    subgraph triggers ["Triggered By"]
        ToolEventSSE["tool_event SSE\n(from backend)"]
        AppCallback["App.tsx\nonToolEvent handler"]
    end

    ToolEventSSE --> AppCallback
    AppCallback --> loading
    AppCallback --> presets
    AppCallback --> viz
```

## Source References

- `frontend/src/App.tsx` -- root component, route dispatch (`/training*` vs MainApp), react-resizable-panels layout, tool event handler, session/model management
- `frontend/src/components/Chat.tsx` -- `fetch` + `ReadableStream` SSE consumer, `TimelineItem` type, markdown rendering, `@`-ref autocomplete, feedback
- `frontend/src/components/MolstarViewer.tsx` -- `MolstarViewerRef` imperative API (loadSdf, loadPdb, query, highlight, measure, getState), Molstar `createPluginUI`
- `frontend/src/components/ToolCallTimeline.tsx` -- renders `TimelineItem[]` with tool status icons and reasoning previews
- `frontend/src/components/StorageStrip.tsx` -- `Artifact` interface, icon mapping by `data_type`, tooltip, `onInjectRef` callback
- `frontend/src/components/ToolManagerPanel.tsx` -- MCP server listing, builtin tool listing, session management
- `frontend/src/components/AddMCPServerModal.tsx` -- MCP server registration form
- `frontend/src/components/TrainingMode.tsx` -- `EventSource` for `/api/training/events`, query management, prompt mode selection
- `frontend/src/lib/molstar/queries.ts` -- MolQL expression builders
- `frontend/src/lib/molstar/selections.ts` -- Molstar query execution, highlight/focus/measurement operations
- `frontend/src/lib/molstar/types.ts` -- shared TypeScript types for viewer state and query results
