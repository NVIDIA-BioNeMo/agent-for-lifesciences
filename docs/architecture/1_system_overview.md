# 1. System Overview

All deployable containers, databases, external APIs, and network boundaries.

```mermaid
flowchart TB

    subgraph internal ["Internal Network (Docker bridge / K8s namespace aevom)"]

        subgraph api_container ["Aevom API Container (FastAPI BFF :8000)"]
            StaticFiles["Static React Build"]
            ChatEndpoint["POST /api/chat (SSE)"]
            RestEndpoints["REST: /api/health, /api/info, /api/models,\n/api/session, /api/feedback, /api/jobs,\n/api/selection, /api/selections,\n/api/storage/*, /api/images/*"]
            SandboxRoutes["/api/sandbox/*"]
            TrainingRoutes["/api/training/*"]
            MCPRoutes["/api/mcp/*"]
            LangGraphAgent["LangGraph StateGraph Agent"]
            ToolRegistry["ToolRegistry"]
        end

        subgraph microservices ["Tool Microservices"]
            RDKit["RDKit Container :8000\nSMILES validation, molecule info,\nSDF generation, 2D depiction"]
            PyExec["PyExec Container :8000\nSandboxed Python execution\n(aevom client library)"]
            Vina["Vina Container :8000\nReceptor/ligand prep,\nAutodock Vina docking"]
        end

        subgraph data_layer ["Data Layer"]
            Postgres[("PostgreSQL 16\ndata_artifacts, jobs,\ntraining_queries")]
            Redis[("Redis 7\nviewer state, selections,\npending ops, MCP sessions,\nsandbox tokens")]
        end

        subgraph observability ["Observability"]
            Phoenix["Arize Phoenix\nOTLP gRPC :4317\nWeb UI :6006\nGraphQL API"]
        end
    end

    subgraph client ["Client Browser"]
        ReactSPA["React SPA (Vite + TypeScript)"]
        Molstar["Molstar 3D Viewer"]
    end

    subgraph external ["External APIs"]
        NVIDIA_LLM["NVIDIA LLM API\n(Nemotron, Mistral, etc.)"]
        OpenFold3["OpenFold3 API\n(BioNeMo)"]
        MSASearch["MSA Search API\n(BioNeMo)"]
        UniProt["UniProt REST API"]
        RCSB["RCSB PDB API"]
        PubChem["PubChem API"]
    end

    client -->|"HTTP / SSE"| api_container
    ChatEndpoint --> LangGraphAgent
    LangGraphAgent --> ToolRegistry

    ToolRegistry -->|"HTTP"| RDKit
    ToolRegistry -->|"HTTP"| PyExec
    ToolRegistry -->|"HTTP"| Vina
    ToolRegistry -->|"HTTPS"| NVIDIA_LLM
    ToolRegistry -->|"HTTPS"| OpenFold3
    ToolRegistry -->|"HTTPS"| MSASearch
    ToolRegistry -->|"HTTPS"| UniProt
    ToolRegistry -->|"HTTPS"| RCSB
    ToolRegistry -->|"HTTPS"| PubChem

    PyExec -->|"HTTP callback"| api_container

    api_container -->|"asyncpg"| Postgres
    api_container -->|"redis-py"| Redis
    api_container -->|"OTLP gRPC"| Phoenix

    RDKit -->|"PNG bytes, base64"| ToolRegistry

    Molstar -->|"POST /api/selection"| RestEndpoints
    ChatEndpoint -->|"SSE: tool_event"| Molstar
    ChatEndpoint -->|"SSE: content, reasoning,\ntool_call_*, storage_update"| ReactSPA
```

## Service Inventory

| Service      | Image / Build                                     | Ports           | Volumes                              | Health Check          |
| ------------ | ------------------------------------------------- | --------------- | ------------------------------------ | --------------------- |
| **backend**  | `backend/Dockerfile` (multi-stage: Node + Python), tagged `aevom-api:latest` | 5180:8000       | `prompts.md` (ro), `backend/api` (dev) | depends_on all others |
| **postgres** | `postgres:16-alpine`                              | 5432 (internal) | `postgres_data`                      | `pg_isready`          |
| **rdkit**    | `services/rdkit/Dockerfile`                       | 8000 (internal) | --                                   | `curl /health`        |
| **pyexec**   | `services/pyexec/Dockerfile`                      | 8000 (internal) | --                                   | `curl /health`        |
| **vina**     | `services/vina/Dockerfile`                        | 8000 (internal) | --                                   | `curl /health`        |
| **admet**    | `services/admet/Dockerfile`                       | 8000 (internal) | --                                   | `curl /ready`         |
| **plip**     | `services/plip/Dockerfile`                        | 8000 (internal) | --                                   | `curl /health`        |
| **redis**    | `redis:7-alpine`                                  | 6379            | --                                   | `redis-cli ping`      |
| **phoenix**  | `arizephoenix/phoenix:version-13.0.3`             | 6080:6006       | `phoenix_data`                       | --                    |

The Compose service is `backend`, the image it builds is `aevom-api:latest`. The
image name matches the registry and the Helm chart; the service name is what the
`aevom-backend-1` container name and the Makefile's health gates derive from.

## Volumes

| Volume          | Purpose                     | Mounted By |
| --------------- | --------------------------- | ---------- |
| `postgres_data` | PostgreSQL data persistence | postgres   |
| `phoenix_data`  | Trace storage               | phoenix    |

No volume is shared between services. 2D molecule depictions used to live on a
`shared_images` volume, which in Kubernetes meant a ReadWriteMany claim; RDKit
now returns PNG bytes inline and the API stores them as `png` artifacts in
Postgres, served by `GET /api/images/{reference}`. That removed the only
correctness trap in the storage layout — RDKit runs behind an HPA, so a file
written by one replica was invisible to the others.

## Network

- Docker Compose: bridge network `192.168.249.0/24`
- Kubernetes: namespace `aevom`, NGINX Ingress at the chart's `ingress.host`

## Source References

- `docker-compose.yml` -- every service, 2 volumes, bridge network
- `deploy/helm/aevom/` -- the Helm chart; `values.yaml` is the deployment contract
- `deploy/helm/README.md` -- install, values table, cutover runbook
- `backend/api/main.py` -- all `/api/*` endpoints, router mounts, lifespan
- `backend/api/tools/registry.py` -- `ToolRegistry` dispatches to microservices and external APIs
- `services/rdkit/`, `services/pyexec/`, `services/vina/`, `services/admet/`, `services/plip/` -- microservice implementations
