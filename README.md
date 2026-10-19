# Aevom

Aevom is a domain-specific harness for agentic molecular science. It lets a frontier language model execute multi-step computational chemistry and structural biology workflows while molecular data, scientific state and tool execution remain outside the model. 

> **Not for production use.** Aevom is research-grade reference software, a
> developer example of building scientific agents on NVIDIA's agent stack. It has
> no user authentication, and it runs model-generated code. Keep it off the public
> internet and read [SECURITY.md](./SECURITY.md) before you deploy it.

See [docs/architecture/](./docs/architecture/1_system_overview.md) for the system architecture.

## Quick Start

### Prerequisites

| Requirement                    | Minimum Version | Notes                                                                                                          |
| ------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------------- |
| Docker                         | 20.10+          | Docker Desktop recommended                                                                                     |
| Docker Compose                 | 2.0+            | Included with Docker Desktop                                                                                   |
| NVIDIA API Key (build)         | —               | Save to `./API_KEY_NV`. From [build.nvidia.com](https://build.nvidia.com/). Required by the **default** model `nemotron-3-5-super-vl` plus `nemotron-3-5-lightning`, `nemotron-3-ultra`, `nemotron-3-super-120b`, `nemotron-3-nano-omni`, `muse-glimmer`, `gpt-oss-20b`, `glm-5-3`, `glm-5-3-flash`. |
| NVIDIA API Key (inference)     | —               | **NVIDIA-internal only.** Save to `./API_KEY_NVINF`. From [inference.nvidia.com](https://inference.nvidia.com/). Required by `gemini-3-pro`, `claude-opus-4-6`, `gpt-5-1`, `gpt-6-astra`, `nemotron-3-super-dynamo`, `nemotron-3-super-rc-nim`, `nemotron-3-ultra-nvinf`. |
| Anthropic API Key              | —               | Optional. Save to `./API_KEY_ANTHROPIC`. From [console.anthropic.com](https://console.anthropic.com/). Adds the newest Claude model of each family to the model dropdown. |
| OpenAI API Key                 | —               | Optional. Save to `./API_KEY_OPENAI`. From [platform.openai.com](https://platform.openai.com/api-keys). Adds the newest GPT Astra, Sol and Luna models to the model dropdown, and GPT Rosalind with `FEAT_ROSALIND=TRUE`. |
| Disk Space                     | 5GB+            | RDKit build requires significant space                                                                         |

**Verify your setup:**

```bash
make check
```

This runs a prerequisite check that validates your environment and provides helpful installation instructions if anything is missing.

### Setup Steps

1. **Clone the repository with its submodules**

   ```bash
   git clone --recursive https://github.com/NVIDIA-BioNeMo/agent-for-lifesciences.git
   cd agent-for-lifesciences
   ```

   `--recursive` also checks out `third_party/bionemo-agent-toolkit`, the
   pinned submodule that supplies the NIM skill instructions (RFdiffusion,
   ProteinMPNN, GenMol, MSA search). Without it the stack still starts, but
   those skills silently drop out of the catalog and the model never learns
   that the NIM tools exist.

   If you forgot `--recursive`, there is no need to clone again. From the
   repository root, run:

   ```bash
   make submodules
   ```

   `make check` reports a missing submodule, so you will be reminded before
   anything builds.

2. **Get the NVIDIA API key(s) for your chosen model**

   Aevom uses two distinct NVIDIA portals depending on the model. The
   [Model Selection](#model-selection) table below tells you which key file
   each model needs. The **default model (`nemotron-3-5-super-vl`) requires
   `API_KEY_NV`**, so one build.nvidia.com key is enough to get started.

   - **`API_KEY_NV`** (build.nvidia.com) — required by the default model and
     all other `integrate.api` models:
     - Visit [build.nvidia.com](https://build.nvidia.com/)
     - Create an account and generate an API key
     - Save it to `./API_KEY_NV`:
       ```bash
       echo 'your-nvidia-build-api-key' > API_KEY_NV
       ```

   - **`API_KEY_NVINF`** (inference.nvidia.com) — **NVIDIA-internal only.**
     That endpoint is not reachable from the public internet, so skip this
     unless you are on the NVIDIA network and set `AEVOM_MODEL` to an
     `inference-api` model:
     - Visit [inference.nvidia.com](https://inference.nvidia.com/)
     - Create an account and generate an API key
     - Save it to `./API_KEY_NVINF`:
       ```bash
       echo 'your-nvidia-inference-api-key' > API_KEY_NVINF
       ```

   - **`API_KEY_ANTHROPIC`** (api.anthropic.com) — optional. With it, Aevom
     lists the models your key can use and offers the newest of each Claude
     family (Opus, Sonnet, Haiku, ...) in the model dropdown:
     - Visit [console.anthropic.com](https://console.anthropic.com/)
     - Create an API key
     - Save it to `./API_KEY_ANTHROPIC`:
       ```bash
       echo 'your-anthropic-api-key' > API_KEY_ANTHROPIC
       ```

   - **`API_KEY_OPENAI`** (api.openai.com) — optional. With it, Aevom offers
     the newest GPT Astra, Sol and Luna models your key can use. The GPT
     Rosalind life-sciences models stay hidden unless you set
     `FEAT_ROSALIND=TRUE` (in `.env` or your shell) before starting the stack:
     - Visit [platform.openai.com](https://platform.openai.com/api-keys)
     - Create an API key
     - Save it to `./API_KEY_OPENAI`:
       ```bash
       echo 'your-openai-api-key' > API_KEY_OPENAI
       ```

   You need at least one key file. An Anthropic or OpenAI key on its own is
   enough: Aevom starts on the best Claude (or, failing that, GPT) model and
   disables the NIM-backed tools (OpenFold3, Boltz-2 and the rest), which need
   `API_KEY_NV`.

   `make check` creates empty placeholders for the key files you skipped, then
   validates the ones you populated before any container starts. An empty key
   file means "not configured".

   If you add or change a key while the stack is running, recreate the backend
   so it sees the new file:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.openshell.yml up -d --force-recreate backend      # make up
   docker compose -f docker-compose.dev.yml -f docker-compose.openshell.yml up -d --force-recreate backend  # make dev
   ```

   Leave out `-f docker-compose.openshell.yml` if you started the stack with
   `PYEXEC_MODE=direct`. Without it, a stack started normally loses its
   OpenShell settings and `python_execute` stops working.

3. **Build and Run**

   ```bash
   make up
   ```

   The first build takes 20-30 minutes as RDKit compiles from source. Subsequent builds are fast due to Docker layer caching.

### macOS (Apple Silicon)

Aevom runs on Apple Silicon Macs under Docker Desktop. Every image builds
natively for `linux/arm64`, so there is no Rosetta emulation and nothing to
patch, but four things differ from Linux:

1. **Install the Xcode Command Line Tools** first. They provide `git`, `make`
   and `python3`, which `make check` needs:
   ```bash
   xcode-select --install
   ```
2. **Give Docker Desktop at least 8 GB of memory** (Settings > Resources >
   Memory limit). The default is half your Mac's RAM. The RDKit build sizes its
   parallelism to the memory available, so less memory still works, but the
   first build takes longer than on Linux. More memory makes it faster; you
   can also set the job count by hand with
   `docker compose build --build-arg BUILD_JOBS=4 rdkit`. If the RDKit build
   stops with `cannot allocate memory` or `Killed`, rerun it with
   `BUILD_JOBS=2`.
3. **Keep the checkout under your home directory.** Docker Desktop only shares
   `/Users`, `/Volumes`, `/private`, `/tmp` and `/var/folders` with containers
   by default; anywhere else, add the path under Settings > Resources > File
   sharing.
4. **CPU only.** There is no NVIDIA GPU, so the `*-gpu` targets refuse to run;
   use `make up` / `make dev`. Docking runs on the CPU build of AutoDock Vina.

`make check` covers all four: it confirms the Command Line Tools run, reports
the Docker VM's memory, warns if the project is outside the shared paths, and
flags host ports another process already holds (Homebrew's redis on 6379 is the
usual one: `brew services stop redis`, or set `REDIS_HOST_PORT` in `.env`).

**From a source tarball** (no `.git` directory), populate the skill submodule
with `make submodules`, which downloads the pinned commit, then continue as
above.

### Model Selection

Set `AEVOM_MODEL` to choose the active LLM. Default is `nemotron-3-5-super-vl`.

```bash
# In .env or inline
AEVOM_MODEL=gemini-3-pro make dev
```

| `AEVOM_MODEL`             | Model                                        | Endpoint        | Reasoning | Key File       |
| ------------------------- | -------------------------------------------- | --------------- | --------- | -------------- |
| `nemotron-3-5-super-vl` *(default)* | Nemotron 3.5 Super VL 120B A12B    | integrate.api   | Yes       | `API_KEY_NV`   |
| `nemotron-3-5-lightning`  | Nemotron 3.5 Lightning 30B A3B               | integrate.api   | Yes       | `API_KEY_NV`   |
| `nemotron-3-ultra`        | Nemotron 3 Ultra 550B A55B                   | integrate.api   | Yes       | `API_KEY_NV`   |
| `nemotron-3-super-120b`   | Nemotron 3 Super 120B A12B                   | integrate.api   | Yes       | `API_KEY_NV`   |
| `nemotron-3-nano-omni`    | Nemotron 3 Nano Omni 30B A3B                 | integrate.api   | Yes       | `API_KEY_NV`   |
| `muse-glimmer`            | Muse Glimmer 30B                             | integrate.api   | Yes       | `API_KEY_NV`   |
| `gpt-oss-20b`             | GPT-OSS 20B                                  | integrate.api   | Yes       | `API_KEY_NV`   |
| `glm-5-3`                 | GLM-5.3                                      | integrate.api   | Yes       | `API_KEY_NV`   |
| `glm-5-3-flash`           | GLM-5.3 Flash                                | integrate.api   | Yes       | `API_KEY_NV`   |
| `gemini-3-pro`            | Gemini 3 Pro                                 | inference-api   | Thinking* | `API_KEY_NVINF`|
| `claude-opus-4-6`         | Claude Opus 4.6 (Bedrock proxy)              | inference-api   | No        | `API_KEY_NVINF`|
| `gpt-5-1`                 | GPT 5.1 (Azure proxy)                        | inference-api   | No        | `API_KEY_NVINF`|
| `gpt-6-astra`             | GPT-6 Astra (Azure proxy)                    | inference-api   | No        | `API_KEY_NVINF`|
| `nemotron-3-super-dynamo` | Nemotron 3 Super Dynamo                      | inference-api   | Yes**     | `API_KEY_NVINF`|
| `nemotron-3-super-rc-nim` | Nemotron 3 Super RC NIM                      | inference-api   | Yes**     | `API_KEY_NVINF`|
| `nemotron-3-ultra-nvinf`  | Nemotron 3 Ultra                             | inference-api   | Yes       | `API_KEY_NVINF`|
| `anthropic/<model id>`    | Discovered from your key, e.g. `anthropic/claude-sonnet-5-5` | api.anthropic.com | Yes | `API_KEY_ANTHROPIC` |
| `openai/<model id>`       | Discovered from your key, e.g. `openai/gpt-6.1-sol` | api.openai.com | Yes | `API_KEY_OPENAI` |

**Endpoint legend:** `integrate.api` = integrate.api.nvidia.com (public),
`inference-api` = inference-api.nvidia.com (**NVIDIA-internal**, not reachable
from the public internet).

The *(default)* marker is the Docker Compose default. The Helm chart and the
k8s base manifests set `nemotron-3-ultra` instead; see
[deploy/helm/README.md](deploy/helm/README.md).

The two Ultra entries are the same weights behind different gateways, and they
are *not* interchangeable: `nemotron-3-ultra` needs thinking switched on
explicitly, `nemotron-3-ultra-nvinf` reasons by default, and the two sample
differently. Benchmark numbers do not transfer between them. The eval suite and
the production cluster pin `nemotron-3-ultra-nvinf`.

Do not send `reasoning_budget` to `nemotron-3-ultra`. The integrate.api gateway
refuses the parameter by failing the request, and reports the refusal as an
in-stream 503 "Service temporarily overloaded" that looks like a capacity
problem and survives every retry.

\*Gemini 3 Pro uses thought signatures rather than readable reasoning content. \*\*Nemotron 3 Super models produce clear-text reasoning in the content field (before `</think>`) rather than using the `reasoning_content` protocol field; the backend parses and extracts this automatically.

Models using `API_KEY_NV` need the key file at the project root. Models using `API_KEY_NVINF` need a separate inference-API key in `API_KEY_NVINF`. See [deploy/helm/README.md](deploy/helm/README.md) for Kubernetes secret setup.

### DevContainer Setup (Optional)

The DevContainer provides a pre-configured development environment with Python 3.11 and Node.js 20. It's **optional**—you can run Aevom with just Docker.

**When to use DevContainer:**

- Consistent environment across team members
- Avoid Python/Node version conflicts on your machine
- Pre-configured VSCode extensions and settings

**When DevContainer is NOT needed:**

- Running Aevom as a demo (just need Docker)
- You already have Python 3.11+ and Node.js 20+ locally

#### GPU Requirements (DevContainer Only)

The DevContainer runs with `--gpus all` for GPU-accelerated workloads. You'll need:

| Requirement              | Notes                                      |
| ------------------------ | ------------------------------------------ |
| NVIDIA GPU               | Any CUDA-capable GPU                       |
| NVIDIA Driver            | 525+ recommended (check with `nvidia-smi`) |
| nvidia-container-toolkit | Enables Docker GPU access                  |

**Install nvidia-container-toolkit (Linux):**

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify GPU access works:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU info displayed. If this fails, the DevContainer will not start.

#### DevContainer Environment Setup

If using DevContainer, create a `.env` file with the host project path:

```bash
echo "HOST_PROJECT_PATH=/path/to/aevom/on/host" > .env
```

This is required because docker-outside-of-docker needs absolute host paths for volume mounts.

### Build & Run Commands

```bash
# Verify prerequisites (includes optional GPU check)
make check

# See all available commands
make help

# Build and start production stack
make up

# Build and start development stack (with hot-reload)
make dev

# python_execute runs in an OpenShell sandbox per chat session by default;
# add PYEXEC_MODE=direct to any start target for the shared pyexec service
make dev PYEXEC_MODE=direct

# Run the backend test suite natively in the devcontainer (~30s)
make test-fast

# Run the full Python test suite (adds the container-only suites)
make test

# Stop all containers
make down

# View logs
make logs

# Clean rebuild from scratch
make rebuild
```

### Vina GPU Acceleration (Optional)

AutoDock Vina can use GPU acceleration for significantly faster molecular docking. This requires the same GPU prerequisites as the DevContainer.

**Default behavior:** `make up` and `make dev` build and run Vina in **CPU mode**.

**GPU commands:**

```bash
# Build with GPU support
make build-gpu

# Start production stack with Vina GPU
make up-gpu

# Start development stack with Vina GPU
make dev-gpu
```

**Requirements for Vina GPU:**

- NVIDIA GPU with CUDA support, on Linux (x86_64 or arm64)
- NVIDIA driver 560+ on x86_64, 575+ on arm64
- nvidia-container-toolkit configured

The GPU build compiles Uni-Dock 1.2.0 from source (the release publishes no
binaries), which adds a few minutes to the first `make build-gpu`.

**arm64 (DGX Spark, GH200):** the same `make build-gpu` / `make dev-gpu`
targets work. Uni-Dock and the nvMolKit conformer worker are built with CUDA
12.9, for sm_90, sm_100 and sm_120/121. A first `make build-gpu` on a DGX Spark
takes about 15 minutes, RDKit included.
Apple Silicon Macs have no NVIDIA GPU, so the GPU targets refuse to run there.

Run `make check` to verify GPU support is available. The GPU check will show whether your system is ready for `make up-gpu` or `make dev-gpu`.

**How it works:** GPU mode uses `docker-compose.gpu.yml` as an override file that:

- Builds the `gpu-final` Dockerfile target (CUDA-enabled Vina)
- Allocates GPU resources to the Vina container

### Access Points

**Production mode (`make up`):**

| Service         | URL                        |
| --------------- | -------------------------- |
| Aevom App       | http://localhost:5180      |
| FastAPI Docs    | http://localhost:5180/docs |
| Phoenix Tracing | http://localhost:6080      |

**Development mode (`make dev`):**

| Service         | URL                        |
| --------------- | -------------------------- |
| Aevom App       | http://localhost:5180      |
| FastAPI Backend | http://localhost:8080      |
| FastAPI Docs    | http://localhost:8080/docs |
| Phoenix Tracing | http://localhost:6080      |

## Troubleshooting

### "docker-compose: command not found" or "no target to make docker-compose"

The Makefile auto-detects whether you have `docker compose` (plugin) or `docker-compose` (standalone). If neither works:

- **Docker Desktop users**: Ensure Docker Desktop is running. The Compose plugin is included.
- **Linux Docker Engine users**: Install the Compose plugin:
  ```bash
  sudo apt-get update
  sudo apt-get install docker-compose-plugin
  ```

### Docker daemon not running

Start Docker:

- **Docker Desktop**: Open the Docker Desktop application
- **Linux**: `sudo systemctl start docker`

### Permission denied on Docker socket

Add your user to the docker group:

```bash
sudo usermod -aG docker $USER
```

Then log out and back in for changes to take effect.

### Build fails with disk space error

The RDKit container build requires approximately 5GB of disk space. Free up space and try again:

```bash
# Clean up unused Docker resources
docker system prune -a
```

### API_KEY_NV file issues

Ensure the file exists and contains your NVIDIA API key:

```bash
# Check if file exists and has content
cat API_KEY_NV
```

If empty or missing, get a key from [NVIDIA Build](https://build.nvidia.com/).

### DevContainer fails to start (GPU issues)

If the DevContainer fails with GPU-related errors:

1. **Verify NVIDIA driver is installed:**

   ```bash
   nvidia-smi
   ```

   If this fails, install the NVIDIA driver for your GPU.

2. **Verify nvidia-container-toolkit is installed:**

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

   If this fails with "could not select device driver", install nvidia-container-toolkit (see [GPU Requirements](#gpu-requirements-devcontainer-only)).

3. **Docker Desktop users (Windows/Mac):** GPU passthrough requires WSL2 (Windows) and is not supported on any Mac (see [macOS (Apple Silicon)](#macos-apple-silicon)). Check [NVIDIA Container Toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your platform.

4. **After installing nvidia-container-toolkit, restart Docker:**

   ```bash
   sudo systemctl restart docker
   ```

## Architecture

```text
┌─────────────────────────────────────── DEPLOYMENT / RUNTIME TRACK ─────────────────────────────────────────┐
│                                                                                                            │
│  User / Browser                                                                                            │
│  [frontend chat + Molstar]                                                                                 │
│          │ POST /api/chat                                                                                  │
│          v                                                                                                 │
│  ┌──────────────────────────────────────────┐                                                              │
│  │ Aevom API + SSE entrypoint               │                                                              │
│  │ file: backend/api/main.py                │                                                              │
│  │ - stream_chat_with_langgraph()           │                                                              │
│  └──────────────────────────────────────────┘                                                              │
│          │ builds graph + tools + prompt                                                                   │
│          v                                                                                                 │
│  ┌──────────────────────────────────────────┐        calls complete()/stream()     ┌──────────────────────┐│
│  │ LangGraph loop                           │------------------------------------->│ Main LLM providers   ││
│  │ file: backend/api/graph/agent.py         │                                      │ file: backend/api/   ││
│  │ agent -> tools -> await_pending_ops      │<-------------------------------------│ llm/config.py        ││
│  │ -> refresh_state -> agent                │  model + tool-call responses         │ + llm/client.py      ││
│  └──────────────────────────────────────────┘                                      │ (Nemotron, GPT-5.1)  ││
│          │ executes tools via wrappers                                             └──────────────────────┘│
│          v                                                                                                 │
│  ┌──────────────────────────────────────────┐                                                              │
│  │ Tool wrapper layer                       │                                                              │
│  │ file: backend/api/graph/tools.py         │                                                              │
│  │ StructuredTool -> ToolRegistry.execute() │                                                              │
│  └──────────────────────────────────────────┘                                                              │
│          │                                                                                                 │
│          v                                                                                                 │
│  ┌───────────────────────────────────────────────┐                                                         │
│  │ Shared ToolRegistry                           │                                                         │
│  │ file: backend/api/tools/registry.py           │                                                         │
│  │ builtin tools + MCP fallback routing          │                                                         │
│  └───────────────┬───────────────────────┬───────┘                                                         │
│                  │                       │                                                                 │
│                  │                       │                                                                 │
│                  v                       v                                                                 │
│  ┌────────────────────────────────┐   ┌──────────────────────────────────────┐                             │
│  │ BioNeMo tool path              │   │ MCP dynamic tool path                │                             │
│  │ file: backend/api/tools/       │   │ files: backend/api/mcp/session_      │                             │
│  │ openfold.py                    │   │ store.py + client.py + adapter.py    │                             │
│  │ OpenFold3 API                  │   │ add/list/remove servers via routes.py│                             │
│  └────────────────────────────────┘   └──────────────────────────────────────┘                             │
│                                                                                                            │
│  ┌───────────────────────────────────────────────┐                                                         │
│  │ LangGraph event -> SSE bridge                 │                                                         │
│  │ file: backend/api/graph/streaming.py          │                                                         │
│  │ emits: reasoning, tool_call_start/end,        │                                                         │
│  │ tool_event, storage_update, content           │                                                         │
│  └───────────────────────────────────────────────┘                                                         │
│          │                                                                                                 │
│          v                                                                                                 │
│  User / Browser receives streamed chat + tool timeline + Molstar events                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


┌───────────────────────────────────────── TRAINING TRACK (GRPO) ────────────────────────────────────────────┐
│                                                                                                            │
│  Offline dataset prep                                                                                      │
│  ┌──────────────────────────────────────────────────┐                                                      │
│  │ CDD -> NeMo-Gym JSONL adapter                    │                                                      │
│  │ file: backend/api/training/dataset_adapter.py    │                                                      │
│  └──────────────────────────────────────────────────┘                                                      │
│          │                                                                                                 │
│          v                                                                                                 │
│  ┌──────────────────────────────────────────────────┐                                                      │
│  │ NeMoRL GRPO trainer                              │                                                      │
│  │ file: ../nemo-rl/examples/nemo_gym/              │                                                      │
│  │ run_grpo_nemo_gym.py                             │                                                      │
│  └──────────────────────────────────────────────────┘                                                      │
│          │ should_use_nemo_gym=true                                                                        │
│          v                                                                                                 │
│  ┌──────────────────────────────────────────────────┐                                                      │
│  │ NeMo-Gym simple agent loop                       │                                                      │
│  │ file: ../nemo-gym/responses_api_agents/          │                                                      │
│  │ simple_agent/app.py                              │                                                      │
│  │ /run -> /seed_session -> /{tool_name} -> /verify |                                                      │
│  └──────────────────────────────────────────────────┘                                                      │
│          │ HTTP                                                                                            │
│          v                                                                                                 │
│  ┌──────────────────────────────────────────────────┐                                                      │
│  │ Aevom NeMo-Gym resource server                   │                                                      │
│  │ file: training/nemo_gym_wrapper/app.py           │                                                      │
│  │ seed_session(), execute_tool(), verify()         │                                                      │
│  └───────────────┬───────────────────────┬──────────┘                                                      │
│                  │                       │                                                                 │
│                  v                       v                                                                 │
│  ┌────────────────────────────────┐   ┌──────────────────────────────────────┐                             │
│  │ Shared ToolRegistry reuse      │   │ Reward logic + annotations           │                             │
│  │ file: backend/api/tools/       │   │ file: backend/api/training/rewards.py│                             │
│  │ registry.py                    │   │ build_reward_annotations()           │                             │
│  └────────────────────────────────┘   └──────────────────────────────────────┘                             │
│                                                                                                            │
│  Note: Training bypasses LangGraph, but reuses the same ToolRegistry contract as runtime.                  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Container Services

| Container  | Purpose                                          |
| ---------- | ------------------------------------------------ |
| `frontend` | React app + FastAPI backend                      |
| `postgres` | PostgreSQL database for data artifacts           |
| `rdkit`    | RDKit cheminformatics microservice               |
| `pyexec`   | Sandboxed Python execution (numpy, scipy, rdkit) |
| `phoenix`  | OpenTelemetry trace collector and UI             |

## Development

The dev container includes Python 3.11 and Node.js 20. All chemistry tools run in microservice containers.

```bash
# Start dev environment with hot-reload
make dev

# Frontend changes: instant hot-reload via Vite
# Backend changes: auto-reload via uvicorn --reload
# PyExec changes: auto-reload via volume mount
```

## Testing

There are two paths, and picking the right one matters for how fast you iterate.

### Fast inner loop (no Docker)

The backend suite — roughly 780 of the repo's ~800 tests — needs no containers.
The devcontainer image ships pytest and the backend dependencies, so run it
directly:

```bash
make test-fast        # whole backend suite, ~30s
```

For a single file or a single test, call `pytest` directly. The root
[pytest.ini](pytest.ini) puts `backend/` on `sys.path`, so this works from any
directory:

```bash
pytest backend/api/tests/test_registry.py
pytest backend/api/tests/test_registry.py -k some_case -x
```

Use this while developing. `make test-fast` preflights the environment
([scripts/check-devcontainer.sh](scripts/check-devcontainer.sh)) and stops with
instructions if you are on the host — where there is no pytest — rather than
failing with a confusing `ModuleNotFoundError`.

Do not `pip install` pytest. If it is missing, the devcontainer image is stale
and needs a rebuild (**Dev Containers: Rebuild Container**); installing into a
running container does not survive one.

### Full gate (containers)

Before pushing, run everything:

```bash
make test
```

The remaining three suites genuinely need their images: `vina` shells out to the
`vina` binary, `pyexec` needs the sandbox's RDKit stack, and `training` needs
`nemo_gym`. Each suite executes inside the container image that already ships
its dependencies. The runner ([scripts/run-tests.sh](scripts/run-tests.sh))
builds the test images (reusing the cache from `make build-dev`) and runs each
suite via `docker compose -f docker-compose.test.yml run`:

| Suite      | Image            | Tests                                                      |
| ---------- | ---------------- | ---------------------------------------------------------- |
| `backend`  | backend          | `backend/api/tests`, `api/tools/tests`, `api/training/tests` |
| `vina`     | vina (`cpu-final`) | `services/vina/tests`                                    |
| `pyexec`   | pyexec           | `services/pyexec/tests`                                    |
| `training` | `aevom-training` | `training/tests` (NeMo-Gym agent suite)                    |

Run a single suite via its `make` target:

```bash
make test-backend    # backend image suite
make test-vina       # vina image suite
make test-pyexec     # pyexec image suite
make test-training   # NeMo-Gym suite (aevom-training image)
```

Or pass suite names directly to the runner (supports arbitrary subsets):

```bash
./scripts/run-tests.sh backend          # one suite
./scripts/run-tests.sh vina pyexec       # a subset
```

The `training` suite runs only when the `aevom-training` image is available
locally; otherwise it is reported as **skipped** (it does not fail the run).
That image is built separately via
`training/slurm/container/build_from_source.sh` (typically on a GPU host). To
fetch it from the registry before running, pass `--pull`:

```bash
./scripts/run-tests.sh --pull training
```

> First run on a fresh devcontainer builds the service images from scratch
> (the RDKit image can take 20-30 minutes); subsequent runs reuse the cache.

## Training Mode (DPO Data Collection)

Aevom includes a training mode for collecting Direct Preference Optimization (DPO) data—pairs of "chosen" (good) and "rejected" (bad) responses for fine-tuning.

**After running `make dev`, open two browser tabs:**

| Mode   | URL                                        | Purpose                               |
| ------ | ------------------------------------------ | ------------------------------------- |
| Oracle | http://localhost:5180/training?mode=oracle | Generates good responses (thumbs up)  |
| Weak   | http://localhost:5180/training?mode=weak   | Generates bad responses (thumbs down) |

**Workflow:**

1. Click a query in the Oracle tab → review → 👍 if good
2. Click the same query in Weak tab → review → 👎 if bad
3. Sidebar tracks pair completion (need 1 up + 1 down per query)

**Export the dataset:**

```bash
python3 scripts/export_training_data.py
```

**API endpoints** for managing training queries:

```bash
# List queries with status
curl http://localhost:8080/api/training/queries

# Add a new query
curl -X POST http://localhost:8080/api/training/queries \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What is the molecular weight of aspirin?", "expected_tool": "python_execute"}'
```

## Documentation

- [docs/architecture/](./docs/architecture/1_system_overview.md) - System architecture
- [multi_turn_tool_chaining.md](./docs/multi_turn_tool_chaining.md) - Tool chaining implementation
- [data_storage_pattern.md](./docs/data_storage_pattern.md) - Data reference pattern
- [python_sandbox_integration.md](./docs/python_sandbox_integration.md) - Python sandbox for LLM-generated code
- [view_injection_pattern.md](./docs/view_injection_pattern.md) - Viewer state injection pattern
- [telemetry.md](./docs/telemetry.md) - Observability setup

## Tips

To download stored molecules and structures, see the
[result storage and analysis guide](examples/storage-access/README.md) for Python
and curl examples, including SDF/PDB files and CSV exports.

## Licensing

Aevom's NVIDIA-authored code is licensed under [Apache-2.0](LICENSE).
Contributions require [DCO sign-off](CONTRIBUTING.md#signing-your-work).
Third-party components retain their own licenses; see [NOTICE](NOTICE).

The pinned [BioNeMo Agent Toolkit](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit)
submodule is at `third_party/bionemo-agent-toolkit`. Its source code is
Apache-2.0, and its skills, documentation, and content additionally carry
CC-BY-4.0 terms. See its [licensing overview](third_party/bionemo-agent-toolkit/LICENSE),
[Apache-2.0 text](third_party/bionemo-agent-toolkit/LICENSE-APACHE-2.0), and
[CC-BY-4.0 text](third_party/bionemo-agent-toolkit/LICENSE-CC-BY-4.0).

The toolkit contains two vendored copies of Google DeepMind's science-skills,
each licensed under Apache-2.0:

- [Plugin copy license](third_party/bionemo-agent-toolkit/plugins/bionemo-agent-toolkit/skills/complexa-binder-design/vendor/science-skills/LICENSE).
- [Workflow copy license](third_party/bionemo-agent-toolkit/workflows/generative_protein_binder_design/complexa-binder-design/vendor/science-skills/LICENSE).

We retain the toolkit's four per-skill Apache-2.0 license files:

- [complexa-binder-design plugin license](third_party/bionemo-agent-toolkit/plugins/bionemo-agent-toolkit/skills/complexa-binder-design/LICENSE).
- [protein-binder-design plugin license](third_party/bionemo-agent-toolkit/plugins/bionemo-agent-toolkit/skills/protein-binder-design/LICENSE).
- [complexa-binder-design workflow license](third_party/bionemo-agent-toolkit/workflows/generative_protein_binder_design/complexa-binder-design/LICENSE).
- [protein-binder-design workflow license](third_party/bionemo-agent-toolkit/workflows/generative_protein_binder_design/protein-binder-design/LICENSE).

These directories are self-contained skills that can be distributed or copied
individually. Keeping their upstream license files preserves the license text
when a skill is separated from the toolkit or Aevom; the repetition does not
change their licensing. We also preserve the science-skills licenses and all
upstream attribution notices.

`make submodules` applies the tracked [toolkit copyright-header correction](third_party/patches/bionemo-agent-toolkit-prune-orphans-copyright.patch)
to `.github/scripts/prune-orphans.sh`, preserving its Apache-2.0 license and
upstream provenance. The upstream commit pin remains unchanged. Source-tarball
and Git checkouts apply the same patch and require the `patch` command.

## Notice and Disclaimer

This software automatically retrieves, accesses or interacts with external materials.  Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.

