# 5. Training Pipeline

End-to-end GRPO training flow across three repositories: `aevom-data-gen`, `aevom`, `nemo-gym`, and `nemo-rl`.

## Training Architecture

```mermaid
flowchart TB
    subgraph data_gen ["Dataset Generation (aevom-data-gen repo)"]
        CDD["CDD Dataset Generator\nWinner/loser molecule pairs\nLabeled sabotage types:\nsteric clash, charge flip,\nH-bond loss, etc."]
    end

    subgraph aevom_training ["Aevom Training Components (aevom repo)"]
        Adapter["dataset_adapter.py\nCDD -> NeMo-Gym JSONL\nMoleculePair -> prompt + seed data"]
        Preloader["TrainingDataPreloader\nPreload molecules into storage\nbefore rollout begins"]

        subgraph resource_server ["NeMo-Gym Resource Server (training/nemo_gym_wrapper/app.py)"]
            SeedSession["POST /seed_session\nPreload winner/loser SDFs\ninto InMemoryStorage\nInitialize fakeredis"]
            ToolExec["POST /{tool_name}\nDelegates to ToolRegistry.execute()\nSame tools as production"]
            Verify["POST /verify\nExtract trajectory\nCompute tiered rewards\nReturn reward float"]
            SandboxEndpoints["Sandbox callback endpoints\n/api/sandbox/data, /store,\n/visualize, /health"]
        end

        subgraph rewards ["Reward Computation"]
            TieredReward["TieredRewardComputer"]
            Tier1["Tier 1: Tool Usage\ndock_attempt: 0.05\npython_exec_success: 0.05\nrdkit_usage: 0.03\nmetric_computed: 0.05\nmax: 0.50"]
            Tier2["Tier 2: Differentiation\ncorrect_differentiation: 0.30\ncorrect_sabotage_type: 0.20\npartial_diagnosis: 0.10"]
            Tier3["Tier 3: Molecule Fix\nmolecule_modified: 0.50\nredock_success: 0.50\naffinity_recovered: 1.00"]
            Judge["LLMJudge\nLogical consistency (0-3)\nData-driven reasoning (0-3)\nScientific accuracy (0-3)\nWeight: 0.30"]
        end

        ConvController["ConversationController\nAdvanced mode: detect hypothesis\nInject 'prove it by editing'\nTrack modification + redocking"]
        ToolDetector["TrajectoryAnalyzer\nExtract docking scores\nDetect tool call patterns\nParse python_exec output"]

        subgraph tool_services ["Tool Services (SLURM)"]
            Orchestrator["ToolServiceOrchestrator\nStarts RDKit, Vina, PyExec\nas uvicorn subprocesses\nGPU pinning for Vina"]
        end

        SharedRegistry["ToolRegistry\n(same as production)\n+ MockRegistry option"]
    end

    subgraph nemo_gym ["NeMo-Gym (nemo-gym repo)"]
        SimpleAgent["SimpleAgent\nresponses_api_agents/simple_agent/app.py\nOrchestratesrollouts:\n/run -> /seed_session\n-> /{tool_name} loop\n-> /verify"]
        VLLMModel["vLLM Model Server\nresponses_api_models/vllm_model/app.py\nPOST /v1/chat/completions\nPOST /v1/responses\nProxies to vLLM backend"]
    end

    subgraph nemo_rl ["NeMo-RL (nemo-rl repo)"]
        GRPOTrainer["GRPO Trainer\nexamples/nemo_gym/run_grpo_nemo_gym.py\nGroup Relative Policy Optimization\nshould_use_nemo_gym=true"]
        PolicyModel["Policy Model (vLLM)\nNemotron-3-Nano-30B\nTP=4, DP=2"]
    end

    subgraph slurm ["SLURM Cluster"]
        SlurmNode["8x H100-80GB per node\nPyxis/Enroot containers\nsbatch: grpo_30b_nanov3.sbatch"]
    end

    CDD -->|"JSON pairs"| Adapter
    Adapter -->|"JSONL dataset"| GRPOTrainer

    GRPOTrainer -->|"manages"| PolicyModel
    GRPOTrainer -->|"spawns"| SimpleAgent
    SimpleAgent -->|"HTTP"| VLLMModel
    VLLMModel -->|"inference"| PolicyModel

    SimpleAgent -->|"HTTP /seed_session"| SeedSession
    SimpleAgent -->|"HTTP /{tool_name}"| ToolExec
    SimpleAgent -->|"HTTP /verify"| Verify

    SeedSession --> Preloader
    ToolExec --> SharedRegistry
    SharedRegistry --> Orchestrator

    Verify --> TieredReward
    TieredReward --> Tier1
    TieredReward --> Tier2
    TieredReward --> Tier3
    TieredReward --> Judge
    Verify --> ToolDetector
    Verify --> ConvController

    GRPOTrainer --> slurm
```

## Training Episode Flow

```mermaid
sequenceDiagram
    participant GRPO as GRPO Trainer
    participant Agent as SimpleAgent
    participant vLLM as vLLM Policy
    participant RS as Resource Server
    participant Tools as ToolRegistry
    participant Reward as TieredRewardComputer

    GRPO->>Agent: /run (prompt with @ref:winner, @ref:loser)
    Agent->>RS: POST /seed_session (preload SDFs, PDB)
    RS->>RS: Store winner/loser in InMemoryStorage
    RS-->>Agent: {session_id, available_tools}

    loop Tool-calling turns (max iterations)
        Agent->>vLLM: POST /v1/responses (messages + tools)
        vLLM-->>Agent: {tool_calls: [{name, args}]}
        Agent->>RS: POST /{tool_name} (arguments)
        RS->>Tools: ToolRegistry.execute(tool_name, args)
        Tools-->>RS: ToolResult
        RS-->>Agent: {result}
    end

    Agent->>vLLM: POST /v1/responses (final, no tools)
    vLLM-->>Agent: {content: diagnosis text}

    Agent->>RS: POST /verify (full trajectory)
    RS->>Reward: compute_reward(trajectory, ground_truth)
    Reward->>Reward: Tier1: check docking, python_exec, RDKit
    Reward->>Reward: Tier2: check differentiation, sabotage type
    Reward->>Reward: Tier3: check molecule fix (if advanced mode)
    Reward->>Reward: Judge: evaluate reasoning quality
    Reward-->>RS: total_reward (0.0 - ~3.0)
    RS-->>Agent: {reward: float}
    Agent-->>GRPO: {reward, trajectory}
```

## Training vs Production Comparison

| Component | Production (K8s) | Training (SLURM) |
|-----------|-----------------|-------------------|
| **LLM** | NVIDIA LLM API (cloud) | Local vLLM (Nemotron-3-Nano-30B) |
| **Orchestration** | LangGraph StateGraph | NeMo-Gym SimpleAgent |
| **Tool Registry** | `ToolRegistry` (shared) | `ToolRegistry` (same code) |
| **Storage** | PostgreSQL + Redis | InMemoryDataArtifactRepository + fakeredis |
| **Tool Services** | Docker containers (rdkit, pyexec, vina) | ToolServiceOrchestrator (uvicorn subprocesses) |
| **Database** | PostgreSQL 16 | None (in-memory) |
| **Frontend** | React SPA + Molstar | None |
| **Deployment** | Kubernetes | SLURM + Pyxis/Enroot |
| **Hardware** | Cloud GPUs | 8x H100-80GB per node (TP=4, DP=2) |

## Source References

- `training/nemo_gym_wrapper/app.py` -- resource server (extends `SimpleResourcesServer`), seed_session, tool execution, verify, sandbox endpoints, fakeredis patching
- `backend/api/training/rewards.py` -- `TieredRewardComputer`, `REWARD_CATALOG`, `DEFAULT_REWARD_CONFIG`, tier1/2/3 computation
- `backend/api/training/judge.py` -- `LLMJudge`, scoring rubric (logical consistency, data-driven, scientific accuracy), YAML-configurable prompts
- `backend/api/training/conversation_controller.py` -- `ConversationController`, hypothesis detection patterns, injection prompt, advanced mode tracking
- `backend/api/training/tool_detector.py` -- `TrajectoryAnalyzer`, docking score extraction, tool call pattern detection
- `backend/api/training/dataset_adapter.py` -- `MoleculePair`, CDD-to-JSONL conversion, `generate_reference_id` reuse
- `backend/api/training/data_preloader.py` -- `TrainingDataPreloader`, molecule preloading into storage
- `training/tool_service_orchestrator.py` -- `ToolServiceOrchestrator`, starts tool services as uvicorn subprocesses with GPU pinning
- `nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py` -- GRPO training entry point
- `nemo-gym/responses_api_agents/simple_agent/app.py` -- SimpleAgent rollout orchestration
- `nemo-gym/responses_api_models/vllm_model/app.py` -- vLLM model server (`/v1/responses`, `/v1/chat/completions`)
- `training/slurm/grpo/grpo_30b_nanov3.sbatch` -- SLURM job script for 30B GRPO training
