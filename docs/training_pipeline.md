# Aevom Training Pipeline

## Overview

The Aevom Training Pipeline enables reinforcement learning (RL) training of large language models for molecular binding analysis tasks. Built on **NeMo-Gym** and **NeMo RL**, the pipeline implements **GRPO (Group Relative Policy Optimization)** to train models that can autonomously diagnose molecular binding differences, propose hypotheses, and even fix defective molecules.

### Training Objective

Given two molecules—one with high binding affinity (winner) and one with poor binding affinity (loser)—the model learns to:

1. **Diagnose**: Use docking, RDKit analysis, and Python code to identify why the molecules differ
2. **Hypothesize**: Form a clear hypothesis about the molecular defect (steric clash, charge flip, H-bond loss, etc.)
3. **Fix** (Advanced Mode): Modify the defective molecule to recover binding affinity

### Reward Structure

The training uses a **tiered, hybrid reward system**:

**Tier 1** is a per-tool, per-call schedule: each scheduled tool earns `attempt`
per call (success or fail) plus `success` per successful call, clamped by that
tool's per-tool `cap`. There is no global Tier-1 cap (the per-tool caps bound it;
effective Tier-1 max = sum of caps = 1.00). Unscheduled tools earn 0. The whole
episode reward is **gated to 0 when the agent makes no tool calls** (judge skipped).

| Tier | Tool / Component | attempt | success | cap | Description |
|------|------------------|---------|---------|-----|-------------|
| 1 | `vina_dock` | 0.02 | 0.02 | 0.08 | Docking (cap = 2 successful calls) |
| 1 | `molecule_info` | 0.02 | 0.02 | 0.08 | RDKit properties (cap = 2 successful calls) |
| 1 | `structure_query` | 0.04 | 0.04 | 0.32 | MolQL selection (cap = 4 successful calls) |
| 1 | `python_execute` | 0.06 | 0.06 | 0.48 | pyexec sandbox (cap = 4 successful calls) |
| 1 | `visualize_pdb` | 0.02 | 0.00 | 0.04 | Display-only (attempt only, cap = 2 attempts) |
| 2 | Sabotage type (`correct_sabotage_id`) | — | — | 0.50 | Named the correct defect type |
| 2 | Partial diagnosis (`partial_sabotage_id`) | — | — | 0.30 | Detected some defect type |
| 3 | Hypothesis with metric | — | — | 0.20 | Hypothesis cites a computed paired metric |
| 3 | Non-trivial edit | — | — | 0.50 | Created a targeted (non-winner-copy) edit |
| 3 | Edit redock success | — | — | 0.50 | Docked the edited molecule |
| 3 | Affinity recovered | — | — | 1.00 | Achieved 80%+ of winner affinity |
| Judge | Reasoning quality | — | — | 0.30 | LLM judge evaluation |

Max total = Tier-1 1.00 + Tier-2 0.50 + Tier-3 2.20 + Judge 0.30 = **4.00**
(floored at 0; trivial-edit penalty -0.50 can only reduce, never raise the max).

---

## Architecture

> **Note:** The deployment architecture uses LangGraph (see [multi_turn_tool_chaining.md](multi_turn_tool_chaining.md)), but the training pipeline bypasses LangGraph entirely. The NeMo-Gym resource server wraps `ToolRegistry` directly via a standalone FastAPI app, providing tool execution and reward verification endpoints.

### High-Level Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SLURM CLUSTER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         NeMo RL Training Process                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │           Policy Model (Nemotron-3-Nano-30B via vLLM)               │  │  │
│  │  │    - Tensor Parallel: 4 GPUs                                        │  │  │
│  │  │    - Data Parallel: 2 replicas                                      │  │  │
│  │  │    - Inference (rollouts) + Training (GRPO gradients)               │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                            │  │
│  │                              │ HTTP (OpenAI-compatible)                   │  │
│  │                              ▼                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    NeMo-Gym Agent Server                            │  │  │
│  │  │    - Orchestrates rollouts (model ↔ resources)                      │  │  │
│  │  │    - Manages multi-turn tool calling loop                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                  │                                              │
│                                  │ HTTP                                         │
│                                  ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                  Aevom Resource Server                                    │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │  │
│  │  │  /seed_session   │  │      /step       │  │     /verify      │         │  │
│  │  │  Load molecules  │  │  Execute tools   │  │  Compute reward  │         │  │
│  │  │  from dataset    │  │  (dock, rdkit)   │  │  (tiered+judge)  │         │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘         │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                        Tool Endpoints                               │  │  │
│  │  │  /vina_dock  /python_exec  /rdkit_*  /load_molecule  /store_sdf     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline Components                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐   │
│  │   Dataset Adapter   │───▶│  Training Tasks     │───▶│  Data Preloader  │   │
│  │                     │    │      (JSONL)        │    │                  │   │
│  │  • CDD → NeMo-Gym   │    │                     │    │  • Load SDFs     │   │
│  │  • Ground truth     │    │  • Winner/Loser     │    │  • Load PDBQT    │   │
│  │  • Preload refs     │    │  • Sabotage type    │    │  • Map refs      │   │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘   │
│                                                                               │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐   │
│  │   Resource Server   │◀──▶│   Tool Executor     │◀──▶│  Tool Registry   │   │
│  │                     │    │                     │    │                  │   │
│  │  • /seed_session    │    │  • Route calls      │    │  • vina_dock     │   │
│  │  • /step            │    │  • Track usage      │    │  • python_exec   │   │
│  │  • /verify          │    │  • Collect results  │    │  • rdkit_*       │   │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘   │
│                                                                               │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐   │
│  │   Reward Computer   │◀───│   Tool Detector     │    │    LLM Judge     │   │
│  │                     │    │                     │    │                  │   │
│  │  • Tier 1: Usage    │    │  • RDKit patterns   │    │  • Logical cons. │   │
│  │  • Tier 2: Diff     │    │  • Metric keywords  │    │  • Data-driven   │   │
│  │  • Tier 3: Fix      │    │  • Sabotage detect  │    │  • Scientific    │   │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘   │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                     Conversation Controller (Advanced Mode)             │  │
│  │                                                                         │  │
│  │  • Hypothesis detection (regex patterns)                                │  │
│  │  • Prompt injection ("prove it by editing...")                          │  │
│  │  • Modification tracking                                                │  │
│  │  • Redocking evaluation                                                 │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Training Data Flow                             │
└─────────────────────────────────────────────────────────────────────────────┘

1. DATASET PREPARATION (offline)
   aevom-data-gen/ ──▶ dataset_adapter.py ──▶ training_tasks.jsonl
   
2. SESSION INITIALIZATION (per episode)
   training_tasks.jsonl ──▶ /seed_session ──▶ preload SDFs, receptor
   
3. ROLLOUT COLLECTION (NeMo RL orchestrates)
   ┌──────────────────────────────────────────────────────────────┐
   │ for step in range(max_steps):                                │
   │   action = policy_model.generate(observation, tools)         │
   │   for tool_call in action.tool_calls:                        │
   │     result = resource_server.call(tool_call.name, args)      │
   │   observation.append(action, results)                        │
   │   if no_tool_calls: break                                    │
   └──────────────────────────────────────────────────────────────┘

4. REWARD COMPUTATION
   trajectory ──▶ /verify ──▶ {
     tier1_reward: tool_usage_score,
     tier2_reward: differentiation_score,
     tier3_reward: molecule_fix_score,
     judge_reward: reasoning_quality_score,
     total_reward: weighted_sum
   }

5. POLICY UPDATE (NeMo RL handles)
   trajectories + rewards ──▶ GRPO loss ──▶ gradient update ──▶ weight sync
```

---

## Components

### 1. Dataset Adapter (`dataset_adapter.py`)

Converts the CDD (Computational Drug Discovery) dataset from `aevom-data-gen` into NeMo-Gym's JSONL format.

**Input**: CDD dataset directory
```
dataset_cdd_v1/
├── complex_001/
│   ├── receptor.pdbqt
│   ├── winner.sdf
│   ├── losers/loser_01_steric_clash.sdf
│   └── metadata.json
```

**Output**: JSONL training tasks
```json
{
  "task_id": "task_000001_complex_001_steric_clash",
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "{system_prompt}"},
      {"role": "user", "content": "Here are two molecules @ref:sdf_winner_001..."}
    ]
  },
  "ground_truth": {
    "sabotage_type": "steric_clash",
    "winner_vina": -9.5,
    "loser_vina": -6.1,
    "modification_desc": "Replaced H on C12 with tert-butyl"
  },
  "preload_refs": {
    "sdf_winner_001": "complex_001/winner.sdf",
    "sdf_loser_001_steric_clash": "complex_001/losers/loser_01_steric_clash.sdf"
  }
}
```

**Usage**:
```bash
python -m api.training.dataset_adapter \
    --dataset /path/to/dataset_cdd_v1 \
    --output /data/training_tasks.jsonl \
    --advanced  # Enable advanced mode prompts
```

### 2. Resource Server (`nemo_gym_server.py`)

NeMo-Gym compatible FastAPI server that provides tools and reward verification.

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/seed_session` | POST | Initialize episode, load molecules |
| `/step` | POST | Execute tool calls |
| `/verify` | POST | Compute final reward |
| `/tool` | POST | Execute single tool |
| `/tools` | GET | Get available tool schemas |
| `/health` | GET | Health check |

**Launch**:
```bash
uvicorn api.training.nemo_gym_server:create_app \
    --host 0.0.0.0 --port 8100 --factory
```

### 3. Tiered Reward Computer (`rewards.py`)

Computes multi-tier rewards based on tool usage, differentiation success, and advanced mode achievements.

**Tier 1: Tool Usage** (accumulated during episode)
- Rewards correct tool usage patterns
- Detects RDKit imports and function calls
- Identifies metric computation

**Tier 2: Differentiation** (episode end)
- Validates correct identification of binding differences
- Matches detected sabotage type against ground truth

**Tier 3: Advanced Mode** (if enabled)
- Rewards hypothesis formation
- Rewards successful molecule modification
- Heavy reward for affinity recovery

### 4. LLM Judge (`judge.py`)

Evaluates reasoning quality using an external LLM.

**Evaluation Criteria**:
- **Logical Consistency** (0-3): Does reasoning flow logically?
- **Data-Driven** (0-3): Are conclusions supported by computed data?
- **Scientific Accuracy** (0-3): Are chemistry concepts correct?
- **Diagnostic Quality** (0-1): Was the defect correctly identified?

**Judge Presets** (in `judge_prompts.py`):
- `default`: Balanced evaluation
- `strict_scientific`: Demanding chemistry professor
- `gentle_mentor`: Supportive feedback
- `binary`: Simple pass/fail
- `chemistry_detailed`: Expert-level analysis

### 5. Conversation Controller (`conversation_controller.py`)

Manages the hypothesis → fix flow in advanced mode.

**Detection Patterns**:
```python
HYPOTHESIS_PATTERNS = [
    r"I\s+have\s+a\s+hypothesis",
    r"I\s+believe\s+(?:the\s+)?(?:reason|cause)\s+is",
    r"My\s+(?:hypothesis|diagnosis)\s+is",
    # ... more patterns
]
```

**Injection Prompt** (the live default in `backend/api/training/conversation_controller.py::DEFAULT_INJECTION_PROMPT`; overridable via `training/configs/advanced_mode.yaml`):
```
Good hypothesis! Now PROVE it by editing the molecule and re-docking.

Modify the loser molecule to address the defect you identified. Make a real
chemical change motivated by your hypothesis -- do NOT simply copy or
regenerate the winner molecule (a Tanimoto-similarity check vs the winner
will detect that and apply a penalty).

**Target**: Achieve at least {target_percentage}% of the winner's binding
affinity ({target_affinity:.2f} kcal/mol).

**Steps**:
1. Use `python_execute` with RDKit to modify the loser molecule's structure
2. Generate new 3D coordinates for the modified molecule
3. Store the new molecule with `aevom.store_sdf(...)` to obtain a new sdf reference
4. Dock the modified molecule using `vina_dock` with the new reference
5. Compare the new binding affinity to the winner and explain whether the
   result confirms or refutes your hypothesis
```

---

## Configuration

### Reward Configuration (`training/configs/rewards.yaml`)

```yaml
tier1:
  # Per-tool, per-call schedule (no global cap; per-tool caps bound Tier-1).
  # Must stay in sync with TOOL_REWARD_SCHEDULE in api/training/rewards.py.
  tool_schedule:
    vina_dock:       { attempt: 0.02, success: 0.02, cap: 0.08 }
    molecule_info:   { attempt: 0.02, success: 0.02, cap: 0.08 }
    structure_query: { attempt: 0.04, success: 0.04, cap: 0.32 }
    python_execute:  { attempt: 0.06, success: 0.06, cap: 0.48 }
    visualize_pdb:   { attempt: 0.02, success: 0.00, cap: 0.04 }
  metric_noise_threshold: 0.05

tier2:
  correct_differentiation: 0.30
  correct_sabotage_type: 0.20
  partial_diagnosis: 0.10

tier3:
  molecule_modified: 0.50
  redock_success: 0.50
  affinity_recovered: 1.00
  target_recovery_fraction: 0.80

judge_weight: 0.30
```

### GRPO Training Configuration (`training/configs/grpo/aevom_grpo.yaml`)

```yaml
policy:
  model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  precision: "bfloat16"
  generation:
    backend: "vllm"
    vllm_cfg:
      tensor_parallel_size: 4
      data_parallel_size: 2
    vllm_kwargs:
      trust_remote_code: true
      gpu_memory_utilization: 0.90
      mamba_ssm_cache_dtype: "bfloat16"

grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 8
  normalize_rewards: true
  use_leave_one_out_baseline: true
  kl_penalty_coef: 0.01
  max_num_steps: 15

optimizer:
  lr: 5.0e-6
  weight_decay: 0.01
```

---

## Infrastructure

### Hardware Requirements

#### Single Node (Recommended)

| Component | Specification |
|-----------|---------------|
| GPUs | 8x H100-80GB or A100-80GB |
| CPU | 64+ cores |
| RAM | 512GB+ |
| Storage | NVMe SSD, 1TB+ |
| Network | 100Gbps+ for multi-node |

**GPU Allocation**:
- GPUs 0-3: Policy model replica 1 (TP=4)
- GPUs 4-7: Policy model replica 2 (TP=4)

#### Multi-Node (For larger experiments)

| Nodes | GPUs | DP Replicas | Throughput |
|-------|------|-------------|------------|
| 1 | 8 | 2 | Baseline |
| 2 | 16 | 4 | ~2x |
| 4 | 32 | 8 | ~4x |

### Software Stack

| Component | Version |
|-----------|---------|
| CUDA | 12.1+ |
| Python | 3.11+ |
| PyTorch | 2.3+ |
| vLLM | 0.5.0+ |
| NeMo | 24.07+ |
| NeMo-Gym | Latest |

### SLURM Deployment

The training pipeline is designed for SLURM clusters with Pyxis/Enroot:

```bash
# Build container
./training/slurm/container/build_enroot.sh

# Submit single-node job
sbatch training/slurm/grpo/train.sbatch

# Submit multi-node job
sbatch training/slurm/grpo/train_multinode.sbatch
```

**Container Contents**:
- NeMo base image
- Aevom backend code
- NeMo-Gym
- vLLM
- Configuration files

---

## Usage

### Quick Start

```bash
# 1. Prepare dataset
python -m api.training.dataset_adapter \
    --dataset /path/to/cdd_dataset \
    --output /data/training_tasks.jsonl

# 2. Build container (on build machine)
cd training/slurm/container
./build_enroot.sh

# 3. Submit training job
export AEVOM_DATA_PATH=/data
export AEVOM_RESULTS_PATH=/results
sbatch train.sbatch
```

### Local Development

```bash
# Start resource server
uvicorn api.training.nemo_gym_server:create_app \
    --host 0.0.0.0 --port 8100 --factory

# Run training script (local mode)
python scripts/run_aevom_grpo.py \
    --config training/configs/grpo/aevom_grpo.yaml \
    --local
```

### Configuration Overrides

```bash
# Override via command line
python scripts/run_aevom_grpo.py \
    --config training/configs/grpo/aevom_grpo.yaml \
    ++training.num_epochs=5 \
    ++grpo.num_prompts_per_step=64
```

---

## Production Safety

The training infrastructure is **completely isolated** from production:

| Aspect | Production | Training |
|--------|------------|----------|
| Deployment | Kubernetes | SLURM |
| Database | PostgreSQL | Local/mock storage |
| LLM | NVIDIA API | Local vLLM |
| Container | Production image | Training image |
| Data | User data | CDD dataset |
| Telemetry | Production Phoenix | Separate instance |

**No shared resources** between training and production deployments.

---

## Files Reference

```
aevom/
├── backend/api/training/
│   ├── __init__.py              # Module exports
│   ├── dataset_adapter.py       # CDD → JSONL conversion
│   ├── data_preloader.py        # Load molecules at episode start
│   ├── nemo_gym_server.py       # Resource server
│   ├── schemas.py               # Request/response types
│   ├── rewards.py               # Tiered reward computation
│   ├── tool_detector.py         # Analyze tool usage
│   ├── judge.py                 # LLM-as-judge
│   ├── judge_prompts.py         # Configurable judge prompts
│   └── conversation_controller.py  # Advanced mode flow
├── training/
│   ├── configs/
│   │   ├── rewards.yaml         # Reward weights
│   │   ├── judge.yaml           # Judge configuration
│   │   ├── advanced_mode.yaml   # Hypothesis detection
│   │   ├── grpo/
│   │   │   ├── aevom_grpo_30b_nanov3.yaml  # Production GRPO config (Megatron, 4-node)
│   │   │   └── aevom_grpo_local_test.yaml  # Local testing (Qwen 0.5B, 1 GPU)
│   │   ├── sft/
│   │   │   ├── sft_nemotron_lora.yaml
│   │   │   └── sft_test.yaml
│   │   └── nemo_gym/
│   │       ├── aevom_resource_server.yaml
│   │       ├── aevom_resource_server_cluster.yaml
│   │       └── aevom_agent.yaml
│   └── slurm/
│       ├── grpo/
│       │   ├── train.sbatch     # Single-node GRPO job
│       │   └── train_multinode.sbatch
│       ├── sft/
│       │   ├── prod.sbatch      # Production SFT job
│       │   └── test.sbatch
│       ├── container/
│       │   ├── Dockerfile       # Training container
│       │   ├── build_enroot.sh  # Build script
│       │   └── entrypoint.sh
│       └── README.md            # Deployment guide
├── scripts/
│   └── run_aevom_grpo.py        # Training launch script
```

---

## See Also

- [NeMo-Gym Documentation](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo RL / NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [aevom-data-gen](../aevom-data-gen/) - Training data generation
