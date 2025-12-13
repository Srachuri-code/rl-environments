# Tool Decathlon → Verifiers Implementation Summary

## What We Built

A **Verifiers-compatible RL environment** wrapping the [Toolathlon benchmark](https://github.com/hkust-nlp/Toolathlon) for large-scale reinforcement learning training.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Trainer                               │
│                  (prime-rl / vf.RLTrainer)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ToolDecathlonEnv (vf.MultiTurnEnv)              │
│  • Per-task tool discovery (K of N tools)                    │
│  • TauBench-style stepping loop                              │
│  • Binary→dense reward extensibility                         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Docker Container (per rollout)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Runtime HTTP Server (task_api.py serve)              │  │
│  │  • FastAPI on port 8000                               │  │
│  │  • Keeps MCP servers alive                            │  │
│  │  • Endpoints: /setup, /execute, /evaluate             │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Toolathlon Infrastructure                            │  │
│  │  • 30+ MCP servers (task-specific subset started)     │  │
│  │  • 7 local tool capabilities (20+ concrete tools)     │  │
│  │  • Task evaluators (reward functions)                 │  │
│  │  • Isolated workspace                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. **Meta-Environment (Not 108 Separate Envs)**
- Single environment package: `tool-decathlon`
- Dataset contains 108 task rows
- Each rollout samples a task → isolated runtime
- Perfect for RL curriculum learning / generalization

### 2. **Persistent Runtime Server (Not Process-Per-Tool)**
**Before:** Each tool call spawned `docker exec uv run python task_api.py execute ...`
- Cost: ~500-1000ms per tool call

**After:** Long-lived FastAPI server inside container
- Cost: ~10-50ms per tool call (HTTP RPC)
- Matches `prime-environments`' MCP connection pattern

### 3. **Dynamic K-of-N Tools Per Task**
- Not all 600+ tools exposed per task
- Runtime reads `state["info"]["mcp_servers"]` + `state["info"]["local_tools"]` from dataset
- Starts only needed MCP servers
- Exposes only needed local tool bundles
- Model sees task-appropriate toolset

### 4. **Toolathlon's Exact Tool Names (No Normalization)**
- Tools exposed as `local-sleep`, `local-search_history`, `filesystem-list_directory`, etc.
- Model learns Toolathlon's conventions
- No alias heuristics, no schema duplication
- Execution handler does direct lookup

### 5. **TauBench-Style Stepping Loop**
Matches [`prime-environments/tau_bench`](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/tau_bench/tau_bench_env.py):
- `setup_state`: create container, start runtime, initialize task
- `env_response`: parse tool_calls → execute via HTTP → return tool messages
- `is_completed`: check `claim_done` or max_turns
- `cleanup_state`: stop container

## File Inventory

### Core Files (Updated)

| File | Lines | Purpose |
|------|-------|---------|
| `docker/task_api.py` | 401 | HTTP runtime server for in-container tool execution |
| `environments/tool_decathlon/tool_decathlon.py` | 386 | Verifiers MultiTurnEnv implementation |
| `scripts/create_hf_dataset.py` | 205 | Dataset creator (task metadata only) |
| `scripts/build_toolathlon_image.sh` | 116 | Docker image builder |
| `scripts/test_no_creds.sh` | 95 | **NEW** - Quick test script |

### Documentation

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Detailed system design |
| `TESTING_INSTRUCTIONS.md` | **NEW** - How to test |
| `COMPLETE_SETUP_GUIDE.md` | Full credential setup (for all 108 tasks) |
| `PRODUCTION_SETUP.md` | Optimization for RL training |

## What Changed From Original Implementation

### Before (Your First Attempt)
- ✅ Meta-environment concept
- ✅ Docker isolation per rollout
- ✅ Toolathlon MCP integration
- ❌ Subclassed `ToolEnv` (wrong abstraction)
- ❌ Process spawn per tool call (slow)
- ❌ Local tools not properly exposed
- ❌ No proper JSON schemas for tools

### After (Current)
- ✅ Clean `MultiTurnEnv` stepping loop
- ✅ Persistent HTTP runtime server
- ✅ All Toolathlon local tools exposed correctly
- ✅ Real JSON schemas from Toolathlon
- ✅ TauBench/SWEAgent-style env_response pattern
- ✅ Ready for RL training at scale

## Testing Status

### ✅ Ready to Test (7 tasks, no credentials)
```bash
./scripts/test_no_creds.sh
```

### ⏸️ Needs Setup (101 tasks, requires credentials)
Follow `COMPLETE_SETUP_GUIDE.md` (~30-45 min one-time setup)

## How This Matches Prime Environments Patterns

Compared to [`prime-environments`](https://github.com/PrimeIntellect-ai/prime-environments):

**Like `tau_bench`:**
- ✅ `MultiTurnEnv` with custom `env_response`
- ✅ Wraps existing benchmark infrastructure
- ✅ Dynamic tool schemas per episode
- ✅ External tool execution (not in-process)

**Like `github_mcp`:**
- ✅ Persistent connection to tool runtime
- ✅ HTTP/RPC for tool calls (not subprocess)
- ✅ Clean separation: schemas vs execution

**Like `mini_swe_agent_bench`:**
- ✅ Docker container per rollout
- ✅ Per-task workspace isolation
- ✅ Evaluation runs at task completion

## Key Insight: Why This Architecture Works for RL

### The Problem Toolathlon Solves
Real-world tool use with authentic services (600+ tools, 32+ apps)

### The Problem We Solve
Convert that benchmark into a **trainable environment** while maintaining fidelity

### The Solution
- **Reuse** Toolathlon's infrastructure (MCPs, eval scripts, tool definitions)
- **Replace** their autonomous agent loop with step-by-step RL control
- **Wrap** everything in Verifiers' standard API

### The Result
- ✅ Full benchmark fidelity (use their exact tools/evals)
- ✅ RL-trainable (step control, rewards, batching)
- ✅ Scalable (isolated containers, async parallel)
- ✅ Simple (~800 lines of glue code total)

## Training Workflow (When Ready)

```python
import verifiers as vf

# Load environment
env = vf.load_environment("tool-decathlon")

# Option 1: Evaluate with API models
results = await env.evaluate(
    client=AsyncOpenAI(),
    model="gpt-4o-mini",
    num_examples=10,
    rollouts_per_example=3
)

# Option 2: Train with vf.RLTrainer
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
trainer = vf.RLTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=vf.grpo_defaults(run_name="tool-decathlon-test")
)
trainer.train()

# Option 3: Train with prime-rl (large-scale)
# See verifiers docs for prime-rl integration
```

## References

- **Toolathlon Paper:** https://arxiv.org/abs/2510.25726
- **Toolathlon Repo:** https://github.com/hkust-nlp/Toolathlon
- **Verifiers Docs:** https://verifiers.readthedocs.io
- **Prime Environments:** https://github.com/PrimeIntellect-ai/prime-environments
- **Prime RL:** https://github.com/PrimeIntellect-ai/prime-rl

## Next Steps

1. **Test locally** (no credentials):
   ```bash
   ./scripts/test_no_creds.sh
   ```

2. **Setup credentials** (unlock all tasks):
   See `COMPLETE_SETUP_GUIDE.md`

3. **Train on GPU cluster**:
   Use `prime-rl` or `vf.RLTrainer`

4. **Publish to Environments Hub**:
   ```bash
   prime env push tool-decathlon
   ```
