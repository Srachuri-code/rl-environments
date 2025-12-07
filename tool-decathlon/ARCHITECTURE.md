# Tool Decathlon Architecture

## Summary

**We wrote 651 lines of glue code to wrap Toolathlon's infrastructure for RL training.**

## The Three Layers

### Layer 1: Toolathlon (In Sandbox)
**What it provides:**
- 108 pre-configured tasks
- 30+ MCP servers (arxiv, github, google cloud, etc.)
- Eval scripts per task
- Task configs, prompts, workspaces

**What we DON'T modify:**
- Their MCP servers
- Their eval scripts  
- Their task definitions

### Layer 2: task_api.py (158 lines)
**Our thin wrapper over Toolathlon's internals:**

```python
setup(task_id)       → Start MCPs for this task, return tools
execute(tool, args)  → Call MCP, return result
evaluate()           → Run eval script, return success/fail
```

**Why we need this:**
- Toolathlon's `TaskAgent` is autonomous (runs to completion)
- We need step-by-step control for RL
- This gives us that control while reusing their infrastructure

### Layer 3: ToolDecathlonEnv (299 lines)
**Glue between verifiers ↔ sandboxes:**

```python
setup_state()    → Create sandbox, call task_api.setup()
env_response()   → Execute tools via task_api.execute()
rubric           → Get reward via task_api.evaluate()
cleanup_state()  → Destroy sandbox
```

## Execution Flow (Step-by-Step)

**Setup:**
```
1. Trainer: "Load environment"
   → ToolDecathlonEnv: Load dataset (just task prompts)
   
2. Trainer: "Sample task: find-alita-paper"
   → ToolDecathlonEnv: setup_state()
   
3. ToolDecathlonEnv: Create prime-sandbox
   → Prime Sandboxes: docker run toolathlon:latest
   
4. ToolDecathlonEnv → Sandbox: python task_api.py setup find-alita-paper
   
5. task_api.py:
   - Reads task_config.json
   - Starts arxiv_local MCP
   - Starts filesystem MCP
   - Returns tools list
   
6. ToolDecathlonEnv → Trainer: Here are your tools [...]
```

**Turn 1:**
```
1. Trainer → Model: Generate response with tools
   
2. Model → Trainer: 
   {"tool_calls": [{"name": "arxiv_local__search", "args": {"query": "Alita"}}]}
   
3. Trainer → ToolDecathlonEnv: Execute this tool call
   
4. ToolDecathlonEnv → Sandbox: 
   python task_api.py execute "arxiv_local__search" '{"query": "Alita"}'
   
5. task_api.py → Toolathlon MCP Manager → ArXiv API
   
6. ArXiv API → task_api.py: [paper results]
   
7. task_api.py → Sandbox stdout: {"results": [...]}
   
8. Sandbox → ToolDecathlonEnv: stdout
   
9. ToolDecathlonEnv → Trainer: 
   {"role": "tool", "content": "Found 3 papers..."}
```

**Turn N (claim_done):**
```
1. Model: {"tool_calls": [{"name": "claim_done"}]}

2. ToolDecathlonEnv: Mark task_done = True

3. ToolDecathlonEnv → Sandbox: python task_api.py evaluate

4. task_api.py:
   - Import evaluator.py for this task
   - Run evaluate(workspace)
   - Check if paper was downloaded correctly
   - Return {"success": True}

5. ToolDecathlonEnv: Store eval_result = True

6. Rubric: reward = 1.0 (success!)

7. Trainer: Use reward for RL update
```

**Cleanup:**
```
ToolDecathlonEnv → Prime Sandboxes: Delete container
→ MCPs auto-shutdown
→ Workspace destroyed
```

## Why Sandboxes?

**Isolation:**
- Each task = fresh container
- No state leakage between episodes
- Parallel execution without conflicts

**Reuse:**
- Toolathlon already built 600+ tools
- We don't replicate - we wrap
- 651 lines vs thousands

**Correctness:**
- Use their exact eval scripts
- Real API calls (not mocks)
- Accurate training signal

## Key Insight: Meta-Environment

This isn't one environment - it's a **meta-environment**:

```python
# Traditional RL: One task
env = MathEnv()  # Always "solve 2+2"

# Meta-RL: Distribution of tasks  
env = ToolDecathlonEnv()  # 108 different tasks!
task = env.sample()  # Random each time
```

**Benefits:**
- Curriculum learning (easy → hard tasks)
- Generalization (train on 80, test on 28)
- Meta-learning (learn to learn new tasks)

## Future: Dense Rewards

**Current architecture already supports this!**

Just modify Toolathlon's `evaluator.py` files:

```python
# Before (sparse)
def evaluate(workspace):
    paper_exists = Path(workspace / "paper.pdf").exists()
    return paper_exists

# After (dense)  
def evaluate(workspace):
    return {
        "paper_downloaded": Path(workspace / "paper.pdf").exists(),
        "correct_paper": check_paper_id(workspace),
        "used_arxiv_tool": check_tool_usage_log(),
        "efficient": get_turn_count() < 5,
    }
```

Then update rubric weights in `tool_decathlon.py`:

```python
Rubric(
    funcs=[
        extract_metric("paper_downloaded"),
        extract_metric("correct_paper"),
        extract_metric("efficient"),
    ],
    weights=[0.5, 1.0, 0.1]
)
```

**No changes to the environment code needed.**

## Communication Protocol

**Verifiers → ToolDecathlonEnv → Sandbox → task_api.py → Toolathlon**

Each arrow is a simple interface:
- Verifiers API: Standard multi-turn RL environment
- Sandbox API: HTTP commands (execute_command)  
- task_api.py: CLI (setup, execute, evaluate)
- Toolathlon: Python imports (their libraries)

All we wrote is the translation between these layers.

## Training at Scale

**Blackwell nodes:**
```python
# Batch size: 1024 tasks in parallel
# Each task = isolated sandbox

for batch in dataloader:  # 1024 tasks
    sandboxes = await create_sandboxes(batch)  # 1024 containers
    # Each runs different Toolathlon task
    
    for turn in range(max_turns):
        responses = model.generate_batch(1024)
        tool_results = await execute_batch(sandboxes, responses)
    
    rewards = await evaluate_batch(sandboxes)  # 1024 eval scripts
    
    loss = rl_loss(rewards)
    optimizer.step()
    
    await destroy_sandboxes(sandboxes)  # Cleanup 1024 containers
```

Perfect parallelization - no shared state, no bottlenecks.
