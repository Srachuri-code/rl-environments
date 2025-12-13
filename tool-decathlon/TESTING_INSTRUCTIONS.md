# Testing Tool Decathlon Environment (No Credentials)

## Quick Test Instructions

### Step 1: Start Docker

```bash
# Start Colima (if using Colima)
colima start --cpu 4 --memory 8 --disk 50

# Or if using Docker Desktop, just make sure it's running
```

### Step 2: Build Docker Image

```bash
cd /Users/skrachur/Desktop/rl-environments/tool-decathlon
./scripts/build_toolathlon_image.sh
```

This will:
- Clone Toolathlon repo (if needed)
- Copy your updated `task_api.py` with HTTP server
- Build `toolathlon:latest` image
- Takes ~5-10 minutes

### Step 3: Install/Reinstall Environment

```bash
cd /Users/skrachur/Desktop/rl-environments/tool-decathlon
uv pip install -e environments/tool_decathlon
```

### Step 4: Run Test Evaluation

**Quick test (1 task, 1 rollout):**
```bash
export OPENAI_API_KEY=your-key-here

vf-eval tool-decathlon \
  -m gpt-4o-mini \
  -n 1 \
  -r 1 \
  --save \
  -a '{"dataset_path": "data/tool_decathlon_dataset_minimal"}'
```

**Full minimal test (7 tasks, 2 rollouts each):**
```bash
vf-eval tool-decathlon \
  -m gpt-4o-mini \
  -n 7 \
  -r 2 \
  --save \
  -a '{"dataset_path": "data/tool_decathlon_dataset_minimal"}'
```

### What to Expect

**Success indicators:**
- ✅ Container creates in ~5-15 seconds
- ✅ Runtime server starts (you'll see HTTP health check logs)
- ✅ Tools are discovered and exposed
- ✅ Model interacts with tools via clean stepping loop
- ✅ Evaluation runs when `claim_done` is called
- ✅ Container cleanup happens

**Expected output:**
```
Loading environment: tool-decathlon...
Container ready with 15 tools
Episode 1/1: arrange-workspace
Turn 1: Model called filesystem-list_directory
Turn 2: Model called excel-read_file
...
Turn N: Model called claim_done
Evaluation: SUCCESS (or FAILED)
Cleanup complete.
```

### Tasks in Minimal Dataset

All of these work without credentials:

1. **arrange-workspace** - File organization
2. **cooking-guidance** - Recipe instructions  
3. **courses-ta-hws** - Homework grading
4. **detect-revised-terms** - PDF comparison
5. **dietary-health** - Diet analysis
6. **excel-data-transformation** - Excel processing
7. **excel-market-research** - Market analysis

### Troubleshooting

**"Docker not found":**
```bash
colima start
```

**"Image not found: toolathlon:latest":**
```bash
./scripts/build_toolathlon_image.sh
```

**"Module not found: tool_decathlon":**
```bash
cd /Users/skrachur/Desktop/rl-environments/tool-decathlon
uv pip install -e environments/tool_decathlon
```

**"Container hangs at startup":**
- First build takes longer (~10 min) due to MCP package downloads
- Check `docker logs <container-name>` for details

**"Runtime server not responding":**
- Check if port 8000 is available
- Look for "Uvicorn running on" in container logs

### Viewing Results

After running `vf-eval --save`, results are saved to:
```
environments/tool_decathlon/outputs/
```

Each rollout includes:
- Full conversation transcript
- Tool calls and responses
- Final reward
- Metadata (task_id, model, timestamp)

### Next Steps After This Works

Once the minimal dataset test passes:

1. **Validate architecture** ✅
2. **Set up credentials** (optional, for all 108 tasks)
3. **Test on GPU cluster** (for RL training)
4. **Scale to full dataset**

## Alternative: Test Without Docker (Debugging)

If Docker issues persist, you can test the core logic:

```python
from tool_decathlon import load_environment
from datasets import load_from_disk

# Load environment (will fail at container creation, but validates dataset/rubric)
env = load_environment(dataset_path="data/tool_decathlon_dataset_minimal")
print(f"Environment loaded: {len(env.dataset)} tasks")
print(f"Sample task: {env.dataset[0]['task_id']}")
```

This validates your environment package is installed correctly.
