# Quick Architecture Test (No Setup Required!)

## Test with Ultra-Minimal Tasks

These 7 tasks use **only local MCPs** - no credentials, no downloads, no web access:

1. **arrange-workspace** - Organize files and PDFs using filesystem, terminal, excel, pdf-tools
2. **cooking-guidance** - Recipe instructions using filesystem, howtocook
3. **courses-ta-hws** - Grade homework using excel, filesystem, terminal
4. **detect-revised-terms** - Compare PDFs using filesystem, pdf-tools
5. **dietary-health** - Analyze diet data using excel, filesystem, howtocook, terminal
6. **excel-data-transformation** - Process Excel files
7. **excel-market-research** - Market analysis in Excel

## Run a Quick Test

### Single Task Test (Fastest)

```bash
vf-eval tool-decathlon \
  -m gpt-4o-mini \
  -n 1 \
  -r 1 \
  -a '{"dataset_path": "data/tool_decathlon_dataset_minimal"}'
```

**Expected:**
- Container initialization: ~10 seconds
- Should complete without hanging
- Verifies entire pipeline works

### Slightly Longer Test

```bash
vf-eval tool-decathlon \
  -m gpt-4o-mini \
  -n 3 \
  -r 2 \
  -a '{"dataset_path": "data/tool_decathlon_dataset_minimal"}'
```

**This tests:**
- 3 different tasks
- 2 rollouts each
- Total 6 episodes
- Should complete in 5-10 minutes (depending on model speed)

## What This Validates

✅ **Docker integration** - Containers create and communicate properly
✅ **task_api.py** - MCP server management works
✅ **Tool execution** - Tools route correctly through containers
✅ **Evaluation** - Toolathlon's eval scripts run and return rewards
✅ **Cleanup** - Containers destroy properly
✅ **verifiers integration** - Full RL environment protocol works

## If This Works

You'll know:
- Your entire architecture is sound
- The async/Docker/MCP integration is correct
- Only missing piece is credentials for the other 101 tasks

Then you can confidently spend the 30 minutes on Toolathlon's setup knowing your implementation is solid!

## Dataset Locations

- **Minimal (7 tasks, local only):** `data/tool_decathlon_dataset_minimal/` ← **Use this for testing!**
- **No-creds (13 tasks, includes web):** `data/tool_decathlon_dataset_no_creds/`
- **Full (30 tasks, needs setup):** `data/tool_decathlon_dataset/`

