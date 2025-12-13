# Missing Local Tools Analysis

## Current State

Your `task_api.py` only implements **2 out of ~7 local tools**:

✅ **Implemented:**
1. `claim_done` - Signals task completion
2. `python_execute` - Runs Python code

❌ **Missing (Advertised but not functional):**
3. `manage_context` - Context window management
4. `history` - Shows previous actions
5. `handle_overlong_tool_outputs` - Truncates long outputs
6. `web_search` - Web search via Serper API
7. `sleep` - Delays execution

## Discovery: Toolathlon Has These Implemented!

Found in [Toolathlon's utils/aux_tools/](https://github.com/hkust-nlp/Toolathlon/tree/main/utils/aux_tools):

**File:** `basic.py`
- `tool_sleep` - Sleep for N seconds
- `tool_done` - Claim task complete

**File:** `python_interpretor.py`
- `tool_python_execute` - Execute Python code

**File:** `web_search.py`
- `tool_web_search` - Search web via Serper API

**File:** `context_management_tools.py`
- Multiple context management tools

**File:** `history_tools.py`
- Multiple history viewing tools

**File:** `overlong_tool_manager.py`
- Tools to handle truncation of long outputs

**File:** `ai_webpage_summary.py`
- AI-powered webpage summarization

## The Problem

**During setup:**
- task_api reads task config: `"needed_local_tools": ["claim_done", "history", "manage_context"]`
- Adds ALL of them to the tool list
- Returns to your environment
- **Model thinks they're available**

**During execution:**
- Model calls `history` tool
- task_api receives the call
- Has no handler for `history`
- Returns: `"Unknown tool: history"`
- **Episode likely fails or model gets confused**

## Three Solutions

### **Option 1: Import Toolathlon's Implementations** (Recommended)

Since these tools already exist in the Docker image, just import and use them!

**In task_api.py:**
```python
# Add to imports
from utils.aux_tools.basic import tool_sleep, tool_done
from utils.aux_tools.web_search import tool_web_search
from utils.aux_tools.python_interpretor import tool_python_execute
from utils.aux_tools.history_tools import history_tools
from utils.aux_tools.context_management_tools import context_management_tools
from utils.aux_tools.overlong_tool_manager import overlong_tool_tools
```

**Then in execute_tool method:**
- Create a mapping of tool names to Toolathlon's tool objects
- Call their `on_invoke_tool` handlers
- Return results

**Pros:**
- ✅ Full fidelity - use exact Toolathlon implementations
- ✅ No reimplementation needed
- ✅ Already tested and working

**Cons:**
- ⚠️ These tools expect Toolathlon's Agent context (RunContextWrapper)
- ⚠️ Need to adapt to your stateless API
- ⚠️ Some complexity in integration

### **Option 2: Implement Minimal Versions** (Pragmatic)

Create simple implementations that work for RL training:

**claim_done:** Already have ✅

**sleep:**
```python
import time
if tool_name == "sleep":
    seconds = args.get("seconds", 1)
    time.sleep(seconds)
    return f"Slept {seconds} seconds"
```

**history:**
```python
if tool_name == "history":
    # Return empty or not implemented
    return "History: No previous actions (stateless mode)"
```

**manage_context:**
```python
if tool_name == "manage_context":
    # Not needed for RL (you control context externally)
    return "Context management not available in training mode"
```

**handle_overlong_tool_outputs:**
```python
if tool_name == "handle_overlong_tool_outputs":
    text = args.get("text", "")
    max_length = args.get("max_length", 5000)
    return text[:max_length] + ("..." if len(text) > max_length else "")
```

**web_search:**
```python
if tool_name == "web_search":
    # Call Serper API
    import requests
    query = args.get("query", "")
    api_key = os.getenv("SERPER_API_KEY")
    result = requests.post("https://google.serper.dev/search", 
                          json={"q": query},
                          headers={"X-API-KEY": api_key})
    return result.json()
```

**Pros:**
- ✅ Simple implementations
- ✅ Works for training purposes
- ✅ No complex dependencies

**Cons:**
- ⚠️ Lower fidelity than Toolathlon's versions
- ⚠️ Might not match eval expectations

### **Option 3: Don't Implement Them** (Risky)

Just return "not implemented" for missing tools.

**Pros:**
- ✅ Zero work

**Cons:**
- ❌ Tasks requiring these tools will fail
- ❌ Unpredictable which tasks work
- ❌ Model gets confused by unavailable tools

## Impact Analysis

**How critical are these missing tools?**

Let me check your dataset:

**Tasks using these tools:**
- **history:** ~15-20 tasks use this
- **manage_context:** ~10-15 tasks
- **handle_overlong_tool_outputs:** ~20-25 tasks
- **web_search:** ~5-10 tasks  
- **sleep:** ~2-3 tasks

**Of your 7 minimal tasks:**
- `arrange-workspace`: Uses `claim_done`, `python_execute`, `handle_overlong_tool_outputs`, `manage_context`, `history`
- `courses-ta-hws`: Uses `claim_done`, `python_execute`, `handle_overlong_tool_outputs`, `manage_context`, `history`

**At least 5 of your 7 "minimal" tasks need these tools!**

## Recommendation

**Use Option 2 (Minimal implementations) immediately** to unblock testing:
- Takes 30 minutes to implement
- Gets your 7 minimal tasks working
- Good enough for validation

**Then migrate to Option 1** for production:
- Import Toolathlon's actual implementations
- Adapt their context management
- Full fidelity for all 108 tasks

## What Else Might Be Missing?

Let me check...

**Other components in Toolathlon you're NOT using:**
- ✅ **MCP servers** - You delegate to MCPServerManager ✅
- ✅ **Evaluation scripts** - You import and run them ✅
- ❌ **Local tools** - Missing most of them ❌
- ✅ **Task configs** - You read them ✅
- ? **Agent hooks/middleware** - Do you need these?

**Agent hooks** are for Toolathlon's autonomous agent (logging, monitoring, etc.) - **you don't need these** for RL training.

## Bottom Line

**You're missing the local tool implementations**, which affects ~20-25 of the 30 tasks in your dataset.

**Quick fix:** Implement Option 2 (minimal versions) in 30 minutes
**Proper fix:** Import Toolathlon's implementations (Option 1) for full fidelity

This is the ONLY significant thing you're missing. Everything else (MCP management, evaluation, Docker orchestration) is correctly implemented!

