#!/usr/bin/env python3
"""
Minimal API for Toolathlon task execution in sandboxes.

This wraps Toolathlon's infrastructure (MCPs, eval) with a simple
step-by-step interface for RL training.

Toolathlon provides: MCPs, eval scripts, task configs
We provide: Step-by-step control for training
"""

import asyncio
import json
import sys
from pathlib import Path

# Add Toolathlon to path
sys.path.insert(0, '/toolathlon')

from utils.data_structures.task_config import TaskConfig
from utils.data_structures.mcp_config import MCPConfig
from utils.mcp.manager import MCPManager
from utils.general.helper import read_json


class TaskAPI:
    """Simple API for task execution."""
    
    def __init__(self):
        self.mcp_manager = None
        self.current_task = None
        self.workspace = None
    
    async def setup(self, task_id: str) -> dict:
        """
        Initialize task: start MCPs, load local tools, create workspace.
        Returns: {"status": "ready", "tools": [...]}
        """
        # Load task config
        task_config_path = f"/toolathlon/tasks/finalpool/{task_id}/task_config.json"
        task_config_dict = read_json(task_config_path)
        
        # Load MCP config
        mcp_config_path = "/toolathlon/configs/mcp_servers_config.json"
        mcp_config_dict = read_json(mcp_config_path) if Path(mcp_config_path).exists() else {}
        mcp_config = MCPConfig.from_dict(mcp_config_dict)
        
        # Create workspace
        self.workspace = f"/toolathlon/agent_workspace/{task_id}"
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        
        # Initialize MCP manager
        self.mcp_manager = MCPManager(mcp_config, self.workspace)
        
        # Start required MCPs
        needed_mcps = task_config_dict.get("needed_mcp_servers", [])
        await self.mcp_manager.start_servers(needed_mcps)
        
        # Get MCP tools
        mcp_tools = await self.mcp_manager.get_all_tools()
        
        # Get local tools (from Toolathlon's aux_tools)
        needed_local_tools = task_config_dict.get("needed_local_tools", [])
        local_tools = self._load_local_tools(needed_local_tools)
        
        self.current_task = task_id
        
        # Combine all tools
        all_tools = self._format_tools(mcp_tools)
        all_tools.extend(local_tools)
        
        return {
            "status": "ready",
            "task_id": task_id,
            "workspace": self.workspace,
            "tools": all_tools,
        }
    
    def _format_tools(self, tools: dict) -> list:
        """Convert MCP tools to OpenAI format."""
        result = []
        for server_name, tool_list in tools.items():
            for tool in tool_list:
                result.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool.name}",
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                })
        return result
    
    def _load_local_tools(self, tool_names: list) -> list:
        """
        Load Toolathlon's local tools (context management, python exec, etc).
        These come from utils/aux_tools/*.py
        """
        from utils.aux_tools.basic import get_all_tools as get_basic_tools
        
        try:
            # Get all available local tools from Toolathlon
            available_tools = get_basic_tools()
            
            # Filter to just what this task needs
            tools = []
            for tool_name in tool_names:
                # Find matching tool definition
                for tool in available_tools:
                    if tool.get("function", {}).get("name") == tool_name:
                        tools.append(tool)
                        break
            
            return tools
            
        except Exception as e:
            # Fallback: return minimal definitions
            print(f"Warning: Could not load local tools: {e}", file=sys.stderr)
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Tool: {name}",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in tool_names
            ]
    
    async def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a single tool call (MCP or local)."""
        if not self.mcp_manager:
            return "Error: Task not initialized"
        
        # Handle claim_done
        if tool_name == "claim_done":
            return "Task marked complete"
        
        # Handle MCP tools (server__tool format)
        if "__" in tool_name:
            server, tool = tool_name.split("__", 1)
            result = await self.mcp_manager.execute_tool(server, tool, args)
            return str(result)
        
        # Handle local tools (from Toolathlon's aux_tools)
        local_tool_result = await self._execute_local_tool(tool_name, args)
        if local_tool_result is not None:
            return local_tool_result
        
        return f"Unknown tool: {tool_name}"
    
    async def _execute_local_tool(self, tool_name: str, args: dict) -> str | None:
        """Execute Toolathlon's local tools."""
        try:
            if tool_name == "python_execute":
                from utils.aux_tools.python_interpretor import execute_python
                result = await execute_python(args.get("code", ""), self.workspace)
                return str(result)
            
            elif tool_name == "manage_context":
                from utils.aux_tools.context_management_tools import manage_context
                result = manage_context(args)
                return str(result)
            
            elif tool_name == "history":
                from utils.aux_tools.history_tools import get_history
                result = get_history(args)
                return str(result)
            
            elif tool_name == "handle_overlong_tool_outputs":
                from utils.aux_tools.overlong_tool_manager import handle_overlong
                result = handle_overlong(args)
                return str(result)
            
            else:
                return None  # Not a local tool
                
        except Exception as e:
            return f"Local tool error: {e}"
    
    async def evaluate(self) -> bool:
        """Run evaluation script for current task."""
        if not self.current_task:
            return False
        
        try:
            # Import task's evaluator
            eval_module = f"tasks.finalpool.{self.current_task}.evaluator"
            evaluator = __import__(eval_module, fromlist=['evaluate'])
            
            # Run evaluation
            result = await evaluator.evaluate(self.workspace)
            return bool(result)
            
        except Exception as e:
            print(f"Eval error: {e}", file=sys.stderr)
            return False
    
    async def cleanup(self):
        """Cleanup MCPs and workspace."""
        if self.mcp_manager:
            await self.mcp_manager.stop_all_servers()


# CLI Interface
async def main():
    """CLI interface for task API."""
    if len(sys.argv) < 2:
        print("Usage: python task_api.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    api = TaskAPI()
    
    if command == "setup":
        task_id = sys.argv[2]
        result = await api.setup(task_id)
        print(json.dumps(result))
    
    elif command == "execute":
        tool_name = sys.argv[2]
        args = json.loads(sys.argv[3])
        result = await api.execute_tool(tool_name, args)
        print(result)
    
    elif command == "evaluate":
        result = await api.evaluate()
        print(json.dumps({"success": result}))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
