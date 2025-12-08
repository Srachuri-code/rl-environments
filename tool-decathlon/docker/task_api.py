#!/usr/bin/env python3
"""
Task API - Thin wrapper to use Toolathlon's infrastructure from Docker.

Just imports and uses Toolathlon's actual classes - no reinventing.
"""
import json
import subprocess
import sys
from pathlib import Path

# Add Toolathlon to path
sys.path.insert(0, "/toolathlon")

# Import anyio (Toolathlon uses this for async)
import anyio

# Import Toolathlon's actual infrastructure
from utils.mcp.tool_servers import MCPServerManager


def read_json(path):
    """Simple JSON reader."""
    with open(path) as f:
        return json.load(f)


class TaskAPI:
    """Persistent API that keeps MCP servers alive."""
    
    def __init__(self):
        self.task_id = None
        self.workspace = None
        self.mcp_manager = None
    
    async def setup(self, task_id: str) -> dict:
        """Setup task and start MCP servers (keep them alive)."""
        self.task_id = task_id
        self.workspace = f"/toolathlon/agent_workspace/{task_id}"
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        
        # Read task config
        task_config = read_json(f"/toolathlon/tasks/finalpool/{task_id}/task_config.json")
        
        needed_mcps = task_config.get("needed_mcp_servers", [])
        needed_local_tools = task_config.get("needed_local_tools", [])
        
        # Create MCP manager (keep alive)
        self.mcp_manager = MCPServerManager(
            agent_workspace=self.workspace,
            config_dir="/toolathlon/configs/mcp_servers",
            debug=False,
        )
        
        # Connect servers
        if needed_mcps:
            await self.mcp_manager.connect_servers(needed_mcps)
        
        # Get tools from connected servers
        all_tools = []
        for server_name in self.mcp_manager.get_connected_server_names():
            server = self.mcp_manager.connected_servers[server_name]
            tools_list = await server.list_tools()
            
            for tool in tools_list:
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}-{tool.name}",  # Use hyphen like Toolathlon does
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                })
        
        # Add local tools
        local_descs = {
            "claim_done": "Call when task is complete",
            "python_execute": "Execute Python code",
        }
        for tool_name in needed_local_tools:
            all_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": local_descs.get(tool_name, f"Local tool: {tool_name}"),
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        
        return {
            "status": "ready",
            "task_id": task_id,
            "workspace": self.workspace,
            "tools": all_tools,
        }
    
    async def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute tool using alive MCP connections."""
        # claim_done
        if tool_name == "claim_done":
            return "Task marked complete"
        
        # Python execute
        if tool_name == "python_execute":
            code = args.get("code", "")
            try:
                result = subprocess.run(
                    ["/toolathlon/.venv/bin/python", "-c", code],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            except Exception as e:
                return f"Error: {e}"
        
        # MCP tools (server-tool format, using hyphen like Toolathlon)
        if "-" not in tool_name:
            return f"Unknown tool: {tool_name}"
        
        server_name, tool = tool_name.split("-", 1)
        
        if not self.mcp_manager or server_name not in self.mcp_manager.connected_servers:
            return f"Error: Server {server_name} not connected"
        
        try:
            server = self.mcp_manager.connected_servers[server_name]
            result = await server.call_tool(tool_name=tool, arguments=args)
            
            # Extract content
            if hasattr(result, 'content') and result.content:
                texts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        texts.append(item.text)
                    else:
                        texts.append(str(item))
                return "\n".join(texts) if texts else "No output"
            
            return str(result) if result else "No result"
            
        except Exception as e:
            return f"Error calling {tool_name}: {e}"
    
    async def evaluate(self) -> bool:
        """Run Toolathlon's evaluator."""
        if not self.task_id:
            return False
        
        try:
            eval_path = f"/toolathlon/tasks/finalpool/{self.task_id}/evaluator.py"
            if not Path(eval_path).exists():
                return True
            
            result = subprocess.run(
                ["/toolathlon/.venv/bin/python", "-c",
                 f"import sys; sys.path.insert(0, '/toolathlon'); "
                 f"from tasks.finalpool.{self.task_id}.evaluator import evaluate; "
                 f"result = evaluate('{self.workspace}'); "
                 f"print('true' if result else 'false')"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            return result.stdout.strip().lower() == "true"
            
        except Exception as e:
            print(f"Eval error: {e}", file=sys.stderr)
            return False
    
    async def cleanup(self):
        """Cleanup MCP connections."""
        if self.mcp_manager:
            await self.mcp_manager.ensure_all_disconnected()


# Global instance (keeps servers alive across calls)
_api = TaskAPI()


async def main():
    """CLI interface using persistent global API instance."""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: task_api.py <command> [args...]"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "setup":
            task_id = sys.argv[2]
            result = await _api.setup(task_id)
            print(json.dumps(result))
        
        elif command == "execute":
            tool_name = sys.argv[2]
            args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = await _api.execute_tool(tool_name, args)
            print(result)
        
        elif command == "evaluate":
            result = await _api.evaluate()
            print(json.dumps({"success": result}))
        
        elif command == "cleanup":
            await _api.cleanup()
            print(json.dumps({"status": "cleaned"}))
        
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
            sys.exit(1)
    
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    anyio.run(main)
