#!/usr/bin/env python3
"""
Task API - Thin wrapper to use Toolathlon's infrastructure from Docker.

Just imports and uses Toolathlon's actual classes - no reinventing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add Toolathlon to path
sys.path.insert(0, "/toolathlon")

# Import anyio (Toolathlon uses this for async)
import anyio
from anyio import Lock

# Import Toolathlon's actual infrastructure
from utils.mcp.tool_servers import MCPServerManager

# Import Toolathlon's local tools (for full parity)
from utils.aux_tools.basic import tool_sleep, tool_done
from utils.aux_tools.python_interpretor import tool_python_execute
from utils.aux_tools.web_search import tool_web_search
from utils.aux_tools.history_tools import history_tools
from utils.aux_tools.context_management_tools import context_management_tools
from utils.aux_tools.overlong_tool_manager import overlong_tool_tools


def read_json(path):
    """Simple JSON reader."""
    with open(path) as f:
        return json.load(f)


class MinimalAgentContext:
    """Minimal context wrapper for Toolathlon's local tools."""
    def __init__(self, workspace: str, messages: list = None):
        self.workspace = workspace
        self.messages = messages or []
        self.agent_workspace = workspace


class TaskAPI:
    """Persistent API that keeps MCP servers alive."""
    
    def __init__(self):
        self.task_id = None
        self.workspace = None
        self.mcp_manager = None
        self.context = None
        self.messages = []
        self._setup_lock: Lock = Lock()
        
        # Map Toolathlon's local tools by their exact names (as defined in Toolathlon)
        self.local_tools_map = {
            tool_sleep.name: tool_sleep,
            tool_done.name: tool_done,
            tool_python_execute.name: tool_python_execute,
            tool_web_search.name: tool_web_search,
        }
        
        # Add all history tools (exact names)
        for tool in history_tools:
            self.local_tools_map[tool.name] = tool
        
        # Add context management tools (exact names)
        for tool in context_management_tools:
            self.local_tools_map[tool.name] = tool
        
        # Add overlong tool handlers (exact names)
        for tool in overlong_tool_tools:
            self.local_tools_map[tool.name] = tool
    
    async def setup(
        self,
        task_id: str,
        mcp_servers: list[str] | None = None,
        local_tools: list[str] | None = None,
    ) -> dict:
        """Setup task and start MCP servers (keep them alive).
        
        Args:
            task_id: Task identifier
            mcp_servers: List of MCP servers to start (from dataset; if None, read from config)
            local_tools: List of local tool capabilities (from dataset; if None, read from config)
        """
        print(f"[SETUP] Starting setup for task: {task_id}", flush=True)
        # Prevent concurrent setup (can happen if the harness retries)
        async with self._setup_lock:
            print(f"[SETUP] Lock acquired", flush=True)
            self.task_id = task_id
            self.workspace = f"/toolathlon/agent_workspace/{task_id}"
            Path(self.workspace).mkdir(parents=True, exist_ok=True)
            print(f"[SETUP] Workspace created: {self.workspace}", flush=True)

            # Create context for local tools
            self.context = MinimalAgentContext(self.workspace, self.messages)
            self.messages = []

            # Use provided tool lists (from dataset) or fall back to reading task config
            if mcp_servers is None or local_tools is None:
                print(f"[SETUP] Reading task config from disk", flush=True)
                task_config = read_json(f"/toolathlon/tasks/finalpool/{task_id}/task_config.json")
                needed_mcps = mcp_servers if mcp_servers is not None else task_config.get("needed_mcp_servers", [])
                needed_local_tools = local_tools if local_tools is not None else task_config.get("needed_local_tools", [])
            else:
                needed_mcps = mcp_servers
                needed_local_tools = local_tools
            
            print(f"[SETUP] Will start {len(needed_mcps)} MCP servers: {needed_mcps}", flush=True)
            print(f"[SETUP] Will enable {len(needed_local_tools)} local tool caps: {needed_local_tools}", flush=True)

            # Create MCP manager (keep alive)
            print(f"[SETUP] Creating MCP manager", flush=True)
            self.mcp_manager = MCPServerManager(
                agent_workspace=self.workspace,
                config_dir="/toolathlon/configs/mcp_servers",
                debug=False,
            )

            # Connect servers
            if needed_mcps:
                print(f"[SETUP] Connecting to {len(needed_mcps)} MCP servers...", flush=True)
                for i, server_name in enumerate(needed_mcps):
                    print(f"[SETUP]   {i+1}/{len(needed_mcps)}: Connecting to {server_name}...", flush=True)
                await self.mcp_manager.connect_servers(needed_mcps)
                print(f"[SETUP] All MCP servers connected", flush=True)

            # Get tools from connected servers
            print(f"[SETUP] Listing tools from MCP servers...", flush=True)
            all_tools = []
            for server_name in self.mcp_manager.get_connected_server_names():
                print(f"[SETUP]   Listing tools from {server_name}...", flush=True)
                server = self.mcp_manager.connected_servers[server_name]
                tools_list = await server.list_tools()
                print(f"[SETUP]   Found {len(tools_list)} tools from {server_name}", flush=True)

                for tool in tools_list:
                    all_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": f"{server_name}-{tool.name}",  # Use hyphen like Toolathlon does
                                "description": tool.description or "",
                                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                            },
                        }
                    )
            
            print(f"[SETUP] Total MCP tools: {len(all_tools)}", flush=True)
            print(f"[SETUP] Expanding local tool capabilities...", flush=True)

            # Add local tools
            # Toolathlon task configs specify *capabilities* (e.g. "history") rather than
            # the concrete tool names (e.g. "search_history", "browse_history", ...).
            # Expand these capability flags into the real local FunctionTool set and
            # preserve their real JSON schemas so models can call them correctly.

            def _tool_to_oai(tool_obj) -> dict[str, Any]:
                """Convert a Toolathlon FunctionTool to OpenAI tool schema.
                
                Uses Toolathlon's exact tool name - no normalization.
                """
                name = getattr(tool_obj, "name", "")
                desc = getattr(tool_obj, "description", "") or ""
                schema = getattr(tool_obj, "params_json_schema", None) or {"type": "object", "properties": {}}

                return {
                    "type": "function",
                    "function": {
                        "name": name,  # Exact name from Toolathlon
                        "description": desc,
                        "parameters": schema,
                    },
                }

            local_tool_objs = []
            for cap in needed_local_tools:
                if cap == "history":
                    local_tool_objs.extend(history_tools)
                elif cap == "manage_context":
                    local_tool_objs.extend(context_management_tools)
                elif cap == "handle_overlong_tool_outputs":
                    local_tool_objs.extend(overlong_tool_tools)
                elif cap in ("sleep", "local-sleep"):
                    local_tool_objs.append(tool_sleep)
                elif cap in ("claim_done", "local-claim_done"):
                    local_tool_objs.append(tool_done)
                elif cap in ("python_execute", "local-python_execute", "local-python-execute"):
                    local_tool_objs.append(tool_python_execute)
                elif cap in ("web_search", "local-web_search"):
                    local_tool_objs.append(tool_web_search)
                else:
                    # Unknown capability flag; ignore silently.
                    pass

            # Deduplicate by tool name (only expose each tool once)
            print(f"[SETUP] Processing {len(local_tool_objs)} local tool objects...", flush=True)
            seen = set()
            for tool_obj in local_tool_objs:
                name = getattr(tool_obj, "name", None)
                if not name or name in seen:
                    continue
                seen.add(name)
                all_tools.append(_tool_to_oai(tool_obj))
            
            print(f"[SETUP] Total tools (MCP + local): {len(all_tools)}", flush=True)
            print(f"[SETUP] Setup complete! Returning tool schemas", flush=True)

            return {
                "status": "ready",
                "task_id": task_id,
                "workspace": self.workspace,
                "tools": all_tools,
            }
    
    async def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute tool using alive MCP connections or local tools."""
        
        # Track tool call for history
        self.messages.append({
            "tool_name": tool_name,
            "args": args,
            "timestamp": str(Path.ctime(Path(self.workspace)))
        })
        
        # Try local tools first (direct lookup - use Toolathlon's exact names)
        if tool_name in self.local_tools_map:
            try:
                tool = self.local_tools_map[tool_name]
                # Update context with current messages
                self.context.messages = self.messages
                # Call Toolathlon's tool handler
                result = await tool.on_invoke_tool(self.context, json.dumps(args))
                
                # Track result
                self.messages[-1]["result"] = str(result)[:500]  # Truncate for memory
                
                return str(result) if result else "Tool executed successfully"
            except Exception as e:
                error_msg = f"Error executing local tool {tool_name}: {e}"
                self.messages[-1]["error"] = str(e)
                return error_msg
        
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
        self.mcp_manager = None
        self.task_id = None
        self.workspace = None
        self.context = None
        self.messages = []


# Global instance (keeps servers alive across calls)
_api = TaskAPI()


async def serve(host: str, port: int):
    """
    Start a tiny HTTP server inside the container.

    This exists to support RL stepping loops: we keep MCP connections alive
    and avoid spawning a new Python process for every tool call.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse
    import uvicorn

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/setup")
    async def setup(payload: dict[str, Any]):
        task_id = payload.get("task_id")
        if not task_id:
            return JSONResponse({"error": "missing task_id"}, status_code=400)
        
        # Tool metadata can be passed from dataset (avoids re-reading config file)
        mcp_servers = payload.get("mcp_servers")  # Optional
        local_tools = payload.get("local_tools")  # Optional
        
        result = await _api.setup(str(task_id), mcp_servers, local_tools)
        return JSONResponse(result)

    @app.post("/execute")
    async def execute(payload: dict[str, Any]):
        tool_name = payload.get("tool_name")
        args = payload.get("args") or {}
        if not tool_name:
            return JSONResponse({"error": "missing tool_name"}, status_code=400)
        result = await _api.execute_tool(str(tool_name), dict(args))
        # Toolathlon tools often return text; return plain text for simplicity.
        return PlainTextResponse(str(result))

    @app.post("/evaluate")
    async def evaluate():
        success = await _api.evaluate()
        return JSONResponse({"success": bool(success)})

    @app.post("/cleanup")
    async def cleanup():
        await _api.cleanup()
        return JSONResponse({"status": "cleaned"})

    # Use Server directly to avoid nested asyncio.run() when already in async context
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


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

        elif command == "serve":
            # Example: python task_api.py serve --host 0.0.0.0 --port 8000
            host = "127.0.0.1"
            port = 8000
            if "--host" in sys.argv:
                host = sys.argv[sys.argv.index("--host") + 1]
            if "--port" in sys.argv:
                port = int(sys.argv[sys.argv.index("--port") + 1])
            await serve(host=host, port=port)
        
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
            sys.exit(1)
    
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    anyio.run(main)
