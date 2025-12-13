#!/usr/bin/env python3
"""
Test which MCP servers can actually connect.

This runs inside a Toolathlon container and tries to connect to each MCP server
to identify which ones work and which ones are broken.
"""
import asyncio
import json
from pathlib import Path
import sys

# Add Toolathlon to path
sys.path.insert(0, "/toolathlon")

from utils.mcp.tool_servers import MCPServerManager


async def test_server(server_name: str, workspace: str) -> dict:
    """Test a single MCP server connection."""
    print(f"\n{'='*60}")
    print(f"Testing: {server_name}")
    print('='*60)
    
    result = {
        "server": server_name,
        "status": "unknown",
        "error": None,
        "num_tools": 0,
        "tools": [],
        "connection_time": 0,
    }
    
    try:
        import time
        start = time.time()
        
        manager = MCPServerManager(
            agent_workspace=workspace,
            config_dir="/toolathlon/configs/mcp_servers",
            debug=False,
        )
        
        print(f"  Connecting...")
        await manager.connect_servers([server_name])
        
        connection_time = time.time() - start
        result["connection_time"] = round(connection_time, 2)
        
        if server_name not in manager.connected_servers:
            result["status"] = "failed"
            result["error"] = "Not in connected_servers after connect()"
            print(f"  ❌ FAILED: Not connected")
            return result
        
        print(f"  ✓ Connected ({connection_time:.2f}s)")
        
        # Try to list tools
        server = manager.connected_servers[server_name]
        tools_list = await asyncio.wait_for(server.list_tools(), timeout=30.0)
        
        result["num_tools"] = len(tools_list)
        result["tools"] = [t.name for t in tools_list[:5]]  # First 5 tools
        result["status"] = "working"
        
        print(f"  ✓ Listed {len(tools_list)} tools")
        if result["tools"]:
            print(f"    Sample: {', '.join(result['tools'][:3])}...")
        
        # Cleanup
        await manager.ensure_all_disconnected()
        print(f"  ✓ Disconnected")
        
    except asyncio.TimeoutError as e:
        result["status"] = "timeout"
        result["error"] = f"Timeout: {str(e)}"
        print(f"  ❌ TIMEOUT: {e}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]
        print(f"  ❌ ERROR: {e}")
    
    return result


async def main():
    """Test all commonly used MCP servers."""
    
    # Get list of all MCP servers from Toolathlon configs
    config_dir = Path("/toolathlon/configs/mcp_servers")
    
    # Common MCP servers used in Toolathlon tasks
    servers_to_test = [
        # Most common (local/basic)
        "filesystem",
        "terminal", 
        "excel",
        "pdf-tools",
        
        # Frequently used
        "arxiv_local",
        "time",
        "fetch",
        "git",
        "memory",
        
        # Less common but in your datasets
        "github",
        "google_cloud",
        "bigquery",
        "canvas",
        "poste",
        "woocommerce",
        "kubernetes",
        "notion",
        "wandb",
        "huggingface",
        "snowflake",
    ]
    
    workspace = "/tmp/test_workspace"
    Path(workspace).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MCP Server Connectivity Test")
    print("="*60)
    print(f"Testing {len(servers_to_test)} MCP servers...")
    print(f"Workspace: {workspace}")
    
    results = []
    for server in servers_to_test:
        result = await test_server(server, workspace)
        results.append(result)
        await asyncio.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    working = [r for r in results if r["status"] == "working"]
    failed = [r for r in results if r["status"] in ("failed", "error")]
    timeout = [r for r in results if r["status"] == "timeout"]
    
    print(f"\n✅ Working ({len(working)}):")
    for r in working:
        print(f"  {r['server']}: {r['num_tools']} tools ({r['connection_time']}s)")
    
    if timeout:
        print(f"\n⏱️  Timeout ({len(timeout)}):")
        for r in timeout:
            print(f"  {r['server']}: {r['error']}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for r in failed:
            print(f"  {r['server']}: {r['error']}")
    
    # Save results
    output_file = "/tmp/mcp_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Full results saved to: {output_file}")
    
    print("\n" + "="*60)
    print(f"Result: {len(working)}/{len(servers_to_test)} servers working")
    print("="*60)


if __name__ == "__main__":
    import anyio
    anyio.run(main)
