"""
Tool Decathlon environment for verifiers.

Thin wrapper over Toolathlon's infrastructure. We just provide glue code to:
1. Load Toolathlon tasks as a verifiers dataset
2. Run tasks in isolated Docker containers
3. Extract rewards from Toolathlon's eval scripts

Everything else (MCPs, tools, evaluation) is handled by Toolathlon.

Paper: https://arxiv.org/abs/2510.25726
GitHub: https://github.com/hkust-nlp/Toolathlon
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import docker
import httpx
import verifiers as vf
from datasets import Dataset, load_from_disk
from loguru import logger
from openai.types.chat import ChatCompletionMessageToolCall


RUNTIME_CONTAINER_PORT = 8000


class ToolDecathlonEnv(vf.MultiTurnEnv):
    """
    Meta-environment wrapping Toolathlon's 108 tasks.
    
    Each task is a sub-environment with:
    - Task-specific prompts (agent harness)
    - Task-specific MCP servers (tools)
    - Task-specific eval script (reward function)
    - Isolated sandbox (workspace)
    
    Architecture:
        Training: Toolathlon runs in prime-sandbox per episode
        Rewards: Extracted from Toolathlon's eval scripts
        Future: Dense rewards by modifying eval scripts
    
    Args:
        dataset_path: Path to preprocessed Toolathlon dataset
        toolathlon_image: Docker image with Toolathlon pre-configured
        max_turns: Maximum turns per episode
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        toolathlon_image: str = "toolathlon:latest",
        max_turns: int = 100,
        runtime_timeout_s: float = 60.0,
        setup_timeout_s: float = 600.0,  # MCP server initialization can be slow (10 min)
        tool_timeout_s: float = 300.0,   # Individual tool calls (5 min)
        **kwargs,
    ):
        env_dir = Path(__file__).parent
        
        # Load dataset
        if dataset_path is None:
            dataset_path = str(env_dir.parent.parent / "data" / "tool_decathlon_dataset")
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Run: python scripts/create_hf_dataset.py"
            )
        
        self.toolathlon_image = toolathlon_image
        self.runtime_timeout_s = runtime_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.tool_timeout_s = tool_timeout_s
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Docker: {e}\n"
                "Make sure Docker is installed and running."
            )
        
        eval_dataset = self._load_dataset(dataset_path)
        rubric = self._create_rubric()
        
        # We implement the stepping loop ourselves (TauBench-style):
        # - setup_state() provisions a per-rollout runtime and injects per-task oai_tools.
        # - env_response() executes tool calls against that runtime.
        super().__init__(eval_dataset=eval_dataset, rubric=rubric, max_turns=max_turns, **kwargs)
    
    def _load_dataset(self, path: str) -> Dataset:
        """Load Toolathlon tasks as verifiers dataset."""
        return load_from_disk(path)
    
    def _create_rubric(self) -> vf.Rubric:
        """
        Create reward rubric (extensible for future dense rewards).
        
        Current: Binary task success from Toolathlon eval
        Future: Add trajectory-based rewards by modifying eval scripts
        """
        
        async def task_success_reward(completion, state, **kwargs) -> float:
            """
            Binary reward from Toolathlon's evaluation.
            
            Future extensions (edit Toolathlon's evaluator.py):
            - Partial credit for progress
            - Efficiency penalties
            - Tool usage quality
            """
            if not state.get("task_done"):
                return 0.0
            
            # Toolathlon eval result (True/False)
            eval_result = state.get("eval_result", False)
            return 1.0 if eval_result else 0.0
        
        # Easy to add more reward functions here
        return vf.Rubric(
            funcs=[task_success_reward],
            weights=[1.0],
        )
    
    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """
        Setup sub-environment (task) in isolated sandbox.
        
        Creates a sandbox with Toolathlon pre-configured and starts
        the task-specific infrastructure (MCPs, workspace, etc).
        """
        # Extract task_id - verifiers passes the full dataset row as state initially
        # Check multiple locations for robustness
        task_id = (
            state.get("task_id") or 
            state.get("info", {}).get("task_id") or
            state.get("task", "unknown")
        )
        
        logger.info(f"Setting up Docker container for task: {task_id}")
        
        # Create Docker container (async via thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        # Use UUID to avoid collisions when parallel rollouts start simultaneously
        container_name = f"toolathlon-{task_id}-{uuid.uuid4().hex[:8]}"
        container = await loop.run_in_executor(
            None,
            lambda: self.docker_client.containers.run(
                self.toolathlon_image,
                command="/bin/bash -c 'tail -f /dev/null'",  # Keep alive
                name=container_name,
                detach=True,
                remove=True,  # Auto-remove when stopped
                mem_limit="4g",
                cpu_count=2,
                ports={f"{RUNTIME_CONTAINER_PORT}/tcp": None},  # random host port
            )
        )
        
        # Make sure state["info"] exists
        state.setdefault("info", {})

        state["container"] = container
        state["task_id"] = task_id
        state["task_done"] = False
        state["eval_result"] = None

        # Start persistent runtime server inside container and do task setup over HTTP.
        runtime_url = await self._start_runtime_server(container)
        state["runtime_url"] = runtime_url

        # Get tool requirements from dataset (already fetched from task_config.json)
        mcp_servers = state.get("info", {}).get("mcp_servers", [])
        local_tools = state.get("info", {}).get("local_tools", [])

        setup_data = await self._runtime_setup(runtime_url, task_id, mcp_servers, local_tools)
        tools = setup_data.get("tools", [])
        
        # Add claim_done tool
        tools.append({
            "type": "function",
            "function": {
                "name": "claim_done",
                "description": "Call when task is complete",
                "parameters": {"type": "object", "properties": {}},
            },
        })
        
        state["info"]["oai_tools"] = tools
        
        logger.info(f"Container ready with {len(tools)} tools")
        return state
    
    async def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check if task is complete."""
        # Respect base guards (max_turns, etc.) when available.
        try:
            if await super().is_completed(messages, state, **kwargs):
                return True
        except Exception:
            # Be robust across verifiers versions.
            pass
        return state.get("task_done", False)
    
    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs
    ) -> Tuple[vf.Messages, vf.State]:
        """
        Execute tool calls in Toolathlon sandbox.
        
        Tool execution is delegated to Toolathlon's MCP servers
        running inside the sandbox.
        """
        responses: list[vf.Message] = []
        last_msg = messages[-1] if messages else {}
        tool_calls = last_msg.get("tool_calls", [])
        runtime_url = state.get("runtime_url")
        if not runtime_url:
            logger.error("No runtime_url in state")
            state["task_done"] = True
            state["eval_result"] = False
            return [], state
        
        for tc in tool_calls:
            # Parse tool call
            if isinstance(tc, ChatCompletionMessageToolCall):
                name = tc.function.name
                args_str = tc.function.arguments
                tc_id = tc.id
            else:
                name = tc.get("function", {}).get("name", "")
                args_str = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", "unknown")
            
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            
            # Execute against runtime server
            if name == "claim_done":
                # Task completion - run eval
                state["task_done"] = True
                eval_result = await self._runtime_evaluate(runtime_url)
                state["eval_result"] = eval_result
                result = f"Task evaluation: {'SUCCESS' if eval_result else 'FAILED'}"
            
            else:
                result = await self._runtime_execute(runtime_url, name, args)
            
            responses.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tc_id,
            })
        
        return responses, state
    
    async def _start_runtime_server(self, container) -> str:
        """
        Start the in-container task_api.py HTTP server and return its host URL.
        """
        loop = asyncio.get_event_loop()

        # Discover mapped host port
        await loop.run_in_executor(None, container.reload)
        ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        host_port = None
        bindings = ports.get(f"{RUNTIME_CONTAINER_PORT}/tcp")
        if bindings:
            host_port = bindings[0].get("HostPort")
        if not host_port:
            # Best-effort reload once more (docker can populate Ports slightly later)
            await asyncio.sleep(0.1)
            await loop.run_in_executor(None, container.reload)
            ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            bindings = ports.get(f"{RUNTIME_CONTAINER_PORT}/tcp")
            if bindings:
                host_port = bindings[0].get("HostPort")

        if not host_port:
            raise RuntimeError("Failed to determine host port mapping for runtime server")

        logger.info(f"Starting runtime server on container port {RUNTIME_CONTAINER_PORT} -> host port {host_port}")

        # First, test if the serve command exists and imports work
        exit_code, test_output = await loop.run_in_executor(
            None,
            lambda: container.exec_run(
                ["/bin/bash", "-c", "cd /toolathlon && /root/.local/bin/uv run python -c 'import task_api; print(\"imports ok\")'"],
                demux=False,
            ),
        )
        
        if exit_code != 0:
            error_msg = test_output.decode() if test_output else "Unknown error"
            logger.error(f"task_api.py import test failed: {error_msg}")
            raise RuntimeError(f"Cannot import task_api.py: {error_msg}")
        
        logger.debug("task_api.py imports successfully")

        # Start server inside container (detached, with output redirected to a log file)
        exec_result = await loop.run_in_executor(
            None,
            lambda: container.exec_run(
                [
                    "/bin/bash",
                    "-c",
                    f"cd /toolathlon && nohup /root/.local/bin/uv run python task_api.py serve --host 0.0.0.0 --port {RUNTIME_CONTAINER_PORT} > /tmp/runtime.log 2>&1 &",
                ],
                detach=False,  # Wait for shell to spawn the background process
            ),
        )
        
        # Give server a moment to start
        await asyncio.sleep(1.0)

        runtime_url = f"http://127.0.0.1:{host_port}"
        logger.info(f"Waiting for runtime server at {runtime_url}/health...")

        # Wait for /health
        deadline = time.time() + self.runtime_timeout_s
        last_error = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    r = await client.get(f"{runtime_url}/health")
                    if r.status_code == 200:
                        logger.info(f"Runtime server healthy at {runtime_url}")
                        return runtime_url
                except Exception as e:
                    last_error = str(e)
                
                if time.time() > deadline:
                    # Try to get server logs for debugging
                    try:
                        exit_code, log_output = await loop.run_in_executor(
                            None,
                            lambda: container.exec_run(["/bin/cat", "/tmp/runtime.log"], demux=False),
                        )
                        logs = log_output.decode() if log_output else "No logs"
                    except Exception:
                        logs = "Could not retrieve logs"
                    
                    error_msg = (
                        f"Runtime server failed to become healthy after {self.runtime_timeout_s}s\n"
                        f"Last error: {last_error}\n"
                        f"Server logs:\n{logs[:2000]}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                await asyncio.sleep(0.5)

    async def _runtime_setup(
        self, runtime_url: str, task_id: str, mcp_servers: list[str], local_tools: list[str]
    ) -> dict[str, Any]:
        """Call runtime /setup with tool metadata from dataset (no disk re-read).
        
        Uses longer timeout because MCP server initialization can be slow.
        """
        async with httpx.AsyncClient(timeout=self.setup_timeout_s) as client:
            r = await client.post(
                f"{runtime_url}/setup",
                json={
                    "task_id": task_id,
                    "mcp_servers": mcp_servers,
                    "local_tools": local_tools,
                },
            )
            r.raise_for_status()
            return r.json()

    async def _runtime_execute(self, runtime_url: str, tool_name: str, args: dict) -> str:
        async with httpx.AsyncClient(timeout=self.tool_timeout_s) as client:
            r = await client.post(f"{runtime_url}/execute", json={"tool_name": tool_name, "args": args})
            r.raise_for_status()
            return r.text

    async def _runtime_evaluate(self, runtime_url: str) -> bool:
        async with httpx.AsyncClient(timeout=self.tool_timeout_s) as client:
            r = await client.post(f"{runtime_url}/evaluate")
            r.raise_for_status()
            data = r.json()
            return bool(data.get("success", False))
    
    async def cleanup_state(self, state: vf.State, **kwargs):
        """Cleanup Docker container (destroys entire task environment)."""
        container = state.get("container")
        if container:
            try:
                loop = asyncio.get_event_loop()
                runtime_url = state.get("runtime_url")
                if runtime_url:
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            await client.post(f"{runtime_url}/cleanup")
                    except Exception:
                        pass
                await loop.run_in_executor(None, container.stop)
                logger.info(f"Cleaned up container for {state.get('task_id')}")
            except Exception as e:
                logger.warning(f"Container cleanup failed: {e}")


def load_environment(
    dataset_path: Optional[str] = None,
    toolathlon_image: str = "toolathlon:latest",
    max_turns: int = 100,
    **kwargs,
) -> ToolDecathlonEnv:
    """
    Load Tool Decathlon meta-environment.
    
    This is a thin wrapper over Toolathlon - all the heavy lifting
    (MCPs, eval, etc) is done by their infrastructure in sandboxes.
    
    Args:
        dataset_path: Path to Toolathlon dataset  
        toolathlon_image: Docker image with Toolathlon setup
        max_turns: Max turns per task
    
    Example:
        env = load_environment()
        
        # Each rollout = isolated Toolathlon instance in sandbox
        for batch in dataloader:
            states = await env.setup_state(batch)  # Create sandboxes
            # ... run your training loop here ...
            await env.cleanup_state(states)  # Destroy sandboxes
    """
    return ToolDecathlonEnv(
        dataset_path=dataset_path,
        toolathlon_image=toolathlon_image,
        max_turns=max_turns,
        **kwargs,
    )
