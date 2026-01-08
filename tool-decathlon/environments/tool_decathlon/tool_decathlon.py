"""
Tool Decathlon environment for verifiers.

Production-ready wrapper over Toolathlon's infrastructure for RL training.
Supports ALL 108 Toolathlon tasks with proper container isolation.

Architecture:
  - Each task runs in an isolated Docker container
  - Container uses Toolathlon's pre-built image (all MCP servers included)
  - Container uses host network to access local services (Canvas, WooCommerce, etc.)
  - Docker socket mounted for K8s tasks
  - task_api.py provides HTTP interface for tool execution

Paper: https://arxiv.org/abs/2510.25726
GitHub: https://github.com/hkust-nlp/Toolathlon
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import docker
import httpx
import random
import verifiers as vf
from datasets import Dataset, load_from_disk
from loguru import logger
from openai.types.chat import ChatCompletionMessageToolCall


RUNTIME_CONTAINER_PORT = 8000

def _get_random_port() -> int:
    """Get a random port in the ephemeral range for host network mode."""
    return random.randint(30000, 60000)


class ToolDecathlonEnv(vf.MultiTurnEnv):
    """
    Meta-environment wrapping Toolathlon's 108 tasks.
    
    Each task is a sub-environment with:
    - Task-specific prompts (agent harness)
    - Task-specific MCP servers (tools)
    - Task-specific eval script (reward function)
    - Isolated sandbox (workspace)
    
    Architecture:
        Training: Toolathlon runs in Docker container per episode
        Rewards: Extracted from Toolathlon's eval scripts
        Future: Dense rewards by modifying eval scripts
    
    Args:
        dataset_path: Path to preprocessed Toolathlon dataset
        toolathlon_image: Docker image with Toolathlon pre-configured
        max_turns: Maximum turns per episode
        use_host_network: Use host network for local service access (Canvas, etc.)
        mount_docker_socket: Mount Docker socket for K8s tasks
        configs_dir: Optional path to custom Toolathlon configs
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        toolathlon_image: str = "toolathlon:latest",
        max_turns: int = 100,
        runtime_timeout_s: float = 120.0,
        setup_timeout_s: float = 600.0,  # MCP server initialization can be slow (10 min)
        tool_timeout_s: float = 300.0,   # Individual tool calls (5 min)
        use_host_network: bool = True,   # Required for local services
        mount_docker_socket: bool = True, # Required for K8s tasks
        configs_dir: Optional[str] = None,
        **kwargs,
    ):
        env_dir = Path(__file__).parent
        project_root = env_dir.parent.parent
        
        # Load dataset
        if dataset_path is None:
            dataset_path = str(project_root / "data" / "tool_decathlon_dataset")
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Run: python scripts/create_hf_dataset.py"
            )
        
        self.toolathlon_image = toolathlon_image
        self.runtime_timeout_s = runtime_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.tool_timeout_s = tool_timeout_s
        self.use_host_network = use_host_network
        self.mount_docker_socket = mount_docker_socket
        
        # Optional custom configs (credentials, tokens, etc.)
        self.configs_dir = configs_dir
        if configs_dir is None:
            # Check if toolathlon-server has configs
            toolathlon_configs = project_root / "toolathlon-server" / "configs"
            if toolathlon_configs.exists():
                self.configs_dir = str(toolathlon_configs)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Docker: {e}\n"
                "Make sure Docker is installed and running."
            )
        
        # Verify image exists
        try:
            self.docker_client.images.get(toolathlon_image)
            logger.info(f"Found Docker image: {toolathlon_image}")
        except docker.errors.ImageNotFound:
            raise RuntimeError(
                f"Docker image not found: {toolathlon_image}\n"
                "Run: bash scripts/build_toolathlon_image.sh"
            )
        
        eval_dataset = self._load_dataset(dataset_path)
        rubric = self._create_rubric()
        
        super().__init__(eval_dataset=eval_dataset, rubric=rubric, max_turns=max_turns, **kwargs)
    
    def _load_dataset(self, path: str) -> Dataset:
        """Load Toolathlon tasks as verifiers dataset."""
        return load_from_disk(path)
    
    def _create_rubric(self) -> vf.Rubric:
        """Create reward rubric."""
        
        async def task_success_reward(completion, state, **kwargs) -> float:
            """Binary reward from Toolathlon's evaluation."""
            if not state.get("task_done"):
                return 0.0
            eval_result = state.get("eval_result", False)
            return 1.0 if eval_result else 0.0
        
        return vf.Rubric(
            funcs=[task_success_reward],
            weights=[1.0],
        )
    
    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """
        Setup task in isolated Docker container.
        
        Container setup follows Toolathlon's run_single_containerized.sh:
        - Host network for local service access
        - Docker socket mounted for K8s tasks
        - Configs copied for MCP credentials
        - Task files available
        """
        # Extract task_id
        task_id = (
            state.get("task_id") or 
            state.get("info", {}).get("task_id") or
            state.get("task", "unknown")
        )
        
        logger.info(f"Setting up Docker container for task: {task_id}")
        
        loop = asyncio.get_event_loop()
        container_name = f"toolathlon-{task_id}-{uuid.uuid4().hex[:8]}"
        
        # Build container run arguments
        run_kwargs = {
            "image": self.toolathlon_image,
            "command": "/bin/bash -c 'tail -f /dev/null'",
            "name": container_name,
            "detach": True,
            "remove": True,
            "mem_limit": "8g",
            "cpu_count": 4,
            "working_dir": "/workspace",
        }
        
        # Host network for local service access (Canvas, WooCommerce, Poste, etc.)
        if self.use_host_network:
            run_kwargs["network_mode"] = "host"
            # With host network, each container needs a unique port
            runtime_port = _get_random_port()
        else:
            # Map container port to random host port
            run_kwargs["ports"] = {f"{RUNTIME_CONTAINER_PORT}/tcp": None}
            runtime_port = RUNTIME_CONTAINER_PORT
        
        # Mount Docker socket for K8s tasks
        if self.mount_docker_socket:
            docker_sock = "/var/run/docker.sock"
            if os.path.exists(docker_sock):
                run_kwargs["volumes"] = {docker_sock: {"bind": docker_sock, "mode": "rw"}}
        
        # Environment variables (for MCP servers that need them)
        run_kwargs["environment"] = {
            "DOCKER_API_VERSION": "1.44",
            "PYTHONUNBUFFERED": "1",
        }
        
        # Create container
        container = await loop.run_in_executor(
            None,
            lambda: self.docker_client.containers.run(**run_kwargs)
        )
        
        # Initialize state
        # Note: Store container ID, not container object (must be JSON serializable)
        state.setdefault("info", {})
        state["container_id"] = container.id
        state["container_name"] = container.name
        state["task_id"] = task_id
        state["task_done"] = False
        state["eval_result"] = None
        state["runtime_port"] = runtime_port
        
        # Store container object separately (not in state that gets serialized)
        if not hasattr(self, '_containers'):
            self._containers = {}
        self._containers[container.id] = container
        
        # Copy configs to container if available
        if self.configs_dir and Path(self.configs_dir).exists():
            await self._copy_configs_to_container(container, self.configs_dir)
        
        # Start runtime server and setup task
        runtime_url = await self._start_runtime_server(container, runtime_port)
        state["runtime_url"] = runtime_url
        
        # Get tool requirements from dataset
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
        
        logger.info(f"Container ready with {len(tools)} tools for task {task_id}")
        return state
    
    async def _copy_configs_to_container(self, container, configs_dir: str):
        """Copy Toolathlon configs to container for MCP credentials."""
        loop = asyncio.get_event_loop()
        try:
            # Create tar archive of configs
            import tarfile
            import io
            
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                tar.add(configs_dir, arcname='configs')
            tar_buffer.seek(0)
            
            # Copy to container
            await loop.run_in_executor(
                None,
                lambda: container.put_archive('/workspace', tar_buffer)
            )
            logger.debug("Copied configs to container")
        except Exception as e:
            logger.warning(f"Failed to copy configs: {e}")
    
    async def is_completed(self, state: vf.State, **kwargs) -> bool:
        """Check if task is complete."""
        try:
            if await super().is_completed(state, **kwargs):
                return True
        except Exception:
            pass
        return state.get("task_done", False)
    
    def get_tools(self, state: vf.State) -> list[dict]:
        """Return tools for this state (dynamic per task)."""
        return state.get("info", {}).get("oai_tools", [])
    
    def _get_container(self, state: vf.State):
        """Get container object from state's container_id."""
        container_id = state.get("container_id")
        if container_id and hasattr(self, '_containers'):
            return self._containers.get(container_id)
        return None
    
    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs
    ) -> Tuple[vf.Messages, vf.State]:
        """Execute tool calls in Toolathlon container."""
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
            
            # Execute tool
            if name == "claim_done" or name == "local-claim_done":
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
    
    async def _start_runtime_server(self, container, runtime_port: int) -> str:
        """Start task_api.py HTTP server in container."""
        loop = asyncio.get_event_loop()
        
        # Determine URL based on network mode
        if self.use_host_network:
            # With host network, server binds to localhost on the unique port
            runtime_url = f"http://127.0.0.1:{runtime_port}"
        else:
            # Get mapped port from Docker
            await loop.run_in_executor(None, container.reload)
            ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            bindings = ports.get(f"{runtime_port}/tcp")
            if not bindings:
                await asyncio.sleep(0.5)
                await loop.run_in_executor(None, container.reload)
                ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
                bindings = ports.get(f"{runtime_port}/tcp")
            
            if not bindings:
                raise RuntimeError("Failed to get port mapping")
            
            host_port = bindings[0].get("HostPort")
            runtime_url = f"http://127.0.0.1:{host_port}"
        
        logger.info(f"Starting runtime server, will be at {runtime_url}")
        
        # Start server in container (uses Toolathlon's venv)
        # PYTHONPATH must include /workspace for utils module imports
        start_cmd = (
            f"cd /workspace && "
            f"PYTHONPATH=/workspace nohup .venv/bin/python task_api.py serve "
            f"--host 0.0.0.0 --port {runtime_port} "
            f"> /tmp/runtime.log 2>&1 &"
        )
        
        await loop.run_in_executor(
            None,
            lambda: container.exec_run(["/bin/bash", "-c", start_cmd], detach=False)
        )
        
        # Wait for server to be healthy
        await asyncio.sleep(2.0)
        
        deadline = time.time() + self.runtime_timeout_s
        last_error = None
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(f"{runtime_url}/health")
                    if r.status_code == 200:
                        logger.info(f"Runtime server healthy at {runtime_url}")
                        return runtime_url
                except Exception as e:
                    last_error = str(e)
                
                await asyncio.sleep(1.0)
        
        # Failed - get logs for debugging
        try:
            _, log_output = await loop.run_in_executor(
                None,
                lambda: container.exec_run(["/bin/cat", "/tmp/runtime.log"])
            )
            logs = log_output.decode() if log_output else "No logs"
        except Exception:
            logs = "Could not retrieve logs"
        
        raise RuntimeError(
            f"Runtime server failed after {self.runtime_timeout_s}s\n"
            f"Last error: {last_error}\n"
            f"Server logs:\n{logs[:3000]}"
        )
    
    async def _runtime_setup(
        self, runtime_url: str, task_id: str, mcp_servers: list[str], local_tools: list[str]
    ) -> dict[str, Any]:
        """Setup task via HTTP API."""
        logger.info(f"Setting up task {task_id} with {len(mcp_servers)} MCP servers")
        
        async with httpx.AsyncClient(timeout=self.setup_timeout_s) as client:
            r = await client.post(
                f"{runtime_url}/setup",
                json={
                    "task_id": task_id,
                    "mcp_servers": mcp_servers,
                    "local_tools": local_tools,
                },
            )
            if r.status_code >= 400:
                error_detail = r.text
                logger.error(f"Setup failed with status {r.status_code}: {error_detail}")
                raise RuntimeError(f"Setup failed: {error_detail}")
            return r.json()
    
    async def _runtime_execute(self, runtime_url: str, tool_name: str, args: dict) -> str:
        """Execute tool via HTTP API."""
        async with httpx.AsyncClient(timeout=self.tool_timeout_s) as client:
            r = await client.post(
                f"{runtime_url}/execute",
                json={"tool_name": tool_name, "args": args}
            )
            r.raise_for_status()
            return r.text
    
    async def _runtime_evaluate(self, runtime_url: str) -> bool:
        """Run evaluation via HTTP API."""
        async with httpx.AsyncClient(timeout=self.tool_timeout_s) as client:
            r = await client.post(f"{runtime_url}/evaluate")
            r.raise_for_status()
            return r.json().get("success", False)
    
    async def cleanup_state(self, state: vf.State, **kwargs):
        """Cleanup Docker container."""
        container = self._get_container(state)
        container_id = state.get("container_id")
        
        if container:
            try:
                loop = asyncio.get_event_loop()
                runtime_url = state.get("runtime_url")
                
                # Cleanup MCP connections
                if runtime_url:
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            await client.post(f"{runtime_url}/cleanup")
                    except Exception:
                        pass
                
                # Stop container
                await loop.run_in_executor(None, container.stop)
                logger.info(f"Cleaned up container for {state.get('task_id')}")
                
                # Remove from internal tracking
                if container_id and hasattr(self, '_containers'):
                    self._containers.pop(container_id, None)
            except Exception as e:
                logger.warning(f"Container cleanup failed: {e}")


def load_environment(
    dataset_path: Optional[str] = None,
    toolathlon_image: str = "toolathlon:latest",
    max_turns: int = 100,
    use_host_network: bool = True,
    mount_docker_socket: bool = True,
    configs_dir: Optional[str] = None,
    **kwargs,
) -> ToolDecathlonEnv:
    """
    Load Tool Decathlon environment for verifiers RL training.
    
    This wraps Toolathlon's 108 tasks for reinforcement learning.
    
    Prerequisites:
        1. Build Docker image: bash scripts/build_toolathlon_image.sh
        2. (Optional) Deploy local services: bash scripts/deploy_services.sh
        3. (Optional) Configure credentials in toolathlon-server/configs/
    
    Args:
        dataset_path: Path to Toolathlon dataset
        toolathlon_image: Docker image with Toolathlon
        max_turns: Max turns per task
        use_host_network: Use host network for local services
        mount_docker_socket: Mount Docker socket for K8s tasks
        configs_dir: Path to Toolathlon configs (credentials)
    
    Example:
        env = load_environment()
        
        # Each rollout = isolated container
        for batch in dataloader:
            states = await env.setup_state(batch)
            # ... training loop ...
            await env.cleanup_state(states)
    """
    return ToolDecathlonEnv(
        dataset_path=dataset_path,
        toolathlon_image=toolathlon_image,
        max_turns=max_turns,
        use_host_network=use_host_network,
        mount_docker_socket=mount_docker_socket,
        configs_dir=configs_dir,
        **kwargs,
    )
