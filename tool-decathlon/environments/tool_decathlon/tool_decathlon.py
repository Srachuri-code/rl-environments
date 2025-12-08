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
import shlex
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
import verifiers as vf
from datasets import Dataset, load_from_disk
from loguru import logger
from openai.types.chat import ChatCompletionMessageToolCall


class ToolDecathlonEnv(vf.ToolEnv):
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
        
        super().__init__(
            eval_dataset=eval_dataset,
            rubric=rubric,
            max_turns=max_turns,
            tools=[],  # Tools managed by Toolathlon's MCPs
            **kwargs,
        )
    
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

        # Generate unique container name
        container_id = uuid.uuid4().hex[:12]
        container_name = f"toolathlon-{task_id}-{container_id}"

        # Create Docker container (async via thread pool to avoid blocking)
        loop = asyncio.get_running_loop()

        def create_container():
            return self.docker_client.containers.run(
                self.toolathlon_image,
                command="/bin/bash -c 'tail -f /dev/null'",  # Keep alive
                name=container_name,
                detach=True,
                remove=True,  # Auto-remove when stopped
                mem_limit="4g",
                cpu_count=2,
            )

        container = await loop.run_in_executor(None, create_container)

        state["container"] = container
        state["container_name"] = container_name
        state["task_id"] = task_id
        state["task_done"] = False
        state["eval_result"] = None

        # Ensure state["info"] exists
        if "info" not in state:
            state["info"] = {}

        # Initialize Toolathlon task in container (async via thread pool)
        # Our task_api.py wrapper starts MCPs and returns tools
        # Use uv run to execute with Toolathlon's Python environment
        def setup_task():
            return container.exec_run(
                ["/bin/bash", "-c", f"cd /toolathlon && /root/.local/bin/uv run python task_api.py setup {task_id}"],
                demux=False,
            )

        exit_code, output = await loop.run_in_executor(None, setup_task)

        if exit_code != 0:
            await loop.run_in_executor(None, container.stop)
            raise RuntimeError(f"Task setup failed: {output.decode()}")

        # Parse setup response
        try:
            setup_data = json.loads(output.decode())
        except json.JSONDecodeError as e:
            await loop.run_in_executor(None, container.stop)
            raise RuntimeError(f"Failed to parse setup response: {e}\nOutput: {output.decode()}")

        tools = setup_data.get("tools", [])

        # Add claim_done tool (only if not already present from task_api)
        tool_names = {t.get("function", {}).get("name") for t in tools}
        if "claim_done" not in tool_names:
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
        responses = []
        last_msg = messages[-1] if messages else {}
        tool_calls = last_msg.get("tool_calls", [])
        container = state.get("container")

        if not container:
            logger.error("No container in state")
            return [], state

        loop = asyncio.get_running_loop()

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

            # Execute in Docker container via task_api
            if name == "claim_done":
                # Task completion - run eval
                state["task_done"] = True
                eval_result = await self._run_toolathlon_eval(container, state)
                state["eval_result"] = eval_result
                result = f"Task evaluation: {'SUCCESS' if eval_result else 'FAILED'}"

            else:
                # Execute tool via task_api (async via thread pool)
                # Use shlex for proper shell escaping
                args_json = json.dumps(args)

                # Create closure with captured variables to avoid loop variable issues
                def execute_tool(tool_name=name, tool_args=args_json):
                    escaped_name = shlex.quote(tool_name)
                    escaped_args = shlex.quote(tool_args)
                    return container.exec_run(
                        ["/bin/bash", "-c", f"cd /toolathlon && /root/.local/bin/uv run python task_api.py execute {escaped_name} {escaped_args}"],
                        demux=False,
                    )

                exit_code, output = await loop.run_in_executor(None, execute_tool)

                if exit_code == 0:
                    result = output.decode().strip()
                else:
                    error_msg = output.decode().strip()
                    logger.warning(f"Tool {name} failed: {error_msg}")
                    result = f"Error: {error_msg}"

            responses.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tc_id,
            })

        return responses, state
    
    async def _run_toolathlon_eval(self, container, state: vf.State) -> bool:
        """
        Run Toolathlon's evaluation script via task_api.

        Returns True if task succeeded, False otherwise.
        Future: Modify eval scripts for dense rewards.
        """
        loop = asyncio.get_running_loop()

        def run_eval():
            return container.exec_run(
                ["/bin/bash", "-c", "cd /toolathlon && /root/.local/bin/uv run python task_api.py evaluate"],
                demux=False,
            )

        exit_code, output = await loop.run_in_executor(None, run_eval)

        if exit_code != 0:
            logger.error(f"Eval failed: {output.decode()}")
            return False

        try:
            result_data = json.loads(output.decode())
            return result_data.get("success", False)
        except Exception as e:
            logger.error(f"Failed to parse eval result: {e}")
            return False

    async def cleanup_state(self, state: vf.State, **kwargs):
        """Cleanup Docker container (destroys entire task environment)."""
        container = state.get("container")
        if container:
            try:
                loop = asyncio.get_running_loop()
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
