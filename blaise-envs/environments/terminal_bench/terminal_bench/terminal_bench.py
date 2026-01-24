"""
Terminal-Bench Pro implementation for verifiers with Docker sandboxing.

Evaluates LLM agents on terminal/command-line tasks across multiple domains
using isolated Docker containers for safe execution.

Uses the alibaba/terminal-bench-pro dataset which contains 200 public tasks
across 8 domains: data processing, games, debugging, system administration,
scientific computing, software engineering, machine learning, and security.

Each task provides:
- instruction: Natural language task description
- config: TOML configuration with metadata and timeouts
- archive: Binary archive containing test files and task data

Evaluation uses pytest-based verification with binary rewards (pass/fail).
"""

import asyncio
import io
import json
import logging
import shutil
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.envs.multiturn_env import MultiTurnEnv

from .utils.docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class TerminalBenchMonitorRubric(vf.Rubric):
    """Monitor rubric for tracking execution metrics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.command_timeout_count)
        self.add_metric(self.rollout_duration_seconds)
        self.add_metric(self.container_error)

    async def command_timeout_count(self, state: vf.State) -> int:
        """Count of command timeouts during rollout."""
        return state.get("command_timeout_count", 0)

    async def rollout_duration_seconds(self, state: vf.State) -> float:
        """Duration of rollout in seconds."""
        start_time = state.get("timing", {}).get("start_time", time.time())
        return time.time() - start_time

    async def container_error(self, state: vf.State) -> int:
        """Whether a container error occurred."""
        return int(state.get("container_error", False))


class TerminalBenchEnv(MultiTurnEnv):
    """
    Terminal-Bench Pro environment for evaluating LLM agents on terminal tasks.

    Uses Docker containers for sandboxed execution of bash commands.
    Each task gets its own container with the task archive extracted.
    """

    def __init__(
        self,
        difficulty: Optional[str] = None,
        category: Optional[str] = None,
        docker_image: str = "python:3.11-slim",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        command_timeout: int = 60,
        task_timeout: int = 600,
        max_turns: int = 100,
        max_command_timeouts: int = 10,
        rollout_timeout_seconds: float = 3600.0,
        runtime: str = "crun",
        **kwargs,
    ):
        eval_dataset, oai_tools = self.create_dataset(
            difficulty=difficulty,
            category=category,
        )
        rubric = self.create_rubric()

        super().__init__(
            eval_dataset=eval_dataset,
            rubric=rubric,
            oai_tools=oai_tools,
            max_turns=max_turns,
            **kwargs,
        )

        self.difficulty = difficulty
        self.category = category
        self.docker_image = docker_image
        self.command_timeout = command_timeout
        self.task_timeout = task_timeout
        self.max_command_timeouts = max_command_timeouts
        self.rollout_timeout_seconds = rollout_timeout_seconds

        # Initialize Docker executor
        self.docker_executor = DockerExecutor(
            default_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            command_timeout=command_timeout,
            runtime=runtime,
        )

        # Track active containers
        self._active_containers: Dict[str, str] = {}

        # Add monitor rubric
        self.add_rubric(TerminalBenchMonitorRubric())

    def create_dataset(
        self,
        difficulty: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[Dataset, List[Dict]]:
        """
        Load Terminal-Bench Pro dataset from HuggingFace and convert to verifiers format.

        Args:
            difficulty: Filter by difficulty level ('easy', 'medium', 'hard')
            category: Filter by category (e.g., 'data-processing', 'debugging')

        Returns:
            Tuple of (dataset, tools) for verifiers environment
        """
        # Load dataset from HuggingFace
        ds = load_dataset("alibabagroup/terminal-bench-pro", split="train")

        # Load task metadata for filtering
        task_info = {}
        try:
            task_info_path = Path(tempfile.gettempdir()) / "terminal_bench_task_info.jsonl"
            if not task_info_path.exists():
                # Download task_info.jsonl from HuggingFace
                from huggingface_hub import hf_hub_download
                info_file = hf_hub_download(
                    repo_id="alibabagroup/terminal-bench-pro",
                    filename="task_info.jsonl",
                    repo_type="dataset",
                )
                shutil.copy(info_file, task_info_path)

            with open(task_info_path, "r") as f:
                for line in f:
                    info = json.loads(line)
                    task_info[info["task_id"]] = info
        except Exception as e:
            logger.warning(f"Could not load task_info.jsonl: {e}")

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Convert to verifiers dataset format
        dataset_rows = []
        for row in ds:
            task_id = row["task_id"]
            instruction = row["instruction"]
            config_str = row["config"]
            archive = row["archive"]

            # Parse TOML config
            try:
                config = tomllib.loads(config_str)
            except Exception:
                config = {}

            # Get metadata from task_info if available
            metadata = task_info.get(task_id, {})
            task_difficulty = metadata.get("difficulty", config.get("metadata", {}).get("difficulty", "unknown"))
            task_category = metadata.get("category", config.get("metadata", {}).get("category", "unknown"))

            # Apply filters
            if difficulty and task_difficulty != difficulty:
                continue
            if category and task_category != category:
                continue

            # Get resource requirements from config
            env_config = config.get("environment", {})

            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ]

            dataset_rows.append({
                "prompt": prompt,
                "task": task_id,
                "info": {
                    "task_id": task_id,
                    "instruction": instruction,
                    "config": config,
                    "config_str": config_str,
                    "archive": archive,
                    "difficulty": task_difficulty,
                    "category": task_category,
                    "tags": metadata.get("tags", config.get("metadata", {}).get("tags", [])),
                    "cpu_cores": env_config.get("cpus", 1),
                    "memory_mb": env_config.get("memory_mb", 2048),
                    "storage_mb": env_config.get("storage_mb", 10240),
                },
            })

        if not dataset_rows:
            raise ValueError(
                f"No tasks found with difficulty={difficulty}, category={category}. "
                "Check filter values."
            )

        # Define bash tool schema
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command in the terminal. Returns stdout and stderr from the command execution.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]

        return Dataset.from_list(dataset_rows), oai_tools

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the terminal agent."""
        return """You are an expert terminal/command-line assistant. Your task is to complete terminal-based tasks by executing bash commands.

You have access to a bash tool that lets you execute commands. Use it to:
- Navigate the filesystem
- Read and write files
- Install packages if needed
- Run scripts and programs
- Perform data processing
- Debug issues

Guidelines:
1. Read the task carefully and understand what needs to be accomplished
2. Break down complex tasks into smaller steps
3. Use appropriate commands for the task at hand
4. Check command outputs and handle errors appropriately
5. Verify your work when possible

When you believe the task is complete, make sure all required outputs are in place and the task requirements are satisfied."""

    def _get_docker_image(self, state: vf.State) -> str:
        """Get Docker image for the task (can be overridden per-task)."""
        # Could extend to use task-specific images based on config
        return self.docker_image

    def create_rubric(self) -> vf.Rubric:
        """Create evaluation rubric using Terminal-Bench's test verification."""

        async def evaluate_task(state, **kwargs) -> float:
            """
            Evaluate task by running the test script inside the container.
            Returns 1.0 if tests pass, 0.0 otherwise.
            """
            container_id = state.get("container_id")
            if not container_id:
                return 0.0

            work_dir = "/workspace"

            # Look for test.sh in the tests directory
            test_paths = [
                f"{work_dir}/tests/test.sh",
                f"{work_dir}/test.sh",
                f"{work_dir}/verify.sh",
            ]

            test_script = None
            for path in test_paths:
                exit_code, _, _ = await self.docker_executor.execute_command(
                    container_id,
                    f"test -f {path} && echo exists",
                    timeout=5,
                )
                if exit_code == 0:
                    test_script = path
                    break

            if not test_script:
                # Check for reward file
                try:
                    content = await self.docker_executor.download_file(
                        container_id,
                        f"{work_dir}/logs/verifier/reward.txt",
                    )
                    return float(content.strip())
                except:
                    return 0.0

            # Create logs directory
            await self.docker_executor.execute_command(
                container_id,
                f"mkdir -p {work_dir}/logs/verifier",
                timeout=10,
            )

            try:
                # Make test script executable and run it
                config = state["info"].get("config", {})
                verifier_timeout = int(config.get("verifier", {}).get("timeout_sec", 600))

                await self.docker_executor.execute_command(
                    container_id,
                    f"chmod +x {test_script}",
                    timeout=10,
                )

                exit_code, stdout, stderr = await self.docker_executor.execute_command(
                    container_id,
                    f"cd {work_dir} && WORK_DIR={work_dir} bash {test_script}",
                    working_dir=work_dir,
                    timeout=verifier_timeout,
                )

                # Check reward file
                try:
                    content = await self.docker_executor.download_file(
                        container_id,
                        f"{work_dir}/logs/verifier/reward.txt",
                    )
                    reward_str = content.strip()
                    return float(reward_str) if reward_str else 0.0
                except:
                    pass

                # Fallback: use exit code
                return 1.0 if exit_code == 0 else 0.0

            except Exception as e:
                logger.error(f"Test execution error: {e}")
                return 0.0

        async def task_completed_metric(state, **kwargs) -> float:
            """Track whether the agent explicitly marked task as complete."""
            return 1.0 if state.get("task_completed", False) else 0.0

        rubric = vf.Rubric(funcs=[evaluate_task], weights=[1.0])
        rubric.add_metric(task_completed_metric)

        return rubric

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize task environment by creating container and extracting archive."""
        state["timing"] = {"start_time": time.time()}
        state["command_timeout_count"] = 0
        state["task_completed"] = False
        state["container_error"] = False
        state["command_count"] = 0

        # Get Docker image
        docker_image = self._get_docker_image(state)
        task_id = state["info"]["task_id"]

        logger.info(f"Setting up container for task {task_id} with image {docker_image}")

        try:
            # Create container
            container_id = await self.docker_executor.create_container(
                image=docker_image,
                working_dir="/workspace",
            )
            state["container_id"] = container_id
            self._active_containers[container_id] = container_id

            # Extract archive to container
            archive_data = state["info"]["archive"]
            if archive_data:
                # Handle different archive data formats
                if isinstance(archive_data, dict) and "bytes" in archive_data:
                    archive_bytes = archive_data["bytes"]
                elif isinstance(archive_data, bytes):
                    archive_bytes = archive_data
                else:
                    archive_bytes = bytes(archive_data)

                # Upload and extract archive
                extracted = await self.docker_executor.upload_archive(
                    container_id,
                    "/workspace",
                    archive_bytes,
                )

                if not extracted:
                    logger.warning(f"Could not extract archive for task {task_id}")

            # Create required directory structure
            await self.docker_executor.execute_command(
                container_id,
                "mkdir -p /workspace/logs/verifier",
                timeout=10,
            )

            logger.debug(f"Container {container_id} is ready for task {task_id}")

        except Exception as e:
            logger.error(f"Setup failed for task {task_id}: {repr(e)}")
            state["container_error"] = True
            state["error"] = vf.InfraError(f"Container setup failed: {e}")

        return state

    @vf.stop
    async def max_commands_reached(self, state: vf.State) -> bool:
        """Stop if too many commands have been executed."""
        max_commands = 500  # Safety limit
        return state.get("command_count", 0) >= max_commands

    @vf.stop
    async def container_exhausted(self, state: vf.State) -> bool:
        """Stop if too many command timeouts."""
        timeout_count = state.get("command_timeout_count", 0)
        if timeout_count >= self.max_command_timeouts:
            logger.warning(f"Container exhausted: {timeout_count} command timeouts")
            state["error"] = vf.InfraError("Too many command timeouts")
            return True
        return False

    @vf.stop
    async def rollout_timeout_reached(self, state: vf.State) -> bool:
        """Stop if wall-clock timeout exceeded."""
        start_time = state.get("timing", {}).get("start_time", time.time())
        elapsed = time.time() - start_time
        if elapsed > self.rollout_timeout_seconds:
            logger.warning(f"Rollout timeout: {elapsed:.0f}s > {self.rollout_timeout_seconds}s")
            state["error"] = vf.InfraError(f"Rollout timeout after {elapsed:.0f}s")
            return True
        return False

    @vf.cleanup
    async def cleanup_container(self, state: vf.State):
        """Clean up container after task completion."""
        container_id = state.get("container_id")
        if container_id:
            try:
                await self.docker_executor.remove_container(container_id)
                self._active_containers.pop(container_id, None)
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")

    @vf.teardown
    async def teardown_containers(self):
        """Clean up all containers on environment destruction."""
        await self.docker_executor.cleanup_all()

    async def execute_bash(self, command: str, state: vf.State) -> str:
        """Execute a bash command in the container."""
        container_id = state.get("container_id")
        if not container_id:
            return "Error: Container not initialized"

        try:
            exit_code, stdout, stderr = await self.docker_executor.execute_command(
                container_id,
                command,
                working_dir="/workspace",
                timeout=self.command_timeout,
            )

            # Handle timeout
            if exit_code == -1 or exit_code == 124:
                state["command_timeout_count"] = state.get("command_timeout_count", 0) + 1
                return f"Error: Command timed out after {self.command_timeout} seconds"

            # Combine output
            output = stdout
            if stderr:
                output = f"{output}\nstderr:\n{stderr}" if output else f"stderr:\n{stderr}"

            # Truncate very long outputs
            max_output_len = 50000
            if len(output) > max_output_len:
                output = output[:max_output_len] + f"\n... (output truncated, {len(output) - max_output_len} chars omitted)"

            return output if output else "(no output)"

        except Exception as e:
            logger.error(f"Command execution failed: {repr(e)}")
            state["container_error"] = True
            return f"Error: {str(e)}"

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        """Process tool calls and return execution results."""
        assert isinstance(messages, list)

        last_message = messages[-1]
        tool_calls = last_message.get("tool_calls", [])

        if not tool_calls:
            # No tool calls - check if this is a completion message
            content = last_message.get("content", "")
            if content and any(phrase in content.lower() for phrase in [
                "task complete", "task is complete", "completed the task",
                "finished", "done", "all requirements satisfied"
            ]):
                state["task_completed"] = True
            return []

        response_messages = []

        for tc in tool_calls:
            # Handle both dict and object formats
            if hasattr(tc, "id"):
                tc_id = tc.id
                func_name = tc.function.name
                func_args = tc.function.arguments
            else:
                tc_id = tc.get("id", "")
                func_name = tc.get("function", {}).get("name", "")
                func_args = tc.get("function", {}).get("arguments", "{}")

            # Parse arguments
            if isinstance(func_args, str):
                try:
                    args = json.loads(func_args)
                except json.JSONDecodeError:
                    args = {"command": func_args}
            else:
                args = func_args

            if func_name == "bash":
                command = args.get("command", "")
                state["command_count"] = state.get("command_count", 0) + 1

                # Execute the command
                output = await self.execute_bash(command, state)

                response_messages.append({
                    "role": "tool",
                    "content": output,
                    "tool_call_id": tc_id,
                })
            else:
                response_messages.append({
                    "role": "tool",
                    "content": f"Error: Unknown tool '{func_name}'. Only 'bash' is available.",
                    "tool_call_id": tc_id,
                })

        return response_messages


def load_environment(
    difficulty: Optional[str] = None,
    category: Optional[str] = None,
    docker_image: str = "python:3.11-slim",
    cpu_cores: int = 1,
    memory_gb: int = 2,
    command_timeout: int = 60,
    task_timeout: int = 600,
    max_turns: int = 100,
    max_command_timeouts: int = 10,
    rollout_timeout_seconds: float = 3600.0,
    runtime: str = "crun",
    **kwargs,
) -> vf.MultiTurnEnv:
    """Load Terminal-Bench Pro environment for verifiers with Docker sandboxing.

    This environment evaluates LLM agents on terminal/command-line tasks
    from the alibaba/terminal-bench-pro dataset using isolated Docker containers.

    Args:
        difficulty: Filter tasks by difficulty ('easy', 'medium', 'hard').
            If None, includes all difficulties.
        category: Filter tasks by category (e.g., 'data-processing', 'debugging',
            'games', 'system-administration', 'scientific-computing',
            'software-engineering', 'machine-learning', 'security').
            If None, includes all categories.
        docker_image: Docker image to use for containers (default: python:3.11-slim).
        cpu_cores: CPU cores per container (default: 1).
        memory_gb: Memory limit in GB per container (default: 2).
        command_timeout: Timeout in seconds for individual bash commands (default: 60).
        task_timeout: Overall timeout in seconds for test verification (default: 600).
        max_turns: Maximum conversation turns (default: 100).
        max_command_timeouts: Max command timeouts before aborting (default: 10).
        rollout_timeout_seconds: Wall-clock timeout for rollout (default: 3600).
        **kwargs: Additional arguments passed to environment constructor.

    Returns:
        A verifiers MultiTurnEnv configured for Terminal-Bench Pro evaluation.

    Example:
        >>> env = load_environment(difficulty="easy", category="data-processing")
        >>> # Run evaluation with vf-eval
    """
    return TerminalBenchEnv(
        difficulty=difficulty,
        category=category,
        docker_image=docker_image,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        command_timeout=command_timeout,
        task_timeout=task_timeout,
        max_turns=max_turns,
        max_command_timeouts=max_command_timeouts,
        rollout_timeout_seconds=rollout_timeout_seconds,
        runtime=runtime,
        **kwargs,
    )
