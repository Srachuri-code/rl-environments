"""
Tool Decathlon environment for verifiers.

Thin wrapper over Toolathlon's infrastructure. We just provide glue code to:
1. Load Toolathlon tasks as a verifiers dataset
2. Run tasks in prime-sandboxes (isolated Docker containers)
3. Extract rewards from Toolathlon's eval scripts

Everything else (MCPs, tools, evaluation) is handled by Toolathlon.

Paper: https://arxiv.org/abs/2510.25726
GitHub: https://github.com/hkust-nlp/Toolathlon
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
from datasets import Dataset, load_from_disk
from loguru import logger
from openai.types.chat import ChatCompletionMessageToolCall
from prime_sandboxes import Sandbox


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
        domains: Filter tasks by domain
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        toolathlon_image: str = "toolathlon:latest",
        max_turns: int = 100,
        domains: Optional[List[str]] = None,
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
        eval_dataset = self._load_dataset(dataset_path, domains)
        rubric = self._create_rubric()
        
        super().__init__(
            eval_dataset=eval_dataset,
            rubric=rubric,
            max_turns=max_turns,
            tools=[],  # Tools managed by Toolathlon's MCPs
            **kwargs,
        )
    
    def _load_dataset(self, path: str, domains: Optional[List[str]]) -> Dataset:
        """Load Toolathlon tasks as verifiers dataset."""
        dataset = load_from_disk(path)
        if domains:
            dataset = dataset.filter(lambda x: x["domain"] in domains)
        return dataset
    
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
        info = state.get("info", {})
        task_id = info.get("task_id", state.get("task_id", "unknown"))
        
        logger.info(f"Setting up sandbox for task: {task_id}")
        
        # Create isolated sandbox
        # Toolathlon is pre-installed in the image
        sandbox = await Sandbox.acreate(
            image=self.toolathlon_image,
            timeout=3600,  # 1 hour per task
            cpu=2,
            memory_gb=4,
        )
        
        state["sandbox"] = sandbox
        state["task_id"] = task_id
        state["task_done"] = False
        state["eval_result"] = None
        
        # Initialize Toolathlon task in sandbox
        # This runs their setup scripts which:
        # - Start required MCP servers
        # - Create workspace
        # - Setup initial files
        init_script = f"""
        cd /toolathlon
        export TOOLATHLON_TASK_ID={task_id}
        
        # Their setup handles everything
        python -c "
from main import setup_task
task = setup_task('{task_id}')
print('READY')
        "
        """
        
        result = await sandbox.run(init_script)
        if "READY" not in result.stdout:
            raise RuntimeError(f"Toolathlon setup failed: {result.stderr}")
        
        # Get tools from Toolathlon's MCPs
        # (They expose them via their API)
        tools_json = await sandbox.run(
            f"cd /toolathlon && python -c \"from main import get_task_tools; import json; print(json.dumps(get_task_tools('{task_id}')))\""
        )
        tools = json.loads(tools_json.stdout)
        
        state["info"]["oai_tools"] = tools
        
        logger.info(f"Sandbox ready with {len(tools)} tools")
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
        sandbox = state["sandbox"]
        
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
            
            # Execute in sandbox
            if name == "claim_done":
                # Task completion - run Toolathlon's eval
                state["task_done"] = True
                eval_result = await self._run_toolathlon_eval(sandbox, state)
                state["eval_result"] = eval_result
                result = f"Task evaluation: {'SUCCESS' if eval_result else 'FAILED'}"
            
            else:
                # Delegate to Toolathlon's MCPs
                exec_script = f"""
                cd /toolathlon
                python -c "
from main import execute_tool
import json
result = execute_tool('{name}', {json.dumps(args)})
print(json.dumps(result))
                "
                """
                exec_result = await sandbox.run(exec_script)
                result = json.loads(exec_result.stdout)
            
            responses.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tc_id,
            })
        
        return responses, state
    
    async def _run_toolathlon_eval(self, sandbox: Sandbox, state: vf.State) -> bool:
        """
        Run Toolathlon's evaluation script in sandbox.
        
        Returns True if task succeeded, False otherwise.
        This is where you can later modify eval scripts for dense rewards.
        """
        task_id = state["task_id"]
        
        eval_script = f"""
        cd /toolathlon
        python -c "
from tasks.finalpool.{task_id}.evaluator import evaluate
import os
workspace = os.path.join('/toolathlon/agent_workspace', '{task_id}')
result = evaluate(workspace)
print('SUCCESS' if result else 'FAILED')
        "
        """
        
        try:
            result = await sandbox.run(eval_script)
            return "SUCCESS" in result.stdout
        except Exception as e:
            logger.error(f"Eval failed: {e}")
            return False
    
    async def cleanup_state(self, state: vf.State, **kwargs):
        """Cleanup sandbox (destroys entire task environment)."""
        sandbox = state.get("sandbox")
        if sandbox:
            try:
                await sandbox.adelete()
                logger.info(f"Cleaned up sandbox for {state.get('task_id')}")
            except Exception as e:
                logger.warning(f"Sandbox cleanup failed: {e}")


def load_environment(
    dataset_path: Optional[str] = None,
    toolathlon_image: str = "toolathlon:latest",
    max_turns: int = 100,
    domains: Optional[List[str]] = None,
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
        domains: Filter by task domain
    
    Example:
        env = load_environment()
        
        # Each rollout = isolated Toolathlon instance in sandbox
        for batch in dataloader:
            states = await env.setup_state(batch)  # Create sandboxes
            ...
            await env.cleanup_state(states)  # Destroy sandboxes
    """
    return ToolDecathlonEnv(
        dataset_path=dataset_path,
        toolathlon_image=toolathlon_image,
        max_turns=max_turns,
        domains=domains,
        **kwargs,
    )
