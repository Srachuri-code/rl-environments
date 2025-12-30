"""
Multi-SWE RL Environment

This environment combines:
- Multi-SWE dataset from PrimeIntellect/Multi-SWE-RL
- Same reward functions as mini_swe_agent_plus
- OpenHands agent harness (tools/actions) - EXACT implementation from MopenHands
- swe-rex for container management (same as Multi-SWE benchmark)
"""

import asyncio
import json
import logging
import pprint
import shlex
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal, Union

import verifiers as vf
from datasets import Dataset, load_dataset
from multi_swe_bench.harness.dataset import Dataset as MultiSWEDataset
from multi_swe_bench.harness.image import Config
from multi_swe_bench.harness.instance import Instance
from multi_swe_bench.harness.report import Report, generate_report
from multi_swe_bench.harness.test_result import TestResult
from swerex.deployment.docker import DockerDeployment
from swerex.deployment.config import DockerDeploymentConfig
from swerex.runtime.abstract import BashAction, CreateBashSessionRequest, WriteFileRequest
from swerex.runtime.remote import RemoteRuntime
from swerex.exceptions import CommandTimeoutError as SweRexTimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential
from verifiers.types import ChatCompletionMessageToolCall, Info, Message, Messages, ProcessedOutputs, State


# Suppress noisy loggers
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("swerex").setLevel(logging.WARNING)


# ============================================================================
# Multi-SWE Dataset Utilities (from mini_swe_agent_plus)
# ============================================================================


def columnar_to_tests(entry):
    """Convert columnar test format to dict format."""
    return {
        name: {"fix": fix, "run": run, "test": test}
        for name, fix, run, test in zip(entry["name"], entry["fix"], entry["run"], entry["test"])
    }


def columnar_to_resolved_issues(entry):
    """Convert columnar resolved issues format to list format."""
    return [
        {"body": body, "number": num, "title": title}
        for body, num, title in zip(entry["body"], entry["number"], entry["title"])
    ]


def restore_row(row):
    """Restore row from columnar format to nested format."""
    row = dict(row)
    test_fields = ["fixed_tests", "p2p_tests", "f2p_tests", "s2p_tests", "n2p_tests"]
    for field in test_fields:
        row[field] = columnar_to_tests(row[field])
    row["resolved_issues"] = columnar_to_resolved_issues(row["resolved_issues"])
    return row


def create_instance(dataset: MultiSWEDataset) -> Instance:
    """Create an Instance from a MultiSWEDataset."""
    config = Config(
        need_clone=False,
        global_env=None,
        clear_env=False,
    )
    return Instance.create(
        pr=dataset,
        config=config,
    )


def validate_report_against_dataset(report: Report, multiswe_ds: MultiSWEDataset) -> tuple[bool, str | None]:
    """
    Validate that a report matches the expected test transitions from the dataset.

    Args:
        report: The Report object to validate
        multiswe_ds: The MultiSWEDataset object containing expected test transitions

    Returns:
        A tuple of (is_valid, error_message)
    """
    if not report.valid:
        return (False, f"Report is not valid: {report.error_msg}")

    # Check p2p_tests (pass-to-pass tests)
    for p2p in multiswe_ds.p2p_tests:
        if p2p not in report.p2p_tests:
            return (
                False,
                f"Missing expected p2p_test: {p2p}. Expected {len(multiswe_ds.p2p_tests)} p2p_tests, found {len(report.p2p_tests)}",
            )

    # Check f2p_tests (fail-to-pass tests) - most critical check
    for f2p in multiswe_ds.f2p_tests:
        if f2p not in report.f2p_tests:
            return (
                False,
                f"Missing expected f2p_test: {f2p}. Expected {len(multiswe_ds.f2p_tests)} f2p_tests, found {len(report.f2p_tests)}",
            )

    # Check s2p_tests (skip-to-pass tests)
    for s2p in multiswe_ds.s2p_tests:
        if s2p not in report.s2p_tests:
            return (
                False,
                f"Missing expected s2p_test: {s2p}. Expected {len(multiswe_ds.s2p_tests)} s2p_tests, found {len(report.s2p_tests)}",
            )

    # Check n2p_tests (none-to-pass tests)
    for n2p in multiswe_ds.n2p_tests:
        if n2p not in report.n2p_tests:
            return (
                False,
                f"Missing expected n2p_test: {n2p}. Expected {len(multiswe_ds.n2p_tests)} n2p_tests, found {len(report.n2p_tests)}",
            )

    return (True, None)


# ============================================================================
# Custom Exception Classes
# ============================================================================


class SandboxError(Exception):
    """Base exception for sandbox-related errors."""
    pass


class CommandTimeoutError(SandboxError):
    """Raised when a command times out."""
    def __init__(self, sandbox_id: str, command: str, timeout: int):
        self.sandbox_id = sandbox_id
        self.command = command
        self.timeout = timeout
        super().__init__(f"Command '{command}' timed out after {timeout}s in sandbox {sandbox_id}")


class ContainerError(SandboxError):
    """Raised when there's a container infrastructure error."""
    pass


# ============================================================================
# Scripts (embedded for Multi-SWE harness)
# ============================================================================

CREATE_FIX_PATCH_SCRIPT = '''#!/bin/bash
# Generate fix.patch from unstaged changes, excluding test files

set -e

REPO_DIR="${1:-$(pwd)}"
cd "$REPO_DIR"

MODIFIED_FILES=$(git diff --name-only)

if [ -z "$MODIFIED_FILES" ]; then
    echo "No unstaged changes found. Creating empty patch." >&2
    touch /home/fix.patch
    exit 0
fi

for file in $MODIFIED_FILES; do
    file_lower=$(echo "$file" | tr '[:upper:]' '[:lower:]')
    if [[ "$file_lower" == *"test"* ]] || \\
       [[ "$file_lower" == *"tests"* ]] || \\
       [[ "$file_lower" == *"e2e"* ]] || \\
       [[ "$file_lower" == *"testing"* ]]; then
        echo "Warning: Test files were modified. Creating empty patch." >&2
        touch /home/fix.patch
        git restore .
        exit 0
    fi
done

git diff -- $MODIFIED_FILES > /home/fix.patch

if [ ! -s /home/fix.patch ]; then
    echo "Generated patch is empty. Creating empty patch file." >&2
    touch /home/fix.patch
fi

git restore .

echo "Created fix.patch at /home/fix.patch" >&2
echo "Unstaged changes have been restored." >&2
'''

# ============================================================================
# OpenHands Prompts - EXACT from MopenHands
# Source: https://github.com/multi-swe-bench/MopenHands
# ============================================================================

# System prompt from openhands/agenthub/codeact_agent/prompts/system_prompt.j2
SYSTEM_PROMPT = """You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
* When configuring git credentials, use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* The assistant MUST NOT include comments in the code unless they are necessary to describe non-obvious behavior.
</IMPORTANT>
"""

# User prompt from evaluation/benchmarks/swe_bench/run_infer.py (Python version)
# NOTE: Multi-SWE Docker images have repos at /home/{repo}, not /workspace/
PROMPT_TEMPLATE = """<uploaded_files>
/home/{workspace_dir_name}
</uploaded_files>
I've uploaded a python code repository in the directory /home/{workspace_dir_name}. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error.
3. Edit the sourcecode of the repo to resolve the issue.
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.
6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:
   - The issue you are fixing
   - The files you modified
   - The functions you changed
   Make sure all these tests pass with your changes.
Your thinking should be thorough and so it's fine if it's very long.
"""

# Format error template
FORMAT_ERROR_TEMPLATE = """Please provide EXACTLY ONE tool call, found {num_actions} tool calls."""


# ============================================================================
# OpenHands Tool Descriptions - EXACT from MopenHands function_calling.py
# Source: https://github.com/multi-swe-bench/MopenHands/blob/main/openhands/agenthub/codeact_agent/function_calling.py
# ============================================================================

_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, the assistant can interact with the running process and send empty `command` to retrieve any additional logs, or send additional text (set `command` to the text) to STDIN of the running process, or send command like `C-c` (Ctrl+C), `C-d` (Ctrl+D), `C-z` (Ctrl+Z) to interrupt the process.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.
"""

_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

_FINISH_DESCRIPTION = """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task."""


# ============================================================================
# Helper functions
# ============================================================================


def _is_transient_error(exception: Exception) -> bool:
    """Check if exception is a transient error that should be retried."""
    error_str = str(exception).lower()
    # Retry on common transient errors
    return any(err in error_str for err in ["timeout", "connection", "unavailable", "refused", "503", "502"])


# Environment variables for sandbox
PATH = "PATH=/testbed/.venv/bin:/root/.local/bin:/root/.cargo/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV_VARS = f"{PATH};PAGER=cat;MANPAGER=cat;LESS=-R;PIP_PROGRESS_BAR=off;TQDM_DISABLE=1;"


# ============================================================================
# swe-rex Sandbox Wrapper
# ============================================================================


class SweRexSandbox:
    """
    Wrapper around swe-rex DockerDeployment for managing containers.
    This is the same approach used by the Multi-SWE benchmark.
    """

    def __init__(
        self,
        docker_image: str,
        session_name: str = "main",
        startup_timeout: int = 120,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize a swe-rex sandbox.

        Args:
            docker_image: Full Docker image name (e.g., "mswebench/pandas-dev_m_pandas:pr-12345")
            session_name: Name for the bash session
            startup_timeout: Timeout for container startup
            logger: Logger instance
        """
        self.docker_image = docker_image
        self.session_name = session_name
        self.startup_timeout = startup_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        self._deployment: DockerDeployment | None = None
        self._runtime: RemoteRuntime | None = None
        self._started = False

    async def start(self) -> None:
        """Start the sandbox container using swe-rex."""
        if self._started:
            return

        config = DockerDeploymentConfig(
            image=self.docker_image,
            port=None,  # Auto-assign port
            startup_timeout=self.startup_timeout,
            pull="never",  # Images should be pre-warmed
            remove_container=True,  # Clean up on stop
            remove_images=False,  # Keep images for reuse
            python_standalone_dir=None,  # Use default Python
        )

        self._deployment = DockerDeployment(logger=self.logger, **config.model_dump())
        
        self.logger.debug(f"Starting swe-rex container for {self.docker_image}")
        await self._deployment.start()
        
        self._runtime = self._deployment.runtime
        
        # Create a bash session for command execution
        await self._runtime.create_session(
            CreateBashSessionRequest(
                session=self.session_name,
                startup_source=["/root/.bashrc"],
                startup_timeout=self.startup_timeout,
            )
        )
        
        self._started = True
        self.logger.debug(f"swe-rex container started: {self._deployment.container_name}")

    async def stop(self) -> None:
        """Stop the sandbox container."""
        if not self._started:
            return

        try:
            if self._deployment:
                await self._deployment.stop()
        except Exception as e:
            self.logger.warning(f"Error stopping swe-rex container: {e}")
        finally:
            self._deployment = None
            self._runtime = None
            self._started = False

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 90,
    ) -> tuple[int, str]:
        """
        Execute a command in the sandbox.

        Args:
            command: Command to execute
            cwd: Working directory (prepended as cd command)
            timeout: Command timeout in seconds

        Returns:
            Tuple of (exit_code, output)
        """
        if not self._started or not self._runtime:
            raise SandboxError("Sandbox not started")

        # Prepend cd if working directory specified
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"

        try:
            result = await self._runtime.run_in_session(
                BashAction(
                    session=self.session_name,
                    command=command,
                    timeout=timeout,
                    set_last_action=True,
                    check="silent",
                )
            )
            return (result.exit_code, result.output)
        except SweRexTimeoutError:
            raise asyncio.TimeoutError(f"Command timed out after {timeout}s")

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the sandbox."""
        if not self._started or not self._runtime:
            raise SandboxError("Sandbox not started")

        await self._runtime.write_file(
            WriteFileRequest(path=path, content=content)
        )

    @property
    def container_name(self) -> str | None:
        """Get the container name."""
        if self._deployment:
            return self._deployment.container_name
        return None


# ============================================================================
# Main Environment Class
# ============================================================================


class MultiSWEOpenHandsEnv(vf.StatefulToolEnv):
    """
    Multi-SWE RL Environment with OpenHands agent harness.

    Uses swe-rex for container management (same as Multi-SWE benchmark).

    Combines:
    - Multi-SWE dataset from PrimeIntellect/Multi-SWE-RL
    - Reward functions from mini_swe_agent_plus
    - OpenHands-style tools: execute_bash, str_replace_editor, finish
    - swe-rex Docker sandbox infrastructure
    """

    def __init__(
        self,
        eval_dataset: Any,
        rubric: vf.Rubric,
        system_prompt: str = SYSTEM_PROMPT,
        max_turns: int = 200,
        turn_timeout: int = 90,
        test_timeout: int = 1800,
        total_timeout_minutes: int = 120,
        startup_timeout: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tools=[],  # We add tools manually via add_tool()
            eval_dataset=eval_dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
        )

        self.turn_timeout = turn_timeout
        self.test_timeout = test_timeout
        self.total_timeout_minutes = total_timeout_minutes
        self.startup_timeout = startup_timeout

        # Track active sandboxes for cleanup
        self.active_sandboxes: dict[str, SweRexSandbox] = {}

        # Add OpenHands-style tools
        # Per MopenHands SWE-bench config: codeact_enable_jupyter=False, codeact_enable_llm_editor=False
        # This means: execute_bash + str_replace_editor + finish (NO execute_ipython_cell)
        self.add_tool(self.execute_bash, args_to_skip=["sandbox_id", "turn_timeout", "working_dir"])
        self.add_tool(self.str_replace_editor, args_to_skip=["sandbox_id", "turn_timeout", "working_dir"])
        self.add_tool(self.finish, args_to_skip=["sandbox_id", "turn_timeout", "working_dir"])

    # ========================================================================
    # Command execution helpers
    # ========================================================================

    @retry(
        retry=retry_if_exception_type(RuntimeError),
        stop=stop_after_delay(180),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _execute_command(
        self, command: str, sandbox_id: str, timeout: int = 90, working_dir: str = None
    ) -> tuple[int, str]:
        """Execute command inside persistent sandbox container."""
        self.logger.debug(f"Executing command in sandbox {sandbox_id}: {command[:100]}...")

        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            raise SandboxError(f"Sandbox {sandbox_id} not found")

        try:
            exit_code, output = await sandbox.exec(
                command=command,
                cwd=working_dir,
                timeout=timeout,
            )
            return (exit_code, output.strip() or "(no output)")
        except asyncio.TimeoutError:
            self.logger.warning(f"Command timed out after {timeout}s: {command}")
            return (
                -1,
                f"The command timed out after {timeout}s and has been killed. Please try a different command.",
            )
        except Exception as e:
            if _is_transient_error(e):
                self.logger.warning(f"Transient error, will retry: {repr(e)}")
                raise RuntimeError(f"Transient error: {repr(e)}")
            self.logger.error(f"Execution error: {repr(e)}")
            self.logger.error(traceback.format_exc())
            return (1, "Command failed due to infrastructure error. Try the same command again!")

    async def execute_command_raise_on_error(
        self, sandbox_id: str, command: str, working_dir: str = None, timeout: int = 90
    ) -> tuple[int, str]:
        """Execute command and raise on error."""
        exit_code, output = await self._execute_command(command, sandbox_id, timeout, working_dir)
        if exit_code != 0:
            raise RuntimeError(
                f"Error executing command: {command} return_code={exit_code} output={output}"
            )
        return (exit_code, output)

    def _format_observation(self, exit_code: int, output: str) -> str:
        """Format tool output in OpenHands observation format."""
        # Truncate long outputs (OpenHands behavior)
        if len(output) > 10000:
            output = output[:5000] + f"\n\n... (truncated {len(output) - 10000} characters) ...\n\n" + output[-5000:]

        return f"OBSERVATION:\n{output}\n[Command finished with exit code {exit_code}]"

    # ========================================================================
    # OpenHands-style Tools - EXACT from MopenHands
    # ========================================================================

    async def execute_bash(
        self,
        command: str,
        is_input: str = "false",
        sandbox_id: str | None = None,
        turn_timeout: int = 90,
        working_dir: str = None,
    ) -> str:
        """
        Execute a bash command in the terminal.
        * Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
        * Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, the assistant can interact with the running process and send empty `command` to retrieve any additional logs, or send additional text (set `command` to the text) to STDIN of the running process, or send command like `C-c` (Ctrl+C), `C-d` (Ctrl+D), `C-z` (Ctrl+Z) to interrupt the process.
        * One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.

        Args:
            command: The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.
            is_input: If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.
        """
        if sandbox_id is None:
            raise ValueError("sandbox_id is required for execute_bash")

        # Handle special inputs
        if is_input == "true":
            # For interactive input, we append to the running process
            cmd = f"echo {shlex.quote(command)}"
        else:
            cmd = command

        full_cmd = f"{ENV_VARS} {cmd}"
        exit_code, output = await self._execute_command(full_cmd, sandbox_id, turn_timeout, working_dir=working_dir)

        return self._format_observation(exit_code, output)

    async def str_replace_editor(
        self,
        command: str,
        path: str,
        file_text: str | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        view_range: list[int] | None = None,
        sandbox_id: str | None = None,
        turn_timeout: int = 90,
        working_dir: str = None,
    ) -> str:
        """
        Custom editing tool for viewing, creating and editing files in plain-text format
        * State is persistent across command calls and discussions with the user
        * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
        * The `create` command cannot be used if the specified `path` already exists as a file
        * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
        * The `undo_edit` command will revert the last edit made to the file at `path`

        Notes for using the `str_replace` command:
        * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
        * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
        * The `new_str` parameter should contain the edited lines that should replace the `old_str`

        Args:
            command: The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
            path: Absolute path to file or directory, e.g. `/home/repo/file.py` or `/home/repo`.
            file_text: Required parameter of `create` command, with the content of the file to be created.
            old_str: Required parameter of `str_replace` command containing the string in `path` to replace.
            new_str: Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
            insert_line: Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
            view_range: Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
        """
        if sandbox_id is None:
            raise ValueError("sandbox_id is required for str_replace_editor")

        if command == "view":
            if view_range:
                start, end = view_range[0], view_range[1] if len(view_range) > 1 else -1
                if end == -1:
                    cmd = f"sed -n '{start},$p' {shlex.quote(path)} | nl -ba -v {start}"
                else:
                    cmd = f"sed -n '{start},{end}p' {shlex.quote(path)} | nl -ba -v {start}"
            else:
                # Check if path is directory or file
                cmd = f"if [ -d {shlex.quote(path)} ]; then find {shlex.quote(path)} -maxdepth 2 -type f | head -100; else cat -n {shlex.quote(path)}; fi"

        elif command == "create":
            if file_text is None:
                return "Error: file_text is required for create command"
            # Check if file already exists
            check_cmd = f"test -f {shlex.quote(path)} && echo 'EXISTS' || echo 'NOT_EXISTS'"
            exit_code, check_output = await self._execute_command(
                f"{ENV_VARS} {check_cmd}", sandbox_id, turn_timeout, working_dir=working_dir
            )
            if "EXISTS" in check_output:
                return f"Error: File already exists at {path}. Cannot overwrite files with `create` command. Use `str_replace` instead."

            cmd = f"cat > {shlex.quote(path)} << 'CREATEFILEEOF'\n{file_text}\nCREATEFILEEOF"

        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str is required for str_replace command"
            new_str = new_str or ""

            # Create a Python script for safe replacement
            replace_script = f'''
import sys
path = {repr(path)}
old_str = {repr(old_str)}
new_str = {repr(new_str)}

with open(path, 'r') as f:
    content = f.read()

count = content.count(old_str)
if count == 0:
    print(f"Error: old_str not found in {{path}}", file=sys.stderr)
    sys.exit(1)
elif count > 1:
    print(f"Error: old_str found {{count}} times in {{path}}. Must be unique.", file=sys.stderr)
    sys.exit(1)

new_content = content.replace(old_str, new_str, 1)
with open(path, 'w') as f:
    f.write(new_content)

print(f"Successfully replaced in {{path}}")
'''
            escaped_script = replace_script.replace("'", "'\\''")
            cmd = f"python3 -c '{escaped_script}'"

        elif command == "insert":
            if insert_line is None or new_str is None:
                return "Error: insert_line and new_str are required for insert command"
            # Use sed to insert after line
            escaped_new_str = new_str.replace("'", "'\\''").replace("\n", "\\n")
            cmd = f"sed -i '{insert_line}a\\{escaped_new_str}' {shlex.quote(path)}"

        elif command == "undo_edit":
            # Try to restore from git
            cmd = f"git checkout -- {shlex.quote(path)} 2>/dev/null || echo 'No backup available for undo'"

        else:
            return f"Error: Unknown command '{command}'. Use one of: view, create, str_replace, insert, undo_edit"

        full_cmd = f"{ENV_VARS} {cmd}"
        exit_code, output = await self._execute_command(full_cmd, sandbox_id, turn_timeout, working_dir=working_dir)

        return self._format_observation(exit_code, output)

    async def finish(
        self,
        sandbox_id: str | None = None,
        turn_timeout: int = 90,
        working_dir: str = None,
    ) -> str:
        """
        Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
        """
        if sandbox_id is None:
            raise ValueError("sandbox_id is required for finish")

        return "OBSERVATION:\n<<<FINISH>>>\nYour task has been completed. The changes will now be evaluated."

    # ========================================================================
    # Sandbox Setup (using swe-rex)
    # ========================================================================

    async def setup_repo(self, sandbox_id: str, state: State):
        """Set up Multi-SWE repository in the sandbox."""
        info = restore_row(state["info"])
        instance_id = info["instance_id"]

        # Install curl if needed
        exit_code, output = await self.execute_command_raise_on_error(
            sandbox_id,
            "which curl || (apt-get update && apt-get install -y curl)",
            working_dir="/home",
            timeout=300,
        )
        self.logger.debug(f"CURL install results for {instance_id=}: {output}")

        # Install uv
        exit_code, output = await self.execute_command_raise_on_error(
            sandbox_id,
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            working_dir="/home",
            timeout=300,
        )
        self.logger.debug(f"UV install results: {output}")

        # Delete pre-existing fix.patch
        await self._execute_command(
            "rm -f /home/fix.patch", sandbox_id, timeout=30, working_dir="/home"
        )

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        """Create per-rollout sandbox using swe-rex."""
        docker_image = state["info"]["docker_image"]
        self.logger.info(f"Setting up swe-rex sandbox for image: {docker_image}")

        # Create unique session ID
        session_id = f"multi-swe-{uuid.uuid4().hex[:12]}"
        full_image = f"mswebench/{docker_image}".lower()

        try:
            # Create and start swe-rex sandbox
            sandbox = SweRexSandbox(
                docker_image=full_image,
                session_name="main",
                startup_timeout=self.startup_timeout,
                logger=self.logger,
            )

            self.logger.debug(f"Starting swe-rex sandbox {session_id}...")
            await sandbox.start()

            # Store sandbox reference using container name as ID
            sandbox_id = sandbox.container_name or session_id
            self.active_sandboxes[sandbox_id] = sandbox
            state["sandbox_id"] = sandbox_id

            # Setup repository
            self.logger.debug(f"Setting up repository for sandbox {sandbox_id}...")
            await self.setup_repo(sandbox_id, state)
            self.logger.debug(f"swe-rex sandbox {sandbox_id} is ready.")

        except Exception as e:
            self.logger.error(f"Error creating swe-rex sandbox:\n\n{repr(e)}")
            self.logger.error(traceback.format_exc())
            state["sandbox_id"] = None
            state["sandbox_error"] = 1

        return state

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Destroy a sandbox."""
        sandbox = self.active_sandboxes.pop(sandbox_id, None)
        if sandbox:
            try:
                await sandbox.stop()
            except Exception as e:
                self.logger.warning(f"Error destroying sandbox {sandbox_id}: {repr(e)}")

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        """Inject sandbox_id and working_dir into tool calls."""
        tools_needing_sandbox = [
            "execute_bash",
            "str_replace_editor",
            "finish",
        ]
        if tool_name in tools_needing_sandbox:
            updated_args = dict(tool_args)
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["turn_timeout"] = self.turn_timeout

            # Set working_dir based on Multi-SWE harness
            info = restore_row(state["info"])
            repo = info["repo"]
            # Multi-SWE Docker images have repos at /home/{repo}
            updated_args["working_dir"] = f"/home/{repo}"
            return updated_args
        else:
            return tool_args

    # ========================================================================
    # Environment Response
    # ========================================================================

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """Process tool calls and return environment messages."""
        assert isinstance(messages, list)
        env_messages = []

        if "tool_calls" in messages[-1]:
            tool_calls = messages[-1]["tool_calls"]
            
            # OpenHands requires single tool call per turn
            # But we MUST respond to ALL tool_call_ids to satisfy OpenAI API
            if len(tool_calls) != 1:
                error_msg = FORMAT_ERROR_TEMPLATE.format(num_actions=len(tool_calls))
                # Respond to EACH tool_call_id with error (OpenAI requires this)
                for tc in tool_calls:
                    tc_id = tc.id if hasattr(tc, 'id') else tc.get("id", "unknown")
                    env_messages.append({
                        "role": "tool",
                        "content": error_msg,
                        "tool_call_id": tc_id,
                    })
                return env_messages

            for tool_call in tool_calls:
                if isinstance(tool_call, ChatCompletionMessageToolCall):
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id or ""
                elif isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args_str = func.get("arguments", "{}")
                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                        tool_call_id = tool_call.get("id", "")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing tool arguments: {e}")
                        tool_message = {
                            "role": "tool",
                            "content": f"Error: Failed to parse tool call arguments. Error: {e}",
                            "tool_call_id": "invalid",
                        }
                        env_messages.append(tool_message)
                        state["is_completed"] = True
                        return env_messages
                else:
                    self.logger.warning(f"Unexpected tool_call type: {type(tool_call)}")
                    continue

                try:
                    tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
                    tool_message: Message = await self.call_tool(tool_name, tool_args, tool_call_id)
                except ValueError as e:
                    tool_message = {
                        "role": "tool",
                        "content": f"Error: Failed to call tool '{tool_name}': {e}",
                        "tool_call_id": tool_call_id,
                    }
                    self.logger.error(f"Error calling tool: {e}")
                except Exception as e:
                    tool_message = {
                        "role": "tool",
                        "content": f"Error executing tool '{tool_name}': {repr(e)}",
                        "tool_call_id": tool_call_id,
                    }
                    self.logger.error(f"Error executing tool '{tool_name}': {repr(e)}")
                    self.logger.error(traceback.format_exc())
                env_messages.append(tool_message)

        self.logger.debug(f"Env Response Messages:\n{pprint.pformat(env_messages)}")
        return env_messages

    # ========================================================================
    # Test Execution
    # ========================================================================

    async def run_tests(self, state: State, test_timeout: int = 1800) -> str:
        """Run tests for Multi-SWE harness."""
        info = restore_row(state["info"])
        instance_id = info["instance_id"]
        repo = info["repo"]
        sandbox_id = state["sandbox_id"]

        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            raise SandboxError(f"Sandbox {sandbox_id} not found")

        # Upload create_fix_patch script using swe-rex
        await sandbox.write_file("/home/create_fix_patch.sh", CREATE_FIX_PATCH_SCRIPT)

        # Run create_fix_patch script
        exit_code, output = await sandbox.exec(
            command=f"{ENV_VARS} bash /home/create_fix_patch.sh",
            cwd=f"/home/{repo}",
            timeout=min(test_timeout, 300),
        )
        self.logger.debug(f"create_fix_patch.sh results for {instance_id=}: {output}")

        # Run fix-run.sh to apply patch and run tests
        try:
            exit_code, output = await sandbox.exec(
                command=f"{ENV_VARS} bash -o pipefail -c 'bash /home/fix-run.sh 2>&1 | tee /home/test_output.txt'",
                cwd=f"/home/{repo}",
                timeout=test_timeout,
            )
            if exit_code != 0:
                self.logger.warning(f"fix-run.sh returned non-zero exit code: {exit_code}")
        except asyncio.TimeoutError:
            self.logger.error(f"Command timed out: bash /home/fix-run.sh after {test_timeout}s")
            state["sandbox_error"] = 1
            return ""
        except Exception as e:
            self.logger.error(f"Command error: bash /home/fix-run.sh: {repr(e)}")
            state["sandbox_error"] = 1
            return ""

        # Read test output
        exit_code, test_output = await self._execute_command(
            "cat /home/test_output.txt", sandbox_id, timeout=60
        )
        return test_output or ""

    async def post_rollout(self, state: State) -> None:
        """Run tests after rollout completes."""
        try:
            state["test_output"] = await self.run_tests(state, test_timeout=self.test_timeout)
            self.logger.debug(f"Test output:\n{state['test_output']}")
            self.logger.debug(f"Total turns taken: {len(state['trajectory'])}")
        except Exception as e:
            state["instance_solved"] = False
            state["error"] = repr(e)
            state["test_output"] = ""
            self.logger.debug(f"Error: {repr(e)}")
            self.logger.debug(traceback.format_exc())
        finally:
            # Cleanup sandbox
            sandbox_id = state.get("sandbox_id")
            if sandbox_id:
                await self.destroy_sandbox(sandbox_id)

    # ========================================================================
    # Stop Conditions
    # ========================================================================

    @vf.stop(priority=1)
    async def is_done(self, state: State) -> bool:
        """Check if rollout should stop."""
        if state.get("sandbox_error") == 1:
            self.logger.error("Sandbox error. Aborting rollout.")
            return True

        completed = False
        last_traj = state["trajectory"][-1] if state["trajectory"] else {}
        last_completion = last_traj.get("completion", [])
        for msg in reversed(last_completion):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                if "<<<FINISH>>>" in msg.get("content", ""):
                    completed = True
                    state["instance_completed"] = completed
                    break
        return completed

    # ========================================================================
    # VLLM Processing
    # ========================================================================

    def process_env_results_vllm(
        self, prompts: list[Messages], completions: list[Messages], states: list[State], *args, **kwargs
    ) -> ProcessedOutputs:
        """Process results for VLLM training."""

        def deserialize_tool_calls(messages: list[dict]) -> list[dict]:
            def deserialize_tool_call(tool_call: dict) -> dict:
                return {
                    **tool_call,
                    "function": {
                        **tool_call["function"],
                        "arguments": json.loads(tool_call["function"]["arguments"]),
                    },
                }

            return [
                {
                    **message,
                    "tool_calls": [deserialize_tool_call(tc) for tc in message.get("tool_calls") or []],
                }
                for message in messages
            ]

        prompts = [deserialize_tool_calls(prompt) for prompt in prompts]
        completions = [deserialize_tool_calls(completion) for completion in completions]

        processed_outputs = vf.Environment.process_env_results_vllm(
            self, prompts, completions, states, *args, **kwargs
        )

        for i, state in enumerate(states):
            if state.get("sandbox_error") == 1:
                processed_outputs.completion_mask[i] = [0] * len(processed_outputs.completion_ids[i])

        return processed_outputs


# ============================================================================
# Rubric (Reward Functions)
# ============================================================================


class MultiSWERubric(vf.Rubric):
    """
    Rubric for Multi-SWE environment using the same reward function
    as mini_swe_agent_plus.
    """

    def __init__(self, dataset: Dataset, **kwargs: Any):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.add_reward_func(self.has_error, 0.0)
        self.add_reward_func(self.solved, 1.0)

    def _calculate_reward_multiswe(self, state: State, info: Info) -> int:
        """
        Calculate reward for Multi-SWE dataset.
        Same implementation as mini_swe_agent_plus.
        """
        multiswe_example = restore_row(info)
        multiswe_ds: MultiSWEDataset = MultiSWEDataset.from_dict(multiswe_example)
        instance: Instance = create_instance(multiswe_ds)
        run_result: Union[str, TestResult] = multiswe_ds.run_result
        test_patch_result: Union[str, TestResult] = multiswe_ds.test_patch_result
        fix_patch_result: Union[str, TestResult] = state["test_output"]

        report = generate_report(instance, run_result, test_patch_result, fix_patch_result)
        is_valid, error_message = validate_report_against_dataset(report, multiswe_ds)
        self.logger.debug(f"Multi-SWE: validate_report_against_dataset: {is_valid=} {error_message=}")
        return int(is_valid)

    def solved(self, state: State, info: Info, **kwargs: Any) -> int:
        """Reward function for solved instances."""
        return self._calculate_reward_multiswe(state, info)

    def has_error(self, state: State) -> int:
        """
        Whether an infra failure occurred in sandboxes.
        If so, the entire group of rollouts will be masked out in training.
        """
        return int(state.get("sandbox_error", 0))


# ============================================================================
# Environment Loader
# ============================================================================


def load_environment(
    dataset_name: Literal["PrimeIntellect/Multi-SWE-RL"] = "PrimeIntellect/Multi-SWE-RL",
    max_turns: int = 200,
    total_timeout_minutes: int = 120,
    test_timeout: int = 1800,
    startup_timeout: int = 120,
    **kwargs: Any,
) -> vf.Environment:
    """
    Load the Multi-SWE environment with OpenHands agent harness.

    Uses swe-rex for container management (same as Multi-SWE benchmark).

    Args:
        dataset_name: The dataset to use. Default is PrimeIntellect/Multi-SWE-RL.
        max_turns: Maximum number of turns per episode.
        total_timeout_minutes: Total timeout for the episode in minutes.
        test_timeout: Timeout for running tests in seconds.
        startup_timeout: Timeout for container startup in seconds.
        **kwargs: Additional arguments passed to the environment.

    Returns:
        A configured Multi-SWE environment.
    """
    split = "train"

    def process_example(x):
        """Process dataset example into environment format."""
        row = restore_row(x)
        resolved_issues = row["resolved_issues"]
        assert len(resolved_issues) == 1
        issue = resolved_issues[0]

        if hints := row.get("hints"):
            problem_statement = issue["title"] + "\n\n" + issue["body"] + "\n\n" + hints
        else:
            problem_statement = issue["title"] + "\n\n" + issue["body"]

        docker_image = f"{x['org']}_m_{x['repo']}:pr-{x['number']}"
        repo = x["repo"]

        # Construct workspace_dir_name in OpenHands style: repo__version
        workspace_dir_name = repo

        # Get base_commit for the prompt
        base_commit = x.get("base_commit", "HEAD")

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        problem_statement=problem_statement,
                        workspace_dir_name=workspace_dir_name,
                        base_commit=base_commit,
                    ),
                }
            ],
            "info": {**x, "docker_image": docker_image},
            "answer": "",
        }

    # Load dataset using HuggingFace's load_dataset
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(process_example)

    rubric = MultiSWERubric(dataset=dataset)

    return MultiSWEOpenHandsEnv(
        eval_dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        test_timeout=test_timeout,
        total_timeout_minutes=total_timeout_minutes,
        startup_timeout=startup_timeout,
    )


if __name__ == "__main__":
    load_environment()
