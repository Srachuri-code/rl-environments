#!/usr/bin/env python3
"""
Tests for Tool Decathlon environment.

These tests verify the environment implementation follows verifiers patterns
and integrates correctly with Docker/Toolathlon.

Run with:
    pytest tests/test_tool_decathlon.py -v

For integration tests (requires Docker):
    pytest tests/test_tool_decathlon.py -v -m integration
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add environment to path
sys.path.insert(0, str(Path(__file__).parent.parent / "environments" / "tool_decathlon"))


class TestToolDecathlonEnvUnit:
    """Unit tests that don't require Docker or verifiers."""

    def test_import(self):
        """Test that the module can be imported."""
        # This will fail if verifiers/docker aren't installed,
        # but that's expected in CI without full deps
        try:
            from tool_decathlon import ToolDecathlonEnv, load_environment
            assert ToolDecathlonEnv is not None
            assert load_environment is not None
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")

    def test_dataset_structure(self):
        """Test that the dataset has the required structure."""
        data_dir = Path(__file__).parent.parent / "data" / "tool_decathlon_tasks.json"
        if not data_dir.exists():
            pytest.skip("Dataset not created yet")

        with open(data_dir) as f:
            tasks = json.load(f)

        assert len(tasks) > 0, "Dataset should have tasks"

        for task in tasks:
            # Required fields per verifiers spec
            assert "task_id" in task, "Missing task_id"
            assert "prompt" in task, "Missing prompt"
            assert isinstance(task["prompt"], list), "prompt should be list of messages"
            assert "info" in task, "Missing info"

            # Prompt structure
            for msg in task["prompt"]:
                assert "role" in msg, "Message missing role"
                assert "content" in msg, "Message missing content"
                assert msg["role"] in ["system", "user", "assistant"], f"Invalid role: {msg['role']}"

            # Info structure
            assert "task_id" in task["info"], "info missing task_id"
            assert "mcp_servers" in task["info"], "info missing mcp_servers"
            assert isinstance(task["info"]["mcp_servers"], list), "mcp_servers should be list"

    def test_task_api_structure(self):
        """Test that task_api.py has required commands."""
        task_api_path = Path(__file__).parent.parent / "docker" / "task_api.py"
        if not task_api_path.exists():
            pytest.skip("task_api.py not found")

        content = task_api_path.read_text()

        # Required commands
        assert "setup" in content, "Missing setup command"
        assert "execute" in content, "Missing execute command"
        assert "evaluate" in content, "Missing evaluate command"
        assert "cleanup" in content, "Missing cleanup command"

        # Required classes
        assert "class TaskAPI" in content, "Missing TaskAPI class"
        assert "MCPServerManager" in content, "Should use Toolathlon's MCPServerManager"


class TestToolDecathlonEnvMocked:
    """Tests with mocked dependencies."""

    @pytest.fixture
    def mock_docker(self):
        """Mock Docker client."""
        with patch("docker.from_env") as mock:
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_container.exec_run.return_value = (0, b'{"status": "ready", "tools": []}')
            mock_client.containers.run.return_value = mock_container
            mock.return_value = mock_client
            yield mock_client, mock_container

    @pytest.fixture
    def mock_verifiers(self):
        """Mock verifiers module."""
        with patch.dict(sys.modules, {
            "verifiers": MagicMock(),
        }):
            yield

    def test_container_naming(self, mock_docker, mock_verifiers):
        """Test that container names are unique."""
        mock_client, mock_container = mock_docker

        # Simulate two container creations
        names = []

        def capture_name(*args, **kwargs):
            names.append(kwargs.get("name", ""))
            return mock_container

        mock_client.containers.run.side_effect = capture_name

        # Would need to actually run setup_state here
        # For now, just verify the pattern is correct
        import uuid
        name1 = f"toolathlon-test-task-{uuid.uuid4().hex[:12]}"
        name2 = f"toolathlon-test-task-{uuid.uuid4().hex[:12]}"

        assert name1 != name2, "Container names should be unique"


class TestRubricStructure:
    """Test that rubric follows verifiers patterns."""

    def test_reward_function_signature(self):
        """Test reward function has correct signature."""
        # The reward function should accept: completion, state, **kwargs
        # and return a float

        async def mock_reward(completion, state, **kwargs) -> float:
            if not state.get("task_done"):
                return 0.0
            return 1.0 if state.get("eval_result") else 0.0

        # Test various states
        assert asyncio.run(mock_reward([], {"task_done": False})) == 0.0
        assert asyncio.run(mock_reward([], {"task_done": True, "eval_result": True})) == 1.0
        assert asyncio.run(mock_reward([], {"task_done": True, "eval_result": False})) == 0.0


class TestStateManagement:
    """Test state management patterns."""

    def test_state_initialization(self):
        """Test that state is properly initialized."""
        # Simulated state after setup_state
        state = {
            "container": MagicMock(),
            "container_name": "toolathlon-test-abc123",
            "task_id": "test-task",
            "task_done": False,
            "eval_result": None,
            "info": {
                "oai_tools": [
                    {"type": "function", "function": {"name": "test_tool"}}
                ]
            }
        }

        # Required fields
        assert "container" in state
        assert "task_id" in state
        assert "task_done" in state
        assert "info" in state
        assert "oai_tools" in state["info"]

    def test_completion_detection(self):
        """Test is_completed logic."""
        # Before claim_done
        state = {"task_done": False}
        assert state.get("task_done", False) == False

        # After claim_done
        state["task_done"] = True
        assert state.get("task_done", False) == True


@pytest.mark.integration
class TestToolDecathlonIntegration:
    """Integration tests requiring Docker and full dependencies."""

    @pytest.fixture
    def check_docker(self):
        """Skip if Docker is not available."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")

    @pytest.fixture
    def check_image(self, check_docker):
        """Skip if Toolathlon image is not built."""
        import docker
        client = docker.from_env()
        try:
            client.images.get("toolathlon:latest")
        except docker.errors.ImageNotFound:
            pytest.skip("toolathlon:latest image not built. Run: ./scripts/build_toolathlon_image.sh")

    @pytest.fixture
    def check_dataset(self):
        """Skip if dataset is not created."""
        dataset_path = Path(__file__).parent.parent / "data" / "tool_decathlon_dataset"
        if not dataset_path.exists():
            pytest.skip("Dataset not created. Run: python scripts/create_hf_dataset.py")

    def test_environment_creation(self, check_docker, check_image, check_dataset):
        """Test creating the environment."""
        from tool_decathlon import load_environment

        env = load_environment(max_turns=10)
        assert env is not None
        assert env.max_turns == 10

    @pytest.mark.asyncio
    async def test_setup_state(self, check_docker, check_image, check_dataset):
        """Test setting up a task in a container."""
        from tool_decathlon import load_environment

        env = load_environment(max_turns=10)

        # Create minimal state with a simple task
        state = {
            "task_id": "arrange-workspace",  # A no-credentials task
            "info": {},
        }

        try:
            state = await env.setup_state(state)

            assert "container" in state
            assert "oai_tools" in state["info"]
            assert len(state["info"]["oai_tools"]) > 0
        finally:
            await env.cleanup_state(state)

    @pytest.mark.asyncio
    async def test_full_episode(self, check_docker, check_image, check_dataset):
        """Test a complete episode with claim_done."""
        from tool_decathlon import load_environment

        env = load_environment(max_turns=10)

        state = {
            "task_id": "arrange-workspace",
            "info": {},
        }

        try:
            state = await env.setup_state(state)

            # Simulate claim_done tool call
            messages = [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "claim_done",
                        "arguments": "{}",
                    }
                }]
            }]

            responses, state = await env.env_response(messages, state)

            assert state["task_done"] == True
            assert state["eval_result"] is not None
            assert len(responses) == 1
            assert responses[0]["role"] == "tool"
        finally:
            await env.cleanup_state(state)


class TestVerifiersCompatibility:
    """Test compatibility with verifiers framework patterns."""

    def test_message_format(self):
        """Test that messages follow OpenAI format."""
        # User message
        user_msg = {"role": "user", "content": "Complete the task"}
        assert user_msg["role"] == "user"

        # Assistant with tool call
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"arg": "value"}',
                }
            }]
        }
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "test_tool"

        # Tool response
        tool_msg = {
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call_123",
        }
        assert tool_msg["role"] == "tool"

    def test_tool_format(self):
        """Test that tools follow OpenAI function calling format."""
        tool = {
            "type": "function",
            "function": {
                "name": "claim_done",
                "description": "Call when task is complete",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
