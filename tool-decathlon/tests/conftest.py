"""
Pytest configuration and fixtures for Tool Decathlon tests.
"""

import sys
from pathlib import Path

import pytest

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "environments" / "tool_decathlon"))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests requiring Docker and full setup"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks async tests"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def docker_available():
    """Check if Docker is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def toolathlon_image_available(docker_available):
    """Check if Toolathlon image is available."""
    if not docker_available:
        return False
    try:
        import docker
        client = docker.from_env()
        client.images.get("toolathlon:latest")
        return True
    except Exception:
        return False
