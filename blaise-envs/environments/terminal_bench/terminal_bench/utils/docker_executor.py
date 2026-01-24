"""
Docker-based command executor for Terminal-Bench environment.

Provides container lifecycle management and command execution using
local Docker containers for sandboxed execution.
"""

import asyncio
import io
import logging
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import docker
    from docker.errors import DockerException, ImageNotFound, NotFound, APIError
    from docker.models.containers import Container
except ImportError as e:
    raise RuntimeError(
        "Missing docker dependency. Install with: pip install docker"
    ) from e


logger = logging.getLogger(__name__)


class DockerExecutor:
    """
    Docker-based executor for running commands in isolated containers.

    Each task gets its own container for sandboxed execution of bash commands.
    """

    def __init__(
        self,
        default_image: str = "python:3.11-slim",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        storage_gb: int = 10,
        command_timeout: int = 60,
        container_timeout_minutes: int = 30,
        network_mode: str = "bridge",
        auto_remove: bool = False,
        runtime: str = "crun",
    ):
        """
        Initialize the Docker executor.

        Args:
            default_image: Default Docker image to use
            cpu_cores: Number of CPU cores to allocate
            memory_gb: Memory limit in GB
            storage_gb: Storage limit in GB (for disk quota if supported)
            command_timeout: Default timeout for commands in seconds
            container_timeout_minutes: Container lifetime timeout in minutes
            network_mode: Docker network mode (bridge, none, host)
            auto_remove: Whether to auto-remove containers on stop
            runtime: OCI runtime to use (crun for speed, runc as fallback)
        """
        self.default_image = default_image
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.storage_gb = storage_gb
        self.command_timeout = command_timeout
        self.container_timeout_minutes = container_timeout_minutes
        self.network_mode = network_mode
        self.auto_remove = auto_remove
        self.runtime = runtime

        self._client: Optional[docker.DockerClient] = None
        self._active_containers: Dict[str, Container] = {}

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            try:
                self._client = docker.from_env()
                # Test connection
                self._client.ping()
            except DockerException as e:
                raise RuntimeError(
                    "Failed to connect to Docker. Ensure Docker is running."
                ) from e
        return self._client

    async def create_container(
        self,
        image: str,
        container_id: Optional[str] = None,
        working_dir: str = "/workspace",
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        """
        Create and start a new container.

        Args:
            image: Docker image to use
            container_id: Optional custom container ID/name
            working_dir: Working directory inside container
            environment: Environment variables
            volumes: Volume mounts {host_path: {"bind": container_path, "mode": "rw"}}

        Returns:
            Container ID string
        """
        container_id = container_id or f"terminal-bench-{uuid.uuid4().hex[:12]}"

        # Pull image if not available
        try:
            await asyncio.to_thread(self.client.images.get, image)
        except ImageNotFound:
            logger.info(f"Pulling image {image}...")
            await asyncio.to_thread(self.client.images.pull, image)

        # Remove existing container with same name if present
        try:
            existing = await asyncio.to_thread(self.client.containers.get, container_id)
            await asyncio.to_thread(existing.remove, force=True)
        except NotFound:
            pass

        # Create container
        container = await asyncio.to_thread(
            self.client.containers.create,
            image,
            name=container_id,
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            tty=True,
            working_dir=working_dir,
            environment=environment or {},
            volumes=volumes or {},
            mem_limit=f"{self.memory_gb}g",
            nano_cpus=int(self.cpu_cores * 1e9),
            network_mode=self.network_mode,
            auto_remove=self.auto_remove,
            runtime=self.runtime,
        )

        # Start container
        await asyncio.to_thread(container.start)

        # Wait for container to be running
        await self._wait_for_ready(container)

        self._active_containers[container_id] = container
        logger.debug(f"Created container {container_id} with image {image}")

        return container_id

    async def _wait_for_ready(
        self,
        container: Container,
        timeout: int = 60,
        poll_interval: float = 0.5,
    ) -> None:
        """Wait for container to reach running state."""
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.to_thread(container.reload)
            if container.status == "running":
                return
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(f"Container failed to start within {timeout}s")

    async def execute_command(
        self,
        container_id: str,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str]:
        """
        Execute a command in a container.

        Args:
            container_id: Container ID
            command: Command to execute
            working_dir: Working directory for command
            timeout: Command timeout in seconds
            environment: Additional environment variables

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        container = self._active_containers.get(container_id)
        if not container:
            try:
                container = await asyncio.to_thread(
                    self.client.containers.get, container_id
                )
            except NotFound:
                raise RuntimeError(f"Container {container_id} not found")

        timeout = timeout or self.command_timeout

        # Wrap command with timeout
        timed_command = f"timeout {timeout} bash -c {repr(command)}"

        try:
            exec_result = await asyncio.wait_for(
                asyncio.to_thread(
                    container.exec_run,
                    timed_command,
                    workdir=working_dir,
                    environment=environment,
                    demux=True,
                ),
                timeout=timeout + 10,  # Extra buffer for exec overhead
            )
        except asyncio.TimeoutError:
            return -1, "", f"Command timed out after {timeout} seconds"

        exit_code = exec_result.exit_code
        stdout_bytes, stderr_bytes = exec_result.output

        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")

        # Exit code 124 means timeout from the timeout command
        if exit_code == 124:
            stderr += f"\nCommand timed out after {timeout} seconds"

        return exit_code, stdout, stderr

    async def upload_bytes(
        self,
        container_id: str,
        container_path: str,
        data: bytes,
        mode: int = 0o644,
    ) -> None:
        """
        Upload bytes to a file in a container.

        Args:
            container_id: Container ID
            container_path: Destination path in container
            data: File content as bytes
            mode: File permissions mode
        """
        container = self._active_containers.get(container_id)
        if not container:
            container = await asyncio.to_thread(
                self.client.containers.get, container_id
            )

        # Create tar archive with the file
        path = Path(container_path)
        tar_stream = io.BytesIO()

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=path.name)
            tarinfo.size = len(data)
            tarinfo.mode = mode
            tar.addfile(tarinfo, io.BytesIO(data))

        tar_stream.seek(0)

        # Ensure parent directory exists
        parent_dir = str(path.parent)
        if parent_dir != "/":
            await self.execute_command(
                container_id,
                f"mkdir -p {parent_dir}",
                timeout=10,
            )

        # Upload file
        await asyncio.to_thread(
            container.put_archive, parent_dir, tar_stream
        )

    async def upload_file(
        self,
        container_id: str,
        container_path: str,
        content: str,
    ) -> None:
        """
        Upload a text file to a container.

        Args:
            container_id: Container ID
            container_path: Destination path in container
            content: File content as string
        """
        await self.upload_bytes(
            container_id,
            container_path,
            content.encode("utf-8"),
            mode=0o755,
        )

    async def upload_archive(
        self,
        container_id: str,
        container_path: str,
        archive_bytes: bytes,
    ) -> bool:
        """
        Upload and extract an archive to a container.

        Args:
            container_id: Container ID
            container_path: Destination directory in container
            archive_bytes: Archive content as bytes (tar.gz, tar, or zip)

        Returns:
            True if extraction succeeded, False otherwise
        """
        container = self._active_containers.get(container_id)
        if not container:
            container = await asyncio.to_thread(
                self.client.containers.get, container_id
            )

        # Ensure destination directory exists
        await self.execute_command(
            container_id,
            f"mkdir -p {container_path}",
            timeout=10,
        )

        # Try to detect and extract archive format
        # First, upload the archive to a temp location
        temp_archive = "/tmp/task_archive"

        # Try tar.gz first
        try:
            # Check if it's a valid tar.gz
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                pass  # Just verify it opens

            await self.upload_bytes(container_id, f"{temp_archive}.tar.gz", archive_bytes)
            exit_code, _, _ = await self.execute_command(
                container_id,
                f"tar -xzf {temp_archive}.tar.gz -C {container_path}",
                timeout=60,
            )
            if exit_code == 0:
                return True
        except:
            pass

        # Try plain tar
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as tar:
                pass

            await self.upload_bytes(container_id, f"{temp_archive}.tar", archive_bytes)
            exit_code, _, _ = await self.execute_command(
                container_id,
                f"tar -xf {temp_archive}.tar -C {container_path}",
                timeout=60,
            )
            if exit_code == 0:
                return True
        except:
            pass

        # Try zip
        try:
            import zipfile
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                pass

            await self.upload_bytes(container_id, f"{temp_archive}.zip", archive_bytes)
            exit_code, _, _ = await self.execute_command(
                container_id,
                f"unzip -o {temp_archive}.zip -d {container_path}",
                timeout=60,
            )
            if exit_code == 0:
                return True
        except:
            pass

        return False

    async def download_file(
        self,
        container_id: str,
        container_path: str,
    ) -> str:
        """
        Download a file from a container.

        Args:
            container_id: Container ID
            container_path: Path in container

        Returns:
            File content as string
        """
        container = self._active_containers.get(container_id)
        if not container:
            container = await asyncio.to_thread(
                self.client.containers.get, container_id
            )

        bits, stat = await asyncio.to_thread(
            container.get_archive, container_path
        )

        # Extract from tar
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)

        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            member = tar.getmembers()[0]
            f = tar.extractfile(member)
            if f:
                return f.read().decode("utf-8", errors="replace")

        raise FileNotFoundError(f"File not found: {container_path}")

    async def stop_container(self, container_id: str, timeout: int = 10) -> None:
        """Stop a container."""
        container = self._active_containers.pop(container_id, None)
        if not container:
            try:
                container = await asyncio.to_thread(
                    self.client.containers.get, container_id
                )
            except NotFound:
                return

        try:
            await asyncio.to_thread(container.stop, timeout=timeout)
        except Exception as e:
            logger.warning(f"Error stopping container {container_id}: {e}")

    async def remove_container(self, container_id: str, force: bool = True) -> None:
        """Remove a container."""
        container = self._active_containers.pop(container_id, None)
        if not container:
            try:
                container = await asyncio.to_thread(
                    self.client.containers.get, container_id
                )
            except NotFound:
                return

        try:
            await asyncio.to_thread(container.remove, force=force)
        except Exception as e:
            logger.warning(f"Error removing container {container_id}: {e}")

    async def cleanup_all(self) -> None:
        """Stop and remove all active containers."""
        container_ids = list(self._active_containers.keys())
        for container_id in container_ids:
            try:
                await self.stop_container(container_id)
                await self.remove_container(container_id)
            except Exception as e:
                logger.warning(f"Error cleaning up container {container_id}: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        for container_id, container in list(self._active_containers.items()):
            try:
                container.stop(timeout=5)
                container.remove(force=True)
            except Exception:
                pass


class CommandResult:
    """Result of a command execution."""

    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        output = self.stdout
        if self.stderr:
            if output:
                output += f"\nstderr:\n{self.stderr}"
            else:
                output = f"stderr:\n{self.stderr}"
        return output or "(no output)"

    @property
    def success(self) -> bool:
        """Whether command succeeded."""
        return self.exit_code == 0
