# crun Integration Guide

## What is crun?

[crun](https://github.com/containers/crun) is a fast, lightweight OCI container runtime written in C. It offers faster container startup times compared to the default `runc` runtime, which is beneficial for RL environments that create many short-lived containers.

## When to Use crun

crun integration is only possible when you have control over Docker container creation. This applies to environments that:

- Use custom `DockerExecutor` classes with `docker-py`
- Call `docker.containers.create()` or `docker run` directly

crun **cannot** be integrated when:

- Using pre-built Docker images with their own runtime management (e.g., swebench images)
- Using third-party libraries that abstract Docker (e.g., `minisweagent`)

## Environments with crun Support

| Environment | crun Support | Reason |
|-------------|--------------|--------|
| terminal_bench | Yes | Uses custom DockerExecutor |
| multi_swe | Yes | Uses custom DockerSandbox |
| swe_bench | No | Uses minisweagent library |

## Integration Steps

### 1. Add runtime parameter to DockerExecutor

```python
class DockerExecutor:
    def __init__(
        self,
        default_image: str = "python:3.11-slim",
        cpu_cores: int = 4,
        memory_gb: int = 4,
        command_timeout: int = 90,
        runtime: str = "crun",  # Add this parameter
    ):
        self.runtime = runtime
        # ... rest of init
```

### 2. Pass runtime to container creation

For `docker-py`:

```python
container = await asyncio.to_thread(
    self.client.containers.create,
    image,
    name=container_id,
    command="tail -f /dev/null",
    detach=True,
    runtime=self.runtime,  # Add this line
    # ... other options
)
```

For subprocess `docker run`:

```python
cmd = [
    "docker", "run",
    "-d",
    "--runtime", self.runtime,  # Add this flag
    "--name", container_name,
    image,
    "sleep", "infinity",
]
```

### 3. Expose runtime in load_environment

```python
def load_environment(
    # ... other params
    runtime: str = "crun",
    **kwargs,
) -> vf.Environment:
    return MyEnvironment(
        runtime=runtime,
        # ... other params
    )
```

### 4. Update README

Add prerequisites section:

```markdown
### Prerequisites

**Docker must be installed and running.** Uses crun runtime by default for faster container startup.

```bash
# Verify Docker is running
docker info

# Install crun (optional, falls back to runc)
# Ubuntu/Debian: sudo apt-get install crun
# Or build from source: https://github.com/containers/crun
```
```

Add runtime to environment arguments table:

```markdown
| `runtime` | str | `"crun"` | OCI runtime (crun for speed, runc as fallback) |
```

## Installing crun

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install crun
```

### Fedora/RHEL

```bash
sudo dnf install crun
```

### From Source

```bash
git clone https://github.com/containers/crun.git
cd crun
./autogen.sh
./configure
make
sudo make install
```

### Verify Installation

```bash
crun --version
docker info | grep -i runtime
```

## Fallback Behavior

If crun is not installed, Docker will fail to create containers with the `crun` runtime. Users should either:

1. Install crun
2. Pass `runtime="runc"` to use the default runtime

Example:

```bash
# Using runc fallback
uv run vf-eval terminal-bench -s -a '{"runtime": "runc"}'
```

## Performance Comparison

| Metric | runc | crun |
|--------|------|------|
| Container startup | ~300ms | ~50ms |
| Memory overhead | Higher | Lower |
| Written in | Go | C |

crun is particularly beneficial for:
- Environments with many short-lived containers
- High-throughput evaluation runs
- Resource-constrained systems
