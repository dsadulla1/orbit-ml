"""Execute generated code in an isolated Docker container.

Each experiment runs in a fresh container:
  * no network, read-only root filesystem, all capabilities dropped, no-new-privileges
  * dataset mounted read-only at /work/data; only the experiment's own folder is writable
  * CPU / memory / process limits and a wall-clock timeout
Hidden labels live outside both mounts, so generated code can never read them.
"""
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import SandboxSettings

SCRIPT = "solution.py"
MAX_LOG_CHARS = 20_000


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class Sandbox(Protocol):
    def run(self, code: str, workdir: Path, data_dir: Path, timeout: int) -> ExecResult: ...


def _tail(text: str, limit: int = MAX_LOG_CHARS) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


class DockerSandbox:
    def __init__(self, settings: SandboxSettings | None = None, docker: str = "docker"):
        self.settings = settings or SandboxSettings()
        self.docker = docker

    def command(self, workdir: Path, data_dir: Path, name: str) -> list[str]:
        s = self.settings
        threads = str(max(1, int(s.cpus)))
        cmd = [
            self.docker, "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=1g",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", str(s.cpus), "--memory", s.memory, "--memory-swap", s.memory,
            "--pids-limit", "512",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", f"OMP_NUM_THREADS={threads}", "-e", f"OPENBLAS_NUM_THREADS={threads}",
            "-e", f"MKL_NUM_THREADS={threads}", "-e", f"ORBIT_N_JOBS={threads}",
            "-v", f"{workdir.resolve()}:/work",
            "-v", f"{data_dir.resolve()}:/work/data:ro",
            "-w", "/work",
        ]
        if hasattr(os, "getuid"):  # run as the host user so outputs stay owned by them
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if s.runtime:
            cmd += ["--runtime", s.runtime]
        return cmd + [s.image, "python", SCRIPT]

    def run(self, code: str, workdir: Path, data_dir: Path, timeout: int) -> ExecResult:
        workdir.mkdir(parents=True, exist_ok=True)
        # Pre-create the mount point so Docker doesn't create it root-owned on the host.
        (workdir / "data").mkdir(exist_ok=True)
        (workdir / SCRIPT).write_text(code, encoding="utf-8")
        name =f"orbit-{uuid.uuid4().hex[:12]}"
        start = time.monotonic()
        try:
            proc = subprocess.run(self.command(workdir, data_dir, name), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except KeyboardInterrupt:
            subprocess.run([self.docker, "kill", name], capture_output=True)
            raise
        except subprocess.TimeoutExpired as e:
            subprocess.run([self.docker, "kill", name], capture_output=True)
            return ExecResult(-1, _tail(_text(e.stdout)), _tail(_text(e.stderr)) +
                              f"\nTimeoutError: experiment exceeded {timeout}s and was killed.",
                              time.monotonic() - start, timed_out=True)
        return ExecResult(proc.returncode, _tail(proc.stdout), _tail(proc.stderr),
                          time.monotonic() - start)


def _text(b) -> str:
    if b is None:
        return ""
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def check_docker(settings: SandboxSettings | None = None) -> tuple[bool, str]:
    """Return (ok, message) describing whether the sandbox can run."""
    settings = settings or SandboxSettings()
    if not shutil.which("docker"):
        return False, "docker CLI not found on PATH."
    info = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                          capture_output=True, text=True)
    if info.returncode != 0:
        return False, f"Docker daemon not reachable: {info.stderr.strip()[:200]}"
    img = subprocess.run(["docker", "image", "inspect", settings.image], capture_output=True)
    if img.returncode != 0:
        return False, (f"Sandbox image {settings.image!r} not found. Build it with: "
                       "docker compose -f docker/compose.yaml build sandbox")
    return True, f"Docker {info.stdout.strip()}, image {settings.image} present."
