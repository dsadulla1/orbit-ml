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
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import SandboxSettings

SCRIPT = "solution.py"
MAX_LOG_CHARS = 20_000
SANDBOX_UID = 1000  # the unprivileged `orbit` user in docker/sandbox.Dockerfile


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


def _read_tail(f, limit: int = MAX_LOG_CHARS) -> str:
    """Last `limit` bytes of a log file, decoded leniently."""
    size = f.seek(0, os.SEEK_END)
    f.seek(max(0, size - limit))
    text = f.read().decode("utf-8", "replace")
    return text if size <= limit else "...[truncated]...\n" + text


def _host_is_root() -> bool:
    return hasattr(os, "getuid") and os.getuid() == 0


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
        if _host_is_root():  # never run generated code as root; use the image's user
            cmd += ["--user", f"{SANDBOX_UID}:{SANDBOX_UID}"]
        elif hasattr(os, "getuid"):  # run as the host user so outputs stay owned by them
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if s.runtime:
            cmd += ["--runtime", s.runtime]
        return cmd + [s.image, "python", SCRIPT]

    def _kill(self, name: str) -> None:
        try:
            subprocess.run([self.docker, "kill", name], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass  # best effort: --rm removes the container once it stops

    def run(self, code: str, workdir: Path, data_dir: Path, timeout: int) -> ExecResult:
        workdir.mkdir(parents=True, exist_ok=True)
        # Pre-create the mount point so Docker doesn't create it root-owned on the host.
        (workdir / "data").mkdir(exist_ok=True)
        (workdir / SCRIPT).write_text(code, encoding="utf-8")
        if _host_is_root():
            for p in (workdir, workdir / "data", workdir / SCRIPT):
                os.chown(p, SANDBOX_UID, SANDBOX_UID)
        name = f"orbit-{uuid.uuid4().hex[:12]}"
        start = time.monotonic()
        # Logs go to temp files outside the writable mount, so a chatty script can't
        # exhaust orchestrator memory; only the tail is kept.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.run(self.command(workdir, data_dir, name),
                                      stdout=out, stderr=err, timeout=timeout)
                returncode, timed_out, note = proc.returncode, False, ""
            except KeyboardInterrupt:
                self._kill(name)
                raise
            except subprocess.TimeoutExpired:
                self._kill(name)
                returncode, timed_out = -1, True
                note = f"\nTimeoutError: experiment exceeded {timeout}s and was killed."
            stdout, stderr = _read_tail(out), _read_tail(err) + note
        _remove_symlinks(workdir)
        return ExecResult(returncode, stdout, stderr, time.monotonic() - start, timed_out)


def _remove_symlinks(workdir: Path) -> None:
    """The orchestrator reads, writes and uploads files in `workdir` after the run. A link
    planted by the script (e.g. stdout.txt -> a host file) would make it touch host files."""
    for p in workdir.iterdir():
        if p.is_symlink():
            p.unlink()


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
