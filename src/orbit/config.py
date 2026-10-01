"""Runtime settings, resolved from environment variables (and an optional .env file)."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    return default


@dataclass
class LLMSettings:
    # Defaults target OpenAI. Point ORBIT_LLM_BASE_URL at any OpenAI-compatible
    # server (vLLM, Ollama, LM Studio, ...) to swap the model.
    base_url: str | None = field(default_factory=lambda: _env("ORBIT_LLM_BASE_URL", "OPENAI_BASE_URL"))
    api_key: str | None = field(default_factory=lambda: _env("ORBIT_LLM_API_KEY", "OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: _env("ORBIT_LLM_MODEL", default="gpt-5-mini"))
    temperature: float | None = field(
        default_factory=lambda: float(t) if (t := _env("ORBIT_LLM_TEMPERATURE")) else None)
    max_tokens: int | None = field(
        default_factory=lambda: int(t) if (t := _env("ORBIT_LLM_MAX_TOKENS")) else None)
    timeout: float = field(default_factory=lambda: float(_env("ORBIT_LLM_TIMEOUT", default="300")))


@dataclass
class SandboxSettings:
    image: str = field(default_factory=lambda: _env("ORBIT_SANDBOX_IMAGE", default="orbit-sandbox:latest"))
    cpus: float = field(default_factory=lambda: float(_env("ORBIT_SANDBOX_CPUS", default="4")))
    memory: str = field(default_factory=lambda: _env("ORBIT_SANDBOX_MEMORY", default="8g"))
    timeout: int = field(default_factory=lambda: int(_env("ORBIT_SANDBOX_TIMEOUT", default="600")))
    runtime: str | None = field(default_factory=lambda: _env("ORBIT_SANDBOX_RUNTIME"))  # e.g. runsc


def tracker_choice(requested: str = "auto") -> str:
    """auto -> wandb when a W&B key is configured, otherwise MLflow."""
    if requested != "auto":
        return requested
    return "wandb" if _env("WANDB_API_KEY") else "mlflow"
