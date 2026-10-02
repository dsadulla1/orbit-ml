"""Serve registered models in a dedicated Docker container and score CSV files against them.

`orbit serve` starts `mlflow models serve` in the `orbit-serve` image (the sandbox's pinned
libraries plus MLflow). The container fetches the model over HTTP from the tracking server on the
Compose network, mounts none of MLflow's data, and publishes its port on 127.0.0.1 only: MLflow's
scoring server has no authentication. The functions here build commands and parse Docker's output
so they can be tested without Docker; the commands in `cli.py` are thin wrappers.
"""
import json
import os
import re
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .problem import sanitize_name

IMAGE = "orbit-serve:latest"
BUILD_COMMAND = "docker compose --env-file .env -f docker/compose.yaml build serve"
NAME_PREFIX = "orbit-serve-"
LABEL_MARK = "orbit.serve"
LABEL_MODEL = "orbit.model"
LABEL_VERSION = "orbit.version"
CONTAINER_PORT = 8080
DEFAULT_PORT = 8090
DEFAULT_ALIAS = "champion"
MLFLOW_CONTAINER = "orbit-mlflow-1"
CHUNK_ROWS = 500
REQUEST_TIMEOUT = 60


class ServeError(Exception):
    """A problem with a message meant for the user."""


@dataclass
class Container:
    name: str
    model: str
    version: str
    port: int | None
    uptime: str


def image_name() -> str:
    return os.environ.get("ORBIT_SERVE_IMAGE") or IMAGE


def memory_limit() -> str:
    return os.environ.get("ORBIT_SERVE_MEMORY") or "4g"


def docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise ServeError("docker CLI not found on PATH.") from None
    except subprocess.TimeoutExpired:
        raise ServeError(f"`docker {args[0]}` timed out.") from None


def check_docker(image: str) -> None:
    """Raises ServeError unless the daemon answers and the serving image exists."""
    info = docker("info", "--format", "{{.ServerVersion}}")
    if info.returncode != 0:
        raise ServeError(f"Docker daemon not reachable: {info.stderr.strip()[:200]}")
    if docker("image", "inspect", image).returncode != 0:
        raise ServeError(f"Serving image {image!r} not found. Build it with: {BUILD_COMMAND}")


# --- Model URIs -------------------------------------------------------------------------------

def model_uri(model: str, version: int | None = None, alias: str | None = None) -> str:
    """`MODEL` is a registered model name or a full `models:/...` URI."""
    if version is not None and alias:
        raise ServeError("Use either --version or --alias, not both.")
    if model.startswith("models:/"):
        if version is not None or alias:
            raise ServeError("--version and --alias only apply to a model name, not a full URI.")
        return model
    if not model.strip() or "/" in model or "@" in model:
        raise ServeError(f"{model!r} is neither a registered model name nor a models:/ URI.")
    if version is not None:
        return f"models:/{model}/{version}"
    return f"models:/{model}@{alias or DEFAULT_ALIAS}"


def parse_uri(uri: str) -> tuple[str, str | None, str | None]:
    """(name, version or stage, alias) of a `models:/` URI; the unused parts are None."""
    rest = uri.removeprefix("models:/")
    if "@" in rest:
        name, _, alias = rest.partition("@")
        return name, None, alias
    name, _, version = rest.partition("/")
    return name, version or None, None


def model_name_of(uri: str) -> str:
    return parse_uri(uri)[0] if uri.startswith("models:/") else uri


def resolve_in_registry(client, uri: str) -> tuple[str, str]:
    """(name, version) the URI points at, checked against the registry. A URI that is not a
    plain version or alias (a stage, `runs:/...`) is passed through unchecked."""
    if not uri.startswith("models:/"):
        return uri, ""
    name, version, alias = parse_uri(uri)
    try:
        if alias:
            found = client.get_model_version_by_alias(name, alias)
        elif version and version.isdigit():
            found = client.get_model_version(name, version)
        else:
            return name, version or ""
        return name, str(found.version)
    except Exception as e:  # noqa: BLE001 - say what exists instead
        raise ServeError(_not_found(client, uri, e)) from None


def _not_found(client, uri: str, error: Exception) -> str:
    try:
        names = sorted(m.name for m in client.search_registered_models(max_results=100))
    except Exception:  # noqa: BLE001
        names = []
    known = ("Registered models: " + ", ".join(names)) if names else "No models are registered yet"
    return f"{uri} not found in the model registry ({str(error)[:150]}). {known}. `orbit models` lists them."


def example_row(payload) -> dict | None:
    """The first row of an MLflow input example, as a record; None when the format is unknown."""
    if isinstance(payload, dict):
        for key in ("dataframe_split", "inputs"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
        if isinstance(payload.get("dataframe_records"), list):
            payload = payload["dataframe_records"]
    if isinstance(payload, dict) and "columns" in payload and payload.get("data"):
        return dict(zip(payload["columns"], payload["data"][0]))
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return None


def fetch_input_example(uri: str) -> dict | None:
    """The model's saved input example (one row), or None when it cannot be fetched."""
    import tempfile

    try:
        import mlflow.artifacts

        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(
                artifact_uri=uri.rstrip("/") + "/input_example.json", dst_path=tmp)
            return example_row(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - only used for the curl hint
        return None


def curl_command(url: str, row: dict | None) -> str:
    if row is None:
        row = {"<feature>": "<value>"}
    body = json.dumps({"dataframe_records": [row]}, allow_nan=False, default=str)
    return (f"curl -s {url}/invocations -H 'Content-Type: application/json' "
            f"-d {shlex.quote(body)}")


# --- Docker discovery -------------------------------------------------------------------------

def discover_mlflow(inspect_json: str | None, fallback_network: str = "orbit_default") -> dict:
    """Where the MLflow server container is, from its `docker inspect` output: its Compose
    `network` (the project's `<project>_default` when it has several), `service` name, `port` and
    `ip` on that network ("" when unknown). Each falls back to a default when the output is
    missing or odd."""
    found = {"network": fallback_network, "service": "mlflow", "port": 5000, "ip": ""}
    try:
        info = json.loads(inspect_json or "")
        info = info[0] if isinstance(info, list) else info
        config = info.get("Config") or {}
        labels = config.get("Labels") or {}
        attached = (info.get("NetworkSettings") or {}).get("Networks") or {}
    except (TypeError, ValueError, IndexError, AttributeError):
        return found
    found["service"] = labels.get("com.docker.compose.service") or found["service"]
    project = labels.get("com.docker.compose.project")
    if project and f"{project}_default" in attached:
        found["network"] = f"{project}_default"
    elif attached:
        found["network"] = next(iter(attached))
    found["ip"] = (attached.get(found["network"]) or {}).get("IPAddress") or ""
    if m := re.search(r"--port[= ](\d+)", " ".join(config.get("Cmd") or [])):
        found["port"] = int(m.group(1))
    return found


def mlflow_endpoint(container: str = MLFLOW_CONTAINER) -> tuple[str, str]:
    """(network, in-network tracking URL) for the MLflow server container. The URL uses the
    container's IP address when known: MLflow's Host-header check accepts private addresses by
    default but not the service name."""
    out = docker("inspect", container)
    found = discover_mlflow(out.stdout if out.returncode == 0 else None)
    return found["network"], f"http://{found['ip'] or found['service']}:{found['port']}"


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(requested: int | None, is_free: Callable[[int], bool] = port_is_free) -> int:
    """`requested` when free (an error when not); else the first free port from the default."""
    if requested is not None:
        if not is_free(requested):
            raise ServeError(f"Port {requested} is already in use on 127.0.0.1.")
        return requested
    for port in range(DEFAULT_PORT, DEFAULT_PORT + 100):
        if is_free(port):
            return port
    raise ServeError(f"No free port between {DEFAULT_PORT} and {DEFAULT_PORT + 99}; pass --port.")


def container_name(model_name: str, port: int, name: str | None = None) -> str:
    if name:
        full = name if name.startswith(NAME_PREFIX) else NAME_PREFIX + name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", full):
            raise ServeError(f"{name!r} is not a valid container name (letters, digits, _ . -).")
        return full
    return f"{NAME_PREFIX}{sanitize_name(model_name).lower() or 'model'}-{port}"


def run_command(name: str, uri: str, version: str, network: str, tracking_uri: str, port: int,
                image: str, memory: str) -> list[str]:
    """The `docker run` argument list. Hardened where it costs nothing (no capabilities,
    no-new-privileges, memory and PID limits, a tmpfs /tmp), but the root filesystem stays
    writable and the network stays on: the server must reach the tracking server."""
    return ["docker", "run", "-d", "--name", name,
            "--label", f"{LABEL_MARK}=1", "--label", f"{LABEL_MODEL}={uri}",
            "--label", f"{LABEL_VERSION}={version}",
            "--network", network,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", memory, "--pids-limit", "512",
            "--tmpfs", "/tmp:rw,mode=1777,size=1g",
            "-p", f"127.0.0.1:{port}:{CONTAINER_PORT}",
            "-e", f"MLFLOW_TRACKING_URI={tracking_uri}",
            image,
            "mlflow", "models", "serve", "-m", uri, "--env-manager", "local",
            "--host", "0.0.0.0", "--port", str(CONTAINER_PORT)]


def start(command: list[str]) -> str:
    """Run the prepared `docker run`; returns the container id."""
    out = docker(*command[1:], timeout=120)
    if out.returncode != 0:
        raise ServeError(f"docker run failed: {out.stderr.strip()[-400:]}")
    return out.stdout.strip()[:12]


def is_running(name: str) -> bool:
    out = docker("inspect", "-f", "{{.State.Running}}", name)
    return out.returncode == 0 and out.stdout.strip() == "true"


def logs_tail(name: str, lines: int = 20) -> str:
    out = docker("logs", "--tail", str(lines), name)
    return (out.stdout + out.stderr).strip()


def wait_ready(port: int, alive: Callable[[], bool], timeout: float = 180, interval: float = 1.0) -> str:
    """Poll the scoring server's /ping until it answers. Returns "ready", "exited" (the
    container stopped first) or "timeout"."""
    url = f"http://127.0.0.1:{port}/ping"
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return "ready"
        except Exception:  # noqa: BLE001 - not up yet
            pass
        if not alive():
            return "exited"
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(interval)


# --- Listing and stopping ---------------------------------------------------------------------

PS_FORMAT = ("{{.Names}}\t{{.Label \"" + LABEL_MARK + "\"}}\t{{.Label \"" + LABEL_MODEL + "\"}}\t"
             "{{.Label \"" + LABEL_VERSION + "\"}}\t{{.Ports}}\t{{.RunningFor}}")


def parse_ps(output: str) -> list[Container]:
    """Orbit serving containers in `docker ps --format PS_FORMAT` output: only names with the
    orbit prefix that also carry the label `orbit.serve=1`."""
    found = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        name, mark, model, version, ports, uptime = (p.strip() for p in parts[:6])
        if not name.startswith(NAME_PREFIX) or mark != "1":
            continue
        m = re.search(r"127\.0\.0\.1:(\d+)->", ports)
        found.append(Container(name, model, version, int(m.group(1)) if m else None, uptime))
    return found


def list_containers() -> list[Container]:
    out = docker("ps", "--filter", f"name={NAME_PREFIX}", "--filter", f"label={LABEL_MARK}=1",
                 "--format", PS_FORMAT)
    if out.returncode != 0:
        raise ServeError(f"docker ps failed: {out.stderr.strip()[:200]}")
    return parse_ps(out.stdout)


def stop_targets(containers: list[Container], target: str) -> list[str]:
    """Names to remove for `--stop TARGET` ("all" or one container name); anything that is not
    a running orbit serving container is refused."""
    if target == "all":
        return [c.name for c in containers]
    names = [c.name for c in containers]
    if target not in names:
        listed = ", ".join(names) or "none running"
        raise ServeError(f"{target!r} is not an orbit serving container ({listed}).")
    return [target]


def remove(names: list[str]) -> None:
    for name in names:
        if not name.startswith(NAME_PREFIX):  # never anything that is not ours
            raise ServeError(f"Refusing to remove {name!r}.")
        out = docker("rm", "-f", name)
        if out.returncode != 0:
            raise ServeError(f"Could not remove {name}: {out.stderr.strip()[:200]}")


# --- Scoring rows -----------------------------------------------------------------------------

def invocations_url(url: str) -> str:
    base = url.strip().rstrip("/")
    return base if base.endswith("/invocations") else base + "/invocations"


def endpoint_for(model: str, containers: list[Container]) -> str:
    """The base URL of the one running container that serves registered model `model`."""
    matches = [c for c in containers if c.port and model in (model_name_of(c.model), c.model)]
    if not matches:
        running = ", ".join(f"{model_name_of(c.model)} (port {c.port})" for c in containers)
        raise ServeError(f"No running endpoint serves {model!r} ({running or 'nothing is served'}). "
                         "Start one with `orbit serve`, or pass the endpoint URL.")
    if len(matches) > 1:
        ports = ", ".join(f"{c.name} (port {c.port})" for c in matches)
        raise ServeError(f"{model!r} is served more than once: {ports}. Pass the endpoint URL.")
    return f"http://127.0.0.1:{matches[0].port}"


def _error_text(e: urllib.error.HTTPError) -> str:
    body = e.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
        message = data.get("message") or data.get("error") or body
    except (ValueError, AttributeError):
        message = body
    message = " ".join(str(message).split())
    message = message.rsplit("Error: ", 1)[-1]  # MLflow nests errors; the last one is the cause
    return f"HTTP {e.code}: {message[:400]}"


def _post(url: str, body: str, timeout: float) -> list:
    request = urllib.request.Request(url, data=body.encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        raise ServeError(f"The endpoint rejected the request: {_error_text(e)}") from None
    except (urllib.error.URLError, OSError) as e:
        raise ServeError(f"Could not reach {url}: {getattr(e, 'reason', e)}") from None
    except ValueError:
        raise ServeError(f"{url} did not return JSON.") from None
    preds = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(preds, list):
        raise ServeError(f"Unexpected response from {url}: {str(payload)[:200]}")
    return preds


def predict_frame(url: str, rows: pd.DataFrame, chunk: int = CHUNK_ROWS,
                  timeout: float = REQUEST_TIMEOUT) -> list:
    """Predictions for `rows` in order, sent `chunk` rows at a time. NaN and infinite values go
    out as JSON null."""
    target = invocations_url(url)
    preds: list = []
    for start_row in range(0, len(rows), chunk):
        part = rows.iloc[start_row:start_row + chunk]
        got = _post(target, '{"dataframe_records": ' + part.to_json(orient="records") + "}", timeout)
        if len(got) != len(part):
            raise ServeError(f"The endpoint returned {len(got)} predictions for {len(part)} rows "
                             f"(rows {start_row + 1}-{start_row + len(part)}).")
        preds.extend(got)
    return preds


def _prediction_frame(preds: list) -> pd.DataFrame:
    """MLflow returns a named Series as a list of one-key records ({"prediction": 0.4}); a plain
    list of numbers is accepted too. The result always has a `prediction` column."""
    if preds and all(isinstance(p, dict) for p in preds):
        frame = pd.DataFrame(preds)
        return frame.rename(columns={frame.columns[0]: "prediction"}) if len(frame.columns) == 1 else frame
    return pd.DataFrame({"prediction": preds})


def predict_csv(url: str, rows_path: Path, out_path: Path, target_col: str | None = None,
                keep_columns: bool = False, chunk: int = CHUNK_ROWS) -> int:
    """Score the rows of a CSV and write `out_path`; returns the number of rows."""
    try:
        rows = pd.read_csv(rows_path)
    except Exception as e:  # noqa: BLE001
        raise ServeError(f"Could not read {rows_path}: {str(e)[:200]}") from None
    if rows.empty:
        raise ServeError(f"{rows_path} has no rows.")
    sent = rows.drop(columns=[target_col]) if target_col and target_col in rows.columns else rows
    preds = predict_frame(url, sent, chunk=chunk)
    out = _prediction_frame(preds)
    if keep_columns:
        out = pd.concat([rows.reset_index(drop=True), out], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return len(rows)
