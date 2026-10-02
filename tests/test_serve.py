import json
import socket
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest
from typer.testing import CliRunner

from orbit import serve as srv
from orbit.cli import app

runner = CliRunner()


# --- helpers ----------------------------------------------------------------------------------

class _Server:
    """A throwaway HTTP server on a free port; `handler(request, path, body)` returns
    (status, payload) and every request is recorded."""

    def __init__(self, handler, port=0):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode() if length else ""
                outer.requests.append((self.command, self.path, body))
                status, payload = handler(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _answer

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    started = []

    def make(handler, port=0):
        started.append(_Server(handler, port))
        return started[-1]

    yield make
    for s in started:
        s.close()


def _unused_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _echo(path, body):
    if path == "/ping":
        return 200, {}
    rows = json.loads(body)["dataframe_records"]
    return 200, {"predictions": [{"prediction": r["x"] * 10} for r in rows]}


PS_LINE = "orbit-serve-wine-8090\t1\tmodels:/wine@champion\t3\t127.0.0.1:8090->8080/tcp\t5 minutes ago"

INSPECT = [{
    "Config": {"Labels": {"com.docker.compose.project": "myproj", "com.docker.compose.service": "tracking"},
               "Cmd": ["mlflow", "server", "--host=0.0.0.0", "--port=5050"]},
    "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "172.17.0.5"},
                                     "myproj_default": {"IPAddress": "172.30.0.2"}}},
}]


# --- URIs ---------------------------------------------------------------------------------------

def test_model_uri_forms():
    assert srv.model_uri("wine") == "models:/wine@champion"
    assert srv.model_uri("wine", alias="staging") == "models:/wine@staging"
    assert srv.model_uri("wine", version=3) == "models:/wine/3"
    assert srv.model_uri("models:/wine/2") == "models:/wine/2"
    for bad in (dict(version=1, alias="a"), ):
        with pytest.raises(srv.ServeError):
            srv.model_uri("wine", **bad)
    with pytest.raises(srv.ServeError, match="full URI"):
        srv.model_uri("models:/wine/2", version=3)
    with pytest.raises(srv.ServeError):
        srv.model_uri("a/b")


def test_parse_uri_and_name():
    assert srv.parse_uri("models:/wine@champion") == ("wine", None, "champion")
    assert srv.parse_uri("models:/wine/3") == ("wine", "3", None)
    assert srv.parse_uri("models:/wine") == ("wine", None, None)
    assert srv.model_name_of("models:/wine@x") == "wine"


class _Client:
    class V:
        version = 4

    def get_model_version_by_alias(self, name, alias):
        if (name, alias) != ("wine", "champion"):
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
        return self.V()

    def get_model_version(self, name, version):
        if (name, version) != ("wine", "4"):
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
        return self.V()

    def search_registered_models(self, max_results=100):
        return [type("M", (), {"name": "wine"})(), type("M", (), {"name": "adult"})()]


def test_resolve_in_registry():
    client = _Client()
    assert srv.resolve_in_registry(client, "models:/wine@champion") == ("wine", "4")
    assert srv.resolve_in_registry(client, "models:/wine/4") == ("wine", "4")
    assert srv.resolve_in_registry(client, "models:/wine/Production") == ("wine", "Production")
    with pytest.raises(srv.ServeError, match="Registered models: adult, wine"):
        srv.resolve_in_registry(client, "models:/nope@champion")
    with pytest.raises(srv.ServeError, match="not found"):
        srv.resolve_in_registry(client, "models:/wine/9")


def test_example_row_formats():
    assert srv.example_row({"columns": ["a", "b"], "data": [[1, "x"], [2, "y"]]}) == {"a": 1, "b": "x"}
    assert srv.example_row({"dataframe_split": {"columns": ["a"], "data": [[5]]}}) == {"a": 5}
    assert srv.example_row({"dataframe_records": [{"a": 1}]}) == {"a": 1}
    assert srv.example_row([{"a": 1}]) == {"a": 1}
    assert srv.example_row({"unexpected": 1}) is None


def test_curl_command_is_quoted_json():
    row = {"a": 1, "name": "it's"}
    cmd = srv.curl_command("http://127.0.0.1:8090", row)
    import shlex
    parts = shlex.split(cmd)
    assert parts[:2] == ["curl", "-s"] and parts[2] == "http://127.0.0.1:8090/invocations"
    assert json.loads(parts[-1]) == {"dataframe_records": [row]}
    assert "<feature>" in srv.curl_command("http://x", None)


# --- docker command -----------------------------------------------------------------------------

def test_run_command_flags():
    cmd = srv.run_command("orbit-serve-wine-8090", "models:/wine@champion", "3", "myproj_default",
                          "http://172.30.0.2:5050", 8090, "orbit-serve:latest", "4g")
    assert cmd[:3] == ["docker", "run", "-d"]
    joined = " ".join(cmd)
    for flag in ("--cap-drop ALL", "--security-opt no-new-privileges", "--memory 4g",
                 "--pids-limit 512", "--tmpfs /tmp:", "--network myproj_default",
                 "-p 127.0.0.1:8090:8080", "-e MLFLOW_TRACKING_URI=http://172.30.0.2:5050",
                 "--label orbit.serve=1", "--label orbit.model=models:/wine@champion",
                 "--name orbit-serve-wine-8090"):
        assert flag in joined, flag
    assert "--read-only" not in cmd and "none" not in cmd
    assert all("0.0.0.0:" not in c for c in cmd)  # never published on every interface
    tail = cmd[cmd.index("orbit-serve:latest") + 1:]
    assert tail == ["mlflow", "models", "serve", "-m", "models:/wine@champion", "--env-manager",
                    "local", "--host", "0.0.0.0", "--port", "8080"]
    assert not any(c.startswith("-v") or c == "--volume" or "mlflow-data" in c for c in cmd)


def test_container_name_slug_and_override():
    assert srv.container_name("Wine Quality!", 8091) == "orbit-serve-wine_quality-8091"
    assert srv.container_name("???", 8090) == "orbit-serve-model-8090"
    assert srv.container_name("wine", 1, name="mine") == "orbit-serve-mine"
    assert srv.container_name("wine", 1, name="orbit-serve-x") == "orbit-serve-x"
    with pytest.raises(srv.ServeError):
        srv.container_name("wine", 1, name="bad name")


def test_memory_and_image_from_environment(monkeypatch):
    monkeypatch.delenv("ORBIT_SERVE_MEMORY", raising=False)
    monkeypatch.delenv("ORBIT_SERVE_IMAGE", raising=False)
    assert srv.memory_limit() == "4g" and srv.image_name() == "orbit-serve:latest"
    monkeypatch.setenv("ORBIT_SERVE_MEMORY", "2g")
    assert srv.memory_limit() == "2g"


def test_discover_mlflow_from_compose_labels():
    found = srv.discover_mlflow(json.dumps(INSPECT))
    assert found == {"network": "myproj_default", "service": "tracking", "port": 5050,
                     "ip": "172.30.0.2"}


def test_discover_mlflow_fallbacks():
    default = {"network": "orbit_default", "service": "mlflow", "port": 5000, "ip": ""}
    for text in (None, "", "not json", "[]", "{}", "[1]"):
        assert srv.discover_mlflow(text) == default
    no_labels = [{"NetworkSettings": {"Networks": {"other_net": {"IPAddress": "10.0.0.3"}}}}]
    assert srv.discover_mlflow(json.dumps(no_labels)) == {
        "network": "other_net", "service": "mlflow", "port": 5000, "ip": "10.0.0.3"}


def test_choose_port():
    assert srv.choose_port(None, lambda p: True) == srv.DEFAULT_PORT
    assert srv.choose_port(None, lambda p: p >= srv.DEFAULT_PORT + 2) == srv.DEFAULT_PORT + 2
    assert srv.choose_port(9000, lambda p: True) == 9000
    with pytest.raises(srv.ServeError, match="already in use"):
        srv.choose_port(9000, lambda p: False)
    with pytest.raises(srv.ServeError, match="No free port"):
        srv.choose_port(None, lambda p: False)


def test_port_is_free_sees_a_listener():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert not srv.port_is_free(s.getsockname()[1])
    assert srv.port_is_free(_unused_port())


# --- list and stop ------------------------------------------------------------------------------

def test_parse_ps_keeps_only_orbit_serving_containers():
    output = "\n".join([
        PS_LINE,
        "orbit-serve-imposter\t\tmodels:/x@y\t1\t127.0.0.1:1->8080/tcp\t1 hour ago",    # no label
        "other-container\t1\tmodels:/x@y\t1\t127.0.0.1:2->8080/tcp\t1 hour ago",       # no prefix
        "orbit-serve-b\t1\tmodels:/b/2\t2\t\t2 hours ago",                              # no port
        "garbage",
    ])
    got = srv.parse_ps(output)
    assert [(c.name, c.model, c.version, c.port, c.uptime) for c in got] == [
        ("orbit-serve-wine-8090", "models:/wine@champion", "3", 8090, "5 minutes ago"),
        ("orbit-serve-b", "models:/b/2", "2", None, "2 hours ago")]


def test_stop_targets_refuse_foreign_names():
    containers = srv.parse_ps(PS_LINE)
    assert srv.stop_targets(containers, "all") == ["orbit-serve-wine-8090"]
    assert srv.stop_targets(containers, "orbit-serve-wine-8090") == ["orbit-serve-wine-8090"]
    for target in ("orbit-mlflow-1", "orbit-serve-other", "wine"):
        with pytest.raises(srv.ServeError, match="not an orbit serving container"):
            srv.stop_targets(containers, target)
    assert srv.stop_targets([], "all") == []


def test_remove_refuses_names_without_the_prefix(monkeypatch):
    calls = []
    monkeypatch.setattr(srv, "docker", lambda *a, **k: calls.append(a) or
                        type("R", (), {"returncode": 0, "stderr": ""})())
    with pytest.raises(srv.ServeError, match="Refusing"):
        srv.remove(["orbit-mlflow-1"])
    assert calls == []
    srv.remove(["orbit-serve-a"])
    assert calls == [("rm", "-f", "orbit-serve-a")]


# --- readiness ----------------------------------------------------------------------------------

def test_wait_ready_success(server):
    s = server(_echo)
    assert srv.wait_ready(s.port, lambda: True, timeout=5, interval=0.05) == "ready"
    assert s.requests[0][1] == "/ping"


def test_wait_ready_after_a_slow_start(server):
    state = {"n": 0}

    def handler(path, body):
        state["n"] += 1
        return (200, {}) if state["n"] > 2 else (503, {})

    s = server(handler)
    assert srv.wait_ready(s.port, lambda: True, timeout=5, interval=0.05) == "ready"
    assert state["n"] == 3


def test_wait_ready_container_exited():
    assert srv.wait_ready(_unused_port(), lambda: False, timeout=5, interval=0.05) == "exited"


def test_wait_ready_timeout():
    assert srv.wait_ready(_unused_port(), lambda: True, timeout=0.3, interval=0.05) == "timeout"


# --- predict client -----------------------------------------------------------------------------

def test_invocations_url():
    assert srv.invocations_url("http://127.0.0.1:8090") == "http://127.0.0.1:8090/invocations"
    assert srv.invocations_url("http://127.0.0.1:8090/") == "http://127.0.0.1:8090/invocations"
    assert srv.invocations_url("http://h:1/invocations") == "http://h:1/invocations"


def test_predict_csv_chunks_nan_and_order(server, tmp_path):
    s = server(_echo)
    rows = pd.DataFrame({"x": range(7), "name": list("abcdefg")})
    rows.loc[3, "name"] = None
    rows["f"] = [0.5, None, 1.5, 2.5, 3.5, 4.5, 5.5]
    src, out = tmp_path / "rows.csv", tmp_path / "sub" / "out.csv"
    rows.to_csv(src, index=False)
    assert srv.predict_csv(s.url, src, out, chunk=3) == 7
    posts = [json.loads(b)["dataframe_records"] for _, p, b in s.requests if p == "/invocations"]
    assert [len(p) for p in posts] == [3, 3, 1]
    assert posts[1][0]["name"] is None and posts[0][1]["f"] is None  # NaN went out as null
    got = pd.read_csv(out)
    assert list(got.columns) == ["prediction"] and got["prediction"].tolist() == [i * 10 for i in range(7)]


def test_predict_csv_keep_columns_and_target_col(server, tmp_path):
    s = server(_echo)
    pd.DataFrame({"x": [1, 2], "y": [5, 6]}).to_csv(tmp_path / "r.csv", index=False)
    srv.predict_csv(s.url + "/invocations", tmp_path / "r.csv", tmp_path / "o.csv", target_col="y",
                    keep_columns=True)
    sent = json.loads(s.requests[0][2])["dataframe_records"]
    assert sent == [{"x": 1}, {"x": 2}]
    got = pd.read_csv(tmp_path / "o.csv")
    assert list(got.columns) == ["x", "y", "prediction"] and got["prediction"].tolist() == [10, 20]


def test_predict_plain_number_predictions(server, tmp_path):
    s = server(lambda p, b: (200, {"predictions": [0.25, 0.75]}))
    pd.DataFrame({"x": [1, 2]}).to_csv(tmp_path / "r.csv", index=False)
    srv.predict_csv(s.url, tmp_path / "r.csv", tmp_path / "o.csv")
    assert pd.read_csv(tmp_path / "o.csv")["prediction"].tolist() == [0.25, 0.75]


def test_predict_shows_the_servers_error(server, tmp_path):
    message = ("Failed to predict data '   x\n0  1'. Error: Failed to enforce schema. "
               "Error: Model is missing inputs ['recency_days'].")
    s = server(lambda p, b: (400, {"error_code": "BAD_REQUEST", "message": message}))
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "r.csv", index=False)
    with pytest.raises(srv.ServeError, match=r"HTTP 400: Model is missing inputs \['recency_days'\]"):
        srv.predict_csv(s.url, tmp_path / "r.csv", tmp_path / "o.csv")
    assert not (tmp_path / "o.csv").exists()


def test_predict_unreachable_wrong_count_and_empty(server, tmp_path):
    pd.DataFrame({"x": [1, 2]}).to_csv(tmp_path / "r.csv", index=False)
    with pytest.raises(srv.ServeError, match="Could not reach"):
        srv.predict_csv(f"http://127.0.0.1:{_unused_port()}", tmp_path / "r.csv", tmp_path / "o.csv")
    s = server(lambda p, b: (200, {"predictions": [1]}))
    with pytest.raises(srv.ServeError, match="1 predictions for 2 rows"):
        srv.predict_csv(s.url, tmp_path / "r.csv", tmp_path / "o.csv")
    (tmp_path / "e.csv").write_text("x\n")
    with pytest.raises(srv.ServeError, match="no rows"):
        srv.predict_csv(s.url, tmp_path / "e.csv", tmp_path / "o.csv")


def test_endpoint_for_named_model():
    one = srv.parse_ps(PS_LINE)
    assert srv.endpoint_for("wine", one) == "http://127.0.0.1:8090"
    assert srv.endpoint_for("models:/wine@champion", one) == "http://127.0.0.1:8090"
    with pytest.raises(srv.ServeError, match="No running endpoint serves 'adult'"):
        srv.endpoint_for("adult", one)
    with pytest.raises(srv.ServeError, match="No running endpoint"):
        srv.endpoint_for("adult", [])
    two = srv.parse_ps(PS_LINE + "\n" + PS_LINE.replace("8090", "8091"))
    with pytest.raises(srv.ServeError, match="more than once"):
        srv.endpoint_for("wine", two)


# --- CLI with a fake docker ---------------------------------------------------------------------

FAKE_DOCKER = """#!/bin/sh
echo "$@" >> "$FAKE_LOG"
case "$1" in
  info) echo 27.0.1 ;;
  image) exit "${FAKE_IMAGE_RC:-0}" ;;
  inspect)
    case "$*" in
      *State.Running*) echo true ;;
      *) cat "$FAKE_INSPECT" ;;
    esac ;;
  run) echo 0123456789abcdef ;;
  ps) cat "$FAKE_PS" ;;
  logs) echo "fake log line" ;;
esac
exit 0
"""


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "docker"
    script.write_text(FAKE_DOCKER)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "inspect.json").write_text(json.dumps(INSPECT))
    (tmp_path / "ps.txt").write_text(PS_LINE + "\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log.txt"))
    monkeypatch.setenv("FAKE_INSPECT", str(tmp_path / "inspect.json"))
    monkeypatch.setenv("FAKE_PS", str(tmp_path / "ps.txt"))
    monkeypatch.delenv("FAKE_IMAGE_RC", raising=False)
    monkeypatch.delenv("ORBIT_SERVE_IMAGE", raising=False)
    monkeypatch.delenv("ORBIT_SERVE_MEMORY", raising=False)

    def calls():
        log = tmp_path / "log.txt"
        return log.read_text().splitlines() if log.exists() else []

    return calls


def test_cli_missing_image_prints_build_command(fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_IMAGE_RC", "1")
    result = runner.invoke(app, ["serve", "wine"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "orbit-serve:latest" in flat and srv.BUILD_COMMAND in flat
    assert not any(c.startswith("run") for c in fake_docker())


def test_cli_serve_needs_a_model_or_a_flag(fake_docker):
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and "Give a model name" in result.output


def test_cli_list(fake_docker):
    result = runner.invoke(app, ["serve", "--list"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "wine" in flat and "8090" in flat and "5 minutes ago" in flat
    assert any(c.startswith("ps ") and "label=orbit.serve=1" in c for c in fake_docker())


def test_cli_list_empty(fake_docker, tmp_path):
    (tmp_path / "ps.txt").write_text("")
    result = runner.invoke(app, ["serve", "--list"])
    assert result.exit_code == 0 and "No orbit serving containers" in result.output


def test_cli_stop_all_and_by_name(fake_docker):
    result = runner.invoke(app, ["serve", "--stop", "all"])
    assert result.exit_code == 0 and "orbit-serve-wine-8090" in result.output
    assert "rm -f orbit-serve-wine-8090" in fake_docker()
    result = runner.invoke(app, ["serve", "--stop", "orbit-mlflow-1"])
    assert result.exit_code == 1 and "not an orbit serving container" in result.output
    assert "rm -f orbit-mlflow-1" not in fake_docker()


def test_cli_serve_happy_path(fake_docker, monkeypatch, server):
    import orbit.cli
    import orbit.tracking

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setattr(orbit.cli, "_check_mlflow", lambda: "[green]ok[/]")
    monkeypatch.setattr(orbit.tracking, "configure_mlflow_env", lambda: None)
    monkeypatch.setattr(srv, "resolve_in_registry", lambda client, uri: ("wine[x]", "3"))
    monkeypatch.setattr(srv, "fetch_input_example", lambda uri: {"a": 1})
    port = _unused_port()
    real_start = srv.start

    def start_and_listen(command):
        out = real_start(command)  # the fake `docker run`
        server(_echo, port)  # the "container" answers /ping once it is up
        return out

    monkeypatch.setattr(srv, "start", start_and_listen)
    result = runner.invoke(app, ["serve", "wine", "--port", str(port), "--alias", "staging"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert f"http://127.0.0.1:{port}/invocations" in flat and "wine[x] version 3" in flat
    assert f"orbit serve --stop orbit-serve-wine_x-{port}" in flat
    assert "dataframe_records" in flat and "curl" in flat
    run = next(c for c in fake_docker() if c.startswith("run "))
    assert f"-p 127.0.0.1:{port}:8080" in run and "--network myproj_default" in run
    assert "MLFLOW_TRACKING_URI=http://172.30.0.2:5050" in run and "models:/wine@staging" in run


def test_cli_serve_container_exits_early(fake_docker, monkeypatch):
    import orbit.cli
    import orbit.tracking

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setattr(orbit.cli, "_check_mlflow", lambda: "[green]ok[/]")
    monkeypatch.setattr(orbit.tracking, "configure_mlflow_env", lambda: None)
    monkeypatch.setattr(srv, "resolve_in_registry", lambda client, uri: ("wine", "1"))
    monkeypatch.setattr(srv, "wait_ready", lambda *a, **k: "exited")
    result = runner.invoke(app, ["serve", "wine", "--port", str(_unused_port())])
    assert result.exit_code == 1 and "fake log line" in result.output
    assert any(c.startswith("rm -f orbit-serve-wine-") for c in fake_docker())


def test_cli_predict_by_model_name(fake_docker, tmp_path, server):
    s = server(_echo)
    (tmp_path / "ps.txt").write_text(PS_LINE.replace("8090", str(s.port)) + "\n")
    rows = tmp_path / "rows.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(rows, index=False)
    out = tmp_path / "scored.csv"
    result = runner.invoke(app, ["predict", "wine", str(rows), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Scored 3 rows" in result.output
    assert pd.read_csv(out)["prediction"].tolist() == [10, 20, 30]
    result = runner.invoke(app, ["predict", "adult", str(rows), "--out", str(out)])
    assert result.exit_code == 1 and "No running endpoint serves 'adult'" in result.output


def test_cli_predict_url_error_is_escaped(server, tmp_path):
    s = server(lambda p, b: (400, {"message": "bad [red]markup[/red]"}))
    rows = tmp_path / "rows.csv"
    pd.DataFrame({"x": [1]}).to_csv(rows, index=False)
    result = runner.invoke(app, ["predict", s.url, str(rows), "--out", str(tmp_path / "o.csv")])
    assert result.exit_code == 1 and "bad [red]markup[/red]" in result.output
