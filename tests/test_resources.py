import json
import stat
from dataclasses import asdict

import pytest

from orbit import html_report, results
from orbit.agent import Agent, Journal, Node, SearchConfig
from orbit.sandbox import DockerSandbox, ExecResult, parse_stats
from orbit.task import TaskSpec, prepare

from conftest import LocalSandbox, ScriptedLLM, clf_reply


def _line(mem, cpu):
    return json.dumps({"MemUsage": mem, "CPUPerc": cpu, "Name": "orbit-x"})


@pytest.mark.parametrize("mem, mib", [
    ("123.4MiB / 8GiB", 123.4), ("1GiB / 8GiB", 1024.0), ("512KiB / 8GiB", 0.5),
    ("1048576B / 8GiB", 1.0), ("2MB / 8GB", 2e6 / 1024**2), ("1GB / 8GB", 1e9 / 1024**2),
    ("1024kB / 8GB", 1024e3 / 1024**2), ("0B / 0B", 0.0)])
def test_parse_stats_memory_units(mem, mib):
    got, cpu = parse_stats(_line(mem, "12.5%"))
    assert got == pytest.approx(mib) and cpu == 12.5


def test_parse_stats_cpu():
    assert parse_stats(_line("1MiB / 2GiB", "123.45%"))[1] == pytest.approx(123.45)
    assert parse_stats(_line("1MiB / 2GiB", "0.00%"))[1] == 0.0


@pytest.mark.parametrize("line", ["", "not json", "[]", "null", "{}", '{"MemUsage": 5}',
                                  _line("--", "--"), _line("12 parsecs / 8GiB", "x%")])
def test_parse_stats_tolerates_malformed_lines(line):
    assert parse_stats(line) == (None, None)


def test_parse_stats_keeps_the_readable_half():
    assert parse_stats(_line("--", "5%")) == (None, 5.0)
    assert parse_stats(_line("2MiB / 1GiB", "--")) == (2.0, None)


def _fake_docker(tmp_path, stats_body):
    """A `docker` that sleeps for `run`, answers `stats` with `stats_body` and ignores the rest."""
    path = tmp_path / "docker"
    path.write_text(f"""#!/bin/sh
case "$1" in
  run) sleep 2; exit 0 ;;
  stats) {stats_body} ;;
esac
exit 0
""")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_run_reports_container_stats(tmp_path, monkeypatch):
    monkeypatch.setattr("orbit.sandbox.STATS_INTERVAL", 0.05)
    counter = tmp_path / "count"
    docker = _fake_docker(tmp_path, f"""
      echo x >> {counter}
      if [ "$(wc -l < {counter})" -le 1 ]; then
        echo '{_line("100MiB / 8GiB", "50.00%")}'
      else
        echo '{_line("200MiB / 8GiB", "150.00%")}'
      fi""")
    result = DockerSandbox(docker=docker).run("print(1)", tmp_path / "exp", tmp_path, 30)
    assert result.ok
    assert result.peak_mem_mb == pytest.approx(200)
    assert result.cpu_pct_peak == pytest.approx(150)
    assert 50 <= result.cpu_pct_mean <= 150


def test_run_survives_failing_stats(tmp_path):
    docker = _fake_docker(tmp_path, "echo 'Error: No such container' >&2; exit 1")
    result = DockerSandbox(docker=docker).run("print(1)", tmp_path / "exp", tmp_path, 30)
    assert result.ok
    assert (result.peak_mem_mb, result.cpu_pct_mean, result.cpu_pct_peak) == (None, None, None)


def test_run_survives_garbage_stats(tmp_path):
    docker = _fake_docker(tmp_path, "echo garbage")
    result = DockerSandbox(docker=docker).run("print(1)", tmp_path / "exp", tmp_path, 30)
    assert result.ok and result.peak_mem_mb is None


def test_run_reports_stats_after_a_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr("orbit.sandbox.STATS_INTERVAL", 0.05)
    docker = _fake_docker(tmp_path, f"echo '{_line('64MiB / 8GiB', '10.00%')}'")
    result = DockerSandbox(docker=docker).run("print(1)", tmp_path / "exp", tmp_path, 1)
    assert result.timed_out and result.peak_mem_mb == pytest.approx(64)


def test_exec_result_fields_are_optional():
    r = ExecResult(0, "", "", 1.0)
    assert r.peak_mem_mb is None and r.cpu_pct_mean is None and r.cpu_pct_peak is None


def test_node_round_trips_with_and_without_resource_fields(tmp_path):
    path = tmp_path / "journal.json"
    journal = Journal(nodes=[Node(id=1, stage="draft", parent=None, peak_mem_mb=512.5, cpu_pct=240.0),
                             Node(id=2, stage="improve", parent=1)])
    journal.save(path)
    loaded = Journal.load(path)
    assert (loaded.nodes[0].peak_mem_mb, loaded.nodes[0].cpu_pct) == (512.5, 240.0)
    assert (loaded.nodes[1].peak_mem_mb, loaded.nodes[1].cpu_pct) == (None, None)

    old = json.loads(path.read_text())  # a journal written before these fields existed
    for n in old["nodes"]:
        n.pop("peak_mem_mb"), n.pop("cpu_pct")
    path.write_text(json.dumps(old))
    assert Journal.load(path).nodes[0].peak_mem_mb is None
    assert "peak_mem_mb" in asdict(Node(id=3, stage="draft", parent=None))


class StatsSandbox(LocalSandbox):
    """Reports a different memory and CPU figure for each fold."""

    def __init__(self, stats):
        self.stats = list(stats)

    def run(self, code, workdir, data_dir, timeout):
        r = super().run(code, workdir, data_dir, timeout)
        r.peak_mem_mb, r.cpu_pct_mean, r.cpu_pct_peak = self.stats.pop(0)
        return r


def test_node_aggregates_stats_over_folds(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    sandbox = StatsSandbox([(100.0, 100.0, 120.0), (300.0, 200.0, 250.0), (200.0, 300.0, 310.0)])
    agent = Agent(task, ScriptedLLM([clf_reply()]), sandbox, SearchConfig(max_steps=1, n_drafts=1))
    node = agent.run().nodes[0]
    assert node.peak_mem_mb == 300.0 and node.cpu_pct == pytest.approx(200.0)


def test_node_without_stats_stays_none(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([clf_reply()]), LocalSandbox(), SearchConfig(max_steps=1, n_drafts=1))
    node = agent.run().nodes[0]
    assert node.peak_mem_mb is None and node.cpu_pct is None
    data = results.collect(task.run_dir)
    assert data["experiments"][0]["peak_mem_mb"] is None and data["experiments"][0]["cpu_pct"] is None
    assert "peak memory" in html_report.render(data)


def test_collect_and_html_show_peak_memory(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([clf_reply()]), StatsSandbox([(1536.4, 80.0, 99.0)]),
                  SearchConfig(max_steps=1, n_drafts=1))
    agent.run()
    data = results.collect(task.run_dir)
    assert data["experiments"][0]["peak_mem_mb"] == pytest.approx(1536.4)
    assert data["experiments"][0]["cpu_pct"] == pytest.approx(80.0)
    page = html_report.render(data)
    assert "<th>peak memory</th>" in page and "1,536 MB" in page
