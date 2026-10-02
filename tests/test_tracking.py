import json

import pytest

from orbit.agent import Agent, SearchConfig
from orbit.task import TaskSpec, prepare
from orbit.tracking import (MLflowTracker, NullTracker, data_fingerprint, make_tracker,
                            resolve_experiment)

from conftest import BROKEN, LocalSandbox, ScriptedLLM, clf_reply


@pytest.fixture
def mlflow_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # artifacts default to ./mlruns; keep them out of the repo
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    import mlflow
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.end_run()  # tests that only start() a tracker leave the parent run open
    mlflow.set_tracking_uri(None)


def test_mlflow_logs_parent_and_nested_runs(clf_csv, tmp_path, mlflow_uri):
    import mlflow

    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    cfg = SearchConfig(max_steps=2, n_drafts=2)
    tracker = MLflowTracker(experiment="orbit-test")
    tracker.start(task, cfg, "scripted")
    agent = Agent(task, ScriptedLLM([clf_reply(), BROKEN]), LocalSandbox(), cfg, tracker=tracker)
    journal = agent.run()
    report = task.run_dir / "report.md"
    report.write_text("# report")
    tracker.log_report(report)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("orbit-test")
    runs = client.search_runs([exp.experiment_id])
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id]

    assert parent.info.status == "FINISHED"
    assert parent.data.params["metric"] == "roc_auc"
    assert parent.data.metrics["best_val_score"] == pytest.approx(journal.get(1).metric)
    assert parent.data.tags["orbit.best_experiment"] == "1"
    assert {a.path for a in client.list_artifacts(parent.info.run_id)} >= {
        "journal.json", "report.md", "data_profile.txt"}

    info = json.loads((task.run_dir / "tracking.json").read_text())
    assert info["run_id"] == parent.info.run_id and info["uri"] == mlflow_uri

    assert len(children) == 2
    assert sorted(r.data.tags["orbit.status"] for r in children) == ["failed", "ok"]
    child_artifacts = {a.path for a in client.list_artifacts(children[0].info.run_id)}
    assert "solution.py" in child_artifacts


def test_mlflow_finish_closes_run_without_test_score(clf_csv, tmp_path, mlflow_uri):
    import mlflow

    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    cfg = SearchConfig(max_steps=1, n_drafts=1)
    tracker = MLflowTracker(experiment="orbit-test")
    tracker.start(task, cfg, "scripted")
    agent = Agent(task, ScriptedLLM([clf_reply()]), LocalSandbox(), cfg, tracker=tracker)
    agent.step()
    (task.run_dir / "nodes" / "001" / "predictions.csv").unlink()  # test scoring fails
    agent.finalize("done")

    run = mlflow.MlflowClient().get_run(tracker.parent.info.run_id)
    assert run.info.status == "FINISHED" and run.data.tags["orbit.stop_reason"] == "done"
    assert mlflow.active_run() is None


def test_mlflow_fails_fast_unless_configured(monkeypatch):
    for key in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
        monkeypatch.setenv(key, "x")  # registers the variable so teardown removes it again
        monkeypatch.delenv(key)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "5")  # explicit user setting wins
    MLflowTracker()
    import os
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "1"
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "15"
    assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "5"


def test_mlflow_console_announcements_are_silenced(monkeypatch):
    import os

    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "x")  # registers it for cleanup
    monkeypatch.delenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT")
    MLflowTracker()
    assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "1"


def _start(clf_csv, tmp_path, experiment, run="run", **spec):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1, **spec), tmp_path / run)
    tracker = MLflowTracker(experiment=experiment)
    tracker.start(task, SearchConfig(max_steps=1, n_drafts=1), "scripted")
    return task, tracker


def test_resolve_experiment(monkeypatch):
    monkeypatch.delenv("ORBIT_EXPERIMENT", raising=False)
    assert resolve_experiment(None, ["data/Wine Quality.csv"]) == "Wine_Quality"
    assert resolve_experiment(None, ["data/wine.csv"], "my run!") == "my_run"
    assert resolve_experiment(None, ["data/wine.csv"], "!!!") == "wine"  # nothing left of --name
    monkeypatch.setenv("ORBIT_EXPERIMENT", "  from env ")
    assert resolve_experiment(None, ["wine.csv"], "x") == "from env"
    assert resolve_experiment("  My Exp/1 ", ["wine.csv"]) == "My Exp/1"  # explicit text is kept
    monkeypatch.setenv("ORBIT_EXPERIMENT", "  ")
    assert resolve_experiment(None, ["wine.csv"]) == "wine"  # blank variable counts as unset
    with pytest.raises(ValueError, match="empty"):
        resolve_experiment("   ", ["wine.csv"])


def test_make_tracker_passes_experiment(monkeypatch):
    assert make_tracker("mlflow", "wine").experiment == "wine"
    assert make_tracker("mlflow").experiment == "orbit"
    assert isinstance(make_tracker("none", "wine"), NullTracker)


def test_runs_of_one_problem_share_an_experiment(clf_csv, tmp_path, mlflow_uri):
    import mlflow

    for run, experiment in (("r1", "problem-a"), ("r2", "problem-a"), ("r3", "problem-b")):
        _start(clf_csv, tmp_path, experiment, run=run)
        mlflow.end_run()
    client = mlflow.MlflowClient()
    counts = {n: len(client.search_runs([client.get_experiment_by_name(n).experiment_id]))
              for n in ("problem-a", "problem-b")}
    assert counts == {"problem-a": 2, "problem-b": 1}


def test_parent_run_tags_and_note(clf_csv, tmp_path, mlflow_uri):
    import mlflow

    task, tracker = _start(clf_csv, tmp_path, "tags", goal="predict " + "x" * 6000,
                           eval_text="area under the curve")
    run = mlflow.MlflowClient().get_run(tracker.parent.info.run_id)
    tags = run.data.tags
    assert tags["orbit.eval"] == "area under the curve" and tags["orbit.target"] == "label"
    assert tags["orbit.primary_table"] == "clf.csv" and tags["orbit.extra_tables"] == ""
    assert tags["orbit.metric"] == "roc_auc" and tags["orbit.validation"] == "single validation split"
    assert tags["orbit.version"] and len(tags["orbit.goal"]) == 5000
    assert "roc_auc" in tags["mlflow.note.content"] and "clf.csv" in tags["mlflow.note.content"]
    assert run.data.params["folds"] == run.data.params["cv_folds"] == "1"
    assert json.loads((task.run_dir / "tracking.json").read_text())["experiment_name"] == "tags"


def test_data_fingerprint_changes_with_the_file(clf_csv, tmp_path, mlflow_uri):
    import re

    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    first = data_fingerprint(task)
    assert re.fullmatch(r"rows=600 sha256=[0-9a-f]{12}", first)
    assert data_fingerprint(task) == first
    clf_csv.write_text(clf_csv.read_text() + "\n")
    assert data_fingerprint(task) != first


def test_deleted_experiment_is_restored(clf_csv, tmp_path, mlflow_uri):
    import mlflow

    client = mlflow.MlflowClient()
    exp_id = client.create_experiment("was-deleted")
    client.delete_experiment(exp_id)
    _, tracker = _start(clf_csv, tmp_path, "was-deleted")
    assert tracker.parent is not None
    exp = client.get_experiment_by_name("was-deleted")
    assert exp.experiment_id == exp_id and exp.lifecycle_stage == "active"
    assert len(client.search_runs([exp_id])) == 1


def test_tracking_failure_stays_non_fatal(clf_csv, tmp_path, mlflow_uri, monkeypatch):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    tracker = MLflowTracker(experiment="x")

    def boom(name):
        raise RuntimeError("server on fire")

    monkeypatch.setattr(tracker.mlflow, "set_experiment", boom)
    tracker.start(task, SearchConfig(max_steps=1, n_drafts=1), "scripted")  # must not raise
    assert tracker.parent is None


def test_experiments_command(clf_csv, tmp_path, mlflow_uri):
    from typer.testing import CliRunner

    from orbit.cli import app

    import mlflow

    for run, experiment in (("r1", "wine [x]"), ("r2", "wine [x]"), ("r3", "diabetes")):
        _start(clf_csv, tmp_path, experiment, run=run)
        mlflow.end_run()
    result = CliRunner().invoke(app, ["experiments"])
    assert result.exit_code == 0
    out = " ".join(result.output.split())
    assert "wine [x]" in out and "diabetes" in out and "Default" not in out
    rows = {line.split("│")[1].strip(): line.split("│")[3].strip()
            for line in result.output.splitlines() if line.count("│") >= 4 and "name" not in line}
    assert rows == {"wine [x]": "2", "diabetes": "1"}


def test_experiments_command_reports_unreachable_server(monkeypatch):
    from typer.testing import CliRunner

    from orbit.cli import app

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # nothing listens here
    for key, value in {"MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "2"}.items():
        monkeypatch.setenv(key, value)
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:9")
    try:
        result = CliRunner().invoke(app, ["experiments"])
    finally:
        mlflow.set_tracking_uri(None)
    assert result.exit_code == 1 and "Could not reach MLflow" in result.output
