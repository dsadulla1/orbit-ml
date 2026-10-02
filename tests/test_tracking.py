import json

import pytest

from orbit.agent import Agent, SearchConfig
from orbit.task import TaskSpec, prepare
from orbit.tracking import MLflowTracker

from conftest import BROKEN, LocalSandbox, ScriptedLLM, clf_reply


@pytest.fixture
def mlflow_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # artifacts default to ./mlruns; keep them out of the repo
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    import mlflow
    mlflow.set_tracking_uri(uri)
    yield uri
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


def test_mlflow_console_announcements_are_silenced():
    import logging

    MLflowTracker()
    assert logging.getLogger("mlflow.tracking.fluent").getEffectiveLevel() >= logging.WARNING
