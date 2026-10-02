import json
import re
import xml.etree.ElementTree as ET

import pytest
from typer.testing import CliRunner

from orbit import html_report, results
from orbit.agent import Agent, SearchConfig
from orbit.cli import _write_report, app
from orbit.task import TaskSpec, prepare
from orbit.tracking import MLflowTracker

from conftest import BROKEN, LocalSandbox, ScriptedLLM, clf_reply


def _run(tmp_path, csv, plan="Logistic regression.", **spec):
    task = prepare(TaskSpec(str(csv), "label", folds=3, goal="Flag the <b>positives</b>",
                            eval_text="ROC AUC", **spec), tmp_path / "run")
    replies = [clf_reply(plan), BROKEN, clf_reply("RF", "ensemble", "RandomForestClassifier", "random_state=0")]
    agent = Agent(task, ScriptedLLM(replies), LocalSandbox(), SearchConfig(max_steps=3, n_drafts=3))
    return task, agent.run()


def test_collect_and_html(clf_csv, tmp_path):
    task, journal = _run(tmp_path, clf_csv, plan="Plan with <script>alert(1)</script> inside.")
    from orbit import report
    md, _ = report.build(task, journal, None)
    (task.run_dir / "report.md").write_text(md)
    data = results.collect(task.run_dir)
    assert data["problem"]["goal"].startswith("Flag") and data["task"]["folds"] == 3
    assert [e["status"] for e in data["experiments"]] == ["ok", "failed", "ok"]
    ok = data["experiments"][0]
    assert len(ok["fold_scores"]) == 3 and ok["val_std"] is not None and data["best"]["code"]
    assert "**Verdict:**" in data["narrative"] and data["narrative_check"].startswith("Narrative source")

    page = html_report.render(data)
    assert "<script" not in page.lower().replace("<script>alert", "") or "&lt;script&gt;" in page
    assert "<b>positives</b>" not in page and "&lt;b&gt;positives&lt;/b&gt;" in page  # escaped
    assert "ValueError" in page and "3-fold cross-validation" in page and "±" in page
    for svg in re.findall(r"<svg.*?</svg>", page, re.S):
        root = ET.fromstring(svg)  # well-formed
        assert root.attrib["viewBox"]
    assert "http" not in " ".join(re.findall(r'(?:src|href)="([^"]*)"', page))


def test_html_without_narrative_or_cv(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    Agent(task, ScriptedLLM([BROKEN]), LocalSandbox(), SearchConfig(max_steps=1, n_drafts=1)).run()
    page = html_report.render(results.collect(task.run_dir))  # nothing worked: no best, no chart
    assert "<svg" not in page and "failed" in page


def test_results_cli_and_report_files(clf_csv, tmp_path):
    task, journal = _run(tmp_path, clf_csv)
    _write_report(task, journal, None)
    assert (task.run_dir / "report.md").exists() and (task.run_dir / "report.html").exists()
    runner = CliRunner()
    r = runner.invoke(app, ["results", str(task.run_dir)])
    flat = "".join(r.output.split())  # the test terminal is narrow: long paths wrap mid-word
    assert r.exit_code == 0 and "Best:experiment" in flat and "report.html" in flat
    assert "Flagthe<b>positives</b>" in flat  # user text is shown literally
    r = runner.invoke(app, ["results", str(task.run_dir), "--json"])
    assert json.loads(r.output)["run"] == "run"
    out = tmp_path / "copy.html"
    r = runner.invoke(app, ["results", str(task.run_dir), "--html", str(out)])
    assert out.exists() and "<!doctype html>" in out.read_text()
    assert runner.invoke(app, ["results"]).exit_code != 0


def test_collect_from_mlflow_matches_the_run_folder(clf_csv, tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    try:
        task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
        cfg = SearchConfig(max_steps=2, n_drafts=2)
        tracker = MLflowTracker(experiment="orbit-results")
        tracker.start(task, cfg, "scripted")
        journal = Agent(task, ScriptedLLM([clf_reply(), BROKEN]), LocalSandbox(), cfg, tracker=tracker).run()
        info = json.loads((task.run_dir / "tracking.json").read_text())
        remote = results.collect_mlflow(info["run_id"])
        local = results.collect(task.run_dir)
        assert [e["id"] for e in remote["experiments"]] == [1, 2]
        assert remote["experiments"][0]["val_score"] == pytest.approx(local["experiments"][0]["val_score"])
        assert remote["experiments"][0]["fold_scores"] == pytest.approx(local["experiments"][0]["fold_scores"])
        assert remote["experiments"][1]["status"] == "failed"
        assert remote["best"]["id"] == journal.best_id == 1
        assert remote["best"]["test_score"] == pytest.approx(journal.get(1).test_metric)
        assert remote["task"]["folds"] == 3 and local["tracker"]["run_id"] == info["run_id"]
        assert local["tracker"]["experiment_name"] == "orbit-results"
    finally:
        mlflow.set_tracking_uri(None)


def test_html_names_the_mlflow_experiment(clf_csv, tmp_path):
    task, _ = _run(tmp_path, clf_csv)
    (task.run_dir / "tracking.json").write_text(json.dumps({
        "uri": "http://localhost:5000", "experiment_id": "3", "run_id": "abc",
        "experiment_name": "wine <b>"}))
    page = html_report.render(results.collect(task.run_dir))
    assert "Open in MLflow: wine &lt;b&gt;" in page
    (task.run_dir / "tracking.json").write_text(json.dumps({
        "uri": "http://localhost:5000", "experiment_id": "3", "run_id": "abc"}))
    assert "Open in MLflow</a>" in html_report.render(results.collect(task.run_dir))


def test_damaged_tracking_file_and_bad_run_ids(clf_csv, tmp_path):
    task, _ = _run(tmp_path, clf_csv)
    (task.run_dir / "tracking.json").write_text("{not json")
    assert results.collect(task.run_dir)["tracker"] is None
    (task.run_dir / "tracking.json").write_text('{"uri": "http://x"}')  # missing ids
    assert results.collect(task.run_dir)["tracker"] is None
    with pytest.raises(ValueError, match="not an MLflow run id"):
        results.collect_mlflow("abc' OR tags.x = '1")


def test_html_links_to_the_traces(clf_csv, tmp_path):
    task, _ = _run(tmp_path, clf_csv)
    (task.run_dir / "tracking.json").write_text(json.dumps({
        "uri": "http://localhost:5000/", "experiment_id": "3", "run_id": "abc"}))
    data = results.collect(task.run_dir)
    assert data["tracker"]["traces_url"] == "http://localhost:5000/#/experiments/3/traces"
    assert 'href="http://localhost:5000/#/experiments/3/traces">Traces</a>' in html_report.render(data)
    (task.run_dir / "tracking.json").write_text(json.dumps({
        "uri": "sqlite:///x.db", "experiment_id": "3", "run_id": "abc"}))
    data = results.collect(task.run_dir)
    assert "traces_url" not in data["tracker"]
    assert "Traces</a>" not in html_report.render(data)
