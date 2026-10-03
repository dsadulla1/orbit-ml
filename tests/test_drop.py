"""`--drop`: columns the user removes from the primary table disappear everywhere."""
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from orbit import prompts, registry, report
from orbit.cli import app
from orbit.export import export_best
from orbit.registry import register_model
from orbit.task import (META_FILE, Task, TaskSpec, load_labeled, parse_drop, prepare,
                        prepare_export_input, setup_hints)

from conftest import LocalSandbox, ScriptedLLM, clf_reply
from test_export import CLF
from test_registry import MIXED_SEARCH, _subprocess
from test_tracking import mlflow_uri  # noqa: F401 - fixture

runner = CliRunner()


@pytest.fixture
def data(tmp_path):
    """Rows in time order: `day` orders them, `duration` is the column to drop."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"day": rng.permutation(n), "f0": rng.normal(size=n), "f1": rng.normal(size=n),
                       "duration": rng.integers(1, 500, size=n), "label": (rng.random(n) < 0.3).astype(int)})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return path


def _csvs(task):
    return [p for p in (task.run_dir / "input").rglob("*.csv")]


@pytest.mark.parametrize("extra", [dict(folds=1), dict(folds=3), dict(folds=1, time_col="day"),
                                   dict(folds=3, time_col="day"), dict(folds=3, time_order=True)])
def test_dropped_column_is_in_no_sandbox_file_or_description(data, tmp_path, extra):
    task = prepare(TaskSpec(str(data), "label", drop=["duration"], **extra), tmp_path / "run")
    files = _csvs(task)
    assert len(files) == 2 * task.n_folds
    for path in files:
        assert "duration" not in pd.read_csv(path).columns, path
    export_dir = prepare_export_input(task)
    assert all("duration" not in pd.read_csv(p).columns for p in export_dir.glob("*.csv"))
    meta = json.loads((export_dir / META_FILE).read_text())
    assert meta["dropped"] == ["duration"]
    assert "duration" not in json.dumps({k: v for k, v in meta.items() if k != "dropped"})
    assert "duration" not in task.profile
    assert "Columns: 4 " in task.profile
    assert "dropped columns: duration (as requested)" in task.spec.notes
    assert "duration" not in json.dumps(task.split_stats)
    assert "duration" not in load_labeled(task.spec).columns
    assert Task.load(task.run_dir).spec.drop == ["duration"]


def test_dropping_changes_nothing_else(data, tmp_path):
    plain = prepare(TaskSpec(str(data), "label", folds=3), tmp_path / "a")
    dropped = prepare(TaskSpec(str(data), "label", folds=3, drop=["duration"]), tmp_path / "b")
    for k in (1, 2, 3):
        for name in ("train.csv", "test.csv"):
            a = pd.read_csv(plain.fold_dir(k) / name).drop(columns=["duration"])
            b = pd.read_csv(dropped.fold_dir(k) / name)
            pd.testing.assert_frame_equal(a, b)


def test_dropping_the_time_column_keeps_the_ordering(data, tmp_path):
    kept = prepare(TaskSpec(str(data), "label", folds=1, time_col="day"), tmp_path / "a")
    gone = prepare(TaskSpec(str(data), "label", folds=1, time_col="day", drop=["day"]), tmp_path / "b")
    train_kept = pd.read_csv(kept.input_dir / "train.csv")
    train_gone = pd.read_csv(gone.input_dir / "train.csv")
    assert "day" not in train_gone.columns
    assert train_kept["day"].is_monotonic_increasing
    pd.testing.assert_frame_equal(train_kept.drop(columns=["day"]), train_gone)
    labeled = load_labeled(gone.spec)
    assert "day" not in labeled.columns
    raw = pd.read_csv(data).sort_values("day", kind="stable")
    assert list(labeled["f0"]) == list(raw["f0"])


def test_unknown_name_lists_close_matches_and_columns(data, tmp_path):
    with pytest.raises(ValueError) as e:
        prepare(TaskSpec(str(data), "label", drop=["Duration"]), tmp_path / "run")
    text = str(e.value)
    assert "'Duration'" in text and "Did you mean 'duration'" in text
    assert "Columns: day, f0, f1, duration, label" in text
    with pytest.raises(ValueError, match="no such column"):
        prepare(TaskSpec(str(data), "label", drop=["zzz"]), tmp_path / "run2")


def test_target_and_id_column_cannot_be_dropped(data, tmp_path):
    with pytest.raises(ValueError, match="target column and cannot be dropped"):
        prepare(TaskSpec(str(data), "label", drop=["label"]), tmp_path / "a")
    with pytest.raises(ValueError, match="id column"):
        prepare(TaskSpec(str(data), "label", id_column="day", drop=["day"]), tmp_path / "b")


def test_parse_drop_accepts_commas_and_repeats():
    assert parse_drop(["duration, day", "f0", "duration", " ", ""]) == ["duration", "day", "f0"]
    assert parse_drop(None) == [] and parse_drop([]) == []


def test_old_task_json_without_drop_loads(data, tmp_path):
    task = prepare(TaskSpec(str(data), "label", folds=1), tmp_path / "run")
    path = task.run_dir / "task.json"
    saved = json.loads(path.read_text())
    del saved["spec"]["drop"]
    path.write_text(json.dumps(saved))
    assert Task.load(task.run_dir).spec.drop == []


def test_date_hint_is_silent_for_a_dropped_date_column(tmp_path):
    n = 800
    df = pd.DataFrame({"signup_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
                       "f0": np.random.default_rng(0).normal(size=n),
                       "label": np.random.default_rng(1).integers(0, 2, size=n)})
    path = tmp_path / "dates.csv"
    df.to_csv(path, index=False)
    spec = TaskSpec(str(path), "label", folds=1)
    assert any("looks like a date" in h for h in setup_hints(df, spec, "classification", ["0", "1"]))
    task = prepare(TaskSpec(str(path), "label", folds=1, drop=["signup_date"]), tmp_path / "run")
    assert not any("looks like a date" in n for n in task.spec.notes)
    assert "dropped columns: signup_date (as requested)" in task.spec.notes


def test_prompt_names_the_dropped_columns(data, tmp_path):
    task = prepare(TaskSpec(str(data), "label", folds=1, drop=["duration", "f1"]), tmp_path / "run")
    text = prompts.data_section(task)
    assert ("The user removed these columns; they are not available and must not be recreated: "
            "duration, f1.") in text
    assert "removed these columns" in prompts.data_section(task, export=True)
    plain = prepare(TaskSpec(str(data), "label", folds=1), tmp_path / "plain")
    assert "removed these columns" not in prompts.data_section(plain)


def test_prompt_does_not_name_a_dropped_time_column(data, tmp_path):
    task = prepare(TaskSpec(str(data), "label", folds=1, time_col="day", drop=["day"]), tmp_path / "run")
    text = prompts.data_section(task)
    assert "later in time than the training rows" in text and "'day'" not in text


def test_report_and_html_list_the_dropped_columns(data, tmp_path):
    from orbit import html_report, results
    from orbit.agent import Agent, SearchConfig

    task = prepare(TaskSpec(str(data), "label", folds=1, drop=["duration"]), tmp_path / "run")
    journal = Agent(task, ScriptedLLM([clf_reply()]), LocalSandbox(),
                    SearchConfig(max_steps=1, n_drafts=1)).run()
    md, _ = report.build(task, journal, None)
    assert "- dropped columns: duration (as requested)" in md
    (task.run_dir / "report.md").write_text(md)
    html = html_report.render(results.collect(task.run_dir))
    assert "dropped columns: duration" in html


def test_explore_drop_with_and_without_target(data):
    r = runner.invoke(app, ["explore", str(data), "-t", "label", "--drop", "duration,f1"])
    assert r.exit_code == 0
    assert "dropped columns: duration, f1 (as requested)" in r.output
    profile = r.output.split("(as requested)")[1]
    assert "duration" not in profile and "f1" not in profile and "f0" in profile
    r = runner.invoke(app, ["explore", str(data), "--drop", "duration"])
    assert r.exit_code == 0 and "duration | int64 | 0 | " in r.output and "(dropped)" in r.output
    assert r.output.count("(dropped)") == 1
    r = runner.invoke(app, ["explore", str(data), "-t", "label", "--drop", "label", "--drop", "nope"])
    assert r.exit_code == 0 and "that is the target" in r.output and "no such column" in r.output


def test_run_passes_repeated_and_comma_separated_drop_to_setup(tmp_path, monkeypatch):
    from orbit.sources import SourceError

    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (True, "ok"))
    monkeypatch.setattr("orbit.cli._llm", lambda: ScriptedLLM([]))
    seen = {}

    def fake_setup(sources, run_dir, **kw):
        seen.update(kw)
        raise SourceError("stop here")

    monkeypatch.setattr("orbit.problem.setup_task", fake_setup)
    runner.invoke(app, ["run", "d.csv", "--out", str(tmp_path), "--drop", "a,b", "--drop", "c"])
    assert seen["drop"] == ["a", "b", "c"]
    assert "--drop" in runner.invoke(app, ["run", "--help"]).output


def test_tracker_tags_the_dropped_columns(data, tmp_path):
    from orbit.agent import SearchConfig
    from orbit.tracking import _run_tags

    task = prepare(TaskSpec(str(data), "label", folds=1, drop=["duration", "f1"]), tmp_path / "run")
    tags = _run_tags(task, SearchConfig())
    assert tags["orbit.dropped"] == "duration,f1" and "Dropped columns: duration,f1" in tags["mlflow.note.content"]
    plain = prepare(TaskSpec(str(data), "label", folds=1), tmp_path / "plain")
    assert "orbit.dropped" not in _run_tags(plain, SearchConfig())


def test_registered_model_does_not_ask_for_a_dropped_column(clf_csv, tmp_path, mlflow_uri):
    import mlflow
    from orbit.agent import Agent, SearchConfig
    from orbit.tracking import MLflowTracker

    df = pd.read_csv(clf_csv)
    df["duration"] = np.arange(len(df)) % 11
    csv = tmp_path / "with_duration.csv"
    df.to_csv(csv, index=False)
    task = prepare(TaskSpec(str(csv), "label", folds=1, drop=["duration"]), tmp_path / "run")
    cfg = SearchConfig(max_steps=1, n_drafts=1)
    tracker = MLflowTracker(experiment="orbit-drop")
    tracker.start(task, cfg, "scripted")
    journal = Agent(task, ScriptedLLM([MIXED_SEARCH]), LocalSandbox(), cfg, tracker=tracker).run()
    export = export_best(task, journal, ScriptedLLM([CLF]), LocalSandbox(), 120, 1)
    assert export.ok, export.error
    res = register_model(task.run_dir, export)
    assert res.version == 1, res.message

    local = mlflow.artifacts.download_artifacts(res.uri)
    features = json.loads(next(__import__("pathlib").Path(local).rglob(registry.FEATURES_FILE)).read_text())
    assert "duration" not in features["columns"] and "duration" not in features["features_order"]
    assert features["features_order"] == [f"f{i}" for i in range(8)]

    sample = pd.read_csv(export.sample_input)
    assert "duration" not in sample.columns
    with_it = tmp_path / "with.csv"
    sample.assign(duration=7).to_csv(with_it, index=False)
    out = _subprocess(tmp_path, mlflow_uri, f"""
        import json
        import mlflow
        import pandas as pd
        model = mlflow.pyfunc.load_model("models:/orbit-drop@champion")
        a = model.predict(pd.read_csv({str(export.sample_input)!r}))
        b = model.predict(pd.read_csv({str(with_it)!r}))
        print(json.dumps({{"without": [float(v) for v in a], "with": [float(v) for v in b]}}))
    """)
    assert len(out["without"]) == len(sample) and out["without"] == out["with"]
