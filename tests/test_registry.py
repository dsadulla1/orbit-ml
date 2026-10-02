import json
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orbit import registry
from orbit.agent import Agent, Journal, SearchConfig
from orbit.export import ExportResult, export_best
from orbit.orbit_model import OrbitModel
from orbit.registry import CHAMPION, RegistryResult, register_model
from orbit.task import TaskSpec, prepare
from orbit.tracking import MLflowTracker

from conftest import LocalSandbox, ScriptedLLM, clf_reply
from test_export import CLF, REG
from test_tracking import mlflow_uri  # noqa: F401 - fixture

REG_SEARCH = """PLAN: Ridge regression.
```python
import pandas as pd
from sklearn.linear_model import Ridge
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
model = Ridge().fit(train.drop(columns=["price"]), train["price"])
pd.DataFrame({"orbit_id": test["orbit_id"],
              "prediction": model.predict(test.drop(columns=["orbit_id"]))}).to_csv("predictions.csv", index=False)
```"""

MIXED_SEARCH = """PLAN: Logistic regression on numeric columns.
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
cols = [c for c in train.columns if c.startswith("f")]
model = LogisticRegression(max_iter=500).fit(train[cols].fillna(0), train["label"])
proba = model.predict_proba(test[cols].fillna(0))[:, list(model.classes_).index("yes")]
pd.DataFrame({"orbit_id": test["orbit_id"], "prediction": proba}).to_csv("predictions.csv", index=False)
```"""

CLEAN_PATH = """
import os, sys
sys.path[:] = [p for p in sys.path if not os.path.exists(os.path.join(p or ".", "orbit", "__init__.py"))]
try:
    import orbit
    raise SystemExit("orbit must not be importable here")
except ImportError:
    pass
"""


def _search_and_export(tmp_path, csv, target, search_reply, export_reply, experiment="orbit-test"):
    task = prepare(TaskSpec(str(csv), target, folds=1), tmp_path / "run")
    cfg = SearchConfig(max_steps=1, n_drafts=1)
    tracker = MLflowTracker(experiment=experiment)
    tracker.start(task, cfg, "scripted")
    journal = Agent(task, ScriptedLLM([search_reply]), LocalSandbox(), cfg, tracker=tracker).run()
    export = export_best(task, journal, ScriptedLLM([export_reply]), LocalSandbox(), 120, 1)
    assert export.ok, export.error
    return task, journal, export


@pytest.fixture
def clf_run(clf_csv, tmp_path, mlflow_uri):
    return _search_and_export(tmp_path, clf_csv, "label", clf_reply(), CLF)


@pytest.fixture
def reg_run(reg_csv, tmp_path, mlflow_uri):
    return _search_and_export(tmp_path, reg_csv, "price", REG_SEARCH, REG)


def _subprocess(tmp_path, uri, body):
    """Run `body` in an interpreter that cannot import orbit; returns the JSON it prints last."""
    script = tmp_path / "serve_check.py"
    script.write_text(CLEAN_PATH + textwrap.dedent(body))
    env = {"PATH": "", "PYTHONPATH": "", "MLFLOW_TRACKING_URI": uri, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


PREDICT_SAMPLE = """
import json
import mlflow
import pandas as pd
model = mlflow.pyfunc.load_model("models:/{name}@champion")
sample = pd.read_csv({sample!r})
out = model.predict(sample)
print(json.dumps([float(v) if not isinstance(v, str) else v for v in out]))
"""


@pytest.mark.parametrize("which", ["clf_run", "reg_run"])
def test_register_and_serve_without_orbit(which, request, tmp_path, mlflow_uri):
    import mlflow

    task, journal, export = request.getfixturevalue(which)
    res = register_model(task.run_dir, export)
    assert res.version == 1 and res.name == "orbit-test" and res.promoted, res.message
    assert res.uri == "models:/orbit-test/1" and CHAMPION in res.aliases

    client = mlflow.MlflowClient()
    version = client.get_model_version("orbit-test", "1")
    assert int(client.get_model_version_by_alias("orbit-test", CHAMPION).version) == 1
    best = journal.get(journal.best_id)
    assert version.tags["orbit.run_id"] == json.loads((task.run_dir / "tracking.json").read_text())["run_id"]
    assert version.tags["orbit.experiment"] == "1"
    assert version.tags["orbit.metric"] == task.metric_name
    assert float(version.tags["orbit.val_score"]) == pytest.approx(best.metric)
    assert float(version.tags["orbit.test_score"]) == pytest.approx(best.test_metric)
    assert float(version.tags["orbit.baseline"]) == pytest.approx(task.baseline)
    assert version.tags["orbit.validation"] == "single validation split"
    assert version.tags["orbit.data_fingerprint"].startswith("rows=")
    assert "orbit.val_std" not in version.tags
    assert "refit on all labeled rows" in version.description
    assert "cross-validation score and the test score" in version.description

    info = mlflow.models.get_model_info(res.uri)
    assert [c.name for c in info.signature.inputs] == list(pd.read_csv(export.sample_input).columns)
    assert info.signature.outputs is not None
    local = Path(mlflow.artifacts.download_artifacts(res.uri))
    files = {p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()}
    assert {"MLmodel", "requirements.txt", "input_example.json"} <= files
    assert len(mlflow.models.Model.load(local).load_input_example(local)) == 5
    assert any(f.endswith("orbit_model.py") for f in files)
    assert any(f.endswith("model.pkl") for f in files) and any(f.endswith("orbit_meta.json") for f in files)
    assert "code/orbit_helpers.py" in files
    assert "mlflow==" in (local / "requirements.txt").read_text()

    saved = json.loads((task.run_dir / "export" / "registry.json").read_text())
    assert saved == {"name": "orbit-test", "version": 1, "uri": res.uri, "aliases": res.aliases,
                     "promoted": True, "message": res.message}

    got = _subprocess(tmp_path, mlflow_uri, PREDICT_SAMPLE.format(
        name="orbit-test", sample=str(export.sample_input)))
    expected = pd.read_csv(export.sample_predictions)["prediction"].tolist()
    assert np.allclose(got, expected, atol=1e-9)


def test_register_does_not_load_the_model(clf_run, mlflow_uri):
    task, _, export = clf_run
    # unpickling this needs lightgbm, which is not installed: logging must not try
    export.model_path.write_bytes(b"clightgbm.sklearn\nLGBMClassifier\n)\x81.")
    res = register_model(task.run_dir, export)
    assert res.version == 1 and res.promoted, res.message


def test_signature_accepts_gaps_odd_types_and_extra_columns(clf_csv, tmp_path, mlflow_uri):
    df = pd.read_csv(clf_csv)
    df["n"] = np.arange(len(df)) % 7                    # an integer column
    df["cat"] = np.where(df["f0"] > 0, "up", "down")    # a text column
    df.loc[0, "f1"] = np.nan                            # a gap in the sample
    csv = tmp_path / "mixed.csv"
    df.to_csv(csv, index=False)
    task, _, export = _search_and_export(tmp_path, csv, "label", MIXED_SEARCH, CLF)
    res = register_model(task.run_dir, export)
    assert res.version == 1, res.message
    body = """
    import json
    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.pyfunc import scoring_server
    sig = mlflow.models.get_model_info("models:/orbit-test@champion").signature
    assert [c.name for c in sig.inputs] == list(pd.read_csv({sample!r}).columns)
    model = mlflow.pyfunc.load_model("models:/orbit-test@champion")
    rows = pd.read_csv({sample!r}).head(6)
    rows.loc[1, "f2"] = np.nan                          # a gap where the sample had none
    rows["n"] = rows["n"].astype(float)
    rows.loc[2, "n"] = np.nan                           # null in an integer column
    rows.loc[3, "n"] = 2.5
    rows.loc[4, "cat"] = None
    rows["label"] = "yes"                               # the target and an id may come along
    rows["orbit_id"] = range(len(rows))
    direct = model.predict(rows)
    payload = {{"dataframe_split": {{"columns": list(rows.columns),
               "data": json.loads(rows.to_json(orient="values"))}}}}
    served = model.predict(scoring_server.infer_and_parse_data(payload, sig.inputs))
    assert np.allclose(direct, served, atol=1e-6)
    try:
        model.predict(rows.drop(columns=["f3"]))
        missing = "accepted"
    except Exception as e:
        missing = str(e)
    print(json.dumps([len(direct), missing]))
    """.format(sample=str(export.sample_input))
    got = _subprocess(tmp_path, mlflow_uri, body)
    assert got[0] == 6 and "missing inputs ['f3']" in got[1]


def test_second_registration_creates_version_2(clf_run, mlflow_uri):
    import mlflow

    task, _, export = clf_run
    assert register_model(task.run_dir, export).version == 1
    second = register_model(task.run_dir, export)
    assert second.version == 2 and second.promoted     # equal score: at least as good
    client = mlflow.MlflowClient()
    assert int(client.get_model_version_by_alias("orbit-test", CHAMPION).version) == 2
    assert {int(v.version) for v in client.search_model_versions("name='orbit-test'")} == {1, 2}


def _champion_tag(key, value):
    import mlflow

    mlflow.MlflowClient().set_model_version_tag("orbit-test", "1", key, value)


@pytest.mark.parametrize("change,kwargs,promoted,words", [
    (("orbit.val_score", "0.0"), {}, True, "replaced"),                  # new is better (roc_auc)
    (("orbit.val_score", "1.1"), {}, False, "kept"),                     # old is better
    (("orbit.metric", "log_loss"), {}, True, "different metric"),
    (("orbit.data_fingerprint", "rows=1 sha256=x"), {}, True, "different data fingerprint"),
    (("orbit.val_score", "1.1"), {"force_champion": True}, True, "forced"),
    (("orbit.val_score", "0.0"), {"promote": False}, False, "switched off"),
    (("orbit.val_score", "nan-ish"), {}, False, "cannot be compared"),
])
def test_promotion_rules(clf_run, mlflow_uri, change, kwargs, promoted, words):
    import mlflow

    task, _, export = clf_run
    assert register_model(task.run_dir, export).version == 1
    _champion_tag(*change)
    res = register_model(task.run_dir, export, **kwargs)
    assert res.version == 2 and res.promoted is promoted and words in res.message, res.message
    champion = mlflow.MlflowClient().get_model_version_by_alias("orbit-test", CHAMPION).version
    assert int(champion) == (2 if promoted else 1)
    assert (CHAMPION in res.aliases) is promoted


def test_promote_false_never_takes_first_alias(clf_run, mlflow_uri):
    import mlflow

    task, _, export = clf_run
    res = register_model(task.run_dir, export, promote=False)
    assert res.version == 1 and not res.promoted and res.aliases == []
    with pytest.raises(mlflow.MlflowException):
        mlflow.MlflowClient().get_model_version_by_alias("orbit-test", CHAMPION)


def test_lower_is_better_metric_direction(mlflow_uri):
    class Client:
        def get_model_version_by_alias(self, name, alias):
            class V:
                version, tags = "3", {"orbit.metric": "rmse", "orbit.val_score": "10.0",
                                      "orbit.data_fingerprint": "f"}
            return V

    tags = {"orbit.metric": "rmse", "orbit.data_fingerprint": "f"}
    assert registry.decide_alias(Client(), "m", {**tags, "orbit.val_score": "9.0"}, True, False)[0]
    assert not registry.decide_alias(Client(), "m", {**tags, "orbit.val_score": "11.0"}, True, False)[0]


def test_model_name_argument_and_experiment_lookup(clf_run, mlflow_uri):
    task, _, export = clf_run
    assert register_model(task.run_dir, export, model_name="custom-name").name == "custom-name"
    info = json.loads((task.run_dir / "tracking.json").read_text())
    del info["experiment_name"]                      # runs from older versions have no name
    (task.run_dir / "tracking.json").write_text(json.dumps(info))
    assert register_model(task.run_dir, export).name == "orbit-test"


def test_pins_match_sandbox_requirements():
    text = (Path(__file__).parent.parent / "docker" / "sandbox-requirements.txt").read_text()
    sandbox = {line.strip().replace("xgboost-cpu", "xgboost") for line in text.splitlines()
               if line.strip() and not line.startswith("#")}
    assert sandbox == set(registry.PINNED)
    pins = registry.pip_requirements()
    assert pins[:-1] == list(registry.PINNED) and re.fullmatch(r"mlflow==\S+", pins[-1])


def test_mlflow_failure_returns_message(clf_run, mlflow_uri, monkeypatch):
    import mlflow

    task, _, export = clf_run

    def boom(*args, **kwargs):
        raise mlflow.MlflowException("server on fire")

    monkeypatch.setattr(mlflow, "MlflowClient", boom)
    res = register_model(task.run_dir, export)
    assert isinstance(res, RegistryResult) and res.version is None and not res.promoted
    assert "server on fire" in res.message
    assert json.loads((task.run_dir / "export" / "registry.json").read_text())["version"] is None


def test_missing_tracking_json_returns_message(clf_run, mlflow_uri):
    task, _, export = clf_run
    (task.run_dir / "tracking.json").unlink()
    res = register_model(task.run_dir, export)
    assert res.version is None and "registry failed" in res.message


def test_keyboard_interrupt_propagates(clf_run, mlflow_uri, monkeypatch):
    import mlflow

    task, _, export = clf_run

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mlflow, "MlflowClient", stop)
    with pytest.raises(KeyboardInterrupt):
        register_model(task.run_dir, export)


class _Fixed:
    def __init__(self, classes, proba, pred=None):
        self.classes_ = np.array(classes)
        self.proba, self.pred, self.seen = proba, pred, None

    def predict_proba(self, rows):
        self.seen = rows
        return self.proba[: len(rows)]

    def predict(self, rows):
        self.seen = rows
        return np.array(self.pred)[: len(rows)]


def _wrapper(model, **meta):
    wrapper = OrbitModel()
    wrapper.model = model
    wrapper.meta = {"target": "label", "id_column": "uid", "needs_proba": True, "positive_label": "yes",
                    **meta}
    return wrapper


def test_wrapper_ignores_target_ids_and_returns_positive_probability():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    model = _Fixed(["no", "yes"], proba)
    rows = pd.DataFrame({"a": [1.0, np.nan, 3.0], "label": ["yes"] * 3, "uid": [7, 8, 9],
                         "orbit_id": [1, 2, 3]}, index=[10, 11, 12])
    out = _wrapper(model).predict(None, rows)
    assert isinstance(out, pd.Series) and out.name == "prediction"
    assert list(out.index) == [0, 1, 2] and out.tolist() == [0.1, 0.8, 0.5]
    assert list(model.seen.columns) == ["a"] and model.seen["a"].isna().sum() == 1


def test_wrapper_matches_classes_as_normalised_text():
    model = _Fixed([0.0, 1.0], np.array([[0.7, 0.3]]))
    assert _wrapper(model, positive_label=1).predict(None, pd.DataFrame({"a": [1]})).tolist() == [0.3]
    assert _wrapper(model, positive_label="1").predict(None, pd.DataFrame({"a": [1]})).tolist() == [0.3]


def test_wrapper_label_and_regression_predictions():
    out = _wrapper(_Fixed(["a", "b"], None, ["b"]), needs_proba=False).predict(None, pd.DataFrame({"x": [1]}))
    assert out.tolist() == ["b"] and out.name == "prediction"


def test_wrapper_clips_for_nonnegative_metrics():
    model = _Fixed([], None, [-3.0, 2.5, -0.1])
    rows = pd.DataFrame({"x": [1, 2, 3]})
    assert _wrapper(model, needs_proba=False, needs_nonneg=True).predict(None, rows).tolist() == [0.0, 2.5, 0.0]
    assert _wrapper(model, needs_proba=False).predict(None, rows).tolist() == [-3.0, 2.5, -0.1]


def test_wrapper_is_free_of_orbit_imports():
    source = Path(registry.__file__).with_name("orbit_model.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+(orbit|\.)", source, re.MULTILINE)


# ---- command line, results and report ----

def _flat(text: str) -> str:
    return "".join(text.split())  # the test terminal is narrow: lines wrap mid-word


@pytest.fixture
def cli(monkeypatch):
    """The `orbit` app with the sandbox swapped for the trusted local one, as in the other CLI tests."""
    from typer.testing import CliRunner

    from orbit.cli import app

    def use(replies):
        llm = ScriptedLLM(replies)
        monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (True, "ok"))
        monkeypatch.setattr("orbit.sandbox.DockerSandbox", lambda settings=None: LocalSandbox())
        monkeypatch.setattr("orbit.cli._llm", lambda: llm)
        return CliRunner(), app, llm

    return use


def test_export_command_registers_and_results_show_the_model(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    runner, app, llm = cli([CLF])
    r = runner.invoke(app, ["export", str(task.run_dir), "--model-name", "my-model"])
    out = _flat(r.output)
    assert r.exit_code == 0, r.output
    assert "Model:exported(experiment#1,1attempt" in out
    assert "Registeredmodelmy-modelversion1,aliaschampion" in out and len(llm.prompts) == 1
    saved = json.loads((task.run_dir / "export" / "registry.json").read_text())
    assert saved["name"] == "my-model" and saved["version"] == 1

    from orbit import html_report, results

    data = results.collect(task.run_dir)
    assert data["model"]["name"] == "my-model" and data["model"]["exported"]
    assert data["model"]["aliases"] == ["champion"] and data["model"]["attempts"] == 1
    r = runner.invoke(app, ["results", str(task.run_dir)])
    assert "Model:my-modelversion1,aliaschampion" in _flat(r.output)
    assert "Registered model my-model, version 1, alias champion" in html_report.render(data)


def test_export_command_no_register_and_force_champion(clf_run, cli, mlflow_uri):
    import mlflow

    task, _, _ = clf_run
    runner, app, _ = cli([CLF, CLF])
    r = runner.invoke(app, ["export", str(task.run_dir), "--no-register"])
    assert r.exit_code == 0 and "Registered" not in r.output and "Model: exported" in r.output
    assert (task.run_dir / "export" / "model.pkl").exists()
    assert not (task.run_dir / "export" / "registry.json").exists()
    assert mlflow.MlflowClient().search_registered_models() == []
    from orbit import results
    assert results.collect(task.run_dir)["model"]["version"] is None
    r = runner.invoke(app, ["results", str(task.run_dir)])
    assert "Model:exported(experiment#1" in _flat(r.output) and "notregistered" in _flat(r.output)

    r = runner.invoke(app, ["export", str(task.run_dir), "--force-champion"])
    assert r.exit_code == 0 and "aliaschampion" in _flat(r.output)


def test_export_command_failure_exits_1(clf_run, cli, mlflow_uri):
    from conftest import BROKEN

    task, _, _ = clf_run
    runner, app, _ = cli([BROKEN] * 2)
    r = runner.invoke(app, ["export", str(task.run_dir), "--attempts", "2"])
    assert r.exit_code == 1 and "Model export failed" in r.output
    assert "noattemptproducedaverifiedmodel" in _flat(r.output)
    assert not (task.run_dir / "export" / "registry.json").exists()
    from orbit import html_report, results
    data = results.collect(task.run_dir)
    assert data["model"]["exported"] is False and len(data["model"]["notes"]) == 2
    assert "Model export failed" in html_report.render(data)
    r = runner.invoke(app, ["results", str(task.run_dir)])
    assert "Model:exportfailed" in _flat(r.output)


def test_export_command_reports_sandbox_unavailable(clf_run, cli, monkeypatch):
    task, _, _ = clf_run
    runner, app, _ = cli([])
    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (False, "no docker"))
    r = runner.invoke(app, ["export", str(task.run_dir)])
    assert r.exit_code == 1 and "Sandbox unavailable" in r.output


def test_export_step_never_raises_for_registry_or_interrupts(clf_run, monkeypatch):
    from orbit import cli as cli_mod
    from orbit.agent import Journal

    task, journal, _ = clf_run
    llm = ScriptedLLM([CLF])

    def broken(run_dir, export, **kwargs):
        return RegistryResult(name="m", message="model registry failed: down")

    monkeypatch.setattr("orbit.registry.register_model", broken)
    assert cli_mod._export_model(task, journal, llm, LocalSandbox(), 120, 1) is True

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("orbit.export.export_best", interrupted)
    assert cli_mod._export_model(task, journal, llm, LocalSandbox(), 120, 1) is False


def test_run_help_mentions_no_export():
    from typer.testing import CliRunner

    from orbit.cli import app

    r = CliRunner().invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
    assert "--no-export" in r.output and "Skip the final model export and registration" in r.output
    assert "--no-register" in CliRunner().invoke(app, ["export", "--help"]).output


def test_models_command(clf_run, cli, mlflow_uri, monkeypatch):
    task, _, export = clf_run
    runner, app, _ = cli([])
    r = runner.invoke(app, ["models"])
    assert r.exit_code == 0 and "No registered models" in r.output
    register_model(task.run_dir, export)
    register_model(task.run_dir, export, promote=False)
    r = runner.invoke(app, ["models"])
    out = r.output
    assert r.exit_code == 0 and "orbit-test" in out and "roc_auc" in out
    row = next(line for line in out.splitlines() if "orbit-test" in line)
    assert [c.strip() for c in row.strip("│ ").split("│")][:3] == ["orbit-test", "2", "1"]

    import mlflow

    def down(*args, **kwargs):
        raise mlflow.MlflowException("connection refused")

    monkeypatch.setattr(mlflow, "MlflowClient", down)
    r = runner.invoke(app, ["models"])
    assert r.exit_code == 1 and "Could not reach MLflow" in r.output


def test_model_line_in_results_and_html_with_link_and_escaping(clf_csv, tmp_path):
    from orbit import html_report, results

    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    Agent(task, ScriptedLLM([clf_reply()]), LocalSandbox(), SearchConfig(max_steps=1, n_drafts=1)).run()
    assert results.collect(task.run_dir)["model"] is None
    export = task.run_dir / "export"
    export.mkdir()
    (export / "export.json").write_text(json.dumps({"ok": True, "node_id": 1, "attempts": 2, "notes": []}))
    (export / "registry.json").write_text(json.dumps({
        "name": "a<b>&c", "version": 3, "uri": "models:/x/3", "aliases": ["champion"],
        "promoted": True, "message": "first champion"}))
    (task.run_dir / "tracking.json").write_text(json.dumps({
        "uri": "http://mlflow.example:5000", "experiment_id": "1", "run_id": "abc"}))
    data = results.collect(task.run_dir)
    assert data["model"]["url"] == "http://mlflow.example:5000/#/models/a%3Cb%3E%26c"
    page = html_report.render(data)
    assert "<b>" not in page and "a&lt;b&gt;&amp;c" in page and 'href="http://mlflow.example:5000/#/models/' in page
    (task.run_dir / "tracking.json").write_text(json.dumps({"uri": "sqlite:///x.db", "experiment_id": "1",
                                                            "run_id": "abc"}))
    assert results.collect(task.run_dir)["model"]["url"] is None
