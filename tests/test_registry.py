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
from test_export_custom import MAIN_CLASS
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
    assert "code/orbit_helpers.py" in files and "artifacts/orbit_features.json" in files
    features = json.loads((local / "artifacts" / "orbit_features.json").read_text())
    assert features["features_order"] == list(pd.read_csv(export.sample_input).columns)
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


@pytest.fixture
def mixed_run(clf_csv, tmp_path, mlflow_uri):
    """A registered model with an integer column and a text column (`cat`) beside the floats."""
    df = pd.read_csv(clf_csv)
    df["n"] = np.arange(len(df)) % 7                    # an integer column
    df["cat"] = np.where(df["f0"] > 0, "up", "down")    # a text column
    df.loc[0, "f1"] = np.nan                            # a gap in the sample
    csv = tmp_path / "mixed.csv"
    df.to_csv(csv, index=False)
    task, _, export = _search_and_export(tmp_path, csv, "label", MIXED_SEARCH, CLF)
    res = register_model(task.run_dir, export)
    assert res.version == 1, res.message
    return task, export


def test_signature_accepts_gaps_odd_types_and_extra_columns(mixed_run, tmp_path, mlflow_uri):
    _, export = mixed_run
    body = """
    import json
    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.pyfunc import scoring_server
    sig = mlflow.models.get_model_info("models:/orbit-test@champion").signature
    assert [c.name for c in sig.inputs] == list(pd.read_csv({sample!r}).columns)
    assert not any(c.required for c in sig.inputs)      # the wrapper checks columns, with clear messages
    model = mlflow.pyfunc.load_model("models:/orbit-test@champion")
    rows = pd.read_csv({sample!r}).head(6)
    rows.loc[1, "f2"] = np.nan                          # a gap where the sample had none
    rows["n"] = rows["n"].astype(float)
    rows.loc[2, "n"] = np.nan                           # null in an integer column
    rows.loc[3, "n"] = 2.5
    rows.loc[4, "cat"] = None
    rows["label"] = "yes"                               # the target and an id may come along
    rows["orbit_id"] = range(len(rows))
    rows["unrelated"] = 1                               # anything else is dropped by MLflow
    direct = model.predict(rows)
    payload = {{"dataframe_split": {{"columns": list(rows.columns),
               "data": json.loads(rows.to_json(orient="values"))}}}}
    served = model.predict(scoring_server.infer_and_parse_data(payload, sig.inputs))
    assert np.allclose(direct, served, atol=1e-6)
    try:
        model.predict(rows.drop(columns=["f3"]))        # reaches the wrapper, which names the column
        missing = "accepted"
    except Exception as e:
        missing = str(e)
    print(json.dumps([len(direct), missing]))
    """.format(sample=str(export.sample_input))
    got = _subprocess(tmp_path, mlflow_uri, body)
    assert got[0] == 6 and got[1].startswith("Missing feature column(s): f3 (1 of 10). Send every feature")


SERVED_CHECKS = """
import asyncio
import json
import logging
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import scoring_server

logged = []
class Catch(logging.Handler):
    def emit(self, record):
        logged.append(record.getMessage())
logging.getLogger("orbit_model").addHandler(Catch())

model = mlflow.pyfunc.load_model("models:/orbit-test@champion")
sample = pd.read_csv({sample!r}).head(4)

def attempt(frame):
    try:
        model.predict(frame)
        return ["ok"]
    except MlflowException as e:
        return [e.error_code, str(e)]

out = {{}}
out["empty"] = attempt(sample.iloc[0:0])
out["no_columns"] = attempt(pd.DataFrame())
out["missing"] = attempt(sample.drop(columns=["f3", "cat"]))
many = sample.copy()
many["f1"] = ["x", "y", "z", "w"]
many["n"] = ["1", "2", "fifty", "4"]                 # numbers as text are fine; "fifty" is not
many["f2"] = [1.5, "nan", "", None]                  # missing, however it is written
out["non_numeric"] = attempt(many)
fine = sample.copy()
fine["n"] = ["1", "2", "3", "4"]
fine["f2"] = [1.5, "nan", "", None]
out["strings_that_are_numbers_or_gaps"] = attempt(fine)
unseen = sample.copy()
unseen["cat"] = ["up", "astronaut", "astronaut", None]
out["unseen"] = attempt(unseen)
out["unseen_logged"] = list(logged)
logged.clear()
out["seen"] = attempt(sample)
out["seen_logged"] = list(logged)
logged.clear()
unseen["cat"] = ["a", "b", "c", "d"]
many_unseen = pd.concat([unseen] * 3)
out["many_unseen"] = attempt(many_unseen)
out["many_unseen_logged"] = list(logged)

app = scoring_server.init(model)
def call(body, ctype="application/json"):
    body = body if isinstance(body, bytes) else json.dumps(body).encode()
    sent = []
    async def receive():
        return {{"type": "http.request", "body": body, "more_body": False}}
    async def send(message):
        sent.append(message)
    scope = {{"type": "http", "asgi": {{"version": "3.0"}}, "http_version": "1.1", "method": "POST",
             "path": "/invocations", "raw_path": b"/invocations", "query_string": b"", "root_path": "",
             "scheme": "http", "server": ("test", 80), "client": ("test", 1),
             "headers": [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode())]}}
    asyncio.run(app(scope, receive, send))
    data = b"".join(m.get("body", b"") for m in sent[1:])
    try:
        data = json.loads(data)
    except ValueError:
        data = data.decode()
    return [sent[0]["status"], data]

records = json.loads(sample.to_json(orient="records"))
out["http_ok"] = call({{"dataframe_records": records}})
out["http_empty"] = call({{"dataframe_records": []}})
out["http_missing"] = call({{"dataframe_records": [{{k: v for k, v in r.items() if k != "f3"}} for r in records]}})
bad = [dict(r) for r in records]
bad[1]["f0"] = "fifty"
out["http_fifty"] = call({{"dataframe_records": bad}})
odd = [dict(r) for r in records]
odd[0]["cat"] = "astronaut"
out["http_unseen"] = call({{"dataframe_records": odd}})
out["http_malformed"] = call(b'{{"dataframe_records": [')
out["http_content_type"] = call(b"a,b 1,2", ctype="text/plain")
out["http_ok_logged"] = [m for m in logged if "astronaut" in m]
print(json.dumps(out, default=str))
"""


def test_served_model_checks_its_input_without_orbit(mixed_run, tmp_path, mlflow_uri):
    _, export = mixed_run
    out = _subprocess(tmp_path, mlflow_uri, SERVED_CHECKS.format(sample=str(export.sample_input)))
    bad = "INVALID_PARAMETER_VALUE"
    assert out["empty"] == [bad, "No rows to score: send at least one record."]
    assert out["no_columns"][0] == bad
    assert out["missing"] == [bad, "Missing feature column(s): f3, cat (2 of 10). Send every feature "
                              "column; columns not needed (the target, orbit_id) are ignored."]
    assert out["non_numeric"] == [bad, "Column 'f1' must be a number; row 1 has 'x'. "
                                  "Column 'n' must be a number; row 3 has 'fifty'."]
    assert out["strings_that_are_numbers_or_gaps"] == ["ok"]
    assert out["unseen"] == ["ok"] and out["seen"] == ["ok"]
    assert out["unseen_logged"] == ["scored 2 value(s) not seen in training: cat: astronaut (2 rows)"]
    assert out["seen_logged"] == []
    assert out["many_unseen_logged"] == ["scored 12 value(s) not seen in training: "
                                         "cat: a (3 rows), b (3 rows), c (3 rows), d (3 rows)"]

    assert out["http_ok"][0] == 200 and len(out["http_ok"][1]["predictions"]) == 4
    status, body = out["http_empty"]
    assert status == 400 and body["error_code"] == bad
    assert body["message"].endswith("Error: No rows to score: send at least one record.")
    assert out["http_missing"][0] == 400 and out["http_missing"][1]["error_code"] == bad
    assert "Error: Missing feature column(s): f3 (1 of 10)." in out["http_missing"][1]["message"]
    assert out["http_fifty"][0] == 400
    assert out["http_fifty"][1]["message"].endswith("Error: Column 'f0' must be a number; row 2 has 'fifty'.")
    assert all(len(out[k][1]["message"]) < 400 for k in ("http_empty", "http_missing", "http_fifty"))
    assert out["http_unseen"][0] == 200 and out["http_ok_logged"]
    assert out["http_malformed"][0] == 400
    assert out["http_content_type"][0] in (400, 415)


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
    assert r.exit_code == 1 and "Exportfailedafter2attempts:" in _flat(r.output)
    assert "exportscriptcrashed" in _flat(r.output) and "orbitexport" in _flat(r.output)
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


# ---- custom model code ----

@pytest.fixture
def custom_run(clf_csv, tmp_path, mlflow_uri):
    from test_export_custom import custom

    return _search_and_export(tmp_path, clf_csv, "label", clf_reply(), custom())


def test_custom_module_is_registered_described_and_served_without_orbit(custom_run, tmp_path, mlflow_uri):
    import hashlib

    import mlflow

    task, _, export = custom_run
    assert export.custom_module is not None
    res = register_model(task.run_dir, export)
    assert res.version == 1, res.message
    version = mlflow.MlflowClient().get_model_version("orbit-test", "1")
    sha = hashlib.sha256(export.custom_module.read_bytes()).hexdigest()
    assert version.tags["orbit.custom_code"] == "true"
    assert "includes custom code from the experiment" in version.description and sha in version.description
    local = Path(mlflow.artifacts.download_artifacts(res.uri))
    files = {p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()}
    assert {"code/orbit_custom.py", "code/orbit_helpers.py"} <= files

    got = _subprocess(tmp_path, mlflow_uri, PREDICT_SAMPLE.format(
        name="orbit-test", sample=str(export.sample_input)))   # no orbit on sys.path in there
    expected = pd.read_csv(export.sample_predictions)["prediction"].tolist()
    assert np.allclose(got, expected, atol=1e-9)


def test_models_without_custom_code_have_no_tag_or_description_line(clf_run, mlflow_uri):
    import mlflow

    task, _, export = clf_run
    register_model(task.run_dir, export)
    version = mlflow.MlflowClient().get_model_version("orbit-test", "1")
    assert "orbit.custom_code" not in version.tags and "custom code" not in version.description
    local = Path(mlflow.artifacts.download_artifacts("models:/orbit-test/1"))
    assert not list(local.rglob("orbit_custom.py"))


# ---- failure block and --hint ----

def test_export_command_failure_block_explains_and_offers_a_retry(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    runner, app, _ = cli([MAIN_CLASS] * 2)
    r = runner.invoke(app, ["export", str(task.run_dir), "--attempts", "2"])
    out = _flat(r.output)
    assert r.exit_code == 1
    assert "Exportfailedafter2attempts:" in out and "FeatureEngineer" in out
    assert "orbit_custom.py" in out
    assert _flat(str(task.run_dir / "export" / "export.json")) in out
    assert _flat(str(task.run_dir / "export" / "attempt_2")) in out
    assert "orbitexport" in out and _flat(f'--hint "Move FeatureEngineer into orbit_custom.py') in out
    first, second = r.output.index("Export failed after"), r.output.index("Details:")
    assert first < second < r.output.index("To retry with guidance")


def test_failure_block_escapes_model_text(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    script = (
        "PLAN: x\n```python\nclass Fe[/red][bold]:\n    pass\n```")
    runner, app, _ = cli([script] * 2)
    r = runner.invoke(app, ["export", str(task.run_dir), "--attempts", "1"])
    assert r.exit_code == 1 and "Exportfailedafter1attempt" in _flat(r.output)
    assert "MarkupError" not in r.output and "Traceback" not in r.output.split("Export failed")[0]


def test_hint_option_reaches_the_prompt_and_is_capped(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    runner, app, llm = cli([CLF, CLF])
    r = runner.invoke(app, ["export", str(task.run_dir), "--no-register", "--hint", "Prefer a plain linear model"])
    assert r.exit_code == 0, r.output
    assert "Hint from the user: Prefer a plain linear model" in llm.prompts[0]
    r = runner.invoke(app, ["export", str(task.run_dir), "--hint", "x" * 1001])
    assert r.exit_code != 0 and "1000" in _flat(r.output) and len(llm.prompts) == 1
    r = runner.invoke(app, ["export", str(task.run_dir), "--no-register", "--hint", "x" * 1000])
    assert r.exit_code == 0 and "x" * 1000 in llm.prompts[1]
    assert "--hint" in runner.invoke(app, ["export", "--help"]).output


def test_reexport_prompt_mentions_the_earlier_failure(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    runner, app, llm = cli([MAIN_CLASS, CLF])
    assert runner.invoke(app, ["export", str(task.run_dir), "--attempts", "1"]).exit_code == 1
    r = runner.invoke(app, ["export", str(task.run_dir), "--no-register"])
    assert r.exit_code == 0
    assert "An earlier export of this run failed with:" in llm.prompts[1]
    assert "FeatureEngineer" in llm.prompts[1]


def test_run_prints_the_failure_block_and_still_exits_zero(clf_csv, tmp_path, monkeypatch, mlflow_uri):
    from typer.testing import CliRunner

    from conftest import BROKEN
    from orbit.cli import app

    class ByPrompt(ScriptedLLM):
        def complete(self, system, user):
            self.replies = [BROKEN if "## Export task" in user else clf_reply()]
            return super().complete(system, user)

    llm = ByPrompt([])
    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (True, "ok"))
    monkeypatch.setattr("orbit.sandbox.DockerSandbox", lambda settings=None: LocalSandbox())
    monkeypatch.setattr("orbit.cli._llm", lambda: llm)
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["run", str(clf_csv), "-t", "label", "--steps", "1", "--drafts", "1",
                                 "--tracker", "mlflow", "--folds", "1", "--out", str(tmp_path / "runs")])
    out = _flat(r.output)
    assert r.exit_code == 0, r.output
    assert "Exportfailedafter3attempts:" in out and "orbitexport" in out and "--hint" in out
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "export" / "export.json").exists() and (run_dir / "journal.json").exists()


# ---- the features file ----

def _input_dir(tmp_path, frame, **meta):
    folder = tmp_path / "input"
    folder.mkdir()
    frame.to_csv(folder / "train.csv", index=False)
    (folder / "orbit_meta.json").write_text(json.dumps({"target": "y", "id_column": "uid", **meta}))
    return folder


def test_features_file_types_categories_and_exclusions(tmp_path):
    frame = pd.DataFrame({
        "uid": range(6), "y": [0, 1, 0, 1, 0, 1], "orbit_id": range(6),
        "age": [30, 41, None, 25, 52, 33],                    # numbers, one missing
        "score": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "job": ["nurse", "clerk", None, "nurse", "10", "2"],  # text, with a gap and numeric-looking values
        "flag": [True, False, True, True, False, True],       # booleans count as text
        "zip": ["00123", "00456", "00123", "00789", "00456", "00123"]})
    features = registry.build_features(_input_dir(tmp_path, frame))
    assert features["features_order"] == ["age", "score", "job", "flag", "zip"]
    assert features["columns"] == {
        "age": {"type": "number"}, "score": {"type": "number"},
        "job": {"type": "text", "categories": ["10", "2", "clerk", "nurse"]},
        "flag": {"type": "text", "categories": ["False", "True"]},
        "zip": {"type": "number"}}      # pandas reads 00123 as a number, as the exported script does
    assert "nurse" in json.dumps(features) and "30" not in json.dumps(features["columns"]["job"])


def test_features_file_truncates_long_category_lists(tmp_path):
    frame = pd.DataFrame({"y": 0, "few": [f"v{i % 500}" for i in range(1000)],
                          "many": [f"v{i}" for i in range(1000)]})
    features = registry.build_features(_input_dir(tmp_path, frame, id_column=None))
    assert len(features["columns"]["few"]["categories"]) == 500          # at the limit: kept, sorted
    assert features["columns"]["few"]["categories"] == sorted(features["columns"]["few"]["categories"])
    assert features["columns"]["many"] == {"type": "text", "categories": None, "truncated": True}


# ---- the wrapper's checks, without MLflow ----

def _checked(rows, features=None, **meta):
    wrapper = _wrapper(_Fixed(["no", "yes"], np.tile([[0.4, 0.6]], (50, 1))), **meta)
    wrapper.features = features
    return wrapper, rows


WRAPPER_FEATURES = {"features_order": ["age", "job"],
                    "columns": {"age": {"type": "number"},
                                "job": {"type": "text", "categories": ["clerk", "nurse"]}}}


def _message(wrapper, rows):
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException) as e:
        wrapper.predict(None, rows)
    assert e.value.error_code == "INVALID_PARAMETER_VALUE" and e.value.get_http_status_code() == 400
    return e.value.message


def test_wrapper_rejects_empty_input_and_missing_columns():
    wrapper, _ = _checked(None, WRAPPER_FEATURES)
    assert _message(wrapper, pd.DataFrame()) == "No rows to score: send at least one record."
    assert _message(wrapper, pd.DataFrame({"age": [], "job": []})) == "No rows to score: send at least one record."
    assert _message(wrapper, pd.DataFrame({"age": [1], "label": ["yes"], "orbit_id": [1]})) == (
        "Missing feature column(s): job (1 of 2). Send every feature column; columns not needed "
        "(the target, orbit_id) are ignored.")
    wide = {"features_order": [f"c{i}" for i in range(14)],
            "columns": {f"c{i}": {"type": "number"} for i in range(14)}}
    wrapper, _ = _checked(None, wide)
    text = _message(wrapper, pd.DataFrame({"c0": [1]}))
    assert "c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 and 3 more (13 of 14)." in text


def test_wrapper_rejects_text_in_numeric_columns_but_not_gaps():
    wrapper, _ = _checked(None, WRAPPER_FEATURES)
    rows = pd.DataFrame({"age": [30, "fifty", "x"], "job": ["clerk"] * 3})
    assert _message(wrapper, rows) == "Column 'age' must be a number; row 2 has 'fifty'."
    gaps = pd.DataFrame({"age": ["31", "nan", "NaN", "", None, 2.5, np.nan], "job": ["clerk"] * 7})
    assert wrapper.predict(None, gaps).tolist() == [0.6] * 7
    three = {"features_order": list("abcd"), "columns": {c: {"type": "number"} for c in "abcd"}}
    wrapper, _ = _checked(None, three)
    text = _message(wrapper, pd.DataFrame({c: ["q" * 60] for c in "abcd"}))
    assert text.count("must be a number") == 3 and "1 more column(s)" in text and "q" * 50 not in text
    nested = pd.DataFrame({"age": [[1, 2], 3], "job": ["clerk", "nurse"]})
    assert "Column 'age' must be a number; row 1 has" in _message(_checked(None, WRAPPER_FEATURES)[0], nested)


def test_wrapper_logs_unseen_categories_once_and_scores_them(caplog):
    wrapper, _ = _checked(None, WRAPPER_FEATURES)
    rows = pd.DataFrame({"age": [1, 2, 3, 4], "job": ["astronaut", "clerk", "astronaut", None]})
    with caplog.at_level("WARNING", logger="orbit_model"):
        assert wrapper.predict(None, rows).tolist() == [0.6] * 4
    assert [r.getMessage() for r in caplog.records] == [
        "scored 2 value(s) not seen in training: job: astronaut (2 rows)"]
    caplog.clear()
    with caplog.at_level("WARNING", logger="orbit_model"):
        wrapper.predict(None, pd.DataFrame({"age": [1], "job": ["nurse"]}))
    assert caplog.records == []


def test_wrapper_unseen_log_is_capped_and_skips_unlisted_categories(caplog):
    cols = [f"t{i}" for i in range(7)]
    features = {"features_order": cols, "columns": {c: {"type": "text", "categories": ["ok"]} for c in cols}}
    features["columns"]["t6"] = {"type": "text", "categories": None, "truncated": True}
    wrapper, _ = _checked(None, features)
    rows = pd.DataFrame({c: [f"v{i}" for i in range(8)] for c in cols})
    with caplog.at_level("WARNING", logger="orbit_model"):
        wrapper.predict(None, rows)
    (text,) = [r.getMessage() for r in caplog.records]
    assert text.startswith("scored 48 value(s) not seen in training: t0: v0 (1 row), v1 (1 row), v2 (1 row), "
                           "v3 (1 row), v4 (1 row), ...; t1:")
    assert "t4:" in text and "t5:" not in text and "t6:" not in text and text.endswith("; 1 more column(s)")


def test_wrapper_without_a_features_file_behaves_as_before(tmp_path):
    import pickle
    from types import SimpleNamespace

    model = _Fixed(["no", "yes"], np.tile([[0.4, 0.6]], (5, 1)))
    (tmp_path / "model.pkl").write_bytes(pickle.dumps(model))
    (tmp_path / "meta.json").write_text(json.dumps({"target": "label", "needs_proba": True,
                                                    "positive_label": "yes"}))
    wrapper = OrbitModel()
    wrapper.load_context(SimpleNamespace(artifacts={"model": str(tmp_path / "model.pkl"),
                                                    "meta": str(tmp_path / "meta.json")}))
    assert wrapper.features is None
    rows = pd.DataFrame({"anything": ["x", 1, None], "label": ["yes"] * 3})
    assert wrapper.predict(None, rows).tolist() == [0.6, 0.6, 0.6]
    assert wrapper.predict(None, pd.DataFrame({"a": [1]})).tolist() == [0.6]
    wrapper.features = None
    wrapper.__dict__.pop("features")      # an object that never had the attribute at all
    assert wrapper.predict(None, rows).tolist() == [0.6] * 3


def test_wrapper_loads_a_features_file(tmp_path):
    import pickle
    from types import SimpleNamespace

    (tmp_path / "model.pkl").write_bytes(pickle.dumps(_Fixed(["no", "yes"], np.zeros((1, 2)))))
    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "features.json").write_text(json.dumps(WRAPPER_FEATURES))
    wrapper = OrbitModel()
    wrapper.load_context(SimpleNamespace(artifacts={
        "model": str(tmp_path / "model.pkl"), "meta": str(tmp_path / "meta.json"),
        "features": str(tmp_path / "features.json")}))
    assert wrapper.features == WRAPPER_FEATURES


# ---- quiet registration ----

def test_registration_output_has_no_mlflow_chatter(clf_run, tmp_path, mlflow_uri):
    task, _, _ = clf_run
    script = tmp_path / "register.py"
    script.write_text("import sys\nfrom pathlib import Path\nfrom orbit.export import load_verified_export\n"
                      "from orbit.registry import register_model\nrun = Path(sys.argv[1])\n"
                      "res = register_model(run, load_verified_export(run))\nprint('RESULT', res.version)\n")
    # CLAUDECODE makes MLflow print a hint when it is imported, unless orbit has switched that off
    env = {"MLFLOW_TRACKING_URI": mlflow_uri, "CLAUDECODE": "1", "HOME": str(tmp_path),
           "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, str(script), str(task.run_dir)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "RESULT 1"
    noise = done.stderr
    for text in ("hint", "type hint", "Successfully registered", "Created version", "Warning", "WARNING"):
        assert text.lower() not in noise.lower(), noise
    assert noise.strip() == ""


# ---- export --reuse ----

def _reuse(cli, run_dir, *args):
    runner, app, llm = cli([])
    r = runner.invoke(app, ["export", str(run_dir), "--reuse", *args])
    assert llm.prompts == []        # no LLM call, whatever happened
    return r


def test_reuse_registers_a_new_version_without_the_llm_or_docker(clf_run, cli, mlflow_uri, monkeypatch):
    import mlflow

    task, _, _ = clf_run
    runner, app, _ = cli([CLF])
    first = runner.invoke(app, ["export", str(task.run_dir)])
    assert first.exit_code == 0, first.output
    summary = (task.run_dir / "export" / "export.json").read_bytes()
    model = (task.run_dir / "export" / "model.pkl").read_bytes()
    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (False, "no docker"))
    monkeypatch.setattr("orbit.cli._llm", lambda: pytest.fail("the LLM must not be needed"))
    r = _reuse(cli, task.run_dir, "--model-name", "again")
    out = _flat(r.output)
    assert r.exit_code == 0, r.output
    assert "Model:exported(experiment#1,1attempt," in out and "Registeredmodelagainversion1,aliaschampion" in out
    r = _reuse(cli, task.run_dir)
    assert r.exit_code == 0 and "Registeredmodelorbit-testversion2" in _flat(r.output)
    assert (task.run_dir / "export" / "export.json").read_bytes() == summary
    assert (task.run_dir / "export" / "model.pkl").read_bytes() == model
    client = mlflow.MlflowClient()
    assert {m.name for m in client.search_registered_models()} == {"orbit-test", "again"}
    local = Path(mlflow.artifacts.download_artifacts("models:/again/1"))
    assert (local / "artifacts" / "orbit_features.json").is_file()
    assert json.loads((task.run_dir / "export" / "registry.json").read_text())["version"] == 2


def test_reuse_refuses_a_changed_or_missing_export(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    export_dir = task.run_dir / "export"
    model = export_dir / "model.pkl"
    original = model.read_bytes()
    model.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))          # same size, other content
    r = _reuse(cli, task.run_dir)
    assert r.exit_code == 1 and "model.pklin" in _flat(r.output) and "nolongermatches" in _flat(r.output)
    model.write_bytes(original + b"x")
    assert _reuse(cli, task.run_dir).exit_code == 1
    model.write_bytes(original)
    (export_dir / "sample_predictions.csv").unlink()
    r = _reuse(cli, task.run_dir)
    assert r.exit_code == 1 and "no verified export in" in r.output and "run without --reuse" in " ".join(r.output.split())
    assert not (export_dir / "registry.json").exists()
    import mlflow
    assert mlflow.MlflowClient().search_registered_models() == []


def test_reuse_needs_an_export_that_worked(clf_run, cli, mlflow_uri, tmp_path):
    task, _, _ = clf_run
    export_json = task.run_dir / "export" / "export.json"
    info = json.loads(export_json.read_text())
    export_json.write_text(json.dumps({**info, "ok": False}))
    assert "noverifiedexportin" in _flat(_reuse(cli, task.run_dir).output)
    export_json.write_text("not json")
    assert _reuse(cli, task.run_dir).exit_code == 1
    export_json.unlink()
    assert _reuse(cli, task.run_dir).exit_code == 1
    other = tmp_path / "plain-run"
    other.mkdir()
    r = _reuse(cli, other)
    assert r.exit_code == 1 and "no verified export in" in r.output
    export_json.write_text(json.dumps(info))
    (task.run_dir / "export" / "input" / "train.csv").unlink()
    r = _reuse(cli, task.run_dir)
    assert r.exit_code == 1 and "inputismissing;runwithout--reuse" in _flat(r.output)


def test_reuse_conflicts_with_no_register_and_hint(clf_run, cli, mlflow_uri):
    task, _, _ = clf_run
    assert _reuse(cli, task.run_dir, "--no-register").exit_code == 2
    assert _reuse(cli, task.run_dir, "--hint", "x").exit_code == 2


def test_reuse_with_custom_module(custom_run, cli, mlflow_uri):
    import mlflow

    task, _, export = custom_run
    r = _reuse(cli, task.run_dir)
    assert r.exit_code == 0 and "Registeredmodelorbit-testversion1" in _flat(r.output)
    assert mlflow.MlflowClient().get_model_version("orbit-test", "1").tags["orbit.custom_code"] == "true"
    local = Path(mlflow.artifacts.download_artifacts("models:/orbit-test/1"))
    assert (local / "code" / "orbit_custom.py").is_file()
    export.custom_module.write_text(export.custom_module.read_text() + "\n# edited\n")
    r = _reuse(cli, task.run_dir)
    assert r.exit_code == 1 and "orbit_custom.pyin" in _flat(r.output) and "nolongermatches" in _flat(r.output)
    export.custom_module.unlink()
    assert "noverifiedexportin" in _flat(_reuse(cli, task.run_dir).output)
