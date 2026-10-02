import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from orbit import export
from orbit.agent import Journal, Node
from orbit.export import ExportResult, export_best
from orbit.task import TaskSpec, prepare

from conftest import BROKEN, LocalSandbox, ScriptedLLM

SCRIPT = """PLAN: Refit on everything inside one pipeline.
```python
import sys; sys.path.insert(0, "data")
import pickle
from orbit_helpers import load, Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import {model}
d = load()
X = d.train_df.drop(columns=[d.target])
model = Pipeline([("prep", Preprocessor(drop=[])), ("m", {model}({kwargs}))]).fit(X, d.y_train)
pickle.dump(model, open("model.pkl", "wb"), protocol=4)
d.save(model.{predict}(d.test_df.drop(columns=["orbit_id"])))
```"""

CLF = SCRIPT.format(model="LogisticRegression", kwargs="max_iter=500", predict="predict_proba")
LABELS = SCRIPT.format(model="LogisticRegression", kwargs="max_iter=500", predict="predict")
REG = SCRIPT.format(model="Ridge", kwargs="", predict="predict")


def _reply(body: str) -> str:
    return f"PLAN: x\n```python\n{body}\n```"


def _task(tmp_path, csv, target, **spec):
    spec.setdefault("folds", 1)
    return prepare(TaskSpec(str(csv), target, **spec), tmp_path / "run")


def _journal(code="print('best')", metric=0.9) -> Journal:
    node = Node(id=1, stage="draft", parent=None, plan="plan text", code=code, metric=metric)
    return Journal(nodes=[node], best_id=1)


def _export(task, replies, journal=None, attempts=3, llm=None):
    llm = llm or ScriptedLLM(replies)
    res = export_best(task, journal or _journal(), llm, LocalSandbox(), exec_timeout=120, cpus=1,
                      max_attempts=attempts)
    return res, llm


@pytest.fixture
def multiclass_csv(tmp_path):
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=3,
                               random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["kind"] = pd.Series(y).map({0: "a", 1: "b", 2: "c"})
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


def _assert_exported(task, res, n_rows):
    export_dir = task.run_dir / "export"
    assert res.ok and res.error is None and res.node_id == 1
    assert res.model_path == export_dir / "model.pkl" and res.model_path.is_file()
    sample = pd.read_csv(res.sample_input)
    preds = pd.read_csv(res.sample_predictions)
    assert len(sample) == len(preds) == n_rows and "orbit_id" not in sample.columns
    assert list(preds.columns) == ["orbit_id", "prediction"]
    # the accepted file is the verified one: it predicts the sample from raw rows
    summary = json.loads((export_dir / "export.json").read_text())
    assert summary["ok"] and summary["node_id"] == 1 and summary["model_bytes"] > 0
    return summary


@pytest.mark.parametrize("folds", [1, 3])
def test_export_probability_task(clf_csv, tmp_path, folds, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label", folds=folds)
    res, llm = _export(task, [CLF])
    summary = _assert_exported(task, res, 200)
    assert res.attempts == 1 and len(llm.prompts) == 1
    assert "run 3 times" not in llm.prompts[0] and "best experiment (#1" in llm.prompts[0].lower()
    assert set(summary) >= {"ok", "node_id", "attempts", "error", "notes", "model_sha256"}
    assert "import" not in json.dumps(summary)               # no code in the summary
    model = pickle.loads(res.model_path.read_bytes())
    proba = model.predict_proba(pd.read_csv(res.sample_input))
    pos = list(model.classes_).index("yes")
    assert np.allclose(proba[:, pos], pd.read_csv(res.sample_predictions)["prediction"])
    assert (task.run_dir / "export" / "verify_1" / "verify.csv").exists()


def test_export_regression_task(reg_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, reg_csv, "price")
    res, _ = _export(task, [REG])
    _assert_exported(task, res, 200)


def test_export_multiclass_label_metric(multiclass_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, multiclass_csv, "kind", folds=3)
    assert task.metric_name == "accuracy"
    res, _ = _export(task, [LABELS])
    _assert_exported(task, res, 200)
    assert set(pd.read_csv(res.sample_predictions)["prediction"]) <= {"a", "b", "c"}


def test_retry_after_missing_model_shows_error(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    no_pickle = CLF.replace('pickle.dump(model, open("model.pkl", "wb"), protocol=4)', "pass")
    res, llm = _export(task, [no_pickle, CLF])
    assert res.ok and res.attempts == 2
    assert "model.pkl was not written" in llm.prompts[1]
    assert "pass" in llm.prompts[1] and "Previous export attempt" in llm.prompts[1]
    assert "Previous export attempt" not in llm.prompts[0]
    assert any(n.startswith("attempt 1:") for n in res.notes)


def test_wrong_columns_fail_verification_then_succeed(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    wrong = _reply("""import sys; sys.path.insert(0, "data")
import pickle
from orbit_helpers import load
from sklearn.linear_model import LogisticRegression
d = load()
cols = [c for c in d.X_train.columns if c != "f0"]
model = LogisticRegression(max_iter=300).fit(d.X_train[cols], d.y_train)
pickle.dump(model, open("model.pkl", "wb"), protocol=4)
d.save(model.predict_proba(d.X_test[cols]))""")
    res, llm = _export(task, [wrong, CLF])
    assert res.ok and res.attempts == 2
    assert "fresh process" in llm.prompts[1] and "f0" in llm.prompts[1]


def test_pickle_with_disallowed_global_is_rejected_before_loading(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    # unpickling this would call print; the scan refuses it, so no load ever happens
    evil = _reply("""import sys; sys.path.insert(0, "data")
import pickle
import numpy as np
from orbit_helpers import load
d = load()
class P:
    def __reduce__(self):
        return (print, ("loaded",))
pickle.dump(P(), open("model.pkl", "wb"), protocol=4)
d.save(np.full(len(d.test_df), 0.5))""")
    res, llm = _export(task, [evil, evil, CLF])
    assert res.ok and res.attempts == 3
    assert "cannot be loaded safely" in llm.prompts[1] and "builtins.print" in llm.prompts[1]
    assert not (task.run_dir / "export" / "verify_1").exists()    # never reached verification
    res, _ = _export(task, [evil], attempts=1)
    assert not res.ok and "builtins.print" in res.error


def test_verification_mismatch_is_rejected(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    lying = CLF.replace('d.save(model.predict_proba(d.test_df.drop(columns=["orbit_id"])))',
                        'd.save(model.predict_proba(d.test_df.drop(columns=["orbit_id"]))[:, ::-1])')
    res, llm = _export(task, [lying, CLF])
    assert res.ok and res.attempts == 2
    assert "does not reproduce" in llm.prompts[1]
    res, _ = _export(task, [lying], attempts=1)
    assert not res.ok and "does not reproduce" in res.error
    assert not (task.run_dir / "export" / "model.pkl").exists()


def test_symlinked_or_oversized_model_is_rejected(clf_csv, tmp_path, monkeypatch, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    link = CLF.replace('pickle.dump(model, open("model.pkl", "wb"), protocol=4)',
                       'pickle.dump(model, open("real.pkl", "wb"), protocol=4)\n'
                       'import os; os.symlink("real.pkl", "model.pkl")')
    res, _ = _export(task, [link], attempts=1)
    assert not res.ok and "model.pkl was not written" in res.error
    monkeypatch.setattr(export, "MAX_MODEL_BYTES", 100)
    res, _ = _export(task, [CLF], attempts=1)
    assert not res.ok and "the limit is" in res.error


def test_all_attempts_failing_does_not_raise(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    res, llm = _export(task, [BROKEN, "no code at all", BROKEN])
    assert isinstance(res, ExportResult) and not res.ok and res.attempts == 3
    assert res.model_path is None and res.sample_input is None
    assert "no attempt produced a verified model" in res.error
    assert "no ```python code block" in llm.prompts[2] or "could not convert" in llm.prompts[2]
    summary = json.loads((task.run_dir / "export" / "export.json").read_text())
    assert not summary["ok"] and summary["attempts"] == 3 and len(summary["notes"]) == 3
    assert not (task.run_dir / "export" / "model.pkl").exists()


def test_stale_results_are_cleared(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    assert _export(task, [CLF])[0].ok
    res, _ = _export(task, [BROKEN], attempts=1)
    assert not res.ok and not (task.run_dir / "export" / "model.pkl").exists()


def test_no_working_experiment(clf_csv, tmp_path):
    task = _task(tmp_path, clf_csv, "label")
    res, llm = _export(task, [], journal=Journal())
    assert not res.ok and "no working experiment" in res.error and not llm.prompts
    failed = Journal(nodes=[Node(id=1, stage="draft", parent=None, code="x")], best_id=None)
    assert not _export(task, [], journal=failed)[0].ok


def test_llm_and_input_errors_are_reported_not_raised(clf_csv, tmp_path):
    task = _task(tmp_path, clf_csv, "label")

    class Down:
        def complete(self, system, user):
            raise ConnectionError("endpoint down")

    res, _ = _export(task, [], llm=Down(), attempts=2)
    assert not res.ok and "endpoint down" in res.error and res.attempts == 2
    clf_csv.unlink()                                        # the source file is gone
    res, _ = _export(task, [CLF])
    assert not res.ok and res.error


def test_keyboard_interrupt_propagates(clf_csv, tmp_path):
    task = _task(tmp_path, clf_csv, "label")

    class Interrupted:
        def complete(self, system, user):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _export(task, [], llm=Interrupted())


def test_hidden_directory_is_never_given_to_the_sandbox(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    seen = []

    class Spy(LocalSandbox):
        def run(self, code, workdir, data_dir, timeout):
            seen.append(data_dir)
            return super().run(code, workdir, data_dir, timeout)

    llm = ScriptedLLM([CLF])
    res = export_best(task, _journal(), llm, Spy(), 120, 1)
    assert res.ok and seen == [task.run_dir / "export" / "input"] * 2    # script, then verifier
