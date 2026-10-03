"""Custom model code, class-label conventions and failure explanations of the export."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from orbit import export
from orbit.export import VERIFIER, example_hint, explain_failure
from orbit.orbit_model import _positive_column as wrapper_column

from test_export import CLF, _assert_exported, _export, _journal, _reply, _task

SCRIPT = """PLAN: A feature transformer in orbit_custom.py, then a classifier.
```python
import sys; sys.path.insert(0, "data")
import importlib, pickle
from orbit_helpers import load, Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import {model}
SRC = {src!r}
open("orbit_custom.py", "w").write(SRC)
importlib.invalidate_caches()
import orbit_custom
d = load()
X = d.train_df.drop(columns=[d.target])
model = Pipeline([("fe", orbit_custom.Interact()), ("prep", Preprocessor(drop=[])),
                  ("m", {model}({kwargs}))]).fit(X, {y})
pickle.dump(model, open("model.pkl", "wb"), protocol=4)
d.save(model.{predict}(d.test_df.drop(columns=["orbit_id"])))
{after}
```"""

MODULE = '''
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def product(a, b):
    return a * b


class Interact(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        first, second = list(X.columns[:2])
        X["interaction"] = product(X[first], X[second])
        return X
'''


def custom(model="LogisticRegression", kwargs="max_iter=500", predict="predict_proba", y="d.y_train",
           src=MODULE, after=""):
    return SCRIPT.format(model=model, kwargs=kwargs, predict=predict, y=y, src=src, after=after)


@pytest.mark.parametrize("folds", [1, 3])
def test_custom_module_travels_with_a_probability_model(clf_csv, tmp_path, folds, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label", folds=folds)
    res, llm = _export(task, [custom()])
    summary = _assert_exported(task, res, 200)
    export_dir = task.run_dir / "export"
    assert res.attempts == 1 and res.custom_module == export_dir / "orbit_custom.py"
    assert res.custom_module.read_text() == MODULE
    assert summary["custom_module"] is True and summary["custom_module_bytes"] == len(MODULE.encode())
    import hashlib
    assert summary["custom_module_sha256"] == hashlib.sha256(MODULE.encode()).hexdigest()
    assert summary["explanation"] is None
    data = res.model_path.read_bytes()
    assert b"orbit_custom" in data and b"__main__" not in data
    assert (export_dir / "verify_1" / "orbit_custom.py").read_text() == MODULE   # the fresh process had it
    assert "orbit_custom" in llm.prompts[0] and "never define" in llm.prompts[0].lower()


def test_custom_module_in_a_regression_model(reg_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, reg_csv, "price")
    res, _ = _export(task, [custom(model="Ridge", kwargs="", predict="predict")])
    summary = _assert_exported(task, res, 200)
    assert res.custom_module is not None and summary["custom_module"] is True


def test_exports_without_custom_code_record_that(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    res, _ = _export(task, [CLF])
    summary = _assert_exported(task, res, 200)
    assert res.custom_module is None and summary["custom_module"] is False
    assert "custom_module_sha256" not in summary


def test_an_unused_module_file_is_not_shipped(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    script = CLF.replace("d = load()", "open('orbit_custom.py', 'w').write('X = 1')\nd = load()")
    res, _ = _export(task, [script])
    assert res.ok and res.custom_module is None
    assert not (task.run_dir / "export" / "orbit_custom.py").exists()


def test_stale_module_of_an_earlier_export_is_removed(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    assert _export(task, [custom()])[0].custom_module
    res, _ = _export(task, [CLF])
    assert res.ok and not (task.run_dir / "export" / "orbit_custom.py").exists()


def test_module_with_forbidden_code_is_rejected_and_the_retry_says_why(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    evil = MODULE.replace("import numpy as np", "import numpy as np\nimport os\nos.getcwd()")
    res, llm = _export(task, [custom(src=evil), custom()])
    assert res.ok and res.attempts == 2
    assert "orbit_custom.py is not allowed" in llm.prompts[1] and "'os'" in llm.prompts[1]
    assert not (task.run_dir / "export" / "verify_1").exists()
    broken = CLF.replace("d = load()", "open('orbit_custom.py', 'w').write('def broken(:')\nd = load()")
    res, _ = _export(task, [broken], attempts=1)
    assert not res.ok and "not valid Python" in res.error
    assert "orbit_custom.py was not accepted" in res.explanation


def test_symlinked_module_is_rejected(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    script = custom().replace('open("orbit_custom.py", "w").write(SRC)',
                              'open("real_module.py", "w").write(SRC)\n'
                              'import os; os.symlink("real_module.py", "orbit_custom.py")')
    res, _ = _export(task, [script], attempts=1)
    assert not res.ok and "orbit_custom.py must be a regular file" in res.error


def test_pickle_that_needs_a_missing_module_gets_a_plain_message(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    script = custom(after='import os; os.remove("orbit_custom.py")')
    res, _ = _export(task, [script], attempts=1)
    assert not res.ok and "orbit_custom.py was not written next to model.pkl" in res.error
    assert "orbit_custom.py was not written" in res.explanation


MAIN_CLASS = """PLAN: Feature code defined in the script.
```python
import sys; sys.path.insert(0, "data")
import pickle
from orbit_helpers import load, Preprocessor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.assign(total=X.iloc[:, 0] + X.iloc[:, 1])
d = load()
X = d.train_df.drop(columns=[d.target])
model = Pipeline([("fe", FeatureEngineer()), ("prep", Preprocessor(drop=[])),
                  ("m", LogisticRegression(max_iter=500))]).fit(X, d.y_train)
pickle.dump(model, open("model.pkl", "wb"), protocol=4)
d.save(model.predict_proba(d.test_df.drop(columns=["orbit_id"])))
```"""


def test_objects_defined_in_the_script_are_named_and_pointed_at_orbit_custom(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    res, llm = _export(task, [MAIN_CLASS, custom()])
    assert res.ok and res.attempts == 2
    retry = llm.prompts[1]
    assert "move FeatureEngineer into orbit_custom.py" in retry and "__main__.FeatureEngineer" in retry
    res, _ = _export(task, [MAIN_CLASS], attempts=1)
    assert not res.ok
    assert "FeatureEngineer" in res.explanation and "orbit_custom.py" in res.explanation
    assert "orbit_custom.py" in example_hint(res.error) and "FeatureEngineer" in example_hint(res.error)


# ---- class labels ----

CASES = [
    (["no", "yes"], "yes", 1), (["yes", "no"], "yes", 0),
    ([0, 1], "1", 1), ([1, 0], "1", 0), ([0.0, 1.0], "1", 1), ([1.0, 0.0], "1", 0),
    ([False, True], "yes", 1), ([True, False], "yes", 0),
    (np.array([0, 1]), "yes", 1), (np.array([0.0, 1.0]), "yes", 1), (np.array([False, True]), "yes", 1),
    ([3, 7], 7, 1), ([3.0, 7.0], "7", 1), ([0, 1], 0, 0), ([0.0, 1.0], 0, 0), ([1, 2], "2", 1),
    ([False, True], "True", 1),
]
WRONG = [(["a", "b"], "yes"), ([1, 2], "yes"), ([0, 1, 2], "yes"), (["0", "1"], "yes"),
         ([0, 2], "yes"), ([2], "yes")]


def _verifier_column():
    namespace = {"np": np}
    start, stop = VERIFIER.index("def _label"), VERIFIER.index("meta = json.load")
    exec(VERIFIER[start:stop], namespace)   # trusted: orbit's own source
    return namespace["_positive_column"]


@pytest.mark.parametrize("classes,positive,column", CASES)
def test_positive_column_accepts_the_sane_conventions(classes, positive, column):
    assert wrapper_column(list(classes), positive) == column
    assert _verifier_column()(list(classes), positive) == column


@pytest.mark.parametrize("classes,positive", WRONG)
def test_wrong_classes_raise_a_message_naming_them(classes, positive):
    for func in (wrapper_column, _verifier_column()):
        with pytest.raises(ValueError) as e:
            func(list(classes), positive)
        text = str(e.value)
        assert str(list(classes)) in text and repr(positive) in text and "[0, 1]" in text


def test_wrapper_and_verifier_messages_are_identical():
    with pytest.raises(ValueError) as a:
        wrapper_column(["a", "b"], "yes")
    with pytest.raises(ValueError) as b:
        _verifier_column()(["a", "b"], "yes")
    assert str(a.value) == str(b.value)


def test_classifier_fitted_on_the_binary_target_is_accepted(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    res, _ = _export(task, [CLF.replace(".fit(X, d.y_train)", ".fit(X, d.y_train_binary)")])
    _assert_exported(task, res, 200)
    model = pickle.loads(res.model_path.read_bytes())
    assert list(model.classes_) == [0, 1]
    got = model.predict_proba(pd.read_csv(res.sample_input))[:, 1]
    assert np.allclose(got, pd.read_csv(res.sample_predictions)["prediction"])


def test_classifier_with_other_labels_fails_with_the_classes_message(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    wrong = custom(y='d.y_train.map({"no": "a", "yes": "b"})')
    res, llm = _export(task, [wrong, custom(y="d.y_train_binary")])
    assert res.ok and res.attempts == 2
    assert "classes_" in llm.prompts[1] and "['a', 'b']" in llm.prompts[1] and "'yes'" in llm.prompts[1]
    assert "d.y_train_binary" in llm.prompts[1]
    res, _ = _export(task, [wrong], attempts=1)
    assert not res.ok and "d.y_train_binary" in res.explanation
    assert "d.y_train_binary" in example_hint(res.error)


# ---- explain_failure ----

EXPLAIN = [
    ("model.pkl references things that cannot be loaded safely: global __main__.create_features is not "
     "allowed", "orbit_custom.py", "create_features"),
    ("model.pkl refers to code defined in the script itself (global __main__.A is not allowed), which "
     "no other program can load. Do not define classes", "orbit_custom.py", "A"),
    ("model.pkl references things that cannot be loaded safely: global os.system is not allowed",
     "not allowed in an exported model", None),
    ("orbit_custom.py is not allowed: line 3: import of 'os' is not allowed", "import of 'os'", None),
    ("orbit_custom.py must be a regular file, not a link or a directory.", "not accepted", None),
    ("model.pkl references things that cannot be loaded safely: global orbit_custom.X is not allowed: "
     "orbit_custom.py was not written next to model.pkl", "was not written", None),
    ("The saved model's class labels do not match the positive label: the model's classes_ are ['a'] "
     "but the positive label is 'yes'", "class labels do not match", "d.y_train_binary"),
    ("The saved model, reloaded in a fresh process, does not reproduce predictions.csv: 3 of 200 "
     "predictions differ", "outside the saved object", None),
    ("model.pkl was not written to the working directory (save the fitted object", "regular model.pkl", None),
    ("Loading the saved model in a fresh process and predicting the raw rows failed (is anything "
     "defined):\nTraceback (most recent call last):\n  File \"verify.py\", line 5\nKeyError: 'age'",
     "KeyError: 'age'", "fresh process"),
    ("The script timed out after 600 seconds.", "time limit", None),
    ("LLM call failed: ConnectionError: endpoint down", "endpoint down", None),
    ("The reply contained no ```python code block.", "no complete Python code block", None),
    ("The reply was cut off inside a code block (output token limit?).", "no complete Python", None),
    ("no working experiment to export", "no working experiment", None),
    ("Script ran but its output was rejected: ids differ", "rejected by orbit", None),
    ("The reloaded model's predictions were rejected: NaN", "rejected by orbit", None),
    ("model.pkl is 600 MB; the limit is 500 MB.", "larger than", None),
    ("Traceback (most recent call last):\n  File \"solution.py\", line 3\nValueError: boom",
     "ValueError: boom", None),
    ("something nobody anticipated", "did not produce a verified model", None),
    ("no attempt produced a verified model; last error: model.pkl references things that cannot be "
     "loaded safely: global __main__.X is not allowed", "orbit_custom.py", "X"),
]


@pytest.mark.parametrize("error,needle,other", EXPLAIN)
def test_explain_failure(error, needle, other):
    text = explain_failure(error)
    assert needle in text and "\n" not in text and len(text) < 500
    if other:
        assert other in text or other in (example_hint(error) or "")
    assert explain_failure("")


def test_example_hints_exist_for_every_retryable_failure():
    for error, _, _ in EXPLAIN:
        hint = example_hint(error)
        if "no working experiment" in error:
            assert hint is None
        else:
            assert hint and '"' not in hint and len(hint) < 200


def test_failed_export_stores_the_explanation(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    res, _ = _export(task, [MAIN_CLASS], attempts=1)
    saved = json.loads((task.run_dir / "export" / "export.json").read_text())
    assert not saved["ok"] and saved["explanation"] == res.explanation and res.explanation
    assert saved["custom_module"] is False


def test_unexpected_exceptions_are_explained_too(clf_csv, tmp_path):
    task = _task(tmp_path, clf_csv, "label")
    clf_csv.unlink()
    res, _ = _export(task, [CLF])
    assert not res.ok and res.explanation


# ---- hint and earlier failures ----

def test_hint_reaches_every_prompt_and_is_capped(clf_csv, tmp_path, orbit_helpers):
    from orbit.prompts import MAX_HINT_CHARS

    task = _task(tmp_path, clf_csv, "label")
    from conftest import LocalSandbox, ScriptedLLM

    scripted = ScriptedLLM([MAIN_CLASS, CLF])
    res = export.export_best(task, _journal(), scripted, LocalSandbox(), 120, 1, max_attempts=2,
                             hint="Use a ridge-like model " + "x" * 2000)
    assert res.ok and all("Hint from the user: Use a ridge-like model" in p for p in scripted.prompts)
    assert "x" * MAX_HINT_CHARS not in scripted.prompts[0]
    assert "Hint from the user" not in _export(task, [CLF])[1].prompts[0]


def test_retry_prompt_carries_the_previous_failure(clf_csv, tmp_path, orbit_helpers):
    task = _task(tmp_path, clf_csv, "label")
    res, _ = _export(task, [MAIN_CLASS], attempts=1)
    assert not res.ok
    res, llm = _export(task, [custom()])
    assert res.ok
    assert "An earlier export of this run failed with:" in llm.prompts[0]
    assert "FeatureEngineer" in llm.prompts[0] and "Raw error:" in llm.prompts[0]
    # a successful export leaves nothing to carry over
    _, again = _export(task, [custom()])
    assert "An earlier export" not in again.prompts[0]


def test_export_prompt_documents_the_module_and_the_class_rule(clf_csv, tmp_path):
    from orbit.prompts import export_prompt

    text = export_prompt(_task(tmp_path, clf_csv, "label"), _journal().nodes[0])
    for needle in ("orbit_custom.py", "import orbit_custom", "d.y_train_binary", "[0, 1]",
                   "no file, network, os or eval"):
        assert needle in text
