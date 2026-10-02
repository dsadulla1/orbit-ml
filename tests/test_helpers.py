import json
import math

import numpy as np
import pandas as pd
import pytest

from orbit import helpers
from orbit.agent import Agent, SearchConfig
from orbit.task import HELPER_FILE, META_FILE, TaskSpec, prepare

from conftest import LocalSandbox, ScriptedLLM


def _task(tmp_path, df, target, **spec):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    spec.setdefault("folds", 1)
    return prepare(TaskSpec(str(path), target, **spec), tmp_path / "run")


def _mixed(n=200):
    rng = np.random.default_rng(0)
    countries = np.array(["UK", "FR", "DE", "ES", "IT"])
    df = pd.DataFrame({
        "num": rng.normal(size=n),
        "country": countries[rng.integers(0, 5, n)],
        "flag": rng.integers(0, 2, n).astype(bool),
        "txt_nan": np.where(rng.random(n) < 0.2, None, "a"),
        "ident": np.arange(n),
        "y": rng.normal(size=n),
    })
    df.loc[::7, "num"] = np.nan
    return df


def test_text_codes_are_shared_and_cover_test_only_categories(tmp_path):
    df = _mixed()
    df["country"] = "UK"
    df.loc[df.index[-3:], "country"] = "RARE"
    task = _task(tmp_path, df, "y", time_col=None)
    d = helpers.load(task.input_dir)
    assert "country" in d.cat_columns
    train_vals = set(d.train_df["country"]) | set(d.test_df["country"])
    mapping = {}
    for frame_raw, frame in ((d.train_df, d.X_train), (d.test_df, d.X_test)):
        for raw, code in zip(frame_raw["country"], frame["country"]):
            assert mapping.setdefault(raw, code) == code
    assert set(mapping) == train_vals
    assert "RARE" in mapping
    assert not d.X_test["country"].isna().any()


def test_codes_are_sorted_string_ranks(tmp_path):
    train = pd.DataFrame({"c": ["b", "a", "c"], "y": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"orbit_id": [1, 2], "c": ["d", "a"]})
    d = _data_from_frames(tmp_path, train, test)
    assert d.X_train["c"].tolist() == [1, 0, 2] and d.X_test["c"].tolist() == [3, 0]
    assert d.X_train["c"].dtype == np.int64


def _data_from_frames(tmp_path, train, test, **meta):
    folder = tmp_path / "in"
    folder.mkdir(exist_ok=True)
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    base = {"target": "y", "kind": "regression", "metric": "rmse", "needs_proba": False,
            "needs_nonneg": False, "positive_label": None, "classes": [], "id_column": None,
            "time_col": None, "n_folds": 1, "extra_names": []}
    base.update(meta)
    (folder / META_FILE).write_text(json.dumps(base))
    return helpers.load(folder)


def test_nan_handling_filled_frames_and_dtypes(tmp_path):
    df = _mixed()
    task = _task(tmp_path, df, "y", id_column="ident")
    d = helpers.load(task.input_dir)
    assert "ident" not in d.X_train.columns and "orbit_id" not in d.X_test.columns
    assert "y" not in d.X_train.columns
    assert d.X_train["num"].isna().any()
    assert list(d.X_train.columns) == list(d.X_test.columns)
    for frame in (d.X_train, d.X_test, d.X_train_filled, d.X_test_filled):
        assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)
    assert not d.X_train_filled.isna().any().any() and not d.X_test_filled.isna().any().any()
    med = d.X_train["num"].median()
    nan_rows = d.X_test["num"].isna()
    assert (d.X_test_filled.loc[nan_rows, "num"] == med).all()
    # missing text stays NaN rather than getting a code
    assert d.X_train["txt_nan"].isna().sum() == d.train_df["txt_nan"].isna().sum() > 0
    assert set(d.cat_columns) == {"country", "flag", "txt_nan"}
    assert (d.test_ids == d.test_df["orbit_id"]).all()


def test_all_missing_column_is_filled_with_zero(tmp_path):
    train = pd.DataFrame({"a": [np.nan] * 4, "b": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"orbit_id": [1, 2], "a": [np.nan, np.nan], "b": [1.0, 2.0]})
    d = _data_from_frames(tmp_path, train, test)
    assert d.X_train_filled["a"].eq(0).all() and d.X_test_filled["a"].eq(0).all()


def test_string_dtype_and_bool_columns(tmp_path):
    train = pd.DataFrame({"s": pd.array(["x", "y", None, "x"], dtype="string"),
                          "b": [True, False, True, False], "y": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"orbit_id": [1, 2], "s": pd.array(["z", "x"], dtype="string"),
                         "b": [False, True]})
    d = _data_from_frames(tmp_path, train, test)
    assert pd.isna(d.X_train["s"].iloc[2]) and d.X_train["s"].iloc[0] == d.X_test["s"].iloc[1]
    assert d.X_test["s"].iloc[0] == 2
    assert d.X_train["b"].tolist() == [1, 0, 1, 0] and d.X_test["b"].tolist() == [0, 1]


def _clf_frames(labels, n=40):
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"f": rng.normal(size=n), "y": [labels[i % len(labels)] for i in range(n)]})
    test = pd.DataFrame({"orbit_id": range(1, 6), "f": rng.normal(size=5)})
    return train, test


def test_binary_labels_and_y_train_binary(tmp_path):
    train, test = _clf_frames(["no", "yes", "no"])
    d = _data_from_frames(tmp_path, train, test, kind="classification", metric="roc_auc",
                          needs_proba=True, positive_label="yes", classes=["no", "yes"])
    assert d.classes == ["no", "yes"] and d.positive_label == "yes"
    assert d.y_train_binary.tolist() == [int(v == "yes") for v in d.y_train]
    assert d.y_train.tolist() == train["y"].tolist()


def test_multiclass_has_no_binary_target(tmp_path):
    train, test = _clf_frames(["a", "b", "c"])
    d = _data_from_frames(tmp_path, train, test, kind="classification", metric="accuracy",
                          classes=["a", "b", "c"])
    assert d.y_train_binary is None and d.positive_label is None


def _proba(tmp_path):
    train, test = _clf_frames(["no", "yes", "no"])
    return _data_from_frames(tmp_path, train, test, kind="classification", metric="roc_auc",
                             needs_proba=True, positive_label="yes", classes=["no", "yes"])


def _read(path):
    return pd.read_csv(path)


def test_save_probabilities_and_2d(tmp_path, capsys):
    d = _proba(tmp_path)
    out = tmp_path / "p.csv"
    d.save(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), out)
    got = _read(out)
    assert list(got.columns) == ["orbit_id", "prediction"] and got["orbit_id"].tolist() == [1, 2, 3, 4, 5]
    assert "5 predictions" in capsys.readouterr().out
    two = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    d.save(two, out)
    assert _read(out)["prediction"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_save_2d_numeric_class_order(tmp_path):
    train, test = _clf_frames([9, 10, 9])
    d = _data_from_frames(tmp_path, train, test, kind="classification", metric="roc_auc",
                          needs_proba=True, positive_label="10", classes=["10", "9"])
    two = np.tile([0.7, 0.3], (5, 1))     # predict_proba order is 9, 10
    d.save(two, tmp_path / "p.csv")
    assert _read(tmp_path / "p.csv")["prediction"].eq(0.3).all()


@pytest.mark.parametrize("bad,match", [
    (np.array([0.1, 0.2]), "2 values but test.csv has 5"),
    (np.array([0.1, 0.2, np.nan, 0.4, 0.5]), "NaN or inf"),
    (np.array([0.1, 0.2, 0.3, 1.4, 0.5]), r"in \[0, 1\]"),
    (np.array([0.1, -0.2, 0.3, 0.4, 0.5]), r"in \[0, 1\]"),
    (np.array(["no", "yes", "no", "yes", "no"]), "not class labels"),
    (np.array([0, 1, 0, 1, 0]), "not class labels"),
    (np.ones((5, 3)) / 3, "1-D"),
])
def test_save_rejects_bad_probabilities(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        _proba(tmp_path).save(bad, tmp_path / "p.csv")
    assert not (tmp_path / "p.csv").exists()


def test_save_class_labels(tmp_path):
    train, test = _clf_frames(["a", "b", "c"])
    d = _data_from_frames(tmp_path, train, test, kind="classification", metric="accuracy",
                          classes=["a", "b", "c"])
    out = tmp_path / "p.csv"
    d.save(np.array(["a", "b", "c", "a", "b"]), out)
    assert _read(out)["prediction"].tolist() == ["a", "b", "c", "a", "b"]
    with pytest.raises(ValueError, match=r"among \['a', 'b', 'c'\].*'z'"):
        d.save(["a", "b", "c", "a", "z"], out)
    with pytest.raises(ValueError, match="has 1 values"):
        d.save(["a"], out)


def test_save_accepts_ints_and_bools_matching_classes(tmp_path):
    train, test = _clf_frames([0, 1, 2])
    d = _data_from_frames(tmp_path, train, test, kind="classification", metric="accuracy",
                          classes=["0", "1", "2"])
    d.save(np.array([0, 1, 2, 1, 0]), tmp_path / "p.csv")
    d.save(np.array([0.0, 1.0, 2.0, 1.0, 0.0]), tmp_path / "p.csv")
    assert _read(tmp_path / "p.csv")["prediction"].tolist() == [0, 1, 2, 1, 0]
    train, test = _clf_frames([True, False])
    b = _data_from_frames(tmp_path / "b", train, test, kind="classification", metric="accuracy",
                          classes=["False", "True"]) if (tmp_path / "b").mkdir() is None else None
    b.save(np.array([True, False, True, False, True]), tmp_path / "b.csv")
    assert _read(tmp_path / "b.csv")["prediction"].tolist() == [True, False, True, False, True]


def test_save_regression_validation_and_clip(tmp_path, capsys):
    train = pd.DataFrame({"f": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"orbit_id": [7, 8], "f": [1.0, 2.0]})
    d = _data_from_frames(tmp_path, train, test)
    out = tmp_path / "p.csv"
    d.save([1.5, -2.0], out)
    assert _read(out)["prediction"].tolist() == [1.5, -2.0]
    for bad, match in (([1.0, np.inf], "NaN or inf"), ([1.0, np.nan], "NaN or inf"),
                       (["a", "b"], "numbers"), ([1.0], "1 values")):
        with pytest.raises(ValueError, match=match):
            d.save(bad, out)
    nn = _data_from_frames(tmp_path / "nn", train, test, metric="rmsle", needs_nonneg=True) \
        if (tmp_path / "nn").mkdir() is None else None
    capsys.readouterr()
    nn.save([1.5, -2.0], out)
    assert _read(out)["prediction"].tolist() == [1.5, 0.0]
    assert "clipped" in capsys.readouterr().out


def test_extra_tables(tmp_path):
    main = pd.DataFrame({"k": range(60), "y": [float(i) for i in range(60)]})
    extra = tmp_path / "lookup.csv"
    pd.DataFrame({"k": [0, 1], "region": ["n", "s"]}).to_csv(extra, index=False)
    task = _task(tmp_path, main, "y", extras=[str(extra)])
    d = helpers.load(task.input_dir)
    assert d.extra_names == ["lookup"]
    assert d.extra("lookup")["region"].tolist() == ["n", "s"]
    with pytest.raises(ValueError, match="No extra table"):
        d.extra("../train")


def test_meta_has_no_labels_or_paths(tmp_path):
    df = pd.DataFrame({"x": range(100), "price": np.round(np.random.default_rng(3).normal(50, 9, 100), 4)})
    task = _task(tmp_path, df, "price")
    text = (task.input_dir / META_FILE).read_text()
    meta = json.loads(text)
    assert meta["target"] == "price" and meta["kind"] == "regression" and meta["classes"] == []
    assert meta["metric"] == "rmse" and meta["n_folds"] == 1 and meta["extra_names"] == []
    assert not any(str(v) in text for v in df["price"].head(20))
    assert str(tmp_path) not in text and "hidden" not in text
    helper_src = (task.input_dir / HELPER_FILE).read_text()
    assert str(tmp_path) not in helper_src


def test_classification_meta_lists_class_names_only(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    meta = json.loads((task.input_dir / META_FILE).read_text())
    assert meta["classes"] == ["no", "yes"] and meta["positive_label"] == "yes"
    assert meta["needs_proba"] is True and meta["needs_nonneg"] is False
    assert set(meta) == {"target", "kind", "metric", "needs_proba", "needs_nonneg", "positive_label",
                         "classes", "id_column", "time_col", "n_folds", "extra_names"}


def test_prepare_writes_meta_and_helper_into_every_fold(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    for k in (1, 2, 3):
        folder = task.fold_dir(k)
        assert json.loads((folder / META_FILE).read_text())["n_folds"] == 3
        assert (folder / HELPER_FILE).read_text() == (
            __import__("pathlib").Path(helpers.__file__).read_text())
        d = helpers.load(folder)
        assert len(d.X_train) == len(d.y_train) and len(d.X_test) == len(d.test_ids)
    single = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run1")
    assert (single.input_dir / META_FILE).exists() and (single.input_dir / HELPER_FILE).exists()


def test_time_aware_task_loads(tmp_path):
    df = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=120).astype(str),
                       "x": range(120), "y": np.arange(120.0)})
    task = _task(tmp_path, df, "y", folds=2, time_col="when")
    d = helpers.load(task.fold_dir(1))
    assert json.loads((task.fold_dir(1) / META_FILE).read_text())["time_col"] == "when"
    assert "when" in d.cat_columns and d.X_train["when"].is_monotonic_increasing


def test_helper_imports_only_numpy_pandas_stdlib():
    import ast
    tree = ast.parse(open(helpers.__file__).read())
    # scikit-learn is optional: its import sits in a try/except ImportError at the top
    optional = {id(n) for t in tree.body if isinstance(t, ast.Try) for n in ast.walk(t)}
    required = [n for n in ast.walk(tree) if id(n) not in optional]
    roots = {n.names[0].name.split(".")[0] for n in required if isinstance(n, ast.Import)}
    roots |= {n.module.split(".")[0] for n in required if isinstance(n, ast.ImportFrom)}
    assert roots <= {"json", "math", "pathlib", "numpy", "pandas"}
    assert "sklearn" in ast.unparse(tree)


HELPER_CLF = """PLAN: Helper-based logistic regression.
```python
import sys; sys.path.insert(0, "data")
from orbit_helpers import load
from sklearn.linear_model import LogisticRegression
d = load()
model = LogisticRegression(max_iter=500).fit(d.X_train_filled, d.y_train_binary)
d.save(model.predict_proba(d.X_test_filled))
```"""

HELPER_REG = """PLAN: Helper-based ridge.
```python
import sys; sys.path.insert(0, "data")
from orbit_helpers import load
from sklearn.linear_model import Ridge
d = load()
d.save(Ridge().fit(d.X_train_filled, d.y_train).predict(d.X_test_filled))
```"""


@pytest.mark.parametrize("folds", [1, 3])
def test_agent_runs_helper_script_end_to_end(clf_csv, tmp_path, folds):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=folds), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([HELPER_CLF]), LocalSandbox(),
                  SearchConfig(exec_timeout=120, max_steps=1, n_drafts=1))
    journal = agent.run()
    node = journal.nodes[0]
    assert not node.buggy, node.error
    assert math.isfinite(node.metric) and node.metric > 0.8
    assert "saved" in (task.run_dir / "nodes" / "001" / "stdout.txt").read_text()


def test_agent_runs_helper_script_for_regression(reg_csv, tmp_path):
    task = prepare(TaskSpec(str(reg_csv), "price", folds=1), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([HELPER_REG]), LocalSandbox(),
                  SearchConfig(exec_timeout=120, max_steps=1, n_drafts=1))
    node = agent.run().nodes[0]
    assert not node.buggy, node.error
    assert math.isfinite(node.metric) and node.metric < task.baseline


def test_save_error_reaches_the_debug_prompt(clf_csv, tmp_path):
    bad = HELPER_CLF.replace("model.predict_proba(d.X_test_filled)", "model.predict(d.X_test_filled)")
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    llm = ScriptedLLM([bad, HELPER_CLF])
    agent = Agent(task, llm, LocalSandbox(), SearchConfig(exec_timeout=120, max_steps=2, n_drafts=1))
    journal = agent.run()
    assert journal.nodes[0].buggy and "needs probabilities" in journal.nodes[0].error
    assert not journal.nodes[1].buggy


# ---- Preprocessor -------------------------------------------------------------------------

def _raw():
    return pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 5.0],
        "city": ["b", "a", None, "b"],
        "flag": [True, False, True, False],
        "ident": [10, 11, 12, 13],
    })


def test_preprocessor_encodes_text_and_fills_numbers():
    raw = _raw()
    out = helpers.Preprocessor(drop=["ident"]).fit(raw).transform(raw)
    assert list(out.columns) == ["num", "city", "flag"]
    assert out["num"].tolist() == [1.0, 3.0, 3.0, 5.0]       # median of 1, 3, 5
    assert out["city"].tolist() == [1, 0, -1, 1]              # sorted ranks, missing -> -1
    assert out["flag"].tolist() == [1, 0, 1, 0]               # False < True as strings
    assert not out.isna().any().any()


def test_preprocessor_unseen_text_and_fill_false():
    pre = helpers.Preprocessor(fill=False).fit(_raw())
    new = pd.DataFrame({"num": [np.nan], "city": ["zz"], "flag": [None], "ident": [1]})
    out = pre.transform(new)
    assert out["city"].tolist() == [-1] and out["flag"].tolist() == [-1]
    assert out["num"].isna().all()
    assert pre.set_params(fill=True).transform(new)["num"].tolist() == [3.0]


def test_preprocessor_all_missing_numeric_uses_zero():
    raw = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    assert helpers.Preprocessor().fit(raw).transform(raw)["a"].tolist() == [0.0, 0.0]


def test_preprocessor_selects_by_name_and_reports_missing():
    pre = helpers.Preprocessor(drop=["ident"]).fit(_raw())
    shuffled = _raw()[["flag", "ident", "city", "num"]].assign(extra=1)
    assert pre.transform(shuffled).equals(pre.transform(_raw()))
    with pytest.raises(ValueError, match=r"\['city', 'flag'\]"):
        pre.transform(_raw()[["num", "ident"]])
    with pytest.raises(ValueError, match="DataFrame"):
        pre.transform(np.zeros((2, 3)))


def test_preprocessor_params_clone_names_and_determinism():
    from sklearn.base import clone
    pre = helpers.Preprocessor(fill=False, drop=["ident"])
    assert clone(pre).get_params() == {"fill": False, "drop": ["ident"]}
    fitted = pre.fit(_raw())
    assert list(fitted.get_feature_names_out()) == ["num", "city", "flag"]
    again = helpers.Preprocessor(fill=False, drop=["ident"]).fit(_raw().iloc[::-1])
    assert again.codes_ == fitted.codes_ and again.medians_ == fitted.medians_
    out = helpers.Preprocessor().set_output(transform="pandas").fit(_raw()).transform(_raw())
    assert isinstance(out, pd.DataFrame)


def test_preprocessor_pipeline_pickles_and_predicts_identically(orbit_helpers):
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    raw = _raw().assign(ident=range(4))
    y = np.array([0, 1, 0, 1])
    model = Pipeline([("prep", orbit_helpers.Preprocessor(drop=["ident"])),
                      ("clf", LogisticRegression())]).fit(raw, y)
    blob = pickle.dumps(model, protocol=4)
    assert b"orbit_helpers" in blob and b"orbit.helpers" not in blob
    loaded = pickle.loads(blob)
    assert np.array_equal(loaded.predict_proba(raw), model.predict_proba(raw))
    assert np.array_equal(loaded.predict_proba(raw.iloc[::-1]), model.predict_proba(raw.iloc[::-1]))
