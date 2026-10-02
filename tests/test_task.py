import json

import numpy as np
import pandas as pd
import pytest

from orbit.task import Task, TaskSpec, prepare


def test_split_hides_labels(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    train = pd.read_csv(task.input_dir / "train.csv")
    test = pd.read_csv(task.input_dir / "test.csv")
    labels = pd.read_csv(task.run_dir / "hidden" / "labels.csv")

    assert "label" in train.columns
    assert "label" not in test.columns and "orbit_id" in test.columns
    assert len(train) + len(test) == 600
    assert set(labels["split"]) == {"val", "test"}
    assert set(labels["orbit_id"]) == set(test["orbit_id"])
    # The hidden directory is outside input/, which is all the sandbox gets.
    assert not any(p.name == "labels.csv" for p in task.input_dir.rglob("*"))


def test_infers_binary_metric_and_positive_class(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    assert task.kind == "classification"
    assert task.metric_name == "roc_auc"
    assert task.positive_label == "yes"          # minority class
    assert task.baseline == pytest.approx(0.5)   # constant prior has no ranking power


def test_regression_defaults(reg_csv, tmp_path):
    task = prepare(TaskSpec(str(reg_csv), "price", folds=1), tmp_path / "run")
    assert task.kind == "regression" and task.metric_name == "rmse"
    assert task.baseline > 0


def test_roundtrip(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", goal="be quick", folds=1), tmp_path / "run")
    loaded = Task.load(task.run_dir)
    assert loaded.spec.goal == "be quick"
    assert loaded.metric_name == task.metric_name
    assert len(loaded.labels("val")) + len(loaded.labels("test")) == 240


@pytest.mark.parametrize("metric,msg", [("rmse", "for regression"), ("nope", "Unknown metric")])
def test_rejects_bad_metric(clf_csv, tmp_path, metric, msg):
    with pytest.raises(ValueError, match=msg):
        prepare(TaskSpec(str(clf_csv), "label", metric=metric, folds=1), tmp_path / "run")


def test_missing_target(clf_csv, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        prepare(TaskSpec(str(clf_csv), "nope", folds=1), tmp_path / "run")


def test_integer_target_with_missing_values_stays_classification(clf_csv, tmp_path):
    df = pd.read_csv(clf_csv)
    df["label"] = (df["label"] == "yes").astype(int).astype(float)
    df.loc[:4, "label"] = None  # missing targets make pandas read 0/1 as 0.0/1.0
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "label", folds=1), tmp_path / "run")
    assert task.kind == "classification" and task.metric_name == "roc_auc"
    assert task.positive_label == "1" and task.classes == ["0", "1"]


def test_regression_metric_overrides_small_integer_target(tmp_path):
    df = pd.DataFrame({"x": range(200), "rating": [i % 5 + 1 for i in range(200)]})
    path = tmp_path / "ratings.csv"
    df.to_csv(path, index=False)
    assert prepare(TaskSpec(str(path), "rating", folds=1), tmp_path / "a").kind == "classification"
    task = prepare(TaskSpec(str(path), "rating", metric="mae", folds=1), tmp_path / "b")
    assert task.kind == "regression" and task.classes == []


def test_rare_class_does_not_break_split(clf_csv, tmp_path):
    df = pd.read_csv(clf_csv)
    df.loc[0, "label"] = "rare"  # a class with a single row cannot be stratified
    path = tmp_path / "rare.csv"
    df.to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "label", metric="accuracy", folds=1), tmp_path / "run")
    assert "rare" in task.classes


def test_cross_validation_layout_hides_labels(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    labels = pd.read_csv(task.run_dir / "hidden" / "labels.csv")
    assert task.n_folds == 3 and labels["orbit_id"].is_unique
    n_test = int((labels["split"] == "test").sum())
    val_ids = set()
    for k in (1, 2, 3):
        d = task.fold_dir(k)
        train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
        assert "label" in train.columns and "label" not in test.columns
        assert set(test["orbit_id"]) == set(task.fold_ids(k))
        fold_val = set(task.labels("val", k).index)
        assert len(test) == len(fold_val) + n_test
        assert not fold_val & val_ids  # folds partition the development rows
        val_ids |= fold_val
        assert len(train) == 600 - n_test - len(fold_val)
    assert len(val_ids) + n_test == 600
    assert not any(p.name == "labels.csv" for p in task.input_dir.rglob("*"))
    assert Task.load(task.run_dir).n_folds == 3


def test_folds_default_depends_on_size(clf_csv, tmp_path):
    assert prepare(TaskSpec(str(clf_csv), "label"), tmp_path / "a").n_folds == 5
    assert prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "b").n_folds == 1


def test_too_few_rows_for_folds(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"x": range(12), "price": [i * 1.5 for i in range(12)]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cross-validation"):
        prepare(TaskSpec(str(path), "price", folds=10), tmp_path / "run")


def test_extra_tables_docs_and_id_column_reach_the_sandbox_and_prompt(clf_csv, tmp_path):
    from orbit.prompts import data_section

    extra = tmp_path / "lookup.tsv"
    pd.DataFrame({"f0": [0.0, 1.0], "region": ["n", "s"]}).to_csv(extra, sep="\t", index=False)
    doc = tmp_path / "about.txt"
    doc.write_text("f0 is the first sensor reading.")
    spec = TaskSpec(str(clf_csv), "label", folds=2, extras=[str(extra)], docs=[str(doc)], id_column="f1",
                    goal="Flag the positives", eval_text="ROC AUC")
    task = prepare(spec, tmp_path / "run")
    for k in (1, 2):
        assert pd.read_csv(task.fold_dir(k) / "extra" / "lookup.csv")["region"].tolist() == ["n", "s"]
    text = data_section(task)
    assert "data/extra/lookup.csv" in text and "region" in text
    assert "first sensor reading" in text and "'f1' only identifies rows" in text
    loaded = Task.load(task.run_dir)
    assert loaded.spec.extras == [str(extra)] and loaded.spec.eval_text == "ROC AUC"
    single = prepare(TaskSpec(str(clf_csv), "label", folds=1, extras=[str(extra)]), tmp_path / "one")
    assert (single.input_dir / "extra" / "lookup.csv").exists()


def test_tab_separated_primary_table(tmp_path):
    path = tmp_path / "t.tsv"
    pd.DataFrame({"x": range(60), "label": ["a", "b"] * 30}).to_csv(path, sep="\t", index=False)
    assert prepare(TaskSpec(str(path), "label", folds=1), tmp_path / "run").kind == "classification"


def test_rare_class_reduces_folds_or_is_rejected(tmp_path):
    def data(positives, name):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"x": range(60), "y": [1] * positives + [0] * (60 - positives)}).to_csv(path, index=False)
        return str(path)

    task = prepare(TaskSpec(data(3, "a"), "y"), tmp_path / "a")
    assert task.n_folds == 2 and any("folds reduced from 5 to 2" in n for n in task.spec.notes)
    assert task.baseline == task.baseline  # finite, not NaN
    assert prepare(TaskSpec(data(2, "b"), "y"), tmp_path / "b").n_folds == 1
    assert prepare(TaskSpec(data(8, "c"), "y"), tmp_path / "c").n_folds == 5
    with pytest.raises(ValueError, match="both classes in every validation fold"):
        prepare(TaskSpec(data(1, "d"), "y"), tmp_path / "d")  # roc_auc cannot be scored on one positive
    assert prepare(TaskSpec(data(1, "e"), "y", metric="accuracy"), tmp_path / "e").baseline > 0.9


def test_baseline_is_the_best_constant_for_each_metric():
    import numpy as np

    from orbit.metrics import METRICS, score
    from orbit.task import _baseline, _best_constant

    y = pd.Series([0.0, 0.0, 10.0, 1000.0, 20.0])
    assert _best_constant(METRICS["rmse"], y) == y.mean()
    assert _best_constant(METRICS["mae"], y) == y.median()
    assert _best_constant(METRICS["rmsle"], y) == pytest.approx(np.expm1(np.log1p(y).mean()))
    # For rmsle the training mean is a weak reference on skewed targets; the baseline must beat it.
    rng = np.random.default_rng(0)
    train = pd.Series(np.where(rng.random(400) < 0.4, 0.0, rng.lognormal(6, 1.5, 400)))
    val = pd.Series(np.where(rng.random(200) < 0.4, 0.0, rng.lognormal(6, 1.5, 200)))
    raw_mean = score(METRICS["rmsle"], val, pd.Series(train.mean(), index=val.index))
    assert _baseline(METRICS["rmsle"], train, val, None) < raw_mean - 0.5
    assert _baseline(METRICS["mae"], train, val, None) <= score(
        METRICS["mae"], val, pd.Series(train.mean(), index=val.index))


# ---- time-aware validation ---------------------------------------------------------------

def _time_csv(tmp_path, n=300, kind="date", shuffle=True, name="ts.csv"):
    df = pd.DataFrame({"f0": range(n), "f1": [i % 7 for i in range(n)],
                       "label": [("yes" if i % 3 == 0 else "no") for i in range(n)]})
    if kind == "date":
        df["when"] = (pd.Timestamp("2020-01-01") + pd.to_timedelta(range(n), unit="D")).strftime("%Y-%m-%d")
    else:
        df["when"] = [1990 + i for i in range(n)]
    if shuffle:
        df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _key(series, kind):
    return pd.to_datetime(series) if kind == "date" else series


def _fold_frames(task):
    labels = pd.read_csv(task.run_dir / "hidden" / "labels.csv")
    for k in range(1, task.n_folds + 1):
        d = task.fold_dir(k)
        train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
        val_ids = labels[(labels["split"] == "val") & (labels["fold"] == (k if task.n_folds > 1 else 1))]["orbit_id"]
        test_ids = labels[labels["split"] == "test"]["orbit_id"]
        yield train, test[test["orbit_id"].isin(val_ids)], test[test["orbit_id"].isin(test_ids)], test


@pytest.mark.parametrize("kind", ["date", "number"])
@pytest.mark.parametrize("folds", [1, 3])
def test_time_split_never_trains_on_the_future(tmp_path, kind, folds):
    path = _time_csv(tmp_path, kind=kind)
    task = prepare(TaskSpec(str(path), "label", folds=folds, time_col="when"), tmp_path / "run")
    assert task.n_folds == folds
    all_dev_max, test_min, seen_val = None, None, []
    for train, val, test, both in _fold_frames(task):
        assert len(val) > 0 and len(test) > 0 and len(both) == len(val) + len(test)
        assert _key(train["when"], kind).max() <= _key(val["when"], kind).min()
        assert "label" not in both.columns
        seen_val.append(val)
        test_min = _key(test["when"], kind).min()
        # every test row is at least as late as every development row
        assert test_min >= _key(train["when"], kind).max()
        assert test_min >= _key(val["when"], kind).max()
    # test rows never appear in a train.csv (rows are identified by their unique f0)
    test_f0 = set(test["f0"])
    for k in range(1, folds + 1):
        assert not test_f0 & set(pd.read_csv(task.fold_dir(k) / "train.csv")["f0"])
    n = 300
    assert len(test) == round(n * 0.2)


def test_time_cv_is_an_expanding_window(tmp_path):
    path = _time_csv(tmp_path, n=300)
    task = prepare(TaskSpec(str(path), "label", folds=3, time_col="when"), tmp_path / "run")
    frames = list(_fold_frames(task))
    trains = [set(t["f0"]) for t, _, _, _ in frames]
    vals = [set(v["f0"]) for _, v, _, _ in frames]
    assert trains[0] < trains[1] < trains[2]               # each fold trains on more rows
    assert all(not t & v for t, v in zip(trains, vals))
    assert trains[1] == trains[0] | vals[0] and trains[2] == trains[1] | vals[1]
    assert not vals[0] & vals[1] and not vals[1] & vals[2]
    dev = len(trains[2]) + len(vals[2])
    assert dev == 240 and len(trains[0]) >= len(vals[0])  # first block is only ever training data


def test_time_labels_schema_and_original_values_unchanged(tmp_path):
    path = _time_csv(tmp_path)
    plain = prepare(TaskSpec(str(path), "label", folds=3), tmp_path / "plain")
    task = prepare(TaskSpec(str(path), "label", folds=3, time_col="when"), tmp_path / "run")
    cols = lambda t: list(pd.read_csv(t.run_dir / "hidden" / "labels.csv").columns)
    assert cols(task) == cols(plain) == ["orbit_id", "split", "fold", "target"]
    labels = pd.read_csv(task.run_dir / "hidden" / "labels.csv")
    assert set(labels.loc[labels["split"] == "test", "fold"]) == {0}
    assert set(labels.loc[labels["split"] == "val", "fold"]) == {1, 2, 3}
    original = pd.read_csv(path).set_index("f0")["when"]
    for train, _, _, both in _fold_frames(task):
        for frame in (train, both):
            assert (frame.set_index("f0")["when"] == original.loc[frame["f0"]].values).all()
    assert list(pd.read_csv(task.fold_dir(1) / "train.csv").columns) == list(pd.read_csv(path).columns)
    assert Task.load(task.run_dir).spec.time_col == "when"


def test_time_ties_keep_file_order(tmp_path):
    n = 200
    df = pd.DataFrame({"f0": range(n), "when": [i // 20 for i in range(n)],
                       "label": [("yes" if i % 3 == 0 else "no") for i in range(n)]})
    path = tmp_path / "ties.csv"
    df.to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "label", folds=1, time_col="when"), tmp_path / "run")
    train = pd.read_csv(task.input_dir / "train.csv")
    assert train["f0"].tolist() == list(range(len(train)))   # stable: ties stay in file order
    (_, val, test, _), = _fold_frames(task)
    assert train["when"].max() <= val["when"].min() <= val["when"].max() <= test["when"].min()


def test_time_unparseable_rows_are_dropped_and_noted(tmp_path):
    path = _time_csv(tmp_path, n=100)
    df = pd.read_csv(path)
    df.loc[[3, 10, 50], "when"] = ["not a date", None, "??"]
    df.to_csv(path, index=False)
    spec = TaskSpec(str(path), "label", folds=1, time_col="when")
    task = prepare(spec, tmp_path / "run")
    assert "dropped 3 rows without a usable time value" in task.spec.notes
    assert task.n_train + len(pd.read_csv(task.input_dir / "test.csv")) == 97


@pytest.mark.parametrize("edit,msg", [
    (lambda df: df.drop(columns="when"), "not found"),
    (lambda df: df.assign(when="2020-01-01"), "constant"),
    (lambda df: df.assign(when=["x"] * 60 + df["when"].tolist()[60:]), "no usable time value"),
])
def test_time_column_errors(tmp_path, edit, msg):
    path = _time_csv(tmp_path, n=100)
    edit(pd.read_csv(path)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=msg):
        prepare(TaskSpec(str(path), "label", folds=1, time_col="when"), tmp_path / "run")


def test_time_column_cannot_be_target_or_reserved_id(tmp_path):
    path = _time_csv(tmp_path, n=100)
    with pytest.raises(ValueError, match="cannot be the target"):
        prepare(TaskSpec(str(path), "label", folds=1, time_col="label"), tmp_path / "run")
    df = pd.read_csv(path)
    df["orbit_id"] = range(len(df))
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="reserved"):
        prepare(TaskSpec(str(path), "label", folds=1, time_col="orbit_id"), tmp_path / "run2")


def test_time_cv_needs_enough_rows(tmp_path):
    path = _time_csv(tmp_path, n=12)
    with pytest.raises(ValueError, match="Too few rows"):
        prepare(TaskSpec(str(path), "label", folds=5, time_col="when"), tmp_path / "run")


def test_time_cv_does_not_reduce_folds_for_rare_classes(tmp_path):
    n = 200
    df = pd.DataFrame({"f0": range(n), "when": range(n), "label": ["no"] * n})
    df.loc[[5, 60, 100, 150, 190], "label"] = "yes"
    path = tmp_path / "rare.csv"
    df.to_csv(path, index=False)
    spec = TaskSpec(str(path), "label", folds=4, time_col="when", metric="accuracy")
    task = prepare(spec, tmp_path / "run")
    assert task.n_folds == 4 and not any("folds reduced" in n for n in task.spec.notes)


def test_time_cv_requires_both_classes_in_every_validation_fold(tmp_path):
    n = 200
    df = pd.DataFrame({"f0": range(n), "when": range(n), "label": ["no"] * n})
    df.loc[[5, 190], "label"] = "yes"   # no positives in the middle blocks
    path = tmp_path / "rare.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="both classes"):
        prepare(TaskSpec(str(path), "label", folds=4, time_col="when"), tmp_path / "run")


def test_older_task_json_without_time_col_loads(clf_csv, tmp_path):
    import json
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    d = json.loads((task.run_dir / "task.json").read_text())
    del d["spec"]["time_col"]
    (task.run_dir / "task.json").write_text(json.dumps(d))
    assert Task.load(task.run_dir).spec.time_col is None


@pytest.mark.parametrize("folds", [1, 3])
def test_time_column_is_mentioned_in_prompt_and_report(tmp_path, folds):
    from orbit import prompts, report
    from orbit.agent import Journal
    path = _time_csv(tmp_path)
    task = prepare(TaskSpec(str(path), "label", folds=folds, time_col="when"), tmp_path / "run")
    assert "later in time (column 'when')" in prompts.data_section(task)
    text = report.facts(task, Journal())["validation"]
    assert text == (f"mean of {folds} forward-chaining folds ordered by when" if folds > 1
                    else "single split ordered by when")
    plain = prepare(TaskSpec(str(path), "label", folds=folds), tmp_path / "plain")
    assert "later in time" not in prompts.data_section(plain)


def test_profile_flags_text_columns(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame({"n": range(40), "country": ["UK", "FR"] * 20, "flag": [True, False] * 20,
                  "y": [0.5 * i for i in range(40)]}).to_csv(path, index=False)
    text = prepare(TaskSpec(str(path), "y", folds=1), tmp_path / "run").profile
    line = {ln.split(" | ")[0]: ln for ln in text.splitlines() if " | " in ln}
    assert "TEXT: encode" in line["country"] and "TEXT" not in line["n"] and "TEXT" not in line["flag"]


# ---- export input -------------------------------------------------------------------------

def test_load_labeled_matches_prepare(tmp_path):
    from orbit.task import load_labeled
    df = pd.DataFrame({"f": range(10), "y": [0.0, 1.0] * 4 + [np.nan, np.nan]})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    out = load_labeled(TaskSpec(str(path), "y"))
    assert len(out) == 8 and str(out["y"].dtype) == "int64"
    with pytest.raises(ValueError, match="not found"):
        load_labeled(TaskSpec(str(path), "nope"))


@pytest.mark.parametrize("folds", [1, 3])
def test_export_input_layout(clf_csv, tmp_path, folds):
    from orbit.task import HELPER_FILE, META_FILE, prepare_export_input
    extra = tmp_path / "lookup.csv"
    pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}).to_csv(extra, index=False)
    spec = TaskSpec(str(clf_csv), "label", folds=folds, extras=[str(extra)])
    task = prepare(spec, tmp_path / "run")
    out = prepare_export_input(task, n_check=50)
    assert out == tmp_path / "run" / "export" / "input"
    source = pd.read_csv(clf_csv)
    train, test = pd.read_csv(out / "train.csv"), pd.read_csv(out / "test.csv")
    pd.testing.assert_frame_equal(train, source)             # all labeled rows, incl. the test split
    assert list(test.columns) == ["orbit_id"] + [c for c in source.columns if c != "label"]
    assert len(test) == 50 and test["orbit_id"].is_unique
    assert set(test["f0"].round(6)) <= set(source["f0"].round(6))   # CSV floats may differ by an ulp
    assert pd.read_csv(out / "extra" / "lookup.csv").shape == (2, 2)
    assert (out / META_FILE).exists() and (out / HELPER_FILE).exists()
    meta = json.loads((out / META_FILE).read_text())
    assert meta["target"] == "label" and meta["extra_names"] == ["lookup"]
    assert not list(out.rglob("labels.csv"))
    again = prepare_export_input(task, n_check=50)
    assert pd.read_csv(again / "test.csv").equals(test)      # seeded sample
    assert len(pd.read_csv(prepare_export_input(task, n_check=10_000) / "test.csv")) == len(source)


def test_export_input_time_ordered_and_missing_target_rows(tmp_path):
    from orbit.task import prepare_export_input
    path = _time_csv(tmp_path)
    df = pd.read_csv(path)
    df.loc[0, "label"] = None                                 # unlabeled rows are not exported
    df.to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "label", folds=1, time_col="when"), tmp_path / "run")
    out = prepare_export_input(task)
    train = pd.read_csv(out / "train.csv")
    assert len(train) == len(df) - 1
    assert train["when"].is_monotonic_increasing
    assert set(pd.read_csv(out / "test.csv").columns) == {"orbit_id", "f0", "f1", "when"}
