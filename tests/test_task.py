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
    pd.DataFrame({"x": range(12), "label": [0, 1] * 6}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cross-validation"):
        prepare(TaskSpec(str(path), "label", folds=10), tmp_path / "run")


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
