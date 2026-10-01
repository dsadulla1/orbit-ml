import pandas as pd
import pytest

from orbit.task import Task, TaskSpec, prepare


def test_split_hides_labels(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label"), tmp_path / "run")
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
    task = prepare(TaskSpec(str(clf_csv), "label"), tmp_path / "run")
    assert task.kind == "classification"
    assert task.metric_name == "roc_auc"
    assert task.positive_label == "yes"          # minority class
    assert task.baseline == pytest.approx(0.5)   # constant prior has no ranking power


def test_regression_defaults(reg_csv, tmp_path):
    task = prepare(TaskSpec(str(reg_csv), "price"), tmp_path / "run")
    assert task.kind == "regression" and task.metric_name == "rmse"
    assert task.baseline > 0


def test_roundtrip(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", goal="be quick"), tmp_path / "run")
    loaded = Task.load(task.run_dir)
    assert loaded.spec.goal == "be quick"
    assert loaded.metric_name == task.metric_name
    assert len(loaded.labels("val")) + len(loaded.labels("test")) == 240


@pytest.mark.parametrize("metric,msg", [("rmse", "for regression"), ("nope", "Unknown metric")])
def test_rejects_bad_metric(clf_csv, tmp_path, metric, msg):
    with pytest.raises(ValueError, match=msg):
        prepare(TaskSpec(str(clf_csv), "label", metric=metric), tmp_path / "run")


def test_missing_target(clf_csv, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        prepare(TaskSpec(str(clf_csv), "nope"), tmp_path / "run")
