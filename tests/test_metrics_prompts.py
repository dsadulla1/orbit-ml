import pandas as pd
import pytest

from orbit.metrics import (METRICS, InvalidPredictions, combine_predictions, match_metric,
                          read_predictions, score)
from orbit.prompts import SYSTEM, parse_response


def _write(tmp_path, df):
    path = tmp_path / "predictions.csv"
    df.to_csv(path, index=False)
    return path


def test_read_predictions_validates(tmp_path):
    ids = pd.Index([1, 2, 3])
    good = _write(tmp_path, pd.DataFrame({"orbit_id": [3, 1, 2], "prediction": [0.1, 0.2, 0.3]}))
    assert list(read_predictions(good, ids)) == [0.2, 0.3, 0.1]  # reordered to expected ids

    with pytest.raises(InvalidPredictions, match="missing 1 of 3"):
        read_predictions(_write(tmp_path, pd.DataFrame({"orbit_id": [1, 2], "prediction": [0, 1]})), ids)
    with pytest.raises(InvalidPredictions, match="must have columns"):
        read_predictions(_write(tmp_path, pd.DataFrame({"pred": [1, 2, 3]})), ids)
    with pytest.raises(InvalidPredictions, match="not written"):
        read_predictions(tmp_path / "absent.csv", ids)


def test_score_probability_metric():
    y = pd.Series(["no", "yes", "yes", "no"])
    assert score(METRICS["roc_auc"], y, pd.Series([0.1, 0.9, 0.8, 0.2]), "yes") == 1.0
    with pytest.raises(InvalidPredictions, match="probabilities"):
        score(METRICS["roc_auc"], y, pd.Series(["no", "yes", "yes", "no"]), "yes")


def test_score_labels_and_regression():
    assert score(METRICS["accuracy"], pd.Series([1, 0, 1]), pd.Series(["1", "0", "0"])) == pytest.approx(2 / 3)
    assert score(METRICS["rmse"], pd.Series([1.0, 3.0]), pd.Series([1.0, 1.0])) == pytest.approx(2 ** 0.5)


def test_parse_response_variants():
    plan, code = parse_response("PLAN: Try RF.\n```python\nprint(1)\n```")
    assert plan == "Try RF." and code == "print(1)"

    plan, code = parse_response("Here is my idea\n```\nx = 1\n```\n```python\nprint('longer')\n```")
    assert code == "print('longer')"  # longest block wins

    plan, code = parse_response("PLAN: no code at all")
    assert code is None and plan == "no code at all"


def test_parse_response_messy_output():
    # A non-Python block before the script must not swallow the script's opening fence.
    plan, code = parse_response("PLAN: x\n```bash\npip install foo\n```\n\n```python\nprint(1)\n```")
    assert code == "print(1)"
    plan, code = parse_response("PLAN: Try RF.\r\n```Python\r\nprint(1)\r\n```\r\nDone.")
    assert plan == "Try RF." and code.strip() == "print(1)"
    plan, code = parse_response("**PLAN:** Use RF.\n```python\nprint(1)\n```")
    assert plan == "Use RF."
    assert parse_response("PLAN: empty\n```python\n```")[1] is None


def test_score_compares_labels_across_csv_types():
    acc = METRICS["accuracy"]
    assert score(acc, pd.Series([0, 1, 1]), pd.Series([0.0, 1.0, 1.0])) == 1.0
    assert score(acc, pd.Series([0, 1, 1]), pd.Series(["0.0", "1", "1"])) == 1.0
    assert score(acc, pd.Series([True, False]), pd.Series(["True", "False"])) == 1.0
    assert score(acc, pd.Series(["a", "b"]), pd.Series(["a", "a"])) == 0.5


def test_score_rejects_non_finite_regression():
    with pytest.raises(InvalidPredictions, match="finite"):
        score(METRICS["rmse"], pd.Series([1.0, 2.0]), pd.Series([float("inf"), 1.0]))


def test_system_prompt_formats_and_warns_about_early_stopping():
    text = SYSTEM.format(libraries="numpy", target="y", id_col="orbit_id", pred_col="prediction",
                         prediction_rule="rule", timeout=60, cpus=2)
    assert "early_stopping_rounds" in text and "lgb.early_stopping" in text and "verbose_eval" in text
    assert "f\"{score:.4f}\"" in text


def test_combine_fold_predictions():
    a, b, c = (pd.Series([0.2, 0.8], index=[1, 2]), pd.Series([0.4, 0.6], index=[1, 2]),
               pd.Series([0.0, 1.0], index=[1, 2]))
    assert combine_predictions(METRICS["roc_auc"], [a, b, c]).round(3).tolist() == [0.2, 0.8]
    assert combine_predictions(METRICS["rmse"], [a]).tolist() == [0.2, 0.8]
    labels = [pd.Series(x, index=[1, 2]) for x in (["a", "b"], ["a", "a"], ["b", "b"])]
    assert combine_predictions(METRICS["accuracy"], labels).tolist() == ["a", "b"]


@pytest.mark.parametrize("text,expected", [
    ("RMSE between log-prices", "rmsle"), ("root mean squared log error", "rmsle"),
    ("RMSE", "rmse"), ("Root Mean Squared Error", "rmse"), ("mean absolute error", "mae"),
    ("ROC AUC", "roc_auc"), ("area under the roc curve", "roc_auc"), ("log loss", "log_loss"),
    ("PR-AUC", "pr_auc"), ("macro F1", "f1_macro"), ("balanced accuracy", "balanced_accuracy"),
    ("accuracy", "accuracy"), ("R^2", "r2"), ("f1_macro", "f1_macro"),
    ("how happy customers are", None), ("", None), (None, None),
    ("RMSE on the catalog data", "rmse"), ("RMSE of the login dialog times", "rmse"),
    ("caucasian cohort accuracy", "accuracy"), ("procedure score", None), ("car2 sales r2", "r2"),
    ("mean absolute percentage error", None), ("MAPE", None), ("RMSE of log1p price", "rmsle")])
def test_match_metric(text, expected):
    assert match_metric(text) == expected


def test_rmsle_and_new_classification_metrics():
    y = pd.Series([100.0, 200.0, 400.0])
    assert score(METRICS["rmsle"], y, y) == 0.0
    assert score(METRICS["rmsle"], y, y * 2) > 0.6
    with pytest.raises(InvalidPredictions, match=">= 0"):
        score(METRICS["rmsle"], y, pd.Series([-1.0, 1.0, 1.0]))
    yc = pd.Series(["a", "a", "a", "b"])
    assert score(METRICS["balanced_accuracy"], yc, pd.Series(["a"] * 4)) == 0.5
    assert score(METRICS["pr_auc"], yc, pd.Series([0.1, 0.2, 0.3, 0.9]), positive_label="b") == 1.0
