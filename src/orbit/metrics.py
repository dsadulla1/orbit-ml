"""Metrics and prediction-file validation. All scoring happens here, outside the sandbox."""
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn import metrics as skm

ID_COL = "orbit_id"
PRED_COL = "prediction"


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str                # "classification" | "regression"
    greater_is_better: bool
    needs_proba: bool
    fn: Callable
    needs_nonneg: bool = False   # targets and predictions must be >= 0 (log-based metrics)


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


def _rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.asarray(p, float)) - np.log1p(np.asarray(y, float))) ** 2)))


METRICS = {
    "roc_auc": Metric("roc_auc", "classification", True, True, skm.roc_auc_score),
    "log_loss": Metric("log_loss", "classification", False, True, skm.log_loss),
    "accuracy": Metric("accuracy", "classification", True, False, skm.accuracy_score),
    "f1_macro": Metric("f1_macro", "classification", True, False,
                       lambda y, p: skm.f1_score(y, p, average="macro")),
    "balanced_accuracy": Metric("balanced_accuracy", "classification", True, False,
                                skm.balanced_accuracy_score),
    "pr_auc": Metric("pr_auc", "classification", True, True, skm.average_precision_score),
    "rmse": Metric("rmse", "regression", False, False, _rmse),
    "rmsle": Metric("rmsle", "regression", False, False, _rmsle, needs_nonneg=True),
    "mae": Metric("mae", "regression", False, False, skm.mean_absolute_error),
    "r2": Metric("r2", "regression", True, False, skm.r2_score),
}


class InvalidPredictions(ValueError):
    """Raised with a message meant to be shown to the agent so it can fix its code."""


def read_predictions(path, expected_ids: pd.Index) -> pd.Series:
    if not path.exists():
        raise InvalidPredictions("predictions.csv was not written to the working directory.")
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 - surface any parse error to the agent
        raise InvalidPredictions(f"predictions.csv could not be parsed: {e}") from e
    missing_cols = {ID_COL, PRED_COL} - set(df.columns)
    if missing_cols:
        raise InvalidPredictions(
            f"predictions.csv must have columns {ID_COL!r} and {PRED_COL!r}; "
            f"missing {sorted(missing_cols)}, found {list(df.columns)[:10]}.")
    if df[ID_COL].duplicated().any():
        raise InvalidPredictions(f"predictions.csv has duplicate {ID_COL} values.")
    preds = df.set_index(ID_COL)[PRED_COL]
    missing = expected_ids.difference(preds.index)
    if len(missing):
        raise InvalidPredictions(
            f"predictions.csv is missing {len(missing)} of {len(expected_ids)} test rows "
            f"(e.g. {ID_COL}={list(missing[:3])}). Predict every row of data/test.csv.")
    preds = preds.loc[expected_ids]
    if preds.isna().any():
        raise InvalidPredictions(f"predictions.csv has {int(preds.isna().sum())} empty predictions.")
    return preds


def check_values(metric: Metric, preds: pd.Series, positive_label=None) -> pd.Series:
    """Validate prediction values for `metric`; returns them as numbers where it needs numbers."""
    if metric.needs_proba:
        p = pd.to_numeric(preds, errors="coerce")
        if p.isna().any() or (p < 0).any() or (p > 1).any():
            raise InvalidPredictions(
                f"{metric.name} needs probabilities in [0, 1] for class {positive_label!r}.")
        return p
    if metric.kind == "classification":
        return preds
    p = pd.to_numeric(preds, errors="coerce")
    if p.isna().any():
        raise InvalidPredictions("Regression predictions must be numeric.")
    if not np.isfinite(p).all():
        raise InvalidPredictions("Regression predictions must be finite (no inf).")
    if metric.needs_nonneg and (p < 0).any():
        raise InvalidPredictions(f"{metric.name} needs predictions >= 0 (clip or exponentiate them).")
    return p


def _labels(s: pd.Series) -> pd.Series:
    """Class labels as strings, so 1, 1.0 and "1" match after CSV round trips."""
    if not pd.api.types.is_bool_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all() and (num == num.round()).all():
            return num.astype("int64").astype(str)
    return s.astype(str)


def score(metric: Metric, y_true: pd.Series, preds: pd.Series, positive_label=None) -> float:
    p = check_values(metric, preds, positive_label)
    if metric.needs_proba:
        y = (y_true.astype(str) == str(positive_label)).astype(int)
        if metric.name == "log_loss":
            return float(metric.fn(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
        return float(metric.fn(y, p))
    if metric.kind == "classification":
        return float(metric.fn(_labels(y_true), _labels(p)))
    return float(metric.fn(y_true.astype(float), p))


def combine_predictions(metric: Metric, folds: list[pd.Series]) -> pd.Series:
    """Merge the predictions the k fold models made for the same rows: the mean for
    probabilities and numbers, the most common label for classes."""
    if len(folds) == 1:
        return folds[0]
    if metric.kind == "classification" and not metric.needs_proba:
        labels = pd.concat([_labels(f) for f in folds], axis=1)
        return labels.mode(axis=1)[0]
    return pd.concat([check_values(metric, f) for f in folds], axis=1).mean(axis=1)


# Wording -> metric, checked in order; the first rule whose words all appear wins. This resolves
# the common phrasings without an LLM; anything else is left to the planner to map or reject.
_METRIC_RULES = [
    (("rmsle",), "rmsle"), (("root mean squared log",), "rmsle"),
    (("log", "rmse"), "rmsle"), (("log", "root mean squared"), "rmsle"),
    (("log1p", "rmse"), "rmsle"), (("log1p", "root mean squared"), "rmsle"),
    (("log loss",), "log_loss"), (("logloss",), "log_loss"), (("cross entropy",), "log_loss"),
    (("pr auc",), "pr_auc"), (("pr_auc",), "pr_auc"), (("average precision",), "pr_auc"),
    (("precision-recall",), "pr_auc"),
    (("roc",), "roc_auc"), (("auc",), "roc_auc"),
    (("balanced accuracy",), "balanced_accuracy"), (("accuracy",), "accuracy"),
    (("f1",), "f1_macro"),
    (("rmse",), "rmse"), (("root mean squared",), "rmse"),
    (("mae",), "mae"), (("mean absolute",), "mae"),
    (("r2",), "r2"), (("r^2",), "r2"), (("r-squared",), "r2"), (("r squared",), "r2"),
    (("coefficient of determination",), "r2"),
]


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").replace("^", "").split())


def _has(text: str, phrase: str) -> bool:
    """Whole-word match, so "log" is not found in "catalog" or "auc" in "caucasian"."""
    return re.search(rf"(?<![a-z0-9]){re.escape(_norm(phrase))}(?![a-z0-9])", text) is not None


def match_metric(text: str | None) -> str | None:
    """Map a free-text evaluation description to a metric name, or None if it is unclear."""
    if not text:
        return None
    t = _norm(text)
    if t.replace(" ", "_") in METRICS:
        return t.replace(" ", "_")
    if _has(t, "mape") or _has(t, "percentage error"):
        return None  # a different metric from MAE; let the planner choose or reject
    for words, name in _METRIC_RULES:
        if all(_has(t, w) for w in words):
            return name
    return None
