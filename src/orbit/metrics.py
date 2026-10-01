"""Metrics and prediction-file validation. All scoring happens here, outside the sandbox."""
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


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


METRICS = {
    "roc_auc": Metric("roc_auc", "classification", True, True, skm.roc_auc_score),
    "log_loss": Metric("log_loss", "classification", False, True, skm.log_loss),
    "accuracy": Metric("accuracy", "classification", True, False, skm.accuracy_score),
    "f1_macro": Metric("f1_macro", "classification", True, False,
                       lambda y, p: skm.f1_score(y, p, average="macro")),
    "rmse": Metric("rmse", "regression", False, False, _rmse),
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


def score(metric: Metric, y_true: pd.Series, preds: pd.Series, positive_label=None) -> float:
    if metric.needs_proba:
        p = pd.to_numeric(preds, errors="coerce")
        if p.isna().any() or (p < 0).any() or (p > 1).any():
            raise InvalidPredictions(
                f"{metric.name} needs probabilities in [0, 1] for class {positive_label!r}.")
        y = (y_true.astype(str) == str(positive_label)).astype(int)
        if metric.name == "log_loss":
            return float(metric.fn(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
        return float(metric.fn(y, p))
    if metric.kind == "classification":
        return float(metric.fn(y_true.astype(str), preds.astype(str)))
    p = pd.to_numeric(preds, errors="coerce")
    if p.isna().any():
        raise InvalidPredictions("Regression predictions must be numeric.")
    return float(metric.fn(y_true.astype(float), p))
