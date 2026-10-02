"""Helper module for generated scripts. orbit copies this file into every sandbox data directory
as `orbit_helpers.py`; it imports only numpy, pandas and the standard library, and is optional.

    import sys; sys.path.insert(0, "data")
    from orbit_helpers import load
    d = load()                     # reads data/train.csv, data/test.csv, data/orbit_meta.json

`d` (a `Data`) offers:
    train_df, test_df      raw frames as read from the CSVs (test_df has the `orbit_id` column)
    target, kind, metric   target column, "classification" or "regression", metric name
    X_train, X_test        numeric feature frames: no id or target columns, text columns encoded
    X_train_filled, X_test_filled   the same with NaN filled by the training median (0 if all missing)
    cat_columns            the encoded text columns (categorical for LightGBM / XGBoost)
    y_train                target values as in train.csv
    classes, positive_label, y_train_binary
                           class labels as strings ([] for regression); for probability metrics the
                           positive label and a 0/1 array for it (else None)
    test_ids               `orbit_id` values in test order
    extra(name), extra_names   additional tables from data/extra/<name>.csv
    save(predictions, path="predictions.csv")
                           validates and writes predictions.csv (see `Data.save`)

Text columns are encoded with one mapping built from the train and test feature values together
(never labels), so every category in the test rows has a code. Codes are the ranks of the sorted
distinct values as strings (0, 1, 2, ...); missing text stays NaN. Encoded columns are int64, or
float64 when they contain NaN.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "orbit_id"
PRED_COL = "prediction"


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _encode(train: pd.DataFrame, test: pd.DataFrame):
    """Numeric copies of the feature frames, and the names of the text columns that were encoded."""
    train, test, cats = train.copy(), test.copy(), []
    for c in train.columns:
        if _is_numeric(train[c]) and _is_numeric(test[c]):
            continue
        cats.append(c)
        seen = pd.concat([train[c], test[c]]).dropna().astype(str)
        mapping = {v: i for i, v in enumerate(sorted(seen.unique()))}
        for df in (train, test):
            codes = df[c].astype(str).where(df[c].notna()).map(mapping)
            df[c] = codes.astype("float64") if codes.isna().any() else codes.astype("int64")
    return train, test, cats


def _label(v) -> str:
    """A class label as a string, so 1, 1.0 and "1" agree."""
    if isinstance(v, (float, np.floating)) and math.isfinite(v) and v == round(v):
        return str(int(v))
    return str(v)


def _model_order(classes: list) -> list:
    """Class order of predict_proba columns: numeric labels sort by value, others as strings."""
    try:
        return sorted(classes, key=float)
    except ValueError:
        return sorted(classes)


class Data:
    def __init__(self, data_dir="data"):
        d = Path(data_dir)
        meta = json.loads((d / "orbit_meta.json").read_text())
        self._d, self._meta = d, meta
        self.target, self.kind, self.metric = meta["target"], meta["kind"], meta["metric"]
        self.classes = list(meta["classes"])
        self.positive_label = meta["positive_label"]
        self.extra_names = list(meta["extra_names"])
        self._proba, self._nonneg = bool(meta["needs_proba"]), bool(meta["needs_nonneg"])
        self.train_df = pd.read_csv(d / "train.csv")
        self.test_df = pd.read_csv(d / "test.csv")
        self.test_ids = self.test_df[ID_COL].to_numpy()
        self.y_train = self.train_df[self.target]
        drop = {self.target, ID_COL, meta.get("id_column")}
        feats = [c for c in self.train_df.columns if c not in drop]
        self.X_train, self.X_test, self.cat_columns = _encode(
            self.train_df[feats], self.test_df[feats])
        med = self.X_train.median().fillna(0)
        self.X_train_filled = self.X_train.fillna(med)
        self.X_test_filled = self.X_test.fillna(med)
        self.y_train_binary = None
        if self._proba:
            self.y_train_binary = (self.y_train.map(_label) == str(self.positive_label)).to_numpy(int)

    def extra(self, name: str) -> pd.DataFrame:
        """An additional table as read from data/extra/<name>.csv."""
        if name not in self.extra_names:
            raise ValueError(f"No extra table {name!r}; available: {self.extra_names}")
        return pd.read_csv(self._d / "extra" / f"{name}.csv")

    def save(self, predictions, path="predictions.csv") -> None:
        """Write `orbit_id`, `prediction` in test order. `predictions` has one entry per test row:
        the probability of `positive_label` (probability metrics; a 2-D array of class
        probabilities in `classes` order is accepted), a class label (label metrics) or a number
        (regression). Raises ValueError with a short reason if they do not fit the metric."""
        n = len(self.test_ids)
        arr = np.asarray(predictions)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 2 and self._proba and arr.shape[1] == len(self.classes):
            arr = arr[:, _model_order(self.classes).index(str(self.positive_label))]
        if arr.ndim != 1:
            raise ValueError(f"predictions must be 1-D with one value per test row; got shape "
                             f"{arr.shape}" + (f" (expected {n} x {len(self.classes)} class "
                                               f"probabilities or {n} values)." if self._proba else "."))
        if len(arr) != n:
            raise ValueError(f"predictions has {len(arr)} values but test.csv has {n} rows; "
                             "predict every row of d.test_df.")
        note = ""
        if self._proba:
            if not pd.api.types.is_float_dtype(arr):
                raise ValueError(f"{self.metric} needs probabilities of {self.positive_label!r} as "
                                 "floats (use predict_proba), not class labels.")
            arr = arr.astype("float64")
            _check_finite(arr)
            if (arr < 0).any() or (arr > 1).any():
                raise ValueError(f"probabilities must be in [0, 1]; got min {arr.min():.4g}, "
                                 f"max {arr.max():.4g}.")
        elif self.kind == "classification":
            labels = np.array([_label(v) for v in arr], dtype=object)
            bad = sorted({v for v in labels if v not in self.classes})
            if bad:
                raise ValueError(f"predictions must be class labels among {self.classes[:20]}; "
                                 f"got {bad[:5]}.")
            arr = labels
        else:
            try:
                arr = arr.astype("float64")
            except (TypeError, ValueError):
                raise ValueError("regression predictions must be numbers.") from None
            _check_finite(arr)
            if self._nonneg and (arr < 0).any():
                arr = np.clip(arr, 0, None)
                note = f" ({self.metric} needs >= 0: negative predictions clipped to 0)"
        pd.DataFrame({ID_COL: self.test_ids, PRED_COL: arr}).to_csv(path, index=False)
        print(f"saved {n} predictions to {path}{note}")


def _check_finite(arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"predictions contain {int((~np.isfinite(arr)).sum())} NaN or inf "
                         "values; fix or impute them before saving.")


def load(data_dir="data") -> Data:
    return Data(data_dir)
