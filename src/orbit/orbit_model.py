"""MLflow pyfunc wrapper shipped inside every model orbit registers (MLflow "models from code").

This file is copied into the logged model and runs wherever the model is served, where the orbit
package is not installed: it must import nothing from orbit.
"""
import json
import logging
import pickle

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import set_model

log = logging.getLogger("orbit_model")

MAX_NAMES = 10        # column names listed in an error
MAX_COLUMNS = 3       # non-numeric columns described in one error
MAX_LOGGED = 5        # columns, and values per column, in the unseen-category log line


def _label(value) -> str:
    """A class label as text, with whole floats written as integers (1.0 -> "1")."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _positive_column(classes, positive) -> int:
    """The predict_proba column of the positive label: the column whose class equals it (as text),
    else the column of 1 when the classes are exactly 0 and 1 (ints, floats or bools: the
    classifier was fitted on the 0/1 target), else an error."""
    labels = [_label(c) for c in classes]
    if _label(positive) in labels:
        return labels.index(_label(positive))
    numbers = [float(c) for c in classes
               if isinstance(c, (bool, int, float, np.bool_, np.integer, np.floating))]
    if len(classes) == 2 and sorted(numbers) == [0.0, 1.0]:
        return numbers.index(1.0)
    raise ValueError(f"the model's classes_ are {list(classes)} but the positive label is "
                     f"{positive!r}: classes_ must be [0, 1] (1 = positive label) or contain the "
                     "original labels")


def _bad_request(message: str) -> MlflowException:
    return MlflowException(message, error_code=INVALID_PARAMETER_VALUE)  # the server answers 400


def _shown(value, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _is_missing_text(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in ("", "nan")


def _non_numbers(column: pd.Series) -> list[int]:
    """Positions of the values in `column` that are present but not numbers: what
    pd.to_numeric(errors="coerce") would silently turn into missing values. The strings "nan"
    and "" count as missing, not as errors."""
    if pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
        return []
    try:
        converted = pd.to_numeric(column, errors="coerce")
    except (TypeError, ValueError):  # lists, dicts and the like
        converted = column.map(lambda v: np.nan if not isinstance(v, (int, float)) else v)
    bad = converted.isna().to_numpy() & column.notna().to_numpy()
    bad &= ~column.map(_is_missing_text).to_numpy(dtype=bool)
    return [int(i) for i in np.flatnonzero(bad)]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


class OrbitModel(mlflow.pyfunc.PythonModel):
    """Takes a DataFrame of raw feature rows and returns a Series named `prediction`: the
    probability of the positive class when the task is scored on probabilities, else the
    model's own prediction."""

    def load_context(self, context):
        # Plain pickle is acceptable here: the file passed orbit's allow-list scan and a
        # reload-and-compare check in a sandbox when it was exported, and it is only ever loaded
        # inside the serving container, never on the machine that ran the search.
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["meta"], encoding="utf-8") as f:
            self.meta = json.load(f)
        # MLflow puts a printout of the request in front of every error it reports, so a long one
        # pushes our message far out of sight: print requests in a few lines.
        for option, value in (("max_rows", 2), ("max_columns", 2), ("width", 80), ("max_colwidth", 20)):
            pd.set_option(f"display.{option}", value)
        # Models registered before the features file existed have none: they are not checked.
        self.features = None
        path = (context.artifacts or {}).get("features")
        if path:
            with open(path, encoding="utf-8") as f:
                self.features = json.load(f)

    def _check(self, rows: pd.DataFrame) -> None:
        """Reject a request the model cannot score sensibly, with a short message that says what
        to send; log (never reject) text values the model has not seen in training."""
        features = getattr(self, "features", None)
        if not features:
            return
        if len(rows) == 0:
            raise _bad_request("No rows to score: send at least one record.")
        columns = features.get("columns") or {}
        order = features.get("features_order") or list(columns)
        missing = [c for c in order if c not in rows.columns]
        if missing:
            names = ", ".join(missing[:MAX_NAMES])
            more = f" and {len(missing) - MAX_NAMES} more" if len(missing) > MAX_NAMES else ""
            raise _bad_request(
                f"Missing feature column(s): {names}{more} ({len(missing)} of {len(order)}). Send "
                "every feature column; columns not needed (the target, orbit_id) are ignored.")
        problems = []
        for name in order:
            if (columns.get(name) or {}).get("type") != "number":
                continue
            bad = _non_numbers(rows[name])
            if bad:
                problems.append((name, bad[0] + 1, rows[name].iloc[bad[0]]))
        if problems:
            text = " ".join(f"Column '{n}' must be a number; row {r} has '{_shown(v)}'."
                            for n, r, v in problems[:MAX_COLUMNS])
            if len(problems) > MAX_COLUMNS:
                text += f" {len(problems) - MAX_COLUMNS} more column(s) have the same problem."
            raise _bad_request(text)
        self._log_unseen(rows, order, columns)

    def _log_unseen(self, rows: pd.DataFrame, order: list, columns: dict) -> None:
        found, total = [], 0
        for name in order:
            spec = columns.get(name) or {}
            known = spec.get("categories")
            if spec.get("type") != "text" or known is None:
                continue
            values = rows[name].dropna()
            unseen = values[~values.map(_label).isin(set(known))]
            if len(unseen):
                counts = unseen.map(_label).value_counts()
                total += int(counts.sum())
                shown = ", ".join(f"{_shown(v, 30)} ({_plural(int(n), 'row')})"
                                  for v, n in counts.head(MAX_LOGGED).items())
                found.append(f"{name}: {shown}" + (", ..." if len(counts) > MAX_LOGGED else ""))
        if found:
            extra = f"; {len(found) - MAX_LOGGED} more column(s)" if len(found) > MAX_LOGGED else ""
            log.warning("scored %d value(s) not seen in training: %s%s", total,
                        "; ".join(found[:MAX_LOGGED]), extra)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        rows = pd.DataFrame(model_input)
        self._check(rows)
        ignore = {self.meta.get("target"), self.meta.get("id_column"), "orbit_id"} - {None}
        rows = rows.drop(columns=[c for c in rows.columns if c in ignore])
        if self.meta.get("needs_proba"):
            column = _positive_column(list(self.model.classes_), self.meta["positive_label"])
            pred = self.model.predict_proba(rows)[:, column]
        else:
            pred = self.model.predict(rows)
            if self.meta.get("needs_nonneg"):  # the metric rejects negative values
                pred = np.clip(np.asarray(pred, dtype=float), 0, None)
        return pd.Series(pred, name="prediction").reset_index(drop=True)


set_model(OrbitModel())
