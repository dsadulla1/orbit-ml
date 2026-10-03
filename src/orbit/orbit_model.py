"""MLflow pyfunc wrapper shipped inside every model orbit registers (MLflow "models from code").

This file is copied into the logged model and runs wherever the model is served, where the orbit
package is not installed: it must import nothing from orbit.
"""
import json
import pickle

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import set_model


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

    def predict(self, context, model_input, params=None):
        rows = pd.DataFrame(model_input)
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
