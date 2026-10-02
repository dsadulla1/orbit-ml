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
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


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
            labels = [_label(c) for c in self.model.classes_]
            column = labels.index(_label(self.meta["positive_label"]))
            pred = self.model.predict_proba(rows)[:, column]
        else:
            pred = self.model.predict(rows)
            if self.meta.get("needs_nonneg"):  # the metric rejects negative values
                pred = np.clip(np.asarray(pred, dtype=float), 0, None)
        return pd.Series(pred, name="prediction").reset_index(drop=True)


set_model(OrbitModel())
