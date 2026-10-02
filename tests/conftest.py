"""Test doubles: a scripted LLM and a subprocess executor for *trusted* fixture code only.

The subprocess executor exists so the search loop can be tested on machines without
Docker. It is intentionally not exposed by the CLI: generated code always runs in Docker.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from orbit.llm import Completion
from orbit.sandbox import SCRIPT, ExecResult

CLASSIFIER = """PLAN: {plan}
```python
import os
import pandas as pd
from sklearn.{module} import {model}

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X, y = train.drop(columns=["label"]), train["label"]
model = {model}({kwargs}).fit(X, y)
proba = model.predict_proba(test.drop(columns=["orbit_id"]))[:, list(model.classes_).index("{pos}")]
pd.DataFrame({{"orbit_id": test["orbit_id"], "prediction": proba}}).to_csv("predictions.csv", index=False)
print("done")
```"""

BROKEN = """PLAN: Gradient boosting with a typo.
```python
import pandas as pd
train = pd.read_csv("data/train.csv")
raise ValueError("could not convert string to float: 'oops'")
```"""

BAD_OUTPUT = """PLAN: Forgets the id column.
```python
import pandas as pd
test = pd.read_csv("data/test.csv")
pd.DataFrame({"prediction": [0.5] * len(test)}).to_csv("predictions.csv", index=False)
```"""


def clf_reply(plan="Logistic regression baseline.", module="linear_model",
              model="LogisticRegression", kwargs="max_iter=500", pos="yes"):
    return CLASSIFIER.format(plan=plan, module=module, model=model, kwargs=kwargs, pos=pos)


class ScriptedLLM:
    """Returns canned replies in order; records every prompt it receives."""
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        text = self.replies.pop(0) if self.replies else clf_reply()
        return Completion(text, prompt_tokens=100, completion_tokens=50)


class LocalSandbox:
    def run(self, code, workdir: Path, data_dir: Path, timeout: int) -> ExecResult:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / SCRIPT).write_text(code)
        if not (workdir / "data").exists():
            shutil.copytree(data_dir, workdir / "data")
        start = time.monotonic()
        try:
            p = subprocess.run([sys.executable, SCRIPT], cwd=workdir, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(-1, "", "TimeoutError", time.monotonic() - start, timed_out=True)
        return ExecResult(p.returncode, p.stdout, p.stderr, time.monotonic() - start)


@pytest.fixture
def clf_csv(tmp_path) -> Path:
    X, y = make_classification(n_samples=600, n_features=8, n_informative=5,
                               weights=[0.7, 0.3], random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["label"] = pd.Series(y).map({0: "no", 1: "yes"})
    path = tmp_path / "clf.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def reg_csv(tmp_path) -> Path:
    X, y = make_regression(n_samples=400, n_features=5, noise=10, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["price"] = y
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def orbit_helpers(monkeypatch):
    """`helpers.py` imported as the module `orbit_helpers`, the name it has inside the sandbox,
    so pickles of its classes refer to the same module path as exported models."""
    import importlib.util
    import orbit.helpers
    spec = importlib.util.spec_from_file_location("orbit_helpers", orbit.helpers.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "orbit_helpers", module)
    spec.loader.exec_module(module)
    return module
