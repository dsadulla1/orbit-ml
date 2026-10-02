"""Export the best experiment as one fitted, pickled model.

After the search the LLM rewrites the best script so it trains on every labeled row and saves a
single scikit-learn style object that takes raw feature rows. Each attempt is checked before it is
accepted:

  1. the script exits cleanly and writes a regular `model.pkl` (at most 500 MB) and valid
     `predictions.csv` for the check rows;
  2. `pickle_scan.scan` finds no disallowed global in the pickle (nothing is unpickled on the host);
  3. a verifier written by orbit loads `model.pkl` in a fresh sandbox container, predicts the raw
     check rows and must reproduce the script's `predictions.csv`.

Layout under `run_dir/export/`: `input/` (see `task.prepare_export_input`), `attempt_<k>/` and
`verify_<k>/` (sandbox working directories), and for an accepted model `model.pkl`,
`sample_input.csv`, `sample_predictions.csv` and `export.json`.
"""
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts
from .agent import ERROR_CHARS, Journal
from .metrics import (ID_COL, PRED_COL, InvalidPredictions, _labels, check_values,
                      read_predictions)
from .pickle_scan import scan
from .sandbox import ExecResult, Sandbox
from .task import Task, prepare_export_input

MAX_MODEL_BYTES = 500 * 1024 * 1024
MODEL_FILE = "model.pkl"
PREDICTIONS_FILE = "predictions.csv"

VERIFIER = '''import json
import pickle
import sys

sys.path.insert(0, "data")
import pandas as pd

meta = json.load(open("data/orbit_meta.json"))
rows = pd.read_csv("data/test.csv")
ids = rows.pop("orbit_id")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
if meta["needs_proba"]:
    labels = [str(int(c)) if isinstance(c, float) and c == int(c) else str(c) for c in model.classes_]
    pred = model.predict_proba(rows)[:, labels.index(str(meta["positive_label"]))]
else:
    pred = model.predict(rows)
pd.DataFrame({"orbit_id": ids, "prediction": pred}).to_csv("verify.csv", index=False)
print("verified", len(rows), "rows")
'''


@dataclass
class ExportResult:
    ok: bool
    model_path: Path | None = None
    sample_input: Path | None = None        # raw features of the check rows (no orbit_id)
    sample_predictions: Path | None = None  # the verified predictions for those rows
    attempts: int = 0
    node_id: int | None = None              # the experiment the model is based on
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _tail(text: str) -> str:
    return text.strip()[-ERROR_CHARS:]


def _run_error(result: ExecResult) -> str:
    return _tail(result.stderr or result.stdout or f"exit code {result.returncode}")


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _mismatch(task: Task, expected: pd.Series, got: pd.Series) -> str | None:
    """Why `got` (the reloaded model's predictions) differs from `expected`, or None if they agree."""
    if task.metric.kind == "classification" and not task.metric.needs_proba:
        bad = int((_labels(expected) != _labels(got.loc[expected.index])).sum())
        return f"{bad} of {len(expected)} predicted labels differ" if bad else None
    a = pd.to_numeric(expected, errors="coerce").to_numpy(float)
    b = pd.to_numeric(got.loc[expected.index], errors="coerce").to_numpy(float)
    if np.allclose(a, b, rtol=1e-5, atol=1e-6):
        return None
    return f"predictions differ by up to {np.nanmax(np.abs(a - b)):.3g}"


def _check_attempt(task: Task, sandbox: Sandbox, exec_timeout: int, export_dir: Path, k: int,
                   result: ExecResult, ids: pd.Index, input_dir: Path) -> tuple[str | None, pd.Series | None]:
    """Validate one attempt; returns (error, predictions). `error` is None when it is acceptable."""
    work = export_dir / f"attempt_{k}"
    if not result.ok:
        return _run_error(result), None
    model = work / MODEL_FILE
    if not _regular(model):
        return (f"{MODEL_FILE} was not written to the working directory (save the fitted object "
                f"with pickle.dump(model, open({MODEL_FILE!r}, 'wb'), protocol=4))."), None
    size = model.stat().st_size
    if size > MAX_MODEL_BYTES:
        return f"{MODEL_FILE} is {size / 1e6:.0f} MB; the limit is {MAX_MODEL_BYTES // 2**20} MB.", None
    if not _regular(work / PREDICTIONS_FILE) and (work / PREDICTIONS_FILE).exists():
        return f"{PREDICTIONS_FILE} must be a regular file.", None
    try:
        preds = read_predictions(work / PREDICTIONS_FILE, ids)
        check_values(task.metric, preds, task.positive_label)
    except InvalidPredictions as e:
        return f"Script ran but its output was rejected: {e}", None
    problems = scan(model.read_bytes())
    if problems:
        return (f"{MODEL_FILE} references things that cannot be loaded safely: "
                + "; ".join(problems[:5]) + ". Use only classes from scikit-learn, lightgbm, xgboost, "
                "numpy, pandas or orbit_helpers, and nothing defined in the script."), None
    vdir = export_dir / f"verify_{k}"
    shutil.rmtree(vdir, ignore_errors=True)
    vdir.mkdir(parents=True)
    shutil.copyfile(model, vdir / MODEL_FILE)
    shutil.copyfile(work / PREDICTIONS_FILE, vdir / PREDICTIONS_FILE)
    ver = sandbox.run(VERIFIER, vdir, input_dir, exec_timeout)
    if not ver.ok or not _regular(vdir / "verify.csv"):
        return ("Loading the saved model in a fresh process and predicting the raw rows failed "
                f"(is anything defined in the script, or a column missing?):\n{_run_error(ver)}"), None
    try:
        got = read_predictions(vdir / "verify.csv", ids)
        check_values(task.metric, got, task.positive_label)
        diff = _mismatch(task, preds, got)
    except InvalidPredictions as e:
        return f"The reloaded model's predictions were rejected: {e}", None
    if diff:
        return (f"The saved model, reloaded in a fresh process, does not reproduce "
                f"{PREDICTIONS_FILE}: {diff}. predictions.csv must come from calling the same "
                "fitted model that is pickled, on the raw rows."), None
    return None, preds


def export_best(task: Task, journal: Journal, llm, sandbox: Sandbox, exec_timeout: int, cpus: float,
                max_attempts: int = 3) -> ExportResult:
    """Refit the best experiment on all labeled rows and save it as one verified `model.pkl`.
    Problems with the LLM, the sandbox or the model come back as `ok=False` and a reason."""
    best = journal.get(journal.best_id) if journal.best_id else None
    if best is None or best.code is None:
        return ExportResult(False, error="no working experiment to export")
    res = ExportResult(False, node_id=best.id)
    export_dir = task.run_dir / "export"
    try:
        for stale in [export_dir / MODEL_FILE, export_dir / "sample_input.csv",
                      export_dir / "sample_predictions.csv", export_dir / "export.json",
                      export_dir / "registry.json"]:
            stale.unlink(missing_ok=True)
        input_dir = prepare_export_input(task)
        ids = pd.Index(pd.read_csv(input_dir / "test.csv")[ID_COL])
        system = prompts.system_prompt(task, exec_timeout, cpus)
        code = error = None
        for k in range(1, max_attempts + 1):
            res.attempts = k
            work = export_dir / f"attempt_{k}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            prompt = prompts.export_prompt(task, best, code, error)
            (work / "prompt.md").write_text(f"# System\n\n{system}\n\n# User\n\n{prompt}",
                                            encoding="utf-8")
            try:
                reply = llm.complete(system, prompt)
            except Exception as e:  # noqa: BLE001 - an LLM hiccup fails the attempt, not the process
                error = f"LLM call failed: {type(e).__name__}: {e}"
                res.notes.append(f"attempt {k}: {error}")
                continue
            (work / "response.md").write_text(reply.text, encoding="utf-8")
            _, new_code = prompts.parse_response(reply.text)
            if new_code is None:
                error = prompts.missing_code_message(reply.text)
                res.notes.append(f"attempt {k}: {error}")
                continue
            code = new_code
            result = sandbox.run(code, work, input_dir, exec_timeout)
            (work / "stdout.txt").write_text(result.stdout, encoding="utf-8")
            (work / "stderr.txt").write_text(result.stderr, encoding="utf-8")
            error, preds = _check_attempt(task, sandbox, exec_timeout, export_dir, k, result, ids,
                                          input_dir)
            if error is None:
                _accept(res, export_dir, input_dir, work, preds)
                break
            res.notes.append(f"attempt {k}: {error.splitlines()[0][:300]}")
        else:
            res.error = f"no attempt produced a verified model; last error: {_tail(error or '')[-500:]}"
    except Exception as e:  # noqa: BLE001 - never raise for export problems
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
    _write_summary(res, export_dir)
    return res


def _accept(res: ExportResult, export_dir: Path, input_dir: Path, work: Path, preds: pd.Series) -> None:
    """Copy the verified model and the check rows next to each other under `export_dir`."""
    model = export_dir / MODEL_FILE
    shutil.copyfile(work / MODEL_FILE, model)
    sample_input = export_dir / "sample_input.csv"
    pd.read_csv(input_dir / "test.csv").drop(columns=[ID_COL]).to_csv(sample_input, index=False)
    sample_pred = export_dir / "sample_predictions.csv"
    preds.rename(PRED_COL).rename_axis(ID_COL).reset_index().to_csv(sample_pred, index=False)
    res.ok, res.model_path, res.sample_input, res.sample_predictions = True, model, sample_input, sample_pred


def _write_summary(res: ExportResult, export_dir: Path) -> None:
    """export.json: the outcome only, no code and no data."""
    summary = {"ok": res.ok, "node_id": res.node_id, "attempts": res.attempts, "error": res.error,
               "notes": res.notes}
    if res.model_path:
        data = res.model_path.read_bytes()
        summary.update(model_file=MODEL_FILE, model_bytes=len(data),
                       model_sha256=hashlib.sha256(data).hexdigest())
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        pass
