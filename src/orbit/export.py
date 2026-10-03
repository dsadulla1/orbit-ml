"""Export the best experiment as one fitted, pickled model.

After the search the LLM rewrites the best script so it trains on every labeled row and saves a
single scikit-learn style object that takes raw feature rows. Each attempt is checked before it is
accepted:

  1. the script exits cleanly and writes a regular `model.pkl` (at most 500 MB) and valid
     `predictions.csv` for the check rows;
  2. `pickle_scan.scan` finds no disallowed global in the pickle (nothing is unpickled on the host).
     Custom classes and functions may only come from `orbit_custom.py`, a module written next to
     `model.pkl`: it must be a regular file and pass `pickle_scan.check_module`, and it is copied
     into the verification directory so the fresh process can import it;
  3. a verifier written by orbit loads `model.pkl` in a fresh sandbox container, predicts the raw
     check rows and must reproduce the script's `predictions.csv`.

Layout under `run_dir/export/`: `input/` (see `task.prepare_export_input`), `attempt_<k>/` and
`verify_<k>/` (sandbox working directories), and for an accepted model `model.pkl`,
`sample_input.csv`, `sample_predictions.csv`, `export.json` and, when the model needs its own code,
`orbit_custom.py`. A failed export still writes `export.json`, with a plain-words explanation.
"""
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts
from .agent import ERROR_CHARS, Journal
from .metrics import (ID_COL, PRED_COL, InvalidPredictions, _labels, check_values,
                      read_predictions)
from .pickle_scan import CUSTOM_FILE, CUSTOM_MODULE, check_module, scan
from .sandbox import ExecResult, Sandbox
from .task import Task, prepare_export_input

MAX_MODEL_BYTES = 500 * 1024 * 1024
MODEL_FILE = "model.pkl"
PREDICTIONS_FILE = "predictions.csv"

VERIFIER = '''import json
import pickle
import sys

sys.path.insert(0, "data")
import numpy as np
import pandas as pd

def _label(value) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _positive_column(classes, positive) -> int:
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


meta = json.load(open("data/orbit_meta.json"))
rows = pd.read_csv("data/test.csv")
ids = rows.pop("orbit_id")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
if meta["needs_proba"]:
    pred = model.predict_proba(rows)[:, _positive_column(list(model.classes_), meta["positive_label"])]
else:
    pred = model.predict(rows)
    if meta.get("needs_nonneg"):
        pred = np.clip(np.asarray(pred, dtype=float), 0, None)
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
    explanation: str | None = None          # the failure in plain words, with the fix
    custom_module: Path | None = None       # export/orbit_custom.py, when the model needs it
    notes: list[str] = field(default_factory=list)


def _exception_line(text: str) -> str:
    """The exception line of a traceback (its last unindented line), else the last line."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    plain = [ln for ln in lines if not ln[0].isspace() and not ln.startswith("Traceback")]
    return (plain or lines or [""])[-1][:300]


def _kind(error: str) -> str:
    """Which known failure `error` (the raw text of a failed attempt) describes."""
    checks = [
        ("none", "no working experiment"),
        ("llm", "LLM call failed"),
        ("nocode", "code block"),
        ("timeout", "timed out"),
        ("custom_missing", f"{CUSTOM_FILE} was not written"),
        ("custom_bad", f"{CUSTOM_FILE} is not allowed"),
        ("custom_bad", f"{CUSTOM_FILE} must be"),
        ("main", "global __main__."),
        ("main", "refers to code defined in the script"),
        ("library", "cannot be loaded safely"),
        ("rejected", "was rejected"),
        ("rejected", "were rejected"),
        ("classes", "positive label"),
        ("classes", "' is not in list"),
        ("mismatch", "does not reproduce"),
        ("nomodel", f"{MODEL_FILE} was not written"),
        ("size", "the limit is"),
        ("fresh", "fresh process"),
        ("crash", "Traceback"),
    ]
    return next((kind for kind, needle in checks if needle in error), "other")


def _main_names(error: str) -> list[str]:
    names = re.findall(r"__main__\.([A-Za-z_][\w.]*)", error)
    return list(dict.fromkeys(names))


def explain_failure(error: str) -> str:
    """A short plain-words reason for a failed export and what fixes it. `error` is the raw text
    of the last failed attempt (or the run's `export.error`)."""
    error = error or ""
    kind = _kind(error)
    if kind == "none":
        return "the run has no working experiment to export; start a search with `orbit run` first"
    if kind == "llm":
        return f"the LLM could not be reached ({_exception_line(error.split('LLM call failed:')[-1])})"
    if kind == "nocode":
        return "the LLM's reply had no complete Python code block, so there was no script to run"
    if kind == "timeout":
        return "the export script did not finish within the time limit; a smaller model or fewer rounds will"
    if kind == "custom_missing":
        return (f"the saved model refers to code in {CUSTOM_MODULE}, but {CUSTOM_FILE} was not written "
                "next to model.pkl")
    if kind == "custom_bad":
        first = error.split(f"{CUSTOM_FILE}", 1)[-1].split("\n")[0].strip(" :.")
        return (f"{CUSTOM_FILE} was not accepted ({first[:200]}). It may import only numpy, pandas, "
                "scipy, scikit-learn, lightgbm, xgboost and standard maths and text modules, and may "
                "not touch files, the network, os or eval")
    if kind == "main":
        names = ", ".join(_main_names(error)) or "the classes or functions it uses"
        return (f"the saved model refers to code defined inside the script ({names}), which no other "
                f"program can load. Custom feature or transformer code must live in {CUSTOM_FILE}, "
                "written next to model.pkl, so it travels with the model")
    if kind == "library":
        return ("the saved model uses a library or function that is not allowed in an exported model; "
                "only scikit-learn, lightgbm, xgboost, numpy, pandas and scipy objects, and code in "
                f"{CUSTOM_FILE}, may be pickled")
    if kind == "classes":
        return ("the saved classifier's class labels do not match the positive label; fit it on "
                "d.y_train_binary (classes_ [0, 1], 1 = positive) or on the original labels d.y_train")
    if kind == "mismatch":
        return ("the saved model, reloaded in a fresh process, does not reproduce its own predictions: "
                "something is done to the predictions outside the saved object")
    if kind == "nomodel":
        return "the script did not write a regular model.pkl file"
    if kind == "size":
        return "model.pkl is larger than the allowed size"
    if kind == "rejected":
        return "the predictions the export produced were rejected by orbit's checks (wrong rows or values)"
    if kind == "fresh":
        tail = error.split("fresh process", 1)[-1]
        return ("the saved model could not be loaded and used in a fresh process: "
                f"{_exception_line(tail)}")
    if kind == "crash":
        return f"the export script crashed: {_exception_line(error)}"
    return "the export did not produce a verified model"


def example_hint(error: str) -> str | None:
    """A concrete `--hint` for retrying after `error`; None when a retry cannot help."""
    error = error or ""
    kind = _kind(error)
    names = ", ".join(_main_names(error)) or "the custom classes and functions"
    hints = {
        "none": None,
        "llm": "Reply with exactly one python code block and keep the script short",
        "nocode": "Reply with exactly one python code block and keep the script short",
        "timeout": "Use fewer boosting rounds or a smaller model so the script runs in a minute",
        "custom_missing": f"Write {CUSTOM_FILE} next to model.pkl before importing it",
        "custom_bad": f"Keep {CUSTOM_FILE} to numpy and pandas code only: no os, sys, open or eval",
        "main": f"Move {names} into {CUSTOM_FILE} and import it; define nothing in the script",
        "library": f"Use only scikit-learn, lightgbm, xgboost, numpy and pandas objects, plus {CUSTOM_FILE}",
        "classes": "Fit the classifier on d.y_train_binary so that classes_ is [0, 1]",
        "mismatch": "Put all post-processing inside the pickled model; predictions.csv must be its own output",
        "nomodel": "Save the fitted object with pickle.dump(model, open('model.pkl', 'wb'), protocol=4)",
        "size": "Use a smaller model, for example fewer trees or a lower max depth",
        "rejected": "Write predictions.csv for every row of test.csv by calling the saved model on it",
        "fresh": "The pipeline must take the raw columns of test.csv and do all feature engineering itself",
        "crash": "Keep the script simple: refit the same pipeline on all rows and pickle it",
    }
    hint = hints.get(kind, "Keep the script simple: refit the same pipeline on all rows and pickle it")
    return hint and hint.replace('"', "'")


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
    close = np.isclose(a, b, rtol=1e-5, atol=1e-6)
    if close.all():
        return None
    worst = int(np.nanargmax(np.abs(a - b)))
    return (f"{int((~close).sum())} of {len(a)} predictions differ (largest: row id "
            f"{expected.index[worst]} has {a[worst]:.6g} in predictions.csv but {b[worst]:.6g} from the "
            "reloaded model). Anything done to the predictions after calling the model (clipping, "
            "inverse transforms) must be inside the pickled object instead: use "
            "TransformedTargetRegressor for target transforms and orbit_helpers.ClipOutput for clipping")


def _custom_module(work: Path, data: bytes) -> tuple[Path | None, str | None]:
    """(path of the attempt's `orbit_custom.py` if the model uses it, error). The file must be a
    regular file that passes `check_module`, whether or not the pickle refers to it."""
    path = work / CUSTOM_FILE
    if not (path.exists() or path.is_symlink()):
        return None, None
    if not _regular(path):
        return None, f"{CUSTOM_FILE} must be a regular file, not a link or a directory."
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return None, f"{CUSTOM_FILE} is not allowed: it cannot be read as UTF-8 text ({e})."
    problems = check_module(source)
    if problems:
        return None, (f"{CUSTOM_FILE} is not allowed: " + "; ".join(problems[:5])
                      + ". Only numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, orbit_helpers "
                      "and standard maths and text modules may be imported; no files, network, os, "
                      "eval or __reduce__.")
    return (path if CUSTOM_MODULE.encode() in data else None), None


def _scan_error(problems: list[str]) -> str:
    main = [p for p in problems if "__main__" in p]
    if main:
        names = ", ".join(_main_names(" ".join(main))) or "those classes and functions"
        return (f"{MODEL_FILE} refers to code defined in the script itself ({'; '.join(problems[:3])}), "
                "which no other program can load. Do not define classes or functions in the script: "
                f"move {names} into {CUSTOM_FILE}. Write that module next to {MODEL_FILE} with "
                f"open({CUSTOM_FILE!r}, 'w').write(SRC), then `import {CUSTOM_MODULE}` and use "
                f"{CUSTOM_MODULE}.<name> in the pipeline, so the pickle refers to {CUSTOM_MODULE}.<name>.")
    return (f"{MODEL_FILE} references things that cannot be loaded safely: " + "; ".join(problems[:5])
            + f". Use only classes from scikit-learn, lightgbm, xgboost, numpy, pandas or "
            f"orbit_helpers, and put your own classes and functions in {CUSTOM_FILE}.")


def _check_attempt(task: Task, sandbox: Sandbox, exec_timeout: int, export_dir: Path, k: int,
                   result: ExecResult, ids: pd.Index, input_dir: Path) -> tuple[str | None, pd.Series | None]:
    """Validate one attempt; returns (error, predictions). `error` is None when it is acceptable."""
    work = export_dir / f"attempt_{k}"
    if result.timed_out:
        return f"The script timed out after {exec_timeout} seconds.", None
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
    data = model.read_bytes()
    custom, error = _custom_module(work, data)
    if error:
        return error, None
    problems = scan(data, custom=custom is not None)
    if problems:
        return _scan_error(problems), None
    vdir = export_dir / f"verify_{k}"
    shutil.rmtree(vdir, ignore_errors=True)
    vdir.mkdir(parents=True)
    shutil.copyfile(model, vdir / MODEL_FILE)
    shutil.copyfile(work / PREDICTIONS_FILE, vdir / PREDICTIONS_FILE)
    if custom is not None:
        shutil.copyfile(custom, vdir / CUSTOM_FILE)
    ver = sandbox.run(VERIFIER, vdir, input_dir, exec_timeout)
    if ver.timed_out:
        return f"Loading the saved model in a fresh process timed out after {exec_timeout} seconds.", None
    if "classes_ are" in ver.stderr and "positive label" in ver.stderr:
        return ("The saved model's class labels do not match the positive label: "
                f"{_exception_line(ver.stderr).removeprefix('ValueError: ')}. Fit the classifier "
                "on d.y_train_binary (1 = positive label) or on the original labels d.y_train."), None
    if not ver.ok or not _regular(vdir / "verify.csv"):
        return ("Loading the saved model in a fresh process and predicting the raw rows failed "
                f"(is anything defined in the script instead of {CUSTOM_FILE}, or a column missing?):\n{_run_error(ver)}"), None
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


def _earlier_failure(export_dir: Path) -> str | None:
    """What the previous export of this run failed with (its explanation and raw error), if it did."""
    try:
        old = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(old, dict) or old.get("ok") or not old.get("error"):
        return None
    why = old.get("explanation") or explain_failure(str(old["error"]))
    return f"{why}. Raw error: {_tail(str(old['error']))}"


def export_best(task: Task, journal: Journal, llm, sandbox: Sandbox, exec_timeout: int, cpus: float,
                max_attempts: int = 3, hint: str | None = None) -> ExportResult:
    """Refit the best experiment on all labeled rows and save it as one verified `model.pkl`.
    Problems with the LLM, the sandbox or the model come back as `ok=False`, a reason and a plain
    explanation. `hint` is extra guidance from the user for the prompt; when the previous export of
    this run failed, the first prompt also says how."""
    best = journal.get(journal.best_id) if journal.best_id else None
    export_dir = task.run_dir / "export"
    if best is None or best.code is None:
        res = ExportResult(False, error="no working experiment to export")
        res.explanation = explain_failure(res.error)
        return res
    res = ExportResult(False, node_id=best.id)
    earlier = _earlier_failure(export_dir)
    error = None
    try:
        for stale in [export_dir / MODEL_FILE, export_dir / CUSTOM_FILE,
                      export_dir / "sample_input.csv", export_dir / "sample_predictions.csv",
                      export_dir / "export.json", export_dir / "registry.json"]:
            stale.unlink(missing_ok=True)
        for stale_dir in [*export_dir.glob("attempt_*"), *export_dir.glob("verify_*")]:
            shutil.rmtree(stale_dir, ignore_errors=True)  # leftovers of an earlier export
        input_dir = prepare_export_input(task)
        ids = pd.Index(pd.read_csv(input_dir / "test.csv")[ID_COL])
        system = prompts.system_prompt(task, exec_timeout, cpus)
        code = None
        for k in range(1, max_attempts + 1):
            res.attempts = k
            work = export_dir / f"attempt_{k}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            prompt = prompts.export_prompt(task, best, code, error, hint=hint, earlier=earlier)
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
            first = error.splitlines()[0][:300]
            res.notes.append(f"attempt {k}: " + (_exception_line(error) if first.startswith("Traceback")
                                                 else first))
        else:
            res.error = f"no attempt produced a verified model; last error: {_tail(error or '')[-500:]}"
            res.explanation = explain_failure(error or "")
    except Exception as e:  # noqa: BLE001 - never raise for export problems
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        res.explanation = explain_failure(res.error)
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
    custom = work / CUSTOM_FILE
    if _regular(custom) and CUSTOM_MODULE.encode() in model.read_bytes():
        res.custom_module = export_dir / CUSTOM_FILE
        shutil.copyfile(custom, res.custom_module)


class ReuseError(Exception):
    """There is no verified export to register again; the message says so in plain words."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_export(run_dir: Path) -> ExportResult:
    """The accepted export of `run_dir`, for registering it again without the LLM. Nothing is
    unpickled: the files must be the regular files `export.json` describes (size and sha256), and
    the export's input folder must still be there."""
    export_dir = Path(run_dir) / "export"
    none = ReuseError(f"no verified export in {export_dir}; run without --reuse")
    try:
        info = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise none from None
    if not isinstance(info, dict) or info.get("ok") is not True:
        raise none
    sample_input, sample_pred = export_dir / "sample_input.csv", export_dir / "sample_predictions.csv"
    model = export_dir / MODEL_FILE
    custom = export_dir / CUSTOM_FILE if info.get("custom_module") else None
    needed = [model, sample_input, sample_pred] + ([custom] if custom else [])
    if not all(_regular(p) for p in needed):
        raise none
    for path, size_key, sha_key in [(model, "model_bytes", "model_sha256"),
                                    *([(custom, "custom_module_bytes", "custom_module_sha256")] if custom else [])]:
        if path.stat().st_size != info.get(size_key) or _sha256(path) != info.get(sha_key):
            raise ReuseError(f"{path.name} in {export_dir} no longer matches export.json (it was "
                             "changed after the export was verified); run without --reuse")
    if not all(_regular(export_dir / "input" / name) for name in ("train.csv", "orbit_meta.json")):
        raise ReuseError(f"{export_dir / 'input'} is missing; run without --reuse")
    return ExportResult(True, model, sample_input, sample_pred, attempts=int(info.get("attempts") or 0),
                        node_id=info.get("node_id"), custom_module=custom,
                        notes=list(info.get("notes") or []))


def _write_summary(res: ExportResult, export_dir: Path) -> None:
    """export.json: the outcome only, no code and no data."""
    summary = {"ok": res.ok, "node_id": res.node_id, "attempts": res.attempts, "error": res.error,
               "explanation": res.explanation, "notes": res.notes, "custom_module": False}
    if res.model_path:
        data = res.model_path.read_bytes()
        summary.update(model_file=MODEL_FILE, model_bytes=len(data),
                       model_sha256=hashlib.sha256(data).hexdigest())
    if res.custom_module:
        source = res.custom_module.read_bytes()
        summary.update(custom_module=True, custom_module_bytes=len(source),
                       custom_module_sha256=hashlib.sha256(source).hexdigest())
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        pass
