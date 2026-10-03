"""Log the exported model to the MLflow model registry, with an alias for the one to serve.

The model is logged on the search's own parent run as a pyfunc model (`orbit_model.py` wrapping
`export/model.pkl`) and registered as a new version of a model named after the experiment. The
alias `champion` marks the version to serve; it moves to a new version only when that version's
cross-validation score is at least as good. Nothing is unpickled here: the file was verified in a
sandbox by `export_best` and is only loaded where the model is served.

Nothing in this module raises for MLflow problems: they come back in `RegistryResult.message`.
"""
import contextlib
import hashlib
import io
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .metrics import METRICS
from .pickle_scan import CUSTOM_FILE
from .task import HELPER_FILE, META_FILE

log = logging.getLogger("orbit")

CHAMPION = "champion"
REGISTRY_FILE = "registry.json"
FEATURES_FILE = "orbit_features.json"
MAX_CATEGORIES = 500   # distinct values of a text column kept in the features file

# The libraries the model was fitted with, as pinned for the sandbox in
# docker/sandbox-requirements.txt (the sandbox installs xgboost-cpu, which is `import xgboost`).
PINNED = ("numpy==2.3.5", "pandas==2.3.3", "scipy==1.18.1", "scikit-learn==1.9.1",
          "xgboost==3.4.1", "lightgbm==4.7.0")


@dataclass
class RegistryResult:
    name: str | None = None
    version: int | None = None
    aliases: list[str] = field(default_factory=list)
    uri: str | None = None                    # models:/<name>/<version>
    promoted: bool = False                    # whether this version now carries `champion`
    message: str = ""


def pip_requirements() -> list[str]:
    import mlflow

    return [*PINNED, f"mlflow=={mlflow.__version__}"]


def _better(metric: str, new: float, old: float) -> bool:
    """Whether `new` is at least as good as `old` for `metric`."""
    return new >= old if METRICS[metric].greater_is_better else new <= old


def _float(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def decide_alias(client, name: str, tags: dict, promote: bool, force: bool) -> tuple[bool, str]:
    """Whether the new version should take `champion`, and why (or why not)."""
    if not promote:
        return False, f"alias {CHAMPION} left where it is (promotion switched off)"
    try:
        current = client.get_model_version_by_alias(name, CHAMPION)
    except Exception:  # noqa: BLE001 - no such alias yet
        return True, f"first {CHAMPION}"
    old = current.tags or {}
    where = f"version {current.version}"
    if force:
        return True, f"replaced {CHAMPION} {where} (forced)"
    for key in ("orbit.metric", "orbit.data_fingerprint"):
        if old.get(key) != tags.get(key):
            return True, (f"replaced {CHAMPION} {where}: it was built for a different "
                          f"{key.removeprefix('orbit.').replace('_', ' ')}, so the scores cannot be compared")
    new_score, old_score = _float(tags.get("orbit.val_score")), _float(old.get("orbit.val_score"))
    if new_score is None or old_score is None:
        return False, f"kept {CHAMPION} on {where}: the validation scores cannot be compared"
    metric = tags.get("orbit.metric")
    if metric in METRICS and _better(metric, new_score, old_score):
        return True, (f"replaced {CHAMPION} {where} (validation {metric} {old_score:.4f} -> "
                      f"{new_score:.4f})")
    return False, (f"kept {CHAMPION} on {where}: its validation {metric} {old_score:.4f} is better "
                   f"than {new_score:.4f}")


def _scores(run_dir: Path, node_id) -> dict:
    from . import results

    data = results.collect(run_dir)
    node = next((e for e in data["experiments"] if e["id"] == node_id), None) or {}
    return {"data": data, "val_score": node.get("val_score"), "val_std": node.get("val_std"),
            "test_score": node.get("test_score")}


def _custom_module(run_dir: Path) -> Path | None:
    path = Path(run_dir) / "export" / CUSTOM_FILE
    return path if path.is_file() and not path.is_symlink() else None


def _custom_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tags(run_dir: Path, export, run_id: str) -> dict[str, str]:
    from .task import Task
    from .tracking import data_fingerprint

    found = _scores(run_dir, export.node_id)
    data, task = found["data"], found["data"]["task"]
    tags = {"orbit.run_id": run_id, "orbit.experiment": export.node_id, "orbit.metric": task["metric"],
            "orbit.baseline": task["baseline"], "orbit.goal": data["problem"]["goal"] or "",
            "orbit.validation": task["validation"]}
    for key in ("val_score", "val_std", "test_score"):
        if found[key] is not None:
            tags[f"orbit.{key}"] = found[key]
    try:
        tags["orbit.data_fingerprint"] = data_fingerprint(Task.load(run_dir))
    except Exception as e:  # noqa: BLE001
        log.warning("registry: no data fingerprint: %s", e)
    if _custom_module(run_dir):
        tags["orbit.custom_code"] = "true"
    return {k: str(v)[:5000] for k, v in tags.items()}


def _description(data: dict, tags: dict, custom: Path | None = None) -> str:
    task, prob = data["task"], data["problem"]
    score = tags.get("orbit.val_score")
    std = f" ± {float(tags['orbit.val_std']):.4f}" if "orbit.val_std" in tags else ""
    lines = [f"**Goal:** {(prob['goal'] or 'predict ' + prob['target'])[:500]}", "",
             f"- Metric: {task['metric']} ({'higher' if task['greater_is_better'] else 'lower'} is better), "
             f"baseline {task['baseline']:.4f}",
             f"- Validation: {task['validation']}"
             + (f", score {float(score):.4f}{std}" if score is not None else ""),
             f"- Held-out test score: {float(tags['orbit.test_score']):.4f}"
             if "orbit.test_score" in tags else "- Held-out test score: not available",
             f"- Based on experiment #{tags['orbit.experiment']} of orbit run `{tags['orbit.run_id']}`",
             "", "This model was refit on all labeled rows after the search, so the honest estimate "
             "of its performance is the cross-validation score and the test score above, not the "
             "score of the refit itself. It takes raw feature rows and returns the probability of "
             "the positive class (probability metrics) or the prediction. Loading it needs the "
             "pinned libraries: " + ", ".join(PINNED) + "."]
    if custom:
        lines += ["", f"This model includes custom code from the experiment: `{CUSTOM_FILE}` "
                  f"(sha256 {_custom_sha256(custom)}). It is shipped with the model, was checked "
                  "statically and is imported when the model is loaded; read it in the model's "
                  "`code/` folder before loading the model anywhere you trust."]
    return "\n".join(lines)


def build_features(input_dir: Path) -> dict:
    """What the model takes, from the export input's train.csv: for every feature column (all but
    the target, the id column and `orbit_id`) its type (`number` or `text`; booleans are text) and,
    for text columns, the sorted distinct values (`categories` is null and `truncated` true above
    MAX_CATEGORIES). Names and category values only, no rows. The wrapper uses it to check requests."""
    meta = json.loads((input_dir / META_FILE).read_text(encoding="utf-8"))
    frame = pd.read_csv(input_dir / "train.csv")
    skip = {meta.get("target"), meta.get("id_column"), "orbit_id"} - {None}
    columns = {}
    for name in (c for c in frame.columns if c not in skip):
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            columns[str(name)] = {"type": "number"}
            continue
        values = sorted({str(v) for v in column.dropna()})
        columns[str(name)] = ({"type": "text", "categories": values} if len(values) <= MAX_CATEGORIES
                              else {"type": "text", "categories": None, "truncated": True})
    return {"features_order": list(columns), "columns": columns}


def _write_features(input_dir: Path) -> Path:
    path = input_dir / FEATURES_FILE
    path.write_text(json.dumps(build_features(input_dir), indent=1), encoding="utf-8")
    return path


def _signature(sample: pd.DataFrame, preds: pd.Series):
    """The sample's column names, untyped and optional. MLflow's inferred types are stricter
    than the fitted pipeline, which does its own preprocessing and copes with gaps: they reject a
    null in an integer column (and cut 1.5 to 1 in one). Untyped columns accept any value. They
    are optional so that MLflow's schema enforcement, whose errors are long, lets a request with a
    missing column or no rows through to the model wrapper, which says what to send."""
    from mlflow.models import ModelSignature, infer_signature
    from mlflow.types import ColSpec, Schema
    from mlflow.types.schema import AnyType

    outputs = infer_signature(model_output=preds).outputs
    return ModelSignature(inputs=Schema([ColSpec(AnyType(), str(c), required=False) for c in sample.columns]),
                          outputs=outputs)


def _log_model(mlflow, export, input_dir: Path, name: str, run_id: str, custom: Path | None = None):
    sample = pd.read_csv(export.sample_input)
    preds = pd.read_csv(export.sample_predictions)["prediction"]
    with mlflow.start_run(run_id=run_id):
        return mlflow.pyfunc.log_model(
            name="model", python_model=str(Path(__file__).with_name("orbit_model.py")),
            artifacts={"model": str(export.model_path), "meta": str(input_dir / META_FILE),
                       "features": str(_write_features(input_dir))},
            code_paths=[str(input_dir / HELPER_FILE), *([str(custom)] if custom else [])],
            pip_requirements=pip_requirements(),
            signature=_signature(sample, preds), input_example=sample.head(5),
            registered_model_name=name)


@contextlib.contextmanager
def _quiet():
    """Hide MLflow's console chatter while logging: its notes that the host's libraries differ from
    the pinned ones (expected: the model targets the serving container, not this machine), warnings,
    and the registration progress lines, which it prints straight to stderr, not through logging."""
    logger = logging.getLogger("mlflow")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(level)


def register_model(run_dir: Path, export, model_name: str | None = None, promote: bool = True,
                   force_champion: bool = False) -> RegistryResult:
    """Register `export.model_path` as a new version of `model_name` (default: the experiment's
    name) and move the `champion` alias to it when it qualifies. Writes `export/registry.json`."""
    run_dir = Path(run_dir)
    result = RegistryResult()
    previous = None
    try:
        from .tracking import configure_mlflow_env

        configure_mlflow_env()  # before the import: MLflow prints a hint when it is imported
        import mlflow

        info = json.loads((run_dir / "tracking.json").read_text())
        uri, run_id = info["uri"], info["run_id"]
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        name = model_name or info.get("experiment_name") or client.get_experiment(
            info["experiment_id"]).name
        result.name = name
        input_dir = run_dir / "export" / "input"
        tags = _tags(run_dir, export, run_id)
        previous = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(uri)  # the registry follows it
        try:
            with _quiet():
                logged = _log_model(mlflow, export, input_dir, name, run_id, _custom_module(run_dir))
        finally:
            mlflow.set_tracking_uri(previous)
        version = int(logged.registered_model_version)
        result.version, result.uri = version, f"models:/{name}/{version}"
        for key, value in tags.items():
            client.set_model_version_tag(name, str(version), key, value)
        client.update_model_version(name, str(version),
                                    _description(_scores(run_dir, export.node_id)["data"], tags,
                                                 _custom_module(run_dir)))
        result.promoted, why = decide_alias(client, name, tags, promote, force_champion)
        if result.promoted:
            client.set_registered_model_alias(name, CHAMPION, str(version))
        result.aliases = list(client.get_model_version(name, str(version)).aliases)
        result.message = why
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001 - registry problems never abort the caller
        result.message = f"model registry failed: {type(e).__name__}: {str(e)[:300]}"
        result.promoted = False
    _write_summary(result, run_dir)
    return result


def _write_summary(result: RegistryResult, run_dir: Path) -> None:
    summary = {"name": result.name, "version": result.version, "uri": result.uri,
               "aliases": result.aliases, "promoted": result.promoted, "message": result.message}
    try:
        path = run_dir / "export" / REGISTRY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        pass
