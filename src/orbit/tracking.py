"""Experiment tracking backends. The orchestrator (never the sandbox) does all logging.

MLflow: one experiment per problem, one parent run per orbit run, one nested child run per experiment.
W&B:    one run per orbit run; experiments are logged as steps plus a summary table.
Tracking problems are reported but never abort a search.
"""
import hashlib
import json
import logging
import os
import statistics
from importlib import metadata
from pathlib import Path

log = logging.getLogger("orbit")

MLFLOW_FAIL_FAST = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "1",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
}
TAG_LIMIT = 5000
ARTIFACTS = ("solution.py", "prompt.md", "response.md", "stdout.txt", "stderr.txt")


def _safe(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("tracking (%s) failed in %s: %s", type(self).__name__, fn.__name__, e)
    return wrapper


def resolve_experiment(explicit: str | None, sources: list[str], name: str | None = None) -> str:
    """The MLflow experiment / W&B project for a run: `--experiment`, then ORBIT_EXPERIMENT, then a
    name derived from the problem (the sanitised `--name`, else the first data source)."""
    from .problem import run_name, sanitize_name

    chosen = explicit if explicit is not None else os.environ.get("ORBIT_EXPERIMENT")
    if chosen is not None and (explicit is not None or chosen.strip()):
        chosen = chosen.strip()
        if not chosen:
            raise ValueError("The experiment name must not be empty.")
        return chosen
    return (sanitize_name(name) if name else "") or run_name(sources)


def data_fingerprint(task) -> str:
    """Row count of the primary table (after splitting) and a hash of the file, to tell data versions apart."""
    import pandas as pd

    fold = task.fold_dir(1)
    rows = sum(len(pd.read_csv(fold / f)) for f in ("train.csv", "test.csv"))
    digest = hashlib.sha256()
    with open(task.spec.data_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"rows={rows} sha256={digest.hexdigest()[:12]}"


def _orbit_version() -> str:
    try:
        return metadata.version("orbit")
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_tags(task, config) -> dict[str, str]:
    from .report import _validation

    spec = task.spec
    tags = {
        "orbit.goal": spec.goal or "", "orbit.eval": spec.eval_text or "",
        "orbit.primary_table": Path(spec.data_path).name,
        "orbit.extra_tables": ",".join(Path(p).name for p in spec.extras),
        "orbit.target": spec.target, "orbit.metric": task.metric_name,
        "orbit.validation": _validation(task), "orbit.version": _orbit_version(),
    }
    try:
        tags["orbit.data_fingerprint"] = data_fingerprint(task)
    except Exception as e:  # noqa: BLE001
        log.warning("tracking: no data fingerprint: %s", e)
    tags["mlflow.note.content"] = "\n".join([
        f"**Goal:** {(spec.goal or 'predict ' + spec.target)[:500]}", "",
        f"- Metric: {task.metric_name} (baseline {task.baseline:.4f})",
        f"- Data: {tags['orbit.primary_table']}, target `{spec.target}`"
        + (f", extra tables {tags['orbit.extra_tables']}" if tags["orbit.extra_tables"] else ""),
        f"- Validation: {tags['orbit.validation']}",
    ])
    return {k: str(v)[:TAG_LIMIT] for k, v in tags.items()}


def configure_mlflow_env() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # quiet import-time banner
    # With a tracking server MLflow prints "View run ... at ..." to stdout after every run; the
    # CLI already shows one line per experiment.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1")
    # MLflow retries a dead server for about 4 minutes per call. Fail fast instead, so an
    # unreachable tracker costs seconds per logged experiment, not minutes.
    for key, value in MLFLOW_FAIL_FAST.items():
        os.environ.setdefault(key, value)


class NullTracker:
    name = "none"

    def start(self, task, config, llm_model): pass
    def log_node(self, node, workdir: Path): pass
    def finish(self, task, journal): pass
    def log_report(self, path: Path): pass


class MLflowTracker:
    name = "mlflow"

    def __init__(self, experiment: str = "orbit"):
        configure_mlflow_env()
        import mlflow
        self.mlflow = mlflow
        self.experiment = experiment
        self.parent = None

    def _set_experiment(self):
        mlflow = self.mlflow
        try:
            mlflow.set_experiment(self.experiment)
        except mlflow.exceptions.MlflowException as e:
            if "deleted" not in str(e).lower():
                raise
            # A deleted experiment keeps its name reserved; bring it back instead of losing the run.
            client = mlflow.MlflowClient()
            client.restore_experiment(client.get_experiment_by_name(self.experiment).experiment_id)
            mlflow.set_experiment(self.experiment)

    @_safe
    def start(self, task, config, llm_model):
        mlflow = self.mlflow
        self._set_experiment()
        self.parent = mlflow.start_run(run_name=task.run_dir.name)
        for key, value in _run_tags(task, config).items():
            try:
                mlflow.set_tags({key: value})
            except Exception as e:  # noqa: BLE001
                log.warning("tracking: could not set tag %s: %s", key, e)
        mlflow.log_params({
            "dataset": Path(task.spec.data_path).name, "target": task.spec.target,
            "metric": task.metric_name, "task_kind": task.kind, "llm_model": llm_model,
            "max_steps": config.max_steps, "n_drafts": config.n_drafts,
            "exec_timeout": config.exec_timeout, "seed": task.spec.seed,
            "cv_folds": task.n_folds, "folds": task.n_folds,
        })
        mlflow.log_metric("baseline", task.baseline)
        mlflow.log_text(task.profile, "data_profile.txt")
        # Lets `orbit results` and the HTML report link to (and fetch from) this run.
        (task.run_dir / "tracking.json").write_text(json.dumps({
            "tracker": "mlflow", "uri": mlflow.get_tracking_uri(),
            "experiment_id": self.parent.info.experiment_id, "experiment_name": self.experiment,
            "run_id": self.parent.info.run_id}))

    @_safe
    def log_node(self, node, workdir: Path):
        mlflow = self.mlflow
        with mlflow.start_run(run_name=f"exp-{node.id:03d}-{node.stage}", nested=True,
                              parent_run_id=self.parent.info.run_id):
            mlflow.log_params({"stage": node.stage, "parent": node.parent or "",
                               "debug_depth": node.debug_depth})
            mlflow.set_tags({"orbit.plan": node.plan[:5000], "orbit.status":
                             "ok" if not node.buggy else "failed"})
            metrics = {"duration_s": node.duration, "prompt_tokens": node.prompt_tokens,
                       "completion_tokens": node.completion_tokens}
            if node.metric is not None:
                metrics["val_score"] = node.metric
            if len(node.fold_scores) > 1:
                metrics["val_score_std"] = statistics.stdev(node.fold_scores)
                metrics.update({f"val_score_fold_{k}": v for k, v in enumerate(node.fold_scores, 1)})
            mlflow.log_metrics(metrics)
            for name in ARTIFACTS:
                if (workdir / name).exists():
                    mlflow.log_artifact(str(workdir / name))
        # Per-step scores on the parent run give a progress curve in the MLflow UI.
        if node.metric is not None:
            mlflow.MlflowClient().log_metric(self.parent.info.run_id, "val_score",
                                             node.metric, step=node.id)

    @_safe
    def finish(self, task, journal):
        client, run_id = self.mlflow.MlflowClient(), self.parent.info.run_id
        try:
            best = journal.get(journal.best_id) if journal.best_id else None
            if best:
                client.log_metric(run_id, "best_val_score", best.metric)
                if best.test_metric is not None:
                    client.log_metric(run_id, "best_test_score", best.test_metric)
                client.set_tag(run_id, "orbit.best_experiment", str(best.id))
            client.set_tag(run_id, "orbit.stop_reason", journal.stop_reason or "")
            client.log_artifact(run_id, str(task.run_dir / "journal.json"))
        finally:  # never leave the parent run RUNNING, e.g. when artifact upload fails
            self.mlflow.end_run()

    @_safe
    def log_report(self, path: Path):
        if self.parent:
            self.mlflow.MlflowClient().log_artifact(self.parent.info.run_id, str(path))


class WandbTracker:
    name = "wandb"

    def __init__(self, project: str = "orbit"):
        import wandb
        self.wandb = wandb
        self.project = project
        self.run = None
        self.rows = []

    @_safe
    def start(self, task, config, llm_model):
        self.run = self.wandb.init(project=self.project, name=task.run_dir.name, config={
            "dataset": Path(task.spec.data_path).name, "target": task.spec.target,
            "metric": task.metric_name, "llm_model": llm_model,
            "max_steps": config.max_steps, "n_drafts": config.n_drafts, "baseline": task.baseline,
        })

    @_safe
    def log_node(self, node, workdir: Path):
        data = {"experiment": node.id, "duration_s": node.duration,
                "tokens": node.prompt_tokens + node.completion_tokens,
                "failed": int(node.buggy)}
        if node.metric is not None:
            data["val_score"] = node.metric
        self.run.log(data, step=node.id)
        self.rows.append([node.id, node.stage, node.parent, node.metric, node.plan])
        if (workdir / "solution.py").exists():
            # Artifact names are project-wide, so scope them to this run.
            art = self.wandb.Artifact(f"{self.run.id}-exp-{node.id:03d}", type="code")
            art.add_file(str(workdir / "solution.py"))
            self.run.log_artifact(art)

    @_safe
    def finish(self, task, journal):
        table = self.wandb.Table(columns=["id", "stage", "parent", "val_score", "plan"],
                                 data=self.rows)
        self.run.log({"experiments": table})
        best = journal.get(journal.best_id) if journal.best_id else None
        if best:
            self.run.summary.update({"best_val_score": best.metric,
                                     "best_test_score": best.test_metric,
                                     "best_experiment": best.id})
        self.run.summary["stop_reason"] = journal.stop_reason
        self.run.finish()

    @_safe
    def log_report(self, path: Path):
        pass  # the run is closed by then; the report lives in the run directory


def make_tracker(name: str, experiment: str | None = None):
    """`experiment` is the MLflow experiment / W&B project; None keeps each backend's default."""
    if name == "mlflow":
        return MLflowTracker(experiment) if experiment else MLflowTracker()
    if name == "wandb":
        try:
            return WandbTracker(experiment) if experiment else WandbTracker()
        except ImportError as e:
            raise SystemExit("W&B tracking needs the extra: uv sync --extra wandb") from e
    return NullTracker()
