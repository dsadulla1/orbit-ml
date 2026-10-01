"""Experiment tracking backends. The orchestrator (never the sandbox) does all logging.

MLflow: one parent run per orbit run, one nested child run per experiment.
W&B:    one run per orbit run; experiments are logged as steps plus a summary table.
Tracking problems are reported but never abort a search.
"""
import logging
from pathlib import Path

log = logging.getLogger("orbit")

ARTIFACTS = ("solution.py", "prompt.md", "response.md", "stdout.txt", "stderr.txt")


def _safe(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("tracking (%s) failed in %s: %s", type(self).__name__, fn.__name__, e)
    return wrapper


class NullTracker:
    name = "none"

    def start(self, task, config, llm_model): pass
    def log_node(self, node, workdir: Path): pass
    def finish(self, task, journal): pass
    def log_report(self, path: Path): pass


class MLflowTracker:
    name = "mlflow"

    def __init__(self, experiment: str = "orbit"):
        import mlflow
        self.mlflow = mlflow
        self.experiment = experiment
        self.parent = None

    @_safe
    def start(self, task, config, llm_model):
        mlflow = self.mlflow
        mlflow.set_experiment(self.experiment)
        self.parent = mlflow.start_run(run_name=task.run_dir.name)
        mlflow.log_params({
            "dataset": Path(task.spec.data_path).name, "target": task.spec.target,
            "metric": task.metric_name, "task_kind": task.kind, "llm_model": llm_model,
            "max_steps": config.max_steps, "n_drafts": config.n_drafts,
            "exec_timeout": config.exec_timeout, "seed": task.spec.seed,
        })
        mlflow.log_metric("baseline", task.baseline)
        mlflow.log_text(task.profile, "data_profile.txt")

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
        best = journal.get(journal.best_id) if journal.best_id else None
        if best:
            client.log_metric(run_id, "best_val_score", best.metric)
            client.log_metric(run_id, "best_test_score", best.test_metric)
            client.set_tag(run_id, "orbit.best_experiment", str(best.id))
        client.set_tag(run_id, "orbit.stop_reason", journal.stop_reason or "")
        client.log_artifact(run_id, str(task.run_dir / "journal.json"))
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
            art = self.wandb.Artifact(f"exp-{node.id:03d}", type="code")
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


def make_tracker(name: str):
    if name == "mlflow":
        return MLflowTracker()
    if name == "wandb":
        try:
            return WandbTracker()
        except ImportError as e:
            raise SystemExit("W&B tracking needs the extra: uv sync --extra wandb") from e
    return NullTracker()
