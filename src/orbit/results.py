"""Collect everything about a finished run into one JSON-friendly dict.

`collect(run_dir)` reads the run folder; `collect_mlflow(run_id)` rebuilds the same structure from
an MLflow server (the parent run and its nested experiment runs), so results can be fetched in
one call without access to the run folder. The CLI table, the JSON export and the HTML report all
render this structure.
"""
import json
import statistics
from pathlib import Path

from . import report as report_mod
from .agent import Journal
from .task import Task


def _last_line(text: str | None, limit: int = 160) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else "")[:limit]


def _read_summary(path: Path) -> tuple[str, str]:
    """(narrative markdown, verification line) from a written report.md."""
    if not path.exists():
        return "", ""
    text = path.read_text(encoding="utf-8")
    if "## Summary" not in text:
        return "", ""
    body = text.split("## Summary", 1)[1]
    narrative, _, rest = body.partition("_Narrative source:")
    check = "Narrative source:" + rest.splitlines()[0].rstrip("_") if rest else ""
    return narrative.strip(), check.strip()


def _tracker_link(run_dir: Path) -> dict | None:
    path = run_dir / "tracking.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    uri = info.get("uri", "")
    if uri.startswith("http"):
        info["url"] = f"{uri.rstrip('/')}/#/experiments/{info['experiment_id']}/runs/{info['run_id']}"
    return info


def collect(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    task, journal = Task.load(run_dir), Journal.load(run_dir / "journal.json")
    f = report_mod.facts(task, journal, top=len(journal.nodes))
    narrative, check = _read_summary(run_dir / "report.md")
    spec = task.spec
    experiments = [{
        "id": n.id, "stage": n.stage, "parent": n.parent, "plan": n.plan,
        "status": "failed" if n.buggy else "ok",
        "val_score": n.metric, "val_std": statistics.stdev(n.fold_scores) if len(n.fold_scores) > 1 else None,
        "fold_scores": n.fold_scores, "test_score": n.test_metric,
        "error": _last_line(n.error), "seconds": round(n.duration, 1),
        "tokens": n.prompt_tokens + n.completion_tokens,
    } for n in journal.nodes]
    best = journal.get(journal.best_id) if journal.best_id else None
    return {
        "run": run_dir.name,
        "problem": {"goal": spec.goal, "eval": spec.eval_text, "sources": spec.sources,
                    "primary_table": Path(spec.data_path).name, "target": spec.target,
                    "extra_tables": [Path(p).name for p in spec.extras],
                    "id_column": spec.id_column, "notes": spec.notes},
        "task": {"kind": task.kind, "metric": task.metric_name,
                 "greater_is_better": task.metric.greater_is_better, "baseline": task.baseline,
                 "folds": task.n_folds, "validation": f["validation"], "classes": task.classes},
        "summary": f, "narrative": narrative, "narrative_check": check,
        "experiments": experiments,
        "best": None if not best else {
            "id": best.id, "code": best.code, "script": f"nodes/{best.id:03d}/solution.py"},
        "stop_reason": journal.stop_reason, "tracker": _tracker_link(run_dir),
    }


def collect_mlflow(run_id: str, client=None) -> dict:
    """The same experiment-level results, read from an MLflow server by parent run id."""
    import mlflow

    client = client or mlflow.MlflowClient()
    parent = client.get_run(run_id)
    children = client.search_runs([parent.info.experiment_id],
                                  filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
                                  order_by=["attributes.start_time ASC"], max_results=1000)
    experiments = []
    for r in children:
        m, p, t = r.data.metrics, r.data.params, r.data.tags
        folds = [v for _, v in sorted((k, v) for k, v in m.items() if k.startswith("val_score_fold_"))]
        experiments.append({
            "id": int(r.info.run_name.split("-")[1]), "stage": p.get("stage"),
            "parent": int(p["parent"]) if p.get("parent") else None, "plan": t.get("orbit.plan", ""),
            "status": t.get("orbit.status"), "val_score": m.get("val_score"),
            "val_std": m.get("val_score_std"), "fold_scores": folds, "test_score": None,
            "seconds": round(m.get("duration_s", 0), 1),
            "tokens": int(m.get("prompt_tokens", 0) + m.get("completion_tokens", 0)),
            "artifact_uri": r.info.artifact_uri})
    pm = parent.data.metrics
    return {
        "run": parent.info.run_name, "run_id": run_id, "experiment_id": parent.info.experiment_id,
        "params": parent.data.params,
        "task": {"metric": parent.data.params.get("metric"), "baseline": pm.get("baseline"),
                 "folds": int(parent.data.params.get("cv_folds", 1))},
        "best": {"id": int(parent.data.tags["orbit.best_experiment"]),
                 "val_score": pm.get("best_val_score"), "test_score": pm.get("best_test_score")}
        if "orbit.best_experiment" in parent.data.tags else None,
        "stop_reason": parent.data.tags.get("orbit.stop_reason"),
        "experiments": sorted(experiments, key=lambda e: e["id"]),
    }
