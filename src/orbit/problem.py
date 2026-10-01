"""From what the user typed to a prepared task: resolve sources, plan the task, split the data."""
import re
import urllib.parse
from pathlib import Path

from .planner import plan
from .sources import resolve
from .task import Task, TaskSpec, prepare


def run_name(sources: list[str]) -> str:
    """A short folder-name stem from the first source (file, folder or URL)."""
    first = sources[0]
    path = urllib.parse.urlparse(first).path if "://" in first else first
    stem = Path(urllib.parse.unquote(path.rstrip("/"))).name.split(".")[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:40] or "task"


def setup_task(sources: list[str], run_dir: Path, goal: str | None = None, eval_text: str | None = None,
               target: str | None = None, folds: int | None = None, seed: int = 42, llm=None) -> Task:
    """Raises SourceError or PlanError with a message meant for the user."""
    resolved = resolve(sources, run_dir / "source")
    named = _is_metric_name(eval_text)  # a bare metric name is a decision; anything else is wording
    p = plan(resolved.tables, resolved.docs, goal=goal, eval_text=None if named else eval_text,
             target=target, metric=eval_text if named else None, llm=llm)
    spec = TaskSpec(str(p.primary), p.target, p.metric, goal, folds=folds, seed=seed,
                    sources=list(sources), eval_text=eval_text, extras=[str(x) for x in p.extras],
                    docs=[str(x) for x in p.docs], id_column=p.id_column,
                    notes=resolved.notes + p.notes)
    return prepare(spec, run_dir)


def _is_metric_name(text: str | None) -> bool:
    """A bare metric name like 'rmse' or 'roc_auc' is a decision, not a description to interpret."""
    from .metrics import METRICS
    return bool(text) and text.strip().lower().replace("-", "_").replace(" ", "_") in METRICS
