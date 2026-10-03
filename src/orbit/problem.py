"""From what the user typed to a prepared task: resolve sources, plan the task, split the data."""
import re
import shutil
import urllib.parse
from pathlib import Path

from .planner import format_note, plan
from .sources import resolve
from .task import Task, TaskSpec, prepare


def sanitize_name(text: str) -> str:
    """Letters, digits, `_` and `-` only, at most 40 characters; empty when nothing is left."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40].strip("_")


def run_name(sources: list[str]) -> str:
    """A short folder-name stem from the first source (file, folder or URL)."""
    first = sources[0]
    path = urllib.parse.urlparse(first).path if "://" in first else first
    return sanitize_name(Path(urllib.parse.unquote(path.rstrip("/"))).name.split(".")[0]) or "task"


def setup_task(sources: list[str], run_dir: Path, goal: str | None = None, eval_text: str | None = None,
               target: str | None = None, folds: int | None = None, seed: int = 42, llm=None,
               within: Path | None = None, time_col: str | None = None,
               table: str | None = None) -> Task:
    """Raises SourceError or PlanError with a message meant for the user. If setup fails and
    `run_dir` did not exist before the call, the folder this call created is removed again (only
    if it lies inside `within`, when given)."""
    existed = run_dir.exists()
    try:
        return _setup(sources, run_dir, goal, eval_text, target, folds, seed, llm, time_col, table)
    except Exception:
        if not existed:
            _remove_created(run_dir, within)
        raise


def _remove_created(run_dir: Path, within: Path | None = None) -> None:
    """Delete a run folder this process just created; never a link, never anything else."""
    real = run_dir.resolve()
    if run_dir.is_symlink() or not run_dir.is_dir() or real == Path.cwd().resolve():
        return
    if within is not None and not (real != within.resolve() and real.is_relative_to(within.resolve())):
        return
    shutil.rmtree(run_dir, ignore_errors=True)


def _setup(sources, run_dir, goal, eval_text, target, folds, seed, llm, time_col, table=None) -> Task:
    resolved = resolve(sources, run_dir / "source")
    named = _is_metric_name(eval_text)  # a bare metric name is a decision; anything else is wording
    p = plan(resolved.tables, resolved.docs, goal=goal, eval_text=None if named else eval_text,
             target=target, metric=eval_text if named else None, llm=llm, table=table)
    used = [p.primary, *p.extras]
    formats = [n for n in map(format_note, used) if n]
    spec = TaskSpec(str(p.primary), p.target, p.metric, goal, folds=folds, seed=seed,
                    sources=list(sources), eval_text=eval_text, extras=[str(x) for x in p.extras],
                    docs=[str(x) for x in p.docs], id_column=p.id_column,
                    notes=resolved.notes + formats + p.notes, time_col=time_col)
    return prepare(spec, run_dir)


def _is_metric_name(text: str | None) -> bool:
    """A bare metric name like 'rmse' or 'roc_auc' is a decision, not a description to interpret."""
    from .metrics import METRICS
    return bool(text) and text.strip().lower().replace("-", "_").replace(" ", "_") in METRICS
