"""orbit command line interface."""
import time
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from . import report as report_mod
from .agent import Agent, Journal, SearchConfig
from .config import LLMSettings, SandboxSettings, load_dotenv, tracker_choice
from .task import Task, TaskSpec, prepare, profile

app = typer.Typer(help="Autonomous ML experiments: an LLM proposes, a sandbox runs, orbit scores.",
                  no_args_is_help=True, add_completion=False)
console = Console()
TRACKERS = ("auto", "mlflow", "wandb", "none")


@app.callback()
def _main():
    load_dotenv()


def _llm():
    from .llm import OpenAICompatLLM
    settings = LLMSettings()
    if not settings.base_url and not settings.api_key:
        raise typer.BadParameter(
            "No LLM configured. Set OPENAI_API_KEY, or ORBIT_LLM_BASE_URL for a local "
            "OpenAI-compatible server (see .env.example).")
    return OpenAICompatLLM(settings)


@app.command()
def explore(data: Path = typer.Argument(..., exists=True, help="CSV file"),
            target: str = typer.Option(..., "--target", "-t")):
    """Print the dataset profile the agent will see."""
    df = pd.read_csv(data)
    if target not in df.columns:
        raise typer.BadParameter(f"{target!r} not found. Columns: {list(df.columns)}",
                                 param_hint="--target")
    console.print(escape(profile(df, target)))


@app.command()
def run(
    data: list[str] = typer.Argument(..., help="Data: CSV/TSV files, folders, zip files or http(s) "
                                               "URLs. Text files (data descriptions) are read too."),
    target: str = typer.Option(None, "--target", "-t",
                               help="Column to predict (default: inferred from --goal and the data)"),
    goal: str = typer.Option(None, help="The problem in plain language, e.g. 'predict each house's price'"),
    eval_text: str = typer.Option(None, "--eval", "--metric", "-m",
                                  help="How to evaluate: a metric name (roc_auc, pr_auc, log_loss, "
                                       "accuracy, balanced_accuracy, f1_macro, rmse, rmsle, mae, r2) "
                                       "or a description such as 'RMSE between log prices' "
                                       "(default: inferred)"),
    steps: int = typer.Option(20, help="Maximum number of experiments"),
    minutes: float = typer.Option(60, help="Time budget for the whole search"),
    tokens: int = typer.Option(None, help="Optional LLM token budget"),
    drafts: int = typer.Option(3, help="Independent initial solutions before refining"),
    folds: int = typer.Option(None, help="Cross-validation folds (default: 5 for small data, "
                                          "otherwise 1 = a single validation split)"),
    timeout: int = typer.Option(None, help="Per-experiment time limit in seconds"),
    tracker: str = typer.Option("auto", help="auto | mlflow | wandb | none"),
    out: Path = typer.Option(Path("runs"), help="Directory for run folders"),
    seed: int = typer.Option(42),
):
    """Solve a prediction problem: orbit builds and compares models, then reports.

    Examples:
      orbit run data.csv -t churn
      orbit run examples/house_prices/ --goal "Predict each house's sale price" --eval "RMSE between log prices"
      orbit run https://example.com/train.csv --goal "Predict survival"
    """
    from .planner import PlanError
    from .problem import run_name, setup_task
    from .sandbox import DockerSandbox, check_docker
    from .sources import SourceError
    from .tracking import make_tracker

    if tracker not in TRACKERS:
        raise typer.BadParameter(f"choose one of {', '.join(TRACKERS)}", param_hint="--tracker")
    sb_settings = SandboxSettings()
    ok, msg = check_docker(sb_settings)
    if not ok:
        console.print(f"[red]Sandbox unavailable:[/] {escape(msg)}")
        raise typer.Exit(1)
    llm = _llm()

    run_dir = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_name(data)}"
    try:
        task = setup_task(data, run_dir, goal=goal, eval_text=eval_text, target=target, folds=folds,
                          seed=seed, llm=llm)
    except (SourceError, PlanError, ValueError) as e:
        console.print(f"[red]Could not set up the task:[/] {escape(str(e))}")
        raise typer.Exit(1)
    cfg = SearchConfig(max_steps=steps, time_budget=minutes * 60, token_budget=tokens,
                       n_drafts=drafts, exec_timeout=timeout or sb_settings.timeout,
                       cpus=sb_settings.cpus, seed=seed)
    trk = make_tracker(tracker_choice(tracker))
    trk.start(task, cfg, llm.model)

    scheme = f"{task.n_folds}-fold CV" if task.n_folds > 1 else "single split"
    console.print(f"[bold]orbit[/] {escape(str(run_dir))}  model={escape(llm.model)}  tracker={trk.name}")
    console.print(f"  data: {escape(Path(task.spec.data_path).name)}, target={escape(task.spec.target)}, "
                  f"metric={task.metric_name}, baseline={task.baseline:.4f}, validation={scheme}")
    if task.spec.extras:
        console.print(f"  extra tables: {escape(', '.join(Path(p).name for p in task.spec.extras))}")
    for note in task.spec.notes:
        console.print(f"  [dim]{escape(note)}[/]")

    def on_node(node, agent):
        best = agent.best()
        status = (f"[green]{node.metric:.4f}[/]" if not node.buggy else
                  f"[red]failed[/] {_last_line(node.error)}")
        star = " [yellow]* new best[/]" if best and best.id == node.id else ""
        console.print(f"  #{node.id:<3} {node.stage:<8} {status}{star}  "
                      f"[dim]{node.duration:.0f}s | {_short(node.plan, 90)}[/]")

    agent = Agent(task, llm, DockerSandbox(sb_settings), cfg, tracker=trk, on_node=on_node)
    journal = agent.run()
    _write_report(task, journal, llm, trk)


def _short(text, limit=70):
    return escape(" ".join((text or "").split())[:limit])  # model text must not act as Rich markup


def _last_line(text, limit=70):
    """Errors are most informative at the end (the exception message)."""
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return _short(lines[-1] if lines else "", limit)  # already escaped by _short


def _write_report(task: Task, journal: Journal, llm, trk=None):
    md, _ = report_mod.build(task, journal, llm)
    path = task.run_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    html_path = _write_html(task.run_dir)
    if trk:
        trk.log_report(path)
        if html_path:
            trk.log_report(html_path)
    console.print()
    console.print(Markdown(md))
    console.print(f"\n[dim]Report written to {escape(str(path))}"
                  + (f" and {escape(str(html_path))}" if html_path else "") + "[/]")


def _write_html(run_dir: Path, path: Path | None = None) -> Path | None:
    """The standalone HTML report; a failure here never costs the run its results."""
    from . import html_report, results

    try:
        path = path or run_dir / "report.html"
        path.write_text(html_report.render(results.collect(run_dir)), encoding="utf-8")
        return path
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Could not write the HTML report: {type(e).__name__}: {escape(str(e))}[/]")
        return None


@app.command()
def compare(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Show every experiment of a run, best first."""
    task, journal = Task.load(run_dir), Journal.load(run_dir / "journal.json")
    table = Table(title=f"{run_dir.name} ({task.metric_name}, baseline {task.baseline:.4f})")
    for col in ("#", "stage", "parent", "val", "test", "secs", "approach / error"):
        table.add_column(col, overflow="fold")
    ranked = journal.ranked(task.metric)
    order = ranked + [n for n in journal.nodes if n.buggy]
    for n in order:
        table.add_row(
            str(n.id), n.stage, str(n.parent or ""),
            f"{n.metric:.4f}" if n.metric is not None else "-",
            f"{n.test_metric:.4f}" if n.test_metric is not None else "",
            f"{n.duration:.0f}",
            escape(n.plan[:120]) if not n.buggy else f"[red]{_last_line(n.error, 120)}[/]",
        )
    console.print(table)


@app.command()
def results(
    run_dir: Path = typer.Argument(None, exists=True, file_okay=False, help="A run folder"),
    json_out: bool = typer.Option(False, "--json", help="Print everything about the run as JSON"),
    html: Path = typer.Option(None, "--html", help="Write the standalone HTML report to this file"),
    mlflow_run: str = typer.Option(None, "--mlflow", help="Fetch a run from the MLflow server by its "
                                                           "parent run id and print it as JSON"),
):
    """Show a run's results: the best model, scores and every experiment.

    Examples:
      orbit results runs/20261001-120000-wine            # summary table
      orbit results runs/20261001-120000-wine --json     # everything, for scripts
      orbit results runs/20261001-120000-wine --html out.html
      orbit results --mlflow 3f9a...                     # same data, straight from MLflow
    """
    import json as json_mod

    from . import results as results_mod

    if mlflow_run:
        console.print_json(json_mod.dumps(results_mod.collect_mlflow(mlflow_run), default=str))
        return
    if run_dir is None:
        raise typer.BadParameter("give a run folder (or --mlflow RUN_ID)")
    if html:
        out = _write_html(run_dir, html)
        if out:
            console.print(f"HTML report written to {escape(str(out))}")
        return
    data = results_mod.collect(run_dir)
    if json_out:
        console.print_json(json_mod.dumps(data, default=str))
        return
    s, task, prob = data["summary"], data["task"], data["problem"]
    console.print(f"[bold]{escape(prob['goal'] or 'Predict ' + prob['target'])}[/]")
    console.print(f"[dim]{escape(prob['primary_table'])}, target {escape(prob['target'])}, {task['metric']} "
                  f"({task['validation']}), baseline {task['baseline']:.4f}[/]")
    best = s.get("best")
    if best:
        std = f" ± {best['val_std_across_folds']}" if best.get("val_std_across_folds") else ""
        console.print(f"Best: experiment #{best['experiment']}  validation {best['val_score']}{std}  "
                      f"test {best['test_score']}  (gain over baseline {best['gain_over_baseline']})")
    table = Table()
    for col in ("#", "stage", "validation", "test", "approach / error"):
        table.add_column(col, overflow="fold")
    for e in sorted(data["experiments"], key=lambda e: (e["val_score"] is None,
                    -(e["val_score"] or 0) if task["greater_is_better"] else (e["val_score"] or 0))):
        ok = e["status"] == "ok"
        score = f"{e['val_score']:.4f}" + (f" ± {e['val_std']:.3f}" if e["val_std"] else "") if ok else "-"
        table.add_row(str(e["id"]), e["stage"], score,
                      f"{e['test_score']:.4f}" if e["test_score"] is not None else "",
                      escape(e["plan"][:110]) if ok else f"[red]{escape(e['error'][:110])}[/]")
    console.print(table)
    console.print(f"[dim]Report: {escape(str(run_dir / 'report.md'))}  |  {escape(str(run_dir / 'report.html'))}[/]")
    if data["tracker"] and data["tracker"].get("url"):
        console.print(f"[dim]MLflow: {escape(data['tracker']['url'])}[/]")


@app.command()
def summarize(run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
              no_llm: bool = typer.Option(False, "--no-llm", help="Deterministic template only")):
    """(Re)write the run's report.md with a data-checked narrative."""
    task, journal = Task.load(run_dir), Journal.load(run_dir / "journal.json")
    _write_report(task, journal, None if no_llm else _llm())


@app.command()
def doctor():
    """Check the LLM endpoint, Docker sandbox and tracker configuration."""
    from .sandbox import check_docker

    s = LLMSettings()
    console.print(escape(f"LLM: model={s.model} base_url={s.base_url or 'https://api.openai.com/v1'}"))
    try:
        reply = _llm().complete("Reply with the single word: pong", "ping")
        console.print(f"  [green]ok[/] replied {escape(repr(reply.text[:40]))}")
    except Exception as e:  # noqa: BLE001
        console.print(f"  [red]failed[/] {type(e).__name__}: {escape(str(e)[:200])}")
    ok, msg = check_docker()
    console.print(f"Sandbox: [{'green' if ok else 'red'}]{'ok' if ok else 'failed'}[/] {escape(msg)}")
    tracker = tracker_choice("auto")
    console.print(f"Tracker (auto): {tracker}")
    if tracker == "mlflow":
        console.print(f"  {_check_mlflow()}")


def _check_mlflow() -> str:
    import os
    import urllib.request

    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("http"):
        return f"[green]ok[/] local store ({escape(uri or 'MLflow default')})"
    try:
        urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=5)
        return f"[green]ok[/] server reachable at {escape(uri)}"
    except Exception as e:  # noqa: BLE001
        return f"[red]failed[/] {escape(uri)}: {escape(str(e))} (start it: docker compose --env-file .env -f docker/compose.yaml up -d mlflow)"


if __name__ == "__main__":
    app()
