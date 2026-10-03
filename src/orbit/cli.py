"""orbit command line interface."""
import time
from pathlib import Path

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
def explore(
    data: list[str] = typer.Argument(..., help="Data: CSV/TSV/Excel files, folders, zip files or "
                                               "http(s) URLs, as for `orbit run`"),
    target: str = typer.Option(None, "--target", "-t", help="Column to profile as the target"),
    goal: str = typer.Option(None, help="Accepted for symmetry with `run`; explore makes no LLM call"),
    table: str = typer.Option(None, "--table", help="Profile this table (file name or part of it) when "
                                                    "several tables hold the target"),
):
    """Show the tables and documents found in the sources and the data profile the agent sees.

    Examples:
      orbit explore data.csv -t churn
      orbit explore examples/house_prices/
      orbit explore "Online Retail.xlsx"
    """
    import tempfile

    from .planner import PlanError, describe_table, format_note, load_for_profile, match_table
    from .sources import SourceError, resolve

    with tempfile.TemporaryDirectory(prefix="orbit-explore-") as tmp:
        try:
            found = resolve(data, Path(tmp) / "source")
            loaded = {p: load_for_profile(p) for p in found.tables}
            notes = found.notes + [n for n in map(format_note, found.tables) if n]
        except (SourceError, ValueError, OSError) as e:
            console.print(f"[red]Could not read the data:[/] {escape(str(e))}")
            raise typer.Exit(1)
        console.print(f"Tables: {escape(', '.join(p.name for p in found.tables))}")
        if found.docs:
            console.print(f"Documents: {escape(', '.join(p.name for p in found.docs))}")
        for note in notes:
            console.print(f"[dim]{escape(note)}[/]")
        frames = {p: df for p, (df, _, _) in loaded.items()}
        if target:
            holders = [p for p, df in frames.items() if target in df.columns]
            if not holders:
                console.print(f"[red]--target {escape(repr(target))} not found.[/] Columns:")
                for p, df in frames.items():
                    console.print(f"  {escape(p.name)}: {escape(', '.join(map(str, df.columns)))}")
                raise typer.Exit(1)
            if table:
                try:
                    wanted = match_table(table, [p.name for p in frames])
                except PlanError as e:
                    console.print(f"[red]{escape(str(e))}[/]")
                    raise typer.Exit(1)
                holders = [p for p in holders if p.name == wanted]
                if not holders:
                    console.print(f"[red]--table {escape(wanted)} has no column {escape(repr(target))}.[/]")
                    raise typer.Exit(1)
            primary = max(holders, key=lambda p: ("train" in p.name.lower(), loaded[p][1]))
            if len(holders) > 1:
                sizes = ", ".join(f"{p.name}: {loaded[p][1]:,} rows" for p in holders)
                console.print(f"[dim]{len(holders)} tables have column {escape(repr(target))} "
                              f"({escape(sizes)}); profiling {escape(primary.name)}[/]")
        for p, (df, total, sampled) in loaded.items():
            console.print()
            if target and p == primary:
                basis = (f"a random sample of {len(df):,} of {total:,} rows" if sampled
                         else f"all {total:,} rows")
                console.print(escape(f"### {p.name} (profile of {basis})"))
                console.print(escape(profile(df, target)))
            else:
                console.print(escape(describe_table(p, df, total_rows=total, sampled=sampled)))


@app.command()
def run(
    data: list[str] = typer.Argument(..., help="Data: CSV/TSV/Excel (.xlsx) files, folders, zip files "
                                               "or http(s) URLs. Text files (data descriptions) are "
                                               "read too."),
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
    table: str = typer.Option(None, "--table", help="File name (or part of it) of the table to learn from "
                                                    "when several tables hold the target"),
    folds: int = typer.Option(None, help="Cross-validation folds (default: 5 for small data, "
                                          "otherwise 1 = a single validation split)"),
    time_col: str = typer.Option(None, "--time-col",
                                 help="Order rows by this column and validate on later rows than "
                                      "the ones used for training"),
    time_order: bool = typer.Option(False, "--time-order",
                                    help="The rows are already in time order: validate on later rows than "
                                         "the ones used for training, without needing a date column"),
    patience: int = typer.Option(None, min=1,
                                 help="Stop after this many working experiments in a row that do "
                                      "not beat the best by more than its spread across folds"),
    timeout: int = typer.Option(None, help="Per-experiment time limit in seconds"),
    tracker: str = typer.Option("auto", help="auto | mlflow | wandb | none"),
    out: Path = typer.Option(Path("runs"), help="Directory for run folders"),
    name: str = typer.Option(None, help="Name for the run folder: <timestamp>-<name> "
                                        "(default: taken from the first data source)"),
    experiment: str = typer.Option(None, "--experiment",
                                   help="MLflow experiment / W&B project to log to (default: named "
                                        "after the problem)"),
    seed: int = typer.Option(42),
    no_export: bool = typer.Option(False, "--no-export",
                                   help="Skip the final model export and registration"),
):
    """Solve a prediction problem: orbit builds and compares models, then reports.

    Examples:
      orbit run data.csv -t churn
      orbit run examples/house_prices/ --goal "Predict each house's sale price" --eval "RMSE between log prices"
      orbit run https://example.com/train.csv --goal "Predict survival"
    """
    from .planner import PlanError
    from .problem import run_name, sanitize_name, setup_task
    from .sandbox import DockerSandbox, check_docker
    from .sources import SourceError
    from .tracking import make_tracker, resolve_experiment

    if tracker not in TRACKERS:
        raise typer.BadParameter(f"choose one of {', '.join(TRACKERS)}", param_hint="--tracker")
    if time_col and time_order:
        raise typer.BadParameter("use either --time-col (a column gives the order) or --time-order "
                                 "(the file order is the time order), not both", param_hint="--time-order")
    try:
        experiment_name = resolve_experiment(experiment, data, name)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--experiment")
    sb_settings = SandboxSettings()
    ok, msg = check_docker(sb_settings)
    if not ok:
        console.print(f"[red]Sandbox unavailable:[/] {escape(msg)}")
        raise typer.Exit(1)
    llm = _llm()

    stem = (sanitize_name(name) if name else "") or run_name(data)
    run_dir = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}"
    try:
        task = setup_task(data, run_dir, goal=goal, eval_text=eval_text, target=target, folds=folds,
                          seed=seed, llm=llm, within=out, time_col=time_col, table=table,
                          time_order=time_order)
    except (SourceError, PlanError, ValueError) as e:
        console.print(f"[red]Could not set up the task:[/] {escape(str(e))}")
        raise typer.Exit(1)
    cfg = SearchConfig(max_steps=steps, time_budget=minutes * 60, token_budget=tokens,
                       n_drafts=drafts, exec_timeout=timeout or sb_settings.timeout,
                       cpus=sb_settings.cpus, seed=seed, patience=patience)
    trk = make_tracker(tracker_choice(tracker), experiment_name)
    trk.start(task, cfg, llm.model)

    scheme = f"{task.n_folds}-fold CV" if task.n_folds > 1 else "single split"
    if task.spec.time_col or task.spec.time_order:
        scheme = (f"{task.n_folds} forward-chaining folds" if task.n_folds > 1 else "single split") \
                 + f" ordered by {escape(task.spec.time_col or 'file order')}"
    console.print(f"[bold]orbit[/] {escape(str(run_dir))}  model={escape(llm.model)}  tracker={trk.name}"
                  + (f"  experiment={escape(experiment_name)}" if trk.name != "none" else ""))
    console.print(f"  data: {escape(Path(task.spec.data_path).name)}, target={escape(task.spec.target)}, "
                  f"metric={task.metric_name}, baseline={task.baseline:.4f}, validation={scheme}")
    if task.spec.extras:
        console.print(f"  extra tables: {escape(', '.join(Path(p).name for p in task.spec.extras))}")
    for note in task.spec.notes:
        console.print(f"  [dim]{escape(note)}[/]")
    info, warning = report_mod.setup_lines(task)
    for line in info:
        console.print(f"  [dim]{escape(line)}[/]")
    if warning:
        console.print(f"  [yellow]{escape(warning)}[/]")

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
    if trk.name == "mlflow" and not no_export:
        _export_model(task, journal, llm, DockerSandbox(sb_settings), cfg.exec_timeout, cfg.cpus)


def _model_line(model: dict) -> str:
    """One plain-text line about the exported model, from `results.collect()["model"]`."""
    if model.get("version") is not None:
        aliases = f", alias {', '.join(model['aliases'])}" if model.get("aliases") else ""
        return f"{model['name']} version {model['version']}{aliases}"
    if model.get("exported"):
        attempts = model.get("attempts") or 0
        return f"exported (experiment #{model.get('experiment')}, {attempts} attempt{'s' * (attempts != 1)})" \
               + (f", not registered: {model['message']}" if model.get("message") else ", not registered")
    return f"export failed: {model.get('error') or 'unknown error'}"


def _export_model(task: Task, journal: Journal, llm, sandbox, exec_timeout: int, cpus: float,
                  register: bool = True, model_name: str | None = None, force_champion: bool = False,
                  attempts: int | None = None) -> bool:
    """Export the best experiment as one verified model and register it in MLflow. Problems are
    printed as warnings and never raise; returns whether the export worked."""
    from .export import export_best
    from .registry import register_model

    console.print("\n[dim]Exporting the best model (refit on all labeled rows) ...[/]")
    try:
        res = export_best(task, journal, llm, sandbox, exec_timeout, cpus,
                          **({"max_attempts": attempts} if attempts else {}))
        if not res.ok:
            console.print(f"[yellow]Model export failed:[/] {escape(res.error or 'unknown error')}")
            for note in res.notes:
                console.print(f"  [dim]{escape(note)}[/]")
            return False
        size = res.model_path.stat().st_size / 1e6
        console.print(f"Model: exported (experiment #{res.node_id}, {res.attempts} attempt"
                      f"{'s' * (res.attempts != 1)}, {size:.1f} MB) -> {escape(str(res.model_path))}")
        if not register:
            return True
        reg = register_model(task.run_dir, res, model_name=model_name, force_champion=force_champion)
    except KeyboardInterrupt:
        console.print("\n[yellow]Model export interrupted; the run's results are intact.[/]")
        return False
    if reg.version is None:
        console.print(f"[yellow]Model registration failed:[/] {escape(reg.message)}")
        return True
    console.print(f"Registered model {escape(reg.name)} version {reg.version}"
                  + (f", alias {escape(', '.join(reg.aliases))}" if reg.aliases else ""))
    console.print(f"  [dim]{escape(reg.message)}[/]")
    return True


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
        if s.get("test_baseline") is not None:
            console.print(f"[dim]Baseline: validation {s['val_baseline']}, test {s['test_baseline']}; "
                          f"gain: validation {s['val_gain']}, test {s['test_gain']}[/]")
    for name, n, target in report_mod.split_rows(s):
        console.print(f"[dim]Split {escape(name)}: {n:,} rows, {escape(target)}[/]")
    if s["distribution_shift"]["flag"]:
        console.print(f"[yellow]The splits have different target distributions "
                      f"({escape(s['distribution_shift']['detail'])}): compare gains over baseline, "
                      "not raw scores.[/]")
    if not s["has_spread"]:
        console.print(f"[dim]{escape(report_mod.no_spread_caveat(s))}[/]")
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
    model = data.get("model")
    if model:
        console.print(f"Model: {escape(_model_line(model))}")
    if data["tracker"] and data["tracker"].get("url"):
        console.print(f"[dim]MLflow: {escape(data['tracker']['url'])}[/]")


@app.command()
def experiments():
    """List the MLflow experiments (one per problem) with their number of orbit runs."""
    import datetime

    from .tracking import configure_mlflow_env

    configure_mlflow_env()
    try:
        import mlflow

        client = mlflow.MlflowClient()
        rows = []
        for exp in client.search_experiments():
            runs, token = 0, None
            while True:
                page = client.search_runs([exp.experiment_id], max_results=1000, page_token=token)
                runs += sum("mlflow.parentRunId" not in r.data.tags for r in page)
                token = page.token
                if not token:
                    break
            if exp.name == "Default" and not runs:
                continue
            rows.append((exp, runs))
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Could not reach MLflow:[/] {escape(str(e)[:300])}")
        raise typer.Exit(1)
    table = Table(title="MLflow experiments")
    for col in ("name", "id", "runs", "last updated"):
        table.add_column(col, overflow="fold")
    for exp, runs in sorted(rows, key=lambda r: r[0].last_update_time or 0, reverse=True):
        updated = (datetime.datetime.fromtimestamp(exp.last_update_time / 1000).strftime("%Y-%m-%d %H:%M")
                   if exp.last_update_time else "")
        table.add_row(escape(exp.name), exp.experiment_id, str(runs), updated)
    console.print(table)


@app.command()
def export(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="A finished run folder"),
    model_name: str = typer.Option(None, "--model-name",
                                   help="Registered model name (default: the run's MLflow experiment)"),
    no_register: bool = typer.Option(False, "--no-register", help="Export only; do not use MLflow"),
    force_champion: bool = typer.Option(False, "--force-champion",
                                        help="Move the champion alias to the new version regardless of score"),
    attempts: int = typer.Option(3, min=1, help="Attempts at a verified export"),
):
    """Refit a run's best experiment on all labeled rows, verify it and register it in MLflow.

    The model lands in RUN_DIR/export/model.pkl; with a tracking.json it is also registered as a
    new version of an MLflow model, and the alias `champion` moves to it when its validation
    score is at least as good as the current champion's.

    Examples:
      orbit export runs/20261001-120000-wine
      orbit export runs/20261001-120000-wine --model-name wine-quality --force-champion
    """
    from .sandbox import DockerSandbox, check_docker

    sb_settings = SandboxSettings()
    ok, msg = check_docker(sb_settings)
    if not ok:
        console.print(f"[red]Sandbox unavailable:[/] {escape(msg)}")
        raise typer.Exit(1)
    llm = _llm()
    task, journal = Task.load(run_dir), Journal.load(run_dir / "journal.json")
    if not _export_model(task, journal, llm, DockerSandbox(sb_settings), sb_settings.timeout,
                         sb_settings.cpus, register=not no_register, model_name=model_name,
                         force_champion=force_champion, attempts=attempts):
        raise typer.Exit(1)


@app.command()
def models():
    """List the models registered in MLflow with the version that carries the alias champion."""
    from .tracking import configure_mlflow_env

    configure_mlflow_env()
    try:
        import mlflow

        client = mlflow.MlflowClient()
        rows = []
        for model in client.search_registered_models(max_results=1000):
            versions = client.search_model_versions(f"name = '{model.name.replace(chr(39), chr(39) * 2)}'")
            latest = max((int(v.version) for v in versions), default=None)
            champion = champion_tags = None
            if "champion" in (model.aliases or {}):
                champion = int(model.aliases["champion"])
                champion_tags = next((v.tags for v in versions if int(v.version) == champion), {})
            rows.append((model.name, latest, champion, champion_tags or {}))
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Could not reach MLflow:[/] {escape(str(e)[:300])}")
        raise typer.Exit(1)
    if not rows:
        console.print("No registered models yet. `orbit run` registers one after each search "
                      "(or use `orbit export RUN_DIR`).")
        return
    table = Table(title="Registered models")
    for col in ("name", "latest", "champion", "metric", "validation", "test"):
        table.add_column(col, overflow="fold")
    for name, latest, champion, tags in sorted(rows):
        table.add_row(escape(name), "" if latest is None else str(latest),
                      "" if champion is None else str(champion), escape(tags.get("orbit.metric", "")),
                      escape(_score(tags.get("orbit.val_score"))), escape(_score(tags.get("orbit.test_score"))))
    console.print(table)


@app.command()
def serve(
    model: str = typer.Argument(None, help="Registered model name or a models:/... URI"),
    version: int = typer.Option(None, "--version", help="Serve this version instead of the alias"),
    alias: str = typer.Option(None, "--alias", help="Serve the version with this alias (default: champion)"),
    port: int = typer.Option(None, "--port", help="Host port on 127.0.0.1 (default: 8090 or the next free one)"),
    name: str = typer.Option(None, "--name", help="Container name (orbit-serve- is prepended if missing)"),
    list_: bool = typer.Option(False, "--list", help="List running serving containers"),
    stop: str = typer.Option(None, "--stop", help="Stop and remove a serving container by name, or `all`"),
):
    """Serve a registered model with MLflow's scoring server in a dedicated Docker container.

    The container has the sandbox's pinned libraries plus MLflow, fetches the model from the
    tracking server and listens on 127.0.0.1 only (no authentication). Build the image once with
    `docker compose --env-file .env -f docker/compose.yaml build serve`.

    Examples:
      orbit serve retail-spend-90d
      orbit serve retail-spend-90d --version 2 --port 8100
      orbit serve --list
      orbit serve --stop all
    """
    from . import serve as srv

    try:
        if list_ or stop:
            _serve_manage(srv, list_, stop)
        elif not model:
            raise srv.ServeError("Give a model name (`orbit models` lists them), or use --list / --stop.")
        else:
            _serve_start(srv, model, version, alias, port, name)
    except srv.ServeError as e:
        console.print(f"[red]{escape(str(e))}[/]")
        raise typer.Exit(1)


def _serve_manage(srv, list_: bool, stop: str | None):
    containers = srv.list_containers()
    if stop:
        names = srv.stop_targets(containers, stop)
        srv.remove(names)
        console.print(f"Stopped {len(names)} serving container(s)" + (f": {escape(', '.join(names))}" if names else "."))
        return
    if not containers:
        console.print("No orbit serving containers are running.")
        return
    table = Table(title="Serving containers")
    for col in ("name", "model", "version", "endpoint", "up"):
        table.add_column(col, overflow="fold")
    for c in containers:
        table.add_row(escape(c.name), escape(c.model), escape(c.version),
                      f"http://127.0.0.1:{c.port}" if c.port else "", escape(c.uptime))
    console.print(table)


def _serve_start(srv, model: str, version: int | None, alias: str | None, port: int | None,
                 name: str | None):
    import os

    from .tracking import configure_mlflow_env

    uri = srv.model_uri(model, version, alias)
    image = srv.image_name()
    srv.check_docker(image)
    tracking = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking.startswith("http"):
        raise srv.ServeError("Serving needs an MLflow tracking server: set MLFLOW_TRACKING_URI "
                             "(see `orbit doctor`).")
    status = _check_mlflow()
    if "failed" in status:
        raise srv.ServeError(f"MLflow is not reachable at {tracking}.")
    configure_mlflow_env()
    import mlflow

    model_name, model_version = srv.resolve_in_registry(
        mlflow.MlflowClient(tracking_uri=tracking, registry_uri=tracking), uri)
    network, inner_uri = srv.mlflow_endpoint()
    host_port = srv.choose_port(port)
    container = srv.container_name(model_name, host_port, name)
    command = srv.run_command(container, uri, model_version, network, inner_uri, host_port, image,
                              srv.memory_limit())
    srv.start(command)
    console.print(f"Started {escape(container)}; loading {escape(uri)} ...")
    state = srv.wait_ready(host_port, lambda: srv.is_running(container))
    if state != "ready":
        tail = srv.logs_tail(container)
        srv.remove([container])
        why = "The container exited" if state == "exited" else "The server was not ready after 180 s"
        raise srv.ServeError(f"{why}; the container was removed." + (f" Last log lines:\n{tail}" if tail else ""))
    url = f"http://127.0.0.1:{host_port}"
    shown = f"{model_name}" + (f" version {model_version}" if model_version else "")
    console.print(f"[green]Serving[/] {escape(shown)} ({escape(uri)}) at {escape(url)}/invocations")
    console.print(f"  container: {escape(container)}   (127.0.0.1 only, no authentication)")
    console.print("  try it:")
    console.print(escape(f"    {srv.curl_command(url, srv.fetch_input_example(uri))}"), soft_wrap=True)
    console.print(f"  score a CSV: orbit predict {escape(url)} rows.csv --out predictions.csv")
    console.print(f"  stop:        orbit serve --stop {escape(container)}")


@app.command()
def predict(
    endpoint: str = typer.Argument(..., help="Endpoint URL (http://127.0.0.1:8090) or a served model name"),
    rows: Path = typer.Argument(..., exists=True, dir_okay=False, help="CSV of raw feature rows"),
    out: Path = typer.Option(Path("predictions.csv"), "--out", help="Where to write the predictions"),
    target_col: str = typer.Option(None, "--target-col", help="Column to leave out of the request"),
    keep_columns: bool = typer.Option(False, "--keep-columns", help="Repeat the input columns in the output"),
):
    """Score the rows of a CSV against a running `orbit serve` endpoint.

    Rows are sent in chunks of 500; the output keeps their order, with one `prediction` column.

    Examples:
      orbit predict retail-spend-90d new_customers.csv --out scored.csv
      orbit predict http://127.0.0.1:8090 new_customers.csv --keep-columns
    """
    from . import serve as srv

    try:
        url = endpoint if endpoint.startswith(("http://", "https://")) else srv.endpoint_for(
            endpoint, srv.list_containers())
        start = time.monotonic()
        count = srv.predict_csv(url, rows, out, target_col=target_col, keep_columns=keep_columns)
    except srv.ServeError as e:
        console.print(f"[red]{escape(str(e))}[/]")
        raise typer.Exit(1)
    console.print(f"Scored {count} rows against {escape(url)} in {time.monotonic() - start:.1f}s "
                  f"-> {escape(str(out))}")


def _score(text) -> str:
    try:
        return f"{float(text):.4f}"
    except (TypeError, ValueError):
        return ""


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
