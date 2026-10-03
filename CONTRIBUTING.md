# Contributing to orbit

Thanks for helping. orbit is a small project with a few rules that protect its main promise:
scores you can trust from code you did not write. Please read [Ground rules](#ground-rules) before
changing anything under `src/orbit/`; the rest of this page is how to get set up and what a good
change looks like.

By contributing you agree that your work is released under the project's [MIT license](LICENSE).

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/) and, to run orbit end to end, Docker.

```bash
git clone https://github.com/dsadulla1/orbit-ml && cd orbit-ml
uv sync                      # add --extra wandb and/or --extra mcp if you work on those
uv run pytest                # no Docker, API key or network needed
```

The whole suite takes several minutes (the MLflow, export and serving tests are the slow ones).
While developing, run the part you touch:

```bash
uv run pytest tests/test_task.py -q
uv run pytest -k "export and not registry" -q
```

To try orbit against a real stack (optional):

```bash
cp .env.example .env         # then edit it; never commit .env
docker compose --env-file .env -f docker/compose.yaml build sandbox
docker compose --env-file .env -f docker/compose.yaml up -d mlflow
uv run orbit doctor          # LLM endpoint, Docker, sandbox image, tracker
uv run orbit run examples/data/wine.csv -t cultivar --steps 6
```

Run Compose from the repository root with `--env-file .env -f docker/compose.yaml`; otherwise it
reads `docker/.env` and falls back to defaults. A local model server is described in the README
(“Using a local model”) and `docker/README.md`.

## Repository map

| Path | What lives there |
|---|---|
| `src/orbit/cli.py` | Typer commands (`explore`, `run`, `results`, `export`, `serve`, ...) |
| `src/orbit/sources.py`, `planner.py`, `problem.py` | Turning files, folders, zips, workbooks and URLs plus a goal into a task |
| `src/orbit/task.py`, `metrics.py` | Splits, folds, hidden labels, baselines; the metric registry and all scoring |
| `src/orbit/agent.py`, `prompts.py`, `llm.py` | The search loop, the prompt contract and the chat-completions client |
| `src/orbit/sandbox.py`, `helpers.py` | `DockerSandbox` and the helper module copied into every sandbox data directory |
| `src/orbit/report.py`, `results.py`, `html_report.py` | Facts, the reviewed narrative, the results structure and the HTML report |
| `src/orbit/tracking.py` | MLflow / W&B logging, experiments, traces, system metrics |
| `src/orbit/export.py`, `pickle_scan.py`, `registry.py`, `orbit_model.py`, `serve.py` | Model export, verification, the model registry and serving |
| `docker/` | Sandbox and serving images, Compose file, pinned sandbox requirements |
| `docs/` | `DESIGN.md` (how it works and why), `mlflow.md`, `mcp.md` |
| `tests/` | `conftest.py` holds the scripted LLM and the fixtures described below |

Read [docs/DESIGN.md](docs/DESIGN.md) before a non-trivial change.

## Ground rules

These are not style preferences. A change that breaks one will not be merged.

1. **Generated code runs only through `DockerSandbox`.** Never add a user-facing way to execute it
   on the host. `tests/conftest.py::LocalSandbox` runs trusted fixture scripts in tests and must
   stay in `tests/`.
2. **Labels stay on the orchestrator side.** `hidden/` is never mounted into a sandbox, all scoring
   happens in `metrics.py` and `agent.py`, and test labels are scored once, for the final best
   experiment. Anything copied into a sandbox input directory (extra tables, helper files, export
   inputs) must not contain labels of rows that are still being validated or tested.
3. **The experiment sandbox gets no network and no credentials.** Tracking and registry calls
   happen in the orchestrator. Do not weaken `DockerSandbox.command`.
4. **Tracker and registry failures never abort a search.** Warn and carry on (see `tracking._safe`).
5. **The LLM protocol is plain text:** `PLAN:` plus one fenced Python block. Do not require native
   tool calling.
6. **Models are only loaded in the serving container.** Exports are verified in the sandbox and
   scanned, never unpickled on the host. A model's own classes and functions live in
   `orbit_custom.py`, which is AST-checked (`pickle_scan.check_module`) and shipped with the model.
7. **Library lists agree.** `docker/sandbox-requirements.txt`, `prompts.LIBRARIES` and the pins in
   `registry.py` must list the same libraries; a test checks the registry pins. Rebuild the images
   after changing them. Keep pandas 2 in the sandbox on purpose: generated code assumes it.
8. **Host code works with pandas 2 and 3.** Use `pd.api.types` checks, never `dtype == object`.
9. **Secrets stay out of the repository.** Never commit `.env`; `.env.example` is the template.
10. **Never put invented results in docs.** Quote only numbers from real runs, with the model and
    settings that produced them.

## Making a change

- **Keep changes small and focused**, one idea per commit. Write the message in the imperative
  (“Add ...”, “Fix ...”) with a short summary line and, when it helps, a body that says why.
- **Add tests with the change.** A bug fix gets a test that fails without it. Tests must not write
  into the repository: use `tmp_path`.
- **Update the docs** that describe the behaviour: `README.md` for anything a user types or sees,
  `docs/DESIGN.md` for how it works, `docs/mlflow.md` for tracking, registry and serving. New
  environment variables go in the README's configuration table and, commented, in `.env.example`.
- **Match the surrounding code**: naming, comment density and idiom. Comments explain why, not what.
  Line endings are LF (`.gitattributes` enforces it).
- **Format user-facing text carefully.** Anything from files, models or users that is printed with
  Rich must go through `rich.markup.escape`; anything put into HTML is escaped by the template.

### Testing conventions

- `ScriptedLLM` (in `tests/conftest.py`) replaces the LLM with canned replies and records prompts.
- `LocalSandbox` runs trusted fixture scripts in a subprocess. Do not use it for anything that is
  not a fixture you wrote.
- Docker and `docker stats` behaviour is tested with a fake `docker` executable placed on `PATH` or
  passed to `DockerSandbox`; HTTP behaviour with small servers on 127.0.0.1 in a thread.
- MLflow tests use a sqlite store in `tmp_path` and `chdir` into it, because MLflow writes
  artifacts to `./mlruns`. They never need a server.
- Leakage-sensitive code (splits, folds, export inputs, extra tables) needs tests that assert what
  a sandbox can and cannot see, not only that scores come out right.

### Prompts

The system prompt is a contract with small local models, so it should stay short and explicit.
Change it with evidence: run the same task a few times before and after and compare how many
experiments fail and why. Failures are in each run's `journal.json` and `nodes/NNN/stderr.txt`.
Say in the pull request what you ran (model, settings, number of runs); do not put such numbers in
the docs unless they come from real runs you describe.

## Recipes

**Add a metric.** Add it to `METRICS` in `metrics.py` (direction, whether it needs probabilities or
non-negative values), teach `match_metric` its wording if it has common names, make sure the
baseline in `task._best_constant` is the best constant for it, list it in the README, and test
scoring, matching and the prediction rule in the prompt.

**Add a library to the sandbox.** Add it to `docker/sandbox-requirements.txt` and `prompts.LIBRARIES`
and, if models may use it, to the pins in `registry.py` and the allow-list in `pickle_scan.py`;
rebuild both images and run the tests.

**Add a command.** Put it in `cli.py` as a thin wrapper around testable functions in a module,
escape printed text, handle expected failures with a short message and a non-zero exit code, add
CliRunner tests and a row to the README's command table.

**Add a tracker.** Implement the tracker interface in `tracking.py` (`start`, `log_node`, `finish`,
`log_report`, `span`), keep every call failure-tolerant, and register it in `make_tracker`.

## Before you open a pull request

- [ ] `uv run pytest` passes.
- [ ] New behaviour has tests; docs and `.env.example` are updated.
- [ ] No ground rule above is bent, and no labels, secrets or generated run folders are committed
      (`runs/`, `mlruns/`, `mlflow.db`, `.env` are git-ignored).
- [ ] The pull request says what changed and why, and how you checked it.

## Reporting bugs and security problems

Open an issue with the command you ran, what you expected, what happened, and the relevant lines
of the run's `journal.json` or `stderr.txt`. Do not paste API keys or private data.

If you find a way for generated code or a model file to escape its container, read labels, reach
the network from the experiment sandbox or reach MLflow's data, please report it privately (for
example through the repository's private vulnerability reporting) instead of in a public issue.
