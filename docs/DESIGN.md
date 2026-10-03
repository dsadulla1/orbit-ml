# Design notes

## Goals

1. Let an LLM run an end-to-end experiment loop on tabular data with no human in the loop.
2. Make the numbers trustworthy even when the generated code is buggy or "creative".
3. Never execute generated code outside an isolated container.
4. Work with any OpenAI-compatible model, including small local ones.

## Components

| Module | Responsibility |
|---|---|
| `sources.py` | Fetch/copy local files, folders, zips and URLs into the run folder (size caps, safe extraction); collect tables (CSV/TSV, and each `.xlsx` sheet converted to CSV) and documents; zips inside zips are followed within depth, archive-count and total-size budgets |
| `planner.py` | Choose the primary table, target and metric from the problem statement; LLM proposals are validated against the files and the metric registry. Also the single table reader `read_table`: it sniffs delimiter, encoding and decimal mark, and profiles use all rows or a seeded random sample (`load_for_profile`) |
| `problem.py` | Resolve sources, plan, and prepare the task in one call |
| `task.py` | Split data into test + k validation folds (or one split), hide labels, profile the data, compute a baseline |
| `helpers.py` | Optional helper for generated scripts, copied into every sandbox input directory as `orbit_helpers.py` (numpy and pandas only): consistent text encoding, NaN-filled frames, validated `predictions.csv`, and `Preprocessor`, a scikit-learn transformer that lets an exported pipeline take raw rows |
| `prompts.py` | System prompt (sandbox contract), draft/improve/debug/export templates, response parsing |
| `llm.py` | Chat-completions client for any OpenAI-compatible endpoint |
| `sandbox.py` | One locked-down `docker run` per experiment |
| `metrics.py` | Prediction-file validation, scoring and eval-wording matching (orchestrator side only; metrics come from a fixed registry, never from generated code) |
| `agent.py` | Search policy, experiment loop, budgets, final test scoring |
| `export.py` | Refit the best experiment on all labeled rows, save one pickled model, verify it by reloading it in the sandbox |
| `pickle_scan.py` | Allow-list scan of a pickle stream (no unpickling) run before an exported model is loaded, and the AST check of the model's `orbit_custom.py` |
| `orbit_model.py` | The MLflow pyfunc wrapper shipped inside every registered model (imports nothing from orbit) |
| `registry.py` | Log the exported model to MLflow, register a version, set tags, description and the `champion` alias |
| `serve.py` | Command builders and parsers behind `orbit serve` and `orbit predict`: model URIs, the serving container's `docker run`, Compose network discovery, readiness polling, listing and stopping, the CSV scoring client |
| `tracking.py` | MLflow / W&B logging (failures are logged, never fatal) |
| `report.py` | Facts → narrative → review (numbers and counts) → revise or fall back to the template → `report.md` |
| `results.py` | One JSON-friendly summary of a run, from the run folder or from an MLflow server |
| `html_report.py` | Standalone `report.html` (Jinja, inline SVG charts, no scripts or external assets) |
| `cli.py` | `explore` (same sources as `run`), `run`, `export`, `models`, `serve`, `predict`, `results`, `compare`, `experiments`, `summarize`, `doctor` |

## Data flow and label hiding

```
data.csv ──split──► test (20%)  ───────────────────────────────┐ held out, scored once
                    development (80%) ──k folds──► fold i:     │
                      input/fold_i/train.csv  (the other k-1 folds: features + target)
                      input/fold_i/test.csv   (fold i + the test rows: features + orbit_id, no labels)
                                     hidden/labels.csv (orbit_id, split, fold, target)
```

Next to every `train.csv` / `test.csv` pair orbit also writes `orbit_meta.json` (target, task kind,
metric, positive label, class names, id and time columns, fold count, extra table names) and
`orbit_helpers.py` (a copy of `helpers.py`); `extra/` holds additional tables when there are any.

Datasets of up to about 10,000 development rows use 5-fold cross-validation (`--folds N` to
change it); larger ones use a single 60/20/20 train / validation / test split (`--folds 1`),
laid out as `input/train.csv` and `input/test.csv`.

With `--time-col COL` the rows are stably sorted by that column (parsed as dates, or used as
numbers) and nothing is shuffled: the latest `test_size` share is the test split. A single split
trains on the earliest rows and validates on the next ones. With k folds the development rows are
cut into k + 1 contiguous blocks; fold i trains on all blocks before block i + 1 and validates on
it (an expanding window, via `TimeSeriesSplit`), so the first block is only ever training data.
Each fold's `test.csv` still holds its validation rows plus all test rows, and `labels.csv` keeps
its columns. Rows without a usable time value are dropped (noted in `spec.notes`), the original
column values are written unchanged, and the stratified fold reduction for rare classes is not
applied. The prompt tells the model that validation and test rows are later than the training rows.

`--time-order` gives the same layout without a column: the order key is the row's position in the
primary table after rows with a missing target are dropped, so nothing is sorted, no column is added
to `train.csv`, `test.csv` or the meta file, and the table's own content is untouched. The two
options are mutually exclusive, `load_labeled` (used by the export) applies the same ordering, and
the prompt says that validation and test rows come later in the file.

`prepare` also stores `split_stats` in `task.json`: for the training rows (of the last fold, the one
with the most history), each validation fold and the test split, the row count and the class shares
(plus the positive share for a binary target) or the mean and standard deviation of the target.
`test_baseline` is the best-constant score on the test labels, fitted on each fold's training rows
and averaged like the validation baseline. They are computed from data `prepare` already holds and
are not written into any sandbox input directory; older `task.json` files without them still load.

Setup also appends notes to `spec.notes` (shown in the setup output and the report): when the
default became a single split because of the row count; and, only for unordered validation of at
least 500 rows, a date-like column (name contains date, time, day, month, year or timestamp, or at
least 95% of a sample of up to 2,000 text values parse as dates; numeric columns need a name with
date, time or year; constant or two-valued columns and the target are ignored) and drift along the
file (five blocks of rows in file order; positive share differing by more than a factor of 2, more
than 0.05 and more than four standard errors, so chance in small blocks stays silent; or block means
of a numeric target differing by more than one standard deviation of the target).

The helper module is a convenience for the model, not a gate: scripts may ignore it and read the
CSVs directly. It contains no labels (the meta file lists class names only, never rows or paths).
`load()` builds numeric feature frames with one text-category mapping shared by the train and test
features (so a category seen only in the test rows cannot crash a script), and `Data.save` checks
the predictions against the metric (length, finiteness, probability range, class names; it clips
negatives for non-negative metrics) before writing `predictions.csv`. Scoring is unchanged and
still validates the file on the orchestrator side.

Only one fold's directory is mounted into a sandbox, read-only, so a script never sees another
fold's labels; `hidden/` is never mounted. A script is run once per fold and must predict every
row in that fold's `test.csv`. The orchestrator scores each fold's validation rows and ranks
experiments by the mean, reporting the spread across folds. Test rows are scored once, for the
final best experiment, from the average of its k fold predictions (the most common label for
class predictions). Choosing among many experiments on the validation score can overfit to it;
the gap between the best experiment's validation and test scores is shown in the report as a
check. Cross-validation costs k sandbox runs per experiment.

## Search policy

Each step chooses one action:

- **draft**: until `n_drafts` independent solutions exist, or while nothing works yet.
- **debug**: with probability `debug_prob` (always, if no experiment works yet), repair the
  newest failed leaf whose debug chain is shorter than `max_debug_depth`. The prompt includes
  the failing script and the tail of its stderr, or the reason its output was rejected.
- **improve**: make one focused change to the best experiment, or, with probability
  `explore_prob`, to one of the next two.

Every prompt carries a compact history (`#id [stage] score :: plan`) so the model avoids
repeating ideas. The search stops at the step, time or token budget, optionally when it stalls
(`--patience N`: N working experiments in a row, failed ones not counted, that do not beat the last
clear improvement by more than that experiment's fold spread, the sample standard deviation of its
fold scores, or 0 for a single split; several small gains therefore add up), after three consecutive
LLM/infrastructure errors, or on Ctrl-C. The best result so far is always scored and reported.

## Response protocol

Replies are free text with a `PLAN:` line and one fenced Python block. When several blocks
appear, the longest one wins. This needs no native tool calling, which small local models and
some servers handle unreliably. `<think>` sections from reasoning models are stripped.

## Sandbox

`docker run --rm --network none --read-only --tmpfs /tmp --cap-drop ALL
--security-opt no-new-privileges --cpus N --memory M --memory-swap M --pids-limit 512
--user <host uid> -v <exp>:/work -v <input>:/work/data:ro orbit-sandbox python solution.py`

- On timeout the container is killed by name.
- While a container runs, a daemon thread polls `docker stats --no-stream` for it and the peak
  memory and CPU go into `ExecResult`. This only observes the container from outside: it adds no
  mounts, options or privileges to the `docker run` command, and polling errors (including polls
  before the container exists) are ignored.
- Threads are capped through `OMP_NUM_THREADS` and related variables; scripts read
  `ORBIT_N_JOBS`.
- The image pins its libraries. pandas stays on 2.x because generated code mostly assumes
  pandas 2 behaviour.
- Optional `--runtime runsc` (gVisor) via `ORBIT_SANDBOX_RUNTIME`.

After each run, orbit deletes any symlinks the script left in its folder before reading or
uploading anything from there. Logs are streamed to temp files outside the mount, and only the
tail is kept. If orbit itself runs as root, experiments still run as the image's unprivileged
user.

Limits of this model: containers share the host kernel; the orchestrator needs Docker daemon
access, which is root-equivalent on the host; the writable experiment folder has no disk quota
(put `runs/` on a volume you can afford to fill); and `/dev/shm` keeps Docker's 64 MB default,
which can constrain joblib's process-based parallelism.

## Model export

`export.export_best` turns the best experiment into one reusable model, after scoring is done.
`task.prepare_export_input` writes `runs/<id>/export/input/`: `train.csv` with every labeled row of
the primary table (read again from the user's source file, never from `hidden/`), `test.csv` with
up to 200 of those rows as features only plus `orbit_id`, the extra tables, `orbit_meta.json` and
`orbit_helpers.py`. These rows include former validation and test rows; that is fine because
scoring has finished and the labels come from the user's own file.

The LLM rewrites the best script so that it trains on all rows and builds one scikit-learn
`Pipeline` (or estimator) that takes raw feature rows and does its own preprocessing
(`orbit_helpers.Preprocessor` handles text encoding and imputation), pickles it with plain
`pickle` (protocol 4; joblib files are not supported) and writes `predictions.csv` for the check
rows. Anything the script did to its predictions (a target transform, clipping) must live inside
the pickled object, which `TransformedTargetRegressor` and `orbit_helpers.ClipOutput` /
`expm1_nonneg` provide; for metrics that reject negative values the verifier and the served model
clip at 0. For probability metrics the saved classifier is fitted on `d.y_train_binary` or on the
original labels. The class-label rule, identical in the verifier and in `orbit_model.py`: use the
column of the positive label when it is among `classes_` (compared as text, so `1.0` equals `1`);
otherwise, when `classes_` is exactly 0 and 1 (ints, floats or bools), use the column of `1`;
otherwise fail with an error naming `classes_` and the positive label. Each attempt (up to three,
later ones see the previous script and the rows that differ, and `orbit export --hint TEXT` adds the
user's guidance) is accepted only if:

1. the script exits cleanly and writes a regular `model.pkl` of at most 500 MB and valid predictions;
2. `pickle_scan.scan` finds no disallowed global in the file. It walks the opcodes without
   unpickling and allows only numpy, pandas, scipy, scikit-learn, LightGBM, XGBoost, joblib,
   `orbit_helpers` and a few standard-library constructors, minus names that read files or run
   code. This is defence in depth, not a guarantee: a pickle is code, and an allow-listed library
   may still contain a harmful callable. The real boundary is that the model is only ever loaded in
   a dedicated container;
   Classes and functions the pipeline needs itself (custom transformers, feature engineering) go in
   one module file, `orbit_custom.py`, written next to `model.pkl`; the script imports it, so the
   pickle refers to `orbit_custom.Name` and never to `__main__`. The scan allows the top-level
   module name `orbit_custom` exactly (not submodules or prefixes) and still applies its deny-list
   to the names inside it. The file itself must be a regular file of at most 200 KB and passes
   `pickle_scan.check_module`, a static `ast` check: it may import only numpy, pandas, scipy,
   scikit-learn, LightGBM, XGBoost, `orbit_helpers` and standard maths, text and typing modules; it
   may not use star or relative imports, `eval`, `exec`, `compile`, `open`, `__import__`, `input`,
   `globals`, `locals`, `vars`, `getattr`/`setattr` with a computed or dunder name, attributes such
   as `__globals__`, `__subclasses__` or `__mro__`, writes to `__dict__`, file-reading attributes
   such as `read_csv` or `load`, or its own `__reduce__`. This is defence in depth as well. The
   verification directory gets a copy of the module so the fresh container can import it;
3. an orbit-written verifier, run in a fresh sandbox container, loads `model.pkl`, predicts the raw
   check rows and reproduces the script's `predictions.csv` (`rtol=1e-5` for numbers, equal labels
   for class predictions). This shows the saved object works from raw rows and is not in-memory state.

The accepted model is copied to `export/model.pkl` next to `sample_input.csv`,
`sample_predictions.csv` and `export.json` (outcome and attempt notes, no code or data), and the
accepted `orbit_custom.py` when the model uses one (`export.json` records `custom_module`, its size
and sha256). Export problems never raise: the result carries `ok=False`, the raw error and
`explanation`, a plain-words reason with the fix (`export.explain_failure`), also stored in
`export.json`. The CLI prints the explanation, where the attempts are and an example
`orbit export RUN_DIR --hint "..."` for the failure. A re-export of a run whose export failed puts
the earlier explanation and error into its first prompt.

`orbit run` (MLflow tracker, unless `--no-export`) and `orbit export RUN_DIR` call `export_best`
and then `registry.register_model`; a failed export is a warning, the run's results stay intact.
`export/registry.json` records the outcome and `results.collect` adds it, with `export.json`, as
the `model` entry shown by `orbit results` and the HTML report. `orbit export RUN_DIR --reuse`
skips `export_best`: `export.load_verified_export` checks the recorded size and sha256 of
`model.pkl` (and `orbit_custom.py`) and hands the files to `register_model` again.

## Serving

`orbit serve` runs a registered model with MLflow's own scoring server (`mlflow models serve`, which
is uvicorn-based on Linux in the pinned MLflow version) inside a dedicated container started from
`orbit-serve:latest` (`docker/serve.Dockerfile`): the sandbox image plus `mlflow` installed under
constraints frozen from the sandbox's pins, so a model unpickles with the libraries it was fitted
and verified with. The container only ever loads models that passed export verification.

Where it runs, and why:

- **Not on the host.** The model is a pickle; the host may lack its libraries, and orbit does not
  load generated pickles outside a container.
- **Not in the MLflow server's container.** Its image lacks the libraries, and that container holds
  the tracking database and every artifact.
- **Not in the sandbox.** The sandbox is built for untrusted code (no network, read-only root);
  serving needs the network and a writable temp directory, and a separate image keeps those
  settings out of the sandbox. `DockerSandbox` is unchanged.

The container joins the Compose network the MLflow container is on (found by inspecting
`orbit-mlflow-1`, falling back to `orbit_default`) and downloads the model through the tracking
server's artifact proxy (`--serve-artifacts`), so it mounts none of MLflow's volume. It reaches the
server by IP address on that network, since MLflow's Host-header check accepts private addresses by
default but not the service name. Hardening that costs nothing is kept: `--cap-drop ALL`,
`no-new-privileges`, a memory limit (`ORBIT_SERVE_MEMORY`, default 4g), `--pids-limit 512`, a tmpfs
`/tmp`, a non-root user. The root filesystem stays writable and the network stays on.

MLflow's scoring server has no authentication, so the port is published as
`127.0.0.1:PORT:8080` and never on another interface. Containers are named `orbit-serve-<model>-<port>`
and carry the labels `orbit.serve=1`, `orbit.model` (URI) and `orbit.version`; `--list` and
`--stop` only consider containers that have both the name prefix and the label, and `orbit predict`
finds a served model by the same labels, and `--list` computes the uptime from each container's
`State.StartedAt` (`docker inspect`), not from Docker's rounded text. `orbit predict` is a plain
`urllib` client: chunks of 500 rows as `dataframe_records`, NaN as null, a 60 s timeout per
request, the server's error message (at most 300 characters) shown on failure.

Requests are checked by the model itself. The registered signature lists the columns as optional
and untyped, so MLflow's schema enforcement, whose errors are long, passes everything through;
`orbit_model.py` then checks the request against `orbit_features.json` (column names, number or
text, the training categories of text columns; built by `registry.build_features` from the export
input's `train.csv`) and answers 400 with a short message, or logs unseen categories at WARNING.
It sets small pandas display options at load time because MLflow prints the request in front of
every error. `orbit predict` fetches the same file with the MLflow SDK and repeats the checks
before sending (`serve.check_rows`), so the two stay in step through their tests; a model
without the file is neither checked by the wrapper nor by the client.

## Reports and number verification

`report.facts()` derives everything worth saying from the journal: the best experiment, its
validation and test scores, gain over the baseline and over the first working experiment, the
runner-up, failure and debug counts, and time and tokens. The LLM is asked to write a
150-word narrative using only those facts. `check_numbers()` then extracts every number in the
narrative and checks that it matches a fact at the precision written (percentages are checked
against fractions too), and `check_claims()` checks written counts of succeeded and failed
experiments ("only one experiment succeeded out of ten") against the run. A narrative with any
problem goes back to the LLM once with the problems listed; if the revision still fails review,
the deterministic template is used instead, and the report says which happened. Without an LLM,
the template produces the same structure.

The facts also hold the per-split rows and target rates, `val_baseline` and `test_baseline`, the
gains over each (`val_gain`, `test_gain`), `has_spread` (false for a single validation split) and
`distribution_shift`. Its `flag` compares the test split with the mean over the validation folds,
and only when every split has at least 30 rows: a binary target's positive share must differ by
more than a factor of 1.5 and by more than 0.03; for several classes, some class share by more than
0.10; for a numeric target, the means by more than half the validation standard deviation.
`folds_differ` applies the same rule between the validation folds, which forward-chaining folds can
trip by themselves. With the flag set the template and the narrator prompt compare gains over each
baseline and explain the gap through the different rates; `review` rejects a narrative that calls
the gap overfitting or leakage (unless it negates that), which goes through the same revise-once,
then-template path. A single validation split is reported as having no spread. `report.md`, the
HTML report and `orbit results` show the splits, the test baseline and gain, and a warning when the
flag is set.

## Tracking

- **MLflow**: one experiment per problem (`--experiment`, then `ORBIT_EXPERIMENT`, then the sanitised
  `--name` or the first data source's name), because scores are only comparable within one problem;
  re-running a problem adds a parent run to the same experiment. If an experiment of that name was
  deleted earlier, the tracker restores it and retries once instead of disabling tracking. One parent run per orbit run; one nested run per experiment with
  params (stage, parent, debug depth), metrics (val score, duration, tokens) and artifacts (script,
  prompt, response, logs). The parent run gets the per-step score curve, the best
  validation and test scores, `journal.json` and `report.md`. It is tagged `orbit.goal`,
  `orbit.eval`, `orbit.primary_table`, `orbit.extra_tables`, `orbit.target`, `orbit.metric`,
  `orbit.validation`, `orbit.data_fingerprint` (rows and a sha256 prefix of the primary table),
  `orbit.version`, and carries a Markdown note (`mlflow.note.content`). `tracking.json` in the run
  folder records the URI, experiment name and id, and run id.
- **Traces** (MLflow): one trace per search step, in the problem's experiment. The root span
  `orbit.step` (node id, stage, parent, final status, validation score, last error line) has child
  spans `llm` (model, token counts), `sandbox` (one per fold: seconds, ok, timed out) and `score`
  (one per fold). Traces are tagged `orbit.node`, `orbit.stage` and `orbit.status`. The OpenAI SDK
  is auto-instrumented, so each chat completion (the report narrator's too) shows up as a span
  nested inside orbit's. The agent only calls `tracker.span(name, **attributes)`, a context manager
  that is a no-op for the other trackers; a failure of the tracing machinery logs one warning and
  turns tracing off for the run, while exceptions from the traced code pass through unchanged.
  `ORBIT_TRACING=off` disables it. Spans are recorded by the orchestrator only; traces contain the
  prompts, like the artifacts.
- **Resource usage** (MLflow): the parent run is started with MLflow's system metrics (CPU,
  memory, and GPU utilisation and memory through `nvidia-ml-py` when the driver is reachable; the
  sampling interval defaults to 5 s). If starting with them fails, the run starts without and a
  warning is logged. `ORBIT_SYSTEM_METRICS=off` disables them. Host metrics cannot tell
  experiments apart, so each sandbox run is also sampled (see Sandbox); `Node.peak_mem_mb`
  (maximum over folds) and `Node.cpu_pct` (mean of the per-fold means) are logged on the
  experiment's run as `peak_mem_mb` and `cpu_pct_mean`, kept in `journal.json`, and shown in the
  HTML report. Both are `None` when no sample was taken.
- **W&B**: one run with per-step metrics, a summary table and the scripts as artifacts.

- **Model registry** (MLflow): `registry.register_model` resumes the search's parent run (the one in
  `tracking.json`) and logs `export/model.pkl` as a pyfunc model "from code": `orbit_model.py`
  (`OrbitModel`) with the pickle and `orbit_meta.json` as artifacts and `orbit_helpers.py` as code,
  so the file runs where orbit is not installed. `load_context` unpickles with plain `pickle`; that
  is acceptable because the file passed the allow-list scan and the sandbox reload check at export
  time and is only loaded inside the serving container. A model with an `orbit_custom.py` also
  ships it as code (next to `orbit_helpers.py`, which MLflow puts on `sys.path` before unpickling),
  tags the version `orbit.custom_code=true` and names the file's sha256 in the version description. `predict` ignores the target, id column and
  `orbit_id` if present and returns a Series named `prediction`. Logging itself never unpickles the
  model, since the host may lack its libraries and generated pickles are not loaded outside a
  container. `pip_requirements` are the sandbox's pins (`registry.PINNED`, tested against
  `docker/sandbox-requirements.txt`, with `xgboost-cpu` as `xgboost`) plus the installed MLflow.
  The signature lists the sample's columns as required but untyped: MLflow's inferred types reject
  a null in an integer column and truncate 1.5 in one to 1, while the pipeline handles both.
  The registered model is named by `--model-name`, else the experiment. Each version is tagged
  `orbit.run_id`, `orbit.experiment`, `orbit.metric`, `orbit.val_score`, `orbit.val_std`,
  `orbit.test_score`, `orbit.baseline`, `orbit.goal`, `orbit.validation`,
  `orbit.data_fingerprint` and has a Markdown description. The alias `champion` moves to the new
  version when absent, forced, when the champion's metric or data fingerprint differs (not
  comparable), or when the new validation score is at least as good (direction from
  `metrics.METRICS`); otherwise it stays. `register_model` never raises for MLflow problems; it
  returns `version=None` and a message.

The sandbox never talks to a tracker; it has no network and no credentials.

## Possible extensions

- Run several experiments in parallel (the policy is sequential today).
- Cross-validated scoring inside the orchestrator for small datasets.
- Resume a run from `journal.json`.
- Seed a search from a user-provided script.
