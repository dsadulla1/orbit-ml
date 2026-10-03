# Tracking, registry and serving with MLflow

How orbit uses MLflow: where runs go, what is recorded, how the best model is registered and how
to serve it. For the design behind it see [DESIGN.md](DESIGN.md) (Tracking, Model export, Serving).

Start the tracking server from the repository root (it listens on 127.0.0.1:5000 only):

```bash
docker compose --env-file .env -f docker/compose.yaml up -d mlflow
```

and set `MLFLOW_TRACKING_URI=http://localhost:5000` in `.env` (the template already does). MLflow
is the default tracker; `--tracker auto|mlflow|wandb|none` overrides it per run, and setting
`WANDB_API_KEY` makes W&B the default (`uv sync --extra wandb`).

## One experiment per problem

Each problem gets its own MLflow experiment, named after `--name` or the first data source (or set
`--experiment NAME` / `ORBIT_EXPERIMENT`), so the runs inside it are comparable. Running the same
problem again adds another run to the same experiment. Each parent run is tagged with its goal,
evaluation wording, tables, target, metric, validation scheme, a data fingerprint (row count and a
hash of the file) and the orbit version, and carries a short note, so attempts can be told apart.
Every experiment of the search is a nested child run with its parameters, score, per-fold scores,
duration, tokens, peak memory and CPU, and artifacts (script, prompt, response, logs).

`orbit experiments` lists the experiments with their number of runs and the time of the latest run. An experiment
deleted in the MLflow UI is restored when its name is used again.

## Traces

Every search step is recorded as a trace in the problem's experiment: the prompt, the model call
(including token counts), each fold's sandbox execution and scoring, with durations and errors, so
you can see how a step unfolded. Traces can be filtered by the tags `orbit.node`, `orbit.stage` and
`orbit.status`, and the HTML report links to them. They contain the prompts, which include the data
profile and any documentation text you supplied, just like the run artifacts do. Set
`ORBIT_TRACING=off` to switch them off. A tracing error is logged once and tracing is then disabled
for the run; the search carries on.

## System metrics and resource use

The parent run records host system metrics (CPU, memory and, when the NVIDIA driver is reachable,
GPU utilisation and memory) every 5 seconds, which is useful when a local LLM server shares the
machine. Every GPU the host's driver exposes is recorded, including ones the model server does not use. Each experiment's run gets `peak_mem_mb` and `cpu_pct_mean`, the sandbox container's peak
memory and mean CPU (100 = one busy core; for cross-validation the maximum and the mean over
folds), sampled with `docker stats` from outside the container, so you can see which experiment
was heavy or close to its memory limit. The HTML report shows peak memory per experiment. Set
`ORBIT_SYSTEM_METRICS=off` to skip the system metrics; the sampling interval follows
`MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`. Short experiments may finish before the first sample,
in which case the values are left empty.

## If MLflow is unreachable

orbit logs a warning per call and carries on: tracker problems never abort a search. It sets short
HTTP retry defaults (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`, `MLFLOW_HTTP_REQUEST_TIMEOUT=15`) so an
outage costs seconds per experiment, not minutes; set those variables yourself to override.

## Model registry

After a search with MLflow, `orbit run` exports the best experiment (see
[Model export](DESIGN.md#model-export)) and registers it as a new version of an MLflow model named
after the experiment (`orbit export RUN_DIR --model-name NAME` does the same for an existing run).
The version is logged on the run's parent run as a pyfunc model that takes raw feature rows and
returns the predicted probability (for probability metrics) or value, with the pickled model, the
helper module, an input example, a signature that only names the input columns (all optional, so
the model's own checks, below, answer requests with a missing column), the features file, and the
sandbox's pinned library versions as its requirements; it needs those versions to load. Version tags record
the run, experiment, metric, validation and test scores, baseline, goal, validation scheme and data
fingerprint.

The features file, `artifacts/orbit_features.json`, lists every feature column in order with its
type (`number` or `text`; booleans count as text) and, for a text column, its distinct values in the
training data (at most 500; beyond that `categories` is null and `truncated` is true). The serving
wrapper uses it to check requests, and `orbit predict` fetches it to check a CSV before sending it.
It holds the column names and the category values of your data, so anyone who can read the model
can read them; it contains no rows. A model registered before this file existed has none and is
served without these checks.

When the pipeline needs its own classes or functions, they travel in one module, `orbit_custom.py`,
which export checked statically and ships as code in the model's `code/` folder. Such a version has
the tag `orbit.custom_code=true`, and its description says that the model includes custom code from
the experiment and gives the file's sha256; read the file in the model's artifacts before loading
the model anywhere you trust. Loading the model imports that module.

The alias `champion` marks the version to serve. It moves to a new version when there is no
champion yet, when the new validation score is at least as good as the champion's, or when the two
cannot be compared (a different metric or data fingerprint); `--force-champion` always moves it.
Otherwise it stays, and the output says so. The model is refit on every labeled row, so the honest
performance estimate is the search's cross-validation score and held-out test score, which are
stored with the version, and not the refit's own predictions. `--no-export` skips this step; export
and registration problems print a warning and never change a run's results. Only MLflow runs
register models; W&B and `none` skip it. `orbit models` lists the registered models.

A failed export prints why in plain words, where to look and how to retry. `export/export.json`
holds the outcome (`error`, `explanation`, one note per attempt), and `export/attempt_N/` the
prompt, reply, script and stderr of each attempt. Retry with extra guidance, for example
`orbit export RUN_DIR --hint "Move the feature code into orbit_custom.py"` (at most 1000
characters); the first prompt of the retry also includes how the previous export failed.

`orbit export RUN_DIR --reuse` skips the LLM and Docker: when `export/export.json` says the export
worked and `model.pkl`, `sample_input.csv`, `sample_predictions.csv` (and `orbit_custom.py`, when
the model has one) are still the files it describes (size and sha256), it registers them again as a
new version, with the usual tags and the usual champion rule (`--model-name`, `--force-champion`
work as before). Use it after a registry problem, to register under another model name, or after
upgrading orbit, so that the new version carries what newer models carry (the features file). With
no verified export it says so and asks you to run without `--reuse`.

## Serving a model

Build the serving image once (it extends the sandbox image, so build that first):

```bash
docker compose --env-file .env -f docker/compose.yaml build serve
```

Then serve a registered model; `MODEL` is a name (the version with the alias `champion`), or a
`models:/...` URI, and `--version` or `--alias` pick another one:

```bash
uv run orbit serve my-model --port 8099
```

orbit checks that the model exists, starts the container, waits until it answers and prints the
endpoint, the container name and a `curl` command built from the model's saved input example. The
request body is `{"dataframe_records": [{...raw feature columns...}]}` and the response is
`{"predictions": [{"prediction": 55.47}]}`. Errors end with a short message after MLflow's own
prefix (`Failed to predict data '...'. Error: <message>`); `orbit predict` shows only the message.

| Request | Response |
| --- | --- |
| A feature column is missing | 400: `Missing feature column(s): a, b (2 of 19). Send every feature column; columns not needed (the target, orbit_id) are ignored.` (at most 10 names, then "and N more") |
| Text in a numeric column (`"fifty"` in `age`) | 400: `Column 'age' must be a number; row 2 has 'fifty'.` (row 1 is the first record; numbers written as text, `"nan"` and `""` are accepted, the last two as missing) |
| A category not seen in training | Scored as the model's pipeline treats any unknown value; one line per request in `docker logs`: `scored 1 value(s) not seen in training: job: astronaut (1 row)` |
| A missing value (`null`) | Scored; the model's pipeline handles it as it did in training |
| An extra column (the target, `orbit_id`, anything else) | Ignored |
| `{"dataframe_records": []}` | 400: `No rows to score: send at least one record.` |
| Malformed JSON | 400 from MLflow: `Invalid input. Ensure that input is a valid JSON formatted string.` |
| A content type other than JSON, CSV or Parquet | 415 from MLflow, listing the types it accepts |
| A very large request | No limit of its own: one request of 60,000 records (23 MB of JSON, the bank-marketing model with 19 columns, a local container) was scored in 3.6 seconds; `orbit predict` sends 500 rows at a time |

A request is never changed silently, with one exception: a category the model has not seen is
scored, because a pipeline's own handling of unknown values is part of the model. Use
`orbit predict --strict` to refuse such rows. Models registered before the features file existed
keep MLflow's own behaviour (a long 400 for a missing column, no other checks).

```bash
uv run orbit predict my-model new_rows.csv --out scored.csv   # or an http://127.0.0.1:PORT URL
uv run orbit serve --list                                    # running endpoints
uv run orbit serve --stop all                                # stop and remove them
```

`orbit predict` sends the rows in chunks of 500 and writes the `prediction` column in the input's
order (`--keep-columns` repeats the input columns first). Given a model name (or a `models:/` URI of
a served model) it first checks the CSV against the model's features file: an empty file, a missing
feature column or text in a numeric column stops it before anything is sent, with the same message
the server gives and the CSV's row number (the first data row is 1). Unseen categories and missing
values are scored and counted in the one-line summary, together with the extra columns that were
ignored; `--strict` stops on unseen categories too (showing up to 5 examples). With a plain URL the
server does the checking. `orbit serve --list` shows each container's uptime from its start time.

The container runs `mlflow models serve` with the sandbox's pinned libraries, so the model loads
exactly as it was fitted. It downloads the model from the tracking server over the Compose network
and mounts none of MLflow's data. It is not the experiment sandbox: it has network access and a
writable `/tmp`, but runs with all capabilities dropped, `no-new-privileges`, a memory limit
(`ORBIT_SERVE_MEMORY`, default 4g) and a process limit. The endpoint is published on 127.0.0.1 only
and has no authentication: put your own proxy in front of it before exposing it to a network.
MLflow's Host-header check applies to the container's requests to the tracking server, so orbit
points it at the server's address on the Compose network; if you restricted `MLFLOW_ALLOWED_HOSTS`,
keep that address allowed. See [Serving](DESIGN.md#serving).
