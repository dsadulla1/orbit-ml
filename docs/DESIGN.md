# Design notes

## Goals

1. Let an LLM run an end-to-end experiment loop on tabular data with no human in the loop.
2. Make the numbers trustworthy even when the generated code is buggy or "creative".
3. Never execute generated code outside an isolated container.
4. Work with any OpenAI-compatible model, including small local ones.

## Components

| Module | Responsibility |
|---|---|
| `sources.py` | Fetch/copy local files, folders, zips and URLs into the run folder (size caps, safe extraction); collect tables and documents |
| `planner.py` | Choose the primary table, target and metric from the problem statement; LLM proposals are validated against the files and the metric registry |
| `problem.py` | Resolve sources, plan, and prepare the task in one call |
| `task.py` | Split data into test + k validation folds (or one split), hide labels, profile the data, compute a baseline |
| `prompts.py` | System prompt (sandbox contract), draft/improve/debug templates, response parsing |
| `llm.py` | Chat-completions client for any OpenAI-compatible endpoint |
| `sandbox.py` | One locked-down `docker run` per experiment |
| `metrics.py` | Prediction-file validation, scoring and eval-wording matching (orchestrator side only; metrics come from a fixed registry, never from generated code) |
| `agent.py` | Search policy, experiment loop, budgets, final test scoring |
| `tracking.py` | MLflow / W&B logging (failures are logged, never fatal) |
| `report.py` | Facts → narrative → review (numbers and counts) → revise or fall back to the template → `report.md` |
| `results.py` | One JSON-friendly summary of a run, from the run folder or from an MLflow server |
| `html_report.py` | Standalone `report.html` (Jinja, inline SVG charts, no scripts or external assets) |
| `cli.py` | `explore`, `run`, `results`, `compare`, `summarize`, `doctor` |

## Data flow and label hiding

```
data.csv ──split──► test (20%)  ───────────────────────────────┐ held out, scored once
                    development (80%) ──k folds──► fold i:     │
                      input/fold_i/train.csv  (the other k-1 folds: features + target)
                      input/fold_i/test.csv   (fold i + the test rows: features + orbit_id, no labels)
                                     hidden/labels.csv (orbit_id, split, fold, target)
```

Datasets of up to about 10,000 development rows use 5-fold cross-validation (`--folds N` to
change it); larger ones use a single 60/20/20 train / validation / test split (`--folds 1`),
laid out as `input/train.csv` and `input/test.csv`.

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
repeating ideas. The search stops at the step, time or token budget, after three consecutive
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

## Tracking

- **MLflow**: one parent run per orbit run; one nested run per experiment with params
  (stage, parent, debug depth), metrics (val score, duration, tokens) and artifacts (script,
  prompt, response, logs). The parent run gets the per-step score curve, the best
  validation and test scores, `journal.json` and `report.md`.
- **W&B**: one run with per-step metrics, a summary table and the scripts as artifacts.

The sandbox never talks to a tracker; it has no network and no credentials.

## Possible extensions

- Run several experiments in parallel (the policy is sequential today).
- Cross-validated scoring inside the orchestrator for small datasets.
- Resume a run from `journal.json`.
- Seed a search from a user-provided script.
