# Design notes

## Goals

1. Let an LLM run an end-to-end experiment loop on tabular data with no human in the loop.
2. Make the numbers trustworthy even when the generated code is buggy or "creative".
3. Never execute generated code outside an isolated container.
4. Work with any OpenAI-compatible model, including small local ones.

## Components

| Module | Responsibility |
|---|---|
| `task.py` | Split data into train / validation / test, hide labels, profile the data, compute a baseline |
| `prompts.py` | System prompt (sandbox contract), draft/improve/debug templates, response parsing |
| `llm.py` | Chat-completions client for any OpenAI-compatible endpoint |
| `sandbox.py` | One locked-down `docker run` per experiment |
| `metrics.py` | Prediction-file validation and scoring (orchestrator side only) |
| `agent.py` | Search policy, experiment loop, budgets, final test scoring |
| `tracking.py` | MLflow / W&B logging (failures are logged, never fatal) |
| `report.py` | Facts → narrative → number verification → `report.md` |
| `cli.py` | `explore`, `run`, `compare`, `summarize`, `doctor` |

## Data flow and label hiding

```
data.csv ──split──► train (60%)  ──► input/train.csv  (features + target)
                    val   (20%)  ─┐
                    test  (20%)  ─┴► input/test.csv   (features + orbit_id, shuffled together)
                                     hidden/labels.csv (orbit_id, split, target)
```

`input/` is mounted read-only into the sandbox; `hidden/` is never mounted. Scripts must
predict every row in `test.csv`. The orchestrator scores the validation rows after each
experiment and the test rows once, for the final best experiment. Choosing among many
experiments on the validation split can overfit to it; the gap between the best experiment's
validation and test scores is shown in the report as a check.

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

Limits of this model: containers share the host kernel, and the orchestrator needs Docker
daemon access, which is root-equivalent on the host.

## Reports and number verification

`report.facts()` derives everything worth saying from the journal: the best experiment, its
validation and test scores, gain over the baseline and over the first working experiment, the
runner-up, failure and debug counts, and time and tokens. The LLM is asked to write a
150-word narrative using only those facts. `check_numbers()` then extracts every number in the
narrative and checks that it matches a fact at the precision written (percentages are checked
against fractions too). Unmatched numbers are flagged in the report. Without an LLM, a
deterministic template produces the same structure.

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
