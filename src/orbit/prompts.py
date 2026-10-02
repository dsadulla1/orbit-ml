"""Prompt templates and response parsing for the experiment agent.

The protocol is deliberately plain: the model replies with a short PLAN and one
fenced Python block. No native tool calling is required, so any chat model behind
an OpenAI-compatible endpoint works.
"""
import re

from .metrics import ID_COL, PRED_COL

LIBRARIES = "numpy, pandas, scipy, scikit-learn, xgboost, lightgbm"

SYSTEM = """You are an expert machine learning engineer running a series of experiments \
to maximise model quality on a tabular dataset. Each reply is ONE experiment: a complete, \
self-contained Python script that will be executed in a sandbox.

Sandbox rules:
- Available libraries: {libraries}. Nothing else; no pip installs, no internet access.
- The libraries are recent releases; use their current APIs:
  * LightGBM: `import lightgbm as lgb`. lgb.train() and fit() do not accept early_stopping_rounds, \
verbose or verbose_eval; use callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)] and put \
verbose=-1 in the params or constructor.
  * Early stopping needs a held-out validation set (valid_sets / eval_set); otherwise skip it. \
XGBoost takes early_stopping_rounds in the estimator's constructor, not in fit().
  * RMSE: sklearn.metrics.root_mean_squared_error (mean_squared_error has no squared= argument).
  * Format numbers with f-strings like f"{{score:.4f}}"; never mix f-strings with .format().
- Text columns must be encoded before use with scikit-learn, XGBoost or LightGBM (ordinal, one-hot or \
target encoding fitted on the training rows only; LightGBM also accepts the pandas category dtype). \
Missing values exist unless the profile says 0%: impute them, or use a model that accepts NaN \
(LightGBM, XGBoost, HistGradientBoosting); Ridge and LogisticRegression reject NaN.
- Read training data from data/train.csv (features + target column {target!r}).
- Read data/test.csv (features + an {id_col!r} column, no labels) and predict EVERY row.
- Write predictions.csv in the current directory with exactly two columns: {id_col!r}, {pred_col!r}.
- {prediction_rule}
- The script must finish within {timeout} seconds using at most {cpus} CPU cores \
(read the core count from the ORBIT_N_JOBS environment variable for n_jobs).
- The official score is computed outside the sandbox on hidden labels. You may print your own \
cross-validation score for reference; keep other output short.
- Use a fixed random seed. Do not try to read anything outside data/.

Reply format (exactly):
PLAN: <one to three sentences: what this experiment tries and why>
```python
<the full script>
```"""

DATA_SECTION = """## Task
Goal: {goal}
Metric: {metric} ({direction}). Baseline that ignores the features scores {baseline:.4f}.
{scoring}

## Training data profile
{profile}{extras}"""

HISTORY_SECTION = """## Experiments so far (official validation scores)
{history}"""

DRAFT = """{data}

{history}

Propose a NEW experiment that takes a meaningfully different approach from the ones above \
(different model family, preprocessing or feature engineering). Keep the first version simple \
and robust; correctness matters more than complexity."""

IMPROVE = """{data}

{history}

## Current best experiment (#{node_id}, score {score:.4f})
PLAN: {plan}
```python
{code}
```

Propose ONE focused improvement to this script (e.g. a feature, a hyperparameter change, \
a better model, ensembling or calibration). Explain the expected gain in the PLAN and return \
the full updated script."""

DEBUG = """{data}

## Broken experiment (#{node_id})
PLAN: {plan}
```python
{code}
```

## Error
```
{error}
```

Fix the root cause while keeping the experiment's intent, and check the rest of the script for \
the same mistake. Do not repeat a fix that has already failed; if unsure, switch to a simpler \
approach. Return the full corrected script."""


def prediction_rule(task) -> str:
    if task.metric.needs_proba:
        return (f"{PRED_COL!r} must be the predicted PROBABILITY (0..1) that "
                f"{task.spec.target!r} == {task.positive_label!r}.")
    if task.kind == "classification":
        return f"{PRED_COL!r} must be the predicted class label (one of {task.classes[:20]})."
    rule = f"{PRED_COL!r} must be the predicted numeric value of {task.spec.target!r}."
    if task.metric.needs_nonneg:
        rule += (" Predictions must be >= 0: clip them, and if you model log1p of the target, "
                 "convert back with expm1 before writing the file.")
    return rule


def system_prompt(task, timeout: int, cpus: float) -> str:
    return SYSTEM.format(libraries=LIBRARIES, target=task.spec.target, id_col=ID_COL,
                         pred_col=PRED_COL, prediction_rule=prediction_rule(task),
                         timeout=timeout, cpus=max(1, int(cpus)))


def data_section(task) -> str:
    direction = "higher is better" if task.metric.greater_is_better else "lower is better"
    goal = task.spec.goal or f"predict {task.spec.target!r} as accurately as possible"
    scoring = ""
    if task.n_folds > 1:
        scoring = (f"Scoring: your script is run {task.n_folds} times, each on a different "
                   f"train/test split of the same data (data/train.csv and data/test.csv change "
                   f"between runs); the score is the mean over the runs. Do not hard-code row "
                   f"counts or ids.")
    if task.spec.time_col:
        scoring = (scoring + " " if scoring else "") + (
            f"Validation and test rows are later in time (column {task.spec.time_col!r}) than the "
            f"training rows of the same run, so features must not rely on information from after a "
            f"row's own time.")
    extras = ""
    if task.spec.id_column:
        extras += (f"\n\nColumn {task.spec.id_column!r} only identifies rows; do not use it as a "
                   f"feature.")
    if task.extra_profile:
        extras += ("\n\n## Additional tables (read-only CSV files under data/extra/; join them to "
                   "the training data when they help)\n" + task.extra_profile)
    if task.docs_text:
        extras += "\n\n## Documentation supplied with the data\n" + task.docs_text
    return DATA_SECTION.format(goal=goal, metric=task.metric_name, direction=direction,
                               baseline=task.baseline, profile=task.profile, scoring=scoring,
                               extras=extras)


def history_section(nodes, max_items: int = 15) -> str:
    if not nodes:
        return HISTORY_SECTION.format(history="(none yet)")
    lines = []
    for n in nodes[-max_items:]:
        result = f"score {n.metric:.4f}" if n.metric is not None else "FAILED"
        lines.append(f"#{n.id} [{n.stage}] {result} :: {_one_line(n.plan)}")
    return HISTORY_SECTION.format(history="\n".join(lines))


def _one_line(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


_PLAN = re.compile(r"PLAN\**\s*:\**\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_PYTHON = ("", "python", "py", "python3")


def parse_response(text: str) -> tuple[str, str | None]:
    """Return (plan, code). Picks the longest fenced block if the model sent several."""
    text = text.replace("\r\n", "\n")
    blocks = []
    # Pair fences in order, so a closing fence is never mistaken for an opening one;
    # blocks in other languages (```bash, ```text) and an unterminated last fence are skipped.
    for part in text.split("```")[1:-1:2]:
        header, _, body = part.partition("\n")
        lang = header.split()[0].lower() if header.split() else ""
        if lang in _PYTHON and body.strip():
            blocks.append(body)
    code = max(blocks, key=len).strip() if blocks else None
    m = _PLAN.search(text)
    plan = m.group(1).strip() if m else text.split("```")[0].strip()
    return _one_line(plan, 600), code
