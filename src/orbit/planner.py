"""Turn a natural-language problem and a set of data files into a concrete task.

Given tables and documents from `sources.resolve`, decide which table holds the labels, which
column is the target, and which supported metric matches the evaluation wording. Explicit
choices (--target, --metric) are never second-guessed. When something is missing, an LLM
proposes it as a JSON block (plain text, no tool calling) and orbit validates every field
against the real files and the metric registry before using it. Metrics are always chosen from
`METRICS`: orbit never executes generated scoring code on the host.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .metrics import METRICS, match_metric

PROFILE_ROWS = 20_000
DOC_CHARS = 3_000
DOCS_TOTAL_CHARS = 8_000

SYSTEM = """You set up machine-learning tasks. You are given a problem statement, how it will be \
evaluated, summaries of the data tables and any documentation. Decide which table holds the \
labels to predict, which column is the target, and which supported metric matches the \
evaluation. Answer with ONE ```json block and nothing else."""

USER = """## Problem
{goal}

## Evaluation
{eval}

## Supported metrics
{metrics}

## Tables
{tables}
{docs}
## Already decided by the user
{fixed}

Reply with exactly this JSON (use null when a field is already decided or does not apply):
```json
{{"primary_table": "<file name of the table that contains the target column>",
  "target": "<column to predict>",
  "metric": "<one supported metric name, or null if none of them fits the evaluation>",
  "id_column": "<column that only identifies rows and must not be used as a feature, or null>",
  "reasoning": "<one sentence>"}}
```"""


class PlanError(ValueError):
    """The task could not be set up; the message tells the user what to specify."""


@dataclass
class Plan:
    primary: Path
    target: str
    metric: str | None
    id_column: str | None = None
    extras: list[Path] = field(default_factory=list)
    docs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    sep = "\t" if ".tsv" in path.name.lower() else ","
    return pd.read_csv(path, sep=sep, nrows=nrows)


def describe_table(path: Path, df: pd.DataFrame, max_cols: int = 40, title: str | None = None) -> str:
    lines = [f"### {title or path.name}  ({len(df):,}{'+' if len(df) >= PROFILE_ROWS else ''} rows, "
             f"{df.shape[1]} columns)", "column | dtype | missing% | example values"]
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        ex = ", ".join(map(str, s.dropna().unique()[:3]))[:60]
        lines.append(f"{c} | {s.dtype} | {100 * s.isna().mean():.0f} | {ex}")
    if df.shape[1] > max_cols:
        lines.append(f"... and {df.shape[1] - max_cols} more columns")
    return "\n".join(lines)


def read_docs(docs: list[Path], per_doc: int = DOC_CHARS, total: int = DOCS_TOTAL_CHARS) -> str:
    parts, used = [], 0
    for p in docs:
        text = p.read_text(encoding="utf-8", errors="replace").strip()[:per_doc]
        if used + len(text) > total:
            break
        used += len(text)
        parts.append(f"### {p.name}\n{text}")
    return "\n\n".join(parts)


def _parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        raise PlanError("the planner reply contained no JSON object")
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise PlanError(f"the planner reply was not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise PlanError("the planner reply was not a JSON object")
    return data


def _check(proposal: dict, frames: dict[str, pd.DataFrame], target, metric, eval_text) -> dict:
    """Validate an LLM proposal against the real tables; raises PlanError with a fixable message."""
    primary = proposal.get("primary_table")
    if primary not in frames:
        raise PlanError(f"primary_table {primary!r} is not one of {sorted(frames)}")
    tgt = target or proposal.get("target")
    if tgt not in frames[primary].columns:
        raise PlanError(f"target {tgt!r} is not a column of {primary}: {list(frames[primary].columns)[:30]}")
    name = metric
    if name is None:
        name = proposal.get("metric")
        if name is not None and name not in METRICS:
            raise PlanError(f"metric {name!r} is not supported; choose from {sorted(METRICS)} or null")
    ident = proposal.get("id_column")
    if ident is not None and ident not in frames[primary].columns:
        ident = None
    return {"primary": primary, "target": tgt, "metric": name, "id_column": ident,
            "reasoning": str(proposal.get("reasoning") or "")[:300]}


def _ask(llm, goal, eval_text, frames, paths, docs, target, metric) -> dict:
    fixed = []
    if target:
        fixed.append(f"target column: {target}")
    if metric:
        fixed.append(f"metric: {metric}")
    prompt = USER.format(
        goal=goal or "(not given: infer a sensible prediction task from the data)",
        eval=eval_text or "(not given)",
        metrics=", ".join(sorted(METRICS)),
        tables="\n\n".join(describe_table(paths[n], f) for n, f in frames.items()),
        docs=(f"\n## Documentation\n{read_docs(docs)}\n" if docs else ""),
        fixed="\n".join(fixed) or "(nothing)")
    error = None
    for _ in range(2):
        text = llm.complete(SYSTEM, prompt + (f"\n\nYour previous reply was rejected: {error}" if error else "")).text
        try:
            return _check(_parse_json(text), frames, target, metric, eval_text)
        except PlanError as e:
            error = str(e)
    raise PlanError(f"could not work out the task from the data ({error}); "
                    "pass --target (and --metric) explicitly")


def plan(tables: list[Path], docs: list[Path], goal: str | None = None, eval_text: str | None = None,
         target: str | None = None, metric: str | None = None, llm=None) -> Plan:
    if metric is not None:
        metric = match_metric(metric) or metric
        if metric not in METRICS:
            raise PlanError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    elif eval_text:
        metric = match_metric(eval_text)

    paths = {p.name: p for p in tables}
    if len(paths) != len(tables):
        raise PlanError("two tables share a file name; rename one so they can be told apart")
    frames = {n: read_table(p, PROFILE_ROWS) for n, p in paths.items()}

    holders = [n for n, f in frames.items() if target and target in f.columns]
    needs_llm = target is None or len(holders) != 1 or (metric is None and eval_text)
    notes = []
    if needs_llm and llm is not None:
        choice = _ask(llm, goal, eval_text, frames, paths, docs, target, metric)
        if choice["reasoning"]:
            notes.append(f"setup: {choice['reasoning']}")
    elif target is None:
        raise PlanError("no --target given and no LLM available to infer it")
    elif not holders:
        raise PlanError(f"target {target!r} is not a column of any table "
                        f"({'; '.join(f'{n}: {list(f.columns)[:8]}' for n, f in frames.items())})")
    else:
        pick = max(holders, key=lambda n: ("train" in n.lower(), len(frames[n])))
        if len(holders) > 1:
            notes.append(f"setup: {len(holders)} tables have column {target!r}; using {pick}")
        choice = {"primary": pick, "target": target, "metric": metric, "id_column": None}
    choice["metric"] = choice["metric"] or metric
    if eval_text and choice["metric"] is None and not match_metric(eval_text):
        raise PlanError(f"cannot map the evaluation {eval_text!r} to a supported metric "
                        f"({', '.join(sorted(METRICS))}); pass --metric")

    primary = choice["primary"]
    wanted = set(frames[primary].columns) - {choice["target"]}
    extras = []
    for n, f in frames.items():
        if n == primary:
            continue
        if set(f.columns) == wanted or set(f.columns) == set(frames[primary].columns):
            notes.append(f"ignored {n}: same columns as {primary}, so it has no extra information")
        else:
            extras.append(paths[n])
    return Plan(primary=paths[primary], target=choice["target"], metric=choice["metric"],
                id_column=choice["id_column"], extras=extras, docs=docs, notes=notes)
