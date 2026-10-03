"""Turn a natural-language problem and a set of data files into a concrete task.

Given tables and documents from `sources.resolve`, decide which table holds the labels, which
column is the target, and which supported metric matches the evaluation wording. Explicit
choices (--target, --metric) are never second-guessed. When something is missing, an LLM
proposes it as a JSON block (plain text, no tool calling) and orbit validates every field
against the real files and the metric registry before using it. Metrics are always chosen from
`METRICS`: orbit never executes generated scoring code on the host.
"""
import codecs
import csv
import functools
import gzip
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS, match_metric

PROFILE_ROWS = 100_000  # tables with more rows are profiled on a random sample of this size
SNIFF_BYTES = 64 * 1024
SNIFF_LINES = 50
DELIMITERS = ",;\t|"
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 decodes any byte, so it is the last resort
_DECIMAL_COMMA = re.compile(r"^\s*-?\d+,\d+\s*$")
_DECIMAL_POINT = re.compile(r"^\s*-?\d+\.\d+\s*$")
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


def _open_bytes(path: Path):
    return gzip.open(path, "rb") if path.name.lower().endswith(".gz") else open(path, "rb")


def _detect_encoding(path: Path) -> str:
    """The first of `ENCODINGS` that decodes the whole file (streamed, so a BOM or a stray Latin-1
    byte late in the file is noticed)."""
    for enc in ENCODINGS[:-1]:
        dec = codecs.getincrementaldecoder(enc)()
        try:
            with _open_bytes(path) as f:
                while chunk := f.read(1 << 20):
                    dec.decode(chunk)
                dec.decode(b"", final=True)
            return enc
        except UnicodeDecodeError:
            continue
        except (OSError, EOFError) as e:
            raise ValueError(f"{path.name}: could not read the file ({e})") from e
    return ENCODINGS[-1]


def _consistent_fields(text: str, delim: str) -> int | None:
    """Field count shared by all of the first records (quotes respected), or None."""
    counts = []
    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=delim):
            counts.append(len(row))
            if len(counts) >= SNIFF_LINES:
                break
    except csv.Error:
        return None
    return counts[0] if counts and len(set(counts)) == 1 and counts[0] > 1 else None


def _sniff_delimiter(text: str) -> str:
    try:
        guess = csv.Sniffer().sniff(text, delimiters=DELIMITERS).delimiter
    except csv.Error:
        guess = None
    if guess and _consistent_fields(text, guess):
        return guess
    fields = {d: _consistent_fields(text, d) for d in DELIMITERS}
    best = [d for d in DELIMITERS if fields[d]]
    return max(best, key=lambda d: fields[d]) if best else ","


def _decimal_comma(path: Path, delim: str, enc: str) -> bool:
    """True when a text column is almost all numbers written like 3,14 and none like 3.14."""
    try:
        sample = pd.read_csv(path, sep=delim, encoding=enc, nrows=2000, dtype=str)
    except (ValueError, OSError):
        return False
    for c in sample.columns:
        vals = sample[c].dropna().astype(str)
        if len(vals) and not vals.str.match(_DECIMAL_POINT).any() \
                and vals.str.match(_DECIMAL_COMMA).mean() >= 0.9:
            return True
    return False


@functools.lru_cache(maxsize=64)
def _format(path: str, mtime: int, size: int) -> tuple:
    p = Path(path)
    enc = _detect_encoding(p)
    with _open_bytes(p) as f:
        raw = f.read(SNIFF_BYTES)
    text = raw.decode(enc, errors="replace")
    if len(raw) >= SNIFF_BYTES and "\n" in text:
        text = text[:text.rindex("\n") + 1]  # drop the record cut off by the read limit
    if ".tsv" in p.name.lower():
        delim = "\t"
    else:
        delim = _sniff_delimiter(text)
    decimal = "," if delim != "," and _decimal_comma(p, delim, enc) else "."
    return delim, enc, decimal


def describe_format(path: Path) -> dict:
    """How a table file is written: `delimiter`, `encoding` and `decimal` mark. `.tsv` files are
    always tab separated; other files are sniffed from their first 64 KB."""
    path = Path(path)
    st = path.stat()
    delim, enc, decimal = _format(str(path.resolve()), st.st_mtime_ns, st.st_size)
    return {"delimiter": delim, "encoding": enc, "decimal": decimal}


def _format_parts(path: Path) -> str:
    """The unusual parts of how a file is written, as text; empty for plain comma / UTF-8 / dot."""
    fmt = describe_format(path)
    parts = []
    if fmt["delimiter"] != ",":
        parts.append(f"delimiter {fmt['delimiter']!r}")
    if fmt["decimal"] != ".":
        parts.append(f"decimal mark {fmt['decimal']!r}")
    if not fmt["encoding"].startswith("utf-8"):
        parts.append(f"{fmt['encoding']} encoding")
    return ", ".join(parts)


def format_note(path: Path) -> str | None:
    """A sentence when the file is not plain comma / UTF-8 / dot-decimal, else None."""
    parts = _format_parts(path)
    return f"read {Path(path).name} with {parts}" if parts else None


def format_notes(paths) -> list[str]:
    """One sentence per way of writing a file: files that share a format are grouped, so four
    semicolon-separated tables give one note, not four."""
    groups: dict[str, list[str]] = {}
    for path in paths:
        parts = _format_parts(Path(path))
        if parts:
            groups.setdefault(parts, []).append(Path(path).name)
    notes = []
    for parts, names in groups.items():
        if len(names) == 1:
            notes.append(f"read {names[0]} with {parts}")
        elif len(names) <= 3:
            notes.append(f"read {', '.join(names[:-1])} and {names[-1]} with {parts}")
        else:
            notes.append(f"read {len(names)} tables ({', '.join(names[:3])} and {len(names) - 3} more) "
                         f"with {parts}")
    return notes


def _check_columns(path: Path, df: pd.DataFrame) -> None:
    if df.shape[1] == 1:
        header = str(df.columns[0])
        seen = [d for d in DELIMITERS if header.count(d) >= 2]
        if seen:
            chars = ", ".join(repr(d) for d in seen)
            raise ValueError(f"{path.name}: could not split the header into columns (the single column "
                             f"{header[:80]!r} contains {chars} repeatedly); check that the file is a "
                             "consistently delimited table (comma, semicolon, tab or pipe)")


def read_table(path: Path, nrows: int | None = None, *, skiprows=None) -> pd.DataFrame:
    """Read a CSV/TSV table (optionally gzipped) whatever its delimiter, encoding or decimal mark."""
    path = Path(path)
    fmt = describe_format(path)
    encodings = [fmt["encoding"]] + [e for e in ENCODINGS if e != fmt["encoding"]]
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=fmt["delimiter"], encoding=enc, decimal=fmt["decimal"],
                             nrows=nrows, skiprows=skiprows)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"{path.name}: the file is empty") from e
    else:  # pragma: no cover - latin-1 decodes every byte
        raise ValueError(f"{path.name}: could not decode the file as text (tried {', '.join(encodings)})")
    _check_columns(path, df)
    return df


def _count_data_rows(path: Path) -> int:
    """Lines after the header, counted without parsing. Quoted fields that contain line breaks make
    this an overestimate, which only matters for files above the sampling threshold."""
    n, last = 0, b"\n"
    with _open_bytes(path) as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return max(0, n + (last != b"\n") - 1)


def load_for_profile(path: Path, max_rows: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, int, bool]:
    """The table for profiling, its total row count and whether it is a sample. Up to `max_rows`
    rows are read in full; a larger table is sampled at random (seeded, original order kept),
    never taken from the top, because files are often ordered by date or class."""
    path = Path(path)
    max_rows = PROFILE_ROWS if max_rows is None else max_rows
    total = _count_data_rows(path)
    if total <= max_rows:
        df = read_table(path)
        return df, len(df), False
    keep = np.zeros(total, dtype=bool)
    keep[np.random.default_rng(seed).choice(total, size=max_rows, replace=False)] = True
    df = read_table(path, skiprows=lambda i: i > 0 and (i > total or not keep[i - 1]))
    return df, total, True


def describe_table(path: Path, df: pd.DataFrame, max_cols: int = 40, title: str | None = None,
                   total_rows: int | None = None, sampled: bool = False,
                   dropped: list[str] | None = None) -> str:
    shown = f"{len(df):,}"
    if sampled and total_rows:
        shown = f"random sample of {len(df):,} of {total_rows:,}"
    lines = [f"### {title or path.name} ({shown} rows, {df.shape[1]} columns)",
             "column | dtype | missing% | example values"]
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        ex = ", ".join(map(str, s.dropna().unique()[:3]))[:60]
        lines.append(f"{c} | {s.dtype} | {100 * s.isna().mean():.0f} | {ex}"
                     + ("  (dropped)" if c in (dropped or ()) else ""))
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
    if not isinstance(primary, str) or primary not in frames:
        raise PlanError(f"primary_table {primary!r} must be one of {sorted(frames)}")
    tgt = target or proposal.get("target")
    if not isinstance(tgt, str) or tgt not in frames[primary].columns:
        raise PlanError(f"target {tgt!r} must be a column name of {primary}: {list(frames[primary].columns)[:30]}")
    name = metric
    if name is None:
        name = proposal.get("metric")
        if name is not None and (not isinstance(name, str) or name not in METRICS):
            raise PlanError(f"metric {name!r} is not supported; choose from {sorted(METRICS)} or null")
    ident = proposal.get("id_column")
    if not isinstance(ident, str) or ident not in frames[primary].columns or ident == tgt:
        ident = None
    return {"primary": primary, "target": tgt, "metric": name, "id_column": ident,
            "reasoning": str(proposal.get("reasoning") or "")[:300]}


def _ask(llm, goal, eval_text, frames, paths, docs, target, metric, sizes, primary=None) -> dict:
    fixed = []
    if primary:
        fixed.append(f"primary table: {primary}")
    if target:
        fixed.append(f"target column: {target}")
    if metric:
        fixed.append(f"metric: {metric}")
    prompt = USER.format(
        goal=goal or "(not given: infer a sensible prediction task from the data)",
        eval=eval_text or "(not given)",
        metrics=", ".join(sorted(METRICS)),
        tables="\n\n".join(describe_table(paths[n], f, total_rows=sizes[n][0], sampled=sizes[n][1])
                          for n, f in frames.items()),
        docs=(f"\n## Documentation\n{read_docs(docs)}\n" if docs else ""),
        fixed="\n".join(fixed) or "(nothing)")
    error = None
    for _ in range(2):
        text = llm.complete(SYSTEM, prompt + (f"\n\nYour previous reply was rejected: {error}" if error else "")).text
        try:
            proposal = _parse_json(text)
            if primary:
                proposal["primary_table"] = primary  # decided by the user
            return _check(proposal, frames, target, metric, eval_text)
        except PlanError as e:
            error = str(e)
    raise PlanError(f"could not work out the task from the data ({error}); "
                    "pass --target (and --metric) explicitly")


def match_table(table: str, names: list[str]) -> str:
    """The table a user asked for by (part of) its file name: an exact name, else a name that
    matches ignoring case, else the only name containing the text."""
    if table in names:
        return table
    lowered = [n for n in names if n.lower() == table.lower()]
    if len(lowered) == 1:
        return lowered[0]
    partial = [n for n in names if table.lower() in n.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise PlanError(f"--table {table!r} matches none of the tables: {', '.join(names)}")
    raise PlanError(f"--table {table!r} matches several tables ({', '.join(partial)}); "
                    "give the full file name")


def plan(tables: list[Path], docs: list[Path], goal: str | None = None, eval_text: str | None = None,
         target: str | None = None, metric: str | None = None, llm=None, table: str | None = None) -> Plan:
    if metric is not None:
        metric = match_metric(metric) or metric
        if metric not in METRICS:
            raise PlanError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    elif eval_text:
        metric = match_metric(eval_text)

    paths = {p.name: p for p in tables}
    if len(paths) != len(tables):
        raise PlanError("two tables share a file name; rename one so they can be told apart")
    loaded = {n: load_for_profile(p) for n, p in paths.items()}
    frames = {n: df for n, (df, _, _) in loaded.items()}
    sizes = {n: (total, sampled) for n, (_, total, sampled) in loaded.items()}

    chosen = match_table(table, list(paths)) if table else None
    if chosen and target and target not in frames[chosen].columns:
        raise PlanError(f"--table {chosen} has no column {target!r}; its columns: "
                        f"{', '.join(map(str, frames[chosen].columns))}")
    holders = [n for n, f in frames.items() if target and target in f.columns and chosen in (None, n)]
    needs_llm = target is None or len(holders) != 1 or (metric is None and eval_text)
    notes = []
    if needs_llm and llm is not None:
        choice = _ask(llm, goal, eval_text, frames, paths, docs, target, metric, sizes, chosen)
        why = ("chosen with --table" if chosen else
               f"chosen by the planner: {choice['reasoning']}" if choice["reasoning"] else "chosen by the planner")
    elif target is None:
        raise PlanError("no --target given and no LLM available to infer it")
    elif not holders:
        raise PlanError(f"target {target!r} is not a column of any table "
                        f"({'; '.join(f'{n}: {list(f.columns)[:8]}' for n, f in frames.items())})")
    else:
        pick = max(holders, key=lambda n: ("train" in n.lower(), sizes[n][0]))
        why = ("chosen with --table" if chosen else
               "the largest of the tables named like 'train'" if "train" in pick.lower()
               else "the table with the most rows")
        choice = {"primary": pick, "target": target, "metric": metric, "id_column": None}
    choice["metric"] = choice["metric"] or metric
    candidates = [n for n, f in frames.items() if choice["target"] in f.columns]
    if len(candidates) > 1:
        others = ", ".join(f"{n} ({sizes[n][0]:,} rows)" for n in candidates if n != choice["primary"])
        notes.append(f"setup: {len(candidates)} tables have column {choice['target']!r}; using "
                     f"{choice['primary']} ({sizes[choice['primary']][0]:,} rows), {why}; "
                     f"other candidates: {others}")
    elif choice.get("reasoning"):
        notes.append(f"setup: {choice['reasoning']}")
    if eval_text and choice["metric"] is None and not match_metric(eval_text):
        raise PlanError(f"cannot map the evaluation {eval_text!r} to a supported metric "
                        f"({', '.join(sorted(METRICS))}); pass --metric")

    primary = choice["primary"]
    wanted = set(frames[primary].columns) - {choice["target"]}
    extras = []
    for n, f in frames.items():
        if n == primary:
            continue
        if choice["target"] in f.columns:
            # Extra tables are readable by the generated code, so one with labels would leak them.
            notes.append(f"ignored {n}: it contains the target column {choice['target']!r}, and generated "
                         f"code must never see labels (combine files first to use it as training data)")
        elif set(f.columns) == wanted:
            notes.append(f"ignored {n}: same columns as {primary} without the target, so it has no extra information")
        else:
            extras.append(paths[n])
    return Plan(primary=paths[primary], target=choice["target"], metric=choice["metric"],
                id_column=choice["id_column"], extras=extras, docs=docs, notes=notes)
