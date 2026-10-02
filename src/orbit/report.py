"""Turn a finished search into a short, data-driven report.

1. `facts()` extracts every number worth reporting from the journal (deterministic).
2. An LLM writes the narrative from those facts only (optional; a template is the fallback).
3. `check_numbers()` verifies that every number in the narrative appears in the facts,
   so hallucinated or mis-rounded figures are flagged instead of published.
"""
import json
import re
import statistics
from pathlib import Path

from .agent import Journal
from .task import Task

NARRATOR_SYSTEM = """You write concise experiment reports for ML practitioners. \
Use ONLY the numbers present in the FACTS JSON, copied with the same precision; never compute \
or invent new figures. The baseline is the best constant prediction that ignores the features, so \
describe gains as a difference from it; never write "x times better" or percentage improvements. \
Describe what a model does only as far as its plan text says; do not credit techniques that the \
plan does not mention. If evidence is weak (small gain, a gain within the cross-validation spread, \
large validation/test gap, many failures) say so plainly. No hype."""

NARRATOR_USER = """FACTS:
```json
{facts}
```

The validation score is described by the "validation" field; when it is a cross-validation \
mean, val_std_across_folds is its spread, and a gain smaller than that spread is within noise.

Write at most 150 words of Markdown with exactly these bold labels, each followed by 1-3 sentences:
**Verdict:** which experiment won and by how much over the baseline.
**Evidence:** validation vs held-out test score, and how the search progressed.
**Caveats:** anything that weakens the conclusion.
**Next steps:** two concrete recommendations."""

REVISE_USER = """FACTS:
```json
{facts}
```

Your draft report:
{draft}

A reviewer found these problems with the draft:
{issues}

Rewrite the report so it is fully consistent with FACTS. Keep the same four bold labels, use only \
numbers from FACTS, and stay under 150 words."""


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _std(node):
    return _r(statistics.stdev(node.fold_scores)) if len(node.fold_scores) > 1 else None


def _validation(task) -> str:
    col = task.spec.time_col
    if col:
        return (f"mean of {task.n_folds} forward-chaining folds ordered by {col}" if task.n_folds > 1
                else f"single split ordered by {col}")
    return f"mean of {task.n_folds}-fold cross-validation" if task.n_folds > 1 else "single validation split"


def facts(task: Task, journal: Journal, top: int = 5) -> dict:
    metric = task.metric
    ranked = journal.ranked(metric)
    best = ranked[0] if ranked else None
    first_ok = next((n for n in journal.nodes if not n.buggy), None)
    duration = (journal.finished_at or journal.started_at) - journal.started_at
    sign = 1 if metric.greater_is_better else -1
    out = {
        "dataset": Path(task.spec.data_path).name,
        "target": task.spec.target,
        "metric": task.metric_name,
        "direction": "higher is better" if metric.greater_is_better else "lower is better",
        "validation": _validation(task),
        "baseline_val_score": _r(task.baseline),
        "baseline_is": "best constant prediction that ignores the features",
        "experiments_run": len(journal.nodes),
        "experiments_failed": sum(n.buggy for n in journal.nodes),
        "debug_attempts": sum(n.stage == "debug" for n in journal.nodes),
        "debug_fixes": sum(n.stage == "debug" and not n.buggy for n in journal.nodes),
        "minutes": _r(duration / 60, 1),
        "total_tokens": journal.tokens,
        "stop_reason": journal.stop_reason,
    }
    if best:
        out["best"] = {
            "experiment": best.id, "stage": best.stage, "plan": best.plan,
            "val_score": _r(best.metric), "test_score": _r(best.test_metric),
            "gain_over_baseline": _r(sign * (best.metric - task.baseline)),
        }
        if _std(best) is not None:
            out["best"]["val_std_across_folds"] = _std(best)
        if best.test_metric is not None:
            out["best"]["val_test_gap"] = _r(abs(best.metric - best.test_metric))
        if first_ok and first_ok.id != best.id:
            out["first_working"] = {"experiment": first_ok.id, "val_score": _r(first_ok.metric)}
            out["gain_from_search"] = _r(sign * (best.metric - first_ok.metric))
        if len(ranked) > 1:
            spread = _std(best) or 0.0
            gap = abs(best.metric - ranked[1].metric)
            out["runner_up"] = {"experiment": ranked[1].id, "val_score": _r(ranked[1].metric),
                                "plan": ranked[1].plan, "gap_to_best": _r(gap),
                                "within_fold_spread": bool(spread and gap <= spread)}
    out["leaderboard"] = [{"experiment": n.id, "stage": n.stage, "val_score": _r(n.metric),
                           "val_std": _std(n), "plan": n.plan[:160]} for n in ranked[:top]]
    return out


_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)  # "1. " / "2) " list items


def _numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _numbers(v)]
    if isinstance(obj, str):
        return [float(m.replace(",", "").rstrip("%")) for m in _NUM.findall(obj)]
    return []


def check_numbers(text: str, fact_dict: dict) -> list[str]:
    """Return numbers in `text` that cannot be traced to `fact_dict` (at the precision written)."""
    known = _numbers(fact_dict)
    unverified = []
    for token in _NUM.findall(_LIST_MARKER.sub("", text)):
        raw = token.replace(",", "")
        is_pct = raw.endswith("%")
        raw = raw.rstrip("%")
        value = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        candidates = known + ([k * 100 for k in known] if is_pct else [])
        if not any(abs(value - k) <= tol for k in candidates):
            unverified.append(token)
    return unverified


_NUMWORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_C = r"(\d+|" + "|".join(_NUMWORDS) + r")"
_CLAIM = re.compile(
    rf"\b{_C}\s+(?:of\s+(?:the\s+)?{_C}\s+)?experiments?\s+(?:\w+\s+)?(succeeded|failed|crashed)\b"
    rf"(?:\s+(?:out of|of)\s+(?:the\s+)?{_C}\b)?", re.IGNORECASE)
_BARE = re.compile(rf"\b{_C}\s+(succeeded|failed|crashed)\b", re.IGNORECASE)  # "7 succeeded and 3 failed"
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_QUALIFIED = re.compile(r"debug|draft|improve|fix|first|best|runner|#\d|experiment \d", re.IGNORECASE)


def _count(token: str) -> int:
    return int(token) if token.isdigit() else _NUMWORDS[token.lower()]


def check_claims(text: str, fact_dict: dict) -> list[str]:
    """Counts of succeeded/failed experiments written in digits or words must match the run.
    Digits are checked by `check_numbers`; this catches a right number with the wrong meaning
    (e.g. "only one experiment succeeded out of ten" when one of ten failed)."""
    run, failed = fact_dict["experiments_run"], fact_dict["experiments_failed"]
    issues = []
    for sentence in _SENTENCE.split(text):
        if _QUALIFIED.search(sentence):
            continue  # claims about a particular stage or experiment are not checked here
        if not _CLAIM.search(sentence) and re.search(r"experiment|runs?\b", sentence, re.IGNORECASE):
            for m in _BARE.finditer(sentence):
                n, verb = m.groups()
                expected = failed if verb.lower() in ("failed", "crashed") else run - failed
                if _count(n) != expected:
                    issues.append(f"claims {n} {verb.lower()}, but {expected} of {run} experiments did")
        for m in _CLAIM.finditer(sentence):
            n, total_a, verb, total_b = m.groups()
            expected = failed if verb.lower() in ("failed", "crashed") else run - failed
            if _count(n) != expected:
                issues.append(f"claims {n} experiments {verb.lower()}, but {expected} of {run} did")
            total = total_a or total_b
            if total and _count(total) != run:
                issues.append(f"claims {total} experiments in total, but {run} were run")
    return issues


def template_narrative(f: dict) -> str:
    if "best" not in f:
        return (f"**Verdict:** none of the {f['experiments_run']} experiments produced valid "
                f"predictions. **Next steps:** inspect the failures in the run folder and "
                "consider a larger time limit or a stronger model.")
    b = f["best"]
    lines = [f"**Verdict:** experiment #{b['experiment']} ({b['stage']}) is best with "
             f"{f['metric']} {b['val_score']} on validation vs a baseline of "
             f"{f['baseline_val_score']} (gain {b['gain_over_baseline']})."]
    ev = ("**Evidence:** held-out test score "
          + (str(b["test_score"]) if b["test_score"] is not None else "unavailable"))
    if "gain_from_search" in f:
        ev += (f"; search improved on the first working experiment "
               f"(#{f['first_working']['experiment']}, {f['first_working']['val_score']}) "
               f"by {f['gain_from_search']}")
    lines.append(ev + f", across {f['experiments_run']} experiments.")
    caveats = []
    if b.get("val_test_gap") is not None and b["val_test_gap"] > 0.02:
        caveats.append(f"validation/test gap of {b['val_test_gap']} suggests some overfitting to validation")
    if f["experiments_failed"]:
        caveats.append(f"{f['experiments_failed']} of {f['experiments_run']} experiments failed")
    if "runner_up" in f and (f["runner_up"]["within_fold_spread"]
                             or abs(f["runner_up"]["val_score"] - b["val_score"]) < 0.005):
        caveats.append(f"runner-up #{f['runner_up']['experiment']} is within noise "
                       f"({f['runner_up']['val_score']})")
    lines.append("**Caveats:** " + ("; ".join(caveats) + "." if caveats else "none significant."))
    lines.append("**Next steps:** confirm the winner with repeated cross-validation on fresh "
                 "seeds, then run a longer search seeded from it.")
    return "\n\n".join(lines)


def leaderboard_md(f: dict) -> str:
    rows = ["| # | stage | val score | approach |", "|---|---|---|---|"]
    for r in f["leaderboard"]:
        plan = r["plan"].replace("|", "/")
        score = f"{r['val_score']} ± {r['val_std']}" if r.get("val_std") is not None else r["val_score"]
        rows.append(f"| {r['experiment']} | {r['stage']} | {score} | {plan} |")
    return "\n".join(rows)


def review(narrative: str, f: dict) -> list[str]:
    """Everything wrong with a narrative that can be checked against the facts."""
    return [f"number {n} does not appear in FACTS" for n in check_numbers(narrative, f)] + \
        check_claims(narrative, f)


def _write(llm, f: dict) -> tuple[str, str, list[str]]:
    """LLM narrative, reviewed against the facts, revised once, else the template.
    Returns (text, source, problems found in earlier drafts)."""
    facts_json = json.dumps(f, indent=2)
    text = llm.complete(NARRATOR_SYSTEM, NARRATOR_USER.format(facts=facts_json)).text.strip()
    source, found = f"LLM ({llm.model})", review(text, f)
    if not found:
        return text, source, []
    revised = llm.complete(NARRATOR_SYSTEM, REVISE_USER.format(
        facts=facts_json, draft=text, issues="\n".join(f"- {p}" for p in found))).text.strip()
    again = review(revised, f)
    if not again:
        return revised, source + ", revised after review", found
    return template_narrative(f), "template (the LLM narrative failed review twice)", found + again


def build(task: Task, journal: Journal, llm=None) -> tuple[str, dict]:
    f = facts(task, journal)
    source, narrative, found = "template", template_narrative(f), []
    if llm is not None and "best" in f:
        try:
            narrative, source, found = _write(llm, f)
        except Exception as e:  # noqa: BLE001
            source = f"template (LLM failed: {type(e).__name__})"
    unverified = review(narrative, f)
    check = ("All numbers verified against run data." if not unverified else
             f"WARNING: unverified statements in narrative: {'; '.join(unverified)}")
    if found and not unverified:
        check += f" Review caught {len(found)} problem(s) in a draft."
    best = f.get("best")
    md = [f"# orbit report: {task.run_dir.name}", "",
          f"Dataset `{f['dataset']}`, target `{f['target']}`, metric **{f['metric']}** "
          f"({f['direction']}). Stopped: {f['stop_reason']}.", "",
          "## Summary", "", narrative, "", f"_Narrative source: {source}. {check}_", ""]
    if best:
        md += [f"## Leaderboard ({f['validation']})", "", leaderboard_md(f), "",
               f"Best experiment code: `nodes/{best['experiment']:03d}/solution.py`", ""]
    md += ["## Run stats", "",
           f"- Experiments: {f['experiments_run']} ({f['experiments_failed']} failed, "
           f"{f['debug_fixes']}/{f['debug_attempts']} debug attempts succeeded)",
           f"- Wall time: {f['minutes']} min, LLM tokens: {f['total_tokens']:,}", ""]
    f["narrative_unverified"] = unverified
    f["review_issues"] = found
    return "\n".join(md), f
