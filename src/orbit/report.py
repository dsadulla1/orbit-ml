"""Turn a finished search into a short, data-driven report.

1. `facts()` extracts every number worth reporting from the journal (deterministic).
2. An LLM writes the narrative from those facts only (optional; a template is the fallback).
3. `check_numbers()` verifies that every number in the narrative appears in the facts,
   so hallucinated or mis-rounded figures are flagged instead of published.
"""
import json
import re
from pathlib import Path

from .agent import Journal
from .task import Task

NARRATOR_SYSTEM = """You write concise experiment reports for ML practitioners. \
Use ONLY the numbers present in the FACTS JSON, copied with the same precision; never compute \
or invent new figures. If evidence is weak (small gain over baseline, large validation/test gap, \
many failures) say so plainly. No hype."""

NARRATOR_USER = """FACTS:
```json
{facts}
```

Write at most 150 words of Markdown with exactly these bold labels, each followed by 1-3 sentences:
**Verdict:** which experiment won and by how much over the baseline.
**Evidence:** validation vs held-out test score, and how the search progressed.
**Caveats:** anything that weakens the conclusion.
**Next steps:** two concrete recommendations."""


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


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
        "baseline_val_score": _r(task.baseline),
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
        if best.test_metric is not None:
            out["best"]["val_test_gap"] = _r(abs(best.metric - best.test_metric))
        if first_ok and first_ok.id != best.id:
            out["first_working"] = {"experiment": first_ok.id, "val_score": _r(first_ok.metric)}
            out["gain_from_search"] = _r(sign * (best.metric - first_ok.metric))
        if len(ranked) > 1:
            out["runner_up"] = {"experiment": ranked[1].id, "val_score": _r(ranked[1].metric),
                                "plan": ranked[1].plan}
    out["leaderboard"] = [{"experiment": n.id, "stage": n.stage, "val_score": _r(n.metric),
                           "plan": n.plan[:160]} for n in ranked[:top]]
    return out


_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


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
    for token in _NUM.findall(text):
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


def template_narrative(f: dict) -> str:
    if "best" not in f:
        return (f"**Verdict:** none of the {f['experiments_run']} experiments produced valid "
                f"predictions. **Next steps:** inspect the failures in the run folder and "
                "consider a larger time limit or a stronger model.")
    b = f["best"]
    lines = [f"**Verdict:** experiment #{b['experiment']} ({b['stage']}) is best with "
             f"{f['metric']} {b['val_score']} on validation vs a baseline of "
             f"{f['baseline_val_score']} (gain {b['gain_over_baseline']})."]
    ev = f"**Evidence:** held-out test score {b['test_score']}"
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
    if "runner_up" in f and abs(f["runner_up"]["val_score"] - b["val_score"]) < 0.005:
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
        rows.append(f"| {r['experiment']} | {r['stage']} | {r['val_score']} | {plan} |")
    return "\n".join(rows)


def build(task: Task, journal: Journal, llm=None) -> tuple[str, dict]:
    f = facts(task, journal)
    source, narrative, unverified = "template", template_narrative(f), []
    if llm is not None and "best" in f:
        try:
            reply = llm.complete(NARRATOR_SYSTEM, NARRATOR_USER.format(facts=json.dumps(f, indent=2)))
            narrative, source = reply.text.strip(), f"LLM ({llm.model})"
            unverified = check_numbers(narrative, f)
        except Exception as e:  # noqa: BLE001
            source = f"template (LLM failed: {type(e).__name__})"
    check = ("All numbers verified against run data." if not unverified else
             f"WARNING: unverified numbers in narrative: {', '.join(unverified)}")
    best = f.get("best")
    md = [f"# orbit report: {task.run_dir.name}", "",
          f"Dataset `{f['dataset']}`, target `{f['target']}`, metric **{f['metric']}** "
          f"({f['direction']}). Stopped: {f['stop_reason']}.", "",
          "## Summary", "", narrative, "", f"_Narrative source: {source}. {check}_", ""]
    if best:
        md += ["## Leaderboard (validation)", "", leaderboard_md(f), "",
               f"Best experiment code: `nodes/{best['experiment']:03d}/solution.py`", ""]
    md += ["## Run stats", "",
           f"- Experiments: {f['experiments_run']} ({f['experiments_failed']} failed, "
           f"{f['debug_fixes']}/{f['debug_attempts']} debug attempts succeeded)",
           f"- Wall time: {f['minutes']} min, LLM tokens: {f['total_tokens']:,}", ""]
    f["narrative_unverified"] = unverified
    return "\n".join(md), f
