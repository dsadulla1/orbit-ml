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

import numpy as np

from .agent import Journal
from .task import Task

NARRATOR_SYSTEM = """You write concise experiment reports for ML practitioners. \
Use ONLY the numbers present in the FACTS JSON, copied with the same precision; never compute \
or invent new figures. The baseline is the best constant prediction that ignores the features, so \
describe gains as a difference from it; never write "x times better" or percentage improvements. \
Describe what a model does only as far as its plan text says; do not credit techniques that the \
plan does not mention. Never claim statistical significance: orbit runs no significance tests. If evidence is weak (small gain, a gain within the cross-validation spread, \
large validation/test gap, many failures) say so plainly. No hype. \
To judge whether the improvement is real, use gain_within_spread and gain_over_spread as given: \
compare the GAIN over the baseline (never the score itself) with val_std_across_folds, say the gain \
is within the spread when gain_within_spread is true and that it exceeds the spread when it is false, \
and do not speculate about performance on the test set beyond the test_score in FACTS. \
When distribution_shift.flag is true the validation and test splits have different target rates or \
means, and therefore different baselines (val_baseline, test_baseline): explain any difference \
between validation and test through those rates and baselines, compare val_gain with test_gain \
instead of raw scores, and never attribute the gap to overfitting or leakage. \
When has_spread is false there is a single validation split: say that no cross-validation spread \
is available, so small differences between experiments are not evidence."""

NARRATOR_USER = """FACTS:
```json
{facts}
```

The validation score is described by the "validation" field; when it is a cross-validation \
mean, val_std_across_folds is its spread, and a gain over the baseline that is not larger than that \
spread is within noise (gain_within_spread; gain_over_spread is their ratio). \
The "splits" field gives the rows and target rate or mean of each split; val_gain and test_gain \
are the gains over the baselines of the validation and test splits.

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
    col = task.spec.time_col or ("file order" if task.spec.time_order else None)
    if col:
        return (f"mean of {task.n_folds} forward-chaining folds ordered by {col}" if task.n_folds > 1
                else f"single split ordered by {col}")
    return f"mean of {task.n_folds}-fold cross-validation" if task.n_folds > 1 else "single validation split"


# Thresholds for calling two splits' target distributions different. Conservative on purpose: a
# difference that small would not change what the baselines say.
SHIFT_MIN_ROWS = 30        # splits smaller than this are too noisy to compare
SHIFT_RATIO = 1.5          # binary: rates differ by more than this factor ...
SHIFT_ABS = 0.03           # ... and by more than this much
SHIFT_CLASS_SHARE = 0.10   # multiclass: largest difference in a class share
SHIFT_MEAN_STDS = 0.5      # regression: difference of means, in standard deviations of validation


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _binary_differs(a: float, b: float) -> bool:
    lo, hi = sorted((a, b))
    return hi - lo > SHIFT_ABS and hi > SHIFT_RATIO * lo


def distribution_shift(task: Task) -> dict:
    """Whether the test split's target distribution differs from the validation folds', from the
    statistics stored at setup. `flag` compares test with the mean over the validation folds:
      binary      positive-class share differs by more than SHIFT_RATIO times and SHIFT_ABS;
      multiclass  some class share differs by more than SHIFT_CLASS_SHARE;
      regression  the means differ by more than SHIFT_MEAN_STDS standard deviations of validation;
    and only when every split has at least SHIFT_MIN_ROWS rows. `folds_differ` applies the same
    rule between the validation folds themselves (forward-chaining folds can drift on their own)."""
    stats = task.split_stats
    none = {"flag": False, "folds_differ": False, "detail": ""}
    if not stats or not stats.get("val") or any(
            sp["rows"] < SHIFT_MIN_ROWS for sp in [stats["test"], *stats["val"]]):
        return none
    val, test = stats["val"], stats["test"]
    if task.kind == "regression":
        v_means = [v["mean"] for v in val]
        v_mean, v_std = float(np.mean(v_means)), float(np.mean([v["std"] for v in val]))
        limit = SHIFT_MEAN_STDS * v_std
        flag = abs(test["mean"] - v_mean) > limit
        folds = len(val) > 1 and max(v_means) - min(v_means) > limit
        what = (f"target mean {_r(test['mean'])} in test vs {_r(v_mean)} in validation"
                f" (standard deviation {_r(v_std)})")
        fold_what = f"target mean by validation fold {_r(min(v_means))} to {_r(max(v_means))}"
    elif len(task.classes) == 2 and stats.get("positive") is not None:
        pos = stats["positive"]
        rates = [v["positive_rate"] for v in val]
        v_rate = float(np.mean(rates))
        flag = _binary_differs(test["positive_rate"], v_rate)
        folds = len(val) > 1 and _binary_differs(min(rates), max(rates))
        what = (f"positive rate (label {pos}) {_pct(round(test['positive_rate'], 3))} in test vs "
                f"{_pct(round(v_rate, 3))} in validation")
        fold_what = (f"positive rate by validation fold {_pct(round(min(rates), 3))} to "
                     f"{_pct(round(max(rates), 3))}")
    else:
        names = list(test["shares"])
        v_share = {c: float(np.mean([v["shares"][c] for v in val])) for c in names}
        worst = max(names, key=lambda c: abs(test["shares"][c] - v_share[c]))
        flag = abs(test["shares"][worst] - v_share[worst]) > SHIFT_CLASS_SHARE
        spreads = {c: max(v["shares"][c] for v in val) - min(v["shares"][c] for v in val)
                   for c in names}
        folds = len(val) > 1 and max(spreads.values()) > SHIFT_CLASS_SHARE
        what = (f"share of class {worst} {_pct(round(test['shares'][worst], 3))} in test vs "
                f"{_pct(round(v_share[worst], 3))} in validation")
        fold_what = f"largest class-share range across validation folds {_pct(round(max(spreads.values()), 3))}"
    detail = what if flag else ""
    if folds:
        detail = (detail + "; folds differ: " if detail else "validation folds differ: ") + fold_what
    return {"flag": bool(flag), "folds_differ": bool(folds), "detail": detail}


def _compact(stat: dict, kind: str, positive, top_classes: int = 6) -> dict:
    out = {"rows": stat["rows"]}
    if kind == "regression":
        out.update(target_mean=_r(stat["mean"]), target_std=_r(stat["std"]))
    elif positive is not None:
        out["positive_rate"] = _r(stat["positive_rate"], 3)
    else:
        top = sorted(stat["shares"].items(), key=lambda kv: -kv[1])[:top_classes]
        out["class_shares"] = {k: _r(v, 3) for k, v in top}
    return out


def _mean_stat(val: list[dict]) -> dict:
    """The validation folds as one split: every statistic averaged over the folds."""
    out = {k: float(np.mean([v[k] for v in val])) for k in ("rows", "positive_rate", "mean", "std")
           if k in val[0]}
    if "shares" in val[0]:
        out["shares"] = {c: float(np.mean([v["shares"][c] for v in val])) for c in val[0]["shares"]}
    return out


def split_summary(task: Task) -> dict:
    """Rows and target rate (or mean) of the training rows, the validation folds (averaged) and the
    test split, in the shape `facts` reports them. Empty for runs prepared before these were stored."""
    stats = task.split_stats
    if not stats or not stats.get("val"):
        return {}
    positive = stats.get("positive") if task.kind == "classification" else None
    val = stats["val"]
    out = {"train": _compact(stats["train"], task.kind, positive),
           "validation": _compact(_mean_stat(val), task.kind, positive),
           "test": _compact(stats["test"], task.kind, positive)}
    out["validation"]["rows"] = int(round(_mean_stat(val)["rows"]))
    out["validation"]["folds"] = len(val)
    if len(val) > 1 and (task.kind == "regression" or positive is not None):
        key, nd = ("mean", 4) if task.kind == "regression" else ("positive_rate", 3)
        out["validation"]["by_fold"] = [_r(v[key], nd) for v in val]
    return out


def split_rows(f: dict) -> list[tuple[str, int, str]]:
    """(split, rows, target description) lines for the report, from `facts()["splits"]`."""
    def target(sp):
        if "positive_rate" in sp:
            return f"positive rate {_pct(sp['positive_rate'])}"
        if "target_mean" in sp:
            return f"mean {sp['target_mean']}, std {sp['target_std']}"
        return ", ".join(f"{k} {_pct(v)}" for k, v in sp.get("class_shares", {}).items())

    sp = f.get("splits") or {}
    if not sp:
        return []
    val = sp["validation"]
    label = f"validation (mean of {val['folds']} folds)" if val["folds"] > 1 else "validation"
    return [("training", sp["train"]["rows"], target(sp["train"])),
            (label, val["rows"], target(val)),
            ("test", sp["test"]["rows"], target(sp["test"]))]


def setup_lines(task: Task) -> tuple[list[str], str]:
    """Plain-text lines for the setup output: the splits and baselines, and a warning (or "") when
    validation and test have different target distributions."""
    f = {"splits": split_summary(task)}
    rows = split_rows(f)
    if not rows:
        return [], ""
    info = ["splits: " + "; ".join(f"{name} {n:,} rows ({target})" for name, n, target in rows)]
    if task.test_baseline is not None:
        info.append(f"baseline {task.metric_name}: validation {_r(task.baseline)}, "
                    f"test {_r(task.test_baseline)}")
    shift = distribution_shift(task)
    warning = (f"validation and test have different target distributions ({shift['detail']}): "
               "compare gains over baseline, not raw scores") if shift["flag"] else ""
    return info, warning


def no_spread_caveat(f: dict) -> str:
    rows = (f.get("splits") or {}).get("validation", {}).get("rows")
    return (f"one validation split{f' of {rows:,} rows' if rows else ''}: no spread across folds is "
            "available, so differences between experiments of a few thousandths are not evidence")


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
        "val_baseline": _r(task.baseline),
        "test_baseline": _r(task.test_baseline),
        "baseline_is": "best constant prediction that ignores the features",
        "has_spread": task.n_folds > 1,
        "splits": split_summary(task),
        "distribution_shift": distribution_shift(task),
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
        out["val_gain"] = out["best"]["gain_over_baseline"]
        out["test_gain"] = (None if best.test_metric is None or task.test_baseline is None
                            else _r(sign * (best.test_metric - task.test_baseline)))
        if _std(best) is not None:
            out["best"]["val_std_across_folds"] = _std(best)
        spread = out["best"].get("val_std_across_folds")
        gain = out["best"]["gain_over_baseline"]
        # The gain over the baseline, not the score, is what the fold spread has to be compared with.
        out["gain_within_spread"] = bool(gain <= spread) if spread else None
        out["gain_over_spread"] = _r(gain / spread, 2) if spread else None
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
    if f.get("gain_within_spread") is False:
        ev += (f"; the gain over the baseline ({f['val_gain']}) exceeds the spread across folds "
               f"({b['val_std_across_folds']})")
    lines.append(ev + f", across {f['experiments_run']} experiments.")
    caveats = []
    shift = f.get("distribution_shift") or {}
    if shift.get("flag"):
        # The splits have different target distributions, so the gap says nothing about overfitting.
        text = f"the validation and test splits have different target distributions ({shift['detail']})"
        if f.get("test_baseline") is not None:
            text += (f", so their baselines differ (validation {f['val_baseline']}, "
                     f"test {f['test_baseline']})")
        if f.get("test_gain") is not None:
            text += (f"; compare gains over each baseline (validation {f['val_gain']}, "
                     f"test {f['test_gain']}) rather than raw scores")
        caveats.append(text)
    elif shift.get("folds_differ"):
        caveats.append(f"{shift['detail']}, so the spread across folds partly reflects that")
    if not shift.get("flag") and b.get("val_test_gap") is not None and b["val_test_gap"] > 0.02:
        caveats.append(f"validation/test gap of {b['val_test_gap']} suggests some overfitting to validation")
    if not f.get("has_spread", True):
        caveats.append(no_spread_caveat(f))
    if f.get("gain_within_spread"):
        caveats.append(f"the gain over the baseline ({f['val_gain']}) is not larger than the spread "
                       f"across folds ({b['val_std_across_folds']}), so it is not evidence of a real "
                       "improvement")
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


def splits_md(f: dict) -> list[str]:
    """The "Splits" section: rows and target rate per split, test baseline, and the warnings."""
    rows = split_rows(f)
    if not rows:
        return []
    md = ["## Splits", "", "| split | rows | target |", "|---|---|---|"]
    md += [f"| {name} | {n:,} | {target} |" for name, n, target in rows]
    md.append("")
    if f.get("test_baseline") is not None:
        md += [f"Baseline {f['metric']}: validation {f['val_baseline']}, test {f['test_baseline']}.", ""]
    if f["distribution_shift"]["flag"]:
        md += [f"**Warning:** the splits have different target distributions "
               f"({f['distribution_shift']['detail']}), so scores and baselines differ between "
               "validation and test; compare gains over each baseline.", ""]
    elif f["distribution_shift"]["folds_differ"]:
        md += [f"**Note:** {f['distribution_shift']['detail']}.", ""]
    if not f["has_spread"]:
        md += [f"**Note:** {no_spread_caveat(f)}.", ""]
    return md


_SIGNIFICANCE = re.compile(r"statistical(?:ly)?\s+significan\w*|\bsignificance\b", re.IGNORECASE)
_OVERFIT = re.compile(r"overfit\w*|\bleak\w*", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:not|no|never|neither|nor|without|cannot|can't|isn't|rather than|instead of)\b",
                      re.IGNORECASE)


def blames_overfitting(text: str) -> bool:
    """Whether a sentence names overfitting or leakage without negating it ("not evidence of
    overfitting" is fine)."""
    return any(_OVERFIT.search(sent) and not _NEGATED.search(sent) for sent in _SENTENCE.split(text))


_CLAUSE = re.compile(r"(?<=[.!?])\s+|\n+|;")
_GAIN_WORD = re.compile(r"gain|improv|\blift|uplift|advantage|(?:over|above|beyond) (?:the )?baseline",
                        re.IGNORECASE)
_NOISE = r"(?:spread|noise|variab\w+|variation|standard deviation)"
_REAL_GAIN = re.compile(r"\b(?:real|clear|genuine|meaningful|true|robust|reliable)\s+(?:\w+\s+)?"
                        r"(?:gain|improvement|improvements|benefit|uplift|signal|advantage)\b"
                        rf"|\b(?:exceed\w*|beat\w*|surpass\w*|above|outside|(?:larger|greater|bigger) than)"
                        rf"\s+(?:the\s+)?(?:[\w-]+\s+){{0,3}}{_NOISE}", re.IGNORECASE)
_WITHIN_NOISE = re.compile(rf"\b(?:within|inside|smaller than|less than|below|under|no larger than|"
                           rf"not larger than)\s+(?:the\s+)?(?:[\w-]+\s+){{0,3}}{_NOISE}", re.IGNORECASE)
_OTHER_SUBJECT = re.compile(r"runner[- ]?up|second[- ]best|next best", re.IGNORECASE)


def _unnegated(text: str, claim: re.Pattern, subject: re.Pattern | None = None) -> bool:
    return any(claim.search(c) and not _NEGATED.search(c) and (subject is None or subject.search(c))
               and not _OTHER_SUBJECT.search(c) for c in _CLAUSE.split(text))


def gain_claims(narrative: str, f: dict) -> list[str]:
    """The narrative's verdict on the gain against the fold spread must match `gain_within_spread`.
    Silent without a spread (single split)."""
    within = f.get("gain_within_spread")
    if within is None:
        return []
    if within and _unnegated(narrative, _REAL_GAIN):
        return ["the gain over the baseline is not larger than the spread across folds "
                f"(gain_over_spread {f.get('gain_over_spread')}), so do not call it a real, clear or "
                "meaningful gain or say it exceeds the spread or the noise"]
    if within is False and _unnegated(narrative, _WITHIN_NOISE, _GAIN_WORD):
        return ["the gain over the baseline is larger than the spread across folds "
                f"(gain_over_spread {f.get('gain_over_spread')}), so do not say it is within the noise "
                "or the spread"]
    return []


def review(narrative: str, f: dict) -> list[str]:
    """Everything wrong with a narrative that can be checked against the facts."""
    found = [f"number {n} does not appear in FACTS" for n in check_numbers(narrative, f)] + \
        check_claims(narrative, f)
    if _SIGNIFICANCE.search(narrative):
        found.append("claims statistical significance, but no significance test was run")
    if (f.get("distribution_shift") or {}).get("flag") and blames_overfitting(narrative):
        found.append("the splits have different target rates, so the gap is not evidence of "
                     "overfitting; explain it through the different rates and baselines and compare "
                     "gains over baseline")
    return found + gain_claims(narrative, f)


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
    md += splits_md(f)
    if task.spec.notes:
        md += ["## Setup notes", ""] + [f"- {n}" for n in task.spec.notes] + [""]
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
