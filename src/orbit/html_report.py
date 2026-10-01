"""Render `results.collect()` as one self-contained HTML file (no scripts, no external assets).

Charts are inline SVG built here; colours come from CSS custom properties so the page follows the
viewer's light/dark setting. The leaderboard table is the accessible view of the same numbers.
"""
import re
from html import escape

from jinja2 import Environment
from markupsafe import Markup

W, ROW, PAD_L, PAD_R, PAD_T, PAD_B = 720, 28, 56, 90, 14, 34


def _fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def narrative_html(md: str) -> Markup:
    """The narrative uses **bold** labels, paragraphs and numbered lists; nothing else."""
    out, items = [], []

    def flush():
        if items:
            out.append("<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>")
            items.clear()

    for block in re.split(r"\n\s*\n", md.strip()):
        for line in block.splitlines():
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape(line.strip()))
            m = re.match(r"^\d+[.)]\s+(.*)", text)
            if m:
                items.append(m.group(1))
            elif text:
                flush()
                out.append(f"<p>{text}</p>")
        flush()
    return Markup("".join(out))


def _scale(lo, hi, a, b):
    span = (hi - lo) or 1.0
    return lambda v: a + (v - lo) / span * (b - a)


def _ticks(lo, hi, n=4):
    span = (hi - lo) or 1.0
    return [lo + span * i / n for i in range(n + 1)]


def score_chart(exps, baseline, best_id, higher_better, metric) -> Markup:
    """One dot per working experiment (score, with +/- spread across folds) against the baseline."""
    ok = [e for e in exps if e["val_score"] is not None]
    if not ok:
        return Markup("")
    lows = [e["val_score"] - (e["val_std"] or 0) for e in ok] + [baseline]
    highs = [e["val_score"] + (e["val_std"] or 0) for e in ok] + [baseline]
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.06 or 0.01
    x = _scale(lo - pad, hi + pad, PAD_L, W - PAD_R)
    h = PAD_T + ROW * len(ok) + PAD_B
    parts = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="Validation {escape(metric)} per experiment">']
    for t in _ticks(lo - pad, hi + pad):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{PAD_T}" y2="{h - PAD_B}"/>'
                     f'<text class="tick" x="{x(t):.1f}" y="{h - PAD_B + 16}" text-anchor="middle">{t:.4g}</text>')
    bx = x(baseline)
    parts.append(f'<line class="base" x1="{bx:.1f}" x2="{bx:.1f}" y1="{PAD_T}" y2="{h - PAD_B}"/>'
                 f'<text class="tick" x="{bx:.1f}" y="{PAD_T - 3}" text-anchor="middle">baseline</text>')
    for i, e in enumerate(ok):
        y = PAD_T + ROW * i + ROW / 2
        s, sd = e["val_score"], e["val_std"] or 0
        best = e["id"] == best_id
        tip = (f"#{e['id']} {e['stage']}: {_fmt(s)}" + (f" ± {_fmt(sd)}" if e["val_std"] else "")
               + f"\n{e['plan'][:120]}")
        parts.append(f'<text class="tick" x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end">#{e["id"]}</text>')
        if sd:
            parts.append(f'<line class="err" x1="{x(s - sd):.1f}" x2="{x(s + sd):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<circle class="dot{" best" if best else ""}" cx="{x(s):.1f}" cy="{y:.1f}" r="{6 if best else 5}">'
                     f'<title>{escape(tip)}</title></circle>')
        if best:
            parts.append(f'<text class="lab" x="{x(s + sd) + 10:.1f}" y="{y + 4:.1f}">best {s:.4g}</text>')
    arrow = "higher is better" if higher_better else "lower is better"
    parts.append(f'<text class="tick" x="{W - PAD_R}" y="{h - 4}" text-anchor="end">{escape(metric)} ({arrow})</text></svg>')
    return Markup("".join(parts))


def progress_chart(exps, higher_better, metric) -> Markup:
    """Best validation score so far, after each experiment."""
    best, pts = None, []
    for e in exps:
        s = e["val_score"]
        if s is not None and (best is None or (s > best if higher_better else s < best)):
            best = s
        pts.append((e["id"], best))
    pts = [p for p in pts if p[1] is not None]
    if len(pts) < 2:
        return Markup("")
    h = 190
    xs = _scale(pts[0][0], pts[-1][0], PAD_L, W - PAD_R)
    vals = [v for _, v in pts]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or 0.01
    ys = _scale(hi + pad, lo - pad, 14, h - 34)
    parts = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Best score so far">']
    for t in _ticks(lo - pad, hi + pad, 3):
        parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                     f'<text class="tick" x="{PAD_L - 8}" y="{ys(t) + 4:.1f}" text-anchor="end">{t:.4g}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{xs(i_):.1f},{ys(v):.1f}" for i, (i_, v) in enumerate(pts))
    parts.append(f'<path class="line" d="{path}"/>')
    for i_, v in pts:
        parts.append(f'<circle class="dot" cx="{xs(i_):.1f}" cy="{ys(v):.1f}" r="4">'
                     f'<title>after #{i_}: best {v:.4f}</title></circle>')
    parts.append(f'<text class="tick" x="{W - PAD_R}" y="{h - 8}" text-anchor="end">experiment number</text></svg>')
    return Markup("".join(parts))


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6e69;
--line:#e3e2de;--series:#2a78d6;--bad:#c2362f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--card:#222221;
--ink:#ffffff;--ink2:#c3c2b7;--muted:#9a998f;--line:#383835;--series:#3987e5;--bad:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#222221;--ink:#ffffff;--ink2:#c3c2b7;
--muted:#9a998f;--line:#383835;--series:#3987e5;--bad:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.6rem;line-height:1.25;margin:0 0 4px}h2{font-size:1.1rem;margin:36px 0 12px}
.sub{color:var(--ink2);margin:0}.meta{color:var(--muted);font-size:.9rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}
.tile,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{color:var(--ink2);font-size:.85rem}.tile .v{font-size:1.7rem;font-weight:600;line-height:1.2}
.tile .d{color:var(--muted);font-size:.8rem}
.card p{margin:.4em 0}.card ol{margin:.3em 0 .3em 1.2em;padding:0}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--line);stroke-width:1}.chart .base{stroke:var(--muted);stroke-width:1;stroke-dasharray:0}
.chart .tick{fill:var(--muted);font-size:11px}.chart .lab{fill:var(--ink);font-size:12px;font-weight:600}
.chart .dot{fill:var(--series);stroke:var(--card);stroke-width:2}.chart .dot.best{stroke:var(--ink);stroke-width:2}
.chart .err{stroke:var(--series);stroke-width:2;opacity:.55;stroke-linecap:round}
.chart .line{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;white-space:nowrap}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
tr.best td{font-weight:600}.fail{color:var(--bad)}
details{margin-top:12px}summary{cursor:pointer;color:var(--ink2)}
pre{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;overflow-x:auto;
font-size:.82rem;line-height:1.45}
a{color:var(--series)}ul.notes{color:var(--ink2);padding-left:1.2em}
@media (max-width:520px){main{padding-top:20px}.tile .v{font-size:1.4rem}}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>orbit report: {{ title }}</title><style>{{ css }}</style></head>
<body><main>
<h1>{{ title }}</h1>
<p class="sub">{{ subtitle }}</p>
<p class="meta">{{ meta }}</p>

<div class="tiles">
{% for t in tiles %}<div class="tile"><div class="k">{{ t.k }}</div><div class="v">{{ t.v }}</div><div class="d">{{ t.d }}</div></div>
{% endfor %}</div>

{% if narrative %}<h2>Summary</h2><div class="card">{{ narrative }}<p class="meta">{{ check }}</p></div>{% endif %}

{% if chart %}<h2>Validation score per experiment</h2>
<div class="card">{{ chart }}<p class="meta">Each dot is one working experiment{% if folds > 1 %}; the bar is the spread across {{ folds }} folds{% endif %}. Differences smaller than the bar are within noise.</p></div>{% endif %}
{% if progress %}<h2>Search progress</h2><div class="card">{{ progress }}</div>{% endif %}

<h2>Experiments</h2>
<div class="card scroll"><table>
<tr><th>#</th><th>stage</th><th>parent</th><th>validation</th><th>test</th><th>seconds</th><th>approach</th></tr>
{% for e in rows %}<tr class="{{ 'best' if e.best }}">
<td class="n">{{ e.id }}</td><td>{{ e.stage }}</td><td class="n">{{ e.parent or '' }}</td>
<td class="n">{% if e.status == 'ok' %}{{ e.val }}{% else %}<span class="fail">failed</span>{% endif %}</td>
<td class="n">{{ e.test }}</td><td class="n">{{ e.seconds }}</td>
<td>{% if e.status == 'ok' %}{{ e.plan }}{% else %}<span class="fail">{{ e.error }}</span> &middot; {{ e.plan }}{% endif %}</td></tr>
{% endfor %}</table></div>

{% if code %}<h2>Best solution</h2>
<div class="card"><p class="meta">{{ script }}</p><details><summary>Show the script</summary><pre>{{ code }}</pre></details></div>{% endif %}

{% if notes %}<h2>How the task was set up</h2><div class="card"><ul class="notes">
{% for n in notes %}<li>{{ n }}</li>{% endfor %}</ul></div>{% endif %}

<p class="meta" style="margin-top:32px">{{ footer }}{% if link %} &middot; <a href="{{ link }}">Open in MLflow</a>{% endif %}</p>
</main></body></html>
"""

_env = Environment(autoescape=True)


def render(data: dict) -> str:
    """HTML for one run, from `results.collect()`."""
    prob, task, s = data["problem"], data["task"], data["summary"]
    exps = data["experiments"]
    best = s.get("best") or {}
    title = prob["goal"] or f"Predict {prob['target']}"
    subtitle = (f"Evaluation: {prob['eval']}" if prob["eval"] else f"Metric: {task['metric']}")
    data_bits = [prob["primary_table"]] + prob["extra_tables"]
    meta = (f"Target {prob['target']} · data {', '.join(data_bits)} · {task['validation']} · "
            f"{s['experiments_run']} experiments · {s['minutes']} min · stopped: {data['stop_reason']}")
    sd = best.get("val_std_across_folds")
    tiles = [
        {"k": f"Best validation {task['metric']}", "v": _fmt(best.get("val_score")),
         "d": f"± {_fmt(sd)} across folds" if sd is not None else task["validation"]},
        {"k": "Held-out test score", "v": _fmt(best.get("test_score")), "d": "scored once, for the winner"},
        {"k": "Baseline (ignores features)", "v": _fmt(task["baseline"]),
         "d": f"gain {_fmt(best.get('gain_over_baseline'))}" if best else ""},
        {"k": "Experiments", "v": f"{s['experiments_run'] - s['experiments_failed']} ok",
         "d": f"{s['experiments_failed']} failed, {s['debug_fixes']}/{s['debug_attempts']} debug fixes"},
    ]
    best_id = (data.get("best") or {}).get("id")
    rows = [{**e, "best": e["id"] == best_id, "val": _fmt(e["val_score"]) + (f" ± {_fmt(e['val_std'], 3)}" if e["val_std"] else ""),
             "test": _fmt(e["test_score"]) if e["test_score"] is not None else ""} for e in exps]
    rows.sort(key=lambda r: (r["val_score"] is None,
                             -(r["val_score"] or 0) if task["greater_is_better"] else (r["val_score"] or 0)))
    tracker = data.get("tracker") or {}
    return _env.from_string(TEMPLATE).render(
        css=Markup(CSS), title=title, subtitle=subtitle, meta=meta, tiles=tiles,
        narrative=narrative_html(data["narrative"]) if data["narrative"] else "", check=data["narrative_check"],
        chart=score_chart(exps, task["baseline"], best_id, task["greater_is_better"], task["metric"]),
        progress=progress_chart(exps, task["greater_is_better"], task["metric"]), folds=task["folds"],
        rows=rows, code=(data.get("best") or {}).get("code"), script=(data.get("best") or {}).get("script"),
        notes=prob["notes"], link=tracker.get("url"), footer=f"orbit run {data['run']}")
