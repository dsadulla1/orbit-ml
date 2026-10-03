"""Per-split statistics, shift detection, the truthful narrative, `--time-order` and setup hints."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from orbit import html_report, prompts, report, results
from orbit.agent import Agent, SearchConfig
from orbit.cli import _write_report, app
from orbit.report import distribution_shift, review, template_narrative
from orbit.task import META_FILE, Task, TaskSpec, load_labeled, prepare, prepare_export_input, setup_hints

from conftest import LocalSandbox, ScriptedLLM, clf_reply

runner = CliRunner()


def _flat(text: str) -> str:
    return re.sub(r"[│╭╮╰╯─┃━\s]", "", text)


def _shifted_csv(tmp_path, labels=("yes", "no"), n=400):
    """File order is time order and the positive rate climbs: 5% in the training rows, 10% in the
    validation rows, 40% in the test rows (for a single 60/20/20 split)."""
    pos, neg = labels
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        if i < 0.6 * n:
            hit = i % 20 == 0
        elif i < 0.8 * n:
            hit = i % 10 == 0
        else:
            hit = i % 5 < 2
        rows.append({"pos": i, "f1": rng.normal(), "label": pos if hit else neg})
    path = tmp_path / "shifted.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _search(task, steps=2, pos="yes"):
    agent = Agent(task, ScriptedLLM([clf_reply(pos=pos), clf_reply("RF", "ensemble",
                  "RandomForestClassifier", "random_state=0")]), LocalSandbox(),
                  SearchConfig(max_steps=steps, n_drafts=steps))
    return agent.run()


@pytest.fixture
def shifted_run(tmp_path):
    task = prepare(TaskSpec(str(_shifted_csv(tmp_path)), "label", folds=1, time_order=True,
                            goal="Flag the positives"), tmp_path / "run")
    return task, _search(task)


# ---- per-split statistics -------------------------------------------------------------------

def test_split_stats_binary(tmp_path):
    n = 100
    y = ["yes" if (i < 60 and i % 10 == 0) or (60 <= i < 80 and i % 4 == 0) or (i >= 80 and i % 2 == 0)
         else "no" for i in range(n)]
    path = tmp_path / "d.csv"
    pd.DataFrame({"f": range(n), "label": y}).to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "label", folds=1, time_order=True, metric="accuracy"),
                   tmp_path / "run")
    st = task.split_stats
    assert st["positive"] == "yes"
    assert (st["train"]["rows"], st["train"]["positive_rate"]) == (60, pytest.approx(6 / 60))
    assert st["train"]["shares"] == {"no": pytest.approx(0.9), "yes": pytest.approx(0.1)}
    assert len(st["val"]) == 1 and st["val"][0]["fold"] == 1
    assert (st["val"][0]["rows"], st["val"][0]["positive_rate"]) == (20, pytest.approx(0.25))
    assert (st["test"]["rows"], st["test"]["positive_rate"]) == (20, pytest.approx(0.5))
    # accuracy of "always no" (the training mode): 0.75 on validation, 0.5 on test
    assert task.baseline == pytest.approx(0.75) and task.test_baseline == pytest.approx(0.5)
    assert json.loads(json.dumps(task.split_stats)) == task.split_stats
    assert Task.load(task.run_dir).split_stats == task.split_stats
    assert Task.load(task.run_dir).test_baseline == pytest.approx(0.5)


def test_split_stats_per_fold_and_test_baseline_for_auc(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    st = task.split_stats
    assert [v["fold"] for v in st["val"]] == [1, 2, 3]
    assert sum(v["rows"] for v in st["val"]) == 480 and st["test"]["rows"] == 120
    assert task.test_baseline == pytest.approx(0.5)      # a constant has no ranking power


def test_split_stats_regression_and_multiclass(tmp_path):
    n = 200
    path = tmp_path / "r.csv"
    pd.DataFrame({"f": range(n), "y": np.arange(n, dtype=float)}).to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "y", folds=1, time_order=True), tmp_path / "run")
    st = task.split_stats
    assert st["positive"] is None
    assert st["train"]["mean"] == pytest.approx(59.5) and st["train"]["rows"] == 120
    assert st["val"][0]["mean"] == pytest.approx(139.5) and st["test"]["mean"] == pytest.approx(179.5)
    assert st["test"]["std"] == pytest.approx(np.arange(160, 200).std())
    # the baseline is the training mean scored on the test labels
    assert task.test_baseline == pytest.approx(np.sqrt(np.mean((np.arange(160, 200) - 59.5) ** 2)))

    path = tmp_path / "m.csv"
    pd.DataFrame({"f": range(100), "c": ["a"] * 50 + ["b"] * 30 + ["c"] * 20}).to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "c", folds=1, time_order=True), tmp_path / "run2")
    assert task.split_stats["positive"] is None
    assert task.split_stats["test"]["shares"] == {"a": 0.0, "b": 0.0, "c": 1.0}
    assert task.split_stats["train"]["shares"]["a"] == pytest.approx(50 / 60)


def test_older_task_json_without_split_stats_loads(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    d = json.loads((task.run_dir / "task.json").read_text())
    del d["split_stats"], d["test_baseline"], d["spec"]["time_order"]
    (task.run_dir / "task.json").write_text(json.dumps(d))
    old = Task.load(task.run_dir)
    assert old.split_stats == {} and old.test_baseline is None and old.spec.time_order is False
    assert distribution_shift(old) == {"flag": False, "folds_differ": False, "detail": ""}
    assert report.split_summary(old) == {} and report.setup_lines(old) == ([], "")


def test_split_stats_are_not_written_into_sandbox_inputs(tmp_path):
    path = _shifted_csv(tmp_path)
    task = prepare(TaskSpec(str(path), "label", folds=1, time_order=True), tmp_path / "run")
    for f in task.input_dir.rglob("*"):
        if f.is_file():
            assert "split_stats" not in f.read_text(errors="ignore")
    assert not any("stats" in f.name for f in task.input_dir.rglob("*"))


# ---- shift detection ------------------------------------------------------------------------

def _binary(test, vals, rows=100):
    stats = {"positive": "p", "train": {"rows": 500}, "test": {"rows": rows, "positive_rate": test},
             "val": [{"fold": i, "rows": rows, "positive_rate": v} for i, v in enumerate(vals, 1)]}
    return SimpleNamespace(split_stats=stats, kind="classification", classes=["n", "p"])


@pytest.mark.parametrize("test,val,flag", [
    (0.20, 0.13, True),      # factor 1.54, difference 0.07
    (0.20, 0.14, False),     # factor 1.43
    (0.0705, 0.04, True),    # difference just above 0.03
    (0.069, 0.04, False),    # difference just below 0.03
    (0.04, 0.025, False),    # a big factor but a tiny difference
    (0.308, 0.074, True),    # the bank file
    (0.13, 0.20, True),      # the other direction
])
def test_binary_shift_thresholds(test, val, flag):
    assert distribution_shift(_binary(test, [val]))["flag"] is flag


def test_shift_needs_enough_rows_and_reports_fold_drift():
    assert distribution_shift(_binary(0.3, [0.05], rows=29))["flag"] is False
    s = distribution_shift(_binary(0.3, [0.054, 0.059, 0.111]))
    assert s["flag"] and s["folds_differ"] and "folds differ" in s["detail"]
    assert "30.0%" in s["detail"] and "in test vs" in s["detail"]
    s = distribution_shift(_binary(0.07, [0.03, 0.07, 0.12]))      # test inside the folds' range
    assert not s["flag"] and s["folds_differ"] and s["detail"].startswith("validation folds differ")
    assert distribution_shift(_binary(0.07, [0.07, 0.075]))["folds_differ"] is False


def _multi(test_a, val_a):
    mk = lambda a: {"rows": 100, "shares": {"a": a, "b": 1 - a}}
    stats = {"positive": None, "train": mk(0.5), "test": mk(test_a),
             "val": [{"fold": 1, **mk(val_a)}]}
    return SimpleNamespace(split_stats=stats, kind="classification", classes=["a", "b", "c"])


def test_multiclass_shift_thresholds():
    assert distribution_shift(_multi(0.50, 0.39))["flag"] is True
    assert distribution_shift(_multi(0.50, 0.41))["flag"] is False
    assert "class a" in distribution_shift(_multi(0.50, 0.39))["detail"]


def _regression(test_mean, val_mean):
    stats = {"positive": None, "train": {"rows": 300, "mean": 9.0, "std": 2.0},
             "test": {"rows": 100, "mean": test_mean, "std": 2.0},
             "val": [{"fold": 1, "rows": 100, "mean": val_mean, "std": 2.0}]}
    return SimpleNamespace(split_stats=stats, kind="regression", classes=[])


def test_regression_shift_thresholds():
    assert distribution_shift(_regression(11.1, 10.0))["flag"] is True    # 1.1 > 0.5 * 2
    assert distribution_shift(_regression(10.9, 10.0))["flag"] is False
    assert distribution_shift(_regression(8.9, 10.0))["flag"] is True
    assert "target mean" in distribution_shift(_regression(11.1, 10.0))["detail"]


# ---- facts and narrative --------------------------------------------------------------------

def test_facts_report_splits_baselines_and_gains(shifted_run):
    task, journal = shifted_run
    f = report.facts(task, journal)
    assert f["splits"]["train"] == {"rows": 240, "positive_rate": 0.05}
    assert f["splits"]["validation"]["rows"] == 80 and f["splits"]["validation"]["positive_rate"] == 0.1
    assert f["splits"]["test"]["positive_rate"] == 0.4
    assert f["val_baseline"] == f["baseline_val_score"] and f["test_baseline"] == 0.5
    b = f["best"]
    assert f["val_gain"] == pytest.approx(b["val_score"] - f["val_baseline"], abs=1e-3)
    assert f["test_gain"] == pytest.approx(b["test_score"] - f["test_baseline"], abs=1e-3)
    assert f["distribution_shift"]["flag"] is True and f["has_spread"] is False
    assert f["validation"] == "single split ordered by file order"


def test_template_narrative_explains_the_shift_without_overfitting(shifted_run):
    task, journal = shifted_run
    f = report.facts(task, journal)
    f["best"]["val_test_gap"] = 0.2           # a large gap that would normally say "overfitting"
    text = template_narrative(f)
    assert "different target distributions" in text and "baselines differ" in text
    assert f"validation {f['val_baseline']}, test {f['test_baseline']}" in text
    assert f"validation {f['val_gain']}, test {f['test_gain']}" in text
    assert "overfit" not in text.lower() and "leak" not in text.lower()
    assert "40.0%" in text and "10.0%" in text
    assert review(text, f) == []              # every number is traceable to the facts


def test_template_narrative_without_shift_keeps_the_overfitting_wording(shifted_run):
    task, journal = shifted_run
    f = report.facts(task, journal)
    f["distribution_shift"] = {"flag": False, "folds_differ": False, "detail": ""}
    f["best"]["val_test_gap"] = 0.2
    assert "suggests some overfitting to validation" in template_narrative(f)
    f["best"]["val_test_gap"] = 0.01
    assert "overfit" not in template_narrative(f)


def test_single_split_caveat_in_narrative_and_report(shifted_run):
    task, journal = shifted_run
    f = report.facts(task, journal)
    text = template_narrative(f)
    assert "one validation split of 80 rows: no spread across folds is available" in text
    assert "a few thousandths are not evidence" in text
    md, _ = report.build(task, journal, None)
    assert "## Splits" in md and "| test | 80 | positive rate 40.0% |" in md
    assert "| validation | 80 | positive rate 10.0% |" in md and "| training | 240 |" in md
    assert "**Warning:** the splits have different target distributions" in md
    assert "Baseline roc_auc: validation" in md and "no spread across folds" in md
    assert "All numbers verified" in md


def test_cross_validated_run_has_a_spread_and_no_single_split_caveat(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    journal = _search(task)
    f = report.facts(task, journal)
    assert f["has_spread"] is True and f["distribution_shift"]["flag"] is False
    assert f["splits"]["validation"]["folds"] == 3 and len(f["splits"]["validation"]["by_fold"]) == 3
    md, _ = report.build(task, journal, None)
    assert "no spread across folds" not in md and "**Warning:**" not in md


def test_narrator_is_told_about_shift_and_spread(shifted_run):
    task, journal = shifted_run
    assert "distribution_shift.flag" in report.NARRATOR_SYSTEM and "overfitting or leakage" in report.NARRATOR_SYSTEM
    assert "has_spread is false" in report.NARRATOR_SYSTEM and "no cross-validation spread" in report.NARRATOR_SYSTEM
    llm = _Narrator("**Verdict:** a.")
    report.build(task, journal, llm)
    assert '"distribution_shift"' in llm.prompts[0] and '"test_baseline"' in llm.prompts[0]


class _Narrator:
    model = "narrator"

    def __init__(self, *texts):
        self.texts, self.prompts = list(texts), []

    def complete(self, system, user):
        from orbit.llm import Completion
        self.prompts.append(user)
        return Completion(self.texts.pop(0) if len(self.texts) > 1 else self.texts[0], 1, 1)


# ---- review ---------------------------------------------------------------------------------

def test_review_rejects_overfitting_and_leakage_explanations_under_shift(shifted_run):
    task, journal = shifted_run
    f = report.facts(task, journal)
    for bad in ("The gap suggests overfitting to the validation set.",
                "A large gap points to data leakage.", "Probably Overfit."):
        issues = review(bad, f)
        assert issues == ["the splits have different target rates, so the gap is not evidence of "
                          "overfitting; explain it through the different rates and baselines and "
                          "compare gains over baseline"]
    assert review("The gap is not evidence of overfitting; the test baseline is higher.", f) == []
    assert review("Rates differ between splits.", f) == []
    f["distribution_shift"] = {"flag": False, "folds_differ": False, "detail": ""}
    assert review("The gap suggests overfitting.", f) == []


def test_review_failure_revises_once_then_falls_back_to_the_template(shifted_run):
    task, journal = shifted_run
    f = report.facts(task, journal)
    good = "**Verdict:** a. **Evidence:** the test rate is higher. **Caveats:** none. **Next steps:** b."
    llm = _Narrator("The gap suggests overfitting.", good)
    text, source, found = report._write(llm, f)
    assert text == good and source.endswith("revised after review") and "not evidence of overfitting" in found[0]
    assert "not evidence of overfitting" in llm.prompts[1]

    llm = _Narrator("It is data leakage.")
    text, source, found = report._write(llm, f)
    assert source.startswith("template") and "overfit" not in text.lower() and len(found) == 2
    md, out = report.build(task, journal, _Narrator("It is data leakage."))
    assert "leakage" not in md.split("## Splits")[0].lower() and out["narrative_unverified"] == []


# ---- results, HTML, CLI ---------------------------------------------------------------------

def test_html_shows_splits_baseline_gain_and_warning_and_escapes(tmp_path):
    path = _shifted_csv(tmp_path, labels=("<b>y</b>", "n"))
    task = prepare(TaskSpec(str(path), "label", folds=1, time_order=True), tmp_path / "run")
    journal = _search(task, pos="<b>y</b>")
    md, _ = report.build(task, journal, None)
    (task.run_dir / "report.md").write_text(md)
    data = results.collect(task.run_dir)
    assert data["task"]["test_baseline"] == 0.5 and data["summary"]["splits"]["test"]["rows"] == 80
    page = html_report.render(data)
    assert "<h2>Splits</h2>" in page and "<th>split</th>" in page
    f = data["summary"]
    assert f"baseline {f['test_baseline']:.4f}, gain {f['test_gain']:.4f}" in page
    assert 'class="warn"' in page and "different target distributions" in page
    assert "no spread across folds is available" in page
    assert "<b>y</b>" not in page and "&lt;b&gt;y&lt;/b&gt;" in page


def test_html_without_shift_has_no_warning(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    journal = _search(task)
    (task.run_dir / "report.md").write_text(report.build(task, journal, None)[0])
    page = html_report.render(results.collect(task.run_dir))
    assert "<h2>Splits</h2>" in page and 'class="warn"' not in page.replace(".warn{", "")
    assert "no spread across folds" not in page and "validation (mean of 3 folds)" in page


def test_html_for_a_run_without_stored_splits_still_renders(shifted_run):
    task, journal = shifted_run
    d = json.loads((task.run_dir / "task.json").read_text())
    d["split_stats"], d["test_baseline"] = {}, None
    (task.run_dir / "task.json").write_text(json.dumps(d))
    page = html_report.render(results.collect(task.run_dir))
    assert "<h2>Splits</h2>" not in page and "scored once, for the winner" in page


def test_orbit_results_prints_splits_baselines_and_warning(shifted_run):
    task, journal = shifted_run
    _write_report(task, journal, None)
    out = _flat(runner.invoke(app, ["results", str(task.run_dir)]).output)
    assert _flat("Split test: 80 rows, positive rate 40.0%") in out
    assert _flat("Split training: 240 rows, positive rate 5.0%") in out
    assert _flat(f"Baseline: validation {task.baseline:.4f}") not in out  # facts are rounded to 4 places
    assert "Baseline:validation" in out and "gain:validation" in out
    assert "Thesplitshavedifferenttargetdistributions" in out and "onevalidationsplitof80rows" in out


def test_setup_lines_show_splits_and_the_shift(shifted_run):
    task, _ = shifted_run
    info, warning = report.setup_lines(task)
    assert info[0].startswith("splits: training 240 rows (positive rate 5.0%); validation 80 rows")
    assert info[1].startswith("baseline roc_auc: validation")
    assert "different target distributions" in warning and "40.0% in test" in warning


# ---- single split note ----------------------------------------------------------------------

def _big_csv(tmp_path, n=13_000):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n),
                       "label": rng.choice(["a", "b"], size=n)})
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    return path


NOTE = ("single validation split (more than 10,000 rows); --folds 5 adds a spread at five times "
        "the sandbox cost")


def test_default_single_split_is_noted_but_an_explicit_one_is_not(tmp_path):
    path = _big_csv(tmp_path)
    task = prepare(TaskSpec(str(path), "label"), tmp_path / "auto")
    assert task.n_folds == 1 and NOTE in task.spec.notes
    explicit = prepare(TaskSpec(str(path), "label", folds=1), tmp_path / "one")
    assert NOTE not in explicit.spec.notes
    small = prepare(TaskSpec(str(path), "label", folds=3), tmp_path / "three")
    assert NOTE not in small.spec.notes


def test_small_files_use_cv_without_the_note(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label"), tmp_path / "run")
    assert task.n_folds == 5 and NOTE not in task.spec.notes


# ---- --time-order ---------------------------------------------------------------------------

def _ordered_csv(tmp_path, n=300, missing=(0, 7, 150)):
    """`pos` is each row's position in the file, so it can be checked against what a script sees."""
    df = pd.DataFrame({"pos": range(n), "f1": [i % 7 for i in range(n)],
                       "label": [("yes" if i % 3 == 0 else "no") for i in range(n)]})
    df.loc[list(missing), "label"] = None
    path = tmp_path / "ordered.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("folds", [1, 3])
def test_time_order_validates_on_later_rows_without_adding_a_column(tmp_path, folds):
    path = _ordered_csv(tmp_path)
    source = pd.read_csv(path)
    task = prepare(TaskSpec(str(path), "label", folds=folds, time_order=True), tmp_path / "run")
    plain = prepare(TaskSpec(str(path), "label", folds=folds), tmp_path / "plain")
    labels = pd.read_csv(task.run_dir / "hidden" / "labels.csv")
    assert list(labels.columns) == ["orbit_id", "split", "fold", "target"]
    assert set(labels.loc[labels["split"] == "test", "fold"]) == {0}
    assert set(labels.loc[labels["split"] == "val", "fold"]) == set(range(1, folds + 1))
    dev_max, test_pos, train_pos = 0, None, set()
    for k in range(1, folds + 1):
        d = task.fold_dir(k)
        train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
        ids = labels[(labels["split"] == "val") & (labels["fold"] == k)]["orbit_id"]
        val = test[test["orbit_id"].isin(ids)]
        tst = test[test["orbit_id"].isin(labels[labels["split"] == "test"]["orbit_id"])]
        assert len(val) and len(tst) and len(val) + len(tst) == len(test)
        assert train["pos"].max() < val["pos"].min()
        assert train["pos"].is_monotonic_increasing                  # file order is kept
        dev_max = max(dev_max, train["pos"].max(), val["pos"].max())
        test_pos = tst["pos"] if test_pos is None else test_pos
        assert set(tst["pos"]) == set(test_pos)
        train_pos |= set(train["pos"])
        assert list(train.columns) == list(source.columns)
        assert list(test.columns) == ["orbit_id", "pos", "f1"]
        assert list(json.loads((d / META_FILE).read_text())) == \
            list(json.loads((plain.fold_dir(k) / META_FILE).read_text()))
        assert json.loads((d / META_FILE).read_text())["time_col"] is None
    assert test_pos.min() > dev_max                                  # every test row is later
    assert not set(test_pos) & train_pos                             # and never in a train file
    assert not {0, 7, 150} & (train_pos | set(test_pos))             # rows without a target are dropped
    assert len(test_pos) == round((300 - 3) * 0.2) and task.spec.time_order is True
    assert Task.load(task.run_dir).spec.time_order is True


def test_time_order_folds_are_an_expanding_window(tmp_path):
    path = _ordered_csv(tmp_path, missing=())
    task = prepare(TaskSpec(str(path), "label", folds=3, time_order=True), tmp_path / "run")
    trains = [set(pd.read_csv(task.fold_dir(k) / "train.csv")["pos"]) for k in (1, 2, 3)]
    assert trains[0] < trains[1] < trains[2]
    assert max(trains[2]) < 240 and sorted(trains[2]) == list(range(len(trains[2])))


def test_time_order_and_time_col_together_is_an_error(tmp_path):
    path = _ordered_csv(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        prepare(TaskSpec(str(path), "label", time_col="pos", time_order=True), tmp_path / "run")
    with pytest.raises(ValueError, match="not both"):
        load_labeled(TaskSpec(str(path), "label", time_col="pos", time_order=True))
    result = runner.invoke(app, ["run", str(path), "-t", "label", "--time-col", "pos", "--time-order"])
    assert result.exit_code != 0 and "notboth" in _flat(result.output)
    from orbit.problem import setup_task
    with pytest.raises(ValueError, match="not both"):
        setup_task([str(path)], tmp_path / "r2", target="label", time_col="pos", time_order=True)
    assert not (tmp_path / "r2").exists()


def test_time_order_export_input_keeps_file_order(tmp_path):
    path = _ordered_csv(tmp_path)
    source = pd.read_csv(path).dropna(subset=["label"]).reset_index(drop=True)
    spec = TaskSpec(str(path), "label", folds=1, time_order=True)
    pd.testing.assert_frame_equal(load_labeled(spec), source)
    task = prepare(spec, tmp_path / "run")
    out = prepare_export_input(task)
    train = pd.read_csv(out / "train.csv")
    assert train["pos"].tolist() == source["pos"].tolist() and list(train.columns) == list(source.columns)
    assert list(pd.read_csv(out / "test.csv").columns) == ["orbit_id", "pos", "f1"]


def test_time_order_wording_in_prompt_report_and_tags(tmp_path):
    from orbit.tracking import _run_tags
    path = _ordered_csv(tmp_path)
    for folds, text in [(1, "single split ordered by file order"),
                        (3, "mean of 3 forward-chaining folds ordered by file order")]:
        task = prepare(TaskSpec(str(path), "label", folds=folds, time_order=True), tmp_path / f"r{folds}")
        assert report._validation(task) == text
        assert _run_tags(task, SearchConfig())["orbit.validation"] == text
        section = prompts.data_section(task)
        assert ("Validation and test rows come later in the file than the training rows; features "
                "must not use information from later rows.") in section
        assert "later in the file" not in prompts.data_section(task, export=True)
    plain = prepare(TaskSpec(str(path), "label", folds=1), tmp_path / "plain")
    assert "later in the file" not in prompts.data_section(plain)


# ---- setup hints ----------------------------------------------------------------------------

def _climbing(n=2000, lo=0.03, hi=0.50, seed=0):
    rng = np.random.default_rng(seed)
    p = lo + (hi - lo) * np.arange(n) / n
    return pd.DataFrame({"f0": rng.normal(size=n), "label": np.where(rng.random(n) < p, "yes", "no")})


def _hints(df, tmp_path, **spec):
    path = tmp_path / "h.csv"
    df.to_csv(path, index=False)
    s = TaskSpec(str(path), "label", **spec)
    df = load_labeled(s)
    from orbit.task import infer_kind
    kind = infer_kind(df["label"])
    return setup_hints(df, s, kind, sorted(map(str, df["label"].unique())) if kind == "classification" else [])


def test_drift_hint_for_a_bank_like_file(tmp_path):
    df = _climbing()
    hints = _hints(df, tmp_path)
    assert len(hints) == 1 and hints[0].startswith("the target rate changes along the file (first fifth ")
    assert "last fifth" in hints[0] and "use --time-order" in hints[0] and "otherwise shuffle the file" in hints[0]
    assert "random validation will be optimistic for later periods" in hints[0]
    first = re.search(r"first fifth ([\d.]+)%", hints[0]).group(1)
    assert 1 < float(first) < 10
    # end to end: the note reaches the task, and --time-order silences it
    path = tmp_path / "bank.csv"
    df.to_csv(path, index=False)
    task = prepare(TaskSpec(str(path), "label", folds=1), tmp_path / "run")
    assert any("changes along the file" in n for n in task.spec.notes)
    ordered = prepare(TaskSpec(str(path), "label", folds=1, time_order=True), tmp_path / "run2")
    assert not any("changes along the file" in n for n in ordered.spec.notes)
    assert ordered.split_stats["test"]["positive_rate"] > 0.3 > ordered.split_stats["train"]["positive_rate"]


def test_drift_hint_for_a_regression_target(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({"f0": rng.normal(size=n), "label": np.arange(n) / n * 10 + rng.normal(size=n)})
    hints = _hints(df, tmp_path)
    assert len(hints) == 1 and hints[0].startswith("the target mean changes along the file (first fifth ")
    df["label"] = rng.permutation(df["label"].to_numpy())
    assert _hints(df, tmp_path) == []


@pytest.mark.parametrize("rate", [0.02, 0.05, 0.3, 0.5])
@pytest.mark.parametrize("n", [500, 1500])
def test_hints_stay_silent_for_shuffled_data(tmp_path, rate, n):
    for seed in range(15):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.integers(0, 5, n),
                           "label": np.where(rng.random(n) < rate, "yes", "no")})
        assert setup_hints(df, TaskSpec("x.csv", "label"), "classification", ["no", "yes"]) == [], seed
        reg = df.assign(label=rng.normal(size=n) * 3 + 5)
        assert setup_hints(reg, TaskSpec("x.csv", "label"), "regression", []) == [], seed


def test_hints_stay_silent_for_small_files_and_ordered_runs(tmp_path):
    assert _hints(_climbing(n=400), tmp_path) == []
    path = tmp_path / "t.csv"
    df = _climbing().assign(when=pd.date_range("2020-01-01", periods=2000).strftime("%Y-%m-%d"))
    df.to_csv(path, index=False)
    for spec in ({"time_order": True}, {"time_col": "when"}):
        task = prepare(TaskSpec(str(path), "label", folds=1, **spec), tmp_path / f"run{len(spec)}{list(spec)[0]}")
        assert not any("looks like a date" in n or "along the file" in n for n in task.spec.notes)


def test_date_hint_by_name_and_by_content(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "f0": rng.normal(size=n),
        "signup_date": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
        "created": pd.date_range("2021-03-01", periods=n, freq="h").strftime("%d/%m/%Y %H:%M"),
        "pdays": rng.integers(0, 999, n),                          # numeric, only "day" in its name
        "year": rng.integers(1990, 2020, n),                       # numeric with a clear name
        "weekday_kind": rng.choice(["wd", "we"], n),               # two values: never a date
        "town": rng.choice(["oslo", "rome", "kyiv", "lima"], n),
        "code": rng.integers(0, 50, n).astype(str),                # numbers written as text
        "label": rng.choice(["yes", "no"], n)})
    hints = _hints(df, tmp_path)
    assert hints == [
        "column 'signup_date' looks like a date or time: validation here is random, so scores are "
        "optimistic if you will predict later periods; use --time-col signup_date to validate on "
        "later rows",
        "column 'created' looks like a date or time: validation here is random, so scores are "
        "optimistic if you will predict later periods; use --time-col created to validate on later rows"]
    only = df[["f0", "pdays", "weekday_kind", "town", "code", "label"]]
    assert _hints(only, tmp_path) == []
    assert _hints(df[["f0", "year", "label"]], tmp_path)[0].startswith("column 'year' looks like")


def test_month_and_weekday_names_are_not_dates(tmp_path):
    rng = np.random.default_rng(1)
    n = 800
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    df = pd.DataFrame({
        "f0": rng.normal(size=n),
        "month": rng.choice(months, n),
        "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "label": rng.choice(["yes", "no"], n)})
    assert _hints(df, tmp_path) == []          # parseable, but they cannot order rows
    spans_years = df.assign(when=pd.date_range("2019-01-01", periods=n, freq="W").strftime("%b %Y"))
    assert any("'when'" in h for h in _hints(spans_years, tmp_path))  # month-year text spans several years


def test_the_target_is_never_flagged_as_a_date(tmp_path):
    n = 600
    df = pd.DataFrame({"f0": range(n), "event_date": ["a", "b", "c"] * (n // 3)})
    s = TaskSpec("x.csv", "event_date")
    assert setup_hints(df, s, "classification", ["a", "b", "c"]) == []
