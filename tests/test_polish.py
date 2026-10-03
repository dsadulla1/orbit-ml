"""Gain versus spread in the narrative, MLflow metric names, the data fingerprint, the experiment
time limit, setup notes in report.md, narrow terminals and the HTML title."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from orbit import html_report, prompts, report, results
from orbit.agent import Agent, Journal, Node, SearchConfig
from orbit.cli import app
from orbit.export import export_best
from orbit.report import review, template_narrative
from orbit.sandbox import ExecResult
from orbit.task import TaskSpec, prepare
from orbit.tracking import MLflowTracker, data_fingerprint

from conftest import BROKEN, LocalSandbox, ScriptedLLM, clf_reply
from test_export import CLF
from test_tracking import mlflow_uri  # noqa: F401 - fixture

runner = CliRunner()


# ---- gain versus spread ---------------------------------------------------------------------

def _cv(tmp_path, csv, scores, folds=3):
    """A finished 3-fold run whose best experiment has the given fold scores."""
    task = prepare(TaskSpec(str(csv), "label", folds=folds), tmp_path / "run")
    node = Node(id=1, stage="draft", parent=None, plan="p", code="x", fold_scores=scores,
                metric=sum(scores) / len(scores), test_metric=0.7)
    return task, Journal(nodes=[node], best_id=1, stop_reason="done")


def _facts(tmp_path, clf_csv, scores):
    task, journal = _cv(tmp_path, clf_csv, scores)
    return report.facts(task, journal), task, journal


def test_facts_compare_the_gain_with_the_spread(tmp_path, clf_csv):
    small, task, _ = _facts(tmp_path, clf_csv, [0.50, 0.55, 0.45])   # gain ~0, spread 0.05
    assert small["gain_within_spread"] is True and small["gain_over_spread"] == pytest.approx(0.0, abs=0.05)
    big, _, _ = _facts(tmp_path / "b", clf_csv, [0.90, 0.91, 0.89])  # gain 0.4, spread 0.01
    assert big["gain_within_spread"] is False and big["gain_over_spread"] > 30
    assert big["gain_over_spread"] == round(big["val_gain"] / big["best"]["val_std_across_folds"], 2)
    task1 = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "single")
    node = Node(id=1, stage="draft", parent=None, plan="p", code="x", metric=0.9, fold_scores=[0.9])
    single = report.facts(task1, Journal(nodes=[node], best_id=1))
    assert single["gain_within_spread"] is None and single["gain_over_spread"] is None


def test_template_states_the_gain_against_the_spread(tmp_path, clf_csv):
    within, _, _ = _facts(tmp_path, clf_csv, [0.50, 0.55, 0.45])
    text = template_narrative(within)
    assert "is not larger than the spread across folds" in text and "not evidence of a real improvement" in text
    assert review(text, within) == []
    big, _, _ = _facts(tmp_path / "b", clf_csv, [0.90, 0.91, 0.89])
    text = template_narrative(big)
    assert "exceeds the spread across folds" in text and "significan" not in text
    assert review(text, big) == []


WITHIN_BAD = ["The best score exceeded the cross-validation spread, indicating a real gain.",
              "This is a clear improvement over the baseline.",
              "The gain beats the noise.",
              "The gain is larger than the spread across folds.",
              "It shows a genuine gain; the winner is meaningful."]
WITHIN_OK = ["The gain is not larger than the spread, so it is not a real improvement.",
             "No clear gain: the improvement is within the spread.",
             "The gain does not exceed the cross-validation spread.",
             "The gain is within noise.",
             "The runner-up beats the spread of the others."]
BEYOND_BAD = ["A gain of 0.30 may be within noise given the spread of 0.0039.",
              "The improvement is inside the fold spread.",
              "The gain is smaller than the noise."]
BEYOND_OK = ["The gain is not within the spread.",
             "The gain exceeds the spread across folds.",
             "The runner-up is within noise.",
             "The gain is never within the noise here."]


@pytest.mark.parametrize("text", WITHIN_BAD)
def test_review_rejects_real_gain_claims_when_the_gain_is_within_the_spread(text):
    issues = review(text, {"experiments_run": 1, "experiments_failed": 0, "gain_within_spread": True,
                           "gain_over_spread": 0.02})
    assert len(issues) == 1 and "not larger than the spread" in issues[0], text


@pytest.mark.parametrize("text", WITHIN_OK)
def test_review_accepts_negated_or_modest_wording_when_within_the_spread(text):
    f = {"experiments_run": 1, "experiments_failed": 0, "gain_within_spread": True, "gain_over_spread": 0.02}
    assert review(text, f) == [], text


@pytest.mark.parametrize("text", BEYOND_BAD)
def test_review_rejects_noise_claims_when_the_gain_exceeds_the_spread(text):
    f = {"experiments_run": 1, "experiments_failed": 0, "gain_within_spread": False, "gain_over_spread": 7.5}
    issues = [i for i in review(text, f) if "number" not in i]
    assert len(issues) == 1 and "larger than the spread" in issues[0], text


@pytest.mark.parametrize("text", BEYOND_OK)
def test_review_accepts_other_wording_when_the_gain_exceeds_the_spread(text):
    f = {"experiments_run": 1, "experiments_failed": 0, "gain_within_spread": False, "gain_over_spread": 7.5}
    assert review(text, f) == [], text


def test_review_has_no_gain_rule_without_a_spread():
    f = {"experiments_run": 1, "experiments_failed": 0, "gain_within_spread": None, "gain_over_spread": None}
    for text in WITHIN_BAD + BEYOND_BAD:
        assert [i for i in review(text, f) if "spread" in i and "FACTS" not in i] == [], text


class _Narrator:
    def __init__(self, *texts):
        self.texts, self.prompts, self.model = list(texts), [], "n"

    def complete(self, system, user):
        from orbit.llm import Completion
        self.prompts.append((system, user))
        return Completion(self.texts.pop(0) if len(self.texts) > 1 else self.texts[0], 1, 1)


def test_narrator_gets_the_gain_facts_and_rules_and_is_revised_once(tmp_path, clf_csv):
    f, task, journal = _facts(tmp_path, clf_csv, [0.50, 0.55, 0.45])
    good = "**Verdict:** a. **Evidence:** b. **Caveats:** the gain is not larger than the spread. **Next steps:** c."
    llm = _Narrator("**Verdict:** x. The best score exceeded the spread, indicating a real gain.", good)
    text, source, found = report._write(llm, f)
    assert text == good and source.endswith("revised after review") and "not larger than the spread" in found[0]
    system, user = llm.prompts[0]
    assert "gain_within_spread" in user and '"gain_over_spread"' in user and "gain_within_spread" in system
    assert "never the score" in system and "speculate" in system
    llm = _Narrator("It is a real gain.")
    text, source, _ = report._write(llm, f)
    assert source.startswith("template") and "not evidence of a real improvement" in text


# ---- setup notes ------------------------------------------------------------------------------

def test_report_md_lists_every_setup_note_and_html_shows_each_once(tmp_path, clf_csv):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3, notes=["read a.csv with ';'"], drop=["f3"]),
                   tmp_path / "run")
    node = Node(id=1, stage="draft", parent=None, plan="p", code="x", fold_scores=[0.8, 0.82, 0.81],
                metric=0.81)
    journal = Journal(nodes=[node], best_id=1, stop_reason="done")
    md, _ = report.build(task, journal, None)
    section = md.split("## Setup notes\n")[1].split("\n## ")[0]
    for note in task.spec.notes:
        assert f"- {note}\n" in section + "\n"
    assert "read a.csv with ';'" in section and "dropped columns: f3 (as requested)" in section
    (task.run_dir / "report.md").write_text(md)
    journal.save(task.run_dir / "journal.json")
    page = html_report.render(results.collect(task.run_dir))
    assert page.count("dropped columns: f3 (as requested)") == 1
    assert "dropped columns: f3" in page            # and the meta line names them
    plain = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "plain")
    assert "## Setup notes" not in report.build(plain, journal, None)[0]


# ---- MLflow metric names and the fingerprint ---------------------------------------------------

def _history(client, run_id, key):
    return [(m.step, m.value) for m in client.get_metric_history(run_id, key)]


def test_parent_run_has_per_step_experiment_and_running_best_scores(clf_csv, tmp_path, mlflow_uri):
    import mlflow

    task = prepare(TaskSpec(str(clf_csv), "label", folds=1), tmp_path / "run")
    cfg = SearchConfig(max_steps=3, n_drafts=3)
    tracker = MLflowTracker(experiment="orbit-names")
    tracker.start(task, cfg, "scripted")
    replies = [clf_reply("C=0.001", kwargs="C=0.001, max_iter=500"), clf_reply(), BROKEN]
    journal = Agent(task, ScriptedLLM(replies), LocalSandbox(), cfg, tracker=tracker).run()
    client = mlflow.MlflowClient()
    parent = next(r for r in client.search_runs([client.get_experiment_by_name("orbit-names").experiment_id])
                  if "mlflow.parentRunId" not in r.data.tags)
    rid = parent.info.run_id
    scores = {n.id: n.metric for n in journal.nodes if not n.buggy}
    assert _history(client, rid, "experiment_val_score") == sorted(scores.items())
    best = [(1, scores[1]), (2, max(scores[1], scores[2]))]
    assert _history(client, rid, "best_val_score_so_far") == best
    assert "val_score" not in parent.data.metrics
    assert parent.data.metrics["best_val_score"] == pytest.approx(max(scores.values()))
    assert "baseline" in parent.data.metrics
    nested = [r for r in client.search_runs([parent.info.experiment_id]) if "mlflow.parentRunId" in r.data.tags]
    assert any("val_score" in r.data.metrics for r in nested)       # nested runs keep their own


def test_running_best_follows_the_metric_direction(reg_csv, tmp_path, mlflow_uri):
    import mlflow

    task = prepare(TaskSpec(str(reg_csv), "price", folds=1), tmp_path / "run")
    tracker = MLflowTracker(experiment="orbit-rmse")
    tracker.start(task, SearchConfig(), "scripted")
    for i, value in enumerate([5.0, 3.0, 4.0], 1):
        tracker.log_node(Node(id=i, stage="draft", parent=None, plan="p", metric=value), tmp_path)
    client = mlflow.MlflowClient()
    rid = tracker.parent.info.run_id
    assert [v for _, v in _history(client, rid, "best_val_score_so_far")] == [5.0, 3.0, 3.0]
    assert [v for _, v in _history(client, rid, "experiment_val_score")] == [5.0, 3.0, 4.0]


def test_fingerprint_counts_labeled_rows_and_follows_drop(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120),
                       "label": ["x", "y"] * 60})
    df.loc[:19, "label"] = np.nan                       # 20 rows without a target
    path = tmp_path / "gaps.csv"
    df.to_csv(path, index=False)
    for spec in (TaskSpec(str(path), "label", folds=3), TaskSpec(str(path), "label", folds=1, drop=["b"])):
        task = prepare(spec, tmp_path / f"run{len(spec.drop)}")
        assert re.fullmatch(r"rows=100 sha256=[0-9a-f]{12}", data_fingerprint(task))


# ---- the experiment-level time limit -----------------------------------------------------------

@pytest.mark.parametrize("limit,folds,expected", [(600, 3, 200), (600, 5, 120), (60, 3, 30),
                                                  (600, 1, 600), (90, 5, 30), (100, 2, 50)])
def test_run_timeout_arithmetic(limit, folds, expected):
    assert prompts.run_timeout(limit, folds) == expected


class _Recorder(LocalSandbox):
    def __init__(self):
        self.timeouts = []

    def run(self, code, workdir, data_dir, timeout):
        self.timeouts.append(timeout)
        return super().run(code, workdir, data_dir, timeout)


@pytest.mark.parametrize("folds,per_run", [(3, 200), (1, 600)])
def test_each_fold_gets_its_share_and_the_prompt_says_so(clf_csv, tmp_path, folds, per_run):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=folds), tmp_path / "run")
    sandbox, llm = _Recorder(), ScriptedLLM([clf_reply()])
    agent = Agent(task, llm, sandbox, SearchConfig(exec_timeout=600, max_steps=1, n_drafts=1))
    agent.run()
    assert sandbox.timeouts == [per_run] * folds
    assert f"finish within {per_run} seconds" in agent.system
    if folds > 1:
        assert "its own time limit" in prompts.data_section(task)
    else:
        assert "its own time limit" not in prompts.data_section(task)


def test_a_fold_that_times_out_fails_the_experiment_naming_the_limit(clf_csv, tmp_path):
    class Slow:
        calls = 0

        def run(self, code, workdir, data_dir, timeout):
            Slow.calls += 1
            workdir.mkdir(parents=True, exist_ok=True)
            return ExecResult(-1, "", f"TimeoutError: experiment exceeded {timeout}s and was killed.",
                              float(timeout), timed_out=True)

    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([clf_reply()]), Slow(), SearchConfig(exec_timeout=300, max_steps=1, n_drafts=1))
    node = agent.step()
    assert node.buggy and "fold 1/3" in node.error and "exceeded 100s" in node.error and Slow.calls == 1


def test_export_keeps_the_full_limit(clf_csv, tmp_path):
    task = prepare(TaskSpec(str(clf_csv), "label", folds=3), tmp_path / "run")
    node = Node(id=1, stage="draft", parent=None, plan="p", code="print(1)", metric=0.9)
    sandbox, llm = _Recorder(), ScriptedLLM([CLF])
    res = export_best(task, Journal(nodes=[node], best_id=1), llm, sandbox, 600, 1)
    assert res.ok, res.error
    assert sandbox.timeouts[0] == 600
    assert "finish within 600 seconds" in prompts.system_prompt(task, 600, 1)


def test_run_prints_the_time_limit_and_help_says_experiment(clf_csv, tmp_path, monkeypatch):
    out = runner.invoke(app, ["run", "--help"]).output
    assert "one experiment" in " ".join(out.split()) and "--drop" in out
    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (True, "ok"))
    monkeypatch.setattr("orbit.sandbox.DockerSandbox", lambda settings=None: LocalSandbox())
    monkeypatch.setattr("orbit.cli._llm", lambda: ScriptedLLM([clf_reply()]))
    r = runner.invoke(app, ["run", str(clf_csv), "-t", "label", "--steps", "1", "--drafts", "1",
                            "--folds", "3", "--timeout", "600", "--tracker", "none", "--no-export",
                            "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "time limit: 600s per experiment, 200s per fold" in r.output


# ---- narrow terminals -------------------------------------------------------------------------

LONG = "bank-marketing-term-deposit-subscription-2026"


def test_serve_list_keeps_long_names_whole_at_80_columns(monkeypatch):
    from orbit import serve
    from orbit.serve import Container

    monkeypatch.setattr(serve, "list_containers", lambda: [
        Container(f"orbit-serve-{LONG}-8090", LONG, "12", 8090, "7m 20s"),
        Container("orbit-serve-wine-8091", "wine", "1", 8091, "up 1 h")])
    r = runner.invoke(app, ["serve", "--list"], env={"COLUMNS": "80"})
    assert r.exit_code == 0
    assert f"orbit-serve-{LONG}-8090" in r.output.splitlines() and LONG in r.output
    assert all(len(line) <= 80 for line in r.output.splitlines() if "http" not in line)
    assert "wine" in r.output and "8091" in r.output and "up up" not in r.output and "up 1 h" in r.output


def _models(monkeypatch, names):
    import mlflow

    class Model:
        def __init__(self, name):
            self.name, self.aliases = name, {"champion": "2"}

    class Version:
        def __init__(self, v):
            self.version = v
            self.tags = {"orbit.metric": "roc_auc", "orbit.val_score": "0.9", "orbit.test_score": "0.91"}

    class Client:
        def search_registered_models(self, max_results):
            return [Model(n) for n in names]

        def search_model_versions(self, f):
            return [Version("1"), Version("2")]

    monkeypatch.setattr(mlflow, "MlflowClient", lambda: Client())


def test_models_keeps_long_names_whole_at_80_columns(monkeypatch):
    _models(monkeypatch, [LONG, "wine"])
    r = runner.invoke(app, ["models"], env={"COLUMNS": "80"})
    assert r.exit_code == 0 and any(LONG in line for line in r.output.splitlines())
    assert "validation 0.9000" in r.output and "champion 2" in r.output
    _models(monkeypatch, ["wine"])                       # short names keep the table
    r = runner.invoke(app, ["models"], env={"COLUMNS": "80"})
    assert "┃ name" in r.output and "│ wine" in r.output


def test_experiments_keeps_long_names_whole_at_80_columns(monkeypatch):
    import mlflow

    class Exp:
        name, experiment_id, last_update_time = LONG, "7", 0

    class Page(list):
        token = None

    class Client:
        def search_experiments(self):
            return [Exp()]

        def search_runs(self, ids, max_results=1000, page_token=None):
            return Page()

    monkeypatch.setattr(mlflow, "MlflowClient", lambda: Client())
    r = runner.invoke(app, ["experiments"], env={"COLUMNS": "80"})
    assert r.exit_code == 0 and any(LONG in line for line in r.output.splitlines())


# ---- HTML title ---------------------------------------------------------------------------------

def test_html_title_is_the_first_sentence_cut_at_100(clf_csv, tmp_path):
    long_goal = ("Predict whether the client will subscribe to a term deposit using demographic and "
                 "campaign data, without anything that is only known after the call. Second sentence <b>here</b>.")
    task = prepare(TaskSpec(str(clf_csv), "label", folds=1, goal=long_goal), tmp_path / "run")
    Agent(task, ScriptedLLM([clf_reply()]), LocalSandbox(), SearchConfig(max_steps=1, n_drafts=1)).run()
    page = html_report.render(results.collect(task.run_dir))
    h1 = re.search(r"<h1>(.*?)</h1>", page).group(1)
    assert len(h1) <= 101 and h1.endswith("…") and h1.startswith("Predict whether the client")
    assert "Second sentence" not in h1
    assert "<title>orbit report: " + h1 + "</title>" in page
    assert "Problem: " + long_goal.replace("<b>here</b>", "&lt;b&gt;here&lt;/b&gt;") in page


def test_short_title_cases():
    assert html_report.short_title("Predict churn. Then more.") == "Predict churn."
    assert html_report.short_title("Predict churn") == "Predict churn"
    assert html_report.short_title("a" * 100) == "a" * 100
    assert html_report.short_title("a" * 101) == "a" * 100 + "…"
    assert html_report.short_title("  ") == "" and html_report.short_title(None) == ""
