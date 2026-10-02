import math

import pandas as pd
from sklearn.datasets import make_classification

from orbit.agent import Agent, Journal, Node, SearchConfig
from orbit.task import TaskSpec, prepare

from conftest import BAD_OUTPUT, BROKEN, LocalSandbox, ScriptedLLM, clf_reply


def _agent(tmp_path, csv, replies, folds=1, **cfg):
    task = prepare(TaskSpec(str(csv), "label", folds=folds), tmp_path / "run")
    llm = ScriptedLLM(replies)
    config = SearchConfig(exec_timeout=120, **cfg)
    return Agent(task, llm, LocalSandbox(), config), llm


def test_full_search(clf_csv, tmp_path):
    replies = [
        clf_reply(),                                                     # 1 draft ok
        BROKEN,                                                          # 2 draft fails
        clf_reply("Random forest.", "ensemble", "RandomForestClassifier",
                  "n_estimators=50, random_state=0"),                    # 3 draft ok
        clf_reply("Fixed: gradient boosting.", "ensemble",
                  "GradientBoostingClassifier", "random_state=0"),       # 4 debug of #2
        clf_reply("Stronger regularisation.", kwargs="C=0.1, max_iter=500"),  # 5 improve
    ]
    agent, llm = _agent(tmp_path, clf_csv, replies, max_steps=5, n_drafts=3, debug_prob=1.0)
    journal = agent.run()

    stages = [n.stage for n in journal.nodes]
    assert stages == ["draft", "draft", "draft", "debug", "improve"]
    assert journal.nodes[1].buggy and "could not convert" in journal.nodes[1].error
    assert journal.nodes[3].parent == 2 and not journal.nodes[3].buggy
    # The debug prompt carries the failing code and its error back to the model.
    assert "could not convert" in llm.prompts[3] and "Broken experiment (#2)" in llm.prompts[3]

    best = journal.get(journal.best_id)
    assert best.metric > 0.8 and best.test_metric is not None
    assert all(n.test_metric is None for n in journal.nodes if n.id != best.id)
    assert journal.stop_reason.startswith("step budget")
    assert journal.tokens == 5 * 150

    reloaded = Journal.load(agent.task.run_dir / "journal.json")
    assert reloaded.best_id == best.id and len(reloaded.nodes) == 5
    assert (agent.task.run_dir / "nodes" / "001" / "solution.py").exists()


def test_rejected_output_is_debuggable(clf_csv, tmp_path):
    agent, llm = _agent(tmp_path, clf_csv, [BAD_OUTPUT, clf_reply()], max_steps=2, n_drafts=1)
    journal = agent.run()
    assert "orbit_id" in journal.nodes[0].error
    assert journal.nodes[1].stage == "debug" and not journal.nodes[1].buggy


def test_missing_code_block(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, ["I would use XGBoost."], max_steps=1, n_drafts=1)
    journal = agent.run()
    assert journal.nodes[0].error.startswith("The reply contained no")
    assert journal.best_id is None


def test_token_budget_stops(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, [], max_steps=10, n_drafts=1, token_budget=300)
    journal = agent.run()
    assert len(journal.nodes) == 2 and "token budget" in journal.stop_reason


def test_llm_errors_abort_gracefully(clf_csv, tmp_path, monkeypatch):
    agent, llm = _agent(tmp_path, clf_csv, [], max_steps=5)

    def boom(system, user):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(llm, "complete", boom)
    monkeypatch.setattr("orbit.agent.time.sleep", lambda s: None)
    journal = agent.run()
    assert "ConnectionError" in journal.stop_reason and journal.nodes == []


def test_invalid_test_rows_rejected_during_search(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, [], max_steps=1, n_drafts=1)
    labels = pd.read_csv(agent.task.run_dir / "hidden" / "labels.csv")
    test_id = int(labels.loc[labels["split"] == "test", "orbit_id"].iloc[0])
    # Valid on every validation row, out of range on one test row.
    reply = clf_reply().replace(
        'pd.DataFrame({"orbit_id"',
        f'proba[test["orbit_id"].to_numpy() == {test_id}] = 7.0\npd.DataFrame({{"orbit_id"'
    )
    agent.llm.replies = [reply]
    journal = agent.run()  # must not crash in finalize
    assert journal.nodes[0].buggy and "probabilities" in journal.nodes[0].error
    assert journal.best_id is None


def test_finalize_survives_unreadable_best_predictions(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, [clf_reply()], max_steps=1, n_drafts=1)
    node = agent.step()
    (agent.task.run_dir / "nodes" / "001" / "predictions.csv").unlink()
    agent.finalize("done")
    assert agent.journal.best_id == node.id and node.test_metric is None
    assert Journal.load(agent.task.run_dir / "journal.json").stop_reason == "done"


def test_cut_off_reply_is_reported(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, ["PLAN: x\n```python\nimport pandas as pd\nX = pd.read"],
                      max_steps=1, n_drafts=1)
    assert "cut off" in agent.run().nodes[0].error


def test_cross_validated_search(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, [clf_reply(), BROKEN], folds=3, max_steps=2, n_drafts=2)
    journal = agent.run()
    ok, bad = journal.nodes
    assert len(ok.fold_scores) == 3 and ok.metric == sum(ok.fold_scores) / 3
    assert ok.test_metric is not None and 0.5 < ok.test_metric <= 1.0
    assert bad.buggy and bad.error.startswith("fold 1/3: ") and bad.fold_scores == []
    workdir = agent.task.run_dir / "nodes" / "001"
    assert all((workdir / f"fold_{k}" / "predictions.csv").exists() for k in (1, 2, 3))
    assert (workdir / "solution.py").exists() and "=== fold 3/3 ===" in (workdir / "stdout.txt").read_text()
    assert Journal.load(agent.task.run_dir / "journal.json").nodes[0].fold_scores == ok.fold_scores


def test_cross_validation_rejects_bad_rows_in_any_fold(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, [BAD_OUTPUT], folds=3, max_steps=1, n_drafts=1)
    node = agent.run().nodes[0]
    assert node.buggy and "fold 1/3" in node.error and "orbit_id" in node.error


def test_non_finite_scores_count_as_failures(clf_csv, tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, clf_csv, [clf_reply()], folds=2, max_steps=1, n_drafts=1)
    monkeypatch.setattr("orbit.agent.score", lambda *a, **k: float("nan"))
    node = agent.run().nodes[0]
    assert node.buggy and "not finite" in node.error and agent.journal.best_id is None
    from orbit.agent import Node
    assert Node(1, "draft", None, metric=float("nan")).buggy


def test_time_ordered_cross_validated_search(tmp_path):
    n = 400
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, weights=[0.7, 0.3],
                               random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["day"] = range(n)
    df["label"] = pd.Series(y).map({0: "no", 1: "yes"})
    df = df.sample(frac=1.0, random_state=3)
    csv = tmp_path / "ts.csv"
    df.to_csv(csv, index=False)
    task = prepare(TaskSpec(str(csv), "label", folds=3, time_col="day"), tmp_path / "run")
    llm = ScriptedLLM([clf_reply()])
    agent = Agent(task, llm, LocalSandbox(), SearchConfig(exec_timeout=120, max_steps=1, n_drafts=1))
    node = agent.run().nodes[0]
    assert len(node.fold_scores) == 3 and not node.buggy
    assert node.test_metric is not None and math.isfinite(node.test_metric)
    assert "later in time (column 'day')" in llm.prompts[0]


# ---- patience -----------------------------------------------------------------------------

def _stale(tmp_path, clf_csv, nodes, patience=3, metric="roc_auc", folds=1):
    """The agent's stale count for hand-built nodes: (metric, fold_scores), None = failed."""
    task = prepare(TaskSpec(str(clf_csv), "label", folds=folds, metric=metric), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([]), LocalSandbox(), SearchConfig(patience=patience))
    for i, item in enumerate(nodes, 1):
        value, fold_scores = item if item else (None, [])
        agent.journal.nodes.append(Node(i, "draft", None, metric=value, fold_scores=fold_scores))
    return agent._stale(), agent._budget_exhausted()


def test_patience_gain_within_spread_does_not_reset(tmp_path, clf_csv):
    spread = [0.80, 0.84, 0.76]   # sample std 0.04
    nodes = [(0.80, spread), (0.82, spread), (0.83, spread)]
    stale, reason = _stale(tmp_path, clf_csv, nodes, folds=3)
    assert stale == 2 and reason is None
    stale, reason = _stale(tmp_path, clf_csv, nodes + [(0.81, spread)], patience=3, folds=3)
    assert stale == 3
    assert reason == "no improvement beyond the fold spread in the last 3 experiments"


def test_patience_small_gains_add_up_to_a_clear_improvement(tmp_path, clf_csv):
    spread = [0.76, 0.80, 0.84]  # std 0.04
    # Each step gains only 0.03 (< 0.04), but compared with the last clear improvement they add up.
    nodes = [(0.80, spread), (0.83, spread), (0.86, spread), (0.89, spread)]
    assert _stale(tmp_path, clf_csv, nodes[:2], folds=3)[0] == 1   # 0.03 above the anchor: not clear
    assert _stale(tmp_path, clf_csv, nodes[:3], folds=3)[0] == 0   # 0.06 above it: counted, new anchor
    assert _stale(tmp_path, clf_csv, nodes, folds=3)[0] == 1       # 0.03 above the new anchor


def test_patience_larger_gain_resets(tmp_path, clf_csv):
    spread = [0.80, 0.84, 0.76]
    nodes = [(0.80, spread), (0.80, spread), (0.80, spread), (0.90, spread)]
    assert _stale(tmp_path, clf_csv, nodes, folds=3)[0] == 0


def test_patience_ignores_failed_experiments(tmp_path, clf_csv):
    nodes = [(0.7, [0.7]), None, (0.7, [0.7]), None, None, None]
    stale, reason = _stale(tmp_path, clf_csv, nodes, patience=2)
    assert stale == 1 and reason is None


def test_patience_single_split_has_no_spread(tmp_path, clf_csv):
    # With one score the spread is 0, so any gain counts.
    assert _stale(tmp_path, clf_csv, [(0.700, [0.700]), (0.701, [0.701])])[0] == 0
    assert _stale(tmp_path, clf_csv, [(0.700, [0.700]), (0.700, [0.700])])[0] == 1


def test_patience_lower_is_better(tmp_path, clf_csv):
    spread = [0.40, 0.44, 0.36]
    stale, _ = _stale(tmp_path, clf_csv, [(0.40, spread), (0.38, spread)], metric="log_loss", folds=3)
    assert stale == 1                       # 0.02 better but within the 0.04 spread
    stale, _ = _stale(tmp_path, clf_csv, [(0.40, spread), (0.30, spread)], metric="log_loss", folds=3)
    assert stale == 0
    stale, _ = _stale(tmp_path, clf_csv, [(0.40, spread), (0.50, spread)], metric="log_loss", folds=3)
    assert stale == 1                       # worse is never an improvement


def test_patience_none_never_stops(tmp_path, clf_csv):
    nodes = [(0.7, [0.7])] * 6
    stale, reason = _stale(tmp_path, clf_csv, nodes, patience=None)
    assert stale == 5 and reason is None


def test_search_stops_when_scores_stall(clf_csv, tmp_path):
    agent, _ = _agent(tmp_path, clf_csv, [clf_reply()] * 8, max_steps=8, n_drafts=8, patience=2)
    journal = agent.run()
    assert len(journal.nodes) == 3        # one improvement, then two identical scores
    assert journal.stop_reason == "no improvement beyond the fold spread in the last 2 experiments"
    assert journal.nodes[journal.best_id - 1].test_metric is not None


def test_debug_prompt_lists_earlier_failures_of_the_chain(clf_csv, tmp_path):
    from orbit.agent import Node

    agent, _ = _agent(tmp_path, clf_csv, [])
    first = Node(1, "draft", None, code="a", error="Traceback\nKeyError: 'country'")
    second = Node(2, "debug", 1, code="b", error="Traceback\nValueError: could not convert string to float",
                  debug_depth=1)
    agent.journal.nodes = [first, second]
    prompt = agent.build_prompt("debug", second)
    assert "Earlier attempts in this chain failed with" in prompt and "#1: KeyError: 'country'" in prompt
    assert "could not convert string to float" in prompt
    assert "Earlier attempts" not in agent.build_prompt("debug", first)  # nothing earlier to list
