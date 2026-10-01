import pandas as pd

from orbit.agent import Agent, Journal, SearchConfig
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
