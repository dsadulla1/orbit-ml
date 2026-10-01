from orbit.agent import Agent, Journal, SearchConfig
from orbit.task import TaskSpec, prepare

from conftest import BAD_OUTPUT, BROKEN, LocalSandbox, ScriptedLLM, clf_reply


def _agent(tmp_path, csv, replies, **cfg):
    task = prepare(TaskSpec(str(csv), "label"), tmp_path / "run")
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
