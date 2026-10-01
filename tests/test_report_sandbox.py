from pathlib import Path

from orbit.agent import Agent, SearchConfig
from orbit.config import SandboxSettings
from orbit.llm import Completion
from orbit.report import build, check_numbers, facts
from orbit.sandbox import DockerSandbox
from orbit.task import TaskSpec, prepare

from conftest import LocalSandbox, ScriptedLLM, clf_reply


def _finished_run(tmp_path, csv):
    task = prepare(TaskSpec(str(csv), "label"), tmp_path / "run")
    agent = Agent(task, ScriptedLLM([clf_reply(), clf_reply("RF", "ensemble",
                  "RandomForestClassifier", "random_state=0")]), LocalSandbox(),
                  SearchConfig(max_steps=2, n_drafts=2))
    return task, agent.run()


def test_check_numbers():
    f = {"best": {"val_score": 0.9123, "gain_over_baseline": 0.4123}, "experiments_run": 12}
    assert check_numbers("Score 0.9123 (+0.41) over 12 runs, i.e. 91.2% AUC.", f) == []
    assert check_numbers("Score 0.95 after 12 runs.", f) == ["0.95"]


def test_template_report(clf_csv, tmp_path):
    task, journal = _finished_run(tmp_path, clf_csv)
    md, f = build(task, journal, llm=None)
    assert "**Verdict:**" in md and "All numbers verified" in md
    assert f["best"]["test_score"] is not None
    assert check_numbers(md.split("## Summary")[1].split("_Narrative")[0], facts(task, journal)) == []


def test_llm_narrative_flags_invented_numbers(clf_csv, tmp_path):
    task, journal = _finished_run(tmp_path, clf_csv)

    class Liar:
        model = "liar"
        def complete(self, system, user):
            return Completion("**Verdict:** experiment #1 reached 0.9999 AUC.")

    md, f = build(task, journal, llm=Liar())
    assert "WARNING: unverified numbers" in md and "0.9999" in f["narrative_unverified"]


def test_docker_command_is_locked_down():
    sb = DockerSandbox(SandboxSettings(image="img", cpus=2, memory="4g", timeout=60, runtime="runsc"))
    cmd = sb.command(Path("work"), Path("data"), "orbit-test")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                 "--memory 4g", "--cpus 2", "--pids-limit", "--runtime runsc", "--rm"):
        assert flag in joined
    assert any(a.endswith(":/work/data:ro") for a in cmd)
    assert cmd[-3:] == ["img", "python", "solution.py"]
