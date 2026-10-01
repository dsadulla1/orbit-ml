import subprocess
from pathlib import Path

import pytest

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


def test_check_numbers_ignores_list_markers():
    f = {"best": {"val_score": 0.9123}, "experiments_run": 12}
    text = "**Next steps:**\n1. Tune the model.\n2) Add features.\n  3. Score 0.95."
    assert check_numbers(text, f) == ["0.95"]


def _can_symlink(tmp_path):
    try:
        (tmp_path / "probe").symlink_to(tmp_path)
        return True
    except OSError:
        return False


def test_docker_sandbox_removes_symlinks_left_by_the_script(tmp_path, monkeypatch):
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks need extra privileges on this system")
    secret = tmp_path / "secret.txt"
    secret.write_text("host file")
    workdir = tmp_path / "exp"

    def fake_run(cmd, **kwargs):
        # A script can plant links to host paths (found via /proc/self/mountinfo).
        for name in ("stdout.txt", "predictions.csv", "prompt.md"):
            (workdir / name).symlink_to(secret)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("orbit.sandbox.subprocess.run", fake_run)
    DockerSandbox().run("print(1)", workdir, tmp_path, 10)
    assert not any(p.is_symlink() for p in workdir.iterdir())
    assert secret.read_text() == "host file"


def test_orphan_think_close_tag_is_stripped():
    from types import SimpleNamespace

    from orbit.config import LLMSettings
    from orbit.llm import OpenAICompatLLM

    llm = OpenAICompatLLM(LLMSettings(base_url="http://localhost:1/v1", api_key="k", model="m"))
    msg = SimpleNamespace(content="reasoning ```python\nlong draft\n```</think>\nPLAN: x\n```python\nok\n```")
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: resp)))
    assert llm.complete("s", "u").text == "PLAN: x\n```python\nok\n```"
