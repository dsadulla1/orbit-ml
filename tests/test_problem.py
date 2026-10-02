import json

import pandas as pd
import pytest

from orbit.cli import app
from orbit.planner import PlanError
from orbit.problem import run_name, setup_task
from orbit.sources import SourceError
from orbit.task import Task

from conftest import ScriptedLLM


def _house(tmp_path):
    d = tmp_path / "house_prices"
    d.mkdir()
    n = 80
    train = pd.DataFrame({"Id": range(n), "Area": range(n), "Rooms": [2, 3] * (n // 2),
                          "SalePrice": [100_000 + 900 * i for i in range(n)]})
    train.to_csv(d / "train.csv", index=False)
    train.drop(columns="SalePrice").to_csv(d / "test.csv", index=False)
    pd.DataFrame({"Rooms": [2, 3], "tax": [1.0, 2.0]}).to_csv(d / "tax.csv", index=False)
    (d / "data_description.txt").write_text("Area is the floor area in square feet.")
    return d


def test_run_name():
    assert run_name(["examples/data/wine.csv"]) == "wine"
    assert run_name(["https://x.org/data/House Prices.csv?raw=1"]) == "House_Prices"
    assert run_name(["some/folder/"]) == "folder"
    assert run_name(["."]) == "task"


def test_natural_language_problem_over_a_folder(tmp_path):
    reply = json.dumps({"primary_table": "train.csv", "target": "SalePrice", "metric": None,
                        "id_column": "Id", "reasoning": "price is the label"})
    llm = ScriptedLLM([f"```json\n{reply}\n```"])
    task = setup_task([str(_house(tmp_path))], tmp_path / "run", goal="Predict the sales price for each house",
                      eval_text="RMSE between log-prices", llm=llm)
    assert (task.metric_name, task.spec.target, task.spec.id_column) == ("rmsle", "SalePrice", "Id")
    assert task.n_folds == 5 and [p.split("/")[-1] for p in task.spec.extras] == ["tax.csv"]
    assert any("ignored test.csv" in n for n in task.spec.notes) and "square feet" in task.docs_text
    assert (task.fold_dir(1) / "extra" / "tax.csv").exists() and len(llm.prompts) == 1
    loaded = Task.load(task.run_dir)
    assert loaded.spec.sources and loaded.spec.eval_text == "RMSE between log-prices"


def test_fully_specified_problem_makes_no_llm_call(tmp_path):
    llm = ScriptedLLM([])
    task = setup_task([str(_house(tmp_path) / "train.csv")], tmp_path / "run", target="SalePrice",
                      eval_text="mae", llm=llm)
    assert task.metric_name == "mae" and llm.prompts == []


def test_user_errors_are_exceptions_with_messages(tmp_path):
    with pytest.raises(SourceError, match="does not exist"):
        setup_task([str(tmp_path / "nope.csv")], tmp_path / "a")
    with pytest.raises(PlanError, match="no --target"):
        setup_task([str(_house(tmp_path) / "train.csv")], tmp_path / "b")
    with pytest.raises(ValueError, match="needs a target >= 0"):
        d = tmp_path / "neg.csv"
        pd.DataFrame({"x": range(40), "y": [-1.0] * 40}).to_csv(d, index=False)
        setup_task([str(d)], tmp_path / "c", target="y", eval_text="rmsle")


def test_run_command_reports_setup_errors(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (True, "ok"))
    monkeypatch.setattr("orbit.cli._llm", lambda: ScriptedLLM([]))
    r = CliRunner().invoke(app, ["run", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "runs")])
    assert r.exit_code == 1 and "Could not set up the task" in r.output and "does not exist" in r.output


def test_sanitize_name():
    from orbit.problem import sanitize_name

    assert sanitize_name("My run: v2!") == "My_run_v2"
    assert sanitize_name("x" * 60) == "x" * 40 and sanitize_name("***") == ""


def test_failed_setup_leaves_no_run_folder(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "r1"
    with pytest.raises(PlanError, match="no --target"):
        setup_task([str(_house(tmp_path) / "train.csv")], run_dir, within=runs)
    assert not run_dir.exists()
    keep = runs / "r2"
    (keep / "source").mkdir(parents=True)
    (keep / "mine.txt").write_text("keep")
    with pytest.raises(SourceError):
        setup_task([str(tmp_path / "nope.csv")], keep, within=runs)
    assert (keep / "mine.txt").read_text() == "keep"
    # a folder outside `within` is never removed
    other = tmp_path / "other" / "r3"
    with pytest.raises(SourceError):
        setup_task([str(tmp_path / "nope.csv")], other, within=runs)
    assert other.exists()


def test_run_command_names_folder_and_cleans_up(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setattr("orbit.sandbox.check_docker", lambda s=None: (True, "ok"))
    monkeypatch.setattr("orbit.cli._llm", lambda: ScriptedLLM([]))
    seen = []

    def fake_setup(sources, run_dir, **kw):
        seen.append(run_dir.name)
        raise SourceError("boom")

    monkeypatch.setattr("orbit.problem.setup_task", fake_setup)
    r = CliRunner().invoke(app, ["run", "data.csv", "--out", str(tmp_path), "--name", "my run!"])
    assert r.exit_code == 1 and seen[0].endswith("-my_run")
