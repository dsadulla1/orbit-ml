import json

import pandas as pd
import pytest

from orbit.planner import PlanError, plan

from conftest import ScriptedLLM


def _j(**kw):
    base = {"primary_table": "train.csv", "target": "SalePrice", "metric": None,
            "id_column": "Id", "reasoning": "SalePrice is the price to predict."}
    return "```json\n" + json.dumps({**base, **kw}) + "\n```"


@pytest.fixture
def house(tmp_path):
    d = tmp_path / "house"
    d.mkdir()
    train = pd.DataFrame({"Id": range(1, 41), "Area": range(40), "Rooms": [2, 3] * 20,
                          "SalePrice": [100_000 + 1000 * i for i in range(40)]})
    train.to_csv(d / "train.csv", index=False)
    train.drop(columns="SalePrice").to_csv(d / "test.csv", index=False)
    pd.DataFrame({"Rooms": [2, 3], "tax": [1.0, 2.0]}).to_csv(d / "tax.csv", index=False)
    (d / "data_description.txt").write_text("SalePrice: the property's sale price in dollars.")
    return {"tables": sorted(d.glob("*.csv")), "docs": [d / "data_description.txt"]}


def test_explicit_target_needs_no_llm(house):
    p = plan(house["tables"], [], target="SalePrice", metric="RMSE")
    assert (p.primary.name, p.target, p.metric) == ("train.csv", "SalePrice", "rmse")
    assert [e.name for e in p.extras] == ["tax.csv"]
    assert any("ignored test.csv" in n for n in p.notes)


def test_llm_infers_task_from_problem_and_docs(house):
    llm = ScriptedLLM([_j()])
    p = plan(house["tables"], house["docs"], goal="Predict the sales price for each house",
             eval_text="RMSE between log-prices", llm=llm)
    assert (p.target, p.metric, p.id_column) == ("SalePrice", "rmsle", "Id")
    assert [e.name for e in p.extras] == ["tax.csv"] and p.notes[0].startswith("setup:")
    sent = llm.prompts[0]
    assert "Predict the sales price" in sent and "sale price in dollars" in sent and "tax.csv" in sent


def test_llm_metric_is_used_when_wording_is_vague(house):
    p = plan(house["tables"], [], goal="price", eval_text="how close the guesses are",
             llm=ScriptedLLM([_j(metric="mae")]))
    assert p.metric == "mae"


def test_bad_replies_are_retried_then_reported(house):
    llm = ScriptedLLM(["no json here", _j()])
    assert plan(house["tables"], [], goal="price", llm=llm).target == "SalePrice"
    assert "rejected" in llm.prompts[1]
    with pytest.raises(PlanError, match="pass --target"):
        plan(house["tables"], [], goal="price",
             llm=ScriptedLLM([_j(target="Nope"), _j(primary_table="x.csv")]))
    with pytest.raises(PlanError, match="pass --target"):
        plan(house["tables"], [], goal="price", llm=ScriptedLLM([_j(metric="made_up")] * 2))


def test_unmappable_evaluation_is_rejected(house):
    with pytest.raises(PlanError, match="cannot map the evaluation"):
        plan(house["tables"], [], goal="price", eval_text="customer delight",
             llm=ScriptedLLM([_j(metric=None)]))
    with pytest.raises(PlanError, match="cannot map the evaluation"):
        plan(house["tables"], [], target="SalePrice", eval_text="customer delight")


def test_missing_pieces_without_an_llm(house):
    with pytest.raises(PlanError, match="no --target"):
        plan(house["tables"], [], goal="price")
    with pytest.raises(PlanError, match="not a column of any table"):
        plan(house["tables"], [], target="Nope")
    with pytest.raises(PlanError, match="unknown metric"):
        plan(house["tables"], [], target="SalePrice", metric="vibes")


def test_target_in_several_tables_prefers_train(tmp_path):
    for name, n in (("train.csv", 30), ("sample.csv", 5)):
        pd.DataFrame({"x": range(n), "y": range(n)}).to_csv(tmp_path / name, index=False)
    p = plan([tmp_path / "sample.csv", tmp_path / "train.csv"], [], target="y")
    assert p.primary.name == "train.csv" and any("using train.csv" in n for n in p.notes)
