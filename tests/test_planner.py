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


def test_extra_table_with_the_target_is_never_exposed(tmp_path):
    base = pd.DataFrame({"id": range(40), "a": range(40), "y": [0, 1] * 20})
    base.to_csv(tmp_path / "train.csv", index=False)
    base.assign(z=1).to_csv(tmp_path / "full.csv", index=False)           # same labels, more columns
    base[["id", "y"]].to_csv(tmp_path / "solution.csv", index=False)      # a labels-only file
    pd.DataFrame({"id": range(40), "w": 1}).to_csv(tmp_path / "side.csv", index=False)
    p = plan(sorted(tmp_path.glob("*.csv")), [], target="y")
    assert [e.name for e in p.extras] == ["side.csv"]
    assert sum("contains the target column" in n for n in p.notes) == 2


@pytest.mark.parametrize("bad", [{"primary_table": ["train.csv"]}, {"target": ["SalePrice"]},
                                 {"metric": ["rmse"]}, {"primary_table": 7}])
def test_non_string_fields_are_retried_not_crashed(house, bad):
    llm = ScriptedLLM([_j(**bad), _j()])
    assert plan(house["tables"], [], goal="price", llm=llm).target == "SalePrice"
    assert "rejected" in llm.prompts[1]


def test_id_column_equal_to_target_is_dropped(house):
    assert plan(house["tables"], [], goal="price", llm=ScriptedLLM([_j(id_column="SalePrice")])).id_column is None


def _versions(tmp_path):
    """Three tables that all hold the target `y`, like the several versions of one dataset."""
    for name, n in (("bank-full.csv", 60), ("bank-sample.csv", 20), ("bank-extra-full.csv", 45)):
        pd.DataFrame({"a": range(n), "y": [i % 2 for i in range(n)]}).to_csv(tmp_path / name, index=False)
    return sorted(tmp_path.glob("*.csv"))


def test_match_table_rules():
    from orbit.planner import match_table

    names = ["bank-full.csv", "bank-sample.csv", "Bank-Extra-full.csv"]
    assert match_table("bank-full.csv", names) == "bank-full.csv"
    assert match_table("BANK-SAMPLE.CSV", names) == "bank-sample.csv"
    assert match_table("sample", names) == "bank-sample.csv"
    assert match_table("extra", names) == "Bank-Extra-full.csv"
    with pytest.raises(PlanError, match="matches none of the tables: bank-full.csv"):
        match_table("nope", names)
    with pytest.raises(PlanError, match="matches several tables .*give the full file name"):
        match_table("full", names)


def test_table_choice_overrides_the_tiebreak_and_says_so(tmp_path):
    tables = _versions(tmp_path)
    default = plan(tables, [], target="y")
    assert default.primary.name == "bank-full.csv"          # most rows
    chosen = plan(tables, [], target="y", table="sample")
    assert chosen.primary.name == "bank-sample.csv"
    note = next(n for n in chosen.notes if "tables have column" in n)
    assert "using bank-sample.csv" in note and "chosen with --table" in note and "bank-full.csv (60 rows)" in note


def test_table_choice_wins_over_the_llm(tmp_path):
    tables = _versions(tmp_path)
    llm = ScriptedLLM([_j(primary_table="bank-full.csv", target="y", id_column=None)])
    p = plan(tables, [], goal="subscribe", llm=llm, table="bank-extra")
    assert p.primary.name == "bank-extra-full.csv" and p.target == "y"
    assert "primary table: bank-extra-full.csv" in llm.prompts[0]


def test_table_choice_errors_are_readable(tmp_path):
    tables = _versions(tmp_path)
    with pytest.raises(PlanError, match="matches none of the tables"):
        plan(tables, [], target="y", table="zzz")
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "other.csv", index=False)
    with pytest.raises(PlanError, match="has no column 'y'; its columns: a"):
        plan(sorted(tmp_path.glob("*.csv")), [], target="y", table="other")
