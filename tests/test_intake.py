"""Reading real-world files: delimiters, encodings, decimal marks, honest profiles, nested archives."""
import gzip
import zipfile

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from orbit import planner
from orbit.cli import app
from orbit.planner import describe_format, describe_table, load_for_profile, plan, read_table
from orbit.problem import setup_task

from conftest import ScriptedLLM

runner = CliRunner()


def _w(tmp_path, name, text, enc="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(enc) if isinstance(text, str) else text)
    return p


# ---- read_table -------------------------------------------------------------------------------

def test_comma_file_is_plain(tmp_path):
    p = _w(tmp_path, "a.csv", "x,y\n1,2\n3,4\n")
    assert describe_format(p) == {"delimiter": ",", "encoding": "utf-8-sig", "decimal": "."}
    assert read_table(p).shape == (2, 2)
    assert planner.format_note(p) is None


def test_semicolon_with_quoted_fields_containing_the_delimiter(tmp_path):
    p = _w(tmp_path, "a.csv", '"id";"name";"v"\n1;"a;b";3\n2;"c;d";4\n3;"e;f";5\n')
    assert describe_format(p)["delimiter"] == ";"
    df = read_table(p)
    assert list(df.columns) == ["id", "name", "v"] and df["name"].tolist() == ["a;b", "c;d", "e;f"]
    assert planner.format_note(p) == "read a.csv with delimiter ';'"


def test_quoted_commas_do_not_fool_a_semicolon_file(tmp_path):
    p = _w(tmp_path, "a.csv", 'id;text\n1;"a, b, c"\n2;"d, e, f"\n3;"g, h, i"\n')
    assert describe_format(p)["delimiter"] == ";" and read_table(p).shape == (3, 2)


@pytest.mark.parametrize("delim", ["\t", "|"])
def test_tab_and_pipe_in_a_csv(tmp_path, delim):
    p = _w(tmp_path, "a.csv", "\n".join(delim.join(r) for r in [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]]))
    assert describe_format(p)["delimiter"] == delim and read_table(p).shape == (2, 3)


def test_bom_is_stripped(tmp_path):
    p = _w(tmp_path, "a.csv", b"\xef\xbb\xbfid,v\n1,2\n")
    assert list(read_table(p).columns) == ["id", "v"]
    assert planner.format_note(p) is None


def test_latin1_bytes(tmp_path):
    p = _w(tmp_path, "a.csv", "name;v\ncafé;1\nnaïve;2\n", enc="latin-1")
    assert describe_format(p)["encoding"] == "cp1252"
    assert read_table(p)["name"].tolist() == ["café", "naïve"]
    assert "cp1252 encoding" in planner.format_note(p)


def test_a_late_non_utf8_byte_is_still_decoded(tmp_path):
    text = "name,v\n" + "plain,1\n" * 50_000 + "café,2\n"
    p = _w(tmp_path, "a.csv", text, enc="latin-1")
    assert read_table(p)["name"].iloc[-1] == "café"


def test_decimal_comma_with_semicolons(tmp_path):
    p = _w(tmp_path, "a.csv", "id;price;label\n1;1,5;a\n2;2,25;b\n3;10,0;a\n4;7,75;b\n")
    assert describe_format(p) == {"delimiter": ";", "encoding": "utf-8-sig", "decimal": ","}
    df = read_table(p)
    assert pd.api.types.is_float_dtype(df["price"]) and df["price"].tolist() == [1.5, 2.25, 10.0, 7.75]
    assert "decimal mark ','" in planner.format_note(p)


def test_dot_decimals_and_text_with_commas_are_left_alone(tmp_path):
    p = _w(tmp_path, "a.csv", 'id;price;note\n1;1.5;"x, y"\n2;2,5;"1,5"\n3;3.5;"z"\n')
    assert describe_format(p)["decimal"] == "."


def test_single_column_file_stays_one_column(tmp_path):
    p = _w(tmp_path, "a.csv", "word\nalpha\nbeta\ngamma\n")
    assert describe_format(p)["delimiter"] == "," and read_table(p).shape == (3, 1)
    q = _w(tmp_path, "b.csv", 'text\n"a, b"\n"c, d"\n"e, f"\n')
    assert read_table(q).shape == (3, 1)


def test_tsv_stays_tab_and_gz_works(tmp_path):
    p = _w(tmp_path, "a.tsv", "a\tb\n1,5\t2\n3,5\t4\n")
    assert describe_format(p)["delimiter"] == "\t" and list(read_table(p).columns) == ["a", "b"]
    g = tmp_path / "a.csv.gz"
    g.write_bytes(gzip.compress("a;b\n1;2\n3;4\n".encode()))
    assert describe_format(g)["delimiter"] == ";" and read_table(g).shape == (2, 2)
    tg = tmp_path / "a.tsv.gz"
    tg.write_bytes(gzip.compress(b"a\tb\n1\t2\n"))
    assert read_table(tg).shape == (1, 2)


def test_unsplittable_header_raises_a_readable_error(tmp_path):
    p = _w(tmp_path, "bad.csv", 'a;b;c\n1;2\n3;4;5;6\n7\n')
    with pytest.raises(ValueError, match=r"bad\.csv: could not split the header.*';'"):
        read_table(p)


def test_nrows_still_works(tmp_path):
    p = _w(tmp_path, "a.csv", "a;b\n" + "".join(f"{i};{i}\n" for i in range(30)))
    assert len(read_table(p, nrows=5)) == 5 and len(read_table(p)) == 30


def test_empty_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        read_table(_w(tmp_path, "e.csv", ""))


# ---- profiles ---------------------------------------------------------------------------------

def _dated(tmp_path, n=30_000, head=20_000):
    """Date ordered: the first `head` rows have y=0 and a constant column, the rest differ."""
    y = np.r_[np.zeros(head, int), np.ones(n - head, int)]
    df = pd.DataFrame({"i": range(n), "const": np.where(np.arange(n) < head, "a", "b"), "y": y})
    p = tmp_path / "dated.csv"
    df.to_csv(p, index=False)
    return p


def test_profile_reads_the_whole_file_up_to_max_rows(tmp_path):
    df, total, sampled = load_for_profile(_dated(tmp_path))
    assert (len(df), total, sampled) == (30_000, 30_000, False)
    assert df["const"].nunique() == 2 and df["y"].mean() == pytest.approx(1 / 3)
    text = describe_table(tmp_path / "dated.csv", df, total_rows=total, sampled=sampled)
    assert text.startswith("### dated.csv (30,000 rows, 3 columns)")


def test_large_file_is_randomly_sampled_and_deterministic(tmp_path):
    p = _dated(tmp_path)
    a, total, sampled = load_for_profile(p, max_rows=5_000, seed=1)
    b, _, _ = load_for_profile(p, max_rows=5_000, seed=1)
    c, _, _ = load_for_profile(p, max_rows=5_000, seed=2)
    assert (len(a), total, sampled) == (5_000, 30_000, True)
    pd.testing.assert_frame_equal(a, b)
    assert not a["i"].equals(c["i"])
    assert a["i"].max() > 25_000 and a["const"].nunique() == 2   # not the head of the file
    assert a["i"].is_monotonic_increasing                         # original order kept
    assert describe_table(p, a, total_rows=total, sampled=sampled).startswith(
        "### dated.csv (random sample of 5,000 of 30,000 rows, 3 columns)")


def test_sampling_works_for_semicolon_gz(tmp_path):
    g = tmp_path / "big.csv.gz"
    g.write_bytes(gzip.compress(("a;b\n" + "".join(f"{i};{i * 2}\n" for i in range(2_000))).encode()))
    df, total, sampled = load_for_profile(g, max_rows=300)
    assert (len(df), total, sampled, df.shape[1]) == (300, 2_000, True, 2)


def test_planner_prompt_uses_the_full_data(tmp_path):
    p = _dated(tmp_path)
    llm = ScriptedLLM(['```json\n{"primary_table": "dated.csv", "target": "y", "metric": null, '
                       '"id_column": null, "reasoning": "y"}\n```'])
    plan([p], [], goal="predict y", llm=llm)
    assert "dated.csv (30,000 rows, 3 columns)" in llm.prompts[0]


def test_explore_shows_format_note_and_row_counts(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _dated(d)
    (d / "semi.csv").write_text("a;b\n" + "".join(f"{i};{i}\n" for i in range(7)))
    r = runner.invoke(app, ["explore", str(d), "-t", "y"], terminal_width=200)
    assert r.exit_code == 0, r.output
    assert "read semi.csv with delimiter ';'" in r.output
    assert "semi.csv (7 rows, 2 columns)" in r.output
    assert "profile of all 30,000 rows" in r.output and "Rows: 30,000" in r.output


def test_explore_says_when_the_profile_is_a_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "PROFILE_ROWS", 1_000)
    r = runner.invoke(app, ["explore", str(_dated(tmp_path)), "-t", "y"], terminal_width=200)
    assert r.exit_code == 0, r.output
    assert "profile of a random sample of 1,000 of 30,000 rows" in r.output


def test_unsplittable_file_is_reported_by_explore(tmp_path):
    p = _w(tmp_path, "bad.csv", 'a;b;c\n1;2\n3;4;5;6\n7\n')
    r = runner.invoke(app, ["explore", str(p)], terminal_width=200)
    assert r.exit_code == 1 and "could not split the header" in r.output


# ---- ambiguity --------------------------------------------------------------------------------

def _bank(d, n, name, extra_col=False):
    rng = np.random.default_rng(len(name))
    df = pd.DataFrame({"age": rng.integers(18, 90, n), "job": rng.choice(["a", "b", "c"], n),
                       "balance": rng.normal(1000, 300, n).round(2), "y": rng.choice(["no", "yes"], n, p=[.8, .2])})
    if extra_col:
        df["emp.var.rate"] = rng.normal(0, 1, n).round(1)
    df.to_csv(d / name, sep=";", index=False, quoting=1)


def test_ambiguity_note_names_the_choice_and_the_others(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _bank(d, 300, "bank-full.csv")
    _bank(d, 40, "bank.csv")
    _bank(d, 120, "bank-extra.csv", extra_col=True)
    p = plan(sorted(d.glob("*.csv")), [], target="y")
    note = next(n for n in p.notes if n.startswith("setup: 3 tables have column 'y'"))
    assert "using bank-full.csv (300 rows), the table with the most rows" in note
    assert "bank-extra.csv (120 rows)" in note and "bank.csv (40 rows)" in note


def test_ambiguity_note_carries_the_planner_reasoning(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _bank(d, 300, "bank-full.csv")
    _bank(d, 40, "bank.csv")
    llm = ScriptedLLM(['```json\n{"primary_table": "bank.csv", "target": "y", "metric": null, '
                       '"id_column": null, "reasoning": "the small file is the sample"}\n```'])
    p = plan(sorted(d.glob("*.csv")), [], goal="predict y", llm=llm)
    note = next(n for n in p.notes if "tables have column" in n)
    assert "using bank.csv (40 rows), chosen by the planner: the small file is the sample" in note
    assert "other candidates: bank-full.csv (300 rows)" in note


def test_train_named_table_wins_without_llm(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _bank(d, 300, "big.csv")
    _bank(d, 50, "train.csv")
    p = plan(sorted(d.glob("*.csv")), [], target="y")
    assert p.primary.name == "train.csv"
    assert any("the largest of the tables named like 'train'" in n for n in p.notes)


# ---- end to end -------------------------------------------------------------------------------

def _bank_zip(tmp_path):
    """Shaped like the UCI Bank Marketing download: zips inside a zip, ';' separated CSVs."""
    d = tmp_path / "build"
    d.mkdir()
    _bank(d, 300, "bank-full.csv")
    _bank(d, 60, "bank.csv")
    _bank(d, 280, "bank-additional-full.csv", extra_col=True)
    _bank(d, 50, "bank-additional.csv", extra_col=True)
    with zipfile.ZipFile(tmp_path / "bank.zip", "w") as z:
        z.write(d / "bank.csv", "bank.csv")
        z.write(d / "bank-full.csv", "bank-full.csv")
        z.writestr("bank-names.txt", "age: client age\ny: has the client subscribed a term deposit")
    with zipfile.ZipFile(tmp_path / "bank-additional.zip", "w") as z:
        z.write(d / "bank-additional-full.csv", "bank-additional/bank-additional-full.csv")
        z.write(d / "bank-additional.csv", "bank-additional/bank-additional.csv")
        z.writestr("bank-additional/bank-additional-names.txt", "emp.var.rate: employment variation")
    outer = tmp_path / "bank+marketing.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.write(tmp_path / "bank.zip", "bank.zip")
        z.write(tmp_path / "bank-additional.zip", "bank-additional.zip")
    return outer


def test_setup_task_on_a_bank_shaped_zip(tmp_path):
    zip_path = _bank_zip(tmp_path)
    task = setup_task([str(zip_path)], tmp_path / "run", target="y", folds=2)
    spec = task.spec
    assert spec.target == "y" and spec.data_path.endswith("bank-full.csv")
    notes = "\n".join(spec.notes)
    assert "read bank-full.csv with delimiter ';'" in notes
    assert "4 tables have column 'y'; using bank-full.csv (300 rows)" in notes
    assert "bank-additional-full.csv (280 rows)" in notes


def test_explore_on_a_bank_shaped_zip_leaves_nothing_behind(tmp_path, monkeypatch):
    zip_path = _bank_zip(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    r = runner.invoke(app, ["explore", str(zip_path), "-t", "y"], terminal_width=200)
    assert r.exit_code == 0, r.output
    for name in ("bank-full.csv", "bank.csv", "bank-additional-full.csv", "bank-additional.csv"):
        assert name in r.output
    assert "bank-names.txt" in r.output and "bank-additional-names.txt" in r.output
    flat = " ".join(r.output.split())
    assert ("read 4 tables (bank-additional-full.csv, bank-additional.csv, bank-full.csv and 1 more) "
            "with delimiter ';'") in flat
    assert flat.count("with delimiter") == 1  # one grouped note, not one per table
    assert "profile of all 300 rows" in r.output and "TARGET" in r.output
    assert list(work.iterdir()) == []


def test_format_notes_group_files_that_share_a_format(tmp_path):
    from orbit.planner import format_notes

    paths = []
    for name in ("a.csv", "b.csv", "c.csv", "d.csv", "e.csv"):
        (tmp_path / name).write_text("x;y\n1;2\n3;4\n")
        paths.append(tmp_path / name)
    plain = tmp_path / "plain.csv"
    plain.write_text("x,y\n1,2\n3,4\n")
    one = tmp_path / "one.csv"
    one.write_text("x|y\n1|2\n")
    notes = format_notes([*paths, plain, one])
    assert notes == ["read 5 tables (a.csv, b.csv, c.csv and 2 more) with delimiter ';'",
                     "read one.csv with delimiter '|'"]
    assert format_notes(paths[:2]) == ["read a.csv and b.csv with delimiter ';'"]
    assert format_notes(paths[:3]) == ["read a.csv, b.csv and c.csv with delimiter ';'"]
    assert format_notes([plain]) == []
