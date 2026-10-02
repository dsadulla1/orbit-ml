import pandas as pd
from typer.testing import CliRunner

from orbit.cli import app

runner = CliRunner()


def test_explore_prints_profile(clf_csv):
    result = runner.invoke(app, ["explore", str(clf_csv), "-t", "label"])
    assert result.exit_code == 0 and "TARGET" in result.output


def test_explore_unknown_target(clf_csv):
    result = runner.invoke(app, ["explore", str(clf_csv), "-t", "nope"])
    assert result.exit_code != 0 and "not found" in result.output


def test_run_rejects_unknown_tracker(clf_csv):
    result = runner.invoke(app, ["run", str(clf_csv), "-t", "label", "--tracker", "mlfow"])
    assert result.exit_code != 0 and "choose one of" in result.output


def test_model_text_is_not_rich_markup():
    from orbit.cli import _last_line, _short

    assert _short("fix [red]this[/red] [bt] now") == r"fix \[red]this\[/red] \[bt] now"
    assert _last_line("Traceback\n  [bt] (8) libfoo.so\nValueError: [/x] bad").startswith("ValueError: \\[/x]")


def test_markup_in_names_does_not_crash_the_terminal_output(tmp_path):
    import pandas as pd

    path = tmp_path / "[bold].csv"
    pd.DataFrame({"[/odd]": range(10), "label": [0, 1] * 5}).to_csv(path, index=False)
    result = runner.invoke(app, ["explore", str(path), "-t", "label"])
    assert result.exit_code == 0 and "[/odd]" in result.output


def test_explore_folder_xlsx_and_failures(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame({"id": range(10), "label": [0, 1] * 5}).to_csv(d / "train.csv", index=False)
    with pd.ExcelWriter(d / "extra.xlsx") as w:
        pd.DataFrame({"id": range(10), "w": range(10)}).to_excel(w, sheet_name="a", index=False)
        pd.DataFrame({"k": [1]}).to_excel(w, sheet_name="b", index=False)
    (d / "notes.txt").write_text("hello")
    r = runner.invoke(app, ["explore", str(d), "-t", "label"])
    assert r.exit_code == 0 and "TARGET" in r.output and "extra__a.csv" in r.output
    assert "notes.txt" in r.output
    r = runner.invoke(app, ["explore", str(d / "extra.xlsx")])
    assert r.exit_code == 0 and "TARGET" not in r.output and "extra__b.csv" in r.output
    r = runner.invoke(app, ["explore", str(d), "-t", "nope"])
    assert r.exit_code == 1 and "not found" in r.output and "train.csv: id, label" in r.output
    r = runner.invoke(app, ["explore", str(tmp_path / "missing")])
    assert r.exit_code == 1 and "Could not read the data" in r.output and "Traceback" not in r.output
