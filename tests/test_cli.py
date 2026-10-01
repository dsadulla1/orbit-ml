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
