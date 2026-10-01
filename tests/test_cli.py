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
