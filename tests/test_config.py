from orbit.config import load_dotenv, tracker_choice


def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA=1\nB = two # trailing note\nC='x # kept'\nD=\"q\"\nE=already\n")
    for k in "ABCD":
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("E", "set")
    load_dotenv(env)
    import os
    assert [os.environ[k] for k in "ABCDE"] == ["1", "two", "x # kept", "q", "set"]
    for k in "ABCD":
        monkeypatch.delenv(k)


def test_tracker_choice(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert tracker_choice() == "mlflow"
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert tracker_choice() == "wandb"
    assert tracker_choice("none") == "none"
