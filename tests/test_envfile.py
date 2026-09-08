from __future__ import annotations

from pathlib import Path

from rlordata.envfile import gpu_rate_usd_per_hour, load_env, parse_env_text


def test_parse_env_text_quotes_comments_and_export() -> None:
    parsed = parse_env_text(
        "# comment\nA=1\nexport B='two words'\nC=\"x # not a comment\"\nD=val # trailing\nE=\n\nBAD LINE\n"
    )
    assert parsed == {"A": "1", "B": "two words", "C": "x # not a comment", "D": "val", "E": ""}
    # .env.example style: blank value followed by an inline comment is blank, not the comment text
    parsed = parse_env_text(
        "RLORDATA_ARTIFACTS=                # durable store for runs/\nX=  #c\n"
    )
    assert parsed == {"RLORDATA_ARTIFACTS": "", "X": ""}


def test_load_env_does_not_override_and_skips_blank(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / ".env"
    p.write_text("RLORDATA_GPU_RATE_USD_PER_HOUR=4.29\nFOO_TEST_ENV=from_file\nEMPTY_TEST_ENV=\n")
    monkeypatch.setenv("FOO_TEST_ENV", "from_shell")
    monkeypatch.delenv("RLORDATA_GPU_RATE_USD_PER_HOUR", raising=False)
    monkeypatch.delenv("EMPTY_TEST_ENV", raising=False)
    applied = load_env(p)
    assert applied == {"RLORDATA_GPU_RATE_USD_PER_HOUR": "4.29"}
    assert gpu_rate_usd_per_hour() == 4.29
    import os

    assert os.environ["FOO_TEST_ENV"] == "from_shell"
    assert "EMPTY_TEST_ENV" not in os.environ
    assert load_env(tmp_path / "missing.env") == {}
