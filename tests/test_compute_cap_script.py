from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rlordata.cli import main as cli_main
from rlordata.sampling.cap import cap_from_correct_lengths
from rlordata.sampling.eval_runner import read_samples
from tests.helpers import REPO, make_world


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "compute_cap", REPO / "scripts" / "compute_cap.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def provisional_run(tmp_path_factory) -> Path:
    """A stub provisional cap run: base model, T=1, n=8, cap 4096, val_candidates (needs >= 500 pool problems)."""
    w = make_world(tmp_path_factory.mktemp("cap"), n_pool=600, n_ood=6)
    cfg = w.eval_config(
        splits=["val_candidates"],
        provisional_cap=4096,
        models=[{"id": "Qwen/Qwen3-4B-Base", "kind": "base", "arm": "base"}],
        decoding={"mean_at_k": {"temperature": 1.0, "top_p": 1.0, "n": 8}},
        output_dir=str(w.root / "runs" / "cap_provisional"),
    )
    assert cli_main(["eval", "--config", str(cfg), "--stub"]) == 0
    return (
        w.root / "runs" / "cap_provisional" / "Qwen__Qwen3-4B-Base" / "val_candidates" / "mean_at_k"
    )


def test_compute_cap_end_to_end_and_refuses_overwrite(
    provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    out = tmp_path / "locked" / "cap.yaml"
    rc = mod.main(["--run-dir", str(provisional_run), "--out", str(out), "--allow-stub-samples"])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert (
        "tokens" in printed and "correct" in printed and "incorrect" in printed
    )  # histogram header
    assert "correct completions truncated at cap" in printed
    data = yaml.safe_load(out.read_text())
    samples = read_samples(provisional_run / "samples.jsonl")
    correct = [s.n_tokens for s in samples if s.correct]
    cap, p99 = cap_from_correct_lengths(correct)
    assert data["max_completion_tokens"] == cap and cap % 256 == 0 and cap >= 512
    assert data["p99_correct_len"] == pytest.approx(p99, abs=0.01)
    assert data["n_correct_used"] == len(correct)
    assert data["n_problems"] == 500 and data["n_samples_total"] == 4000
    for key in (
        "computed_on",
        "config_hash",
        "run_id",
        "git_sha",
        "computed_at",
        "rule",
        "frac_correct_truncated_at_cap",
    ):
        assert key in data, key
    assert data["config_hash"] == (provisional_run / "config_hash.txt").read_text().strip()
    assert data["run_id"] == json.loads((provisional_run / "meta.json").read_text())["run_id"]
    assert data["frac_correct_truncated_at_cap"] == pytest.approx(
        float(np.mean(np.array(correct) > cap)), abs=1e-6
    )  # the record is rounded to 6 decimals
    assert data["frac_correct_truncated_at_cap"] < 0.01
    # refuses to overwrite, file unchanged
    before = out.read_text()
    rc = mod.main(["--run-dir", str(provisional_run), "--out", str(out), "--allow-stub-samples"])
    assert rc == 2 and "REFUSING" in capsys.readouterr().err
    assert out.read_text() == before


def test_compute_cap_refuses_stub_samples_and_protocol_mismatch(
    provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    rc = mod.main(["--run-dir", str(provisional_run), "--out", str(tmp_path / "cap.yaml")])
    assert rc == 2 and "not vllm" in capsys.readouterr().err
    assert not (tmp_path / "cap.yaml").exists()
    # stub samples can never target the real locked path
    cfg = yaml.safe_load((provisional_run / "config.yaml").read_text())
    issues = mod.validate_run(cfg, allow_stub=True, out=REPO / "configs" / "locked" / "cap.yaml")
    assert any("real configs/locked/cap.yaml" in i for i in issues)
    # protocol mismatches are all reported
    bad = dict(cfg)
    bad["decoding"] = {**cfg["decoding"], "temperature": 0.0, "n": 1}
    bad["max_completion_tokens"] = 2048
    bad["split"] = "test_300"
    bad["model"] = {"id": "Qwen/Qwen3-4B", "kind": "instruct"}
    issues = mod.validate_run(bad, allow_stub=True, out=tmp_path / "x.yaml")
    assert len(issues) == 6
