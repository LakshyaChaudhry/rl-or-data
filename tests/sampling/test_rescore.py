from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from rlordata.cli import main
from rlordata.data.generator import read_jsonl
from rlordata.sampling.cap import write_cap_yaml
from rlordata.sampling.eval_runner import model_slug, read_samples, write_samples
from rlordata.sampling.rescore import (
    answers_from_pools,
    rescore_file,
    rescore_run_dir,
    rescore_samples,
)
from tests.helpers import make_world


def test_rescore_restores_verdicts_and_simulates_cap(tmp_path: Path) -> None:
    w = make_world(tmp_path, n_pool=400, n_ood=6)
    assert w.tier_with_stub() == 0
    samples = read_samples(w.samples_path)
    answers = answers_from_pools([read_jsonl(w.pool_path), read_jsonl(w.ood_path)])
    # corrupt the stored verdicts; rescoring with the current verifier must restore them exactly
    corrupted = [
        replace(s, correct=not s.correct, reward=1.0 - s.reward, extraction_failed=False)
        for s in samples
    ]
    restored, summary = rescore_samples(corrupted, answers=answers, cap=None)
    assert [(s.correct, s.extraction_failed, s.extracted_answer) for s in restored] == [
        (s.correct, s.extraction_failed, s.extracted_answer) for s in samples
    ]
    assert summary["n_verdicts_changed"] == len(samples) and summary["n_cap_truncated"] == 0
    assert restored[0].extra["rescore"] == {
        "cap": None,
        "cap_truncated": False,
        "original_n_tokens": samples[0].n_tokens,
    }
    # cap simulation: every completion longer than the cap becomes truncated and unscored
    cap = 300
    capped, summary = rescore_samples(samples, answers=answers, cap=cap)
    longer = [s for s in samples if s.n_tokens > cap]
    assert summary["n_cap_truncated"] == len(longer) > 0
    for s in capped:
        if s.extra["rescore"]["cap_truncated"]:
            assert s.truncated and s.n_tokens == cap and not s.correct and s.extraction_failed
        else:
            assert s.n_tokens <= cap
    assert summary["accuracy_after"] <= summary["accuracy_before"]


def test_rescore_file_keeps_raw_and_is_idempotent(tmp_path: Path) -> None:
    w = make_world(tmp_path, n_pool=400, n_ood=6)
    assert w.tier_with_stub() == 0
    answers = answers_from_pools([read_jsonl(w.pool_path), read_jsonl(w.ood_path)])
    original = w.samples_path.read_bytes()
    rescore_file(w.samples_path, answers=answers, cap=300)
    raw = w.samples_path.with_name("tiering_pass8.raw.jsonl")
    assert raw.read_bytes() == original
    once = w.samples_path.read_bytes()
    rescore_file(w.samples_path, answers=answers, cap=300)  # from raw again, not compounding
    assert w.samples_path.read_bytes() == once
    assert raw.read_bytes() == original


def test_rescore_run_dir_updates_metrics_and_meta(tmp_path: Path) -> None:
    w = make_world(tmp_path, n_pool=600, n_ood=6)
    cfg = w.eval_config(
        splits=["val_candidates"],
        provisional_cap=4096,
        models=[{"id": "Qwen/Qwen3-4B-Base", "kind": "base", "arm": "base"}],
        decoding={"mean_at_k": {"temperature": 1.0, "top_p": 1.0, "n": 8}},
        output_dir=str(w.root / "runs" / "cap_provisional"),
    )
    assert main(["eval", "--config", str(cfg), "--stub"]) == 0
    run = (
        w.root
        / "runs"
        / "cap_provisional"
        / model_slug("Qwen/Qwen3-4B-Base")
        / "val_candidates"
        / "mean_at_k"
    )
    before = json.loads((run / "metrics.json").read_text())
    summary = rescore_run_dir(run, answers=answers_from_pools([read_jsonl(w.pool_path)]), cap=None)
    after = json.loads((run / "metrics.json").read_text())
    assert summary["n_verdicts_changed"] == 0  # same verifier -> same verdicts
    assert after["accuracy"] == before["accuracy"] and after["run_id"] == before["run_id"]
    assert after["rescore"]["n_samples"] == 4000 and "per_tier_truncation_rate" in after
    meta = json.loads((run / "meta.json").read_text())
    assert len(meta["rescore_history"]) == 1 and (run / "samples.raw.jsonl").exists()


def test_tier_provisional_then_rescore_from_matches_direct_tiering(tmp_path: Path, capsys) -> None:
    """Provisional sampling + offline rescore at the locked cap == tiering directly at that cap."""
    w = make_world(tmp_path)
    # provisional: no cap.yaml, splits go to a _provisional dir, completions stored
    rc = main(["tier", "--config", str(w.tier_config), "--stub", "--provisional-cap", "4096"])
    out = capsys.readouterr().out
    assert rc == 0 and "PROVISIONAL TIERING" in out
    assert (w.splits_dir.parent / "splits_provisional" / "test_300.jsonl").exists()
    assert not (w.splits_dir / "test_300.jsonl").exists()
    raw_samples = read_samples(w.samples_path)
    assert len(raw_samples) == (800 + 30) * 8
    # lock a cap below the provisional one and rescore
    write_cap_yaml(w.cap_path, {"max_completion_tokens": 512})
    rc = main(["tier", "--config", str(w.tier_config), "--rescore-from", str(w.samples_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "rescored" in out and "cap-truncated" in out
    rescored_test = read_jsonl(w.splits_dir / "test_300.jsonl")
    assert len(rescored_test) == 60
    run_dirs = sorted(w.tier_run_dir.iterdir())
    rd = next(d for d in run_dirs if "rescore" in d.name)
    cfg = yaml.safe_load((rd / "config.yaml").read_text())
    assert (
        cfg["kind"] == "tier_rescore"
        and cfg["max_completion_tokens"] == 512
        and cfg["rescore"]["cap"] == 512
    )
    assert (rd / "tiering_pass8.jsonl").exists() and (
        w.samples_path.with_name("tiering_pass8.raw.jsonl")
    ).exists()
    # pass8 in the tiered pool equals the count of correct rescored samples per problem
    tiered = read_jsonl(w.splits_dir / "pool_tiered.jsonl")
    rescored = read_samples(w.samples_path)
    correct: dict[str, int] = {}
    for s in rescored:
        correct[s.problem_id] = correct.get(s.problem_id, 0) + int(s.correct)
    assert all(p.pass8 == correct.get(p.problem_id, 0) for p in tiered)
    assert all(s.n_tokens <= 512 for s in rescored)
    # direct tiering at the locked cap (fresh world, same seeds) gives the same pass8 for every problem
    w2 = make_world(tmp_path / "direct")
    write_cap_yaml(w2.cap_path, {"max_completion_tokens": 512})
    assert w2.tier_with_stub() == 0
    direct = {p.problem_id: p.pass8 for p in read_jsonl(w2.splits_dir / "pool_tiered.jsonl")}
    via_rescore = {p.problem_id: p.pass8 for p in tiered}
    # The stub's completions are seeded per (seed, prompt, sample) and its truncation is length-based,
    # so the simulated cap must agree with sampling at that cap on every problem.
    assert via_rescore == direct
    capsys.readouterr()


def test_rescore_from_refuses_mismatched_samples(tmp_path: Path) -> None:
    w = make_world(tmp_path, n_pool=400, n_ood=6)
    assert w.tier_with_stub() == 0
    write_cap_yaml(w.cap_path, {"max_completion_tokens": 2048})
    samples = read_samples(w.samples_path)
    write_samples(samples[:-8], w.samples_path)  # drop one problem's samples
    import pytest

    with pytest.raises(SystemExit):
        main(["tier", "--config", str(w.tier_config), "--rescore-from", str(w.samples_path)])
