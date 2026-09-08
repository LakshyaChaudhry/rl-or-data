from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from rlordata.data.generator import read_jsonl
from rlordata.data.tiers import pass8_histogram, sample_pass8, tier_counts, tier_from_pass8
from rlordata.run_dir import REQUIRED_RUN_FILES
from rlordata.sampling.cap import write_cap_yaml
from rlordata.sampling.eval_runner import read_samples
from rlordata.sampling.stub_sampler import StubSampler
from rlordata.types import Problem
from tests.helpers import make_world, small_pool


def test_sample_pass8_counts_and_generation_order(tmp_path: Path) -> None:
    probs = small_pool(n=30)
    cap_path = tmp_path / "cap.yaml"
    write_cap_yaml(cap_path, {"max_completion_tokens": 1024})
    sampler = StubSampler.from_problems(
        probs,
        model_id="m",
        model_kind="base",
        max_completion_tokens=1024,
        seed=1,
        cap_path=cap_path,
    )
    tiered, samples = sample_pass8(
        probs, sampler, k=8, temperature=1.0, top_p=1.0, run_id="r", config_hash="h", seed=1
    )
    assert len(tiered) == 30 and len(samples) == 240
    by_pid: dict[str, list] = {}
    for s in samples:
        by_pid.setdefault(s.problem_id, []).append(s)
    for p in tiered:
        group = by_pid[p.problem_id]
        assert [s.extra["sample_idx"] for s in group] == list(range(8))
        assert p.pass8 == sum(s.correct for s in group)
        assert p.tier == tier_from_pass8(p.pass8)
        assert all(
            s.data_condition == "pool" and s.arm == "base" and s.run_id == "r" for s in group
        )
    # deterministic
    tiered2, samples2 = sample_pass8(
        probs, sampler, k=8, temperature=1.0, top_p=1.0, run_id="r", config_hash="h", seed=1
    )
    assert tiered2 == tiered and samples2 == samples
    hist = pass8_histogram(tiered)
    assert hist.startswith("pass@8 histogram (n=30)") and len(hist.splitlines()) == 10
    assert sum(tier_counts(tiered).values()) == 30
    with pytest.raises(ValueError):
        bad = StubSampler.from_problems(
            probs,
            model_id="m",
            model_kind="instruct",
            max_completion_tokens=1024,
            seed=1,
            cap_path=cap_path,
        )
        sample_pass8(
            probs, bad, k=8, temperature=1.0, top_p=1.0, run_id="r", config_hash="h", seed=1
        )


def test_tier_cli_with_stub_writes_everything(tmp_path: Path, capsys) -> None:
    w = make_world(tmp_path)
    rc = w.tier_with_stub()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "DRY RUN" in out and "pass@8 histogram" in out and "pool tier counts" in out
    assert "[sanity] split disjointness: OK" in out

    # splits + tiered pool + post-hoc ood
    for name in (
        "train_easy_100",
        "train_mixed_100",
        "val_mixed_100",
        "test_300",
        "train_curated",
        "ood_hard_200",
        "pool_tiered",
    ):
        assert (w.splits_dir / f"{name}.jsonl").exists(), name
    test = read_jsonl(w.splits_dir / "test_300.jsonl")
    assert Counter(p.tier for p in test) == {"easy": 20, "medium": 20, "hard": 20}
    assert all(p.split == "test_300" and p.pass8 is not None for p in test)
    curated = read_jsonl(w.splits_dir / "train_curated.jsonl")
    mixed_ids = {p.problem_id for p in read_jsonl(w.splits_dir / "train_mixed_100.jsonl")}
    assert curated and all(1 <= p.pass8 <= 7 and p.problem_id in mixed_ids for p in curated)
    ood = read_jsonl(w.splits_dir / "ood_hard_200.jsonl")
    assert len(ood) == 30 and all(
        p.pass8 is not None and p.tier != "untiered" and p.split == "ood_hard_200" for p in ood
    )
    pool_tiered = read_jsonl(w.splits_dir / "pool_tiered.jsonl")
    assert len(pool_tiered) == 800 and all(p.pass8 is not None for p in pool_tiered)

    # all 8 completions per problem, generation order, pool + ood
    samples = read_samples(w.samples_path)
    assert len(samples) == (800 + 30) * 8
    per = Counter(s.problem_id for s in samples)
    assert set(per.values()) == {8}
    assert {s.data_condition for s in samples} == {"pool", "ood_hard_200"}
    pass8 = {p.problem_id: p.pass8 for p in pool_tiered + ood}
    correct = Counter(s.problem_id for s in samples if s.correct)
    assert all(pass8[pid] == correct.get(pid, 0) for pid in pass8)

    # run directory provenance
    run_dirs = list(w.tier_run_dir.iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    for f in REQUIRED_RUN_FILES + ("tier_summary.json", "tiering_pass8.jsonl", "test_300.jsonl"):
        assert (rd / f).exists(), f
    cfg = yaml.safe_load((rd / "config.yaml").read_text())
    assert cfg["k"] == 8 and cfg["temperature"] == 1.0 and cfg["cap_is_provisional"] is True
    assert cfg["sampler"]["sampler"] == "stub" and cfg["max_completion_tokens"] == 4096
    meta = json.loads((rd / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["n_samples"] == len(samples)

    # second run: pool_tiered has pass8 -> re-running on it builds splits without sampling
    cfg2 = yaml.safe_load(w.tier_config.read_text())
    cfg2["input"] = str(w.splits_dir / "pool_tiered.jsonl")
    cfg2["post_hoc_tier_inputs"] = [str(w.splits_dir / "ood_hard_200.jsonl")]
    cfg2["output_dir"] = str(tmp_path / "splits2")
    p2 = tmp_path / "tier2.yaml"
    p2.write_text(yaml.safe_dump(cfg2))
    from rlordata.cli import main

    assert main(["tier", "--config", str(p2)]) == 0
    assert "need_sampling=False" in capsys.readouterr().out
    assert read_jsonl(tmp_path / "splits2" / "test_300.jsonl") == test


def test_tier_cli_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    w = make_world(tmp_path, n_pool=200, n_ood=6)
    assert w.tier_with_stub(["--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not w.splits_dir.exists() and not w.samples_path.exists()


def test_build_splits_independent_of_spec_key_order() -> None:
    from rlordata.data.tiers import build_splits
    from tests.helpers import TINY_SPLITS

    pool = [
        Problem(**{**p.to_dict(), "pass8": i % 9, "tier": tier_from_pass8(i % 9)})
        for i, p in enumerate(small_pool(n=800))
    ]
    a = build_splits(pool, seed=1, split_spec=TINY_SPLITS)
    reordered = {
        k: (dict(sorted(v.items(), reverse=True)) if "derived_from" not in v else v)
        for k, v in sorted(TINY_SPLITS.items())
    }
    b = build_splits(pool, seed=1, split_spec=reordered)
    assert {k: [p.problem_id for p in v] for k, v in a.items()} == {
        k: [p.problem_id for p in v] for k, v in b.items()
    }


def test_splits_are_tier_interleaved_so_prefixes_are_stratified() -> None:
    from rlordata.data.tiers import build_splits, interleave_tiers
    from tests.helpers import TINY_SPLITS

    pool = [
        Problem(**{**p.to_dict(), "pass8": i % 9, "tier": tier_from_pass8(i % 9)})
        for i, p in enumerate(small_pool(n=800))
    ]
    splits = build_splits(pool, seed=1, split_spec=TINY_SPLITS)
    test = splits["test_300"]
    assert [p.tier for p in test[:6]] == ["easy", "medium", "hard", "easy", "medium", "hard"]
    first = Counter(p.tier for p in test[:20])  # "first N by index" is stratified to within 1
    assert max(first.values()) - min(first.values()) <= 1
    assert Counter(p.tier for p in test) == {"easy": 20, "medium": 20, "hard": 20}
    val = splits["val_mixed_100"]
    assert Counter(p.tier for p in val[:9]) == {"easy": 3, "medium": 3, "hard": 3}
    # interleave keeps within-tier order and is a pure reordering
    again = interleave_tiers(test)
    assert again == test and sorted(p.problem_id for p in again) == sorted(
        p.problem_id for p in test
    )
    assert [p.problem_id for p in again if p.tier == "hard"] == [
        p.problem_id for p in test if p.tier == "hard"
    ]
