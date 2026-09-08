from __future__ import annotations

import json
from pathlib import Path

import yaml

from rlordata.analysis import sanity
from rlordata.data.tiers import build_splits, tier_from_pass8
from rlordata.types import Problem
from tests.helpers import TINY_SPLITS, small_pool


def _tiered_pool(n: int = 800) -> list[Problem]:
    return [
        Problem(**{**p.to_dict(), "pass8": i % 9, "tier": tier_from_pass8(i % 9)})
        for i, p in enumerate(small_pool(n))
    ]


def test_disjointness_passes_on_real_splits_and_catches_leaks() -> None:
    splits = build_splits(_tiered_pool(), seed=3, split_spec=TINY_SPLITS)
    assert sanity.check_split_disjointness(splits) == []
    # leak by id
    leaked = dict(splits)
    leaked["val_mixed_100"] = splits["val_mixed_100"] + [splits["test_300"][0]]
    issues = sanity.check_split_disjointness(leaked)
    assert any("shared problem_id" in i for i in issues)
    # leak by structure only (same pipeline, different range)
    src = splits["test_300"][1]
    pipeline = json.loads(json.dumps(src.pipeline))
    pipeline["range"] = {"lo": pipeline["range"]["lo"] + 1, "hi": pipeline["range"]["hi"] + 1}
    clone = Problem(**{**src.to_dict(), "problem_id": "clone", "pipeline": pipeline})
    leaked2 = dict(splits)
    leaked2["train_easy_100"] = splits["train_easy_100"] + [clone]
    issues = sanity.check_split_disjointness(leaked2)
    assert any("shared pipeline structure" in i for i in issues) and not any(
        "shared problem_id" in i for i in issues
    )
    # curated must be a subset of mixed
    broken = dict(splits)
    broken["train_curated"] = splits["train_curated"] + [splits["test_300"][2]]
    assert any("not a subset" in i for i in sanity.check_split_disjointness(broken))
    # duplicates within a split
    dup = dict(splits)
    dup["test_300"] = splits["test_300"] + [splits["test_300"][0]]
    assert any("duplicate problem_id" in i for i in sanity.check_split_disjointness(dup))
    # transfer sets have no structure; only id overlap is checked
    transfer = [
        Problem(
            problem_id=f"t{i}",
            text="q",
            answer=1,
            pipeline={"source": "gsm8k", "index": i},
            range_scale="S",
            n_filters=0,
            n_transforms=0,
            total_steps=0,
        )
        for i in range(3)
    ]
    ok = dict(splits)
    ok["gsm8k_500"] = transfer
    assert sanity.check_split_disjointness(ok) == []


def _run_dir(root: Path, name: str, cfg: dict, metrics: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics))
    return d


def test_protocol_identity_and_rate_flags(tmp_path: Path) -> None:
    base = {
        "max_completion_tokens": 1024,
        "prompt_template": "T {problem_text}",
        "answer_regex": "^Answer",
        "max_prompt_tokens": 1024,
    }
    good = {"accuracy": 0.5, "truncation_rate": 0.01, "extraction_failure_rate": 0.02}
    a = _run_dir(tmp_path, "a", base, good)
    b = _run_dir(tmp_path, "b", dict(base), good)
    assert sanity.check_protocol_identical([a, b]) == []
    assert sanity.check_run_dirs([a, b]) == []
    c = _run_dir(
        tmp_path, "c", {**base, "max_completion_tokens": 2048}, {**good, "truncation_rate": 0.2}
    )
    issues = sanity.check_run_dirs([a, c])
    assert any("max_completion_tokens differs" in i for i in issues)
    assert any("truncation rate 0.200 > 0.05" in i for i in issues)
    d = _run_dir(tmp_path, "d", {**base, "prompt_template": "other", "cap_is_provisional": True})
    issues = sanity.check_run_dirs([a, d])
    assert any("prompt_template differs" in i for i in issues)
    assert any("provisional cap" in i for i in issues)
    assert any("no metrics.json" in i for i in issues)
    e = _run_dir(tmp_path, "e", {"max_completion_tokens": 1024})
    assert any("lacks" in i for i in sanity.check_protocol_identical([a, e]))
    assert sanity.check_rates(
        {"truncation_rate": 0.0, "extraction_failure_rate": 0.5}, label="x"
    ) == ["x: extraction-failure rate 0.500 > 0.05"]
    assert sanity.check_protocol_identical([a]) == []


def test_cli_on_splits_dir(tmp_path: Path, capsys) -> None:
    from rlordata.data.generator import write_jsonl

    splits = build_splits(_tiered_pool(), seed=3, split_spec=TINY_SPLITS)
    d = tmp_path / "splits"
    for name, probs in splits.items():
        write_jsonl(probs, d / f"{name}.jsonl")
    write_jsonl(_tiered_pool(100), d / "pool_tiered.jsonl")  # ignored by the checker
    assert sanity.main(["--splits-dir", str(d)]) == 0
    assert "OK" in capsys.readouterr().out
    write_jsonl(splits["test_300"][:2], d / "leak.jsonl")
    assert sanity.main(["--splits-dir", str(d)]) == 1
    assert sanity.main([]) == 2
