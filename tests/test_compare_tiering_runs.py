from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_tiering_runs", REPO_ROOT / "scripts" / "compare_tiering_runs.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _s(
    pid: str, idx: int, completion: str, n_tokens: int, *, truncated: bool, correct: bool
) -> dict:
    return {
        "data_condition": "pool",
        "problem_id": pid,
        "extra": {"sample_idx": idx},
        "completion": completion,
        "n_tokens": n_tokens,
        "truncated": truncated,
        "correct": correct,
    }


def _key(s: dict) -> tuple[str, str, int]:
    return (s["data_condition"], s["problem_id"], s["extra"]["sample_idx"])


def test_identical_and_continued_samples_pass() -> None:
    ref = [
        _s("a", 0, "short Answer: 1", 5, truncated=False, correct=True),
        _s("a", 1, "x" * 8, 8, truncated=True, correct=False),  # hit the reference cap (8)
    ]
    cand = [
        _s("a", 0, "short Answer: 1", 5, truncated=False, correct=True),
        _s("a", 1, "x" * 8 + " Answer: 1", 11, truncated=False, correct=True),
    ]
    out = mod.compare({_key(s): s for s in ref}, {_key(s): s for s in cand})
    assert out["ok"]
    assert out["n_identical"] == 1 and out["n_continued_past_reference_cap"] == 1
    assert out["pass8_changes"] == {"pool:a": (1, 2)}


def test_divergence_is_flagged() -> None:
    ref = [_s("a", 0, "Answer: 1", 3, truncated=False, correct=True)]
    cand = [_s("a", 0, "Answer: 2", 3, truncated=False, correct=False)]
    out = mod.compare({_key(s): s for s in ref}, {_key(s): s for s in cand})
    assert not out["ok"] and out["n_mismatch"] == 1
    out = mod.compare({_key(s): s for s in ref}, {})
    assert not out["ok"] and out["n_missing_in_candidate"] == 1
