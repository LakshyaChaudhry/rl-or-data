from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rlordata.sampling.cap import (
    PROVISIONAL_CAP,
    CapError,
    cap_from_correct_lengths,
    ceil_to_multiple,
    compute_cap,
    length_histogram,
    load_locked_cap,
    resolve_cap,
    write_cap_yaml,
)


def test_ceil_to_multiple() -> None:
    assert ceil_to_multiple(1, 256) == 256
    assert ceil_to_multiple(256, 256) == 256
    assert ceil_to_multiple(257, 256) == 512
    assert ceil_to_multiple(1237.5, 256) == 1280


def test_cap_formula_matches_spec_rule() -> None:
    lengths = list(range(1, 1001))  # p99 (linear interpolation) = 990.01
    cap, p99 = cap_from_correct_lengths(lengths)
    assert p99 == pytest.approx(float(np.percentile(lengths, 99)))
    assert cap == ceil_to_multiple(1.25 * p99, 256) == 1280
    # floor at 512
    cap_small, _ = cap_from_correct_lengths([10, 20, 30])
    assert cap_small == 512
    # multiples of 256 only
    for n in (300, 700, 2000, 3000):
        cap_n, _ = cap_from_correct_lengths(list(range(1, n)))
        assert cap_n % 256 == 0 and cap_n >= 512
    with pytest.raises(CapError):
        cap_from_correct_lengths([])


def test_resolve_cap_rules(tmp_path: Path) -> None:
    cap_path = tmp_path / "cap.yaml"
    # missing, not allowed
    with pytest.raises(CapError):
        resolve_cap(None, cap_path=cap_path)
    with pytest.raises(CapError):
        resolve_cap(PROVISIONAL_CAP, cap_path=cap_path)
    # missing, allowed -> requested
    assert (
        resolve_cap(PROVISIONAL_CAP, cap_path=cap_path, allow_provisional=True) == PROVISIONAL_CAP
    )
    with pytest.raises(CapError):
        resolve_cap(None, cap_path=cap_path, allow_provisional=True)
    # locked
    write_cap_yaml(cap_path, {"max_completion_tokens": 1024})
    assert load_locked_cap(cap_path) == 1024
    assert resolve_cap(None, cap_path=cap_path) == 1024
    assert resolve_cap(1024, cap_path=cap_path) == 1024
    with pytest.raises(CapError):
        resolve_cap(1280, cap_path=cap_path)
    with pytest.raises(CapError):  # even with allow_provisional the locked value wins
        resolve_cap(PROVISIONAL_CAP, cap_path=cap_path, allow_provisional=True)


def test_write_cap_yaml_refuses_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "locked" / "cap.yaml"
    write_cap_yaml(p, {"max_completion_tokens": 768, "p99_correct_len": 600.0})
    data = yaml.safe_load(p.read_text())
    assert data["max_completion_tokens"] == 768
    assert p.read_text().startswith("# locked (SPEC §7)")
    with pytest.raises(CapError):
        write_cap_yaml(p, {"max_completion_tokens": 512})
    assert yaml.safe_load(p.read_text())["max_completion_tokens"] == 768
    with pytest.raises(CapError):
        write_cap_yaml(tmp_path / "other.yaml", {"p99_correct_len": 1.0})
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: 1\n")
    with pytest.raises(CapError):
        load_locked_cap(bad)


def test_compute_cap_diagnostics() -> None:
    correct = [100] * 98 + [1000, 5000]
    incorrect = [4096] * 5 + [50] * 5
    res = compute_cap(
        correct,
        incorrect,
        n_problems=20,
        provisional_truncated=[False] * 100 + [True] * 5 + [False] * 5,
    )
    assert res.n_correct_used == 100 and res.n_samples_total == 110 and res.n_problems == 20
    assert res.max_completion_tokens % 256 == 0
    assert res.frac_correct_truncated_at_cap == pytest.approx(
        np.mean(np.array(correct) > res.max_completion_tokens)
    )
    assert res.provisional_truncation_rate == pytest.approx(5 / 110)
    hist = length_histogram(correct, incorrect, bin_width=1024)
    assert "correct" in hist.splitlines()[0]
    assert len(hist.splitlines()) >= 2
