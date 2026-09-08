from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rlordata.sampling.cap import (
    ANCHOR_CAP,
    PROVISIONAL_CAP,
    CapError,
    ceil_to_multiple,
    cell_key,
    compute_cap,
    length_histogram,
    load_locked_cap,
    measured_cap_from_lengths,
    per_cell_table,
    resolve_cap,
    write_cap_yaml,
)


def test_ceil_to_multiple() -> None:
    assert ceil_to_multiple(1, 256) == 256
    assert ceil_to_multiple(256, 256) == 256
    assert ceil_to_multiple(257, 256) == 512
    assert ceil_to_multiple(1237.5, 256) == 1280
    assert cell_key("S", 3) == "S3" and cell_key("M", 5) == "M5"


def test_measured_cap_has_no_floor() -> None:
    lengths = list(range(1, 1001))  # p99 (linear interpolation) = 990.01
    cap, p99 = measured_cap_from_lengths(lengths)
    assert p99 == pytest.approx(float(np.percentile(lengths, 99)))
    assert cap == ceil_to_multiple(1.25 * p99, 256) == 1280
    cap_small, _ = measured_cap_from_lengths([10, 20, 30])
    assert cap_small == 256  # no 512 floor any more; the 2048 anchor floor lives in compute_cap
    with pytest.raises(CapError):
        measured_cap_from_lengths([])


def _result(cells: dict[str, list[int]], incorrect: list[int] | None = None):
    incorrect = incorrect if incorrect is not None else [4096, 4096, 300]
    n = sum(len(v) for v in cells.values()) + len(incorrect)
    return compute_cap(cells, incorrect, n_problems=10, provisional_truncated=[False] * n)


def test_floor_binds_when_every_cell_is_short() -> None:
    # longest cell p99 ~ 1600 -> 1.25 × 1600 = 2000 <= 2048 -> anchor floor binds
    cells = {"S2": [100] * 200, "M5": list(range(400, 1601))}
    r = _result(cells)
    assert r.max_completion_tokens == ANCHOR_CAP == 2048
    assert r.binding_term == "floor" and r.measured_cap <= 2048
    assert r.driving_cell == "M5" and r.p99_max == pytest.approx(np.percentile(cells["M5"], 99))
    assert r.violations() == [] and r.cells_over_limit == ()
    assert r.frac_correct_over_anchor == 0.0 and r.frac_correct_truncated_at_cap == 0.0


def test_measurement_binds_and_is_a_multiple_of_256() -> None:
    cells = {
        "S2": [100] * 200,
        "M4": list(range(1000, 2001)),
    }  # p99 ≈ 1990 -> 1.25× = 2487.5 -> 2560
    r = _result(cells)
    assert r.binding_term == "measured"
    assert (
        r.max_completion_tokens == r.measured_cap == ceil_to_multiple(1.25 * r.p99_max, 256) == 2560
    )
    assert r.max_completion_tokens % 256 == 0 and r.max_completion_tokens > ANCHOR_CAP
    assert r.driving_cell == "M4"
    assert r.violations() == []


def test_driving_cell_is_largest_p99_not_largest_n() -> None:
    big = {"S2": [900] * 500 + [1000] * 5}  # n=505, p99 ≈ 1000
    small = {"M5": [1500] * 25 + [2100] * 5}  # n=30, p99 ≈ 2100
    r = _result({**big, **small})
    assert r.driving_cell == "M5" and r.per_cell["M5"].n_correct == 30
    assert r.p99_max == pytest.approx(np.percentile(small["M5"], 99))
    assert r.binding_term == "measured" and r.max_completion_tokens == ceil_to_multiple(
        1.25 * r.p99_max, 256
    )
    assert set(r.per_cell_p99) == {"S2", "M5"}


def test_cell_over_one_percent_is_a_violation() -> None:
    # 49 short + 1 huge: p99 interpolates well below the outlier, so 1/50 = 2 % of the cell
    # exceeds the chosen cap -> loud failure, nothing may be written.
    cells = {"S2": [200] * 300, "M5": [300] * 49 + [9000]}
    r = _result(cells)
    assert r.per_cell["M5"].frac_correct_over_cap == pytest.approx(1 / 50)
    assert r.cells_over_limit == ("M5",)
    violations = r.violations()
    assert any("cell M5" in v for v in violations)
    table = per_cell_table(r)
    assert "OVER 1% LIMIT" in table and "driving cell" in table and "binding term" in table


def test_compute_cap_diagnostics_and_table() -> None:
    cells = {"S2": [100] * 98 + [1000, 2500], "M3": [800] * 50 + [2100]}
    incorrect = [4096] * 5 + [50] * 5
    r = compute_cap(
        cells,
        incorrect,
        n_problems=20,
        provisional_truncated=[False] * 151 + [True] * 5 + [False] * 5,
    )
    assert r.n_correct_used == 151 and r.n_samples_total == 161 and r.n_problems == 20
    assert r.provisional_truncation_rate == pytest.approx(5 / 161)
    corr = np.array(cells["S2"] + cells["M3"])
    assert r.frac_correct_over_anchor == pytest.approx(np.mean(corr > 2048))
    assert r.frac_correct_truncated_at_cap == pytest.approx(np.mean(corr > r.max_completion_tokens))
    d = r.to_dict()
    assert d["per_cell"]["S2"]["n_correct"] == 100 and isinstance(d["cells_over_limit"], list)
    table = per_cell_table(r)
    assert (
        "S2" in table
        and "M3" in table
        and f"cap = max({ANCHOR_CAP}, {r.measured_cap}) = {r.max_completion_tokens}" in table
    )
    hist = length_histogram(corr.tolist(), incorrect, bin_width=1024)
    assert "correct" in hist.splitlines()[0] and len(hist.splitlines()) >= 2
    with pytest.raises(CapError):
        compute_cap({"S2": []}, [1], n_problems=1, provisional_truncated=[False])
    with pytest.raises(AssertionError):
        compute_cap({"S2": [1, 2]}, [1], n_problems=1, provisional_truncated=[False])


def test_resolve_cap_rules(tmp_path: Path) -> None:
    cap_path = tmp_path / "cap.yaml"
    with pytest.raises(CapError):
        resolve_cap(None, cap_path=cap_path)
    with pytest.raises(CapError):
        resolve_cap(PROVISIONAL_CAP, cap_path=cap_path)
    assert (
        resolve_cap(PROVISIONAL_CAP, cap_path=cap_path, allow_provisional=True) == PROVISIONAL_CAP
    )
    with pytest.raises(CapError):
        resolve_cap(None, cap_path=cap_path, allow_provisional=True)
    write_cap_yaml(cap_path, {"max_completion_tokens": 2048})
    assert load_locked_cap(cap_path) == 2048
    assert resolve_cap(None, cap_path=cap_path) == 2048
    assert resolve_cap(2048, cap_path=cap_path) == 2048
    with pytest.raises(CapError):
        resolve_cap(2304, cap_path=cap_path)
    with pytest.raises(CapError):  # even with allow_provisional the locked value wins
        resolve_cap(PROVISIONAL_CAP, cap_path=cap_path, allow_provisional=True)


def test_write_cap_yaml_refuses_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "locked" / "cap.yaml"
    write_cap_yaml(p, {"max_completion_tokens": 2048, "binding_term": "floor"})
    data = yaml.safe_load(p.read_text())
    assert data["max_completion_tokens"] == 2048
    assert p.read_text().startswith("# locked (SPEC §7 v1.3)")
    with pytest.raises(CapError):
        write_cap_yaml(p, {"max_completion_tokens": 2304})
    assert yaml.safe_load(p.read_text())["max_completion_tokens"] == 2048
    with pytest.raises(CapError):
        write_cap_yaml(tmp_path / "other.yaml", {"p99_max": 1.0})
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: 1\n")
    with pytest.raises(CapError):
        load_locked_cap(bad)
