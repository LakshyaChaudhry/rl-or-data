"""Results packet (analysis/packet.py): line budget, verbatim grading sheet, blank verdicts, flat CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from rlordata.analysis import packet, report
from tests.analysis.test_report import (  # noqa: F401  (fixtures)
    _copy,
    _different_cap,
    _fast_core_bootstrap,
    store,
)


def _run(cfg: dict[str, Any], tmp: Path, out: str) -> tuple[int, int, Path]:
    path = tmp / f"{out}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    common = ["--config", str(path), "--out", str(tmp / out), "--jobs", "1"]
    rc_report = report.main([*common, "--no-figures"])
    rc_packet = packet.main([*common, "--date", "2026-01-01"])
    return rc_report, rc_packet, tmp / out


@pytest.fixture(scope="module")
def packets(
    store: tuple[Path, dict[str, Any]],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("packet")
    a, b = _run(store[1], tmp, "a"), _run(store[1], tmp, "b")
    assert (a[0], b[0]) == (0, 0)
    assert (a[1], b[1]) == (0, 0), "packet over the line budget on the fixture store"
    return a[2], b[2]


def test_packet_is_within_budget_ordered_and_reproducible(packets: tuple[Path, Path]) -> None:
    a, b = packets
    text = (a / "results_packet.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= packet.MAX_LINES
    heads = [m.group(1) for m in re.finditer(r"^## (\d+)\. ", text, flags=re.M)]
    assert heads == [str(i) for i in range(13)]
    assert not text.startswith("SANITY FAILED")
    for name in ("results_packet.md", "results_flat.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), (
            f"{name} differs between two runs"
        )


def test_grading_sheet_is_verbatim_and_verdicts_blank(packets: tuple[Path, Path]) -> None:
    out = packets[0]
    text = (out / "results_packet.md").read_text(encoding="utf-8")
    sheet = (out / "hypotheses.md").read_text(encoding="utf-8")
    assert sheet.rstrip("\n") in text
    verdicts = [line for line in text.splitlines() if line.startswith("**Verdict")]
    assert len(verdicts) == 6
    assert all(re.fullmatch(r"\*\*Verdict \(Laksh\):\*\* _+", v) for v in verdicts)
    assert "no threshold registered" in text  # H2: the packet never invents one


def test_tables_are_rectangular(packets: tuple[Path, Path]) -> None:
    """A literal '|' inside a cell would silently shift every column to its right."""
    text = (packets[0] / "results_packet.md").read_text(encoding="utf-8")
    body, _, rest = text.partition("## 11. ")
    own = body + "## 12. " + rest.partition("## 12. ")[2]  # the verbatim sheet is report.py's
    width = None
    for line in own.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        n = line.count("|")
        assert width in (None, n), line[:120]
        width = n


def test_flat_csv_schema(packets: tuple[Path, Path]) -> None:
    with (packets[0] / "results_flat.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == packet.FLAT_HEADER
    assert {r["seed"] for r in rows} >= {"1", "2", "3", "mean", "std"}
    acc = [r for r in rows if r["metric"] == "greedy_accuracy" and r["seed"] != "std"]
    assert acc and all(r["ci_lo"] and r["ci_hi"] and int(r["n"]) > 0 for r in acc)
    keys = [(r["arm"], r["seed"], r["split"], r["metric"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_stale_grading_sheet_is_refused(
    store: tuple[Path, dict[str, Any]],  # noqa: F811
    tmp_path: Path,
) -> None:
    _, _, out = _run(store[1], tmp_path, "x")
    sheet = out / "hypotheses.md"
    sheet.write_text(
        sheet.read_text(encoding="utf-8").replace("______", "supported", 1), encoding="utf-8"
    )
    cfg = tmp_path / "x.yaml"
    with pytest.raises(SystemExit, match="stale or edited"):
        packet.main(["--config", str(cfg), "--out", str(out), "--jobs", "1"])


def test_sanity_failure_is_the_first_line(
    store: tuple[Path, dict[str, Any]],  # noqa: F811
    tmp_path: Path,
) -> None:
    root, cfg = _copy(store, tmp_path)
    _different_cap(root / "runs")
    rc_report, _, out = _run(cfg, tmp_path, "bad")
    assert rc_report == 1
    lines = (out / "results_packet.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "SANITY FAILED"
    assert any("| FAIL |" in line for line in lines)
    assert sum("DEPENDS ON A FAILED SANITY CHECK" in line for line in lines) >= 9
