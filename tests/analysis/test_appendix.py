"""tasks/06b B/C: secondary contrasts, truncation bounds, and the exploratory reader's fences."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rlordata.analysis import appendix, loader, report
from rlordata.train.rft import run_name
from tests.analysis.store_fixture import CHOSEN
from tests.analysis.test_report import (  # noqa: F401  (fixtures)
    _copy,
    _fast_core_bootstrap,
    outputs,
    store,
)
from tests.helpers import REPO


def test_fixture_without_the_arm_has_no_secondary_contrast(
    store: tuple[Path, dict[str, Any]],  # noqa: F811
) -> None:
    ds = loader.load_dataset(store[1])
    assert report.active_contrasts(ds) == report.CONTRASTS
    assert "Secondary" not in report.hypotheses_markdown(ds, report.build_results(ds), "0" * 64)


def test_bounds_bracket_the_observed_delta_and_match_the_contrast_table(
    outputs: tuple[Path, Path],  # noqa: F811
) -> None:
    out = outputs[0]
    with (out / "appendix" / "truncation_bounds.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (out / "tables" / "contrasts.csv").open(encoding="utf-8") as f:
        table = {(r["contrast"], r["metric"]): float(r["mean_delta"]) for r in csv.DictReader(f)}
    names = [s[0] for s in appendix.SCENARIOS]
    keys = {(r["contrast"], r["split"]) for r in rows}
    assert keys and len(rows) == 4 * len(keys)
    for contrast, split in keys:
        d = {
            r["scenario"]: float(r["mean_delta"])
            for r in rows
            if (r["contrast"], r["split"]) == (contrast, split)
        }
        obs, lo, hi, both = (d[n] for n in names)
        assert lo <= obs <= hi and lo <= both <= hi
        metric = "test_greedy" if split == "test_300" else "ood_greedy"
        assert obs == pytest.approx(table[(contrast, metric)], abs=1e-12)
    text = (out / "appendix" / "truncation_bounds.md").read_text(encoding="utf-8")
    assert "Post hoc" in text and "uninformative" in text
    # never on a criterion line or in the results tables
    assert "bound" not in (out / "tables" / "results.md").read_text(encoding="utf-8").lower()
    assert not (
        out / "appendix" / "exploratory_cap.md"
    ).exists()  # fixture config names no such tree


def test_secondary_arm_adds_a_section_and_leaves_the_confirmatory_sheet_alone(
    store: tuple[Path, dict[str, Any]],  # noqa: F811
    tmp_path: Path,
) -> None:
    root, cfg = _copy(store, tmp_path)
    before_ds = loader.load_dataset(cfg)
    before = report.hypotheses_markdown(before_ds, report.build_results(before_ds), "0" * 64)
    lr, ep = CHOSEN["rft_curated"]
    for s in cfg["seeds"]:  # a stand-in arm: byte copies of RFT-Curated, so S1 must be exactly 0
        shutil.copytree(
            root / "runs" / "rft" / "rft_curated" / run_name(s, lr, ep),
            root / "runs" / "rft" / "iter_rft_curated" / f"seed{s}",
        )
    cfg["arms"]["iter_rft_curated"] = {
        "label": "IterRFT-Curated", "method": "iter_rft", "data_condition": "train_curated",
        "run_pattern": "rft/iter_rft_curated/seed{seed}", "secondary": True,
    }  # fmt: skip
    ds = loader.load_dataset(cfg)
    assert [c.key for c in report.active_contrasts(ds)][-2:] == [
        "S1_iter_vs_rft",
        "S2_grpo_vs_iter",
    ]
    results = report.build_results(ds)
    s1, s2, h3 = (
        results["contrasts"][k]["metrics"]["test_greedy"]["mean_delta"]
        for k in ("S1_iter_vs_rft", "S2_grpo_vs_iter", "H3")
    )
    assert s1 == 0.0 and s2 + s1 == pytest.approx(h3, abs=1e-12)  # H3 = (ii) + (i) exactly
    sheet = report.hypotheses_markdown(ds, results, "0" * 64)
    head, _, tail = sheet.partition("## Secondary")
    assert head.rstrip() == before.rstrip(), "the confirmatory sections changed"
    assert "registered after unblinding" in tail and "omits sampling variance" in tail
    assert sheet.count("**Verdict (Laksh):** ___") == 7 and tail.count("**Verdict") == 1
    wrong = report.wrong_markdown(ds, results, "0" * 64)
    assert "ran no sweep" in wrong and "Registered after unblinding" in wrong
    assert [r["contrast"] for r in appendix.bounds_rows(ds)].count("S1_iter_vs_rft") == 8


def test_results_loader_refuses_the_exploratory_tree() -> None:
    cfg = loader.load_config(REPO / "configs" / "analysis" / "default.yaml")
    run_root = Path("/somewhere/runs")
    with pytest.raises(loader.ExcludedPathError):
        loader.guard(
            run_root / "exploratory_cap8704" / "base" / "test_300" / "greedy", run_root, cfg
        )
    assert cfg["exploratory"]["cap_multiple"] == 2 and cfg["known_missing"] == []
    assert cfg["arms"]["iter_rft_curated"]["secondary"] is True


@pytest.mark.parametrize(
    "metrics, config",
    [
        ({"max_completion_tokens": 8704}, {"exploratory": True, "cap_deviation": "x", "max_completion_tokens": 8704}),
        ({"exploratory": True, "cap_deviation": "x", "max_completion_tokens": 9000},
         {"exploratory": True, "cap_deviation": "x", "max_completion_tokens": 9000}),
    ],
)  # fmt: skip
def test_exploratory_reader_refuses_unstamped_or_off_rule_units(
    tmp_path: Path, metrics: dict[str, Any], config: dict[str, Any]
) -> None:
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "samples.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        appendix._read_exploratory_unit(tmp_path, 4352, 2)
