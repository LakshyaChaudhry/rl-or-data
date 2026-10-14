"""Paper outputs: the exploratory GRPO-RFT contrasts and the paper=True figures, on the synthetic store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rlordata.analysis import exploratory_contrasts, loader, plots, report
from rlordata.core import evaluate
from tests.analysis.store_fixture import build_store


@pytest.fixture(scope="module", autouse=True)
def _fast_core_bootstrap() -> Any:
    real = evaluate.bootstrap_ci

    def quick(
        scores: Any, n_boot: int = 200, alpha: float = 0.05, seed: int = 0
    ) -> tuple[float, float]:
        return real(scores, n_boot=n_boot, alpha=alpha, seed=seed)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(evaluate, "bootstrap_ci", quick)
        yield


@pytest.fixture(scope="module")
def ds(tmp_path_factory: pytest.TempPathFactory) -> loader.Dataset:
    cfg = build_store(tmp_path_factory.mktemp("store"))
    cfg["n_boot"] = 200
    return loader.load_dataset(cfg)


def test_exploratory_contrasts_match_arm_means_and_bracket_the_bounds(ds: loader.Dataset) -> None:
    rows = exploratory_contrasts.exploratory_rows(ds)
    assert set(rows) == {"X_easy_grpo_vs_rft", "X_mixed_grpo_vs_rft"}
    for r in rows.values():
        a = [
            float(u.scores.mean())
            for u in (x.units[("test_300", "greedy")] for x in ds.arms[r["a"]])
        ]
        b = [
            float(u.scores.mean())
            for u in (x.units[("test_300", "greedy")] for x in ds.arms[r["b"]])
        ]
        assert np.allclose(r["test"]["delta_per_seed"], np.subtract(a, b))
        lo, hi = r["bounds"]["test_300_lower"][0], r["bounds"]["test_300_upper"][0]
        assert lo <= r["test"]["mean_delta"] + 1e-12 <= hi + 2e-12
    md = exploratory_contrasts.markdown(rows, "abc123")
    assert "NOT pre-registered" in md and md.count("| test_300 |") == 2


def test_paper_figures_render_without_captions_or_type3_fonts(
    ds: loader.Dataset, tmp_path: Path
) -> None:
    results = report.build_results(ds)
    rc = {**plots.RC, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(rc):
        plots.fig_per_tier(ds, results, tmp_path / "tier.pdf", paper=True)
        plots.fig_zero_std(ds, tmp_path / "zero.pdf", paper=True)
        plots.fig_pass_at_k(ds, results, tmp_path / "passk.pdf", paper=True)
        plots.fig_eval_length(ds, results, tmp_path / "length.pdf", paper=True)
    for name in ("tier", "zero", "passk", "length"):
        pdf = (tmp_path / f"{name}.pdf").read_bytes()
        assert b"/Type3" not in pdf, name
        assert (tmp_path / f"{name}.png").exists(), name
        assert b"SPEC" not in pdf, f"{name}: protocol wording leaked into a paper figure"
