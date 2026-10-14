"""Paper versions of development figures (c) per-tier, (f) zero-advantage, (d) pass@k, (i) eval length.

Same functions, data and layout as ``plots.py`` (``paper=True``): no burned-in caption, no
truncation-flag hollows, legend above the panels, per-row y-scaling, vector PDF + PNG preview.

    uv run python -m rlordata.analysis.paper_figures --run-root runs/store_mirror/runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from rlordata.analysis import plots, report
from rlordata.analysis.loader import load_config, load_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/analysis/default.yaml"))
    ap.add_argument("--run-root", type=Path, default=Path("runs/store_mirror/runs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/figures/paper"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/.cache/units"))
    args = ap.parse_args(argv)
    ds = load_dataset(load_config(args.config, run_root=args.run_root), cache_dir=args.cache)
    results = report.build_results(ds)
    with plt.rc_context(
        {**plots.RC, "pdf.fonttype": 42, "ps.fonttype": 42}
    ):  # TrueType, not Type 3
        plots.fig_per_tier(ds, results, args.out / "fig_per_tier.pdf", paper=True)
        plots.fig_zero_std(ds, args.out / "fig_zero_std.pdf", paper=True)
        plots.fig_pass_at_k(ds, results, args.out / "fig_pass_at_k.pdf", paper=True)
        plots.fig_eval_length(ds, results, args.out / "fig_eval_length.pdf", paper=True)
    for name in ("per_tier", "zero_std", "pass_at_k", "eval_length"):
        print(args.out / f"fig_{name}.pdf")


if __name__ == "__main__":
    main()
