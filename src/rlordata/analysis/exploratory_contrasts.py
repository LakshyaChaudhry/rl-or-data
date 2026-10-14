"""Exploratory GRPO - RFT contrasts on the Easy and Mixed prompt sets. NOT pre-registered.

These are the two pairs matched on both prompts and rollouts (19,200 vs 19,200). They were defined on
2026-09-27, after the test results were seen, for the write-up. They reuse the registered machinery
(``report.contrast_block``, the SPEC §10 criterion, and the truncation-bound scoring of ``appendix``) so
the numbers are computed exactly like the registered contrasts, but they never enter ``contrasts.csv``
or the grading sheet.

    uv run python -m rlordata.analysis.exploratory_contrasts --run-root runs/store_mirror/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlordata.analysis import appendix, report, stats
from rlordata.analysis.loader import Dataset, analysis_config_hash, load_config, load_dataset

SPECS = (
    report.ContrastSpec("X_easy_grpo_vs_rft", "exploratory", "grpo_easy", "rft_easy", False),
    report.ContrastSpec("X_mixed_grpo_vs_rft", "exploratory", "grpo_mixed", "rft_mixed", False),
)
TEST_KEYS = (
    "delta_per_seed", "mean_delta", "std_delta", "ci_low", "ci_high", "pooled_seed_std",
    "truncation_a_per_seed", "truncation_b_per_seed", "flagged_seeds_a", "flagged_seeds_b",
)  # fmt: skip
OOD_KEYS = (
    "delta_per_seed", "mean_delta", "std_delta", "ci_low", "ci_high",
    "truncation_a_per_seed", "truncation_b_per_seed",
)  # fmt: skip


def _contrast(block: dict[str, Any]) -> stats.Contrast:
    return stats.Contrast(**{k: block[k] for k in stats.Contrast.__dataclass_fields__})


def exploratory_rows(ds: Dataset) -> dict[str, Any]:
    """{contrast key: test / ood blocks, the SPEC §10 criterion, and truncation-bound corners}."""
    n_boot, seed = int(ds.cfg["n_boot"]), int(ds.cfg["seed"])
    out: dict[str, Any] = {}
    for sp in SPECS:
        t = report.contrast_block(ds, sp, "test_greedy", n_boot=n_boot, seed=seed)
        o = report.contrast_block(ds, sp, "ood_greedy", n_boot=n_boot, seed=seed)
        assert t is not None and o is not None, f"{sp.key}: missing units"
        crit = stats.evaluate_criterion(_contrast(t), _contrast(o))
        bounds: dict[str, list[float]] = {}  # same corners as appendix.bounds_rows
        for split in ("test_300", "ood_hard_200"):
            ua = appendix._arm_units(ds, sp.a, split)
            ub = appendix._arm_units(ds, sp.b, split)
            for name, (up_a, up_b) in {"lower": (False, True), "upper": (True, False)}.items():
                c = stats.paired_contrast(
                    [r.seed for r in ds.arms[sp.a]],
                    [appendix._scores(u, up_a) for u in ua],
                    [appendix._scores(u, up_b) for u in ub],
                    n_boot=n_boot,
                    seed=seed,
                )
                bounds[f"{split}_{name}"] = [c.mean_delta, c.ci_low, c.ci_high]
        out[sp.key] = {
            "a": sp.a,
            "b": sp.b,
            "test": {k: t[k] for k in TEST_KEYS},
            "ood": {k: o[k] for k in OOD_KEYS},
            "criterion": {
                "threshold_2x_pooled_std": crit.threshold,
                "magnitude_ok": crit.magnitude_ok,
                "same_sign_on_ood": crit.same_sign_on_ood,
                "met": crit.met,
            },
            "bounds": bounds,
        }
    return out


def markdown(rows: dict[str, Any], cfg_hash: str) -> str:
    lines = [
        "# Exploratory GRPO − RFT contrasts on the Easy and Mixed prompt sets (NOT pre-registered)",
        "",
        f"Defined 2026-09-27, after the test results were seen. Analysis config hash `{cfg_hash}`; "
        "Δ = A − B paired seed-wise; CI = paired per-problem bootstrap; criterion = SPEC §10, evaluated "
        "mechanically. Bounds: lower = B's truncated completions all right and A's all wrong; upper = the reverse.",
        "",
        "| contrast | split | Δ per seed | mean Δ [95 % CI] | 2 × pooled seed sd | bounds [lower, upper] |",
        "|---|---|---|---|---|---|",
    ]
    for key, r in rows.items():
        for split, blk, b in (
            ("test_300", r["test"], "test_300"),
            ("ood_hard_200", r["ood"], "ood_hard_200"),
        ):
            thr = f"{r['criterion']['threshold_2x_pooled_std']:.3f}" if split == "test_300" else ""
            lines.append(
                f"| {r['a']} − {r['b']} | {split} | "
                + " / ".join(f"{d:+.3f}" for d in blk["delta_per_seed"])
                + f" | {blk['mean_delta']:+.3f} [{blk['ci_low']:+.3f}, {blk['ci_high']:+.3f}] | {thr} | "
                + f"[{r['bounds'][b + '_lower'][0]:+.3f}, {r['bounds'][b + '_upper'][0]:+.3f}] |"
            )
        lines.append(f"| {key} | criterion | | met = {r['criterion']['met']} | | |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/analysis/default.yaml"))
    ap.add_argument("--run-root", type=Path, default=Path("runs/store_mirror/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/phase5/appendix"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/.cache/units"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, run_root=args.run_root)
    ds = load_dataset(cfg, cache_dir=args.cache)
    rows = exploratory_rows(ds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exploratory_grpo_vs_rft.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.out / "exploratory_grpo_vs_rft.md").write_text(
        markdown(rows, analysis_config_hash(cfg)[:12])
    )
    print(args.out / "exploratory_grpo_vs_rft.md")


if __name__ == "__main__":
    main()
