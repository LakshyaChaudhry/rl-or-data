"""tasks/07: the compute log — GPU type, GPU-hours and cost per arm, from every meta.json in a run root.

    uv run python scripts/compute_log.py            # -> reports/compute_log.md, reports/compute_log.csv

Read-only on the run root. Every `meta.json` with a `gpu_hours_actual` is counted exactly once and
classified by its path only (nothing here reads an accuracy). Hours are the ones the code metered
around training and generation; an instance is billed for more than that (setup, weight downloads,
model loads between jobs, store syncs, the idle-guard window), and that overhead is not recorded
anywhere in the run root.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ARM_LABELS = {
    "rft_easy": "RFT-Easy", "rft_mixed": "RFT-Mixed", "rft_curated": "RFT-Curated",
    "grpo_easy": "GRPO-Easy", "grpo_mixed": "GRPO-Mixed", "grpo_curated": "GRPO-Curated",
    "iter_rft_curated": "IterRFT-Curated (secondary)",
    "grpo_random_reward": "C1 random reward", "grpo_format_only": "C2 format only",
}  # fmt: skip
COLUMNS = (
    ("train", "training, result-bearing seeds"),
    ("sweep", "training, other sweep configs"),
    ("eval_final", "final evals"),
    ("eval_selection", "val evals (selection / checkpoints / rounds)"),
    ("eval_superseded", "superseded evals (not results)"),
)
OTHER_LABELS = {
    "eval": "base + reference model evals (Phase 1; base gsm8k mean@8 from tasks/06b)",
    "tier": "pass@8 tiering of the pool (Phase 1)",
    "cap_provisional": "provisional cap run (Phase 1)",
    "transfer_pick": "transfer-set pick (Phase 1)",
    "rft/draw_seed1": "RFT base draw, 192 samples per prompt (shared by every RFT arm and seed)",
    "exploratory_cap8704": "EXPLORATORY cap-8,704 re-eval (tasks/06b; never a SPEC §10 number)",
    "rft_noeos_ablation": "RFT runs before the EOS amendment (never results; PREREGISTRATION §4)",
    "grpo/_failed": "failed GRPO attempt: bf16-merge collapse of grpo_mixed_s1 (never a result)",
    "gemma_smoke": "Gemma smoke test (stub-level; Gemma was never evaluated)",
    "dev": "development runs",
}


def _chosen(arm_dir: Path) -> str | None:
    p = arm_dir / "chosen.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text(encoding="utf-8"))
    return f"_lr{float(c['learning_rate']):g}_ep{int(c['epochs'])}"


def classify(rel: str, is_unit: bool, chosen: dict[str, str | None]) -> tuple[str, str] | None:
    """(group, column) for one meta.json path relative to the run root; None = a duplicate to skip."""
    parts = rel.split("/")
    top = parts[0]
    if top == "rft" and parts[1].startswith(("rft_", "iter_rft_")):
        arm = parts[1]
        if "/round_" in rel and not is_unit:
            return None  # iterated RFT: the run-level meta.json already sums its rounds
        if is_unit:
            if "_stale" in rel:
                return arm, "eval_superseded"
            return arm, "eval_final" if "/eval/final/" in rel else "eval_selection"
        suffix = chosen.get(arm)
        return arm, "train" if suffix is None or parts[2].endswith(suffix) else "sweep"
    if top == "grpo" and not parts[1].startswith("_"):
        arm = parts[1].rsplit("_s", 1)[0]
        if not is_unit:
            return arm, "train"
        return arm, "eval_final" if "/eval/final/" in rel else "eval_selection"
    if top in ("rft", "grpo"):
        return f"{top}/{parts[1]}", "other"
    if top.startswith(("_stub", "tier_", "out", "greedy", "mean_at_k", "pass_at_k")):
        return "dev", "other"
    return top, "other"


def collect(
    run_root: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], set[str], list[str]]:
    chosen = {d.name: _chosen(d) for d in (run_root / "rft").glob("*") if d.is_dir()}
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    gpus: set[str] = set()
    unmetered: list[str] = []
    for meta in sorted(run_root.rglob("meta.json")):
        rel = meta.relative_to(run_root).as_posix()
        d: dict[str, Any] = json.loads(meta.read_text(encoding="utf-8"))
        where = classify(rel, (meta.parent / "metrics.json").exists(), chosen)
        if where is None:
            continue
        if d.get("gpu_type"):
            gpus.add(str(d["gpu_type"]))
        if d.get("gpu_hours_actual") is None:
            unmetered.append(rel)
            continue
        hours[where[0]][where[1]] += float(d["gpu_hours_actual"])
        counts[where[0]][where[1]] += 1
    return hours, counts, gpus, unmetered


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis/default.yaml")
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_root = Path(args.run_root or cfg["run_root"])
    billed = float(cfg["gpu_rate"]["billed_usd_per_hour"])
    recorded = float(cfg["gpu_rate"]["recorded_usd_per_hour"])
    hours, counts, gpus, unmetered = collect(run_root)

    rows: list[list[Any]] = []
    lines = [
        "# Compute log (tasks/07)",
        "",
        f"Generated by `scripts/compute_log.py` from every `meta.json` under `{run_root}` (read-only). "
        f"GPU: {', '.join(sorted(gpus)) or 'unknown'}, one per instance (Lambda, us-west-3). **GPU-hours are the "
        "primary quantity**: the hours the code metered around training and generation. Dollars are those "
        f"hours × ${billed:g}/h, the rate Lambda billed (`meta.json` files record ${recorded:g}/h). An instance "
        "is billed for more than the metered hours — setup, weight downloads, model loads between jobs, store "
        "syncs, the 30-minute idle-guard window — and that overhead is recorded nowhere in the run root, so "
        "the invoice is the only source for total spend.",
        "",
        "## Per arm",
        "",
        "| arm | "
        + " | ".join(label for _, label in COLUMNS)
        + f" | total GPU-h | $ at ${billed:g}/h |",
        "|---|" + "---|" * (len(COLUMNS) + 2),
    ]
    arm_total = 0.0
    for arm, label in ARM_LABELS.items():
        if arm not in hours:
            continue
        cells = [hours[arm].get(c, 0.0) for c, _ in COLUMNS]
        n = [counts[arm].get(c, 0) for c, _ in COLUMNS]
        total = sum(cells)
        arm_total += total
        lines.append(
            f"| {label} | "
            + " | ".join(f"{h:.2f} ({k})" if k else "—" for h, k in zip(cells, n, strict=True))
            + f" | {total:.2f} | ${total * billed:,.0f} |"
        )
        rows += [[label, c, n[i], f"{cells[i]:.4f}"] for i, (c, _) in enumerate(COLUMNS) if n[i]]
    lines += [
        f"| **all arms and controls** | | | | | | **{arm_total:.2f}** | **${arm_total * billed:,.0f}** |",
        "",
        "Cell = GPU-hours (number of runs or eval units). RFT: 9 configurations per arm were trained on seed 1 "
        "and one was selected on val_mixed_100; the 8 others are the sweep column. GRPO: one fixed recipe, "
        "nothing swept; its val evals are the step-100/200/300 checkpoints. IterRFT: training includes the "
        "on-policy sampling of rounds 2–3; registered after unblinding (PREREGISTRATION §6).",
        "",
        "## Everything else",
        "",
        "| what | GPU-h (metered runs / units) | $ |",
        "|---|---|---|",
    ]
    other_total = 0.0
    for group in (*OTHER_LABELS, *sorted(set(hours) - set(OTHER_LABELS) - set(ARM_LABELS))):
        if group not in hours:
            continue
        h, k = sum(hours[group].values()), sum(counts[group].values())
        other_total += h
        lines.append(f"| {OTHER_LABELS.get(group, group)} | {h:.2f} ({k}) | ${h * billed:,.0f} |")
        rows.append([OTHER_LABELS.get(group, group), "other", k, f"{h:.4f}"])
    grand = arm_total + other_total
    lines += [
        f"| **everything else** | **{other_total:.2f}** | **${other_total * billed:,.0f}** |",
        "",
        f"**Metered total: {grand:.2f} GPU-hours = ${grand * billed:,.0f} at ${billed:g}/h** "
        f"(${grand * recorded:,.0f} at the ${recorded:g}/h written into the meta.json files). "
        f"{len(unmetered)} meta.json file(s) carry no `gpu_hours_actual` and are not counted: "
        f"{', '.join(unmetered) or '—'}.",
        "",
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compute_log.md").write_text("\n".join(lines), encoding="utf-8")
    with (out / "compute_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["group", "category", "n_meta_files", "gpu_hours"])
        w.writerows(rows)
    print(f"[compute_log] {grand:.2f} GPU-h metered -> {out / 'compute_log.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
