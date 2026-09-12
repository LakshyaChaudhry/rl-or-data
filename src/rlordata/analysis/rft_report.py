"""Per-arm RFT report (tasks/03 §6): seeds as rows, μ ± σ across seeds with every seed shown.

No arm-vs-arm comparison lives here (that is Phase 4 / SPEC §10); this only aggregates one arm.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rlordata.train.common import read_json, write_json
from rlordata.train.rft import run_name

COLUMNS = (
    ("val", "val_mixed_100/greedy"),
    ("test", "test_300/greedy"),
    ("ood", "ood_hard_200/greedy"),
)


def _metrics(run_dir: Path, key: str) -> dict[str, Any] | None:
    p = run_dir / "eval" / "final" / key / "metrics.json"
    return read_json(p) if p.exists() else None


def seed_row(run_dir: Path, seed: int) -> dict[str, Any]:
    budgets = read_json(run_dir / "budgets.json")
    meta = read_json(run_dir / "meta.json")
    row: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "config_hash": (run_dir / "config_hash.txt").read_text().strip(),
    }
    for label, key in COLUMNS:
        m = _metrics(run_dir, key)
        if m is None:
            continue
        row[label] = {
            "accuracy": m["accuracy"],
            "ci": [m["ci_low"], m["ci_high"]],
            "truncation_rate": m["truncation_rate"],
            "extraction_failure_rate": m["extraction_failure_rate"],
            "answer_line_rate": m.get("answer_line_rate"),
            "per_tier": m.get("per_tier"),
            "flags": m.get("flags"),
        }
    mk = _metrics(run_dir, "test_300/mean_at_k")
    pk = _metrics(run_dir, "test_300/pass_at_k")
    row["test_mean_at_8"] = None if mk is None else mk["accuracy"]
    row["test_pass_at_8"] = None if pk is None else pk["pass_at_k"].get("8")
    row["test_pass_at_k"] = None if pk is None else pk["pass_at_k"]
    transfer = {}
    for m in sorted((run_dir / "eval" / "final").glob("*/greedy/metrics.json")):
        d = read_json(m)
        if d["split"] not in {k for _, k in COLUMNS} and d["split"] not in (
            "val_mixed_100",
            "test_300",
            "ood_hard_200",
        ):
            transfer[d["split"]] = {
                "accuracy": d["accuracy"],
                "ci": [d["ci_low"], d["ci_high"]],
                "truncation_rate": d["truncation_rate"],
            }
    row["transfer"] = transfer
    row["budgets"] = {
        k: budgets.get(k)
        for k in (
            "prompts",
            "completions_available",
            "completions_consumed",
            "training_tokens",
            "optimizer_steps",
        )
    }
    row["gpu_hours"] = meta.get("gpu_hours_actual")
    row["cost_usd"] = meta.get("cost_usd_actual")
    return row


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mu, sd


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for label, _ in COLUMNS:
        vals = [r[label]["accuracy"] for r in rows if label in r]
        if vals:
            mu, sd = mean_std(vals)
            out[label] = {"mean": mu, "std": sd, "per_seed": vals}
    for key in ("test_mean_at_8", "test_pass_at_8"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            mu, sd = mean_std(vals)
            out[key] = {"mean": mu, "std": sd, "per_seed": vals}
    return out


def format_table(
    arm: str, rows: list[dict[str, Any]], agg: dict[str, Any], *, learning_rate: float, epochs: int
) -> str:
    def cell(r: dict[str, Any], label: str) -> str:
        if label not in r:
            return f"{'—':>21}"
        m = r[label]
        flag = "!" if (m.get("flags") or {}).get("truncation_gt_5pct") else " "
        return f"{m['accuracy']:.3f} [{m['ci'][0]:.3f},{m['ci'][1]:.3f}]{flag}"

    lines = [
        f"# {arm} — chosen lr={learning_rate:g}, epochs={epochs} (selected on val_mixed_100 only)",
        "",
        f"{'seed':>4} {'val greedy [95% CI]':>22} {'test greedy [95% CI]':>22} {'ood greedy [95% CI]':>22} "
        f"{'mean@8':>7} {'pass@8':>7} {'trunc%':>7} {'xfail%':>7} {'prompts':>7} {'avail':>7} {'used':>7} "
        f"{'tokens':>10} {'steps':>6} {'GPU-h':>6} {'$':>7}",
    ]
    for r in rows:
        t = r.get("test", {})
        b = r["budgets"]
        lines.append(
            f"{r['seed']:>4} {cell(r, 'val'):>22} {cell(r, 'test'):>22} {cell(r, 'ood'):>22} "
            f"{(r.get('test_mean_at_8') if r.get('test_mean_at_8') is not None else float('nan')):>7.3f} "
            f"{(r.get('test_pass_at_8') if r.get('test_pass_at_8') is not None else float('nan')):>7.3f} "
            f"{100 * t.get('truncation_rate', float('nan')):>7.1f} {100 * t.get('extraction_failure_rate', float('nan')):>7.1f} "
            f"{b['prompts']:>7} {b['completions_available']:>7} {b['completions_consumed']:>7} "
            f"{b['training_tokens']:>10} {b['optimizer_steps']:>6} {(r['gpu_hours'] or 0):>6.2f} {(r['cost_usd'] or 0):>7.2f}"
        )
    lines.append("")
    for label, _ in COLUMNS:
        if label in agg:
            a = agg[label]
            lines.append(
                f"{label} greedy: {a['mean']:.3f} ± {a['std']:.3f} over {len(a['per_seed'])} seeds {[f'{v:.3f}' for v in a['per_seed']]}"
            )
    for key in ("test_mean_at_8", "test_pass_at_8"):
        if key in agg:
            a = agg[key]
            lines.append(
                f"{key}: {a['mean']:.3f} ± {a['std']:.3f} {[f'{v:.3f}' for v in a['per_seed']]}"
            )
    lines.append("")
    lines.append(
        "'!' = truncation > 5 % on that split (not a headline number). ood truncation is reported, not flagged."
    )
    lines.append("No arm-vs-arm claim here (Phase 4).")
    return "\n".join(lines)


def write_arm_report(arm_dir: Path, *, seeds: list[int], learning_rate: float, epochs: int) -> str:
    arm = arm_dir.name
    rows = []
    for seed in seeds:
        run_dir = arm_dir / run_name(seed, learning_rate, epochs)
        if (run_dir / "budgets.json").exists():
            rows.append(seed_row(run_dir, seed))
    agg = aggregate(rows)
    text = format_table(arm, rows, agg, learning_rate=learning_rate, epochs=epochs)
    write_json(
        arm_dir / "report.json",
        {
            "arm": arm,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "rows": rows,
            "aggregate": agg,
        },
    )
    (arm_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    args = ap.parse_args(argv)
    arm_dir = Path(args.arm_dir)
    chosen = read_json(arm_dir / "chosen.json")
    seeds = sorted(
        {
            int(p.name.split("_")[0][4:])
            for p in arm_dir.glob("seed*_lr*_ep*")
            if (p / "budgets.json").exists()
        }
    )
    print(
        write_arm_report(
            arm_dir,
            seeds=seeds,
            learning_rate=float(chosen["learning_rate"]),
            epochs=int(chosen["epochs"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
