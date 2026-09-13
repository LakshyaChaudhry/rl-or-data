"""tasks/04b §3 — run TRL and Laksh's reference GRPO loop on the tiny 0.6B config; diff the curves.

    uv run python scripts/compare_reference_vs_trl.py [--config configs/grpo/tiny_0p6b.yaml] [--seed 1]
                                                       [--skip-trl] [--skip-reference] [--no-eval]

Writes ``outputs/reference_check.json`` (both training-reward curves, per-step differences over the
first 10 steps, final val greedy accuracy with CI for each when an eval ran) and
``outputs/figures/reference_check.png``. Never a result: the dev config is stamped non-result-bearing.

Contract for Laksh's ``src/rlordata/train/reference_grpo.py`` (his file; this script only calls it):

    run_reference(cfg: dict, *, seed: int, run_dir: Path) -> Path
        Trains on the same resolved config the TRL side uses (``cfg["training"]``, ``cfg["dev_overrides"]``,
        ``cfg["max_completion_tokens"]``, ``cfg["model_id"]``, ``cfg["data_condition"]``), using
        core.completion_logprobs / core.verify / core.group_advantages / core.grpo_loss, beta = 0.
        Must write ``run_dir/train_log.jsonl`` with one row per step ``{"step": int, "reward_mean": float}``
        and save the final PEFT adapter at ``run_dir/adapter/final``. Returns run_dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlordata.envfile import load_env  # noqa: E402
from rlordata.train.common import load_arm_config, read_json, write_json  # noqa: E402


def _curve(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "train_log.jsonl"
    if not path.exists():
        return []
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [
        {"step": int(r["step"]), "reward_mean": r.get("reward_mean")} for r in rows if "step" in r
    ]


def _val_greedy(run_dir: Path) -> dict[str, Any] | None:
    m = run_dir / "eval" / "val" / "val_mixed_100" / "greedy" / "metrics.json"
    if not m.exists():
        return None
    d = read_json(m)
    return {k: d.get(k) for k in ("accuracy", "ci_low", "ci_high", "truncation_rate", "n_problems")}


def _ci_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
    if not a or not b or a.get("ci_low") is None or b.get("ci_low") is None:
        return None
    return (
        a["ci_low"] <= b["accuracy"] <= a["ci_high"]
        and b["ci_low"] <= a["accuracy"] <= b["ci_high"]
    )


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/grpo/tiny_0p6b.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-trl", action="store_true", help="reuse an existing TRL dev run")
    ap.add_argument("--skip-reference", action="store_true", help="reuse an existing reference run")
    ap.add_argument("--no-eval", action="store_true", help="curves only (no vLLM on this machine)")
    ap.add_argument("--out", default="outputs/reference_check.json")
    args = ap.parse_args(argv)

    cfg = load_arm_config(args.config)
    if not cfg.get("dev"):
        raise SystemExit(
            f"{args.config} is not a dev config (dev: true); refusing — this check is never a result"
        )
    root = Path(cfg.get("output_dir", "runs/dev/grpo_tiny"))
    trl_dir = root / f"trl_s{args.seed}"
    ref_dir = root / f"reference_s{args.seed}"

    if not args.skip_trl:
        from rlordata.train.grpo_trl import train_grpo

        rc = train_grpo(cfg, SimpleNamespace(seed=args.seed, run_dir=str(trl_dir), force=True))
        if rc != 0:
            return rc
    if not args.skip_reference:
        try:
            from rlordata.train.reference_grpo import run_reference  # Laksh's file
        except ImportError as e:
            raise SystemExit(
                "src/rlordata/train/reference_grpo.py is not implemented yet (Laksh, tasks/04b). "
                f"Import error: {e}. Run with --skip-reference to produce the TRL half only."
            ) from e
        run_reference(cfg, seed=args.seed, run_dir=ref_dir)

    if not args.no_eval:
        from rlordata.train.grpo_trl import evaluate_grpo_checkpoints

        for d in (trl_dir, ref_dir):
            if (d / "budgets.json").exists() or (d / "adapter" / "final").exists():
                evaluate_grpo_checkpoints(cfg, SimpleNamespace(seed=args.seed, run_dir=str(d)))

    trl_curve, ref_curve = _curve(trl_dir), _curve(ref_dir)
    by_step = {r["step"]: r["reward_mean"] for r in ref_curve}
    first10 = [
        {
            "step": r["step"],
            "trl": r["reward_mean"],
            "reference": by_step.get(r["step"]),
            "diff": (
                None
                if by_step.get(r["step"]) is None or r["reward_mean"] is None
                else r["reward_mean"] - by_step[r["step"]]
            ),
        }
        for r in trl_curve[:10]
    ]
    trl_val, ref_val = _val_greedy(trl_dir), _val_greedy(ref_dir)
    report = {
        "config": args.config,
        "seed": args.seed,
        "result_bearing": False,
        "trl": {"run_dir": str(trl_dir), "curve": trl_curve, "val_greedy": trl_val},
        "reference": {"run_dir": str(ref_dir), "curve": ref_curve, "val_greedy": ref_val},
        "first_10_steps": first10,
        "both_curves_rise": _rises(trl_curve) and _rises(ref_curve),
        "val_within_each_others_ci": _ci_overlap(trl_val, ref_val),
    }
    out = Path(args.out)
    write_json(out, report)
    _plot(report, out.with_suffix("").parent / "figures" / "reference_check.png")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trl", "reference")}, indent=2))
    print(f"[reference-check] wrote {out}")
    return 0


def _rises(curve: list[dict[str, Any]]) -> bool | None:
    vals = [r["reward_mean"] for r in curve if r.get("reward_mean") is not None]
    if len(vals) < 4:
        return None
    k = max(1, len(vals) // 4)
    return sum(vals[-k:]) / k > sum(vals[:k]) / k


def _plot(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in ("trl", "reference"):
        c = report[name]["curve"]
        if c:
            ax.plot([r["step"] for r in c], [r["reward_mean"] for r in c], label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("mean training reward")
    ax.set_title("tasks/04b: TRL vs reference GRPO (0.6B dev, not a result)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
