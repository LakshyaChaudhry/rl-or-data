"""SFT hyperparameter sweep for one RFT arm (tasks/03 §4) — the "deflated SFT baseline" guard.

Per arm, seed 1 only: every (learning_rate, epochs) on the locked grid in
``configs/locked/training.yaml`` → train → evaluate the final adapter on the selection split
(``val_mixed_100``, SPEC §10) → ``runs/rft/<arm>/sweep.json`` with every config's accuracy, CI,
truncation and optimizer steps → ``chosen.json`` with the winner. Ties go to fewer epochs, then
to the lower learning rate. Seeds 2 and 3 then reuse the chosen config unchanged
(``make rft-finals``).

This script never reads the held-out splits; ``tests/test_rft_sweep_guard.py`` greps it for
their names and fails if they appear.

    uv run python scripts/rft_sweep.py --config configs/rft/mixed.yaml [--stub --allow-cpu]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from rlordata.cli import main as cli_main
from rlordata.envfile import load_env
from rlordata.train.common import load_arm_config, read_json, write_json
from rlordata.train.rft import run_name
from rlordata.train.rft_eval import val_greedy_accuracy

SWEEP_SEED = 1  # tasks/03 §4: the sweep runs on seed 1 only


def sweep_grid(cfg: dict) -> list[tuple[float, int]]:
    grid = cfg["training"]["rft"]["sweep"]
    return [
        (float(lr), int(ep)) for lr, ep in itertools.product(grid["learning_rate"], grid["epochs"])
    ]


def choose(results: list[dict]) -> dict:
    """Highest selection accuracy; ties → fewer epochs → lower learning rate."""
    finished = [r for r in results if r.get("status") == "finished"]
    if not finished:
        raise SystemExit("no finished sweep runs to choose from")
    return sorted(finished, key=lambda r: (-r["val_accuracy"], r["epochs"], r["learning_rate"]))[0]


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="configs/rft/<arm>.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--base-eval-dir", default=None)
    ap.add_argument("--stub", action="store_true", help="StubSampler for the eval (dry runs only)")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--n-problems", type=int, default=None, help="--stub only")
    ap.add_argument("--only", default=None, help="comma-separated lr:epochs pairs to run (debug)")
    args = ap.parse_args(argv)

    cfg = load_arm_config(args.config)
    arm = cfg["arm"]
    output_dir = Path(args.output_dir or cfg.get("output_dir", "runs/rft"))
    arm_dir = output_dir / arm
    common: list[str] = ["--config", args.config, "--seed", str(SWEEP_SEED)]
    for flag, val in (
        ("--output-dir", args.output_dir),
        ("--splits-dir", args.splits_dir),
        ("--samples-dir", args.samples_dir),
        ("--base-eval-dir", args.base_eval_dir),
    ):
        if val:
            common += [flag, str(val)]
    grid = sweep_grid(cfg)
    if args.only:
        wanted = {(float(a), int(b)) for a, b in (x.split(":") for x in args.only.split(","))}
        grid = [g for g in grid if g in wanted]
    results: list[dict] = []
    for lr, epochs in grid:
        run_dir = arm_dir / run_name(SWEEP_SEED, lr, epochs)
        print(f"\n=== sweep {arm}: lr={lr:g} epochs={epochs} -> {run_dir}", flush=True)
        train_args = [*common, "--stage", "train", "--lr", str(lr), "--epochs", str(epochs)]
        if args.allow_cpu:
            train_args.append("--allow-cpu")
        rc = cli_main(["rft", *train_args])
        if rc != 0:
            results.append(
                {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "run_dir": str(run_dir),
                    "status": "train_failed",
                }
            )
            continue
        eval_args = [*common, "--stage", "eval", "--run-dir", str(run_dir), "--eval-set", "val"]
        if args.stub:
            eval_args.append("--stub")
        if args.n_problems is not None:
            eval_args += ["--n-problems", str(args.n_problems)]
        rc = cli_main(["rft", *eval_args])
        if rc != 0:
            results.append(
                {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "run_dir": str(run_dir),
                    "status": "eval_failed",
                }
            )
            continue
        m = val_greedy_accuracy(run_dir)
        budgets = read_json(run_dir / "budgets.json")
        results.append(
            {
                "learning_rate": lr,
                "epochs": epochs,
                "run_dir": str(run_dir),
                "status": "finished",
                "selection_split": m["split"],
                "val_accuracy": m["accuracy"],
                "ci_low": m["ci_low"],
                "ci_high": m["ci_high"],
                "truncation_rate": m["truncation_rate"],
                "extraction_failure_rate": m["extraction_failure_rate"],
                "answer_line_rate": m.get("answer_line_rate"),
                "optimizer_steps": budgets["optimizer_steps"],
                "training_tokens": budgets["training_tokens"],
                "config_hash": (run_dir / "config_hash.txt").read_text().strip(),
            }
        )
        write_json(
            arm_dir / "sweep.json",
            {"arm": arm, "seed": SWEEP_SEED, "grid": grid, "results": results},
        )
    chosen = choose(results)
    write_json(
        arm_dir / "chosen.json",
        {
            "arm": arm,
            "learning_rate": chosen["learning_rate"],
            "epochs": chosen["epochs"],
            "selected_on": chosen["selection_split"],
            "val_accuracy": chosen["val_accuracy"],
            "tie_break": "fewer epochs, then lower learning rate",
            "sweep_run_dir": chosen["run_dir"],
        },
    )
    print("\n=== sweep results (selection split only) ===")
    for r in sorted(results, key=lambda r: (r["learning_rate"], r["epochs"])):
        if r["status"] != "finished":
            print(f"  lr={r['learning_rate']:g} ep={r['epochs']}: {r['status']}")
            continue
        print(
            f"  lr={r['learning_rate']:g} ep={r['epochs']}: acc {r['val_accuracy']:.3f} "
            f"[{r['ci_low']:.3f},{r['ci_high']:.3f}] trunc {100 * r['truncation_rate']:.1f}% "
            f"steps {r['optimizer_steps']}"
        )
    print(
        f"chosen: lr={chosen['learning_rate']:g} epochs={chosen['epochs']} (acc {chosen['val_accuracy']:.3f})"
    )
    print(json.dumps({"chosen": chosen["run_dir"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
