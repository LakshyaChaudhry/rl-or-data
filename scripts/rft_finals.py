"""Final RFT runs for one arm (tasks/03 §§4–6): seeds 2 and 3 with the sweep's chosen config, then
the final eval (val + test + ood + transfer, once) of every seed, then the per-arm report.

Seed 1's final run is the sweep run with the chosen config (same config, same seed → the same run;
training it twice would only add GPU hours). It is evaluated on the final sets here, once.

    uv run python scripts/rft_finals.py --config configs/rft/mixed.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlordata.analysis.rft_report import write_arm_report
from rlordata.cli import main as cli_main
from rlordata.envfile import load_env
from rlordata.train.common import load_arm_config, read_json
from rlordata.train.rft import run_name


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--base-eval-dir", default=None)
    ap.add_argument("--seeds", default=None, help="override training.yaml seeds, e.g. 1,2,3")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--n-problems", type=int, default=None, help="--stub only")
    args = ap.parse_args(argv)

    cfg = load_arm_config(args.config)
    arm = cfg["arm"]
    output_dir = Path(args.output_dir or cfg.get("output_dir", "runs/rft"))
    arm_dir = output_dir / arm
    chosen_path = arm_dir / "chosen.json"
    if not chosen_path.exists():
        raise SystemExit(f"{chosen_path} missing: run the sweep first (scripts/rft_sweep.py)")
    chosen = read_json(chosen_path)
    lr, epochs = float(chosen["learning_rate"]), int(chosen["epochs"])
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["training"]["seeds"])]
    common: list[str] = ["--config", args.config]
    for flag, val in (
        ("--output-dir", args.output_dir),
        ("--splits-dir", args.splits_dir),
        ("--samples-dir", args.samples_dir),
        ("--base-eval-dir", args.base_eval_dir),
    ):
        if val:
            common += [flag, str(val)]
    print(f"=== finals {arm}: chosen lr={lr:g} epochs={epochs}; seeds {seeds}")
    for seed in seeds:
        run_dir = arm_dir / run_name(seed, lr, epochs)
        train_args = [
            *common,
            "--seed",
            str(seed),
            "--stage",
            "train",
            "--lr",
            str(lr),
            "--epochs",
            str(epochs),
        ]
        if args.allow_cpu:
            train_args.append("--allow-cpu")
        if cli_main(["rft", *train_args]) != 0:  # skips itself when budgets.json exists
            raise SystemExit(f"training failed for seed {seed}")
        eval_args = [
            *common,
            "--seed",
            str(seed),
            "--stage",
            "eval",
            "--run-dir",
            str(run_dir),
            "--eval-set",
            "final",
        ]
        if args.stub:
            eval_args.append("--stub")
        if args.n_problems is not None:
            eval_args += ["--n-problems", str(args.n_problems)]
        if cli_main(["rft", *eval_args]) != 0:
            raise SystemExit(f"final eval failed for seed {seed}")
    report = write_arm_report(arm_dir, seeds=seeds, learning_rate=lr, epochs=epochs)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
