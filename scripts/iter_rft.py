"""Iterated RFT end to end for the configured seeds (tasks/06b). AGENT-OWNED.

Per seed: round 1 train → val eval → [sample r → train r → val eval] for r = 2..R → finalize →
final eval (val + the held-out sets + transfer, once). Every stage is idempotent (finished work is
skipped), so a killed queue job simply resumes. Each stage runs in its own process by default so a
vLLM engine and a trainer never share a CUDA context; ``--in-process`` is for dry runs and tests.

    uv run python scripts/iter_rft.py --config configs/rft/iter_curated.yaml --seeds 1,2,3

This script decides nothing from any evaluation: the final adapter is the last round's, always.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rlordata.cli import main as cli_main
from rlordata.envfile import load_env
from rlordata.train.common import load_arm_config
from rlordata.train.iter_rft import round_dir, seed_dir


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rft/iter_curated.yaml")
    ap.add_argument("--seeds", default=None, help="e.g. 1,2,3 (default: training.yaml seeds)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--base-eval-dir", default=None)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--n-problems", type=int, default=None, help="--stub only")
    ap.add_argument("--in-process", action="store_true", help="dry runs/tests: no subprocesses")
    ap.add_argument("--skip-final-eval", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_arm_config(args.config)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["training"]["seeds"])]
    rounds = int(cfg["rounds"])
    output_dir = Path(args.output_dir or cfg.get("output_dir", "runs/rft"))
    shared: list[str] = ["--config", args.config]
    for flag, val in (("--output-dir", args.output_dir), ("--splits-dir", args.splits_dir)):
        if val:
            shared += [flag, str(val)]

    def run(cmd: list[str]) -> None:
        print(f"=== rlordata {' '.join(cmd)}", flush=True)
        if args.in_process:
            rc = cli_main(cmd)
        else:
            rc = subprocess.run(
                [sys.executable, "-m", "rlordata.cli", *cmd], check=False
            ).returncode
        if rc != 0:
            raise SystemExit(f"stage failed (exit {rc}): {' '.join(cmd)}")

    def iter_stage(seed: int, stage: str, r: int | None = None) -> None:
        cmd = ["iter-rft", *shared, "--seed", str(seed), "--stage", stage]
        if r is not None:
            cmd += ["--round", str(r)]
        if args.samples_dir:
            cmd += ["--samples-dir", str(args.samples_dir)]
        if args.stub:
            cmd.append("--stub")
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        run(cmd)

    def evaluate(seed: int, run_dir: Path, eval_set: str) -> None:
        cmd = ["rft", *shared, "--seed", str(seed), "--stage", "eval", "--run-dir", str(run_dir),
               "--eval-set", eval_set]  # fmt: skip
        if args.base_eval_dir:
            cmd += ["--base-eval-dir", str(args.base_eval_dir)]
        if args.stub:
            cmd.append("--stub")
        if args.n_problems is not None:
            cmd += ["--n-problems", str(args.n_problems)]
        run(cmd)

    for seed in seeds:
        sdir = seed_dir(output_dir, cfg["arm"], seed)
        for r in range(1, rounds + 1):
            if r > 1:
                iter_stage(seed, "sample", r)
            iter_stage(seed, "train", r)
            evaluate(seed, round_dir(sdir, r), "val")
        iter_stage(seed, "finalize")
        if not args.skip_final_eval:
            evaluate(seed, sdir, "final")
    print(f"[iter-rft] done: seeds {seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
