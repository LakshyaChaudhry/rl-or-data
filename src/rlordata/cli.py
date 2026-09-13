"""Command-line entry point. Subcommands are implemented by tasks in tasks/.

rlordata gen  --config configs/data/pool.yaml       # tasks/01
rlordata tier --config configs/data/tiering.yaml    # tasks/01 + 02a (live pass@8; --stub for dry runs)
rlordata eval --config configs/eval/base.yaml       # tasks/02 + 02a (--stub is the local acceptance test)
rlordata rft  --config configs/rft/all.yaml         # tasks/03
rlordata grpo --config configs/grpo/mixed100.yaml   # tasks/04
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlordata")
    sub = parser.add_subparsers(dest="cmd", required=True)
    parsers: dict[str, argparse.ArgumentParser] = {}
    for name in ("gen", "tier", "eval", "rft", "grpo"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--seed", type=int, default=None, help="override config seed")
        p.add_argument("--dry-run", action="store_true")
        parsers[name] = p
    for name in ("tier", "eval"):
        parsers[name].add_argument(
            "--stub",
            action="store_true",
            help="scripted StubSampler: no GPU, no weights, never a result",
        )
        parsers[name].add_argument("--output-dir", default=None, help="override config output_dir")
    parsers["eval"].add_argument(
        "--force", action="store_true", help="re-run units whose samples exist (incl. test_300)"
    )
    parsers["eval"].add_argument(
        "--n-problems", type=int, default=None, help="first N problems per split (dry runs)"
    )
    parsers["eval"].add_argument("--splits-dir", default=None, help="override config splits_dir")
    parsers["eval"].add_argument("--pool", default=None, help="override config pool path")
    parsers["eval"].add_argument(
        "--models",
        default=None,
        help="comma-separated model ids to run (others skipped); e.g. one model per process",
    )
    rft = parsers["rft"]
    rft.add_argument(
        "--stage",
        choices=["draw", "select", "train", "eval"],
        default="train",
        help="tasks/03: draw (192/prompt, once) | select (print stats) | train | eval",
    )
    rft.add_argument("--lr", type=float, default=None, help="train: learning rate (sweep grid)")
    rft.add_argument("--epochs", type=int, default=None, help="train: epochs (sweep grid)")
    rft.add_argument(
        "--run-dir", default=None, help="train: override run dir; eval: run to evaluate"
    )
    rft.add_argument("--eval-set", choices=["val", "final"], default="val")
    rft.add_argument("--output-dir", default=None, help="override config output_dir (runs/rft)")
    rft.add_argument("--splits-dir", default=None)
    rft.add_argument("--samples-dir", default=None, help="where the draw files live (data/samples)")
    rft.add_argument(
        "--base-eval-dir", default=None, help="base-model eval dir for the sanity gate"
    )
    rft.add_argument("--micro-batch-size", type=int, default=None)
    rft.add_argument("--force", action="store_true", help="retrain / re-evaluate finished units")
    rft.add_argument("--stub", action="store_true", help="StubSampler for draw/eval dry runs")
    rft.add_argument(
        "--n-total", type=int, default=None, help="draw: samples per prompt (--stub only)"
    )
    rft.add_argument(
        "--n-problems", type=int, default=None, help="eval: first N problems (--stub only)"
    )
    rft.add_argument(
        "--allow-cpu", action="store_true", help="train without CUDA (smoke tests only)"
    )
    grpo = parsers["grpo"]
    grpo.add_argument(
        "--stage",
        choices=["train", "eval"],
        default="train",
        help="tasks/04: train (TRL GRPOTrainer) | eval (checkpoint val + final test/ood)",
    )
    grpo.add_argument(
        "--run-dir", default=None, help="train: override run dir; eval: run to evaluate"
    )
    grpo.add_argument("--output-dir", default=None, help="override config output_dir (runs/grpo)")
    grpo.add_argument("--splits-dir", default=None)
    grpo.add_argument(
        "--base-eval-dir", default=None, help="base-model eval dir for the sanity gate"
    )
    grpo.add_argument("--force", action="store_true", help="retrain / re-evaluate finished units")
    grpo.add_argument(
        "--resume", action="store_true", help="resume GRPO from the latest adapter checkpoint"
    )
    parsers["tier"].add_argument(
        "--samples-output", default=None, help="override config samples_output"
    )
    parsers["tier"].add_argument("--run-dir", default=None, help="override config run_dir")
    parsers["tier"].add_argument(
        "--provisional-cap",
        type=int,
        default=None,
        help="sample before cap.yaml exists (splits not final; rescore later)",
    )
    parsers["tier"].add_argument(
        "--rescore-from",
        default=None,
        help="rebuild pass8 + splits from stored completions with the current verifier and the locked cap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Dispatch is filled in by the agent as each task lands. Keep this file thin.
    dispatch: dict = {}
    try:
        from rlordata.data import generator

        dispatch["gen"] = generator.cli_main
    except (ImportError, AttributeError):
        pass
    try:
        from rlordata.data import tiers

        dispatch["tier"] = tiers.cli_main
    except (ImportError, AttributeError):
        pass
    try:
        from rlordata.sampling import eval_runner

        dispatch["eval"] = eval_runner.cli_main
    except (ImportError, AttributeError):
        pass
    try:
        from rlordata.train import rft

        dispatch["rft"] = rft.cli_main
    except (ImportError, AttributeError):
        pass
    try:
        from rlordata.train import grpo_trl

        dispatch["grpo"] = grpo_trl.cli_main
    except (ImportError, AttributeError):
        pass
    if args.cmd not in dispatch:
        print(
            f"'{args.cmd}' is not implemented yet — see tasks/ for the task that adds it.",
            file=sys.stderr,
        )
        return 2
    return int(dispatch[args.cmd](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
