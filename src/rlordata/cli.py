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
    if args.cmd not in dispatch:
        print(
            f"'{args.cmd}' is not implemented yet — see tasks/ for the task that adds it.",
            file=sys.stderr,
        )
        return 2
    return int(dispatch[args.cmd](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
