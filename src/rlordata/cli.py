"""Command-line entry point. Subcommands are implemented by tasks in tasks/.

rlordata gen  --config configs/data/pool.yaml       # tasks/01
rlordata tier --config configs/data/tiering.yaml    # tasks/01 (needs GPU)
rlordata eval --config configs/eval/base.yaml       # tasks/02
rlordata rft  --config configs/rft/all.yaml         # tasks/03
rlordata grpo --config configs/grpo/mixed100.yaml   # tasks/04
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rlordata")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "tier", "eval", "rft", "grpo"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--seed", type=int, default=None, help="override config seed")
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

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
    if args.cmd not in dispatch:
        print(
            f"'{args.cmd}' is not implemented yet — see tasks/ for the task that adds it.",
            file=sys.stderr,
        )
        return 2
    return int(dispatch[args.cmd](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
