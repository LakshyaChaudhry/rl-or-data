"""Rescore stored completions after an extractor or cap change (no GPU, no resampling). AGENT-OWNED.

    uv run python scripts/rescore_samples.py --run-dir runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k
    uv run python scripts/rescore_samples.py --run-dir runs/eval/<model>/<split>/<decoding> --cap-yaml configs/locked/cap.yaml

Re-verifies every completion with the CURRENT ``core.verify`` against the pool's answers (joined by
``problem_id``), optionally simulating a lower cap, and rewrites samples.jsonl + metrics.json with
provenance in meta.json. The first-ever original is kept as ``samples.raw.jsonl``. See
``rlordata.sampling.rescore``. For the tiering samples use ``rlordata tier --rescore-from`` instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rlordata.data.generator import read_jsonl  # noqa: E402
from rlordata.sampling.cap import load_locked_cap  # noqa: E402
from rlordata.sampling.rescore import answers_from_pools, rescore_run_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run-dir", required=True, action="append", help="run directory (repeatable)"
    )
    parser.add_argument(
        "--pool",
        action="append",
        default=None,
        help="pool JSONL(s) for answers (default: data/pool/*.jsonl)",
    )
    parser.add_argument(
        "--cap-yaml", default=None, help="simulate this locked cap on completions longer than it"
    )
    parser.add_argument(
        "--cap", type=int, default=None, help="explicit cap (tests); --cap-yaml wins"
    )
    args = parser.parse_args(argv)
    pools = args.pool or ["data/pool/pool.jsonl", "data/pool/ood_hard_200.jsonl"]
    answers = answers_from_pools([read_jsonl(p) for p in pools if Path(p).exists()])
    cap = load_locked_cap(args.cap_yaml) if args.cap_yaml else args.cap
    if args.cap_yaml and cap is None:
        print(f"REFUSING: {args.cap_yaml} does not exist", file=sys.stderr)
        return 2
    for d in args.run_dir:
        summary = rescore_run_dir(Path(d), answers=answers, cap=cap)
        print(f"{d}:")
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k not in ("raw_sha256",)},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
