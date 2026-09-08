"""Classify extraction failures in a samples.jsonl so a high failure rate can be understood before tiering.

    uv run python scripts/inspect_extraction.py runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k/samples.jsonl

Buckets (first match wins, in this order):
  truncated              hit the cap; no answer line possible
  answer_trailing_text   an 'Answer:' line exists but has trailing text ('Answer: 16.', 'Answer: 16 numbers')
  answer_non_integer     an 'Answer:' line exists but the value is not a plain integer ('Answer: **16**', '$16$', 'sixteen')
  answer_case_or_markdown a variant such as 'answer:', '**Answer:**', 'Final answer:' but no exact line
  answer_in_prose        'the answer is N' phrasing, no 'Answer:' line
  no_answer_mention      nothing resembling an answer
The SPEC §5 regex is deliberately strict (leniency becomes reward hacking); this script does not
change scoring, it only explains it. Prints counts and a few tail excerpts per bucket.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rlordata.core.verify import ANSWER_RE  # noqa: E402

_TRAILING = re.compile(r"^Answer:\s*(-?\d+)\S.*$|^Answer:\s*(-?\d+)\s+\S.*$", re.MULTILINE)
_NONINT = re.compile(r"^Answer:\s*(?!-?\d+\s*$).+$", re.MULTILINE)
_VARIANT = re.compile(
    r"(?im)^\s*(\*\*answer\*\*|\*\*answer:\*\*|answer\s*:|final answer\s*:|answer\s*=)"
)
_PROSE = re.compile(r"(?i)\bthe (final )?answer is\b")


def bucket(completion: str, truncated: bool) -> str:
    if ANSWER_RE.search(completion):
        return "parsed"  # should not happen for extraction_failed samples
    if truncated:
        return "truncated"
    if _TRAILING.search(completion):
        return "answer_trailing_text"
    if _NONINT.search(completion):
        return "answer_non_integer"
    if _VARIANT.search(completion):
        return "answer_case_or_markdown"
    if _PROSE.search(completion):
        return "answer_in_prose"
    return "no_answer_mention"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("samples", help="samples.jsonl")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tail", type=int, default=240, help="chars of the completion tail to show"
    )
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.samples).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = [r for r in rows if r["extraction_failed"]]
    print(
        f"{len(rows)} samples, {len(failed)} extraction failures ({len(failed) / max(1, len(rows)):.1%})"
    )
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    for r in failed:
        b = bucket(r["completion"], bool(r["truncated"]))
        counts[b] += 1
        examples[b].append(r)
    for b, n in counts.most_common():
        print(f"  {b:<24} {n:>6}  ({n / len(failed):.1%} of failures)")
    rng = random.Random(args.seed)
    for b, _ in counts.most_common():
        print(f"\n=== {b} — {min(args.examples, len(examples[b]))} example tail(s) ===")
        for r in rng.sample(examples[b], min(args.examples, len(examples[b]))):
            tail = r["completion"][-args.tail :].replace("\n", "\\n")
            print(f"  [answer={r['extra'].get('answer')} n_tokens={r['n_tokens']}] ...{tail}")
    # How many failures contain the right integer anywhere (format-only miss vs genuinely wrong)?
    right_somewhere = sum(
        1
        for r in failed
        if re.search(rf"(?<!\d)-?{re.escape(str(r['extra'].get('answer')))}(?!\d)", r["completion"])
    )
    print(
        f"\nfailures whose text contains the correct integer somewhere: {right_somewhere} ({right_somewhere / max(1, len(failed)):.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
