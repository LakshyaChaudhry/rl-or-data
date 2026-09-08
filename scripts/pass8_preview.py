"""Preview the pass@8 tier distribution from the provisional cap run, under alternative extraction rules.

    uv run python scripts/pass8_preview.py runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k/samples.jsonl

The provisional run already holds 8 T=1.0 completions for 500 pool problems, so the tier counts that
`make tier` would produce can be previewed here before spending GPU time. Three extraction rules are
compared (the SPEC §5 rule is what the pipeline scores with; the others are for the amendment decision):

  strict   SPEC §5: last line matching ^Answer:\\s*(-?\\d+)\\s*$
  boxed    strict + \\boxed{N}, **Answer:** N, Final Answer: N, and a trailing '.'/',' after the integer
  prose    boxed + 'the (final) answer is N'

Every rule still demands the exact integer and takes the LAST match. Nothing here changes scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rlordata.core.verify import ANSWER_RE  # noqa: E402
from rlordata.data.tiers import tier_from_pass8  # noqa: E402

_INT = r"(-?\d[\d,]*)"
RULES: dict[str, list[re.Pattern[str]]] = {
    "strict": [ANSWER_RE],
    "boxed": [
        re.compile(
            r"^\s*(?:\*\*)?(?:Final )?Answer:?(?:\*\*)?:?\s*\$?\\?(?:boxed\{)?\s*"
            + _INT
            + r"\s*\}?\$?\s*[.,]?\s*$",
            re.M | re.I,
        ),
        re.compile(r"\\boxed\{\s*" + _INT + r"\s*\}"),
    ],
    "prose": [
        re.compile(
            r"^\s*(?:\*\*)?(?:Final )?Answer:?(?:\*\*)?:?\s*\$?\\?(?:boxed\{)?\s*"
            + _INT
            + r"\s*\}?\$?\s*[.,]?\s*$",
            re.M | re.I,
        ),
        re.compile(r"\\boxed\{\s*" + _INT + r"\s*\}"),
        re.compile(r"the (?:final )?answer is:?\s*\$?\\?(?:boxed\{|\()?\s*" + _INT, re.I),
    ],
}


def extract(completion: str, rule: str) -> int | None:
    last: tuple[int, int] | None = None  # (position, value)
    for pat in RULES[rule]:
        for m in pat.finditer(completion):
            raw = next(g for g in m.groups() if g is not None)
            try:
                value = int(raw.replace(",", ""))
            except ValueError:
                continue
            if last is None or m.start() > last[0]:
                last = (m.start(), value)
    return None if last is None else last[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("samples")
    parser.add_argument(
        "--pool-size", type=int, default=6000, help="extrapolate tier counts to this pool size"
    )
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.samples).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems = sorted({r["problem_id"] for r in rows})
    print(f"{len(rows)} samples over {len(problems)} problems")
    for rule in RULES:
        pass8: Counter[str] = Counter({p: 0 for p in problems})
        n_extracted = 0
        n_correct = 0
        for r in rows:
            value = extract(r["completion"], rule) if not r["truncated"] else None
            if rule == "strict":
                assert (value is not None) == (not r["extraction_failed"]) or r["truncated"], (
                    "strict rule must match the stored verdicts"
                )
                value = r["extracted_answer"]
            if value is not None:
                n_extracted += 1
                if value == r["extra"]["answer"]:
                    n_correct += 1
                    pass8[r["problem_id"]] += 1
        hist = Counter(pass8.values())
        tiers = Counter(tier_from_pass8(v) for v in pass8.values())
        scale = args.pool_size / len(problems)
        print(
            f"\n[{rule}] extraction {n_extracted / len(rows):.1%}  mean@8 {n_correct / len(rows):.3f}"
        )
        print("  pass8 histogram: " + "  ".join(f"{k}:{hist.get(k, 0)}" for k in range(9)))
        print(
            "  tiers on these problems: "
            + ", ".join(f"{t} {tiers.get(t, 0)}" for t in ("easy", "medium", "hard"))
            + f"   -> extrapolated to {args.pool_size}: "
            + ", ".join(
                f"{t} ~{round(tiers.get(t, 0) * scale)}" for t in ("easy", "medium", "hard")
            )
        )
    print(
        "\nSPEC §6 needs: easy >= 233 (100 train_easy + 33 train_mixed + 33 val + 100 test)  "
        "medium >= 166  hard >= 168, before structure-disjointness losses."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
