"""Show that an analysis change left every earlier number alone (tasks/06b B).

Compares the CSV tables of two analysis snapshots row by row: every row of OLD must reappear in NEW
byte for byte; rows that exist only in NEW are listed by whom they belong to. Exit code 1 when an old
row is missing or changed, unless it is one of the rows named with --expect-changed (a `who,metric`
pair, e.g. the base gsm8k_500 mean@8 unit that went from `missing` to a value).

    uv run python scripts/compare_phase_tables.py results/phase4 results/phase5 \
        --expect-changed base,gsm8k_mean8
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TABLES = ("units.csv", "arms.csv", "per_tier.csv", "budgets.csv", "truncation.csv", "contrasts.csv")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--expect-changed", action="append", default=[], help="who,metric (repeatable)")
    args = ap.parse_args(argv)
    expected = {tuple(x.split(",", 1)) for x in args.expect_changed}
    bad = 0
    for name in TABLES:
        with (args.old / "tables" / name).open(encoding="utf-8") as f:
            old = list(csv.reader(f))
        with (args.new / "tables" / name).open(encoding="utf-8") as f:
            new = list(csv.reader(f))
        new_rows = {tuple(r) for r in new[1:]}
        old_rows = {tuple(r) for r in old[1:]}
        header = old[0]
        gone = [r for r in old[1:] if tuple(r) not in new_rows]
        unexpected = []
        for r in gone:
            row = dict(zip(header, r, strict=True))
            if (row.get("who", r[0]), row.get("metric", "")) not in expected:
                unexpected.append(r)
        added = sorted({r[0] for r in new[1:] if tuple(r) not in old_rows})
        print(
            f"{name}: header identical: {old[0] == new[0]}; {len(old) - 1} old rows, "
            f"{len(gone) - len(unexpected)} changed as expected, {len(unexpected)} changed unexpectedly; "
            f"rows only in new belong to: {', '.join(added) or '—'}"
        )
        for r in unexpected[:10]:
            print("   UNEXPECTED:", ",".join(r)[:200])
        bad += len(unexpected) + (old[0] != new[0])
    print("OK: every earlier number is byte-identical" if not bad else f"FAILED: {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
