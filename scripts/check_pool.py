"""Regenerate the pool and ood set from their configs and diff against the committed JSONL files.

    uv run python scripts/check_pool.py [--pool-config configs/data/pool.yaml] [--ood-config configs/data/ood.yaml]

The committed ``data/pool/*.jsonl`` are the reviewed protocol data (SPEC §6). This check, run on every new
machine by ``setup/setup_gpu.sh``, proves the generator reproduces them byte-for-byte there (numpy
``Generator`` streams are not guaranteed stable across numpy versions). Exit 0 = identical; 1 = drift
(the committed files stay the source of truth — do not regenerate over them); 2 = files missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from rlordata.data.generator import generate_pool, meta_path_for, read_jsonl  # noqa: E402


def check(config_path: Path) -> tuple[bool, str]:
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config.get("output", "data/pool/pool.jsonl"))
    if not out.exists():
        return False, f"{out}: MISSING (commit it or run `make gen` / `make gen-ood`)"
    committed = read_jsonl(out)
    regenerated = generate_pool(config, seed=int(config.get("seed", 0)))
    meta = meta_path_for(out)
    meta_hash = json.loads(meta.read_text())["config_hash"][:12] if meta.exists() else "no meta"
    if len(committed) != len(regenerated):
        return (
            False,
            f"{out}: {len(committed)} committed vs {len(regenerated)} regenerated problems",
        )
    diffs = [
        i
        for i, (a, b) in enumerate(zip(committed, regenerated, strict=True))
        if (a.problem_id, a.text, a.answer) != (b.problem_id, b.text, b.answer)
    ]
    if diffs:
        return False, (
            f"{out}: {len(diffs)} problem(s) differ from a fresh regeneration (first index {diffs[0]}); "
            f"numpy RNG drift? Keep the committed file; do not regenerate."
        )
    return (
        True,
        f"{out}: identical to regeneration ({len(committed)} problems, config hash {meta_hash})",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool-config", default="configs/data/pool.yaml")
    parser.add_argument("--ood-config", default="configs/data/ood.yaml")
    args = parser.parse_args(argv)
    rc = 0
    for cfg in (Path(args.pool_config), Path(args.ood_config)):
        ok, msg = check(cfg)
        print(("OK   " if ok else "FAIL ") + msg)
        if not ok:
            rc = 2 if "MISSING" in msg else max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
