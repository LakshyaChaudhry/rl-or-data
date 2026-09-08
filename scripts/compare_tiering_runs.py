"""Cross-check two tiering sample files sampled at different caps (determinism audit). AGENT-OWNED.

    uv run python scripts/compare_tiering_runs.py \\
        --reference runs/tier/tier_Qwen__Qwen3-4B-Base_k8_seed1_provisional/tiering_pass8.jsonl \\
        --candidate data/samples/tiering_pass8.jsonl

Sampling is batch-invariant and seeded per sample, so a completion's first ``cap`` tokens must not
depend on ``max_tokens``. Given the same problems, k, T, top_p and seed, sampled once at a lower
(provisional) cap and once at the locked cap, this script joins samples by
``(data_condition, problem_id, extra.sample_idx)`` and checks:

- reference completions that did NOT hit the reference cap are bitwise identical in the candidate;
- reference completions that DID hit the cap are a prefix of the candidate's completion;
- reports per-problem pass8 changes (only ever caused by continued completions).

Exit status 1 on any mismatch. Nothing here changes scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                key = (s["data_condition"], s["problem_id"], int(s["extra"]["sample_idx"]))
                assert key not in out, f"duplicate sample key {key}"
                out[key] = s
    return out


def compare(
    reference: dict[tuple[str, str, int], dict[str, Any]],
    candidate: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Return a summary dict; ``summary['ok']`` is False on any divergence."""
    ref_keys, cand_keys = set(reference), set(candidate)
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    ref_cap = max(int(s["n_tokens"]) for s in reference.values()) if reference else 0
    n_identical = n_prefix = 0
    mismatches: list[dict[str, Any]] = []
    pass8_ref: Counter[tuple[str, str]] = Counter()
    pass8_cand: Counter[tuple[str, str]] = Counter()
    for key in sorted(ref_keys & cand_keys):
        r, c = reference[key], candidate[key]
        pass8_ref[key[:2]] += int(r["correct"])
        pass8_cand[key[:2]] += int(c["correct"])
        hit_cap = bool(r["truncated"]) or int(r["n_tokens"]) >= ref_cap
        if not hit_cap:
            if c["completion"] == r["completion"] and int(c["n_tokens"]) == int(r["n_tokens"]):
                n_identical += 1
            else:
                mismatches.append(
                    {
                        "key": key,
                        "kind": "not_identical",
                        "ref_n": r["n_tokens"],
                        "cand_n": c["n_tokens"],
                    }
                )
        elif c["completion"].startswith(r["completion"]) and int(c["n_tokens"]) >= int(
            r["n_tokens"]
        ):
            n_prefix += 1
        else:
            mismatches.append(
                {"key": key, "kind": "not_prefix", "ref_n": r["n_tokens"], "cand_n": c["n_tokens"]}
            )
    changed = {
        f"{k[0]}:{k[1][:12]}": (pass8_ref[k], pass8_cand[k])
        for k in pass8_ref
        if pass8_ref[k] != pass8_cand[k]
    }
    return {
        "ok": not (missing or extra or mismatches),
        "n_reference": len(reference),
        "n_candidate": len(candidate),
        "reference_cap": ref_cap,
        "n_missing_in_candidate": len(missing),
        "n_extra_in_candidate": len(extra),
        "n_identical": n_identical,
        "n_continued_past_reference_cap": n_prefix,
        "n_mismatch": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "n_problems_pass8_changed": len(changed),
        "pass8_changes": dict(sorted(changed.items())[:50]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", required=True, help="samples at the lower (provisional) cap")
    parser.add_argument("--candidate", required=True, help="samples at the locked cap")
    args = parser.parse_args(argv)
    summary = compare(_read(Path(args.reference)), _read(Path(args.candidate)))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
