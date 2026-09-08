"""Phase 1: compute the locked token cap (SPEC §7 v1.3) from the provisional cap run. AGENT-OWNED (tasks/02, 02b).

    uv run python scripts/compute_cap.py \\
        --run-dir runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k \\
        --pool data/pool/pool.jsonl

Reads ``samples.jsonl`` + ``config.yaml`` (+ ``meta.json``) from the provisional run directory
(Qwen3-4B-Base, T=1.0, n=8, provisional cap 4096, on ``val_candidates``), joins every sample to its
pool problem by ``problem_id`` to get the cell ``(range_scale, total_steps)``, and applies

    cap = max(2048, ceil_to_256(1.25 × max over cells of p99(n_tokens of CORRECT completions)))

It prints the per-cell table (n_correct, p50, p99, % over 2048, % over the cap), which cell set
the cap, which term bound (``floor`` or ``measured``), the overall fraction of correct completions
above 2048 (the anchor-paper comparison number), and the correct/incorrect length histogram. It
writes ``configs/locked/cap.yaml`` with provenance and refuses to overwrite it. If correct
completions truncated at the chosen cap reach 1 % overall or in any cell, nothing is written.

Stub-sampler runs are refused for the default output path; ``--allow-stub-samples`` exists only so
the script can be tested end-to-end into a temporary path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rlordata.data.candidates import VAL_CANDIDATES_NAME  # noqa: E402
from rlordata.data.generator import read_jsonl  # noqa: E402
from rlordata.run_dir import git_sha, now_iso, read_run_config, read_run_meta  # noqa: E402
from rlordata.sampling.cap import (  # noqa: E402
    ANCHOR_CAP,
    DEFAULT_CAP_PATH,
    PROVISIONAL_CAP,
    CapError,
    CapResult,
    cell_key,
    compute_cap,
    length_histogram,
    per_cell_table,
    write_cap_yaml,
)
from rlordata.sampling.eval_runner import read_samples  # noqa: E402
from rlordata.types import Problem, Sample  # noqa: E402

EXPECTED_MODEL = "Qwen/Qwen3-4B-Base"
EXPECTED_N = 8
EXPECTED_T = 1.0


def validate_run(config: dict, *, allow_stub: bool, out: Path) -> list[str]:
    """Protocol checks on the provisional run's resolved config; returns problems."""
    issues: list[str] = []
    dec = config.get("decoding", {})
    model = config.get("model", {})
    if float(dec.get("temperature", -1)) != EXPECTED_T:
        issues.append(f"temperature {dec.get('temperature')} != {EXPECTED_T}")
    if int(dec.get("n", -1)) != EXPECTED_N:
        issues.append(f"n {dec.get('n')} != {EXPECTED_N}")
    if int(config.get("max_completion_tokens", -1)) != PROVISIONAL_CAP:
        issues.append(
            f"max_completion_tokens {config.get('max_completion_tokens')} != {PROVISIONAL_CAP}"
        )
    if config.get("split") != VAL_CANDIDATES_NAME:
        issues.append(f"split {config.get('split')!r} != {VAL_CANDIDATES_NAME!r}")
    if model.get("kind") != "base":
        issues.append(f"model kind {model.get('kind')!r} != 'base'")
    if model.get("id") != EXPECTED_MODEL:
        issues.append(f"model id {model.get('id')!r} != {EXPECTED_MODEL!r}")
    sampler = (config.get("sampler") or {}).get("sampler")
    if sampler != "vllm":
        if not allow_stub:
            issues.append(
                f"sampler {sampler!r} is not vllm (pass --allow-stub-samples only for tests)"
            )
        elif out.resolve() == (REPO_ROOT / DEFAULT_CAP_PATH).resolve():
            issues.append("stub samples may never be written to the real configs/locked/cap.yaml")
    return issues


def group_by_cell(
    samples: list[Sample], pool: list[Problem]
) -> tuple[dict[str, list[int]], list[int], list[str]]:
    """Join samples to pool problems by problem_id (source of truth) -> correct lengths per cell.

    Returns ``(correct_by_cell, incorrect_lengths, issues)``. ``Sample.extra`` cell fields, when
    present, are cross-checked against the pool and any disagreement is an issue.
    """
    by_id = {p.problem_id: p for p in pool}
    correct_by_cell: dict[str, list[int]] = {}
    incorrect: list[int] = []
    issues: list[str] = []
    missing = 0
    mismatched = 0
    for s in samples:
        p = by_id.get(s.problem_id)
        if p is None:
            missing += 1
            continue
        cell = cell_key(p.range_scale, p.total_steps)
        ex = s.extra or {}
        if (
            "range_scale" in ex
            and "total_steps" in ex
            and cell_key(ex["range_scale"], ex["total_steps"]) != cell
        ):
            mismatched += 1
        if s.correct:
            correct_by_cell.setdefault(cell, []).append(int(s.n_tokens))
        else:
            incorrect.append(int(s.n_tokens))
    if missing:
        issues.append(
            f"{missing} sample(s) have a problem_id that is not in the pool (wrong --pool?)"
        )
    if mismatched:
        issues.append(f"{mismatched} sample(s) carry a cell in extra that disagrees with the pool")
    return correct_by_cell, incorrect, issues


def cap_record(
    result: CapResult, *, config: dict, run_dir: Path, meta: dict, pool_path: Path
) -> dict:
    return {
        # --- the constant ---
        "max_completion_tokens": result.max_completion_tokens,
        "anchor_cap": result.anchor_cap,
        "measured_cap": result.measured_cap,
        "binding_term": result.binding_term,
        "driving_cell": result.driving_cell,
        "p99_max": round(result.p99_max, 2),
        "p99_correct_len": round(
            result.p99_max, 2
        ),  # kept for readers of the v1.2 field name (= p99 of the driving cell)
        "per_cell_p99": {k: round(v, 2) for k, v in result.per_cell_p99.items()},
        "per_cell_stats": {
            k: {
                "n_correct": s.n_correct,
                "p50": round(s.p50, 2),
                "p99": round(s.p99, 2),
                "measured_cap": s.measured_cap,
                "frac_correct_over_2048": round(s.frac_correct_over_anchor, 6),
                "frac_correct_over_cap": round(s.frac_correct_over_cap, 6),
            }
            for k, s in result.per_cell.items()
        },
        "frac_correct_over_2048": round(result.frac_correct_over_anchor, 6),
        "frac_correct_truncated_at_cap": round(result.frac_correct_truncated_at_cap, 6),
        "frac_all_truncated_at_cap": round(result.frac_all_truncated_at_cap, 6),
        "provisional_truncation_rate": round(result.provisional_truncation_rate, 6),
        # --- provenance ---
        "n_correct_used": result.n_correct_used,
        "n_samples_total": result.n_samples_total,
        "n_problems": result.n_problems,
        "computed_on": (
            f"{VAL_CANDIDATES_NAME} (500 random pool problems, seed 1), T=1.0, n=8, "
            f"provisional cap {PROVISIONAL_CAP}; cells = range_scale x total_steps from {pool_path}"
        ),
        "pool": str(pool_path),
        "model_id": config.get("model", {}).get("id"),
        "config_hash": (run_dir / "config_hash.txt").read_text().strip(),
        "run_id": config.get("run_id"),
        "run_dir": str(run_dir),
        "git_sha": git_sha(),
        "run_git_sha": meta.get("git_sha"),
        "computed_at": now_iso(),
        "rule": "max(2048, ceil_to_256(1.25 * max_over_cells(p99(n_tokens of correct completions))))",
        "sampler": (config.get("sampler") or {}).get("sampler"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True, help="provisional cap run directory")
    parser.add_argument(
        "--pool", default="data/pool/pool.jsonl", help="pool JSONL for the problem_id -> cell join"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_CAP_PATH), help="cap.yaml path (default: the locked one)"
    )
    parser.add_argument("--allow-stub-samples", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    pool_path = Path(args.pool)
    if out.exists():
        print(f"REFUSING: {out} already exists (locked protocol constant).", file=sys.stderr)
        return 2
    config = read_run_config(run_dir)
    meta = read_run_meta(run_dir) if (run_dir / "meta.json").exists() else {}
    issues = validate_run(config, allow_stub=args.allow_stub_samples, out=out)
    if issues:
        print("REFUSING: the run does not match the provisional cap protocol:", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        return 2

    samples = read_samples(run_dir / "samples.jsonl")
    if not samples:
        print("REFUSING: no samples", file=sys.stderr)
        return 2
    if not pool_path.exists():
        print(f"REFUSING: pool {pool_path} not found (needed for the cell join)", file=sys.stderr)
        return 2
    correct_by_cell, incorrect, join_issues = group_by_cell(samples, read_jsonl(pool_path))
    if join_issues:
        print("REFUSING: sample/pool join problems:", file=sys.stderr)
        for i in join_issues:
            print(f"  - {i}", file=sys.stderr)
        return 2
    n_problems = len({s.problem_id for s in samples})
    try:
        result = compute_cap(
            correct_by_cell,
            incorrect,
            n_problems=n_problems,
            provisional_truncated=[s.truncated for s in samples],
        )
    except CapError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    correct_all = [x for v in correct_by_cell.values() for x in v]
    print(f"provisional run: {run_dir}")
    print(
        f"  samples={result.n_samples_total} problems={n_problems} correct={result.n_correct_used} "
        f"({result.n_correct_used / result.n_samples_total:.3f}) "
        f"truncated@{PROVISIONAL_CAP}={result.provisional_truncation_rate:.4f}"
    )
    print(
        f"  correct len: mean={result.mean_correct_len:.1f} max={result.max_correct_len}; "
        f"incorrect len mean={result.mean_incorrect_len:.1f}"
    )
    print("per-cell table (correct completions only):")
    print(per_cell_table(result))
    print(length_histogram(correct_all, incorrect))

    violations = result.violations()
    if violations:
        print(
            "REFUSING: correct-completion truncation at the chosen cap reaches the 1 % limit:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print("Report this; do not hand-edit the cap (SPEC §7).", file=sys.stderr)
        return 3

    record = cap_record(result, config=config, run_dir=run_dir, meta=meta, pool_path=pool_path)
    path = write_cap_yaml(out, record)
    print(f"wrote {path}")
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    if result.binding_term == "measured":
        print(
            f"NOTE: the measured term raised the cap above the anchor {ANCHOR_CAP} "
            f"(cell {result.driving_cell}, p99 {result.p99_max:.1f}); this is recorded in cap.yaml."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
