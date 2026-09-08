"""Phase 1: compute the locked token cap (SPEC §7) from the provisional cap run. AGENT-OWNED (tasks/02).

    uv run python scripts/compute_cap.py --run-dir runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k

Reads ``samples.jsonl`` + ``config.yaml`` (+ ``meta.json``) from the provisional run directory
(Qwen3-4B-Base, T=1.0, n=8, cap 4096, on ``val_candidates``), keeps correct completions, takes the
p99 of ``n_tokens``, × 1.25, ceil to a multiple of 256, floor 512, and writes
``configs/locked/cap.yaml`` with provenance. Refuses to overwrite an existing cap.yaml. Prints the
length histogram (correct vs incorrect) and the fraction of correct completions that would be
truncated at the cap (must be < 1 %, else nothing is written).

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
from rlordata.run_dir import git_sha, now_iso, read_run_config, read_run_meta  # noqa: E402
from rlordata.sampling.cap import (  # noqa: E402
    DEFAULT_CAP_PATH,
    MAX_CORRECT_TRUNCATED_FRAC,
    PROVISIONAL_CAP,
    CapError,
    compute_cap,
    length_histogram,
    write_cap_yaml,
)
from rlordata.sampling.eval_runner import read_samples  # noqa: E402

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True, help="provisional cap run directory")
    parser.add_argument(
        "--out", default=str(DEFAULT_CAP_PATH), help="cap.yaml path (default: the locked one)"
    )
    parser.add_argument("--allow-stub-samples", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out)
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
    correct = [s.n_tokens for s in samples if s.correct]
    incorrect = [s.n_tokens for s in samples if not s.correct]
    n_problems = len({s.problem_id for s in samples})
    try:
        result = compute_cap(
            correct,
            incorrect,
            n_problems=n_problems,
            provisional_truncated=[s.truncated for s in samples],
        )
    except CapError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    print(f"provisional run: {run_dir}")
    print(
        f"  samples={result.n_samples_total} problems={n_problems} correct={result.n_correct_used} "
        f"({result.n_correct_used / result.n_samples_total:.3f}) truncated@{PROVISIONAL_CAP}={result.provisional_truncation_rate:.4f}"
    )
    print(
        f"  correct len: mean={result.mean_correct_len:.1f} p99={result.p99_correct_len:.1f} max={result.max_correct_len}; "
        f"incorrect len mean={result.mean_incorrect_len:.1f}"
    )
    print(
        f"  cap = max(512, ceil_256(1.25 × {result.p99_correct_len:.1f})) = {result.max_completion_tokens}"
    )
    print(
        f"  correct completions truncated at cap: {result.frac_correct_truncated_at_cap:.4%} "
        f"(all: {result.frac_all_truncated_at_cap:.4%})"
    )
    print(length_histogram(correct, incorrect))

    if result.frac_correct_truncated_at_cap >= MAX_CORRECT_TRUNCATED_FRAC:
        print(
            f"REFUSING: {result.frac_correct_truncated_at_cap:.3%} of correct completions would be truncated "
            f"(limit {MAX_CORRECT_TRUNCATED_FRAC:.0%}). Report this; do not hand-edit the cap.",
            file=sys.stderr,
        )
        return 3

    record = {
        "max_completion_tokens": result.max_completion_tokens,
        "p99_correct_len": round(result.p99_correct_len, 2),
        "n_correct_used": result.n_correct_used,
        "n_samples_total": result.n_samples_total,
        "n_problems": n_problems,
        "computed_on": f"{VAL_CANDIDATES_NAME} (500 random pool problems, seed 1), T=1.0, n=8, provisional cap {PROVISIONAL_CAP}",
        "model_id": config.get("model", {}).get("id"),
        "config_hash": config.get("run_id") and (run_dir / "config_hash.txt").read_text().strip(),
        "run_id": config.get("run_id"),
        "run_dir": str(run_dir),
        "git_sha": git_sha(),
        "run_git_sha": meta.get("git_sha"),
        "computed_at": now_iso(),
        "frac_correct_truncated_at_cap": round(result.frac_correct_truncated_at_cap, 6),
        "frac_all_truncated_at_cap": round(result.frac_all_truncated_at_cap, 6),
        "provisional_truncation_rate": round(result.provisional_truncation_rate, 6),
        "rule": "max(512, ceil_to_256(1.25 * p99(n_tokens of correct completions)))",
        "sampler": (config.get("sampler") or {}).get("sampler"),
    }
    path = write_cap_yaml(out, record)
    print(f"wrote {path}")
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
