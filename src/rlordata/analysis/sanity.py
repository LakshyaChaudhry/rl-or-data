"""Mechanical checks that run before any result is reported (CLAUDE.md 'Scientific standards').

Implemented (tasks/02a §8):
  - split disjointness (``problem_id`` and ``structure_id``) across train/val/test/ood
  - identical cap, prompt template, answer regex and prompt-length limit across the resolved
    configs of runs being compared
  - truncation > 5 % and extraction-failure > 5 % flags on metrics.json; on ``ood_hard_200``
    truncation is reported per tier, prominently, but never flagged (SPEC §7 v1.3)

Still to implement (tasks/03+):
  - LoRA adapter actually loaded (parameter count delta, adapter hash)
  - tokenizer identity between vLLM and trainer
  - eval checkpoint == final checkpoint (no early stopping on test)

    python -m rlordata.analysis.sanity --splits-dir data/splits --run-dirs runs/eval/*/test_300/greedy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from rlordata.data.generator import read_jsonl
from rlordata.data.tiers import structure_id
from rlordata.sampling.eval_runner import REPORT_ONLY_TRUNCATION_SPLITS
from rlordata.types import Problem

TRUNCATION_MAX = 0.05  # SPEC §7
EXTRACTION_FAILURE_MAX = 0.05  # design choice; flagged, not headline-blocking
PROTOCOL_FIELDS = (
    "max_completion_tokens",
    "prompt_template",
    "answer_regex",
    "extraction_rule",
    "max_prompt_tokens",
)
# (subset, superset) pairs where overlap is by construction, not leakage.
DERIVED_SPLITS: tuple[tuple[str, str], ...] = (("train_curated", "train_mixed_100"),)
# Files in data/splits that are not splits.
NON_SPLIT_FILES = ("pool_tiered",)


def check_split_disjointness(
    splits: dict[str, list[Problem]],
    *,
    derived: Iterable[tuple[str, str]] = DERIVED_SPLITS,
) -> list[str]:
    """Return human-readable problems; empty list means every pair is disjoint.

    Checks pairwise ``problem_id`` and ``structure_id`` overlap. Derived pairs (curated ⊂ mixed)
    are exempt from the pairwise check but must be genuine subsets. Transfer sets
    (``pipeline["source"]`` set) skip the structure check since they have no counting pipeline.
    """
    issues: list[str] = []
    derived_set = {tuple(p) for p in derived}
    names = [n for n in splits if n not in NON_SPLIT_FILES]
    ids = {n: {p.problem_id for p in splits[n]} for n in names}
    structs = {
        n: {structure_id(p.pipeline) for p in splits[n] if "source" not in p.pipeline}
        for n in names
    }
    for n in names:
        if len(ids[n]) != len(splits[n]):
            issues.append(
                f"{n}: {len(splits[n]) - len(ids[n])} duplicate problem_id(s) within the split"
            )
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if (a, b) in derived_set or (b, a) in derived_set:
                sub, sup = (a, b) if (a, b) in derived_set else (b, a)
                if not ids[sub] <= ids[sup]:
                    issues.append(
                        f"{sub} is not a subset of {sup} ({len(ids[sub] - ids[sup])} stray ids)"
                    )
                continue
            common = ids[a] & ids[b]
            if common:
                issues.append(
                    f"{a} ∩ {b}: {len(common)} shared problem_id(s), e.g. {sorted(common)[0][:12]}"
                )
            common_s = structs[a] & structs[b]
            if common_s:
                issues.append(
                    f"{a} ∩ {b}: {len(common_s)} shared pipeline structure(s), e.g. {sorted(common_s)[0][:12]}"
                )
    return issues


def load_splits_dir(path: str | Path) -> dict[str, list[Problem]]:
    d = Path(path)
    return {
        p.stem: read_jsonl(p) for p in sorted(d.glob("*.jsonl")) if p.stem not in NON_SPLIT_FILES
    }


def load_resolved_config(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_protocol_identical(
    run_dirs: Iterable[str | Path],
    *,
    fields: Iterable[str] = PROTOCOL_FIELDS,
) -> list[str]:
    """Every run being compared must share the cap, template, regex and prompt-length limit."""
    dirs = [Path(d) for d in run_dirs]
    if len(dirs) < 2:
        return []
    configs = {d: load_resolved_config(d) for d in dirs}
    issues: list[str] = []
    for field in fields:
        values: dict[str, list[str]] = {}
        for d, cfg in configs.items():
            if field not in cfg:
                issues.append(f"{d}: resolved config lacks {field!r}")
                continue
            values.setdefault(json.dumps(cfg[field], sort_keys=True), []).append(str(d))
        if len(values) > 1:
            desc = "; ".join(f"{v[:40]!s} in {len(ds)} run(s)" for v, ds in values.items())
            issues.append(f"{field} differs across runs: {desc}")
    provisional = [str(d) for d, cfg in configs.items() if cfg.get("cap_is_provisional")]
    if provisional:
        issues.append(f"{len(provisional)} run(s) used a provisional cap: {provisional[:3]}")
    return issues


def check_rates(
    metrics: dict[str, Any],
    *,
    truncation_max: float = TRUNCATION_MAX,
    extraction_failure_max: float = EXTRACTION_FAILURE_MAX,
    label: str = "",
    split: str | None = None,
) -> list[str]:
    """Flag truncation / extraction-failure rates above threshold (CLAUDE.md, SPEC §7).

    ``split`` (default: ``metrics["split"]``) selects the policy: on ``REPORT_ONLY_TRUNCATION_SPLITS``
    (ood_hard_200) truncation is never an issue; use :func:`truncation_report` to print it.
    """
    issues: list[str] = []
    prefix = f"{label}: " if label else ""
    split = split if split is not None else metrics.get("split")
    tr = float(metrics.get("truncation_rate", 0.0))
    ef = float(metrics.get("extraction_failure_rate", 0.0))
    if tr > truncation_max and split not in REPORT_ONLY_TRUNCATION_SPLITS:
        issues.append(
            f"{prefix}truncation rate {tr:.3f} > {truncation_max:.2f} — not a headline number"
        )
    if ef > extraction_failure_max:
        issues.append(f"{prefix}extraction-failure rate {ef:.3f} > {extraction_failure_max:.2f}")
    return issues


def truncation_report(metrics: dict[str, Any], *, label: str = "") -> str:
    """One prominent line: overall + per-tier truncation and extraction-failure, at-cap rate."""
    prefix = f"{label}: " if label else ""
    tiers = metrics.get("per_tier_truncation_rate") or {}
    ext = metrics.get("per_tier_extraction_failure_rate") or {}
    per_tier = ", ".join(
        f"{t} trunc {100 * r:.1f}% / extract-fail {100 * ext.get(t, 0.0):.1f}%"
        for t, r in sorted(tiers.items())
    )
    at_cap = metrics.get("at_cap_rate")
    at_cap_s = "" if at_cap is None else f"; at cap {100 * at_cap:.1f}%"
    policy = (
        " [reported, not flagged]" if metrics.get("split") in REPORT_ONLY_TRUNCATION_SPLITS else ""
    )
    return (
        f"{prefix}truncation {100 * float(metrics.get('truncation_rate', 0.0)):.1f}% overall"
        f"{at_cap_s}{policy}" + (f" — {per_tier}" if per_tier else "")
    )


def check_run_dirs(run_dirs: Iterable[str | Path]) -> list[str]:
    """Protocol identity across run dirs plus per-run rate flags from metrics.json."""
    dirs = [Path(d) for d in run_dirs]
    issues = check_protocol_identical(dirs)
    for d in dirs:
        m = d / "metrics.json"
        if not m.exists():
            issues.append(f"{d}: no metrics.json")
            continue
        with m.open(encoding="utf-8") as f:
            issues.extend(check_rates(json.load(f), label=str(d)))
    return issues


def run_dir_reports(run_dirs: Iterable[str | Path]) -> list[str]:
    """Per-run truncation reports (always printed; the only place ood_hard_200 truncation surfaces)."""
    out: list[str] = []
    for d in (Path(d) for d in run_dirs):
        m = d / "metrics.json"
        if m.exists():
            with m.open(encoding="utf-8") as f:
                out.append(truncation_report(json.load(f), label=str(d)))
    return out


def format_problems(title: str, issues: list[str]) -> str:
    if not issues:
        return f"[sanity] {title}: OK"
    return f"[sanity] {title}: {len(issues)} issue(s)\n" + "\n".join(f"  - {i}" for i in issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rlordata.analysis.sanity")
    parser.add_argument(
        "--splits-dir", default=None, help="check split disjointness in this directory"
    )
    parser.add_argument("--run-dirs", nargs="*", default=[], help="run directories to compare")
    args = parser.parse_args(argv)
    all_issues: list[str] = []
    if args.splits_dir:
        issues = check_split_disjointness(load_splits_dir(args.splits_dir))
        print(format_problems(f"split disjointness ({args.splits_dir})", issues))
        all_issues += issues
    if args.run_dirs:
        for line in run_dir_reports(args.run_dirs):
            print(f"[sanity] {line}")
        issues = check_run_dirs(args.run_dirs)
        print(format_problems(f"protocol + rates ({len(args.run_dirs)} run dirs)", issues))
        all_issues += issues
    if not args.splits_dir and not args.run_dirs:
        parser.print_help(sys.stderr)
        return 2
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
