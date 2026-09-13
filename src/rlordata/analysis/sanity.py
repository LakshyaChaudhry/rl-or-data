"""Mechanical checks that run before any result is reported (CLAUDE.md 'Scientific standards').

Implemented (tasks/02a §8):
  - split disjointness (``problem_id`` and ``structure_id``) across train/val/test/ood
  - identical cap, prompt template, answer regex and prompt-length limit across the resolved
    configs of runs being compared
  - truncation > 5 % and extraction-failure > 5 % flags on metrics.json; on ``ood_hard_200``
    truncation is reported per tier, prominently, but never flagged (SPEC §7 v1.3)

Added by tasks/03:
  - LoRA adapter is non-trivial: every ``lora_B`` tensor is non-zero (PEFT initialises B to zero,
    so an untrained or unapplied adapter has ‖ΔW‖ = 0) — :func:`check_adapter_nontrivial`
  - the evaluated adapter is the run's final (last-epoch) one — :func:`check_final_checkpoint`
  - the merged model's greedy outputs differ from the base model's on ≥ 10 % of val prompts —
    :func:`check_outputs_differ`
  - tokenizer identity between vLLM and the trainer is asserted in ``train/rft_eval.py`` from the
    hashes both record (``tokenizer_sha256_trainer`` / ``tokenizer_sha256_vllm``)

    python -m rlordata.analysis.sanity --splits-dir data/splits --run-dirs runs/eval/*/test_300/greedy
"""

from __future__ import annotations

import argparse
import hashlib
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


# ---------------------------------------------------------------------------
# tasks/03 §5: trained-adapter checks
# ---------------------------------------------------------------------------

MIN_OUTPUT_DIFFERENCE = 0.10


def check_adapter_nontrivial(adapter_dir: str | Path) -> list[str]:
    """‖ΔW‖ > 0: every LoRA ``B`` matrix must be non-zero (PEFT zero-initialises B)."""
    d = Path(adapter_dir)
    issues: list[str] = []
    weights = d / "adapter_model.safetensors"
    if not (d / "adapter_config.json").exists():
        return [f"{d}: no adapter_config.json (not a PEFT adapter directory)"]
    if not weights.exists():
        return [f"{d}: no adapter_model.safetensors"]
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - ml extra missing
        return [f"{d}: safetensors not installed; cannot verify the adapter"]
    n_b = 0
    zero_b: list[str] = []
    total_sq = 0.0
    with safe_open(str(weights), framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open has no __iter__
            if "lora_B" not in key:
                continue
            t = f.get_tensor(key).float()
            n_b += 1
            sq = float((t * t).sum())
            total_sq += sq
            if sq == 0.0:
                zero_b.append(key)
    if n_b == 0:
        issues.append(f"{d}: adapter has no lora_B tensors")
    if zero_b:
        issues.append(
            f"{d}: {len(zero_b)}/{n_b} lora_B tensors are all-zero (adapter not trained or not applied), "
            f"e.g. {zero_b[0]}"
        )
    if n_b and total_sq == 0.0:
        issues.append(f"{d}: ‖ΔW‖ = 0")
    return issues


def check_final_checkpoint(run_dir: str | Path, *, eval_set: str | None = None) -> list[str]:
    """The adapter that gets evaluated must be the run's final checkpoint (RFT last-epoch / GRPO step 300).

    For GRPO runs ``eval_set="final"`` (test/ood/transfer) is only allowed on the step-300 adapter;
    ``"val"`` may evaluate any recorded step checkpoint (the val curve at 100/200/300).
    """
    d = Path(run_dir)
    b = d / "budgets.json"
    if not b.exists():
        return [f"{d}: no budgets.json (training did not finish)"]
    with b.open(encoding="utf-8") as f:
        budgets = json.load(f)
    issues: list[str] = []
    final = Path(budgets.get("final_adapter", ""))
    if not final or not final.exists():
        issues.append(f"{d}: final adapter {final} missing")

    epochs = budgets.get("epoch_adapters") or []
    if "epochs" in budgets or epochs:
        if int(budgets.get("epochs", -1)) != len(epochs):
            issues.append(
                f"{d}: {len(epochs)} epoch adapters saved but {budgets.get('epochs')} epochs planned"
            )
        src = final / "SOURCE.txt"
        if epochs and src.exists() and Path(epochs[-1]).name not in src.read_text():
            issues.append(
                f"{d}: final adapter is not a copy of the last epoch ({Path(epochs[-1]).name})"
            )
    elif "step_adapters" in budgets or "optimizer_steps" in budgets:
        # GRPO (tasks/04): the adapter under evaluation must be one this run saved, and the
        # test/ood/transfer eval must be the step-300 (= max_steps) one.
        step_adapters = {str(k): str(v) for k, v in (budgets.get("step_adapters") or {}).items()}
        known = set(step_adapters.values()) | {str(budgets.get("final_adapter_trained", ""))}
        max_steps = int(budgets.get("max_steps", budgets.get("optimizer_steps", 300)) or 300)
        eval_step = budgets.get("eval_step")
        if str(final) not in known and eval_step is None:
            issues.append(f"{d}: adapter {final} is not one of this run's saved checkpoints")
        if eval_step is not None and step_adapters.get(str(eval_step)) != str(final):
            issues.append(
                f"{d}: budgets.eval_step={eval_step} but final_adapter={final} is not that step's adapter"
            )
        if eval_set == "final":
            step300 = step_adapters.get(str(max_steps))
            if (
                step300 is None
                or str(final) != step300
                or (eval_step is not None and int(eval_step) != max_steps)
            ):
                issues.append(
                    f"{d}: test/ood eval requested on {final} (eval_step={eval_step}); "
                    f"only the step-{max_steps} adapter may be evaluated on the held-out sets"
                )

    meta = d / "meta.json"
    if meta.exists():
        with meta.open(encoding="utf-8") as f:
            if json.load(f).get("status") != "finished":
                issues.append(f"{d}: training status is not 'finished'")
    return issues


def check_grpo_reward_budget(run_dir: str | Path, *, expected: int = 19200) -> list[str]:
    """Exactly ``expected`` completions were sampled (count reward_records.jsonl)."""
    d = Path(run_dir)
    path = d / "reward_records.jsonl"
    if not path.exists():
        return [f"{d}: no reward_records.jsonl"]
    n = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    issues: list[str] = []
    if n != int(expected):
        issues.append(f"{d}: reward records={n}, expected {expected} (SPEC §8 / tasks/04 §6)")
    budgets = d / "budgets.json"
    if budgets.exists():
        with budgets.open(encoding="utf-8") as f:
            b = json.load(f)
        if int(b.get("completions_consumed", -1)) != n:
            issues.append(
                f"{d}: budgets.completions_consumed={b.get('completions_consumed')} != records {n}"
            )
    return issues


def check_c1_reward_near_half(
    run_dir: str | Path, *, lo: float = 0.4, hi: float = 0.6
) -> list[str]:
    """C1 training reward should stay ≈ 0.5 while ``correct`` is logged separately."""
    d = Path(run_dir)
    path = d / "reward_records.jsonl"
    if not path.exists():
        return [f"{d}: no reward_records.jsonl"]
    rewards: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("reward_name") not in ("random_bernoulli", "random_bernoulli_0_5"):
                continue
            rewards.append(float(row["reward"]))
    if not rewards:
        return [f"{d}: no random_bernoulli reward records (not a C1 run?)"]
    mean = sum(rewards) / len(rewards)
    if not (lo <= mean <= hi):
        return [f"{d}: C1 mean training reward={mean:.3f} outside [{lo}, {hi}]"]
    return []


def check_prompt_hashes_match(run_dir: str | Path, splits_dir: str | Path) -> list[str]:
    """Trainer prompt bytes equal ``format_prompt(problem, 'base')`` for each problem_id."""
    from rlordata.sampling.prompts import format_prompt

    d = Path(run_dir)
    cfg_path = d / "config.yaml"
    if not cfg_path.exists():
        return [f"{d}: no config.yaml"]
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    stored = cfg.get("prompt_sha256_by_problem_id") or {}
    if not stored:
        return [f"{d}: config missing prompt_sha256_by_problem_id"]
    dc = cfg.get("data_condition")
    split_path = Path(splits_dir) / f"{dc}.jsonl"
    if not split_path.exists():
        return [f"{d}: split {split_path} missing for prompt-hash check"]
    from rlordata.data.generator import read_jsonl

    issues: list[str] = []
    for p in read_jsonl(split_path):
        if p.problem_id not in stored:
            issues.append(f"{d}: missing prompt hash for {p.problem_id[:12]}")
            continue
        h = hashlib.sha256(format_prompt(p, "base").encode("utf-8")).hexdigest()
        if h != stored[p.problem_id]:
            issues.append(f"{d}: prompt hash drift on {p.problem_id[:12]}")
    return issues


def check_outputs_differ(
    base_samples: list[Any],
    new_samples: list[Any],
    *,
    min_fraction: float = MIN_OUTPUT_DIFFERENCE,
) -> list[str]:
    """The merged model's greedy completions must differ from the base's on ≥ ``min_fraction``.

    Greedy is deterministic, so identical completions on (almost) every prompt means the adapter
    was not merged / not applied; the eval would silently be a base-model eval.
    """
    base = {s.problem_id: s.completion for s in base_samples}
    new = {s.problem_id: s.completion for s in new_samples}
    common = sorted(set(base) & set(new))
    if not common:
        return ["no common problems between the base and adapter val samples"]
    differ = sum(1 for pid in common if base[pid] != new[pid])
    frac = differ / len(common)
    if frac < min_fraction:
        return [
            f"greedy outputs differ from the base model on only {differ}/{len(common)} val prompts "
            f"({100 * frac:.1f}% < {100 * min_fraction:.0f}%): adapter not applied?"
        ]
    return []


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
