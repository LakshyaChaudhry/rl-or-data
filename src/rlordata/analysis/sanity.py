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

Added by tasks/05 (cross-run, :func:`cross_run_checks`): unit integrity (stored metrics equal the
metrics recomputed from samples.jsonl), one protocol for every compared unit, committed split
digests, prompt bytes identical across base-model arms, final checkpoints only, seeds actually
differ (adapter hashes, greedy outputs), budgets present, shared hyperparameters, val-only
selection, plus listed-not-fatal notes (SPEC §7 truncation flags, git_dirty, exclusions, what
could not be verified from a mirror).

    python -m rlordata.analysis.sanity --splits-dir data/splits --run-dirs runs/eval/*/test_300/greedy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import yaml

from rlordata.data.generator import read_jsonl
from rlordata.data.tiers import structure_id
from rlordata.sampling.eval_runner import REPORT_ONLY_TRUNCATION_SPLITS
from rlordata.types import Problem

if TYPE_CHECKING:
    from rlordata.analysis.loader import Dataset, Run, Unit

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


def check_final_checkpoint(
    run_dir: str | Path, *, eval_set: str | None = None, root: str | Path | None = None
) -> list[str]:
    """The adapter that gets evaluated must be the run's final checkpoint (RFT last-epoch / GRPO step 300).

    For GRPO runs ``eval_set="final"`` (test/ood/transfer) is only allowed on the step-300 adapter;
    ``"val"`` may evaluate any recorded step checkpoint (the val curve at 100/200/300).

    ``root`` is the directory the recorded ``runs/...`` paths are relative to (the store root when
    reading a mirror; default: the working directory, as on the training box).
    """
    d = Path(run_dir)
    b = d / "budgets.json"
    if not b.exists():
        return [f"{d}: no budgets.json (training did not finish)"]
    with b.open(encoding="utf-8") as f:
        budgets = json.load(f)
    issues: list[str] = []
    base = Path(root) if root is not None else Path()
    final = Path(budgets.get("final_adapter", ""))
    if not budgets.get("final_adapter") or not (base / final).exists():
        issues.append(f"{d}: final adapter {final} missing")

    epochs = budgets.get("epoch_adapters") or []
    if "epochs" in budgets or epochs:
        if int(budgets.get("epochs", -1)) != len(epochs):
            issues.append(
                f"{d}: {len(epochs)} epoch adapters saved but {budgets.get('epochs')} epochs planned"
            )
        src = base / final / "SOURCE.txt"
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


# ---------------------------------------------------------------------------
# tasks/05: cross-run checks over a loaded Dataset (fail loudly)
# ---------------------------------------------------------------------------

GRPO_COMPLETIONS = 19200  # SPEC §8: 300 steps × 8 prompts × 8 generations
REQUIRED_BUDGET_KEYS = (
    "prompts",
    "completions_available",
    "completions_consumed",
    "training_tokens",
    "optimizer_steps",
)
STORED_METRIC_KEYS = (
    "n_problems",
    "accuracy",
    "ci_low",
    "ci_high",
    "truncation_rate",
    "extraction_failure_rate",
    "answer_line_rate",
    "mean_completion_tokens",
)
GRPO_CONFIG_MAY_DIFFER = ("seed", "output_dir", "run_name", "logging_dir")
LORA_KEYS = ("r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora", "task_type")
COUNTING_SPLITS = ("val_mixed_100", "test_300", "ood_hard_200")


@dataclass
class CheckResult:
    name: str
    detail: str  # what was checked, with counts
    failures: list[str] = dc_field(default_factory=list)
    notes: list[str] = dc_field(default_factory=list)  # listed in the output, never fatal


@dataclass
class CrossRunReport:
    checks: list[CheckResult]

    @property
    def failures(self) -> list[str]:
        return [f"{c.name}: {f}" for c in self.checks for f in c.failures]

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": not self.failures,
            "checks": [
                {"name": c.name, "detail": c.detail, "failures": c.failures, "notes": c.notes}
                for c in self.checks
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Cross-run sanity (tasks/05 item 1)",
            "",
            f"**{'FAILED' if self.failures else 'passed'}** — {len(self.checks)} checks, "
            f"{len(self.failures)} failure(s), {sum(len(c.notes) for c in self.checks)} note(s). "
            "Failures abort `make analysis`; notes are listed and carried into the tables and the "
            "grading sheet.",
            "",
        ]
        for c in self.checks:
            lines.append(f"## {c.name} — {'FAIL' if c.failures else 'ok'}")
            lines.append("")
            lines.append(c.detail)
            lines.append("")
            for f_ in c.failures:
                lines.append(f"- **FAIL** {f_}")
            for n in c.notes:
                lines.append(f"- note: {n}")
            if c.failures or c.notes:
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _units(ds: Dataset) -> list[tuple[Run, Unit]]:
    return [(r, u) for r in ds.all_runs() for u in r.units.values()]


def _who(run: Run) -> str:
    return f"{run.key}/seed{run.seed}" if run.kind in ("arm", "control") else run.key


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    return a == b


def check_units_complete(ds: Dataset) -> CheckResult:
    known = {
        (k["who"], k["split"], k["decoding"]): k["why"] for k in ds.cfg.get("known_missing") or []
    }
    res = CheckResult(
        "units present",
        f"{len(_units(ds))} units loaded for {len(ds.all_runs())} models/runs; "
        f"{len(ds.missing)} expected unit(s) absent.",
    )
    for m in ds.missing:
        why = known.get((m["who"], m["split"], m["decoding"]))
        line = f"{m['who']} {m['split']}/{m['decoding']} does not exist"
        if why is None:
            res.failures.append(line)
        else:
            res.notes.append(f"{line} — known: {why}. Rendered as 'missing', never as zero.")
    for key, runs in {**ds.arms, **ds.controls}.items():
        want = [
            int(s)
            for s in (ds.cfg["arms"] | (ds.cfg.get("controls") or {}))[key].get(
                "seeds", ds.cfg["seeds"]
            )
        ]
        if [r.seed for r in runs] != want:
            res.failures.append(f"{key}: seeds {[r.seed for r in runs]} != configured {want}")
    return res


def check_exclusions(ds: Dataset) -> CheckResult:
    from rlordata.analysis.loader import excluded_present, excluded_reason

    res = CheckResult(
        "never-results stay out",
        "Runs are loaded from an explicit allow-list; every loaded path is re-checked against the "
        "never-results fragments. Present under the run root but never read as results:",
    )
    for run, unit in _units(ds):
        why = excluded_reason(unit.path, ds.run_root, ds.cfg)
        if why is not None:
            res.failures.append(f"{_who(run)} loaded {unit.rel}, which is not a result: {why}")
    for item in excluded_present(ds.run_root, ds.cfg):
        if item["paths"]:
            shown = ", ".join(item["paths"][:4]) + (
                f", … ({len(item['paths'])} paths)" if len(item["paths"]) > 4 else ""
            )
            res.notes.append(f"excluded `{item['fragment']}` ({item['why']}): {shown}")
    return res


def check_unit_integrity(ds: Dataset) -> CheckResult:
    res = CheckResult(
        "unit integrity",
        "Every unit: status finished, n_samples == planned, sampler vllm, thinking off, samples carry "
        "the unit's config hash and seed, no truncated completion is scored correct (SPEC §5), and the "
        "metrics recomputed from samples.jsonl with core.evaluate equal the stored metrics.json "
        f"({', '.join(STORED_METRIC_KEYS)}, pass_at_k, per_tier).",
    )
    for run, u in _units(ds):
        tag = f"{_who(run)} {u.split}/{u.decoding}"
        if u.meta.get("status") != "finished":
            res.failures.append(f"{tag}: status {u.meta.get('status')!r}")
        n = u.metrics["n_samples"]
        if not (u.meta.get("n_samples") == u.meta.get("n_samples_planned") == n):
            res.failures.append(
                f"{tag}: n_samples {u.meta.get('n_samples')} / planned "
                f"{u.meta.get('n_samples_planned')} / in samples.jsonl {n}"
            )
        sampler = (u.config.get("sampler") or {}).get("sampler")
        if sampler != "vllm" or u.stored_metrics.get("sampler") != "vllm":
            res.failures.append(f"{tag}: sampler {sampler!r} (only vLLM numbers are results)")
        if u.config.get("thinking") is not False:
            res.failures.append(f"{tag}: thinking={u.config.get('thinking')!r}")
        if u.sample_field_values["config_hash"] != [u.config_hash]:
            res.failures.append(
                f"{tag}: samples carry config hashes {u.sample_field_values['config_hash']}"
            )
        if u.sample_field_values["seed"] != [u.seed]:
            res.failures.append(f"{tag}: samples carry seeds {u.sample_field_values['seed']}")
        if u.n_truncated_correct:
            res.failures.append(
                f"{tag}: {u.n_truncated_correct} truncated completion(s) scored correct"
            )
        for k in (*STORED_METRIC_KEYS, "pass_at_k", "per_tier"):
            if not _close(u.metrics.get(k), u.stored_metrics.get(k)):
                res.failures.append(
                    f"{tag}: stored {k}={u.stored_metrics.get(k)!r} != recomputed {u.metrics.get(k)!r}"
                )
    return res


def check_protocol(ds: Dataset) -> CheckResult:
    dirs = [u.path for _, u in _units(ds)]
    res = CheckResult(
        "one protocol",
        f"{', '.join(PROTOCOL_FIELDS)} identical in the resolved config of all {len(dirs)} units; "
        "cap equal to configs/locked/cap.yaml; no provisional cap.",
        failures=check_protocol_identical(dirs),
    )
    cap_path = Path("configs/locked/cap.yaml")
    if cap_path.exists():
        with cap_path.open(encoding="utf-8") as f:
            cap = int((yaml.safe_load(f) or {}).get("max_completion_tokens", -1))
        bad = sorted({int(u.config["max_completion_tokens"]) for _, u in _units(ds)} - {cap})
        if bad:
            res.failures.append(f"units evaluated at cap {bad}, locked cap is {cap}")
    return res


def check_split_digests(ds: Dataset, splits_dir: str | Path) -> CheckResult:
    from rlordata.sampling.eval_runner import parse_subset, problem_ids_digest

    res = CheckResult(
        "committed splits",
        f"Pairwise disjointness of {splits_dir} by problem_id and pipeline structure; every counting "
        "unit's problem_ids_sha256 and sample problem ids equal the committed split (pass@k: its "
        "first-100 prefix); gsm8k_500 digests identical across units.",
    )
    splits = load_splits_dir(splits_dir)
    res.failures += check_split_disjointness(splits)
    gsm: dict[str, list[str]] = {}
    for run, u in _units(ds):
        tag = f"{_who(run)} {u.split}/{u.decoding}"
        digest = u.config.get("problem_ids_sha256")
        if u.split not in COUNTING_SPLITS:
            gsm.setdefault(str(digest), []).append(tag)
            continue
        if u.split not in splits:
            res.failures.append(f"{tag}: split file {u.split}.jsonl missing in {splits_dir}")
            continue
        problems = splits[u.split]
        if u.config.get("subset"):
            problems = problems[: parse_subset(u.config["subset"])[1]]
        if digest != problem_ids_digest(problems):
            res.failures.append(f"{tag}: problem_ids_sha256 differs from the committed split")
        if tuple(sorted(p.problem_id for p in problems)) != u.problem_ids:
            res.failures.append(f"{tag}: samples.jsonl problem ids differ from the committed split")
    if len(gsm) > 1:
        res.failures.append(
            f"gsm8k_500 evaluated on {len(gsm)} different problem sets: {sorted(gsm)}"
        )
    elif gsm:
        res.notes.append(
            f"gsm8k_500 digest {next(iter(gsm))[:16]}… is identical in all "
            f"{sum(len(v) for v in gsm.values())} units; it is compared between units only (the set is "
            "downloaded, not committed)."
        )
    return res


def check_prompt_drift(ds: Dataset) -> CheckResult:
    res = CheckResult(
        "prompt bytes",
        "For every base-model unit (base, arms, controls) the sha256 over (problem_id, prompt) is "
        "identical across runs for the same split/subset — no prompt-template drift between arms.",
    )
    groups: dict[tuple[str, str | None], dict[str, list[str]]] = {}
    n_instruct = 0
    for run, u in _units(ds):
        if (u.config.get("model") or {}).get("kind") != "base":
            n_instruct += 1
            continue
        groups.setdefault((u.split, u.config.get("subset")), {}).setdefault(
            u.prompt_digest, []
        ).append(_who(run))
    for (split, subset), by_digest in sorted(groups.items(), key=str):
        if len(by_digest) > 1:
            desc = "; ".join(f"{d[:10]}…: {sorted(set(w))[:4]}" for d, w in by_digest.items())
            res.failures.append(
                f"{subset or split}: {len(by_digest)} distinct prompt sets — {desc}"
            )
    if n_instruct:
        res.notes.append(
            f"{n_instruct} instruct reference units wrap the same template in their chat template "
            "(SPEC §5); they are covered by the prompt_template field check only."
        )
    return res


def check_final_checkpoints(ds: Dataset) -> CheckResult:
    root = ds.run_root.parent  # recorded paths are "runs/..."
    res = CheckResult(
        "final checkpoints only",
        "Each trained run: training finished; every eval/final unit names the run's final adapter "
        "(RFT last epoch = adapter/final, GRPO adapter/step_300 with trainer_state.global_step == "
        "max_steps) and eval_set 'final'; RFT runs use the val-chosen lr/epochs and append_eos.",
    )
    unverifiable = []
    for run in ds.trained_runs():
        who = _who(run)
        res.failures += check_final_checkpoint(run.run_dir, eval_set="final", root=root)
        budgets = run.budgets or {}
        final = str(budgets.get("final_adapter", ""))
        for u in run.units.values():
            tag = f"{who} {u.split}/{u.decoding}"
            if u.sanity.get("adapter") != final or u.config.get("adapter") != final:
                res.failures.append(
                    f"{tag}: evaluated {u.sanity.get('adapter')!r}, final adapter is {final!r}"
                )
            if u.sanity.get("eval_set") != "final":
                res.failures.append(f"{tag}: eval_set {u.sanity.get('eval_set')!r}")
            if u.config.get("train_run_dir") != f"{ds.run_root.name}/{run.rel}":
                res.failures.append(f"{tag}: train_run_dir {u.config.get('train_run_dir')!r}")
            if u.sanity.get("protocol_vs_base") != "ok":
                res.failures.append(f"{tag}: protocol_vs_base {u.sanity.get('protocol_vs_base')!r}")
        if run.method == "grpo":
            state = root / final / "trainer_state.json"
            if state.exists():
                with state.open(encoding="utf-8") as f:
                    step = json.load(f).get("global_step")
                if step != budgets.get("max_steps"):
                    res.failures.append(
                        f"{who}: {final} is at step {step}, not {budgets.get('max_steps')}"
                    )
            if not (root / final / "adapter_model.safetensors").exists():
                unverifiable.append(who)
        if run.method == "rft":
            cfg_path = run.run_dir / "config.yaml"
            rcfg = load_resolved_config(run.run_dir) if cfg_path.exists() else {}
            if rcfg.get("append_eos") is not True:
                res.failures.append(
                    f"{who}: append_eos is {rcfg.get('append_eos')!r} (PREREGISTRATION §4)"
                )
            chosen = run.chosen or {}
            if not (
                _close(budgets.get("learning_rate"), chosen.get("learning_rate"))
                and budgets.get("epochs") == chosen.get("epochs")
            ):
                res.failures.append(
                    f"{who}: trained lr={budgets.get('learning_rate')} ep={budgets.get('epochs')}, "
                    f"chosen.json says lr={chosen.get('learning_rate')} ep={chosen.get('epochs')}"
                )
    if unverifiable:
        res.notes.append(
            f"NOT VERIFIED here: the weights of the evaluated GRPO adapter (adapter/step_300) are not in "
            f"this run root for {len(unverifiable)} run(s) ({', '.join(unverifiable[:3])}, …); only "
            "adapter/final (saved by the trainer right after step 300) is. That the two hold the same "
            "weights follows from train/grpo_trl.py, not from a byte comparison."
        )
    return res


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_hashes(ds: Dataset, *, cache_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """{who: {path, sha256 | None}} for every trained run's final adapter weights.

    Uses the evaluated adapter when its weights are present, else ``final_adapter_trained``.
    Hashes are cached by (path, size, mtime_ns); the cache never changes a value.
    """
    root = ds.run_root.parent
    cache: dict[str, Any] = {}
    if cache_path is not None and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cache = json.load(f)
    out: dict[str, dict[str, Any]] = {}
    for run in ds.trained_runs():
        budgets = run.budgets or {}
        rel = None
        for cand in (budgets.get("final_adapter"), budgets.get("final_adapter_trained")):
            if cand and (root / cand / "adapter_model.safetensors").exists():
                rel = f"{cand}/adapter_model.safetensors"
                break
        if rel is None:
            out[_who(run)] = {"path": None, "sha256": None}
            continue
        st = (root / rel).stat()
        key = f"{rel}|{st.st_size}|{st.st_mtime_ns}"
        if key not in cache:
            cache[key] = _sha256_file(root / rel)
        out[_who(run)] = {"path": rel, "sha256": cache[key]}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return out


def _store_checksums(path: str | Path | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1].strip().lstrip("*")] = parts[0]
    return out


def check_seeds_differ(ds: Dataset, *, cache_path: Path | None = None) -> CheckResult:
    res = CheckResult(
        "seeds actually differ",
        "sha256 of every trained run's final adapter weights is pairwise distinct across all runs "
        "(and equals the store's checksum listing when one is configured); greedy test_300 "
        "completions differ between seeds of an arm; greedy val completions differ from the base "
        f"model's on ≥ {100 * MIN_OUTPUT_DIFFERENCE:.0f} % of prompts (adapter really applied).",
    )
    hashes = adapter_hashes(ds, cache_path=cache_path)
    store = _store_checksums(ds.cfg.get("store_checksums"))
    seen: dict[str, str] = {}
    for who, h in hashes.items():
        if h["sha256"] is None:
            res.failures.append(f"{who}: no adapter weights under the run root; cannot verify")
            continue
        if h["sha256"] in seen:
            res.failures.append(f"{who} and {seen[h['sha256']]} have byte-identical adapters")
        seen[h["sha256"]] = who
        if store:
            want = store.get(h["path"])
            if want is None:
                res.notes.append(f"{who}: {h['path']} is not in the store checksum listing")
            elif want != h["sha256"]:
                res.failures.append(f"{who}: adapter sha256 differs from the store's listing")
        res.notes.append(f"{who}: {h['sha256'][:16]}… ({h['path']})")
    key = ("test_300", "greedy")
    for arm, runs in {**ds.arms, **ds.controls}.items():
        for i, a in enumerate(runs):
            for b in runs[i + 1 :]:
                if key in a.units and key in b.units:
                    ua, ub = a.units[key], b.units[key]
                    same = sum(
                        x == y for x, y in zip(ua.completion_sha, ub.completion_sha, strict=True)
                    )
                    if same == len(ua.completion_sha):
                        res.failures.append(
                            f"{arm}: seeds {a.seed} and {b.seed} give identical greedy test completions"
                        )
    vkey = ("val_mixed_100", "greedy")
    if vkey in ds.base.units:
        bu = ds.base.units[vkey]
        base_s = [
            SimpleNamespace(problem_id=p, completion=c)
            for p, c in zip(bu.problem_ids, bu.completion_sha, strict=True)
        ]
        for run in ds.trained_runs():
            if vkey in run.units:
                u = run.units[vkey]
                new_s = [
                    SimpleNamespace(problem_id=p, completion=c)
                    for p, c in zip(u.problem_ids, u.completion_sha, strict=True)
                ]
                res.failures += [f"{_who(run)}: {i}" for i in check_outputs_differ(base_s, new_s)]
    return res


def check_budgets(ds: Dataset) -> CheckResult:
    res = CheckResult(
        "budgets present",
        f"budgets.json with {', '.join(REQUIRED_BUDGET_KEYS)} for every trained run; GRPO runs consumed "
        f"exactly {GRPO_COMPLETIONS:,} completions (reward_records.jsonl line count) and are marked "
        "result_bearing; the C1 control's training reward averages ≈ 0.5.",
    )
    expected = int(
        ds.cfg.get("grpo_completions", GRPO_COMPLETIONS)
    )  # overridden only by test fixtures
    for run in ds.trained_runs():
        who = _who(run)
        if run.budgets is None:
            res.failures.append(f"{who}: no budgets.json")
            continue
        lacking = [k for k in REQUIRED_BUDGET_KEYS if run.budgets.get(k) is None]
        if lacking:
            res.failures.append(f"{who}: budgets.json lacks {lacking}")
        if (run.meta or {}).get("status") != "finished":
            res.failures.append(f"{who}: training status {(run.meta or {}).get('status')!r}")
        if run.method == "grpo":
            res.failures += check_grpo_reward_budget(run.run_dir, expected=expected)
            if run.budgets.get("result_bearing") is not True:
                res.failures.append(
                    f"{who}: result_bearing is {run.budgets.get('result_bearing')!r}"
                )
            if run.budgets.get("reward") == "random_bernoulli" or "random_reward" in run.rel:
                res.failures += check_c1_reward_near_half(run.run_dir)
    return res


def check_shared_hyperparameters(ds: Dataset) -> CheckResult:
    root = ds.run_root.parent
    res = CheckResult(
        "shared hyperparameters",
        f"grpo_config.json identical across GRPO runs except {', '.join(GRPO_CONFIG_MAY_DIFFER)}; LoRA "
        f"shape ({', '.join(LORA_KEYS)}, target_modules) identical in every adapter_config.json; trainer "
        "and vLLM tokenizer hashes identical everywhere.",
    )
    grpo: dict[str, list[str]] = {}
    lora: dict[str, list[str]] = {}
    toks: dict[str, list[str]] = {}
    for run in ds.trained_runs():
        who = _who(run)
        gc = run.run_dir / "grpo_config.json"
        if run.method == "grpo":
            if not gc.exists():
                res.failures.append(f"{who}: no grpo_config.json")
            else:
                with gc.open(encoding="utf-8") as f:
                    g = {k: v for k, v in json.load(f).items() if k not in GRPO_CONFIG_MAY_DIFFER}
                grpo.setdefault(json.dumps(g, sort_keys=True), []).append(who)
        budgets = run.budgets or {}
        for cand in (budgets.get("final_adapter_trained"), budgets.get("final_adapter")):
            ac = root / str(cand) / "adapter_config.json"
            if cand and ac.exists():
                with ac.open(encoding="utf-8") as f:
                    a = json.load(f)
                shape = {k: a.get(k) for k in LORA_KEYS} | {
                    "target_modules": sorted(a.get("target_modules") or [])
                }
                lora.setdefault(json.dumps(shape, sort_keys=True), []).append(who)
                break
        else:
            res.failures.append(f"{who}: no adapter_config.json")
        for u in run.units.values():
            pair = (u.config.get("tokenizer_sha256_trainer"), u.config.get("tokenizer_sha256_vllm"))
            if pair[0] != pair[1] or pair[0] is None:
                res.failures.append(f"{who} {u.split}/{u.decoding}: tokenizer hashes {pair}")
            toks.setdefault(str(pair[0]), []).append(who)
    for name, groups in (("grpo_config", grpo), ("LoRA shape", lora), ("tokenizer", toks)):
        if len(groups) > 1:
            desc = "; ".join(f"{sorted(set(w))[:3]}" for w in groups.values())
            res.failures.append(f"{name} differs between runs: {desc}")
    if lora:
        res.notes.append(f"LoRA shape in all adapters: {next(iter(lora))}")
    return res


def check_selection_val_only(ds: Dataset) -> CheckResult:
    res = CheckResult(
        "model selection reads val only",
        "Each RFT arm's chosen.json was selected on val_mixed_100 and equals the argmax of sweep.json's "
        "val accuracy under the recorded tie-break (fewer epochs, then lower learning rate). GRPO ran "
        "one fixed recipe: nothing was selected. The analysis itself selects nothing.",
    )
    for arm, rec in ds.selection.items():
        chosen, sweep = rec["chosen"], rec["sweep"]
        if chosen.get("selected_on") != "val_mixed_100":
            res.failures.append(f"{arm}: chosen.json selected_on={chosen.get('selected_on')!r}")
        results = sweep.get("results") or []
        if not results:
            res.failures.append(f"{arm}: sweep.json has no results; selection cannot be re-derived")
            continue
        if any(r.get("selection_split") != "val_mixed_100" for r in results):
            res.failures.append(
                f"{arm}: a sweep result was scored on something other than val_mixed_100"
            )
        best = min(
            results,
            key=lambda r: (-float(r["val_accuracy"]), int(r["epochs"]), float(r["learning_rate"])),
        )
        if not (
            _close(best["learning_rate"], chosen.get("learning_rate"))
            and best["epochs"] == chosen.get("epochs")
        ):
            res.failures.append(
                f"{arm}: chosen lr={chosen.get('learning_rate')} ep={chosen.get('epochs')} but the val "
                f"argmax is lr={best['learning_rate']} ep={best['epochs']}"
            )
        accs = sorted((float(r["val_accuracy"]) for r in results), reverse=True)
        res.notes.append(
            f"{arm}: {len(results)} configs tried on seed 1; chosen lr={chosen.get('learning_rate'):g} "
            f"ep={chosen.get('epochs')} at val {accs[0]:.3f} (n=100); runner-up {accs[1]:.3f}, "
            f"worst {accs[-1]:.3f} — 'best on val', not 'optimal' (PREREGISTRATION §5.1)."
        )
    return res


def check_provenance_notes(ds: Dataset) -> CheckResult:
    res = CheckResult(
        "provenance (listed, not fatal)",
        "Units or training runs recorded with a dirty git tree, and the GPU rate meta.json used.",
    )
    dirty_units = [
        f"{_who(r)} {u.split}/{u.decoding} @ {str(u.meta.get('git_sha'))[:7]}"
        for r, u in _units(ds)
        if u.meta.get("git_dirty")
    ]
    dirty_runs = [_who(r) for r in ds.trained_runs() if (r.meta or {}).get("git_dirty")]
    if dirty_units:
        res.notes.append(
            f"{len(dirty_units)} eval unit(s) record git_dirty=true: " + "; ".join(dirty_units)
        )
    if dirty_runs:
        res.notes.append(f"training runs with git_dirty=true: {', '.join(dirty_runs)}")
    rates = sorted(
        {
            float(u.meta["gpu_rate_usd_per_hour"])
            for _, u in _units(ds)
            if u.meta.get("gpu_rate_usd_per_hour")
        }
    )
    billed = (ds.cfg.get("gpu_rate") or {}).get("billed_usd_per_hour")
    res.notes.append(
        f"meta.json cost fields use ${'/'.join(f'{r:g}' for r in rates)}/h; Lambda billed ${billed}/h. "
        "GPU-hours are the primary quantity; every dollar figure names its rate."
    )
    return res


def check_truncation_flags(ds: Dataset) -> CheckResult:
    res = CheckResult(
        "SPEC §7 truncation and extraction-failure flags (listed, not fatal)",
        "A run with truncation > 5 % on test_300 is flagged and its numbers are not headline numbers; "
        "on ood_hard_200 truncation is reported, not flagged. One line per model/run that has any "
        "flag; flags are carried into every table and contrast that touches them. Under SPEC §5 v1.6 "
        "a truncated completion is always an extraction failure, so the two rates are not "
        "independent evidence.",
    )
    for run in ds.all_runs():
        flagged = []
        for u in run.units.values():
            if u.split == "gsm8k_500":
                continue
            if check_rates(u.metrics, split=u.split):
                m = u.metrics
                flagged.append(
                    f"{u.split}/{u.decoding} trunc {100 * m['truncation_rate']:.1f}% "
                    f"xfail {100 * m['extraction_failure_rate']:.1f}%"
                )
        test = run.units.get(("test_300", "greedy"))
        headline = test is not None and test.metrics["truncation_rate"] > TRUNCATION_MAX
        if flagged:
            res.notes.append(
                f"{_who(run)}{' — NOT HEADLINE (test_300 greedy > 5 %)' if headline else ''}: "
                + "; ".join(flagged)
            )
    for key, runs in ds.arms.items():
        over = [
            r.seed
            for r in runs
            if ("test_300", "greedy") in r.units
            and r.units[("test_300", "greedy")].metrics["truncation_rate"] > TRUNCATION_MAX
        ]
        if over:
            res.notes.append(
                f"{key}: {len(over)}/{len(runs)} seeds exceed 5 % truncation on test_300 greedy (seeds {over})"
            )
    return res


def cross_run_checks(
    ds: Dataset, *, splits_dir: str | Path, cache_dir: Path | None = None
) -> CrossRunReport:
    """All tasks/05 item-1 checks. Any failure must abort the analysis (``report.main`` does)."""
    return CrossRunReport(
        [
            check_units_complete(ds),
            check_exclusions(ds),
            check_unit_integrity(ds),
            check_protocol(ds),
            check_split_digests(ds, splits_dir),
            check_prompt_drift(ds),
            check_final_checkpoints(ds),
            check_seeds_differ(
                ds, cache_path=None if cache_dir is None else cache_dir / "adapter_sha256.json"
            ),
            check_budgets(ds),
            check_shared_hyperparameters(ds),
            check_selection_val_only(ds),
            check_truncation_flags(ds),
            check_provenance_notes(ds),
        ]
    )


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
