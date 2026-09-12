"""``rlordata eval``: base + reference-model evals per ``configs/eval/*.yaml``. AGENT-OWNED; tasks/02a.

For each model × split × decoding it formats prompts (``sampling.prompts``), samples through
the single generation path (``VLLMSampler``, or ``StubSampler`` with ``--stub``), verifies with
``core.verify``, and writes ``<output_dir>/<model>/<split>/<decoding>/``:

    samples.jsonl     one ``types.Sample`` per (problem, sample), all CLAUDE.md fields
    metrics.json      ``core.evaluate.compute_metrics`` + pass@k + threshold flags
    config.yaml, config_hash.txt, meta.json, NOTES.md   (``run_dir.start_run``)

plus ``<output_dir>/summary.md`` / ``summary.json`` (model × split × {greedy, mean@8, pass@8, trunc%}).

Protocol guards (CLAUDE.md): the cap comes from ``configs/locked/cap.yaml`` for every model
(``sampling.cap.resolve_cap``); ``test_300`` is evaluated once per model and re-running it needs
``--force``; thinking is never enabled; instruct models get the locked TEMPLATE inside their chat
template. ``--stub`` never touches a GPU and is the acceptance test for the pipeline layout. Finished units
(``samples.jsonl`` present) are skipped so an interrupted run resumes; ``--models a,b`` restricts a
run to some model ids (one model per process avoids vLLM teardown issues between models).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rlordata.artifacts import sync_run_dir
from rlordata.core.evaluate import compute_metrics, pass_at_k
from rlordata.core.verify import (
    ANSWER_RULE_TEXT,
    EXTRACTION_RULE,
    has_answer_line,
    verify_batch,
)
from rlordata.data.candidates import VAL_CANDIDATES_NAME, val_candidates
from rlordata.data.generator import read_jsonl
from rlordata.envfile import gpu_rate_usd_per_hour, load_env
from rlordata.run_dir import RunHandle, finish_run, format_cost, start_run
from rlordata.sampling.cap import DEFAULT_CAP_PATH, PROVISIONAL_CAP, load_locked_cap, resolve_cap
from rlordata.sampling.prompts import TEMPLATE, chat_template_kwargs, format_prompts
from rlordata.sampling.vllm_sampler import MAX_PROMPT_TOKENS, Completion
from rlordata.types import Problem, Sample

TRUNCATION_FLAG = 0.05  # SPEC §7: > 5 % truncation on test_300 is flagged
# SPEC §7 v1.3: on ood_hard_200 truncation is reported per tier, not flagged (it is a finding, not a defect).
REPORT_ONLY_TRUNCATION_SPLITS = ("ood_hard_200",)
EXTRACTION_FLAG = (
    0.05  # design choice (tasks/02a §8 gives no number); reported, not headline-blocking
)
DEFAULT_EST_TOKENS_PER_SEC = (
    3000.0  # conservative batched vLLM throughput for a 4B model on one H100
)
_SUBSET_RE = re.compile(r"^(?P<split>.+)_first(?P<n>\d+)$")
_TRANSFER_RE = re.compile(r"^(?P<kind>rg|gsm8k)(?:_(?P<task>[a-z_]+))?_(?P<n>\d+)$")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: str  # "base" | "instruct"
    arm: str  # "base" | "ref" | trained-arm name
    lora_path: str | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return model_slug(self.id)

    @staticmethod
    def from_config(entry: dict[str, Any]) -> ModelSpec:
        kind = str(entry["kind"])
        if kind not in ("base", "instruct"):
            raise ValueError(f"model {entry.get('id')!r}: kind must be base|instruct, got {kind!r}")
        return ModelSpec(
            id=str(entry["id"]),
            kind=kind,
            arm=str(entry.get("arm", "base" if kind == "base" else "ref")),
            lora_path=entry.get("lora_path"),
            llm_kwargs=dict(entry.get("vllm_kwargs", {}) or {}),
        )


@dataclass(frozen=True)
class DecodingSpec:
    name: str
    temperature: float
    n: int
    top_p: float = 1.0
    subset: str | None = None  # e.g. "test_300_first100" (pass@k on n=64)
    ks: tuple[int, ...] | None = None

    @staticmethod
    def from_config(name: str, entry: dict[str, Any]) -> DecodingSpec:
        temperature = float(entry.get("temperature", 0.0))
        n = int(entry.get("n", 1))
        if temperature == 0.0 and n != 1:
            raise ValueError(f"decoding {name}: greedy (T=0) must use n=1, got n={n}")
        ks = entry.get("ks")
        return DecodingSpec(
            name=name,
            temperature=temperature,
            n=n,
            top_p=float(entry.get("top_p", 1.0)),
            subset=entry.get("subset"),
            ks=tuple(int(k) for k in ks) if ks else None,
        )


@dataclass
class EvalUnit:
    model: ModelSpec
    split: str  # directory name; for subsets the parent split (e.g. test_300)
    decoding: DecodingSpec
    problems: list[Problem]

    def out_dir(self, root: Path) -> Path:
        return root / self.model.slug / self.split / self.decoding.name

    def run_id(self, seed: int) -> str:
        return f"eval_{self.model.slug}_{self.split}_{self.decoding.name}_seed{seed}"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def parse_subset(subset: str) -> tuple[str, int]:
    """``"test_300_first100"`` -> ``("test_300", 100)``."""
    m = _SUBSET_RE.match(subset)
    if not m:
        raise ValueError(f"unrecognised subset spec {subset!r}; expected '<split>_first<N>'")
    return m.group("split"), int(m.group("n"))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def load_split(
    name: str,
    *,
    splits_dir: Path,
    pool_path: Path,
    ood_path: Path | None = None,
) -> list[Problem]:
    """Resolve a split name to problems, in file order.

    ``val_candidates`` comes from the pool (``data.candidates``); ``data/splits/<name>.jsonl``
    otherwise; ``ood_hard_200`` falls back to the untiered pool file; transfer sets
    (``rg_<task>_<n>``, ``gsm8k_<n>``) come from ``data.transfer``.
    """
    if name == VAL_CANDIDATES_NAME:
        return val_candidates(read_jsonl(pool_path))
    path = splits_dir / f"{name}.jsonl"
    if path.exists():
        return read_jsonl(path)
    if name == "ood_hard_200" and ood_path is not None and Path(ood_path).exists():
        return [Problem(**{**p.to_dict(), "split": name}) for p in read_jsonl(ood_path)]
    m = _TRANSFER_RE.match(name)
    if m:
        from rlordata.data import transfer

        n = int(m.group("n"))
        if m.group("kind") == "gsm8k":
            return transfer.load_gsm8k_test(n=n)
        return transfer.load_reasoning_gym(m.group("task"), n=n)
    raise FileNotFoundError(f"split {name!r}: {path} not found (run `rlordata tier` first)")


def problem_ids_digest(problems: Iterable[Problem]) -> str:
    h = hashlib.sha256()
    for p in problems:
        h.update(p.problem_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_units(
    config: dict[str, Any],
    *,
    splits_dir: Path,
    pool_path: Path,
    ood_path: Path | None,
    n_problems: int | None = None,
) -> list[EvalUnit]:
    """Model-major list of (model, split, decoding) units with their problems loaded."""
    models = [ModelSpec.from_config(m) for m in config["models"]]
    decodings = [
        DecodingSpec.from_config(name, entry) for name, entry in config["decoding"].items()
    ]
    split_names = [str(s) for s in config["splits"]]
    cache: dict[str, list[Problem]] = {}

    def get(name: str) -> list[Problem]:
        if name not in cache:
            probs = load_split(name, splits_dir=splits_dir, pool_path=pool_path, ood_path=ood_path)
            if n_problems is not None:
                probs = probs[: int(n_problems)]
            if not probs:
                raise ValueError(f"split {name!r} is empty")
            cache[name] = probs
        return cache[name]

    units: list[EvalUnit] = []
    for model in models:
        for split in split_names:
            for dec in decodings:
                if dec.subset is None:
                    units.append(EvalUnit(model, split, dec, get(split)))
        for dec in decodings:
            if dec.subset is not None:
                parent, first_n = parse_subset(dec.subset)
                probs = get(parent)[:first_n]
                units.append(EvalUnit(model, parent, dec, probs))
    return units


def resolved_unit_config(
    unit: EvalUnit,
    *,
    seed: int,
    cap: int,
    cap_path: str | Path,
    sampler_desc: dict[str, Any],
    chat_kwargs: dict[str, Any] | None,
    source_config: str,
) -> dict[str, Any]:
    return {
        "kind": "eval",
        "run_id": unit.run_id(seed),
        "source_config": source_config,
        "model": {
            "id": unit.model.id,
            "kind": unit.model.kind,
            "arm": unit.model.arm,
            "lora_path": unit.model.lora_path,
        },
        "chat_template_kwargs": chat_kwargs,
        "thinking": False,
        "split": unit.split,
        "subset": unit.decoding.subset,
        "n_problems": len(unit.problems),
        "problem_ids_sha256": problem_ids_digest(unit.problems),
        "decoding": {
            "name": unit.decoding.name,
            "temperature": unit.decoding.temperature,
            "top_p": unit.decoding.top_p,
            "n": unit.decoding.n,
            "ks": list(unit.decoding.ks) if unit.decoding.ks else None,
            "repetition_penalty": 1.0,
        },
        "seed": seed,
        "max_completion_tokens": cap,
        "cap_yaml": str(cap_path),
        "cap_is_provisional": bool(sampler_desc.get("cap_is_provisional", False)),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "prompt_template": TEMPLATE,
        "answer_regex": ANSWER_RULE_TEXT,
        "extraction_rule": EXTRACTION_RULE,
        "sampler": sampler_desc,
    }


# ---------------------------------------------------------------------------
# Running one unit
# ---------------------------------------------------------------------------


def build_samples(
    problems: list[Problem],
    prompts: list[str],
    completions: list[list[Completion]],
    *,
    run_id: str,
    config_hash: str,
    seed: int,
    arm: str,
    data_condition: str,
    extra: dict[str, Any] | None = None,
) -> list[Sample]:
    """Verify every completion with ``core.verify`` and build ``types.Sample`` records."""
    assert len(problems) == len(prompts) == len(completions)
    out: list[Sample] = []
    base_extra = dict(extra or {})
    for i, (problem, prompt, comps) in enumerate(zip(problems, prompts, completions, strict=True)):
        verdicts = verify_batch(
            [problem] * len(comps),
            [c.text for c in comps],
            [bool(c.truncated) for c in comps],
        )
        for j, (c, v) in enumerate(zip(comps, verdicts, strict=True)):
            out.append(
                Sample(
                    run_id=run_id,
                    config_hash=config_hash,
                    seed=seed,
                    arm=arm,
                    data_condition=data_condition,
                    problem_id=problem.problem_id,
                    tier=problem.tier,
                    prompt=prompt,
                    completion=c.text,
                    extracted_answer=v.extracted,
                    correct=bool(v.reward == 1.0),
                    reward=float(v.reward),
                    n_tokens=int(c.n_tokens),
                    truncated=bool(c.truncated),
                    extraction_failed=bool(v.extraction_failed),
                    extra={
                        **base_extra,
                        "problem_idx": i,
                        "sample_idx": j,
                        "finish_reason": c.finish_reason,
                        "answer": problem.answer,
                        "pass8": problem.pass8,
                        "split": problem.split,
                        "range_scale": problem.range_scale,
                        "total_steps": problem.total_steps,
                    },
                )
            )
    return out


def write_samples(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")


def read_samples(path: Path) -> list[Sample]:
    out: list[Sample] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Sample(**json.loads(line)))
    return out


def _rate(values: list[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def metrics_for(
    samples: list[Sample],
    *,
    ks: Iterable[int] | None = None,
    seed: int = 0,
    cap: int | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """``compute_metrics`` + unbiased pass@k + per-tier truncation / extraction-failure / at-cap rates + flags.

    ``cap`` enables the at-cap rate (completions whose length reached the cap exactly). ``split``
    selects the truncation policy: flagged when > 5 %, except on ``REPORT_ONLY_TRUNCATION_SPLITS``
    where it is reported but never flagged (SPEC §7 v1.3).
    """
    m = compute_metrics(samples, seed=seed)
    counts: dict[str, tuple[int, int]] = {}
    for s in samples:
        n, c = counts.get(s.problem_id, (0, 0))
        counts[s.problem_id] = (n + 1, c + int(s.correct))
    ns = {n for n, _ in counts.values()}
    n_per_problem = ns.pop() if len(ns) == 1 else None
    pass_k: dict[str, float] = {}
    if n_per_problem is not None:
        wanted = [k for k in (ks or (1, 2, 4, 8, 16, 32, 64)) if k <= n_per_problem]
        for k in wanted:
            pass_k[str(k)] = float(
                sum(pass_at_k(n, c, k) for n, c in counts.values()) / len(counts)
            )
    tier_n: dict[str, int] = {}
    seen: set[str] = set()
    by_tier: dict[str, list[Sample]] = {}
    for s in samples:
        by_tier.setdefault(s.tier, []).append(s)
        if s.problem_id not in seen:
            seen.add(s.problem_id)
            tier_n[s.tier] = tier_n.get(s.tier, 0) + 1
    tiers = sorted(by_tier)
    at_cap = None if cap is None else _rate([s.n_tokens >= cap for s in samples])
    over = bool(m.truncation_rate > TRUNCATION_FLAG)
    policy = "report_only" if split in REPORT_ONLY_TRUNCATION_SPLITS else "flag"
    out = asdict(m)
    out.update(
        {
            "n_samples": len(samples),
            "samples_per_problem": n_per_problem,
            "pass_at_k": pass_k,
            "per_tier_n": tier_n,
            "per_tier_n_samples": {t: len(by_tier[t]) for t in tiers},
            "per_tier_truncation_rate": {
                t: _rate([s.truncated for s in by_tier[t]]) for t in tiers
            },
            "per_tier_extraction_failure_rate": {
                t: _rate([s.extraction_failed for s in by_tier[t]]) for t in tiers
            },
            # SPEC §10: format compliance is reported, never rewarded — the fraction of answers
            # read from an explicit answer line (§5 layers (a)/(b)) rather than the fallback (c).
            "answer_line_rate": _rate(
                [has_answer_line(s.completion, truncated=bool(s.truncated)) for s in samples]
            ),
            "per_tier_answer_line_rate": {
                t: _rate(
                    [has_answer_line(s.completion, truncated=bool(s.truncated)) for s in by_tier[t]]
                )
                for t in tiers
            },
            "at_cap_rate": at_cap,
            "per_tier_at_cap_rate": (
                None
                if cap is None
                else {t: _rate([s.n_tokens >= cap for s in by_tier[t]]) for t in tiers}
            ),
            "max_completion_tokens": cap,
            "truncation_over_5pct": over,
            "truncation_flag_policy": policy,
            "flags": {
                "truncation_gt_5pct": bool(over and policy == "flag"),
                "extraction_failure_gt_5pct": bool(m.extraction_failure_rate > EXTRACTION_FLAG),
            },
        }
    )
    return out


def run_unit(
    unit: EvalUnit,
    sampler: Any,
    out_dir: Path,
    *,
    seed: int,
    resolved: dict[str, Any],
    chat_kwargs: dict[str, Any] | None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Sample, verify, write samples.jsonl + metrics.json + provenance. Returns the metrics.

    ``run_id`` overrides the default ``eval_<model>_<split>_<decoding>_seed<seed>`` (trained-arm
    evals name the arm and training run so they never collide with the base eval's ids)."""
    run_id = run_id or unit.run_id(seed)
    handle: RunHandle = start_run(
        out_dir,
        resolved,
        run_id=run_id,
        extra_meta={
            "model_id": unit.model.id,
            "split": unit.split,
            "decoding": unit.decoding.name,
            "n_problems": len(unit.problems),
            "n_samples_planned": len(unit.problems) * unit.decoding.n,
            "sampler": sampler.name,
        },
    )
    prompts = format_prompts(
        unit.problems,
        unit.model.kind,
        sampler.tokenizer,
        model_id=unit.model.id,
        chat_kwargs=chat_kwargs,
    )
    t0 = time.monotonic()
    completions = sampler.sample(
        prompts, n=unit.decoding.n, temperature=unit.decoding.temperature, top_p=unit.decoding.top_p
    )
    sample_s = time.monotonic() - t0
    samples = build_samples(
        unit.problems,
        prompts,
        completions,
        run_id=run_id,
        config_hash=handle.config_hash,
        seed=seed,
        arm=unit.model.arm,
        data_condition=unit.split,
        extra={
            "model_id": unit.model.id,
            "model_kind": unit.model.kind,
            "decoding": unit.decoding.name,
            "temperature": unit.decoding.temperature,
            "top_p": unit.decoding.top_p,
            "subset": unit.decoding.subset,
        },
    )
    write_samples(samples, out_dir / "samples.jsonl")
    metrics = metrics_for(
        samples,
        ks=unit.decoding.ks,
        seed=seed,
        cap=int(resolved["max_completion_tokens"]),
        split=unit.split,
    )
    metrics.update(
        {
            "run_id": run_id,
            "config_hash": handle.config_hash,
            "model_id": unit.model.id,
            "arm": unit.model.arm,
            "split": unit.split,
            "subset": unit.decoding.subset,
            "decoding": unit.decoding.name,
            "temperature": unit.decoding.temperature,
            "n": unit.decoding.n,
            "seed": seed,
            "max_completion_tokens": resolved["max_completion_tokens"],
            "sampler": sampler.name,
        }
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    gen_tokens = sum(s.n_tokens for s in samples)
    finish_run(
        handle,
        n_samples=len(samples),
        sampling_wall_clock_s=round(sample_s, 3),
        generated_tokens=gen_tokens,
        tokens_per_s=round(gen_tokens / sample_s, 1) if sample_s > 0 else None,
        accuracy=metrics["accuracy"],
        truncation_rate=metrics["truncation_rate"],
        extraction_failure_rate=metrics["extraction_failure_rate"],
    )
    if metrics["flags"]["truncation_gt_5pct"]:
        flag = " TRUNCATION>5%"
    elif metrics["truncation_over_5pct"]:
        flag = " trunc>5% (reported, not flagged on this split)"
    else:
        flag = ""
    per_tier = " ".join(
        f"{t[0]}={100 * r:.1f}%" for t, r in sorted(metrics["per_tier_truncation_rate"].items())
    )
    print(
        f"  [{unit.model.slug} | {unit.split} | {unit.decoding.name}] n={metrics['n_problems']} "
        f"acc={metrics['accuracy']:.3f} [{metrics['ci_low']:.3f},{metrics['ci_high']:.3f}] "
        f"trunc={metrics['truncation_rate']:.3f} (per tier: {per_tier}) at_cap={metrics['at_cap_rate']:.3f} "
        f"extract_fail={metrics['extraction_failure_rate']:.3f}{flag}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def collect_metrics(output_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/*/*/metrics.json")):
        with path.open(encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def summarize(metrics: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Main table (model × split: greedy [CI], mean@8, pass@8, trunc %, at-cap %) plus a per-tier
    truncation / extraction-failure table for every split and decoding (SPEC §7 v1.3)."""
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for m in metrics:
        by_key.setdefault((m["model_id"], m["split"]), {})[m["decoding"]] = m
    rows: list[dict[str, Any]] = []
    for (model_id, split), decs in sorted(by_key.items()):
        greedy = decs.get("greedy")
        mean8 = decs.get("mean_at_k")
        passk = decs.get("pass_at_k")
        p8 = None
        if mean8 and "8" in mean8.get("pass_at_k", {}):
            p8 = mean8["pass_at_k"]["8"]
        elif passk and "8" in passk.get("pass_at_k", {}):
            p8 = passk["pass_at_k"]["8"]
        flagged = any(v["flags"]["truncation_gt_5pct"] for v in decs.values())
        reported = any(v.get("truncation_over_5pct") for v in decs.values()) and not flagged
        rows.append(
            {
                "model_id": model_id,
                "split": split,
                "n_problems": (greedy or mean8 or passk or {}).get("n_problems"),
                "greedy": None if not greedy else greedy["accuracy"],
                "greedy_ci": None if not greedy else [greedy["ci_low"], greedy["ci_high"]],
                "mean_at_8": None if not mean8 else mean8["accuracy"],
                "pass_at_8": p8,
                "pass_at_k_n64": None if not passk else passk.get("pass_at_k"),
                "trunc_greedy": None if not greedy else greedy["truncation_rate"],
                "trunc_t1": None if not mean8 else mean8["truncation_rate"],
                "at_cap_greedy": None if not greedy else greedy.get("at_cap_rate"),
                "at_cap_t1": None if not mean8 else mean8.get("at_cap_rate"),
                "extraction_fail_greedy": None if not greedy else greedy["extraction_failure_rate"],
                "per_tier_truncation": {
                    k: v.get("per_tier_truncation_rate") for k, v in decs.items()
                },
                "per_tier_extraction_failure": {
                    k: v.get("per_tier_extraction_failure_rate") for k, v in decs.items()
                },
                "per_tier_at_cap": {k: v.get("per_tier_at_cap_rate") for k, v in decs.items()},
                "config_hashes": {k: v["config_hash"][:12] for k, v in decs.items()},
                "flagged": flagged,
                "truncation_reported_not_flagged": reported,
            }
        )

    def fmt(x: float | None) -> str:
        return "  -  " if x is None else f"{x:.3f}"

    def pct(x: float | None) -> str:
        return "  - " if x is None else f"{100 * x:4.1f}"

    lines = [
        f"{'model':<34} {'split':<16} {'n':>4} {'greedy [95% CI]':<22} {'mean@8':>7} {'pass@8':>7} "
        f"{'trunc% g/T1':>12} {'at-cap% g/T1':>13} flag",
        "-" * 132,
    ]
    for r in rows:
        ci = "" if r["greedy_ci"] is None else f" [{r['greedy_ci'][0]:.3f},{r['greedy_ci'][1]:.3f}]"
        if r["flagged"]:
            flag = "TRUNC>5%"
        elif r["truncation_reported_not_flagged"]:
            flag = "trunc>5% (reported)"
        else:
            flag = ""
        lines.append(
            f"{r['model_id']:<34} {r['split']:<16} {str(r['n_problems']):>4} {fmt(r['greedy']) + ci:<22} "
            f"{fmt(r['mean_at_8']):>7} {fmt(r['pass_at_8']):>7} {pct(r['trunc_greedy'])}/{pct(r['trunc_t1']):<6} "
            f"{pct(r['at_cap_greedy'])}/{pct(r['at_cap_t1']):<7} {flag}"
        )

    # Per-tier table: truncation % / extraction-failure % per tier, per split and decoding.
    seen_tiers = {t for r in rows for d in r["per_tier_truncation"].values() if d for t in d}
    tiers = [t for t in ("easy", "medium", "hard") if t in seen_tiers] + sorted(
        seen_tiers - {"easy", "medium", "hard"}
    )
    lines += [
        "",
        "per-tier truncation % / extraction-failure % (per split and decoding; ood_hard_200 is reported, never flagged)",
        f"{'model':<34} {'split':<16} {'decoding':<10} " + " ".join(f"{t:>13}" for t in tiers),
        "-" * (62 + 14 * max(1, len(tiers))),
    ]
    for r in rows:
        for dec in sorted(r["per_tier_truncation"]):
            tr = r["per_tier_truncation"][dec] or {}
            ef = r["per_tier_extraction_failure"][dec] or {}
            cells = " ".join(
                f"{100 * tr[t]:5.1f}/{100 * ef.get(t, 0.0):<5.1f}" if t in tr else f"{'-':>13}"
                for t in tiers
            )
            lines.append(f"{r['model_id']:<34} {r['split']:<16} {dec:<10} {cells}")
    return "\n".join(lines), rows


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------


def make_sampler(
    model: ModelSpec,
    *,
    stub: bool,
    cap: int,
    seed: int,
    cap_path: str | Path,
    allow_provisional_cap: bool,
    problems: list[Problem],
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.9,
) -> Any:
    if stub:
        from rlordata.sampling.stub_sampler import StubSampler

        return StubSampler.from_problems(
            problems,
            model_id=model.id,
            model_kind=model.kind,  # type: ignore[arg-type]
            lora_path=model.lora_path,
            max_completion_tokens=cap,
            seed=seed,
            allow_provisional_cap=allow_provisional_cap,
            cap_path=cap_path,
        )
    from rlordata.sampling.vllm_sampler import VLLMSampler

    return VLLMSampler(
        model.id,
        model.kind,  # type: ignore[arg-type]
        lora_path=model.lora_path,
        max_completion_tokens=cap,
        seed=seed,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        allow_provisional_cap=allow_provisional_cap,
        cap_path=cap_path,
        llm_kwargs=model.llm_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _estimate_gpu_hours(units: list[EvalUnit], cap: int, tokens_per_s: float) -> float:
    worst_tokens = sum(len(u.problems) * u.decoding.n * cap for u in units)
    return worst_tokens / max(tokens_per_s, 1e-9) / 3600.0


def cli_main(args: Any) -> int:
    """Entry point for ``rlordata eval`` (see ``cli.py`` for the flags)."""
    load_env()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    stub = bool(getattr(args, "stub", False))
    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))
    n_problems = getattr(args, "n_problems", None)
    seed = int(args.seed if getattr(args, "seed", None) is not None else config.get("seed", 1))
    output_dir = Path(getattr(args, "output_dir", None) or config.get("output_dir", "runs/eval/"))
    splits_dir = Path(getattr(args, "splits_dir", None) or config.get("splits_dir", "data/splits/"))
    pool_path = Path(getattr(args, "pool", None) or config.get("pool", "data/pool/pool.jsonl"))
    ood_path = Path(config.get("ood_path", "data/pool/ood_hard_200.jsonl"))
    cap_path = Path(config.get("cap_yaml", DEFAULT_CAP_PATH))
    est_tps = float(config.get("est_tokens_per_sec", DEFAULT_EST_TOKENS_PER_SEC))

    # --- cap (SPEC §7) -------------------------------------------------------
    provisional_cfg = config.get("provisional_cap")
    allow_provisional = False
    if provisional_cfg is not None:
        # The one run allowed before cap.yaml exists. resolve_cap raises if a locked cap exists
        # and differs, so this config cannot be re-run once the cap is locked.
        allow_provisional = True
        cap = resolve_cap(int(provisional_cfg), cap_path=cap_path, allow_provisional=True)
        print(
            f"PROVISIONAL CAP RUN: max_completion_tokens={cap} (cap.yaml absent; this run defines it)"
        )
    elif stub and load_locked_cap(cap_path) is None:
        allow_provisional = True
        cap = PROVISIONAL_CAP
        print(
            "="
            * 78
            + f"\nDRY RUN (stub sampler): {cap_path} does not exist; using provisional cap {cap}.\n"
            "Nothing here is a result. The real sampler refuses to run without cap.yaml.\n"
            + "="
            * 78
        )
    else:
        cap = resolve_cap(None, cap_path=cap_path)  # raises CapError if cap.yaml is missing
    if stub:
        print("STUB SAMPLER: scripted completions, no GPU, no model weights. Not a result.")

    # --- plan --------------------------------------------------------------
    units = plan_units(
        config, splits_dir=splits_dir, pool_path=pool_path, ood_path=ood_path, n_problems=n_problems
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional model filter (run one model per process; finished units are skipped anyway).
    only = getattr(args, "models", None)
    if only:
        wanted = {m.strip() for m in str(only).split(",") if m.strip()}
        unknown = wanted - {u.model.id for u in units}
        if unknown:
            print(f"--models: unknown model id(s) {sorted(unknown)}", file=sys.stderr)
            return 2
        units = [u for u in units if u.model.id in wanted]

    # Resume semantics: a unit whose samples.jsonl exists is skipped. test_300 is evaluated once
    # per model (SPEC §10 / tasks/02): its finished units are never re-sampled without --force,
    # and --force says so loudly.
    done = [u for u in units if (u.out_dir(output_dir) / "samples.jsonl").exists()]
    test_done = [u for u in done if u.split == "test_300"]
    if force and test_done:
        print(
            f"WARNING: --force re-samples {len(test_done)} finished test_300 unit(s); test_300 is meant to be "
            "evaluated once per model (SPEC §10). Only do this deliberately.",
            file=sys.stderr,
        )
    todo = [u for u in units if force or u not in done]
    skipped = [u for u in units if u not in todo]
    rate = gpu_rate_usd_per_hour()
    est_hours = _estimate_gpu_hours(todo, cap, est_tps)
    n_comp = sum(len(u.problems) * u.decoding.n for u in todo)
    print(
        f"plan: {len(units)} unit(s), {len(todo)} to run, {len(skipped)} already done (resume) | "
        f"seed={seed} cap={cap} sampler={'stub' if stub else 'vllm'} output={output_dir}"
    )
    for u in units:
        if u in todo:
            mark = "run"
        elif u.split == "test_300":
            mark = "skip (done; test_300 is evaluated once per model — --force to redo)"
        else:
            mark = "skip (done)"
        print(
            f"  {u.model.slug:<34} {u.split:<16} {u.decoding.name:<10} n_problems={len(u.problems):<4} "
            f"n={u.decoding.n}  {mark}"
        )
    print(
        f"cost estimate (START, worst case: every completion hits the cap; assumes {est_tps:.0f} tok/s): "
        f"{n_comp} completions → {format_cost(est_hours, rate)}"
    )
    if dry_run:
        print("dry-run: not sampling")
        return 0

    t_start = time.monotonic()
    status = "finished"
    try:
        seen_models: list[ModelSpec] = []
        for u in todo:
            if u.model not in seen_models:
                seen_models.append(u.model)
        for model in seen_models:
            model_units = [u for u in todo if u.model == model]
            all_problems = [p for u in model_units for p in u.problems]
            print(
                f"\n== model {model.id} ({model.kind}, arm={model.arm}) — {len(model_units)} unit(s)"
            )
            chat_kwargs = chat_template_kwargs(model.id) if model.kind == "instruct" else None
            sampler = make_sampler(
                model,
                stub=stub,
                cap=cap,
                seed=seed,
                cap_path=cap_path,
                allow_provisional_cap=allow_provisional,
                problems=all_problems,
                dtype=str(config.get("dtype", "bfloat16")),
                gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            )
            try:
                for u in model_units:
                    resolved = resolved_unit_config(
                        u,
                        seed=seed,
                        cap=cap,
                        cap_path=cap_path,
                        sampler_desc=sampler.describe(),
                        chat_kwargs=chat_kwargs,
                        source_config=str(config_path),
                    )
                    out_dir = u.out_dir(output_dir)
                    run_unit(
                        u, sampler, out_dir, seed=seed, resolved=resolved, chat_kwargs=chat_kwargs
                    )
                    sync_run_dir(out_dir, quiet=True)
            finally:
                sampler.close()
    except BaseException:
        status = "failed"
        raise
    finally:
        table, rows = summarize(collect_metrics(output_dir))
        (output_dir / "summary.md").write_text(
            f"# eval summary — {output_dir}\n\nstatus: {status}\n\n```\n{table}\n```\n",
            encoding="utf-8",
        )
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"status": status, "seed": seed, "cap": cap, "rows": rows},
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        print("\n" + table)
        hours = (time.monotonic() - t_start) / 3600.0
        print(f"cost estimate (END, actual wall-clock): {format_cost(hours, rate)}")
        sync_run_dir(output_dir)
    return 0
