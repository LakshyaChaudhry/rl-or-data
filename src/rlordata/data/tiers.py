"""Difficulty tiering by base-model pass@8 and split construction (SPEC §6). AGENT-OWNED; tasks/01, 02a.

Contract:
    tier_from_pass8(pass8: int) -> Tier
    build_splits(pool_with_pass8, seed) -> dict[str, list[Problem]]
    structure_id(pipeline) — canonical_id of pipeline with range removed
    sample_pass8(problems, sampler, k=8, ...) -> (tiered problems, all k completions in order)
    cli_main(args) — ``rlordata tier`` (live pass@8 through the sampler; ``--stub`` for dry runs)
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.artifacts import sync_run_dir
from rlordata.core.verify import ANSWER_RE
from rlordata.data.generator import canonical_id, read_jsonl, write_jsonl
from rlordata.envfile import gpu_rate_usd_per_hour, load_env
from rlordata.run_dir import finish_run, format_cost, start_run
from rlordata.sampling.cap import DEFAULT_CAP_PATH, PROVISIONAL_CAP, load_locked_cap, resolve_cap
from rlordata.sampling.eval_runner import ModelSpec, build_samples, make_sampler
from rlordata.sampling.prompts import TEMPLATE, format_prompt
from rlordata.sampling.vllm_sampler import MAX_PROMPT_TOKENS
from rlordata.types import Problem, Sample, Tier

# Default thresholds (SPEC §6 / configs/data/tiering.yaml).
_EASY_MIN = 6
_MEDIUM_MIN = 2


def tier_from_pass8(
    pass8: int,
    *,
    easy_min_pass8: int = _EASY_MIN,
    medium_min_pass8: int = _MEDIUM_MIN,
) -> Tier:
    """Map pass@8 count to easy / medium / hard."""
    if pass8 < 0 or pass8 > 8:
        raise ValueError(f"pass8 must be in 0..8, got {pass8}")
    if pass8 >= easy_min_pass8:
        return "easy"
    if pass8 >= medium_min_pass8:
        return "medium"
    return "hard"


def structure_id(pipeline: dict[str, Any]) -> str:
    """Canonical id of the pipeline with ``range`` removed (structure disjointness)."""
    without_range = {k: v for k, v in pipeline.items() if k != "range"}
    return canonical_id(without_range)


def interleave_tiers(problems: list[Problem]) -> list[Problem]:
    """Round-robin the problems across tiers (easy, medium, hard, easy, ...), keeping the
    within-tier order. Any prefix of the result is then tier-stratified to within one problem,
    which is what "the first 100 of test_300 by index" (pass@k subset, SPEC §10) relies on."""
    order = ("easy", "medium", "hard")
    queues: dict[str, list[Problem]] = {t: [p for p in problems if p.tier == t] for t in order}
    others = [p for p in problems if p.tier not in order]
    out: list[Problem] = []
    while any(queues.values()):
        for t in order:
            if queues[t]:
                out.append(queues[t].pop(0))
    return out + others


def build_splits(
    pool_with_pass8: list[Problem],
    seed: int,
    *,
    split_spec: dict[str, Any] | None = None,
    easy_min_pass8: int = _EASY_MIN,
    medium_min_pass8: int = _MEDIUM_MIN,
) -> dict[str, list[Problem]]:
    """Build SPEC §6 splits; disjoint by ``problem_id`` and by ``structure_id``.

    ``train_curated`` is derived from ``train_mixed_100`` (1 ≤ pass8 ≤ 7), not sampled. Each
    split is emitted tier-interleaved (:func:`interleave_tiers`) so that a by-index prefix is
    stratified.
    """
    if split_spec is None:
        split_spec = {
            "train_easy_100": {"easy": 100},
            "train_mixed_100": {"easy": 33, "medium": 33, "hard": 34},
            "val_mixed_100": {"easy": 33, "medium": 33, "hard": 34},
            "test_300": {"easy": 100, "medium": 100, "hard": 100},
            "train_curated": {
                "derived_from": "train_mixed_100",
                "pass8_min": 1,
                "pass8_max": 7,
            },
        }

    rng = np.random.default_rng(seed)

    # Ensure every problem has tier + pass8.
    enriched: list[Problem] = []
    for p in pool_with_pass8:
        if p.pass8 is None:
            raise ValueError(f"problem {p.problem_id} missing pass8")
        tier = tier_from_pass8(
            p.pass8,
            easy_min_pass8=easy_min_pass8,
            medium_min_pass8=medium_min_pass8,
        )
        if p.tier != tier:
            p = Problem(**{**p.to_dict(), "tier": tier})
        enriched.append(p)

    by_tier: dict[str, list[Problem]] = defaultdict(list)
    for p in enriched:
        by_tier[p.tier].append(p)
    for tier in by_tier:
        # Deterministic shuffle within tier.
        idxs = rng.permutation(len(by_tier[tier]))
        by_tier[tier] = [by_tier[tier][int(i)] for i in idxs]

    used_ids: set[str] = set()
    used_structures: set[str] = set()
    splits: dict[str, list[Problem]] = {}

    # Sample primary splits before derived ones. Prefer test/val before train so
    # evaluation sets get priority when structure collisions bite.
    order = [
        "test_300",
        "val_mixed_100",
        "train_mixed_100",
        "train_easy_100",
        "train_medium_100",
        "train_hard_100",
        "train_easy_500",
        "train_mixed_500",
    ]
    sampled_keys = [k for k in order if k in split_spec and "derived_from" not in split_spec[k]]
    # Any other non-derived keys not listed.
    for key, spec in split_spec.items():
        if "derived_from" in spec:
            continue
        if key not in sampled_keys:
            sampled_keys.append(key)

    for key in sampled_keys:
        spec = split_spec[key]
        chosen: list[Problem] = []
        # Canonical tier order so the split never depends on the config's key order
        # (a YAML round-trip with sorted keys used to change which problems were picked).
        for tier_name in ("easy", "medium", "hard"):
            if tier_name not in spec:
                continue
            need_n = int(spec[tier_name])
            picked = 0
            remaining: list[Problem] = []
            for p in by_tier[tier_name]:
                if picked >= need_n:
                    remaining.append(p)
                    continue
                sid = structure_id(p.pipeline)
                if p.problem_id in used_ids or sid in used_structures:
                    remaining.append(p)
                    continue
                tagged = Problem(**{**p.to_dict(), "split": key, "tier": tier_name})  # type: ignore[arg-type]
                chosen.append(tagged)
                used_ids.add(p.problem_id)
                used_structures.add(sid)
                picked += 1
            by_tier[tier_name] = remaining
            if picked < need_n:
                raise RuntimeError(
                    f"split {key}: need {need_n} {tier_name}, only got {picked} "
                    f"after structure/id disjointness"
                )
        splits[key] = interleave_tiers(chosen)

    for key, spec in split_spec.items():
        if "derived_from" not in spec:
            continue
        parent = splits[spec["derived_from"]]
        lo = int(spec.get("pass8_min", 1))
        hi = int(spec.get("pass8_max", 7))
        curated = [
            Problem(**{**p.to_dict(), "split": key})
            for p in parent
            if p.pass8 is not None and lo <= p.pass8 <= hi
        ]
        splits[key] = curated

    return splits


# ---------------------------------------------------------------------------
# Live pass@8 path (tasks/02a §6)
# ---------------------------------------------------------------------------


def sample_pass8(
    problems: list[Problem],
    sampler: Any,
    *,
    k: int,
    temperature: float,
    top_p: float,
    run_id: str,
    config_hash: str,
    seed: int,
    arm: str = "base",
    data_condition: str = "pool",
    easy_min_pass8: int = _EASY_MIN,
    medium_min_pass8: int = _MEDIUM_MIN,
) -> tuple[list[Problem], list[Sample]]:
    """Sample ``k`` completions per problem (base template, one sampler call), verify, tier.

    Returns the problems with ``pass8`` / ``tier`` filled and every completion as a
    ``types.Sample`` in generation order (``extra.sample_idx``), which ``train_curated`` and RFT
    read later.
    """
    assert k >= 1 and len(problems) > 0
    if getattr(sampler, "model_kind", "base") != "base":
        raise ValueError("tiering uses the base policy with the plain TEMPLATE (SPEC §6.1)")
    prompts = [format_prompt(p, "base") for p in problems]
    completions = sampler.sample(prompts, n=k, temperature=temperature, top_p=top_p)
    samples = build_samples(
        problems,
        prompts,
        completions,
        run_id=run_id,
        config_hash=config_hash,
        seed=seed,
        arm=arm,
        data_condition=data_condition,
        extra={
            "model_id": sampler.model_id,
            "decoding": "pass8",
            "temperature": temperature,
            "top_p": top_p,
            "k": k,
        },
    )
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.problem_id] = counts.get(s.problem_id, 0) + int(s.correct)
    tiered = [
        Problem(
            **{
                **p.to_dict(),
                "pass8": counts.get(p.problem_id, 0),
                "tier": tier_from_pass8(
                    counts.get(p.problem_id, 0),
                    easy_min_pass8=easy_min_pass8,
                    medium_min_pass8=medium_min_pass8,
                ),
            }
        )
        for p in problems
    ]
    return tiered, samples


def _pass8_from_samples(
    problems: list[Problem],
    samples: list[Sample],
    *,
    k: int,
    easy_min: int,
    medium_min: int,
    label: str,
) -> list[Problem]:
    """Tier ``problems`` from stored (rescored) samples; every problem must have exactly ``k``."""
    counts: dict[str, int] = {}
    n_seen: dict[str, int] = {}
    for s in samples:
        n_seen[s.problem_id] = n_seen.get(s.problem_id, 0) + 1
        counts[s.problem_id] = counts.get(s.problem_id, 0) + int(s.correct)
    bad = [p.problem_id for p in problems if n_seen.get(p.problem_id, 0) != k]
    if bad:
        raise SystemExit(
            f"{label}: {len(bad)} problem(s) do not have exactly {k} stored samples (first {bad[0][:12]}); "
            "the samples file does not match the pool"
        )
    return [
        Problem(
            **{
                **p.to_dict(),
                "pass8": counts.get(p.problem_id, 0),
                "tier": tier_from_pass8(
                    counts.get(p.problem_id, 0),
                    easy_min_pass8=easy_min,
                    medium_min_pass8=medium_min,
                ),
            }
        )
        for p in problems
    ]


def tier_counts(problems: list[Problem]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in problems:
        out[p.tier] = out.get(p.tier, 0) + 1
    return dict(sorted(out.items()))


def pass8_histogram(problems: list[Problem], *, k: int = 8, width: int = 40) -> str:
    counts = [0] * (k + 1)
    for p in problems:
        if p.pass8 is not None:
            counts[int(p.pass8)] += 1
    scale = max(1, max(counts))
    lines = [f"pass@{k} histogram (n={sum(counts)}):"]
    for i, c in enumerate(counts):
        lines.append(f"  {i:>2}/{k}  {c:>6}  {'#' * int(round(width * c / scale))}")
    return "\n".join(lines)


def _write_samples_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")


def cli_main(args: Any) -> int:
    """``rlordata tier --config configs/data/tiering.yaml [--stub] [--output-dir D] [--samples-output F]``.

    If the input pool already carries ``pass8`` on every problem, only the splits are built.
    Otherwise the base policy is sampled k times per problem (vLLM, or the stub with ``--stub``),
    the completions are stored in generation order, the pool is tiered and the SPEC §6 splits are
    written. ``post_hoc_tier_inputs`` (e.g. ``ood_hard_200``) are tiered but not filtered.

    ``--provisional-cap N`` samples before cap.yaml exists (splits not final);
    ``--rescore-from samples.jsonl`` re-verifies stored completions with the current verifier and
    the locked cap and rebuilds the splits without sampling.
    """
    load_env()
    with Path(args.config).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pool_path = Path(config["input"])
    pool = read_jsonl(pool_path)
    seed = int(args.seed if args.seed is not None else config.get("seed", 1))
    tiers_cfg = config.get("tiers", {})
    easy_min = int(tiers_cfg.get("easy_min_pass8", _EASY_MIN))
    medium_min = int(tiers_cfg.get("medium_min_pass8", _MEDIUM_MIN))
    k = int(config.get("k", 8))
    temperature = float(config.get("temperature", 1.0))
    top_p = float(config.get("top_p", 1.0))
    model_id = str(config.get("model_id", "Qwen/Qwen3-4B-Base"))
    model_kind = str(config.get("model_kind", "base"))
    if model_kind != "base":
        raise SystemExit("tiering must use the base policy (SPEC §6.1)")
    stub = bool(getattr(args, "stub", False))
    dry_run = bool(getattr(args, "dry_run", False))
    out_dir = Path(getattr(args, "output_dir", None) or config.get("output_dir", "data/splits/"))
    samples_out = Path(
        getattr(args, "samples_output", None)
        or config.get("samples_output", "data/samples/tiering_pass8.jsonl")
    )
    run_root = Path(getattr(args, "run_dir", None) or config.get("run_dir", "runs/tier/"))
    cap_path = Path(config.get("cap_yaml", DEFAULT_CAP_PATH))
    post_hoc_inputs = [Path(p) for p in config.get("post_hoc_tier_inputs", []) or []]
    extra_pools: dict[str, list[Problem]] = {}
    for p in post_hoc_inputs:
        if p.exists():
            extra_pools[p.stem] = read_jsonl(p)
        else:
            print(f"post-hoc input {p} not found; skipping", file=sys.stderr)

    missing = [p for p in pool if p.pass8 is None]
    need_sampling = bool(missing) or any(
        any(q.pass8 is None for q in v) for v in extra_pools.values()
    )
    n_to_sample = len(pool) if missing else 0
    n_to_sample += sum(len(v) for v in extra_pools.values() if any(q.pass8 is None for q in v))

    print(
        f"tier: pool={pool_path} n={len(pool)} k={k} T={temperature} top_p={top_p} seed={seed} "
        f"model={model_id} sampler={'stub' if stub else 'vllm'} need_sampling={need_sampling} "
        f"post_hoc={list(extra_pools)}"
    )
    if dry_run:
        print(f"dry-run: would sample {n_to_sample} problems × {k}; not writing anything")
        return 0

    samples: list[Sample] = []
    handle = None
    rescore_from = getattr(args, "rescore_from", None)
    provisional_cap = getattr(args, "provisional_cap", None)
    if rescore_from:
        # Offline path: pass8 from stored completions, re-verified with the current core.verify and
        # the LOCKED cap (simulated on completions longer than it). No sampling.
        from rlordata.sampling.rescore import answers_from_pools, rescore_file, sha256_file

        cap = resolve_cap(None, cap_path=cap_path)  # the real cap must exist for real splits
        src = Path(rescore_from)
        answers = answers_from_pools([pool, *extra_pools.values()])
        rescored, rescore_summary = rescore_file(src, answers=answers, cap=cap)
        if src.resolve() != samples_out.resolve():
            _write_samples_jsonl(rescored, samples_out)
        samples = rescored
        by_condition: dict[str, list[Sample]] = {}
        for s in samples:
            by_condition.setdefault(s.data_condition, []).append(s)
        pool = _pass8_from_samples(
            pool,
            by_condition.get("pool", []),
            k=k,
            easy_min=easy_min,
            medium_min=medium_min,
            label="pool",
        )
        for name in list(extra_pools):
            extra_pools[name] = _pass8_from_samples(
                extra_pools[name],
                by_condition.get(name, []),
                k=k,
                easy_min=easy_min,
                medium_min=medium_min,
                label=name,
            )
        run_id = f"tier_rescore_{model_id.replace('/', '__')}_k{k}_seed{seed}"
        resolved = {
            "kind": "tier_rescore",
            "run_id": run_id,
            "source_config": str(args.config),
            "rescore_from": str(src),
            "rescore_from_sha256": sha256_file(src.with_name(src.stem + ".raw.jsonl")),
            "model": {"id": model_id, "kind": "base", "arm": "base"},
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_completion_tokens": cap,
            "cap_yaml": str(cap_path),
            "cap_is_provisional": False,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "prompt_template": TEMPLATE,
            "answer_regex": ANSWER_RE.pattern,
            "thinking": False,
            "tiers": {"easy_min_pass8": easy_min, "medium_min_pass8": medium_min},
            "splits": config.get("splits"),
            "input": str(pool_path),
            "n_pool": len(pool),
            "post_hoc_tier_inputs": {name: len(v) for name, v in extra_pools.items()},
            "output_dir": str(out_dir),
            "samples_output": str(samples_out),
            "rescore": rescore_summary,
        }
        handle = start_run(
            run_root / run_id,
            resolved,
            run_id=run_id,
            extra_meta={"k": k, "rescore": rescore_summary},
        )
        print(
            f"rescored {rescore_summary['n_samples']} stored completions with the current verifier at cap {cap}: "
            f"verdicts changed {rescore_summary['n_verdicts_changed']}, cap-truncated {rescore_summary['n_cap_truncated']} "
            f"({rescore_summary['frac_cap_truncated']:.2%}), accuracy {rescore_summary['accuracy_before']:.3f} -> "
            f"{rescore_summary['accuracy_after']:.3f}, extraction failure {rescore_summary['extraction_failure_before']:.3f} -> "
            f"{rescore_summary['extraction_failure_after']:.3f}"
        )
        _write_samples_jsonl(samples, handle.run_dir / "tiering_pass8.jsonl")
        need_sampling = False
    if need_sampling:
        if provisional_cap is not None:
            # Sample before the cap is locked. Valid because sampling is batch-invariant and seeded:
            # a completion's first `cap` tokens do not depend on max_tokens, so `--rescore-from` can
            # apply the locked cap afterwards. Splits written here are NOT final.
            cap = resolve_cap(int(provisional_cap), cap_path=cap_path, allow_provisional=True)
            allow_provisional = True
            if getattr(args, "output_dir", None) is None:
                out_dir = out_dir.parent / (out_dir.name.rstrip("/") + "_provisional")
            print(
                "="
                * 78
                + f"\nPROVISIONAL TIERING at cap {cap} ({cap_path} absent). Completions -> {samples_out}.\n"
                f"Splits -> {out_dir} are NOT final: once cap.yaml exists run\n"
                f"  rlordata tier --config {args.config} --rescore-from {samples_out}\n" + "=" * 78
            )
        elif stub and load_locked_cap(cap_path) is None:
            cap = PROVISIONAL_CAP
            allow_provisional = True
            print(
                "=" * 78
                + f"\nDRY RUN (stub sampler): {cap_path} missing; provisional cap {cap}. Not a result.\n"
                + "=" * 78
            )
        else:
            cap = resolve_cap(None, cap_path=cap_path)  # cap.yaml must exist before tiering
            allow_provisional = False
        if stub:
            print("STUB SAMPLER: scripted completions, no GPU. Not a result.")
        all_problems = pool + [q for v in extra_pools.values() for q in v]
        sampler = make_sampler(
            ModelSpec(id=model_id, kind="base", arm="base"),
            stub=stub,
            cap=cap,
            seed=seed,
            cap_path=cap_path,
            allow_provisional_cap=allow_provisional,
            problems=all_problems,
        )
        run_id = (
            f"tier_{model_id.replace('/', '__')}_k{k}_seed{seed}"
            + ("_stub" if stub else "")
            + ("_provisional" if provisional_cap is not None and not stub else "")
        )
        resolved = {
            "kind": "tier",
            "run_id": run_id,
            "source_config": str(args.config),
            "model": {"id": model_id, "kind": "base", "arm": "base"},
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_completion_tokens": cap,
            "cap_yaml": str(cap_path),
            "cap_is_provisional": allow_provisional,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "prompt_template": TEMPLATE,
            "answer_regex": ANSWER_RE.pattern,
            "thinking": False,
            "tiers": {"easy_min_pass8": easy_min, "medium_min_pass8": medium_min},
            "splits": config.get("splits"),
            "input": str(pool_path),
            "n_pool": len(pool),
            "post_hoc_tier_inputs": {name: len(v) for name, v in extra_pools.items()},
            "output_dir": str(out_dir),
            "samples_output": str(samples_out),
            "sampler": sampler.describe(),
        }
        handle = start_run(
            run_root / run_id,
            resolved,
            run_id=run_id,
            extra_meta={"n_to_sample": n_to_sample, "k": k},
        )
        rate = gpu_rate_usd_per_hour()
        est_hours = n_to_sample * k * cap / 3000.0 / 3600.0
        print(
            f"cost estimate (START, worst case, 3000 tok/s): {n_to_sample * k} completions → {format_cost(est_hours, rate)}"
        )
        try:
            if missing:
                pool, pool_samples = sample_pass8(
                    pool,
                    sampler,
                    k=k,
                    temperature=temperature,
                    top_p=top_p,
                    run_id=run_id,
                    config_hash=handle.config_hash,
                    seed=seed,
                    data_condition="pool",
                    easy_min_pass8=easy_min,
                    medium_min_pass8=medium_min,
                )
                samples.extend(pool_samples)
            for name, probs in list(extra_pools.items()):
                if any(q.pass8 is None for q in probs):
                    tiered, extra_samples = sample_pass8(
                        probs,
                        sampler,
                        k=k,
                        temperature=temperature,
                        top_p=top_p,
                        run_id=run_id,
                        config_hash=handle.config_hash,
                        seed=seed,
                        data_condition=name,
                        easy_min_pass8=easy_min,
                        medium_min_pass8=medium_min,
                    )
                    extra_pools[name] = tiered
                    samples.extend(extra_samples)
        finally:
            sampler.close()
        _write_samples_jsonl(samples, samples_out)
        _write_samples_jsonl(samples, handle.run_dir / "tiering_pass8.jsonl")
        print(f"wrote {len(samples)} completions (generation order) -> {samples_out}")

    enriched = [
        Problem(
            **{
                **p.to_dict(),
                "tier": tier_from_pass8(
                    int(p.pass8), easy_min_pass8=easy_min, medium_min_pass8=medium_min
                ),
            }
        )
        for p in pool
    ]
    splits = build_splits(
        enriched,
        seed=seed,
        split_spec=config.get("splits"),
        easy_min_pass8=easy_min,
        medium_min_pass8=medium_min,
    )
    for name, probs in extra_pools.items():
        splits[name] = [Problem(**{**q.to_dict(), "split": name}) for q in probs]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(enriched, out_dir / "pool_tiered.jsonl")
    print(f"wrote tiered pool ({len(enriched)}) -> {out_dir / 'pool_tiered.jsonl'}")
    for name, problems in splits.items():
        write_jsonl(problems, out_dir / f"{name}.jsonl")
        print(
            f"wrote {len(problems):>5} -> {out_dir / f'{name}.jsonl'}  tiers={tier_counts(problems)}"
        )

    print(f"\npool tier counts: {tier_counts(enriched)}")
    print(pass8_histogram(enriched, k=k))
    for name, probs in extra_pools.items():
        print(f"\n{name} tier counts (post hoc, unfiltered): {tier_counts(probs)}")
        print(pass8_histogram(probs, k=k))
    curated = splits.get("train_curated", [])
    print(f"\ntrain_curated: {len(curated)} prompts, tiers={tier_counts(curated)}")

    from rlordata.analysis.sanity import check_split_disjointness, format_problems

    issues = check_split_disjointness(splits)
    print(format_problems("split disjointness", issues))

    summary = {
        "pool_tier_counts": tier_counts(enriched),
        "pass8_counts": {str(i): sum(1 for p in enriched if p.pass8 == i) for i in range(k + 1)},
        "splits": {name: {"n": len(v), "tiers": tier_counts(v)} for name, v in splits.items()},
        "disjointness_issues": issues,
    }
    if handle is not None:
        with (handle.run_dir / "tier_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        for name in list(splits) + ["pool_tiered"]:
            shutil.copy2(out_dir / f"{name}.jsonl", handle.run_dir / f"{name}.jsonl")
        finish_run(
            handle,
            n_samples=len(samples),
            status="finished" if not issues else "finished_with_issues",
        )
        print(
            f"cost estimate (END, actual): {format_cost(handle.elapsed_s / 3600.0, gpu_rate_usd_per_hour())}"
        )
        sync_run_dir(handle.run_dir)
        sync_run_dir(samples_out)
    sync_run_dir(out_dir)
    return 1 if issues else 0
