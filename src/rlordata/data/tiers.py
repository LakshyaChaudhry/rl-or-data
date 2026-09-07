"""Difficulty tiering by base-model pass@8 and split construction (SPEC §6). AGENT-OWNED; tasks/01.

Contract:
    tier_from_pass8(pass8: int) -> Tier
    build_splits(pool_with_pass8, seed) -> dict[str, list[Problem]]
    structure_id(pipeline) — canonical_id of pipeline with range removed
    cli_main(args) — ``rlordata tier`` (needs sampler from tasks/02 for live pass@8)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.data.generator import canonical_id, read_jsonl, write_jsonl
from rlordata.types import Problem, Tier

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


def build_splits(
    pool_with_pass8: list[Problem],
    seed: int,
    *,
    split_spec: dict[str, Any] | None = None,
    easy_min_pass8: int = _EASY_MIN,
    medium_min_pass8: int = _MEDIUM_MIN,
) -> dict[str, list[Problem]]:
    """Build SPEC §6 splits; disjoint by ``problem_id`` and by ``structure_id``.

    ``train_curated`` is derived from ``train_mixed_100`` (1 ≤ pass8 ≤ 7), not sampled.
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
        for tier_name, need in spec.items():
            if tier_name not in ("easy", "medium", "hard"):
                continue
            need_n = int(need)
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
        splits[key] = chosen

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


def cli_main(args: Any) -> int:
    """``rlordata tier --config configs/data/tiering.yaml``.

    Live pass@8 sampling requires ``rlordata.sampling.vllm_sampler`` (tasks/02).
    Until then, if the input JSONL already has ``pass8`` set, we only build splits.
    """
    with Path(args.config).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pool_path = Path(config["input"])
    pool = read_jsonl(pool_path)
    seed = int(args.seed if args.seed is not None else config.get("seed", 1))
    tiers_cfg = config.get("tiers", {})
    easy_min = int(tiers_cfg.get("easy_min_pass8", _EASY_MIN))
    medium_min = int(tiers_cfg.get("medium_min_pass8", _MEDIUM_MIN))

    missing = [p for p in pool if p.pass8 is None]
    if missing:
        try:
            from rlordata.sampling.vllm_sampler import VLLMSampler  # type: ignore
        except (ImportError, AttributeError) as exc:
            raise SystemExit(
                "pass8 missing on pool and vllm_sampler is not implemented yet "
                "(tasks/02). Pre-populate pass8 or finish the sampler first."
            ) from exc

        # Placeholder wiring for tasks/02: sampler API may refine this.
        k = int(config.get("k", 8))
        _ = VLLMSampler  # noqa: F841 — real call filled when sampler lands
        raise SystemExit(
            f"tier CLI sampler path not fully wired ({len(missing)} untiered, k={k}); "
            "use a pool with pass8 for split-only runs until tasks/02."
        )

    enriched = [
        Problem(
            **{
                **p.to_dict(),
                "tier": tier_from_pass8(
                    int(p.pass8),
                    easy_min_pass8=easy_min,
                    medium_min_pass8=medium_min,
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

    out_dir = Path(config.get("output_dir", "data/splits/"))
    if args.dry_run:
        summary = {k: len(v) for k, v in splits.items()}
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("dry-run: not writing splits")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, problems in splits.items():
        write_jsonl(problems, out_dir / f"{name}.jsonl")
        print(f"wrote {len(problems)} -> {out_dir / f'{name}.jsonl'}")
    return 0
