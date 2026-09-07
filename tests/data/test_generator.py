"""AGENT-OWNED acceptance tests for tasks/01.

The reference executor below is written from SPEC §4 / tasks/01 text only — it must not
import ``rlordata.data.generator.execute_pipeline``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from rlordata.data.generator import (
    canonical_id,
    execute_pipeline,
    generate_pool,
    read_jsonl,
    render,
    write_jsonl,
)
from rlordata.data.tiers import build_splits, structure_id, tier_from_pass8
from rlordata.types import Problem

# ---------------------------------------------------------------------------
# Independent reference executor (SPEC §4 / tasks/01 — do not import generator)
# ---------------------------------------------------------------------------

_ANSWER_ABS_MAX = 10**9


def _ref_is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def _ref_is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def _ref_digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _ref_reverse_digits(n: int) -> int:
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])


def _ref_apply_filter(values: list[int], filt: dict[str, Any]) -> list[int]:
    name = filt["name"]
    if name == "even":
        return [v for v in values if v % 2 == 0]
    if name == "odd":
        return [v for v in values if v % 2 != 0]
    if name == "positive":
        return [v for v in values if v > 0]
    if name == "negative":
        return [v for v in values if v < 0]
    if name == "divisible_by":
        n = int(filt["n"])
        return [v for v in values if n != 0 and v % n == 0]
    if name == "not_divisible_by":
        n = int(filt["n"])
        return [v for v in values if n != 0 and v % n != 0]
    if name == "below_threshold":
        t = int(filt["t"])
        return [v for v in values if v < t]
    if name == "above_threshold":
        t = int(filt["t"])
        return [v for v in values if v > t]
    if name == "digit_sum_equals":
        s = int(filt["s"])
        return [v for v in values if _ref_digit_sum(v) == s]
    if name == "contains_digit":
        d = str(int(filt["d"]))
        return [v for v in values if d in str(abs(v))]
    if name == "prime":
        return [v for v in values if _ref_is_prime(v)]
    if name == "perfect_square":
        return [v for v in values if _ref_is_perfect_square(v)]
    raise ValueError(f"unknown filter: {name}")


def _ref_apply_transform(values: list[int], transform: dict[str, Any]) -> list[int]:
    name = transform["name"]
    if name == "add":
        k = int(transform["k"])
        return [v + k for v in values]
    if name == "multiply":
        k = int(transform["k"])
        return [v * k for v in values]
    if name == "square":
        return [v * v for v in values]
    if name == "absolute_value":
        return [abs(v) for v in values]
    if name == "modulo":
        m = int(transform["m"])
        return [v % m for v in values]
    if name == "digit_sum":
        return [_ref_digit_sum(v) for v in values]
    if name == "reverse_digits":
        return [_ref_reverse_digits(v) for v in values]
    raise ValueError(f"unknown transform: {name}")


def _ref_final_op(values: list[int], op: dict[str, Any]) -> int:
    if not values:
        raise ValueError("empty set before final op")
    name = op["name"]
    if name == "count":
        return len(values)
    if name == "unique_count":
        return len(set(values))
    if name == "zero_count":
        return sum(1 for v in values if v == 0)
    if name == "sum":
        return int(sum(values))
    if name == "product_mod":
        m = int(op["m"])
        acc = 1
        for v in values:
            acc = (acc * v) % m
        return int(acc)
    if name == "mean":
        # Integer-rounded; round half to even (Python round).
        return int(round(sum(values) / len(values)))
    if name == "median":
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return int(s[n // 2])
        # Even length: lower median (document in generator).
        return int(s[n // 2 - 1])
    if name == "mode":
        counts = Counter(values)
        top = max(counts.values())
        # Smallest value among ties.
        return int(min(v for v, c in counts.items() if c == top))
    if name == "min":
        return int(min(values))
    if name == "max":
        return int(max(values))
    if name == "range":
        return int(max(values) - min(values))
    if name == "bitwise_and":
        acc = values[0]
        for v in values[1:]:
            acc &= v
        return int(acc)
    if name == "bitwise_or":
        acc = values[0]
        for v in values[1:]:
            acc |= v
        return int(acc)
    if name == "bitwise_xor":
        acc = values[0]
        for v in values[1:]:
            acc ^= v
        return int(acc)
    raise ValueError(f"unknown op: {name}")


def reference_execute(pipeline: dict[str, Any]) -> int:
    """Independent ground-truth executor written from SPEC text, not generator code."""
    lo = int(pipeline["range"]["lo"])
    hi = int(pipeline["range"]["hi"])
    values = list(range(lo, hi + 1))
    for filt in pipeline["filters"]:
        values = _ref_apply_filter(values, filt)
    for transform in pipeline["transforms"]:
        values = _ref_apply_transform(values, transform)
    return _ref_final_op(values, pipeline["op"])


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_pool_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "configs" / "data" / "pool.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def _small_pool_config(n: int = 200, seed: int = 0) -> dict[str, Any]:
    cfg = _load_pool_config()
    cfg = deepcopy(cfg)
    cfg["n_problems"] = n
    cfg["seed"] = seed
    return cfg


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_determinism_identical_ids_and_texts() -> None:
    cfg = _small_pool_config(n=64, seed=12345)
    a = generate_pool(cfg, seed=12345)
    b = generate_pool(cfg, seed=12345)
    assert [p.problem_id for p in a] == [p.problem_id for p in b]
    assert [p.text for p in a] == [p.text for p in b]
    assert [p.answer for p in a] == [p.answer for p in b]


# ---------------------------------------------------------------------------
# 2. Independent reference executor agreement
# ---------------------------------------------------------------------------


def test_reference_executor_agrees_on_pool() -> None:
    cfg = _small_pool_config(n=2000, seed=20260901)
    pool = generate_pool(cfg, seed=20260901)
    assert len(pool) == 2000
    for p in pool:
        assert reference_execute(p.pipeline) == execute_pipeline(p.pipeline) == p.answer


# ---------------------------------------------------------------------------
# 3. canonical_id stability / sensitivity
# ---------------------------------------------------------------------------


def test_canonical_id_ignores_key_order_and_changes_on_edit() -> None:
    pipe = {
        "range": {"lo": 1, "hi": 10},
        "filters": [{"name": "even"}, {"name": "divisible_by", "n": 3}],
        "transforms": [{"name": "add", "k": 1}],
        "op": {"name": "count"},
    }
    reordered = {
        "op": {"name": "count"},
        "transforms": [{"k": 1, "name": "add"}],
        "filters": [{"name": "even"}, {"n": 3, "name": "divisible_by"}],
        "range": {"hi": 10, "lo": 1},
    }
    assert canonical_id(pipe) == canonical_id(reordered)
    # Whitespace / dump style must not matter: canonical_id uses sort_keys + separators.
    assert canonical_id(pipe) == canonical_id(json.loads(json.dumps(pipe)))

    changed = deepcopy(pipe)
    changed["filters"][1]["n"] = 4
    assert canonical_id(changed) != canonical_id(pipe)

    changed_range = deepcopy(pipe)
    changed_range["range"]["hi"] = 11
    assert canonical_id(changed_range) != canonical_id(pipe)


# ---------------------------------------------------------------------------
# 4. Split disjointness by problem_id and structure
# ---------------------------------------------------------------------------


def test_split_disjointness_by_id_and_structure() -> None:
    cfg = _small_pool_config(n=2500, seed=7)
    pool = generate_pool(cfg, seed=7)
    # Synthetic pass8: cycle so all tiers are populated.
    enriched: list[Problem] = []
    for i, p in enumerate(pool):
        pass8 = i % 9  # 0..8
        enriched.append(
            Problem(
                **{
                    **p.to_dict(),
                    "pass8": pass8,
                    "tier": tier_from_pass8(pass8),
                }
            )
        )

    splits = build_splits(enriched, seed=99)
    required = ("train_easy_100", "train_mixed_100", "val_mixed_100", "test_300", "train_curated")
    for key in required:
        assert key in splits

    # problem_id disjoint across primary sampled splits (curated is subset of mixed).
    primary = ("train_easy_100", "train_mixed_100", "val_mixed_100", "test_300")
    seen_ids: set[str] = set()
    seen_structs: set[str] = set()
    for key in primary:
        ids = {p.problem_id for p in splits[key]}
        structs = {structure_id(p.pipeline) for p in splits[key]}
        assert ids.isdisjoint(seen_ids)
        assert structs.isdisjoint(seen_structs)
        seen_ids |= ids
        seen_structs |= structs

    curated_ids = {p.problem_id for p in splits["train_curated"]}
    mixed_ids = {p.problem_id for p in splits["train_mixed_100"]}
    assert curated_ids <= mixed_ids
    for p in splits["train_curated"]:
        assert p.pass8 is not None and 1 <= p.pass8 <= 7


# ---------------------------------------------------------------------------
# 5. Knob monotonicity
# ---------------------------------------------------------------------------


def test_knob_monotonicity_steps_and_range_span() -> None:
    base = _load_pool_config()
    base = deepcopy(base)
    base["n_problems"] = 300

    cfg_s = deepcopy(base)
    cfg_s["range_scale_weights"] = {"S": 1.0}
    cfg_s["total_steps"] = {"min": 2, "max": 3}

    cfg_m = deepcopy(base)
    cfg_m["range_scale_weights"] = {"M": 1.0}
    cfg_m["total_steps"] = {"min": 3, "max": 4}

    cfg_l = deepcopy(base)
    cfg_l["range_scale_weights"] = {"L": 1.0}
    cfg_l["total_steps"] = {"min": 6, "max": 8}
    # L is allowed when weights request it (ood path).
    cfg_l["n_filters"] = {"min": 1, "max": 4}
    cfg_l["n_transforms"] = {"min": 0, "max": 3}

    pool_s = generate_pool(cfg_s, seed=1)
    pool_m = generate_pool(cfg_m, seed=1)
    pool_l = generate_pool(cfg_l, seed=1)

    def mean_steps(pool: list[Problem]) -> float:
        return sum(p.total_steps for p in pool) / len(pool)

    def mean_span(pool: list[Problem]) -> float:
        spans = [p.pipeline["range"]["hi"] - p.pipeline["range"]["lo"] + 1 for p in pool]
        return sum(spans) / len(spans)

    assert mean_steps(pool_s) < mean_steps(pool_m) < mean_steps(pool_l)
    assert mean_span(pool_s) < mean_span(pool_m) < mean_span(pool_l)


# ---------------------------------------------------------------------------
# 6. ≥3 distinct surface renderings per operator
# ---------------------------------------------------------------------------


def test_each_operator_has_at_least_three_paraphrases() -> None:
    # Build one minimal pipeline per operator and render with many seeds.
    from rlordata.data import generator as gen

    operators = gen.all_operator_names()
    assert len(operators) >= 10
    for op_name in operators:
        pipeline = gen.minimal_pipeline_for_operator(op_name)
        texts = {render(pipeline, __import__("numpy").random.default_rng(s)) for s in range(60)}
        assert len(texts) >= 3, f"{op_name} only produced {len(texts)} distinct renderings"


# ---------------------------------------------------------------------------
# 7. Bounded answers, no empty filtered sets
# ---------------------------------------------------------------------------


def test_answers_bounded_and_no_empty_filtered_sets() -> None:
    cfg = _small_pool_config(n=500, seed=42)
    pool = generate_pool(cfg, seed=42)
    for p in pool:
        assert isinstance(p.answer, int)
        assert abs(p.answer) <= _ANSWER_ABS_MAX
        assert not math.isnan(float(p.answer))
        assert not math.isinf(float(p.answer))
        # Re-run filters only: must be non-empty before transforms/final op.
        lo = p.pipeline["range"]["lo"]
        hi = p.pipeline["range"]["hi"]
        values = list(range(lo, hi + 1))
        for filt in p.pipeline["filters"]:
            values = _ref_apply_filter(values, filt)
        assert values, "generator emitted empty filtered set"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    cfg = _small_pool_config(n=20, seed=3)
    pool = generate_pool(cfg, seed=3)
    path = tmp_path / "pool.jsonl"
    write_jsonl(pool, path)
    loaded = read_jsonl(path)
    assert len(loaded) == len(pool)
    assert [p.problem_id for p in loaded] == [p.problem_id for p in pool]
    assert [p.answer for p in loaded] == [p.answer for p in pool]


def test_tier_from_pass8_thresholds() -> None:
    assert tier_from_pass8(8) == "easy"
    assert tier_from_pass8(6) == "easy"
    assert tier_from_pass8(5) == "medium"
    assert tier_from_pass8(2) == "medium"
    assert tier_from_pass8(1) == "hard"
    assert tier_from_pass8(0) == "hard"
