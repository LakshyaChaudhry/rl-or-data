"""Procedural counting-problem generator (SPEC §4). AGENT-OWNED; tasks/01.

Rounding / tie-break conventions (documented here; SPEC §4 left them open):
- ``mean``: integer-rounded with round-half-to-even (Python ``round``).
- ``median`` (even length): lower median — ``sorted(values)[n // 2 - 1]``.
- ``mode``: smallest value among frequency ties.
- ``reverse_digits``: reverse decimal digits of ``abs(n)``, restore sign; leading zeros drop
  (e.g. 120 → 21).
- ``digit_sum`` (transform or filter): sum of decimal digits of ``abs(n)``.
- Bitwise final ops: rejected at sample time if any value is negative (unambiguous NL).
- Reject / resample: empty set after filters; ``|answer| > 10**9``; bitwise-with-negatives;
  invalid params (e.g. modulo 0).

Contract:
    generate_pool(config: dict, seed: int) -> list[Problem]
    execute_pipeline(pipeline: dict) -> int
    render(pipeline: dict, rng) -> str
    canonical_id(pipeline: dict) -> str
    write_jsonl(problems, path) / read_jsonl(path)
    cli_main(args) -> int
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.types import Problem

ANSWER_ABS_MAX = 10**9

DEFAULT_RANGE_SCALE: dict[str, dict[str, int]] = {
    "S": {"lo_min": 1, "hi_max": 50},
    "M": {"lo_min": 1, "hi_max": 200},
    "L": {"lo_min": 1, "hi_max": 1000},
}

FILTER_NAMES = (
    "even",
    "odd",
    "positive",
    "negative",
    "divisible_by",
    "not_divisible_by",
    "below_threshold",
    "above_threshold",
    "digit_sum_equals",
    "contains_digit",
    "prime",
    "perfect_square",
)

TRANSFORM_NAMES = (
    "add",
    "multiply",
    "square",
    "absolute_value",
    "modulo",
    "digit_sum",
    "reverse_digits",
)

FINAL_OP_NAMES = (
    "count",
    "unique_count",
    "zero_count",
    "sum",
    "product_mod",
    "mean",
    "median",
    "mode",
    "min",
    "max",
    "range",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
)

# ≥3 paraphrase templates per operator. Placeholders: {n},{t},{s},{d},{k},{m}
_FILTER_TEMPLATES: dict[str, list[str]] = {
    "even": [
        "First, keep only the numbers that are even.",
        "Next, retain solely the even integers.",
        "Then, filter the list to even values only.",
    ],
    "odd": [
        "First, keep only the numbers that are odd.",
        "Next, retain solely the odd integers.",
        "Then, filter the list to odd values only.",
    ],
    "positive": [
        "First, keep only the numbers that are positive.",
        "Next, retain solely the strictly positive integers.",
        "Then, filter the list to values greater than zero.",
    ],
    "negative": [
        "First, keep only the numbers that are negative.",
        "Next, retain solely the strictly negative integers.",
        "Then, filter the list to values less than zero.",
    ],
    "divisible_by": [
        "First, keep only the numbers that are divisible by {n}.",
        "Next, retain solely integers divisible by {n}.",
        "Then, filter the list to multiples of {n}.",
    ],
    "not_divisible_by": [
        "First, keep only the numbers that are not divisible by {n}.",
        "Next, retain solely integers that are not multiples of {n}.",
        "Then, discard every value divisible by {n}.",
    ],
    "below_threshold": [
        "First, keep only the numbers that are strictly below {t}.",
        "Next, retain solely integers less than {t}.",
        "Then, filter the list to values < {t}.",
    ],
    "above_threshold": [
        "First, keep only the numbers that are strictly above {t}.",
        "Next, retain solely integers greater than {t}.",
        "Then, filter the list to values > {t}.",
    ],
    "digit_sum_equals": [
        "First, keep only the numbers whose digits sum to {s}.",
        "Next, retain solely integers with digit sum equal to {s}.",
        "Then, filter the list to values whose digit sum is {s}.",
    ],
    "contains_digit": [
        "First, keep only the numbers that contain the digit {d}.",
        "Next, retain solely integers whose decimal form includes digit {d}.",
        "Then, filter the list to values containing digit {d}.",
    ],
    "prime": [
        "First, keep only the prime numbers.",
        "Next, retain solely the primes.",
        "Then, filter the list to prime values only.",
    ],
    "perfect_square": [
        "First, keep only the perfect squares.",
        "Next, retain solely perfect-square integers.",
        "Then, filter the list to perfect squares only.",
    ],
}

_TRANSFORM_TEMPLATES: dict[str, list[str]] = {
    "add": [
        "Then, add {k} to each remaining number.",
        "Next, increase every remaining value by {k}.",
        "After that, replace each remaining number x with x + {k}.",
    ],
    "multiply": [
        "Then, multiply each remaining number by {k}.",
        "Next, scale every remaining value by {k}.",
        "After that, replace each remaining number x with x * {k}.",
    ],
    "square": [
        "Then, square each remaining number.",
        "Next, replace every remaining value x with x squared.",
        "After that, replace each remaining number x with x * x.",
    ],
    "absolute_value": [
        "Then, replace each remaining number with its absolute value.",
        "Next, take the absolute value of every remaining integer.",
        "After that, replace each remaining number x with |x|.",
    ],
    "modulo": [
        "Then, replace each remaining number with its residue modulo {m}.",
        "Next, take every remaining value modulo {m}.",
        "After that, replace each remaining number x with x mod {m}.",
    ],
    "digit_sum": [
        "Then, replace each remaining number with the sum of its digits "
        "(using the absolute value's digits).",
        "Next, map every remaining value to the digit sum of its absolute value.",
        "After that, replace each remaining number x with the digit sum of |x|.",
    ],
    "reverse_digits": [
        "Then, reverse the decimal digits of each remaining number "
        "(absolute value's digits, restoring the sign; leading zeros drop).",
        "Next, reverse the digits of |x| for each remaining x and restore the sign.",
        "After that, replace each remaining number by reversing the digits of its absolute value, "
        "keeping the original sign.",
    ],
}

_OP_TEMPLATES: dict[str, list[str]] = {
    "count": [
        "Of these numbers, count how many values remain.",
        "Finally, report how many numbers are left.",
        "What is the number of remaining values?",
    ],
    "unique_count": [
        "Of these numbers, count how many unique values remain.",
        "Finally, report the number of distinct remaining values.",
        "What is the count of unique remaining numbers?",
    ],
    "zero_count": [
        "Of these numbers, count how many are equal to zero.",
        "Finally, report how many remaining values equal 0.",
        "What is the number of zeros among the remaining values?",
    ],
    "sum": [
        "Of these numbers, compute their sum.",
        "Finally, report the sum of the remaining values.",
        "What is the sum of the remaining numbers?",
    ],
    "product_mod": [
        "Of these numbers, compute the product of all remaining values modulo {m}.",
        "Finally, report (product of the remaining numbers) mod {m}.",
        "What is the product of the remaining values modulo {m}?",
    ],
    "mean": [
        "Of these numbers, compute the mean and round to the nearest integer (round half to even).",
        "Finally, report the integer mean of the remaining values, using round-half-to-even.",
        "What is the mean of the remaining numbers, rounded to an integer with round half to even?",
    ],
    "median": [
        "Of these numbers, compute the median (for an even count, use the lower median).",
        "Finally, report the median of the remaining values; if the count is even, use the lower median.",
        "What is the median of the remaining numbers (lower median when the length is even)?",
    ],
    "mode": [
        "Of these numbers, compute the mode (if there is a tie, choose the smallest value).",
        "Finally, report the mode of the remaining values; break ties by taking the smallest.",
        "What is the mode of the remaining numbers (smallest value on ties)?",
    ],
    "min": [
        "Of these numbers, report the minimum.",
        "Finally, what is the smallest remaining value?",
        "What is the minimum of the remaining numbers?",
    ],
    "max": [
        "Of these numbers, report the maximum.",
        "Finally, what is the largest remaining value?",
        "What is the maximum of the remaining numbers?",
    ],
    "range": [
        "Of these numbers, report the range (maximum minus minimum).",
        "Finally, what is max(remaining) - min(remaining)?",
        "What is the range of the remaining numbers (max − min)?",
    ],
    "bitwise_and": [
        "Of these numbers, compute the bitwise AND of all remaining values.",
        "Finally, report the bitwise AND folded over the remaining integers.",
        "What is the bitwise AND of the remaining numbers?",
    ],
    "bitwise_or": [
        "Of these numbers, compute the bitwise OR of all remaining values.",
        "Finally, report the bitwise OR folded over the remaining integers.",
        "What is the bitwise OR of the remaining numbers?",
    ],
    "bitwise_xor": [
        "Of these numbers, compute the bitwise XOR of all remaining values.",
        "Finally, report the bitwise XOR folded over the remaining integers.",
        "What is the bitwise XOR of the remaining numbers?",
    ],
}

_RANGE_TEMPLATES = [
    "Consider the integers from {lo} to {hi}, inclusive.",
    "Start with the inclusive integer range [{lo}, {hi}].",
    "Take all integers between {lo} and {hi}, including both endpoints.",
]


def all_operator_names() -> list[str]:
    """Every filter / transform / final-op name (for paraphrase coverage tests)."""
    return list(FILTER_NAMES) + list(TRANSFORM_NAMES) + list(FINAL_OP_NAMES)


def minimal_pipeline_for_operator(op_name: str) -> dict[str, Any]:
    """A tiny valid pipeline that mentions ``op_name`` (for rendering tests)."""
    base: dict[str, Any] = {
        "range": {"lo": 1, "hi": 20},
        "filters": [{"name": "positive"}],
        "transforms": [],
        "op": {"name": "count"},
    }
    if op_name in FILTER_NAMES:
        filt: dict[str, Any] = {"name": op_name}
        if op_name in ("divisible_by", "not_divisible_by"):
            filt["n"] = 3
        elif op_name in ("below_threshold", "above_threshold"):
            filt["t"] = 10
        elif op_name == "digit_sum_equals":
            filt["s"] = 5
        elif op_name == "contains_digit":
            filt["d"] = 1
        base["filters"] = [filt]
        return base
    if op_name in TRANSFORM_NAMES:
        tr: dict[str, Any] = {"name": op_name}
        if op_name in ("add", "multiply"):
            tr["k"] = 2
        elif op_name == "modulo":
            tr["m"] = 5
        base["transforms"] = [tr]
        return base
    if op_name in FINAL_OP_NAMES:
        op: dict[str, Any] = {"name": op_name}
        if op_name == "product_mod":
            op["m"] = 97
        base["op"] = op
        return base
    raise ValueError(f"unknown operator: {op_name}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    root = int(math.isqrt(n))
    for d in range(3, root + 1, 2):
        if n % d == 0:
            return False
    return True


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n


def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _reverse_digits(n: int) -> int:
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])


def _apply_filter(values: list[int], filt: dict[str, Any]) -> list[int]:
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
        if n == 0:
            raise ValueError("divisible_by n=0")
        return [v for v in values if v % n == 0]
    if name == "not_divisible_by":
        n = int(filt["n"])
        if n == 0:
            raise ValueError("not_divisible_by n=0")
        return [v for v in values if v % n != 0]
    if name == "below_threshold":
        t = int(filt["t"])
        return [v for v in values if v < t]
    if name == "above_threshold":
        t = int(filt["t"])
        return [v for v in values if v > t]
    if name == "digit_sum_equals":
        s = int(filt["s"])
        return [v for v in values if _digit_sum(v) == s]
    if name == "contains_digit":
        d = str(int(filt["d"]))
        return [v for v in values if d in str(abs(v))]
    if name == "prime":
        return [v for v in values if _is_prime(v)]
    if name == "perfect_square":
        return [v for v in values if _is_perfect_square(v)]
    raise ValueError(f"unknown filter: {name}")


def _apply_transform(values: list[int], transform: dict[str, Any]) -> list[int]:
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
        if m == 0:
            raise ValueError("modulo m=0")
        return [v % m for v in values]
    if name == "digit_sum":
        return [_digit_sum(v) for v in values]
    if name == "reverse_digits":
        return [_reverse_digits(v) for v in values]
    raise ValueError(f"unknown transform: {name}")


def _final_op(values: list[int], op: dict[str, Any]) -> int:
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
        if m == 0:
            raise ValueError("product_mod m=0")
        acc = 1
        for v in values:
            acc = (acc * v) % m
        return int(acc)
    if name == "mean":
        return int(round(sum(values) / len(values)))
    if name == "median":
        ordered = sorted(values)
        n = len(ordered)
        if n % 2 == 1:
            return int(ordered[n // 2])
        return int(ordered[n // 2 - 1])
    if name == "mode":
        counts = Counter(values)
        top = max(counts.values())
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


def execute_pipeline(pipeline: dict[str, Any]) -> int:
    """Execute a pipeline and return the integer answer.

    Pipeline shape: ``{range: {lo, hi}, filters: [...], transforms: [...], op: {...}}``.
    """
    lo = int(pipeline["range"]["lo"])
    hi = int(pipeline["range"]["hi"])
    assert lo <= hi, f"invalid range [{lo}, {hi}]"
    values = list(range(lo, hi + 1))
    for filt in pipeline["filters"]:
        values = _apply_filter(values, filt)
    for transform in pipeline["transforms"]:
        values = _apply_transform(values, transform)
    return _final_op(values, pipeline["op"])


def _filtered_values(pipeline: dict[str, Any]) -> list[int]:
    lo = int(pipeline["range"]["lo"])
    hi = int(pipeline["range"]["hi"])
    values = list(range(lo, hi + 1))
    for filt in pipeline["filters"]:
        values = _apply_filter(values, filt)
    return values


def _values_before_op(pipeline: dict[str, Any]) -> list[int]:
    values = _filtered_values(pipeline)
    for transform in pipeline["transforms"]:
        values = _apply_transform(values, transform)
    return values


# ---------------------------------------------------------------------------
# Canonical id / IO
# ---------------------------------------------------------------------------


def canonical_id(pipeline: dict[str, Any]) -> str:
    """sha256 over ``json.dumps(pipeline, sort_keys=True, separators=(',', ':'))``."""
    payload = json.dumps(pipeline, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(problems: list[Problem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for problem in problems:
            f.write(json.dumps(problem.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[Problem]:
    problems: list[Problem] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            problems.append(Problem(**raw))
    return problems


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pick(rng: np.random.Generator, templates: list[str], **kwargs: Any) -> str:
    idx = int(rng.integers(0, len(templates)))
    return templates[idx].format(**kwargs)


def render(pipeline: dict[str, Any], rng: np.random.Generator) -> str:
    """Natural-language rendering with ≥3 paraphrases per operator, chosen by ``rng``."""
    lo = int(pipeline["range"]["lo"])
    hi = int(pipeline["range"]["hi"])
    parts: list[str] = [_pick(rng, _RANGE_TEMPLATES, lo=lo, hi=hi)]

    for i, filt in enumerate(pipeline["filters"]):
        name = filt["name"]
        templates = _FILTER_TEMPLATES[name]
        # First filter uses "First," templates; later ones still fine (templates vary).
        _ = i
        parts.append(
            _pick(
                rng,
                templates,
                n=filt.get("n"),
                t=filt.get("t"),
                s=filt.get("s"),
                d=filt.get("d"),
            )
        )

    for transform in pipeline["transforms"]:
        name = transform["name"]
        parts.append(
            _pick(
                rng,
                _TRANSFORM_TEMPLATES[name],
                k=transform.get("k"),
                m=transform.get("m"),
            )
        )

    op = pipeline["op"]
    parts.append(_pick(rng, _OP_TEMPLATES[op["name"]], m=op.get("m")))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _parse_total_steps(config: dict[str, Any]) -> tuple[int, ...]:
    raw = config["total_steps"]
    if isinstance(raw, dict):
        return tuple(range(int(raw["min"]), int(raw["max"]) + 1))
    return tuple(int(x) for x in raw)


def _sample_range(
    rng: np.random.Generator, scale: str, scale_cfg: dict[str, int]
) -> tuple[int, int]:
    lo_min = int(scale_cfg["lo_min"])
    hi_max = int(scale_cfg["hi_max"])
    # Span of [lo, hi] is at most the scale's hi_max - lo_min + 1 by construction.
    lo = int(rng.integers(lo_min, hi_max + 1))
    hi = int(rng.integers(lo, hi_max + 1))
    return lo, hi


def _sample_filter(rng: np.random.Generator, lo: int, hi: int) -> dict[str, Any]:
    name = str(rng.choice(FILTER_NAMES))
    filt: dict[str, Any] = {"name": name}
    if name in ("divisible_by", "not_divisible_by"):
        filt["n"] = int(rng.integers(2, 13))
    elif name in ("below_threshold", "above_threshold"):
        # Threshold inside or just outside the range so the filter is often non-vacuous.
        filt["t"] = int(rng.integers(lo, hi + 2))
    elif name == "digit_sum_equals":
        filt["s"] = int(rng.integers(1, 19))
    elif name == "contains_digit":
        filt["d"] = int(rng.integers(0, 10))
    return filt


def _sample_transform(rng: np.random.Generator) -> dict[str, Any]:
    name = str(rng.choice(TRANSFORM_NAMES))
    transform: dict[str, Any] = {"name": name}
    if name == "add":
        transform["k"] = int(rng.integers(-9, 10))
        if transform["k"] == 0:
            transform["k"] = 1
    elif name == "multiply":
        transform["k"] = int(rng.integers(2, 6))
    elif name == "modulo":
        transform["m"] = int(rng.integers(2, 21))
    return transform


def _sample_op(rng: np.random.Generator) -> dict[str, Any]:
    name = str(rng.choice(FINAL_OP_NAMES))
    op: dict[str, Any] = {"name": name}
    if name == "product_mod":
        op["m"] = int(rng.integers(2, 97))
    return op


def _sample_counts(
    rng: np.random.Generator, config: dict[str, Any], total_steps: int
) -> tuple[int, int]:
    f_min = int(config.get("n_filters", {}).get("min", 1))
    f_max = int(config.get("n_filters", {}).get("max", 4))
    t_min = int(config.get("n_transforms", {}).get("min", 0))
    t_max = int(config.get("n_transforms", {}).get("max", 3))
    # n_filters + n_transforms + 1 = total_steps
    need = total_steps - 1
    candidates: list[tuple[int, int]] = []
    for n_f in range(f_min, f_max + 1):
        n_t = need - n_f
        if t_min <= n_t <= t_max:
            candidates.append((n_f, n_t))
    if not candidates:
        raise ValueError(
            f"no (n_filters, n_transforms) satisfy total_steps={total_steps} "
            f"with filters[{f_min},{f_max}] transforms[{t_min},{t_max}]"
        )
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def _try_sample_pipeline(rng: np.random.Generator, config: dict[str, Any]) -> dict[str, Any] | None:
    range_scale = config.get("range_scale", DEFAULT_RANGE_SCALE)
    weights_cfg = config["range_scale_weights"]
    scales = list(weights_cfg.keys())
    weights = np.asarray([float(weights_cfg[s]) for s in scales], dtype=float)
    weights = weights / weights.sum()
    scale = str(rng.choice(scales, p=weights))
    scale_cfg = range_scale[scale]

    step_choices = _parse_total_steps(config)
    total_steps = int(step_choices[int(rng.integers(0, len(step_choices)))])
    n_filters, n_transforms = _sample_counts(rng, config, total_steps)

    lo, hi = _sample_range(rng, scale, scale_cfg)
    filters = [_sample_filter(rng, lo, hi) for _ in range(n_filters)]
    transforms = [_sample_transform(rng) for _ in range(n_transforms)]
    op = _sample_op(rng)

    pipeline = {
        "range": {"lo": lo, "hi": hi},
        "filters": filters,
        "transforms": transforms,
        "op": op,
        # Carry scale for Problem construction (stripped from structure_id / not in hash? )
        # SPEC: problem_id = sha256(canonical pipeline). Keep only execution fields in id.
    }
    # Attach scale outside canonical fields via return pairing — see generate_pool.
    try:
        filtered = _filtered_values(pipeline)
        if not filtered:
            return None
        before_op = filtered
        for transform in transforms:
            before_op = _apply_transform(before_op, transform)
        if not before_op:
            return None
        if op["name"].startswith("bitwise") and any(v < 0 for v in before_op):
            return None
        answer = _final_op(before_op, op)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if abs(answer) > ANSWER_ABS_MAX:
        return None

    return {"pipeline": pipeline, "range_scale": scale, "answer": answer}


def generate_pool(config: dict[str, Any], seed: int) -> list[Problem]:
    """Generate ``config['n_problems']`` problems deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    n_problems = int(config["n_problems"])
    paraphrases = int(config.get("paraphrases_per_operator", 3))
    for mapping in (_FILTER_TEMPLATES, _TRANSFORM_TEMPLATES, _OP_TEMPLATES):
        for name, templates in mapping.items():
            if len(templates) < paraphrases:
                raise RuntimeError(
                    f"operator {name} has {len(templates)} templates; need >= {paraphrases}"
                )

    problems: list[Problem] = []
    seen_ids: set[str] = set()
    # Cap rejection loops so a bad config fails loudly.
    max_attempts = n_problems * 200
    attempts = 0
    while len(problems) < n_problems:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"failed to sample {n_problems} valid problems after {max_attempts} attempts "
                f"(got {len(problems)})"
            )
        sampled = _try_sample_pipeline(rng, config)
        if sampled is None:
            continue
        pipeline = sampled["pipeline"]
        pid = canonical_id(pipeline)
        if pid in seen_ids:
            continue
        text = render(pipeline, rng)
        answer = int(sampled["answer"])
        # Prefer the answer from execute_pipeline for a single code path in the record.
        answer = execute_pipeline(pipeline)
        n_filters = len(pipeline["filters"])
        n_transforms = len(pipeline["transforms"])
        problem = Problem(
            problem_id=pid,
            text=text,
            answer=answer,
            pipeline=pipeline,
            range_scale=sampled["range_scale"],  # type: ignore[arg-type]
            n_filters=n_filters,
            n_transforms=n_transforms,
            total_steps=n_filters + n_transforms + 1,
        )
        seen_ids.add(pid)
        problems.append(problem)
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_stats(problems: list[Problem]) -> None:
    by_scale: Counter[str] = Counter(p.range_scale for p in problems)
    by_steps: Counter[int] = Counter(p.total_steps for p in problems)
    by_op: Counter[str] = Counter(str(p.pipeline["op"]["name"]) for p in problems)
    print(f"n_problems: {len(problems)}")
    print("per range_scale:")
    for k in sorted(by_scale):
        print(f"  {k}: {by_scale[k]}")
    print("per total_steps:")
    for k in sorted(by_steps):
        print(f"  {k}: {by_steps[k]}")
    print("per final op:")
    for k in sorted(by_op):
        print(f"  {k}: {by_op[k]}")


def cli_main(args: Any) -> int:
    """``rlordata gen --config ... [--seed N] [--dry-run]``."""
    with Path(args.config).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    problems = generate_pool(config, seed=seed)
    _print_stats(problems)
    if args.dry_run:
        print("dry-run: not writing JSONL")
        return 0
    out = Path(config.get("output", "data/pool/pool.jsonl"))
    write_jsonl(problems, out)
    print(f"wrote {len(problems)} problems -> {out}")
    return 0
