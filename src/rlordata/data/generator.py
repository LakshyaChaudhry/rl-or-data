"""Procedural counting-problem generator (SPEC §4, v1.2). AGENT-OWNED; tasks/01 + tasks/01b.

Rounding / tie-break conventions (documented here; SPEC §4 left them open):
- ``mean``: integer-rounded with round-half-to-even (Python ``round``).
- ``median`` (even length): lower median — ``sorted(values)[n // 2 - 1]``.
- ``mode``: smallest value among frequency ties.
- ``reverse_digits``: reverse decimal digits of ``abs(n)``, restore sign; leading zeros drop
  (e.g. 120 → 21).
- ``digit_sum`` (transform or filter): sum of decimal digits of ``abs(n)``.
- ``modulo`` / ``product_mod``: non-negative remainder (Python ``%``).
- Bitwise final ops: rejected at sample time if any value is negative (unambiguous NL).

v1.2 range bands (SPEC §4, tasks/01b §1): ``range_scale`` is the span ``hi - lo``; the span is
drawn uniformly inside the scale's band, then ``lo`` uniformly in
``[lo_min, lo_min + span_max - span]`` so that ``hi = lo + span <= lo_min + span_max``. The
upper bound on ``lo`` is a design choice (the amendment fixes only ``lo >= lo_min``): it keeps
every scale's numbers inside its magnitude window (S ≤ 51, M ≤ 201, L ≤ 1001), as in v1.1.

v1.2 rejection rules (SPEC §4 "every step must do work"), each with a named reason that is
counted per cell: ``duplicate_consecutive_filters``, ``empty_set``, ``noop_filter``,
``noop_transform``, ``too_few_at_final_op``, ``mode_without_repeat``,
``unique_count_without_repeat``, ``bitwise_negative``, ``invalid_params``,
``answer_too_large``, ``duplicate_id``.

v1.2 stratification (SPEC §4 "balanced pool"): cells = ``scales_in_pool`` × ``total_steps``;
target per cell = ``n_problems // n_cells``, and when ``n_problems`` is not divisible the
remainder is spread one extra problem per cell over the first cells in (scale, steps) order
(e.g. 200 over 3 cells → 67/67/66). Cells are filled in order; the pool is emitted grouped by
cell. Cell counts and per-reason rejection counts are printed and stored next to the pool
in ``<output>.meta.json``.

Rendering (SPEC §4 v1.2): connectives are chosen by position — "First" for step 1, "Then" /
"Next" (varied) in the middle, "Finally" / "Of these numbers" before the final op — and only
the operation phrase (≥3 paraphrases per operator) is paraphrased.

Contract:
    generate_pool(config: dict, seed: int) -> list[Problem]
    generate_pool_with_stats(config: dict, seed: int) -> tuple[list[Problem], PoolStats]
    execute_pipeline(pipeline: dict) -> int
    trace_pipeline(pipeline: dict) -> list[list[int]]
    reject_reason(pipeline: dict, rules: RejectRules = ...) -> str | None
    render(pipeline: dict, rng) -> str
    canonical_id(pipeline: dict) -> str
    write_jsonl(problems, path) / read_jsonl(path) / write_pool_meta(path, stats)
    cli_main(args) -> int
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.types import Problem

ANSWER_ABS_MAX = 10**9
DEFAULT_LO_MIN = 1

# Span bands (SPEC §4 v1.2): scale = hi - lo. Disjoint by construction.
DEFAULT_RANGE_SCALE: dict[str, dict[str, int]] = {
    "S": {"span_min": 10, "span_max": 50},
    "M": {"span_min": 51, "span_max": 200},
    "L": {"span_min": 201, "span_max": 1000},
}

# Dead filters for lo >= 1 (SPEC §4 v1.2). Code path kept; excluded from sampling by default.
DEFAULT_EXCLUDED_FILTERS: tuple[str, ...] = ("positive", "negative")

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

REJECT_REASONS = (
    "duplicate_consecutive_filters",
    "empty_set",
    "noop_filter",
    "noop_transform",
    "too_few_at_final_op",
    "mode_without_repeat",
    "unique_count_without_repeat",
    "bitwise_negative",
    "invalid_params",
    "answer_too_large",
    "duplicate_id",
)

# ---------------------------------------------------------------------------
# Rendering: connectives by position + ≥3 operation phrases per operator
# ---------------------------------------------------------------------------

CONNECTIVE_FIRST = "First"
CONNECTIVES_MIDDLE: tuple[str, ...] = ("Then", "Next")
CONNECTIVES_LAST: tuple[str, ...] = ("Finally", "Of these numbers")

_RANGE_TEMPLATES = [
    "Consider the integers from {lo} to {hi}, inclusive.",
    "Start with the inclusive integer range [{lo}, {hi}].",
    "Take all integers between {lo} and {hi}, including both endpoints.",
]

# Operation phrases: imperative clauses, no connective, no trailing period, and no ". " inside
# (tests split sentences on ". "). Placeholders: {n},{t},{s},{d},{k},{m},{m1}
_FILTER_PHRASES: dict[str, list[str]] = {
    "even": [
        "keep only the numbers that are even",
        "retain only the even integers",
        "discard every odd number, keeping the even ones",
    ],
    "odd": [
        "keep only the numbers that are odd",
        "retain only the odd integers",
        "discard every even number, keeping the odd ones",
    ],
    "positive": [
        "keep only the numbers that are positive",
        "retain only the strictly positive integers",
        "keep only the values greater than zero",
    ],
    "negative": [
        "keep only the numbers that are negative",
        "retain only the strictly negative integers",
        "keep only the values less than zero",
    ],
    "divisible_by": [
        "keep only the numbers that are divisible by {n}",
        "retain only the multiples of {n}",
        "keep only the values that leave no remainder when divided by {n}",
    ],
    "not_divisible_by": [
        "keep only the numbers that are not divisible by {n}",
        "discard every multiple of {n}",
        "retain only the values that are not multiples of {n}",
    ],
    "below_threshold": [
        "keep only the numbers that are strictly less than {t}",
        "retain only the values below {t} (not including {t} itself)",
        "discard every number that is {t} or greater",
    ],
    "above_threshold": [
        "keep only the numbers that are strictly greater than {t}",
        "retain only the values above {t} (not including {t} itself)",
        "discard every number that is {t} or smaller",
    ],
    "digit_sum_equals": [
        "keep only the numbers whose digits sum to {s}",
        "retain only the values with a digit sum of exactly {s}",
        "keep only the numbers whose decimal digits add up to {s}",
    ],
    "contains_digit": [
        "keep only the numbers that contain the digit {d}",
        "retain only the values whose decimal representation includes the digit {d}",
        "discard every number that does not have a {d} among its digits",
    ],
    "prime": [
        "keep only the prime numbers",
        "retain only the primes",
        "discard every number that is not prime",
    ],
    "perfect_square": [
        "keep only the perfect squares",
        "retain only the values that are perfect squares",
        "discard every number that is not a perfect square",
    ],
}

_TRANSFORM_PHRASES: dict[str, list[str]] = {
    "add": [
        "add {k} to each remaining number",
        "increase every remaining value by {k}",
        "replace each remaining number x with x + {k}",
    ],
    # Rendering of ``add`` with a negative k (phrases keyed by an internal name, not an operator).
    "add_negative": [
        "subtract {k} from each remaining number",
        "decrease every remaining value by {k}",
        "replace each remaining number x with x - {k}",
    ],
    "multiply": [
        "multiply each remaining number by {k}",
        "scale every remaining value by a factor of {k}",
        "replace each remaining number x with {k} * x",
    ],
    "square": [
        "square each remaining number",
        "replace every remaining value x with x squared",
        "replace each remaining number x with x * x",
    ],
    "absolute_value": [
        "replace each remaining number with its absolute value",
        "take the absolute value of every remaining number",
        "replace each remaining number x with |x|",
    ],
    "modulo": [
        "replace each remaining number with its remainder modulo {m}, taken in the range 0 to {m1}",
        "take every remaining value modulo {m}, using the non-negative remainder (0 to {m1})",
        "replace each remaining number x with x mod {m}, where the result lies in 0 to {m1}",
    ],
    "digit_sum": [
        "replace each remaining number with the sum of its decimal digits (ignoring any minus sign)",
        "map every remaining value to the digit sum of its absolute value",
        "replace each remaining number x with the sum of the digits of |x|",
    ],
    "reverse_digits": [
        "reverse the decimal digits of each remaining number (reverse the digits of its absolute "
        "value, keep the original sign, and drop any leading zeros)",
        "replace every remaining value x with the digits of |x| written in reverse order, with "
        "the original sign restored and leading zeros dropped",
        "replace each remaining number by reversing the digits of its absolute value, keeping "
        "the original sign and dropping any leading zeros",
    ],
}

_OP_PHRASES: dict[str, list[str]] = {
    "count": [
        "count how many values remain",
        "report how many numbers are left",
        "give the number of remaining values",
    ],
    "unique_count": [
        "count how many distinct values remain",
        "report the number of unique values among the remaining numbers",
        "give the count of distinct remaining values (each repeated value counts once)",
    ],
    "zero_count": [
        "count how many of the remaining values are equal to zero",
        "report how many remaining numbers equal 0",
        "give the number of zeros among the remaining values",
    ],
    "sum": [
        "compute the sum of the remaining values",
        "report the total of all remaining numbers",
        "add up all the remaining values and give the result",
    ],
    "product_mod": [
        "compute the product of all remaining values modulo {m} (the non-negative remainder)",
        "report the non-negative remainder when the product of the remaining numbers is divided by {m}",
        "multiply all remaining values together and give the result mod {m}, as a non-negative remainder",
    ],
    "mean": [
        "compute the mean of the remaining values, rounded to the nearest integer (round half to even)",
        "report the average of the remaining numbers rounded to an integer, rounding halves to the even neighbour",
        "give the integer-rounded mean of the remaining values, using round-half-to-even",
    ],
    "median": [
        "compute the median of the remaining values (if the count is even, use the lower of the two middle values)",
        "report the median of the remaining numbers, taking the lower middle value when the count is even",
        "give the median of the remaining values, using the smaller of the two middle values for an even count",
    ],
    "mode": [
        "compute the mode of the remaining values (if several values tie for most frequent, choose the smallest)",
        "report the most frequent remaining value, breaking ties by taking the smallest",
        "give the mode of the remaining numbers, choosing the smallest value in case of a tie",
    ],
    "min": [
        "report the minimum of the remaining values",
        "give the smallest remaining number",
        "find the least value among the remaining numbers",
    ],
    "max": [
        "report the maximum of the remaining values",
        "give the largest remaining number",
        "find the greatest value among the remaining numbers",
    ],
    "range": [
        "report the range of the remaining values (maximum minus minimum)",
        "give the difference between the largest and the smallest remaining number",
        "compute the maximum minus the minimum over the remaining values",
    ],
    "bitwise_and": [
        "compute the bitwise AND of all remaining values",
        "report the result of combining all remaining numbers with bitwise AND",
        "fold bitwise AND across the remaining values and give the result",
    ],
    "bitwise_or": [
        "compute the bitwise OR of all remaining values",
        "report the result of combining all remaining numbers with bitwise OR",
        "fold bitwise OR across the remaining values and give the result",
    ],
    "bitwise_xor": [
        "compute the bitwise XOR of all remaining values",
        "report the result of combining all remaining numbers with bitwise XOR",
        "fold bitwise XOR across the remaining values and give the result",
    ],
}


def all_operator_names() -> list[str]:
    """Every filter / transform / final-op name (for paraphrase coverage tests)."""
    return list(FILTER_NAMES) + list(TRANSFORM_NAMES) + list(FINAL_OP_NAMES)


def minimal_pipeline_for_operator(op_name: str) -> dict[str, Any]:
    """A tiny valid pipeline that mentions ``op_name`` (for rendering tests)."""
    base: dict[str, Any] = {
        "range": {"lo": 1, "hi": 20},
        "filters": [{"name": "even"}],
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


def trace_pipeline(pipeline: dict[str, Any]) -> list[list[int]]:
    """Multiset after the range and after each filter / transform, in pipeline order.

    Returns ``1 + n_filters + n_transforms`` lists; the last is what the final op sees.
    Pipeline shape: ``{range: {lo, hi}, filters: [...], transforms: [...], op: {...}}``.
    """
    lo = int(pipeline["range"]["lo"])
    hi = int(pipeline["range"]["hi"])
    assert lo <= hi, f"invalid range [{lo}, {hi}]"
    values = list(range(lo, hi + 1))
    trace = [values]
    for filt in pipeline["filters"]:
        values = _apply_filter(values, filt)
        trace.append(values)
    for transform in pipeline["transforms"]:
        values = _apply_transform(values, transform)
        trace.append(values)
    return trace


def execute_pipeline(pipeline: dict[str, Any]) -> int:
    """Execute a pipeline and return the integer answer (single code path for records)."""
    return _final_op(trace_pipeline(pipeline)[-1], pipeline["op"])


# ---------------------------------------------------------------------------
# Config and rejection rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RejectRules:
    """SPEC §4 v1.2 rejection rules (``reject:`` block of configs/data/pool.yaml)."""

    noop_steps: bool = True
    duplicate_consecutive_filters: bool = True
    min_values_at_final_op: int = 3
    mode_without_repeat: bool = True
    unique_count_without_repeat: bool = True
    max_abs_answer: int = ANSWER_ABS_MAX

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> RejectRules:
        raw = dict(raw or {})
        # The v1.2 amendment spells this key ``unique_count_withepeat`` in pool.yaml; accept
        # both spellings so a later typo fix cannot silently disable the rule.
        unique_flag = raw.get(
            "unique_count_without_repeat", raw.get("unique_count_withepeat", True)
        )
        return cls(
            noop_steps=bool(raw.get("noop_steps", True)),
            duplicate_consecutive_filters=bool(raw.get("duplicate_consecutive_filters", True)),
            min_values_at_final_op=int(raw.get("min_values_at_final_op", 3)),
            mode_without_repeat=bool(raw.get("mode_without_repeat", True)),
            unique_count_without_repeat=bool(unique_flag),
            max_abs_answer=int(raw.get("max_abs_answer", ANSWER_ABS_MAX)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noop_steps": self.noop_steps,
            "duplicate_consecutive_filters": self.duplicate_consecutive_filters,
            "min_values_at_final_op": self.min_values_at_final_op,
            "mode_without_repeat": self.mode_without_repeat,
            "unique_count_without_repeat": self.unique_count_without_repeat,
            "max_abs_answer": self.max_abs_answer,
        }


@dataclass(frozen=True)
class GenConfig:
    """Resolved generator config (defaults filled; see module docstring)."""

    n_problems: int
    lo_min: int
    bands: dict[str, tuple[int, int]]  # scale -> (span_min, span_max)
    scales: tuple[str, ...]
    stratify: bool
    total_steps: tuple[int, ...]
    n_filters: tuple[int, int]
    n_transforms: tuple[int, int]
    excluded_filters: frozenset[str]
    reject: RejectRules
    paraphrases_per_operator: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GenConfig:
        raw_bands = config.get("range_scale", DEFAULT_RANGE_SCALE)
        bands: dict[str, tuple[int, int]] = {}
        for scale, spec in raw_bands.items():
            if "span_min" not in spec or "span_max" not in spec:
                raise ValueError(
                    f"range_scale[{scale}] must define span_min/span_max (v1.2 bands), got {spec}"
                )
            span_min, span_max = int(spec["span_min"]), int(spec["span_max"])
            if not 1 <= span_min <= span_max:
                raise ValueError(f"range_scale[{scale}]: bad band [{span_min}, {span_max}]")
            bands[str(scale)] = (span_min, span_max)
        ordered = sorted(bands.items(), key=lambda kv: kv[1][0])
        for (a, band_a), (b, band_b) in zip(ordered, ordered[1:], strict=False):
            if band_a[1] >= band_b[0]:
                raise ValueError(f"range_scale bands overlap: {a}={band_a}, {b}={band_b}")

        scales = tuple(str(s) for s in config.get("scales_in_pool", list(bands)))
        for s in scales:
            if s not in bands:
                raise ValueError(f"scales_in_pool contains unknown scale {s!r}; have {list(bands)}")
        if not scales:
            raise ValueError("scales_in_pool is empty")

        raw_steps = config.get("total_steps", {"min": 2, "max": 8})
        if isinstance(raw_steps, dict):
            total_steps = tuple(range(int(raw_steps["min"]), int(raw_steps["max"]) + 1))
        else:
            total_steps = tuple(int(x) for x in raw_steps)
        if not total_steps:
            raise ValueError("total_steps is empty")

        n_filters_cfg = config.get("n_filters", {}) or {}
        n_transforms_cfg = config.get("n_transforms", {}) or {}
        excluded = config.get("excluded_filters", list(DEFAULT_EXCLUDED_FILTERS))
        excluded_set = frozenset(str(x) for x in (excluded or []))
        for name in excluded_set:
            if name not in FILTER_NAMES:
                raise ValueError(f"excluded_filters: unknown filter {name!r}")
        if not [f for f in FILTER_NAMES if f not in excluded_set]:
            raise ValueError("excluded_filters removes every filter")

        return cls(
            n_problems=int(config["n_problems"]),
            lo_min=int(config.get("lo_min", DEFAULT_LO_MIN)),
            bands=bands,
            scales=scales,
            stratify=bool(config.get("stratify", True)),
            total_steps=total_steps,
            n_filters=(int(n_filters_cfg.get("min", 1)), int(n_filters_cfg.get("max", 4))),
            n_transforms=(
                int(n_transforms_cfg.get("min", 0)),
                int(n_transforms_cfg.get("max", 3)),
            ),
            excluded_filters=excluded_set,
            reject=RejectRules.from_config(config.get("reject")),
            paraphrases_per_operator=int(config.get("paraphrases_per_operator", 3)),
        )

    def hi_max(self, scale: str) -> int:
        """Largest value any range of ``scale`` may reach: ``lo_min + span_max``."""
        return self.lo_min + self.bands[scale][1]

    @property
    def allowed_filters(self) -> tuple[str, ...]:
        return tuple(f for f in FILTER_NAMES if f not in self.excluded_filters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_problems": self.n_problems,
            "lo_min": self.lo_min,
            "range_scale": {
                k: {"span_min": v[0], "span_max": v[1], "hi_max": self.lo_min + v[1]}
                for k, v in self.bands.items()
            },
            "scales_in_pool": list(self.scales),
            "stratify": self.stratify,
            "total_steps": list(self.total_steps),
            "n_filters": {"min": self.n_filters[0], "max": self.n_filters[1]},
            "n_transforms": {"min": self.n_transforms[0], "max": self.n_transforms[1]},
            "excluded_filters": sorted(self.excluded_filters),
            "reject": self.reject.to_dict(),
            "paraphrases_per_operator": self.paraphrases_per_operator,
        }


def _check_candidate(pipeline: dict[str, Any], rules: RejectRules) -> tuple[str | None, int | None]:
    """Apply the v1.2 rejection rules; return ``(reason, None)`` or ``(None, answer)``."""
    filters = pipeline["filters"]
    if rules.duplicate_consecutive_filters:
        for a, b in zip(filters, filters[1:], strict=False):
            if a == b:
                return "duplicate_consecutive_filters", None

    n_filters = len(filters)
    try:
        trace = trace_pipeline(pipeline)
    except (ValueError, ZeroDivisionError, OverflowError):
        return "invalid_params", None

    for i in range(1, len(trace)):
        before, after = trace[i - 1], trace[i]
        if not after:
            return "empty_set", None
        if rules.noop_steps and sorted(before) == sorted(after):
            return ("noop_filter" if i <= n_filters else "noop_transform"), None

    final_values = trace[-1]
    if len(final_values) < rules.min_values_at_final_op:
        return "too_few_at_final_op", None

    op = pipeline["op"]
    has_repeat = len(set(final_values)) < len(final_values)
    if op["name"] == "mode" and rules.mode_without_repeat and not has_repeat:
        return "mode_without_repeat", None
    if op["name"] == "unique_count" and rules.unique_count_without_repeat and not has_repeat:
        return "unique_count_without_repeat", None
    if op["name"].startswith("bitwise") and any(v < 0 for v in final_values):
        return "bitwise_negative", None

    try:
        answer = _final_op(final_values, op)
    except (ValueError, ZeroDivisionError, OverflowError):
        return "invalid_params", None
    if abs(answer) > rules.max_abs_answer:
        return "answer_too_large", None
    return None, int(answer)


def reject_reason(pipeline: dict[str, Any], rules: RejectRules | None = None) -> str | None:
    """Name of the SPEC §4 v1.2 rule a pipeline violates, or ``None`` if it is acceptable."""
    reason, _ = _check_candidate(pipeline, rules or RejectRules())
    return reason


# ---------------------------------------------------------------------------
# Canonical id / IO
# ---------------------------------------------------------------------------


def canonical_id(pipeline: dict[str, Any]) -> str:
    """sha256 over ``json.dumps(pipeline, sort_keys=True, separators=(',', ':'))``."""
    payload = json.dumps(pipeline, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(resolved_config: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of a resolved config."""
    payload = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
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


def _pick(rng: np.random.Generator, templates: list[str] | tuple[str, ...], **kwargs: Any) -> str:
    idx = int(rng.integers(0, len(templates)))
    return templates[idx].format(**kwargs)


def _filter_phrase(rng: np.random.Generator, filt: dict[str, Any]) -> str:
    return _pick(
        rng,
        _FILTER_PHRASES[filt["name"]],
        n=filt.get("n"),
        t=filt.get("t"),
        s=filt.get("s"),
        d=filt.get("d"),
    )


def _transform_phrase(rng: np.random.Generator, transform: dict[str, Any]) -> str:
    name = transform["name"]
    k = transform.get("k")
    m = transform.get("m")
    if name == "add" and k is not None and int(k) < 0:
        return _pick(rng, _TRANSFORM_PHRASES["add_negative"], k=-int(k))
    m1 = int(m) - 1 if m is not None else None
    return _pick(rng, _TRANSFORM_PHRASES[name], k=k, m=m, m1=m1)


def _op_phrase(rng: np.random.Generator, op: dict[str, Any]) -> str:
    return _pick(rng, _OP_PHRASES[op["name"]], m=op.get("m"))


def render(pipeline: dict[str, Any], rng: np.random.Generator) -> str:
    """Natural-language rendering: position-aware connectives, paraphrased operation phrases.

    Sentence layout: ``<range>. First, <step 1>. Then|Next, <step i>. ... Finally|Of these
    numbers, <final op>.`` Steps are filters then transforms, in pipeline order.
    """
    lo = int(pipeline["range"]["lo"])
    hi = int(pipeline["range"]["hi"])
    parts: list[str] = [_pick(rng, _RANGE_TEMPLATES, lo=lo, hi=hi)]

    phrases: list[str] = [_filter_phrase(rng, f) for f in pipeline["filters"]]
    phrases += [_transform_phrase(rng, t) for t in pipeline["transforms"]]
    for i, phrase in enumerate(phrases):
        connective = CONNECTIVE_FIRST if i == 0 else _pick(rng, CONNECTIVES_MIDDLE)
        parts.append(f"{connective}, {phrase}.")

    last = _pick(rng, CONNECTIVES_LAST)
    parts.append(f"{last}, {_op_phrase(rng, pipeline['op'])}.")
    return " ".join(parts)


def _check_paraphrase_coverage(minimum: int) -> None:
    for mapping in (_FILTER_PHRASES, _TRANSFORM_PHRASES, _OP_PHRASES):
        for name, templates in mapping.items():
            if len(templates) < minimum:
                raise RuntimeError(
                    f"operator {name} has {len(templates)} templates; need >= {minimum}"
                )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_range(rng: np.random.Generator, cfg: GenConfig, scale: str) -> tuple[int, int]:
    """Span uniform in the band, then lo uniform with ``hi = lo + span <= lo_min + span_max``."""
    span_min, span_max = cfg.bands[scale]
    span = int(rng.integers(span_min, span_max + 1))
    lo = int(rng.integers(cfg.lo_min, cfg.lo_min + span_max - span + 1))
    return lo, lo + span


def _sample_filter(
    rng: np.random.Generator, lo: int, hi: int, allowed: tuple[str, ...]
) -> dict[str, Any]:
    name = str(allowed[int(rng.integers(0, len(allowed)))])
    filt: dict[str, Any] = {"name": name}
    if name in ("divisible_by", "not_divisible_by"):
        filt["n"] = int(rng.integers(2, 13))
    elif name in ("below_threshold", "above_threshold"):
        # Threshold inside or just outside the range; no-op / empty outcomes are rejected.
        filt["t"] = int(rng.integers(lo, hi + 2))
    elif name == "digit_sum_equals":
        filt["s"] = int(rng.integers(1, 19))
    elif name == "contains_digit":
        filt["d"] = int(rng.integers(0, 10))
    return filt


def _sample_transform(rng: np.random.Generator) -> dict[str, Any]:
    name = str(TRANSFORM_NAMES[int(rng.integers(0, len(TRANSFORM_NAMES)))])
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
    name = str(FINAL_OP_NAMES[int(rng.integers(0, len(FINAL_OP_NAMES)))])
    op: dict[str, Any] = {"name": name}
    if name == "product_mod":
        op["m"] = int(rng.integers(2, 97))
    return op


def _sample_counts(rng: np.random.Generator, cfg: GenConfig, total_steps: int) -> tuple[int, int]:
    """Pick ``(n_filters, n_transforms)`` uniformly among pairs with ``n_f + n_t + 1 == steps``."""
    f_min, f_max = cfg.n_filters
    t_min, t_max = cfg.n_transforms
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


def _sample_candidate(
    rng: np.random.Generator, cfg: GenConfig, scale: str, total_steps: int
) -> dict[str, Any]:
    """One unchecked pipeline for a given (scale, total_steps) cell."""
    n_filters, n_transforms = _sample_counts(rng, cfg, total_steps)
    lo, hi = _sample_range(rng, cfg, scale)
    allowed = cfg.allowed_filters
    filters = [_sample_filter(rng, lo, hi, allowed) for _ in range(n_filters)]
    transforms = [_sample_transform(rng) for _ in range(n_transforms)]
    return {
        "range": {"lo": lo, "hi": hi},
        "filters": filters,
        "transforms": transforms,
        "op": _sample_op(rng),
    }


# ---------------------------------------------------------------------------
# Stratified generation with per-cell accounting
# ---------------------------------------------------------------------------


@dataclass
class CellStats:
    """Accounting for one (range_scale × total_steps) cell."""

    scale: str
    total_steps: int
    target: int | None  # None when not stratified
    accepted: int = 0
    attempts: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return self.attempts - self.accepted

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.attempts if self.attempts else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "range_scale": self.scale,
            "total_steps": self.total_steps,
            "target": self.target,
            "accepted": self.accepted,
            "attempts": self.attempts,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejection_rate, 4),
            "reasons": dict(sorted(self.reasons.items())),
        }


@dataclass
class PoolStats:
    """Generation provenance + cell table; stored next to the pool as ``*.meta.json``."""

    seed: int
    config: dict[str, Any]  # resolved GenConfig.to_dict()
    cells: list[CellStats]
    wall_clock_s: float = 0.0

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    def to_dict(self) -> dict[str, Any]:
        total_attempts = sum(c.attempts for c in self.cells)
        total_accepted = sum(c.accepted for c in self.cells)
        reasons: Counter[str] = Counter()
        for c in self.cells:
            reasons.update(c.reasons)
        return {
            "seed": self.seed,
            "config_hash": self.config_hash,
            "config": self.config,
            "n_problems": total_accepted,
            "n_attempts": total_attempts,
            "rejection_rate": round(
                (total_attempts - total_accepted) / total_attempts if total_attempts else 0.0, 4
            ),
            "rejection_reasons": dict(sorted(reasons.items())),
            "cells": [c.to_dict() for c in self.cells],
            "wall_clock_s": round(self.wall_clock_s, 3),
        }

    def format_table(self) -> str:
        lines = ["cell table (range_scale × total_steps):"]
        lines.append(
            f"  {'scale':<5} {'steps':>5} {'target':>7} {'accepted':>8} {'attempts':>8} {'reject%':>8}"
        )
        for c in self.cells:
            target = "-" if c.target is None else str(c.target)
            lines.append(
                f"  {c.scale:<5} {c.total_steps:>5} {target:>7} {c.accepted:>8} "
                f"{c.attempts:>8} {100 * c.rejection_rate:>7.1f}%"
            )
        lines.append("rejection reasons per cell:")
        for c in self.cells:
            top = ", ".join(f"{k}={v}" for k, v in c.reasons.most_common())
            lines.append(f"  {c.scale} {c.total_steps}: {top or '-'}")
        return "\n".join(lines)


def _cell_targets(cfg: GenConfig) -> list[CellStats]:
    cells = [
        CellStats(scale=s, total_steps=k, target=None) for s in cfg.scales for k in cfg.total_steps
    ]
    if cfg.stratify:
        base, remainder = divmod(cfg.n_problems, len(cells))
        for i, cell in enumerate(cells):
            cell.target = base + (1 if i < remainder else 0)
    return cells


def _attempt(
    rng: np.random.Generator,
    cfg: GenConfig,
    cell: CellStats,
    seen_ids: set[str],
) -> Problem | None:
    """One sampling attempt for ``cell``; records the rejection reason on failure."""
    cell.attempts += 1
    pipeline = _sample_candidate(rng, cfg, cell.scale, cell.total_steps)
    reason, answer = _check_candidate(pipeline, cfg.reject)
    if reason is not None:
        cell.reasons[reason] += 1
        return None
    pid = canonical_id(pipeline)
    if pid in seen_ids:
        cell.reasons["duplicate_id"] += 1
        return None
    text = render(pipeline, rng)
    recomputed = execute_pipeline(pipeline)
    assert recomputed == answer, f"answer mismatch for {pid}: {recomputed} != {answer}"
    n_filters = len(pipeline["filters"])
    n_transforms = len(pipeline["transforms"])
    assert n_filters + n_transforms + 1 == cell.total_steps
    problem = Problem(
        problem_id=pid,
        text=text,
        answer=int(recomputed),
        pipeline=pipeline,
        range_scale=cell.scale,  # type: ignore[arg-type]
        n_filters=n_filters,
        n_transforms=n_transforms,
        total_steps=cell.total_steps,
    )
    seen_ids.add(pid)
    cell.accepted += 1
    return problem


def generate_pool_with_stats(config: dict[str, Any], seed: int) -> tuple[list[Problem], PoolStats]:
    """Generate ``config['n_problems']`` problems deterministically from ``seed``, with stats.

    Stratified mode fills each (scale × total_steps) cell to its target in order; the
    non-stratified mode draws a cell uniformly per attempt until ``n_problems`` are accepted.
    """
    cfg = GenConfig.from_dict(config)
    _check_paraphrase_coverage(cfg.paraphrases_per_operator)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    cells = _cell_targets(cfg)
    problems: list[Problem] = []
    seen_ids: set[str] = set()

    if cfg.stratify:
        for cell in cells:
            assert cell.target is not None
            max_attempts = max(20_000, 1_000 * cell.target)
            while cell.accepted < cell.target:
                if cell.attempts >= max_attempts:
                    raise RuntimeError(
                        f"cell ({cell.scale}, steps={cell.total_steps}): only "
                        f"{cell.accepted}/{cell.target} accepted after {cell.attempts} attempts; "
                        f"reasons={dict(cell.reasons.most_common())}"
                    )
                problem = _attempt(rng, cfg, cell, seen_ids)
                if problem is not None:
                    problems.append(problem)
    else:
        by_key = {(c.scale, c.total_steps): c for c in cells}
        max_attempts = max(20_000, 1_000 * cfg.n_problems)
        attempts = 0
        while len(problems) < cfg.n_problems:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"failed to sample {cfg.n_problems} valid problems after {max_attempts} "
                    f"attempts (got {len(problems)})"
                )
            scale = cfg.scales[int(rng.integers(0, len(cfg.scales)))]
            steps = cfg.total_steps[int(rng.integers(0, len(cfg.total_steps)))]
            problem = _attempt(rng, cfg, by_key[(scale, steps)], seen_ids)
            if problem is not None:
                problems.append(problem)

    stats = PoolStats(
        seed=int(seed),
        config=cfg.to_dict(),
        cells=cells,
        wall_clock_s=time.perf_counter() - started,
    )
    return problems, stats


def generate_pool(config: dict[str, Any], seed: int) -> list[Problem]:
    """Generate ``config['n_problems']`` problems deterministically from ``seed``."""
    problems, _ = generate_pool_with_stats(config, seed)
    return problems


# ---------------------------------------------------------------------------
# Provenance sidecar + CLI
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def meta_path_for(output: str | Path) -> Path:
    """``data/pool/pool.jsonl`` -> ``data/pool/pool.meta.json``."""
    output = Path(output)
    return output.with_name(output.stem + ".meta.json")


def write_pool_meta(path: str | Path, stats: PoolStats, *, output: str | Path) -> None:
    """Store cell counts, rejection accounting, and provenance next to the pool JSONL."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        pkg_version: str | None = version("rlordata")
    except PackageNotFoundError:
        pkg_version = None
    record = {
        "output": str(output),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "rlordata_version": pkg_version,
        "numpy_version": np.__version__,
        **stats.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")


def _print_stats(problems: list[Problem], stats: PoolStats) -> None:
    by_scale: Counter[str] = Counter(p.range_scale for p in problems)
    by_steps: Counter[int] = Counter(p.total_steps for p in problems)
    by_op: Counter[str] = Counter(str(p.pipeline["op"]["name"]) for p in problems)
    print(f"n_problems: {len(problems)}")
    print(
        f"seed: {stats.seed}  config_hash: {stats.config_hash[:12]}  "
        f"wall-clock: {stats.wall_clock_s:.1f}s"
    )
    print(stats.format_table())
    total = stats.to_dict()
    print(
        f"overall: {total['n_attempts']} attempts, rejection rate {100 * total['rejection_rate']:.1f}%"
    )
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
    problems, stats = generate_pool_with_stats(config, seed=seed)
    _print_stats(problems, stats)
    if args.dry_run:
        print("dry-run: not writing JSONL")
        return 0
    out = Path(config.get("output", "data/pool/pool.jsonl"))
    write_jsonl(problems, out)
    meta = meta_path_for(out)
    write_pool_meta(meta, stats, output=out)
    print(f"wrote {len(problems)} problems -> {out}")
    print(f"wrote cell table + provenance -> {meta}")
    return 0
