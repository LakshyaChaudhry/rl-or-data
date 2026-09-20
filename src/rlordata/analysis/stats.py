"""Seed aggregation and paired comparisons (SPEC §10). AGENT-OWNED; uses core.evaluate for per-run metrics.

Per-run accuracy, CI, truncation and pass@k come from ``core.evaluate`` (through
``sampling.eval_runner.metrics_for``); nothing here recomputes them. This module only combines
runs:

  arm summary     mean ± std across seeds (sample std, ddof = 1), every seed kept; a bootstrap CI
                  over problems of the seed-averaged per-problem score (``core.evaluate.bootstrap_ci``)
  paired contrast seed i vs seed i differences, their mean and std, the pooled seed std of the two
                  arms, and a paired per-problem bootstrap CI of the difference
  criterion       SPEC §10, evaluated mechanically: (a) |Δ test_300 greedy| > 2 × pooled seed std
                  and (b) sign(Δ) equal on ood_hard_200. No verdicts are produced here.
  ratio           H2's (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed) with a joint
                  problem-level bootstrap

Every function is deterministic given its ``seed``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rlordata.analysis.rft_report import mean_std
from rlordata.core.evaluate import bootstrap_ci

CRITERION_MULTIPLIER = 2.0  # SPEC §10 (a)


@dataclass(frozen=True)
class ArmSummary:
    seeds: tuple[int, ...]
    per_seed: tuple[float, ...]
    mean: float
    std: float  # sample std across seeds (ddof = 1); 0.0 with one seed, nan with none
    ci_low: float  # bootstrap over problems of the seed-averaged per-problem score
    ci_high: float
    n_problems: int
    seed_min: float
    seed_max: float


@dataclass(frozen=True)
class Contrast:
    """A − B on one split/metric, paired seed-wise."""

    seeds: tuple[int, ...]
    a_per_seed: tuple[float, ...]
    b_per_seed: tuple[float, ...]
    delta_per_seed: tuple[float, ...]
    mean_delta: float
    std_delta: float  # sample std of the seed-wise differences
    pooled_seed_std: float  # nan when either arm has fewer than two seeds
    ci_low: float  # paired per-problem bootstrap of mean_delta
    ci_high: float
    n_problems: int
    seed_ranges_overlap: bool


@dataclass(frozen=True)
class Criterion:
    """SPEC §10 pre-registered success criterion, evaluated mechanically."""

    threshold: float  # 2 × pooled seed std on test_300
    magnitude_ok: bool | None  # (a); None when the pooled std is undefined (single seed)
    same_sign_on_ood: bool  # (b)
    met: bool | None  # (a) and (b); None when (a) is undefined


@dataclass(frozen=True)
class Ratio:
    value: float  # mean(num) / mean(den)
    numerator: float
    denominator: float
    per_seed: tuple[float, ...]
    ci_low: float
    ci_high: float
    frac_boot_den_nonpositive: float  # share of resamples whose denominator is <= 0
    ci_is_bounded: bool  # False when the denominator is not separated from 0 at the 95 % level
    n_problems: int


def _stack(per_seed_scores: Sequence[np.ndarray]) -> np.ndarray:
    """[S, P] from S per-problem vectors of identical length."""
    arr = np.asarray(per_seed_scores, dtype=np.float64)
    assert arr.ndim == 2, "expected one [P] score vector per seed, all of the same length"
    return arr


def summarize_arm(
    seeds: Sequence[int],
    per_seed_scores: Sequence[np.ndarray],  # S × [P], same problem order in every seed
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> ArmSummary:
    arr = _stack(per_seed_scores)  # [S, P]
    assert arr.shape[0] == len(seeds)
    values = [float(v) for v in arr.mean(axis=1)]
    mu, sd = mean_std(values)
    lo, hi = bootstrap_ci(arr.mean(axis=0), n_boot=n_boot, seed=seed)
    return ArmSummary(
        seeds=tuple(int(s) for s in seeds),
        per_seed=tuple(values),
        mean=mu,
        std=sd,
        ci_low=lo,
        ci_high=hi,
        n_problems=int(arr.shape[1]),
        seed_min=min(values),
        seed_max=max(values),
    )


def pooled_seed_std(a: Sequence[float], b: Sequence[float]) -> float:
    """sqrt of the pooled sample variance of two arms' per-seed values (nan if either has < 2)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = float(np.var(np.asarray(a, dtype=np.float64), ddof=1))
    vb = float(np.var(np.asarray(b, dtype=np.float64), ddof=1))
    return math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))


def ranges_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return max(min(a), min(b)) <= min(max(a), max(b))


def paired_contrast(
    seeds: Sequence[int],
    a_scores: Sequence[np.ndarray],  # S × [P]
    b_scores: Sequence[np.ndarray],  # S × [P]; b may hold a single vector (an unseeded baseline)
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Contrast:
    """A − B. With one B vector (the base model) it is paired against every seed of A."""
    a = _stack(a_scores)  # [S, P]
    b = _stack(b_scores)  # [S, P] or [1, P]
    assert a.shape[1] == b.shape[1], "arms were not scored on the same problems"
    assert b.shape[0] in (1, a.shape[0]) and a.shape[0] == len(seeds)
    a_seed = a.mean(axis=1)
    b_seed = np.broadcast_to(b.mean(axis=1), a_seed.shape)
    delta = a_seed - b_seed
    mu, sd = mean_std([float(d) for d in delta])
    per_problem = a.mean(axis=0) - b.mean(axis=0)  # [P]
    lo, hi = bootstrap_ci(per_problem, n_boot=n_boot, seed=seed)
    return Contrast(
        seeds=tuple(int(s) for s in seeds),
        a_per_seed=tuple(float(v) for v in a_seed),
        b_per_seed=tuple(float(v) for v in b.mean(axis=1)),
        delta_per_seed=tuple(float(d) for d in delta),
        mean_delta=mu,
        std_delta=sd,
        pooled_seed_std=pooled_seed_std(list(a_seed), list(b.mean(axis=1))),
        ci_low=lo,
        ci_high=hi,
        n_problems=int(a.shape[1]),
        seed_ranges_overlap=ranges_overlap(list(a_seed), list(b.mean(axis=1))),
    )


def evaluate_criterion(test: Contrast, ood: Contrast) -> Criterion:
    """SPEC §10: |Δ test| > 2 × pooled seed std of the two arms, and the same sign of Δ on ood."""
    threshold = CRITERION_MULTIPLIER * test.pooled_seed_std
    s_test, s_ood = np.sign(test.mean_delta), np.sign(ood.mean_delta)
    same_sign = bool(s_test != 0 and s_test == s_ood)
    if math.isnan(threshold):
        return Criterion(
            threshold=threshold, magnitude_ok=None, same_sign_on_ood=same_sign, met=None
        )
    magnitude_ok = bool(abs(test.mean_delta) > threshold)
    return Criterion(
        threshold=threshold,
        magnitude_ok=magnitude_ok,
        same_sign_on_ood=same_sign,
        met=bool(magnitude_ok and same_sign),
    )


def ratio_of_contrasts(
    num_a: Sequence[np.ndarray],  # S × [P]
    den_a: Sequence[np.ndarray],  # S × [P]
    common_b: Sequence[np.ndarray],  # S × [P]
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Ratio:
    """(num_a − common_b) / (den_a − common_b), seed-wise and with a joint problem bootstrap."""
    na, da, cb = _stack(num_a), _stack(den_a), _stack(common_b)
    assert na.shape == da.shape == cb.shape
    num_p = na.mean(axis=0) - cb.mean(axis=0)  # [P]
    den_p = da.mean(axis=0) - cb.mean(axis=0)  # [P]
    num, den = float(num_p.mean()), float(den_p.mean())
    seed_num = na.mean(axis=1) - cb.mean(axis=1)
    seed_den = da.mean(axis=1) - cb.mean(axis=1)
    per_seed = tuple(
        float(n / d) if d != 0 else float("nan") for n, d in zip(seed_num, seed_den, strict=True)
    )
    rng = np.random.default_rng(seed)
    p = num_p.shape[0]
    idx = rng.integers(0, p, size=(n_boot, p))  # [B, P]
    b_num = num_p[idx].mean(axis=1)  # [B]
    b_den = den_p[idx].mean(axis=1)  # [B]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = b_num / b_den
    finite = ratios[np.isfinite(ratios)]
    frac_nonpos = float(np.mean(b_den <= 0))
    frac_nonneg = float(np.mean(b_den >= 0))
    lo = float(np.quantile(finite, 0.025)) if finite.size else float("nan")
    hi = float(np.quantile(finite, 0.975)) if finite.size else float("nan")
    return Ratio(
        value=num / den if den != 0 else float("nan"),
        numerator=num,
        denominator=den,
        per_seed=per_seed,
        ci_low=lo,
        ci_high=hi,
        frac_boot_den_nonpositive=frac_nonpos,
        ci_is_bounded=bool(frac_nonpos < 0.025 or frac_nonneg < 0.025),
        n_problems=int(p),
    )
