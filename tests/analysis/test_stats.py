from __future__ import annotations

import math

import numpy as np

from rlordata.analysis import stats


def _scores(
    rng: np.random.Generator, p: float, n_seeds: int = 3, n_problems: int = 200
) -> list[np.ndarray]:
    return [(rng.random(n_problems) < p).astype(np.float64) for _ in range(n_seeds)]


def test_pooled_seed_std_is_the_pooled_sample_variance() -> None:
    a, b = [0.70, 0.72, 0.74], [0.60, 0.66, 0.63]
    want = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)  # equal n: mean of the variances
    assert math.isclose(stats.pooled_seed_std(a, b), want)
    assert math.isnan(stats.pooled_seed_std([0.5], b))


def test_paired_contrast_pairs_seed_i_with_seed_i() -> None:
    rng = np.random.default_rng(0)
    a, b = _scores(rng, 0.8), _scores(rng, 0.6)
    c = stats.paired_contrast([1, 2, 3], a, b, n_boot=300, seed=0)
    want = [float(x.mean() - y.mean()) for x, y in zip(a, b, strict=True)]
    assert np.allclose(c.delta_per_seed, want)
    assert math.isclose(c.mean_delta, float(np.mean(want)))
    assert math.isclose(c.std_delta, float(np.std(want, ddof=1)))
    assert c.ci_low <= c.mean_delta <= c.ci_high
    assert c.n_problems == 200
    # deterministic given the seed; a different seed moves the interval
    assert c == stats.paired_contrast([1, 2, 3], a, b, n_boot=300, seed=0)
    assert c.ci_low != stats.paired_contrast([1, 2, 3], a, b, n_boot=300, seed=1).ci_low


def test_single_baseline_is_paired_with_every_seed_and_has_no_pooled_std() -> None:
    rng = np.random.default_rng(1)
    a, base = _scores(rng, 0.7, n_seeds=1), _scores(rng, 0.5, n_seeds=1)
    c = stats.paired_contrast([1], a, base, n_boot=200, seed=0)
    assert math.isnan(c.pooled_seed_std)
    crit = stats.evaluate_criterion(c, c)
    assert crit.magnitude_ok is None and crit.met is None and crit.same_sign_on_ood is True


def _contrast(delta: float, pooled: float) -> stats.Contrast:
    return stats.Contrast(
        (1, 2, 3), (0.0,) * 3, (0.0,) * 3, (delta,) * 3, delta, 0.0, pooled, delta, delta, 10, False
    )


def test_criterion_needs_magnitude_and_the_same_sign_on_ood() -> None:
    big, small = _contrast(0.10, 0.02), _contrast(0.03, 0.02)
    assert stats.evaluate_criterion(big, _contrast(0.01, 0.5)).met is True
    assert stats.evaluate_criterion(big, _contrast(-0.01, 0.5)).met is False  # sign flips on ood
    assert (
        stats.evaluate_criterion(big, _contrast(0.0, 0.5)).met is False
    )  # zero is not 'same sign'
    assert (
        stats.evaluate_criterion(small, _contrast(0.01, 0.5)).met is False
    )  # within 2 x pooled std
    assert stats.evaluate_criterion(_contrast(-0.10, 0.02), _contrast(-0.2, 0.5)).met is True
    assert math.isclose(stats.evaluate_criterion(big, big).threshold, 0.04)


def test_ratio_reports_an_unbounded_interval_when_the_denominator_straddles_zero() -> None:
    rng = np.random.default_rng(2)
    common = _scores(rng, 0.6)
    clear = stats.ratio_of_contrasts(
        _scores(rng, 0.7), _scores(rng, 0.9), common, n_boot=500, seed=0
    )
    assert clear.ci_is_bounded and clear.ci_low <= clear.value <= clear.ci_high
    assert math.isclose(clear.value, clear.numerator / clear.denominator)
    assert len(clear.per_seed) == 3
    vague = stats.ratio_of_contrasts(
        _scores(rng, 0.7), _scores(rng, 0.6), common, n_boot=500, seed=0
    )
    assert not vague.ci_is_bounded and vague.frac_boot_den_nonpositive > 0.025


def test_summarize_arm_keeps_every_seed() -> None:
    rng = np.random.default_rng(3)
    s = stats.summarize_arm([1, 2, 3], _scores(rng, 0.75), n_boot=300, seed=0)
    assert len(s.per_seed) == 3 and s.seed_min <= s.mean <= s.seed_max
    assert math.isclose(s.std, float(np.std(s.per_seed, ddof=1)))
    assert s.ci_low <= s.mean <= s.ci_high
    assert stats.ranges_overlap([0.1, 0.3], [0.3, 0.5]) and not stats.ranges_overlap(
        [0.1, 0.2], [0.3, 0.5]
    )
