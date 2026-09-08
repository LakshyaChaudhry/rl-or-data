"""evaluate.py — metrics and uncertainty. Hand-written (Laksh).

INTUITION
    A number without an error bar is an anecdote. Every accuracy we report comes with a bootstrap
    CI over problems (does the result depend on which problems we happened to sample?) and, for
    trained arms, a spread over seeds (does it depend on the training run's randomness?).

PRECISE (SPEC §10)
    accuracy       mean(correct) over problems (greedy: one sample/problem; mean@k: mean over k)
    pass@k         unbiased estimator (Chen et al. 2021):
                   for a problem with n samples and c correct,  pass@k = 1 - C(n-c, k) / C(n, k)
                   (use log-comb or the product form to avoid overflow; == 1 if n - c < k)
    bootstrap CI   resample problems with replacement B=10,000 times; percentile 2.5 / 97.5
    truncation     fraction of samples with truncated == True; extraction_failed likewise

TESTS YOU WRITE (tests/core/test_evaluate.py)
    - pass@k with c == n -> 1.0; c == 0 -> 0.0; n=8,c=1,k=1 -> 1/8; n=8,c=1,k=8 -> 1.0
    - bootstrap CI contains the point estimate and narrows with more problems
    - per-tier accuracy sums back to overall with the right weights
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlordata.types import Sample


@dataclass(frozen=True)
class Metrics:
    n_problems: int
    accuracy: float
    ci_low: float
    ci_high: float
    truncation_rate: float
    extraction_failure_rate: float
    mean_completion_tokens: float
    per_tier: dict[str, float]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem with n samples, c correct."""
    assert 0 <= c <= n and 1 <= k <= n

    if n - c < k:
        return 1.0
    
    acc = 1.0
    for i in range(k):
        acc *= (n - c - i) / (n - i)
    return 1.0 - acc
    
    #raise NotImplementedError("Laksh: implement pass_at_k")


def bootstrap_ci(
    per_problem_scores: np.ndarray,  # [P] in [0,1]
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over problems."""
    assert per_problem_scores.ndim == 1

    rng = np.random.default_rng(seed)
    p = per_problem_scores.shape[0]
    if p == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, p, size=p)  # resample problems with replacement
        means[b] = per_problem_scores[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi

    #raise NotImplementedError("Laksh: implement bootstrap_ci")


def compute_metrics(samples: list[Sample], seed: int = 0) -> Metrics:
    """Aggregate a list of Samples (one or more per problem) into Metrics.

    If a problem has k samples, its score is mean(correct) over them (this gives mean@k);
    with one sample per problem it is plain accuracy.
    """

    if not samples:
        raise ValueError("compute_metrics requires at least one sample")
    by_problem: dict[str, list[Sample]] = {}
    for s in samples:
        by_problem.setdefault(s.problem_id, []).append(s)
    problem_ids = sorted(by_problem.keys())
    scores = []
    tiers = []
    for pid in problem_ids:
        group = by_problem[pid]
        scores.append(float(np.mean([float(s.correct) for s in group])))
        tiers.append(group[0].tier)
    scores_arr = np.asarray(scores, dtype=np.float64)
    acc = float(scores_arr.mean())
    ci_low, ci_high = bootstrap_ci(scores_arr, seed=seed)
    trunc = float(np.mean([float(s.truncated) for s in samples]))
    ext_fail = float(np.mean([float(s.extraction_failed) for s in samples]))
    mean_tok = float(np.mean([s.n_tokens for s in samples]))
    per_tier: dict[str, float] = {}
    for tier in sorted(set(tiers)):
        mask = [t == tier for t in tiers]
        per_tier[tier] = float(scores_arr[mask].mean())
    return Metrics(
        n_problems=len(problem_ids),
        accuracy=acc,
        ci_low=ci_low,
        ci_high=ci_high,
        truncation_rate=trunc,
        extraction_failure_rate=ext_fail,
        mean_completion_tokens=mean_tok,
        per_tier=per_tier,
    )

    #raise NotImplementedError("Laksh: implement compute_metrics")
