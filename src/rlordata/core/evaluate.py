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
    raise NotImplementedError("Laksh: implement pass_at_k")


def bootstrap_ci(
    per_problem_scores: np.ndarray,  # [P] in [0,1]
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over problems."""
    assert per_problem_scores.ndim == 1
    raise NotImplementedError("Laksh: implement bootstrap_ci")


def compute_metrics(samples: list[Sample], seed: int = 0) -> Metrics:
    """Aggregate a list of Samples (one or more per problem) into Metrics.

    If a problem has k samples, its score is mean(correct) over them (this gives mean@k);
    with one sample per problem it is plain accuracy.
    """
    raise NotImplementedError("Laksh: implement compute_metrics")
