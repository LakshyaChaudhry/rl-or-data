"""``val_candidates``: the untiered pool subset the provisional cap run samples (tasks/02 §2).

Defined once here so the cap run, ``scripts/compute_cap.py`` and any later check agree:
500 pool problems drawn without replacement with ``numpy.random.default_rng(1)``, returned in
pool order. Deterministic given the pool file (which is itself deterministic given its seed).
"""

from __future__ import annotations

import numpy as np

from rlordata.types import Problem

VAL_CANDIDATES_N = 500
VAL_CANDIDATES_SEED = 1
VAL_CANDIDATES_NAME = "val_candidates"


def val_candidate_indices(
    n_pool: int, *, n: int = VAL_CANDIDATES_N, seed: int = VAL_CANDIDATES_SEED
) -> list[int]:
    """Sorted pool indices of the candidates. ``n_pool`` must be >= ``n``."""
    if n_pool < n:
        raise ValueError(f"pool has {n_pool} problems, need at least {n} for val_candidates")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pool, size=n, replace=False)  # [n]
    return sorted(int(i) for i in idx)


def val_candidates(
    pool: list[Problem], *, n: int = VAL_CANDIDATES_N, seed: int = VAL_CANDIDATES_SEED
) -> list[Problem]:
    """The candidate problems, tagged ``split="val_candidates"``, tier left as in the pool."""
    return [
        Problem(**{**pool[i].to_dict(), "split": VAL_CANDIDATES_NAME})
        for i in val_candidate_indices(len(pool), n=n, seed=seed)
    ]
