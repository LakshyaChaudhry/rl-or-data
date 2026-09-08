from __future__ import annotations

import pytest

from rlordata.data.candidates import (
    VAL_CANDIDATES_N,
    VAL_CANDIDATES_NAME,
    VAL_CANDIDATES_SEED,
    val_candidate_indices,
    val_candidates,
)
from tests.helpers import small_pool


def test_indices_deterministic_sorted_unique_and_snapshot() -> None:
    a = val_candidate_indices(6000)
    b = val_candidate_indices(6000)
    assert a == b and len(a) == VAL_CANDIDATES_N == 500 and VAL_CANDIDATES_SEED == 1
    assert a == sorted(a) and len(set(a)) == len(a)
    assert all(0 <= i < 6000 for i in a)
    # Guard against accidental RNG changes: first indices for the real pool size.
    assert a[:8] == [33, 39, 65, 81, 90, 97, 98, 109]
    assert val_candidate_indices(6000, seed=2) != a
    with pytest.raises(ValueError):
        val_candidate_indices(499)


def test_val_candidates_are_pool_problems_in_pool_order() -> None:
    pool = small_pool(n=800)
    cands = val_candidates(pool)
    assert len(cands) == 500
    ids = [p.problem_id for p in cands]
    assert len(set(ids)) == 500
    pool_pos = {p.problem_id: i for i, p in enumerate(pool)}
    positions = [pool_pos[i] for i in ids]
    assert positions == sorted(positions)
    assert all(p.split == VAL_CANDIDATES_NAME for p in cands)
    assert val_candidates(pool) == cands
