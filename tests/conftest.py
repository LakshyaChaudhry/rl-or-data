"""Shared fixtures. Core tests skip cleanly until Laksh implements the function under test."""

from __future__ import annotations

import pytest

from rlordata.types import Problem


def _implemented(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 — any other error means it IS implemented (and maybe wrong)
        return True
    return True


@pytest.fixture
def toy_problem() -> Problem:
    return Problem(
        problem_id="toy",
        text="Consider the integers from 1 to 100, inclusive. First, keep only the numbers that are even. "
        "Then, keep only the numbers that are divisible by 3. Of these numbers, count how many values remain.",
        answer=16,
        pipeline={
            "range": [1, 100],
            "filters": ["even", {"divisible_by": 3}],
            "transforms": [],
            "op": "count",
        },
        range_scale="M",
        n_filters=2,
        n_transforms=0,
        total_steps=3,
    )


def skip_unless_implemented(fn, *args, **kwargs):
    if not _implemented(fn, *args, **kwargs):
        pytest.skip(f"{fn.__name__} not implemented yet (Laksh)")
