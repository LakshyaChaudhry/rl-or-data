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


@pytest.fixture(autouse=True, scope="session")
def _isolated_artifacts_store(tmp_path_factory: pytest.TempPathFactory):
    """Never let a test sync into the real durable store.

    CLI entry points call ``load_env()`` and the tier/eval runners sync their outputs to
    ``RLORDATA_ARTIFACTS`` at exit; on the GPU box that is the NFS store, and stub/dry-run test
    outputs (``runs/splits``, ``runs/greedy``, ``runs/tiering_pass8.jsonl`` ...) were landing next
    to — and over — real runs. ``load_env`` never overrides an existing variable, so setting it
    here wins. Session-scoped so it precedes module-scoped fixtures that run the CLI (a
    function-scoped patch came too late: ``load_env`` had already put the real path in
    ``os.environ`` for the whole process).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RLORDATA_ARTIFACTS", str(tmp_path_factory.mktemp("artifacts")))
        yield
