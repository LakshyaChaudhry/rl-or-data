import pytest

from rlordata.core.rft_select import SFTExample, rft_select
from rlordata.sampling.prompts import render_template
from rlordata.types import Problem, Sample

pytestmark = pytest.mark.core


def _problem(i: int, tier: str = "medium") -> Problem:
    return Problem(
        problem_id=f"p{i}",
        text=f"Consider the integers from 1 to {10 + i}, inclusive. First, keep only the even numbers. Of these numbers, count how many values remain.",
        answer=5,
        pipeline={
            "range": {"lo": 1, "hi": 10 + i},
            "filters": [],
            "transforms": [],
            "op": {"name": "count"},
        },
        range_scale="S",
        n_filters=1,
        n_transforms=0,
        total_steps=2,
        tier=tier,  # type: ignore[arg-type]
    )


def _sample(p: Problem, j: int, correct: bool, completion: str | None = None) -> Sample:
    text = completion or f"steps...\nAnswer: {p.answer if correct else p.answer + 1 + j}"
    return Sample(
        run_id="r",
        config_hash="h",
        seed=1,
        arm="base",
        data_condition="train_mixed_100",
        problem_id=p.problem_id,
        tier=p.tier,
        prompt=render_template(p.text),
        completion=text,
        extracted_answer=p.answer if correct else p.answer + 1,
        correct=correct,
        reward=1.0 if correct else 0.0,
        n_tokens=20,
        truncated=False,
        extraction_failed=False,
        extra={"sample_idx": j},
    )


def test_zero_of_eight_dropped_by_both():
    p = _problem(0, "hard")
    ss = [_sample(p, j, False) for j in range(8)]
    samples = {p.problem_id: ss}
    assert rft_select([p], samples, mode="all") == []
    assert rft_select([p], samples, mode="curated") == []


def test_eight_of_eight_kept_by_all_dropped_by_curated():
    p = _problem(1, "easy")
    # Distinct correct completions so dedup does not collapse them.
    ss = [_sample(p, j, True, completion=f"Answer: {p.answer} #{j}") for j in range(8)]
    samples = {p.problem_id: ss}
    all_ex = rft_select([p], samples, mode="all")
    assert len(all_ex) == 8
    assert all(isinstance(e, SFTExample) for e in all_ex)
    assert rft_select([p], samples, mode="curated") == []


def test_three_of_eight_both_keep_exactly_three_after_dedup():
    p = _problem(2, "medium")
    ss = []
    for j in range(8):
        if j in (1, 3, 5):
            ss.append(_sample(p, j, True, completion=f"good-{j}"))
        else:
            ss.append(_sample(p, j, False))
    # Duplicate one correct completion — should not create a 4th example.
    ss.append(_sample(p, 8, True, completion="good-1"))
    samples = {p.problem_id: ss}
    for mode in ("all", "curated"):
        ex = rft_select([p], samples, mode=mode)
        assert len(ex) == 3
        assert {e.completion for e in ex} == {"good-1", "good-3", "good-5"}


def test_first_eight_uses_sample_order_not_reward_order():
    """pass8 is counted on ss[:8] in list order, even if later samples are all correct."""
    p = _problem(3, "hard")
    # First 8 all wrong → curated drops; samples 8..15 all correct would not save it.
    ss = [_sample(p, j, False) for j in range(8)]
    ss += [_sample(p, j, True, completion=f"late-{j}") for j in range(8, 16)]
    samples = {p.problem_id: ss}
    assert rft_select([p], samples, mode="curated") == []
    # all mode still keeps the late correct ones
    assert len(rft_select([p], samples, mode="all")) == 8

    # Flip: first 8 all correct → curated drops even if later ones mix.
    p2 = _problem(4, "easy")
    ss2 = [_sample(p2, j, True, completion=f"early-{j}") for j in range(8)]
    ss2 += [_sample(p2, j, j % 2 == 0, completion=f"late-{j}") for j in range(8, 12)]
    assert rft_select([p2], {p2.problem_id: ss2}, mode="curated") == []


def test_max_per_problem_caps_after_dedup():
    p = _problem(5, "medium")
    ss = [_sample(p, j, True, completion=f"c-{j}") for j in range(8)]
    samples = {p.problem_id: ss}
    ex = rft_select([p], samples, mode="all", max_per_problem=2)
    assert len(ex) == 2
    assert [e.completion for e in ex] == ["c-0", "c-1"]
