from __future__ import annotations

import pytest

from rlordata.data import transfer
from rlordata.sampling.prompts import TEMPLATE, format_prompt

MISSING = [
    name
    for name, ok in (
        ("reasoning-gym", transfer.reasoning_gym_available()),
        ("datasets", transfer.datasets_available()),
    )
    if not ok
]


def test_parse_int_answer_cases() -> None:
    assert transfer.parse_int_answer("27") == 27
    assert transfer.parse_int_answer(" -5 ") == -5
    assert transfer.parse_int_answer("1,000") == 1000
    assert transfer.parse_int_answer(30) == 30
    assert transfer.parse_int_answer("['12.5', '-3.00']") is None  # number_filtering format
    assert transfer.parse_int_answer("3.5") is None
    assert transfer.parse_int_answer(True) is None


def test_parse_gsm8k_answer() -> None:
    field = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72"
    assert transfer.parse_gsm8k_answer(field) == 72
    assert transfer.parse_gsm8k_answer("x\n#### 1,000") == 1000
    assert transfer.parse_gsm8k_answer("x\n#### -3") == -3
    with pytest.raises(ValueError):
        transfer.parse_gsm8k_answer("no marker")
    with pytest.raises(ValueError):
        transfer.parse_gsm8k_answer("x\n#### 2.5")


def test_gsm8k_problems_from_rows_render_with_template() -> None:
    rows = [{"question": f"Q{i}?", "answer": f"steps\n#### {i * 3}"} for i in range(5)]
    probs = transfer.gsm8k_problems(rows, n=3)
    assert [p.answer for p in probs] == [0, 3, 6]
    assert all(
        p.split == "gsm8k_3" and p.pipeline["source"] == "gsm8k" and p.tier == "untiered"
        for p in probs
    )
    assert len({p.problem_id for p in probs}) == 3
    assert transfer.gsm8k_problems(rows, n=3) == probs  # deterministic ids
    assert format_prompt(probs[0], "base") == TEMPLATE.format(problem_text="Q0?")
    with pytest.raises(ValueError):
        transfer.gsm8k_problems(rows, n=10)


def test_number_filtering_is_rejected_as_non_integer() -> None:
    with pytest.raises(transfer.TransferTaskIncompatibleError):
        transfer.load_reasoning_gym("number_filtering")
    with pytest.raises(ValueError):
        transfer.load_reasoning_gym("not_a_task")


@pytest.mark.skipif(
    not transfer.reasoning_gym_available(), reason=f"optional deps missing: {MISSING}"
)
@pytest.mark.parametrize("task", ["basic_arithmetic", "count_primes"])
def test_reasoning_gym_candidates_load_300_integer_problems(task: str) -> None:
    probs = transfer.load_reasoning_gym(task)
    assert len(probs) == 300
    assert all(isinstance(p.answer, int) for p in probs)
    assert len({p.problem_id for p in probs}) == 300
    assert all(p.split == f"rg_{task}_300" and p.pipeline["task"] == task for p in probs)
    again = transfer.load_reasoning_gym(task)
    assert [p.problem_id for p in again] == [p.problem_id for p in probs]


@pytest.mark.slow
@pytest.mark.skipif(not transfer.datasets_available(), reason=f"optional deps missing: {MISSING}")
def test_gsm8k_test_first_500_downloads() -> None:
    try:
        probs = transfer.load_gsm8k_test()
    except Exception as exc:  # noqa: BLE001 — network
        pytest.skip(f"GSM8K download failed: {exc}")
    assert len(probs) == 500 and all(isinstance(p.answer, int) for p in probs)
