"""AGENT-OWNED tests of the SPEC §5 v1.5 extraction rule, on the exact forms seen in the provisional
cap run (Qwen3-4B-Base, T=1.0, 2026-09-08). Laksh's cases live in tests/core/test_verify.py."""

from __future__ import annotations

import pytest

from rlordata.core.verify import ANSWER_RULE_TEXT, EXTRACTION_RULE, extract_answer, verify

ACCEPTED = {
    "Some reasoning...\nAnswer: 16": 16,
    "**Answer:** 256": 256,
    "**Answer**: 256": 256,
    "Final Answer: 0": 0,
    "**Final Answer:** 40500": 40500,
    "answer: 16": 16,
    "Answer:16": 16,
    "Answer: -5": -5,
    "Answer: 007": 7,
    "Answer: 1214.": 1214,
    "Answer: 441,720": 441720,
    "Answer: $16$": 16,
    "Answer: \\boxed{16}": 16,
    "Answer: $\\boxed{16}$": 16,
    "there are 8 numbers.\n\n\\[\n\\boxed{8}\n\\]": 8,
    "the final answer is \\( \\boxed{56} \\).": 56,
    "4. **Final Answer:**\n   \\boxed{14}": 14,
    "\\boxed{1,000}": 1000,
    "Answer: 12\nActually wait.\nAnswer: 16": 16,
    "Answer: 12\n... later \\boxed{16}": 16,  # last by position, across forms
    "\\boxed{12}\nAnswer: 16": 16,
    "Answer: 16   \n": 16,
}
REJECTED = [
    "",
    "Answer: <integer>",
    "Answer: ed{9}\\)",
    "The smallest remaining number is 6.",
    "Thus, the final answer is 68.",  # prose only
    "Answer: 16 numbers",  # trailing text
    "Answer: 3.5",
    "Answer: sixteen",
    "Answer: 1,2",  # not a digit-group comma
    "Answers: 16",
    "boxed{16}",  # no backslash
]


@pytest.mark.parametrize(("text", "expected"), list(ACCEPTED.items()))
def test_accepted_forms(text: str, expected: int) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize("text", REJECTED)
def test_rejected_forms(text: str) -> None:
    assert extract_answer(text) is None


def test_verify_uses_the_rule(toy_problem) -> None:
    v = verify(toy_problem, "So we get\n\\boxed{16}")
    assert v.reward == 1.0 and v.extracted == 16 and not v.extraction_failed
    v = verify(toy_problem, "**Answer:** 15")
    assert v.reward == 0.0 and v.extracted == 15 and not v.extraction_failed
    v = verify(toy_problem, "the answer is 16")
    assert v.reward == 0.0 and v.extracted is None and v.extraction_failed


def test_rule_is_recorded() -> None:
    assert EXTRACTION_RULE == "v1.5"
    assert "boxed" in ANSWER_RULE_TEXT and "||" in ANSWER_RULE_TEXT
