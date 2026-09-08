"""AGENT-OWNED tests of the SPEC §5 v1.6 extraction rule, on the exact forms seen in the provisional
cap run and the k=8 tiering run (Qwen3-4B-Base, T=1.0, 2026-09-08). Laksh's cases live in
tests/core/test_verify.py.

v1.6 is layered:
  (a) an explicit answer line, once presentation-only decoration is stripped — ACCEPTED_STRICT;
  (b) \\boxed{N} anywhere;
  (c) fallback: the LAST integer in the completion — FALLBACK_LAST_INT.
Only a completion with no integer at all, or one that hit the cap, fails extraction — REJECTED.
The strict layer still runs first, so a well-formed answer line beats a later incidental number.
"""

from __future__ import annotations

import pytest

from rlordata.core.verify import ANSWER_RULE_TEXT, EXTRACTION_RULE, extract_answer, verify

ACCEPTED_STRICT = {
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
    # --- decoration-only widening (SPEC §12 v1.6) ---
    "**Answer: 36**": 36,  # bold spanning label AND value — the main v1.5 miss
    "**Final Answer: 36**": 36,
    "*Answer: 36*": 36,
    "__Answer: 36__": 36,
    "Answer: **36**": 36,
    "### Answer: 36": 36,
    "#### Answer: 36": 36,
    "> Answer: 36": 36,
    "- Answer: 36": 36,
    "`Answer: 36`": 36,
    '"Answer: 36"': 36,
    "Answer: \\(36\\)": 36,
    "\\[ \\text{Answer: } 36 \\]": 36,
    "Final answer: Answer: 36": 36,
    "Answer: <36>": 36,
    "Answer: <integer>36</integer>": 36,
    "Answer: <integer> 36": 36,
    "**Answer:** 1,234": 1234,
    # label on its own line, value on the next — v1.5 accepted this because \s* spans newlines,
    # so the line-by-line normaliser must keep it (regression guard).
    "**Final Answer:**\n5": 5,
    "**Final Answer**\n5": 5,
    "Answer:\n16": 16,
    "Final Answer:\n\n  42  ": 42,
    "**Final Answer:**\n**5**": 5,
    "**Final Answer:\n45.": 45,  # emphasis opened before the label, never closed
    # the strict layer wins over a later incidental integer
    "Answer: 16\nthat took 3 steps": 16,
}

FALLBACK_LAST_INT = {
    "36": 36,
    "**36**": 36,
    "\\[ 36 \\]": 36,
    "The answer is 36.": 36,
    "Therefore, the final answer is 36.": 36,
    "The smallest remaining number is 6.": 6,
    "Answers: 16": 16,
    "boxed{16}": 16,  # no backslash: not \boxed, but the trailing integer still counts
    "Answer: 16 numbers": 16,
    "Answer: 36 and 37": 37,  # the LAST integer, not a free choice among candidates
    "so it could be 5, or 7, or 12": 12,  # listing candidates does not help
    "Step 4: the result is\n42": 42,
}

REJECTED = [
    "",
    "Answer: <integer>",  # the prompt template echoed back, no value
    "Answer:",
    "Answer: sixteen",
    'print(f"Answer: {n}")',
    "no numbers here at all",
    "I could not determine the value.",
]


@pytest.mark.parametrize(("text", "expected"), list(ACCEPTED_STRICT.items()))
def test_accepted_strict_forms(text: str, expected: int) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize(("text", "expected"), list(FALLBACK_LAST_INT.items()))
def test_fallback_takes_the_last_integer(text: str, expected: int) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize("text", REJECTED)
def test_rejected_forms(text: str) -> None:
    assert extract_answer(text) is None


def test_truncated_never_guesses() -> None:
    """A completion that hit the cap never committed to an answer (SPEC §5 v1.6 layer (c))."""
    assert extract_answer("reasoning so far 42", truncated=True) is None
    assert extract_answer("Answer: 16", truncated=True) is None
    assert extract_answer("Answer: 16", truncated=False) == 16


def test_verify_uses_the_rule(toy_problem) -> None:
    v = verify(toy_problem, "So we get\n\\boxed{16}")
    assert v.reward == 1.0 and v.extracted == 16 and not v.extraction_failed
    v = verify(toy_problem, "**Answer:** 15")
    assert v.reward == 0.0 and v.extracted == 15 and not v.extraction_failed
    v = verify(toy_problem, "the answer is 16")  # prose now scores via the fallback layer
    assert v.reward == 1.0 and v.extracted == 16 and not v.extraction_failed
    v = verify(toy_problem, "I could not work it out")
    assert v.reward == 0.0 and v.extracted is None and v.extraction_failed
    v = verify(toy_problem, "part way through, 16", truncated=True)
    assert v.reward == 0.0 and v.extracted is None and v.extraction_failed


def test_rule_is_recorded() -> None:
    assert EXTRACTION_RULE == "v1.6"
    assert "boxed" in ANSWER_RULE_TEXT and "||" in ANSWER_RULE_TEXT
    assert "fallback" in ANSWER_RULE_TEXT
