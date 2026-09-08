"""verify.py — the reward function. Hand-written (Laksh). Pure Python.

INTUITION
    The verifier is the experiment. It turns a completion (a string) into a reward (a number).
    Everything downstream — RFT filtering, GRPO advantages, every accuracy number — is only as
    honest as this function. It must be strict, deterministic, and boring.

PRECISE (SPEC §5)
    Extract the last line matching  ^Answer:\\s*(-?\\d+)\\s*$  . If none matches: extraction failed,
    reward 0. Otherwise reward 1.0 iff int(match) == problem.answer, else 0.0.
    Primary arms use binary correctness only. Format/length rewards live in configs/controls and are
    implemented as *separate* reward functions in train/ (agent-owned), never here.

TESTS YOU WRITE (tests/core/test_verify.py)
    - exact answer on last line -> 1.0
    - correct answer present but not on the last line in the required form -> 0.0 + extraction_failed
    - negative numbers, leading zeros, trailing whitespace, "Answer: 16." (trailing period must fail)
    - two "Answer:" lines -> the LAST one counts
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rlordata.types import Problem

ANSWER_RE = re.compile(r"^Answer:\s*(-?\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Verdict:
    reward: float  # 1.0 or 0.0
    extracted: int | None
    extraction_failed: bool


def extract_answer(completion: str) -> int | None:
    """Return the integer from the LAST well-formed 'Answer: <int>' line, or None.

    Implement with ANSWER_RE. Do not be lenient: leniency here becomes reward hacking later.
    """
    matches = ANSWER_RE.findall(completion)
    if not matches:
        return None
    return int(matches[-1])
    # raise NotImplementedError("Laksh: implement extract_answer")


def verify(problem: Problem, completion: str) -> Verdict:
    """Binary correctness verdict for one completion (SPEC §5)."""
    extracted = extract_answer(completion)
    if extracted is None:
        return Verdict(reward=0.0, extracted=None, extraction_failed=True)
    if extracted == problem.answer:
        return Verdict(reward=1.0, extracted=extracted, extraction_failed=False)
    return Verdict(reward=0.0, extracted=extracted, extraction_failed=False)


def verify_batch(problems: list[Problem], completions: list[str]) -> list[Verdict]:
    """Elementwise verify; len(problems) == len(completions)."""
    assert len(problems) == len(completions)
    return [verify(p, c) for p, c in zip(problems, completions, strict=True)]
