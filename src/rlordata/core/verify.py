"""verify.py — the reward function. Hand-written (Laksh). Pure Python.

INTUITION
    The verifier is the experiment. It turns a completion (a string) into a reward (a number).
    Everything downstream — RFT filtering, GRPO advantages, every accuracy number — is only as
    honest as this function. It must be strict, deterministic, and boring.

PRECISE (SPEC §5, v1.5)
    The answer is the LAST integer, by position, written in an explicit final-answer form:
      (a) an answer line: `Answer: <int>` — case-insensitive, optional `Final `, optional markdown
          bold (`**Answer:**` / `**Answer**:`), integer optionally wrapped in `$…$` or `\\boxed{…}`,
          optional single trailing `.` or `,`; nothing else on the line;
      (b) `\\boxed{<int>}` anywhere.
    `<int>` is -?\\d+ with optional digit-group commas (removed). If nothing matches: extraction
    failed, reward 0. Otherwise reward 1.0 iff int == problem.answer, else 0.0.
    v1.4 was the single form ^Answer:\\s*(-?\\d+)\\s*$; widened on 2026-09-08 (SPEC §12 v1.5) after
    the base model wrote \\boxed{} / **Answer:** in 61 % of T=1.0 samples. Still no prose, no
    trailing text: leniency beyond an explicit final-answer form becomes reward hacking later.
    Primary arms use binary correctness only. Format/length rewards live in configs/controls and are
    implemented as *separate* reward functions in train/ (agent-owned), never here.

TESTS YOU WRITE (tests/core/test_verify.py)
    - exact answer on last line -> 1.0
    - correct answer present but not on the last line in the required form -> 0.0 + extraction_failed
    - negative numbers, leading zeros, trailing whitespace, "Answer: 16." (trailing period accepted, v1.5)
    - "\\boxed{16}" anywhere, "**Answer:** 16", "Answer: 1,000" -> 1000; "Answer: 16 numbers" fails
    - two "Answer:" lines -> the LAST one counts
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rlordata.types import Problem

EXTRACTION_RULE = "v1.5"  # SPEC §5 / §12; recorded in every run's resolved config
_INT = r"(-?\d{1,3}(?:,\d{3})+|-?\d+)"  # digit-group commas allowed only in groups of three
ANSWER_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?(?:final\s+)?answer(?::\s*\*\*|\*\*:|:)\s*\$?(?:\\boxed\{)?\s*"
    + _INT
    + r"\s*\}?\$?\s*[.,]?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{\s*" + _INT + r"\s*\}")
ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (ANSWER_LINE_RE, BOXED_RE)
ANSWER_RE = ANSWER_LINE_RE  # backwards-compatible name (the answer-line pattern)
ANSWER_RULE_TEXT = " || ".join(p.pattern for p in ANSWER_PATTERNS)  # what run configs record


@dataclass(frozen=True)
class Verdict:
    reward: float  # 1.0 or 0.0
    extracted: int | None
    extraction_failed: bool


def extract_answer(completion: str) -> int | None:
    """Return the integer from the LAST explicit final-answer form (by position), or None.

    Implemented with ANSWER_PATTERNS (SPEC §5 v1.5). Do not widen further: leniency beyond an
    explicit final-answer form becomes reward hacking later.
    """
    last: tuple[int, int] | None = None  # (position, value)
    for pattern in ANSWER_PATTERNS:
        for m in pattern.finditer(completion):
            value = int(m.group(1).replace(",", ""))
            if last is None or m.start() > last[0]:
                last = (m.start(), value)
    return None if last is None else last[1]


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
