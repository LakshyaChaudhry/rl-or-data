"""verify.py — the reward function. Hand-written (Laksh). Pure Python.

INTUITION
    The verifier is the experiment. It turns a completion (a string) into a reward (a number).
    Everything downstream — RFT filtering, GRPO advantages, every accuracy number — is only as
    honest as this function. It must be strict, deterministic, and boring.

PRECISE (SPEC §5, v1.6)
    The answer is the LAST integer, by position, written in an explicit final-answer form:
      (a) an answer line that, after removing presentation-only decoration, reduces exactly to
          `Answer: <int>` — case-insensitive, optional `Final `; decoration removed is: leading
          markdown headings/blockquotes/list bullets (`###`, `>`, `-`); symmetric emphasis, code
          span, quote or LaTeX wrapping of the WHOLE line (`**…**`, `__…__`, `*…*`, `_…_`,
          `` `…` ``, `"…"`, `\\[…\\]`, `\\(…\\)`); `\\text{…}`; a duplicated `Answer:` label;
          and wrapping of the VALUE by `$…$`, `\\boxed{…}`, `\\(…\\)`, `<…>`, `<integer>…</integer>`
          or emphasis; plus one optional trailing `.` or `,`. Nothing else may remain on the line;
      (b) `\\boxed{<int>}` anywhere;
      (c) FALLBACK — if neither (a) nor (b) appears, the LAST integer in the completion.
          A completion that hit the cap (`truncated=True`) never committed to an answer and
          always fails extraction; it is never guessed from a trailing integer.
    `<int>` is -?\\d+ with optional digit-group commas (removed). If nothing matches: extraction
    failed, reward 0. Otherwise reward 1.0 iff int == problem.answer, else 0.0.
    v1.4 was the single form ^Answer:\\s*(-?\\d+)\\s*$; widened on 2026-09-08 (SPEC §12 v1.5) after
    the base model wrote \\boxed{} / **Answer:** in 61 % of T=1.0 samples. v1.6 (2026-09-08) widened
    again after an audit of the tiering run found v1.5 rejected `**Answer: 36**` — bold spanning
    label AND value — and other decoration-only variants, mislabelling 339 problems `hard` that the
    model had actually solved. Layer (c) was added in the same amendment: an audit found
    23.6 % of the `hard` tier were formatting casualties, not hard problems, and the anchor
    paper (Bauer et al. §3.3) scores correctness on the NUMBER, shaping format with a separate
    additive term. Taking the LAST integer is what prevents gaming — a policy that lists
    candidates is scored on the one it ends with — not demanding a particular format. Format is
    not rewarded here: answer-line compliance is REPORTED per run and the format bonus stays a
    control condition (§8, C2).
    Primary arms use binary correctness only. Format/length rewards live in configs/controls and are
    implemented as *separate* reward functions in train/ (agent-owned), never here.

TESTS YOU WRITE (tests/core/test_verify.py)
    - exact answer on last line -> 1.0
    - no integer anywhere, or a truncated completion -> 0.0 + extraction_failed
    - negative numbers, leading zeros, trailing whitespace, "Answer: 16." (trailing period accepted, v1.5)
    - "\\boxed{16}" anywhere, "**Answer:** 16", "Answer: 1,000" -> 1000; "Answer: 16 numbers" fails
    - two "Answer:" lines -> the LAST one counts
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rlordata.types import Problem

EXTRACTION_RULE = "v1.6"  # SPEC §5 / §12; recorded in every run's resolved config
_INT = r"(-?\d{1,3}(?:,\d{3})+|-?\d+)"  # digit-group commas allowed only in groups of three
# The answer line, once decoration is stripped by _normalise_answer_line (v1.6).
ANSWER_LINE_RE = re.compile(
    r"^\s*(?:final\s+)?answer\s*:\s*\$?(?:\\boxed\{)?\s*" + _INT + r"\s*\}?\$?\s*[.,]?\s*$",
    re.IGNORECASE,
)

# --- presentation-only decoration removed before ANSWER_LINE_RE is applied (SPEC §5 v1.6) ---
_DECORATION = re.compile(r"^(?:[#>]+\s*|[-*+\u2022]\s+)+")
_WRAPPERS = (
    ("**", "**"),
    ("__", "__"),
    ("*", "*"),
    ("_", "_"),
    ("`", "`"),
    ('"', '"'),
    ("\u201c", "\u201d"),
    ("\\[", "\\]"),
    ("\\(", "\\)"),
)
_TEXT_CMD = re.compile(r"\\text\{([^{}]*)\}")
_LABEL_BOTH = re.compile(r"(?i)^(\*\*|__|\*|_)\s*((?:final\s+)?answer)\s*:\s*\1\s*")
_LABEL_WORD = re.compile(r"(?i)^(\*\*|__|\*|_)\s*((?:final\s+)?answer)\s*\1\s*:\s*")
_DUP_LABEL = re.compile(r"(?i)^(?:final\s+)?answer\s*:\s*(?=(?:final\s+)?answer\s*:)")
# an emphasis marker opened before the label and never closed ("**Final Answer:\n45.")
_LEAD_EMPH = re.compile(r"(?i)^(?:\*\*|__|\*|_)(?=\s*(?:final\s+)?answer\b)")
_VALUE_WRAPPERS = (
    re.compile(r"(?i)(answer\s*:\s*)<integer>\s*(-?[\d,]+)\s*(?:</integer>)?\s*$"),
    re.compile(r"(?i)(answer\s*:\s*)<\s*(-?[\d,]+)\s*>\s*$"),
    re.compile(r"(?i)(answer\s*:\s*)(?:\*\*|__|\*|_)\s*(-?[\d,]+)\s*(?:\*\*|__|\*|_)\s*[.,]?\s*$"),
    re.compile(r"(?i)(answer\s*:\s*)\\\(\s*(-?[\d,]+)\s*\\\)\s*[.,]?\s*$"),
)


_LABEL_ONLY = re.compile(r"(?i)^(?:final\s+)?answer\s*:?\s*$")
_BARE_INT = re.compile(r"^" + _INT + r"\s*[.,]?$")


def _normalise_answer_line(line: str) -> str:
    """Strip presentation-only decoration from one line (SPEC §5 v1.6). Never changes the integer."""
    s = _DECORATION.sub("", line.strip()).strip()
    changed = True
    while changed:  # symmetric wrappers around the whole line, outermost first
        changed = False
        for lo, hi in _WRAPPERS:
            if len(s) > len(lo) + len(hi) and s.startswith(lo) and s.endswith(hi):
                s = s[len(lo) : -len(hi)].strip()
                changed = True
                break
    s = _TEXT_CMD.sub(r"\1", s).strip()
    s = _LABEL_BOTH.sub(r"\2: ", s).strip()
    s = _LABEL_WORD.sub(r"\2: ", s).strip()
    s = _LEAD_EMPH.sub("", s).strip()  # only what the paired rules above left behind
    s = _DUP_LABEL.sub("", s).strip()
    for pattern in _VALUE_WRAPPERS:
        s = pattern.sub(r"\1\2", s).strip()
    return s


BOXED_RE = re.compile(r"\\boxed\{\s*" + _INT + r"\s*\}")
# Fallback layer (SPEC §5 v1.6): when no explicit final-answer form is present, the answer is the
# LAST integer in the completion. Anti-gaming comes from taking the LAST value — a policy that
# lists candidates is scored on the one it ends with — not from demanding a particular format.
_ANY_INT = re.compile(_INT)
ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (ANSWER_LINE_RE, BOXED_RE)
ANSWER_RE = ANSWER_LINE_RE  # backwards-compatible name (the answer-line pattern)
ANSWER_RULE_TEXT = (  # what run configs record
    "normalise_answer_line(line) then: "
    + " || ".join(p.pattern for p in ANSWER_PATTERNS)
    + " || fallback: last integer in the completion (not when truncated)"
)


@dataclass(frozen=True)
class Verdict:
    reward: float  # 1.0 or 0.0
    extracted: int | None
    extraction_failed: bool


def _last_integer(completion: str) -> int | None:
    """The last integer in the completion, scanning non-empty lines from the end (SPEC §5 v1.6)."""
    for line in reversed(completion.splitlines()):
        if not line.strip():
            continue
        found = list(_ANY_INT.finditer(line))
        if found:
            return int(found[-1].group(0).replace(",", ""))
    return None


def _extract_explicit(completion: str) -> int | None:
    """Layers (a)+(b) only: the LAST integer in an explicit final-answer form, or None (SPEC §5)."""
    last: tuple[int, int] | None = None  # (position, value)
    for m in BOXED_RE.finditer(completion):
        value = int(m.group(1).replace(",", ""))
        if last is None or m.start() > last[0]:
            last = (m.start(), value)
    raw = completion.splitlines(keepends=True)
    norm = [_normalise_answer_line(line) for line in raw]
    offsets: list[int] = []
    pos = 0
    for line in raw:
        offsets.append(pos)
        pos += len(line)
    for i, text in enumerate(norm):
        value: int | None = None
        m = ANSWER_LINE_RE.match(text)
        if m is not None:
            value = int(m.group(1).replace(",", ""))
        elif _LABEL_ONLY.match(text):
            # v1.5 allowed the label and the value on separate lines ("**Final Answer:**\n5"),
            # because \s* spans newlines; keep that form (SPEC §5 v1.6).
            for j in range(i + 1, len(norm)):
                if not norm[j]:
                    continue
                nxt = _BARE_INT.match(norm[j])
                if nxt is not None:
                    value = int(nxt.group(1).replace(",", ""))
                break
        if value is not None and (last is None or offsets[i] > last[0]):
            last = (offsets[i], value)
    return None if last is None else last[1]


def has_answer_line(completion: str, *, truncated: bool = False) -> bool:
    """True iff the answer came from an explicit final-answer form, not the fallback.

    This is what runs report as ``answer_line_rate`` (SPEC §10). It is a diagnostic only:
    format is never rewarded in the primary arms (§8).
    """
    return not truncated and _extract_explicit(completion) is not None


def extract_answer(completion: str, *, truncated: bool = False) -> int | None:
    """Return the answer integer under the layered SPEC §5 v1.6 rule, or None.

    Layers: (a) an answer line, once _normalise_answer_line strips presentation-only decoration;
    (b) ``\\boxed{N}`` anywhere; (c) fallback — the last integer in the completion. A truncated
    completion never committed to an answer and is never guessed from a trailing integer.
    Taking the LAST integer is what prevents gaming; the format itself is not rewarded.
    """
    if truncated:
        return None
    explicit = _extract_explicit(completion)
    if explicit is not None:
        return explicit
    return _last_integer(completion)


def verify(problem: Problem, completion: str, *, truncated: bool = False) -> Verdict:
    """Binary correctness verdict for one completion (SPEC §5)."""
    extracted = extract_answer(completion, truncated=truncated)
    if extracted is None:
        return Verdict(reward=0.0, extracted=None, extraction_failed=True)
    if extracted == problem.answer:
        return Verdict(reward=1.0, extracted=extracted, extraction_failed=False)
    return Verdict(reward=0.0, extracted=extracted, extraction_failed=False)


def verify_batch(
    problems: list[Problem],
    completions: list[str],
    truncated: list[bool] | None = None,
) -> list[Verdict]:
    """Elementwise verify; len(problems) == len(completions) == len(truncated) if given."""
    assert len(problems) == len(completions)
    flags = [False] * len(completions) if truncated is None else truncated
    assert len(flags) == len(completions)
    return [verify(p, c, truncated=t) for p, c, t in zip(problems, completions, flags, strict=True)]
