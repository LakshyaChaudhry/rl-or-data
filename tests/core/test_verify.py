"""Test CASES authored by Laksh. Agent may add fixtures, not expected values."""

import pytest

from rlordata.core import verify as vf
from tests.conftest import skip_unless_implemented

pytestmark = pytest.mark.core


def test_exact_last_line(toy_problem):
    skip_unless_implemented(vf.extract_answer, "Answer: 16")
    v = vf.verify(toy_problem, "Some reasoning...\nAnswer: 16")
    assert v.reward == 1.0 and v.extracted == 16 and not v.extraction_failed


def test_trailing_period_accepted_v15(toy_problem):
    # SPEC v1.5 (2026-09-08): a single trailing '.' or ',' after the integer is allowed.
    # (v1.4 expected this to fail; expected value changed under the amendment, authorized by Laksh.)
    skip_unless_implemented(vf.extract_answer, "Answer: 16")
    v = vf.verify(toy_problem, "Answer: 16.")
    assert v.reward == 1.0 and v.extracted == 16 and not v.extraction_failed


def test_last_answer_line_wins(toy_problem):
    skip_unless_implemented(vf.extract_answer, "Answer: 16")
    v = vf.verify(toy_problem, "Answer: 12\nActually wait.\nAnswer: 16")
    assert v.reward == 1.0 and v.extracted == 16


# TODO(Laksh): negative numbers, leading zeros, whitespace, answer-not-on-last-line, empty completion.


def test_wrong_but_parseable(toy_problem):
    v = vf.verify(toy_problem, "Answer: 99")
    assert v.reward == 0.0 and v.extracted == 99 and not v.extraction_failed


def test_empty_completion(toy_problem):
    v = vf.verify(toy_problem, "")
    assert v.reward == 0.0 and v.extraction_failed
