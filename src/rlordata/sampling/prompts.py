"""Prompt template (SPEC §5). Locked text; agent implements the instruct-model wrapping only."""

from __future__ import annotations

from typing import Literal

from rlordata.types import Problem

# locked: do not edit the template text without a SPEC changelog entry
TEMPLATE = (
    "Solve the following problem. Think step by step, then give the final answer on the last line "
    'in the form "Answer: <integer>".\n\nProblem: {problem_text}\n'
)


def format_prompt(problem: Problem, model_kind: Literal["base", "instruct"], tokenizer=None) -> str:  # noqa: ANN001
    """Base models get TEMPLATE verbatim. Instruct models get TEMPLATE inside their chat template
    with thinking disabled (agent implements per-model kwargs; tasks/02)."""
    if model_kind == "base":
        return TEMPLATE.format(problem_text=problem.text)
    raise NotImplementedError("tasks/02: instruct wrapping with thinking disabled")
