"""rft_select.py — rejection sampling + curation. Hand-written (Laksh).

INTUITION
    Rejection-sampling fine-tuning (RFT / STaR / ReST): let the model try many times, keep the
    tries the verifier accepts, and imitate those. It is the simplest thing that "learns from
    its own samples." Arm 1 (RFT-all) keeps every correct completion. Arm 2 (RFT-curated)
    additionally drops prompts that are trivially easy or hopeless for the base model — the same
    prompts GRPO silently ignores, because a group with identical rewards has zero advantage.

PRECISE (SPEC §8)
    Input: for each problem, K sampled completions with binary rewards.
    RFT-all:      keep (problem, completion) pairs with reward == 1.
    RFT-curated:  let pass8 = number correct among the FIRST 8 samples (fixed order, seed-controlled);
                  keep a problem only if 1 <= pass8 <= 7; then keep its correct completions.
    Dedup exact-duplicate completions per problem. Optionally cap correct completions per problem
    at `max_per_problem` (config; default None) to avoid easy prompts dominating the SFT set —
    if you cap, cap identically for both arms and record it.

TESTS YOU WRITE (tests/core/test_rft_select.py)
    - problem with 0/8 correct -> dropped by curated, and contributes nothing to all (no positives)
    - problem with 8/8 correct -> kept by all, dropped by curated
    - problem with 3/8 correct -> both keep exactly its 3 correct completions (after dedup)
    - the "first 8" rule uses sample order, not reward order
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rlordata.types import Problem, Sample


@dataclass(frozen=True)
class SFTExample:
    problem_id: str
    prompt: str
    completion: str
    tier: str


def rft_select(
    problems: list[Problem],
    samples: dict[str, list[Sample]],  # problem_id -> K samples in generation order
    mode: Literal["all", "curated"],
    max_per_problem: int | None = None,
) -> list[SFTExample]:
    """Build the SFT dataset for Arm 1 ("all") or Arm 2 ("curated")."""
    raise NotImplementedError("Laksh: implement rft_select")
