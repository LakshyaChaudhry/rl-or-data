"""Shared record types. These are the contracts between components (SPEC §13).

Agent may add optional fields; may not rename or remove existing ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Tier = Literal["easy", "medium", "hard", "untiered"]


@dataclass(frozen=True)
class Problem:
    """One procedurally generated counting problem (SPEC §4)."""

    problem_id: str  # sha256 of canonical pipeline JSON
    text: str  # natural-language rendering
    answer: int  # ground truth, computed by executing the pipeline
    pipeline: dict[str, Any]  # canonical structure: range, filters, transforms, op
    range_scale: Literal["S", "M", "L"]
    n_filters: int
    n_transforms: int
    total_steps: int  # n_filters + n_transforms + 1
    tier: Tier = "untiered"
    pass8: int | None = None  # base-model correct count out of 8 (set by tiering)
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sample:
    """One model completion for one problem (eval or training rollout)."""

    run_id: str
    config_hash: str
    seed: int
    arm: str
    data_condition: str
    problem_id: str
    tier: Tier
    prompt: str
    completion: str
    extracted_answer: int | None
    correct: bool
    reward: float
    n_tokens: int
    truncated: bool
    extraction_failed: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
