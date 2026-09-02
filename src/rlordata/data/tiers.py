"""Difficulty tiering by the base model's own pass@8 and split construction (SPEC §6). AGENT-OWNED; tasks/01.

Contract:
    tier_from_pass8(pass8: int) -> Tier            # easy >=6, medium 2-5, hard <=1
    build_splits(pool_with_pass8, seed) -> dict[str, list[Problem]]
        keys: train_easy_100, train_mixed_100, val_mixed_100, test_300, (ood_hard_200 built separately)
    Disjointness is enforced by problem_id AND by pipeline structure (pipeline with range removed).
"""

from __future__ import annotations
