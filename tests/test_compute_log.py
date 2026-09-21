"""tasks/07 compute log: every meta.json lands in exactly one column, classified by path only."""

from __future__ import annotations

import pytest

from scripts.compute_log import classify

CHOSEN = {"rft_mixed": "_lr5e-05_ep4", "iter_rft_curated": None}


@pytest.mark.parametrize(
    "rel, is_unit, want",
    [
        ("rft/rft_mixed/seed2_lr5e-05_ep4/meta.json", False, ("rft_mixed", "train")),
        ("rft/rft_mixed/seed1_lr0.0001_ep8/meta.json", False, ("rft_mixed", "sweep")),
        ("rft/rft_mixed/seed1_lr5e-05_ep4/eval/final/test_300/greedy/meta.json", True, ("rft_mixed", "eval_final")),
        ("rft/rft_mixed/seed1_lr1e-05_ep2/eval/val/val_mixed_100/greedy/meta.json", True, ("rft_mixed", "eval_selection")),
        ("rft/rft_mixed/seed1_lr1e-05_ep2/eval/val_bf16merge_stale/val_mixed_100/greedy/meta.json", True, ("rft_mixed", "eval_superseded")),
        ("rft/iter_rft_curated/seed1/meta.json", False, ("iter_rft_curated", "train")),
        ("rft/iter_rft_curated/seed1/round_2/meta.json", False, None),  # summed in the run-level file
        ("rft/iter_rft_curated/seed1/round_2/eval/val/val_mixed_100/greedy/meta.json", True, ("iter_rft_curated", "eval_selection")),
        ("rft/draw_seed1/meta.json", False, ("rft/draw_seed1", "other")),
        ("grpo/grpo_curated_s3/meta.json", False, ("grpo_curated", "train")),
        ("grpo/grpo_random_reward_s1/eval_checkpoints/step_100/val_mixed_100/greedy/meta.json", True, ("grpo_random_reward", "eval_selection")),
        ("grpo/_failed/grpo_mixed_s1_bf16merge_x/meta.json", False, ("grpo/_failed", "other")),
        ("exploratory_cap8704/base/test_300/greedy/meta.json", True, ("exploratory_cap8704", "other")),
        ("_stub_dryrun_20260913/x/meta.json", False, ("dev", "other")),
    ],
)  # fmt: skip
def test_classify(rel: str, is_unit: bool, want: tuple[str, str] | None) -> None:
    assert classify(rel, is_unit, CHOSEN) == want
