"""tasks/03 §4: the sweep never touches the held-out splits — grep the code, fail on their names."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN = ("test_300", "ood_hard_200", "ood_hard", "test_3")
SWEEP_FILES = ("scripts/rft_sweep.py",)
# tasks/04: nothing in the GRPO training path may name the held-out splits either; only the shared
# eval module (train/rft_eval.py) touches them, once, at the final checkpoint.
GRPO_TRAIN_FILES = (
    "src/rlordata/train/grpo_trl.py",
    "src/rlordata/train/rewards.py",
    "src/rlordata/train/callbacks.py",
    "scripts/run_queue.py",
)


def test_sweep_code_never_names_heldout_splits() -> None:
    for rel in SWEEP_FILES:
        text = Path(rel).read_text(encoding="utf-8")
        for name in FORBIDDEN:
            assert name not in text, (
                f"{rel} mentions {name!r}: the sweep may only read val_mixed_100"
            )
        assert "val_mixed_100" in text


def test_grpo_training_code_never_names_heldout_splits() -> None:
    for rel in GRPO_TRAIN_FILES:
        text = Path(rel).read_text(encoding="utf-8")
        for name in FORBIDDEN:
            assert name not in text, (
                f"{rel} mentions {name!r}: GRPO training never sees held-out splits"
            )
