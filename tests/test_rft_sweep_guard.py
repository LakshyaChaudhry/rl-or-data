"""tasks/03 §4: the sweep never touches the held-out splits — grep the code, fail on their names."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN = ("test_300", "ood_hard_200", "ood_hard", "test_3")
SWEEP_FILES = ("scripts/rft_sweep.py",)


def test_sweep_code_never_names_heldout_splits() -> None:
    for rel in SWEEP_FILES:
        text = Path(rel).read_text(encoding="utf-8")
        for name in FORBIDDEN:
            assert name not in text, (
                f"{rel} mentions {name!r}: the sweep may only read val_mixed_100"
            )
        assert "val_mixed_100" in text
