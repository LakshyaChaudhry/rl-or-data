"""tasks/03 end to end on a tiny world (stub sampler, tiny model, CPU). Never a result."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rlordata.train.rft_pipeline_dry import core_implemented, run_dry_pipeline, tokenizer_available

os.environ.setdefault("HF_HUB_OFFLINE", "1")
pytest.importorskip("torch")
pytest.importorskip("peft")


@pytest.mark.slow
@pytest.mark.skipif(
    not core_implemented(), reason="core.sft_loss / core.rft_select not implemented yet (Laksh)"
)
def test_rft_pipeline_dry(tmp_path: Path) -> None:
    if not tokenizer_available():
        pytest.skip("Qwen tokenizer not cached")
    assert run_dry_pipeline(tmp_path / "w", verbose=False) == 0
