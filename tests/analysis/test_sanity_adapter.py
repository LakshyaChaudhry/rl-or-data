"""tasks/03 §5 sanity: adapter non-trivial, final checkpoint, outputs differ."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlordata.analysis.sanity import (
    check_adapter_nontrivial,
    check_final_checkpoint,
    check_outputs_differ,
)
from rlordata.types import Sample

torch = pytest.importorskip("torch")


def _adapter(path: Path, *, zero_b: bool) -> Path:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 64}))
    b = torch.zeros(8, 4) if zero_b else torch.randn(8, 4)
    save_file(
        {
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.randn(4, 8),
            "base_model.model.layers.0.q_proj.lora_B.weight": b,
        },
        str(path / "adapter_model.safetensors"),
    )
    return path


def test_adapter_nontrivial(tmp_path: Path) -> None:
    assert check_adapter_nontrivial(_adapter(tmp_path / "ok", zero_b=False)) == []
    issues = check_adapter_nontrivial(_adapter(tmp_path / "zero", zero_b=True))
    assert issues and "all-zero" in issues[0]
    assert check_adapter_nontrivial(tmp_path / "missing")


def test_final_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    e1, e2 = run / "adapter" / "epoch_1", run / "adapter" / "epoch_2"
    final = run / "adapter" / "final"
    for d in (e1, e2, final):
        _adapter(d, zero_b=False)
    (final / "SOURCE.txt").write_text("copy of epoch_2\n")
    (run / "meta.json").write_text(json.dumps({"status": "finished"}))
    (run / "budgets.json").write_text(
        json.dumps({"final_adapter": str(final), "epoch_adapters": [str(e1), str(e2)], "epochs": 2})
    )
    assert check_final_checkpoint(run) == []
    (run / "budgets.json").write_text(
        json.dumps({"final_adapter": str(final), "epoch_adapters": [str(e1)], "epochs": 2})
    )
    assert any("epoch adapters" in i for i in check_final_checkpoint(run))
    (final / "SOURCE.txt").write_text("copy of epoch_1\n")
    (run / "budgets.json").write_text(
        json.dumps({"final_adapter": str(final), "epoch_adapters": [str(e1), str(e2)], "epochs": 2})
    )
    assert any("last epoch" in i for i in check_final_checkpoint(run))
    assert check_final_checkpoint(tmp_path / "nothing")


def _s(pid: str, completion: str) -> Sample:
    return Sample(
        run_id="r",
        config_hash="h",
        seed=1,
        arm="a",
        data_condition="d",
        problem_id=pid,
        tier="easy",
        prompt="p",
        completion=completion,
        extracted_answer=None,
        correct=False,
        reward=0.0,
        n_tokens=1,
        truncated=False,
        extraction_failed=True,
    )


def test_outputs_differ() -> None:
    base = [_s(f"p{i}", f"c{i}") for i in range(10)]
    same = [_s(f"p{i}", f"c{i}") for i in range(10)]
    assert check_outputs_differ(base, same)
    changed = [_s(f"p{i}", f"c{i}" if i else "different") for i in range(10)]
    assert check_outputs_differ(base, changed, min_fraction=0.10) == []
    assert check_outputs_differ(base, changed, min_fraction=0.20)
    assert check_outputs_differ(base, [_s("zzz", "x")])
