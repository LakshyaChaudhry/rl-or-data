"""tasks/03 §3: the SFT loop on a tiny model. Runs once core.sft_loss and core.rft_select land."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rlordata.core.rft_select import SFTExample
from rlordata.sampling.prompts import render_template
from rlordata.train.rft import build_lora_model, train_rft
from rlordata.train.rft_pipeline_dry import core_implemented, tokenizer_available, write_tiny_model

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("peft")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    if not tokenizer_available():
        pytest.skip("Qwen tokenizer not cached")
    return write_tiny_model(tmp_path_factory.mktemp("tiny") / "model")


def _examples(n: int = 6) -> list[SFTExample]:
    out = []
    for i in range(n):
        text = f"Consider the integers from 1 to {20 + i}, inclusive. First, keep only the even numbers. Of these numbers, count how many values remain."
        out.append(
            SFTExample(
                f"p{i}",
                render_template(text),
                f"Even numbers: 2, 4, ...\nAnswer: {(20 + i) // 2}",
                "easy",
            )
        )
    return out


def _train(tiny_model_dir: Path, run_dir: Path, seed: int, epochs: int = 2):
    from rlordata.train.common import load_yaml

    tr = load_yaml("configs/locked/training.yaml")
    model, tok = build_lora_model(
        str(tiny_model_dir),
        tr,
        seed=seed,
        dtype="float32",
        device="cpu",
        gradient_checkpointing=False,
    )
    return train_rft(
        model,
        tok,
        _examples(),
        run_dir=run_dir,
        seed=seed,
        learning_rate=1e-4,
        epochs=epochs,
        batch_size=4,
        micro_batch_size=2,
        grad_clip=1.0,
        warmup_ratio=0.1,
        weight_decay=0.0,
        log_every=1,
    ), model


@pytest.mark.skipif(
    not core_implemented(), reason="core.sft_loss / core.rft_select not implemented yet (Laksh)"
)
def test_train_loop_writes_log_adapters_and_budgets(tiny_model_dir: Path, tmp_path: Path) -> None:
    result, model = _train(tiny_model_dir, tmp_path / "run", seed=1)
    assert result.optimizer_steps == 2 * 2  # ceil(6/4)=2 steps per epoch × 2 epochs
    assert result.training_tokens > 0 and result.total_tokens > result.training_tokens
    log = [
        json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in log] == [1, 2, 3, 4]
    assert all({"loss", "lr", "tokens_seen", "wall_clock_s", "epoch"} <= set(r) for r in log)
    assert log[-1]["tokens_seen"] == result.training_tokens
    assert [p.name for p in result.epoch_adapter_dirs] == ["epoch_1", "epoch_2"]
    assert (result.final_adapter_dir / "adapter_model.safetensors").exists()
    assert "epoch_2" in (result.final_adapter_dir / "SOURCE.txt").read_text()
    from rlordata.analysis.sanity import check_adapter_nontrivial, check_final_checkpoint

    assert check_adapter_nontrivial(result.final_adapter_dir) == []
    from rlordata.train.common import write_json

    write_json(
        tmp_path / "run" / "budgets.json",
        {
            "final_adapter": str(result.final_adapter_dir),
            "epoch_adapters": [str(p) for p in result.epoch_adapter_dirs],
            "epochs": 2,
        },
    )
    assert check_final_checkpoint(tmp_path / "run") == []


@pytest.mark.skipif(
    not core_implemented(), reason="core.sft_loss / core.rft_select not implemented yet (Laksh)"
)
def test_training_is_seed_deterministic(tiny_model_dir: Path, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    a, _ = _train(tiny_model_dir, tmp_path / "a", seed=1, epochs=1)
    b, _ = _train(tiny_model_dir, tmp_path / "b", seed=1, epochs=1)
    c, _ = _train(tiny_model_dir, tmp_path / "c", seed=2, epochs=1)
    wa = load_file(str(a.final_adapter_dir / "adapter_model.safetensors"))
    wb = load_file(str(b.final_adapter_dir / "adapter_model.safetensors"))
    wc = load_file(str(c.final_adapter_dir / "adapter_model.safetensors"))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert any(not torch.equal(wa[k], wc[k]) for k in wa if "lora_A" in k)


@pytest.mark.skipif(
    not core_implemented(), reason="core.sft_loss / core.rft_select not implemented yet (Laksh)"
)
def test_loss_is_token_mean_over_the_full_batch(tiny_model_dir: Path, tmp_path: Path) -> None:
    """micro_batch 2 × accumulation 2 must match a single 4-example step in the logged loss."""
    from rlordata.train.common import load_yaml

    tr = load_yaml("configs/locked/training.yaml")
    losses = []
    for micro in (4, 2, 1):
        model, tok = build_lora_model(
            str(tiny_model_dir),
            tr,
            seed=1,
            dtype="float32",
            device="cpu",
            gradient_checkpointing=False,
        )
        train_rft(
            model,
            tok,
            _examples(4),
            run_dir=tmp_path / f"m{micro}",
            seed=1,
            learning_rate=1e-4,
            epochs=1,
            batch_size=4,
            micro_batch_size=micro,
            grad_clip=1.0,
            warmup_ratio=0.0,
            weight_decay=0.0,
            log_every=1,
        )
        losses.append(
            json.loads((tmp_path / f"m{micro}" / "train_log.jsonl").read_text().splitlines()[0])[
                "loss"
            ]
        )
    assert abs(losses[0] - losses[1]) < 1e-4 and abs(losses[0] - losses[2]) < 1e-4
