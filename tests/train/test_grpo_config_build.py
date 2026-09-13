"""Build the real TRL GRPOConfig for every GRPO arm and control, without a GPU (tasks/04 §2).

Skipped where TRL is not installed (the Mac); runs under `make test` on the GPU box, so a keyword the
installed TRL/transformers pair rejects fails here instead of at the first training job. On a Mac,
run it against the locked TRL with:  uv run --with trl==1.13.0 pytest tests/train/test_grpo_config_build.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

trl = pytest.importorskip("trl")

from rlordata.train.common import load_arm_config  # noqa: E402
from rlordata.train.grpo_trl import MAX_PROMPT_LENGTH, build_grpo_config  # noqa: E402

CONFIGS = [
    "configs/grpo/mixed100.yaml",
    "configs/grpo/curated.yaml",
    "configs/grpo/easy100.yaml",
    "configs/controls/random_reward.yaml",
    "configs/controls/format_only.yaml",
]


@pytest.fixture(autouse=True)
def _no_cuda_bf16(monkeypatch):
    """transformers refuses bf16 on a machine without a bf16-capable device; the values are what we test."""
    import transformers.training_args as ta

    if not getattr(ta, "is_torch_bf16_gpu_available", lambda: True)():
        monkeypatch.setattr(ta, "is_torch_bf16_gpu_available", lambda: True, raising=False)


@pytest.mark.parametrize("path", CONFIGS)
def test_grpo_config_builds_with_pinned_values(path: str, tmp_path: Path) -> None:
    cfg = load_arm_config(path)
    try:
        c = build_grpo_config(cfg, run_dir=tmp_path, seed=1)
    except ValueError as e:  # only a missing bf16 device is tolerated off-GPU
        if "bf16" not in str(e).lower():
            raise
        pytest.skip(f"bf16 unavailable on this machine: {e}")
    g = cfg["training"]["grpo"]
    assert c.max_steps == 300 and c.num_generations == 8 and c.generation_batch_size == 64
    assert c.per_device_train_batch_size * c.gradient_accumulation_steps == 64
    # 4 × 16 (8 × 8 OOMed on an 80 GB H100). One optimizer step per generation batch, and the dapo
    # normalizer ratio grad_accum / steps_per_generation stays 1, so the update equals 8 × 8's.
    assert c.per_device_train_batch_size == 4 and c.gradient_accumulation_steps == 16
    assert c.steps_per_generation == c.gradient_accumulation_steps
    assert c.max_steps * c.generation_batch_size == g["total_sampled_completions"] == 19200
    assert c.learning_rate == 5e-5 and c.lr_scheduler_type.value == "cosine"
    assert c.get_warmup_steps(c.max_steps) == 30  # 10 % of 300, same ceil rule as the RFT trainer
    assert c.max_grad_norm == 1.0 and c.weight_decay == 0.0
    assert c.temperature == 1.0 and c.top_p == 1.0
    assert c.max_completion_length == cfg["max_completion_tokens"] == 4352
    assert c.vllm_max_model_length == MAX_PROMPT_LENGTH + 4352
    assert c.beta == 0.0 and c.epsilon == 0.2 and c.epsilon_high in (None, 0.2)
    assert c.scale_rewards == "group" and c.loss_type == "dapo" and c.num_iterations == 1
    assert c.importance_sampling_level == "token" and c.delta is None
    assert c.top_entropy_quantile == 1.0 and c.mask_truncated_completions is False
    assert c.vllm_importance_sampling_correction is False
    assert c.use_vllm is True and c.vllm_mode == "colocate"
    assert c.save_steps == 25 and c.seed == 1
    assert c.gradient_checkpointing is True
    assert not getattr(c, "use_liger_kernel", False)
