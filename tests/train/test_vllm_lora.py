"""NativeLoraSync / vllm_llm_with_lora against fakes: no GPU, no vLLM engine (tasks/04; notebook 2026-09-14).

The GPU-side check (vLLM-native-LoRA log-probs ≈ fp32 base+adapter) lives in the diagnostics under
runs/grpo/_diag/; these tests pin the plumbing: LoRA enabled at construction, a fresh LoRARequest id
per sync, the request passed to generate, stale adapter versions deleted, refusals when misused.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlordata.train import vllm_lora as vl

vllm = pytest.importorskip("vllm")


class _FakeLLM:
    def __init__(self, *, lora: bool = True) -> None:
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(lora_config=object() if lora else None)
        )
        self.calls: list[dict] = []
        self.resets = 0

    def generate(self, prompts, sampling_params=None, **kwargs):
        self.calls.append({"prompts": prompts, **kwargs})
        return ["ok"]

    def reset_prefix_cache(self) -> None:
        self.resets += 1


class _FakeModel:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.safetensors").write_bytes(b"x")
        self.saved.append(p)


def _fake_trainer(*, mode: str = "colocate", sleep: bool = False, lora: bool = True):
    vg = SimpleNamespace(mode=mode, enable_sleep_mode=sleep, llm=_FakeLLM(lora=lora))
    vg.sync_weights = lambda: pytest.fail("TRL's merge-based sync must never run")
    return SimpleNamespace(vllm_generation=vg, model=_FakeModel())


def test_vllm_llm_with_lora_patches_trl_module_attr_and_restores() -> None:
    trl_vg = pytest.importorskip("trl.generation.vllm_generation")
    orig = trl_vg.LLM
    with vl.vllm_llm_with_lora():
        assert isinstance(trl_vg.LLM, functools.partial)
        assert trl_vg.LLM.func is orig
        kw = trl_vg.LLM.keywords
        assert kw["enable_lora"] is True
        assert kw["max_lora_rank"] == 64  # SPEC §9 r=64 (vllm_sampler.LORA_MAX_RANK)
        assert kw["max_loras"] == 1
    assert trl_vg.LLM is orig


def test_sync_saves_adapter_and_hands_vllm_a_fresh_request(tmp_path: Path) -> None:
    t = _fake_trainer()
    sync = vl.NativeLoraSync(t, tmp_path / "lora", keep=2)
    vg = t.vllm_generation
    assert vg.sync_weights == sync.sync_weights  # TRL's merge path replaced

    vg.sync_weights()
    vg.sync_weights()
    assert [p.name for p in t.model.saved] == ["v1", "v2"]
    assert sync.request.lora_int_id == 2 and sync.request.lora_name == "policy-v2"
    assert sync.request.lora_path == str(tmp_path / "lora" / "v2")
    assert vg.llm.resets == 2

    out = vg.llm.generate([{"prompt_token_ids": [1, 2]}], sampling_params=None, use_tqdm=False)
    assert out == ["ok"]
    assert vg.llm.calls[-1]["lora_request"] is sync.request

    vg.sync_weights()  # v3: v1 is stale (keep=2)
    assert not (tmp_path / "lora" / "v1").exists()
    assert (tmp_path / "lora" / "v2").exists() and (tmp_path / "lora" / "v3").exists()
    assert sync.describe()["syncs"] == 3
    sync.close()
    assert not (tmp_path / "lora").exists()


def test_generate_before_first_sync_is_refused(tmp_path: Path) -> None:
    t = _fake_trainer()
    vl.NativeLoraSync(t, tmp_path / "lora")
    with pytest.raises(RuntimeError, match="before the first adapter sync"):
        t.vllm_generation.llm.generate([])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lora": False}, "without enable_lora"),
        ({"mode": "server"}, "colocate"),
        ({"sleep": True}, "sleep_mode"),
    ],
)
def test_refuses_unsupported_configurations(tmp_path: Path, kwargs: dict, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        vl.NativeLoraSync(_fake_trainer(**kwargs), tmp_path / "lora")


def test_mismatch_logging_trainer_is_a_grpo_trainer() -> None:
    trl = pytest.importorskip("trl")
    cls = vl.mismatch_logging_trainer_cls()
    assert issubclass(cls, trl.GRPOTrainer)
