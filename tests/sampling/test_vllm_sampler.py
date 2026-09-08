from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rlordata.sampling.cap import PROVISIONAL_CAP, CapError, write_cap_yaml
from rlordata.sampling.vllm_sampler import MAX_PROMPT_TOKENS, Completion, VLLMSampler


def test_module_imports_without_vllm() -> None:
    import rlordata.sampling.vllm_sampler as mod

    assert hasattr(mod, "VLLMSampler")
    assert MAX_PROMPT_TOKENS == 1024
    c = Completion(text="Answer: 1", n_tokens=3, truncated=False, finish_reason="stop")
    assert c.to_dict()["truncated"] is False


def test_constructor_refuses_wrong_cap_before_touching_vllm(tmp_path: Path) -> None:
    cap_path = tmp_path / "cap.yaml"
    write_cap_yaml(cap_path, {"max_completion_tokens": 1024})
    with pytest.raises(CapError):
        VLLMSampler(
            "Qwen/Qwen3-4B-Base", "base", max_completion_tokens=2048, seed=1, cap_path=cap_path
        )
    # No cap file and no provisional permission: refused too.
    with pytest.raises(CapError):
        VLLMSampler(
            "Qwen/Qwen3-4B-Base",
            "base",
            max_completion_tokens=PROVISIONAL_CAP,
            seed=1,
            cap_path=tmp_path / "none.yaml",
        )
    with pytest.raises(ValueError):
        VLLMSampler(
            "Qwen/Qwen3-4B-Base", "chat", max_completion_tokens=1024, seed=1, cap_path=cap_path
        )  # type: ignore[arg-type]


def test_constructor_with_valid_cap_needs_vllm(tmp_path: Path) -> None:
    """With a valid cap the constructor proceeds to the lazy vllm import (ImportError on a Mac)."""
    if "vllm" in sys.modules or _has("vllm"):
        pytest.skip("vllm installed; the gpu-marked test covers construction")
    cap_path = tmp_path / "cap.yaml"
    write_cap_yaml(cap_path, {"max_completion_tokens": 1024})
    with pytest.raises(ImportError):
        VLLMSampler(
            "Qwen/Qwen3-4B-Base", "base", max_completion_tokens=1024, seed=1, cap_path=cap_path
        )


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


@pytest.mark.gpu
def test_real_sampler_is_deterministic_given_seed(tmp_path: Path) -> None:
    """GPU only: Qwen3-0.6B-Base (SPEC §3 smoke-test model), provisional cap into tmp_path."""
    pytest.importorskip("vllm")
    from rlordata.sampling.prompts import TEMPLATE

    prompts = [
        TEMPLATE.format(
            problem_text="Consider the integers from 1 to 10, inclusive. Count how many are even."
        ),
        TEMPLATE.format(
            problem_text="Consider the integers from 1 to 20, inclusive. Count how many are odd."
        ),
    ]
    s = VLLMSampler(
        "Qwen/Qwen3-0.6B-Base",
        "base",
        max_completion_tokens=256,
        seed=7,
        allow_provisional_cap=True,
        cap_path=tmp_path / "cap.yaml",
        gpu_memory_utilization=0.5,
    )
    try:
        a = s.sample(prompts, n=2, temperature=1.0)
        b = s.sample(prompts, n=2, temperature=1.0)
        assert [[c.text for c in row] for row in a] == [[c.text for c in row] for row in b]
        assert all(len(row) == 2 for row in a)
        assert all(c.n_tokens <= 256 for row in a for c in row)
        assert all(c.truncated == (c.finish_reason == "length") for row in a for c in row)
        g = s.sample(prompts, n=1, temperature=0.0)
        assert len(g) == 2 and len(g[0]) == 1
    finally:
        s.close()
