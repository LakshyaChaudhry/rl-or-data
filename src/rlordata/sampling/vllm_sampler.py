"""The single generation path for every number in the paper (SPEC §10, CLAUDE.md). AGENT-OWNED; tasks/02.

Contract:
    class VLLMSampler:
        __init__(model_id, model_kind, lora_path=None, max_completion_tokens=<from configs/locked/cap.yaml>,
                 seed=..., dtype="bfloat16", gpu_memory_utilization=0.9, allow_provisional_cap=False)
        sample(prompts: list[str], n: int, temperature: float, top_p: float = 1.0) -> list[list[Completion]]
    Completion has: text, n_tokens, truncated (finish_reason == "length").
    Prompt formatting uses rlordata.sampling.prompts.format_prompt(problem, model_kind)   # base vs instruct
    Refuses to construct if the cap passed differs from configs/locked/cap.yaml once that file exists.

Implementation notes (tasks/02a):
- ``vllm`` is imported lazily inside ``__init__`` so this module imports on a Mac; the cap check
  runs *before* the import so the refusal is testable without a GPU.
- One ``LLM.generate`` call for all prompts × n. Per-prompt ``SamplingParams`` carry
  ``seed = base_seed + prompt_index * n`` so every (prompt, sample) pair has its own seed and the
  result is deterministic given ``seed`` and prompt order.
- No repetition penalty, ``top_p`` as given (SPEC §7). Greedy is ``temperature=0``.
- Max prompt tokens 1024 (SPEC §7): prompts longer than that raise instead of being truncated.
- Thinking mode is never enabled here; instruct wrapping (thinking off) lives in ``prompts.py``.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rlordata.sampling.cap import DEFAULT_CAP_PATH, resolve_cap

MAX_PROMPT_TOKENS = 1024  # SPEC §7
LORA_MAX_RANK = 64  # SPEC §9 (r=64)
ModelKind = Literal["base", "instruct"]


@dataclass(frozen=True)
class Completion:
    """One sampled completion. ``n_tokens`` counts generated tokens only."""

    text: str
    n_tokens: int
    truncated: bool  # finish_reason == "length"
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMSampler:
    """vLLM offline sampler; see module docstring for the contract."""

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        model_kind: ModelKind,
        lora_path: str | Path | None = None,
        max_completion_tokens: int | None = None,
        seed: int = 0,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        allow_provisional_cap: bool = False,
        *,
        cap_path: str | Path = DEFAULT_CAP_PATH,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if model_kind not in ("base", "instruct"):
            raise ValueError(f"model_kind must be 'base' or 'instruct', got {model_kind!r}")
        # Cap rule first: raises CapError before any GPU work or vllm import.
        self.max_completion_tokens = resolve_cap(
            max_completion_tokens, cap_path=cap_path, allow_provisional=allow_provisional_cap
        )
        self.cap_is_provisional = not Path(cap_path).exists()
        self.model_id = model_id
        self.model_kind: ModelKind = model_kind
        self.lora_path = str(lora_path) if lora_path is not None else None
        self.seed = int(seed)
        self.dtype = dtype
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.llm_kwargs = dict(llm_kwargs or {})

        from vllm import LLM, SamplingParams  # lazy: GPU box only

        self._SamplingParams = SamplingParams
        self._lora_request = None
        kwargs: dict[str, Any] = {
            "model": model_id,
            "dtype": dtype,
            "seed": self.seed,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_prompt_tokens + self.max_completion_tokens,
            "trust_remote_code": False,
        }
        if self.lora_path is not None:
            from vllm.lora.request import LoRARequest

            kwargs.update({"enable_lora": True, "max_lora_rank": LORA_MAX_RANK})
            self._lora_request = LoRARequest("adapter", 1, self.lora_path)
        kwargs.update(self.llm_kwargs)
        self._llm = LLM(**kwargs)
        self.tokenizer = self._llm.get_tokenizer()
        try:
            import vllm

            self.vllm_version = getattr(vllm, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            self.vllm_version = "unknown"

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Everything about the sampler that belongs in a resolved run config."""
        return {
            "sampler": self.name,
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "lora_path": self.lora_path,
            "max_completion_tokens": self.max_completion_tokens,
            "cap_is_provisional": self.cap_is_provisional,
            "max_prompt_tokens": self.max_prompt_tokens,
            "seed": self.seed,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "repetition_penalty": 1.0,
            "seed_scheme": "per_prompt: seed + prompt_index * n; vLLM child seeds add sample index",
            "vllm_version": self.vllm_version,
            "llm_kwargs": self.llm_kwargs,
        }

    def prompt_token_counts(self, prompts: list[str]) -> list[int]:
        return [len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    def sample(
        self, prompts: list[str], n: int, temperature: float, top_p: float = 1.0
    ) -> list[list[Completion]]:
        """All prompts × n in one vLLM call. Returns ``[len(prompts)][n]`` completions."""
        assert isinstance(prompts, list) and len(prompts) > 0, "prompts must be a non-empty list"
        assert all(isinstance(p, str) and p for p in prompts), "prompts must be non-empty strings"
        assert n >= 1, "n must be >= 1"
        assert temperature >= 0.0 and 0.0 < top_p <= 1.0
        too_long = [
            i for i, c in enumerate(self.prompt_token_counts(prompts)) if c > self.max_prompt_tokens
        ]
        if too_long:
            raise ValueError(
                f"{len(too_long)} prompt(s) exceed max_prompt_tokens={self.max_prompt_tokens} "
                f"(first index {too_long[0]}); SPEC §7 forbids silently truncating prompts"
            )
        params = [
            self._SamplingParams(
                n=n,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=self.max_completion_tokens,
                seed=self.seed + i * n,
                repetition_penalty=1.0,
            )
            for i in range(len(prompts))
        ]
        gen_kwargs: dict[str, Any] = {"use_tqdm": True}
        if self._lora_request is not None:
            gen_kwargs["lora_request"] = self._lora_request
        outputs = self._llm.generate(prompts, params, **gen_kwargs)
        results: list[list[Completion]] = []
        for req in outputs:
            comps = [
                Completion(
                    text=o.text,
                    n_tokens=len(o.token_ids),
                    truncated=(o.finish_reason == "length"),
                    finish_reason=str(o.finish_reason),
                )
                for o in req.outputs
            ]
            assert len(comps) == n, f"vLLM returned {len(comps)} completions, expected {n}"
            results.append(comps)
        assert len(results) == len(prompts)
        return results

    def close(self) -> None:
        """Release the engine so the next model fits on the GPU."""
        llm = getattr(self, "_llm", None)
        if llm is not None:
            del self._llm
            del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
