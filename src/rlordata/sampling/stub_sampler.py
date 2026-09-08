"""Scripted stand-in for :class:`VLLMSampler` for tests and dry runs. AGENT-OWNED; tasks/02a.

Same constructor and ``sample`` signature as the real sampler, same cap rule, no GPU. Completions
are scripted from the problem's true answer (looked up by problem text inside the prompt), so
``core.verify`` produces a controllable mix of correct / incorrect / unparseable / truncated
outcomes. Every draw is seeded from ``(seed, prompt, sample_index)`` so results are deterministic
and independent of batch composition or order. Never used for a result-bearing number.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rlordata.sampling.cap import DEFAULT_CAP_PATH, PROVISIONAL_CAP, resolve_cap
from rlordata.sampling.vllm_sampler import MAX_PROMPT_TOKENS, Completion
from rlordata.types import Problem

_PROBLEM_LINE = re.compile(r"Problem: (.+)")


class FakeChatTokenizer:
    """Minimal ``apply_chat_template`` for instruct-model dry runs; records the kwargs it saw."""

    name_or_path = "fake-chat-tokenizer"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        assert tokenize is False, "prompts are rendered as text; vLLM tokenizes"
        self.calls.append(dict(kwargs))
        body = "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in messages)
        if add_generation_prompt:
            body += "<|assistant|>\n"
        return body

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        # Whitespace tokens are a fine proxy for a prompt-length guard in dry runs.
        return {"input_ids": [0] * len(text.split())}


def _u32(*parts: Any) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


class StubSampler:
    """Scripted sampler with the :class:`VLLMSampler` interface."""

    name = "stub"

    def __init__(
        self,
        model_id: str,
        model_kind: Literal["base", "instruct"],
        lora_path: str | Path | None = None,
        max_completion_tokens: int | None = None,
        seed: int = 0,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        allow_provisional_cap: bool = False,
        *,
        cap_path: str | Path = DEFAULT_CAP_PATH,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
        answers: dict[str, int] | None = None,
        p_correct: float = 0.5,
        p_truncated: float = 0.05,
        p_unparseable: float = 0.05,
        difficulty_spread: bool = True,
        mean_len: float = 180.0,
    ) -> None:
        if model_kind not in ("base", "instruct"):
            raise ValueError(f"model_kind must be 'base' or 'instruct', got {model_kind!r}")
        self.max_completion_tokens = resolve_cap(
            max_completion_tokens, cap_path=cap_path, allow_provisional=allow_provisional_cap
        )
        self.cap_is_provisional = not Path(cap_path).exists()
        self.model_id = model_id
        self.model_kind = model_kind
        self.lora_path = str(lora_path) if lora_path is not None else None
        self.seed = int(seed)
        self.dtype = dtype
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.answers: dict[str, int] = dict(answers or {})
        assert 0.0 <= p_correct <= 1.0 and 0.0 <= p_truncated <= 1.0 and 0.0 <= p_unparseable <= 1.0
        self.p_correct = float(p_correct)
        self.p_truncated = float(p_truncated)
        self.p_unparseable = float(p_unparseable)
        self.difficulty_spread = bool(difficulty_spread)
        self.mean_len = float(mean_len)
        self.tokenizer = FakeChatTokenizer()
        self.vllm_version = "stub"
        self.n_calls = 0

    @classmethod
    def from_problems(cls, problems: Iterable[Problem], **kwargs: Any) -> StubSampler:
        answers = {p.text: int(p.answer) for p in problems}
        return cls(answers=answers, **kwargs)

    def add_problems(self, problems: Iterable[Problem]) -> None:
        for p in problems:
            self.answers[p.text] = int(p.answer)

    def describe(self) -> dict[str, Any]:
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
            "seed_scheme": "stub: sha256(seed, prompt, sample_index)",
            "vllm_version": self.vllm_version,
            "stub_params": {
                "p_correct": self.p_correct,
                "p_truncated": self.p_truncated,
                "p_unparseable": self.p_unparseable,
                "difficulty_spread": self.difficulty_spread,
                "mean_len": self.mean_len,
            },
        }

    def prompt_token_counts(self, prompts: list[str]) -> list[int]:
        return [len(self.tokenizer(p)["input_ids"]) for p in prompts]

    # ------------------------------------------------------------------
    def _answer_for(self, prompt: str) -> int | None:
        m = _PROBLEM_LINE.search(prompt)
        if m is not None:
            ans = self.answers.get(m.group(1).strip())
            if ans is not None:
                return ans
        for text, ans in self.answers.items():
            if text in prompt:
                return ans
        return None

    def _problem_p_correct(self, prompt: str) -> float:
        if not self.difficulty_spread:
            return self.p_correct
        # Per-problem "difficulty": uniform in [0, 1], fixed by the problem text (not the seed),
        # so the same problem is easy/hard across seeds the way a real model would be.
        m = _PROBLEM_LINE.search(prompt)
        key = m.group(1).strip() if m else prompt
        return _u32("difficulty", key) / 2**32

    def _one(self, prompt: str, j: int, temperature: float, answer: int | None) -> Completion:
        rng = np.random.default_rng(_u32(self.seed, prompt, j, temperature))
        cap = self.max_completion_tokens
        u = rng.random()
        if u < self.p_truncated:
            text = "Let me work through this carefully. " * 8 + "The remaining values are"
            return Completion(text=text, n_tokens=cap, truncated=True, finish_reason="length")
        # Natural length is drawn independently of the cap; a completion longer than the cap is cut
        # off and loses its answer line, exactly like a real model (so sampling at cap A and
        # rescoring at a lower cap B agrees with sampling at B directly).
        length = int(max(8, rng.lognormal(np.log(self.mean_len), 0.45)))
        if length > cap:
            text = "Let me work through this carefully. " * 8 + "The remaining values are"
            return Completion(text=text, n_tokens=cap, truncated=True, finish_reason="length")
        if answer is None:
            wrong = 0
            body = f"Step 1: I could not identify the problem.\nAnswer: {wrong}"
            return Completion(text=body, n_tokens=length, truncated=False, finish_reason="stop")
        if u < self.p_truncated + self.p_unparseable:
            # No integer anywhere: under SPEC §5 v1.6 the fallback layer would otherwise read a
            # trailing number, so an unparseable stub completion must contain no digits at all.
            body = "First, list the numbers.\nThen apply the rule.\nI cannot determine it."
            return Completion(text=body, n_tokens=length, truncated=False, finish_reason="stop")
        p_ok = self._problem_p_correct(prompt)
        correct = rng.random() < p_ok
        value = answer if correct else answer + int(rng.integers(1, 4)) * int(rng.choice([-1, 1]))
        body = f"Step 1: list the numbers.\nStep 2: apply each rule.\nStep 3: compute.\nAnswer: {value}"
        return Completion(text=body, n_tokens=length, truncated=False, finish_reason="stop")

    def sample(
        self, prompts: list[str], n: int, temperature: float, top_p: float = 1.0
    ) -> list[list[Completion]]:
        assert isinstance(prompts, list) and len(prompts) > 0, "prompts must be a non-empty list"
        assert all(isinstance(p, str) and p for p in prompts)
        assert n >= 1 and temperature >= 0.0 and 0.0 < top_p <= 1.0
        too_long = [
            i for i, c in enumerate(self.prompt_token_counts(prompts)) if c > self.max_prompt_tokens
        ]
        if too_long:
            raise ValueError(
                f"{len(too_long)} prompt(s) exceed max_prompt_tokens={self.max_prompt_tokens}"
            )
        self.n_calls += 1
        out: list[list[Completion]] = []
        for prompt in prompts:
            answer = self._answer_for(prompt)
            # Greedy (T=0) is deterministic: every sample is the same draw.
            out.append(
                [
                    self._one(prompt, 0 if temperature == 0.0 else j, temperature, answer)
                    for j in range(n)
                ]
            )
        return out

    def close(self) -> None:
        return None


__all__ = ["FakeChatTokenizer", "StubSampler", "PROVISIONAL_CAP"]
