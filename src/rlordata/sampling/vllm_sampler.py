"""The single generation path for every number in the paper (SPEC §10, CLAUDE.md). AGENT-OWNED; tasks/02.

Contract:
    class VLLMSampler:
        __init__(model_id, lora_path=None, max_completion_tokens=<from configs/locked/cap.yaml>, seed=...)
        sample(prompts: list[str], n: int, temperature: float, top_p: float = 1.0) -> list[list[Completion]]
    Completion has: text, n_tokens, truncated (finish_reason == "length").
    Prompt formatting uses rlordata.sampling.prompts.format_prompt(problem, model_kind)   # base vs instruct
    Refuses to construct if the cap passed differs from configs/locked/cap.yaml once that file exists.
"""

from __future__ import annotations
