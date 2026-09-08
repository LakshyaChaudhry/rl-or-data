"""Prompt template (SPEC §5). Locked text; agent implements the instruct-model wrapping only.

Instruct reference models receive the *same* TEMPLATE text as the single user turn of their own
chat template, with thinking disabled (SPEC §5). The thinking-off kwargs are per model and live in
:data:`THINKING_OFF_KWARGS`; an instruct model that is not registered there raises, so a thinking
model can never be evaluated with thinking silently enabled. The exact kwargs used are recorded
in every run's resolved config by the eval runner.
"""

from __future__ import annotations

from typing import Any, Literal

from rlordata.types import Problem

# locked: do not edit the template text without a SPEC changelog entry
TEMPLATE = (
    "Solve the following problem. Think step by step, then give the final answer on the last line "
    'in the form "Answer: <integer>".\n\nProblem: {problem_text}\n'
)

ModelKind = Literal["base", "instruct"]

# Chat-template kwargs that disable thinking, per reference model (tasks/02a "Models" table).
# Verified 2026-09-07 against the model cards: Qwen3 and Gemma 4 expose ``enable_thinking``;
# Qwen2.5-Instruct and Llama-3.1-Instruct have no thinking mode.
THINKING_OFF_KWARGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-4B": {"enable_thinking": False},
    "Qwen/Qwen2.5-7B-Instruct": {},
    "meta-llama/Llama-3.1-8B-Instruct": {},
    "google/gemma-4-E4B-it": {"enable_thinking": False},
}


def chat_template_kwargs(model_id: str) -> dict[str, Any]:
    """Thinking-off kwargs for a registered instruct model; unknown ids raise (never guess)."""
    try:
        return dict(THINKING_OFF_KWARGS[model_id])
    except KeyError as exc:
        raise ValueError(
            f"no thinking-off chat-template kwargs registered for instruct model {model_id!r}; "
            "add it to rlordata.sampling.prompts.THINKING_OFF_KWARGS after checking its model card"
        ) from exc


def render_template(problem_text: str) -> str:
    return TEMPLATE.format(problem_text=problem_text)


def format_prompt(
    problem: Problem,
    model_kind: ModelKind,
    tokenizer=None,  # noqa: ANN001 — HF tokenizer or FakeChatTokenizer (duck-typed)
    *,
    model_id: str | None = None,
    chat_kwargs: dict[str, Any] | None = None,
) -> str:
    """Base models get TEMPLATE verbatim. Instruct models get TEMPLATE inside their chat template
    with thinking disabled (per-model kwargs from :data:`THINKING_OFF_KWARGS` or ``chat_kwargs``)."""
    text = render_template(problem.text)
    if model_kind == "base":
        return text
    if model_kind != "instruct":
        raise ValueError(f"model_kind must be 'base' or 'instruct', got {model_kind!r}")
    if tokenizer is None:
        raise ValueError("instruct wrapping needs the model's tokenizer (apply_chat_template)")
    if chat_kwargs is None:
        if model_id is None:
            raise ValueError(
                "instruct wrapping needs model_id (to look up thinking-off kwargs) or chat_kwargs"
            )
        chat_kwargs = chat_template_kwargs(model_id)
    for key, value in chat_kwargs.items():
        if key == "enable_thinking" and value is not False:
            raise ValueError("thinking must be disabled for every reference model (SPEC §5)")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )


def format_prompts(
    problems: list[Problem],
    model_kind: ModelKind,
    tokenizer=None,  # noqa: ANN001
    *,
    model_id: str | None = None,
    chat_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Vectorised :func:`format_prompt`; resolves the chat kwargs once."""
    if model_kind == "instruct" and chat_kwargs is None:
        if model_id is None:
            raise ValueError("instruct wrapping needs model_id or chat_kwargs")
        chat_kwargs = chat_template_kwargs(model_id)
    return [
        format_prompt(p, model_kind, tokenizer, model_id=model_id, chat_kwargs=chat_kwargs)
        for p in problems
    ]
