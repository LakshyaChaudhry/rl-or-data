from __future__ import annotations

import pytest

from rlordata.sampling.prompts import (
    TEMPLATE,
    THINKING_OFF_KWARGS,
    chat_template_kwargs,
    format_prompt,
    format_prompts,
)
from rlordata.sampling.stub_sampler import FakeChatTokenizer


def test_base_prompt_is_template_verbatim(toy_problem) -> None:
    assert format_prompt(toy_problem, "base") == TEMPLATE.format(problem_text=toy_problem.text)
    assert format_prompt(toy_problem, "base").endswith("\n")


def test_instruct_wrapping_uses_same_text_and_thinking_off(toy_problem) -> None:
    tok = FakeChatTokenizer()
    out = format_prompt(toy_problem, "instruct", tok, model_id="Qwen/Qwen3-4B")
    assert TEMPLATE.format(problem_text=toy_problem.text) in out
    assert out.startswith("<|user|>") and out.endswith("<|assistant|>\n")
    assert tok.calls == [{"enable_thinking": False}]

    tok2 = FakeChatTokenizer()
    format_prompt(toy_problem, "instruct", tok2, model_id="Qwen/Qwen2.5-7B-Instruct")
    assert tok2.calls == [{}]
    tok3 = FakeChatTokenizer()
    format_prompt(toy_problem, "instruct", tok3, model_id="google/gemma-4-E4B-it")
    assert tok3.calls == [{"enable_thinking": False}]
    tok4 = FakeChatTokenizer()
    format_prompt(toy_problem, "instruct", tok4, model_id="meta-llama/Llama-3.1-8B-Instruct")
    assert tok4.calls == [{}]


def test_every_reference_model_is_registered() -> None:
    for mid in (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-4-E4B-it",
    ):
        kw = chat_template_kwargs(mid)
        assert kw.get("enable_thinking", False) is False
    assert chat_template_kwargs("Qwen/Qwen3-4B") is not THINKING_OFF_KWARGS["Qwen/Qwen3-4B"]  # copy


def test_instruct_errors(toy_problem) -> None:
    tok = FakeChatTokenizer()
    with pytest.raises(ValueError):
        format_prompt(toy_problem, "instruct", tok, model_id="some/unregistered-instruct")
    with pytest.raises(ValueError):
        format_prompt(toy_problem, "instruct", None, model_id="Qwen/Qwen3-4B")
    with pytest.raises(ValueError):
        format_prompt(toy_problem, "instruct", tok)  # neither model_id nor chat_kwargs
    with pytest.raises(ValueError):
        format_prompt(toy_problem, "instruct", tok, chat_kwargs={"enable_thinking": True})
    with pytest.raises(ValueError):
        format_prompt(toy_problem, "chat", tok)  # type: ignore[arg-type]


def test_format_prompts_vectorised(toy_problem) -> None:
    tok = FakeChatTokenizer()
    outs = format_prompts([toy_problem, toy_problem], "instruct", tok, model_id="Qwen/Qwen3-4B")
    assert len(outs) == 2 and outs[0] == outs[1]
    assert format_prompts([toy_problem], "base") == [format_prompt(toy_problem, "base")]


@pytest.mark.slow
def test_real_qwen3_tokenizer_renders_thinking_off(toy_problem) -> None:
    """Needs network / HF cache: loads the real Qwen3-4B tokenizer."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load Qwen/Qwen3-4B tokenizer: {exc}")
    out = format_prompt(toy_problem, "instruct", tok, model_id="Qwen/Qwen3-4B")
    assert TEMPLATE.format(problem_text=toy_problem.text) in out
    assert "<|im_start|>user" in out and out.rstrip().endswith(
        ("<|im_start|>assistant", "</think>")
    )
    # enable_thinking=False makes the Qwen3 template emit an empty think block in the assistant turn
    assert "<think>\n\n</think>" in out
