"""tasks/03: selection bookkeeping, tokenization, batching, configs (no training, no GPU)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rlordata.core.rft_select import SFTExample
from rlordata.sampling.prompts import TEMPLATE, render_template
from rlordata.train import rft
from rlordata.train.common import load_yaml, lora_kwargs_from_training, tokenizer_hash
from rlordata.train.rft import (
    Selection,
    assert_curated_matches,
    batches_for_epoch,
    collate,
    encode_example,
    epoch_order,
    run_name,
    select_examples,
)
from rlordata.types import Problem, Sample

os.environ.setdefault("HF_HUB_OFFLINE", "1")

VLLM_DTYPES = {"auto", "half", "float16", "bfloat16", "float", "float32"}


@pytest.mark.parametrize("arm", ["easy", "mixed", "curated"])
def test_draw_sampler_gets_a_vllm_dtype(arm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Arm configs say ``dtype: bf16``; vLLM's ModelConfig rejects that (the first GPU draw crashed)."""
    seen: dict[str, object] = {}

    def fake_make_sampler(spec: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr("rlordata.sampling.eval_runner.make_sampler", fake_make_sampler)
    cfg = {
        **load_yaml(Path("configs/rft") / f"{arm}.yaml"),
        "max_completion_tokens": 4352,
        "cap_yaml": "configs/locked/cap.yaml",
    }
    rft._sampler_for_draw(cfg, seed=1, stub=False, problems=[])
    assert seen["dtype"] in VLLM_DTYPES
    assert seen["dtype"] == "bfloat16"


def _problem(i: int, tier: str = "medium", pass8: int = 3) -> Problem:
    return Problem(
        problem_id=f"p{i}",
        text=f"Consider the integers from 1 to {10 + i}, inclusive. First, keep only the even numbers. Of these numbers, count how many values remain.",
        answer=5,
        pipeline={
            "range": {"lo": 1, "hi": 10 + i},
            "filters": [],
            "transforms": [],
            "op": {"name": "count"},
        },
        range_scale="S",
        n_filters=1,
        n_transforms=0,
        total_steps=2,
        tier=tier,  # type: ignore[arg-type]
        pass8=pass8,
    )


def _sample(p: Problem, j: int, correct: bool) -> Sample:
    return Sample(
        run_id="r",
        config_hash="h",
        seed=1,
        arm="base",
        data_condition="train_mixed_100",
        problem_id=p.problem_id,
        tier=p.tier,
        prompt=render_template(p.text),
        completion=f"steps...\nAnswer: {p.answer if correct else p.answer + 1}",
        extracted_answer=p.answer if correct else p.answer + 1,
        correct=correct,
        reward=1.0 if correct else 0.0,
        n_tokens=20,
        truncated=False,
        extraction_failed=False,
        extra={"sample_idx": j},
    )


def _fake_rft_select(problems, samples, mode, max_per_problem=None):
    """Test double for the selection *bookkeeping* only (core.rft_select is Laksh's)."""
    out = []
    for p in problems:
        ss = samples[p.problem_id]
        c8 = sum(s.correct for s in ss[:8])
        if mode == "curated" and not (1 <= c8 <= 7):
            continue
        seen = set()
        for s in ss:
            if s.correct and s.completion not in seen:
                seen.add(s.completion)
                out.append(SFTExample(p.problem_id, s.prompt, s.completion, p.tier))
    return out


def test_select_budgets(monkeypatch) -> None:
    monkeypatch.setattr(rft, "rft_select", _fake_rft_select)
    problems = [_problem(0, "easy", 8), _problem(1, "medium", 3), _problem(2, "hard", 0)]
    draws = {
        "p0": [
            _sample(problems[0], j, True) for j in range(12)
        ],  # 12 correct, all identical text → 1 after dedup
        "p1": [
            _sample(problems[1], j, j % 4 == 0) for j in range(12)
        ],  # 3 correct, identical text → 1
        "p2": [_sample(problems[2], j, False) for j in range(12)],
    }
    sel = select_examples(problems, draws, mode="all", max_per_problem=None, samples_per_prompt=12)
    b = sel.budgets
    assert b["prompts"] == 2 and b["prompts_parent_split"] == 3
    assert b["completions_available"] == 36 and b["completions_consumed"] == 2
    assert b["correct_before_dedup"] == 15 and b["correct_after_dedup"] == 2
    assert b["per_tier_prompts"] == {"easy": 1, "medium": 1}
    curated = select_examples(
        problems, draws, mode="curated", max_per_problem=None, samples_per_prompt=12
    )
    assert curated.problem_ids == {"p1"}
    with pytest.raises(ValueError):
        select_examples(problems, {"p0": draws["p0"]}, mode="all", max_per_problem=None)
    with pytest.raises(ValueError):
        select_examples(problems, draws, mode="best", max_per_problem=None)


def test_assert_curated_matches(tmp_path: Path) -> None:
    from rlordata.data.generator import write_jsonl

    write_jsonl([_problem(1)], tmp_path / "train_curated.jsonl")
    ex = SFTExample("p1", "prompt", "c", "medium")
    assert_curated_matches(Selection([ex], {}), tmp_path / "train_curated.jsonl")
    with pytest.raises(SystemExit):
        assert_curated_matches(
            Selection([SFTExample("p2", "prompt", "c", "medium")], {}),
            tmp_path / "train_curated.jsonl",
        )


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen tokenizer not cached (offline)")


def test_encode_example_roundtrip_and_mask(tokenizer) -> None:
    p = _problem(3)
    ex = SFTExample(
        p.problem_id,
        render_template(p.text),
        "Let's count.\n2, 4, 6, 8, 10, 12.\n**Answer: 6**\n",
        p.tier,
    )
    enc = encode_example(tokenizer, ex)
    assert tokenizer.decode(enc.input_ids) == ex.prompt + ex.completion
    assert enc.completion_mask[: enc.n_prompt_tokens] == [0] * enc.n_prompt_tokens
    assert enc.completion_mask[enc.n_prompt_tokens :] == [1] * enc.n_completion_tokens
    assert enc.n_prompt_tokens == len(tokenizer(ex.prompt, add_special_tokens=False)["input_ids"])
    assert tokenizer.eos_token_id not in enc.input_ids  # nothing appended (tasks/03 §3)
    with pytest.raises(ValueError, match="TEMPLATE"):
        encode_example(tokenizer, SFTExample("x", "Problem: y", "Answer: 1", "easy"))
    with pytest.raises(ValueError, match="empty"):
        encode_example(tokenizer, SFTExample("x", render_template("y"), "", "easy"))


def test_collate_right_pads(tokenizer) -> None:
    torch = pytest.importorskip("torch")
    p = _problem(4)
    a = encode_example(tokenizer, SFTExample("a", render_template(p.text), "Answer: 5", "easy"))
    b = encode_example(
        tokenizer,
        SFTExample("b", render_template(p.text), "Longer reasoning here.\nAnswer: 5", "easy"),
    )
    t = collate([a, b], pad_id=tokenizer.pad_token_id)
    assert (
        t["input_ids"].shape
        == t["attention_mask"].shape
        == t["completion_mask"].shape
        == (2, len(b.input_ids))
    )
    assert int(t["attention_mask"][0].sum()) == len(a.input_ids)
    assert int(t["completion_mask"][0].sum()) == a.n_completion_tokens
    assert torch.all(t["input_ids"][0, len(a.input_ids) :] == tokenizer.pad_token_id)


def test_epoch_order_is_seeded() -> None:
    assert epoch_order(10, 1, 0) == epoch_order(10, 1, 0)
    assert epoch_order(10, 1, 0) != epoch_order(10, 2, 0)
    assert epoch_order(10, 1, 0) != epoch_order(10, 1, 1)
    assert sorted(epoch_order(10, 1, 0)) == list(range(10))
    batches = batches_for_epoch(10, 1, 0, 4)
    assert [len(b) for b in batches] == [4, 4, 2]


def test_lora_kwargs_come_from_locked_training_yaml() -> None:
    tr = load_yaml("configs/locked/training.yaml")
    kw = lora_kwargs_from_training(tr)
    assert kw["r"] == 64 and kw["lora_alpha"] == 16 and kw["lora_dropout"] == 0.0
    assert set(kw["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert tr["rft"]["sweep"] == {"learning_rate": [1e-5, 5e-5, 1e-4], "epochs": [2, 4, 8]}


def test_run_name_and_tokenizer_hash(tokenizer) -> None:
    assert run_name(2, 5e-5, 4) == "seed2_lr5e-05_ep4"
    h = tokenizer_hash(tokenizer)
    assert len(h) == 64 and h == tokenizer_hash(tokenizer)
    assert TEMPLATE in render_template("x") or render_template("x").startswith(
        TEMPLATE.split("{")[0]
    )
