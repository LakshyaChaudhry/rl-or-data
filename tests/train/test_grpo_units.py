"""Unit tests for GRPO harness pieces that do not need GPU/TRL (tasks/04)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlordata.sampling.prompts import format_prompt
from rlordata.train.grpo_trl import (
    GENERATION_BATCH_SIZE,
    arm_run_name,
    build_dataset,
    prompt_bytes_hash,
)
from rlordata.train.rewards import RewardRecorder, make_reward_fn
from rlordata.types import Problem


def _problem(i: int, tier: str = "medium") -> Problem:
    return Problem(
        problem_id=f"pid{i}",
        text=(
            f"Consider the integers from 1 to {10 + i}, inclusive. "
            "First, keep only the even numbers. Of these numbers, count how many values remain."
        ),
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
    )


def test_arm_run_name():
    assert arm_run_name({"arm": "grpo", "data_condition": "train_mixed_100"}, 1) == "grpo_mixed_s1"
    assert arm_run_name({"arm": "grpo", "data_condition": "train_easy_100"}, 2) == "grpo_easy_s2"
    assert arm_run_name({"arm": "grpo", "data_condition": "train_curated"}, 1) == "grpo_curated_s1"
    assert arm_run_name({"arm": "grpo_random_reward", "data_condition": "train_mixed_100"}, 1) == (
        "grpo_random_reward_s1"
    )


def test_generation_batch_budget():
    assert GENERATION_BATCH_SIZE == 64
    assert 300 * GENERATION_BATCH_SIZE == 19200


def test_build_dataset_plain_prompt():
    pytest.importorskip("datasets")
    problems = [_problem(0), _problem(1, "hard")]
    ds = build_dataset(problems)
    assert list(ds["problem_id"]) == ["pid0", "pid1"]
    assert ds["prompt"][0] == format_prompt(problems[0], "base")
    assert "{problem_text}" not in ds["prompt"][0]


def test_prompt_bytes_hash_stable():
    problems = [_problem(0)]
    a = prompt_bytes_hash(problems)
    b = prompt_bytes_hash(problems)
    assert a == b
    assert len(a["pid0"]) == 64


def test_verify_binary_reward(tmp_path: Path):
    rec = RewardRecorder(tmp_path / "reward_records.jsonl")
    fn = make_reward_fn("verify_binary", recorder=rec, seed=1, max_completion_length=128)
    p = _problem(0)
    good = "reasoning\nAnswer: 5"
    bad = "reasoning\nAnswer: 9"
    rewards = fn(
        [good, bad],
        problem_id=[p.problem_id, p.problem_id],
        answer=[p.answer, p.answer],
        tier=[p.tier, p.tier],
    )
    assert rewards == [1.0, 0.0]
    lines = (tmp_path / "reward_records.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(x) for x in lines]
    assert rows[0]["correct"] is True and rows[1]["correct"] is False


def test_random_bernoulli_independent_of_content(tmp_path: Path):
    rec = RewardRecorder(tmp_path / "r.jsonl")
    fn = make_reward_fn("random_bernoulli", recorder=rec, seed=7)
    p = _problem(0)
    a = fn(
        ["Answer: 5", "garbage"],
        problem_id=[p.problem_id, p.problem_id],
        answer=[p.answer, p.answer],
        tier=[p.tier, p.tier],
    )
    rec2 = RewardRecorder(tmp_path / "r2.jsonl")
    fn2 = make_reward_fn("random_bernoulli", recorder=rec2, seed=7)
    b = fn2(
        ["totally different", "also different"],
        problem_id=[p.problem_id, p.problem_id],
        answer=[p.answer, p.answer],
        tier=[p.tier, p.tier],
    )
    assert a == b
    rows = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().strip().splitlines()]
    assert rows[0]["correct"] is True


def test_format_only_reward(tmp_path: Path):
    rec = RewardRecorder(tmp_path / "r.jsonl")
    fn = make_reward_fn("format_only", recorder=rec, seed=1)
    p = _problem(0)
    rewards = fn(
        ["Answer: 5", "no integer here"],
        problem_id=p.problem_id,
        answer=p.answer,
        tier=p.tier,
    )
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0


# --- callbacks: reward records are tagged with the step being trained (tasks/04 §4) ---


class _State:
    def __init__(self, global_step: int) -> None:
        self.global_step = global_step


def test_callback_tags_records_with_current_step_and_accumulates_tokens(tmp_path: Path):
    from rlordata.train.callbacks import GrpoDiagnosticsCallback

    rec = RewardRecorder(tmp_path / "reward_records.jsonl")
    fn = make_reward_fn("verify_binary", recorder=rec, seed=1, max_completion_length=8)
    cb = GrpoDiagnosticsCallback(
        run_dir=tmp_path, recorder=rec, num_generations=2, est_gpu_hours=0.0
    )
    p_easy, p_hard = _problem(0, "easy"), _problem(1, "hard")
    for n in (1, 2):
        # HF order: on_step_begin (global_step == n-1) → rewards inside training_step → on_log (== n)
        cb.on_step_begin(None, _State(n - 1), None)
        fn(
            ["Answer: 5", "Answer: 9", "Answer: 5", "Answer: 5"],
            problem_id=[p_easy.problem_id] * 2 + [p_hard.problem_id] * 2,
            answer=[5] * 4,
            tier=["easy", "easy", "hard", "hard"],
            completion_ids=[[1, 2, 3], [1, 2], [1, 2, 3, 4, 5], [1, 2, 3, 4]],
        )
        cb.on_log(None, _State(n), None, logs={"reward": 0.75, "frac_reward_zero_std": 0.5})
    rows = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["cumulative_completions"] == 4 and rows[1]["cumulative_completions"] == 8
    assert rows[0]["cumulative_training_tokens"] == 3 + 2 + 5 + 4
    assert rows[1]["cumulative_training_tokens"] == 2 * (3 + 2 + 5 + 4)
    assert rows[0]["per_tier"]["easy"]["frac_reward_zero_std"] == 0.0  # group [1, 0]
    assert rows[0]["per_tier"]["hard"]["frac_reward_zero_std"] == 1.0  # group [1, 1]
    assert rows[0]["reward_mean"] == 0.75 and rows[0]["frac_reward_zero_std"] == 0.5
    recs = [json.loads(x) for x in (tmp_path / "reward_records.jsonl").read_text().splitlines()]
    assert {r["step"] for r in recs} == {1, 2}
    assert all(r["truncated"] is False for r in recs)


def test_reward_at_cap_is_truncated_and_scores_zero(tmp_path: Path):
    """A completion that used every allowed token never committed to an answer (SPEC §5/§7)."""
    rec = RewardRecorder(tmp_path / "r.jsonl")
    fn = make_reward_fn("verify_binary", recorder=rec, seed=1, max_completion_length=8)
    p = _problem(0)
    rewards = fn(
        ["Answer: 5", "Answer: 5"],
        problem_id=[p.problem_id] * 2,
        answer=[5, 5],
        tier=["easy", "easy"],
        completion_ids=[[1] * 8, [1] * 7],
    )
    assert rewards == [0.0, 1.0]
    recs = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert recs[0]["truncated"] is True and recs[0]["n_tokens"] == 8
    assert recs[0]["extraction_failed"] is True and recs[0]["correct"] is False


# --- sanity: only the step-300 adapter may be evaluated on the held-out sets (tasks/04 §5/§6) ---


def _grpo_budgets(tmp_path: Path, *, final: str, eval_step: int | None) -> Path:
    from rlordata.train.common import write_json

    run = tmp_path / "grpo_mixed_s1"
    for name in ("step_100", "step_200", "step_300", "final"):
        (run / "adapter" / name).mkdir(parents=True, exist_ok=True)
    b = {
        "final_adapter": str(run / "adapter" / final),
        "final_adapter_trained": str(run / "adapter" / "final"),
        "step_adapters": {str(s): str(run / "adapter" / f"step_{s}") for s in (100, 200, 300)},
        "max_steps": 300,
        "optimizer_steps": 300,
    }
    if eval_step is not None:
        b["eval_step"] = eval_step
    write_json(run / "budgets.json", b)
    return run


def test_final_checkpoint_gate_refuses_test_on_mid_checkpoint(tmp_path: Path):
    from rlordata.analysis.sanity import check_final_checkpoint

    run = _grpo_budgets(tmp_path, final="step_100", eval_step=100)
    assert check_final_checkpoint(run, eval_set="val") == []
    issues = check_final_checkpoint(run, eval_set="final")
    assert issues and "step-300" in issues[0]

    run = _grpo_budgets(tmp_path, final="step_300", eval_step=300)
    assert check_final_checkpoint(run, eval_set="final") == []

    # pointer and eval_step disagree → refuse either way
    run = _grpo_budgets(tmp_path, final="step_200", eval_step=300)
    assert check_final_checkpoint(run, eval_set="val")


def test_grpo_reward_budget_gate(tmp_path: Path):
    from rlordata.analysis.sanity import check_c1_reward_near_half, check_grpo_reward_budget
    from rlordata.train.common import write_json

    run = tmp_path / "r"
    run.mkdir()
    rec = RewardRecorder(run / "reward_records.jsonl")
    fn = make_reward_fn("random_bernoulli", recorder=rec, seed=3)
    p = _problem(0)
    for i in range(50):
        rec.set_step(
            i + 1
        )  # as the callback does: the draw is seeded by (seed, problem_id, step, i)
        fn([f"x{i}"] * 4, problem_id=p.problem_id, answer=p.answer, tier=p.tier)
    write_json(run / "budgets.json", {"completions_consumed": 200})
    assert check_grpo_reward_budget(run, expected=200) == []
    assert check_grpo_reward_budget(run, expected=19200)
    assert check_c1_reward_near_half(run) == []
