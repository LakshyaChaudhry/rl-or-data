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
