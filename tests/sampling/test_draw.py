"""tasks/03 §1: the 192-sample draw (tiering 8 first + new draws), written once, read-only."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rlordata.cli import main
from rlordata.data.generator import read_jsonl
from rlordata.sampling.cap import PROVISIONAL_CAP, write_cap_yaml
from rlordata.sampling.draw import (
    DRAW_SEED_OFFSET,
    SAMPLES_PER_PROMPT,
    TIERING_K,
    draw_path,
    draw_split,
    load_tiering_samples,
    read_draw,
    summarize_draw,
    write_draw,
)
from rlordata.sampling.stub_sampler import StubSampler
from tests.helpers import World, load_yaml, make_world

N_TOTAL = 12  # dry-run size; the protocol value is asserted separately


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    w = make_world(tmp_path_factory.mktemp("draw_world"))
    assert w.tier_with_stub() == 0
    write_cap_yaml(w.cap_path, {"max_completion_tokens": PROVISIONAL_CAP})
    return w


def _arm_config(world: World, tmp: Path, arm: str = "mixed") -> Path:
    cfg = load_yaml(f"configs/rft/{arm}.yaml")
    cfg.update(
        {
            "cap_yaml": str(world.cap_path),
            "splits_dir": str(world.splits_dir),
            "samples_dir": str(tmp / "samples"),
            "tiering_samples": str(world.samples_path),
            "output_dir": str(tmp / "runs"),
        }
    )
    p = tmp / f"rft_{arm}.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return p


def test_protocol_constants() -> None:
    assert SAMPLES_PER_PROMPT == 192 and TIERING_K == 8  # SPEC §8 / §6.1
    assert SAMPLES_PER_PROMPT - TIERING_K == 184
    assert draw_path("train_mixed_100", 1).name == "base_train_mixed_100_k192_seed1.jsonl"


def test_tiering_samples_loaded_in_generation_order(world: World) -> None:
    problems = read_jsonl(world.splits_dir / "train_mixed_100.jsonl")
    tiering = load_tiering_samples(world.samples_path, problems)
    assert set(tiering) == {p.problem_id for p in problems}
    for p in problems:
        idx = [s.extra["sample_idx"] for s in tiering[p.problem_id]]
        assert idx == list(range(TIERING_K))
        assert sum(s.correct for s in tiering[p.problem_id]) == p.pass8


def test_tiering_loader_rejects_mismatched_pass8(world: World, tmp_path: Path) -> None:
    problems = read_jsonl(world.splits_dir / "train_mixed_100.jsonl")
    p = problems[0]
    wrong = type(p)(**{**p.to_dict(), "pass8": (p.pass8 or 0) + 1 if (p.pass8 or 0) < 8 else 0})
    with pytest.raises(ValueError, match="pass8"):
        load_tiering_samples(world.samples_path, [wrong])


def test_draw_split_puts_tiering_first_and_uses_offset_seed(world: World) -> None:
    problems = read_jsonl(world.splits_dir / "train_easy_100.jsonl")[:5]
    tiering = load_tiering_samples(world.samples_path, problems)
    sampler = StubSampler.from_problems(
        problems,
        model_id="Qwen/Qwen3-4B-Base",
        model_kind="base",
        max_completion_tokens=PROVISIONAL_CAP,
        seed=1,  # the run seed: must be refused
        cap_path=world.cap_path,
    )
    with pytest.raises(AssertionError, match="DRAW_SEED_OFFSET"):
        draw_split(
            problems,
            tiering,
            sampler,
            split="train_easy_100",
            run_id="r",
            config_hash="h",
            seed=1,
            n_total=N_TOTAL,
        )
    sampler = StubSampler.from_problems(
        problems,
        model_id="Qwen/Qwen3-4B-Base",
        model_kind="base",
        max_completion_tokens=PROVISIONAL_CAP,
        seed=1 + DRAW_SEED_OFFSET,
        cap_path=world.cap_path,
    )
    samples = draw_split(
        problems,
        tiering,
        sampler,
        split="train_easy_100",
        run_id="r",
        config_hash="h",
        seed=1,
        n_total=N_TOTAL,
    )
    assert len(samples) == len(problems) * N_TOTAL
    by_pid: dict[str, list] = {}
    for s in samples:
        by_pid.setdefault(s.problem_id, []).append(s)
    for p in problems:
        ss = by_pid[p.problem_id]
        assert [s.extra["sample_idx"] for s in ss] == list(range(N_TOTAL))
        assert [s.completion for s in ss[:TIERING_K]] == [
            s.completion for s in tiering[p.problem_id]
        ]
        assert all(s.extra["source"] == "tiering" for s in ss[:TIERING_K])
        assert all(s.extra["source"] == "draw" for s in ss[TIERING_K:])
        assert all(s.data_condition == "train_easy_100" for s in ss)
        assert all(s.extra["sampler_seed"] == 1 + DRAW_SEED_OFFSET for s in ss[TIERING_K:])
    # Determinism: the same sampler seed gives the same draw.
    again = draw_split(
        problems,
        tiering,
        sampler,
        split="train_easy_100",
        run_id="r",
        config_hash="h",
        seed=1,
        n_total=N_TOTAL,
    )
    assert [s.completion for s in again] == [s.completion for s in samples]


def test_write_draw_is_read_only_and_refuses_overwrite(tmp_path: Path, world: World) -> None:
    problems = read_jsonl(world.splits_dir / "train_easy_100.jsonl")[:2]
    tiering = load_tiering_samples(world.samples_path, problems)
    sampler = StubSampler.from_problems(
        problems,
        model_id="m",
        model_kind="base",
        max_completion_tokens=PROVISIONAL_CAP,
        seed=1 + DRAW_SEED_OFFSET,
        cap_path=world.cap_path,
    )
    samples = draw_split(
        problems,
        tiering,
        sampler,
        split="train_easy_100",
        run_id="r",
        config_hash="h",
        seed=1,
        n_total=N_TOTAL,
    )
    path = write_draw(samples, tmp_path / "d.jsonl")
    assert oct(os.stat(path).st_mode & 0o777) == oct(0o444)
    with pytest.raises(FileExistsError):
        write_draw(samples, path)
    back = read_draw(path, expect_n=N_TOTAL)
    assert set(back) == {p.problem_id for p in problems}
    summary = summarize_draw(back)
    assert summary["n_samples"] == 2 * N_TOTAL and summary["samples_per_problem"] == N_TOTAL
    assert sum(summary["pass8_histogram"].values()) == 2
    assert 0.0 <= summary["mean_pass_rate"] <= 1.0
    with pytest.raises(ValueError):
        read_draw(path, expect_n=N_TOTAL + 1)


def test_cli_draw_stage_writes_both_splits_once(world: World, tmp_path: Path) -> None:
    cfg = _arm_config(world, tmp_path)
    assert (
        main(["rft", "--config", str(cfg), "--stage", "draw", "--stub", "--n-total", str(N_TOTAL)])
        == 0
    )
    for split in ("train_easy_100", "train_mixed_100"):
        p = draw_path(split, 1, samples_dir=tmp_path / "samples")
        assert p.exists() and oct(os.stat(p).st_mode & 0o777) == oct(0o444)
        n_probs = len(read_jsonl(world.splits_dir / f"{split}.jsonl"))
        assert len(read_draw(p, expect_n=N_TOTAL)) == n_probs
    run_dir = tmp_path / "runs" / "draw_seed1_stub"
    assert (run_dir / "draw_summary.json").exists() and (run_dir / "config.yaml").exists()
    # Second call: nothing re-drawn.
    mtimes = {
        s: os.stat(draw_path(s, 1, samples_dir=tmp_path / "samples")).st_mtime
        for s in ("train_easy_100", "train_mixed_100")
    }
    assert (
        main(["rft", "--config", str(cfg), "--stage", "draw", "--stub", "--n-total", str(N_TOTAL)])
        == 0
    )
    assert mtimes == {
        s: os.stat(draw_path(s, 1, samples_dir=tmp_path / "samples")).st_mtime for s in mtimes
    }


def test_cli_draw_refuses_non_protocol_n_without_stub(world: World, tmp_path: Path) -> None:
    (tmp_path / "b").mkdir(exist_ok=True)
    cfg = _arm_config(world, tmp_path / "b")
    with pytest.raises(SystemExit):
        main(["rft", "--config", str(cfg), "--stage", "draw", "--n-total", "12"])
