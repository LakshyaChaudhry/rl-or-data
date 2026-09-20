"""tasks/06b: iterated RFT — round seeds, round-1 reuse of the base draw, design guards, dry run."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import yaml

from rlordata.sampling import cap as cap_mod
from rlordata.train import iter_rft
from rlordata.train.common import load_arm_config
from rlordata.types import Problem, Sample
from tests.helpers import REPO


def _problem(i: int, tier: str = "easy") -> Problem:
    return Problem(
        problem_id=f"p{i}", text=f"q{i}", answer=i, pipeline={}, range_scale="S",
        n_filters=1, n_transforms=0, total_steps=2, tier=tier, pass8=4,
    )  # fmt: skip


def _sample(p: Problem, j: int, *, correct: bool, completion: str | None = None) -> Sample:
    from rlordata.sampling.prompts import format_prompt

    return Sample(
        run_id="r", config_hash="h", seed=1, arm="base", data_condition="train_mixed_100",
        problem_id=p.problem_id, tier="untiered", prompt=format_prompt(p, "base"),
        completion=completion or f"work {j}\nAnswer: {p.answer if correct else p.answer + 1}",
        extracted_answer=p.answer if correct else p.answer + 1, correct=correct,
        reward=float(correct), n_tokens=100 + j, truncated=False, extraction_failed=False,
        extra={"sample_idx": j},
    )  # fmt: skip


def test_round_sampler_seeds_are_deterministic_distinct_and_never_a_training_seed() -> None:
    seeds = {(s, r): iter_rft.round_sampler_seed(s, r) for s in (1, 2, 3, 4, 5) for r in (1, 2, 3)}
    assert seeds == {(s, r): iter_rft.round_sampler_seed(s, r) for s, r in seeds}
    assert len(set(seeds.values())) == len(seeds)
    assert not set(seeds.values()) & {1, 2, 3, 4, 5}


def test_round1_is_the_first_n_of_the_base_draw_in_generation_order() -> None:
    problems = [_problem(0), _problem(1, "hard")]
    draw = {p.problem_id: [_sample(p, j, correct=j % 2 == 0) for j in range(12)] for p in problems}
    out = iter_rft.round1_samples(problems, draw, 4)
    assert [(s.problem_id, s.extra["sample_idx"]) for s in out] == [
        (p.problem_id, j) for p in problems for j in range(4)
    ]
    assert {s.tier for s in out if s.problem_id == "p1"} == {"hard"}  # frozen tier stamped
    assert all(s.extra["round"] == 1 and s.data_condition == "train_curated" for s in out)
    assert [s.completion for s in out[:4]] == [s.completion for s in draw["p0"][:4]]
    with pytest.raises(SystemExit):
        iter_rft.round1_samples(problems, {"p0": draw["p0"]}, 4)  # a curated prompt without a draw


def test_round_diagnostics_counts() -> None:
    from rlordata.train.rft import select_examples

    problems = [_problem(0), _problem(1, "hard"), _problem(2, "medium")]
    samples = (
        [_sample(problems[0], j, correct=True, completion="same\nAnswer: 0") for j in range(4)]
        + [_sample(problems[1], j, correct=False) for j in range(4)]
        + [_sample(problems[2], j, correct=j < 2) for j in range(4)]
    )
    by_pid: dict[str, list[Sample]] = {}
    for s in samples:
        by_pid.setdefault(s.problem_id, []).append(s)
    sel = select_examples(problems, by_pid, mode="all", max_per_problem=None, samples_per_prompt=4)
    d = iter_rft.round_diagnostics(problems, samples, sel)
    assert d["n_prompts_all_correct"] == 1 and d["n_prompts_zero_correct"] == 1
    assert d["correct_before_dedup"] == 6 and d["kept_after_dedup"] == len(sel.examples)
    assert d["pass_rate_per_tier"] == {"easy": 1.0, "hard": 0.0, "medium": 0.5}
    assert d["kept"]["truncation_rate"] == 0.0 and d["sampled"]["mean_tokens"] > 0


def test_the_committed_config_is_the_registered_design() -> None:
    cfg = load_arm_config(REPO / "configs/rft/iter_curated.yaml")
    iter_rft.validate_config(cfg)
    assert (cfg["rounds"], cfg["samples_per_round"], cfg["epochs"]) == (3, 64, 4)
    assert float(cfg["learning_rate"]) == 1e-5 and cfg["data_condition"] == "train_curated"
    n_curated = sum(1 for _ in (REPO / "data/splits/train_curated.jsonl").open())
    assert n_curated * cfg["samples_per_round"] * cfg["rounds"] == 14_016
    chosen = REPO / "results/phase2/rft/rft_curated/chosen.json"
    got = json.loads(chosen.read_text())
    assert (got["learning_rate"], got["epochs"]) == (cfg["learning_rate"], cfg["epochs"])


@pytest.mark.parametrize(
    "change",
    [
        {"select": {"mode": "curated", "max_per_problem": None}},  # would re-curate a frozen set
        {"select": {"mode": "all", "max_per_problem": 8}},
        {"data_condition": "train_mixed_100"},
        {"learning_rate": 3e-5},  # off the locked grid
        {"epochs": 3},
    ],
)
def test_design_guards_refuse_other_experiments(change: dict) -> None:
    cfg = load_arm_config(REPO / "configs/rft/iter_curated.yaml") | change
    with pytest.raises(SystemExit):
        iter_rft.validate_config(cfg)


def test_result_bearing_stages_need_the_preregistration_commit() -> None:
    cfg = load_arm_config(REPO / "configs/rft/iter_curated.yaml")
    iter_rft.assert_preregistered(cfg)  # this checkout descends from it
    with pytest.raises(SystemExit):
        iter_rft.assert_preregistered(cfg | {"preregistration_commit": "0" * 40})
    with pytest.raises(SystemExit):
        iter_rft.assert_preregistered(cfg | {"preregistration_commit": None})


def test_iterated_rft_code_never_names_a_held_out_split() -> None:
    for rel in ("src/rlordata/train/iter_rft.py", "scripts/iter_rft.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert not re.search(r"test_300|ood_hard_200|gsm8k", text), rel


def test_only_the_approved_exploratory_cap_passes_the_cap_guard(
    tmp_path: Path, monkeypatch
) -> None:
    cap_path = tmp_path / "cap.yaml"
    cap_path.write_text(yaml.safe_dump({"max_completion_tokens": 4352}), encoding="utf-8")
    monkeypatch.delenv(cap_mod.EXPLORATORY_CAP_ENV, raising=False)
    with pytest.raises(cap_mod.CapError):
        cap_mod.resolve_cap(8704, cap_path=cap_path)
    monkeypatch.setenv(cap_mod.EXPLORATORY_CAP_ENV, "yes please")
    with pytest.raises(cap_mod.CapError):
        cap_mod.resolve_cap(8704, cap_path=cap_path)
    monkeypatch.setenv(cap_mod.EXPLORATORY_CAP_ENV, cap_mod.EXPLORATORY_CAP_DEVIATION_ID)
    assert cap_mod.resolve_cap(8704, cap_path=cap_path) == 8704
    assert cap_mod.resolve_cap(None, cap_path=cap_path) == 4352  # the default stays the locked cap
    for other in (8192, 4353, 16384):
        with pytest.raises(cap_mod.CapError):
            cap_mod.resolve_cap(other, cap_path=cap_path)


def test_exploratory_units_cannot_be_written_where_results_live(tmp_path: Path) -> None:
    from scripts.exploratory_cap_eval import refuse_result_dirs

    refuse_result_dirs(tmp_path / "runs" / "exploratory_cap8704")
    for bad in (
        tmp_path / "runs" / "eval" / "exploratory_x",
        tmp_path / "runs" / "rft" / "arm" / "seed1" / "eval" / "final" / "exploratory_x",
        tmp_path / "runs" / "cap8704",
    ):
        with pytest.raises(SystemExit):
            refuse_result_dirs(bad)
    queue = yaml.safe_load((REPO / "queue_tasks06b.yaml").read_text())["jobs"]
    assert [j["name"] for j in queue][0] == "eval_base_gsm8k_mean8" and len(queue) == 6
    cfg = yaml.safe_load((REPO / "configs/eval/exploratory_cap.yaml").read_text())
    assert cfg["cap_multiple"] == 2 and cfg["splits"] == ["test_300", "ood_hard_200"]
    assert len(cfg["runs"]) == 9 and "exploratory" in Path(cfg["output_dir"]).name


def _exploratory_cap_dry(root: Path, arm: Path) -> None:
    """The exploratory script on the dry world: 2 × the world's cap, stamped, in its own tree."""
    from scripts.exploratory_cap_eval import main as exploratory_main

    world_cfg = yaml.safe_load((root / "configs" / "iter_rft_curated.yaml").read_text())
    locked = yaml.safe_load(Path(world_cfg["cap_yaml"]).read_text())["max_completion_tokens"]
    cfg = {
        "cap_yaml": world_cfg["cap_yaml"],
        "cap_multiple": 2,
        "splits": ["test_300"],
        "splits_dir": world_cfg["splits_dir"],
        "pool": world_cfg["pool"],
        "ood_path": world_cfg["ood_path"],
        "output_dir": str(root / "runs" / "exploratory_dry"),
        "model_id": world_cfg["model_id"],
        "base": {"name": "base", "seed": 1},
        "runs": [{"name": "iter_s1", "dir": str(arm / "seed1"), "seed": 1}],
    }
    path = root / "configs" / "exploratory.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert exploratory_main(["--config", str(path), "--stub", "--n-problems", "4"]) == 0
    for name in ("base", "iter_s1"):
        unit = root / "runs" / "exploratory_dry" / name / "test_300" / "greedy"
        metrics = json.loads((unit / "metrics.json").read_text())
        resolved = yaml.safe_load((unit / "config.yaml").read_text())
        assert metrics["exploratory"] is True and resolved["exploratory"] is True
        assert resolved["max_completion_tokens"] == 2 * locked == metrics["max_completion_tokens"]
        assert resolved["kind"] == "exploratory_cap_eval"
    # the primary final eval of the same run is untouched and still at the locked cap
    primary = yaml.safe_load(
        (arm / "seed1" / "eval" / "final" / "test_300" / "greedy" / "config.yaml").read_text()
    )
    assert primary["max_completion_tokens"] == locked and "exploratory" not in primary
    os.environ.pop("RLORDATA_EXPLORATORY_CAP_DEVIATION", None)


@pytest.mark.slow
def test_iter_rft_pipeline_dry(tmp_path: Path) -> None:
    """3 rounds × 2 seeds on the tiny world: layout, continuation, budgets, final = last round."""
    import torch
    from safetensors.torch import load_file

    from rlordata.train.rft_pipeline_dry import (
        core_implemented,
        run_iter_dry_pipeline,
        tokenizer_available,
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not tokenizer_available() or not core_implemented():
        pytest.skip("tokenizer not cached or core functions not implemented")
    assert run_iter_dry_pipeline(tmp_path, keep=True) == 0
    arm = tmp_path / "runs" / "rft" / "iter_rft_curated"
    for seed in (1, 2):
        d = arm / f"seed{seed}"
        budgets = json.loads((d / "budgets.json").read_text())
        rounds = [json.loads((d / f"round_{r}" / "budgets.json").read_text()) for r in (1, 2, 3)]
        n_prompts = budgets["prompts_parent_split"]
        assert budgets["completions_available"] == n_prompts * 4 * 3 and budgets["rounds"] == 3
        for key in ("completions_consumed", "training_tokens", "optimizer_steps"):
            assert budgets[key] == sum(r[key] for r in rounds)
        # continuation: round r starts from round r-1's final adapter, and training moved it
        assert rounds[0]["init_adapter"] is None
        for r in (1, 2):
            assert rounds[r]["init_adapter"] == rounds[r - 1]["final_adapter"]
        weights = [
            load_file(str(d / f"round_{r}" / "adapter" / "final" / "adapter_model.safetensors"))
            for r in (1, 2, 3)
        ]
        for a, b in zip(weights, weights[1:], strict=False):
            assert any(not torch.equal(a[k], b[k]) for k in a)
        # final = the last round's last epoch, byte for byte
        final = load_file(str(d / "adapter" / "final" / "adapter_model.safetensors"))
        assert all(torch.equal(final[k], weights[2][k]) for k in final)
        assert "round_3" in (d / "adapter" / "final" / "SOURCE.txt").read_text()
        # round 1 data is the base draw's first 4 per prompt; rounds 2-3 were sampled from the policy
        first = [json.loads(line) for line in (d / "round_1" / "samples.jsonl").open()]
        assert {s["extra"]["round_source"] for s in first} == {"base_draw_first_n"}
        assert max(s["extra"]["sample_idx"] for s in first) == 3
        second = [json.loads(line) for line in (d / "round_2" / "samples.jsonl").open()]
        assert {s["extra"]["sampler_seed"] for s in second} == {1000 * seed + 2}
        assert {s["extra"]["policy_adapter"] for s in second} == {rounds[0]["final_adapter"]}
        # val after every round, the final sets once, diagnostics present
        info = json.loads((d / "rounds.json").read_text())["rounds"]
        assert [r["round"] for r in info] == [1, 2, 3] and all(r["val_greedy"] for r in info)
        assert all("onpolicy" in r["diagnostics"] for r in info)
        assert (d / "eval" / "final" / "summary.json").exists()
        assert yaml.safe_load((d / "config.yaml").read_text())["append_eos"] is True
    _exploratory_cap_dry(tmp_path, arm)
    s1 = (arm / "seed1" / "round_1" / "samples.jsonl").read_text()
    assert s1 == (arm / "seed2" / "round_1" / "samples.jsonl").read_text()  # one shared draw
