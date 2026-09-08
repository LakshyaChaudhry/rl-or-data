from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rlordata.cli import main
from rlordata.run_dir import REQUIRED_RUN_FILES
from rlordata.sampling.cap import CapError, write_cap_yaml
from rlordata.sampling.eval_runner import (
    DecodingSpec,
    ModelSpec,
    metrics_for,
    model_slug,
    parse_subset,
    read_samples,
    summarize,
)
from rlordata.types import Sample
from tests.helpers import World, make_world

REQUIRED_SAMPLE_FIELDS = (
    "run_id",
    "config_hash",
    "seed",
    "arm",
    "data_condition",
    "problem_id",
    "tier",
    "completion",
    "extracted_answer",
    "correct",
    "n_tokens",
    "truncated",
    "extraction_failed",
)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    w = make_world(tmp_path_factory.mktemp("world"))
    assert w.tier_with_stub() == 0
    return w


def test_specs_and_helpers() -> None:
    assert model_slug("Qwen/Qwen3-4B-Base") == "Qwen__Qwen3-4B-Base"
    assert parse_subset("test_300_first100") == ("test_300", 100)
    with pytest.raises(ValueError):
        parse_subset("test_300")
    with pytest.raises(ValueError):
        DecodingSpec.from_config("greedy", {"temperature": 0.0, "n": 8})
    with pytest.raises(ValueError):
        ModelSpec.from_config({"id": "x", "kind": "chat"})
    d = DecodingSpec.from_config(
        "pass_at_k", {"temperature": 1.0, "n": 64, "subset": "test_300_first100", "ks": [1, 8]}
    )
    assert d.ks == (1, 8) and d.subset == "test_300_first100"


def _sample(pid: str, correct: bool, tier: str = "easy") -> Sample:
    return Sample(
        run_id="r",
        config_hash="h",
        seed=1,
        arm="base",
        data_condition="d",
        problem_id=pid,
        tier=tier,  # type: ignore[arg-type]
        prompt="p",
        completion="c",
        extracted_answer=1,
        correct=correct,
        reward=float(correct),
        n_tokens=10,
        truncated=False,
        extraction_failed=False,
    )


def test_metrics_per_tier_truncation_and_flag_policy() -> None:
    def s(pid: str, tier: str, truncated: bool, ext: bool, n_tokens: int) -> Sample:
        return Sample(
            run_id="r",
            config_hash="h",
            seed=1,
            arm="base",
            data_condition="d",
            problem_id=pid,
            tier=tier,  # type: ignore[arg-type]
            prompt="p",
            completion="c",
            extracted_answer=None if ext else 1,
            correct=False,
            reward=0.0,
            n_tokens=n_tokens,
            truncated=truncated,
            extraction_failed=ext or truncated,
        )

    samples = (
        [s("e1", "easy", False, False, 100), s("e1", "easy", False, False, 120)]
        + [s("m1", "medium", True, True, 2048), s("m1", "medium", False, False, 500)]
        + [s("h1", "hard", True, True, 2048), s("h1", "hard", True, True, 2048)]
    )
    m = metrics_for(samples, cap=2048, split="test_300")
    assert m["per_tier_truncation_rate"] == {"easy": 0.0, "hard": 1.0, "medium": 0.5}
    assert m["per_tier_extraction_failure_rate"] == {"easy": 0.0, "hard": 1.0, "medium": 0.5}
    assert m["per_tier_at_cap_rate"] == {"easy": 0.0, "hard": 1.0, "medium": 0.5}
    assert m["per_tier_n_samples"] == {"easy": 2, "hard": 2, "medium": 2}
    assert m["at_cap_rate"] == pytest.approx(0.5) and m["truncation_rate"] == pytest.approx(0.5)
    assert m["truncation_over_5pct"] is True and m["truncation_flag_policy"] == "flag"
    assert m["flags"]["truncation_gt_5pct"] is True
    # ood_hard_200: reported, never flagged
    m_ood = metrics_for(samples, cap=2048, split="ood_hard_200")
    assert (
        m_ood["truncation_over_5pct"] is True and m_ood["truncation_flag_policy"] == "report_only"
    )
    assert m_ood["flags"]["truncation_gt_5pct"] is False
    assert m_ood["per_tier_truncation_rate"] == m["per_tier_truncation_rate"]
    # no cap given -> at-cap rates absent, nothing else changes
    assert metrics_for(samples)["at_cap_rate"] is None


def test_metrics_for_pass_at_k_and_flags() -> None:
    samples = (
        [_sample("a", True)] * 8
        + [_sample("b", False)] * 7
        + [_sample("b", True)]
        + [_sample("c", False)] * 8
    )
    m = metrics_for(samples, ks=(1, 2, 4, 8))
    assert m["samples_per_problem"] == 8 and m["n_problems"] == 3
    assert m["pass_at_k"]["8"] == pytest.approx(2 / 3)
    assert m["pass_at_k"]["1"] == pytest.approx((1 + 1 / 8 + 0) / 3)
    assert m["accuracy"] == pytest.approx((1 + 1 / 8 + 0) / 3)
    assert m["flags"] == {"truncation_gt_5pct": False, "extraction_failure_gt_5pct": False}
    assert m["per_tier_n"] == {"easy": 3}


def test_eval_stub_end_to_end_produces_every_file(world: World, capsys) -> None:
    cfg = world.eval_config()
    rc = main(["eval", "--config", str(cfg), "--stub"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "DRY RUN" in out and "cost estimate (START" in out and "cost estimate (END" in out
    base = yaml.safe_load(cfg.read_text())
    out_dir = Path(base["output_dir"])
    models = [ModelSpec.from_config(m) for m in base["models"]]
    assert len(models) == 5
    n_units = 0
    for m in models:
        for split in ("val_mixed_100", "test_300", "ood_hard_200"):
            for dec in ("greedy", "mean_at_k"):
                d = out_dir / m.slug / split / dec
                for f in REQUIRED_RUN_FILES + ("samples.jsonl", "metrics.json"):
                    assert (d / f).exists(), d / f
                n_units += 1
        d = out_dir / m.slug / "test_300" / "pass_at_k"
        assert (d / "samples.jsonl").exists() and (d / "metrics.json").exists()
        n_units += 1
    assert n_units == 35
    assert (out_dir / "summary.md").exists() and (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert len(summary["rows"]) == 15 and summary["status"] == "finished"
    table = (out_dir / "summary.md").read_text()
    assert (
        "greedy [95% CI]" in table and "mean@8" in table and "pass@8" in table and "trunc%" in table
    )
    assert "per-tier truncation % / extraction-failure %" in table and "at-cap%" in table
    for r in summary["rows"]:
        assert set(r["per_tier_truncation"]) >= {"greedy", "mean_at_k"}
        assert set(r["per_tier_truncation"]["greedy"]) == {"easy", "medium", "hard"}
        if r["split"] == "ood_hard_200":
            assert r["flagged"] is False  # never flagged, only reported

    # one unit in detail: samples carry every CLAUDE.md field; metrics from core.evaluate
    d = out_dir / models[0].slug / "test_300" / "mean_at_k"
    samples = read_samples(d / "samples.jsonl")
    assert len(samples) == 60 * 8
    rec = json.loads((d / "samples.jsonl").read_text().splitlines()[0])
    for f in REQUIRED_SAMPLE_FIELDS:
        assert f in rec, f
    assert rec["data_condition"] == "test_300" and rec["arm"] == "base"
    metrics = json.loads((d / "metrics.json").read_text())
    for key in (
        "accuracy",
        "ci_low",
        "ci_high",
        "truncation_rate",
        "extraction_failure_rate",
        "mean_completion_tokens",
        "per_tier",
        "pass_at_k",
        "flags",
        "config_hash",
        "n_problems",
    ):
        assert key in metrics, key
    assert metrics["ci_low"] <= metrics["accuracy"] <= metrics["ci_high"]
    assert (
        metrics["n_problems"] == 60
        and metrics["samples_per_problem"] == 8
        and "8" in metrics["pass_at_k"]
    )
    assert set(metrics["per_tier"]) == {"easy", "medium", "hard"}
    assert (
        metrics["config_hash"] == rec["config_hash"] == (d / "config_hash.txt").read_text().strip()
    )
    resolved = yaml.safe_load((d / "config.yaml").read_text())
    assert resolved["max_completion_tokens"] == 4096 and resolved["cap_is_provisional"] is True
    assert resolved["prompt_template"].startswith("Solve the following problem.")
    assert resolved["decoding"] == {
        "name": "mean_at_k",
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 8,
        "ks": None,
        "repetition_penalty": 1.0,
    }
    assert resolved["thinking"] is False and resolved["chat_template_kwargs"] is None
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["n_samples"] == 480 and "gpu_type" in meta

    # instruct model: chat kwargs recorded and the prompt went through the chat template
    d_inst = out_dir / model_slug("Qwen/Qwen3-4B") / "val_mixed_100" / "greedy"
    resolved_i = yaml.safe_load((d_inst / "config.yaml").read_text())
    assert resolved_i["chat_template_kwargs"] == {"enable_thinking": False}
    first = json.loads((d_inst / "samples.jsonl").read_text().splitlines()[0])
    assert first["prompt"].startswith("<|user|>") and first["arm"] == "ref"
    d_llama = out_dir / model_slug("meta-llama/Llama-3.1-8B-Instruct") / "val_mixed_100" / "greedy"
    assert yaml.safe_load((d_llama / "config.yaml").read_text())["chat_template_kwargs"] == {}

    # pass@k unit: first 100 of test_300 by index (here all 60), n=64, ks from config
    d_pk = out_dir / models[0].slug / "test_300" / "pass_at_k"
    mp = json.loads((d_pk / "metrics.json").read_text())
    assert mp["samples_per_problem"] == 64
    assert sorted(int(k) for k in mp["pass_at_k"]) == [1, 2, 4, 8, 16, 32, 64]
    assert mp["pass_at_k"]["64"] >= mp["pass_at_k"]["8"] >= mp["pass_at_k"]["1"]
    test_ids = [
        json.loads(line)["problem_id"]
        for line in (world.splits_dir / "test_300.jsonl").read_text().splitlines()
    ]
    pk_ids = []
    for s in read_samples(d_pk / "samples.jsonl"):
        if s.problem_id not in pk_ids:
            pk_ids.append(s.problem_id)
    assert pk_ids == test_ids[:100]

    # ood_hard_200 came from the post-hoc tiered split (tiers assigned)
    d_ood = out_dir / models[0].slug / "ood_hard_200" / "greedy"
    m_ood = json.loads((d_ood / "metrics.json").read_text())
    assert m_ood["n_problems"] == 30 and "untiered" not in m_ood["per_tier"]
    assert m_ood["truncation_flag_policy"] == "report_only"
    assert m_ood["flags"]["truncation_gt_5pct"] is False
    assert set(m_ood["per_tier_truncation_rate"]) == set(m_ood["per_tier"])
    assert m_ood["max_completion_tokens"] == 4096
    assert metrics["truncation_flag_policy"] == "flag" and "per_tier_at_cap_rate" in metrics


def test_test_300_is_evaluated_once_unless_forced(world: World, capsys) -> None:
    cfg = world.eval_config()
    rc = main(["eval", "--config", str(cfg), "--stub"])
    err = capsys.readouterr().err
    assert rc == 2 and "test_300" in err and "--force" in err
    # resume: remove one val unit and every test_300 unit -> only those re-run, others untouched
    out_dir = Path(yaml.safe_load(cfg.read_text())["output_dir"])
    slug = model_slug("Qwen/Qwen3-4B-Base")
    keep = out_dir / slug / "ood_hard_200" / "greedy" / "meta.json"
    before = keep.read_text()
    import shutil

    shutil.rmtree(out_dir / slug / "val_mixed_100" / "greedy")
    for m in yaml.safe_load(cfg.read_text())["models"]:
        shutil.rmtree(out_dir / model_slug(m["id"]) / "test_300")
    rc = main(["eval", "--config", str(cfg), "--stub"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "16 to run, 19 already done" in out
    assert keep.read_text() == before
    assert (out_dir / slug / "val_mixed_100" / "greedy" / "samples.jsonl").exists()
    # --force re-runs everything, including test_300
    assert main(["eval", "--config", str(cfg), "--stub", "--force"]) == 0
    assert "35 to run, 0 already done" in capsys.readouterr().out


def test_dry_run_and_n_problems(world: World, tmp_path: Path, capsys) -> None:
    cfg = world.eval_config(output_dir=str(tmp_path / "out"))
    assert main(["eval", "--config", str(cfg), "--stub", "--dry-run"]) == 0
    assert "dry-run: not sampling" in capsys.readouterr().out
    assert not (tmp_path / "out" / "summary.md").exists()
    cfg2 = world.eval_config(
        output_dir=str(tmp_path / "out2"),
        models=[{"id": "Qwen/Qwen3-4B-Base", "kind": "base", "arm": "base"}],
    )
    assert main(["eval", "--config", str(cfg2), "--stub", "--n-problems", "7"]) == 0
    m = json.loads(
        (
            tmp_path / "out2" / "Qwen__Qwen3-4B-Base" / "test_300" / "pass_at_k" / "metrics.json"
        ).read_text()
    )
    assert m["n_problems"] == 7


def test_real_sampler_refuses_without_cap_and_locked_cap_is_used(
    world: World, tmp_path: Path
) -> None:
    cfg = world.eval_config(output_dir=str(tmp_path / "out"))
    with pytest.raises(CapError):
        main(["eval", "--config", str(cfg)])  # no --stub, no cap.yaml
    cap_path = tmp_path / "cap.yaml"
    write_cap_yaml(cap_path, {"max_completion_tokens": 768})
    cfg2 = world.eval_config(
        output_dir=str(tmp_path / "out2"),
        cap_yaml=str(cap_path),
        models=[{"id": "Qwen/Qwen3-4B-Base", "kind": "base", "arm": "base"}],
        splits=["val_mixed_100"],
        decoding={"greedy": {"temperature": 0.0, "n": 1}},
    )
    assert main(["eval", "--config", str(cfg2), "--stub"]) == 0
    resolved = yaml.safe_load(
        (
            tmp_path / "out2" / "Qwen__Qwen3-4B-Base" / "val_mixed_100" / "greedy" / "config.yaml"
        ).read_text()
    )
    assert resolved["max_completion_tokens"] == 768 and resolved["cap_is_provisional"] is False
    # a provisional-cap config cannot run once the cap is locked to a different value
    cfg3 = world.eval_config(
        output_dir=str(tmp_path / "out3"),
        cap_yaml=str(cap_path),
        provisional_cap=4096,
        splits=["val_candidates"],
    )
    with pytest.raises(CapError):
        main(["eval", "--config", str(cfg3), "--stub"])


def test_summarize_handles_partial_results() -> None:
    table, rows = summarize(
        [
            {
                "model_id": "m",
                "split": "s",
                "decoding": "greedy",
                "accuracy": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "truncation_rate": 0.0,
                "extraction_failure_rate": 0.0,
                "n_problems": 10,
                "config_hash": "abc",
                "flags": {"truncation_gt_5pct": False},
            },
        ]
    )
    assert (
        rows[0]["mean_at_8"] is None
        and rows[0]["pass_at_8"] is None
        and "0.500 [0.400,0.600]" in table
    )
