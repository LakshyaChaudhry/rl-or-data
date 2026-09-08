from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rlordata.cli import main as cli_main
from rlordata.data.generator import read_jsonl
from rlordata.sampling.cap import ANCHOR_CAP, cell_key, compute_cap
from rlordata.sampling.eval_runner import read_samples, write_samples
from rlordata.types import Sample
from tests.helpers import REPO, make_world

PROVENANCE_FIELDS = (
    "max_completion_tokens",
    "anchor_cap",
    "binding_term",
    "measured_cap",
    "per_cell_p99",
    "driving_cell",
    "frac_correct_over_2048",
    "p99_correct_len",
    "n_correct_used",
    "computed_on",
    "config_hash",
    "run_id",
    "git_sha",
    "computed_at",
    "rule",
    "frac_correct_truncated_at_cap",
    "per_cell_stats",
    "pool",
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "compute_cap", REPO / "scripts" / "compute_cap.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A stub provisional cap run: base model, T=1, n=8, cap 4096, val_candidates (needs >= 500 pool problems)."""
    w = make_world(tmp_path_factory.mktemp("cap"), n_pool=600, n_ood=6)
    cfg = w.eval_config(
        splits=["val_candidates"],
        provisional_cap=4096,
        models=[{"id": "Qwen/Qwen3-4B-Base", "kind": "base", "arm": "base"}],
        decoding={"mean_at_k": {"temperature": 1.0, "top_p": 1.0, "n": 8}},
        output_dir=str(w.root / "runs" / "cap_provisional"),
    )
    assert cli_main(["eval", "--config", str(cfg), "--stub"]) == 0
    return w


@pytest.fixture(scope="module")
def provisional_run(world) -> Path:
    return (
        world.root
        / "runs"
        / "cap_provisional"
        / "Qwen__Qwen3-4B-Base"
        / "val_candidates"
        / "mean_at_k"
    )


def _cells_from(samples: list[Sample], pool) -> tuple[dict[str, list[int]], list[int]]:
    by_id = {p.problem_id: p for p in pool}
    cells: dict[str, list[int]] = {}
    incorrect: list[int] = []
    for s in samples:
        p = by_id[s.problem_id]
        (
            cells.setdefault(cell_key(p.range_scale, p.total_steps), []) if s.correct else incorrect
        ).append(s.n_tokens)
    return cells, incorrect


def test_compute_cap_end_to_end_and_refuses_overwrite(
    world, provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    out = tmp_path / "locked" / "cap.yaml"
    rc = mod.main(
        [
            "--run-dir",
            str(provisional_run),
            "--pool",
            str(world.pool_path),
            "--out",
            str(out),
            "--allow-stub-samples",
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0, printed
    # per-cell table, driving cell, binding term, anchor comparison, histogram
    assert (
        "per-cell table" in printed
        and "driving cell (largest p99)" in printed
        and "binding term:" in printed
    )
    assert "correct completions over 2048 (anchor comparison)" in printed
    assert "incorrect" in printed and "tokens" in printed
    data = yaml.safe_load(out.read_text())
    for key in PROVENANCE_FIELDS:
        assert key in data, key
    # independent recomputation from the samples + pool join
    samples = read_samples(provisional_run / "samples.jsonl")
    cells, incorrect = _cells_from(samples, read_jsonl(world.pool_path))
    r = compute_cap(
        cells, incorrect, n_problems=500, provisional_truncated=[s.truncated for s in samples]
    )
    assert (
        data["max_completion_tokens"] == r.max_completion_tokens == max(ANCHOR_CAP, r.measured_cap)
    )
    assert data["max_completion_tokens"] % 256 == 0 and data["max_completion_tokens"] >= 2048
    assert data["anchor_cap"] == 2048 and data["binding_term"] == r.binding_term in (
        "floor",
        "measured",
    )
    assert data["measured_cap"] == r.measured_cap and data["driving_cell"] == r.driving_cell
    assert set(data["per_cell_p99"]) == set(cells) and len(cells) == 8  # 8 pool cells: S/M × 2..5
    assert data["per_cell_p99"][data["driving_cell"]] == max(data["per_cell_p99"].values())
    assert data["p99_correct_len"] == pytest.approx(r.p99_max, abs=0.01)
    assert data["n_correct_used"] == sum(len(v) for v in cells.values())
    assert data["n_problems"] == 500 and data["n_samples_total"] == 4000
    assert data["config_hash"] == (provisional_run / "config_hash.txt").read_text().strip()
    assert data["run_id"] == json.loads((provisional_run / "meta.json").read_text())["run_id"]
    correct_all = np.array([x for v in cells.values() for x in v])
    assert data["frac_correct_over_2048"] == pytest.approx(
        float(np.mean(correct_all > 2048)), abs=1e-6
    )
    assert data["frac_correct_truncated_at_cap"] == pytest.approx(
        float(np.mean(correct_all > data["max_completion_tokens"])), abs=1e-6
    )
    assert data["frac_correct_truncated_at_cap"] < 0.01
    assert "2048" in data["rule"] and "512" not in data["rule"]
    # refuses to overwrite, file unchanged
    before = out.read_text()
    rc = mod.main(
        [
            "--run-dir",
            str(provisional_run),
            "--pool",
            str(world.pool_path),
            "--out",
            str(out),
            "--allow-stub-samples",
        ]
    )
    assert rc == 2 and "REFUSING" in capsys.readouterr().err
    assert out.read_text() == before


def test_compute_cap_refuses_stub_samples_wrong_pool_and_protocol_mismatch(
    world, provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    rc = mod.main(
        [
            "--run-dir",
            str(provisional_run),
            "--pool",
            str(world.pool_path),
            "--out",
            str(tmp_path / "cap.yaml"),
        ]
    )
    assert rc == 2 and "not vllm" in capsys.readouterr().err
    assert not (tmp_path / "cap.yaml").exists()
    # a pool that does not contain the sampled problems -> join failure, nothing written
    rc = mod.main(
        [
            "--run-dir",
            str(provisional_run),
            "--pool",
            str(world.ood_path),
            "--out",
            str(tmp_path / "cap.yaml"),
            "--allow-stub-samples",
        ]
    )
    assert rc == 2 and "not in the pool" in capsys.readouterr().err
    assert not (tmp_path / "cap.yaml").exists()
    # stub samples can never target the real locked path
    cfg = yaml.safe_load((provisional_run / "config.yaml").read_text())
    issues = mod.validate_run(cfg, allow_stub=True, out=REPO / "configs" / "locked" / "cap.yaml")
    assert any("real configs/locked/cap.yaml" in i for i in issues)
    bad = dict(cfg)
    bad["decoding"] = {**cfg["decoding"], "temperature": 0.0, "n": 1}
    bad["max_completion_tokens"] = 2048
    bad["split"] = "test_300"
    bad["model"] = {"id": "Qwen/Qwen3-4B", "kind": "instruct"}
    issues = mod.validate_run(bad, allow_stub=True, out=tmp_path / "x.yaml")
    assert len(issues) == 6


def _synthetic_run(
    world, tmp_path: Path, provisional_run: Path, lengths_by_cell: dict[str, list[int]]
) -> Path:
    """Clone the provisional run dir but rewrite samples so correct lengths follow ``lengths_by_cell``."""
    import shutil

    run = tmp_path / "run"
    shutil.copytree(provisional_run, run)
    pool = read_jsonl(world.pool_path)
    by_cell: dict[str, list] = {}
    for p in pool:
        by_cell.setdefault(cell_key(p.range_scale, p.total_steps), []).append(p)
    template = read_samples(provisional_run / "samples.jsonl")[0]
    samples: list[Sample] = []
    for cell, lengths in lengths_by_cell.items():
        probs = by_cell[cell]
        for i, n_tok in enumerate(lengths):
            p = probs[i % len(probs)]
            samples.append(
                Sample(
                    **{
                        **template.to_dict(),
                        "problem_id": p.problem_id,
                        "n_tokens": int(n_tok),
                        "correct": True,
                        "reward": 1.0,
                        "truncated": False,
                        "extraction_failed": False,
                        "extra": {
                            **template.extra,
                            "range_scale": p.range_scale,
                            "total_steps": p.total_steps,
                        },
                    }
                )
            )
    # a few incorrect ones
    samples.append(
        Sample(
            **{
                **template.to_dict(),
                "correct": False,
                "reward": 0.0,
                "n_tokens": 4096,
                "truncated": True,
            }
        )
    )
    write_samples(samples, run / "samples.jsonl")
    return run


def test_script_floor_binds_on_short_synthetic_cells(
    world, provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    run = _synthetic_run(
        world, tmp_path, provisional_run, {"S2": [120] * 100, "M5": [600] * 90 + [1500] * 10}
    )
    out = tmp_path / "cap.yaml"
    assert (
        mod.main(
            [
                "--run-dir",
                str(run),
                "--pool",
                str(world.pool_path),
                "--out",
                str(out),
                "--allow-stub-samples",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    data = yaml.safe_load(out.read_text())
    assert (
        data["max_completion_tokens"] == 2048
        and data["binding_term"] == "floor"
        and data["driving_cell"] == "M5"
    )
    assert "binding term: floor" in printed and "M5" in printed


def test_script_measurement_binds_and_records_raise(
    world, provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    run = _synthetic_run(
        world, tmp_path, provisional_run, {"S2": [120] * 100, "M4": list(range(1500, 2300))}
    )
    out = tmp_path / "cap.yaml"
    assert (
        mod.main(
            [
                "--run-dir",
                str(run),
                "--pool",
                str(world.pool_path),
                "--out",
                str(out),
                "--allow-stub-samples",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    data = yaml.safe_load(out.read_text())
    assert (
        data["binding_term"] == "measured"
        and data["max_completion_tokens"] == data["measured_cap"] > 2048
    )
    assert data["max_completion_tokens"] % 256 == 0 and data["driving_cell"] == "M4"
    assert "NOTE: the measured term raised the cap" in printed


def test_script_fails_loudly_when_a_cell_exceeds_one_percent(
    world, provisional_run: Path, tmp_path: Path, capsys
) -> None:
    mod = _load_script()
    run = _synthetic_run(
        world, tmp_path, provisional_run, {"S2": [120] * 300, "M5": [300] * 49 + [9000]}
    )
    out = tmp_path / "cap.yaml"
    rc = mod.main(
        [
            "--run-dir",
            str(run),
            "--pool",
            str(world.pool_path),
            "--out",
            str(out),
            "--allow-stub-samples",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 3 and "cell M5" in err and "1 % limit" in err
    assert not out.exists()
